// name: src/core/src/runtime/telemetry/bank.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core telemetry bank accumulation, slow-loop self-tuning, and snapshot projection.

use super::*;
use crate::drivers::{
    DistortionTension, EGO_TRIGGER_BASIN_CAPTURE, EGO_TRIGGER_BOUNDARY_CHATTER,
    EGO_TRIGGER_BOUNDARY_STAGNATION, EGO_TRIGGER_INTERNAL_CONFLICT, EGO_TRIGGER_LIMINAL_EXTERIOR,
    LiminalEgoState,
};
use crate::foundation::{Pod, branchless_clamp_f32};
use crate::liminal::{
    EpistemicRegime, LacunaPartition8, NUM_LIMINAL_THRESHOLD_CHANNELS, NUM_STIGMERGIC_GRANULES,
};
use crate::runtime::boundary_delta::LiminalBoundaryDelta;
use crate::runtime::context::CoreStateStack;
use crate::runtime::gateway::signals::ManifoldProbeBaseline;
use crate::runtime::horizon::EpistemicHorizonReading;
use crate::stigmergy::{OverlapDiscoveryReport, StigmergicActionDecision, StigmergicThresholds};

/// Standalone `#![no_std]` zero-alloc `3 + System` telemetry bank stored beside compute engines.
///
/// Hot `Coarse`-mode control and counters (`policy`, `active_mode`, and `spine.total_steps` /
/// `spine.qos_throttle_events`) reside in Cache Line 0 (`0..64`), while `detailed` is isolated at
/// offset `384` (Cache Line 6) to avoid polluting Hot/Warm cache lines.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreTelemetryBank {
    policy: TelemetryCapturePolicy,
    active_mode: TelemetryVerbosityMode,
    derivative_alpha_q7: u8,
    ema_initialized: bool,
    reserved_cl0: u8,
    /// Vertical `+ System` spine counters (hot `Coarse` counters packed within `0..64`).
    pub spine: SystemSpineTelemetry,
    attribution_counts: [u32; 8],
    whole_token_bypass_events: u64,
    immediate_tension_damps: u64,
    slow_tension_excites: u64,
    pframe_chain_evals: u64,
    pframe_total_hops: u64,
    exterior_sandbox_stages: u64,
    unobserved_aperture_zeros: u64,
    /// Layer 0 (`Liminal`) counters.
    pub liminal: LiminalTelemetry,
    /// Layer 1 (`Stigmergic`) counters.
    pub stigmergy: StigmergicTelemetry,
    /// Layer 2 (`Ego`) counters.
    pub ego: EgoTelemetry,
    /// Consecutive quiescent steps required before retracting from `Detailed` back to `base`.
    pub retract_dwell_threshold: u16,
    entropy_ema_q15: u16,
    coherence_ema_q15: u16,
    guard_ema_q15: u16,
    assimilation_drift_q15: u16,
    assimilation_drift_dwell: u16,
    sandbox_coherence_improve_dwell: u16,
    reserved_warm_pad: u16,
    guard_relaxation_count: u32,
    instability_recovery_count: u32,
    /// Slide-in detailed counters (untouched unless `active_mode == Detailed`; CL6-isolated).
    pub detailed: DetailedLayerTelemetry,
    abstract_token_count: u32,
    concrete_token_count: u32,
    boundary_churn_count: u32,
    logits_skip_count: u32,
}

const _: () = {
    assert!(core::mem::size_of::<TelemetryCapturePolicy>() == 4);
    assert!(core::mem::align_of::<CoreTelemetryBank>() == 64);
    assert!(core::mem::size_of::<CoreTelemetryBank>() == 576);
    assert!(core::mem::offset_of!(CoreTelemetryBank, spine) == 8);
    assert!(core::mem::offset_of!(CoreTelemetryBank, guard_relaxation_count) == 376);
    assert!(core::mem::offset_of!(CoreTelemetryBank, instability_recovery_count) == 380);
    assert!(core::mem::offset_of!(CoreTelemetryBank, detailed) == 384);
    assert!(core::mem::offset_of!(CoreTelemetryBank, abstract_token_count) == 560);
    assert!(core::mem::offset_of!(CoreTelemetryBank, logits_skip_count) == 572);
};

impl Default for CoreTelemetryBank {
    #[inline]
    fn default() -> Self {
        Self::new(TelemetryCapturePolicy::default())
    }
}

/// Per-step telemetry inputs for [`CoreTelemetryBank::record_step`].
#[derive(Debug, Clone, Copy)]
pub struct StepTelemetryInput<'a> {
    /// Stigmergic action decision selected for this step.
    pub decision: StigmergicActionDecision,
    /// Epistemic horizon reading for the current step.
    pub horizon: &'a EpistemicHorizonReading,
    /// Temporal boundary velocity analyzer reading.
    pub delta: &'a LiminalBoundaryDelta,
    /// Multi-attractor overlap and discovery excursion report.
    pub overlap: &'a OverlapDiscoveryReport,
    /// Active subjective Ego state snapshot.
    pub ego: &'a LiminalEgoState,
    /// Active entropy controller branch mode (`0`, `1`, `2`).
    pub entropy_gate_mode: u32,
    /// Bitmask of latent horizon slots slid in on this step.
    pub slid_in_bits: u32,
    /// Bitmask of latent horizon slots slid out on this step.
    pub slid_out_bits: u32,
    /// Whether cold-path horizon escalation fired on this step.
    pub escalated: bool,
    /// Whether system QoS throttling was active on this step.
    pub qos_throttled: bool,
}

impl CoreTelemetryBank {
    /// Creates a new [`CoreTelemetryBank`] governed by `policy`.
    #[inline]
    #[must_use]
    pub const fn new(policy: TelemetryCapturePolicy) -> Self {
        let initial_mode = match policy {
            TelemetryCapturePolicy::Pinned(m) => m,
            TelemetryCapturePolicy::Auto { base, .. } => base,
        };
        Self {
            policy,
            active_mode: initial_mode,
            derivative_alpha_q7: crate::stigmergy::substrate::DEFAULT_DERIVATIVE_ALPHA_Q7,
            ema_initialized: false,
            reserved_cl0: 0,
            spine: SystemSpineTelemetry {
                total_steps: 0,
                continuous_standard_steps: 0,
                qos_throttle_events: 0,
                entropy_gate_bifurcation_events: 0,
                entropy_gate_repulsion_events: 0,
                mode_expansions: 0,
                mode_retractions: 0,
                slow_loop_epochs: 0,
                experience_cache_hits: 0,
                experience_cache_stores: 0,
                experience_cache_evictions: 0,
                slow_loop_repulse_bias_q7: 0,
                slow_loop_dwell_delta: 0,
                reserved: [0; 6],
            },
            attribution_counts: [0; 8],
            whole_token_bypass_events: 0,
            immediate_tension_damps: 0,
            slow_tension_excites: 0,
            pframe_chain_evals: 0,
            pframe_total_hops: 0,
            exterior_sandbox_stages: 0,
            unobserved_aperture_zeros: 0,
            liminal: LiminalTelemetry {
                regime_counts: [0; 3],
                chatter_events: 0,
                stationary_shortcut_hits: 0,
                slide_in_count: 0,
                slide_out_count: 0,
                passive_stall_events: 0,
            },
            stigmergy: StigmergicTelemetry {
                decision_counts: [0; 7],
                overlap_discovery_hits: 0,
                quorum_core_hits: 0,
            },
            ego: EgoTelemetry {
                intervention_count: 0,
                tiebreak_count: 0,
                escalation_swaps: 0,
                unaligned_intervention_count: 0,
            },
            retract_dwell_threshold: 2,
            entropy_ema_q15: 0,
            coherence_ema_q15: 0,
            guard_ema_q15: 0,
            assimilation_drift_q15: 0,
            assimilation_drift_dwell: 0,
            sandbox_coherence_improve_dwell: 0,
            reserved_warm_pad: 0,
            guard_relaxation_count: 0,
            instability_recovery_count: 0,
            detailed: DetailedLayerTelemetry {
                liminal_channel_chatter_counts: [0; NUM_LIMINAL_THRESHOLD_CHANNELS],
                stigmergic_granule_overlap_hits: [0; NUM_STIGMERGIC_GRANULES],
                ego_trigger_counts: [0; 5],
                detailed_captured_steps: 0,
            },
            abstract_token_count: 0,
            concrete_token_count: 0,
            boundary_churn_count: 0,
            logits_skip_count: 0,
        }
    }

    /// Initializes non-zero fields of [`CoreTelemetryBank`] directly inside an already-zeroed target
    /// buffer without materializing a 576-byte stack temporary.
    ///
    /// # Safety
    /// Caller guarantees `ptr` points to a valid, properly aligned, all-zero `CoreTelemetryBank`.
    #[inline]
    pub(crate) unsafe fn init_in_place_on_zeroed(ptr: *mut Self, policy: TelemetryCapturePolicy) {
        let initial_mode = match policy {
            TelemetryCapturePolicy::Pinned(m) => m,
            TelemetryCapturePolicy::Auto { base, .. } => base,
        };
        // SAFETY: `ptr` is valid, aligned, and zero-initialized.
        unsafe {
            core::ptr::addr_of_mut!((*ptr).policy).write(policy);
            core::ptr::addr_of_mut!((*ptr).active_mode).write(initial_mode);
            core::ptr::addr_of_mut!((*ptr).derivative_alpha_q7)
                .write(crate::stigmergy::substrate::DEFAULT_DERIVATIVE_ALPHA_Q7);
            core::ptr::addr_of_mut!((*ptr).retract_dwell_threshold).write(2);
        }
    }

    /// Returns the Slow-Loop Q0.7 derivative weight $\alpha \in [0, 127]$ (read-only on the Hot Path).
    #[inline(always)]
    #[must_use]
    pub const fn derivative_alpha_q7(&self) -> u8 {
        self.derivative_alpha_q7
    }

    /// Sets the Slow-Loop Q0.7 derivative weight $\alpha \in [0, 127]$ (e.g., `0` for pure integral ablation).
    #[inline(always)]
    pub fn set_derivative_alpha_q7(&mut self, alpha_q7: u8) {
        self.derivative_alpha_q7 = alpha_q7.min(127);
    }

    /// Creates a bank pinned directly to a fixed [`TelemetryVerbosityMode`].
    #[inline]
    #[must_use]
    pub const fn with_mode(mode: TelemetryVerbosityMode) -> Self {
        Self::new(TelemetryCapturePolicy::Pinned(mode))
    }

    /// Returns the currently active [`TelemetryVerbosityMode`].
    #[inline(always)]
    #[must_use]
    pub const fn mode(&self) -> TelemetryVerbosityMode {
        self.active_mode
    }

    /// Returns the configured [`TelemetryCapturePolicy`].
    #[inline(always)]
    #[must_use]
    pub const fn policy(&self) -> TelemetryCapturePolicy {
        self.policy
    }

    /// Updates the [`TelemetryCapturePolicy`] and aligns `active_mode` immediately if pinned or base changes.
    #[inline]
    pub fn set_policy(&mut self, policy: TelemetryCapturePolicy) {
        self.policy = policy;
        let next = match policy {
            TelemetryCapturePolicy::Pinned(m) => m,
            TelemetryCapturePolicy::Auto { base, .. } => base,
        };
        if (next as u8) > (self.active_mode as u8) {
            self.spine.mode_expansions = self.spine.mode_expansions.saturating_add(1);
        } else if (next as u8) < (self.active_mode as u8) {
            self.spine.mode_retractions = self.spine.mode_retractions.saturating_add(1);
        }
        self.active_mode = next;
    }

    /// Clears all accumulated counters across `3 + System` domains while retaining policy and dwell config.
    #[inline]
    pub fn reset(&mut self) {
        self.liminal = LiminalTelemetry::default();
        self.stigmergy = StigmergicTelemetry::default();
        self.ego = EgoTelemetry::default();
        self.spine = SystemSpineTelemetry::default();
        self.detailed = DetailedLayerTelemetry::default();
        self.ema_initialized = false;
        self.entropy_ema_q15 = 0;
        self.coherence_ema_q15 = 0;
        self.guard_ema_q15 = 0;
        self.assimilation_drift_q15 = 0;
        self.assimilation_drift_dwell = 0;
        self.sandbox_coherence_improve_dwell = 0;
        self.derivative_alpha_q7 = crate::stigmergy::substrate::DEFAULT_DERIVATIVE_ALPHA_Q7;
        self.attribution_counts = [0; 8];
        self.whole_token_bypass_events = 0;
        self.immediate_tension_damps = 0;
        self.slow_tension_excites = 0;
        self.pframe_chain_evals = 0;
        self.pframe_total_hops = 0;
        self.exterior_sandbox_stages = 0;
        self.unobserved_aperture_zeros = 0;
        self.guard_relaxation_count = 0;
        self.instability_recovery_count = 0;
        self.abstract_token_count = 0;
        self.concrete_token_count = 0;
        self.boundary_churn_count = 0;
        self.logits_skip_count = 0;
    }

    /// Records a `SystemSpine` whole-token stationary bypass event without polluting
    /// stigmergic action saturation counters.
    #[inline(always)]
    pub fn record_whole_token_bypass(&mut self) {
        self.whole_token_bypass_events = self.whole_token_bypass_events.saturating_add(1);
    }

    /// Records abstract vs. concrete token residency and boundary-churn transitions.
    #[inline(always)]
    pub fn record_decode_residency(&mut self, whole_token_abstract: bool, boundary_churn: bool) {
        if whole_token_abstract {
            self.abstract_token_count = self.abstract_token_count.saturating_add(1);
        } else {
            self.concrete_token_count = self.concrete_token_count.saturating_add(1);
        }
        if boundary_churn {
            self.boundary_churn_count = self.boundary_churn_count.saturating_add(1);
        }
    }

    /// Records a skipped host D2H logits observation (`SpineLogitsObserveMode::SkipToDeviceArgmax`).
    #[inline(always)]
    pub fn record_logits_skip(&mut self) {
        self.logits_skip_count = self.logits_skip_count.saturating_add(1);
    }

    /// Records a transient contradiction-guard pin relaxation event.
    #[inline(always)]
    pub fn record_guard_relaxation(&mut self) {
        self.guard_relaxation_count = self.guard_relaxation_count.saturating_add(1);
    }

    /// Records an exit instability or `Lacuna::Undefined` recovery event.
    #[inline(always)]
    pub fn record_instability_recovery(&mut self) {
        self.instability_recovery_count = self.instability_recovery_count.saturating_add(1);
    }

    /// Returns cumulative abstract-bypass decode tokens recorded on this bank.
    #[inline(always)]
    #[must_use]
    pub const fn abstract_token_count(&self) -> u32 {
        self.abstract_token_count
    }

    /// Returns cumulative concrete compute-executed decode tokens recorded on this bank.
    #[inline(always)]
    #[must_use]
    pub const fn concrete_token_count(&self) -> u32 {
        self.concrete_token_count
    }

    /// Returns cumulative decode residency boundary-churn transitions (`abstract <-> concrete`).
    #[inline(always)]
    #[must_use]
    pub const fn boundary_churn_count(&self) -> u32 {
        self.boundary_churn_count
    }

    /// Returns cumulative host D2H logits observation skips.
    #[inline(always)]
    #[must_use]
    pub const fn logits_skip_count(&self) -> u32 {
        self.logits_skip_count
    }

    /// Returns cumulative transient contradiction-guard pin relaxations.
    #[inline(always)]
    #[must_use]
    pub const fn guard_relaxation_count(&self) -> u32 {
        self.guard_relaxation_count
    }

    /// Returns cumulative exit instability / `Lacuna::Undefined` recovery events.
    #[inline(always)]
    #[must_use]
    pub const fn instability_recovery_count(&self) -> u32 {
        self.instability_recovery_count
    }

    /// Records lightweight `2 Primitives x 3 Drivers` structural attribution, P-Frame chain hops,
    /// `DistortionTension` events, exterior $O \setminus S$ sandbox staging, and pre-inference bypass hits.
    #[inline]
    pub fn record_two_by_two_step(
        &mut self,
        tag_raw: u8,
        whole_token_bypass: bool,
        lacuna_partition: &LacunaPartition8,
        tension: &DistortionTension,
        pframe_hops: u8,
    ) {
        let idx = usize::from(tag_raw & 0x07);
        self.attribution_counts[idx] = self.attribution_counts[idx].saturating_add(1);
        if whole_token_bypass {
            self.whole_token_bypass_events = self.whole_token_bypass_events.saturating_add(1);
        }
        if lacuna_partition.is_sandbox_eligible() {
            self.exterior_sandbox_stages = self.exterior_sandbox_stages.saturating_add(1);
        }
        if lacuna_partition.is_unobserved_aperture_zero() {
            self.unobserved_aperture_zeros = self.unobserved_aperture_zeros.saturating_add(1);
        }
        if pframe_hops > 0 {
            self.pframe_chain_evals = self.pframe_chain_evals.saturating_add(1);
            self.pframe_total_hops = self
                .pframe_total_hops
                .saturating_add(u64::from(pframe_hops));
        }
        self.immediate_tension_damps = self
            .immediate_tension_damps
            .max(u64::from(tension.immediate_damping_events));
        self.slow_tension_excites = self
            .slow_tension_excites
            .max(u64::from(tension.slow_excitation_events));
    }

    /// Updates `0 fdiv` rolling Q0.15 EMAs ($\alpha = 1/32$) and gauge-relative assimilation drift
    /// against `baseline` in `Standard` or `Detailed` mode, returning the updated drift dwell count.
    ///
    /// In [`TelemetryVerbosityMode::Coarse`], EMA updates and drift dwell accumulation pause
    /// immediately to keep Hot-Path overhead zero.
    #[inline]
    pub fn update_assimilation_ema(
        &mut self,
        entropy: f32,
        coherence: f32,
        guard: f32,
        baseline: &ManifoldProbeBaseline,
        drift_threshold: f32,
    ) -> u16 {
        if matches!(self.active_mode, TelemetryVerbosityMode::Coarse) {
            return self.assimilation_drift_dwell;
        }
        let ent_q15 = f32_unit_to_q15(entropy);
        let coh_q15 = f32_unit_to_q15(coherence);
        let grd_q15 = f32_unit_to_q15(guard);
        let init = self.ema_initialized;
        let prev_coh = self.coherence_ema_q15;

        self.entropy_ema_q15 = step_ema_q15(self.entropy_ema_q15, ent_q15, init);
        self.coherence_ema_q15 = step_ema_q15(self.coherence_ema_q15, coh_q15, init);
        self.guard_ema_q15 = step_ema_q15(self.guard_ema_q15, grd_q15, init);
        self.ema_initialized = true;

        let h0 = f32_unit_to_q15(baseline.baseline_entropy);
        let c0 = f32_unit_to_q15(baseline.baseline_inter_layer_coherence);
        let g0 = f32_unit_to_q15(baseline.baseline_guard_noise_floor);

        let coh_drop = c0.saturating_sub(self.coherence_ema_q15);
        let ent_excess = self.entropy_ema_q15.saturating_sub(h0);
        let guard_excess = self.guard_ema_q15.saturating_sub(g0);
        let drift_raw =
            u32::from(coh_drop) + (u32::from(ent_excess) >> 1) + (u32::from(guard_excess) >> 1);
        self.assimilation_drift_q15 = drift_raw.min(u32::from(EMA_Q15_ONE)) as u16;

        let tau_q15 = f32_unit_to_q15(drift_threshold).max(1);
        let hysteresis_floor = tau_q15.saturating_sub(tau_q15 >> 3);
        let enough_samples = self.spine.continuous_standard_steps >= 16;
        if enough_samples && self.assimilation_drift_q15 >= tau_q15 {
            self.assimilation_drift_dwell = self.assimilation_drift_dwell.saturating_add(1);
        } else if self.assimilation_drift_q15 < hysteresis_floor {
            self.assimilation_drift_dwell = 0;
        }

        if init && (coh_q15 >= c0 || coh_q15 > prev_coh) {
            self.sandbox_coherence_improve_dwell =
                self.sandbox_coherence_improve_dwell.saturating_add(1);
        } else {
            self.sandbox_coherence_improve_dwell = 0;
        }

        self.assimilation_drift_dwell
    }

    /// Returns the latest gauge-relative assimilation drift $D_{\mathrm{assim}} \in [0.0, 1.0]$.
    #[inline(always)]
    #[must_use]
    pub fn assimilation_drift(&self) -> f32 {
        f32::from(self.assimilation_drift_q15) * EMA_Q15_TO_F32
    }

    /// Returns the consecutive assimilation drift dwell steps.
    #[inline(always)]
    #[must_use]
    pub const fn assimilation_drift_dwell(&self) -> u16 {
        self.assimilation_drift_dwell
    }

    /// Returns the consecutive coherence improvement steps observed for sandbox settlement.
    #[inline(always)]
    #[must_use]
    pub const fn sandbox_coherence_improve_dwell(&self) -> u16 {
        self.sandbox_coherence_improve_dwell
    }

    /// Resets the assimilation drift dwell counter (e.g., after sandbox settlement or release).
    #[inline(always)]
    pub const fn reset_assimilation_dwell(&mut self) {
        self.assimilation_drift_dwell = 0;
    }

    /// Resets the sandbox coherence improvement dwell counter upon fresh sandbox nucleation.
    #[inline(always)]
    pub const fn reset_sandbox_improve_dwell(&mut self) {
        self.sandbox_coherence_improve_dwell = 0;
    }

    /// Synchronizes Tier-1 residual experience cache cumulative metrics (`hits`, `stores`,
    /// `evictions`) onto `+ System` telemetry in zero-alloc mode.
    #[inline(always)]
    pub fn sync_experience_metrics(&mut self, hits: u64, stores: u64, evictions: u64) {
        self.spine.experience_cache_hits = hits;
        self.spine.experience_cache_stores = stores;
        self.spine.experience_cache_evictions = evictions;
    }

    /// Evaluates automatic verbosity policy with dwell anti-chattering and returns the updated mode.
    #[inline]
    pub fn evaluate_policy(
        &mut self,
        destabilized: bool,
        quiescent_dwell: u16,
        qos_throttled: bool,
    ) -> TelemetryVerbosityMode {
        let next = match self.policy {
            TelemetryCapturePolicy::Pinned(m) => m,
            TelemetryCapturePolicy::Auto {
                base,
                spike_to_detailed,
                qos_throttle_to_coarse,
            } => {
                if qos_throttle_to_coarse && qos_throttled {
                    TelemetryVerbosityMode::Coarse
                } else if spike_to_detailed && destabilized {
                    TelemetryVerbosityMode::Detailed
                } else if quiescent_dwell >= self.retract_dwell_threshold {
                    base
                } else {
                    self.active_mode
                }
            }
        };
        if (next as u8) > (self.active_mode as u8) {
            self.spine.mode_expansions = self.spine.mode_expansions.saturating_add(1);
        } else if (next as u8) < (self.active_mode as u8) {
            self.spine.mode_retractions = self.spine.mode_retractions.saturating_add(1);
        }
        self.active_mode = next;
        next
    }

    /// Records a single step across `3 + System` telemetry domains according to `self.active_mode`.
    ///
    /// In [`TelemetryVerbosityMode::Coarse`], returns immediately after incrementing `total_steps`
    /// (and `qos_throttle_events` when throttled) so Hot-Path L1D stays clean.
    #[inline]
    pub fn record_step(&mut self, input: StepTelemetryInput<'_>) {
        let StepTelemetryInput {
            decision,
            horizon,
            delta,
            overlap,
            ego,
            entropy_gate_mode,
            slid_in_bits,
            slid_out_bits,
            escalated,
            qos_throttled,
        } = input;
        self.spine.total_steps = self.spine.total_steps.saturating_add(1);
        if qos_throttled {
            self.spine.qos_throttle_events = self.spine.qos_throttle_events.saturating_add(1);
        }
        if matches!(self.active_mode, TelemetryVerbosityMode::Coarse) {
            return;
        }

        self.spine.continuous_standard_steps =
            self.spine.continuous_standard_steps.saturating_add(1);
        if entropy_gate_mode == 1 {
            self.spine.entropy_gate_bifurcation_events =
                self.spine.entropy_gate_bifurcation_events.saturating_add(1);
        } else if entropy_gate_mode >= 2 {
            self.spine.entropy_gate_repulsion_events =
                self.spine.entropy_gate_repulsion_events.saturating_add(1);
        }

        // Layer 0: Liminal
        let reg_idx = usize::from(horizon.regime_code.min(2));
        self.liminal.regime_counts[reg_idx] = self.liminal.regime_counts[reg_idx].saturating_add(1);
        if delta.chattering_channel_mask != 0 {
            self.liminal.chatter_events = self.liminal.chatter_events.saturating_add(1);
        }
        if delta.stationary_shortcut_ready != 0 {
            self.liminal.stationary_shortcut_hits =
                self.liminal.stationary_shortcut_hits.saturating_add(1);
        }
        if slid_in_bits != 0 {
            self.liminal.slide_in_count = self.liminal.slide_in_count.saturating_add(1);
        }
        if slid_out_bits != 0 {
            self.liminal.slide_out_count = self.liminal.slide_out_count.saturating_add(1);
        }

        // Layer 1: Stigmergic
        let dec_idx = (decision as usize).min(6);
        self.stigmergy.decision_counts[dec_idx] =
            self.stigmergy.decision_counts[dec_idx].saturating_add(1);
        if overlap.discovery_triggered != 0 {
            self.stigmergy.overlap_discovery_hits =
                self.stigmergy.overlap_discovery_hits.saturating_add(1);
        }
        if overlap.overlap_core_mask != 0 {
            self.stigmergy.quorum_core_hits = self.stigmergy.quorum_core_hits.saturating_add(1);
        }

        // Layer 2: Ego & cross-layer distortion accounting
        if ego.last_intervened_mask != 0 {
            self.ego.intervention_count = self.ego.intervention_count.saturating_add(1);
            if (ego.last_intervened_mask
                & (EGO_TRIGGER_INTERNAL_CONFLICT | EGO_TRIGGER_BASIN_CAPTURE))
                != 0
            {
                self.ego.tiebreak_count = self.ego.tiebreak_count.saturating_add(1);
            }
            let unaligned = matches!(decision, StigmergicActionDecision::RepulsionEscape)
                || (horizon.is_internal_conflict() && overlap.overlap_core_mask == 0);
            if unaligned {
                self.ego.unaligned_intervention_count =
                    self.ego.unaligned_intervention_count.saturating_add(1);
            }
        } else {
            let stalled = (horizon.is_internal_conflict() || horizon.is_liminal())
                && delta.stationary_shortcut_ready == 0
                && matches!(
                    decision,
                    StigmergicActionDecision::FullOrchestration
                        | StigmergicActionDecision::CrystallizeMacroBasin
                        | StigmergicActionDecision::RepulsionEscape
                );
            if stalled {
                self.liminal.passive_stall_events =
                    self.liminal.passive_stall_events.saturating_add(1);
            }
        }
        if escalated {
            self.ego.escalation_swaps = self.ego.escalation_swaps.saturating_add(1);
        }

        if matches!(self.active_mode, TelemetryVerbosityMode::Detailed) {
            self.record_detailed_breakdown(delta, overlap, ego);
        }
    }

    #[inline]
    fn record_detailed_breakdown(
        &mut self,
        delta: &LiminalBoundaryDelta,
        overlap: &OverlapDiscoveryReport,
        ego: &LiminalEgoState,
    ) {
        self.detailed.detailed_captured_steps =
            self.detailed.detailed_captured_steps.saturating_add(1);

        let chatter = delta.chattering_channel_mask;
        for (ch, cnt) in self
            .detailed
            .liminal_channel_chatter_counts
            .iter_mut()
            .enumerate()
            .take(NUM_LIMINAL_THRESHOLD_CHANNELS)
        {
            if ((chatter >> ch) & 1) != 0 {
                *cnt = cnt.saturating_add(1);
            }
        }

        let core_mask = overlap.overlap_core_mask;
        for (g, cnt) in self
            .detailed
            .stigmergic_granule_overlap_hits
            .iter_mut()
            .enumerate()
            .take(NUM_STIGMERGIC_GRANULES)
        {
            if ((core_mask >> g) & 1) != 0 {
                *cnt = cnt.saturating_add(1);
            }
        }

        const EGO_FLAGS: [u8; 5] = [
            EGO_TRIGGER_BASIN_CAPTURE,
            EGO_TRIGGER_INTERNAL_CONFLICT,
            EGO_TRIGGER_LIMINAL_EXTERIOR,
            EGO_TRIGGER_BOUNDARY_CHATTER,
            EGO_TRIGGER_BOUNDARY_STAGNATION,
        ];
        for (idx, &flag) in EGO_FLAGS.iter().enumerate() {
            if (ego.last_intervened_mask & flag) != 0 {
                self.detailed.ego_trigger_counts[idx] =
                    self.detailed.ego_trigger_counts[idx].saturating_add(1);
            }
        }
    }

    /// Accumulates counters from a pulled [`CoreTelemetrySnapshot`] into this bank, enabling
    /// multi-pass or multi-probe aggregation using the exact same core distortion/ratio laws.
    #[inline]
    pub fn accumulate_snapshot(&mut self, snap: &CoreTelemetrySnapshot) {
        for (dst, &src) in self
            .liminal
            .regime_counts
            .iter_mut()
            .zip(snap.liminal.regime_counts.iter())
        {
            *dst = dst.saturating_add(src);
        }
        self.liminal.chatter_events = self
            .liminal
            .chatter_events
            .saturating_add(snap.liminal.chatter_events);
        self.liminal.stationary_shortcut_hits = self
            .liminal
            .stationary_shortcut_hits
            .saturating_add(snap.liminal.stationary_shortcut_hits);
        self.liminal.slide_in_count = self
            .liminal
            .slide_in_count
            .saturating_add(snap.liminal.slide_in_count);
        self.liminal.slide_out_count = self
            .liminal
            .slide_out_count
            .saturating_add(snap.liminal.slide_out_count);
        self.liminal.passive_stall_events = self
            .liminal
            .passive_stall_events
            .saturating_add(snap.liminal.passive_stall_events);

        for (dst, &src) in self
            .stigmergy
            .decision_counts
            .iter_mut()
            .zip(snap.stigmergy.decision_counts.iter())
        {
            *dst = dst.saturating_add(src);
        }
        self.stigmergy.overlap_discovery_hits = self
            .stigmergy
            .overlap_discovery_hits
            .saturating_add(snap.stigmergy.overlap_discovery_hits);
        self.stigmergy.quorum_core_hits = self
            .stigmergy
            .quorum_core_hits
            .saturating_add(snap.stigmergy.quorum_core_hits);

        self.ego.intervention_count = self
            .ego
            .intervention_count
            .saturating_add(snap.ego.intervention_count);
        self.ego.tiebreak_count = self
            .ego
            .tiebreak_count
            .saturating_add(snap.ego.tiebreak_count);
        self.ego.escalation_swaps = self
            .ego
            .escalation_swaps
            .saturating_add(snap.ego.escalation_swaps);
        self.ego.unaligned_intervention_count = self
            .ego
            .unaligned_intervention_count
            .saturating_add(snap.ego.unaligned_intervention_count);

        self.spine.total_steps = self
            .spine
            .total_steps
            .saturating_add(snap.spine.total_steps);
        self.spine.continuous_standard_steps = self
            .spine
            .continuous_standard_steps
            .saturating_add(snap.spine.continuous_standard_steps);
        self.spine.qos_throttle_events = self
            .spine
            .qos_throttle_events
            .saturating_add(snap.spine.qos_throttle_events);
        self.spine.entropy_gate_bifurcation_events = self
            .spine
            .entropy_gate_bifurcation_events
            .saturating_add(snap.spine.entropy_gate_bifurcation_events);
        self.spine.entropy_gate_repulsion_events = self
            .spine
            .entropy_gate_repulsion_events
            .saturating_add(snap.spine.entropy_gate_repulsion_events);
        self.spine.mode_expansions = self
            .spine
            .mode_expansions
            .saturating_add(snap.spine.mode_expansions);
        self.spine.mode_retractions = self
            .spine
            .mode_retractions
            .saturating_add(snap.spine.mode_retractions);
        self.spine.experience_cache_hits = snap.spine.experience_cache_hits;
        self.spine.experience_cache_stores = snap.spine.experience_cache_stores;
        self.spine.experience_cache_evictions = snap.spine.experience_cache_evictions;
        for (dst, &src) in self
            .attribution_counts
            .iter_mut()
            .zip(snap.attribution_counts.iter())
        {
            *dst = dst.saturating_add(src);
        }
        self.whole_token_bypass_events = self
            .whole_token_bypass_events
            .saturating_add(snap.whole_token_bypass_events);
        self.immediate_tension_damps = self
            .immediate_tension_damps
            .saturating_add(snap.immediate_tension_damps);
        self.slow_tension_excites = self
            .slow_tension_excites
            .saturating_add(snap.slow_tension_excites);
        self.pframe_chain_evals = self
            .pframe_chain_evals
            .saturating_add(snap.pframe_chain_evals);
        self.pframe_total_hops = self
            .pframe_total_hops
            .saturating_add(snap.pframe_total_hops);
        self.exterior_sandbox_stages = self
            .exterior_sandbox_stages
            .saturating_add(snap.exterior_sandbox_stages);
        self.unobserved_aperture_zeros = self
            .unobserved_aperture_zeros
            .saturating_add(snap.unobserved_aperture_zeros);
        self.guard_relaxation_count = self
            .guard_relaxation_count
            .saturating_add(u32::try_from(snap.guard_relaxation_count).unwrap_or(u32::MAX));
        self.instability_recovery_count = self
            .instability_recovery_count
            .saturating_add(u32::try_from(snap.instability_recovery_count).unwrap_or(u32::MAX));
        self.abstract_token_count = self
            .abstract_token_count
            .saturating_add(u32::try_from(snap.abstract_token_count).unwrap_or(u32::MAX));
        self.concrete_token_count = self
            .concrete_token_count
            .saturating_add(u32::try_from(snap.concrete_token_count).unwrap_or(u32::MAX));
        self.boundary_churn_count = self
            .boundary_churn_count
            .saturating_add(u32::try_from(snap.boundary_churn_count).unwrap_or(u32::MAX));
        self.logits_skip_count = self
            .logits_skip_count
            .saturating_add(u32::try_from(snap.logits_skip_count).unwrap_or(u32::MAX));
    }

    /// Computes a Lazy-Pull cross-layer [`LayerDistortionReport`] diagnosing whether the core
    /// is aligned, passively saturated/stalled, or actively overdriven by unaligned Ego bias.
    #[inline]
    #[must_use]
    pub fn compute_distortion_report(&self) -> LayerDistortionReport {
        if self.spine.continuous_standard_steps == 0 {
            return LayerDistortionReport::default();
        }
        let denom = self.spine.continuous_standard_steps as f32;
        let inv_denom = 1.0_f32 / denom;
        let skip_idx = StigmergicActionDecision::InertialLayerSkip as usize;
        let mut max_decision = 0_u64;
        let mut hhi = 0.0_f32;
        for (idx, &cnt) in self.stigmergy.decision_counts.iter().enumerate() {
            // Intentional stationary shortcuts (`InertialLayerSkip` / `SelfRef` passthrough)
            // are healthy low-entropy skips, not pathological stigmergic over-integration.
            if idx == skip_idx {
                continue;
            }
            if cnt > max_decision {
                max_decision = cnt;
            }
            let p = (cnt as f32) * inv_denom;
            hhi += p * p;
        }
        let action_saturation_ratio =
            branchless_clamp_f32((max_decision as f32) * inv_denom, 0.0, 1.0);
        let concentration = branchless_clamp_f32((hhi - (1.0 / 7.0)) / (6.0 / 7.0), 0.0, 1.0);
        let sat_excess = branchless_clamp_f32((action_saturation_ratio - 0.45) / 0.55, 0.0, 1.0);
        let sat_load = 0.5 * concentration + 0.5 * sat_excess;
        let passive_stall_rate = branchless_clamp_f32(
            (self.liminal.passive_stall_events as f32) * inv_denom,
            0.0,
            1.0,
        );
        let active_mismatch_rate = branchless_clamp_f32(
            (self.ego.unaligned_intervention_count as f32) * inv_denom,
            0.0,
            1.0,
        );
        let ego_ratio =
            branchless_clamp_f32((self.ego.intervention_count as f32) * inv_denom, 0.0, 1.0);

        // Preserve total saturation load without diluting it by `ego_ratio`; attribute
        // whether saturation persists passively (low Ego) or despite/from Ego (active).
        let passive_distortion_ratio = branchless_clamp_f32(
            if passive_stall_rate > sat_load * (1.0 - 0.5 * ego_ratio) {
                passive_stall_rate
            } else {
                sat_load * (1.0 - 0.5 * ego_ratio)
            },
            0.0,
            1.0,
        );
        let active_distortion_ratio = branchless_clamp_f32(
            if active_mismatch_rate > sat_load * ego_ratio {
                active_mismatch_rate
            } else {
                sat_load * ego_ratio
            },
            0.0,
            1.0,
        );
        let mismatch_ratio = branchless_clamp_f32(
            ((self.liminal.passive_stall_events + self.ego.unaligned_intervention_count) as f32)
                * inv_denom,
            0.0,
            1.0,
        );

        let enough_samples = self.spine.continuous_standard_steps >= 16;
        let is_distorted = enough_samples
            && (sat_load >= 0.40 || passive_stall_rate >= 0.35 || active_mismatch_rate >= 0.35);
        let ego_cooldown_recommended = is_distorted
            && ((active_distortion_ratio >= passive_distortion_ratio && ego_ratio >= 0.40)
                || active_mismatch_rate >= 0.35);
        let ego_wakeup_recommended = is_distorted && !ego_cooldown_recommended;
        let kind = if ego_cooldown_recommended {
            LayerDistortionKind::ActiveOverdrive
        } else if ego_wakeup_recommended {
            LayerDistortionKind::PassiveSaturation
        } else {
            LayerDistortionKind::Aligned
        };

        LayerDistortionReport {
            action_saturation_ratio,
            passive_distortion_ratio,
            active_distortion_ratio,
            mismatch_ratio,
            kind,
            ego_wakeup_recommended,
            ego_cooldown_recommended,
        }
    }

    /// Computes a Lazy-Pull [`CoreTelemetrySnapshot`] combining `3 + System` counters, derived ratios,
    /// and live `stack` topological state without performing allocations.
    #[inline]
    #[must_use]
    pub fn snapshot(
        &self,
        stack: &CoreStateStack,
        horizon: &EpistemicHorizonReading,
        active_anchors: u8,
        spawned_anchors: u8,
        active_horizon_slots_mask: u32,
    ) -> CoreTelemetrySnapshot {
        let denom = self.spine.continuous_standard_steps.max(1) as f32;
        let has_samples = self.spine.continuous_standard_steps > 0;
        let mut decision_ratios_pct = [0.0_f32; 7];
        if has_samples {
            for (dst, &cnt) in decision_ratios_pct
                .iter_mut()
                .zip(self.stigmergy.decision_counts.iter())
            {
                *dst = (cnt as f32 / denom) * 100.0;
            }
        }
        let excursion_cnt =
            self.liminal.regime_counts[1].saturating_add(self.liminal.regime_counts[2]);
        let topology = LiminalTopologyTelemetrySnapshot::capture(
            &stack.liminal,
            horizon,
            active_anchors,
            spawned_anchors,
        );
        let distortion = self.compute_distortion_report();
        let detailed = if matches!(self.active_mode, TelemetryVerbosityMode::Detailed)
            || self.detailed.detailed_captured_steps > 0
        {
            Some(self.detailed)
        } else {
            None
        };

        CoreTelemetrySnapshot {
            mode: self.active_mode,
            policy: self.policy,
            liminal: self.liminal,
            stigmergy: self.stigmergy,
            ego: self.ego,
            spine: self.spine,
            detailed,
            decision_ratios_pct,
            liminal_excursion_ratio: if has_samples {
                excursion_cnt as f32 / denom
            } else {
                0.0
            },
            stationary_shortcut_ratio: if has_samples {
                self.liminal.stationary_shortcut_hits as f32 / denom
            } else {
                0.0
            },
            ego_intervention_ratio: if has_samples {
                self.ego.intervention_count as f32 / denom
            } else {
                0.0
            },
            topology,
            distortion,
            active_horizon_slots_mask,
            entropy_ema: f32::from(self.entropy_ema_q15) * EMA_Q15_TO_F32,
            coherence_ema: f32::from(self.coherence_ema_q15) * EMA_Q15_TO_F32,
            guard_ema: f32::from(self.guard_ema_q15) * EMA_Q15_TO_F32,
            assimilation_drift: f32::from(self.assimilation_drift_q15) * EMA_Q15_TO_F32,
            assimilation_drift_dwell: self.assimilation_drift_dwell,
            attribution_counts: self.attribution_counts,
            whole_token_bypass_events: self.whole_token_bypass_events,
            immediate_tension_damps: self.immediate_tension_damps,
            slow_tension_excites: self.slow_tension_excites,
            pframe_chain_evals: self.pframe_chain_evals,
            pframe_total_hops: self.pframe_total_hops,
            exterior_sandbox_stages: self.exterior_sandbox_stages,
            unobserved_aperture_zeros: self.unobserved_aperture_zeros,
            abstract_token_count: u64::from(self.abstract_token_count),
            concrete_token_count: u64::from(self.concrete_token_count),
            boundary_churn_count: u64::from(self.boundary_churn_count),
            logits_skip_count: u64::from(self.logits_skip_count),
            guard_relaxation_count: u64::from(self.guard_relaxation_count),
            instability_recovery_count: u64::from(self.instability_recovery_count),
        }
    }

    /// Evaluates bounded Slow-Loop autonomous threshold adjustments at an epoch boundary.
    ///
    /// Skips adjustment (`applied = false`) when `active_mode == Coarse` or fewer than
    /// `min_samples.max(1)` continuous `Standard`/`Detailed` steps have accumulated so denominator
    /// skew never destabilizes thresholds.
    #[inline]
    pub fn evaluate_slow_loop_epoch(
        &mut self,
        base: &StigmergicThresholds,
        min_samples: u64,
    ) -> SlowLoopTelemetryAdjustment {
        let distortion = self.compute_distortion_report();
        let req = min_samples.max(1);
        if matches!(self.active_mode, TelemetryVerbosityMode::Coarse)
            || self.spine.continuous_standard_steps < req
        {
            return SlowLoopTelemetryAdjustment {
                applied: false,
                sampled_steps: self.spine.continuous_standard_steps,
                tuned_thresholds: *base,
                suggested_slide_out_dwell_delta: 0,
                suggested_repulse_bias_q7: 0,
                suggested_derivative_alpha_q7: self.derivative_alpha_q7,
                distortion,
            };
        }

        let denom = self.spine.continuous_standard_steps as f32;
        let chatter_ratio = self.liminal.chatter_events as f32 / denom;
        let excursion_ratio = (self.liminal.regime_counts[1]
            .saturating_add(self.liminal.regime_counts[2])) as f32
            / denom;
        let skip_ratio = self.stigmergy.decision_counts[1] as f32 / denom;
        let rank1_ratio = self.stigmergy.decision_counts[4] as f32 / denom;
        let repulse_ratio = self.stigmergy.decision_counts[6] as f32 / denom;

        // Layer 0 chatter -> bounded dwell hysteresis extension (prevents slot flapping).
        let dwell_delta: i8 = if chatter_ratio > 0.50 {
            2
        } else if chatter_ratio > 0.25 {
            1
        } else {
            0
        };

        // Layer 1 decision balance -> homeostatic +/-15% adjustments keeping the monotone ladder intact.
        let density_scale = branchless_clamp_f32(
            1.0 - (repulse_ratio - 0.35) * 0.25 + (excursion_ratio - 0.40) * 0.10,
            0.88,
            1.12,
        );
        let rank1_scale = branchless_clamp_f32(
            1.0 + (rank1_ratio - 0.40) * 0.50 - (skip_ratio - 0.25) * 0.20,
            0.85,
            1.25,
        );
        let skip_entropy_scale = branchless_clamp_f32(
            1.0 + (rank1_ratio - 0.55) * 0.20 - (skip_ratio - 0.10) * 0.35,
            0.85,
            1.08,
        );
        let contradict_scale = branchless_clamp_f32(
            1.0 + (repulse_ratio - 0.45) * 0.25 - (excursion_ratio - 0.30) * 0.15,
            0.88,
            1.12,
        );
        let next_crystallize = branchless_clamp_f32(
            base.crystallize_density_thresh * branchless_clamp_f32(rank1_scale, 0.90, 1.25),
            base.subhead_route_density_thresh,
            4.50,
        );

        let next_express = branchless_clamp_f32(
            base.express_steer_density_thresh * density_scale,
            0.05,
            next_crystallize,
        );
        let next_subhead = branchless_clamp_f32(
            base.subhead_route_density_thresh * density_scale,
            next_express,
            next_crystallize,
        );
        let next_rank1 = branchless_clamp_f32(
            base.rank1_inject_density_thresh * rank1_scale,
            next_subhead,
            next_crystallize,
        );
        let next_skip_entropy = branchless_clamp_f32(
            base.inertial_skip_entropy_thresh * skip_entropy_scale,
            0.05,
            0.55,
        );

        let tuned_thresholds = StigmergicThresholds {
            inertial_skip_entropy_thresh: next_skip_entropy,
            express_steer_density_thresh: next_express,
            subhead_route_density_thresh: next_subhead,
            rank1_inject_density_thresh: next_rank1,
            crystallize_density_thresh: next_crystallize,
            contradiction_coherence_thresh: branchless_clamp_f32(
                base.contradiction_coherence_thresh * contradict_scale,
                0.05,
                0.95,
            ),
            ..*base
        };

        // Layer 2 / System bounded repulsion bias in [-16, 16].
        let raw_bias = ((excursion_ratio - 0.35) * 32.0).clamp(-16.0, 16.0) as i8;

        // Slow-Loop relative derivative/integral balance alpha (Q0.7, read-only on Hot Path):
        // PassiveSaturation (over-integration) -> push toward derivative (+12);
        // ActiveOverdrive (over-differentiation) -> pull back toward integral (-12);
        // Aligned -> gently relax toward nominal default.
        let next_alpha = match distortion.kind {
            LayerDistortionKind::PassiveSaturation => self
                .derivative_alpha_q7
                .saturating_add(12)
                .min(crate::stigmergy::substrate::MAX_DERIVATIVE_ALPHA_Q7),
            LayerDistortionKind::ActiveOverdrive => self
                .derivative_alpha_q7
                .saturating_sub(12)
                .max(crate::stigmergy::substrate::MIN_DERIVATIVE_ALPHA_Q7),
            LayerDistortionKind::Aligned => {
                let nominal = crate::stigmergy::substrate::DEFAULT_DERIVATIVE_ALPHA_Q7;
                if self.derivative_alpha_q7 > nominal {
                    self.derivative_alpha_q7.saturating_sub(2).max(nominal)
                } else {
                    self.derivative_alpha_q7.saturating_add(2).min(nominal)
                }
            }
        };
        self.derivative_alpha_q7 = next_alpha;

        self.spine.slow_loop_epochs = self.spine.slow_loop_epochs.saturating_add(1);
        self.spine.slow_loop_dwell_delta = dwell_delta;
        self.spine.slow_loop_repulse_bias_q7 = raw_bias;

        SlowLoopTelemetryAdjustment {
            applied: true,
            sampled_steps: self.spine.continuous_standard_steps,
            tuned_thresholds,
            suggested_slide_out_dwell_delta: dwell_delta,
            suggested_repulse_bias_q7: raw_bias,
            suggested_derivative_alpha_q7: next_alpha,
            distortion,
        }
    }
}

/// Compile-time check pinning [`EpistemicRegime`] indices to `[0, 1, 2]`.
const _: () = {
    assert!(EpistemicRegime::KnownCoherent as usize == 0);
    assert!(EpistemicRegime::InternalConflict as usize == 1);
    assert!(EpistemicRegime::Liminal as usize == 2);
};

// SAFETY: `LiminalTopologyTelemetrySnapshot` is `#[repr(C, align(16))]` of 32 bytes with zero padding traps.
unsafe impl Pod for LiminalTopologyTelemetrySnapshot {}
