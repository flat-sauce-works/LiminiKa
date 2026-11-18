// name: src/core/src/traits/gateway/telemetry_bank.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core-First Lazy-Pull `3 + System` telemetry bank, adaptive verbosity modes
//! (`Coarse` / `Standard` / `Detailed`), and bounded Slow-Loop self-tuning.
//!
//! Keeps hot-path compute structs (`AutonomousCoreLayerStack` 128 B and `ActionHubEngine`
//! 2,816 B) untouched while tracking separate `Layer 0` (`Liminal`), `Layer 1` (`Stigmergic`),
//! `Layer 2` (`Ego`), and vertical `+ System` execution spine telemetry in `#![no_std]` zero-alloc
//! memory. Ratios, percentages, and bounded Slow-Loop threshold adjustments are computed lazily
//! when pulled or at coarse epoch boundaries.

use super::core_layers::{
    AutonomousCoreLayerStack, LiminalBoundaryDelta, LiminalEgoField,
    StigmergicVennSerendipityReport,
};
use super::epistemic::{
    EpistemicHorizonReading, EpistemicRegime, LiminalTopologyTelemetrySnapshot,
};
use crate::action_hub::{StigmergicActionDecision, StigmergicThresholds};
use crate::traits::primitives::{
    branchless_clamp_f32, EGO_TRIGGER_BASIN_CAPTURE, EGO_TRIGGER_BOUNDARY_CHATTER,
    EGO_TRIGGER_BOUNDARY_STAGNATION, EGO_TRIGGER_INTERNAL_CONFLICT, EGO_TRIGGER_LIMINAL_EXTERIOR,
    NUM_LIMINAL_APERTURE_CHANNELS,
};
use crate::traits::telemetry::NUM_STIGMERGIC_GRANULES;

/// Granularity tier controlling how much telemetry is captured on each step.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TelemetryVerbosityMode {
    /// Minimal-overhead tier for game-concurrent operation: only increments `SystemSpineTelemetry::total_steps`
    /// and reads live compute state lazily on Pull.
    Coarse = 0,
    /// Default tier capturing continuous `3 + System` summary counters (`Standard` + `Detailed` share
    /// `continuous_standard_steps` so Slow-Loop denominators remain consistent across promotions).
    #[default]
    Standard = 1,
    /// Full research/diagnostic tier sliding in [`DetailedLayerTelemetry`] per-channel, per-granule,
    /// and per-trigger counters.
    Detailed = 2,
}

/// Capture policy deciding whether verbosity stays manually pinned or slides dynamically between tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetryCapturePolicy {
    /// Fixes telemetry to a single mode for reproducible benchmarking and ablations.
    Pinned(TelemetryVerbosityMode),
    /// Adapts telemetry granularity dynamically while preserving dwell anti-chattering.
    Auto {
        /// Nominal baseline tier (`Coarse` or `Standard`).
        base: TelemetryVerbosityMode,
        /// Promotes temporarily to [`TelemetryVerbosityMode::Detailed`] during destabilization.
        spike_to_detailed: bool,
        /// Drops to [`TelemetryVerbosityMode::Coarse`] when `System` QoS throttling is active.
        qos_throttle_to_coarse: bool,
    },
}

impl Default for TelemetryCapturePolicy {
    #[inline(always)]
    fn default() -> Self {
        Self::Auto {
            base: TelemetryVerbosityMode::Standard,
            spike_to_detailed: true,
            qos_throttle_to_coarse: true,
        }
    }
}

/// Layer 0 (`Liminal`) telemetry domain: cognitive boundaries, chatter, and slide-in aperture activity.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LiminalLayerTelemetry {
    /// Step counts per [`EpistemicRegime`] (`[0]=KnownCoherent`, `[1]=InternalConflict`, `[2]=Liminal`).
    pub regime_counts: [u64; 3],
    /// Steps observing boundary chatter (`chattering_channel_mask != 0`).
    pub chatter_events: u64,
    /// Steps eligible for quiescent Layer 0 stationary shortcut (`stationary_shortcut_ready != 0`).
    pub stationary_shortcut_hits: u64,
    /// Dynamic slide-in transitions (`slid_in_bits != 0`).
    pub slide_in_count: u64,
    /// Dynamic slide-out retractions (`slid_out_bits != 0`).
    pub slide_out_count: u64,
    /// Steps stuck in conflict or liminal drift without Ego intervention (`last_intervened_mask == 0`).
    pub passive_stall_events: u64,
}

/// Layer 1 (`Stigmergic`) telemetry domain: 7-regime `ActionHub` decisions, Venn overlap, and quorums.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StigmergicLayerTelemetry {
    /// Per-regime [`StigmergicActionDecision`] invocation counts (`0..=6`).
    pub decision_counts: [u64; 7],
    /// Steps where Venn serendipity discovered exterior overlap.
    pub venn_serendipity_hits: u64,
    /// Steps where micro-swarm granules maintained non-empty Venn core consensus.
    pub quorum_core_hits: u64,
}

/// Layer 2 (`Ego`) telemetry domain: top-down interventions, tie-breaks, and cold-path escalations.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EgoLayerTelemetry {
    /// Steps where Layer 2 Ego actively intervened (`last_intervened_mask != 0`).
    pub intervention_count: u64,
    /// Steps where Ego resolved boundary conflict or basin capture.
    pub tiebreak_count: u64,
    /// Steps where cold-path escalation swapped external phases into a working lane.
    pub escalation_swaps: u64,
    /// Steps where Ego intervened yet ended in repulsion or unaligned conflict without Venn quorum.
    pub unaligned_intervention_count: u64,
}

/// Classification of cross-layer geometric/control distortion relative to Layer 2 Ego alignment.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayerDistortionKind {
    /// Core layers and ActionHub remain balanced and aligned (`0`).
    #[default]
    Aligned = 0,
    /// Passive saturation or drift without sufficient Ego corrective intervention (`1`).
    PassiveSaturation = 1,
    /// Excessive or unaligned Ego overdrive causing saturation or conflict mismatch (`2`).
    ActiveOverdrive = 2,
}

/// Lazy-computed diagnostic summary of cross-layer distortion (`Distortion`) used to trigger
/// corrective Ego awakening on passive stalls or Ego cooldown on active overdrive.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LayerDistortionReport {
    /// Largest decision share among the 7 `ActionHub` regimes (`0.0..=1.0`).
    pub action_saturation_ratio: f32,
    /// Passive distortion score combining unguided stalls and single-mode saturation (`0.0..=1.0`).
    pub passive_distortion_ratio: f32,
    /// Active overdrive distortion score combining unaligned Ego steps and saturated Ego bias (`0.0..=1.0`).
    pub active_distortion_ratio: f32,
    /// Overall cross-layer mismatch rate (`0.0..=1.0`).
    pub mismatch_ratio: f32,
    /// Dominant distortion state.
    pub kind: LayerDistortionKind,
    /// `true` when passive distortion warrants waking/boosting Layer 2 Ego.
    pub ego_wakeup_recommended: bool,
    /// `true` when active distortion warrants cooling down Layer 2 Ego thrust/defiance.
    pub ego_cooldown_recommended: bool,
}

/// Vertical `+ System` execution spine telemetry: cross-layer EDBC, QoS, mode switches, and Slow-Loop state.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SystemSpineTelemetry {
    /// Monotonic total steps observed (incremented across all modes, including `Coarse`).
    pub total_steps: u64,
    /// Continuous denominator of steps recorded in `Standard` or `Detailed` (safe for Slow-Loop).
    pub continuous_standard_steps: u64,
    /// Steps experiencing `System` QoS throttling.
    pub qos_throttle_events: u64,
    /// Steps in EDBC bifurcation branch mode (`active_branch_mode == 1`).
    pub edbc_bifurcation_events: u64,
    /// Steps in EDBC repulsion branch mode (`active_branch_mode >= 2`).
    pub edbc_repulsion_events: u64,
    /// Automatic or manual upward verbosity transitions.
    pub mode_promotions: u64,
    /// Automatic or manual downward verbosity transitions.
    pub mode_demotions: u64,
    /// Number of Slow-Loop epoch evaluations performed.
    pub slow_loop_epochs: u64,
    /// Number of auxiliary residual spillover cache hits (`RamPhaseKvCache` recall).
    pub experience_cache_hits: u64,
    /// Number of verified auxiliary residual experiences promoted into RAM cache.
    pub experience_cache_stores: u64,
    /// Number of stale or contradicted cache entries evicted below pheromone floor.
    pub experience_cache_evictions: u64,
    /// Latest bounded Slow-Loop repulse bias adjustment in Q0.7 (`-16..=16`).
    pub slow_loop_repulse_bias_q7: i8,
    /// Latest bounded Slow-Loop `min_slide_out_dwell` delta (`-1..=2`).
    pub slow_loop_dwell_delta: i8,
    /// Explicit C alignment padding.
    pub reserved: [u8; 6],
}

/// Optional/slide-in fine-grained telemetry captured exclusively while active mode is [`TelemetryVerbosityMode::Detailed`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DetailedLayerTelemetry {
    /// Per-channel Layer 0 boundary chatter counts (`0..8`).
    pub liminal_channel_chatter_counts: [u64; NUM_LIMINAL_APERTURE_CHANNELS],
    /// Per-granule Layer 1 Venn core activation counts (`0..8`).
    pub stigmergic_granule_venn_hits: [u64; NUM_STIGMERGIC_GRANULES],
    /// Per-flag Layer 2 Ego trigger breakdown:
    /// `[0]=BasinCapture, [1]=InternalConflict, [2]=LiminalExterior, [3]=BoundaryChatter, [4]=BoundaryStagnation`.
    pub ego_trigger_counts: [u64; 5],
    /// Steps captured while [`TelemetryVerbosityMode::Detailed`] was slid in.
    pub detailed_captured_steps: u64,
}

/// Result returned by Slow-Loop epoch evaluation over continuous `Standard`/`Detailed` counters.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SlowLoopTelemetryAdjustment {
    /// `true` when enough continuous samples existed and active mode was not `Coarse`.
    pub applied: bool,
    /// Continuous `Standard`/`Detailed` steps evaluated for this adjustment.
    pub sampled_steps: u64,
    /// Bounded `StigmergicThresholds` calibrated from telemetry rates (clamped within `±15%` of base).
    pub tuned_thresholds: StigmergicThresholds,
    /// Bounded recommendation for `min_slide_out_dwell` offset (`-1..=2`).
    pub suggested_slide_out_dwell_delta: i8,
    /// Bounded recommendation for repulsion bias in Q0.7 (`-16..=16`).
    pub suggested_repulse_bias_q7: i8,
    /// Cross-layer distortion diagnosis evaluated over the epoch samples.
    pub distortion: LayerDistortionReport,
}

/// Lazy-Pull unified telemetry snapshot combining `3 + System` domains, derived ratios, and
/// live [`LiminalTopologyTelemetrySnapshot`] state.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CoreTelemetrySnapshot {
    /// Active telemetry verbosity at snapshot capture time.
    pub mode: TelemetryVerbosityMode,
    /// Configured verbosity policy.
    pub policy: TelemetryCapturePolicy,
    /// Layer 0 (`Liminal`) telemetry counters.
    pub layer0: LiminalLayerTelemetry,
    /// Layer 1 (`Stigmergic`) telemetry counters.
    pub layer1: StigmergicLayerTelemetry,
    /// Layer 2 (`Ego`) telemetry counters.
    pub layer2: EgoLayerTelemetry,
    /// Vertical `+ System` spine counters.
    pub system: SystemSpineTelemetry,
    /// Detailed breakdown if `Detailed` mode captured data or is currently active.
    pub detailed: Option<DetailedLayerTelemetry>,
    /// Lazy-computed `ActionDecision` percentage share (`0.0..=100.0`) over `continuous_standard_steps`.
    pub decision_ratios_pct: [f32; 7],
    /// Lazy-computed share of steps spent in `InternalConflict` or `Liminal` regime (`0.0..=1.0`).
    pub liminal_excursion_ratio: f32,
    /// Lazy-computed share of steps eligible for Layer 0 stationary shortcut (`0.0..=1.0`).
    pub stationary_shortcut_ratio: f32,
    /// Lazy-computed share of steps with active Layer 2 Ego intervention (`0.0..=1.0`).
    pub ego_intervention_ratio: f32,
    /// Compact 32-byte topological snapshot of live Layer 0 boundaries.
    pub topology: LiminalTopologyTelemetrySnapshot,
    /// Cross-layer geometric/control distortion metrics (`Distortion`).
    pub distortion: LayerDistortionReport,
    /// Bitmask of external `LatentHorizonSlot`s currently slid in.
    pub active_horizon_slots_mask: u32,
}

/// Standalone `#![no_std]` zero-alloc `3 + System` telemetry bank stored beside compute engines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreTelemetryBank {
    policy: TelemetryCapturePolicy,
    active_mode: TelemetryVerbosityMode,
    /// Consecutive quiescent steps required before demoting from `Detailed` back to `base`.
    pub retract_dwell_threshold: u16,
    /// Layer 0 (`Liminal`) counters.
    pub layer0: LiminalLayerTelemetry,
    /// Layer 1 (`Stigmergic`) counters.
    pub layer1: StigmergicLayerTelemetry,
    /// Layer 2 (`Ego`) counters.
    pub layer2: EgoLayerTelemetry,
    /// Vertical `+ System` spine counters.
    pub system: SystemSpineTelemetry,
    /// Slide-in detailed counters (untouched unless `active_mode == Detailed`).
    pub detailed: DetailedLayerTelemetry,
}

impl Default for CoreTelemetryBank {
    #[inline]
    fn default() -> Self {
        Self::new(TelemetryCapturePolicy::default())
    }
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
            retract_dwell_threshold: 2,
            layer0: LiminalLayerTelemetry {
                regime_counts: [0; 3],
                chatter_events: 0,
                stationary_shortcut_hits: 0,
                slide_in_count: 0,
                slide_out_count: 0,
                passive_stall_events: 0,
            },
            layer1: StigmergicLayerTelemetry {
                decision_counts: [0; 7],
                venn_serendipity_hits: 0,
                quorum_core_hits: 0,
            },
            layer2: EgoLayerTelemetry {
                intervention_count: 0,
                tiebreak_count: 0,
                escalation_swaps: 0,
                unaligned_intervention_count: 0,
            },
            system: SystemSpineTelemetry {
                total_steps: 0,
                continuous_standard_steps: 0,
                qos_throttle_events: 0,
                edbc_bifurcation_events: 0,
                edbc_repulsion_events: 0,
                mode_promotions: 0,
                mode_demotions: 0,
                slow_loop_epochs: 0,
                experience_cache_hits: 0,
                experience_cache_stores: 0,
                experience_cache_evictions: 0,
                slow_loop_repulse_bias_q7: 0,
                slow_loop_dwell_delta: 0,
                reserved: [0; 6],
            },
            detailed: DetailedLayerTelemetry {
                liminal_channel_chatter_counts: [0; NUM_LIMINAL_APERTURE_CHANNELS],
                stigmergic_granule_venn_hits: [0; NUM_STIGMERGIC_GRANULES],
                ego_trigger_counts: [0; 5],
                detailed_captured_steps: 0,
            },
        }
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
            self.system.mode_promotions = self.system.mode_promotions.saturating_add(1);
        } else if (next as u8) < (self.active_mode as u8) {
            self.system.mode_demotions = self.system.mode_demotions.saturating_add(1);
        }
        self.active_mode = next;
    }

    /// Clears all accumulated counters across `3 + System` domains while retaining policy and dwell config.
    #[inline]
    pub fn reset(&mut self) {
        self.layer0 = LiminalLayerTelemetry::default();
        self.layer1 = StigmergicLayerTelemetry::default();
        self.layer2 = EgoLayerTelemetry::default();
        self.system = SystemSpineTelemetry::default();
        self.detailed = DetailedLayerTelemetry::default();
    }

    /// Synchronizes Tier-1 residual experience cache cumulative metrics (`hits`, `stores`,
    /// `evictions`) onto `+ System` telemetry in zero-alloc mode.
    #[inline(always)]
    pub fn sync_experience_metrics(&mut self, hits: u64, stores: u64, evictions: u64) {
        self.system.experience_cache_hits = hits;
        self.system.experience_cache_stores = stores;
        self.system.experience_cache_evictions = evictions;
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
            self.system.mode_promotions = self.system.mode_promotions.saturating_add(1);
        } else if (next as u8) < (self.active_mode as u8) {
            self.system.mode_demotions = self.system.mode_demotions.saturating_add(1);
        }
        self.active_mode = next;
        next
    }

    /// Records a single step across `3 + System` telemetry domains according to `self.active_mode`.
    ///
    /// In [`TelemetryVerbosityMode::Coarse`], returns immediately after incrementing `total_steps`
    /// (and `qos_throttle_events` when throttled) so Hot-Path L1D stays clean.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn record_step(
        &mut self,
        decision: StigmergicActionDecision,
        horizon: &EpistemicHorizonReading,
        delta: &LiminalBoundaryDelta,
        venn: &StigmergicVennSerendipityReport,
        ego: &LiminalEgoField,
        edbc_branch_mode: u32,
        slid_in_bits: u32,
        slid_out_bits: u32,
        escalated: bool,
        qos_throttled: bool,
    ) {
        self.system.total_steps = self.system.total_steps.saturating_add(1);
        if qos_throttled {
            self.system.qos_throttle_events = self.system.qos_throttle_events.saturating_add(1);
        }
        if matches!(self.active_mode, TelemetryVerbosityMode::Coarse) {
            return;
        }

        self.system.continuous_standard_steps =
            self.system.continuous_standard_steps.saturating_add(1);
        if edbc_branch_mode == 1 {
            self.system.edbc_bifurcation_events =
                self.system.edbc_bifurcation_events.saturating_add(1);
        } else if edbc_branch_mode >= 2 {
            self.system.edbc_repulsion_events = self.system.edbc_repulsion_events.saturating_add(1);
        }

        // Layer 0: Liminal
        let reg_idx = usize::from(horizon.regime_code.min(2));
        self.layer0.regime_counts[reg_idx] = self.layer0.regime_counts[reg_idx].saturating_add(1);
        if delta.chattering_channel_mask != 0 {
            self.layer0.chatter_events = self.layer0.chatter_events.saturating_add(1);
        }
        if delta.stationary_shortcut_ready != 0 {
            self.layer0.stationary_shortcut_hits =
                self.layer0.stationary_shortcut_hits.saturating_add(1);
        }
        if slid_in_bits != 0 {
            self.layer0.slide_in_count = self.layer0.slide_in_count.saturating_add(1);
        }
        if slid_out_bits != 0 {
            self.layer0.slide_out_count = self.layer0.slide_out_count.saturating_add(1);
        }

        // Layer 1: Stigmergic
        let dec_idx = (decision as usize).min(6);
        self.layer1.decision_counts[dec_idx] =
            self.layer1.decision_counts[dec_idx].saturating_add(1);
        if venn.serendipity_triggered != 0 {
            self.layer1.venn_serendipity_hits = self.layer1.venn_serendipity_hits.saturating_add(1);
        }
        if venn.venn_core_mask != 0 {
            self.layer1.quorum_core_hits = self.layer1.quorum_core_hits.saturating_add(1);
        }

        // Layer 2: Ego & cross-layer distortion accounting
        if ego.last_intervened_mask != 0 {
            self.layer2.intervention_count = self.layer2.intervention_count.saturating_add(1);
            if (ego.last_intervened_mask
                & (EGO_TRIGGER_INTERNAL_CONFLICT | EGO_TRIGGER_BASIN_CAPTURE))
                != 0
            {
                self.layer2.tiebreak_count = self.layer2.tiebreak_count.saturating_add(1);
            }
            let unaligned = matches!(decision, StigmergicActionDecision::RepulsionEscape)
                || (horizon.is_internal_conflict() && venn.venn_core_mask == 0);
            if unaligned {
                self.layer2.unaligned_intervention_count =
                    self.layer2.unaligned_intervention_count.saturating_add(1);
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
                self.layer0.passive_stall_events =
                    self.layer0.passive_stall_events.saturating_add(1);
            }
        }
        if escalated {
            self.layer2.escalation_swaps = self.layer2.escalation_swaps.saturating_add(1);
        }

        if matches!(self.active_mode, TelemetryVerbosityMode::Detailed) {
            self.record_detailed_breakdown(delta, venn, ego);
        }
    }

    #[inline]
    fn record_detailed_breakdown(
        &mut self,
        delta: &LiminalBoundaryDelta,
        venn: &StigmergicVennSerendipityReport,
        ego: &LiminalEgoField,
    ) {
        self.detailed.detailed_captured_steps =
            self.detailed.detailed_captured_steps.saturating_add(1);

        let chatter = delta.chattering_channel_mask;
        for (ch, cnt) in self
            .detailed
            .liminal_channel_chatter_counts
            .iter_mut()
            .enumerate()
            .take(NUM_LIMINAL_APERTURE_CHANNELS)
        {
            if ((chatter >> ch) & 1) != 0 {
                *cnt = cnt.saturating_add(1);
            }
        }

        let venn_mask = venn.venn_core_mask;
        for (g, cnt) in self
            .detailed
            .stigmergic_granule_venn_hits
            .iter_mut()
            .enumerate()
            .take(NUM_STIGMERGIC_GRANULES)
        {
            if ((venn_mask >> g) & 1) != 0 {
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
            .layer0
            .regime_counts
            .iter_mut()
            .zip(snap.layer0.regime_counts.iter())
        {
            *dst = dst.saturating_add(src);
        }
        self.layer0.chatter_events = self
            .layer0
            .chatter_events
            .saturating_add(snap.layer0.chatter_events);
        self.layer0.stationary_shortcut_hits = self
            .layer0
            .stationary_shortcut_hits
            .saturating_add(snap.layer0.stationary_shortcut_hits);
        self.layer0.slide_in_count = self
            .layer0
            .slide_in_count
            .saturating_add(snap.layer0.slide_in_count);
        self.layer0.slide_out_count = self
            .layer0
            .slide_out_count
            .saturating_add(snap.layer0.slide_out_count);
        self.layer0.passive_stall_events = self
            .layer0
            .passive_stall_events
            .saturating_add(snap.layer0.passive_stall_events);

        for (dst, &src) in self
            .layer1
            .decision_counts
            .iter_mut()
            .zip(snap.layer1.decision_counts.iter())
        {
            *dst = dst.saturating_add(src);
        }
        self.layer1.venn_serendipity_hits = self
            .layer1
            .venn_serendipity_hits
            .saturating_add(snap.layer1.venn_serendipity_hits);
        self.layer1.quorum_core_hits = self
            .layer1
            .quorum_core_hits
            .saturating_add(snap.layer1.quorum_core_hits);

        self.layer2.intervention_count = self
            .layer2
            .intervention_count
            .saturating_add(snap.layer2.intervention_count);
        self.layer2.tiebreak_count = self
            .layer2
            .tiebreak_count
            .saturating_add(snap.layer2.tiebreak_count);
        self.layer2.escalation_swaps = self
            .layer2
            .escalation_swaps
            .saturating_add(snap.layer2.escalation_swaps);
        self.layer2.unaligned_intervention_count = self
            .layer2
            .unaligned_intervention_count
            .saturating_add(snap.layer2.unaligned_intervention_count);

        self.system.total_steps = self.system.total_steps.saturating_add(snap.system.total_steps);
        self.system.continuous_standard_steps = self
            .system
            .continuous_standard_steps
            .saturating_add(snap.system.continuous_standard_steps);
        self.system.qos_throttle_events = self
            .system
            .qos_throttle_events
            .saturating_add(snap.system.qos_throttle_events);
        self.system.edbc_bifurcation_events = self
            .system
            .edbc_bifurcation_events
            .saturating_add(snap.system.edbc_bifurcation_events);
        self.system.edbc_repulsion_events = self
            .system
            .edbc_repulsion_events
            .saturating_add(snap.system.edbc_repulsion_events);
        self.system.mode_promotions = self
            .system
            .mode_promotions
            .saturating_add(snap.system.mode_promotions);
        self.system.mode_demotions = self
            .system
            .mode_demotions
            .saturating_add(snap.system.mode_demotions);
        self.system.experience_cache_hits = snap.system.experience_cache_hits;
        self.system.experience_cache_stores = snap.system.experience_cache_stores;
        self.system.experience_cache_evictions = snap.system.experience_cache_evictions;
    }

    /// Computes a Lazy-Pull cross-layer [`LayerDistortionReport`] diagnosing whether the core
    /// is aligned, passively saturated/stalled, or actively overdriven by unaligned Ego bias.
    #[inline]
    #[must_use]
    pub fn compute_distortion_report(&self) -> LayerDistortionReport {
        if self.system.continuous_standard_steps == 0 {
            return LayerDistortionReport::default();
        }
        let denom = self.system.continuous_standard_steps as f32;
        let inv_denom = 1.0_f32 / denom;
        let mut max_decision = 0_u64;
        let mut hhi = 0.0_f32;
        for &cnt in &self.layer1.decision_counts {
            if cnt > max_decision {
                max_decision = cnt;
            }
            let p = (cnt as f32) * inv_denom;
            hhi += p * p;
        }
        let action_saturation_ratio = branchless_clamp_f32((max_decision as f32) * inv_denom, 0.0, 1.0);
        let concentration = branchless_clamp_f32((hhi - (1.0 / 7.0)) / (6.0 / 7.0), 0.0, 1.0);
        let sat_excess =
            branchless_clamp_f32((action_saturation_ratio - 0.45) / 0.55, 0.0, 1.0);
        let sat_load = 0.5 * concentration + 0.5 * sat_excess;
        let passive_stall_rate =
            branchless_clamp_f32((self.layer0.passive_stall_events as f32) * inv_denom, 0.0, 1.0);
        let active_mismatch_rate = branchless_clamp_f32(
            (self.layer2.unaligned_intervention_count as f32) * inv_denom,
            0.0,
            1.0,
        );
        let ego_ratio =
            branchless_clamp_f32((self.layer2.intervention_count as f32) * inv_denom, 0.0, 1.0);

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
            ((self.layer0.passive_stall_events + self.layer2.unaligned_intervention_count) as f32)
                * inv_denom,
            0.0,
            1.0,
        );

        let enough_samples = self.system.continuous_standard_steps >= 16;
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
        stack: &AutonomousCoreLayerStack,
        horizon: &EpistemicHorizonReading,
        active_crystals: u8,
        spawned_daughters: u8,
        active_horizon_slots_mask: u32,
    ) -> CoreTelemetrySnapshot {
        let denom = self.system.continuous_standard_steps.max(1) as f32;
        let has_samples = self.system.continuous_standard_steps > 0;
        let mut decision_ratios_pct = [0.0_f32; 7];
        if has_samples {
            for (dst, &cnt) in decision_ratios_pct
                .iter_mut()
                .zip(self.layer1.decision_counts.iter())
            {
                *dst = (cnt as f32 / denom) * 100.0;
            }
        }
        let excursion_cnt = self.layer0.regime_counts[1].saturating_add(self.layer0.regime_counts[2]);
        let topology = LiminalTopologyTelemetrySnapshot::capture(
            &stack.liminal_layer,
            horizon,
            active_crystals,
            spawned_daughters,
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
            layer0: self.layer0,
            layer1: self.layer1,
            layer2: self.layer2,
            system: self.system,
            detailed,
            decision_ratios_pct,
            liminal_excursion_ratio: if has_samples {
                excursion_cnt as f32 / denom
            } else {
                0.0
            },
            stationary_shortcut_ratio: if has_samples {
                self.layer0.stationary_shortcut_hits as f32 / denom
            } else {
                0.0
            },
            ego_intervention_ratio: if has_samples {
                self.layer2.intervention_count as f32 / denom
            } else {
                0.0
            },
            topology,
            distortion,
            active_horizon_slots_mask,
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
            || self.system.continuous_standard_steps < req
        {
            return SlowLoopTelemetryAdjustment {
                applied: false,
                sampled_steps: self.system.continuous_standard_steps,
                tuned_thresholds: *base,
                suggested_slide_out_dwell_delta: 0,
                suggested_repulse_bias_q7: 0,
                distortion,
            };
        }

        let denom = self.system.continuous_standard_steps as f32;
        let chatter_ratio = self.layer0.chatter_events as f32 / denom;
        let excursion_ratio =
            (self.layer0.regime_counts[1].saturating_add(self.layer0.regime_counts[2])) as f32
                / denom;
        let skip_ratio = self.layer1.decision_counts[1] as f32 / denom;
        let rank1_ratio = self.layer1.decision_counts[4] as f32 / denom;
        let repulse_ratio = self.layer1.decision_counts[6] as f32 / denom;

        // Layer 0 chatter -> bounded dwell hysteresis extension (prevents slot flapping).
        let dwell_delta: i8 = if chatter_ratio > 0.50 {
            2
        } else if chatter_ratio > 0.25 {
            1
        } else {
            0
        };

        // Layer 1 decision balance -> homeostatic ±15% adjustments keeping the monotone ladder intact.
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
            1.0 + (rank1_ratio - 0.45) * 0.35 - (skip_ratio - 0.25) * 0.35,
            0.85,
            1.20,
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

        self.system.slow_loop_epochs = self.system.slow_loop_epochs.saturating_add(1);
        self.system.slow_loop_dwell_delta = dwell_delta;
        self.system.slow_loop_repulse_bias_q7 = raw_bias;

        SlowLoopTelemetryAdjustment {
            applied: true,
            sampled_steps: self.system.continuous_standard_steps,
            tuned_thresholds,
            suggested_slide_out_dwell_delta: dwell_delta,
            suggested_repulse_bias_q7: raw_bias,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_telemetry_bank_modes_pull_and_slow_loop_bounds() {
        let mut bank = CoreTelemetryBank::default();
        assert_eq!(bank.mode(), TelemetryVerbosityMode::Standard);

        let stack = AutonomousCoreLayerStack::new();
        let coherent_horizon = EpistemicHorizonReading {
            regime_code: EpistemicRegime::KnownCoherent as u8,
            ..EpistemicHorizonReading::default()
        };
        let conflict_horizon = EpistemicHorizonReading {
            regime_code: EpistemicRegime::InternalConflict as u8,
            ..EpistemicHorizonReading::default()
        };
        let mut delta = LiminalBoundaryDelta {
            stationary_shortcut_ready: 1,
            ..LiminalBoundaryDelta::default()
        };
        let venn = StigmergicVennSerendipityReport::default();
        let mut ego = LiminalEgoField::new();

        // 1. Auto mode promotes cleanly to Detailed on destabilization, records breakdown, then demotes after dwell.
        assert_eq!(
            bank.evaluate_policy(true, 0, false),
            TelemetryVerbosityMode::Detailed
        );
        delta.chattering_channel_mask = 1 << 2;
        ego.last_intervened_mask = EGO_TRIGGER_BOUNDARY_CHATTER;
        bank.record_step(
            StigmergicActionDecision::RepulsionEscape,
            &conflict_horizon,
            &delta,
            &venn,
            &ego,
            2,
            1 << 0,
            0,
            true,
            false,
        );
        assert_eq!(bank.detailed.detailed_captured_steps, 1);
        assert_eq!(bank.detailed.liminal_channel_chatter_counts[2], 1);
        assert_eq!(bank.detailed.ego_trigger_counts[3], 1);
        assert_eq!(bank.layer2.escalation_swaps, 1);

        // Retracts smoothly back to Standard once quiescent_dwell >= retract_dwell_threshold (2).
        assert_eq!(
            bank.evaluate_policy(false, 1, false),
            TelemetryVerbosityMode::Detailed
        );
        assert_eq!(
            bank.evaluate_policy(false, 2, false),
            TelemetryVerbosityMode::Standard
        );

        // 2. QoS throttle drops to Coarse; denominator `continuous_standard_steps` does not skew.
        let prev_continuous = bank.system.continuous_standard_steps;
        assert_eq!(
            bank.evaluate_policy(false, 2, true),
            TelemetryVerbosityMode::Coarse
        );
        bank.record_step(
            StigmergicActionDecision::InertialLayerSkip,
            &coherent_horizon,
            &delta,
            &venn,
            &LiminalEgoField::new(),
            0,
            0,
            0,
            false,
            true,
        );
        assert_eq!(bank.system.total_steps, 2);
        assert_eq!(bank.system.continuous_standard_steps, prev_continuous);
        assert_eq!(bank.system.qos_throttle_events, 1);

        // Slow-Loop skips while Coarse is active.
        let base = StigmergicThresholds::default();
        let skipped_adj = bank.evaluate_slow_loop_epoch(&base, 1);
        assert!(!skipped_adj.applied);

        // Return to Standard, add samples, and verify bounded Slow-Loop adjustment + Lazy-Pull ratios.
        assert_eq!(
            bank.evaluate_policy(false, 2, false),
            TelemetryVerbosityMode::Standard
        );
        for _ in 0..9 {
            bank.record_step(
                StigmergicActionDecision::ExpressPhaseSteer,
                &coherent_horizon,
                &LiminalBoundaryDelta {
                    stationary_shortcut_ready: 1,
                    ..LiminalBoundaryDelta::default()
                },
                &venn,
                &LiminalEgoField::new(),
                0,
                0,
                0,
                false,
                false,
            );
        }
        let adj = bank.evaluate_slow_loop_epoch(&base, 8);
        assert!(adj.applied);
        assert_eq!(adj.sampled_steps, 10);
        assert!(adj.tuned_thresholds.express_steer_density_thresh >= base.express_steer_density_thresh * 0.85);
        assert!(adj.tuned_thresholds.express_steer_density_thresh <= base.express_steer_density_thresh * 1.15);

        let snap = bank.snapshot(&stack, &coherent_horizon, 2, 1, 0);
        assert_eq!(snap.system.total_steps, 11);
        assert_eq!(snap.system.continuous_standard_steps, 10);
        assert!((snap.decision_ratios_pct[2] - 90.0).abs() < 1e-4);
        assert!((snap.decision_ratios_pct[6] - 10.0).abs() < 1e-4);
        assert!(snap.detailed.is_some());
    }

    #[test]
    fn test_layer_distortion_report_and_slow_loop_rank1_skip_balance() {
        let mut bank = CoreTelemetryBank::default();
        let conflict = EpistemicHorizonReading {
            regime_code: EpistemicRegime::InternalConflict as u8,
            conflict_severity: 0.25,
            ..EpistemicHorizonReading::default()
        };
        let stagnant_delta = LiminalBoundaryDelta::default();
        let venn = StigmergicVennSerendipityReport::default();

        // 1. Passive saturation under Rank-1 dominance without Ego intervention -> PassiveSaturation.
        for _ in 0..20 {
            bank.record_step(
                StigmergicActionDecision::FullOrchestration,
                &conflict,
                &stagnant_delta,
                &venn,
                &LiminalEgoField::new(),
                0,
                0,
                0,
                false,
                false,
            );
        }
        let pass_dist = bank.compute_distortion_report();
        assert_eq!(pass_dist.kind, LayerDistortionKind::PassiveSaturation);
        assert!(pass_dist.ego_wakeup_recommended);
        assert!(!pass_dist.ego_cooldown_recommended);

        let base = StigmergicThresholds::default();
        let adj = bank.evaluate_slow_loop_epoch(&base, 16);
        assert!(adj.applied);
        assert!(adj.tuned_thresholds.rank1_inject_density_thresh > base.rank1_inject_density_thresh);
        assert!(adj.tuned_thresholds.inertial_skip_entropy_thresh > base.inertial_skip_entropy_thresh);
        assert!(adj.tuned_thresholds.validate().is_ok());

        // 2. Active overdrive under unaligned Ego intervention -> ActiveOverdrive.
        bank.reset();
        let mut active_ego =
            LiminalEgoField::from_preset(crate::LiminalEgoPreset::High);
        active_ego.last_intervened_mask = EGO_TRIGGER_INTERNAL_CONFLICT;
        for _ in 0..20 {
            bank.record_step(
                StigmergicActionDecision::RepulsionEscape,
                &conflict,
                &stagnant_delta,
                &venn,
                &active_ego,
                2,
                0,
                0,
                false,
                false,
            );
        }
        let act_dist = bank.compute_distortion_report();
        assert_eq!(act_dist.kind, LayerDistortionKind::ActiveOverdrive);
        assert!(act_dist.ego_cooldown_recommended);
        assert!(!act_dist.ego_wakeup_recommended);

        // 3. High saturation (76%) with ~48% Ego must NOT falsely read as Aligned, and accumulate_snapshot merges cleanly.
        let mut bank_a = CoreTelemetryBank::default();
        let mut active_mid_ego = LiminalEgoField::from_preset(crate::LiminalEgoPreset::Medium);
        let aligned_horizon = EpistemicHorizonReading {
            regime_code: EpistemicRegime::KnownCoherent as u8,
            ..EpistemicHorizonReading::default()
        };
        for i in 0..25 {
            let dec = if i < 19 {
                StigmergicActionDecision::FullOrchestration
            } else {
                StigmergicActionDecision::ExpressPhaseSteer
            };
            active_mid_ego.last_intervened_mask = if i < 12 { EGO_TRIGGER_BASIN_CAPTURE } else { 0 };
            bank_a.record_step(
                dec,
                &aligned_horizon,
                &stagnant_delta,
                &venn,
                &active_mid_ego,
                0,
                0,
                0,
                false,
                false,
            );
        }
        let sat_dist = bank_a.compute_distortion_report();
        assert_ne!(sat_dist.kind, LayerDistortionKind::Aligned);
        let mut merged = CoreTelemetryBank::default();
        merged.accumulate_snapshot(&bank_a.snapshot(&AutonomousCoreLayerStack::new(), &aligned_horizon, 0, 0, 0));
        assert_eq!(merged.compute_distortion_report().kind, sat_dist.kind);
    }
}
