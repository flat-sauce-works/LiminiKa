// name: src/core/src/runtime/context/orchestration.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Runtime step execution, lifecycle actions, and latent-horizon orchestration on [`RuntimeContext`].

use super::{
    abi_version, abi_version_string, query_capability, CoreStateStack, Result, RuntimeContext,
};
use crate::drivers::{
    DistortionTension, EgoDifferentialNet, EntropyState, SpineDecodeSignal, SpineStepPlan,
    SystemSpineState, TokenBypassObservation, HOST_IMMUTABLE_PINNED_COMPUTE_MASK,
    SPINE_CONTROL_FLAG_PREV_RESIDENCY_VALID, SPINE_CONTROL_FLAG_PREV_WAS_ABSTRACT,
    SPINE_ENTROPY_CALM_THRESH_Q15, SPINE_ENTROPY_GUARD_RELAX_MAX_Q15, SPINE_ENTROPY_Q15_ONE,
    SPINE_ENTROPY_REWARD_LOW_THRESH_Q15, SPINE_ENTROPY_SURGE_THRESH_Q15,
    SPINE_ENTROPY_TENSION_FLOOR_Q15, SPINE_SENTINEL_AUTO_U8, SPINE_STEP_EVENT_BOUNDARY_CHURN,
    SPINE_STEP_EVENT_GUARD_REESCALATED, SPINE_STEP_EVENT_INSTABILITY_RECOVERY,
    SPINE_STEP_EVENT_LOGITS_OBSERVE_SKIPPED, SPINE_STEP_EVENT_NONE,
    SPINE_STEP_EVENT_TRANSIENT_GUARD_RELAXED,
};
use crate::foundation::{
    branchless_clamp_f32, branchless_max_f32, branchless_min_f32, store_q7_words_64,
    swar_words_nonzero_64, AlignedBuffer16, AlignedSlice16, AlignedSliceMut32, PhaseSteering,
    RopeSteeringTarget, TaggedPtr, ACTION_MODE_LAYER_SKIP, ACTION_MODE_SUBHEAD_ROUTE,
};
use crate::liminal::{
    branchless_select_f32, branchless_select_u32, branchless_select_u64, branchless_select_u8,
    AttributionTag, BitCut8, DriverKind, EpistemicRegime, Lacuna, LacunaPartition8, PrimitiveKind,
    RuntimeConfig, Scoped, LMNK_ERR_INVALID_ARGUMENT, MAX_ATTENTION_HEADS,
};
use crate::runtime::boundary_delta::LiminalBoundaryDelta;
use crate::runtime::gateway::{
    CausalProvenanceReport, GatewayFrame, ManifoldGatewaySignal, ROOT_CAUSE_SYSTEM_QOS_THROTTLE,
};
use crate::runtime::horizon::{
    apply_latent_horizon_slots, EpistemicHorizonReading, HorizonEscalationProvider,
    HorizonEscalationRequest, HorizonTriggerInput, LatentHorizonSlot,
};
use crate::runtime::storage::KvKeyframeHeader;
use crate::stigmergy::{
    trail_phase_words_mut, trail_phase_words_ref, LedgerSlot, OverlapDiscoveryReport,
    PhaseDeltaSlice, PointerSubstrate, PointerTrail, SandboxLifecycleEvent,
    StigmergicActionDecision, StigmergicSubstrateOps, WorkbenchLanePlacement,
    LEDGER_METRIC_FAST_PATH_HIT, LEDGER_METRIC_PHASE_STEERED, LEDGER_METRIC_REPULSION_APPLIED,
    MAX_ATTRACTOR_PULL_FORCE, TRAIL_FLAG_REPULSION_ACTIVE, WORKBENCH_LANE_FLAG_REPULSION,
};

// ===================================================================
// Inherent Runtime Orchestration & Lifecycle Methods
// ===================================================================

impl RuntimeContext {
    /// Returns the active semantic ABI version tuple `(major, minor, patch)`.
    #[inline(always)]
    #[must_use]
    pub const fn abi_version(&self) -> (u32, u32, u32) {
        abi_version()
    }

    /// Returns the static ABI version string ([`crate::liminal::LMNK_ABI_VERSION_STRING`]).
    #[inline(always)]
    #[must_use]
    pub const fn abi_version_string(&self) -> &'static str {
        abi_version_string()
    }

    /// Queries active compute capability bitmask flags without branching.
    #[inline(always)]
    #[must_use]
    pub const fn query_capability(&self) -> u64 {
        query_capability()
    }

    /// Populates `config` with this runtime context's active raw configuration descriptor.
    ///
    /// # Errors
    /// Never fails in the default implementation.
    #[inline(always)]
    pub fn init_default_config(&self, config: &mut RuntimeConfig) -> Result<()> {
        *config = self.config.raw;
        Ok(())
    }

    /// Resets transient phase accumulators and step counters without freeing pre-allocated tables.
    ///
    /// # Errors
    /// Propagates any reset error from [`PointerSubstrate::reset`] or [`RuntimeContext::clear_all_domains`].
    #[inline]
    pub fn reset(&mut self) -> Result<()> {
        self.entropy_controller.reset();
        self.substrate.reset()?;
        self.clear_all_domains()?;
        self.substrate.bloom_filter.clear();
        self.substrate.ledger_slot = LedgerSlot {
            mode: u32::from(self.config.raw.ledger_mode),
            ..Default::default()
        };
        self.core_layers = CoreStateStack::new();
        self.last_boundary_delta = LiminalBoundaryDelta::default();
        self.last_horizon = EpistemicHorizonReading::default();
        self.last_overlap_report = OverlapDiscoveryReport::default();
        self.active_horizon_slots_mask = 0;
        self.sandbox_workbench_lane = None;
        self.last_sandbox_event = SandboxLifecycleEvent::Idle;
        self.sandbox_settlements = 0;
        self.compute_pool.reset_turn_bypass_state();
        let prev_baseline_pin = self.system_spine.baseline_pinned_compute_mask;
        let prev_pre_bypass = self.system_spine.is_pre_bypass_enabled();
        let mut reset_spine = SystemSpineState::with_policy(
            self.config.raw.guard_hold_tokens,
            self.config.raw.max_observe_stride,
        );
        if prev_baseline_pin != 0 {
            reset_spine.apply_host_pin_floor(prev_baseline_pin);
        }
        reset_spine.set_pre_bypass_enabled(prev_pre_bypass);
        self.system_spine = reset_spine;
        self.last_lacuna_partition = LacunaPartition8::default();
        self.distortion_tension = DistortionTension::new(false);
        self.ego_differential_net = EgoDifferentialNet::default();
        Ok(())
    }

    /// Updates runtime spine policy parameters (`0 alloc`) while monotonically raising the
    /// permanent host compute pin floor (`baseline_pinned_compute_mask`) and re-clamping
    /// [`crate::drivers::AnonymousComputePool`] masks so `configure_spine` and `configure_compute_pool` commute.
    ///
    /// Passing `0` for `baseline_pinned_compute_mask` preserves the active permanent floor while
    /// enforcing [`HOST_IMMUTABLE_PINNED_COMPUTE_MASK`].
    #[inline]
    pub fn configure_spine(
        &mut self,
        baseline_pinned_compute_mask: u64,
        guard_hold_tokens: u8,
        max_observe_stride: u8,
        pre_bypass_enabled: bool,
    ) {
        self.system_spine.apply_host_pin_floor(
            baseline_pinned_compute_mask | HOST_IMMUTABLE_PINNED_COMPUTE_MASK,
        );
        self.system_spine.guard_hold_tokens = guard_hold_tokens;
        self.system_spine.set_max_observe_stride(max_observe_stride);
        self.system_spine.set_pre_bypass_enabled(pre_bypass_enabled);
        self.config.raw.guard_hold_tokens = guard_hold_tokens;
        self.config.raw.max_observe_stride = self.system_spine.max_observe_stride;
        let effective_pinned = self.system_spine.pinned_compute_mask
            | self.system_spine.baseline_pinned_compute_mask
            | HOST_IMMUTABLE_PINNED_COMPUTE_MASK;
        self.compute_pool.set_slot_masks(
            self.compute_pool.active_mask,
            effective_pinned,
            self.compute_pool.repulsion_mask,
            self.compute_pool.reversible_mask,
        );
    }

    /// Configures anonymous compute pool slot masks (`active`, `repulsion`, `reversible`) and
    /// optionally re-anchors the Gauge-Zero baseline pressure (`0 alloc`).
    ///
    /// - Host-pinned slots (`pinned_compute_mask | baseline_pinned_compute_mask |
    ///   HOST_IMMUTABLE_PINNED_COMPUTE_MASK`) are never stripped and are clamped out of
    ///   `repulsion_mask` and `reversible_mask` alongside any bits outside `active_mask`.
    /// - Passing `0` for `baseline_pressure_q15` preserves the active baseline pressure
    ///   (default `8192`); any non-zero value calls
    ///   [`crate::drivers::AnonymousComputePool::reanchor_baseline_pressure`] (clamped to `32767`).
    /// - Commutes with [`Self::configure_spine`]: calling either order yields identical
    ///   `SystemSpineState` and `AnonymousComputePool` masks.
    #[inline]
    pub fn configure_compute_pool(
        &mut self,
        active_mask: u64,
        repulsion_mask: u64,
        reversible_mask: u64,
        baseline_pressure_q15: u16,
    ) {
        self.system_spine
            .apply_host_pin_floor(HOST_IMMUTABLE_PINNED_COMPUTE_MASK);
        let effective_pinned = self.system_spine.pinned_compute_mask
            | self.system_spine.baseline_pinned_compute_mask
            | HOST_IMMUTABLE_PINNED_COMPUTE_MASK;
        self.compute_pool.set_slot_masks(
            active_mask,
            effective_pinned,
            repulsion_mask,
            reversible_mask,
        );
        if baseline_pressure_q15 != 0 {
            self.compute_pool
                .reanchor_baseline_pressure(baseline_pressure_q15);
        }
    }

    /// Initializes per-turn mechanical spine policy at prompt / turn boundaries (`0 alloc`).
    ///
    /// Resets per-turn bypass counters, clears stale observed entropy, arms or restores transient
    /// turn-guard compute pins without mutating `baseline_pinned_compute_mask`, and seeds
    /// `last_executed_mask` to `compute_pool.active_mask`.
    #[inline]
    pub fn begin_turn_spine_policy(&mut self, guard_turn: bool, unobserved_locked: bool) {
        self.compute_pool.reset_turn_bypass_state();
        self.system_spine.reset_turn_bypass_state();
        self.system_spine.last_observed_entropy_q15 = 0;
        self.system_spine.set_transient_guard_relaxed(false);
        self.system_spine.set_turn_guard_active(guard_turn);
        self.system_spine.unobserved_locked = u8::from(unobserved_locked);
        let active_mask = self.compute_pool.active_mask;
        if guard_turn {
            self.system_spine.arm_transient_guard_pin(active_mask);
        } else {
            self.system_spine.restore_baseline_pin_mask();
        }
        self.system_spine.last_executed_mask = active_mask;
    }

    /// Evaluates the 4-stage decode control step in [`RuntimeContext`] using static-generic
    /// [`SpineDecodeSignal`] (`0 alloc`, `0 fdiv`), returning a pre-execution [`SpineStepPlan`]:
    ///
    /// 1. Residency & boundary-churn accounting plus [`Lacuna`] exit-entropy handling.
    /// 2. Compute exit reward commitment onto `last_executed_mask` (when `lacuna.is_defined()`).
    /// 3. Transient contradiction-guard pin relaxation or immediate instability re-escalation.
    /// 4. Pre-execution compute dispatch, whole-token bypass gating, next logits observation mode
    ///    selection, and `last_executed_mask` update.
    #[inline]
    #[must_use]
    pub fn step_decode_control<S: SpineDecodeSignal>(
        &mut self,
        signal: &S,
        external_guard_active: bool,
    ) -> SpineStepPlan {
        const Q15_TO_F32: f32 = 1.0_f32 / 32_768.0_f32;

        let step_was_whole_bypass = self.system_spine.whole_token_bypass_active != 0;
        let lacuna = signal.exit_entropy_lacuna();
        let action_repulse = StigmergicActionDecision::from_u8(self.substrate.last_decision)
            == StigmergicActionDecision::RepulsionEscape;
        let entropy_repulse = self.entropy_controller.is_repulsion_mode();
        let eff_repulsion = signal.repulsion_active() || action_repulse || entropy_repulse;

        let raw_sens = signal.boundary_sensitivity_q7();
        let sens_q7 = if raw_sens == SPINE_SENTINEL_AUTO_U8 {
            self.last_lacuna_partition.boundary_sensitivity_q7
        } else {
            raw_sens.min(127)
        };
        let min_calm_dwell = signal.min_calm_dwell().max(2);
        let pre_bypass_active =
            signal.pre_bypass_enabled() && self.system_spine.is_pre_bypass_enabled();
        let mut event_flags = SPINE_STEP_EVENT_NONE;

        // Stage 1: Residency tracking, boundary-churn detection, and `Lacuna` exit-entropy update.
        let prev_valid =
            (self.system_spine.control_flags & SPINE_CONTROL_FLAG_PREV_RESIDENCY_VALID) != 0;
        let prev_was_abstract =
            (self.system_spine.control_flags & SPINE_CONTROL_FLAG_PREV_WAS_ABSTRACT) != 0;
        let churn = prev_valid && (prev_was_abstract != step_was_whole_bypass);
        if churn {
            event_flags |= SPINE_STEP_EVENT_BOUNDARY_CHURN;
        }
        self.system_spine.control_flags |= SPINE_CONTROL_FLAG_PREV_RESIDENCY_VALID;
        if step_was_whole_bypass {
            self.system_spine.control_flags |= SPINE_CONTROL_FLAG_PREV_WAS_ABSTRACT;
        } else {
            self.system_spine.control_flags &= !SPINE_CONTROL_FLAG_PREV_WAS_ABSTRACT;
        }
        self.telemetry
            .record_decode_residency(step_was_whole_bypass, churn);

        let entropy_surge = match lacuna {
            Lacuna::SelfRef | Lacuna::Exterior => {
                let ent_q15 = signal
                    .observed_exit_entropy_q15()
                    .min(SPINE_ENTROPY_Q15_ONE);
                self.system_spine.last_observed_entropy_q15 = ent_q15;
                self.system_spine.unobserved_bypass_run = 0;
                if step_was_whole_bypass {
                    self.substrate.last_entropy_h_tilde = f32::from(ent_q15) * Q15_TO_F32;
                }
                let ent_above_floor_q15 = ent_q15.saturating_sub(SPINE_ENTROPY_TENSION_FLOOR_Q15);
                let exit_entropy_q7 =
                    (((u32::from(ent_above_floor_q15) * 127) + 16_384) >> 15).min(127) as u8;
                let guard_prox_q7 = if eff_repulsion { 88 } else { sens_q7 >> 2 };
                let active_overdrive = eff_repulsion && self.core_layers.ego.is_active();
                self.distortion_tension.step_hot_path_branchless(
                    self.ego_differential_net.last_delta_norm_q7,
                    sens_q7,
                    exit_entropy_q7,
                    guard_prox_q7,
                    active_overdrive,
                );
                let surge = ent_q15 > SPINE_ENTROPY_SURGE_THRESH_Q15;
                if surge {
                    self.signal_compute_instability();
                    event_flags |= SPINE_STEP_EVENT_INSTABILITY_RECOVERY;
                }
                surge
            }
            Lacuna::Unobserved => {
                self.system_spine.unobserved_bypass_run =
                    self.system_spine.unobserved_bypass_run.saturating_add(1);
                self.telemetry.record_logits_skip();
                event_flags |= SPINE_STEP_EVENT_LOGITS_OBSERVE_SKIPPED;
                false
            }
            Lacuna::Undefined => {
                self.system_spine.unobserved_bypass_run = 0;
                self.signal_compute_instability();
                event_flags |= SPINE_STEP_EVENT_INSTABILITY_RECOVERY;
                true
            }
        };

        // Stage 2: Commit completed-step compute reward when observation is well-defined.
        if lacuna.is_defined() {
            let raw_reward = signal.exit_reward_override_q7();
            let reward_q7 = if raw_reward == SPINE_SENTINEL_AUTO_U8 {
                if self.system_spine.last_observed_entropy_q15 < SPINE_ENTROPY_REWARD_LOW_THRESH_Q15
                {
                    56
                } else {
                    96
                }
            } else {
                raw_reward.min(127)
            };
            let reward_mask = if self.system_spine.last_executed_mask != 0 {
                self.system_spine.last_executed_mask
            } else {
                self.compute_pool.active_mask
            };
            self.compute_pool.commit_exit_reward(reward_mask, reward_q7);
        }

        // Stage 3: Transient contradiction-guard relaxation or instability re-escalation.
        let liminal_stationary = u8::from(
            lacuna.is_defined()
                && (self.system_spine.last_observed_entropy_q15 < SPINE_ENTROPY_CALM_THRESH_Q15)
                && (sens_q7 < 80),
        );
        if self.system_spine.is_transient_guard_relaxed() {
            if eff_repulsion || entropy_surge || !lacuna.is_defined() {
                self.system_spine.set_transient_guard_relaxed(false);
                self.system_spine.set_turn_guard_active(true);
                let active_mask = self.compute_pool.active_mask;
                self.system_spine.arm_transient_guard_pin(active_mask);
                self.compute_pool.reset_turn_bypass_state();
                self.system_spine.reset_turn_bypass_counters_only();
                self.system_spine.unobserved_bypass_run = 0;
                event_flags |= SPINE_STEP_EVENT_GUARD_REESCALATED;
            }
        } else if pre_bypass_active
            && (self.system_spine.turn_guard_active() || external_guard_active)
            && self.system_spine.guard_hold_tokens > 0
            && self.system_spine.passive_saturation_probe_steps == 0
            && self.system_spine.unobserved_locked == 0
            && lacuna.is_defined()
            && (self.system_spine.last_observed_entropy_q15 < SPINE_ENTROPY_GUARD_RELAX_MAX_Q15)
            && self.system_spine.can_relax_transient_guard_pin(
                self.system_spine.guard_hold_tokens,
                liminal_stationary,
                eff_repulsion,
                entropy_surge,
                min_calm_dwell,
            )
        {
            self.system_spine.set_transient_guard_relaxed(true);
            self.system_spine.set_turn_guard_active(false);
            self.system_spine.restore_baseline_pin_mask();
            self.compute_pool.re_bypass_cooldown = 0;
            self.system_spine.on_transient_guard_relaxed(min_calm_dwell);
            self.telemetry.record_guard_relaxation();
            event_flags |= SPINE_STEP_EVENT_TRANSIENT_GUARD_RELAXED;
        }

        // Stage 4: Pre-execution dispatch, whole-token bypass evaluation, and next observe mode.
        let static_guard = !self.system_spine.is_transient_guard_relaxed()
            && (self.system_spine.turn_guard_active() || external_guard_active);
        let eff_guard_active =
            static_guard || (self.system_spine.passive_saturation_probe_steps > 0);
        self.system_spine.passive_saturation_probe_steps = self
            .system_spine
            .passive_saturation_probe_steps
            .saturating_sub(1);

        let mut dispatch = self.compute_pool.step_dispatch_with_spine(
            signal.pressure_reading(),
            &self.system_spine,
            eff_guard_active,
            eff_repulsion,
        );
        let veto_mask = if pre_bypass_active {
            let irreversible_step = u8::from(
                (signal.pressure_reading().irreversible_step != 0)
                    || (self.system_spine.irreversible_step_lock != 0),
            );
            let ego_active_boundary = u8::from(self.core_layers.ego.is_active() && sens_q7 >= 80);
            let obs = TokenBypassObservation {
                irreversible_step,
                liminal_stationary,
                boundary_sensitivity_q7: sens_q7,
                repulsion_active: u8::from(eff_repulsion),
                exit_entropy_rising: u8::from(entropy_surge),
                ego_active_boundary,
                min_calm_dwell,
                _pad: 0,
            };
            self.evaluate_whole_token_bypass_with_spine(obs, eff_guard_active, &mut dispatch)
        } else {
            dispatch.whole_token_steering_bypass = 0;
            self.system_spine.whole_token_bypass_active = 0;
            let veto = self.system_spine.evaluate_pool_bypass_veto_branchless(
                self.compute_pool.reversible_mask,
                eff_guard_active,
                eff_repulsion,
            );
            self.system_spine.last_veto_mask = veto;
            veto
        };

        let next_step_index = signal.decode_step_index().saturating_add(1);
        let next_whole_token_bypass = dispatch.whole_token_steering_bypass != 0;
        let (next_mode_enum, effective_observe_stride) =
            self.system_spine.select_next_observe_mode(
                next_step_index,
                signal.greedy_sampling(),
                next_whole_token_bypass,
                sens_q7,
            );
        self.system_spine.last_executed_mask = dispatch.execute_mask;

        SpineStepPlan {
            dispatch,
            next_observe_mode: next_mode_enum.code(),
            veto_mask,
            event_flags,
            effective_observe_stride,
            reserved_align: [0; 20],
        }
    }

    /// Executes a per-token Hot Path inference step in zero-allocation mode ($C_{\mathrm{alloc}} = 0$).
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `query_tensor` and `key_tensor` alias.
    #[inline]
    pub fn step_token(
        &mut self,
        token_id: u32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: Option<&mut PointerTrail>,
    ) -> Result<()> {
        if let (Some(q), Some(k)) = (query_tensor.as_deref(), key_tensor.as_deref()) {
            q.ensure_disjoint_mut(k)?;
        }

        let mut state_out = EntropyState::default();
        self.entropy_controller.eval_stateful(0.0, &mut state_out)?;

        let next_step = self.substrate.step_counter.wrapping_add(1);
        self.substrate.step_counter = next_step;
        let cap = self.config.raw.substrate_capacity;
        let cur_active = self.substrate.active_count;
        self.substrate.active_count =
            branchless_select_u32(cur_active < cap, cur_active.wrapping_add(1), cur_active);

        let mut repulsion_active = false;
        if let Some(trail) = trail_out {
            trail.prev_ptr = branchless_select_u64(
                trail.current_ptr != 0,
                trail.current_ptr,
                u64::from(self.substrate.last_selected_trail_id),
            );
            trail.current_ptr = u64::from(token_id);
            trail.step_count = next_step as u32;
            trail.stigmergic_density = branchless_select_f32(
                trail.stigmergic_density > 1.0,
                trail.stigmergic_density,
                1.0,
            );

            let (mut phase_words, total_pull) =
                self.evaluate_multi_domain_superposition_words(trail);

            let (_density, rep_trig) = self
                .substrate
                .bloom_filter
                .apply_phase_conjugate_repulsion_words(
                    &mut phase_words,
                    self.config.repulsion_threshold,
                    self.config.repulsion_scale_u7,
                );
            repulsion_active = rep_trig;

            trail.attractor_pull_force = total_pull;
            *trail_phase_words_mut(trail) = phase_words;
            trail.flags = state_out.active_branch_mode
                | branchless_select_u32(repulsion_active, TRAIL_FLAG_REPULSION_ACTIVE, 0);
        }
        self.substrate.last_selected_trail_id = token_id;

        let metric = LEDGER_METRIC_PHASE_STEERED
            | branchless_select_u8(repulsion_active, LEDGER_METRIC_REPULSION_APPLIED, 0);
        self.substrate.ledger_slot.push_telemetry_metric(metric);
        Ok(())
    }

    /// Serializes active runtime context state into a `.lmnk` binary snapshot buffer.
    ///
    /// # Errors
    /// Returns [`crate::liminal::LMNK_ERR_BUFFER_TOO_SMALL`] if `buffer.len() < *required_size`.
    #[inline]
    pub fn serialize(&self, buffer: &mut [u8], required_size: &mut usize) -> Result<()> {
        self.serialize_to_bytes(buffer, required_size)
    }

    /// Deserializes a `.lmnk` binary snapshot buffer to restore context state in-place.
    ///
    /// # Errors
    /// Returns [`crate::liminal::LMNK_ERR_CONTAINER_CORRUPTED`] or [`crate::liminal::LMNK_ERR_VERSION_MISMATCH`]
    /// if the snapshot header magic, version, or payload checksum is invalid.
    #[inline]
    pub fn deserialize(&mut self, buffer: &[u8]) -> Result<()> {
        self.deserialize_from_bytes(buffer)
    }

    /// Seeks to a specific token position in the KV-keyframe cache without forward passes.
    ///
    /// # Errors
    /// Never fails in the default implementation.
    #[inline]
    pub fn seek_to_token(
        &mut self,
        token_index: u32,
        header_out: &mut KvKeyframeHeader,
    ) -> Result<()> {
        header_out.token_index = token_index;
        header_out.frame_type = 0;
        Ok(())
    }

    /// Executes a unified vertical orchestration pass that pierces all 3 core stack slots
    /// (`Slot 0: Liminal` -> `Slot 1: Stigmergic` -> `Slot 2: Ego`), dynamically slides
    /// dormant [`LatentHorizonSlot`] boundaries in or out based on live stability/anomaly
    /// telemetry, optionally escalates unresolved excursion to an external
    /// [`HorizonEscalationProvider`], updates `trail_out` / `StigmergicSubstrate`, and steers
    /// `query_tensor` in strict zero-allocation mode ($C_{\mathrm{alloc}} = 0$).
    ///
    /// Returns `(frame, report, slid_in_bits, slid_out_bits, escalated)`.
    ///
    /// # Errors
    /// Propagates validation or steering errors from `geometry`, `query_tensor`, `StigmergicSubstrate`,
    /// or `escalation_provider`.
    #[inline]
    pub fn pierce_and_orchestrate_step<
        S: ManifoldGatewaySignal + ?Sized,
        G: RopeSteeringTarget + ?Sized,
        H: LatentHorizonSlot + ?Sized,
        E: HorizonEscalationProvider + ?Sized,
    >(
        &mut self,
        signal: &S,
        geometry: &G,
        latent_slots: &mut H,
        escalation_provider: &mut E,
        trail_out: &mut PointerTrail,
        mut query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> Result<(GatewayFrame, CausalProvenanceReport, u32, u32, bool)> {
        let (head_dim, num_heads, _expected_len) =
            self.substrate.validate_substrate_cascade_params(
                self.phase_steering.head_dim(),
                self.phase_steering.num_heads(),
                self.phase_steering.phase_step_min_rad(),
                self.phase_steering.phase_clamp_max_rad(),
                query_tensor.as_deref(),
                None,
            )?;
        let geom = geometry.rope_geometry();
        if geom.head_dim != head_dim || geom.num_heads != num_heads {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }

        // 1. Compute pure `SystemSpine` I-Frame baseline when Ego is active, then pierce
        //    `Liminal` -> `Stigmergy` -> `EgoDifferentialNet` and enforce physical `pinned` isolation.
        let prev_liminal = self.core_layers.liminal;
        let prev_overlap = self.core_layers.stigmergy.overlap_mask;
        let (mut frame, pframe_delta_q7, pframe_hops) = if self.core_layers.ego.is_active() {
            let mut sys_baseline =
                signal.synthesize_gateway_with_aperture_from_stack(&self.core_layers);
            self.core_layers.modulate_gateway_frame_branchless(
                &mut sys_baseline,
                &self.last_horizon,
                &self.last_overlap_report,
            );
            let (mut ego_frame, _ego_w) = signal.synthesize_gateway_with_core_layers(
                &mut self.core_layers,
                &self.last_boundary_delta,
                &self.last_horizon,
                &self.last_overlap_report,
            );
            self.core_layers.liminal.active_aperture_mask =
                self.system_spine.mask_ego_aperture_edits(
                    prev_liminal.active_aperture_mask,
                    self.core_layers.liminal.active_aperture_mask,
                );
            let (d_q7, hops, pframe_matched_mask) = self
                .ego_differential_net
                .evaluate_pframe_chain_with_mask_branchless(
                    &self.core_layers.ego,
                    &self.system_spine,
                    &self.last_lacuna_partition,
                    &self.core_layers.stigmergy,
                    &self.distortion_tension,
                    &mut ego_frame,
                );
            if hops > 0 {
                self.ego_differential_net
                    .record_matched_activations(pframe_matched_mask);
            }
            self.system_spine
                .enforce_pinned_isolation_branchless(&sys_baseline, &mut ego_frame);
            (ego_frame, d_q7, hops)
        } else {
            let mut base_frame =
                signal.synthesize_gateway_with_aperture_from_stack(&self.core_layers);
            self.core_layers.modulate_gateway_frame_branchless(
                &mut base_frame,
                &self.last_horizon,
                &self.last_overlap_report,
            );
            (base_frame, 0, 0)
        };
        let mut frm_w = *frame.phase_words_ref();
        self.last_boundary_delta = LiminalBoundaryDelta::compute_branchless(
            &prev_liminal,
            &self.core_layers.liminal,
            prev_overlap,
            self.last_overlap_report.overlap_core_mask,
            &self.last_boundary_delta,
        );

        // 2. Evaluate stability / anomaly trigger with dwell hysteresis and slide slots in/out.
        let entropy_st = self.entropy_controller.state();
        let mut trigger = HorizonTriggerInput {
            entropy_gate_mode: entropy_st.active_branch_mode,
            moving_z_entropy: entropy_st.moving_z_entropy,
            repulsion_severity: frame.repulsion_severity,
            causal_root_cause: frame.attribution_tag,
            epistemic_regime: self.last_horizon.regime_code,
            stationary_shortcut_ready: self.last_boundary_delta.stationary_shortcut_ready != 0,
            ego_active_triggers: self.core_layers.ego.last_intervened_mask,
            unstable_dwell_steps: 0,
            quiescent_dwell_steps: 0,
        };
        let (unstable_dwell, quiescent_dwell) = self
            .horizon_dwell
            .step_branchless(trigger.is_destabilized(), trigger.stationary_shortcut_ready);
        trigger.unstable_dwell_steps = unstable_dwell;
        trigger.quiescent_dwell_steps = quiescent_dwell;

        let (slid_in, slid_out) = apply_latent_horizon_slots(
            latent_slots,
            &mut self.active_horizon_slots_mask,
            &trigger,
            &mut frame,
            &mut self.core_layers,
        );
        if slid_in != 0 || slid_out != 0 || self.active_horizon_slots_mask != 0 {
            frm_w = *frame.phase_words_ref();
        }

        // 3. Cold-path escalation swap-in when destabilized, or cleanly retract the leased lane.
        let is_conflict = trigger.epistemic_regime == EpistemicRegime::InternalConflict as u8
            || trigger.repulsion_severity > 0.25;
        let esc_req = HorizonEscalationRequest {
            trigger,
            placement: WorkbenchLanePlacement::AutoFree,
            lane_flags: if is_conflict {
                WORKBENCH_LANE_FLAG_REPULSION
            } else {
                0
            },
            layer_scale_q7: 64,
            desired_aperture_mask: self.core_layers.liminal.active_aperture_mask,
            current_frame_energy: frame.effective_energy,
        };
        let escalated = self.substrate.apply_horizon_escalation(
            &esc_req,
            &mut self.escalated_workbench_lane,
            escalation_provider,
        )?;

        // 4. Step substrate pointer trail + Bloom filter + RoPE phase steering (zero-alloc).
        let bloom_snapshot = self.substrate.bloom_filter;
        let mut ledger_slot = self.substrate.ledger_slot;
        let (bypass_hit, _eff_ptr, _density, repulsion_active, post_rep_words) = self
            .substrate
            .step_substrate_with_bloom_repulsion_to_words(
                TaggedPtr(frame.tagged_ptr),
                &mut ledger_slot,
                &bloom_snapshot,
                self.config.repulsion_threshold,
                self.config.repulsion_scale_u7,
                trail_out,
            )?;
        *trail_phase_words_mut(trail_out) = if swar_words_nonzero_64(&frm_w) {
            frm_w
        } else {
            post_rep_words
        };
        trail_out.attractor_pull_force = branchless_min_f32(
            branchless_max_f32(trail_out.attractor_pull_force, frame.effective_energy),
            MAX_ATTRACTOR_PULL_FORCE,
        );

        let mut steered = false;
        if (frame.action_mode_flags & ACTION_MODE_LAYER_SKIP) == 0 {
            if let Some(q) = query_tensor.as_deref_mut() {
                let final_w = *trail_phase_words_ref(trail_out);
                if swar_words_nonzero_64(&final_w) {
                    let mut q7_stack = AlignedBuffer16::new([0i8; MAX_ATTENTION_HEADS]);
                    store_q7_words_64(&mut q7_stack.data, &final_w);
                    let p_aligned = AlignedSlice16::new(&q7_stack.data[..num_heads])?;
                    if let Some(active_deltas) = Scoped::from_boundary(PhaseDeltaSlice(p_aligned))
                        .cut_by_lacuna(frame.phase_clamp_rad > 0.0, frame.phase_clamp_rad, 0.0)
                        .into_active()
                    {
                        self.phase_steering
                            .apply_phase_steering_for_geometry_safe_active(
                                geometry,
                                q,
                                active_deltas,
                                frame.phase_clamp_rad,
                            )?;
                        steered = true;
                    }
                }
            }
        }

        let metric = branchless_select_u8(bypass_hit, LEDGER_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(steered, LEDGER_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(repulsion_active, LEDGER_METRIC_REPULSION_APPLIED, 0);
        ledger_slot.push_telemetry_metric(metric);
        self.substrate.ledger_slot = ledger_slot;

        let step_decision = if repulsion_active {
            StigmergicActionDecision::RepulsionEscape
        } else if steered && (frame.action_mode_flags & ACTION_MODE_SUBHEAD_ROUTE) != 0 {
            StigmergicActionDecision::SubHeadRoute
        } else if steered {
            StigmergicActionDecision::ExpressPhaseSteer
        } else if (frame.action_mode_flags & ACTION_MODE_LAYER_SKIP) != 0 {
            StigmergicActionDecision::InertialLayerSkip
        } else {
            StigmergicActionDecision::PassiveSidecar
        };
        let qos_throttled = frame.attribution_tag == ROOT_CAUSE_SYSTEM_QOS_THROTTLE;
        let _ = self.telemetry.evaluate_policy(
            trigger.is_destabilized(),
            quiescent_dwell,
            qos_throttled,
        );
        let exit_ent_q7 =
            (branchless_clamp_f32(entropy_st.moving_z_entropy, 0.0, 1.0) * 127.0) as u8;
        let guard_prox_q7 =
            (branchless_clamp_f32(frame.repulsion_severity, 0.0, 1.0) * 127.0) as u8;
        let active_overdrive = matches!(step_decision, StigmergicActionDecision::RepulsionEscape)
            && self.core_layers.ego.is_active();
        self.distortion_tension.step_hot_path_branchless(
            pframe_delta_q7,
            self.last_lacuna_partition.boundary_sensitivity_q7,
            exit_ent_q7,
            guard_prox_q7,
            active_overdrive,
        );
        let step_tag = AttributionTag::new(
            if self.last_lacuna_partition.is_sandbox_eligible()
                || self.core_layers.stigmergy.overlap_mask != 0
            {
                PrimitiveKind::Stigmergy
            } else {
                PrimitiveKind::Liminal
            },
            if pframe_hops > 0 || self.core_layers.ego.last_intervened_mask != 0 {
                DriverKind::Ego
            } else {
                DriverKind::System
            },
        );
        self.telemetry.record_two_by_two_step(
            step_tag.raw(),
            self.system_spine.whole_token_bypass_active != 0,
            &self.last_lacuna_partition,
            &self.distortion_tension,
            pframe_hops,
        );
        self.telemetry.record_step(
            step_decision,
            &self.last_horizon,
            &self.last_boundary_delta,
            &self.last_overlap_report,
            &self.core_layers.ego,
            entropy_st.active_branch_mode,
            slid_in,
            slid_out,
            escalated,
            qos_throttled,
        );
        let _ = self.telemetry.update_assimilation_ema(
            entropy_st.moving_z_entropy,
            frame.fact_persona_coherence,
            frame.repulsion_severity,
            &self.probe_baseline,
            0.15_f32,
        );
        let exterior_granule_mask = BitCut8::split(
            self.last_overlap_report.lacuna_exterior_mask
                | self.last_overlap_report.discovery_excursion_mask,
            self.last_overlap_report.overlap_core_mask,
        )
        .outside;
        let sb_event = self.substrate.step_sandbox_hypothesis_lifecycle(
            &mut self.telemetry,
            &mut self.sandbox_workbench_lane,
            step_decision,
            self.core_layers.ego.is_active(),
            exterior_granule_mask,
            trail_phase_words_ref(trail_out),
            frame.effective_energy,
            3,
            3,
        );
        self.last_sandbox_event = sb_event;
        if matches!(sb_event, SandboxLifecycleEvent::Settled(_)) {
            self.sandbox_settlements = self.sandbox_settlements.saturating_add(1);
        }

        Ok((
            frame,
            frame.to_causal_report(),
            slid_in,
            slid_out,
            escalated,
        ))
    }
}
