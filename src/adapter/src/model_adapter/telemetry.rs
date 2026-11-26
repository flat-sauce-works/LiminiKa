// name: src/adapter/src/model_adapter/telemetry.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Accessors, hot-path tension stepping, bypass/compute dispatch, and telemetry methods on [`ModelSteeringAdapter`].

use super::super::*;
use super::ModelSteeringAdapter;

/// Runtime observability summary projecting [`CoreTelemetrySnapshot`] counters onto
/// the `liminal` Abstraction <-> Concretion coordinate system.
///
/// Every counter maps directly onto `v0.3.0` [`CoreTelemetryBank`] fields via [`CoreTelemetrySnapshot`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LiminalBoundaryFlowSummary {
    /// Total monotonic runtime steps (`spine.total_steps`).
    pub total_steps: u64,
    /// Continuous `Standard`/`Detailed` steps (`spine.continuous_standard_steps`).
    pub continuous_standard_steps: u64,
    /// Abstract ground-state steps: whole-token pre-inference bypass (`whole_token_bypass_events`).
    pub whole_token_bypass_events: u64,
    /// Abstract ground-state steps: Layer 0 stationary shortcut eligibility (`liminal.stationary_shortcut_hits`).
    pub stationary_shortcut_hits: u64,
    /// Abstract ground-state steps: `Lacuna::Unobserved` zero-aperture locks (`unobserved_aperture_zeros`).
    pub unobserved_aperture_zeros: u64,
    /// Completed decode tokens executed in abstract whole-token bypass (`abstract_token_count`).
    pub abstract_token_count: u64,
    /// Completed decode tokens executed through concrete layers (`concrete_token_count`).
    pub concrete_token_count: u64,
    /// Decode residency transitions (`abstract <-> concrete`, `boundary_churn_count`).
    pub boundary_churn_count: u64,
    /// Host D2H logits observation skips (`logits_skip_count`).
    pub logits_skip_count: u64,
    /// Transient contradiction-guard pin relaxations (`guard_relaxation_count`).
    pub guard_relaxation_count: u64,
    /// Exit instability and `Lacuna::Undefined` recoveries (`instability_recovery_count`).
    pub instability_recovery_count: u64,
    /// Boundary gate transitions: dynamic horizon slide-ins (`liminal.slide_in_count`).
    pub slide_in_count: u64,
    /// Boundary gate transitions: dynamic horizon slide-outs (`liminal.slide_out_count`).
    pub slide_out_count: u64,
    /// Boundary gate transitions: chatter events (`liminal.chatter_events`).
    pub chatter_events: u64,
    /// Concrete excited-state steps: Layer 2 Ego interventions (`ego.intervention_count`).
    pub ego_interventions: u64,
    /// Concrete excited-state steps: P-Frame differential chain evaluations (`pframe_chain_evals`).
    pub pframe_chain_evals: u64,
    /// Concrete excited-state steps: cumulative P-Frame hops traversed (`pframe_total_hops`).
    pub pframe_total_hops: u64,
    /// Concrete excited-state steps: `Lacuna::Exterior` sandbox lane stagings (`exterior_sandbox_stages`).
    pub exterior_sandbox_stages: u64,
    /// Concrete-to-Abstract settlements: verified sandbox settlements (`sandbox_settlements()`).
    pub sandbox_settlements: u64,
    /// Homeostatic pull toward abstraction: Hot-Path immediate tension damps (`immediate_tension_damps`).
    pub immediate_tension_damps: u64,
    /// Homeostatic push toward concretion: Slow-Loop stagnation excitations (`slow_tension_excites`).
    pub slow_tension_excites: u64,
    /// Active phase anchor population (`topology.active_anchors`).
    pub active_anchors: u8,
    /// Ratio of abstract ground-state hits (`stationary_shortcut_hits + unobserved_aperture_zeros`)
    /// to `continuous_standard_steps` (clamped to `[0.0, 1.0]`, `0.0` when denominator is `0`).
    pub abstract_residency_ratio: f32,
    /// Ratio of verified `sandbox_settlements` to `exterior_sandbox_stages`
    /// (clamped to `[0.0, 1.0]`, `0.0` when denominator is `0`).
    pub settlement_ratio: f32,
    /// Homeostatic balance ratio `immediate_tension_damps / (immediate_tension_damps + slow_tension_excites)`
    /// (`0.0` when denominator is `0`).
    pub tension_balance_ratio: f32,
    /// Ratio of `ego_interventions` to `continuous_standard_steps` (`0.0` when denominator is `0`).
    pub ego_excitation_ratio: f32,
}

#[inline]
fn safe_clamped_ratio(numerator: u64, denominator: u64) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        (numerator as f64 / denominator as f64).clamp(0.0, 1.0) as f32
    }
}

impl ModelSteeringAdapter {
    /// Returns a shared reference to the embedded 5,120-byte [`RuntimeContext`].
    #[must_use]
    pub fn runtime_context(&self) -> &RuntimeContext {
        &self.ctx
    }

    /// Returns a mutable reference to the embedded 5,120-byte [`RuntimeContext`].
    #[cfg(test)]
    pub(crate) fn runtime_context_mut(&mut self) -> &mut RuntimeContext {
        &mut self.ctx
    }

    /// Re-anchors the baseline pressure reference of [`AnonymousComputePool`] in Q0.15 (`0..=32767`).
    pub fn reanchor_compute_pressure(&mut self, baseline_q15: u16) {
        self.ctx
            .compute_pool_mut()
            .reanchor_baseline_pressure(baseline_q15);
    }

    /// Resets per-turn decode bypass state and clears `last_entropy_h_tilde` before a turn starts.
    pub fn reset_turn_decode_state(&mut self) {
        self.ctx.reset_turn_bypass_state();
        self.ctx.substrate_mut().last_entropy_h_tilde = 0.0;
    }

    /// Returns the per-regime [`StigmergicActionDecision`] invocation counts (`0..=6`) from
    /// Core Layer 1 (`StigmergicLayerTelemetry`).
    #[must_use]
    pub fn substrate_decision_counts(&self) -> &[u64; 7] {
        &self.ctx.telemetry_bank().stigmergy.decision_counts
    }

    /// Resets the `2 Primitives x 2 Drivers` telemetry bank and restores initial neutral entropy (`0.35`)
    /// for a fresh evaluation pass.
    pub fn reset_substrate_stats(&mut self) {
        self.ctx.reset_telemetry();
        self.ctx.substrate_mut().last_entropy_h_tilde = 0.35;
    }

    /// Resets core layers, horizon dwell/slots, spillover cache, thresholds, and `2 Primitives x 2 Drivers`
    /// telemetry back to clean defaults for fair ablation pass isolation.
    pub fn reset_horizon_and_core_state(&mut self) {
        let saved_hops = self.ctx.ego_differential_net().max_hops;
        *self.ctx.core_layers_mut() = CoreStateStack::new();
        self.configured_ego = LiminalEgoState::new();
        *self.ctx.last_boundary_delta_mut() = LiminalBoundaryDelta::default();
        *self.ctx.last_horizon_mut() = EpistemicHorizonReading {
            regime_code: EpistemicRegime::KnownCoherent as u8,
            ..EpistemicHorizonReading::default()
        };
        *self.ctx.last_overlap_report_mut() = OverlapDiscoveryReport::default();
        self.last_ensemble_report = None;
        *self.ctx.horizon_dwell_mut() = HorizonDwellState::default();
        *self.ctx.active_horizon_slots_mask_mut() = 0;
        self.coherence_guard_slot = CoherenceHorizonGuardSlot::default();
        *self.residual_spillover = DefaultResidualSpilloverSlot::default();
        self.threshold_policy.base = self.calibrated_base_thresholds;
        {
            let sub = self.ctx.substrate_mut();
            sub.thresholds = self.calibrated_base_thresholds;
            sub.last_decision = 0;
            sub.last_anomaly_severity = 0.0;
            let _ = sub.release_all_lanes();
        }
        self.escalation_boundary_dirty = true;
        self.last_escalated_regime = EpistemicRegime::KnownCoherent as u8;
        *self.ctx.escalated_workbench_lane_mut() = None;
        *self.ctx.sandbox_workbench_lane_mut() = None;
        self.active_workbench_preset = WorkbenchPreset::BalancedDialog;
        let preserved_base = self.ctx.system_spine().baseline_pinned_compute_mask;
        let preserved_hold = self.ctx.system_spine().guard_hold_tokens;
        let preserved_stride = self.ctx.system_spine().max_observe_stride;
        let preserved_pre_bypass = self.ctx.system_spine().is_pre_bypass_enabled();
        let mut fresh_spine = SystemSpineState::new();
        fresh_spine.set_baseline_pin_mask(preserved_base);
        fresh_spine.guard_hold_tokens = preserved_hold;
        fresh_spine.set_max_observe_stride(preserved_stride);
        fresh_spine.set_pre_bypass_enabled(preserved_pre_bypass);
        *self.ctx.system_spine_mut() = fresh_spine;
        *self.ctx.last_lacuna_partition_mut() = LacunaPartition8::default();
        *self.ctx.distortion_tension_mut() = DistortionTension::new(false);
        *self.ctx.ego_differential_net_mut() = EgoDifferentialNet::new_empty(saved_hops);
        self.sandbox_settlement_streak = 0;
        self.prev_sandbox_coherence = 0.0;
        self.prev_sandbox_assim_drift = 1.0;
        self.crystallized_anchors.clear();
        self.ctx.set_sandbox_settlements(0);
        self.ctx.reset_turn_bypass_state();
        let policy = self.ctx.telemetry_bank().policy();
        self.ctx.set_telemetry_policy(policy);
        self.reset_substrate_stats();
    }

    /// Returns a shared reference to the `2 Primitives x 2 Drivers` [`CoreTelemetryBank`].
    #[must_use]
    pub fn telemetry_bank(&self) -> &CoreTelemetryBank {
        self.ctx.telemetry_bank()
    }

    /// Returns a shared reference to the label-free [`AnonymousComputePool`].
    #[must_use]
    pub fn compute_pool(&self) -> &AnonymousComputePool {
        self.ctx.compute_pool()
    }

    /// Returns the 64-byte unified [`CorePrimitiveState`] (`Liminal` + `Stigmergy`).
    #[must_use]
    pub fn core_primitive_state(&self) -> CorePrimitiveState {
        self.ctx.core_layers().core_primitive_state()
    }

    /// Returns a shared reference to the mechanical [`SystemSpineState`] (`pinned` isolation barrier).
    #[must_use]
    pub fn system_spine(&self) -> &SystemSpineState {
        self.ctx.system_spine()
    }

    /// Returns the most recent branchless 4-region [`LacunaPartition8`].
    #[must_use]
    pub fn last_lacuna_partition(&self) -> &LacunaPartition8 {
        self.ctx.last_lacuna_partition()
    }

    /// Returns a shared reference to the asymmetric homeostatic [`DistortionTension`] controller.
    #[must_use]
    pub fn distortion_tension(&self) -> &DistortionTension {
        self.ctx.distortion_tension()
    }

    /// Returns a shared reference to the P-Frame differential chain [`EgoDifferentialNet`].
    #[must_use]
    pub fn ego_differential_net(&self) -> &EgoDifferentialNet {
        self.ctx.ego_differential_net()
    }

    /// Returns cumulative multi-turn Known Unknown ($O \setminus S$) sandbox settlements.
    #[must_use]
    pub fn sandbox_settlements(&self) -> u64 {
        u64::from(self.ctx.sandbox_settlements())
    }

    /// Updates the maximum hop bound (`1..=4`) of [`EgoDifferentialNet`] and rebuilds active P-Frame nodes.
    pub fn set_ego_max_hops(&mut self, max_hops: u8) {
        self.ctx.ego_differential_net_mut().max_hops = max_hops.clamp(1, 4);
        if self.configured_ego.is_active() {
            self.sync_ego_differential_net_from_configured();
        }
    }

    /// Advances 1 Hot-Path [`DistortionTension`] step using the latest P-Frame delta norm and boundary sensitivity.
    pub fn step_distortion_tension_hot_path(
        &mut self,
        exit_entropy_q7: u8,
        guard_proximity_q7: u8,
        active_overdrive: bool,
    ) {
        let delta_norm_q7 = self.ctx.ego_differential_net().last_delta_norm_q7;
        let sens_q7 = self.ctx.last_lacuna_partition().boundary_sensitivity_q7;
        self.ctx.distortion_tension_mut().step_hot_path_branchless(
            delta_norm_q7,
            sens_q7,
            exit_entropy_q7,
            guard_proximity_q7,
            active_overdrive,
        );
    }

    #[inline(always)]
    fn external_guard_active(&self) -> bool {
        (self.coherence_guard_slot.activations > self.coherence_guard_slot.retractions)
            && (self.ctx.last_horizon().conflict_severity > 0.25)
    }

    /// Initializes per-turn mechanical spine policy on [`RuntimeContext`].
    pub(crate) fn begin_turn_spine_policy(&mut self, guard_turn: bool, unobserved_locked: bool) {
        self.ctx
            .begin_turn_spine_policy(guard_turn, unobserved_locked);
    }

    /// Evaluates one 4-stage decode control step in [`RuntimeContext`] with adapter guard context.
    #[must_use]
    pub fn step_decode_control<S: SpineDecodeSignal>(&mut self, signal: &S) -> SpineStepPlan {
        let external_guard = self.external_guard_active();
        self.ctx.step_decode_control(signal, external_guard)
    }

    /// Configures the decode-token hold window before a transient contradiction-guard compute pin
    /// may relax toward low-cost bypass (`0` holds the entire turn).
    pub fn set_guard_hold_tokens(&mut self, tokens: u8) {
        self.ctx.system_spine_mut().guard_hold_tokens = tokens;
    }

    /// Returns the configured decode-token hold window before transient guard relaxation (`0` = full turn).
    #[must_use]
    pub fn guard_hold_tokens(&self) -> u8 {
        self.ctx.system_spine().guard_hold_tokens
    }

    /// Configures maximum logits observation stride (`>= 1`, where `1` observes every token).
    pub fn set_max_observe_stride(&mut self, stride: u8) {
        self.ctx.system_spine_mut().set_max_observe_stride(stride);
    }

    /// Returns effective maximum logits observation stride (`>= 1`).
    #[must_use]
    pub fn max_observe_stride(&self) -> u8 {
        self.ctx.system_spine().effective_max_observe_stride()
    }

    /// Enables or disables pre-inference whole-token steering bypass on [`SystemSpineState`].
    pub fn set_pre_bypass_enabled(&mut self, enabled: bool) {
        self.ctx.system_spine_mut().set_pre_bypass_enabled(enabled);
    }

    /// Returns `true` when pre-inference whole-token steering bypass is enabled.
    #[must_use]
    pub fn is_pre_bypass_enabled(&self) -> bool {
        self.ctx.system_spine().is_pre_bypass_enabled()
    }

    /// Returns `true` when the active turn's transient contradiction-guard pin has been relaxed.
    #[must_use]
    pub fn is_transient_guard_relaxed(&self) -> bool {
        self.ctx.system_spine().is_transient_guard_relaxed()
    }

    /// Resolves step/turn-local effective `(guard_active, repulsion_active)` signals for
    /// `SystemSpineState` bypass veto gating without latching onto session-persistent
    /// `StigmergicSubstrate::slot_repulsion_mask`.
    #[must_use]
    pub fn effective_spine_guard_and_repulsion(&self, obs_repulsion: bool) -> (bool, bool) {
        let spine = self.ctx.system_spine();
        let static_guard = !spine.is_transient_guard_relaxed()
            && (spine.turn_guard_active() || self.external_guard_active());
        let guard_active = static_guard || spine.passive_saturation_probe_steps > 0;
        let action_repulse = liminika_core::stigmergy::StigmergicActionDecision::from_u8(
            self.ctx.substrate().last_decision,
        )
            == liminika_core::stigmergy::StigmergicActionDecision::RepulsionEscape;
        let entropy_repulse = self.ctx.entropy_controller().is_repulsion_mode();
        let repulsion_active = obs_repulsion || action_repulse || entropy_repulse;
        (guard_active, repulsion_active)
    }

    /// Evaluates whether a transient contradiction-guard compute pin may safely relax toward
    /// low-cost bypass after holding the opening window, or re-escalates immediately if
    /// repulsion or exit entropy instability returns after relaxation.
    #[must_use]
    pub fn evaluate_transient_guard_relaxation(
        &mut self,
        last_observed_entropy: f32,
        liminal_stationary: u8,
        repulsion_active: bool,
        exit_entropy_rising: bool,
        min_calm_dwell: u8,
    ) -> bool {
        let (_, eff_repulsion) = self.effective_spine_guard_and_repulsion(repulsion_active);
        let ext_guard = self.external_guard_active();
        if self.ctx.system_spine().is_transient_guard_relaxed() {
            if eff_repulsion || exit_entropy_rising {
                let active_mask = self.ctx.compute_pool().active_mask;
                {
                    let spine = self.ctx.system_spine_mut();
                    spine.set_transient_guard_relaxed(false);
                    spine.set_turn_guard_active(true);
                    spine.arm_transient_guard_pin(active_mask);
                    spine.reset_turn_bypass_counters_only();
                    spine.unobserved_bypass_run = 0;
                }
                self.ctx.compute_pool_mut().reset_turn_bypass_state();
            }
            return false;
        }

        let hold_tokens = self.ctx.system_spine().guard_hold_tokens;
        let has_static_guard = self.ctx.system_spine().turn_guard_active() || ext_guard;
        if !has_static_guard
            || hold_tokens == 0
            || self.ctx.system_spine().passive_saturation_probe_steps > 0
            || self.ctx.system_spine().unobserved_locked != 0
            || !last_observed_entropy.is_finite()
            || last_observed_entropy >= 0.72
        {
            return false;
        }

        if !self.ctx.can_relax_transient_guard_pin(
            hold_tokens,
            liminal_stationary,
            eff_repulsion,
            exit_entropy_rising,
            min_calm_dwell,
        ) {
            return false;
        }

        {
            let spine = self.ctx.system_spine_mut();
            spine.set_transient_guard_relaxed(true);
            spine.set_turn_guard_active(false);
            spine.restore_baseline_pin_mask();
        }
        self.ctx.on_transient_guard_relaxed(min_calm_dwell);
        self.ctx.telemetry_bank_mut().record_guard_relaxation();
        true
    }

    /// Evaluates token-level whole-steering bypass on `obs` jointly with [`SystemSpineState`]
    /// physical `pinned` / guard / repulsion vetoes, returning the active `SPINE_BYPASS_VETO_*`
    /// bitmask.
    #[must_use]
    pub fn evaluate_whole_token_bypass_with_spine(
        &mut self,
        mut obs: TokenBypassObservation,
        dispatch: &mut ComputeDispatchMask,
    ) -> u8 {
        let (guard_active, repulsion_active) =
            self.effective_spine_guard_and_repulsion(obs.repulsion_active != 0);
        obs.repulsion_active = u8::from(repulsion_active);
        let spine = self.ctx.system_spine_mut();
        spine.passive_saturation_probe_steps =
            spine.passive_saturation_probe_steps.saturating_sub(1);
        self.ctx
            .evaluate_whole_token_bypass_with_spine(obs, guard_active, dispatch)
    }

    /// Evaluates token-level whole-steering bypass on `obs`, mutating `dispatch` and mirroring
    /// active bypass state onto [`SystemSpineState`].
    pub fn evaluate_whole_token_bypass(
        &mut self,
        obs: TokenBypassObservation,
        dispatch: &mut ComputeDispatchMask,
    ) -> bool {
        // Intentional discard: the veto bitmask is not needed by this wrapper, which reports
        // `dispatch.whole_token_steering_bypass` as its boolean result instead.
        let _ = self.evaluate_whole_token_bypass_with_spine(obs, dispatch);
        dispatch.whole_token_steering_bypass != 0
    }

    /// Exports active [`EgoDifferentialNet`] P-Frame nodes as model-agnostic
    /// [`LogicalPFrameDescriptor`]s for unified container persistence.
    #[must_use]
    pub fn export_logical_pframe_topology(&self) -> Vec<LogicalPFrameDescriptor> {
        let net = self.ctx.ego_differential_net();
        let mut out = Vec::new();
        for (idx, (routing, cold)) in net
            .node_routing
            .iter()
            .zip(net.node_cold.iter())
            .enumerate()
        {
            if ((net.active_node_mask >> idx) & 1) != 0 && routing.is_active() {
                out.push(LogicalPFrameDescriptor {
                    node_id: cold.node_id,
                    anchor_ref_id: cold.anchor_ref_id,
                    next_hop_idx: routing.next_hop_idx,
                    hop_depth: routing.hop_depth,
                    lacuna_trigger_mask: routing.lacuna_trigger_mask,
                    scope_region_selector: routing.scope_region_selector,
                    pheromone_q7: routing.pheromone_q7,
                    gain_q7: routing.gain_q7,
                    intent_delta: cold.intent_delta,
                });
            }
        }
        out
    }

    /// Re-bakes persisted [`LogicalPFrameDescriptor`]s into this model's 64-head Q7
    /// [`EgoDifferentialNet`] upon unified container load or cross-model transcoding.
    pub fn bake_logical_pframe_topology(&mut self, descriptors: &[LogicalPFrameDescriptor]) {
        let mut net = EgoDifferentialNet::new_empty(self.ctx.ego_differential_net().max_hops);
        for (slot, desc) in descriptors
            .iter()
            .take(liminika_core::drivers::MAX_EGO_PFRAME_NODES)
            .enumerate()
        {
            let local_ego = if desc.intent_delta.is_active() {
                LiminalEgoState::from_intent_axes(&desc.intent_delta)
            } else {
                self.configured_ego
            };
            let shift_base = local_ego.transgression_bias_q7;
            let mut phases = [0 as lmnk_q7_t; MAX_ATTENTION_HEADS];
            for (g, &gv_q7) in local_ego
                .sovereign_granule_phase_q7
                .iter()
                .enumerate()
                .take(NUM_STIGMERGIC_GRANULES)
            {
                let gv = i32::from(gv_q7);
                let signed = if (desc.hop_depth & 1) == 0 && ((g & 1) == 1) {
                    -gv
                } else {
                    gv
                };
                let head_shift = match desc.hop_depth {
                    1 => 4_i32,
                    2 => i32::from(shift_base >> 2),
                    _ => -i32::from(shift_base >> 2),
                };
                let base_s = (signed + head_shift).clamp(-127, 127);
                let base_h = g << 3;
                for k in 0..8 {
                    let mod_off = ((k as i32) - 3) * (head_shift >> 4);
                    phases[base_h + k] = (base_s + mod_off).clamp(-127, 127) as lmnk_q7_t;
                }
            }
            let mut node = DistortionDelta::new_empty(desc.node_id, desc.anchor_ref_id);
            node.next_hop_idx = desc.next_hop_idx;
            node.hop_depth = desc.hop_depth;
            node.lacuna_trigger_mask = desc.lacuna_trigger_mask;
            node.scope_region_selector = desc.scope_region_selector;
            node.pheromone_q7 = desc.pheromone_q7;
            node.gain_q7 = desc.gain_q7;
            node.intent_delta = desc.intent_delta;
            node.delta_phase_q7 = phases;
            node.flags = DISTORTION_DELTA_FLAG_ACTIVE | DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID;
            net.set_node(slot, node);
        }
        *self.ctx.ego_differential_net_mut() = net;
    }

    fn sync_ego_differential_net_from_configured(&mut self) {
        let axes = self.configured_ego.extract_intent_axes();
        let max_hops = self.ctx.ego_differential_net().max_hops;
        *self.ctx.ego_differential_net_mut() =
            EgoDifferentialNet::from_ego_state_and_axes(&self.configured_ego, &axes, max_hops);
        *self.ctx.distortion_tension_mut() =
            DistortionTension::new(self.configured_ego.is_active());
    }

    /// Evaluates external `+ System` Gauge-Zero pressure jointly with [`SystemSpineState`]
    /// physical `pinned` / guard / repulsion vetoes and emits a pre-execution
    /// [`ComputeDispatchMask`] for this adapter's anonymous compute slots.
    #[must_use]
    pub fn step_compute_dispatch_with_spine(
        &mut self,
        reading: SystemPressureReading,
        repulsion_active: bool,
    ) -> ComputeDispatchMask {
        let (guard_active, eff_repulsion) =
            self.effective_spine_guard_and_repulsion(repulsion_active);
        let spine = *self.ctx.system_spine();
        self.ctx.compute_pool_mut().step_dispatch_with_spine(
            reading,
            &spine,
            guard_active,
            eff_repulsion,
        )
    }

    /// Evaluates external `+ System` Gauge-Zero pressure and emits a pre-execution
    /// [`ComputeDispatchMask`] for this adapter's anonymous compute slots.
    #[must_use]
    pub fn step_compute_dispatch(&mut self, reading: SystemPressureReading) -> ComputeDispatchMask {
        self.step_compute_dispatch_with_spine(reading, false)
    }

    /// Updates slot contribution pheromones strictly from an exit scalar reward (`0..=127`)
    /// without mid-layer D2H synchronization.
    pub fn commit_compute_exit_reward(&mut self, executed_mask: u64, reward_q7: u8) {
        self.ctx
            .compute_pool_mut()
            .commit_exit_reward(executed_mask, reward_q7);
    }

    /// Triggers immediate 1-step full-layer recovery across the compute pool and [`SystemSpineState`]
    /// with asymmetric re-bypass dwell hysteresis (called on exit entropy spike, `InternalConflict`, or
    /// `LiminalExterior`).
    pub fn signal_compute_instability(&mut self) {
        self.ctx.signal_compute_instability();
    }

    /// Updates the [`TelemetryCapturePolicy`] (`Pinned` vs `Auto` `Coarse`/`Standard`/`Detailed`).
    pub fn set_telemetry_policy(&mut self, policy: TelemetryCapturePolicy) {
        self.ctx.set_telemetry_policy(policy);
    }

    /// Pulls a Lazy-Pull [`CoreTelemetrySnapshot`] computed inside `liminika-core`.
    #[must_use]
    pub fn telemetry_snapshot(&self) -> CoreTelemetrySnapshot {
        self.ctx.telemetry_snapshot()
    }

    /// Evaluates bounded Slow-Loop epoch threshold adjustments over continuous `Standard`/`Detailed`
    /// samples without touching Hot-Path loops.
    #[must_use]
    pub fn evaluate_slow_loop_telemetry(
        &mut self,
        min_samples: u64,
    ) -> SlowLoopTelemetryAdjustment {
        let base = self.calibrated_base_thresholds;
        self.ctx
            .telemetry_bank_mut()
            .evaluate_slow_loop_epoch(&base, min_samples)
    }

    fn push_crystallized_anchor(&mut self, record: PhaseAnchorRecord) {
        if self.crystallized_anchors.len() >= Self::MAX_CRYSTALLIZED_ANCHOR_BUFFER {
            self.crystallized_anchors.remove(0);
        }
        self.crystallized_anchors.push(record);
    }

    /// Returns Cold-Path [`PhaseAnchorRecord`]s consolidated by [`EgoDifferentialNet`] during
    /// Slow-Loop epochs (bounded FIFO up to [`Self::MAX_CRYSTALLIZED_ANCHOR_BUFFER`]).
    #[must_use]
    pub fn crystallized_anchors(&self) -> &[PhaseAnchorRecord] {
        &self.crystallized_anchors
    }

    /// Drains and returns all buffered Cold-Path crystallized [`PhaseAnchorRecord`]s in FIFO order.
    #[must_use]
    pub fn drain_crystallized_anchors(&mut self) -> Vec<PhaseAnchorRecord> {
        std::mem::take(&mut self.crystallized_anchors)
    }

    /// Exports the embedded [`RuntimeContext`] into a validated 512-byte standalone `.lmnk`
    /// snapshot container (`LMNK_SECTION_RUNTIME_SNAPSHOT`) without exposing mutable internals.
    ///
    /// # Errors
    /// Returns an error if core snapshot serialization fails.
    pub fn export_runtime_snapshot(
        &self,
    ) -> Result<[u8; LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES], String> {
        let mut buf = [0_u8; LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES];
        let mut req = 0_usize;
        self.ctx
            .serialize_to_bytes(&mut buf, &mut req)
            .map_err(|e| format!("RuntimeContext::serialize_to_bytes failed: {e}"))?;
        if req != LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES {
            return Err(format!(
                "unexpected runtime snapshot byte length: {req} != {LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES}"
            ));
        }
        Ok(buf)
    }

    /// Restores [`RuntimeContext`] state from a 512-byte snapshot container (or multi-section
    /// `.lmnkc` containing `LMNK_SECTION_RUNTIME_SNAPSHOT`) while strictly enforcing this host's
    /// monotonic `Spine` pin floor (`apply_host_spine_pin_floor`) so deserialized bytes can never
    /// weaken irreversible boundary pins.
    ///
    /// # Errors
    /// Returns an error if container validation, topology matching, or snapshot decoding fails.
    pub fn restore_runtime_snapshot(&mut self, bytes: &[u8]) -> Result<(), String> {
        let (active_slots, geom_pinned_slots, _) =
            Self::compute_slot_masks(self.geometry.num_layers);
        let host_pin_floor =
            geom_pinned_slots | self.ctx.system_spine().baseline_pinned_compute_mask;
        self.ctx
            .deserialize_from_bytes(bytes)
            .map_err(|e| format!("RuntimeContext::deserialize_from_bytes failed: {e}"))?;
        self.ctx.apply_host_spine_pin_floor(host_pin_floor);
        let final_pinned = self.ctx.system_spine().baseline_pinned_compute_mask & active_slots;
        let reversible = active_slots & !final_pinned;
        self.ctx
            .compute_pool_mut()
            .set_slot_masks(active_slots, final_pinned, 0, reversible);
        self.configured_ego = self.ctx.core_layers().ego;
        if self.configured_ego.is_active() {
            self.sync_ego_differential_net_from_configured();
        }
        Ok(())
    }

    /// Bridges live adapter state into a domain-agnostic [`EquilibriumState`]: drains all buffered
    /// Cold-Path crystallized anchors, captures the 512-byte runtime snapshot, and reconciles both
    /// via flat renormalization (`EquilibriumState::ingest_crystallized_anchors`).
    ///
    /// # Errors
    /// Returns an error if snapshot export, topology validation, or equilibrium reconciliation fails.
    #[must_use = "EquilibriumJoinReport summarizes convergence and bounded working-set updates"]
    pub fn sync_equilibrium_state(
        &mut self,
        equilibrium: &mut EquilibriumState,
    ) -> Result<EquilibriumJoinReport, String> {
        let snap = self.export_runtime_snapshot()?;
        equilibrium
            .update_runtime_snapshot_from_container(&snap)
            .map_err(|e| e.to_string())?;
        let drained = self.drain_crystallized_anchors();
        let report = equilibrium
            .ingest_crystallized_anchors(&drained)
            .map_err(|e| e.to_string())?;
        Ok(report)
    }

    /// Restores this adapter's [`RuntimeContext`] from an [`EquilibriumState`]'s persisted
    /// `LMNK_SECTION_RUNTIME_SNAPSHOT` (when present) while enforcing host `Spine` pin floors.
    ///
    /// Returns `Ok(true)` when a runtime snapshot was restored and `Ok(false)` for legacy `.lmnkc`
    /// equilibrium files carrying anchor records only.
    ///
    /// # Errors
    /// Returns an error if the stored snapshot fails validation or topology checks.
    pub fn restore_from_equilibrium(
        &mut self,
        equilibrium: &EquilibriumState,
    ) -> Result<bool, String> {
        let Some(snap_512) = equilibrium.runtime_snapshot_container_bytes() else {
            return Ok(false);
        };
        self.restore_runtime_snapshot(&snap_512)?;
        Ok(true)
    }

    /// Evaluates bounded Slow-Loop epoch adjustments and applies them to [`AdaptiveThresholdPolicy`],
    /// [`StigmergicSubstrate`] thresholds, [`DistortionTension`], and [`EgoDifferentialNet`] GOP
    /// maintenance when `applied == true`.
    #[must_use]
    pub fn apply_slow_loop_telemetry_epoch(
        &mut self,
        min_samples: u64,
    ) -> SlowLoopTelemetryAdjustment {
        let adj = self.evaluate_slow_loop_telemetry(min_samples);
        if adj.applied {
            self.threshold_policy.base = adj.tuned_thresholds;
            let neutral_trail = lmnk_pointer_trail_t::default();
            let resolved = {
                let probe = *self.ctx.probe_baseline();
                self.threshold_policy.resolve_thresholds(
                    &self.system_constraint,
                    &probe,
                    &neutral_trail,
                    probe.baseline_entropy,
                )
            };
            self.ctx.substrate_mut().thresholds = resolved;
            let next_dwell = (2_i16 + i16::from(adj.suggested_slide_out_dwell_delta)).clamp(1, 8);
            self.coherence_guard_slot.slide_out_dwell = next_dwell as u16;
            self.apply_distortion_ego_feedback(&adj.distortion);
            let passive_sat = adj.distortion.ego_wakeup_recommended
                || matches!(
                    adj.distortion.kind,
                    liminika_core::runtime::LayerDistortionKind::PassiveSaturation
                );
            if passive_sat {
                let spine = self.ctx.system_spine_mut();
                spine.passive_saturation_probe_steps = spine.passive_saturation_probe_steps.max(2);
            }
            let chatter = self.ctx.last_boundary_delta().chatter_potential_u8;
            let ego_active = self.configured_ego.is_active();
            self.ctx.distortion_tension_mut().step_slow_loop_excitation(
                passive_sat,
                chatter,
                ego_active,
            );
            let next_epoch = {
                let spine = self.ctx.system_spine_mut();
                spine.slow_loop_epoch = spine.slow_loop_epoch.saturating_add(1);
                spine.slow_loop_epoch
            };
            if let Some(anchor) = self
                .ctx
                .ego_differential_net_mut()
                .consolidate_and_evaporate_cold_path(next_epoch)
            {
                self.push_crystallized_anchor(anchor);
            }
        }
        adj
    }

    /// Applies closed-loop `Distortion <-> Ego` homeostatic control:
    /// wakes Ego with `EGO_TRIGGER_LAYER_DISTORTION` on passive stalls/saturation, cools down
    /// Ego thrust/defiance on active overdrive, and restores configured baselines when aligned.
    pub fn apply_distortion_ego_feedback(&mut self, report: &LayerDistortionReport) {
        if !self.configured_ego.is_active() {
            return;
        }
        let ego = &mut self.ctx.core_layers_mut().ego;
        if report.ego_wakeup_recommended {
            ego.trigger_mask |= EGO_TRIGGER_LAYER_DISTORTION | EGO_TRIGGER_BASIN_CAPTURE;
            ego.counter_basin_thrust_q7 = ego.counter_basin_thrust_q7.max(48);
            if ego.transgression_bias_q7.abs() < 24 {
                ego.transgression_bias_q7 = 24;
            }
        } else if report.ego_cooldown_recommended {
            ego.trigger_mask &= !(EGO_TRIGGER_LAYER_DISTORTION | EGO_TRIGGER_INTERNAL_CONFLICT);
            ego.counter_basin_thrust_q7 = self.configured_ego.counter_basin_thrust_q7 >> 1;
            ego.irrational_defiance_q7 = self.configured_ego.irrational_defiance_q7 >> 1;
        } else {
            let epochs = ego.intervention_epochs;
            let last_mask = ego.last_intervened_mask;
            *ego = self.configured_ego;
            ego.intervention_epochs = epochs;
            ego.last_intervened_mask = last_mask;
        }
    }

    /// Returns a shared reference to the embedded [`StigmergicSubstrate`].
    #[must_use]
    pub fn substrate(&self) -> &StigmergicSubstrate {
        self.ctx.substrate()
    }

    /// Returns a mutable reference to the embedded [`StigmergicSubstrate`].
    #[cfg(test)]
    pub(crate) fn substrate_mut(&mut self) -> &mut StigmergicSubstrate {
        self.ctx.substrate_mut()
    }

    /// Returns the validated [`RopeGeometry`].
    #[must_use]
    pub const fn geometry(&self) -> &RopeGeometry {
        &self.geometry
    }

    /// Updates the active RoPE layer bitmask (`1` = steer, `0` = skip NoPE/disabled layer).
    pub fn set_layer_rope_mask(&mut self, mask: u64) {
        self.geometry.layer_rope_mask = mask;
        self.ctx.probe_baseline_mut().steered_layers =
            Self::compute_steered_layer_count(&self.geometry);
    }

    /// Returns the latest Cold-Path [`ManifoldProbeBaseline`] calibrated for the active model.
    #[must_use]
    pub fn probe_baseline(&self) -> &ManifoldProbeBaseline {
        self.ctx.probe_baseline()
    }

    /// Returns the live [`SystemConstraint<()>`] telemetry snapshot.
    #[must_use]
    pub const fn system_constraint(&self) -> &SystemConstraint<()> {
        &self.system_constraint
    }

    /// Writes `circuit` into the embedded [`liminika_core::abi::lmnk_ledger_slot_t`] so every subsequent
    /// [`StigmergicSubstrate`] and the rank-1 residual step modulates [`GatewayFrame`] dynamically from this circuit.
    pub(crate) fn set_active_circuit(&mut self, circuit: CognitiveTrajectoryLedger) {
        circuit.write_to_ledger_slot(self.ctx.ledger_slot_mut());
    }

    /// Clears the active [`CognitiveTrajectoryLedger`] flag and trajectory register from the adapter.
    pub(crate) fn clear_active_circuit(&mut self) {
        let slot = self.ctx.ledger_slot_mut();
        slot.telemetry_mini_ledger = [0u8; 8];
        slot.auto_tune_flags &= !LEDGER_FLAG_COGNITIVE_TRAJECTORY;
    }

    /// Returns the currently active [`CognitiveTrajectoryLedger`] if attached to the adapter's ledger slot.
    #[must_use]
    pub fn active_circuit(&self) -> Option<CognitiveTrajectoryLedger> {
        let slot = self.ctx.ledger_slot();
        if (slot.auto_tune_flags & LEDGER_FLAG_COGNITIVE_TRAJECTORY) != 0 {
            Some(CognitiveTrajectoryLedger::from_ledger_slot(slot))
        } else {
            None
        }
    }

    /// Evaluates `bank` against `(bias_words, input_words)` at `scale`, applies homeostatic
    /// plasticity (`reinforce_bank`), and pushes the consolidated ledger back into the adapter's
    /// live Hot-Path ledger slot while co-adapting both Core Layer 0 (`Liminal`) and Core Layer 1 (`Stigmergic`).
    pub fn reinforce_active_circuit_with_bank<B: CognitiveCircuitBankOps>(
        &mut self,
        bank: &mut B,
        bias_words: &[u64; NUM_STIGMERGIC_GRANULES],
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        scale: StigmergicScaleGranularity,
        converged: bool,
    ) -> CognitiveCircuitActivation {
        let act = bank.evaluate_bank_activation(bias_words, input_words, scale);
        bank.reinforce_bank(&act, converged);
        let ledger = bank.to_consolidated_ledger();
        self.set_active_circuit(ledger);
        let horizon = *self.ctx.last_horizon();
        let overlap = *self.ctx.last_overlap_report();
        self.ctx
            .core_layers_mut()
            .adapt_layers_branchless(&ledger, &horizon, &overlap, None);
        act
    }

    /// Returns a shared reference to the live Core Layer 0 [`LiminalThresholdState`] governing
    /// dynamic `Liminal` channel thresholds and cognitive window gains.
    #[must_use]
    pub fn threshold_state(&self) -> &LiminalThresholdState {
        &self.ctx.core_layers().liminal
    }

    /// Returns a shared reference to the live Core Layer 1 [`StigmergicGranuleState`].
    #[must_use]
    pub fn stigmergic_granule_state(&self) -> &StigmergicGranuleState {
        &self.ctx.core_layers().stigmergy
    }

    /// Returns a shared reference to the unified 3-layer [`CoreStateStack`].
    #[must_use]
    pub fn core_layer_stack(&self) -> &CoreStateStack {
        self.ctx.core_layers()
    }

    /// Returns the most recently observed [`EpistemicHorizonReading`].
    #[must_use]
    pub fn last_horizon(&self) -> &EpistemicHorizonReading {
        self.ctx.last_horizon()
    }

    /// Returns the most recently observed [`OverlapDiscoveryReport`].
    #[must_use]
    pub fn last_overlap_report(&self) -> &OverlapDiscoveryReport {
        self.ctx.last_overlap_report()
    }

    /// Returns a shared reference to the live 64-byte Core Layer 2 [`LiminalEgoState`].
    #[must_use]
    pub fn ego_state(&self) -> &LiminalEgoState {
        &self.ctx.core_layers().ego
    }

    /// Replaces the adapter's active Core Layer 2 [`LiminalEgoState`].
    pub fn set_ego_state(&mut self, ego: LiminalEgoState) {
        self.ctx.core_layers_mut().ego = ego;
        self.configured_ego = ego;
        self.sync_ego_differential_net_from_configured();
    }

    /// Configures Core Layer 2 [`LiminalEgoState`] from a coarse [`LiminalEgoPreset`] (`Custom`, `Low`, `Medium`, `High`).
    pub fn configure_ego_from_preset(&mut self, preset: LiminalEgoPreset) {
        let ego = LiminalEgoState::from_preset(preset);
        self.ctx.core_layers_mut().ego = ego;
        self.configured_ego = ego;
        self.sync_ego_differential_net_from_configured();
    }

    /// Configures Core Layer 2 [`LiminalEgoState`] from intermediate normalized [`EgoIntentAxes`].
    pub fn configure_ego_from_intent_axes(&mut self, axes: &EgoIntentAxes) {
        let ego = LiminalEgoState::from_intent_axes(axes);
        self.ctx.core_layers_mut().ego = ego;
        self.configured_ego = ego;
        self.sync_ego_differential_net_from_configured();
    }

    /// Smoothly blends an [`EgoIntentAxes`] overlay onto the current Ego state (`weight_q7` in `0..=127`).
    pub fn blend_ego_intent_axes(&mut self, overlay: &EgoIntentAxes, weight_q7: u8) {
        let base_axes = self.ctx.core_layers().ego.extract_intent_axes();
        let blended = base_axes.blend_lerp_q7(overlay, weight_q7);
        let ego = LiminalEgoState::from_intent_axes(&blended);
        self.ctx.core_layers_mut().ego = ego;
        self.configured_ego = ego;
        self.sync_ego_differential_net_from_configured();
    }

    /// Derives Core Layer 2 [`LiminalEgoState`] deterministically from a natural-language Ego description.
    pub fn configure_ego_from_natural_language(&mut self, description: &str) {
        let ego = crate::ego_intent::ego_state_from_natural_language(description);
        self.ctx.core_layers_mut().ego = ego;
        self.configured_ego = ego;
        self.sync_ego_differential_net_from_configured();
    }

    /// Overlays natural-language Ego cues onto the existing Core Layer 2 [`LiminalEgoState`].
    pub fn overlay_ego_from_natural_language(&mut self, description: &str) {
        let ego = crate::ego_intent::overlay_ego_state_from_natural_language(
            self.ctx.core_layers().ego,
            description,
        );
        self.ctx.core_layers_mut().ego = ego;
        self.configured_ego = ego;
        self.sync_ego_differential_net_from_configured();
    }

    /// Returns the most recent temporal [`LiminalBoundaryDelta`] between aperture epochs.
    #[must_use]
    pub fn last_boundary_delta(&self) -> &LiminalBoundaryDelta {
        self.ctx.last_boundary_delta()
    }

    /// Returns the most recent [`StigmergicFragmentEnsembleReport`] computed during recall sync.
    #[must_use]
    pub const fn last_ensemble_report(&self) -> Option<&StigmergicFragmentEnsembleReport> {
        self.last_ensemble_report.as_ref()
    }

    /// Returns the bitmask of external [`LatentHorizonSlot`] boundaries currently slid into the
    /// adapter's active cognitive window.
    #[must_use]
    pub fn active_horizon_slots_mask(&self) -> u32 {
        self.ctx.active_horizon_slots_mask()
    }

    /// Returns the live consecutive destabilization / quiescence dwell tracker [`HorizonDwellState`].
    #[must_use]
    pub fn horizon_dwell(&self) -> HorizonDwellState {
        self.ctx.horizon_dwell()
    }

    /// Returns a shared reference to the built-in [`CoherenceHorizonGuardSlot`].
    #[must_use]
    pub const fn coherence_guard_slot(&self) -> &CoherenceHorizonGuardSlot {
        &self.coherence_guard_slot
    }

    /// Returns a shared reference to the auxiliary [`DefaultResidualSpilloverSlot`] experience cache.
    #[must_use]
    pub fn residual_spillover(&self) -> &DefaultResidualSpilloverSlot {
        &self.residual_spillover
    }

    /// Returns a mutable reference to the auxiliary [`DefaultResidualSpilloverSlot`] experience cache.
    pub fn residual_spillover_mut(&mut self) -> &mut DefaultResidualSpilloverSlot {
        &mut self.residual_spillover
    }

    /// Stages peeled 64-head residual phases from a saturated [`DissipativeOverflowReport`].
    pub fn stage_overflow_residual(&mut self, report: &DissipativeOverflowReport) {
        self.residual_spillover.stage_overflow_report(report);
    }

    /// Records verified post-step spillover outcome (`resolved && coherence > 0.0`) into the
    /// Tier-1 RAM cache and updates `+ System` experience telemetry counters.
    pub fn commit_spillover_experience(
        &mut self,
        residual_phases: &[lmnk_q7_t; MAX_ATTENTION_HEADS],
        coherence: f32,
        resolved: bool,
    ) -> (bool, u32) {
        let res = self.residual_spillover.commit_verified_experience(
            0,
            residual_phases,
            coherence,
            resolved,
        );
        self.ctx.telemetry_bank_mut().sync_experience_metrics(
            self.residual_spillover.cache_hits,
            self.residual_spillover.cache_stores,
            self.residual_spillover.cache_evictions,
        );
        res
    }

    /// Captures a read-only [`LiminalBoundaryFlowSummary`] projecting live [`CoreTelemetrySnapshot`]
    /// counters onto the `liminal` Abstraction <-> Concretion coordinate system.
    #[must_use]
    pub fn liminal_boundary_flow_summary(&self) -> LiminalBoundaryFlowSummary {
        let snap = self.telemetry_snapshot();
        let settlements = self.sandbox_settlements();
        let abstract_hits = snap
            .liminal
            .stationary_shortcut_hits
            .saturating_add(snap.unobserved_aperture_zeros);
        let tension_total = snap
            .immediate_tension_damps
            .saturating_add(snap.slow_tension_excites);

        LiminalBoundaryFlowSummary {
            total_steps: snap.spine.total_steps,
            continuous_standard_steps: snap.spine.continuous_standard_steps,
            whole_token_bypass_events: snap.whole_token_bypass_events,
            stationary_shortcut_hits: snap.liminal.stationary_shortcut_hits,
            unobserved_aperture_zeros: snap.unobserved_aperture_zeros,
            abstract_token_count: snap.abstract_token_count,
            concrete_token_count: snap.concrete_token_count,
            boundary_churn_count: snap.boundary_churn_count,
            logits_skip_count: snap.logits_skip_count,
            guard_relaxation_count: snap.guard_relaxation_count,
            instability_recovery_count: snap.instability_recovery_count,
            slide_in_count: snap.liminal.slide_in_count,
            slide_out_count: snap.liminal.slide_out_count,
            chatter_events: snap.liminal.chatter_events,
            ego_interventions: snap.ego.intervention_count,
            pframe_chain_evals: snap.pframe_chain_evals,
            pframe_total_hops: snap.pframe_total_hops,
            exterior_sandbox_stages: snap.exterior_sandbox_stages,
            sandbox_settlements: settlements,
            immediate_tension_damps: snap.immediate_tension_damps,
            slow_tension_excites: snap.slow_tension_excites,
            active_anchors: snap.topology.active_anchors,
            abstract_residency_ratio: safe_clamped_ratio(
                abstract_hits,
                snap.spine.continuous_standard_steps,
            ),
            settlement_ratio: safe_clamped_ratio(settlements, snap.exterior_sandbox_stages),
            tension_balance_ratio: safe_clamped_ratio(snap.immediate_tension_damps, tension_total),
            ego_excitation_ratio: safe_clamped_ratio(
                snap.ego.intervention_count,
                snap.spine.continuous_standard_steps,
            ),
        }
    }

    /// Renders a live Mermaid `flowchart LR` diagram and Markdown correspondence table
    /// overlaying runtime [`LiminalBoundaryFlowSummary`] counters onto the `liminal`
    /// Abstraction <-> Concretion boundary.
    ///
    /// Node labels strictly avoid raw `<` and `>` characters for GitHub Mermaid parser safety.
    #[must_use]
    pub fn render_liminal_flow_mermaid(&self) -> String {
        use std::fmt::Write as _;

        let s = self.liminal_boundary_flow_summary();
        let mut out = String::with_capacity(1400);
        let _ = writeln!(out, "```mermaid");
        let _ = writeln!(out, "flowchart LR");
        let _ = writeln!(out, "    subgraph ABS[\"Abstract Realm (Ground State)\"]");
        let _ = writeln!(
            out,
            "        ABS_BYP[\"SelfRef Bypass: whole_byp={} shortcut={} unobs={} (residency={:.1}%)\"]",
            s.whole_token_bypass_events,
            s.stationary_shortcut_hits,
            s.unobserved_aperture_zeros,
            s.abstract_residency_ratio * 100.0
        );
        let _ = writeln!(
            out,
            "        ABS_IFR[\"Indivisible(Grounded) I-Frame Anchors: active={} settled={}\"]",
            s.active_anchors, s.sandbox_settlements
        );
        let _ = writeln!(out, "    end");
        let _ = writeln!(out, "    subgraph LIM[\"liminal Boundary Gate\"]");
        let _ = writeln!(
            out,
            "        LIM_GATE[\"steps={} (std={}) slide_in={} slide_out={} chatter={} churn={}\"]",
            s.total_steps,
            s.continuous_standard_steps,
            s.slide_in_count,
            s.slide_out_count,
            s.chatter_events,
            s.boundary_churn_count
        );
        let _ = writeln!(out, "    end");
        let _ = writeln!(out, "    subgraph CON[\"Concrete Realm (Excited State)\"]");
        let _ = writeln!(
            out,
            "        CON_EGO[\"Ego P-Frame: interv={} ({:.1}%) evals={} hops={}\"]",
            s.ego_interventions,
            s.ego_excitation_ratio * 100.0,
            s.pframe_chain_evals,
            s.pframe_total_hops
        );
        let _ = writeln!(
            out,
            "        CON_EXT[\"Exterior Sandbox: staged={} settled={} ({:.1}%)\"]",
            s.exterior_sandbox_stages,
            s.sandbox_settlements,
            s.settlement_ratio * 100.0
        );
        let _ = writeln!(out, "    end");
        let _ = writeln!(
            out,
            "    ABS_BYP <-->|\"damp={} / excite={} (bal={:.1}%)\"| LIM_GATE",
            s.immediate_tension_damps,
            s.slow_tension_excites,
            s.tension_balance_ratio * 100.0
        );
        let _ = writeln!(
            out,
            "    LIM_GATE -->|\"Ego Excitation ({})\"| CON_EGO",
            s.ego_interventions
        );
        let _ = writeln!(
            out,
            "    LIM_GATE <-->|\"Exterior Stage ({}) / Settle ({})\"| CON_EXT",
            s.exterior_sandbox_stages, s.sandbox_settlements
        );
        let _ = writeln!(
            out,
            "    CON_EXT -->|\"settle_sandbox_lane ({})\"| ABS_IFR",
            s.sandbox_settlements
        );
        let _ = writeln!(out, "```");
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "| Boundary Axis | `CoreTelemetryBank` Counter | Live Value | Derived Ratio |"
        );
        let _ = writeln!(out, "| :--- | :--- | ---: | ---: |");
        let _ = writeln!(
            out,
            "| Abstract Residency | `stationary_shortcut_hits` + `unobserved_aperture_zeros` (`whole_token_bypass_events={}`) | {} + {} / {} | {:.1}% |",
            s.whole_token_bypass_events,
            s.stationary_shortcut_hits,
            s.unobserved_aperture_zeros,
            s.continuous_standard_steps,
            s.abstract_residency_ratio * 100.0
        );
        let _ = writeln!(
            out,
            "| Boundary Gate Flux | `liminal.slide_in_count` / `slide_out_count` / `chatter_events` | {} / {} / {} | - |",
            s.slide_in_count, s.slide_out_count, s.chatter_events
        );
        let _ = writeln!(
            out,
            "| Concrete Ego P-Frame | `ego.intervention_count` (`pframe_chain_evals={}`, `pframe_total_hops={}`) | {} / {} | {:.1}% |",
            s.pframe_chain_evals,
            s.pframe_total_hops,
            s.ego_interventions,
            s.continuous_standard_steps,
            s.ego_excitation_ratio * 100.0
        );
        let _ = writeln!(
            out,
            "| Exterior Settlement | `sandbox_settlements` / `exterior_sandbox_stages` | {} / {} | {:.1}% |",
            s.sandbox_settlements,
            s.exterior_sandbox_stages,
            s.settlement_ratio * 100.0
        );
        let _ = writeln!(
            out,
            "| Tension Balance | `immediate_tension_damps` / (`damps` + `slow_tension_excites`) | {} / ({} + {}) | {:.1}% |",
            s.immediate_tension_damps,
            s.immediate_tension_damps,
            s.slow_tension_excites,
            s.tension_balance_ratio * 100.0
        );
        let _ = writeln!(
            out,
            concat!(
                "| Decode Spine Control | `abstract_token_count` / `concrete_token_count` ",
                "/ `logits_skip_count` | {} / {} / {} (churn={}, relax={}, recov={}) | - |"
            ),
            s.abstract_token_count,
            s.concrete_token_count,
            s.logits_skip_count,
            s.boundary_churn_count,
            s.guard_relaxation_count,
            s.instability_recovery_count
        );
        out
    }
}
