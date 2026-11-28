// name: src/core/src/runtime/context/accessors.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Thin accessors, workbench slot controls, and helper methods on [`RuntimeContext`].

use super::{Config, CoreStateStack, OrchestratorStatusSnapshot, Result, RuntimeContext};
use crate::drivers::{
    AnonymousComputePool, ComputeDispatchMask, DistortionTension, EgoDifferentialNet,
    EntropyController, EnvelopeBounds, EnvelopeFeedback, EnvelopeState, SpinePinSnapshot,
    SystemPressureReading, SystemSpineState, TokenBypassObservation,
};
use crate::foundation::{
    AlignedSlice16, AlignedSliceMut16, SWAR_SPLAT_BYTE_64, branchless_clamp_f32,
    branchless_flush_subnormal_f32, branchless_min_f32, branchless_q7_inhibit_words_64,
    branchless_relu_f32, coherence_from_word_arrays_with_both_uniform_flags_64,
    fast_inv_pos_normal_f32, is_word_array_uniform_64, load_q7_words_64,
    swar_slerp_words_with_uniform_flag_64, swar_words_equal_64, swar_words_nonzero_64,
};
use crate::liminal::{
    BOUNDARY_SIGNAL_FLAG_IRREVERSIBLE_INTENT, BOUNDARY_SIGNAL_FLAG_OBSERVED,
    BOUNDARY_SIGNAL_KNOWN_FLAGS, BoundarySignal, DEFAULT_ENTROPY_EPS, LMNK_ERR_INVALID_ARGUMENT,
    LacunaPartition8, MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS, NUM_STIGMERGIC_GRANULES, PhaseQ7,
    RuntimeConfig,
};
use crate::runtime::boundary_delta::LiminalBoundaryDelta;
use crate::runtime::gateway::{
    CoreLayerEvalContext, GatewayFrame, ManifoldProbeBaseline, SandboxLifecycleInput,
};
use crate::runtime::horizon::{EpistemicHorizonReading, HorizonDwellState};
use crate::runtime::telemetry::{CoreTelemetryBank, CoreTelemetrySnapshot, TelemetryCapturePolicy};
use crate::stigmergy::{
    ContrastivePhaseBloomFilter, LedgerSlot, MAX_ATTRACTOR_PULL_FORCE, OverlapDiscoveryReport,
    PhaseSteeringEngine, PointerTrail, SandboxLifecycleEvent, StigmergicSubstrate,
    StigmergicThresholds, trail_phase_words_ref,
};

/// Geometry and threshold parameters for [`RuntimeContext::rebind_phase_and_substrate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContextRebindParams {
    /// Attention head dimension (`d_head`).
    pub head_dim: u32,
    /// Number of attention heads (`<= 64`).
    pub num_heads: u32,
    /// Active rotary dimension (`<= head_dim`, even).
    pub rotary_dim: u32,
    /// RoPE layout code (`0` = Interleaved, `1` = HalfSplit).
    pub rope_layout: u8,
    /// Substrate trail capacity (`>= 1`).
    pub substrate_capacity: u32,
    /// Stigmergic threshold ladder for the rebound substrate.
    pub thresholds: StigmergicThresholds,
    /// Baseline entropy observation seeded into the rebound substrate.
    pub baseline_entropy: f32,
}

impl RuntimeContext {
    /// Returns an immutable reference to the 128-byte 3-slot [`CoreStateStack`]
    /// (`Slot 0: Liminal`, `Slot 1: Stigmergic`, `Slot 2: Ego`).
    #[inline(always)]
    #[must_use]
    pub const fn core_layers(&self) -> &CoreStateStack {
        &self.core_layers
    }

    /// Returns a mutable reference to the 128-byte 3-slot [`CoreStateStack`].
    #[inline(always)]
    #[must_use]
    pub fn core_layers_mut(&mut self) -> &mut CoreStateStack {
        &mut self.core_layers
    }

    /// Returns an immutable reference to the latest Core Stack Slot 0 [`LiminalBoundaryDelta`].
    #[inline(always)]
    #[must_use]
    pub const fn last_boundary_delta(&self) -> &LiminalBoundaryDelta {
        &self.last_boundary_delta
    }

    /// Returns the active [`ManifoldProbeBaseline`] gauge origin used for assimilation drift.
    #[inline(always)]
    #[must_use]
    pub const fn probe_baseline(&self) -> &ManifoldProbeBaseline {
        &self.probe_baseline
    }

    /// Updates the [`ManifoldProbeBaseline`] gauge origin used for assimilation drift and, when a
    /// calibrated normalized baseline entropy (`probe_epoch > 0`, `0 < baseline_entropy <= 1`) is
    /// provided, seeds [`EnvelopeState::baseline_pressure_q15`] via [`EnvelopeState::seed_baseline_pressure_q15`].
    #[inline(always)]
    pub const fn set_probe_baseline(&mut self, baseline: ManifoldProbeBaseline) {
        self.probe_baseline = baseline;
        if baseline.probe_epoch > 0
            && baseline.baseline_entropy > 0.0
            && baseline.baseline_entropy <= 1.0
        {
            let q15 = (baseline.baseline_entropy * 32_768.0 + 0.5) as u16;
            self.envelope
                .seed_baseline_pressure_q15(q15, &self.envelope_bounds);
        }
    }

    /// Returns a mutable reference to the active [`ManifoldProbeBaseline`] gauge origin.
    #[inline(always)]
    #[must_use]
    pub fn probe_baseline_mut(&mut self) -> &mut ManifoldProbeBaseline {
        &mut self.probe_baseline
    }

    /// Returns an immutable reference to the embedded [`PhaseSteeringEngine`].
    #[inline(always)]
    #[must_use]
    pub const fn phase_steering(&self) -> &PhaseSteeringEngine {
        &self.phase_steering
    }

    /// Returns the physical workbench lane currently leased for an isolated sandbox hypothesis.
    #[inline(always)]
    #[must_use]
    pub const fn sandbox_workbench_lane(&self) -> Option<u8> {
        self.sandbox_workbench_lane
    }

    /// Returns a mutable reference to the isolated sandbox hypothesis workbench lane lease.
    #[inline(always)]
    #[must_use]
    pub fn sandbox_workbench_lane_mut(&mut self) -> &mut Option<u8> {
        &mut self.sandbox_workbench_lane
    }

    /// Returns the most recent [`SandboxLifecycleEvent`] emitted by [`Self::pierce_and_orchestrate_step`].
    #[inline(always)]
    #[must_use]
    pub const fn last_sandbox_event(&self) -> SandboxLifecycleEvent {
        self.last_sandbox_event
    }

    /// Returns the cumulative count of settled sandbox hypothesis lanes recorded in this context.
    #[inline(always)]
    #[must_use]
    pub const fn sandbox_settlements(&self) -> u32 {
        self.sandbox_settlements
    }

    /// Sets the cumulative count of settled sandbox hypothesis lanes recorded in this context.
    #[inline(always)]
    pub const fn set_sandbox_settlements(&mut self, count: u32) {
        self.sandbox_settlements = count;
    }

    /// Increments the cumulative count of settled sandbox hypothesis lanes using saturating arithmetic.
    #[inline(always)]
    pub const fn increment_sandbox_settlements(&mut self) {
        self.sandbox_settlements = self.sandbox_settlements.saturating_add(1);
    }

    /// Returns a mutable reference to the latest Core Stack Slot 0 [`LiminalBoundaryDelta`].
    #[inline(always)]
    #[must_use]
    pub fn last_boundary_delta_mut(&mut self) -> &mut LiminalBoundaryDelta {
        &mut self.last_boundary_delta
    }

    /// Returns an immutable reference to the latest [`EpistemicHorizonReading`].
    #[inline(always)]
    #[must_use]
    pub const fn last_horizon(&self) -> &EpistemicHorizonReading {
        &self.last_horizon
    }

    /// Returns a mutable reference to the latest [`EpistemicHorizonReading`].
    #[inline(always)]
    #[must_use]
    pub fn last_horizon_mut(&mut self) -> &mut EpistemicHorizonReading {
        &mut self.last_horizon
    }

    /// Returns an immutable reference to the latest [`OverlapDiscoveryReport`].
    #[inline(always)]
    #[must_use]
    pub const fn last_overlap_report(&self) -> &OverlapDiscoveryReport {
        &self.last_overlap_report
    }

    /// Returns a mutable reference to the latest [`OverlapDiscoveryReport`].
    #[inline(always)]
    #[must_use]
    pub fn last_overlap_report_mut(&mut self) -> &mut OverlapDiscoveryReport {
        &mut self.last_overlap_report
    }

    /// Returns the bitmask of external [`crate::runtime::horizon::LatentHorizonSlot`] boundaries currently slid
    /// into the
    /// active cognitive window.
    #[inline(always)]
    #[must_use]
    pub const fn active_horizon_slots_mask(&self) -> u32 {
        self.active_horizon_slots_mask
    }

    /// Returns a mutable reference to the active [`crate::runtime::horizon::LatentHorizonSlot`] bitmask.
    #[inline(always)]
    #[must_use]
    pub fn active_horizon_slots_mask_mut(&mut self) -> &mut u32 {
        &mut self.active_horizon_slots_mask
    }

    /// Returns the consecutive destabilization / quiescence dwell tracker [`HorizonDwellState`].
    #[inline(always)]
    #[must_use]
    pub const fn horizon_dwell(&self) -> HorizonDwellState {
        self.horizon_dwell
    }

    /// Returns a mutable reference to the consecutive destabilization / quiescence dwell tracker.
    #[inline(always)]
    #[must_use]
    pub fn horizon_dwell_mut(&mut self) -> &mut HorizonDwellState {
        &mut self.horizon_dwell
    }

    /// Returns a shared reference to the embedded `3 + System` [`CoreTelemetryBank`].
    #[inline(always)]
    #[must_use]
    pub const fn telemetry_bank(&self) -> &CoreTelemetryBank {
        &self.telemetry
    }

    /// Returns a mutable reference to the embedded `3 + System` [`CoreTelemetryBank`].
    #[inline(always)]
    #[must_use]
    pub fn telemetry_bank_mut(&mut self) -> &mut CoreTelemetryBank {
        &mut self.telemetry
    }

    /// Returns a shared reference to the label-free [`AnonymousComputePool`].
    #[inline(always)]
    #[must_use]
    pub const fn compute_pool(&self) -> &AnonymousComputePool {
        &self.compute_pool
    }

    /// Returns a mutable reference to the label-free [`AnonymousComputePool`].
    #[inline(always)]
    #[must_use]
    pub fn compute_pool_mut(&mut self) -> &mut AnonymousComputePool {
        &mut self.compute_pool
    }

    /// Returns a shared reference to the [`SystemSpineState`] (`DriverKind::System`).
    #[inline(always)]
    #[must_use]
    pub const fn system_spine(&self) -> &SystemSpineState {
        &self.system_spine
    }

    /// Returns a mutable reference to the [`SystemSpineState`].
    #[inline(always)]
    #[must_use]
    pub fn system_spine_mut(&mut self) -> &mut SystemSpineState {
        &mut self.system_spine
    }

    /// Returns a read-only reference to the embedded [`EnvelopeState`] (`DriverKind::Envelope`).
    #[inline(always)]
    #[must_use]
    pub const fn envelope(&self) -> &EnvelopeState {
        &self.envelope
    }

    /// Returns a read-only reference to the active [`EnvelopeBounds`] configuration (`offset 416`).
    #[inline(always)]
    #[must_use]
    pub const fn envelope_bounds(&self) -> &EnvelopeBounds {
        &self.envelope_bounds
    }

    /// Validates and updates [`RuntimeContext::envelope_bounds`] atomically (`0 alloc`).
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] without modifying internal state if
    /// [`EnvelopeBounds::validate_invariants`] returns `false`.
    #[inline(always)]
    pub fn configure_envelope_bounds(&mut self, bounds: EnvelopeBounds) -> Result<()> {
        if !bounds.validate_invariants() {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        self.envelope_bounds = bounds;
        Ok(())
    }

    /// Seeds `EnvelopeState::baseline_pressure_q15` clamped to the active [`EnvelopeBounds`].
    #[inline(always)]
    pub const fn seed_envelope_baseline_pressure_q15(&mut self, baseline_q15: u16) {
        self.envelope
            .seed_baseline_pressure_q15(baseline_q15, &self.envelope_bounds);
    }

    /// Extracts a 16-byte [`SpinePinSnapshot`] from [`EnvelopeState`] for isolated spine queries.
    #[inline(always)]
    #[must_use]
    pub(crate) const fn spine_pin_snapshot(&self) -> SpinePinSnapshot {
        SpinePinSnapshot::new(
            self.envelope.pinned_compute_mask,
            self.envelope.pinned_slot_mask,
            self.envelope.unobserved_locked,
            self.envelope.irreversible_step_lock,
        )
    }

    /// Sets the permanent baseline compute pin mask on [`EnvelopeState`] while enforcing
    /// [`crate::liminal::HOST_IMMUTABLE_PINNED_COMPUTE_MASK`].
    #[inline(always)]
    pub const fn set_baseline_pin_mask(&mut self, mask: u64) {
        self.envelope.set_baseline_pin_mask(mask);
    }

    /// Arms a transient turn-guard compute pin mask on [`EnvelopeState`] without raising
    /// the permanent baseline pin floor.
    #[inline(always)]
    pub const fn arm_transient_guard_pin(&mut self, active_mask: u64) {
        self.envelope.arm_transient_guard_pin(active_mask);
    }

    /// Restores the active compute pin mask on [`EnvelopeState`] back to its baseline floor.
    #[inline(always)]
    pub const fn restore_baseline_pin_mask(&mut self) {
        self.envelope.restore_baseline_pin_mask();
    }

    /// Monotonically adds compute slot pin bits into the active mask on [`EnvelopeState`].
    #[inline(always)]
    pub const fn pin_compute_slots(&mut self, mask: u64) {
        self.envelope.pin_compute_slots(mask);
    }

    /// Updates the zero-aperture unobserved lock on [`EnvelopeState`].
    #[inline(always)]
    pub const fn set_unobserved_locked(&mut self, locked: bool) {
        self.envelope.unobserved_locked = if locked { 1 } else { 0 };
    }

    /// Updates the irreversible step lock on [`EnvelopeState`].
    #[inline(always)]
    pub const fn set_irreversible_step_lock(&mut self, locked: bool) {
        self.envelope.irreversible_step_lock = if locked { 1 } else { 0 };
    }

    /// Resets transient safety-envelope state while preserving permanent host pin floors.
    #[inline(always)]
    pub const fn reset_envelope_preserving_pin_floor(&mut self) {
        self.envelope.reset_preserving_pin_floor();
    }

    /// Evaluates one branchless [`EnvelopeState`] safety-envelope step in-place (`0 alloc`, `0 fdiv`).
    #[inline(always)]
    #[must_use]
    pub fn step_envelope(
        &mut self,
        bounds: &EnvelopeBounds,
        signal: &BoundarySignal,
    ) -> EnvelopeFeedback {
        self.envelope.step(bounds, signal)
    }

    /// Emits a 32-byte domain-agnostic [`BoundarySignal`] summarizing this context's live
    /// [`LacunaPartition8`], [`EnvelopeState`], dominant lane, and caller-supplied metrics.
    #[inline]
    #[must_use]
    pub fn emit_boundary_signal(
        &self,
        pressure_q15: u16,
        chatter_q7: u8,
        epoch_id: u32,
        flags: u8,
    ) -> BoundarySignal {
        let clamped_pressure = pressure_q15.min(32_768);
        let delta_q15 = clamped_pressure
            .abs_diff(self.envelope.last_pressure_q15)
            .min(32_768);
        let mut sig = BoundarySignal::from_partition(
            self.last_lacuna_partition,
            clamped_pressure,
            delta_q15,
            chatter_q7,
            epoch_id,
        );
        sig.contraction_q7 = self.envelope.contraction_q7;
        sig.dominant_lane = (self.find_dominant_domain().0 as u8) & 0x07;
        let explicit_obs = (!sig.is_fault())
            && !self.last_lacuna_partition.is_unobserved_aperture_zero()
            && self.envelope.unobserved_locked == 0
            && (clamped_pressure > 0 || chatter_q7 > 0);
        let obs_flag = if explicit_obs {
            BOUNDARY_SIGNAL_FLAG_OBSERVED
        } else {
            0
        };
        let irrev_flag = if self.envelope.irreversible_step_lock != 0 {
            BOUNDARY_SIGNAL_FLAG_IRREVERSIBLE_INTENT
        } else {
            0
        };
        sig.signal_flags |= (flags & BOUNDARY_SIGNAL_KNOWN_FLAGS) | obs_flag | irrev_flag;
        sig
    }

    /// Returns the most recent branchless [`LacunaPartition8`].
    #[inline(always)]
    #[must_use]
    pub const fn last_lacuna_partition(&self) -> &LacunaPartition8 {
        &self.last_lacuna_partition
    }

    /// Returns a mutable reference to the most recent [`LacunaPartition8`].
    #[inline(always)]
    #[must_use]
    pub fn last_lacuna_partition_mut(&mut self) -> &mut LacunaPartition8 {
        &mut self.last_lacuna_partition
    }

    /// Returns a shared reference to the live [`DistortionTension`] homeostatic state.
    #[inline(always)]
    #[must_use]
    pub const fn distortion_tension(&self) -> &DistortionTension {
        &self.distortion_tension
    }

    /// Returns a mutable reference to the live [`DistortionTension`] homeostatic state.
    #[inline(always)]
    #[must_use]
    pub fn distortion_tension_mut(&mut self) -> &mut DistortionTension {
        &mut self.distortion_tension
    }

    /// Returns a shared reference to the inline [`EgoDifferentialNet`] P-Frame chain.
    #[inline(always)]
    #[must_use]
    pub const fn ego_differential_net(&self) -> &EgoDifferentialNet {
        &self.ego_differential_net
    }

    /// Returns a mutable reference to the inline [`EgoDifferentialNet`] P-Frame chain.
    #[inline(always)]
    #[must_use]
    pub fn ego_differential_net_mut(&mut self) -> &mut EgoDifferentialNet {
        &mut self.ego_differential_net
    }

    /// Evaluates external `+ System` Gauge-Zero pressure and emits a pre-execution
    /// [`ComputeDispatchMask`] without heap allocation or `fdiv`.
    #[inline]
    #[must_use]
    pub fn step_compute_dispatch(&mut self, reading: SystemPressureReading) -> ComputeDispatchMask {
        self.compute_pool
            .step_dispatch(reading, self.envelope.pinned_compute_mask)
    }

    /// Updates slot contribution pheromones strictly from an exit scalar reward (`0..=127`)
    /// without mid-layer D2H synchronization.
    #[inline]
    pub fn commit_compute_exit_reward(&mut self, executed_mask: u64, reward_q7: u8) {
        self.compute_pool
            .commit_exit_reward(executed_mask, reward_q7);
    }

    /// Resets per-turn bypass dwell, cooldown, and concrete streak counters across both the
    /// compute pool and [`SystemSpineState`].
    #[inline(always)]
    pub const fn reset_turn_bypass_state(&mut self) {
        self.compute_pool.reset_turn_bypass_state();
        self.system_spine.reset_turn_bypass_state();
    }

    /// Returns `true` when [`SystemSpineState`] allows relaxing a transient contradiction-guard
    /// compute pin back toward bypass.
    #[inline(always)]
    #[must_use]
    pub const fn can_relax_transient_guard_pin(
        &self,
        guard_hold_tokens: u8,
        liminal_stationary: u8,
        repulsion_active: bool,
        exit_entropy_rising: bool,
        min_calm_dwell: u8,
    ) -> bool {
        self.system_spine.can_relax_transient_guard_pin(
            guard_hold_tokens,
            liminal_stationary,
            repulsion_active,
            exit_entropy_rising,
            min_calm_dwell,
        )
    }

    /// Clears transient guard-induced cooldown lag on both the compute pool and [`SystemSpineState`]
    /// and seeds `bypass_dwell_steps` once post-hold relaxation is authorized.
    #[inline(always)]
    pub const fn on_transient_guard_relaxed(&mut self, min_calm_dwell: u8) {
        self.compute_pool.re_bypass_cooldown = 0;
        self.system_spine.on_transient_guard_relaxed(min_calm_dwell);
    }

    /// Triggers exit instability recovery across both [`AnonymousComputePool`] and [`SystemSpineState`],
    /// arming asymmetric re-bypass cooldowns and recording an instability recovery telemetry event.
    #[inline]
    pub fn signal_compute_instability(&mut self) {
        let cooldown = self.compute_pool.config.min_re_bypass_dwell.max(2);
        self.compute_pool.signal_exit_instability();
        self.system_spine.arm_bypass_cooldown(cooldown);
        self.telemetry.record_instability_recovery();
    }

    /// Updates the [`TelemetryCapturePolicy`] governing verbosity transitions (`Coarse` / `Standard` / `Detailed`).
    #[inline(always)]
    pub fn set_telemetry_policy(&mut self, policy: TelemetryCapturePolicy) {
        self.telemetry.set_policy(policy);
    }

    /// Resets accumulated telemetry counters across `3 + System` domains.
    #[inline(always)]
    pub fn reset_telemetry(&mut self) {
        self.telemetry.reset();
    }

    /// Computes a Lazy-Pull [`CoreTelemetrySnapshot`] combining `3 + System` counters, derived
    /// ratios, and live Core Stack Slot 0 topological state.
    #[inline]
    #[must_use]
    pub fn telemetry_snapshot(&self) -> CoreTelemetrySnapshot {
        self.telemetry.snapshot(
            &self.core_layers,
            &self.last_horizon,
            self.substrate.slot_enabled_mask.count_ones() as u8,
            0,
            self.active_horizon_slots_mask,
        )
    }

    /// Returns a single-glance [`OrchestratorStatusSnapshot`] summarizing concrete core engines,
    /// 3-slot core stack state, and dynamically active horizon slots.
    #[inline(always)]
    #[must_use]
    pub fn orchestrator_status(&self) -> OrchestratorStatusSnapshot {
        OrchestratorStatusSnapshot {
            step_counter: self.substrate.step_counter,
            active_trail_count: self.substrate.active_count,
            entropy_gate_mode: self.entropy_controller.state().active_branch_mode,
            attractor_anchor_count: self.substrate.slot_enabled_mask.count_ones(),
            slot_enabled_mask: self.substrate.slot_enabled_mask,
            slot_pinned_mask: self.substrate.slot_pinned_mask,
            slot_repulsion_mask: self.substrate.slot_repulsion_mask,
            slot_sandbox_mask: self.substrate.slot_sandbox_mask,
            liminal_threshold_mask: self.core_layers.liminal.active_aperture_mask,
            stigmergic_overlap_core_mask: self.core_layers.stigmergy.overlap_mask,
            ego_active: self.core_layers.ego.is_active(),
            ego_intervention_epochs: self.core_layers.ego.intervention_epochs,
            stationary_shortcut_ready: self.last_boundary_delta.stationary_shortcut_ready != 0,
            active_horizon_slots_mask: self.active_horizon_slots_mask,
            unstable_dwell_steps: self.horizon_dwell.unstable_dwell_steps,
            quiescent_dwell_steps: self.horizon_dwell.quiescent_dwell_steps,
        }
    }

    /// Returns a reconstructed value copy of the active high-level [`Config`].
    #[inline(always)]
    #[must_use]
    pub const fn config(&self) -> Config {
        Config {
            raw: self.raw_config,
            hysteresis_band: self.derived_config.hysteresis_band,
            tau_flux: self.derived_config.tau_flux,
            bifurcation_threshold: self.derived_config.bifurcation_threshold,
            repulsion_threshold: self.derived_config.repulsion_threshold,
            repulsion_scale_u7: self.derived_config.repulsion_scale_u7,
            slot_coupling_matrix: self.derived_config.slot_coupling_matrix,
        }
    }

    /// Returns an immutable reference to the active raw C-ABI [`RuntimeConfig`].
    #[inline(always)]
    #[must_use]
    pub const fn raw_config(&self) -> &RuntimeConfig {
        &self.raw_config
    }

    /// Returns an immutable reference to the underlying [`StigmergicSubstrate`].
    #[inline(always)]
    #[must_use]
    pub const fn substrate(&self) -> &StigmergicSubstrate {
        &self.substrate
    }

    /// Returns a mutable reference to the underlying [`StigmergicSubstrate`].
    #[inline(always)]
    #[must_use]
    pub fn substrate_mut(&mut self) -> &mut StigmergicSubstrate {
        &mut self.substrate
    }

    /// Returns an immutable reference to the underlying [`EntropyController`].
    #[inline(always)]
    #[must_use]
    pub const fn entropy_controller(&self) -> &EntropyController {
        &self.entropy_controller
    }

    /// Returns a mutable reference to the underlying [`EntropyController`].
    #[inline(always)]
    #[must_use]
    pub fn entropy_controller_mut(&mut self) -> &mut EntropyController {
        &mut self.entropy_controller
    }

    /// Returns an immutable reference to the 64-byte [`ContrastivePhaseBloomFilter`].
    #[inline(always)]
    #[must_use]
    pub const fn bloom_filter(&self) -> &ContrastivePhaseBloomFilter {
        &self.substrate.bloom_filter
    }

    /// Returns a mutable reference to the 64-byte [`ContrastivePhaseBloomFilter`].
    #[inline(always)]
    #[must_use]
    pub fn bloom_filter_mut(&mut self) -> &mut ContrastivePhaseBloomFilter {
        &mut self.substrate.bloom_filter
    }

    /// Returns an immutable reference to the active [`LedgerSlot`] scratchpad.
    #[inline(always)]
    #[must_use]
    pub const fn ledger_slot(&self) -> &LedgerSlot {
        &self.substrate.ledger_slot
    }

    /// Returns a mutable reference to the active [`LedgerSlot`] scratchpad.
    #[inline(always)]
    #[must_use]
    pub fn ledger_slot_mut(&mut self) -> &mut LedgerSlot {
        &mut self.substrate.ledger_slot
    }

    /// Returns an immutable 16-byte aligned slice view over the 8-lane Q7 working phase bank (`512` bytes).
    #[inline(always)]
    #[must_use]
    pub const fn slot_phase_bank(&self) -> AlignedSlice16<'_, PhaseQ7> {
        self.substrate.slot_phase_bank.relax_slice::<16>()
    }

    /// Returns a mutable 16-byte aligned slice view over the 8-lane Q7 working phase bank (`512` bytes).
    #[inline(always)]
    #[must_use]
    pub fn slot_phase_bank_mut(&mut self) -> AlignedSliceMut16<'_, PhaseQ7> {
        self.substrate.slot_phase_bank.relax_slice_mut::<16>()
    }

    /// Returns a mutable reference to the 64-byte (`8 x u64`) aligned word view of slot `slot_idx`.
    #[inline(always)]
    pub(super) fn slot_words_mut(
        &mut self,
        slot_idx: usize,
    ) -> Result<&mut [u64; NUM_STIGMERGIC_GRANULES]> {
        if slot_idx >= MAX_ATTRACTOR_DOMAINS {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        // SAFETY: `self.substrate.slot_phase_bank` is `AlignedBuffer64<PhaseQ7, 512>`, so every
        // 64-byte slot at byte offset `slot_idx * 64` is 64-byte aligned (`>= 8-byte` alignment of `u64`).
        Ok(unsafe {
            &mut *(self
                .substrate
                .slot_phase_bank
                .as_mut_slice()
                .as_mut_ptr()
                .add(slot_idx * MAX_ATTENTION_HEADS)
                as *mut [u64; NUM_STIGMERGIC_GRANULES])
        })
    }

    /// Sets the 64-head Q7 phase profile for a specific Workbench Lane (`0..=7`) and enables it.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `slot_id >= MAX_ATTRACTOR_DOMAINS`.
    #[inline]
    pub fn set_slot_phase_profile(
        &mut self,
        slot_id: u8,
        phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> Result<()> {
        let dst_w = self.slot_words_mut(usize::from(slot_id))?;
        *dst_w = load_q7_words_64(phases);
        self.substrate.slot_enabled_mask |= 1u8 << slot_id;
        Ok(())
    }

    /// Returns the active 8-lane weight array (`[f32; MAX_ATTRACTOR_DOMAINS]`).
    #[inline(always)]
    #[must_use]
    pub const fn slot_weights(&self) -> &[f32; MAX_ATTRACTOR_DOMAINS] {
        &self.substrate.slot_weights
    }

    /// Returns the 8-bit bitmask of enabled Workbench Lanes (`bit i == 1` means lane `i` is active).
    #[inline(always)]
    #[must_use]
    pub const fn slot_enabled_mask(&self) -> u8 {
        self.substrate.slot_enabled_mask
    }

    /// Returns the 8-bit bitmask of explicitly pinned Workbench Lanes (`0..=7`).
    #[inline(always)]
    #[must_use]
    pub const fn slot_pinned_mask(&self) -> u8 {
        self.substrate.slot_pinned_mask
    }

    /// Returns the 8-bit bitmask of anti-phase repulsion Workbench Lanes (`0..=7`).
    #[inline(always)]
    #[must_use]
    pub const fn slot_repulsion_mask(&self) -> u8 {
        self.substrate.slot_repulsion_mask
    }

    /// Returns the 8-bit bitmask of isolated exterior-only sandbox Workbench Lanes (`0..=7`).
    #[inline(always)]
    #[must_use]
    pub const fn slot_sandbox_mask(&self) -> u8 {
        self.substrate.slot_sandbox_mask
    }

    /// Returns the anonymous workbench lane (`0..=7`) currently leased by cold-path escalation, if any.
    #[inline(always)]
    #[must_use]
    pub const fn escalated_workbench_lane(&self) -> Option<u8> {
        self.escalated_workbench_lane
    }

    /// Returns a mutable reference to the cold-path escalated workbench lane lease.
    #[inline(always)]
    #[must_use]
    pub fn escalated_workbench_lane_mut(&mut self) -> &mut Option<u8> {
        &mut self.escalated_workbench_lane
    }

    /// Rebinds `phase_steering`, `substrate`, and geometry fields inside `config.raw` in-place
    /// when host model geometry changes, avoiding a full 5 KB stack copy.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `params` fail architectural invariant validation.
    pub fn rebind_phase_and_substrate(&mut self, params: ContextRebindParams) -> Result<()> {
        let ContextRebindParams {
            head_dim,
            num_heads,
            rotary_dim,
            rope_layout,
            substrate_capacity,
            thresholds,
            baseline_entropy,
        } = params;
        if substrate_capacity == 0
            || head_dim > u16::MAX as u32
            || num_heads > u16::MAX as u32
            || rotary_dim > head_dim
            || (rotary_dim & 1) != 0
            || rope_layout > 1
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        thresholds.validate()?;
        let new_phase_steering = PhaseSteeringEngine::new(head_dim, num_heads)?;
        self.raw_config.head_dim = head_dim;
        self.raw_config.num_heads = num_heads;
        self.raw_config.rotary_dim = rotary_dim;
        self.raw_config.rope_layout = rope_layout;
        self.raw_config.substrate_capacity = substrate_capacity;
        let sub_ptr = core::ptr::addr_of_mut!(self.substrate);
        // SAFETY: `sub_ptr` is a valid, 64B-aligned mutable pointer to `self.substrate` (`Pod`);
        // zeroing followed immediately by `StigmergicSubstrate::init_in_place_on_zeroed`
        // initializes all fields cleanly in-place without stack staging.
        unsafe {
            core::ptr::write_bytes(sub_ptr, 0u8, 1);
            StigmergicSubstrate::init_in_place_on_zeroed(
                sub_ptr,
                &self.raw_config,
                self.derived_config.repulsion_threshold,
                self.derived_config.repulsion_scale_u7,
                self.derived_config.slot_coupling_matrix,
            );
        }
        self.substrate.thresholds = thresholds;
        self.substrate.last_entropy_h_tilde = baseline_entropy;
        self.phase_steering = new_phase_steering;
        Ok(())
    }

    /// Builds a read-only [`CoreLayerEvalContext`] when Core Stack Slot 0 or Slot 2 is active.
    #[inline]
    #[must_use]
    pub fn build_core_layer_eval_context(&self) -> Option<CoreLayerEvalContext<'_>> {
        (self.core_layers.liminal.adaptation_epochs > 0 || self.core_layers.ego.is_active())
            .then_some(CoreLayerEvalContext {
                stack: &self.core_layers,
                delta: &self.last_boundary_delta,
                horizon: &self.last_horizon,
                overlap: &self.last_overlap_report,
                derivative_alpha_q7: self.telemetry.derivative_alpha_q7(),
                lacuna_partition: Some(&self.last_lacuna_partition),
                envelope: Some(&self.envelope),
                ego_net: Some(&self.ego_differential_net),
                tension: Some(&self.distortion_tension),
            })
    }

    /// Returns a field-disjoint split borrow pairing an optional read-only [`CoreLayerEvalContext`],
    /// `&ManifoldProbeBaseline`, and `&mut StigmergicSubstrate` for gateway orchestration calls.
    #[inline]
    #[must_use]
    pub fn split_for_substrate_gateway(
        &mut self,
    ) -> (
        Option<CoreLayerEvalContext<'_>>,
        &ManifoldProbeBaseline,
        &mut StigmergicSubstrate,
    ) {
        let core_ctx = (self.core_layers.liminal.adaptation_epochs > 0
            || self.core_layers.ego.is_active())
        .then_some(CoreLayerEvalContext {
            stack: &self.core_layers,
            delta: &self.last_boundary_delta,
            horizon: &self.last_horizon,
            overlap: &self.last_overlap_report,
            derivative_alpha_q7: self.telemetry.derivative_alpha_q7(),
            lacuna_partition: Some(&self.last_lacuna_partition),
            envelope: Some(&self.envelope),
            ego_net: Some(&self.ego_differential_net),
            tension: Some(&self.distortion_tension),
        });
        (core_ctx, &self.probe_baseline, &mut self.substrate)
    }

    /// Evaluates whole-token steering bypass jointly with [`SystemSpineState`] physical vetoes
    /// and records telemetry when bypass activates.
    #[inline]
    #[must_use]
    pub fn evaluate_whole_token_bypass_with_spine(
        &mut self,
        obs: TokenBypassObservation,
        guard_active: bool,
        dispatch: &mut ComputeDispatchMask,
    ) -> u8 {
        let pins = self.spine_pin_snapshot();
        let veto = self
            .compute_pool
            .evaluate_whole_token_steering_bypass_with_spine(
                obs,
                &mut self.system_spine,
                pins,
                guard_active,
                dispatch,
            );
        if dispatch.whole_token_steering_bypass != 0 {
            self.telemetry.record_whole_token_bypass();
        }
        veto
    }

    /// Evaluates the embedded [`EgoDifferentialNet`] P-Frame chain against current core state,
    /// mutates `frame` in-place, and records matched activations.
    #[inline]
    pub fn evaluate_ego_pframe_chain_mut(&mut self, frame: &mut GatewayFrame) -> (u8, u8, u8) {
        let (d_norm, hops, matched_mask) = self
            .ego_differential_net
            .evaluate_pframe_chain_with_mask_branchless(
                &self.core_layers.ego,
                &self.envelope,
                &self.last_lacuna_partition,
                &self.core_layers.stigmergy,
                &self.distortion_tension,
                frame,
            );
        self.ego_differential_net.last_delta_norm_q7 = d_norm;
        self.ego_differential_net.last_hops_traversed = hops;
        if hops > 0 {
            self.ego_differential_net
                .record_matched_activations(matched_mask);
        }
        (d_norm, hops, matched_mask)
    }

    /// Advances the isolated exterior-only sandbox workbench lifecycle in [`StigmergicSubstrate`]
    /// and updates `last_sandbox_event`.
    #[inline]
    #[must_use]
    pub fn step_sandbox_hypothesis_lifecycle(
        &mut self,
        input: SandboxLifecycleInput<'_>,
    ) -> SandboxLifecycleEvent {
        let ev = self.substrate.step_sandbox_hypothesis_lifecycle(
            &mut self.telemetry,
            &mut self.sandbox_workbench_lane,
            input,
        );
        self.last_sandbox_event = ev;
        if matches!(ev, SandboxLifecycleEvent::Settled(_)) {
            self.sandbox_settlements = self.sandbox_settlements.saturating_add(1);
        }
        ev
    }

    /// Returns the active 64-bit ($8 \times 8$) Attractor Working Lane coupling bitmatrix.
    #[inline(always)]
    #[must_use]
    pub const fn slot_coupling_matrix(&self) -> u64 {
        self.substrate.coupling_matrix_8x8
    }

    /// Updates the active 64-bit ($8 \times 8$) Attractor Working Lane coupling bitmatrix.
    #[inline(always)]
    pub fn set_slot_coupling_matrix(&mut self, matrix: u64) {
        self.substrate.coupling_matrix_8x8 = matrix;
        self.derived_config.slot_coupling_matrix = matrix;
    }

    /// Returns the total number of processed token steps.
    #[inline(always)]
    #[must_use]
    pub const fn step_counter(&self) -> u64 {
        self.substrate.step_counter
    }

    /// Returns the active pointer trail count in the substrate.
    #[inline(always)]
    #[must_use]
    pub const fn active_trail_count(&self) -> u32 {
        self.substrate.active_count
    }

    /// Evaluates the multi-domain attractor superposition plus `self.substrate.slot_phase_bank`
    /// directly in 8 register-resident `u64` SWAR words (`0` intermediate 64-byte stack buffers
    /// or store-reload round-trips).
    #[inline]
    #[must_use]
    pub(crate) fn evaluate_multi_domain_superposition_words(
        &self,
        trail: &PointerTrail,
    ) -> ([u64; NUM_STIGMERGIC_GRANULES], f32) {
        let tw = *trail_phase_words_ref(trail);
        let tn = swar_words_nonzero_64(&tw);
        let init_w = tw;
        let init_nonzero = tn;
        let init_is_uniform = !init_nonzero || is_word_array_uniform_64(&init_w);
        let mut run_w = init_w;
        let mut run_is_uniform = init_is_uniform;
        let mut running_pull = 0.0_f32;

        if let Ok(bank_2d) = self
            .substrate
            .slot_phase_bank
            .as_slot_bank::<MAX_ATTRACTOR_DOMAINS, MAX_ATTENTION_HEADS>()
        {
            let (rep_cut, attract_cut) = self.substrate.lane_role_quadrants();
            let weights = &self.substrate.slot_weights;
            let mut standard_bits = attract_cut.outside;
            while standard_bits != 0 {
                let idx = standard_bits.trailing_zeros() as usize;
                standard_bits &= standard_bits - 1;
                let base_e = branchless_flush_subnormal_f32(branchless_relu_f32(weights[idx]));
                if base_e > DEFAULT_ENTROPY_EPS {
                    // SAFETY: `self.substrate.slot_phase_bank` is `AlignedBuffer64<PhaseQ7, 512>`, so every
                    // 64-byte row `bank_2d[idx]` is 64-byte aligned (`>= 8-byte` alignment of `u64`).
                    let slot_w = unsafe {
                        &*(bank_2d[idx].as_ptr() as *const [u64; NUM_STIGMERGIC_GRANULES])
                    };
                    if swar_words_nonzero_64(slot_w) {
                        let slot_is_uniform = is_word_array_uniform_64(slot_w);
                        let coh = if init_nonzero {
                            coherence_from_word_arrays_with_both_uniform_flags_64(
                                &init_w,
                                init_is_uniform,
                                slot_w,
                                slot_is_uniform,
                            )
                        } else {
                            0.0
                        };
                        let res_pull = branchless_flush_subnormal_f32(
                            base_e * branchless_clamp_f32(1.0 + coh, 0.25, 2.0),
                        );
                        let total_w = running_pull + res_pull;
                        if total_w > DEFAULT_ENTROPY_EPS && !swar_words_equal_64(&run_w, slot_w) {
                            let ratio_f32 = branchless_clamp_f32(
                                (res_pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                                0.0,
                                128.0,
                            );
                            let s = (ratio_f32 as u8) as u64;
                            let bias_restore =
                                (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                            let (next_w, next_uni) = swar_slerp_words_with_uniform_flag_64(
                                &run_w,
                                run_is_uniform,
                                slot_w,
                                slot_is_uniform,
                                s,
                                bias_restore,
                            );
                            run_w = next_w;
                            run_is_uniform = next_uni;
                        }
                        running_pull = total_w;
                    }
                }
            }

            let mut rep_bits = rep_cut.outside;
            while rep_bits != 0 {
                let rep_idx = rep_bits.trailing_zeros() as usize;
                rep_bits &= rep_bits - 1;
                // SAFETY: `bank_2d[rep_idx]` is 64-byte aligned.
                let rep_words = unsafe {
                    &*(bank_2d[rep_idx].as_ptr() as *const [u64; NUM_STIGMERGIC_GRANULES])
                };
                if swar_words_nonzero_64(rep_words) {
                    let rep_strength =
                        branchless_clamp_f32(branchless_relu_f32(weights[rep_idx]), 0.0, 1.0);
                    running_pull = branchless_q7_inhibit_words_64(
                        &mut run_w,
                        running_pull,
                        rep_words,
                        rep_strength,
                    );
                }
            }
        }

        let clamped_pull = branchless_min_f32(
            branchless_flush_subnormal_f32(running_pull),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        (run_w, clamped_pull)
    }
}
