// name: src/adapter/src/model_adapter.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model geometry descriptors (`RopeGeometry`) and unified steering adapter
//! ([`ModelSteeringAdapter`]) bridging external LLM runtimes to [`liminika_core`].

use super::*;

/// Memory pairing convention used by an LLM's Rotary Position Embedding (RoPE).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeLayout {
    /// Adjacent pairs `(x[2i], x[2i + 1])` (original RoPE / some GGUF layouts).
    Interleaved,
    /// Split-half pairs `(x[i], x[i + rotary_dim / 2])` (Hugging Face / NeoX / Candle).
    HalfSplit,
}

/// Explicit per-model RoPE geometry descriptor.
///
/// Capturing layout, partial-RoPE bounds, GQA ratios, and NoPE layer masks up front
/// prevents silent geometric corruption across heterogeneous LLM architectures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RopeGeometry {
    /// Pairwise rotation layout (`Interleaved` or `HalfSplit`).
    pub layout: RopeLayout,
    /// Full dimension of each attention head (`d_head`).
    pub head_dim: usize,
    /// Active rotated dimensions (`<= head_dim`; smaller for partial RoPE models).
    pub rotary_dim: usize,
    /// Number of query heads (`H_Q`).
    pub num_q_heads: usize,
    /// Number of key/value heads (`H_KV`, `<= H_Q` for GQA).
    pub num_kv_heads: usize,
    /// Total transformer layers (`<= 64` for `layer_rope_mask`).
    pub num_layers: usize,
    /// Bitmask where bit `i == 1` enables RoPE steering on layer `i` (`0` skips NoPE layers).
    pub layer_rope_mask: u64,
}

impl RopeGeometry {
    /// Returns the validated geometry preset for `microsoft/Phi-4-mini-instruct`.
    #[must_use]
    pub const fn phi4_mini() -> Self {
        Self {
            layout: RopeLayout::HalfSplit,
            head_dim: 128,
            rotary_dim: 96,
            num_q_heads: 24,
            num_kv_heads: 8,
            num_layers: 32,
            layer_rope_mask: 0xFFFF_FFFF,
        }
    }

    /// Validates structural invariants so invalid model configurations fail closed
    /// before any steering pass runs.
    ///
    /// # Errors
    ///
    /// Returns an error string if `head_dim`, `rotary_dim`, head counts, or `num_layers`
    /// violate RoPE or GQA divisibility constraints.
    pub fn validate(&self) -> Result<(), String> {
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            return Err(format!(
                "head_dim ({}) must be a positive even integer",
                self.head_dim
            ));
        }
        if self.rotary_dim == 0
            || !self.rotary_dim.is_multiple_of(2)
            || self.rotary_dim > self.head_dim
        {
            return Err(format!(
                "rotary_dim ({}) must be even, > 0, and <= head_dim ({})",
                self.rotary_dim, self.head_dim
            ));
        }
        if self.num_q_heads == 0 || self.num_kv_heads == 0 {
            return Err("num_q_heads and num_kv_heads must be > 0".to_string());
        }
        if !self.num_q_heads.is_multiple_of(self.num_kv_heads) {
            return Err(format!(
                "num_q_heads ({}) must be divisible by num_kv_heads ({})",
                self.num_q_heads, self.num_kv_heads
            ));
        }
        if self.num_layers == 0 || self.num_layers > 64 {
            return Err(format!(
                "num_layers ({}) must be in 1..=64 for bitmask routing",
                self.num_layers
            ));
        }
        Ok(())
    }

    /// Returns `true` if layer `layer_idx` uses RoPE and should receive phase steering.
    #[inline]
    #[must_use]
    pub const fn is_rope_layer(&self, layer_idx: usize) -> bool {
        if layer_idx >= self.num_layers || layer_idx >= 64 {
            return false;
        }
        ((self.layer_rope_mask >> layer_idx) & 1) != 0
    }

    /// Total elements in one token's active rotated query slice (`num_q_heads * rotary_dim`).
    #[inline]
    #[must_use]
    pub const fn q_rot_len(&self) -> usize {
        self.num_q_heads * self.rotary_dim
    }
}

#[repr(C, align(32))]
#[derive(Clone, Copy)]
pub(crate) struct Align32Chunk([f32; 8]);

/// Safe 32-byte aligned `f32` buffer for passing arbitrary-length slices to [`liminika_core`].
pub(crate) struct AlignedF32Buffer32 {
    chunks: Vec<Align32Chunk>,
    len: usize,
}

impl AlignedF32Buffer32 {
    fn new_zeroed(len: usize) -> Self {
        let num_chunks = len.div_ceil(8).max(1);
        Self {
            chunks: vec![Align32Chunk([0.0; 8]); num_chunks],
            len,
        }
    }

    fn as_slice(&self) -> &[f32] {
        let ptr = self.chunks.as_ptr().cast::<f32>();
        // SAFETY: `Align32Chunk` is `#[repr(C, align(32))]` wrapping `[f32; 8]`, so
        // `chunks.len() * 8 >= self.len` contiguous `f32` elements are valid and initialized.
        unsafe { core::slice::from_raw_parts(ptr, self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        let ptr = self.chunks.as_mut_ptr().cast::<f32>();
        // SAFETY: Same contiguous `[f32; 8]` layout as `as_slice`, with exclusive `&mut self`.
        unsafe { core::slice::from_raw_parts_mut(ptr, self.len) }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(crate) struct Align16Chunk([gcso_q7_t; 16]);

/// Safe 16-byte aligned `gcso_q7_t` buffer for [`AlignedSlice16`] / [`AlignedSliceMut16`].
pub(crate) struct AlignedQ7Buffer16 {
    chunks: Vec<Align16Chunk>,
    len: usize,
}

impl AlignedQ7Buffer16 {
    fn from_slice(src: &[gcso_q7_t]) -> Self {
        let mut buf = Self::new_zeroed(src.len());
        buf.as_mut_slice().copy_from_slice(src);
        buf
    }

    fn new_zeroed(len: usize) -> Self {
        let num_chunks = len.div_ceil(16).max(1);
        Self {
            chunks: vec![Align16Chunk([0; 16]); num_chunks],
            len,
        }
    }

    fn as_slice(&self) -> &[gcso_q7_t] {
        let ptr = self.chunks.as_ptr().cast::<gcso_q7_t>();
        // SAFETY: `Align16Chunk` is `#[repr(C, align(16))]` wrapping `[gcso_q7_t; 16]`.
        unsafe { core::slice::from_raw_parts(ptr, self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [gcso_q7_t] {
        let ptr = self.chunks.as_mut_ptr().cast::<gcso_q7_t>();
        // SAFETY: Same contiguous `[gcso_q7_t; 16]` layout as `as_slice`, with exclusive `&mut self`.
        unsafe { core::slice::from_raw_parts_mut(ptr, self.len) }
    }
}

impl RopeSteeringTarget for RopeGeometry {
    #[inline(always)]
    fn rope_geometry(&self) -> RopeGeometryDescriptor {
        RopeGeometryDescriptor {
            layout: match self.layout {
                RopeLayout::Interleaved => RopePairLayout::Interleaved,
                RopeLayout::HalfSplit => RopePairLayout::HalfSplit,
            },
            // `ModelSteeringAdapter` operates directly on the sliced `q_rot` buffer (`rotary_dim` per head).
            head_dim: self.rotary_dim,
            rotary_dim: self.rotary_dim,
            num_heads: self.num_q_heads,
        }
    }
}

/// Unified model-connection and optimization adapter over [`liminika_core`].
///
/// Bridges a target LLM's [`RopeGeometry`] to [`PhaseSteeringEngine`],
/// [`EntropyController`], and [`ResidualAdapterEngine`] without polluting `liminika-core`
/// with model-specific parsing or runtime dependencies.
pub struct ModelSteeringAdapter {
    geometry: RopeGeometry,
    phase_engine: PhaseSteeringEngine,
    action_hub: Box<ActionHubEngine>,
    ledger_slot: gcso_ledger_slot_t,
    decision_counts: [u64; 7],
    entropy_controller: EntropyController,
    residual_adapter: ResidualAdapterEngine,
    system_constraint: SystemConstraint<()>,
    probe_baseline: ManifoldProbeBaseline,
    threshold_policy: AdaptiveThresholdPolicy,
    core_layers: AutonomousCoreLayerStack,
    ego_field: LiminalEgoField,
    last_boundary_delta: LiminalBoundaryDelta,
    last_horizon: EpistemicHorizonReading,
    last_venn_report: StigmergicVennSerendipityReport,
}

impl ModelSteeringAdapter {
    fn compute_steered_layer_count(geometry: &RopeGeometry) -> u16 {
        let mut count = 0u16;
        for l in 0..geometry.num_layers.min(64) {
            if geometry.is_rope_layer(l) {
                count += 1;
            }
        }
        count.max(1)
    }

    /// Creates a new [`ModelSteeringAdapter`] after validating `geometry`.
    ///
    /// # Errors
    ///
    /// Returns an error if `geometry.validate()` fails or [`PhaseSteeringEngine::new`]
    /// rejects the `(rotary_dim, num_q_heads)` pair.
    pub fn new(geometry: RopeGeometry) -> Result<Self, String> {
        geometry.validate()?;
        let phase_engine =
            PhaseSteeringEngine::new(geometry.rotary_dim as u32, geometry.num_q_heads as u32)
                .map_err(|e| format!("PhaseSteeringEngine::new failed: {e}"))?;
        let mut action_hub = Box::new(
            ActionHubEngine::with_dimensions_and_thresholds(
                geometry.rotary_dim as u32,
                geometry.num_q_heads as u32,
                16,
                StigmergicThresholds::default(),
            )
            .map_err(|e| format!("ActionHubEngine::with_dimensions_and_thresholds failed: {e}"))?,
        );
        action_hub.last_entropy_h_tilde = 0.35;
        let ledger_slot = gcso_ledger_slot_t::default();
        let entropy_controller = EntropyController::new(gcso_edbc_state_t::default());
        let residual_adapter = ResidualAdapterEngine::new(1.0, 0, 1);
        let system_constraint = SystemConstraint::new((), 0.0, 0.0, 0xFF);
        let mut probe_baseline = ManifoldProbeBaseline::new();
        probe_baseline.active_heads = geometry.num_q_heads.min(64) as u16;
        probe_baseline.steered_layers = Self::compute_steered_layer_count(&geometry);
        let threshold_policy = AdaptiveThresholdPolicy::default();

        Ok(Self {
            geometry,
            phase_engine,
            action_hub,
            ledger_slot,
            decision_counts: [0; 7],
            entropy_controller,
            residual_adapter,
            system_constraint,
            probe_baseline,
            threshold_policy,
            core_layers: AutonomousCoreLayerStack::new(),
            ego_field: LiminalEgoField::new(),
            last_boundary_delta: LiminalBoundaryDelta::default(),
            last_horizon: EpistemicHorizonReading::default(),
            last_venn_report: StigmergicVennSerendipityReport::default(),
        })
    }

    /// Returns the per-regime [`StigmergicActionDecision`] invocation counts (`0..=6`).
    #[must_use]
    pub const fn action_hub_decision_counts(&self) -> &[u64; 7] {
        &self.decision_counts
    }

    /// Resets the per-regime [`StigmergicActionDecision`] invocation counters and restores
    /// initial neutral entropy (`0.35`) for a fresh evaluation pass.
    pub fn reset_action_hub_stats(&mut self) {
        self.decision_counts = [0; 7];
        self.action_hub.last_entropy_h_tilde = 0.35;
    }

    /// Returns a shared reference to the embedded [`ActionHubEngine`].
    #[must_use]
    pub fn action_hub(&self) -> &ActionHubEngine {
        &self.action_hub
    }

    /// Returns a mutable reference to the embedded [`ActionHubEngine`].
    pub fn action_hub_mut(&mut self) -> &mut ActionHubEngine {
        &mut self.action_hub
    }

    /// Returns the validated [`RopeGeometry`].
    #[must_use]
    pub const fn geometry(&self) -> &RopeGeometry {
        &self.geometry
    }

    /// Updates the active RoPE layer bitmask (`1` = steer, `0` = skip NoPE/disabled layer).
    pub fn set_layer_rope_mask(&mut self, mask: u64) {
        self.geometry.layer_rope_mask = mask;
        self.probe_baseline.steered_layers = Self::compute_steered_layer_count(&self.geometry);
    }

    /// Returns the latest Cold-Path [`ManifoldProbeBaseline`] calibrated for the active model.
    #[must_use]
    pub const fn probe_baseline(&self) -> &ManifoldProbeBaseline {
        &self.probe_baseline
    }

    /// Returns the live [`SystemConstraint<()>`] telemetry snapshot.
    #[must_use]
    pub const fn system_constraint(&self) -> &SystemConstraint<()> {
        &self.system_constraint
    }

    /// Writes `circuit` into the embedded [`gcso_ledger_slot_t`] so every subsequent
    /// [`ActionHubEngine`] and SRL step modulates [`GatewayFrame`] dynamically from this circuit.
    pub fn set_active_circuit(&mut self, circuit: CognitiveTrajectoryLedger) {
        circuit.write_to_ledger_slot(&mut self.ledger_slot);
    }

    /// Clears the active [`CognitiveTrajectoryLedger`] flag and trajectory register from the adapter.
    pub fn clear_active_circuit(&mut self) {
        self.ledger_slot.telemetry_mini_ledger = [0u8; 8];
        self.ledger_slot.auto_tune_flags &= !LEDGER_FLAG_COGNITIVE_TRAJECTORY;
    }

    /// Returns the currently active [`CognitiveTrajectoryLedger`] if attached to the adapter's ledger slot.
    #[must_use]
    pub const fn active_circuit(&self) -> Option<CognitiveTrajectoryLedger> {
        if (self.ledger_slot.auto_tune_flags & LEDGER_FLAG_COGNITIVE_TRAJECTORY) != 0 {
            Some(CognitiveTrajectoryLedger::from_ledger_slot(
                &self.ledger_slot,
            ))
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
        let horizon = self.last_horizon;
        let venn = self.last_venn_report;
        self.core_layers
            .adapt_layers_branchless(&ledger, &horizon, &venn, None);
        act
    }

    /// Returns a shared reference to the live Core Layer 0 [`LiminalApertureField`] governing
    /// dynamic `Liminal` channel thresholds and cognitive window gains.
    #[must_use]
    pub const fn aperture_field(&self) -> &LiminalApertureField {
        &self.core_layers.liminal_layer
    }

    /// Returns a mutable reference to the live Core Layer 0 [`LiminalApertureField`].
    pub fn aperture_field_mut(&mut self) -> &mut LiminalApertureField {
        &mut self.core_layers.liminal_layer
    }

    /// Returns a shared reference to the live Core Layer 1 [`StigmergicGranuleField`].
    #[must_use]
    pub const fn stigmergic_granule_field(&self) -> &StigmergicGranuleField {
        &self.core_layers.stigmergic_layer
    }

    /// Returns a shared reference to the unified 2-layer [`AutonomousCoreLayerStack`].
    #[must_use]
    pub const fn core_layer_stack(&self) -> &AutonomousCoreLayerStack {
        &self.core_layers
    }

    /// Returns the most recently observed [`EpistemicHorizonReading`].
    #[must_use]
    pub const fn last_horizon(&self) -> &EpistemicHorizonReading {
        &self.last_horizon
    }

    /// Returns the most recently observed [`StigmergicVennSerendipityReport`].
    #[must_use]
    pub const fn last_venn_serendipity_report(&self) -> &StigmergicVennSerendipityReport {
        &self.last_venn_report
    }

    /// Returns a shared reference to the live 64-byte [`LiminalEgoField`].
    #[must_use]
    pub const fn ego_field(&self) -> &LiminalEgoField {
        &self.ego_field
    }

    /// Returns a mutable reference to the live 64-byte [`LiminalEgoField`].
    pub fn ego_field_mut(&mut self) -> &mut LiminalEgoField {
        &mut self.ego_field
    }

    /// Replaces the adapter's active [`LiminalEgoField`].
    pub fn set_ego_field(&mut self, ego: LiminalEgoField) {
        self.ego_field = ego;
    }

    /// Configures [`LiminalEgoField`] from a coarse [`LiminalEgoPreset`] (`Custom`, `Low`, `Medium`, `High`).
    pub fn configure_ego_from_preset(&mut self, preset: LiminalEgoPreset) {
        self.ego_field = LiminalEgoField::from_preset(preset);
    }

    /// Derives [`LiminalEgoField`] deterministically from a natural-language Ego description.
    pub fn configure_ego_from_natural_language(&mut self, description: &str) {
        self.ego_field = LiminalEgoField::from_natural_language(description);
    }

    /// Returns the most recent temporal [`LiminalBoundaryDelta`] between aperture epochs.
    #[must_use]
    pub const fn last_boundary_delta(&self) -> &LiminalBoundaryDelta {
        &self.last_boundary_delta
    }

    /// Updates `last_horizon` and adapts Core Layer 0 (`LiminalApertureField`) branchlessly.
    pub fn adapt_liminal_aperture(
        &mut self,
        horizon: &EpistemicHorizonReading,
        overflow_opt: Option<&DissipativeOverflowReport>,
    ) {
        let venn = self.last_venn_report;
        self.adapt_core_layers(horizon, &venn, overflow_opt);
    }

    /// Updates both Core Layer 0 (`Liminal`) and Core Layer 1 (`Stigmergic`) branchlessly, computes
    /// the temporal [`LiminalBoundaryDelta`], and applies [`LiminalEgoField`] boundary interventions.
    pub fn adapt_core_layers(
        &mut self,
        horizon: &EpistemicHorizonReading,
        venn_report: &StigmergicVennSerendipityReport,
        overflow_opt: Option<&DissipativeOverflowReport>,
    ) {
        let prev_layer0 = self.core_layers.liminal_layer;
        let prev_venn = self.last_venn_report.venn_core_mask;
        self.last_horizon = *horizon;
        self.last_venn_report = *venn_report;
        let ledger = self.active_circuit().unwrap_or_default();
        self.core_layers
            .adapt_layers_branchless(&ledger, horizon, venn_report, overflow_opt);
        self.last_boundary_delta = LiminalBoundaryDelta::compute_branchless(
            &prev_layer0,
            &self.core_layers.liminal_layer,
            prev_venn,
            venn_report.venn_core_mask,
            &self.last_boundary_delta,
        );
        if self.ego_field.is_active() {
            let preview = GatewayFrame {
                effective_energy: horizon.total_memory_support,
                ..GatewayFrame::default()
            };
            let _ = self.ego_field.intervene_on_layers_and_frame_branchless(
                &mut self.core_layers,
                &self.last_boundary_delta,
                horizon,
                venn_report,
                preview,
            );
        }
    }

    /// Captures the live [`LiminalTopologyTelemetrySnapshot`] and renders a self-introspecting
    /// 2-layer Mermaid `flowchart TD` diagram covering Core Layer 0 (`Liminal`), Core Layer 1
    /// (`StigmergicGranuleField`), cognitive aperture masks, and crystal populations.
    #[must_use]
    pub fn render_liminal_topology_mermaid(
        &self,
        active_crystals: u8,
        spawned_daughters: u8,
    ) -> String {
        let snap = LiminalTopologyTelemetrySnapshot::capture(
            &self.core_layers.liminal_layer,
            &self.last_horizon,
            active_crystals,
            spawned_daughters,
        );
        let mut buf = [0u8; 2048];
        let written = snap.format_core_layers_mermaid(
            &self.core_layers.stigmergic_layer,
            &self.last_venn_report,
            &mut buf,
        );
        String::from_utf8_lossy(&buf[..written]).into_owned()
    }

    /// Returns a mutable reference to the dynamic [`AdaptiveThresholdPolicy`].
    pub fn threshold_policy_mut(&mut self) -> &mut AdaptiveThresholdPolicy {
        &mut self.threshold_policy
    }

    /// Updates hardware/runtime telemetry (`vram_pressure`, `gpu_frame_overrun`, active swarm mask)
    /// from an external environment probe and immediately refreshes cached thresholds.
    pub fn update_environment_telemetry(
        &mut self,
        vram_pressure: f32,
        gpu_frame_overrun: f32,
        active_swarm_mask: u8,
    ) {
        self.system_constraint =
            SystemConstraint::new((), vram_pressure, gpu_frame_overrun, active_swarm_mask);
        let neutral_trail = gcso_pointer_trail_t::default();
        self.action_hub.thresholds = self.threshold_policy.resolve_thresholds(
            &self.system_constraint,
            &self.probe_baseline,
            &neutral_trail,
            self.probe_baseline.baseline_entropy,
        );
    }

    /// Cold-Path model calibration probe run before startup or periodically when drift / model
    /// changes occur. Measures the model's empirical inter-layer phase trajectory coherence and
    /// neutral entropy distribution, updates [`ManifoldProbeBaseline`], and equilibrates
    /// [`StigmergicThresholds`] without manual constant tuning.
    ///
    /// # Errors
    ///
    /// Returns an error if any probe Q-slice length mismatches `self.geometry().q_rot_len()`.
    pub fn calibrate_manifold_probe(
        &mut self,
        per_layer_sample_q: &[&[f32]],
        baseline_ref_per_layer: &[&[f32]],
        observed_entropies: &[f32],
    ) -> Result<ManifoldProbeBaseline, String> {
        let mut prev_q7: Option<Vec<gcso_q7_t>> = None;
        let mut coh_sum = 0.0_f64;
        let mut coh_count = 0_usize;

        for (layer_idx, (q_sample, q_ref)) in per_layer_sample_q
            .iter()
            .zip(baseline_ref_per_layer.iter())
            .enumerate()
        {
            if !self.geometry.is_rope_layer(layer_idx) {
                continue;
            }
            let cur_q7 = self.bake_layer_procrustes_q7(q_ref, q_sample)?;
            if let Some(prev) = prev_q7.as_ref() {
                let mut dot = 0.0_f64;
                let mut n_a = 0.0_f64;
                let mut n_b = 0.0_f64;
                for (&a, &b) in prev.iter().zip(cur_q7.iter()) {
                    let fa = f64::from(a);
                    let fb = f64::from(b);
                    dot += fa * fb;
                    n_a += fa * fa;
                    n_b += fb * fb;
                }
                let denom = (n_a * n_b).sqrt();
                if denom > 1e-12 {
                    let cos = (dot / denom).clamp(-1.0, 1.0);
                    if cos > 0.0 {
                        coh_sum += cos;
                    }
                    coh_count += 1;
                }
            }
            prev_q7 = Some(cur_q7);
        }

        if coh_count > 0 {
            self.probe_baseline.baseline_inter_layer_coherence =
                (coh_sum / coh_count as f64).clamp(0.05, 1.0) as f32;
        }
        let mut ent_sum = 0.0_f64;
        let mut ent_count = 0_usize;
        for &h in observed_entropies {
            if h.is_finite() && h >= 0.0 {
                ent_sum += f64::from(h.clamp(0.0, 1.0));
                ent_count += 1;
            }
        }
        if ent_count > 0 {
            self.probe_baseline.baseline_entropy = (ent_sum / ent_count as f64) as f32;
        }
        self.probe_baseline.active_heads = self.geometry.num_q_heads.min(64) as u16;
        self.probe_baseline.steered_layers = Self::compute_steered_layer_count(&self.geometry);
        self.probe_baseline.probe_epoch = self.probe_baseline.probe_epoch.saturating_add(1);

        let neutral_trail = gcso_pointer_trail_t::default();
        self.action_hub.thresholds = self.threshold_policy.resolve_thresholds(
            &self.system_constraint,
            &self.probe_baseline,
            &neutral_trail,
            self.probe_baseline.baseline_entropy,
        );
        Ok(self.probe_baseline)
    }

    /// Safely re-allocates and re-initializes the Cold-Path engines (`PhaseSteeringEngine`,
    /// `ActionHubEngine`, and `ledger_slot`) when the underlying LLM or RoPE geometry changes
    /// during a periodic environment probe, returning the new `probe_epoch`.
    ///
    /// # Errors
    ///
    /// Returns an error if `new_geometry.validate()` fails or `liminika-core` rejects dimensions.
    pub fn rebind_geometry_cold_path(&mut self, new_geometry: RopeGeometry) -> Result<u32, String> {
        new_geometry.validate()?;
        let new_phase_engine = PhaseSteeringEngine::new(
            new_geometry.rotary_dim as u32,
            new_geometry.num_q_heads as u32,
        )
        .map_err(|e| format!("PhaseSteeringEngine::new failed: {e}"))?;
        let mut next_probe = self.probe_baseline;
        next_probe.active_heads = new_geometry.num_q_heads.min(64) as u16;
        next_probe.steered_layers = Self::compute_steered_layer_count(&new_geometry);
        next_probe.probe_epoch = next_probe.probe_epoch.saturating_add(1);

        let neutral_trail = gcso_pointer_trail_t::default();
        let calibrated = self.threshold_policy.resolve_thresholds(
            &self.system_constraint,
            &next_probe,
            &neutral_trail,
            next_probe.baseline_entropy,
        );
        let mut new_action_hub = Box::new(
            ActionHubEngine::with_dimensions_and_thresholds(
                new_geometry.rotary_dim as u32,
                new_geometry.num_q_heads as u32,
                16,
                calibrated,
            )
            .map_err(|e| format!("ActionHubEngine::with_dimensions_and_thresholds failed: {e}"))?,
        );
        new_action_hub.last_entropy_h_tilde = next_probe.baseline_entropy;

        self.geometry = new_geometry;
        self.phase_engine = new_phase_engine;
        self.action_hub = new_action_hub;
        self.ledger_slot = gcso_ledger_slot_t::default();
        self.decision_counts = [0; 7];
        self.probe_baseline = next_probe;
        Ok(self.probe_baseline.probe_epoch)
    }

    /// Returns a mutable reference to the embedded [`EntropyController`] (EDBC).
    pub fn entropy_controller_mut(&mut self) -> &mut EntropyController {
        &mut self.entropy_controller
    }

    /// Returns a mutable reference to the embedded [`ResidualAdapterEngine`] (Rank-1 residual).
    pub fn residual_adapter_mut(&mut self) -> &mut ResidualAdapterEngine {
        &mut self.residual_adapter
    }

    /// Computes per-head Q7 phase deltas from negative (`source`) and positive (`target`)
    /// post-RoPE query centroids via [`PhaseSteeringGeometryExt::compute_procrustes_phase_delta_for_geometry`].
    ///
    /// # Errors
    ///
    /// Returns an error if either slice length does not match `num_q_heads * rotary_dim`
    /// or if `liminika-core` returns an argument error.
    pub fn bake_layer_procrustes_q7(
        &self,
        neg_q_rot: &[f32],
        pos_q_rot: &[f32],
    ) -> Result<Vec<gcso_q7_t>, String> {
        let expected = self.geometry.q_rot_len();
        if neg_q_rot.len() != expected || pos_q_rot.len() != expected {
            return Err(format!(
                "expected q_rot length {}, got neg={} pos={}",
                expected,
                neg_q_rot.len(),
                pos_q_rot.len()
            ));
        }

        let mut neg_buf = AlignedF32Buffer32::new_zeroed(expected);
        let mut pos_buf = AlignedF32Buffer32::new_zeroed(expected);
        neg_buf.as_mut_slice().copy_from_slice(neg_q_rot);
        pos_buf.as_mut_slice().copy_from_slice(pos_q_rot);

        let src_aligned = AlignedSlice32::new(neg_buf.as_slice())
            .map_err(|e| format!("AlignedSlice32 src error: {e}"))?;
        let dst_aligned = AlignedSlice32::new(pos_buf.as_slice())
            .map_err(|e| format!("AlignedSlice32 dst error: {e}"))?;
        let mut q7_buf = AlignedQ7Buffer16::new_zeroed(self.geometry.num_q_heads);
        {
            let mut q7_mut = AlignedSliceMut16::new(q7_buf.as_mut_slice())
                .map_err(|e| format!("AlignedSliceMut16 error: {e}"))?;
            self.phase_engine
                .compute_procrustes_phase_delta_for_geometry(
                    &self.geometry,
                    &src_aligned,
                    &dst_aligned,
                    &mut q7_mut,
                )
                .map_err(|e| format!("compute_procrustes_phase_delta_for_geometry error: {e}"))?;
        }

        Ok(q7_buf.as_slice().to_vec())
    }

    /// Applies RIPA-bounded phase steering to a single token's rotated query slice
    /// (`num_q_heads * rotary_dim`) using [`PhaseSteeringGeometryExt::apply_phase_steering_for_geometry_safe`].
    ///
    /// Layers disabled in `geometry.layer_rope_mask` (NoPE layers) are skipped in O(1).
    ///
    /// # Errors
    ///
    /// Returns an error if slice lengths mismatch or `liminika-core` rejects the inputs.
    pub fn steer_layer_q_rot(
        &self,
        layer_idx: usize,
        q_rot_inout: &mut [f32],
        q7_deltas: &[gcso_q7_t],
        max_rad: f32,
    ) -> Result<(), String> {
        if max_rad <= 0.0 || !self.geometry.is_rope_layer(layer_idx) {
            return Ok(());
        }
        let expected = self.geometry.q_rot_len();
        if q_rot_inout.len() != expected {
            return Err(format!(
                "expected q_rot_inout length {}, got {}",
                expected,
                q_rot_inout.len()
            ));
        }
        if q7_deltas.len() != self.geometry.num_q_heads {
            return Err(format!(
                "expected {} q7 deltas, got {}",
                self.geometry.num_q_heads,
                q7_deltas.len()
            ));
        }

        let mut work_buf = AlignedF32Buffer32::new_zeroed(expected);
        work_buf.as_mut_slice().copy_from_slice(q_rot_inout);

        let q7_buf = AlignedQ7Buffer16::from_slice(q7_deltas);
        let q7_aligned = AlignedSlice16::new(q7_buf.as_slice())
            .map_err(|e| format!("AlignedSlice16 error: {e}"))?;
        {
            let mut tensor_mut = AlignedSliceMut32::new(work_buf.as_mut_slice())
                .map_err(|e| format!("AlignedSliceMut32 error: {e}"))?;
            self.phase_engine
                .apply_phase_steering_for_geometry_safe(
                    &self.geometry,
                    &mut tensor_mut,
                    &q7_aligned,
                    max_rad,
                )
                .map_err(|e| format!("apply_phase_steering_for_geometry_safe error: {e}"))?;
        }

        q_rot_inout.copy_from_slice(work_buf.as_slice());

        Ok(())
    }

    fn marshal_q_rot_and_logits<R, F>(
        &mut self,
        layer_idx: usize,
        q_rot_inout: &mut [f32],
        logits_opt: Option<&mut [f32]>,
        f: F,
    ) -> Result<Option<R>, String>
    where
        F: FnOnce(
            &mut Self,
            &mut AlignedSliceMut32<'_, f32>,
            Option<&mut AlignedSliceMut32<'_, f32>>,
        ) -> Result<R, String>,
    {
        if !self.geometry.is_rope_layer(layer_idx) {
            return Ok(None);
        }
        let expected = self.geometry.q_rot_len();
        if q_rot_inout.len() != expected {
            return Err(format!(
                "expected q_rot_inout length {}, got {}",
                expected,
                q_rot_inout.len()
            ));
        }

        let mut work_buf = AlignedF32Buffer32::new_zeroed(expected);
        work_buf.as_mut_slice().copy_from_slice(q_rot_inout);

        let mut logits_buf = logits_opt.as_ref().map(|l| {
            let mut b = AlignedF32Buffer32::new_zeroed(l.len());
            b.as_mut_slice().copy_from_slice(l);
            b
        });

        let out = {
            let mut tensor_mut = AlignedSliceMut32::new(work_buf.as_mut_slice())
                .map_err(|e| format!("AlignedSliceMut32 error: {e}"))?;
            let mut logits_mut_opt = match logits_buf.as_mut() {
                Some(b) => Some(
                    AlignedSliceMut32::new(b.as_mut_slice())
                        .map_err(|e| format!("AlignedSliceMut32 logits error: {e}"))?,
                ),
                None => None,
            };
            f(self, &mut tensor_mut, logits_mut_opt.as_mut())?
        };

        q_rot_inout.copy_from_slice(work_buf.as_slice());
        if let (Some(dst), Some(src)) = (logits_opt, logits_buf) {
            dst.copy_from_slice(src.as_slice());
        }

        Ok(Some(out))
    }

    /// Steers a single token's rotated query slice (`num_q_heads * rotary_dim`) using a
    /// [`ManifoldGatewaySignal`] composite so structural hyperparameters (`ripa_clamp_rad`,
    /// `qdps_min_step_rad`, `effective_inv_temp`) are derived autonomously from Gateway physics
    /// rather than manual scalar tuning.
    ///
    /// Layers disabled in `geometry.layer_rope_mask` (NoPE layers) are skipped in O(1),
    /// returning `Ok(None)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `q_rot_inout` length mismatches or `liminika-core` rejects the inputs.
    pub fn steer_layer_with_gateway<S: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        layer_idx: usize,
        signal: &S,
        trail_inout: &mut gcso_pointer_trail_t,
        unresolved_damping_u7: u8,
        q_rot_inout: &mut [f32],
        logits_opt: Option<&mut [f32]>,
    ) -> Result<Option<(GatewayFrame, CausalProvenanceReport)>, String> {
        self.marshal_q_rot_and_logits(layer_idx, q_rot_inout, logits_opt, |this, q, logits| {
            this.phase_engine
                .steer_gateway_signal_for_geometry_safe(
                    signal,
                    &this.geometry,
                    trail_inout,
                    unresolved_damping_u7,
                    q,
                    None,
                    logits,
                )
                .map_err(|e| format!("steer_gateway_signal_for_geometry_safe error: {e}"))
        })
    }

    /// Routes [`ManifoldGatewaySignal`] through the embedded [`ActionHubEngine`] via
    /// [`ActionHubGatewayExt::ingest_gateway_and_steer_for_geometry_safe`], allowing the hub's
    /// stigmergic trail density, FastPathLedger fast-path state, and frame energy to autonomously modulate
    /// steering across layers and tokens.
    ///
    /// # Errors
    ///
    /// Returns an error if `q_rot_inout` length mismatches or `liminika-core` rejects the inputs.
    pub fn steer_layer_with_action_hub<S: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        layer_idx: usize,
        signal: &S,
        trail_inout: &mut gcso_pointer_trail_t,
        unresolved_damping_u7: u8,
        q_rot_inout: &mut [f32],
        logits_opt: Option<&mut [f32]>,
    ) -> Result<Option<(GatewayFrame, CausalProvenanceReport)>, String> {
        self.marshal_q_rot_and_logits(layer_idx, q_rot_inout, logits_opt, |this, q, logits| {
            this.action_hub
                .ingest_gateway_and_steer_for_geometry_safe(
                    signal,
                    &this.geometry,
                    &mut this.ledger_slot,
                    trail_inout,
                    unresolved_damping_u7,
                    Some(q),
                    logits,
                )
                .map_err(|e| format!("ingest_gateway_and_steer_for_geometry_safe error: {e}"))
        })
    }

    /// Orchestrates a layer's RoPE steering via [`ActionHubEngine`] using stigmergic
    /// evaporation + resonance-weighted reinforcement and records the selected
    /// [`StigmergicActionDecision`] inside `action_hub.last_decision`.
    ///
    /// # Errors
    ///
    /// Returns an error if `q_rot_inout` length mismatches or `liminika-core` rejects the inputs.
    #[allow(clippy::too_many_arguments)]
    pub fn orchestrate_layer_rope_with_action_hub<S: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        layer_idx: usize,
        signal: &S,
        trail_inout: &mut gcso_pointer_trail_t,
        entropy_h_tilde_opt: Option<f32>,
        unresolved_damping_u7: u8,
        q_rot_inout: &mut [f32],
        logits_opt: Option<&mut [f32]>,
    ) -> Result<
        Option<(
            GatewayFrame,
            CausalProvenanceReport,
            StigmergicActionDecision,
        )>,
        String,
    > {
        self.marshal_q_rot_and_logits(layer_idx, q_rot_inout, logits_opt, |this, q, logits| {
            let entropy_h_tilde =
                entropy_h_tilde_opt.unwrap_or(this.action_hub.last_entropy_h_tilde);
            let scale = StigmergicScaleGranularity::from_layer_ratio(
                layer_idx,
                this.geometry.num_layers.max(1),
            );
            let (frame, report, decision) = this
                .action_hub
                .orchestrate_gateway_layer_with_policy_and_scale_safe(
                    signal,
                    &this.geometry,
                    &this.threshold_policy,
                    &this.system_constraint,
                    &this.probe_baseline,
                    scale,
                    entropy_h_tilde,
                    &mut this.ledger_slot,
                    trail_inout,
                    unresolved_damping_u7,
                    Some(q),
                    None,
                    logits,
                )
                .map_err(|e| {
                    format!("orchestrate_gateway_layer_with_policy_and_scale_safe error: {e}")
                })?;
            this.action_hub.last_decision = decision as u8;
            this.decision_counts[(decision as usize).min(6)] += 1;
            Ok((frame, report, decision))
        })
    }

    /// Executes norm-adaptive Rank-1 residual residual injection only when [`ActionHubEngine`]'s latest
    /// [`StigmergicActionDecision`] authorizes full orchestration (`FullOrchestration` or
    /// `CrystallizeMacroBasin`). Returns `Ok(None)` when bypassed by ActionHub.
    ///
    /// # Errors
    ///
    /// Returns an error if slice lengths mismatch or `liminika-core` rejects the inputs.
    pub fn orchestrate_layer_srl_with_action_hub<S: ManifoldGatewaySignal + ?Sized>(
        &self,
        signal: &S,
        hidden_inout: &mut [f32],
        u_vec: &[f32],
        v_vec: &[f32],
    ) -> Result<Option<(GatewayFrame, f32)>, String> {
        let decision = StigmergicActionDecision::from_u8(self.action_hub.last_decision);
        if matches!(
            decision,
            StigmergicActionDecision::FullOrchestration
                | StigmergicActionDecision::CrystallizeMacroBasin
        ) {
            let res = self.apply_srl_with_gateway(signal, hidden_inout, u_vec, v_vec)?;
            Ok(Some(res))
        } else {
            Ok(None)
        }
    }

    /// Applies norm-adaptive Rank-1 residual residual injection driven purely by a [`ManifoldGatewaySignal`]
    /// via [`GatewayRank1SteeringExt::apply_gateway_rank1_hidden_fused`] without manual scale tuning.
    ///
    /// # Errors
    ///
    /// Returns an error if slice lengths mismatch or `liminika-core` rejects the inputs.
    pub fn apply_srl_with_gateway<S: ManifoldGatewaySignal + ?Sized>(
        &self,
        signal: &S,
        hidden_inout: &mut [f32],
        u_vec: &[f32],
        v_vec: &[f32],
    ) -> Result<(GatewayFrame, f32), String> {
        let d = hidden_inout.len();
        if d == 0 || u_vec.len() != d || v_vec.len() != d {
            return Err(format!(
                "Rank-1 residual hidden/u/v length mismatch: hidden={}, u={}, v={}",
                d,
                u_vec.len(),
                v_vec.len()
            ));
        }

        let mut y_buf = AlignedF32Buffer32::new_zeroed(d);
        let mut x_buf = AlignedF32Buffer32::new_zeroed(d);
        let mut u_buf = AlignedF32Buffer32::new_zeroed(d);
        let mut v_buf = AlignedF32Buffer32::new_zeroed(d);
        y_buf.as_mut_slice().copy_from_slice(hidden_inout);
        x_buf.as_mut_slice().copy_from_slice(hidden_inout);
        u_buf.as_mut_slice().copy_from_slice(u_vec);
        v_buf.as_mut_slice().copy_from_slice(v_vec);

        let (mut frame, frm_w) = signal.synthesize_gateway_frame_with_words();
        if let Some(circuit) = self.active_circuit() {
            let _ = circuit.modulate_gateway_frame_at_scale(
                &mut frame,
                &frm_w,
                &frm_w,
                StigmergicScaleGranularity::MacroCoalition,
            );
        }
        if self.core_layers.liminal_layer.adaptation_epochs > 0 || self.ego_field.is_active() {
            frame = self.core_layers.modulate_gateway_frame_branchless(
                frame,
                &self.last_horizon,
                &self.last_venn_report,
            );
            if self.ego_field.is_active() {
                let mut ego = self.ego_field;
                let mut stack = self.core_layers;
                frame = ego.intervene_on_layers_and_frame_branchless(
                    &mut stack,
                    &self.last_boundary_delta,
                    &self.last_horizon,
                    &self.last_venn_report,
                    frame,
                );
            }
        }

        let gain = {
            let mut y_mut = AlignedSliceMut32::new(y_buf.as_mut_slice())
                .map_err(|e| format!("AlignedSliceMut32 y error: {e}"))?;
            let x_aligned = AlignedSlice32::new(x_buf.as_slice())
                .map_err(|e| format!("AlignedSlice32 x error: {e}"))?;
            let u_aligned = AlignedSlice32::new(u_buf.as_slice())
                .map_err(|e| format!("AlignedSlice32 u error: {e}"))?;
            let v_aligned = AlignedSlice32::new(v_buf.as_slice())
                .map_err(|e| format!("AlignedSlice32 v error: {e}"))?;
            self.residual_adapter
                .apply_gateway_frame_rank1_hidden_fused(
                    &frame, &mut y_mut, &x_aligned, &u_aligned, &v_aligned,
                )
                .map_err(|e| format!("apply_gateway_frame_rank1_hidden_fused error: {e}"))?
        };

        hidden_inout.copy_from_slice(y_buf.as_slice());
        Ok((frame, gain))
    }

    /// Evaluates Top-2 sparse surrogate entropy via [`EntropyController::eval_surrogate_and_step`]
    /// and updates `action_hub.last_entropy_h_tilde` without rescaling `logits`.
    ///
    /// # Errors
    ///
    /// Returns an error if `logits` is empty or `EntropyController` reports an error.
    pub fn measure_edbc_entropy(&mut self, logits: &[f32]) -> Result<gcso_edbc_state_t, String> {
        if logits.is_empty() {
            return Err("logits slice must not be empty".to_string());
        }
        let mut max1 = f32::NEG_INFINITY;
        let mut max2 = f32::NEG_INFINITY;
        for &v in logits.iter() {
            if v > max1 {
                max2 = max1;
                max1 = v;
            } else if v > max2 {
                max2 = v;
            }
        }
        let sum_exp: f32 = logits
            .iter()
            .map(|&v| (v - max1).exp())
            .sum::<f32>()
            .max(1e-12);
        let p_max = 1.0 / sum_exp;
        let p_2nd = if max2.is_finite() {
            (max2 - max1).exp() / sum_exp
        } else {
            0.0
        };

        let mut state_out = gcso_edbc_state_t::default();
        let _temp = self
            .entropy_controller
            .eval_surrogate_and_step(p_max, p_2nd, 0.0, 0.0, 1.0, 0.1, 0.1, &mut state_out)
            .map_err(|e| format!("eval_surrogate_and_step error: {e}"))?;
        self.action_hub.last_entropy_h_tilde = state_out.moving_z_entropy;
        Ok(state_out)
    }

    /// Evaluates Top-2 sparse surrogate entropy via [`Self::measure_edbc_entropy`]
    /// and scales `logits` by the resulting adaptive `sample_temperature`.
    ///
    /// # Errors
    ///
    /// Returns an error if `logits` is empty or `EntropyController` reports an error.
    pub fn step_edbc_logits(&mut self, logits: &mut [f32]) -> Result<gcso_edbc_state_t, String> {
        let state_out = self.measure_edbc_entropy(logits)?;
        let temp = state_out.sample_temperature;
        if temp.is_finite() && temp > 1e-4 && (temp - 1.0).abs() > 1e-4 {
            let inv_temp = 1.0 / temp;
            for v in logits.iter_mut() {
                *v *= inv_temp;
            }
        }
        Ok(state_out)
    }

    /// Applies Rank-1 Rank-1 residual residual injection (`y <- y + s * u * (v^T * y)`) to a hidden-state
    /// vector via [`Rank1ResidualAdapter::eval_rank1_slices_fused`].
    ///
    /// # Errors
    ///
    /// Returns an error if slice lengths mismatch or `ResidualAdapterEngine` rejects inputs.
    pub fn apply_srl_hidden_fused(
        &self,
        hidden_inout: &mut [f32],
        u_vec: &[f32],
        v_vec: &[f32],
        scale_factor: f32,
    ) -> Result<(), String> {
        if scale_factor == 0.0 {
            return Ok(());
        }
        let d = hidden_inout.len();
        if d == 0 || u_vec.len() != d || v_vec.len() != d {
            return Err(format!(
                "Rank-1 residual hidden/u/v length mismatch: hidden={}, u={}, v={}",
                d,
                u_vec.len(),
                v_vec.len()
            ));
        }

        let mut y_buf = AlignedF32Buffer32::new_zeroed(d);
        let mut x_buf = AlignedF32Buffer32::new_zeroed(d);
        let mut u_buf = AlignedF32Buffer32::new_zeroed(d);
        let mut v_buf = AlignedF32Buffer32::new_zeroed(d);
        y_buf.as_mut_slice().copy_from_slice(hidden_inout);
        x_buf.as_mut_slice().copy_from_slice(hidden_inout);
        u_buf.as_mut_slice().copy_from_slice(u_vec);
        v_buf.as_mut_slice().copy_from_slice(v_vec);

        let mut y_mut = AlignedSliceMut32::new(y_buf.as_mut_slice())
            .map_err(|e| format!("AlignedSliceMut32 y error: {e}"))?;
        let x_aligned = AlignedSlice32::new(x_buf.as_slice())
            .map_err(|e| format!("AlignedSlice32 x error: {e}"))?;
        let u_aligned = AlignedSlice32::new(u_buf.as_slice())
            .map_err(|e| format!("AlignedSlice32 u error: {e}"))?;
        let v_aligned = AlignedSlice32::new(v_buf.as_slice())
            .map_err(|e| format!("AlignedSlice32 v error: {e}"))?;

        self.residual_adapter
            .eval_rank1_slices_fused(
                &mut y_mut,
                &x_aligned,
                &u_aligned,
                &v_aligned,
                None,
                scale_factor,
            )
            .map_err(|e| format!("eval_rank1_slices_fused error: {e}"))?;

        hidden_inout.copy_from_slice(y_buf.as_slice());
        Ok(())
    }
}
