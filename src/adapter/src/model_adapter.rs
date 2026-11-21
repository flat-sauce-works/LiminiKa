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

/// Adapter-level [`LatentHorizonSlot`] that remains dormant during coherent nominal tokens
/// and dynamically widens Core Layer 0 guard apertures + enables `ACTION_MODE_PHASE_REPULSE`
/// when instability, epistemic conflict, or `Liminal` void excursion persists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoherenceHorizonGuardSlot {
    /// Bitmask (`1 << k`) claimed inside `active_horizon_slots_mask` (default `1 << 0`).
    pub slot_bit: u32,
    /// Minimum consecutive destabilized steps before sliding in (default `1`).
    pub slide_in_dwell: u16,
    /// Minimum consecutive quiescent steps before retracting out (default `2`).
    pub slide_out_dwell: u16,
    /// Layer 0 aperture bits activated while this boundary is slid in.
    pub guard_aperture_bits: u32,
    /// Total times `on_slide_in_modulate` fired.
    pub activations: u64,
    /// Total times `on_slide_out` cleanly retracted this boundary.
    pub retractions: u64,
    saved_aperture_mask: u32,
}

impl Default for CoherenceHorizonGuardSlot {
    fn default() -> Self {
        Self {
            slot_bit: 1,
            slide_in_dwell: 1,
            slide_out_dwell: 2,
            guard_aperture_bits: (1 << 3) | (1 << 7),
            activations: 0,
            retractions: 0,
            saved_aperture_mask: 0,
        }
    }
}

impl LatentHorizonSlot for CoherenceHorizonGuardSlot {
    #[inline(always)]
    fn horizon_slot_mask(&self) -> u32 {
        self.slot_bit
    }

    #[inline(always)]
    fn min_slide_in_dwell(&self) -> u16 {
        self.slide_in_dwell.max(1)
    }

    #[inline(always)]
    fn min_slide_out_dwell(&self) -> u16 {
        self.slide_out_dwell.max(1)
    }

    #[inline]
    fn on_slide_in_modulate(
        &mut self,
        frame: &mut GatewayFrame,
        stack: &mut AutonomousCoreLayerStack,
        trigger: &HorizonTriggerInput,
    ) {
        if self.activations == self.retractions {
            self.saved_aperture_mask = stack.liminal_layer.active_aperture_mask;
            self.activations = self.activations.saturating_add(1);
        }
        stack.liminal_layer.active_aperture_mask |= self.guard_aperture_bits;
        if trigger.repulsion_severity > 0.25 {
            frame.action_mode_flags |= ACTION_MODE_PHASE_REPULSE;
        }
    }

    #[inline]
    fn on_slide_out(
        &mut self,
        _frame: &mut GatewayFrame,
        stack: &mut AutonomousCoreLayerStack,
        _trigger: &HorizonTriggerInput,
    ) {
        let added_only = self.guard_aperture_bits & !self.saved_aperture_mask;
        stack.liminal_layer.active_aperture_mask &= !added_only;
        self.retractions = self.retractions.saturating_add(1);
    }
}

/// Cold-path [`HorizonEscalationProvider`] backed by a compiled [`TextAttractorCorpus`],
/// swapping the strongest Guard (under conflict/repulsion) or Fact phase signature into an
/// anonymous free workbench lane only when Hot-Path triggers detect destabilization.
#[derive(Debug)]
pub struct CorpusRecallEscalationProvider<'a> {
    corpus: &'a TextAttractorCorpus,
    /// Minimum lane weight assigned to escalated corpus entries.
    pub min_weight: f32,
    /// Number of cold-path escalation swaps performed.
    pub escalation_calls: u64,
}

impl<'a> CorpusRecallEscalationProvider<'a> {
    /// Creates a new escalation provider backed by `corpus`.
    #[must_use]
    pub const fn new(corpus: &'a TextAttractorCorpus) -> Self {
        Self {
            corpus,
            min_weight: 0.5,
            escalation_calls: 0,
        }
    }
}

impl HorizonEscalationProvider for CorpusRecallEscalationProvider<'_> {
    fn try_escalate_horizon(
        &mut self,
        req: &HorizonEscalationRequest,
        slot_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<Option<f32>> {
        if !req.trigger.is_destabilized() || self.corpus.entries().is_empty() {
            return Ok(None);
        }
        let prefer_guard = req.trigger.epistemic_regime == EpistemicRegime::InternalConflict as u8
            || req.trigger.repulsion_severity > 0.25;
        let preferred_role = if prefer_guard {
            AttractorRole::Guard
        } else {
            AttractorRole::Fact
        };

        let selected = self
            .corpus
            .entries()
            .iter()
            .filter(|e| e.role == preferred_role && !e.per_layer_q7.is_empty())
            .max_by(|a, b| a.energy.total_cmp(&b.energy))
            .or_else(|| {
                self.corpus
                    .entries()
                    .iter()
                    .filter(|e| !e.per_layer_q7.is_empty())
                    .max_by(|a, b| a.energy.total_cmp(&b.energy))
            });

        let Some(entry) = selected else {
            return Ok(None);
        };
        let n_layers = entry.per_layer_q7.len();
        if n_layers == 0 {
            return Ok(None);
        }
        if n_layers == 1 {
            *slot_phase_out = entry.per_layer_q7[0];
        } else {
            let scale_q7 = usize::from(req.layer_scale_q7.min(127));
            let scaled = scale_q7 * (n_layers - 1);
            let idx0 = (scaled / 127).min(n_layers - 1);
            let idx1 = (idx0 + 1).min(n_layers - 1);
            let frac = (scaled % 127) as i32;
            let w0 = 127 - frac;
            let a0 = &entry.per_layer_q7[idx0];
            let a1 = &entry.per_layer_q7[idx1];
            for (dst, (&v0, &v1)) in slot_phase_out.iter_mut().zip(a0.iter().zip(a1.iter())) {
                let interp = (i32::from(v0) * w0 + i32::from(v1) * frac + 63) / 127;
                *dst = interp.clamp(-127, 127) as gcso_q7_t;
            }
        }
        self.escalation_calls = self.escalation_calls.saturating_add(1);
        Ok(Some(entry.energy.max(self.min_weight)))
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
    substrate: Box<StigmergicSubstrate>,
    ledger_slot: gcso_ledger_slot_t,
    telemetry: CoreTelemetryBank,
    entropy_controller: EntropyController,
    residual_adapter: ResidualAdapterEngine,
    system_constraint: SystemConstraint<()>,
    probe_baseline: ManifoldProbeBaseline,
    threshold_policy: AdaptiveThresholdPolicy,
    calibrated_base_thresholds: StigmergicThresholds,
    core_layers: AutonomousCoreLayerStack,
    configured_ego: LiminalEgoState,
    last_boundary_delta: LiminalBoundaryDelta,
    last_horizon: EpistemicHorizonReading,
    last_venn_report: StigmergicOverlapSerendipityReport,
    last_ensemble_report: Option<StigmergicFragmentEnsembleReport>,
    horizon_dwell: HorizonDwellState,
    active_horizon_slots_mask: u32,
    coherence_guard_slot: CoherenceHorizonGuardSlot,
    residual_spillover: Box<DefaultResidualSpilloverSlot>,
    escalation_boundary_dirty: bool,
    last_escalated_regime: u8,
    escalated_workbench_lane: Option<u8>,
    sandbox_workbench_lane: Option<u8>,
    active_workbench_preset: WorkbenchPreset,
    compute_pool: AnonymousComputePool,
    system_spine: SystemSpineState,
    last_venn_partition: EpistemicPartition,
    distortion_tension: DistortionTension,
    ego_differential_net: EgoDifferentialNet,
    sandbox_promotion_streak: u8,
    prev_sandbox_coherence: f32,
    prev_sandbox_assim_drift: f32,
    sandbox_promotions: u64,
    turn_guard_active: bool,
    passive_saturation_probe_steps: u8,
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
        let mut substrate = Box::new(
            StigmergicSubstrate::with_dimensions_and_thresholds(
                geometry.rotary_dim as u32,
                geometry.num_q_heads as u32,
                16,
                StigmergicThresholds::default(),
            )
            .map_err(|e| {
                format!("StigmergicSubstrate::with_dimensions_and_thresholds failed: {e}")
            })?,
        );
        substrate.last_entropy_h_tilde = 0.35;
        let ledger_slot = gcso_ledger_slot_t::default();
        let entropy_controller = EntropyController::new(gcso_edbc_state_t::default());
        let residual_adapter = ResidualAdapterEngine::new(1.0, 0, 1);
        let system_constraint = SystemConstraint::new((), 0.0, 0.0, 0xFF);
        let mut probe_baseline = ManifoldProbeBaseline::new();
        probe_baseline.active_heads = geometry.num_q_heads.min(64) as u16;
        probe_baseline.steered_layers = Self::compute_steered_layer_count(&geometry);
        let threshold_policy = AdaptiveThresholdPolicy::default();
        let calibrated_base_thresholds = threshold_policy.base;
        let mut compute_pool = AnonymousComputePool::default();
        let n_slots = geometry.num_layers.min(64);
        let active_slots = if n_slots == 64 {
            u64::MAX
        } else {
            (1_u64 << n_slots) - 1
        };
        let pinned_slots = if n_slots == 0 {
            0
        } else if n_slots >= 12 {
            // Protect head (0..4) and tail ((n-4)..n) layers as irreversible boundary anchors.
            0x0000_000F_u64 | (0x0000_000F_u64 << (n_slots - 4))
        } else {
            1_u64 | (1_u64 << (n_slots - 1))
        };
        let reversible_slots = active_slots & !pinned_slots;
        compute_pool.set_slot_masks(active_slots, pinned_slots, 0, reversible_slots);

        Ok(Self {
            geometry,
            phase_engine,
            substrate,
            ledger_slot,
            telemetry: CoreTelemetryBank::default(),
            entropy_controller,
            residual_adapter,
            system_constraint,
            probe_baseline,
            threshold_policy,
            calibrated_base_thresholds,
            core_layers: AutonomousCoreLayerStack::new(),
            configured_ego: LiminalEgoState::new(),
            last_boundary_delta: LiminalBoundaryDelta::default(),
            last_horizon: EpistemicHorizonReading {
                total_memory_support: 1.0,
                peak_positive_affinity: 1.0,
                horizon_gate: 1.0,
                regime_code: EpistemicRegime::KnownCoherent as u8,
                ..EpistemicHorizonReading::default()
            },
            last_venn_report: StigmergicOverlapSerendipityReport::default(),
            last_ensemble_report: None,
            horizon_dwell: HorizonDwellState::default(),
            active_horizon_slots_mask: 0,
            coherence_guard_slot: CoherenceHorizonGuardSlot::default(),
            residual_spillover: Box::default(),
            escalation_boundary_dirty: true,
            last_escalated_regime: EpistemicRegime::KnownCoherent as u8,
            escalated_workbench_lane: None,
            sandbox_workbench_lane: None,
            active_workbench_preset: WorkbenchPreset::BalancedDialog,
            compute_pool,
            system_spine: SystemSpineState::new(),
            last_venn_partition: EpistemicPartition::default(),
            distortion_tension: DistortionTension::new(false),
            ego_differential_net: EgoDifferentialNet::new_empty(1),
            sandbox_promotion_streak: 0,
            prev_sandbox_coherence: 0.0,
            prev_sandbox_assim_drift: 1.0,
            sandbox_promotions: 0,
            turn_guard_active: false,
            passive_saturation_probe_steps: 0,
        })
    }

    /// Returns the per-regime [`StigmergicActionDecision`] invocation counts (`0..=6`) from
    /// Core Layer 1 (`StigmergicLayerTelemetry`).
    #[must_use]
    pub const fn substrate_decision_counts(&self) -> &[u64; 7] {
        &self.telemetry.layer1.decision_counts
    }

    /// Resets the `2 Primitives x 2 Drivers` telemetry bank and restores initial neutral entropy (`0.35`)
    /// for a fresh evaluation pass.
    pub fn reset_substrate_stats(&mut self) {
        self.telemetry.reset();
        self.substrate.last_entropy_h_tilde = 0.35;
    }

    /// Resets core layers, horizon dwell/slots, spillover cache, thresholds, and `2 Primitives x 2 Drivers`
    /// telemetry back to clean defaults for fair ablation pass isolation.
    pub fn reset_horizon_and_core_state(&mut self) {
        let saved_hops = self.ego_differential_net.max_hops;
        self.core_layers = AutonomousCoreLayerStack::new();
        self.configured_ego = LiminalEgoState::new();
        self.last_boundary_delta = LiminalBoundaryDelta::default();
        self.last_horizon = EpistemicHorizonReading {
            regime_code: EpistemicRegime::KnownCoherent as u8,
            ..EpistemicHorizonReading::default()
        };
        self.last_venn_report = StigmergicOverlapSerendipityReport::default();
        self.last_ensemble_report = None;
        self.horizon_dwell = HorizonDwellState::default();
        self.active_horizon_slots_mask = 0;
        self.coherence_guard_slot = CoherenceHorizonGuardSlot::default();
        *self.residual_spillover = DefaultResidualSpilloverSlot::default();
        self.threshold_policy.base = self.calibrated_base_thresholds;
        self.substrate.thresholds = self.calibrated_base_thresholds;
        self.substrate.last_decision = 0;
        self.substrate.last_anomaly_severity = 0.0;
        self.escalation_boundary_dirty = true;
        self.last_escalated_regime = EpistemicRegime::KnownCoherent as u8;
        self.escalated_workbench_lane = None;
        self.sandbox_workbench_lane = None;
        let _ = self.substrate.release_all_lanes();
        self.active_workbench_preset = WorkbenchPreset::BalancedDialog;
        self.system_spine = SystemSpineState::new();
        self.last_venn_partition = EpistemicPartition::default();
        self.distortion_tension = DistortionTension::new(false);
        self.ego_differential_net = EgoDifferentialNet::new_empty(saved_hops);
        self.sandbox_promotion_streak = 0;
        self.prev_sandbox_coherence = 0.0;
        self.prev_sandbox_assim_drift = 1.0;
        self.sandbox_promotions = 0;
        self.turn_guard_active = false;
        self.passive_saturation_probe_steps = 0;
        self.telemetry.set_policy(self.telemetry.policy());
        self.reset_substrate_stats();
    }

    /// Returns a shared reference to the `2 Primitives x 2 Drivers` [`CoreTelemetryBank`].
    #[must_use]
    pub const fn telemetry_bank(&self) -> &CoreTelemetryBank {
        &self.telemetry
    }

    /// Returns a shared reference to the label-free [`AnonymousComputePool`].
    #[must_use]
    pub const fn compute_pool(&self) -> &AnonymousComputePool {
        &self.compute_pool
    }

    /// Returns a mutable reference to the label-free [`AnonymousComputePool`].
    #[must_use]
    pub fn compute_pool_mut(&mut self) -> &mut AnonymousComputePool {
        &mut self.compute_pool
    }

    /// Returns the 64-byte unified [`CorePrimitiveState`] (`Liminal` + `Stigmergy`).
    #[must_use]
    pub const fn core_primitive_state(&self) -> CorePrimitiveState {
        self.core_layers.core_primitive_state()
    }

    /// Returns a shared reference to the mechanical [`SystemSpineState`] (`pinned` isolation barrier).
    #[must_use]
    pub const fn system_spine(&self) -> &SystemSpineState {
        &self.system_spine
    }

    /// Returns the most recent branchless 4-region [`EpistemicPartition`].
    #[must_use]
    pub const fn last_venn_partition(&self) -> &EpistemicPartition {
        &self.last_venn_partition
    }

    /// Returns a shared reference to the asymmetric homeostatic [`DistortionTension`] controller.
    #[must_use]
    pub const fn distortion_tension(&self) -> &DistortionTension {
        &self.distortion_tension
    }

    /// Returns a shared reference to the P-Frame differential chain [`EgoDifferentialNet`].
    #[must_use]
    pub const fn ego_differential_net(&self) -> &EgoDifferentialNet {
        &self.ego_differential_net
    }

    /// Returns cumulative multi-turn Known Unknown ($O \setminus S$) sandbox promotions.
    #[must_use]
    pub const fn sandbox_promotions(&self) -> u64 {
        self.sandbox_promotions
    }

    /// Updates the maximum hop bound (`1..=4`) of [`EgoDifferentialNet`] and rebuilds active P-Frame nodes.
    pub fn set_ego_max_hops(&mut self, max_hops: u8) {
        self.ego_differential_net.max_hops = max_hops.clamp(1, 4);
        if self.configured_ego.is_active() {
            self.sync_ego_differential_net_from_configured();
        }
    }

    /// Advances 1 Hot-Path [`DistortionTension`] step using the latest P-Frame delta norm and Venn sensitivity.
    pub fn step_distortion_tension_hot_path(
        &mut self,
        exit_entropy_q7: u8,
        guard_proximity_q7: u8,
        active_overdrive: bool,
    ) {
        self.distortion_tension.step_hot_path_branchless(
            self.ego_differential_net.last_delta_norm_q7,
            self.last_venn_partition.boundary_sensitivity_q7,
            exit_entropy_q7,
            guard_proximity_q7,
            active_overdrive,
        );
    }

    /// Resolves step/turn-local effective `(guard_active, repulsion_active)` signals for
    /// `SystemSpineState` bypass veto gating without latching onto session-persistent
    /// `StigmergicSubstrate::slot_repulsion_mask`.
    #[must_use]
    pub fn effective_spine_guard_and_repulsion(&self, obs_repulsion: bool) -> (bool, bool) {
        let guard_slot_slid_in = (self.coherence_guard_slot.activations
            > self.coherence_guard_slot.retractions)
            && (self.last_horizon.conflict_severity > 0.25);
        let guard_active =
            self.turn_guard_active || guard_slot_slid_in || self.passive_saturation_probe_steps > 0;
        let action_repulse = liminika_core::stigmergy::StigmergicActionDecision::from_u8(
            self.substrate.last_decision,
        )
            == liminika_core::stigmergy::StigmergicActionDecision::RepulsionEscape;
        let entropy_repulse = self.entropy_controller.is_repulsion_mode();
        let repulsion_active = obs_repulsion || action_repulse || entropy_repulse;
        (guard_active, repulsion_active)
    }

    /// Evaluates token-level whole-steering bypass on `obs` jointly with [`SystemSpineState`]
    /// physical `pinned` / guard / repulsion vetoes, returning the active `SPINE_BYPASS_VETO_*`
    /// bitmask.
    pub fn evaluate_whole_token_bypass_with_spine(
        &mut self,
        mut obs: TokenBypassObservation,
        dispatch: &mut ComputeDispatchMask,
    ) -> u8 {
        let (guard_active, repulsion_active) =
            self.effective_spine_guard_and_repulsion(obs.repulsion_active != 0);
        obs.repulsion_active = u8::from(repulsion_active);
        self.passive_saturation_probe_steps = self.passive_saturation_probe_steps.saturating_sub(1);
        let veto = self
            .compute_pool
            .evaluate_whole_token_steering_bypass_with_spine(
                obs,
                &mut self.system_spine,
                guard_active,
                dispatch,
            );
        if dispatch.whole_token_steering_bypass != 0 {
            self.telemetry.record_whole_token_bypass();
        }
        veto
    }

    /// Evaluates token-level whole-steering bypass on `obs`, mutating `dispatch` and mirroring
    /// active bypass state onto [`SystemSpineState`].
    pub fn evaluate_whole_token_bypass(
        &mut self,
        obs: TokenBypassObservation,
        dispatch: &mut ComputeDispatchMask,
    ) -> bool {
        let _ = self.evaluate_whole_token_bypass_with_spine(obs, dispatch);
        dispatch.whole_token_steering_bypass != 0
    }

    /// Exports active [`EgoDifferentialNet`] P-Frame nodes as model-agnostic
    /// [`LogicalPFrameDescriptor`]s for `.gcso` v2 persistence.
    #[must_use]
    pub fn export_logical_pframe_topology(&self) -> Vec<LogicalPFrameDescriptor> {
        let mut out = Vec::new();
        for (idx, node) in self.ego_differential_net.nodes.iter().enumerate() {
            if ((self.ego_differential_net.active_node_mask >> idx) & 1) != 0 && node.is_active() {
                out.push(LogicalPFrameDescriptor {
                    node_id: node.node_id,
                    anchor_ref_id: node.anchor_ref_id,
                    next_hop_idx: node.next_hop_idx,
                    hop_depth: node.hop_depth,
                    venn_trigger_mask: node.venn_trigger_mask,
                    venn_region_selector: node.venn_region_selector,
                    pheromone_q7: node.pheromone_q7,
                    gain_q7: node.gain_q7,
                    intent_delta: node.intent_delta,
                });
            }
        }
        out
    }

    /// Re-bakes persisted [`LogicalPFrameDescriptor`]s into this model's 64-head Q7
    /// [`EgoDifferentialNet`] upon `.gcso` v2 load or cross-model transcoding.
    pub fn bake_logical_pframe_topology(&mut self, descriptors: &[LogicalPFrameDescriptor]) {
        let mut net = EgoDifferentialNet::new_empty(self.ego_differential_net.max_hops);
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
            let mut phases = [0 as gcso_q7_t; MAX_ATTENTION_HEADS];
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
                    phases[base_h + k] = (base_s + mod_off).clamp(-127, 127) as gcso_q7_t;
                }
            }
            let mut node = DistortionDelta::new_empty(desc.node_id, desc.anchor_ref_id);
            node.next_hop_idx = desc.next_hop_idx;
            node.hop_depth = desc.hop_depth;
            node.venn_trigger_mask = desc.venn_trigger_mask;
            node.venn_region_selector = desc.venn_region_selector;
            node.pheromone_q7 = desc.pheromone_q7;
            node.gain_q7 = desc.gain_q7;
            node.intent_delta = desc.intent_delta;
            node.delta_phase_q7 = phases;
            node.flags = DISTORTION_DELTA_FLAG_ACTIVE | DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID;
            net.nodes[slot] = node;
            net.active_node_mask |= 1u8 << slot;
        }
        self.ego_differential_net = net;
    }

    fn sync_ego_differential_net_from_configured(&mut self) {
        let axes = self.configured_ego.extract_intent_axes();
        let max_hops = self.ego_differential_net.max_hops;
        self.ego_differential_net =
            EgoDifferentialNet::from_ego_state_and_axes(&self.configured_ego, &axes, max_hops);
        self.distortion_tension = DistortionTension::new(self.configured_ego.is_active());
    }

    /// Evaluates external `+ System` Gauge-Zero pressure jointly with [`SystemSpineState`]
    /// physical `pinned` / guard / repulsion vetoes and emits a pre-execution
    /// [`ComputeDispatchMask`] for this adapter's anonymous compute slots.
    pub fn step_compute_dispatch_with_spine(
        &mut self,
        reading: SystemPressureReading,
        repulsion_active: bool,
    ) -> ComputeDispatchMask {
        let (guard_active, eff_repulsion) =
            self.effective_spine_guard_and_repulsion(repulsion_active);
        self.compute_pool.step_dispatch_with_spine(
            reading,
            &self.system_spine,
            guard_active,
            eff_repulsion,
        )
    }

    /// Evaluates external `+ System` Gauge-Zero pressure and emits a pre-execution
    /// [`ComputeDispatchMask`] for this adapter's anonymous compute slots.
    pub fn step_compute_dispatch(&mut self, reading: SystemPressureReading) -> ComputeDispatchMask {
        self.step_compute_dispatch_with_spine(reading, false)
    }

    /// Updates slot contribution pheromones strictly from an exit scalar reward (`0..=127`)
    /// without mid-layer D2H synchronization.
    pub fn commit_compute_exit_reward(&mut self, executed_mask: u64, reward_q7: u8) {
        self.compute_pool
            .commit_exit_reward(executed_mask, reward_q7);
    }

    /// Triggers immediate 1-step full-layer recovery in the anonymous compute pool with asymmetric
    /// re-bypass dwell hysteresis (called on exit entropy spike, `InternalConflict`, or `LiminalExterior`).
    pub fn signal_compute_instability(&mut self) {
        self.compute_pool.signal_exit_instability();
    }

    /// Updates the [`TelemetryCapturePolicy`] (`Pinned` vs `Auto` `Coarse`/`Standard`/`Detailed`).
    pub fn set_telemetry_policy(&mut self, policy: TelemetryCapturePolicy) {
        self.telemetry.set_policy(policy);
    }

    /// Pulls a Lazy-Pull [`CoreTelemetrySnapshot`] computed inside `liminika-core`.
    #[must_use]
    pub fn telemetry_snapshot(&self) -> CoreTelemetrySnapshot {
        self.telemetry.snapshot(
            &self.core_layers,
            &self.last_horizon,
            0,
            0,
            self.active_horizon_slots_mask,
        )
    }

    /// Evaluates bounded Slow-Loop epoch threshold adjustments over continuous `Standard`/`Detailed`
    /// samples without touching Hot-Path loops.
    pub fn evaluate_slow_loop_telemetry(
        &mut self,
        min_samples: u64,
    ) -> SlowLoopTelemetryAdjustment {
        let base = self.calibrated_base_thresholds;
        self.telemetry.evaluate_slow_loop_epoch(&base, min_samples)
    }

    /// Evaluates bounded Slow-Loop epoch adjustments and applies them to [`AdaptiveThresholdPolicy`],
    /// [`StigmergicSubstrate`] thresholds, [`DistortionTension`], and [`EgoDifferentialNet`] GOP
    /// maintenance when `applied == true`.
    pub fn apply_slow_loop_telemetry_epoch(
        &mut self,
        min_samples: u64,
    ) -> SlowLoopTelemetryAdjustment {
        let adj = self.evaluate_slow_loop_telemetry(min_samples);
        if adj.applied {
            self.threshold_policy.base = adj.tuned_thresholds;
            let neutral_trail = gcso_pointer_trail_t::default();
            self.substrate.thresholds = self.threshold_policy.resolve_thresholds(
                &self.system_constraint,
                &self.probe_baseline,
                &neutral_trail,
                self.probe_baseline.baseline_entropy,
            );
            let next_dwell = (2_i16 + i16::from(adj.suggested_slide_out_dwell_delta)).clamp(1, 8);
            self.coherence_guard_slot.slide_out_dwell = next_dwell as u16;
            self.apply_distortion_ego_feedback(&adj.distortion);
            let passive_sat = adj.distortion.ego_wakeup_recommended
                || matches!(
                    adj.distortion.kind,
                    liminika_core::runtime::LayerDistortionKind::PassiveSaturation
                );
            if passive_sat {
                self.passive_saturation_probe_steps = self.passive_saturation_probe_steps.max(2);
            }
            self.distortion_tension.step_slow_loop_excitation(
                passive_sat,
                self.last_boundary_delta.chatter_potential_u8,
                self.configured_ego.is_active(),
            );
            self.system_spine.slow_loop_epoch = self.system_spine.slow_loop_epoch.saturating_add(1);
            let _ = self
                .ego_differential_net
                .consolidate_and_evaporate_cold_path(self.system_spine.slow_loop_epoch);
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
        if report.ego_wakeup_recommended {
            self.core_layers.ego_layer.trigger_mask |=
                EGO_TRIGGER_LAYER_DISTORTION | EGO_TRIGGER_BASIN_CAPTURE;
            self.core_layers.ego_layer.counter_basin_thrust_q7 =
                self.core_layers.ego_layer.counter_basin_thrust_q7.max(48);
            if self.core_layers.ego_layer.transgression_bias_q7.abs() < 24 {
                self.core_layers.ego_layer.transgression_bias_q7 = 24;
            }
        } else if report.ego_cooldown_recommended {
            self.core_layers.ego_layer.trigger_mask &=
                !(EGO_TRIGGER_LAYER_DISTORTION | EGO_TRIGGER_INTERNAL_CONFLICT);
            self.core_layers.ego_layer.counter_basin_thrust_q7 =
                self.configured_ego.counter_basin_thrust_q7 >> 1;
            self.core_layers.ego_layer.irrational_defiance_q7 =
                self.configured_ego.irrational_defiance_q7 >> 1;
        } else {
            let epochs = self.core_layers.ego_layer.intervention_epochs;
            let last_mask = self.core_layers.ego_layer.last_intervened_mask;
            self.core_layers.ego_layer = self.configured_ego;
            self.core_layers.ego_layer.intervention_epochs = epochs;
            self.core_layers.ego_layer.last_intervened_mask = last_mask;
        }
    }

    /// Returns a shared reference to the embedded [`StigmergicSubstrate`].
    #[must_use]
    pub fn substrate(&self) -> &StigmergicSubstrate {
        &self.substrate
    }

    /// Returns a mutable reference to the embedded [`StigmergicSubstrate`].
    pub fn substrate_mut(&mut self) -> &mut StigmergicSubstrate {
        &mut self.substrate
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
    /// [`StigmergicSubstrate`] and the rank-1 residual step modulates [`GatewayFrame`] dynamically from this circuit.
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

    /// Returns a shared reference to the live Core Layer 0 [`LiminalThresholdState`] governing
    /// dynamic `Liminal` channel thresholds and cognitive window gains.
    #[must_use]
    pub const fn threshold_state(&self) -> &LiminalThresholdState {
        &self.core_layers.liminal_layer
    }

    /// Returns a mutable reference to the live Core Layer 0 [`LiminalThresholdState`].
    pub fn threshold_state_mut(&mut self) -> &mut LiminalThresholdState {
        &mut self.core_layers.liminal_layer
    }

    /// Returns a shared reference to the live Core Layer 1 [`StigmergicGranuleState`].
    #[must_use]
    pub const fn stigmergic_granule_state(&self) -> &StigmergicGranuleState {
        &self.core_layers.stigmergic_layer
    }

    /// Returns a shared reference to the unified 3-layer [`AutonomousCoreLayerStack`].
    #[must_use]
    pub const fn core_layer_stack(&self) -> &AutonomousCoreLayerStack {
        &self.core_layers
    }

    /// Returns a mutable reference to the unified 3-layer [`AutonomousCoreLayerStack`].
    pub fn core_layer_stack_mut(&mut self) -> &mut AutonomousCoreLayerStack {
        &mut self.core_layers
    }

    /// Returns the most recently observed [`EpistemicHorizonReading`].
    #[must_use]
    pub const fn last_horizon(&self) -> &EpistemicHorizonReading {
        &self.last_horizon
    }

    /// Returns the most recently observed [`StigmergicOverlapSerendipityReport`].
    #[must_use]
    pub const fn last_venn_serendipity_report(&self) -> &StigmergicOverlapSerendipityReport {
        &self.last_venn_report
    }

    /// Returns a shared reference to the live 64-byte Core Layer 2 [`LiminalEgoState`].
    #[must_use]
    pub const fn ego_state(&self) -> &LiminalEgoState {
        &self.core_layers.ego_layer
    }

    /// Returns a mutable reference to the live 64-byte Core Layer 2 [`LiminalEgoState`].
    pub fn ego_state_mut(&mut self) -> &mut LiminalEgoState {
        &mut self.core_layers.ego_layer
    }

    /// Replaces the adapter's active Core Layer 2 [`LiminalEgoState`].
    pub fn set_ego_state(&mut self, ego: LiminalEgoState) {
        self.core_layers.ego_layer = ego;
        self.configured_ego = ego;
        self.sync_ego_differential_net_from_configured();
    }

    /// Configures Core Layer 2 [`LiminalEgoState`] from a coarse [`LiminalEgoPreset`] (`Custom`, `Low`, `Medium`, `High`).
    pub fn configure_ego_from_preset(&mut self, preset: LiminalEgoPreset) {
        let ego = LiminalEgoState::from_preset(preset);
        self.core_layers.ego_layer = ego;
        self.configured_ego = ego;
        self.sync_ego_differential_net_from_configured();
    }

    /// Configures Core Layer 2 [`LiminalEgoState`] from intermediate normalized [`EgoIntentAxes`].
    pub fn configure_ego_from_intent_axes(&mut self, axes: &EgoIntentAxes) {
        let ego = LiminalEgoState::from_intent_axes(axes);
        self.core_layers.ego_layer = ego;
        self.configured_ego = ego;
        self.sync_ego_differential_net_from_configured();
    }

    /// Smoothly blends an [`EgoIntentAxes`] overlay onto the current Ego state (`weight_q7` in `0..=127`).
    pub fn blend_ego_intent_axes(&mut self, overlay: &EgoIntentAxes, weight_q7: u8) {
        let base_axes = self.core_layers.ego_layer.extract_intent_axes();
        let blended = base_axes.blend_lerp_q7(overlay, weight_q7);
        let ego = LiminalEgoState::from_intent_axes(&blended);
        self.core_layers.ego_layer = ego;
        self.configured_ego = ego;
        self.sync_ego_differential_net_from_configured();
    }

    /// Derives Core Layer 2 [`LiminalEgoState`] deterministically from a natural-language Ego description.
    pub fn configure_ego_from_natural_language(&mut self, description: &str) {
        let ego = crate::ego_intent::ego_state_from_natural_language(description);
        self.core_layers.ego_layer = ego;
        self.configured_ego = ego;
        self.sync_ego_differential_net_from_configured();
    }

    /// Overlays natural-language Ego cues onto the existing Core Layer 2 [`LiminalEgoState`].
    pub fn overlay_ego_from_natural_language(&mut self, description: &str) {
        let ego = crate::ego_intent::overlay_ego_state_from_natural_language(
            self.core_layers.ego_layer,
            description,
        );
        self.core_layers.ego_layer = ego;
        self.configured_ego = ego;
        self.sync_ego_differential_net_from_configured();
    }

    /// Returns the most recent temporal [`LiminalBoundaryDelta`] between aperture epochs.
    #[must_use]
    pub const fn last_boundary_delta(&self) -> &LiminalBoundaryDelta {
        &self.last_boundary_delta
    }

    /// Returns the most recent [`StigmergicFragmentEnsembleReport`] computed during recall sync.
    #[must_use]
    pub const fn last_ensemble_report(&self) -> Option<&StigmergicFragmentEnsembleReport> {
        self.last_ensemble_report.as_ref()
    }

    /// Returns the bitmask of external [`LatentHorizonSlot`] boundaries currently slid into the
    /// adapter's active cognitive window.
    #[must_use]
    pub const fn active_horizon_slots_mask(&self) -> u32 {
        self.active_horizon_slots_mask
    }

    /// Returns the live consecutive destabilization / quiescence dwell tracker [`HorizonDwellState`].
    #[must_use]
    pub const fn horizon_dwell(&self) -> HorizonDwellState {
        self.horizon_dwell
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
        residual_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        coherence: f32,
        resolved: bool,
    ) -> (bool, u32) {
        let res = self.residual_spillover.commit_verified_experience(
            0,
            residual_phases,
            coherence,
            resolved,
        );
        self.telemetry.sync_experience_metrics(
            self.residual_spillover.cache_hits,
            self.residual_spillover.cache_stores,
            self.residual_spillover.cache_evictions,
        );
        res
    }

    /// Updates Core Layer 0 (`Liminal`), Layer 1 (`Stigmergic`), and Layer 2 (`Ego`)
    /// branchlessly, leveraging `LiminalBoundaryDelta::stationary_shortcut_ready` to skip
    /// redundant adaptation churn when boundaries are quiescent without stagnation triggers.
    pub fn adapt_core_layers(
        &mut self,
        horizon: &EpistemicHorizonReading,
        venn_report: &StigmergicOverlapSerendipityReport,
        overflow_opt: Option<&DissipativeOverflowReport>,
    ) {
        let stagnation_drive =
            (self.core_layers.ego_layer.trigger_mask & EGO_TRIGGER_BOUNDARY_STAGNATION) != 0;
        let can_shortcut = self.core_layers.liminal_layer.adaptation_epochs > 0
            && self.last_boundary_delta.stationary_shortcut_ready == 1
            && self.last_horizon.regime_code == horizon.regime_code
            && venn_report.serendipity_triggered == 0
            && self.last_venn_report.venn_core_mask == venn_report.venn_core_mask
            && overflow_opt.is_none()
            && !stagnation_drive;
        self.last_horizon = *horizon;
        self.last_venn_report = *venn_report;
        if can_shortcut {
            return;
        }

        let prev_layer0 = self.core_layers.liminal_layer;
        let prev_venn = self.core_layers.stigmergic_layer.venn_overlap_mask;
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
        if self.core_layers.ego_layer.is_active() {
            let preview = GatewayFrame {
                effective_energy: horizon.total_memory_support,
                ..GatewayFrame::default()
            };
            let _ = self
                .core_layers
                .intervene_ego_and_modulate_frame_branchless(
                    &self.last_boundary_delta,
                    horizon,
                    venn_report,
                    preview,
                );
        }
    }

    /// Synchronizes all three core layers (`Liminal`, `Stigmergic`, `Ego`), temporal boundary
    /// delta (`LiminalBoundaryDelta`), and fragmented micro-ensemble consensus directly from a
    /// live [`DynamicAttractorRecall`] outcome.
    pub fn sync_core_layers_from_recall(
        &mut self,
        recall: &DynamicAttractorRecall<'_>,
        query_words_opt: Option<&[u64; NUM_STIGMERGIC_GRANULES]>,
    ) -> (
        EpistemicHorizonReading,
        StigmergicOverlapSerendipityReport,
        Option<StigmergicFragmentEnsembleReport>,
    ) {
        let has_guard_conflict = recall.guard_conflict_margin > 0.0;
        let guard_turn = has_guard_conflict
            || matches!(
                recall.action_decision,
                liminika_core::stigmergy::StigmergicActionDecision::RepulsionEscape
            );
        self.turn_guard_active = guard_turn;
        if guard_turn {
            self.system_spine
                .pin_compute_slots(self.compute_pool.active_mask);
        } else {
            self.system_spine.clear_pinned_compute_slots();
        }
        self.last_venn_partition = recall.venn_partition;
        self.system_spine.ood_void_locked = u8::from(
            matches!(recall.void_state, VoidBoundaryState::LiminalExterior)
                && !recall.is_known_unknown_exterior
                && recall.guard_conflict_margin <= 0.05,
        );
        if !has_guard_conflict && self.configured_ego.is_active() {
            self.distortion_tension.effective_ego_scale_q7 =
                self.distortion_tension.effective_ego_scale_q7.max(48);
            self.distortion_tension.suppression_cooldown_steps = 0;
        }
        let regime_code = if has_guard_conflict
            || matches!(recall.void_state, VoidBoundaryState::LiminalBoundary)
        {
            EpistemicRegime::InternalConflict as u8
        } else if matches!(recall.void_state, VoidBoundaryState::LiminalExterior) {
            EpistemicRegime::Liminal as u8
        } else {
            EpistemicRegime::KnownCoherent as u8
        };
        let is_liminal = regime_code == EpistemicRegime::Liminal as u8;
        let horizon = EpistemicHorizonReading {
            total_memory_support: (recall.top1_fact_score.max(0.0) * 2.0).max(if is_liminal {
                0.0
            } else {
                0.35
            }),
            peak_positive_affinity: recall.top1_fact_score.max(0.0),
            internal_conflict_energy: if has_guard_conflict {
                recall.guard_conflict_margin.max(0.0)
            } else {
                (recall.top2_fact_score.max(0.0) * 0.5).max(0.0)
            },
            horizon_gate: if is_liminal { 0.0 } else { 1.0 },
            conflict_severity: if has_guard_conflict {
                recall.guard_conflict_margin.clamp(0.15, 1.0)
            } else if regime_code == EpistemicRegime::InternalConflict as u8 {
                0.25
            } else {
                0.0
            },
            dominant_anchor_id: recall
                .recalled_facts
                .first()
                .map_or(0, |(e, _)| e.anchor_id),
            opposing_anchor_id: recall.recalled_facts.get(1).map_or(0, |(e, _)| e.anchor_id),
            regime_code,
            reserved_align: [0; 3],
        };

        let q_words = query_words_opt
            .copied()
            .unwrap_or(*recall.gateway_frame.phase_words_ref());
        let mut drifted_words = q_words;
        for (g, w) in drifted_words.iter_mut().enumerate() {
            let tol = self.core_layers.stigmergic_layer.granule_noise_tolerance_q7[g];
            let nudge = u64::from(tol.min(16)).wrapping_mul(SWAR_SPLAT_BYTE_64);
            *w = swar_q7_circular_add_x8(*w, nudge);
        }

        let mut fact_fragments: Vec<ObjectiveFact<[i8; MAX_ATTENTION_HEADS]>> =
            Vec::with_capacity(recall.recalled_facts.len());
        for &(entry, score) in &recall.recalled_facts {
            let q7 = entry
                .per_layer_q7
                .last()
                .copied()
                .unwrap_or([0_i8; MAX_ATTENTION_HEADS]);
            let scaled_e = (entry.energy * score.clamp(0.25, 1.25)).max(0.25);
            fact_fragments.push(ObjectiveFact::new(q7, scaled_e, 0, entry.anchor_id));
        }

        let aperture_floor =
            f32::from(self.core_layers.liminal_layer.threshold_floor_q7[1]) * (1.0 / 127.0);
        let venn = evaluate_stigmergic_overlap_and_serendipity_words_64(
            &fact_fragments,
            &q_words,
            &drifted_words,
            aperture_floor,
            self.core_layers.stigmergic_layer.serendipity_floor_q7,
        );

        self.adapt_core_layers(&horizon, &venn, None);
        let ensemble_report = if fact_fragments.len() >= 2 {
            let (_frame, _words, rep) = self.synthesize_fragment_ensemble(&fact_fragments);
            let agreed_or_tie = rep.quorum_agreed_granule_mask | rep.ego_tiebreak_granule_mask;
            for (g, ph) in self
                .core_layers
                .stigmergic_layer
                .granule_pheromone_q7
                .iter_mut()
                .enumerate()
                .take(NUM_STIGMERGIC_GRANULES)
            {
                let bit = 1u8 << g;
                if (agreed_or_tie & bit) != 0 {
                    *ph = ph.saturating_add(4).min(127);
                }
            }
            Some(rep)
        } else {
            None
        };
        self.last_ensemble_report = ensemble_report;
        self.escalation_boundary_dirty = true;
        let preset = if horizon.regime_code == EpistemicRegime::InternalConflict as u8 {
            WorkbenchPreset::GuardDefense
        } else if horizon.regime_code == EpistemicRegime::Liminal as u8 {
            WorkbenchPreset::CreativeLiminal
        } else if !recall.recalled_facts.is_empty() && recall.top1_fact_score >= 0.20 {
            WorkbenchPreset::FactualGrounding
        } else {
            WorkbenchPreset::BalancedDialog
        };

        let lane_budget = usize::from(preset.recommended_lane_budget());
        let mut specs: Vec<WorkbenchSlotSpec> = Vec::with_capacity(lane_budget);
        let repulsion_spec = has_guard_conflict.then(|| {
            let mut rep_phases = [0_i8; MAX_ATTENTION_HEADS];
            for (dst, &src) in rep_phases
                .iter_mut()
                .zip(recall.gateway_frame.phase_deltas.iter())
            {
                *dst = src.saturating_neg();
            }
            WorkbenchSlotSpec::auto_repulsion(
                recall.guard_conflict_margin.clamp(0.35, 1.0),
                rep_phases,
            )
        });
        if preset.prioritizes_repulsion() {
            if let Some(spec) = repulsion_spec {
                if specs.len() < lane_budget {
                    specs.push(spec);
                }
            }
        }
        for (fact_rank, &(entry, score)) in recall.recalled_facts.iter().take(3).enumerate() {
            if specs.len() >= lane_budget {
                break;
            }
            if let Some(&phases) = entry.per_layer_q7.last() {
                let (min_s, max_s, min_w, max_w) = match fact_rank {
                    0 => (0.25_f32, 1.0_f32, 0.15_f32, 1.0_f32),
                    1 => (0.20_f32, 0.90_f32, 0.10_f32, 0.90_f32),
                    _ => (0.15_f32, 0.80_f32, 0.10_f32, 0.80_f32),
                };
                specs.push(WorkbenchSlotSpec::auto_free(
                    (entry.energy * score.clamp(min_s, max_s)).clamp(min_w, max_w),
                    phases,
                ));
            }
        }
        if specs.len() < lane_budget {
            if let Some(&(entry, score)) = recall.recalled_personas.first() {
                if let Some(&phases) = entry.per_layer_q7.last() {
                    specs.push(WorkbenchSlotSpec::auto_free(
                        (entry.energy * score.clamp(0.20, 1.0)).clamp(0.15, 1.0),
                        phases,
                    ));
                }
            }
        }
        if !preset.prioritizes_repulsion() {
            if let Some(spec) = repulsion_spec {
                if specs.len() < lane_budget {
                    specs.push(spec);
                }
            }
        }
        let prev_was_sandbox = self.sandbox_workbench_lane.is_some();
        let _ = self.substrate.apply_preset_lanes(&specs);
        self.sandbox_workbench_lane = None;
        let assim_drift = self.telemetry.assimilation_drift();
        if prev_was_sandbox || recall.is_known_unknown_exterior {
            let improved = prev_was_sandbox
                && recall.top1_fact_score > self.prev_sandbox_coherence
                && assim_drift <= self.prev_sandbox_assim_drift + 0.02;
            self.sandbox_promotion_streak = if improved {
                self.sandbox_promotion_streak.saturating_add(1)
            } else {
                0
            };
            self.prev_sandbox_coherence = recall.top1_fact_score;
            self.prev_sandbox_assim_drift = assim_drift;
        }
        if !has_guard_conflict && recall.is_known_unknown_exterior {
            if let Some(&(ext_entry, ext_score)) = recall.exterior_sandbox_facts.first() {
                if let Some(&ext_phases) = ext_entry.per_layer_q7.last() {
                    let ext_weight =
                        (ext_entry.energy * ext_score.clamp(0.20, 0.90)).clamp(0.15, 0.95);
                    if let Ok(lane) = self
                        .substrate
                        .allocate_lane(&WorkbenchSlotSpec::auto_sandbox(ext_weight, ext_phases))
                    {
                        self.sandbox_workbench_lane = Some(lane);
                        if self.sandbox_promotion_streak >= 2
                            && self.substrate.promote_sandbox_lane(lane).unwrap_or(false)
                        {
                            self.sandbox_promotions = self.sandbox_promotions.saturating_add(1);
                            self.sandbox_workbench_lane = None;
                            self.sandbox_promotion_streak = 0;
                        }
                    }
                }
            }
        }
        if let Some(esc_lane) = self.escalated_workbench_lane {
            if (self.substrate.slot_enabled_mask & (1u8 << esc_lane)) == 0 {
                self.escalated_workbench_lane = None;
            }
        }
        if self.core_layers.ego_layer.is_active() {
            let (_p_frame, d_norm, hops, matched_mask) = self
                .ego_differential_net
                .evaluate_pframe_chain_with_mask_branchless(
                    &self.core_layers.ego_layer,
                    &self.system_spine,
                    &self.last_venn_partition,
                    &self.core_layers.stigmergic_layer,
                    &self.distortion_tension,
                    recall.gateway_frame,
                );
            self.ego_differential_net.last_delta_norm_q7 = d_norm;
            self.ego_differential_net.last_hops_traversed = hops;
            if hops > 0 {
                self.ego_differential_net
                    .record_matched_activations(matched_mask);
            }
        }
        self.active_workbench_preset = preset;
        (horizon, venn, ensemble_report)
    }

    /// Returns the anonymous workbench lane (`0..=7`) currently leased by cold-path escalation, if any.
    #[must_use]
    pub const fn escalated_workbench_lane(&self) -> Option<u8> {
        self.escalated_workbench_lane
    }

    /// Returns the currently active [`WorkbenchPreset`] selected for the 8-lane working bank.
    #[must_use]
    pub const fn active_workbench_preset(&self) -> WorkbenchPreset {
        self.active_workbench_preset
    }

    /// Switches the 8-lane working phase workbench to `preset` and atomically loads `specs`
    /// into [`StigmergicSubstrate`] (preserving pinned lanes), returning the number of lanes updated.
    ///
    /// # Errors
    ///
    /// Returns an error if any `spec` has invalid weight, out-of-range explicit lane (`>= 8`), or
    /// if an `AutoFree` allocation fails because all 8 lanes are pinned.
    pub fn switch_workbench_preset(
        &mut self,
        preset: WorkbenchPreset,
        specs: &[WorkbenchSlotSpec],
    ) -> Result<u8, String> {
        let updated = self
            .substrate
            .apply_preset_lanes(specs)
            .map_err(|e| format!("apply_preset_lanes error: {e}"))?;
        if let Some(esc_lane) = self.escalated_workbench_lane {
            if (self.substrate.slot_enabled_mask & (1u8 << esc_lane)) == 0 {
                self.escalated_workbench_lane = None;
            }
        }
        self.active_workbench_preset = preset;
        Ok(updated)
    }

    /// Captures the live [`LiminalTopologyTelemetrySnapshot`] and renders a self-introspecting
    /// 3-layer Mermaid `flowchart TD` diagram covering Core Layer 0 (`Liminal`), Core Layer 1
    /// (`StigmergicGranuleState`), Core Layer 2 (`LiminalEgoState`), cognitive aperture masks,
    /// and crystal populations.
    #[must_use]
    pub fn render_liminal_topology_mermaid(
        &self,
        active_crystals: u8,
        spawned_daughters: u8,
    ) -> String {
        let distortion = self.telemetry_snapshot().distortion;
        self.render_liminal_topology_mermaid_with_distortion(
            active_crystals,
            spawned_daughters,
            &distortion,
        )
    }

    /// Renders the live 3-layer Mermaid `flowchart TD` diagram using an explicit
    /// [`LayerDistortionReport`] (such as a multi-probe cumulative report).
    #[must_use]
    pub fn render_liminal_topology_mermaid_with_distortion(
        &self,
        active_crystals: u8,
        spawned_daughters: u8,
        distortion: &LayerDistortionReport,
    ) -> String {
        let snap = LiminalTopologyTelemetrySnapshot::capture(
            &self.core_layers.liminal_layer,
            &self.last_horizon,
            active_crystals,
            spawned_daughters,
        );
        let mut buf = [0u8; 2048];
        let written = snap.format_three_core_layers_mermaid_with_distortion(
            &self.core_layers,
            &self.last_venn_report,
            Some(distortion),
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
        self.substrate.thresholds = self.threshold_policy.resolve_thresholds(
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
        self.substrate.thresholds = self.threshold_policy.resolve_thresholds(
            &self.system_constraint,
            &self.probe_baseline,
            &neutral_trail,
            self.probe_baseline.baseline_entropy,
        );
        Ok(self.probe_baseline)
    }

    /// Safely re-allocates and re-initializes the Cold-Path engines (`PhaseSteeringEngine`,
    /// `StigmergicSubstrate`, and `ledger_slot`) when the underlying LLM or RoPE geometry changes
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
        let mut new_substrate = Box::new(
            StigmergicSubstrate::with_dimensions_and_thresholds(
                new_geometry.rotary_dim as u32,
                new_geometry.num_q_heads as u32,
                16,
                calibrated,
            )
            .map_err(|e| {
                format!("StigmergicSubstrate::with_dimensions_and_thresholds failed: {e}")
            })?,
        );
        new_substrate.last_entropy_h_tilde = next_probe.baseline_entropy;

        self.geometry = new_geometry;
        self.phase_engine = new_phase_engine;
        self.substrate = new_substrate;
        self.ledger_slot = gcso_ledger_slot_t::default();
        self.sandbox_workbench_lane = None;
        self.telemetry.reset();
        self.probe_baseline = next_probe;
        Ok(self.probe_baseline.probe_epoch)
    }

    /// Returns a mutable reference to the embedded [`EntropyController`].
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

    /// Applies soft-bounded (`PhaseClamp`) phase steering to a single token's rotated query slice
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
        let Some(active_q7) =
            liminika_core::liminal::Scoped::from_boundary(PhaseDeltaSlice(q7_aligned))
                .cut_by_lacuna(max_rad.is_finite() && max_rad > 0.0, max_rad, 0.0)
                .into_active()
        else {
            return Ok(());
        };
        {
            let mut tensor_mut = AlignedSliceMut32::new(work_buf.as_mut_slice())
                .map_err(|e| format!("AlignedSliceMut32 error: {e}"))?;
            self.phase_engine
                .apply_phase_steering_for_geometry_safe_active(
                    &self.geometry,
                    &mut tensor_mut,
                    active_q7,
                    max_rad,
                )
                .map_err(|e| format!("apply_phase_steering_for_geometry_safe_active error: {e}"))?;
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
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn build_core_layer_ctx<'a>(
        core_layers: &'a AutonomousCoreLayerStack,
        last_boundary_delta: &'a LiminalBoundaryDelta,
        last_horizon: &'a EpistemicHorizonReading,
        last_venn_report: &'a StigmergicOverlapSerendipityReport,
        derivative_alpha_q7: u8,
        last_venn_partition: &'a EpistemicPartition,
        system_spine: &'a SystemSpineState,
        ego_differential_net: &'a EgoDifferentialNet,
        distortion_tension: &'a DistortionTension,
    ) -> Option<CoreLayerEvalContext<'a>> {
        (core_layers.liminal_layer.adaptation_epochs > 0 || core_layers.ego_layer.is_active())
            .then_some(CoreLayerEvalContext {
                stack: core_layers,
                delta: last_boundary_delta,
                horizon: last_horizon,
                venn: last_venn_report,
                derivative_alpha_q7,
                venn_partition: Some(last_venn_partition),
                spine: Some(system_spine),
                ego_net: Some(ego_differential_net),
                tension: Some(distortion_tension),
            })
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
            let core_ctx = Self::build_core_layer_ctx(
                &this.core_layers,
                &this.last_boundary_delta,
                &this.last_horizon,
                &this.last_venn_report,
                this.telemetry.derivative_alpha_q7(),
                &this.last_venn_partition,
                &this.system_spine,
                &this.ego_differential_net,
                &this.distortion_tension,
            );
            this.phase_engine
                .steer_gateway_signal_with_core_layers_for_geometry_safe(
                    signal,
                    &this.geometry,
                    core_ctx,
                    trail_inout,
                    unresolved_damping_u7,
                    q,
                    None,
                    logits,
                )
                .map_err(|e| {
                    format!("steer_gateway_signal_with_core_layers_for_geometry_safe error: {e}")
                })
        })
    }

    /// Routes [`ManifoldGatewaySignal`] through the embedded [`StigmergicSubstrate`] via
    /// [`SubstrateGatewayExt::ingest_gateway_and_steer_with_core_layers_for_geometry_safe`],
    /// allowing the substrate's stigmergic trail density, FastPathLedger fast-path state, and live
    /// `2 Primitives x 2 Drivers` stack (`Liminal`/`Stigmergy` x `System`/`Ego`) to autonomously modulate steering.
    ///
    /// # Errors
    ///
    /// Returns an error if `q_rot_inout` length mismatches or `liminika-core` rejects the inputs.
    pub fn steer_layer_with_substrate<S: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        layer_idx: usize,
        signal: &S,
        trail_inout: &mut gcso_pointer_trail_t,
        unresolved_damping_u7: u8,
        q_rot_inout: &mut [f32],
        logits_opt: Option<&mut [f32]>,
    ) -> Result<Option<(GatewayFrame, CausalProvenanceReport)>, String> {
        self.marshal_q_rot_and_logits(layer_idx, q_rot_inout, logits_opt, |this, q, logits| {
            let core_ctx = Self::build_core_layer_ctx(
                &this.core_layers,
                &this.last_boundary_delta,
                &this.last_horizon,
                &this.last_venn_report,
                this.telemetry.derivative_alpha_q7(),
                &this.last_venn_partition,
                &this.system_spine,
                &this.ego_differential_net,
                &this.distortion_tension,
            );
            this.substrate
                .ingest_gateway_and_steer_with_core_layers_for_geometry_safe(
                    signal,
                    &this.geometry,
                    core_ctx,
                    &mut this.ledger_slot,
                    trail_inout,
                    unresolved_damping_u7,
                    Some(q),
                    logits,
                )
                .map_err(|e| {
                    format!(
                        "ingest_gateway_and_steer_with_core_layers_for_geometry_safe error: {e}"
                    )
                })
        })
    }

    /// Orchestrates a layer's RoPE steering via [`StigmergicSubstrate`] using stigmergic
    /// evaporation + resonance-weighted reinforcement, the live 3-layer core stack, and the
    /// built-in dormant-until-destabilized [`CoherenceHorizonGuardSlot`].
    ///
    /// # Errors
    ///
    /// Returns an error if `q_rot_inout` length mismatches or `liminika-core` rejects the inputs.
    #[allow(clippy::too_many_arguments)]
    pub fn orchestrate_layer_rope_with_substrate<S: ManifoldGatewaySignal + ?Sized>(
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
        let mut guard_slot = self.coherence_guard_slot;
        let mut no_escalator = ();
        let res = self.orchestrate_layer_rope_with_horizon_slots(
            layer_idx,
            signal,
            &mut guard_slot,
            &mut no_escalator,
            trail_inout,
            entropy_h_tilde_opt,
            unresolved_damping_u7,
            q_rot_inout,
            logits_opt,
        );
        self.coherence_guard_slot = guard_slot;
        res
    }

    /// Extended layer-wise orchestration pass pairing [`StigmergicSubstrate`] + 3-layer core stack
    /// piercing with caller-supplied [`LatentHorizonSlot`] boundaries (dynamic slide-in/out with
    /// dwell hysteresis) and an optional cold-path [`HorizonEscalationProvider`].
    ///
    /// # Errors
    ///
    /// Returns an error if `q_rot_inout` length mismatches, `escalation_provider` fails, or
    /// `liminika-core` rejects the inputs.
    #[allow(clippy::too_many_arguments)]
    pub fn orchestrate_layer_rope_with_horizon_slots<
        S: ManifoldGatewaySignal + ?Sized,
        H: LatentHorizonSlot + ?Sized,
        E: HorizonEscalationProvider + ?Sized,
    >(
        &mut self,
        layer_idx: usize,
        signal: &S,
        horizon_slots: &mut H,
        escalation_provider: &mut E,
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
                entropy_h_tilde_opt.unwrap_or(this.substrate.last_entropy_h_tilde);
            let scale = StigmergicScaleGranularity::from_layer_ratio(
                layer_idx,
                this.geometry.num_layers.max(1),
            );
            let pre_trigger = HorizonTriggerInput {
                entropy_gate_mode: this.entropy_controller.state().active_branch_mode,
                moving_z_entropy: entropy_h_tilde,
                repulsion_severity: this.last_horizon.conflict_severity,
                causal_root_cause: ROOT_CAUSE_NOMINAL,
                epistemic_regime: this.last_horizon.regime_code,
                stationary_shortcut_ready: this.last_boundary_delta.stationary_shortcut_ready != 0,
                ego_active_triggers: this.core_layers.ego_layer.last_intervened_mask,
                unstable_dwell_steps: this.horizon_dwell.unstable_dwell_steps,
                quiescent_dwell_steps: this.horizon_dwell.quiescent_dwell_steps,
            };
            let num_layers_denom = this.geometry.num_layers.saturating_sub(1).max(1);
            let layer_scale_q7 = ((layer_idx * 127) / num_layers_denom).min(127) as u8;
            let conflict_active = pre_trigger.repulsion_severity > 0.25;
            let esc_req = HorizonEscalationRequest {
                trigger: pre_trigger,
                placement: WorkbenchLanePlacement::AutoFree,
                lane_flags: if conflict_active {
                    WORKBENCH_LANE_FLAG_REPULSION
                } else {
                    0
                },
                layer_scale_q7,
                desired_aperture_mask: this.core_layers.liminal_layer.active_aperture_mask,
                current_frame_energy: this.last_horizon.total_memory_support,
            };
            let lane_active = this
                .escalated_workbench_lane
                .is_some_and(|l| l < 8 && (this.substrate.slot_enabled_mask & (1u8 << l)) != 0);
            let should_check_escalation = this.escalation_boundary_dirty
                || pre_trigger.epistemic_regime != this.last_escalated_regime
                || (lane_active && pre_trigger.can_retract())
                || (!lane_active && pre_trigger.is_destabilized());
            let escalated = if should_check_escalation {
                let sw = this
                    .substrate
                    .apply_horizon_escalation(
                        &esc_req,
                        &mut this.escalated_workbench_lane,
                        escalation_provider,
                    )
                    .map_err(|e| format!("apply_horizon_escalation error: {e}"))?;
                this.escalation_boundary_dirty = false;
                this.last_escalated_regime = pre_trigger.epistemic_regime;
                sw
            } else {
                false
            };

            let core_ctx = Self::build_core_layer_ctx(
                &this.core_layers,
                &this.last_boundary_delta,
                &this.last_horizon,
                &this.last_venn_report,
                this.telemetry.derivative_alpha_q7(),
                &this.last_venn_partition,
                &this.system_spine,
                &this.ego_differential_net,
                &this.distortion_tension,
            );
            let (mut frame, report, decision) = this
                .substrate
                .orchestrate_gateway_layer_with_core_layers_safe(
                    signal,
                    &this.geometry,
                    core_ctx,
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
                    format!("orchestrate_gateway_layer_with_core_layers_safe error: {e}")
                })?;

            let mut trigger = HorizonTriggerInput {
                entropy_gate_mode: this.entropy_controller.state().active_branch_mode,
                moving_z_entropy: entropy_h_tilde,
                repulsion_severity: frame.repulsion_severity,
                causal_root_cause: frame.dominant_cause_code,
                epistemic_regime: this.last_horizon.regime_code,
                stationary_shortcut_ready: this.last_boundary_delta.stationary_shortcut_ready != 0,
                ego_active_triggers: this.core_layers.ego_layer.last_intervened_mask,
                unstable_dwell_steps: 0,
                quiescent_dwell_steps: 0,
            };
            let (u_dwell, q_dwell) = this
                .horizon_dwell
                .step_branchless(trigger.is_destabilized(), trigger.stationary_shortcut_ready);
            trigger.unstable_dwell_steps = u_dwell;
            trigger.quiescent_dwell_steps = q_dwell;
            let (slid_in, slid_out) = apply_latent_horizon_slots(
                horizon_slots,
                &mut this.active_horizon_slots_mask,
                &trigger,
                &mut frame,
                &mut this.core_layers,
            );

            let qos_throttled = frame.dominant_cause_code == ROOT_CAUSE_SYSTEM_QOS_THROTTLE;
            let _ =
                this.telemetry
                    .evaluate_policy(trigger.is_destabilized(), q_dwell, qos_throttled);
            this.telemetry.record_step(
                decision,
                &this.last_horizon,
                &this.last_boundary_delta,
                &this.last_venn_report,
                &this.core_layers.ego_layer,
                trigger.entropy_gate_mode,
                slid_in,
                slid_out,
                escalated,
                qos_throttled,
            );
            let _ = this.telemetry.update_assimilation_ema(
                entropy_h_tilde,
                frame.fact_persona_coherence,
                frame.repulsion_severity,
                &this.probe_baseline,
                0.15_f32,
            );
            let exterior_granule_mask = BitCut8::split(
                this.last_venn_report.void_exterior_mask
                    | this.last_venn_report.serendipity_excursion_mask,
                this.last_venn_report.venn_core_mask,
            )
            .outside;
            let sb_event = this.substrate.step_sandbox_hypothesis_lifecycle(
                &mut this.telemetry,
                &mut this.sandbox_workbench_lane,
                decision,
                this.configured_ego.is_active(),
                exterior_granule_mask,
                liminika_core::stigmergy::trail_phase_words_ref(trail_inout),
                frame.effective_energy,
                3,
                3,
            );
            match sb_event {
                SandboxLifecycleEvent::Promoted(_) => {
                    this.sandbox_promotions = this.sandbox_promotions.saturating_add(1);
                    this.sandbox_promotion_streak = 0;
                }
                SandboxLifecycleEvent::Released(_) => {
                    this.sandbox_promotion_streak = 0;
                }
                SandboxLifecycleEvent::Idle | SandboxLifecycleEvent::Nucleated(_) => {}
            }
            this.substrate.last_decision = decision as u8;
            Ok((frame, report, decision))
        })
    }

    /// Orchestrates a layer's RoPE steering while pairing [`CoherenceHorizonGuardSlot`] and the
    /// auxiliary [`DefaultResidualSpilloverSlot`] experience cache (with an optional external
    /// [`HorizonEscalationProvider`]), syncing hits/stores/evictions to [`CoreTelemetryBank`].
    ///
    /// # Errors
    ///
    /// Returns an error if `q_rot_inout` length mismatches, `escalation_provider` fails, or
    /// `liminika-core` rejects the inputs.
    #[allow(clippy::too_many_arguments)]
    pub fn orchestrate_layer_rope_with_spillover<
        S: ManifoldGatewaySignal + ?Sized,
        E: HorizonEscalationProvider + ?Sized,
    >(
        &mut self,
        layer_idx: usize,
        signal: &S,
        escalation_provider: &mut E,
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
        let mut guard_slot = self.coherence_guard_slot;
        let mut spillover_box = std::mem::take(&mut self.residual_spillover);
        let res = {
            let mut combined = (&mut guard_slot, &mut *spillover_box);
            self.orchestrate_layer_rope_with_horizon_slots(
                layer_idx,
                signal,
                &mut combined,
                escalation_provider,
                trail_inout,
                entropy_h_tilde_opt,
                unresolved_damping_u7,
                q_rot_inout,
                logits_opt,
            )
        };
        self.coherence_guard_slot = guard_slot;
        self.telemetry.sync_experience_metrics(
            spillover_box.cache_hits,
            spillover_box.cache_stores,
            spillover_box.cache_evictions,
        );
        self.residual_spillover = spillover_box;
        res
    }

    /// Executes norm-adaptive Rank-1 residual injection only when [`StigmergicSubstrate`]'s latest
    /// [`StigmergicActionDecision`] authorizes full orchestration (`FullOrchestration` or
    /// `CrystallizeMacroBasin`), persisting Core Layer 0/1/2 Ego interventions onto `self.core_layers`.
    /// Returns `Ok(None)` when bypassed by the substrate.
    ///
    /// # Errors
    ///
    /// Returns an error if slice lengths mismatch or `liminika-core` rejects the inputs.
    pub fn orchestrate_layer_srl_with_substrate<S: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        signal: &S,
        hidden_inout: &mut [f32],
        u_vec: &[f32],
        v_vec: &[f32],
    ) -> Result<Option<(GatewayFrame, f32)>, String> {
        let decision = StigmergicActionDecision::from_u8(self.substrate.last_decision);
        if matches!(
            decision,
            StigmergicActionDecision::FullOrchestration
                | StigmergicActionDecision::CrystallizeMacroBasin
        ) {
            let res = self.apply_srl_with_gateway_mut(signal, hidden_inout, u_vec, v_vec)?;
            Ok(Some(res))
        } else {
            Ok(None)
        }
    }

    /// Evaluates a slice of fragmented signals (`fragments`) as a subspace micro-ensemble over
    /// the live 3-layer [`AutonomousCoreLayerStack`], updating Ego intervention state in place.
    pub fn synthesize_fragment_ensemble<S: ManifoldGatewaySignal>(
        &mut self,
        fragments: &[S],
    ) -> (
        GatewayFrame,
        [u64; NUM_STIGMERGIC_GRANULES],
        StigmergicFragmentEnsembleReport,
    ) {
        S::synthesize_fragment_ensemble_with_core_layers(
            fragments,
            &mut self.core_layers,
            &self.last_boundary_delta,
            &self.last_horizon,
            &self.last_venn_report,
        )
    }

    fn modulate_srl_circuit_frame(
        circuit: Option<CognitiveTrajectoryLedger>,
        mut frame: GatewayFrame,
        frm_w: &[u64; NUM_STIGMERGIC_GRANULES],
    ) -> GatewayFrame {
        if let Some(c) = circuit {
            let _ = c.modulate_gateway_frame_at_scale(
                &mut frame,
                frm_w,
                frm_w,
                StigmergicScaleGranularity::MacroCoalition,
            );
        }
        frame
    }

    fn apply_srl_frame_internal(
        &self,
        frame: &GatewayFrame,
        hidden_inout: &mut [f32],
        u_vec: &[f32],
        v_vec: &[f32],
    ) -> Result<f32, String> {
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
                    frame, &mut y_mut, &x_aligned, &u_aligned, &v_aligned,
                )
                .map_err(|e| format!("apply_gateway_frame_rank1_hidden_fused error: {e}"))?
        };

        hidden_inout.copy_from_slice(y_buf.as_slice());
        Ok(gain)
    }

    /// Applies norm-adaptive Rank-1 residual injection driven purely by a [`ManifoldGatewaySignal`]
    /// via [`ResidualAdapterEngine::apply_gateway_frame_rank1_hidden_fused`] without manual
    /// scale tuning (shared `&self` snapshot path for read-only forward passes).
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
        let frame = if self.core_layers.liminal_layer.adaptation_epochs > 0
            || self.core_layers.ego_layer.is_active()
        {
            let raw_sys = signal.synthesize_gateway_with_aperture_from_stack(&self.core_layers);
            let sys_baseline = self.core_layers.modulate_gateway_frame_branchless(
                raw_sys,
                &self.last_horizon,
                &self.last_venn_report,
            );
            let ego_base = self.core_layers.modulate_gateway_frame_with_ego_readonly(
                &self.last_boundary_delta,
                &self.last_horizon,
                &self.last_venn_report,
                raw_sys,
            );
            let (p_frame, _, hops) = self.ego_differential_net.evaluate_pframe_chain_branchless(
                &self.core_layers.ego_layer,
                &self.system_spine,
                &self.last_venn_partition,
                &self.core_layers.stigmergic_layer,
                &self.distortion_tension,
                ego_base,
            );
            let cand = if hops > 0 { p_frame } else { ego_base };
            let cand_w = *cand.phase_words_ref();
            let modulated = Self::modulate_srl_circuit_frame(self.active_circuit(), cand, &cand_w);
            self.system_spine
                .enforce_pinned_isolation_branchless(&sys_baseline, modulated)
        } else {
            let (raw_frame, frm_w) = signal.synthesize_gateway_frame_with_words();
            Self::modulate_srl_circuit_frame(self.active_circuit(), raw_frame, &frm_w)
        };
        let gain = self.apply_srl_frame_internal(&frame, hidden_inout, u_vec, v_vec)?;
        Ok((frame, gain))
    }

    /// Mutable counterpart to [`Self::apply_srl_with_gateway`] that evaluates dynamic cognitive
    /// apertures and persists Core Layer 0/1/2 Ego interventions onto `self.core_layers`.
    ///
    /// # Errors
    ///
    /// Returns an error if slice lengths mismatch or `liminika-core` rejects the inputs.
    pub fn apply_srl_with_gateway_mut<S: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        signal: &S,
        hidden_inout: &mut [f32],
        u_vec: &[f32],
        v_vec: &[f32],
    ) -> Result<(GatewayFrame, f32), String> {
        let circuit = self.active_circuit();
        let frame = if self.core_layers.liminal_layer.adaptation_epochs > 0
            || self.core_layers.ego_layer.is_active()
        {
            let raw_sys = signal.synthesize_gateway_with_aperture_from_stack(&self.core_layers);
            let sys_baseline = self.core_layers.modulate_gateway_frame_branchless(
                raw_sys,
                &self.last_horizon,
                &self.last_venn_report,
            );
            let (ego_frame, _ego_w) = signal.synthesize_gateway_with_core_layers(
                &mut self.core_layers,
                &self.last_boundary_delta,
                &self.last_horizon,
                &self.last_venn_report,
            );
            let (p_frame, d_norm, hops, matched_mask) = self
                .ego_differential_net
                .evaluate_pframe_chain_with_mask_branchless(
                    &self.core_layers.ego_layer,
                    &self.system_spine,
                    &self.last_venn_partition,
                    &self.core_layers.stigmergic_layer,
                    &self.distortion_tension,
                    ego_frame,
                );
            self.ego_differential_net.last_delta_norm_q7 = d_norm;
            let cand = if hops > 0 {
                self.ego_differential_net
                    .record_matched_activations(matched_mask);
                p_frame
            } else {
                ego_frame
            };
            let cand_w = *cand.phase_words_ref();
            let modulated = Self::modulate_srl_circuit_frame(circuit, cand, &cand_w);
            self.system_spine
                .enforce_pinned_isolation_branchless(&sys_baseline, modulated)
        } else {
            let (raw_frame, frm_w) = signal.synthesize_gateway_frame_with_words();
            Self::modulate_srl_circuit_frame(circuit, raw_frame, &frm_w)
        };
        let gain = self.apply_srl_frame_internal(&frame, hidden_inout, u_vec, v_vec)?;
        Ok((frame, gain))
    }

    /// Evaluates Top-2 sparse surrogate entropy via [`EntropyController::eval_surrogate_and_step`]
    /// and updates `substrate.last_entropy_h_tilde` without rescaling `logits`.
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
        self.substrate.last_entropy_h_tilde = state_out.moving_z_entropy;
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

    /// Applies Rank-1 residual injection (`y <- y + s * u * (v^T * y)`) to a hidden-state
    /// vector via [`ResidualAdapterEngine::eval_rank1_slices_fused_active`].
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
        let active_scale =
            match liminika_core::liminal::Scoped::from_boundary(ResidualScale(scale_factor))
                .cut_by_lacuna(scale_factor != 0.0, scale_factor, 0.0)
            {
                liminika_core::liminal::ScopedCut::Bottom(_) => {
                    return Err(format!(
                        "eval_rank1_slices_fused_active error: {}",
                        liminika_core::liminal::GCSO_ERROR_INVALID_ARGUMENT
                    ));
                }
                cut => match cut.into_active() {
                    Some(active) => active,
                    None => return Ok(()),
                },
            };
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
            .eval_rank1_slices_fused_active(
                &mut y_mut,
                &x_aligned,
                &u_aligned,
                &v_aligned,
                None,
                active_scale,
            )
            .map_err(|e| format!("eval_rank1_slices_fused_active error: {e}"))?;

        hidden_inout.copy_from_slice(y_buf.as_slice());
        Ok(())
    }
}
