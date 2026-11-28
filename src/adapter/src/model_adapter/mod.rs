// name: src/adapter/src/model_adapter/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model geometry descriptors (`RopeGeometry`) and unified steering adapter
//! ([`ModelSteeringAdapter`]) bridging external LLM runtimes to [`liminika_core`].

use super::*;

mod layer_selection;
mod recall_sync;
mod steering;
mod telemetry;

pub use layer_selection::{
    AdaptiveLayerSelectConfig, DEFAULT_CANDIDATE_BAND_END_RATIO,
    DEFAULT_CANDIDATE_BAND_START_RATIO, DEFAULT_TARGET_LAYER_RATIO, active_layers_mask,
    clip_layer_mask, ratio_band_layer_mask, select_adaptive_layer_mask, top_half_layer_mask,
};
pub use steering::LayerRopeStepInput;
pub use telemetry::LiminalBoundaryFlowSummary;

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
    /// Constructs an explicit per-model RoPE geometry descriptor.
    #[must_use]
    pub const fn new(
        layout: RopeLayout,
        head_dim: usize,
        rotary_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        num_layers: usize,
        layer_rope_mask: u64,
    ) -> Self {
        Self {
            layout,
            head_dim,
            rotary_dim,
            num_q_heads,
            num_kv_heads,
            num_layers,
            layer_rope_mask,
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
        if self.head_dim == 0 || (self.head_dim & 1) != 0 {
            return Err(format!(
                "head_dim ({}) must be a positive even integer",
                self.head_dim
            ));
        }
        if self.rotary_dim == 0 || (self.rotary_dim & 1) != 0 || self.rotary_dim > self.head_dim {
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
    pub(super) fn new_zeroed(len: usize) -> Self {
        let num_chunks = len.div_ceil(8).max(1);
        Self {
            chunks: vec![Align32Chunk([0.0; 8]); num_chunks],
            len,
        }
    }

    pub(super) fn as_slice(&self) -> &[f32] {
        let ptr = self.chunks.as_ptr().cast::<f32>();
        // SAFETY: `Align32Chunk` is `#[repr(C, align(32))]` wrapping `[f32; 8]`, so
        // `chunks.len() * 8 >= self.len` contiguous `f32` elements are valid and initialized.
        unsafe { core::slice::from_raw_parts(ptr, self.len) }
    }

    pub(super) fn as_mut_slice(&mut self) -> &mut [f32] {
        let ptr = self.chunks.as_mut_ptr().cast::<f32>();
        // SAFETY: Same contiguous `[f32; 8]` layout as `as_slice`, with exclusive `&mut self`.
        unsafe { core::slice::from_raw_parts_mut(ptr, self.len) }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(crate) struct Align16Chunk([lmnk_q7_t; 16]);

/// Safe 16-byte aligned `lmnk_q7_t` buffer for [`AlignedSlice16`] / [`AlignedSliceMut16`].
pub(crate) struct AlignedQ7Buffer16 {
    chunks: Vec<Align16Chunk>,
    len: usize,
}

impl AlignedQ7Buffer16 {
    pub(super) fn from_slice(src: &[lmnk_q7_t]) -> Self {
        let mut buf = Self::new_zeroed(src.len());
        buf.as_mut_slice().copy_from_slice(src);
        buf
    }

    pub(super) fn new_zeroed(len: usize) -> Self {
        let num_chunks = len.div_ceil(16).max(1);
        Self {
            chunks: vec![Align16Chunk([0; 16]); num_chunks],
            len,
        }
    }

    pub(super) fn as_slice(&self) -> &[lmnk_q7_t] {
        let ptr = self.chunks.as_ptr().cast::<lmnk_q7_t>();
        // SAFETY: `Align16Chunk` is `#[repr(C, align(16))]` wrapping `[lmnk_q7_t; 16]`.
        unsafe { core::slice::from_raw_parts(ptr, self.len) }
    }

    pub(super) fn as_mut_slice(&mut self) -> &mut [lmnk_q7_t] {
        let ptr = self.chunks.as_mut_ptr().cast::<lmnk_q7_t>();
        // SAFETY: Same contiguous `[lmnk_q7_t; 16]` layout as `as_slice`, with exclusive `&mut self`.
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
/// when instability, epistemic conflict, or `Liminal` exterior excursion persists.
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
        stack: &mut CoreStateStack,
        trigger: &HorizonTriggerInput,
    ) {
        if self.activations == self.retractions {
            self.saved_aperture_mask = stack.liminal.active_aperture_mask;
            self.activations = self.activations.saturating_add(1);
        }
        stack.liminal.active_aperture_mask |= self.guard_aperture_bits;
        if trigger.repulsion_severity > 0.25 {
            frame.action_mode_flags |= ACTION_MODE_PHASE_REPULSE;
        }
    }

    #[inline]
    fn on_slide_out(
        &mut self,
        _frame: &mut GatewayFrame,
        stack: &mut CoreStateStack,
        _trigger: &HorizonTriggerInput,
    ) {
        let added_only = self.guard_aperture_bits & !self.saved_aperture_mask;
        stack.liminal.active_aperture_mask &= !added_only;
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
        slot_phase_out: &mut [lmnk_q7_t; MAX_ATTENTION_HEADS],
    ) -> LmnkResult<Option<f32>> {
        if !req.trigger.is_destabilized() || self.corpus.entries().is_empty() {
            return Ok(None);
        }
        let repulsion_lane = (req.lane_flags & WORKBENCH_LANE_FLAG_REPULSION) != 0;
        let prefer_guard = repulsion_lane
            || req.trigger.epistemic_regime == EpistemicRegime::InternalConflict as u8
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
                if repulsion_lane {
                    // Never borrow an attractive Fact/Persona into a repulsion-flagged lane
                    // when no repulsive Guard attractor has been observed in the corpus.
                    None
                } else {
                    self.corpus
                        .entries()
                        .iter()
                        .filter(|e| !e.per_layer_q7.is_empty())
                        .max_by(|a, b| a.energy.total_cmp(&b.energy))
                }
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
                let num = i32::from(v0) * w0 + i32::from(v1) * frac;
                let interp = if num >= 0 {
                    (num + 63) / 127
                } else {
                    (num - 63) / 127
                };
                *dst = interp.clamp(-127, 127) as lmnk_q7_t;
            }
        }
        if entry.role == AttractorRole::Guard {
            // Workbench lanes are strictly additive in core orchestration; convert Guard poles
            // into saturating anti-phase (`-R`, `i8::MIN -> +127`) matching `auto_repulsion`.
            for dst in slot_phase_out.iter_mut() {
                *dst = dst.saturating_neg();
            }
        }
        self.escalation_calls = self.escalation_calls.saturating_add(1);
        Ok(Some(entry.energy.max(self.min_weight)))
    }
}

/// Unified model-connection and optimization adapter over [`liminika_core`].
///
/// Embeds a heap-allocated [`RuntimeContext`] directly via `Box<RuntimeContext>` (zero
/// duplicated core fields) and bridges a target LLM's [`RopeGeometry`] to
/// [`liminika_core::stigmergy::PhaseSteeringEngine`], [`EntropyController`], and
/// [`ResidualAdapterEngine`] without leaking model-specific parsing or runtime dependencies into
/// `liminika-core`.
pub struct ModelSteeringAdapter {
    geometry: RopeGeometry,
    ctx: Box<RuntimeContext>,
    residual_adapter: ResidualAdapterEngine,
    system_constraint: SystemConstraint<()>,
    threshold_policy: AdaptiveThresholdPolicy,
    calibrated_base_thresholds: StigmergicThresholds,
    configured_ego: LiminalEgoState,
    last_ensemble_report: Option<StigmergicFragmentEnsembleReport>,
    coherence_guard_slot: CoherenceHorizonGuardSlot,
    residual_spillover: Box<DefaultResidualSpilloverSlot>,
    escalation_boundary_dirty: bool,
    last_escalated_regime: u8,
    active_workbench_preset: WorkbenchPreset,
    sandbox_settlement_streak: u8,
    prev_sandbox_coherence: f32,
    prev_sandbox_assim_drift: f32,
    crystallized_anchors: Vec<PhaseAnchorRecord>,
}

impl Drop for ModelSteeringAdapter {
    fn drop(&mut self) {
        self.ctx.zeroize_in_place();
        self.configured_ego = LiminalEgoState::new();
        self.residual_adapter = ResidualAdapterEngine::default();
        self.coherence_guard_slot = CoherenceHorizonGuardSlot::default();
        *self.residual_spillover = DefaultResidualSpilloverSlot::default();
        self.last_ensemble_report = None;
        self.crystallized_anchors.clear();
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

impl ModelSteeringAdapter {
    /// Maximum number of Cold-Path crystallized [`PhaseAnchorRecord`] entries retained in the
    /// bounded FIFO capture queue before oldest entries are evicted.
    pub const MAX_CRYSTALLIZED_ANCHOR_BUFFER: usize = 64;

    /// Default decode-token hold window before a transient contradiction-guard compute pin may
    /// relax toward low-cost bypass (`0` holds the entire turn; opt-in via `--guard-hold-tokens=<N>`).
    pub const DEFAULT_GUARD_HOLD_TOKENS: u8 = 0;
    pub(super) fn compute_steered_layer_count(geometry: &RopeGeometry) -> u16 {
        let mut count = 0u16;
        for l in 0..geometry.num_layers.min(64) {
            if geometry.is_rope_layer(l) {
                count += 1;
            }
        }
        count.max(1)
    }

    pub(super) const fn compute_slot_masks(num_layers: usize) -> (u64, u64, u64) {
        let n_slots = if num_layers < 64 { num_layers } else { 64 };
        let active_slots = if n_slots == 64 {
            u64::MAX
        } else {
            (1_u64 << n_slots) - 1
        };
        let pinned_slots = if n_slots == 0 {
            0
        } else if n_slots == 1 {
            1
        } else if n_slots < 12 {
            1_u64 | (1_u64 << (n_slots - 1))
        } else {
            // Proportionally protect head (`0..w`) and tail (`(n-w)..n`) boundary anchors with
            // `w = clamp(n / 8, 2, 4)` (exact head-4 + tail-4 parity at 32 layers).
            let raw = n_slots / 8;
            let pin_width = if raw < 2 {
                2
            } else if raw > 4 {
                4
            } else {
                raw
            };
            let low = (1_u64 << pin_width) - 1;
            low | (low << (n_slots - pin_width))
        };
        let reversible_slots = active_slots & !pinned_slots;
        (active_slots, pinned_slots, reversible_slots)
    }

    /// Creates a new [`ModelSteeringAdapter`] after validating `geometry`.
    ///
    /// # Errors
    ///
    /// Returns an error if `geometry.validate()` fails or [`RuntimeContext::new_boxed`]
    /// rejects the `(rotary_dim, num_q_heads)` pair.
    pub fn new(geometry: RopeGeometry) -> Result<Self, String> {
        geometry.validate()?;
        let mut cfg = Config::new()
            .with_head_dim(geometry.rotary_dim as u32)
            .with_num_heads(geometry.num_q_heads as u32)
            .with_substrate_capacity(16);
        cfg.raw.rotary_dim = geometry.rotary_dim as u32;
        cfg.raw.rope_layout = match geometry.layout {
            RopeLayout::Interleaved => 0,
            RopeLayout::HalfSplit => 1,
        };
        let mut ctx = RuntimeContext::new_boxed(cfg)
            .map_err(|e| format!("RuntimeContext::new_boxed failed: {e}"))?;

        ctx.substrate_mut().last_entropy_h_tilde = 0.35;
        let threshold_policy = AdaptiveThresholdPolicy::default();
        let calibrated_base_thresholds = threshold_policy.base;
        ctx.substrate_mut().thresholds = calibrated_base_thresholds;
        *ctx.entropy_controller_mut() = EntropyController::new(lmnk_entropy_state_t::default());

        let residual_adapter = ResidualAdapterEngine::new(1.0, 0, 1);
        let system_constraint = SystemConstraint::new((), 0.0, 0.0, 0xFF);
        ctx.probe_baseline_mut().active_heads = geometry.num_q_heads.min(64) as u16;
        ctx.probe_baseline_mut().steered_layers = Self::compute_steered_layer_count(&geometry);

        let (active_slots, pinned_slots, reversible_slots) =
            Self::compute_slot_masks(geometry.num_layers);
        ctx.compute_pool_mut()
            .set_slot_masks(active_slots, pinned_slots, 0, reversible_slots);
        ctx.set_baseline_pin_mask(pinned_slots);
        {
            let spine = ctx.system_spine_mut();
            spine.guard_hold_tokens = Self::DEFAULT_GUARD_HOLD_TOKENS;
            spine.set_max_observe_stride(1);
        }

        *ctx.last_horizon_mut() = EpistemicHorizonReading {
            total_memory_support: 1.0,
            peak_positive_affinity: 1.0,
            horizon_gate: 1.0,
            regime_code: EpistemicRegime::KnownCoherent as u8,
            ..EpistemicHorizonReading::default()
        };

        Ok(Self {
            geometry,
            ctx,
            residual_adapter,
            system_constraint,
            threshold_policy,
            calibrated_base_thresholds,
            configured_ego: LiminalEgoState::new(),
            last_ensemble_report: None,
            coherence_guard_slot: CoherenceHorizonGuardSlot::default(),
            residual_spillover: Box::default(),
            escalation_boundary_dirty: true,
            last_escalated_regime: EpistemicRegime::KnownCoherent as u8,
            active_workbench_preset: WorkbenchPreset::BalancedDialog,
            sandbox_settlement_streak: 0,
            prev_sandbox_coherence: 0.0,
            prev_sandbox_assim_drift: 1.0,
            crystallized_anchors: Vec::new(),
        })
    }
}
