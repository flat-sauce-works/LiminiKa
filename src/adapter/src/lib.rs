// name: src/adapter/src/lib.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model connection, RoPE geometry mapping, and optimization adapter over [`liminika_core`].
//!
//! Keeps [`liminika_core`] strictly `#![no_std]` and model-agnostic while providing a
//! unified, zero-external-dependency bridge for external LLM runtimes (Candle, `llama.cpp`,
//! and CLI evaluation harnesses).

use liminika_core::abi::{gcso_daes_slot_t, gcso_edbc_state_t, gcso_pointer_trail_t, gcso_q7_t};
use liminika_core::action_hub::{ActionHubEngine, StigmergicActionDecision, StigmergicThresholds};
use liminika_core::crystal::CrystallizedAttractorRecord;
use liminika_core::entropy_control::EntropyController;
use liminika_core::phase_steering::PhaseSteeringEngine;
use liminika_core::residual_adapter::ResidualAdapterEngine;
use liminika_core::storage::GcsoFourSectionView;
use liminika_core::traits::aligned::{
    AlignedSlice16, AlignedSlice32, AlignedSliceMut16, AlignedSliceMut32,
};
use liminika_core::traits::gateway::{
    ActionHubGatewayExt, AdaptiveThresholdPolicy, CausalProvenanceReport, CausalProvenanceSignal,
    GatewayFrame, GatewayPhaseSteeringExt, GatewaySrlSteeringExt, GuardPhaseAnchor,
    ManifoldGatewaySignal, ManifoldProbeBaseline, ObjectiveFact, StigmergicThresholdPolicy,
    SubjectivePersona, SystemConstraint, CAUSAL_AXIS_CONTRADICTION_GUARD,
    CAUSAL_AXIS_OBJECTIVE_FACT, CAUSAL_AXIS_SUBJECTIVE_PERSONA,
};
use liminika_core::traits::primitives::{
    ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA, ATTRACTOR_SLOT_REPULSION, MAX_ATTENTION_HEADS,
};
use liminika_core::traits::rope::{
    PhaseSteeringGeometryExt, RopeGeometryDescriptor, RopePairLayout, RopeSteeringTarget,
};
use liminika_core::traits::telemetry::SwarmResolutionReport;
use liminika_core::traits::SrlAdapter;

pub mod pooling;
pub mod reflection;
pub mod stratum;

pub use pooling::*;
pub use reflection::*;
pub use stratum::*;

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
struct Align32Chunk([f32; 8]);

/// Safe 32-byte aligned `f32` buffer for passing arbitrary-length slices to [`liminika_core`].
struct AlignedF32Buffer32 {
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
struct Align16Chunk([gcso_q7_t; 16]);

/// Safe 16-byte aligned `gcso_q7_t` buffer for [`AlignedSlice16`] / [`AlignedSliceMut16`].
struct AlignedQ7Buffer16 {
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
    daes_slot: gcso_daes_slot_t,
    decision_counts: [u64; 7],
    entropy_controller: EntropyController,
    residual_adapter: ResidualAdapterEngine,
    system_constraint: SystemConstraint<()>,
    probe_baseline: ManifoldProbeBaseline,
    threshold_policy: AdaptiveThresholdPolicy,
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
        let daes_slot = gcso_daes_slot_t::default();
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
            daes_slot,
            decision_counts: [0; 7],
            entropy_controller,
            residual_adapter,
            system_constraint,
            probe_baseline,
            threshold_policy,
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
    /// `ActionHubEngine`, and `daes_slot`) when the underlying LLM or RoPE geometry changes
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
        self.daes_slot = gcso_daes_slot_t::default();
        self.decision_counts = [0; 7];
        self.probe_baseline = next_probe;
        Ok(self.probe_baseline.probe_epoch)
    }

    /// Returns a mutable reference to the embedded [`EntropyController`] (EDBC).
    pub fn entropy_controller_mut(&mut self) -> &mut EntropyController {
        &mut self.entropy_controller
    }

    /// Returns a mutable reference to the embedded [`ResidualAdapterEngine`] (SRL).
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
    /// stigmergic trail density, DAES fast-path state, and frame energy to autonomously modulate
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
                    &mut this.daes_slot,
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
            let (frame, report, decision) = this
                .action_hub
                .orchestrate_gateway_layer_with_policy_safe(
                    signal,
                    &this.geometry,
                    &this.threshold_policy,
                    &this.system_constraint,
                    &this.probe_baseline,
                    entropy_h_tilde,
                    &mut this.daes_slot,
                    trail_inout,
                    unresolved_damping_u7,
                    Some(q),
                    None,
                    logits,
                )
                .map_err(|e| format!("orchestrate_gateway_layer_with_policy_safe error: {e}"))?;
            this.action_hub.last_decision = decision as u8;
            this.decision_counts[(decision as usize).min(6)] += 1;
            Ok((frame, report, decision))
        })
    }

    /// Executes norm-adaptive SRL residual injection only when [`ActionHubEngine`]'s latest
    /// [`StigmergicActionDecision`] authorizes full orchestration (`FullOrchestrationSrl` or
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
            StigmergicActionDecision::FullOrchestrationSrl
                | StigmergicActionDecision::CrystallizeMacroBasin
        ) {
            let res = self.apply_srl_with_gateway(signal, hidden_inout, u_vec, v_vec)?;
            Ok(Some(res))
        } else {
            Ok(None)
        }
    }

    /// Applies norm-adaptive SRL residual injection driven purely by a [`ManifoldGatewaySignal`]
    /// via [`GatewaySrlSteeringExt::apply_gateway_srl_hidden_fused`] without manual scale tuning.
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
                "SRL hidden/u/v length mismatch: hidden={}, u={}, v={}",
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

        let (frame, gain) = {
            let mut y_mut = AlignedSliceMut32::new(y_buf.as_mut_slice())
                .map_err(|e| format!("AlignedSliceMut32 y error: {e}"))?;
            let x_aligned = AlignedSlice32::new(x_buf.as_slice())
                .map_err(|e| format!("AlignedSlice32 x error: {e}"))?;
            let u_aligned = AlignedSlice32::new(u_buf.as_slice())
                .map_err(|e| format!("AlignedSlice32 u error: {e}"))?;
            let v_aligned = AlignedSlice32::new(v_buf.as_slice())
                .map_err(|e| format!("AlignedSlice32 v error: {e}"))?;
            self.residual_adapter
                .apply_gateway_srl_hidden_fused(
                    signal, &mut y_mut, &x_aligned, &u_aligned, &v_aligned,
                )
                .map_err(|e| format!("apply_gateway_srl_hidden_fused error: {e}"))?
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

    /// Applies Rank-1 SRL residual injection (`y <- y + s * u * (v^T * y)`) to a hidden-state
    /// vector via [`SrlAdapter::eval_rank1_slices_fused`].
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
                "SRL hidden/u/v length mismatch: hidden={}, u={}, v={}",
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

/// Semantic origin role of a natural-language text attractor inside [`TextAttractorCorpus`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttractorRole {
    /// Factual / declarative knowledge chunk (`ObjectiveFact` origin).
    Fact,
    /// Stylistic / behavioral / personality log (`SubjectivePersona` origin).
    Persona,
    /// Contrastive / negative guard boundary (`ContradictionGuard` origin).
    Guard,
}

/// Geometric classification of a query's position relative to the compiled attractor manifold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoidBoundaryState {
    /// Query clearly resides inside a resonant attractor basin.
    InsideBasin,
    /// Query sits on the `cvoid` / liminal boundary between two competing attractors.
    LiminalBoundary,
    /// Query falls in empty ungrounded space (`cvoid`) below minimum resonance; avoid forced steering.
    CvoidEmptySpace,
}

/// Single natural-language-backed attractor entry compiled for the active model's RoPE manifold.
#[derive(Debug, Clone)]
pub struct TextAttractorEntry {
    /// Unique stable anchor identifier (`1..`).
    pub anchor_id: u32,
    /// Origin role (`Fact`, `Persona`, or `Guard`).
    pub role: AttractorRole,
    /// Lightweight natural-language source text (reproducible source of truth).
    pub text: String,
    /// Per-layer 64-head Q7 phase signatures (`num_layers` entries).
    pub per_layer_q7: Vec<[gcso_q7_t; MAX_ATTENTION_HEADS]>,
    /// Base gravitational pull energy for gateway composition.
    pub energy: f32,
    /// L2-normalized centered residual Q signature across steered layers for high-resolution semantic recall.
    residual_sig: Vec<f32>,
}

impl CausalProvenanceSignal for TextAttractorEntry {
    fn causal_provenance_bits(&self) -> u32 {
        match self.role {
            AttractorRole::Fact => CAUSAL_AXIS_OBJECTIVE_FACT,
            AttractorRole::Persona => CAUSAL_AXIS_SUBJECTIVE_PERSONA,
            AttractorRole::Guard => CAUSAL_AXIS_CONTRADICTION_GUARD,
        }
    }
}

impl ManifoldGatewaySignal for TextAttractorEntry {
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        let q7 = self
            .per_layer_q7
            .last()
            .copied()
            .unwrap_or([0_i8; MAX_ATTENTION_HEADS]);
        let layer_idx = self.per_layer_q7.len().saturating_sub(1) as u64;
        match self.role {
            AttractorRole::Fact => ObjectiveFact::new(q7, self.energy, layer_idx, self.anchor_id)
                .synthesize_gateway_frame(),
            AttractorRole::Persona => {
                SubjectivePersona::new(q7, self.energy, layer_idx, self.anchor_id)
                    .synthesize_gateway_frame()
            }
            AttractorRole::Guard => GuardPhaseAnchor::new(q7, self.energy.clamp(0.0, 1.0), 0.5)
                .synthesize_gateway_frame(),
        }
    }
}

/// Dynamic top-K recall result from [`TextAttractorCorpus::recall_for_query`].
#[derive(Debug, Clone)]
pub struct DynamicAttractorRecall<'a> {
    /// Whether the query landed inside a basin, on a liminal boundary, or in `cvoid` empty space.
    pub void_state: VoidBoundaryState,
    /// Highest geometric resonance score among Fact attractors.
    pub top1_fact_score: f32,
    /// Second-highest geometric resonance score among Fact attractors.
    pub top2_fact_score: f32,
    /// Highest geometric resonance score among Persona attractors.
    pub top1_persona_score: f32,
    /// Smallest per-attractor local `cvoid` depth observed across all attractors (`0.0` = deep in a basin).
    pub min_local_void_depth: f32,
    /// Highest Guard resonance minus highest Fact/Persona resonance (`> 0` indicates guard-dominant conflict).
    pub guard_conflict_margin: f32,
    /// `ActionHub` regime selected by the upper boundary stratum (`PassiveSidecar`, `InertialLayerSkip`,
    /// `ExpressPhaseSteer`, `PspmSubHeadRoute`, `FullOrchestrationSrl`, `CrystallizeMacroBasin`, or `RepulsionEscape`).
    pub action_decision: StigmergicActionDecision,
    /// Authoritative 128-byte core [`GatewayFrame`] synthesized and arbitrated by the pipeline.
    pub gateway_frame: GatewayFrame,
    /// Cross-channel probe telemetry and core [`CausalProvenanceReport`].
    pub telemetry: StratumProbeTelemetry,
    /// Top-K recalled Fact entries (empty when `void_state == CvoidEmptySpace`).
    pub recalled_facts: Vec<(&'a TextAttractorEntry, f32)>,
    /// Top-K recalled Persona entries (suppressed only on Guard repulsion).
    pub recalled_personas: Vec<(&'a TextAttractorEntry, f32)>,
}

impl<'a> DynamicAttractorRecall<'a> {
    /// Materializes per-layer `ObjectiveFact<[i8; 64]>` anchors for
    /// [`ResonantSignalSlice`](liminika_core::traits::gateway::ResonantSignalSlice).
    #[must_use]
    pub fn build_fact_anchors_per_layer(
        &self,
        num_layers: usize,
    ) -> Vec<Vec<ObjectiveFact<[gcso_q7_t; MAX_ATTENTION_HEADS]>>> {
        (0..num_layers)
            .map(|l| {
                self.recalled_facts
                    .iter()
                    .filter_map(|(entry, score)| {
                        entry.per_layer_q7.get(l).map(|&arr| {
                            let scaled_e = (entry.energy * score.clamp(0.25, 1.25)).max(0.25);
                            ObjectiveFact::new(arr, scaled_e, 0, entry.anchor_id)
                        })
                    })
                    .collect()
            })
            .collect()
    }

    /// Materializes per-layer `SubjectivePersona<[i8; 64]>` anchors for
    /// [`ResonantSignalSlice`](liminika_core::traits::gateway::ResonantSignalSlice).
    #[must_use]
    pub fn build_persona_anchors_per_layer(
        &self,
        num_layers: usize,
    ) -> Vec<Vec<SubjectivePersona<[gcso_q7_t; MAX_ATTENTION_HEADS]>>> {
        (0..num_layers)
            .map(|l| {
                self.recalled_personas
                    .iter()
                    .filter_map(|(entry, score)| {
                        entry.per_layer_q7.get(l).map(|&arr| {
                            let scaled_e = (entry.energy * score.clamp(0.25, 1.25)).max(0.25);
                            SubjectivePersona::new(arr, scaled_e, 0, entry.anchor_id)
                        })
                    })
                    .collect()
            })
            .collect()
    }
}

/// Policy parameters controlling top-K attractor recall and `cvoid` boundary classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecallPolicy {
    /// Maximum number of fact attractors to return when inside a basin or liminal boundary.
    pub top_k_facts: usize,
    /// Maximum number of persona attractors to return.
    pub top_k_personas: usize,
    /// Minimum top-1 geometric coherence required to stay inside an attractor basin (below this -> `CvoidEmptySpace`).
    pub min_coherence: f32,
    /// Minimum top-1 minus top-2 score gap to avoid `LiminalBoundary`.
    pub boundary_margin: f32,
    /// Guard conflict margin (`max_guard_score - top1_fact_score`) above which fact injection is suppressed.
    pub guard_suppress_margin: f32,
}

impl Default for RecallPolicy {
    fn default() -> Self {
        Self {
            top_k_facts: 2,
            top_k_personas: 1,
            min_coherence: 0.35,
            boundary_margin: 0.03,
            guard_suppress_margin: 0.0,
        }
    }
}

/// Geometric boundary measurements evaluated by a [`RecallBoundaryGate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecallBoundaryMetrics {
    /// Highest geometric resonance score among Fact attractors.
    pub top1_fact_score: f32,
    /// Second-highest geometric resonance score among Fact attractors.
    pub top2_fact_score: f32,
    /// Highest geometric resonance score among Persona attractors.
    pub top1_persona_score: f32,
    /// Highest geometric resonance score among Guard attractors.
    pub max_guard_score: f32,
    /// `max_guard_score - max(top1_fact_score, top1_persona_score)`.
    pub guard_conflict_margin: f32,
}

/// Decision produced by a [`RecallBoundaryGate`] for a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecallGateDecision {
    /// Classified boundary state (`InsideBasin`, `LiminalBoundary`, or `CvoidEmptySpace`).
    pub void_state: VoidBoundaryState,
    /// Whether forced Fact/Persona attractor injection should be suppressed (e.g., `cvoid` or Guard dominance).
    pub suppress_fact_injection: bool,
}

/// Trait abstracting `cvoid` boundary classification and Guard-conflict suppression policies.
pub trait RecallBoundaryGate {
    /// Evaluates boundary metrics under the active [`RecallPolicy`].
    fn evaluate(
        &self,
        metrics: &RecallBoundaryMetrics,
        policy: &RecallPolicy,
    ) -> RecallGateDecision;
}

/// Relational neighborhood context observed from a single attractor's vantage point, enabling
/// asymmetric, perspective-dependent `cvoid` measurements (analogous to viewing a chain from the
/// left edge vs. the right edge or per-head Attention affinity).
#[derive(Debug, Clone, Copy)]
pub struct AttractorPerspectiveContext<'a> {
    /// Target attractor whose local frame is being evaluated.
    pub entry: &'a TextAttractorEntry,
    /// Raw cosine similarity between the query and this attractor.
    pub raw_cosine: f32,
    /// Mean cosine from this attractor to its local nearest neighbors inside the corpus.
    pub neighbor_mean_cosine: f32,
    /// Standard deviation of cosine similarity around this attractor's local neighborhood.
    pub neighbor_std_cosine: f32,
    /// Corpus-wide average of `neighbor_mean_cosine`.
    pub global_neighbor_mean: f32,
    /// Query's mean cosine similarity across the corpus (query-side baseline).
    pub query_mean_cosine: f32,
    /// Standard deviation of the query's cosine similarity across the corpus.
    pub query_std_cosine: f32,
}

/// Per-attractor measurement returned by [`AttractorVoidPerspective::measure_perspective`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerspectiveVoidReading {
    /// Perspective-calibrated affinity score as viewed from this attractor's angle.
    pub perspective_affinity: f32,
    /// Non-negative local `cvoid` gap seen from this attractor (`0.0` = well inside basin).
    pub local_void_depth: f32,
}

/// Trait computing perspective-dependent affinity and local `cvoid` void depth per attractor.
///
/// Rather than judging every attractor from a single isotropic origin, each attractor can have a
/// distinct local density and boundary geometry so central hubs do not eclipse peripheral basins.
pub trait AttractorVoidPerspective {
    /// Measures affinity and local `cvoid` depth from `ctx.entry`'s vantage point.
    fn measure_perspective(
        &self,
        ctx: &AttractorPerspectiveContext<'_>,
        min_coherence: f32,
    ) -> PerspectiveVoidReading;
}

/// Default [`AttractorVoidPerspective`] implementing bidirectional CSLS hubness normalization and
/// asymmetric Z-score scaling across each attractor's local neighborhood.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AsymmetricDominoVoidPerspective {
    /// Number of nearest corpus neighbors defining each attractor's local baseline.
    pub neighbor_k: usize,
    /// Damping weight applied to centered attractor hubness (`neighbor_mean - global_mean`).
    pub hubness_damping: f32,
    /// Weight applied to the query/attractor bidirectional Z-score modulated by positive raw cosine.
    pub perspective_z_gain: f32,
}

impl Default for AsymmetricDominoVoidPerspective {
    fn default() -> Self {
        Self {
            neighbor_k: 4,
            hubness_damping: 0.50,
            perspective_z_gain: 0.025,
        }
    }
}

impl AttractorVoidPerspective for AsymmetricDominoVoidPerspective {
    fn measure_perspective(
        &self,
        ctx: &AttractorPerspectiveContext<'_>,
        min_coherence: f32,
    ) -> PerspectiveVoidReading {
        let hub_shift = ctx.neighbor_mean_cosine - ctx.global_neighbor_mean;
        let z_from_attractor =
            (ctx.raw_cosine - ctx.neighbor_mean_cosine) / ctx.neighbor_std_cosine.max(0.05);
        let z_from_query =
            (ctx.raw_cosine - ctx.query_mean_cosine) / ctx.query_std_cosine.max(0.05);
        let rel_z = (0.5 * (z_from_attractor + z_from_query)).clamp(-3.0, 3.0);
        let positive_support = ctx.raw_cosine.max(0.0);
        // Gate hubness compensation by positive raw support so unrelated singleton entries never
        // gain phantom positive affinity when `raw_cosine <= 0`, while true dense hubs still damp.
        let support_gate = (positive_support / min_coherence.max(0.05)).clamp(0.0, 1.0);
        let damped_hub = if hub_shift < 0.0 {
            hub_shift * support_gate * support_gate
        } else {
            hub_shift * support_gate
        };
        let perspective_affinity = ctx.raw_cosine - self.hubness_damping * damped_hub
            + self.perspective_z_gain * positive_support * rel_z;
        let local_void_depth = (min_coherence - perspective_affinity).max(0.0);
        PerspectiveVoidReading {
            perspective_affinity,
            local_void_depth,
        }
    }
}

/// Default [`RecallBoundaryGate`] enforcing both `cvoid` coherence floor and Guard dominance suppression.
#[derive(Debug, Default, Clone, Copy)]
pub struct CoherenceGuardGate;

impl RecallBoundaryGate for CoherenceGuardGate {
    fn evaluate(
        &self,
        metrics: &RecallBoundaryMetrics,
        policy: &RecallPolicy,
    ) -> RecallGateDecision {
        if metrics.guard_conflict_margin > policy.guard_suppress_margin {
            return RecallGateDecision {
                void_state: VoidBoundaryState::CvoidEmptySpace,
                suppress_fact_injection: true,
            };
        }
        let margin = metrics.top1_fact_score - metrics.top2_fact_score;
        let sharp_liminal_match = metrics.top1_fact_score >= policy.min_coherence * 0.65
            && margin >= policy.boundary_margin
            && metrics.top1_fact_score >= metrics.top2_fact_score.max(0.0) * 1.5;

        if metrics.top1_fact_score < policy.min_coherence && !sharp_liminal_match {
            RecallGateDecision {
                void_state: VoidBoundaryState::CvoidEmptySpace,
                suppress_fact_injection: true,
            }
        } else if sharp_liminal_match && metrics.top1_fact_score < policy.min_coherence
            || margin < policy.boundary_margin
        {
            RecallGateDecision {
                void_state: VoidBoundaryState::LiminalBoundary,
                suppress_fact_injection: false,
            }
        } else {
            RecallGateDecision {
                void_state: VoidBoundaryState::InsideBasin,
                suppress_fact_injection: false,
            }
        }
    }
}

/// Model-agnostic Cold-Path corpus that compiles natural-language logs, facts, personas, and
/// contrastive guards into a dynamic attractor manifold with explicit `cvoid` boundary detection.
#[derive(Debug, Default, Clone)]
pub struct TextAttractorCorpus {
    entries: Vec<TextAttractorEntry>,
    next_anchor_id: u32,
    common_mode_basis: Vec<Vec<f32>>,
}

impl TextAttractorCorpus {
    /// Creates an empty [`TextAttractorCorpus`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_anchor_id: 1,
            common_mode_basis: Vec::new(),
        }
    }

    /// Returns all stored attractor entries.
    #[must_use]
    pub fn entries(&self) -> &[TextAttractorEntry] {
        &self.entries
    }

    pub(crate) fn build_steered_residual_sig(
        geom: &RopeGeometry,
        q_layers: &[&[f32]],
        ref_layers: &[&[f32]],
    ) -> Vec<f32> {
        let mut sig = Vec::new();
        for (l, (q, r)) in q_layers.iter().zip(ref_layers.iter()).enumerate() {
            if !geom.is_rope_layer(l) {
                continue;
            }
            let start = sig.len();
            for (&qv, &rv) in q.iter().zip(r.iter()) {
                sig.push(qv - rv);
            }
            // Equalize per-layer contribution so late-layer norm spikes cannot drown mid-layer semantics.
            Self::l2_normalize_in_place(&mut sig[start..]);
        }
        Self::l2_normalize_in_place(&mut sig);
        sig
    }

    fn l2_normalize_in_place(v: &mut [f32]) {
        let norm_sq: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        let norm = norm_sq.sqrt();
        if norm > 1e-12 {
            let inv = (1.0 / norm) as f32;
            for x in v.iter_mut() {
                *x *= inv;
            }
        }
    }

    pub(crate) fn project_and_normalize(v: &[f32], basis: &[Vec<f32>]) -> Vec<f32> {
        let mut out = v.to_vec();
        if !basis.is_empty() {
            for dir in basis {
                if dir.len() != out.len() {
                    continue;
                }
                let dot: f64 = out
                    .iter()
                    .zip(dir.iter())
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                let d = dot as f32;
                for (x, &c) in out.iter_mut().zip(dir.iter()) {
                    *x -= d * c;
                }
            }
            Self::l2_normalize_in_place(&mut out);
        }
        out
    }

    pub(crate) fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut dot = 0.0_f64;
        let mut na = 0.0_f64;
        let mut nb = 0.0_f64;
        for i in 0..len {
            let x = f64::from(a[i]);
            let y = f64::from(b[i]);
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        let denom = (na * nb).sqrt();
        if denom > 1e-12 {
            (dot / denom).clamp(-1.0, 1.0) as f32
        } else {
            0.0
        }
    }

    /// Estimates up to `max_components` orthonormal common-mode directions from `sigs` via
    /// deflated power iteration so shared template/attention-sink directions can be projected out.
    pub(crate) fn fit_common_mode_subspace_from_sigs(
        sigs: &[Vec<f32>],
        max_components: usize,
    ) -> Vec<Vec<f32>> {
        let mut basis: Vec<Vec<f32>> = Vec::new();
        if sigs.len() < 4 || max_components == 0 {
            return basis;
        }
        let dim = sigs[0].len();
        if dim == 0 || sigs.iter().any(|s| s.len() != dim) {
            return basis;
        }
        let rank_cap = max_components.min(sigs.len() / 3);
        let mut work: Vec<Vec<f32>> = sigs.to_vec();

        for _ in 0..rank_cap {
            let mut v = vec![0.0_f32; dim];
            for w in &work {
                for (dst, &src) in v.iter_mut().zip(w.iter()) {
                    *dst += src;
                }
            }
            for dir in &basis {
                let dot: f64 = v
                    .iter()
                    .zip(dir.iter())
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                let d = dot as f32;
                for (dst, &c) in v.iter_mut().zip(dir.iter()) {
                    *dst -= d * c;
                }
            }
            Self::l2_normalize_in_place(&mut v);
            if v.iter().all(|&x| x == 0.0) {
                break;
            }
            for _ in 0..8 {
                let mut next = vec![0.0_f32; dim];
                for w in &work {
                    let dot: f64 = w
                        .iter()
                        .zip(v.iter())
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum();
                    let d = dot as f32;
                    for (dst, &src) in next.iter_mut().zip(w.iter()) {
                        *dst += d * src;
                    }
                }
                for dir in &basis {
                    let dot: f64 = next
                        .iter()
                        .zip(dir.iter())
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum();
                    let d = dot as f32;
                    for (dst, &c) in next.iter_mut().zip(dir.iter()) {
                        *dst -= d * c;
                    }
                }
                let norm_sq: f64 = next.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
                if norm_sq < 1e-12 {
                    break;
                }
                Self::l2_normalize_in_place(&mut next);
                v = next;
            }
            let norm_sq: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            if norm_sq < 0.5 {
                break;
            }
            for w in &mut work {
                let dot: f64 = w
                    .iter()
                    .zip(v.iter())
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                let d = dot as f32;
                for (dst, &c) in w.iter_mut().zip(v.iter()) {
                    *dst -= d * c;
                }
            }
            basis.push(v);
        }
        basis
    }

    /// Estimates up to `max_components` orthonormal common-mode directions from corpus signatures
    /// via deflated power iteration.
    pub fn fit_common_mode_subspace(&mut self, max_components: usize) {
        let sigs: Vec<Vec<f32>> = self
            .entries
            .iter()
            .map(|e| e.residual_sig.clone())
            .collect();
        self.common_mode_basis = Self::fit_common_mode_subspace_from_sigs(&sigs, max_components);
    }

    /// Estimates and stores the dominant unsupervised common-mode subspace (up to 3 components)
    /// so recall projects out shared prompt/template bias.
    pub fn fit_common_mode_projection(&mut self) {
        self.fit_common_mode_subspace(3);
    }

    /// Appends external orthonormal prior directions (e.g., startup base-model framing modes)
    /// into the common-mode projection basis via modified Gram-Schmidt orthogonalization.
    pub fn extend_common_mode_basis(&mut self, extra_dirs: &[Vec<f32>], max_total: usize) {
        let Some(expected_dim) = self
            .entries
            .first()
            .map(|e| e.residual_sig.len())
            .filter(|&d| d > 0)
        else {
            return;
        };
        for dir in extra_dirs {
            if self.common_mode_basis.len() >= max_total {
                break;
            }
            if dir.len() != expected_dim {
                continue;
            }
            let mut orth = dir.clone();
            for existing in &self.common_mode_basis {
                let dot: f64 = orth
                    .iter()
                    .zip(existing.iter())
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                let d = dot as f32;
                for (dst, &c) in orth.iter_mut().zip(existing.iter()) {
                    *dst -= d * c;
                }
            }
            let norm_sq: f64 = orth.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            if norm_sq >= 0.25 {
                Self::l2_normalize_in_place(&mut orth);
                self.common_mode_basis.push(orth);
            }
        }
    }

    /// Returns common-mode-projected normalized signatures across the corpus.
    fn projected_corpus_sigs(&self) -> Vec<Vec<f32>> {
        self.entries
            .iter()
            .map(|e| Self::project_and_normalize(&e.residual_sig, &self.common_mode_basis))
            .collect()
    }

    /// Calibrates [`RecallPolicy`] from the empirical off-diagonal similarity distribution of the
    /// corpus (after common-mode projection), avoiding hardcoded model-specific thresholds.
    #[must_use]
    pub fn calibrate_recall_policy(
        &self,
        top_k_facts: usize,
        top_k_personas: usize,
    ) -> RecallPolicy {
        if self.entries.len() < 2 {
            return RecallPolicy {
                top_k_facts,
                top_k_personas,
                ..RecallPolicy::default()
            };
        }
        let projected = self.projected_corpus_sigs();
        let mut sims = Vec::new();
        for i in 0..projected.len() {
            for j in (i + 1)..projected.len() {
                sims.push(f64::from(Self::cosine_f32(&projected[i], &projected[j])));
            }
        }
        let n = sims.len().max(1) as f64;
        let mean = sims.iter().sum::<f64>() / n;
        let var = sims.iter().map(|&s| (s - mean) * (s - mean)).sum::<f64>() / n;
        let std = var.sqrt();
        let min_coherence = (mean + 1.25 * std).clamp(0.15, 0.75) as f32;
        let boundary_margin = (0.30 * std).clamp(0.01, 0.12) as f32;
        RecallPolicy {
            top_k_facts,
            top_k_personas,
            min_coherence,
            boundary_margin,
            guard_suppress_margin: 0.0,
        }
    }

    /// Compiles and appends a single natural-language text chunk into the attractor corpus using
    /// the active model's `adapter` and per-layer baseline reference activations.
    ///
    /// # Errors
    ///
    /// Returns an error if `per_layer_q.len() != baseline_ref_per_layer.len()` or any layer slice
    /// mismatches `adapter.geometry().q_rot_len()`.
    pub fn ingest_text_activations(
        &mut self,
        adapter: &ModelSteeringAdapter,
        role: AttractorRole,
        text: impl Into<String>,
        per_layer_q: &[&[f32]],
        baseline_ref_per_layer: &[&[f32]],
        energy: f32,
    ) -> Result<u32, String> {
        if per_layer_q.len() != baseline_ref_per_layer.len() {
            return Err(format!(
                "layer count mismatch: per_layer_q={}, baseline={}",
                per_layer_q.len(),
                baseline_ref_per_layer.len()
            ));
        }
        let mut per_layer_q7 = Vec::with_capacity(per_layer_q.len());
        for (q, r) in per_layer_q.iter().zip(baseline_ref_per_layer.iter()) {
            let q7 = adapter.bake_layer_procrustes_q7(r, q)?;
            let mut arr64 = [0i8; MAX_ATTENTION_HEADS];
            let n = q7.len().min(MAX_ATTENTION_HEADS);
            arr64[..n].copy_from_slice(&q7[..n]);
            per_layer_q7.push(arr64);
        }
        let residual_sig = Self::build_steered_residual_sig(
            adapter.geometry(),
            per_layer_q,
            baseline_ref_per_layer,
        );
        let id = self.next_anchor_id;
        self.next_anchor_id = self.next_anchor_id.saturating_add(1);
        self.entries.push(TextAttractorEntry {
            anchor_id: id,
            role,
            text: text.into(),
            per_layer_q7,
            energy: energy.max(0.1),
            residual_sig,
        });
        Ok(id)
    }

    /// Re-bakes or scores a query's per-layer Q activations against the corpus purely via
    /// steered-layer geometric cosine similarity (zero keyword heuristics), using the default
    /// [`CoherenceGuardGate`] to classify [`VoidBoundaryState`].
    ///
    /// # Errors
    ///
    /// Returns an error if query/baseline layer slices mismatch `adapter.geometry()`.
    pub fn recall_for_query(
        &self,
        adapter: &ModelSteeringAdapter,
        query_per_layer_q: &[&[f32]],
        baseline_ref_per_layer: &[&[f32]],
        policy: &RecallPolicy,
    ) -> Result<DynamicAttractorRecall<'_>, String> {
        self.recall_for_query_with_gate(
            adapter,
            query_per_layer_q,
            baseline_ref_per_layer,
            policy,
            &CoherenceGuardGate,
        )
    }

    /// Scores a query's per-layer Q activations against the corpus using the default
    /// [`AsymmetricDominoVoidPerspective`] and delegates boundary/guard gating to a
    /// caller-supplied [`RecallBoundaryGate`] trait object.
    ///
    /// # Errors
    ///
    /// Returns an error if query/baseline layer slices mismatch `adapter.geometry()`.
    pub fn recall_for_query_with_gate(
        &self,
        adapter: &ModelSteeringAdapter,
        query_per_layer_q: &[&[f32]],
        baseline_ref_per_layer: &[&[f32]],
        policy: &RecallPolicy,
        gate: &dyn RecallBoundaryGate,
    ) -> Result<DynamicAttractorRecall<'_>, String> {
        self.recall_for_query_with_perspective(
            adapter,
            query_per_layer_q,
            baseline_ref_per_layer,
            policy,
            gate,
            &AsymmetricDominoVoidPerspective::default(),
        )
    }

    /// Scores a query's per-layer Q activations against the corpus using caller-supplied
    /// [`AttractorVoidPerspective`] and [`RecallBoundaryGate`] trait objects inside the default
    /// 3-stratum Society-of-Mind pipeline.
    ///
    /// # Errors
    ///
    /// Returns an error if query/baseline layer slices mismatch `adapter.geometry()`.
    pub fn recall_for_query_with_perspective(
        &self,
        adapter: &ModelSteeringAdapter,
        query_per_layer_q: &[&[f32]],
        baseline_ref_per_layer: &[&[f32]],
        policy: &RecallPolicy,
        gate: &dyn RecallBoundaryGate,
        perspective: &dyn AttractorVoidPerspective,
    ) -> Result<DynamicAttractorRecall<'_>, String> {
        let mut pipeline = StratumPipeline::new();
        pipeline.push_stratum(Box::new(LocalFeatureWorkerStratum::default()));
        pipeline.push_stratum(Box::new(HeadAngleWorkerStratum::default()));
        pipeline.push_stratum(Box::new(LocalTokenResonanceStratum::default()));
        pipeline.push_stratum(Box::new(PerspectiveDominoStratum { perspective }));
        pipeline.push_stratum(Box::new(ActionHubBoundaryStratum { gate }));
        self.recall_for_query_with_strata(
            adapter,
            query_per_layer_q,
            baseline_ref_per_layer,
            policy,
            &pipeline,
        )
    }

    /// Scores a query's per-layer Q activations through a dynamically composable [`StratumPipeline`]
    /// (`n - 1 <-> n <-> n + 1`), enabling runtime addition or removal of strata.
    ///
    /// # Errors
    ///
    /// Returns an error if query/baseline layer slices mismatch `adapter.geometry()` or a stratum fails.
    pub fn recall_for_query_with_strata(
        &self,
        adapter: &ModelSteeringAdapter,
        query_per_layer_q: &[&[f32]],
        baseline_ref_per_layer: &[&[f32]],
        policy: &RecallPolicy,
        pipeline: &StratumPipeline<'_>,
    ) -> Result<DynamicAttractorRecall<'_>, String> {
        self.recall_for_query_with_strata_and_local_scores(
            adapter,
            query_per_layer_q,
            baseline_ref_per_layer,
            policy,
            pipeline,
            None,
        )
    }

    /// Scores a query's per-layer Q activations through a [`StratumPipeline`] while supplying an
    /// independent `local_token_scores` channel (e.g., ColBERT-style MaxSim point-cloud scores).
    ///
    /// # Errors
    ///
    /// Returns an error if query/baseline layer slices mismatch `adapter.geometry()`,
    /// `local_token_scores` length mismatches `self.entries().len()`, or a stratum fails.
    pub fn recall_for_query_with_strata_and_local_scores(
        &self,
        adapter: &ModelSteeringAdapter,
        query_per_layer_q: &[&[f32]],
        baseline_ref_per_layer: &[&[f32]],
        policy: &RecallPolicy,
        pipeline: &StratumPipeline<'_>,
        local_token_scores: Option<&[f32]>,
    ) -> Result<DynamicAttractorRecall<'_>, String> {
        let obs = local_token_scores.map(|scores| LocalChannelObservation {
            combined_scores: scores.to_vec(),
            ..Default::default()
        });
        self.recall_for_query_with_strata_observation(
            adapter,
            query_per_layer_q,
            baseline_ref_per_layer,
            policy,
            pipeline,
            obs,
        )
    }

    /// Scores a query's per-layer Q activations through a [`StratumPipeline`] while preserving an
    /// uncollapsed [`LocalChannelObservation`] (bidirectional Chamfer scores + per-token swarm
    /// deposits) for [`liminika_core::action_hub::ActionHubEngine`] trail decomposition.
    ///
    /// # Errors
    ///
    /// Returns an error if query/baseline layer slices mismatch `adapter.geometry()`,
    /// observation vectors mismatch `self.entries().len()`, or a stratum fails.
    pub fn recall_for_query_with_strata_observation(
        &self,
        adapter: &ModelSteeringAdapter,
        query_per_layer_q: &[&[f32]],
        baseline_ref_per_layer: &[&[f32]],
        policy: &RecallPolicy,
        pipeline: &StratumPipeline<'_>,
        local_observation: Option<LocalChannelObservation>,
    ) -> Result<DynamicAttractorRecall<'_>, String> {
        if query_per_layer_q.len() != baseline_ref_per_layer.len() {
            return Err(format!(
                "query/baseline layer mismatch: {} vs {}",
                query_per_layer_q.len(),
                baseline_ref_per_layer.len()
            ));
        }
        let geom = adapter.geometry();
        let expected = geom.q_rot_len();
        for (l, (q, r)) in query_per_layer_q
            .iter()
            .zip(baseline_ref_per_layer.iter())
            .enumerate()
        {
            if !geom.is_rope_layer(l) {
                continue;
            }
            if q.len() != expected || r.len() != expected {
                return Err(format!(
                    "layer {l}: expected q_rot length {expected}, got query={} baseline={}",
                    q.len(),
                    r.len()
                ));
            }
        }
        let raw_sig =
            Self::build_steered_residual_sig(geom, query_per_layer_q, baseline_ref_per_layer);
        let query_sig = Self::project_and_normalize(&raw_sig, &self.common_mode_basis);
        let entry_sigs = self.projected_corpus_sigs();
        let raw_cosines: Vec<f32> = entry_sigs
            .iter()
            .map(|sig| Self::cosine_f32(&query_sig, sig))
            .collect();
        let n_entries = entry_sigs.len();
        let (token_channel, q_to_d_scores, d_to_q_scores, token_deposits) = match local_observation
        {
            Some(obs) => {
                if obs.combined_scores.len() != n_entries {
                    return Err(format!(
                        "local_token_scores length mismatch: expected {n_entries}, got {}",
                        obs.combined_scores.len()
                    ));
                }
                (
                    obs.combined_scores,
                    obs.q_to_d_scores,
                    obs.d_to_q_scores,
                    obs.token_deposits,
                )
            }
            None => (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
        };
        let initial_signal = StratumSignal {
            candidate_scores: raw_cosines.clone(),
            local_token_scores: token_channel,
            q_to_d_scores,
            d_to_q_scores,
            token_deposits,
            linked_fact_indices: Vec::new(),
            decomposition_triggered: false,
            local_void_depths: raw_cosines
                .iter()
                .map(|&s| (policy.min_coherence - s).max(0.0))
                .collect(),
            stigmergic_weights: vec![1.0_f32; n_entries],
            boundary_metrics: RecallBoundaryMetrics {
                top1_fact_score: -1.0,
                top2_fact_score: -1.0,
                top1_persona_score: -1.0,
                max_guard_score: -1.0,
                guard_conflict_margin: 0.0,
            },
            gate_decision: RecallGateDecision {
                void_state: VoidBoundaryState::CvoidEmptySpace,
                suppress_fact_injection: true,
            },
            action_decision: StigmergicActionDecision::PassiveSidecar,
            gateway_frame: GatewayFrame::default(),
            swarm_report: SwarmResolutionReport::default(),
            local_channel_preferred: false,
            telemetry: StratumProbeTelemetry::default(),
        };
        let stratum_ctx = StratumContext {
            entries: &self.entries,
            raw_sig: &raw_sig,
            query_sig: &query_sig,
            entry_sigs: &entry_sigs,
            layer_chunk_dim: expected,
            num_q_heads: geom.num_q_heads,
            rotary_dim: geom.rotary_dim,
            policy,
        };
        let (signal, directive) = pipeline.evaluate(&stratum_ctx, initial_signal)?;

        let mut fact_scores: Vec<(&TextAttractorEntry, f32)> = Vec::new();
        let mut persona_scores: Vec<(&TextAttractorEntry, f32)> = Vec::new();
        for (entry, &score) in self.entries.iter().zip(signal.candidate_scores.iter()) {
            match entry.role {
                AttractorRole::Fact => fact_scores.push((entry, score)),
                AttractorRole::Persona => persona_scores.push((entry, score)),
                AttractorRole::Guard => {}
            }
        }
        fact_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        persona_scores.sort_by(|a, b| b.1.total_cmp(&a.1));

        let suppress_facts = directive.repulse_guard
            || directive.suppress_injection
            || signal.gate_decision.suppress_fact_injection;
        let recalled_facts = if suppress_facts {
            Vec::new()
        } else if signal.decomposition_triggered && !signal.linked_fact_indices.is_empty() {
            let max_coalition = policy.top_k_facts.max(signal.linked_fact_indices.len());
            signal
                .linked_fact_indices
                .iter()
                .filter_map(|&(idx, score)| {
                    self.entries
                        .get(idx)
                        .filter(|e| e.role == AttractorRole::Fact)
                        .map(|e| (e, signal.candidate_scores[idx].max(score)))
                })
                .take(max_coalition)
                .collect()
        } else {
            fact_scores
                .into_iter()
                .enumerate()
                .filter(|(rank, (_, s))| *rank == 0 || *s >= policy.min_coherence * 0.75)
                .map(|(_, pair)| pair)
                .take(policy.top_k_facts)
                .collect()
        };
        // Persona grounding operates orthogonally to factual `cvoid` gating and is suppressed only on Guard repulsion.
        let recalled_personas = if directive.repulse_guard {
            Vec::new()
        } else {
            persona_scores
                .into_iter()
                .filter(|(_, s)| *s >= policy.min_coherence * 0.75)
                .take(policy.top_k_personas)
                .collect()
        };

        Ok(DynamicAttractorRecall {
            void_state: signal.gate_decision.void_state,
            top1_fact_score: signal.boundary_metrics.top1_fact_score,
            top2_fact_score: signal.boundary_metrics.top2_fact_score,
            top1_persona_score: signal.boundary_metrics.top1_persona_score,
            min_local_void_depth: directive.min_local_void_depth,
            guard_conflict_margin: signal.boundary_metrics.guard_conflict_margin,
            action_decision: signal.action_decision,
            gateway_frame: signal.gateway_frame,
            telemetry: signal.telemetry,
            recalled_facts,
            recalled_personas,
        })
    }
}

/// Transition telemetry evaluated by [`TimelineCutDetector`] to choose between a P-Frame delta
/// and a full I-Frame keyframe (Scene Cut) in a `.gcso` GOP timeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameTransitionMetrics {
    /// Mean normalized L1 phase change relative to the previous frame (`0.0..=2.0`).
    pub phase_l1_delta: f32,
    /// Current query boundary classification (`CvoidEmptySpace` triggers scene-cut keyframe).
    pub void_state: VoidBoundaryState,
    /// Number of P-Frames elapsed since the last I-Frame.
    pub frames_since_keyframe: usize,
    /// Whether a new attractor cluster crystallized at this step.
    pub new_cluster_spawned: bool,
}

/// Trait abstracting GOP scene-cut / keyframe insertion policy for `.gcso` timelines.
pub trait TimelineCutDetector {
    /// Returns `true` when the timeline should emit a full I-Frame keyframe instead of a P-Frame.
    fn should_emit_keyframe(&self, metrics: &FrameTransitionMetrics) -> bool;
}

/// Default adaptive GOP detector inserting I-Frames on max GOP length, cluster crystallization,
/// `cvoid` boundary transitions, or large phase shifts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveGopCutDetector {
    /// Maximum P-Frames allowed between I-Frames.
    pub max_gop_len: usize,
    /// Phase L1 delta threshold triggering a scene-cut I-Frame.
    pub phase_delta_cut_thresh: f32,
}

impl Default for AdaptiveGopCutDetector {
    fn default() -> Self {
        Self {
            max_gop_len: 16,
            phase_delta_cut_thresh: 0.35,
        }
    }
}

impl TimelineCutDetector for AdaptiveGopCutDetector {
    fn should_emit_keyframe(&self, metrics: &FrameTransitionMetrics) -> bool {
        metrics.frames_since_keyframe >= self.max_gop_len
            || metrics.new_cluster_spawned
            || matches!(metrics.void_state, VoidBoundaryState::CvoidEmptySpace)
            || metrics.phase_l1_delta >= self.phase_delta_cut_thresh
    }
}

/// Lightweight natural-language source record stored in the `.gcso` Subtitle / Source Track for
/// model-agnostic reproducibility and cross-model transcoding.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceTextRecord {
    /// Monotonic frame index (`0..`).
    pub frame_index: u64,
    /// Unix epoch timestamp in seconds.
    pub timestamp_epoch_sec: u64,
    /// Attractor role (`Fact`, `Persona`, or `Guard`).
    pub role: AttractorRole,
    /// Raw natural-language chunk.
    pub text: String,
    /// Base gravitational energy associated with this text chunk.
    pub energy: f32,
}

/// Video-codec-inspired GOP frame payload stored in the `.gcso` PPRC Track.
#[derive(Debug, Clone, PartialEq)]
pub enum GopFrameKind {
    /// Full I-Frame snapshot containing a validated 4-section ZIMMS `.gcso` container and active policy.
    KeyframeI {
        /// Serialized 4-section ZIMMS container bytes (`GcsoFourSectionView`).
        zimms_4section: Vec<u8>,
        /// Calibrated recall policy at this keyframe.
        recall_policy: RecallPolicy,
    },
    /// Incremental P-Frame storing Q7 phase motion delta and energy delta relative to the prior frame.
    DeltaP {
        /// Per-head Q7 phase motion delta (`64` heads).
        delta_q7: [gcso_q7_t; MAX_ATTENTION_HEADS],
        /// Incremental energy change.
        energy_delta: f32,
        /// Target anchor ID updated by this step.
        target_anchor_id: u32,
    },
}

/// Single frame in a [`GcsoDualTrackTimeline`] pairing a Subtitle/Source record with a GOP payload.
#[derive(Debug, Clone, PartialEq)]
pub struct GopTimelineFrame {
    /// Monotonic frame index.
    pub frame_index: u64,
    /// Model-agnostic natural-language source track entry.
    pub source: SourceTextRecord,
    /// Compiled I-Frame or P-Frame payload.
    pub kind: GopFrameKind,
}

/// Reconstructed state returned by zero-forward GOP seek (`AttractorContainerCodec::seek_zero_forward`).
#[derive(Debug, Clone, PartialEq)]
pub struct SeekedContainerState {
    /// Requested target frame index.
    pub target_frame_index: u64,
    /// Frame index of the nearest preceding I-Frame used as reconstruction base.
    pub keyframe_index: u64,
    /// Number of P-Frames accumulated on top of the I-Frame.
    pub p_frames_applied: usize,
    /// Active [`RecallPolicy`] restored from the keyframe.
    pub recall_policy: RecallPolicy,
    /// Reconstructed 64-head Q7 phase state after applying P-Frame deltas.
    pub active_phase_q7: [gcso_q7_t; MAX_ATTENTION_HEADS],
    /// Reconstructed total gravitational energy.
    pub total_energy: f32,
    /// Anchor IDs present in the keyframe's ZIMMS crystal section.
    pub anchor_ids: Vec<u32>,
}

/// Context passed into [`AttractorContainerCodec::record_step`] to keep recording extensible.
pub struct TimelineStepContext<'a> {
    /// Active steering adapter providing model geometry.
    pub adapter: &'a ModelSteeringAdapter,
    /// Active compiled attractor corpus.
    pub corpus: &'a TextAttractorCorpus,
    /// Active calibrated recall policy.
    pub policy: &'a RecallPolicy,
    /// Scene-cut / keyframe detector trait object.
    pub detector: &'a dyn TimelineCutDetector,
}

/// Trait abstracting dual-track `.gcso` container recording, zero-forward GOP seek, and
/// cross-model natural-language track export.
pub trait AttractorContainerCodec {
    /// Appends a timeline frame (emitting an I-Frame or P-Frame per `ctx.detector`).
    /// Returns `Ok(true)` if an I-Frame keyframe was emitted, `Ok(false)` for a P-Frame.
    ///
    /// # Errors
    /// Returns an error if ZIMMS serialization or frame ordering fails.
    fn record_step(
        &mut self,
        ctx: &TimelineStepContext<'_>,
        source: SourceTextRecord,
        metrics: &FrameTransitionMetrics,
    ) -> Result<bool, String>;

    /// Reconstructs attractor phase state at `target_frame_index` by seeking to the nearest
    /// preceding I-Frame and applying subsequent P-Frames with zero LLM forward passes.
    ///
    /// # Errors
    /// Returns an error if `target_frame_index` is out of range or no preceding I-Frame exists.
    fn seek_zero_forward(&self, target_frame_index: u64) -> Result<SeekedContainerState, String>;

    /// Exports the model-agnostic Subtitle / Source Track up to `up_to_frame` (inclusive) for
    /// cross-model re-baking when `EnvProbe` detects a model change.
    fn export_subtitle_track(&self, up_to_frame: Option<u64>) -> Vec<SourceTextRecord>;

    /// Serializes the dual-track container into a portable byte stream.
    fn to_bytes(&self) -> Vec<u8>;
}

/// Abstract encoder projecting a [`SourceTextRecord`] from the `.gcso` Subtitle Track into
/// per-layer `q`-space activations for a target model's geometry during cross-model re-baking.
pub trait SubtitleActivationEncoder {
    /// Encodes `record` into per-layer `q`-space activation vectors (`[layer][rotary_dim]`).
    ///
    /// # Errors
    /// Returns an error if tokenization, forward-pass activation capture, or encoding fails.
    fn encode_subtitle_activations(
        &mut self,
        record: &SourceTextRecord,
    ) -> Result<Vec<Vec<f32>>, String>;
}

/// Dependencies required to resume or re-bake a [`GcsoDualTrackTimeline`] for a target model.
pub struct RebakeContext<'a> {
    /// Target model's steering adapter (defines the target [`RopeGeometry`]).
    pub adapter: &'a ModelSteeringAdapter,
    /// Recall boundary policy applied when constructing re-baked ZIMMS keyframes.
    pub policy: &'a RecallPolicy,
    /// Adaptive GOP scene-cut detector used during re-baking.
    pub detector: &'a dyn TimelineCutDetector,
    /// Optional per-layer reference centroids (`[layer][rotary_dim]`). When empty, the mean of
    /// the encoded subtitle activations is computed automatically as the reference centroid.
    pub centroid_layers: &'a [&'a [f32]],
    /// Optional inclusive upper frame limit for partial re-baking.
    pub up_to_frame: Option<u64>,
}

/// Outcome of [`GcsoDualTrackTimeline::resume_or_rebake`] after `EnvProbe` fingerprint checking.
#[derive(Debug, Clone)]
pub enum TimelineResumeOutcome {
    /// Target model geometry matched `model_fingerprint`; state was restored via zero-forward seek.
    FastResumed {
        /// Reconstructed state at the requested target frame.
        seek_state: SeekedContainerState,
    },
    /// Target model geometry differed; Subtitle Track records were re-baked into the new space.
    Rebaked {
        /// Newly re-baked dual-track timeline bound to `target_fingerprint`.
        timeline: GcsoDualTrackTimeline,
        /// Re-baked dynamic attractor corpus aligned with the target model.
        corpus: TextAttractorCorpus,
        /// Reconstructed state at the requested target frame on the re-baked timeline.
        seek_state: SeekedContainerState,
        /// Fingerprint recorded in the source container.
        source_fingerprint: u64,
        /// Fingerprint of the target model's [`RopeGeometry`].
        target_fingerprint: u64,
        /// Number of subtitle records re-baked into the new timeline.
        rebaked_frames: usize,
    },
}

const GCSO_DUAL_MAGIC: &[u8; 8] = b"GCSODUAL";
const GCSO_DUAL_VERSION: u32 = 1;

/// Dual-Track (Subtitle Natural-Language Track + PPRC Video-Style GOP Track) `.gcso` container.
#[derive(Debug, Clone, PartialEq)]
pub struct GcsoDualTrackTimeline {
    model_fingerprint: u64,
    frames: Vec<GopTimelineFrame>,
    seek_index: Vec<usize>,
}

impl GcsoDualTrackTimeline {
    /// Computes a deterministic 64-bit fingerprint for a [`RopeGeometry`] so `EnvProbe` can detect
    /// model switches requiring subtitle-track transcoding.
    #[must_use]
    pub fn geometry_fingerprint(geom: &RopeGeometry) -> u64 {
        let mut h = 0xCBF2_9CE4_8422_2325_u64;
        for val in [
            geom.num_layers as u64,
            geom.num_q_heads as u64,
            geom.num_kv_heads as u64,
            geom.head_dim as u64,
            geom.rotary_dim as u64,
            geom.layer_rope_mask,
            match geom.layout {
                RopeLayout::Interleaved => 1,
                RopeLayout::HalfSplit => 2,
            },
        ] {
            h ^= val;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
        h
    }

    /// Creates an empty [`GcsoDualTrackTimeline`] bound to `model_fingerprint`.
    #[must_use]
    pub const fn new(model_fingerprint: u64) -> Self {
        Self {
            model_fingerprint,
            frames: Vec::new(),
            seek_index: Vec::new(),
        }
    }

    /// Returns the geometry fingerprint this compiled GOP track was baked for.
    #[must_use]
    pub const fn model_fingerprint(&self) -> u64 {
        self.model_fingerprint
    }

    /// Returns all recorded timeline frames.
    #[must_use]
    pub fn frames(&self) -> &[GopTimelineFrame] {
        &self.frames
    }

    /// Returns frame indices of all I-Frame keyframes in the seek table.
    #[must_use]
    pub fn keyframe_positions(&self) -> &[usize] {
        &self.seek_index
    }

    fn slot_for_role(role: AttractorRole) -> u8 {
        match role {
            AttractorRole::Fact => ATTRACTOR_SLOT_KNOWLEDGE,
            AttractorRole::Persona => ATTRACTOR_SLOT_PERSONA,
            AttractorRole::Guard => ATTRACTOR_SLOT_REPULSION,
        }
    }

    fn build_zimms_snapshot(
        corpus: &TextAttractorCorpus,
        timestamp_epoch_sec: u64,
    ) -> Result<Vec<u8>, String> {
        let mut crystals = Vec::with_capacity(corpus.entries().len().max(1));
        for entry in corpus.entries() {
            let mut rec = CrystallizedAttractorRecord::new_empty(
                entry.anchor_id,
                Self::slot_for_role(entry.role),
                0,
            );
            rec.base_energy = entry.energy;
            rec.phase_gravity_mass = entry.energy;
            rec.sample_count = 1;
            if let Some(first_layer) = entry.per_layer_q7.last() {
                rec.phase_profile = *first_layer;
            }
            rec.checksum_crc32 = 0;
            crystals.push(rec);
        }
        if crystals.is_empty() {
            let empty_rec = CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_KNOWLEDGE, 0);
            crystals.push(empty_rec);
        }
        let trail = gcso_pointer_trail_t::default();
        let trails = [trail];
        let req_len = GcsoFourSectionView::required_byte_len_with_reports(
            crystals.len(),
            trails.len(),
            false,
            false,
            0,
        )
        .map_err(|e| format!("required_byte_len error: {e}"))?;
        let mut aligned = vec![0_u8; req_len + 128];
        let off = aligned.as_ptr().align_offset(128);
        let written = GcsoFourSectionView::write_four_section_with_reports_to_bytes(
            &crystals,
            &trails,
            None,
            None,
            &[],
            timestamp_epoch_sec,
            &mut aligned[off..off + req_len],
        )
        .map_err(|e| format!("write_four_section error: {e}"))?;
        Ok(aligned[off..off + written].to_vec())
    }

    fn latest_phase_q7(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        if let Some(last) = self.frames.last() {
            if let Ok(state) = self.seek_zero_forward(last.frame_index) {
                return state.active_phase_q7;
            }
        }
        [0_i8; MAX_ATTENTION_HEADS]
    }

    /// Writes the serialized `.gcso` dual-track container to `path`, returning the bytes written.
    ///
    /// # Errors
    /// Returns an error if creating parent directories or writing the file fails.
    pub fn save_to_path(&self, path: impl AsRef<std::path::Path>) -> Result<usize, String> {
        let p = path.as_ref();
        if let Some(parent) = p.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "failed to create parent directory {}: {e}",
                    parent.display()
                )
            })?;
        }
        let bytes = self.to_bytes();
        std::fs::write(p, &bytes)
            .map_err(|e| format!("failed to write .gcso container {}: {e}", p.display()))?;
        Ok(bytes.len())
    }

    /// Loads and validates a `.gcso` dual-track container from `path`.
    ///
    /// # Errors
    /// Returns an error if reading the file or parsing the container fails.
    pub fn load_from_path(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let p = path.as_ref();
        let bytes = std::fs::read(p)
            .map_err(|e| format!("failed to read .gcso container {}: {e}", p.display()))?;
        Self::from_bytes(&bytes)
    }

    /// Compiles a sequence of model-agnostic [`SourceTextRecord`] entries into a new
    /// [`GcsoDualTrackTimeline`] and [`TextAttractorCorpus`] for `ctx.adapter.geometry()`
    /// using `encoder`.
    ///
    /// # Errors
    /// Returns an error if `subtitles` is empty, activation encoding fails, or GOP recording fails.
    pub fn bake_from_subtitle_records<E: SubtitleActivationEncoder + ?Sized>(
        subtitles: &[SourceTextRecord],
        ctx: &RebakeContext<'_>,
        encoder: &mut E,
    ) -> Result<(Self, TextAttractorCorpus), String> {
        if subtitles.is_empty() {
            return Err("cannot bake an empty subtitle track".to_string());
        }

        let target_fp = Self::geometry_fingerprint(ctx.adapter.geometry());
        let mut encoded_records = Vec::with_capacity(subtitles.len());
        for record in subtitles {
            encoded_records.push(encoder.encode_subtitle_activations(record)?);
        }

        let num_layers = encoded_records[0].len();
        let auto_centroids: Vec<Vec<f32>> = if ctx.centroid_layers.is_empty() {
            let mut sums: Vec<Vec<f32>> = encoded_records[0]
                .iter()
                .map(|layer| vec![0.0_f32; layer.len()])
                .collect();
            for rec_layers in &encoded_records {
                if rec_layers.len() != num_layers {
                    return Err(
                        "inconsistent layer count from SubtitleActivationEncoder".to_string()
                    );
                }
                for (l, src) in rec_layers.iter().enumerate() {
                    if src.len() != sums[l].len() {
                        return Err(format!(
                            "inconsistent rotary dimension at layer {l}: {} vs {}",
                            src.len(),
                            sums[l].len()
                        ));
                    }
                    for (dst, &val) in sums[l].iter_mut().zip(src.iter()) {
                        *dst += val;
                    }
                }
            }
            let inv_n = 1.0_f32 / (encoded_records.len() as f32);
            for layer_sum in &mut sums {
                for val in layer_sum.iter_mut() {
                    *val *= inv_n;
                }
            }
            sums
        } else {
            Vec::new()
        };

        let auto_refs: Vec<&[f32]> = auto_centroids.iter().map(Vec::as_slice).collect();
        let ref_layers: &[&[f32]] = if ctx.centroid_layers.is_empty() {
            &auto_refs
        } else {
            ctx.centroid_layers
        };

        let mut rebaked_corpus = TextAttractorCorpus::new();
        for (record, q_layers) in subtitles.iter().zip(encoded_records.iter()) {
            let q_refs: Vec<&[f32]> = q_layers.iter().map(Vec::as_slice).collect();
            rebaked_corpus.ingest_text_activations(
                ctx.adapter,
                record.role,
                &record.text,
                &q_refs,
                ref_layers,
                record.energy,
            )?;
        }
        rebaked_corpus.fit_common_mode_projection();

        let mut rebaked_timeline = Self::new(target_fp);
        let mut frames_since_keyframe = 0_usize;
        let step_ctx = TimelineStepContext {
            adapter: ctx.adapter,
            corpus: &rebaked_corpus,
            policy: ctx.policy,
            detector: ctx.detector,
        };
        for (idx, record) in subtitles.iter().enumerate() {
            let scene_cut = idx > 0 && record.role != subtitles[idx - 1].role;
            let emitted_i = rebaked_timeline.record_step(
                &step_ctx,
                record.clone(),
                &FrameTransitionMetrics {
                    phase_l1_delta: if idx == 0 { 0.0 } else { 0.08 },
                    void_state: if scene_cut {
                        VoidBoundaryState::CvoidEmptySpace
                    } else {
                        VoidBoundaryState::InsideBasin
                    },
                    frames_since_keyframe,
                    new_cluster_spawned: scene_cut,
                },
            )?;
            frames_since_keyframe = if emitted_i {
                0
            } else {
                frames_since_keyframe + 1
            };
        }

        Ok((rebaked_timeline, rebaked_corpus))
    }

    /// Checks `EnvProbe` compatibility against `ctx.adapter.geometry()` and either performs a
    /// zero-forward GOP seek (`FastResumed`) or automatically re-bakes the Subtitle Track through
    /// `encoder` into the target model's geometry (`Rebaked`).
    ///
    /// # Errors
    /// Returns an error if seeking, encoding subtitle activations, or recording re-baked frames fails.
    pub fn resume_or_rebake<E: SubtitleActivationEncoder + ?Sized>(
        &self,
        ctx: &RebakeContext<'_>,
        encoder: &mut E,
        target_frame: u64,
    ) -> Result<TimelineResumeOutcome, String> {
        let target_fp = Self::geometry_fingerprint(ctx.adapter.geometry());
        if self.model_fingerprint == target_fp {
            let seek_state = self.seek_zero_forward(target_frame)?;
            return Ok(TimelineResumeOutcome::FastResumed { seek_state });
        }

        let subtitles = self.export_subtitle_track(ctx.up_to_frame);
        let (rebaked_timeline, rebaked_corpus) =
            Self::bake_from_subtitle_records(&subtitles, ctx, encoder)?;

        let effective_target = subtitles
            .iter()
            .filter(|r| r.frame_index <= target_frame)
            .map(|r| r.frame_index)
            .next_back()
            .unwrap_or(subtitles[0].frame_index);
        let seek_state = rebaked_timeline.seek_zero_forward(effective_target)?;

        Ok(TimelineResumeOutcome::Rebaked {
            timeline: rebaked_timeline,
            corpus: rebaked_corpus,
            seek_state,
            source_fingerprint: self.model_fingerprint,
            target_fingerprint: target_fp,
            rebaked_frames: subtitles.len(),
        })
    }

    /// Deserializes a portable [`GcsoDualTrackTimeline`] byte stream produced by [`Self::to_bytes`].
    ///
    /// # Errors
    /// Returns an error if magic, version, or frame payload lengths are invalid.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut cursor = 0_usize;
        let read_exact = |cur: &mut usize, n: usize| -> Result<&[u8], String> {
            let end = cur.checked_add(n).ok_or_else(|| "overflow".to_string())?;
            if end > bytes.len() {
                return Err("unexpected EOF in GCSO dual-track container".to_string());
            }
            let slice = &bytes[*cur..end];
            *cur = end;
            Ok(slice)
        };
        let magic = read_exact(&mut cursor, 8)?;
        if magic != GCSO_DUAL_MAGIC {
            return Err("invalid GCSO dual-track magic".to_string());
        }
        let ver = u32::from_le_bytes(read_exact(&mut cursor, 4)?.try_into().map_err(|_| "ver")?);
        if ver != GCSO_DUAL_VERSION {
            return Err(format!("unsupported GCSO dual-track version: {ver}"));
        }
        let model_fingerprint =
            u64::from_le_bytes(read_exact(&mut cursor, 8)?.try_into().map_err(|_| "fp")?);
        let frame_count =
            u32::from_le_bytes(read_exact(&mut cursor, 4)?.try_into().map_err(|_| "cnt")?) as usize;

        let mut frames = Vec::with_capacity(frame_count);
        let mut seek_index = Vec::new();
        for idx in 0..frame_count {
            let frame_index =
                u64::from_le_bytes(read_exact(&mut cursor, 8)?.try_into().map_err(|_| "fidx")?);
            let ts = u64::from_le_bytes(read_exact(&mut cursor, 8)?.try_into().map_err(|_| "ts")?);
            let role = match read_exact(&mut cursor, 1)?[0] {
                0 => AttractorRole::Fact,
                1 => AttractorRole::Persona,
                2 => AttractorRole::Guard,
                r => return Err(format!("unknown AttractorRole byte: {r}")),
            };
            let energy =
                f32::from_le_bytes(read_exact(&mut cursor, 4)?.try_into().map_err(|_| "en")?);
            let text_len =
                u32::from_le_bytes(read_exact(&mut cursor, 4)?.try_into().map_err(|_| "tlen")?)
                    as usize;
            let text = std::str::from_utf8(read_exact(&mut cursor, text_len)?)
                .map_err(|e| format!("invalid UTF-8 in source track: {e}"))?
                .to_string();
            let kind_tag = read_exact(&mut cursor, 1)?[0];
            let kind = match kind_tag {
                0 => {
                    let top_k_facts = u32::from_le_bytes(
                        read_exact(&mut cursor, 4)?.try_into().map_err(|_| "tkf")?,
                    ) as usize;
                    let top_k_personas = u32::from_le_bytes(
                        read_exact(&mut cursor, 4)?.try_into().map_err(|_| "tkp")?,
                    ) as usize;
                    let min_coherence = f32::from_le_bytes(
                        read_exact(&mut cursor, 4)?.try_into().map_err(|_| "mc")?,
                    );
                    let boundary_margin = f32::from_le_bytes(
                        read_exact(&mut cursor, 4)?.try_into().map_err(|_| "bm")?,
                    );
                    let guard_suppress_margin = f32::from_le_bytes(
                        read_exact(&mut cursor, 4)?.try_into().map_err(|_| "gsm")?,
                    );
                    let z_len = u32::from_le_bytes(
                        read_exact(&mut cursor, 4)?.try_into().map_err(|_| "zlen")?,
                    ) as usize;
                    let zimms_4section = read_exact(&mut cursor, z_len)?.to_vec();
                    seek_index.push(idx);
                    GopFrameKind::KeyframeI {
                        zimms_4section,
                        recall_policy: RecallPolicy {
                            top_k_facts,
                            top_k_personas,
                            min_coherence,
                            boundary_margin,
                            guard_suppress_margin,
                        },
                    }
                }
                1 => {
                    let q7_bytes = read_exact(&mut cursor, MAX_ATTENTION_HEADS)?;
                    let mut delta_q7 = [0_i8; MAX_ATTENTION_HEADS];
                    for (dst, &b) in delta_q7.iter_mut().zip(q7_bytes.iter()) {
                        *dst = b as i8;
                    }
                    let energy_delta = f32::from_le_bytes(
                        read_exact(&mut cursor, 4)?.try_into().map_err(|_| "ed")?,
                    );
                    let target_anchor_id = u32::from_le_bytes(
                        read_exact(&mut cursor, 4)?.try_into().map_err(|_| "aid")?,
                    );
                    GopFrameKind::DeltaP {
                        delta_q7,
                        energy_delta,
                        target_anchor_id,
                    }
                }
                other => return Err(format!("unknown GopFrameKind tag: {other}")),
            };
            frames.push(GopTimelineFrame {
                frame_index,
                source: SourceTextRecord {
                    frame_index,
                    timestamp_epoch_sec: ts,
                    role,
                    text,
                    energy,
                },
                kind,
            });
        }
        Ok(Self {
            model_fingerprint,
            frames,
            seek_index,
        })
    }
}

impl AttractorContainerCodec for GcsoDualTrackTimeline {
    fn record_step(
        &mut self,
        ctx: &TimelineStepContext<'_>,
        source: SourceTextRecord,
        metrics: &FrameTransitionMetrics,
    ) -> Result<bool, String> {
        let frame_index = source.frame_index;
        let emit_i = self.frames.is_empty() || ctx.detector.should_emit_keyframe(metrics);
        let fp = Self::geometry_fingerprint(ctx.adapter.geometry());
        if self.frames.is_empty() {
            self.model_fingerprint = fp;
        }
        if emit_i {
            let zimms_4section =
                Self::build_zimms_snapshot(ctx.corpus, source.timestamp_epoch_sec)?;
            let idx = self.frames.len();
            self.seek_index.push(idx);
            self.frames.push(GopTimelineFrame {
                frame_index,
                source,
                kind: GopFrameKind::KeyframeI {
                    zimms_4section,
                    recall_policy: *ctx.policy,
                },
            });
            Ok(true)
        } else {
            let prev_q7 = self.latest_phase_q7();
            let target_entry = ctx.corpus.entries().last();
            let curr_q7 = target_entry
                .and_then(|e| e.per_layer_q7.last().copied())
                .unwrap_or([0_i8; MAX_ATTENTION_HEADS]);
            let mut delta_q7 = [0_i8; MAX_ATTENTION_HEADS];
            for h in 0..MAX_ATTENTION_HEADS {
                delta_q7[h] = curr_q7[h].saturating_sub(prev_q7[h]);
            }
            let target_anchor_id = target_entry.map_or(0, |e| e.anchor_id);
            let energy_delta = source.energy;
            self.frames.push(GopTimelineFrame {
                frame_index,
                source,
                kind: GopFrameKind::DeltaP {
                    delta_q7,
                    energy_delta,
                    target_anchor_id,
                },
            });
            Ok(false)
        }
    }

    fn seek_zero_forward(&self, target_frame_index: u64) -> Result<SeekedContainerState, String> {
        let target_pos = self
            .frames
            .iter()
            .position(|f| f.frame_index == target_frame_index)
            .ok_or_else(|| format!("frame_index {target_frame_index} not found"))?;
        let key_pos = self
            .seek_index
            .iter()
            .copied()
            .rfind(|&pos| pos <= target_pos)
            .ok_or_else(|| format!("no preceding I-Frame for target {target_frame_index}"))?;

        let key_frame = &self.frames[key_pos];
        let (zimms_bytes, recall_policy) = match &key_frame.kind {
            GopFrameKind::KeyframeI {
                zimms_4section,
                recall_policy,
            } => (zimms_4section, *recall_policy),
            GopFrameKind::DeltaP { .. } => {
                return Err("corrupt seek index pointing to DeltaP".to_string())
            }
        };

        let mut aligned = vec![0_u8; zimms_bytes.len() + 128];
        let off = aligned.as_ptr().align_offset(128);
        aligned[off..off + zimms_bytes.len()].copy_from_slice(zimms_bytes);
        let view =
            GcsoFourSectionView::from_zimms_bytes_zero_copy(&aligned[off..off + zimms_bytes.len()])
                .map_err(|e| format!("ZIMMS zero-copy view validation failed: {e}"))?;

        let mut active_phase_q7 = view
            .crystals
            .last()
            .map_or([0_i8; MAX_ATTENTION_HEADS], |c| c.phase_profile);
        let mut total_energy: f32 = view.crystals.iter().map(|c| c.base_energy).sum();
        let anchor_ids: Vec<u32> = view.crystals.iter().map(|c| c.anchor_id).collect();

        let mut p_frames_applied = 0_usize;
        for f in &self.frames[(key_pos + 1)..=target_pos] {
            if let GopFrameKind::DeltaP {
                delta_q7,
                energy_delta,
                ..
            } = &f.kind
            {
                for h in 0..MAX_ATTENTION_HEADS {
                    active_phase_q7[h] = active_phase_q7[h].saturating_add(delta_q7[h]);
                }
                total_energy += *energy_delta;
                p_frames_applied += 1;
            }
        }

        Ok(SeekedContainerState {
            target_frame_index,
            keyframe_index: key_frame.frame_index,
            p_frames_applied,
            recall_policy,
            active_phase_q7,
            total_energy,
            anchor_ids,
        })
    }

    fn export_subtitle_track(&self, up_to_frame: Option<u64>) -> Vec<SourceTextRecord> {
        self.frames
            .iter()
            .filter(|f| up_to_frame.is_none_or(|limit| f.frame_index <= limit))
            .map(|f| f.source.clone())
            .collect()
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(GCSO_DUAL_MAGIC);
        out.extend_from_slice(&GCSO_DUAL_VERSION.to_le_bytes());
        out.extend_from_slice(&self.model_fingerprint.to_le_bytes());
        out.extend_from_slice(&(self.frames.len() as u32).to_le_bytes());
        for f in &self.frames {
            out.extend_from_slice(&f.frame_index.to_le_bytes());
            out.extend_from_slice(&f.source.timestamp_epoch_sec.to_le_bytes());
            let role_byte = match f.source.role {
                AttractorRole::Fact => 0_u8,
                AttractorRole::Persona => 1_u8,
                AttractorRole::Guard => 2_u8,
            };
            out.push(role_byte);
            out.extend_from_slice(&f.source.energy.to_le_bytes());
            let tb = f.source.text.as_bytes();
            out.extend_from_slice(&(tb.len() as u32).to_le_bytes());
            out.extend_from_slice(tb);
            match &f.kind {
                GopFrameKind::KeyframeI {
                    zimms_4section,
                    recall_policy,
                } => {
                    out.push(0_u8);
                    out.extend_from_slice(&(recall_policy.top_k_facts as u32).to_le_bytes());
                    out.extend_from_slice(&(recall_policy.top_k_personas as u32).to_le_bytes());
                    out.extend_from_slice(&recall_policy.min_coherence.to_le_bytes());
                    out.extend_from_slice(&recall_policy.boundary_margin.to_le_bytes());
                    out.extend_from_slice(&recall_policy.guard_suppress_margin.to_le_bytes());
                    out.extend_from_slice(&(zimms_4section.len() as u32).to_le_bytes());
                    out.extend_from_slice(zimms_4section);
                }
                GopFrameKind::DeltaP {
                    delta_q7,
                    energy_delta,
                    target_anchor_id,
                } => {
                    out.push(1_u8);
                    for &q in delta_q7 {
                        out.push(q as u8);
                    }
                    out.extend_from_slice(&energy_delta.to_le_bytes());
                    out.extend_from_slice(&target_anchor_id.to_le_bytes());
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phi4_mini_preset_validates_and_skips_nope_layers() {
        let mut geom = RopeGeometry::phi4_mini();
        assert!(geom.validate().is_ok());
        assert_eq!(geom.q_rot_len(), 24 * 96);

        geom.layer_rope_mask &= !(1 << 5);
        assert!(geom.is_rope_layer(4));
        assert!(!geom.is_rope_layer(5));
    }

    #[test]
    fn adapter_bakes_and_steers_half_split_and_interleaved() {
        for layout in [RopeLayout::HalfSplit, RopeLayout::Interleaved] {
            let mut geom = RopeGeometry::phi4_mini();
            geom.layout = layout;
            geom.layer_rope_mask &= !(1 << 1); // Layer 1 is NoPE

            let mut adapter = ModelSteeringAdapter::new(geom).expect("adapter init");
            assert!(adapter.residual_adapter_mut().validate().is_ok());
            let _ = adapter.entropy_controller_mut();

            let len = adapter.geometry().q_rot_len();
            let neg: Vec<f32> = (0..len).map(|i| ((i % 13) as f32) * 0.1 - 0.5).collect();
            let pos: Vec<f32> = (0..len).map(|i| ((i % 7) as f32) * 0.15 - 0.3).collect();

            let q7 = adapter
                .bake_layer_procrustes_q7(&neg, &pos)
                .expect("bake should succeed");
            assert_eq!(q7.len(), 24);

            let mut q_layer0 = neg.clone();
            adapter
                .steer_layer_q_rot(0, &mut q_layer0, &q7, 0.10)
                .expect("steer layer 0");

            // NoPE layer 1 must remain untouched.
            let mut q_layer1 = neg.clone();
            adapter
                .steer_layer_q_rot(1, &mut q_layer1, &q7, 0.10)
                .expect("skip layer 1");
            assert_eq!(q_layer1, neg);
        }
    }

    #[test]
    fn adapter_steps_edbc_and_applies_srl_residual() {
        let mut adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
        let mut logits = vec![4.0, 2.0, 1.0, 0.5];
        let state = adapter
            .step_edbc_logits(&mut logits)
            .expect("edbc step should succeed");
        assert!(state.sample_temperature > 0.0);

        let mut hidden = vec![1.0, 0.5, -0.5, 0.25];
        let u = vec![0.5, 0.5, 0.5, 0.5];
        let v = vec![1.0, 0.0, 0.0, 0.0];
        adapter
            .apply_srl_hidden_fused(&mut hidden, &u, &v, 0.5)
            .expect("srl fused should succeed");
        assert!((hidden[0] - 1.25).abs() < 1e-5);
    }

    #[test]
    fn adapter_steers_via_gateway_trait_composition() {
        use liminika_core::traits::gateway::{
            ObjectiveFact, SubjectivePersona, CAUSAL_AXIS_OBJECTIVE_FACT,
            CAUSAL_AXIS_SUBJECTIVE_PERSONA,
        };

        let mut adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
        let len = adapter.geometry().q_rot_len();
        let orig: Vec<f32> = (0..len).map(|i| ((i % 11) as f32) * 0.1 - 0.4).collect();
        let mut q_rot = orig.clone();
        let mut logits = vec![2.0, 1.0, 0.5, -0.5];
        let mut trail = gcso_pointer_trail_t::default();

        let fact = ObjectiveFact::new([20i8; 64], 2.5, 0, 42);
        let persona = SubjectivePersona::new([-12i8; 64], 1.8, 0, 7);
        let composite_signal = (fact, persona);

        let (frame, report) = adapter
            .steer_layer_with_gateway(
                0,
                &composite_signal,
                &mut trail,
                128,
                &mut q_rot,
                Some(&mut logits),
            )
            .expect("gateway steering should succeed")
            .expect("layer 0 is active");

        assert!(frame.ripa_clamp_rad > 0.0);
        assert_ne!(report.provenance_bits & CAUSAL_AXIS_OBJECTIVE_FACT, 0);
        assert_ne!(report.provenance_bits & CAUSAL_AXIS_SUBJECTIVE_PERSONA, 0);
        assert_ne!(q_rot, orig);
        assert_eq!(trail.step_count, 1);

        let mut hidden = vec![10.0, 0.0, 0.0, 0.0];
        let u = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![1.0, 0.0, 0.0, 0.0];
        let (_, gain) = adapter
            .apply_srl_with_gateway(&composite_signal, &mut hidden, &u, &v)
            .expect("gateway srl should succeed");
        assert!(gain > 0.0);
        assert!(hidden[0] > 10.0);
    }

    #[test]
    fn adapter_resonates_arbitrary_slice_without_hardcoded_count() {
        use liminika_core::traits::gateway::{ObjectiveFact, ResonantSignalSlice};

        let mut adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
        let len = adapter.geometry().q_rot_len();
        let mut q_rot: Vec<f32> = (0..len).map(|i| ((i % 9) as f32) * 0.1 - 0.3).collect();
        let mut trail = gcso_pointer_trail_t::default();

        // Arbitrary number of candidates (e.g. 3, not tied to 8) matching 24 active Q heads.
        let mut a1 = [0i8; 64];
        a1[..24].fill(60);
        let mut a2 = [0i8; 64];
        a2[..24].fill(-60);
        let signals = [
            ObjectiveFact::new(a1, 2.0, 0, 10),
            ObjectiveFact::new(a2, 2.0, 0, 20),
            ObjectiveFact::new([0i8; 64], 2.0, 0, 30),
        ];
        let probe = [60i8; 24];
        let resonant = ResonantSignalSlice::from_q7_probe(&signals, &probe);

        let (frame, _) = adapter
            .steer_layer_with_gateway(0, &resonant, &mut trail, 128, &mut q_rot, None)
            .expect("resonant steering should succeed")
            .expect("layer 0 active");

        assert_eq!(frame.target_anchor_id, 10);
        assert!(frame.phase_deltas[0] > 30);
    }

    #[test]
    fn adapter_steers_via_action_hub_and_builds_stigmergic_density() {
        use liminika_core::traits::gateway::{ObjectiveFact, ResonantSignalSlice};

        let mut adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
        let len = adapter.geometry().q_rot_len();
        let mut q_rot: Vec<f32> = (0..len).map(|i| ((i % 7) as f32) * 0.1 - 0.25).collect();
        let orig = q_rot.clone();
        let mut trail = gcso_pointer_trail_t::default();

        let mut a1 = [0i8; 64];
        a1[..24].fill(55);
        let signals = [ObjectiveFact::new(a1, 2.5, 0, 77)];
        let probe = [55i8; 24];
        let resonant = ResonantSignalSlice::from_q7_probe(&signals, &probe);

        let (frame, _) = adapter
            .steer_layer_with_action_hub(0, &resonant, &mut trail, 128, &mut q_rot, None)
            .expect("action_hub steering should succeed")
            .expect("layer 0 active");

        assert_eq!(frame.target_anchor_id, 77);
        assert_ne!(q_rot, orig);
        assert_eq!(adapter.action_hub().step_counter, 1);
        assert!(trail.stigmergic_density > 0.0);
    }

    #[test]
    fn action_hub_orchestrator_gates_rope_and_srl_by_stigmergic_density() {
        use liminika_core::traits::gateway::{ObjectiveFact, ResonantSignalSlice};

        let mut adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
        let len = adapter.geometry().q_rot_len();
        let mut trail = gcso_pointer_trail_t::default();

        let mut a1 = [0i8; 64];
        a1[..24].fill(55);
        let signals = [ObjectiveFact::new(a1, 2.5, 0, 77)];
        let probe = [55i8; 24];
        let resonant = ResonantSignalSlice::from_q7_probe(&signals, &probe);

        let mut hidden = vec![10.0, 0.0, 0.0, 0.0];
        let u = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![1.0, 0.0, 0.0, 0.0];

        // First step from zero density crosses express_steer (0.35) but not srl_inject (1.50).
        let mut q_rot1: Vec<f32> = (0..len).map(|i| ((i % 7) as f32) * 0.1 - 0.25).collect();
        let orig1 = q_rot1.clone();
        let (_, _, d1) = adapter
            .orchestrate_layer_rope_with_action_hub(
                0,
                &resonant,
                &mut trail,
                None,
                128,
                &mut q_rot1,
                None,
            )
            .expect("step 1")
            .expect("active");
        assert_eq!(d1, StigmergicActionDecision::ExpressPhaseSteer);
        assert_ne!(q_rot1, orig1);
        assert!(adapter
            .orchestrate_layer_srl_with_action_hub(&resonant, &mut hidden, &u, &v)
            .expect("srl gate")
            .is_none());

        // Repeated resonant steps accumulate density above srl_inject_density_thresh (1.50).
        let mut final_decision = d1;
        for _ in 0..8 {
            let mut q_rot_k: Vec<f32> = (0..len).map(|i| ((i % 7) as f32) * 0.1 - 0.25).collect();
            let (_, _, dk) = adapter
                .orchestrate_layer_rope_with_action_hub(
                    0,
                    &resonant,
                    &mut trail,
                    None,
                    128,
                    &mut q_rot_k,
                    None,
                )
                .expect("step k")
                .expect("active");
            final_decision = dk;
        }
        assert!(matches!(
            final_decision,
            StigmergicActionDecision::FullOrchestrationSrl
                | StigmergicActionDecision::CrystallizeMacroBasin
        ));
        let srl_res = adapter
            .orchestrate_layer_srl_with_action_hub(&resonant, &mut hidden, &u, &v)
            .expect("srl gate");
        assert!(srl_res.is_some());
        assert!(hidden[0] > 10.0);
    }

    #[test]
    fn cold_path_probe_calibrates_thresholds_and_rebinds_geometry() {
        let geom = RopeGeometry::phi4_mini();
        let mut adapter = ModelSteeringAdapter::new(geom).expect("phi4 geometry valid");
        adapter.set_layer_rope_mask(0xFFFF_0000);
        let q_len = adapter.geometry().q_rot_len();

        let ref_vec = vec![0.0_f32; q_len];
        let probe_vec: Vec<f32> = (0..q_len)
            .map(|i| if (i & 1) == 0 { 0.85 } else { 0.45 })
            .collect();
        let samples: Vec<&[f32]> = vec![probe_vec.as_slice(); 32];
        let refs: Vec<&[f32]> = vec![ref_vec.as_slice(); 32];

        let baseline = adapter
            .calibrate_manifold_probe(&samples, &refs, &[0.42, 0.46])
            .expect("calibration probe should succeed");
        assert_eq!(baseline.probe_epoch, 1);
        assert_eq!(baseline.active_heads, 24);
        assert_eq!(baseline.steered_layers, 16);
        assert!(baseline.baseline_entropy > 0.40);

        let low_load_srl = adapter.action_hub().thresholds.srl_inject_density_thresh;
        adapter.update_environment_telemetry(0.85, 0.70, 0x0F);
        let high_load_srl = adapter.action_hub().thresholds.srl_inject_density_thresh;
        assert!(high_load_srl > low_load_srl);

        let mut alt_geom = RopeGeometry::phi4_mini();
        alt_geom.num_q_heads = 32;
        alt_geom.head_dim = 128;
        alt_geom.rotary_dim = 128;
        alt_geom.layer_rope_mask = 0xFFFF_FFFF;
        let new_epoch = adapter
            .rebind_geometry_cold_path(alt_geom)
            .expect("cold-path geometry rebind should succeed");
        assert_eq!(new_epoch, 2);
        assert_eq!(adapter.probe_baseline().active_heads, 32);
        assert_eq!(adapter.probe_baseline().steered_layers, 32);
    }

    #[test]
    fn text_attractor_corpus_recalls_basin_and_detects_cvoid_boundary() {
        let geom = RopeGeometry::phi4_mini();
        let mut adapter = ModelSteeringAdapter::new(geom).expect("phi4 geometry valid");
        adapter.set_layer_rope_mask(0xFFFF_0000);
        let q_len = adapter.geometry().q_rot_len();

        let ref_vec = vec![0.0_f32; q_len];
        let half = adapter.geometry().rotary_dim / 2;
        let rot = adapter.geometry().rotary_dim;
        let mut fact_a = vec![0.0_f32; q_len];
        let mut fact_b = vec![0.0_f32; q_len];
        let mut void_q = vec![0.0_f32; q_len];
        for h in 0..adapter.geometry().num_q_heads {
            let base = h * rot;
            for i in 0..half {
                let (sa, ca) = (0.35 + 0.02 * h as f32).sin_cos();
                let (sb, cb) = (-0.45 - 0.01 * h as f32).sin_cos();
                let (sv, cv) = (2.80 - 0.02 * h as f32).sin_cos();
                fact_a[base + i] = ca;
                fact_a[base + half + i] = sa;
                fact_b[base + i] = cb;
                fact_b[base + half + i] = sb;
                void_q[base + i] = cv;
                void_q[base + half + i] = sv;
            }
        }

        let refs: Vec<&[f32]> = vec![ref_vec.as_slice(); 32];
        let qa_slices: Vec<&[f32]> = vec![fact_a.as_slice(); 32];
        let qb_slices: Vec<&[f32]> = vec![fact_b.as_slice(); 32];
        let qv_slices: Vec<&[f32]> = vec![void_q.as_slice(); 32];

        let mut corpus = TextAttractorCorpus::new();
        corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Fact A",
                &qa_slices,
                &refs,
                2.5,
            )
            .expect("ingest A");
        corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Fact B",
                &qb_slices,
                &refs,
                2.5,
            )
            .expect("ingest B");

        let policy = RecallPolicy {
            top_k_facts: 1,
            top_k_personas: 1,
            min_coherence: 0.50,
            boundary_margin: 0.05,
            ..RecallPolicy::default()
        };
        let recall_a = corpus
            .recall_for_query(&adapter, &qa_slices, &refs, &policy)
            .expect("recall A");
        assert_eq!(recall_a.void_state, VoidBoundaryState::InsideBasin);
        assert_eq!(recall_a.recalled_facts[0].0.text, "Fact A");

        let recall_void = corpus
            .recall_for_query(&adapter, &qv_slices, &refs, &policy)
            .expect("recall void");
        assert_eq!(recall_void.void_state, VoidBoundaryState::CvoidEmptySpace);
        assert!(recall_void.recalled_facts.is_empty());
    }

    #[test]
    fn dual_track_timeline_records_gop_seeks_zero_forward_and_roundtrips() {
        let geom = RopeGeometry::phi4_mini();
        let adapter = ModelSteeringAdapter::new(geom).expect("phi4 geometry");
        let q_len = adapter.geometry().q_rot_len();
        let ref_vec = vec![0.0_f32; q_len];
        let mut q_a = vec![0.0_f32; q_len];
        let mut q_b = vec![0.0_f32; q_len];
        let rot = adapter.geometry().rotary_dim;
        let half = rot / 2;
        for h in 0..adapter.geometry().num_q_heads {
            let base = h * rot;
            for i in 0..half {
                let (sa, ca) = (0.30 + 0.01 * h as f32).sin_cos();
                let (sb, cb) = (0.42 + 0.01 * h as f32).sin_cos();
                q_a[base + i] = ca;
                q_a[base + half + i] = sa;
                q_b[base + i] = cb;
                q_b[base + half + i] = sb;
            }
        }
        let refs: Vec<&[f32]> = vec![ref_vec.as_slice(); 32];
        let qa_slices: Vec<&[f32]> = vec![q_a.as_slice(); 32];
        let qb_slices: Vec<&[f32]> = vec![q_b.as_slice(); 32];

        let mut corpus = TextAttractorCorpus::new();
        let id1 = corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Turn 0 fact",
                &qa_slices,
                &refs,
                2.0,
            )
            .expect("ingest 0");
        assert_eq!(id1, 1);

        let detector = AdaptiveGopCutDetector::default();
        let policy = RecallPolicy::default();
        let mut timeline = GcsoDualTrackTimeline::new(GcsoDualTrackTimeline::geometry_fingerprint(
            adapter.geometry(),
        ));

        // Frame 0: initial frame -> always emits I-Frame keyframe.
        let is_i0 = timeline
            .record_step(
                &TimelineStepContext {
                    adapter: &adapter,
                    corpus: &corpus,
                    policy: &policy,
                    detector: &detector,
                },
                SourceTextRecord {
                    frame_index: 0,
                    timestamp_epoch_sec: 1_700_000_000,
                    role: AttractorRole::Fact,
                    text: "Turn 0 fact".to_string(),
                    energy: 2.0,
                },
                &FrameTransitionMetrics {
                    phase_l1_delta: 0.0,
                    void_state: VoidBoundaryState::InsideBasin,
                    frames_since_keyframe: 0,
                    new_cluster_spawned: false,
                },
            )
            .expect("frame 0");
        assert!(is_i0);

        // Frame 1: in-basin small phase delta -> emits P-Frame delta.
        corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Persona,
                "Turn 1 persona",
                &qb_slices,
                &refs,
                1.5,
            )
            .expect("ingest 1");
        let is_i1 = timeline
            .record_step(
                &TimelineStepContext {
                    adapter: &adapter,
                    corpus: &corpus,
                    policy: &policy,
                    detector: &detector,
                },
                SourceTextRecord {
                    frame_index: 1,
                    timestamp_epoch_sec: 1_700_000_010,
                    role: AttractorRole::Persona,
                    text: "Turn 1 persona".to_string(),
                    energy: 1.5,
                },
                &FrameTransitionMetrics {
                    phase_l1_delta: 0.08,
                    void_state: VoidBoundaryState::InsideBasin,
                    frames_since_keyframe: 1,
                    new_cluster_spawned: false,
                },
            )
            .expect("frame 1");
        assert!(!is_i1);

        // Frame 2: cvoid scene cut -> automatically emits a new I-Frame keyframe.
        let is_i2 = timeline
            .record_step(
                &TimelineStepContext {
                    adapter: &adapter,
                    corpus: &corpus,
                    policy: &policy,
                    detector: &detector,
                },
                SourceTextRecord {
                    frame_index: 2,
                    timestamp_epoch_sec: 1_700_000_020,
                    role: AttractorRole::Guard,
                    text: "Turn 2 scene cut".to_string(),
                    energy: 2.0,
                },
                &FrameTransitionMetrics {
                    phase_l1_delta: 0.10,
                    void_state: VoidBoundaryState::CvoidEmptySpace,
                    frames_since_keyframe: 2,
                    new_cluster_spawned: true,
                },
            )
            .expect("frame 2");
        assert!(is_i2);
        assert_eq!(timeline.keyframe_positions(), &[0, 2]);

        // Zero-forward seek to frame 1 reconstructs from keyframe 0 + 1 P-Frame.
        let seek1 = timeline.seek_zero_forward(1).expect("seek frame 1");
        assert_eq!(seek1.keyframe_index, 0);
        assert_eq!(seek1.p_frames_applied, 1);
        assert!((seek1.total_energy - 3.5).abs() < 1e-5);

        // Portable binary serialization roundtrip & subtitle track export.
        let bytes = timeline.to_bytes();
        let decoded = GcsoDualTrackTimeline::from_bytes(&bytes).expect("decode dual-track");
        assert_eq!(decoded, timeline);
        let subtitles = decoded.export_subtitle_track(Some(1));
        assert_eq!(subtitles.len(), 2);
        assert_eq!(subtitles[0].text, "Turn 0 fact");
        assert_eq!(subtitles[1].text, "Turn 1 persona");

        // Disk persistence roundtrip inside repository-contained `containers/`.
        let repo_container = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("containers")
            .join("dual_track_test.gcso");
        let written = timeline
            .save_to_path(&repo_container)
            .expect("save .gcso to repository containers/");
        assert_eq!(written, bytes.len());
        let loaded = GcsoDualTrackTimeline::load_from_path(&repo_container)
            .expect("load .gcso from repository containers/");
        assert_eq!(loaded, timeline);
        let _ = std::fs::remove_file(&repo_container);

        // Also verify automatic creation of missing nested parent directories when `.gitkeep` is absent.
        let nested_base = std::env::temp_dir().join(format!(
            "liminika_adapter_autocreated_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&nested_base);
        let nested_path = nested_base.join("containers").join("nested_test.gcso");
        assert!(!nested_base.exists());
        let nested_written = timeline
            .save_to_path(&nested_path)
            .expect("save .gcso should auto-create missing parent directories");
        assert_eq!(nested_written, bytes.len());
        let nested_loaded =
            GcsoDualTrackTimeline::load_from_path(&nested_path).expect("load auto-created .gcso");
        assert_eq!(nested_loaded, timeline);
        let _ = std::fs::remove_dir_all(&nested_base);
    }

    #[test]
    fn dual_track_timeline_fast_resumes_and_rebakes_across_models() {
        struct DeterministicHashEncoder {
            num_layers: usize,
            rotary_dim: usize,
        }

        impl SubtitleActivationEncoder for DeterministicHashEncoder {
            fn encode_subtitle_activations(
                &mut self,
                record: &SourceTextRecord,
            ) -> Result<Vec<Vec<f32>>, String> {
                let seed = record
                    .text
                    .bytes()
                    .fold(0x9E37_79B9_7F4A_7C15_u64, |acc, b| {
                        acc.wrapping_mul(131).wrapping_add(u64::from(b))
                    });
                let mut layers = Vec::with_capacity(self.num_layers);
                for l in 0..self.num_layers {
                    let mut v = Vec::with_capacity(self.rotary_dim);
                    for d in 0..self.rotary_dim {
                        let phase = ((seed ^ ((l as u64) << 8) ^ (d as u64)) % 100) as f32 * 0.01;
                        v.push((phase - 0.5) * 2.0);
                    }
                    layers.push(v);
                }
                Ok(layers)
            }
        }

        let src_adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("src init");
        let len = src_adapter.geometry().q_rot_len();
        let centroid = vec![0.0_f32; len];
        let refs: Vec<&[f32]> = vec![centroid.as_slice(); 32];
        let q_a: Vec<f32> = (0..len)
            .map(|i| if i % 2 == 0 { 0.7 } else { -0.3 })
            .collect();
        let qa_slices: Vec<&[f32]> = vec![q_a.as_slice(); 32];

        let mut corpus = TextAttractorCorpus::new();
        corpus
            .ingest_text_activations(
                &src_adapter,
                AttractorRole::Fact,
                "Fact zero",
                &qa_slices,
                &refs,
                1.5,
            )
            .expect("ingest 0");
        corpus
            .ingest_text_activations(
                &src_adapter,
                AttractorRole::Persona,
                "Persona one",
                &qa_slices,
                &refs,
                1.2,
            )
            .expect("ingest 1");

        let detector = AdaptiveGopCutDetector::default();
        let policy = RecallPolicy::default();
        let mut timeline = GcsoDualTrackTimeline::new(GcsoDualTrackTimeline::geometry_fingerprint(
            src_adapter.geometry(),
        ));
        let step_ctx = TimelineStepContext {
            adapter: &src_adapter,
            corpus: &corpus,
            policy: &policy,
            detector: &detector,
        };
        timeline
            .record_step(
                &step_ctx,
                SourceTextRecord {
                    frame_index: 0,
                    timestamp_epoch_sec: 1_700_000_000,
                    role: AttractorRole::Fact,
                    text: "Fact zero".to_string(),
                    energy: 1.5,
                },
                &FrameTransitionMetrics {
                    phase_l1_delta: 0.0,
                    void_state: VoidBoundaryState::InsideBasin,
                    frames_since_keyframe: 0,
                    new_cluster_spawned: false,
                },
            )
            .expect("step 0");
        timeline
            .record_step(
                &step_ctx,
                SourceTextRecord {
                    frame_index: 1,
                    timestamp_epoch_sec: 1_700_000_010,
                    role: AttractorRole::Persona,
                    text: "Persona one".to_string(),
                    energy: 1.2,
                },
                &FrameTransitionMetrics {
                    phase_l1_delta: 0.05,
                    void_state: VoidBoundaryState::InsideBasin,
                    frames_since_keyframe: 1,
                    new_cluster_spawned: false,
                },
            )
            .expect("step 1");

        // 1. Same model geometry -> FastResumed (zero-forward seek).
        let mut same_encoder = DeterministicHashEncoder {
            num_layers: src_adapter.geometry().num_layers,
            rotary_dim: len,
        };
        let same_ctx = RebakeContext {
            adapter: &src_adapter,
            policy: &policy,
            detector: &detector,
            centroid_layers: &[],
            up_to_frame: None,
        };
        match timeline
            .resume_or_rebake(&same_ctx, &mut same_encoder, 1)
            .expect("fast resume")
        {
            TimelineResumeOutcome::FastResumed { seek_state } => {
                assert_eq!(seek_state.target_frame_index, 1);
            }
            TimelineResumeOutcome::Rebaked { .. } => {
                panic!("expected FastResumed for matching geometry");
            }
        }

        // 2. Different model geometry -> Rebaked automatically from Subtitle Track.
        let mut alt_geom = RopeGeometry::phi4_mini();
        alt_geom.head_dim = 128;
        alt_geom.rotary_dim = 128;
        let alt_adapter = ModelSteeringAdapter::new(alt_geom).expect("alt init");
        let alt_len = alt_adapter.geometry().q_rot_len();
        let mut alt_encoder = DeterministicHashEncoder {
            num_layers: alt_adapter.geometry().num_layers,
            rotary_dim: alt_len,
        };
        let alt_ctx = RebakeContext {
            adapter: &alt_adapter,
            policy: &policy,
            detector: &detector,
            centroid_layers: &[],
            up_to_frame: None,
        };
        match timeline
            .resume_or_rebake(&alt_ctx, &mut alt_encoder, 1)
            .expect("cross-model rebake")
        {
            TimelineResumeOutcome::Rebaked {
                timeline: rebaked_tl,
                corpus: rebaked_corpus,
                seek_state,
                source_fingerprint,
                target_fingerprint,
                rebaked_frames,
            } => {
                assert_ne!(source_fingerprint, target_fingerprint);
                assert_eq!(rebaked_frames, 2);
                assert_eq!(rebaked_corpus.entries().len(), 2);
                assert_eq!(rebaked_tl.model_fingerprint(), target_fingerprint);
                assert_eq!(seek_state.target_frame_index, 1);
            }
            TimelineResumeOutcome::FastResumed { .. } => {
                panic!("expected Rebaked for mismatched geometry");
            }
        }
    }

    #[test]
    fn coherence_guard_gate_flags_low_fact_coherence_as_cvoid() {
        let gate = CoherenceGuardGate;
        let policy = RecallPolicy {
            min_coherence: 0.16,
            boundary_margin: 0.05,
            guard_suppress_margin: 0.0,
            top_k_facts: 2,
            top_k_personas: 1,
        };
        let decision = gate.evaluate(
            &RecallBoundaryMetrics {
                top1_fact_score: 0.077,
                top2_fact_score: 0.043,
                top1_persona_score: 0.240,
                max_guard_score: 0.100,
                guard_conflict_margin: -0.140,
            },
            &policy,
        );
        assert!(decision.suppress_fact_injection);
        assert_eq!(decision.void_state, VoidBoundaryState::CvoidEmptySpace);
    }

    #[test]
    fn coherence_guard_gate_detects_sharp_winner_liminal_without_weakening_ood_cvoid() {
        let gate = CoherenceGuardGate;
        let policy = RecallPolicy {
            min_coherence: 0.19,
            boundary_margin: 0.06,
            guard_suppress_margin: 0.0,
            top_k_facts: 2,
            top_k_personas: 1,
        };
        // Sharp unimodal winner just below strict min_coherence enters LiminalBoundary.
        let liminal = gate.evaluate(
            &RecallBoundaryMetrics {
                top1_fact_score: 0.17,
                top2_fact_score: 0.06,
                top1_persona_score: 0.05,
                max_guard_score: 0.02,
                guard_conflict_margin: -0.15,
            },
            &policy,
        );
        assert!(!liminal.suppress_fact_injection);
        assert_eq!(liminal.void_state, VoidBoundaryState::LiminalBoundary);

        // Flat out-of-domain noise remains strictly CvoidEmptySpace.
        let flat_ood = gate.evaluate(
            &RecallBoundaryMetrics {
                top1_fact_score: 0.15,
                top2_fact_score: 0.12,
                top1_persona_score: 0.05,
                max_guard_score: 0.02,
                guard_conflict_margin: -0.13,
            },
            &policy,
        );
        assert!(flat_ood.suppress_fact_injection);
        assert_eq!(flat_ood.void_state, VoidBoundaryState::CvoidEmptySpace);
    }

    #[test]
    fn perspective_void_damps_dense_hub_and_preserves_peripheral_attractor() {
        let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
        let len = adapter.geometry().q_rot_len();
        let centroid = vec![0.0_f32; len];
        let refs: Vec<&[f32]> = vec![centroid.as_slice(); 32];

        // Construct a crowded hub cluster (#1..#3) near axis 0 and a distinct peripheral
        // attractor (#4) along orthogonal axis 1.
        let mut hub_a = vec![0.0_f32; len];
        let mut hub_b = vec![0.0_f32; len];
        let mut hub_c = vec![0.0_f32; len];
        let mut periph = vec![0.0_f32; len];
        hub_a[0] = 1.0;
        hub_a[2] = 0.25;
        hub_b[0] = 0.95;
        hub_b[4] = 0.30;
        hub_c[0] = 0.95;
        hub_c[6] = 0.30;
        periph[1] = 1.0;

        let mut corpus = TextAttractorCorpus::new();
        let id_hub = corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Hub attractor A",
                &vec![hub_a.as_slice(); 32],
                &refs,
                1.5,
            )
            .expect("hub a");
        corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Hub attractor B",
                &vec![hub_b.as_slice(); 32],
                &refs,
                1.5,
            )
            .expect("hub b");
        corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Hub attractor C",
                &vec![hub_c.as_slice(); 32],
                &refs,
                1.5,
            )
            .expect("hub c");
        let id_periph = corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Peripheral attractor",
                &vec![periph.as_slice(); 32],
                &refs,
                1.5,
            )
            .expect("periph");

        // Query has equal raw projection on hub_a's axis 0 and peripheral's axis 1 (slightly
        // higher raw weight toward hub), yet from the peripheral attractor's perspective the query
        // stands out sharply whereas hub_a sits inside a crowded neighborhood.
        let mut query = vec![0.0_f32; len];
        query[0] = 0.62;
        query[1] = 0.58;
        let q_slices = vec![query.as_slice(); 32];
        let policy = RecallPolicy {
            min_coherence: 0.30,
            boundary_margin: 0.02,
            guard_suppress_margin: 0.0,
            top_k_facts: 2,
            top_k_personas: 1,
        };
        let recall = corpus
            .recall_for_query(&adapter, &q_slices, &refs, &policy)
            .expect("recall");
        assert_ne!(id_hub, id_periph);
        assert_eq!(recall.recalled_facts[0].0.anchor_id, id_periph);
        assert!(recall.min_local_void_depth < 0.05);
    }

    #[test]
    fn layer_angle_perspective_separates_specialized_late_layer_attractor() {
        let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
        let len = adapter.geometry().q_rot_len();
        let num_layers = adapter.geometry().num_layers;
        let centroid = vec![0.0_f32; len];
        let refs: Vec<&[f32]> = vec![centroid.as_slice(); num_layers];

        // Hub shares strong early-layer alignment (layers 0..24), while specialized attractor B
        // dominates the final 8 semantic layers.
        let mut hub_layers = vec![vec![0.0_f32; len]; num_layers];
        let mut spec_layers = vec![vec![0.0_f32; len]; num_layers];
        let mut query_layers = vec![vec![0.0_f32; len]; num_layers];
        for l in 0..num_layers {
            if l < 24 {
                hub_layers[l][0] = 1.0;
                spec_layers[l][1] = 1.0;
                query_layers[l][0] = 0.65;
                query_layers[l][1] = 0.55;
            } else {
                hub_layers[l][0] = 0.2;
                hub_layers[l][2] = 0.98;
                spec_layers[l][3] = 1.0;
                query_layers[l][3] = 1.0;
            }
        }
        let hub_refs: Vec<&[f32]> = hub_layers.iter().map(Vec::as_slice).collect();
        let spec_refs: Vec<&[f32]> = spec_layers.iter().map(Vec::as_slice).collect();
        let q_refs: Vec<&[f32]> = query_layers.iter().map(Vec::as_slice).collect();

        let mut corpus = TextAttractorCorpus::new();
        corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Early hub",
                &hub_refs,
                &refs,
                1.5,
            )
            .expect("ingest hub");
        let id_spec = corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Late specialist",
                &spec_refs,
                &refs,
                1.5,
            )
            .expect("ingest spec");

        let policy = RecallPolicy {
            min_coherence: 0.20,
            boundary_margin: 0.01,
            guard_suppress_margin: 0.0,
            top_k_facts: 2,
            top_k_personas: 1,
        };
        let recall = corpus
            .recall_for_query(&adapter, &q_refs, &refs, &policy)
            .expect("recall");
        assert_eq!(recall.recalled_facts[0].0.anchor_id, id_spec);
    }

    #[test]
    fn society_stratum_pipeline_supports_dynamic_add_remove_and_action_hub_arbitration() {
        let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
        let len = adapter.geometry().q_rot_len();
        let centroid = vec![0.0_f32; len];
        let refs: Vec<&[f32]> = vec![centroid.as_slice(); 32];

        let mut fact_vec = vec![0.0_f32; len];
        let mut guard_vec = vec![0.0_f32; len];
        fact_vec[0] = 1.0;
        guard_vec[1] = 1.0;

        let mut corpus = TextAttractorCorpus::new();
        corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Coherent fact",
                &vec![fact_vec.as_slice(); 32],
                &refs,
                1.5,
            )
            .expect("fact");
        corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Guard,
                "Contrastive guard",
                &vec![guard_vec.as_slice(); 32],
                &refs,
                1.5,
            )
            .expect("guard");

        let policy = RecallPolicy {
            min_coherence: 0.30,
            boundary_margin: 0.05,
            guard_suppress_margin: 0.0,
            top_k_facts: 2,
            top_k_personas: 1,
        };

        // 1. Default 5-stratum pipeline can dynamically remove & re-insert intermediate strata.
        let mut pipeline = StratumPipeline::default();
        assert_eq!(
            pipeline.stratum_names(),
            vec![
                "local_feature_worker",
                "head_angle_worker",
                "local_token_resonance",
                "perspective_domino_cvoid",
                "action_hub_boundary",
            ]
        );
        let removed = pipeline.remove_stratum("perspective_domino_cvoid");
        assert!(removed.is_some());
        assert_eq!(
            pipeline.stratum_names(),
            vec![
                "local_feature_worker",
                "head_angle_worker",
                "local_token_resonance",
                "action_hub_boundary",
            ]
        );

        // Even with the perspective stratum removed, ActionHub still arbitrates cleanly.
        let recall_fact = corpus
            .recall_for_query_with_strata(
                &adapter,
                &vec![fact_vec.as_slice(); 32],
                &refs,
                &policy,
                &pipeline,
            )
            .expect("fact recall");
        assert_eq!(
            recall_fact.action_decision,
            StigmergicActionDecision::CrystallizeMacroBasin
        );

        // Re-insert the perspective stratum at index 3 and verify Guard dominance triggers RepulsionEscape.
        if let Some(mid) = removed {
            pipeline.insert_stratum(3, mid);
        }
        assert_eq!(pipeline.stratum_names().len(), 5);
        let recall_guard = corpus
            .recall_for_query_with_strata(
                &adapter,
                &vec![guard_vec.as_slice(); 32],
                &refs,
                &policy,
                &pipeline,
            )
            .expect("guard recall");
        assert_eq!(
            recall_guard.action_decision,
            StigmergicActionDecision::RepulsionEscape
        );
        assert_eq!(recall_guard.void_state, VoidBoundaryState::CvoidEmptySpace);
    }

    #[test]
    fn head_angle_worker_stratum_separates_specialized_head_attractor() {
        // Use phi4_mini geometry where head 0 is shared noise and head 1 holds the specialized signal.
        let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
        let len = adapter.geometry().q_rot_len();
        let rotary_dim = len / 24;
        let num_layers = adapter.geometry().num_layers;
        let centroid = vec![0.0_f32; len];
        let refs: Vec<&[f32]> = vec![centroid.as_slice(); num_layers];

        let mut spec_a = vec![0.0_f32; len];
        let mut spec_b = vec![0.0_f32; len];
        for i in 0..4 {
            spec_a[i] = 0.25;
            spec_b[i] = 0.25;
        }
        spec_a[rotary_dim] = 1.0;
        spec_b[rotary_dim + 1] = 1.0;

        let mut corpus = TextAttractorCorpus::new();
        let id_a = corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "specialized head A",
                &vec![spec_a.as_slice(); num_layers],
                &refs,
                1.5,
            )
            .expect("ingest spec_a");
        let _id_b = corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "specialized head B",
                &vec![spec_b.as_slice(); num_layers],
                &refs,
                1.5,
            )
            .expect("ingest spec_b");

        let policy = RecallPolicy {
            min_coherence: 0.20,
            boundary_margin: 0.01,
            guard_suppress_margin: 0.0,
            top_k_facts: 2,
            top_k_personas: 1,
        };
        let recall = corpus
            .recall_for_query_with_strata(
                &adapter,
                &vec![spec_a.as_slice(); num_layers],
                &refs,
                &policy,
                &StratumPipeline::default(),
            )
            .expect("head-angle recall");
        assert_eq!(
            recall.recalled_facts.first().map(|(e, _)| e.anchor_id),
            Some(id_a)
        );
        assert!(recall.top1_fact_score > recall.top2_fact_score);
    }

    #[test]
    fn persona_recall_survives_factual_cvoid_and_routes_via_gateway_frame() {
        let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
        let len = adapter.geometry().q_rot_len();
        let num_layers = adapter.geometry().num_layers;
        let centroid = vec![0.0_f32; len];
        let refs: Vec<&[f32]> = vec![centroid.as_slice(); num_layers];

        let mut fact_vec = vec![0.0_f32; len];
        let mut persona_vec = vec![0.0_f32; len];
        fact_vec[0] = 1.0;
        persona_vec[1] = 1.0;

        let mut corpus = TextAttractorCorpus::new();
        corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Orthogonal fact",
                &vec![fact_vec.as_slice(); num_layers],
                &refs,
                1.5,
            )
            .expect("fact");
        let id_persona = corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Persona,
                "Target persona",
                &vec![persona_vec.as_slice(); num_layers],
                &refs,
                1.5,
            )
            .expect("persona");

        let policy = RecallPolicy {
            min_coherence: 0.30,
            boundary_margin: 0.02,
            guard_suppress_margin: 0.0,
            top_k_facts: 2,
            top_k_personas: 1,
        };
        let recall = corpus
            .recall_for_query_with_strata(
                &adapter,
                &vec![persona_vec.as_slice(); num_layers],
                &refs,
                &policy,
                &StratumPipeline::default(),
            )
            .expect("persona recall");

        assert!(recall.recalled_facts.is_empty());
        assert_eq!(
            recall.recalled_personas.first().map(|(e, _)| e.anchor_id),
            Some(id_persona)
        );
        assert_eq!(recall.gateway_frame.dominant_slot, ATTRACTOR_SLOT_PERSONA);
        assert_eq!(
            recall.action_decision,
            StigmergicActionDecision::ExpressPhaseSteer
        );
    }

    #[test]
    fn local_token_resonance_arbitrates_channel_disagreement_via_pspm_subhead_route() {
        let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
        let len = adapter.geometry().q_rot_len();
        let num_layers = adapter.geometry().num_layers;
        let centroid = vec![0.0_f32; len];
        let refs: Vec<&[f32]> = vec![centroid.as_slice(); num_layers];

        let mut f1 = vec![0.0_f32; len];
        let mut f2 = vec![0.0_f32; len];
        let mut f3 = vec![0.0_f32; len];
        f1[0] = 0.9;
        f1[1] = 0.4;
        f2[0] = 0.85;
        f2[2] = 0.5;
        f3[0] = 0.85;
        f3[3] = 0.5;

        let mut corpus = TextAttractorCorpus::new();
        corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Ambiguous hub fact",
                &vec![f1.as_slice(); num_layers],
                &refs,
                1.5,
            )
            .expect("f1");
        let id_target = corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Target specialist fact",
                &vec![f2.as_slice(); num_layers],
                &refs,
                1.5,
            )
            .expect("f2");
        corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Peer fact",
                &vec![f3.as_slice(); num_layers],
                &refs,
                1.5,
            )
            .expect("f3");

        let mut query = vec![0.0_f32; len];
        query[0] = 1.0;
        let policy = RecallPolicy {
            min_coherence: 0.30,
            boundary_margin: 0.02,
            guard_suppress_margin: 0.0,
            top_k_facts: 2,
            top_k_personas: 1,
        };
        // Local channel sharply selects id_target while global channel prefers f1 slightly -> disagreement triggers PspmSubHeadRoute.
        let recall = corpus
            .recall_for_query_with_strata_and_local_scores(
                &adapter,
                &vec![query.as_slice(); num_layers],
                &refs,
                &policy,
                &StratumPipeline::default(),
                Some(&[0.10, 0.85, 0.12]),
            )
            .expect("disagreement recall");
        assert!(recall.telemetry.channel_disagreement);
        assert_eq!(
            recall.action_decision,
            StigmergicActionDecision::PspmSubHeadRoute
        );
        assert_eq!(
            recall.recalled_facts.first().map(|(e, _)| e.anchor_id),
            Some(id_target)
        );
    }

    #[test]
    fn support_gated_perspective_prevents_phantom_boost_on_non_matching_outlier() {
        let perspective = AsymmetricDominoVoidPerspective::default();
        let entry = TextAttractorEntry {
            anchor_id: 99,
            role: AttractorRole::Guard,
            text: "Outlier".to_string(),
            per_layer_q7: vec![],
            energy: 2.0,
            residual_sig: vec![],
        };
        let outlier_ctx = AttractorPerspectiveContext {
            entry: &entry,
            raw_cosine: -0.10,
            neighbor_mean_cosine: -0.20,
            neighbor_std_cosine: 0.08,
            global_neighbor_mean: 0.15,
            query_mean_cosine: 0.0,
            query_std_cosine: 0.08,
        };
        let reading = perspective.measure_perspective(&outlier_ctx, 0.25);
        assert!(
            (reading.perspective_affinity - (-0.10)).abs() < 1e-6,
            "negative raw_cosine must not gain phantom affinity from negative hub_shift: got {}",
            reading.perspective_affinity
        );
    }

    #[test]
    fn per_role_local_token_resonance_arbitrates_fact_despite_persona_hub() {
        let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
        let len = adapter.geometry().q_rot_len();
        let num_layers = adapter.geometry().num_layers;
        let centroid = vec![0.0_f32; len];
        let refs: Vec<&[f32]> = vec![centroid.as_slice(); num_layers];

        let mut f_hub = vec![0.0_f32; len];
        let mut f_target = vec![0.0_f32; len];
        let mut f_peer = vec![0.0_f32; len];
        let mut persona_hub = vec![0.0_f32; len];
        f_hub[0] = 0.70;
        f_hub[1] = 0.70;
        f_target[2] = 1.0;
        f_peer[3] = 1.0;
        persona_hub[0] = 0.90;
        persona_hub[4] = 0.43;

        let mut corpus = TextAttractorCorpus::new();
        corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Generic hub fact",
                &vec![f_hub.as_slice(); num_layers],
                &refs,
                1.5,
            )
            .expect("f_hub");
        let id_target = corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Specialist fact",
                &vec![f_target.as_slice(); num_layers],
                &refs,
                1.5,
            )
            .expect("f_target");
        corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Peer fact",
                &vec![f_peer.as_slice(); num_layers],
                &refs,
                1.5,
            )
            .expect("f_peer");
        corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Persona,
                "Persona style hub",
                &vec![persona_hub.as_slice(); num_layers],
                &refs,
                1.5,
            )
            .expect("persona_hub");

        // Persona hub dominates mixed global cosine (`0.90`), yet local channel sharply isolates id_target inside Facts.
        let mut query = vec![0.0_f32; len];
        query[0] = 1.0;
        let policy = RecallPolicy {
            min_coherence: 0.25,
            boundary_margin: 0.02,
            guard_suppress_margin: 0.0,
            top_k_facts: 2,
            top_k_personas: 1,
        };
        let recall = corpus
            .recall_for_query_with_strata_and_local_scores(
                &adapter,
                &vec![query.as_slice(); num_layers],
                &refs,
                &policy,
                &StratumPipeline::default(),
                Some(&[0.01, 0.45, -0.02, -0.05]),
            )
            .expect("per-role arbitration");
        assert_eq!(
            recall.recalled_facts.first().map(|(e, _)| e.anchor_id),
            Some(id_target)
        );
        assert_eq!(
            recall.action_decision,
            StigmergicActionDecision::PspmSubHeadRoute
        );
    }

    #[test]
    fn action_hub_compound_decomposition_links_disjoint_specialists_while_respecting_cvoid() {
        let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
        let len = adapter.geometry().q_rot_len();
        let num_layers = adapter.geometry().num_layers;
        let centroid = vec![0.0_f32; len];
        let refs: Vec<&[f32]> = vec![centroid.as_slice(); num_layers];

        let mut f_hub = vec![0.0_f32; len];
        let mut f_shared_peer = vec![0.0_f32; len];
        let mut f_spec_a = vec![0.0_f32; len];
        let mut f_spec_b = vec![0.0_f32; len];
        f_hub[0] = 0.85;
        f_hub[1] = 0.52;
        f_shared_peer[0] = 0.85;
        f_shared_peer[4] = 0.52;
        f_spec_a[2] = 1.0;
        f_spec_b[3] = 1.0;

        let mut corpus = TextAttractorCorpus::new();
        let id_hub = corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Domain overview hub",
                &vec![f_hub.as_slice(); num_layers],
                &refs,
                1.5,
            )
            .expect("hub");
        corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Domain shared sibling",
                &vec![f_shared_peer.as_slice(); num_layers],
                &refs,
                1.5,
            )
            .expect("peer");
        let id_a = corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Specialist A",
                &vec![f_spec_a.as_slice(); num_layers],
                &refs,
                1.5,
            )
            .expect("spec_a");
        let id_b = corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Specialist B",
                &vec![f_spec_b.as_slice(); num_layers],
                &refs,
                1.5,
            )
            .expect("spec_b");

        let mut query = vec![0.0_f32; len];
        query[0] = 0.90;
        query[1] = 0.30;
        query[2] = 0.22;
        query[3] = 0.22;
        let policy = RecallPolicy {
            min_coherence: 0.20,
            boundary_margin: 0.04,
            guard_suppress_margin: 0.0,
            top_k_facts: 2,
            top_k_personas: 1,
        };

        // Shared qualifier tokens spike across both entry 0 & 1, whereas disjoint specialist
        // tokens spike cleanly onto entry 2 and entry 3.
        let obs = LocalChannelObservation {
            combined_scores: vec![0.18, 0.02, 0.08, 0.08],
            q_to_d_scores: vec![0.16, 0.03, 0.10, 0.10],
            d_to_q_scores: vec![0.05, -0.02, 0.08, 0.08],
            token_deposits: vec![
                TokenResonanceDeposit {
                    disc_weight: 0.35,
                    entry_excess: vec![0.40, 0.35, -0.25, -0.25],
                    token_phase_words: [0; 8],
                },
                TokenResonanceDeposit {
                    disc_weight: 0.35,
                    entry_excess: vec![0.40, 0.34, -0.25, -0.25],
                    token_phase_words: [0; 8],
                },
                TokenResonanceDeposit {
                    disc_weight: 0.45,
                    entry_excess: vec![-0.15, -0.15, 0.60, -0.15],
                    token_phase_words: [0; 8],
                },
                TokenResonanceDeposit {
                    disc_weight: 0.45,
                    entry_excess: vec![-0.15, -0.15, -0.15, 0.60],
                    token_phase_words: [0; 8],
                },
            ],
        };

        let recall = corpus
            .recall_for_query_with_strata_observation(
                &adapter,
                &vec![query.as_slice(); num_layers],
                &refs,
                &policy,
                &StratumPipeline::default(),
                Some(obs),
            )
            .expect("compound recall");
        let ids: Vec<u32> = recall
            .recalled_facts
            .iter()
            .map(|(e, _)| e.anchor_id)
            .collect();
        assert_eq!(ids.len(), 2, "expected 2 decomposed facts, got {ids:?}");
        assert!(
            ids.contains(&id_a) && ids.contains(&id_b),
            "expected specialists {id_a} and {id_b}, got {ids:?} (hub={id_hub})"
        );
        assert_eq!(
            recall.action_decision,
            StigmergicActionDecision::PspmSubHeadRoute
        );

        // Weak noise OOD query below basin support must stay suppressed in cvoid.
        let mut ood_q = vec![0.0_f32; len];
        ood_q[15] = 1.0;
        let weak_obs = LocalChannelObservation {
            combined_scores: vec![0.01, 0.0, 0.01, 0.01],
            q_to_d_scores: vec![0.01, 0.0, 0.01, 0.01],
            d_to_q_scores: vec![0.0, 0.0, 0.0, 0.0],
            token_deposits: vec![TokenResonanceDeposit {
                disc_weight: 0.05,
                entry_excess: vec![0.01, -0.01, 0.01, -0.01],
                token_phase_words: [0; 8],
            }],
        };
        let ood_recall = corpus
            .recall_for_query_with_strata_observation(
                &adapter,
                &vec![ood_q.as_slice(); num_layers],
                &refs,
                &policy,
                &StratumPipeline::default(),
                Some(weak_obs),
            )
            .expect("ood recall");
        assert_eq!(ood_recall.void_state, VoidBoundaryState::CvoidEmptySpace);
        assert!(ood_recall.recalled_facts.is_empty());
        assert_eq!(
            ood_recall
                .telemetry
                .cognitive_trajectory
                .latest_step()
                .map(|s| s.op()),
            Some(liminika_core::traits::gateway::CognitiveOpKind::Observe)
        );
    }

    #[test]
    fn residual_peeling_decomposer_chains_three_disjoint_specialists_beyond_top_k_two() {
        let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
        let len = adapter.geometry().q_rot_len();
        let num_layers = adapter.geometry().num_layers;
        let centroid = vec![0.0_f32; len];
        let refs: Vec<&[f32]> = vec![centroid.as_slice(); num_layers];

        let mut f_hub = vec![0.0_f32; len];
        let mut f_a = vec![0.0_f32; len];
        let mut f_b = vec![0.0_f32; len];
        let mut f_c = vec![0.0_f32; len];
        f_hub[0] = 1.0;
        f_a[1] = 1.0;
        f_b[2] = 1.0;
        f_c[3] = 1.0;

        let mut corpus = TextAttractorCorpus::new();
        let _id_hub = corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Overview hub",
                &vec![f_hub.as_slice(); num_layers],
                &refs,
                1.5,
            )
            .expect("hub");
        let id_a = corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Specialist alpha",
                &vec![f_a.as_slice(); num_layers],
                &refs,
                1.5,
            )
            .expect("spec_a");
        let id_b = corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Specialist beta",
                &vec![f_b.as_slice(); num_layers],
                &refs,
                1.5,
            )
            .expect("spec_b");
        let id_c = corpus
            .ingest_text_activations(
                &adapter,
                AttractorRole::Fact,
                "Specialist gamma",
                &vec![f_c.as_slice(); num_layers],
                &refs,
                1.5,
            )
            .expect("spec_c");

        let mut query = vec![0.0_f32; len];
        query[0] = 0.70;
        query[1] = 0.25;
        query[2] = 0.25;
        query[3] = 0.25;
        let policy = RecallPolicy {
            min_coherence: 0.20,
            boundary_margin: 0.04,
            guard_suppress_margin: 0.0,
            top_k_facts: 2,
            top_k_personas: 1,
        };

        let obs = LocalChannelObservation {
            combined_scores: vec![0.14, 0.12, 0.10, 0.09],
            q_to_d_scores: vec![0.10, 0.12, 0.11, 0.10],
            d_to_q_scores: vec![0.02, 0.08, 0.08, 0.07],
            token_deposits: vec![
                TokenResonanceDeposit {
                    disc_weight: 0.40,
                    entry_excess: vec![-0.12, 0.55, -0.12, -0.12],
                    token_phase_words: [0; 8],
                },
                TokenResonanceDeposit {
                    disc_weight: 0.40,
                    entry_excess: vec![-0.12, -0.12, 0.52, -0.12],
                    token_phase_words: [0; 8],
                },
                TokenResonanceDeposit {
                    disc_weight: 0.40,
                    entry_excess: vec![-0.12, -0.12, -0.12, 0.48],
                    token_phase_words: [0; 8],
                },
            ],
        };

        let recall = corpus
            .recall_for_query_with_strata_observation(
                &adapter,
                &vec![query.as_slice(); num_layers],
                &refs,
                &policy,
                &StratumPipeline::default(),
                Some(obs),
            )
            .expect("3-hop recall");
        let ids: Vec<u32> = recall
            .recalled_facts
            .iter()
            .map(|(e, _)| e.anchor_id)
            .collect();
        assert_eq!(ids.len(), 3, "expected 3 chained specialists, got {ids:?}");
        assert!(
            ids.contains(&id_a) && ids.contains(&id_b) && ids.contains(&id_c),
            "expected {id_a}, {id_b}, {id_c} in coalition, got {ids:?}"
        );
        assert_eq!(
            recall.action_decision,
            StigmergicActionDecision::PspmSubHeadRoute
        );
        assert_eq!(
            recall
                .telemetry
                .cognitive_trajectory
                .latest_step()
                .map(|s| s.op()),
            Some(liminika_core::traits::gateway::CognitiveOpKind::DecomposePeel)
        );
    }
}
