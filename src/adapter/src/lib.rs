// name: src/adapter/src/lib.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model connection, RoPE geometry mapping, and optimization adapter over [`liminika_core`].
//!
//! Keeps [`liminika_core`] strictly `#![no_std]` and model-agnostic while providing a
//! unified, zero-external-dependency bridge for external LLM runtimes (Candle, `llama.cpp`,
//! and CLI evaluation harnesses).

use liminika_core::abi::{gcso_daes_slot_t, gcso_edbc_state_t, gcso_pointer_trail_t, gcso_q7_t};
use liminika_core::action_hub::{ActionHubEngine, StigmergicActionDecision, StigmergicThresholds};
use liminika_core::entropy_control::EntropyController;
use liminika_core::phase_steering::PhaseSteeringEngine;
use liminika_core::residual_adapter::ResidualAdapterEngine;
use liminika_core::traits::aligned::{
    AlignedSlice16, AlignedSlice32, AlignedSliceMut16, AlignedSliceMut32,
};
use liminika_core::traits::gateway::{
    ActionHubGatewayExt, AdaptiveThresholdPolicy, CausalProvenanceReport, GatewayFrame,
    GatewayPhaseSteeringExt, GatewaySrlSteeringExt, ManifoldGatewaySignal, ManifoldProbeBaseline,
    ObjectiveFact, StigmergicThresholdPolicy, SubjectivePersona, SystemConstraint,
};
use liminika_core::traits::primitives::MAX_ATTENTION_HEADS;
use liminika_core::traits::rope::{
    PhaseSteeringGeometryExt, RopeGeometryDescriptor, RopePairLayout, RopeSteeringTarget,
};
use liminika_core::traits::SrlAdapter;

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

/// Dynamic top-K recall result from [`TextAttractorCorpus::recall_for_query`].
#[derive(Debug, Clone)]
pub struct DynamicAttractorRecall<'a> {
    /// Whether the query landed inside a basin, on a liminal boundary, or in `cvoid` empty space.
    pub void_state: VoidBoundaryState,
    /// Highest geometric resonance score among Fact attractors.
    pub top1_fact_score: f32,
    /// Second-highest geometric resonance score among Fact attractors.
    pub top2_fact_score: f32,
    /// Highest Guard resonance minus highest Fact resonance (`> 0` indicates guard-dominant conflict).
    pub guard_conflict_margin: f32,
    /// Top-K recalled Fact entries (empty when `void_state == CvoidEmptySpace`).
    pub recalled_facts: Vec<(&'a TextAttractorEntry, f32)>,
    /// Top-K recalled Persona entries.
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
}

impl Default for RecallPolicy {
    fn default() -> Self {
        Self {
            top_k_facts: 2,
            top_k_personas: 1,
            min_coherence: 0.35,
            boundary_margin: 0.03,
        }
    }
}

/// Model-agnostic Cold-Path corpus that compiles natural-language logs, facts, personas, and
/// contrastive guards into a dynamic attractor manifold with explicit `cvoid` boundary detection.
#[derive(Debug, Default, Clone)]
pub struct TextAttractorCorpus {
    entries: Vec<TextAttractorEntry>,
    next_anchor_id: u32,
    common_mode_dir: Vec<f32>,
}

impl TextAttractorCorpus {
    /// Creates an empty [`TextAttractorCorpus`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_anchor_id: 1,
            common_mode_dir: Vec::new(),
        }
    }

    /// Returns all stored attractor entries.
    #[must_use]
    pub fn entries(&self) -> &[TextAttractorEntry] {
        &self.entries
    }

    fn build_steered_residual_sig(
        geom: &RopeGeometry,
        q_layers: &[&[f32]],
        ref_layers: &[&[f32]],
    ) -> Vec<f32> {
        let mut sig = Vec::new();
        for (l, (q, r)) in q_layers.iter().zip(ref_layers.iter()).enumerate() {
            if !geom.is_rope_layer(l) {
                continue;
            }
            for (&qv, &rv) in q.iter().zip(r.iter()) {
                sig.push(qv - rv);
            }
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

    fn project_and_normalize(v: &[f32], common_dir: &[f32]) -> Vec<f32> {
        let mut out = v.to_vec();
        if common_dir.len() == out.len() && !common_dir.is_empty() {
            let dot: f64 = out
                .iter()
                .zip(common_dir.iter())
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let d = dot as f32;
            for (x, &c) in out.iter_mut().zip(common_dir.iter()) {
                *x -= d * c;
            }
            Self::l2_normalize_in_place(&mut out);
        }
        out
    }

    fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
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

    /// Estimates and stores the dominant unsupervised common-mode singular direction across
    /// ingested entries via power iteration, so recall projects out shared prompt/template bias.
    pub fn fit_common_mode_projection(&mut self) {
        if self.entries.len() < 4 {
            self.common_mode_dir.clear();
            return;
        }
        let dim = self.entries[0].residual_sig.len();
        if dim == 0 || self.entries.iter().any(|e| e.residual_sig.len() != dim) {
            self.common_mode_dir.clear();
            return;
        }
        let mut v = vec![0.0_f32; dim];
        for e in &self.entries {
            for (dst, &src) in v.iter_mut().zip(e.residual_sig.iter()) {
                *dst += src;
            }
        }
        Self::l2_normalize_in_place(&mut v);
        if v.iter().all(|&x| x == 0.0) {
            v[0] = 1.0;
        }
        for _ in 0..8 {
            let mut next = vec![0.0_f32; dim];
            for e in &self.entries {
                let dot: f64 = e
                    .residual_sig
                    .iter()
                    .zip(v.iter())
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                let d = dot as f32;
                for (dst, &src) in next.iter_mut().zip(e.residual_sig.iter()) {
                    *dst += d * src;
                }
            }
            Self::l2_normalize_in_place(&mut next);
            v = next;
        }
        self.common_mode_dir = v;
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
        let projected: Vec<Vec<f32>> = self
            .entries
            .iter()
            .map(|e| Self::project_and_normalize(&e.residual_sig, &self.common_mode_dir))
            .collect();
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
    /// steered-layer active-head geometric cosine similarity (zero keyword heuristics),
    /// returning top-K recalled attractors and the query's [`VoidBoundaryState`].
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
        let query_sig = Self::project_and_normalize(&raw_sig, &self.common_mode_dir);

        let mut fact_scores: Vec<(&TextAttractorEntry, f32)> = Vec::new();
        let mut persona_scores: Vec<(&TextAttractorEntry, f32)> = Vec::new();
        let mut max_guard_score = -1.0_f32;

        for entry in &self.entries {
            let entry_sig = Self::project_and_normalize(&entry.residual_sig, &self.common_mode_dir);
            let mean_cos = Self::cosine_f32(&query_sig, &entry_sig);
            match entry.role {
                AttractorRole::Fact => fact_scores.push((entry, mean_cos)),
                AttractorRole::Persona => persona_scores.push((entry, mean_cos)),
                AttractorRole::Guard => {
                    if mean_cos > max_guard_score {
                        max_guard_score = mean_cos;
                    }
                }
            }
        }

        fact_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        persona_scores.sort_by(|a, b| b.1.total_cmp(&a.1));

        let top1_fact_score = fact_scores.first().map_or(-1.0, |x| x.1);
        let top2_fact_score = fact_scores.get(1).map_or(-1.0, |x| x.1);
        let guard_conflict_margin = max_guard_score - top1_fact_score;

        let void_state = if top1_fact_score < policy.min_coherence {
            VoidBoundaryState::CvoidEmptySpace
        } else if (top1_fact_score - top2_fact_score) < policy.boundary_margin {
            VoidBoundaryState::LiminalBoundary
        } else {
            VoidBoundaryState::InsideBasin
        };

        let recalled_facts = if matches!(void_state, VoidBoundaryState::CvoidEmptySpace) {
            Vec::new()
        } else {
            fact_scores
                .into_iter()
                .filter(|(_, s)| *s >= policy.min_coherence * 0.75)
                .take(policy.top_k_facts)
                .collect()
        };
        let recalled_personas = persona_scores
            .into_iter()
            .take(policy.top_k_personas)
            .collect();

        Ok(DynamicAttractorRecall {
            void_state,
            top1_fact_score,
            top2_fact_score,
            guard_conflict_margin,
            recalled_facts,
            recalled_personas,
        })
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
}
