// name: src/adapter/src/lib.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model connection, RoPE geometry mapping, and optimization adapter over [`liminika_core`].
//!
//! Keeps [`liminika_core`] strictly `#![no_std]` and model-agnostic while providing a
//! unified, zero-external-dependency bridge for external LLM runtimes (Candle, `llama.cpp`,
//! and CLI evaluation harnesses).

use liminika_core::abi::{gcso_edbc_state_t, gcso_q7_t};
use liminika_core::entropy_control::EntropyController;
use liminika_core::phase_steering::PhaseSteeringEngine;
use liminika_core::residual_adapter::ResidualAdapterEngine;
use liminika_core::traits::aligned::{
    AlignedSlice16, AlignedSlice32, AlignedSliceMut16, AlignedSliceMut32,
};
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
    entropy_controller: EntropyController,
    residual_adapter: ResidualAdapterEngine,
}

impl ModelSteeringAdapter {
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
        let entropy_controller = EntropyController::new(gcso_edbc_state_t::default());
        let residual_adapter = ResidualAdapterEngine::new(1.0, 0, 1);

        Ok(Self {
            geometry,
            phase_engine,
            entropy_controller,
            residual_adapter,
        })
    }

    /// Returns the validated [`RopeGeometry`].
    #[must_use]
    pub const fn geometry(&self) -> &RopeGeometry {
        &self.geometry
    }

    /// Updates the active RoPE layer bitmask (`1` = steer, `0` = skip NoPE/disabled layer).
    pub fn set_layer_rope_mask(&mut self, mask: u64) {
        self.geometry.layer_rope_mask = mask;
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

    /// Evaluates Top-2 sparse surrogate entropy via [`EntropyController::eval_surrogate_and_step`]
    /// and scales `logits` by the resulting adaptive `sample_temperature`.
    ///
    /// # Errors
    ///
    /// Returns an error if `logits` is empty or `EntropyController` reports an error.
    pub fn step_edbc_logits(&mut self, logits: &mut [f32]) -> Result<gcso_edbc_state_t, String> {
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
        let temp = self
            .entropy_controller
            .eval_surrogate_and_step(p_max, p_2nd, 0.0, 0.0, 1.0, 0.1, 0.1, &mut state_out)
            .map_err(|e| format!("eval_surrogate_and_step error: {e}"))?;

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
}
