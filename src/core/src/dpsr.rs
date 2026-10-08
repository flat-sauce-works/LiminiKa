// name: src/core/src/dpsr.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dynamic Phase-Shifted RoPE (DPSR) core kernel and steering engine implementation.
//!
//! This module implements the `PhaseSteering` trait defined in `traits.rs` (Ground Truth).
//! It guarantees zero dynamic memory allocations on the Hot Path, operating strictly over
//! aligned slice wrappers (`AlignedSliceMut32`, `AlignedSlice16`) in O(d_head) time complexity per head.

#![allow(clippy::module_name_repetitions)]
#![allow(clippy::inline_always)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]

#[cfg(not(feature = "std"))]
use core::f32::consts::PI;
#[cfg(feature = "std")]
use std::f32::consts::PI;

use crate::abi::{gcso_q7_t, GcsoStatus, GCSO_ERROR_EDBC_SINGULARITY, GCSO_ERROR_INVALID_ARGUMENT};
use crate::traits::{
    AlignedSlice16, AlignedSlice32, AlignedSliceMut16, AlignedSliceMut32, GcsoResult, PhaseSteering,
};

/// Scale factor for Q7 quantized phase representation (beta_Q7 = pi / 128).
const Q7_PHASE_SCALE: f32 = PI / 128.0;

/// Inverse scale factor to map radians back to Q7 fixed-point representation.
const INV_Q7_PHASE_SCALE: f32 = 128.0 / PI;

/// Cross-environment single-precision floating point sine function wrapper.
#[inline(always)]
fn sin_f32(val: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        val.sin()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sinf(val)
    }
}

/// Cross-environment single-precision floating point cosine function wrapper.
#[inline(always)]
fn cos_f32(val: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        val.cos()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::cosf(val)
    }
}

/// Cross-environment single-precision floating point hyperbolic tangent wrapper.
#[inline(always)]
fn tanh_f32(val: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        val.tanh()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::tanhf(val)
    }
}

/// Cross-environment single-precision floating point square root wrapper.
#[inline(always)]
fn sqrt_f32(val: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        val.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrtf(val)
    }
}

/// Cross-environment single-precision floating point four-quadrant arctangent wrapper.
#[inline(always)]
fn atan2_f32(y: f32, x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        y.atan2(x)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::atan2f(y, x)
    }
}

/// Cross-environment single-precision floating point rounding function wrapper.
#[inline(always)]
fn round_f32(val: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        val.round()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::roundf(val)
    }
}

/// Primary DPSR kernel engine executing inline SO(2)^(d_head/2) dynamic phase steering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DpsrEngine {
    head_dim: u32,
    num_heads: u32,
}

impl DpsrEngine {
    /// Creates a new `DpsrEngine` instance with validated head configuration.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `head_dim` is odd, zero, or `num_heads` is zero.
    #[inline]
    pub fn new(head_dim: u32, num_heads: u32) -> GcsoResult<Self> {
        if head_dim == 0 || (head_dim % 2 != 0) || num_heads == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(Self {
            head_dim,
            num_heads,
        })
    }

    /// Helper method to validate tensor buffer lengths against configured head parameters.
    #[inline(always)]
    fn validate_tensor_bounds(&self, tensor_len: usize, phase_len: usize) -> GcsoResult<()> {
        let expected_tensor_len = (self.head_dim as usize) * (self.num_heads as usize);
        if tensor_len < expected_tensor_len || phase_len < (self.num_heads as usize) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(())
    }
}

impl PhaseSteering for DpsrEngine {
    #[inline(always)]
    fn head_dim(&self) -> u32 {
        self.head_dim
    }

    #[inline(always)]
    fn num_heads(&self) -> u32 {
        self.num_heads
    }

    #[inline]
    fn apply_phase_steering(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        self.validate_tensor_bounds(query_tensor.len(), phase_deltas.len())?;

        let head_dim = self.head_dim as usize;
        let num_heads = self.num_heads as usize;

        let q_buf = query_tensor.as_mut_slice();
        let p_buf = phase_deltas.as_slice();

        for (h, head_q) in q_buf.chunks_exact_mut(head_dim).take(num_heads).enumerate() {
            let theta = (p_buf[h] as f32) * Q7_PHASE_SCALE;
            if theta == 0.0 {
                continue;
            }

            let cos_t = cos_f32(theta);
            let sin_t = sin_f32(theta);

            for pair in head_q.chunks_exact_mut(2) {
                let q0 = pair[0];
                let q1 = pair[1];

                pair[0] = q0 * cos_t - q1 * sin_t;
                pair[1] = q0 * sin_t + q1 * cos_t;
            }
        }

        Ok(())
    }

    #[inline]
    fn apply_phase_steering_fused(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        self.validate_tensor_bounds(query_tensor.len(), phase_deltas.len())?;
        self.validate_tensor_bounds(key_tensor.len(), phase_deltas.len())?;

        let head_dim = self.head_dim as usize;
        let num_heads = self.num_heads as usize;

        let q_buf = query_tensor.as_mut_slice();
        let k_buf = key_tensor.as_mut_slice();
        let p_buf = phase_deltas.as_slice();

        let q_heads = q_buf.chunks_exact_mut(head_dim).take(num_heads);
        let k_heads = k_buf.chunks_exact_mut(head_dim).take(num_heads);

        for (h, (head_q, head_k)) in q_heads.zip(k_heads).enumerate() {
            let theta = (p_buf[h] as f32) * Q7_PHASE_SCALE;
            if theta == 0.0 {
                continue;
            }

            let cos_t = cos_f32(theta);
            let sin_t = sin_f32(theta);

            for (pair_q, pair_k) in head_q.chunks_exact_mut(2).zip(head_k.chunks_exact_mut(2)) {
                // Modulate Query
                let q0 = pair_q[0];
                let q1 = pair_q[1];
                pair_q[0] = q0 * cos_t - q1 * sin_t;
                pair_q[1] = q0 * sin_t + q1 * cos_t;

                // Modulate Key
                let k0 = pair_k[0];
                let k1 = pair_k[1];
                pair_k[0] = k0 * cos_t - k1 * sin_t;
                pair_k[1] = k0 * sin_t + k1 * cos_t;
            }
        }

        Ok(())
    }

    #[inline]
    fn apply_phase_steering_safe(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        if max_rad <= 0.0 || max_rad.is_nan() || max_rad.is_infinite() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.validate_tensor_bounds(query_tensor.len(), phase_deltas.len())?;

        let head_dim = self.head_dim as usize;
        let num_heads = self.num_heads as usize;
        let pairs_per_head = head_dim / 2;

        // RIPA restriction: restrict phase steering strictly to upper d_head / 4 dimensions (d_head / 8 pairs)
        let low_freq_pairs = (head_dim / 8).max(1);
        let start_pair = if pairs_per_head > low_freq_pairs {
            pairs_per_head - low_freq_pairs
        } else {
            0
        };

        let q_buf = query_tensor.as_mut_slice();
        let p_buf = phase_deltas.as_slice();

        for (h, head_q) in q_buf.chunks_exact_mut(head_dim).take(num_heads).enumerate() {
            let raw_theta = (p_buf[h] as f32) * Q7_PHASE_SCALE;
            // RIPA soft clamping: theta_safe = max_rad * tanh(raw_theta / max_rad)
            let theta = max_rad * tanh_f32(raw_theta / max_rad);

            if theta == 0.0 {
                continue;
            }

            let cos_t = cos_f32(theta);
            let sin_t = sin_f32(theta);

            for pair in head_q.chunks_exact_mut(2).skip(start_pair) {
                let q0 = pair[0];
                let q1 = pair[1];

                pair[0] = q0 * cos_t - q1 * sin_t;
                pair[1] = q0 * sin_t + q1 * cos_t;
            }
        }

        Ok(())
    }

    #[inline]
    fn qdps_filter_step(
        &self,
        phase_deltas: &mut AlignedSliceMut16<'_, gcso_q7_t>,
        min_step_rad: f32,
    ) -> GcsoResult<()> {
        if min_step_rad < 0.0 || min_step_rad.is_nan() || min_step_rad.is_infinite() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if phase_deltas.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let p_buf = phase_deltas.as_mut_slice();
        for delta in p_buf.iter_mut() {
            let rad = ((*delta as f32) * Q7_PHASE_SCALE).abs();
            if rad < min_step_rad {
                *delta = 0;
            }
        }

        Ok(())
    }

    #[inline]
    fn lazy_unwrap_override(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        context_accum: &AlignedSlice32<'_, f32>,
    ) -> GcsoResult<()> {
        self.validate_tensor_bounds(query_tensor.len(), context_accum.len())?;

        let head_dim = self.head_dim as usize;
        let num_heads = self.num_heads as usize;

        let q_buf = query_tensor.as_mut_slice();
        let c_buf = context_accum.as_slice();

        for (h, head_q) in q_buf.chunks_exact_mut(head_dim).take(num_heads).enumerate() {
            let theta = c_buf[h];
            if theta == 0.0 {
                continue;
            }

            let cos_t = cos_f32(theta);
            let sin_t = sin_f32(theta);

            for pair in head_q.chunks_exact_mut(2) {
                let q0 = pair[0];
                let q1 = pair[1];

                pair[0] = q0 * cos_t - q1 * sin_t;
                pair[1] = q0 * sin_t + q1 * cos_t;
            }
        }

        Ok(())
    }

    #[inline]
    fn slerp_norm_guard_stable(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        norm_lower: f32,
        norm_upper: f32,
    ) -> GcsoResult<()> {
        if norm_lower <= 0.0 || norm_upper < norm_lower || tensor.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let buf = tensor.as_mut_slice();
        let mut norm_sq = 0.0f32;
        for &val in buf.iter() {
            norm_sq += val * val;
        }

        if norm_sq.is_nan() || norm_sq.is_infinite() {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        let norm = sqrt_f32(norm_sq);

        // Zero-norm defense threshold
        if norm > 1e-12 {
            let scale = if norm < norm_lower {
                norm_lower / norm
            } else if norm > norm_upper {
                norm_upper / norm
            } else {
                1.0
            };

            if scale != 1.0 {
                for val in buf.iter_mut() {
                    *val *= scale;
                }
            }
        }

        Ok(())
    }

    #[inline]
    fn fused_logit_shift(
        &self,
        logits: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        if logits.is_empty() || phase_deltas.len() < (self.num_heads as usize) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let num_heads = self.num_heads as usize;
        let p_buf = phase_deltas.as_slice();

        let mut aggregate_phase_bias = 0.0f32;
        for &p in &p_buf[..num_heads] {
            aggregate_phase_bias += (p as f32) * Q7_PHASE_SCALE;
        }
        let mean_bias = aggregate_phase_bias / (num_heads as f32);

        for logit in logits.as_mut_slice().iter_mut() {
            *logit += mean_bias;
        }

        Ok(())
    }

    #[inline]
    fn compute_procrustes_phase_delta(
        &self,
        source: &AlignedSlice32<'_, f32>,
        target: &AlignedSlice32<'_, f32>,
        phase_out: &mut AlignedSliceMut16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        let expected_len = (self.head_dim as usize) * (self.num_heads as usize);
        if source.len() < expected_len
            || target.len() < expected_len
            || phase_out.len() < (self.num_heads as usize)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let head_dim = self.head_dim as usize;
        let num_heads = self.num_heads as usize;

        let src_buf = source.as_slice();
        let tgt_buf = target.as_slice();
        let out_buf = phase_out.as_mut_slice();

        let src_heads = src_buf.chunks_exact(head_dim).take(num_heads);
        let tgt_heads = tgt_buf.chunks_exact(head_dim).take(num_heads);

        for (h, (head_src, head_tgt)) in src_heads.zip(tgt_heads).enumerate() {
            let mut sum_sin = 0.0f32;
            let mut sum_cos = 0.0f32;

            for (pair_src, pair_tgt) in head_src.chunks_exact(2).zip(head_tgt.chunks_exact(2)) {
                let u0 = pair_src[0];
                let u1 = pair_src[1];
                let v0 = pair_tgt[0];
                let v1 = pair_tgt[1];

                // Compute cross-product (sin) and dot-product (cos) for 2D orientation
                sum_sin += u0 * v1 - u1 * v0;
                sum_cos += u0 * v0 + u1 * v1;
            }

            let angle = atan2_f32(sum_sin, sum_cos);
            let q_val = round_f32(angle * INV_Q7_PHASE_SCALE);
            let clamped_q = q_val.clamp(-128.0, 127.0) as i8;

            out_buf[h] = clamped_q;
        }

        Ok(())
    }
}

// ===================================================================
// Core Unit Tests for DpsrEngine
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dpsr_engine_creation() {
        assert!(DpsrEngine::new(128, 32).is_ok());
        assert_eq!(DpsrEngine::new(127, 32), Err(GCSO_ERROR_INVALID_ARGUMENT));
        assert_eq!(DpsrEngine::new(0, 32), Err(GCSO_ERROR_INVALID_ARGUMENT));
        assert_eq!(DpsrEngine::new(128, 0), Err(GCSO_ERROR_INVALID_ARGUMENT));
    }

    #[test]
    fn test_apply_phase_steering_identity() {
        #[repr(align(64))]
        struct AlignedQuery([f32; 256]);
        #[repr(align(16))]
        struct AlignedPhases([gcso_q7_t; 2]);

        let mut query = AlignedQuery([1.0; 256]);
        let phases = AlignedPhases([0; 2]);

        let engine = DpsrEngine::new(128, 2).unwrap();
        let mut q_slice = AlignedSliceMut32::new(&mut query.0).unwrap();
        let p_slice = AlignedSlice16::new(&phases.0).unwrap();

        engine.apply_phase_steering(&mut q_slice, &p_slice).unwrap();
        assert_eq!(q_slice[0], 1.0);
        assert_eq!(q_slice[1], 1.0);
    }

    #[test]
    fn test_apply_phase_steering_rotation_90_deg() {
        #[repr(align(64))]
        struct AlignedQuery([f32; 4]);
        #[repr(align(16))]
        struct AlignedPhases([gcso_q7_t; 1]);

        // [q0, q1] = [1.0, 0.0], pi/2 rotation in Q7 is 64
        let mut query = AlignedQuery([1.0, 0.0, 1.0, 0.0]);
        let phases = AlignedPhases([64]); // 64 * (pi/128) = pi/2

        let engine = DpsrEngine::new(4, 1).unwrap();
        let mut q_slice = AlignedSliceMut32::new(&mut query.0).unwrap();
        let p_slice = AlignedSlice16::new(&phases.0).unwrap();

        engine.apply_phase_steering(&mut q_slice, &p_slice).unwrap();

        // cos(pi/2) approx 0, sin(pi/2) approx 1 -> [0.0, 1.0]
        assert!(q_slice[0].abs() < 1e-5);
        assert!((q_slice[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_qdps_filter_step() {
        #[repr(align(16))]
        struct AlignedPhases([gcso_q7_t; 4]);
        let mut phases = AlignedPhases([1, 10, -1, 30]);

        let engine = DpsrEngine::new(128, 4).unwrap();
        let mut p_slice = AlignedSliceMut16::new(&mut phases.0).unwrap();

        // min_step_rad = 0.15 rad (~8.6 deg).
        // Q7=1 corresponds to ~0.0245 rad < 0.15 -> filtered to 0.
        // Q7=10 corresponds to ~0.245 rad > 0.15 -> kept.
        engine.qdps_filter_step(&mut p_slice, 0.15).unwrap();

        assert_eq!(p_slice[0], 0);
        assert_eq!(p_slice[1], 10);
        assert_eq!(p_slice[2], 0);
        assert_eq!(p_slice[3], 30);
    }

    #[test]
    fn test_slerp_norm_guard_stable() {
        #[repr(align(64))]
        struct AlignedBuffer([f32; 4]);
        let mut buf = AlignedBuffer([3.0, 4.0, 0.0, 0.0]); // norm = 5.0

        let engine = DpsrEngine::new(4, 1).unwrap();
        let mut slice = AlignedSliceMut32::new(&mut buf.0).unwrap();

        // Clamp norm to range [1.0, 2.5]
        engine
            .slerp_norm_guard_stable(&mut slice, 1.0, 2.5)
            .unwrap();

        let new_norm = sqrt_f32(slice[0] * slice[0] + slice[1] * slice[1]);
        assert!((new_norm - 2.5).abs() < 1e-5);
    }
}
