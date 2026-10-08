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
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::chunks_exact_to_as_chunks)]
#![allow(clippy::implicit_saturating_sub)]

#[cfg(not(feature = "std"))]
use core::f32::consts::PI;
#[cfg(feature = "std")]
use std::f32::consts::PI;

use crate::abi::{gcso_q7_t, GCSO_ERROR_EDBC_SINGULARITY, GCSO_ERROR_INVALID_ARGUMENT};
use crate::traits::{
    AlignedSlice16, AlignedSlice32, AlignedSliceMut16, AlignedSliceMut32, GcsoResult, PhaseSteering,
};

/// Scale factor for Q7 quantized phase representation (beta_Q7 = pi / 128).
const Q7_PHASE_SCALE: f32 = PI / 128.0;

/// Inverse scale factor to map radians back to Q7 fixed-point representation.
const INV_Q7_PHASE_SCALE: f32 = 128.0 / PI;

/// Cross-environment single-precision floating point sine function wrapper (zero external crates).
#[inline(always)]
fn sin_f32(val: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        val.sin()
    }
    #[cfg(not(feature = "std"))]
    {
        let mut x = val % (2.0 * PI);
        if x > PI {
            x -= 2.0 * PI;
        } else if x < -PI {
            x += 2.0 * PI;
        }
        let x2 = x * x;
        let x3 = x * x2;
        let x5 = x3 * x2;
        let x7 = x5 * x2;
        x - (x3 / 6.0) + (x5 / 120.0) - (x7 / 5040.0)
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
        sin_f32(val + (PI * 0.5))
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
        if val > 4.0 {
            1.0
        } else if val < -4.0 {
            -1.0
        } else {
            let x2 = val * val;
            let num = val * (27.0 + x2);
            let den = 27.0 + 9.0 * x2;
            num / den
        }
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
        if val <= 0.0 {
            return 0.0;
        }
        let mut x = val;
        for _ in 0..6 {
            x = 0.5 * (x + val / x);
        }
        x
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
        if val >= 0.0 {
            (val + 0.5) as i32 as f32
        } else {
            (val - 0.5) as i32 as f32
        }
    }
}

/// Cross-environment single-precision floating point atan2 wrapper.
#[inline(always)]
fn atan2_f32(y: f32, x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        y.atan2(x)
    }
    #[cfg(not(feature = "std"))]
    {
        if x == 0.0 {
            if y > 0.0 {
                PI * 0.5
            } else if y < 0.0 {
                -PI * 0.5
            } else {
                0.0
            }
        } else {
            let ax = if x < 0.0 { -x } else { x };
            let ay = if y < 0.0 { -y } else { y };
            let (min_val, max_val) = if ax < ay { (ax, ay) } else { (ay, ax) };
            let ratio = min_val / max_val;
            let r2 = ratio * ratio;
            let mut angle = ratio
                * (0.999_866_3 - 0.330_299_5 * r2 + 0.180_141 * r2 * r2
                    - 0.085_133_ * r2 * r2 * r2);
            if ax < ay {
                angle = (PI * 0.5) - angle;
            }
            if x < 0.0 {
                angle = PI - angle;
            }
            if y < 0.0 {
                angle = -angle;
            }
            angle
        }
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
        if head_dim == 0 || !head_dim.is_multiple_of(2) || num_heads == 0 {
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

            let (pairs, _) = head_q.as_chunks_mut::<2>();
            for pair in pairs {
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

            let (pairs_q, _) = head_q.as_chunks_mut::<2>();
            let (pairs_k, _) = head_k.as_chunks_mut::<2>();

            for (pair_q, pair_k) in pairs_q.iter_mut().zip(pairs_k.iter_mut()) {
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
        let start_pair = pairs_per_head.saturating_sub(low_freq_pairs);

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

            let (pairs, _) = head_q.as_chunks_mut::<2>();

            for pair in pairs.iter_mut().skip(start_pair) {
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

        let p_buf = phase_deltas.as_mut_slice();
        let num_heads = (self.num_heads as usize).min(p_buf.len());

        let min_step_q7 = round_f32(min_step_rad * INV_Q7_PHASE_SCALE) as i32;

        for delta in p_buf.iter_mut().take(num_heads) {
            let val = i32::from(*delta);
            if val.abs() < min_step_q7 {
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
        let expected_len = (self.head_dim as usize) * (self.num_heads as usize);
        if query_tensor.len() < expected_len || context_accum.len() < expected_len {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let q_buf = query_tensor.as_mut_slice();
        let c_buf = context_accum.as_slice();

        for (q, c) in q_buf.iter_mut().zip(c_buf.iter()) {
            *q += *c;
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
        if norm_lower >= norm_upper
            || norm_lower <= 0.0
            || norm_lower.is_nan()
            || norm_upper.is_nan()
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let buf = tensor.as_mut_slice();
        let mut sum_sq = 0.0f32;

        for v in buf.iter() {
            sum_sq += v * v;
        }

        if sum_sq.is_nan() || sum_sq.is_infinite() {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        let current_norm = sqrt_f32(sum_sq);

        if current_norm < 1e-12 {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        let scale = if current_norm < norm_lower {
            norm_lower / current_norm
        } else if current_norm > norm_upper {
            norm_upper / current_norm
        } else {
            1.0
        };

        if scale != 1.0 {
            for v in buf.iter_mut() {
                *v *= scale;
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

        let l_buf = logits.as_mut_slice();
        let p_buf = phase_deltas.as_slice();

        let num_heads = self.num_heads as usize;
        let mut sum_phase = 0.0f32;

        for &p in p_buf.iter().take(num_heads) {
            sum_phase += (p as f32) * Q7_PHASE_SCALE;
        }

        let avg_phase_shift = sum_phase / (num_heads as f32);

        for logit in l_buf.iter_mut() {
            *logit += avg_phase_shift;
        }

        Ok(())
    }

    /// Computes the Orthogonal Procrustes phase delta in SO(2)^(d/2) subspace.
    /// Maps continuous 2D vector rotations to Q7 quantized phase representation.
    #[inline]
    fn compute_procrustes_phase_delta(
        &self,
        source: &AlignedSlice32<'_, f32>,
        target: &AlignedSlice32<'_, f32>,
        phase_out: &mut AlignedSliceMut16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        let src_slice = source.as_slice();
        let tgt_slice = target.as_slice();
        let out_slice = phase_out.as_mut_slice();

        let head_dim = self.head_dim as usize;
        let num_heads = self.num_heads as usize;
        let expected_tensor_len = head_dim * num_heads;

        if src_slice.len() < expected_tensor_len
            || tgt_slice.len() < expected_tensor_len
            || out_slice.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        for h in 0..num_heads {
            let src_head = &src_slice[h * head_dim..(h + 1) * head_dim];
            let tgt_head = &tgt_slice[h * head_dim..(h + 1) * head_dim];

            let mut sum_cos = 0.0f32;
            let mut sum_sin = 0.0f32;

            for pair_idx in 0..(head_dim / 2) {
                let u_x = src_head[pair_idx * 2];
                let u_y = src_head[pair_idx * 2 + 1];
                let v_x = tgt_head[pair_idx * 2];
                let v_y = tgt_head[pair_idx * 2 + 1];

                // Inner product and determinant for SO(2) phase delta: arg(v * conj(u))
                sum_cos += u_x * v_x + u_y * v_y;
                sum_sin += u_x * v_y - u_y * v_x;
            }

            let norm_sq = sum_cos * sum_cos + sum_sin * sum_sin;
            if norm_sq > 1e-12 {
                let theta = atan2_f32(sum_sin, sum_cos);
                let q7_val = round_f32((theta / PI) * 128.0) as i32;
                out_slice[h] = q7_val.clamp(-128, 127) as i8;
            } else {
                out_slice[h] = 0;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dpsr_engine_creation() {
        assert!(DpsrEngine::new(128, 32).is_ok());
        assert!(DpsrEngine::new(127, 32).is_err()); // Odd head dimension
        assert!(DpsrEngine::new(0, 32).is_err());
        assert!(DpsrEngine::new(128, 0).is_err());
    }

    #[test]
    fn test_qdps_filter() {
        let engine = DpsrEngine::new(128, 4).unwrap();

        // Enforce 16-byte alignment using repr(align(16))
        #[repr(align(16))]
        struct AlignedPhases([gcso_q7_t; 16]);

        let mut raw_buf = AlignedPhases([1, 5, -2, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut aligned = AlignedSliceMut16::new(&mut raw_buf.0).unwrap();

        // 5 * Q7_PHASE_SCALE is ~0.122 rad
        assert!(engine.qdps_filter_step(&mut aligned, 0.1).is_ok());

        let result = aligned.as_ref();
        assert_eq!(result[0], 0); // Filtered out (< 0.1 rad)
        assert_eq!(result[1], 5); // Retained
        assert_eq!(result[2], 0); // Filtered out
        assert_eq!(result[3], 10); // Retained
    }
}
