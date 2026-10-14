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
#![allow(clippy::chunks_exact_to_as_chunks)]
#![allow(clippy::implicit_saturating_sub)]
#![allow(clippy::doc_markdown)]

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

/// Cross-environment single-precision floating point ceiling function wrapper.
#[inline(always)]
fn ceil_f32(val: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        val.ceil()
    }
    #[cfg(not(feature = "std"))]
    {
        if val.is_nan() || val.is_infinite() {
            return val;
        }
        let ival = val as i32;
        if (ival as f32) < val {
            (ival + 1) as f32
        } else {
            ival as f32
        }
    }
}

/// Fused single-precision sine and cosine evaluator optimizing math functions across std and no_std.
///
/// Applies argument reduction followed by quadrant folding to `[-PI/2, PI/2]` and Horner's polynomial
/// evaluation for high numerical accuracy across the full range of floating-point values in `no_std`.
#[inline(always)]
fn sincos_f32(val: f32) -> (f32, f32) {
    #[cfg(feature = "std")]
    {
        val.sin_cos()
    }
    #[cfg(not(feature = "std"))]
    {
        if val.is_nan() || val.is_infinite() {
            return (0.0, 1.0);
        }

        // Argument reduction to [-PI, PI]
        let double_pi = 2.0 * PI;
        let k = round_f32(val / double_pi);
        let mut x = val - k * double_pi;
        if x > PI {
            x -= double_pi;
        } else if x < -PI {
            x += double_pi;
        }

        // Quadrant reduction to [-PI/2, PI/2] to guarantee high polynomial accuracy
        let (x_folded, sign_sin, sign_cos) = if x > PI * 0.5 {
            (PI - x, 1.0f32, -1.0f32)
        } else if x < -PI * 0.5 {
            (-PI - x, -1.0f32, -1.0f32)
        } else {
            (x, 1.0f32, 1.0f32)
        };

        let x2 = x_folded * x_folded;

        // Horner's scheme polynomial for sin(x) on [-PI/2, PI/2]
        let sin_poly = x_folded
            * (1.0
                + x2 * (-0.166_666_67
                    + x2 * (0.008_333_334
                        + x2 * (-0.000_198_412_7
                            + x2 * (0.000_002_755_732 + x2 * -0.000_000_025_052_108)))));

        // Horner's scheme polynomial for cos(x) on [-PI/2, PI/2]
        let cos_poly = 1.0
            + x2 * (-0.5
                + x2 * (0.041_666_668
                    + x2 * (-0.001_388_888_9
                        + x2 * (0.000_024_801_587 + x2 * -0.000_000_275_573_19))));

        (sign_sin * sin_poly, sign_cos * cos_poly)
    }
}

/// Cross-environment single-precision floating point hyperbolic tangent wrapper using Padé (5,4) approximation.
#[inline(always)]
fn tanh_f32(val: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        val.tanh()
    }
    #[cfg(not(feature = "std"))]
    {
        if val.is_nan() {
            return 0.0;
        }
        if val > 4.97 {
            1.0
        } else if val < -4.97 {
            -1.0
        } else {
            let x2 = val * val;
            let x4 = x2 * x2;
            let num = val * (945.0 + 105.0 * x2 + x4);
            let den = 945.0 + 420.0 * x2 + 15.0 * x4;
            num / den
        }
    }
}

/// Cross-environment single-precision floating point square root wrapper with Newton-Raphson iterations.
#[inline(always)]
fn sqrt_f32(val: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        val.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        if val <= 0.0 || val.is_nan() || val.is_infinite() {
            return 0.0;
        }
        let i = val.to_bits();
        // Bit-hack initial estimate for sqrt(val)
        let guess_bits = (i >> 1) + 0x1fbf_0000;
        let mut y = f32::from_bits(guess_bits);
        // 3 iterations of Newton-Raphson method for high numerical stability
        y = 0.5 * (y + val / y);
        y = 0.5 * (y + val / y);
        y = 0.5 * (y + val / y);
        y
    }
}

/// Cross-environment single-precision floating point atan2 wrapper using Remez polynomial approximation.
#[inline(always)]
fn atan2_f32(y: f32, x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        y.atan2(x)
    }
    #[cfg(not(feature = "std"))]
    {
        if x.is_nan() || y.is_nan() {
            return 0.0;
        }
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
                * (0.999_866_3 - 0.330_299_5 * r2 + 0.180_141 * r2 * r2 - 0.085_133 * r2 * r2 * r2);
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
        if head_dim == 0 || (head_dim % 2 != 0) || num_heads == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(Self {
            head_dim,
            num_heads,
        })
    }

    /// Helper method to validate tensor buffer lengths against configured head parameters safely.
    #[inline(always)]
    fn validate_tensor_bounds(&self, tensor_len: usize, phase_len: usize) -> GcsoResult<()> {
        let expected_tensor_len = (self.head_dim as usize)
            .checked_mul(self.num_heads as usize)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
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

            let (sin_t, cos_t) = sincos_f32(theta);

            // Process 4 pairs (8 floats) per iteration for SIMD vectorization & ILP
            let mut pairs_chunk = head_q.chunks_exact_mut(8);
            for chunk in pairs_chunk.by_ref() {
                let q0 = chunk[0];
                let q1 = chunk[1];
                let q2 = chunk[2];
                let q3 = chunk[3];
                let q4 = chunk[4];
                let q5 = chunk[5];
                let q6 = chunk[6];
                let q7 = chunk[7];

                chunk[0] = q0 * cos_t - q1 * sin_t;
                chunk[1] = q0 * sin_t + q1 * cos_t;
                chunk[2] = q2 * cos_t - q3 * sin_t;
                chunk[3] = q2 * sin_t + q3 * cos_t;
                chunk[4] = q4 * cos_t - q5 * sin_t;
                chunk[5] = q4 * sin_t + q5 * cos_t;
                chunk[6] = q6 * cos_t - q7 * sin_t;
                chunk[7] = q6 * sin_t + q7 * cos_t;
            }

            let remainder = pairs_chunk.into_remainder();
            for pair in remainder.chunks_exact_mut(2) {
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

            let (sin_t, cos_t) = sincos_f32(theta);

            // Process 4 pairs (8 floats) simultaneously for Query and Key for maximum memory bandwidth
            let mut q_chunks = head_q.chunks_exact_mut(8);
            let mut k_chunks = head_k.chunks_exact_mut(8);

            for (q_chunk, k_chunk) in q_chunks.by_ref().zip(k_chunks.by_ref()) {
                // Query rotation
                let q0 = q_chunk[0];
                let q1 = q_chunk[1];
                let q2 = q_chunk[2];
                let q3 = q_chunk[3];
                let q4 = q_chunk[4];
                let q5 = q_chunk[5];
                let q6 = q_chunk[6];
                let q7 = q_chunk[7];

                q_chunk[0] = q0 * cos_t - q1 * sin_t;
                q_chunk[1] = q0 * sin_t + q1 * cos_t;
                q_chunk[2] = q2 * cos_t - q3 * sin_t;
                q_chunk[3] = q2 * sin_t + q3 * cos_t;
                q_chunk[4] = q4 * cos_t - q5 * sin_t;
                q_chunk[5] = q4 * sin_t + q5 * cos_t;
                q_chunk[6] = q6 * cos_t - q7 * sin_t;
                q_chunk[7] = q6 * sin_t + q7 * cos_t;

                // Key rotation
                let k0 = k_chunk[0];
                let k1 = k_chunk[1];
                let k2 = k_chunk[2];
                let k3 = k_chunk[3];
                let k4 = k_chunk[4];
                let k5 = k_chunk[5];
                let k6 = k_chunk[6];
                let k7 = k_chunk[7];

                k_chunk[0] = k0 * cos_t - k1 * sin_t;
                k_chunk[1] = k0 * sin_t + k1 * cos_t;
                k_chunk[2] = k2 * cos_t - k3 * sin_t;
                k_chunk[3] = k2 * sin_t + k3 * cos_t;
                k_chunk[4] = k4 * cos_t - k5 * sin_t;
                k_chunk[5] = k4 * sin_t + k5 * cos_t;
                k_chunk[6] = k6 * cos_t - k7 * sin_t;
                k_chunk[7] = k6 * sin_t + k7 * cos_t;
            }

            let q_rem = q_chunks.into_remainder();
            let k_rem = k_chunks.into_remainder();
            for (pair_q, pair_k) in q_rem.chunks_exact_mut(2).zip(k_rem.chunks_exact_mut(2)) {
                let q0 = pair_q[0];
                let q1 = pair_q[1];
                pair_q[0] = q0 * cos_t - q1 * sin_t;
                pair_q[1] = q0 * sin_t + q1 * cos_t;

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
        let start_elem = start_pair * 2;

        let q_buf = query_tensor.as_mut_slice();
        let p_buf = phase_deltas.as_slice();

        for (h, head_q) in q_buf.chunks_exact_mut(head_dim).take(num_heads).enumerate() {
            let raw_theta = (p_buf[h] as f32) * Q7_PHASE_SCALE;
            if raw_theta == 0.0 {
                continue;
            }

            // RIPA soft clamping: theta_safe = max_rad * tanh(raw_theta / max_rad)
            let theta = max_rad * tanh_f32(raw_theta / max_rad);
            if theta == 0.0 {
                continue;
            }

            let (sin_t, cos_t) = sincos_f32(theta);

            let target_subslice = &mut head_q[start_elem..];
            let mut chunks = target_subslice.chunks_exact_mut(8);

            for chunk in chunks.by_ref() {
                let q0 = chunk[0];
                let q1 = chunk[1];
                let q2 = chunk[2];
                let q3 = chunk[3];
                let q4 = chunk[4];
                let q5 = chunk[5];
                let q6 = chunk[6];
                let q7 = chunk[7];

                chunk[0] = q0 * cos_t - q1 * sin_t;
                chunk[1] = q0 * sin_t + q1 * cos_t;
                chunk[2] = q2 * cos_t - q3 * sin_t;
                chunk[3] = q2 * sin_t + q3 * cos_t;
                chunk[4] = q4 * cos_t - q5 * sin_t;
                chunk[5] = q4 * sin_t + q5 * cos_t;
                chunk[6] = q6 * cos_t - q7 * sin_t;
                chunk[7] = q6 * sin_t + q7 * cos_t;
            }

            let remainder = chunks.into_remainder();
            for pair in remainder.chunks_exact_mut(2) {
                let q0 = pair[0];
                let q1 = pair[1];
                pair[0] = q0 * cos_t - q1 * sin_t;
                pair[1] = q0 * sin_t + q1 * cos_t;
            }
        }

        Ok(())
    }

    #[inline]
    fn apply_soft_phase_damping(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        damping_factor: f32,
    ) -> GcsoResult<()> {
        if damping_factor < 0.0 || damping_factor.is_nan() || damping_factor.is_infinite() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        if (damping_factor - 1.0).abs() < 1e-6 {
            return Ok(());
        }

        let q_buf = query_tensor.as_mut_slice();
        let expected_q_len = (self.head_dim as usize)
            .checked_mul(self.num_heads as usize)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;

        if q_buf.len() < expected_q_len {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let mut chunks = q_buf[..expected_q_len].chunks_exact_mut(8);
        for chunk in chunks.by_ref() {
            chunk[0] *= damping_factor;
            chunk[1] *= damping_factor;
            chunk[2] *= damping_factor;
            chunk[3] *= damping_factor;
            chunk[4] *= damping_factor;
            chunk[5] *= damping_factor;
            chunk[6] *= damping_factor;
            chunk[7] *= damping_factor;
        }

        let rem = chunks.into_remainder();
        for val in rem {
            *val *= damping_factor;
        }

        Ok(())
    }

    /// QDPS discrete filter: cuts off phase rotation steps falling below `min_step_rad`.
    ///
    /// Optimized to execute entirely in the integer Q7 domain without floating-point
    /// conversions in Hot Path loops. Uses `ceil_f32` helper for full no_std compliance.
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

        // Pre-calculate minimum Q7 integer threshold to execute integer-domain comparison safely in no_std
        let min_step_q7_raw = ceil_f32(min_step_rad / Q7_PHASE_SCALE) as i32;
        if min_step_q7_raw <= 0 {
            return Ok(());
        }

        // If threshold exceeds the Q7 maximum range (127), clear all elements to 0
        if min_step_q7_raw > 127 {
            p_buf[..num_heads].fill(0);
            return Ok(());
        }

        let min_step_q7 = min_step_q7_raw;

        for delta in p_buf.iter_mut().take(num_heads) {
            // Safely compute absolute value via i32 cast to avoid i8::abs(-128) panic in debug builds
            if (*delta as i32).abs() < min_step_q7 {
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
        let head_dim = self.head_dim as usize;
        let num_heads = self.num_heads as usize;
        let expected_q_len = head_dim * num_heads;

        if query_tensor.len() < expected_q_len || context_accum.len() < num_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let q_buf = query_tensor.as_mut_slice();
        let c_buf = context_accum.as_slice();

        for (h, head_q) in q_buf.chunks_exact_mut(head_dim).take(num_heads).enumerate() {
            let theta = c_buf[h];
            if theta == 0.0 {
                continue;
            }

            let (sin_t, cos_t) = sincos_f32(theta);

            let mut chunks = head_q.chunks_exact_mut(8);
            for chunk in chunks.by_ref() {
                let q0 = chunk[0];
                let q1 = chunk[1];
                let q2 = chunk[2];
                let q3 = chunk[3];
                let q4 = chunk[4];
                let q5 = chunk[5];
                let q6 = chunk[6];
                let q7 = chunk[7];

                chunk[0] = q0 * cos_t - q1 * sin_t;
                chunk[1] = q0 * sin_t + q1 * cos_t;
                chunk[2] = q2 * cos_t - q3 * sin_t;
                chunk[3] = q2 * sin_t + q3 * cos_t;
                chunk[4] = q4 * cos_t - q5 * sin_t;
                chunk[5] = q4 * sin_t + q5 * cos_t;
                chunk[6] = q6 * cos_t - q7 * sin_t;
                chunk[7] = q6 * sin_t + q7 * cos_t;
            }

            let remainder = chunks.into_remainder();
            for pair in remainder.chunks_exact_mut(2) {
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
        if norm_lower > norm_upper || norm_lower < 0.0 || norm_lower.is_nan() || norm_upper.is_nan()
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let buf = tensor.as_mut_slice();
        let mut sum_sq0 = 0.0f32;
        let mut sum_sq1 = 0.0f32;
        let mut sum_sq2 = 0.0f32;
        let mut sum_sq3 = 0.0f32;

        let chunks = buf.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            sum_sq0 += chunk[0] * chunk[0];
            sum_sq1 += chunk[1] * chunk[1];
            sum_sq2 += chunk[2] * chunk[2];
            sum_sq3 += chunk[3] * chunk[3];
            sum_sq0 += chunk[4] * chunk[4];
            sum_sq1 += chunk[5] * chunk[5];
            sum_sq2 += chunk[6] * chunk[6];
            sum_sq3 += chunk[7] * chunk[7];
        }

        let mut sum_sq = (sum_sq0 + sum_sq1) + (sum_sq2 + sum_sq3);
        for &v in remainder {
            sum_sq += v * v;
        }

        if sum_sq.is_nan() || sum_sq.is_infinite() {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        let current_norm = sqrt_f32(sum_sq);

        // Zero-norm defense: skip rescaling if norm is virtually zero to prevent division by zero
        if current_norm < 1e-12 {
            return Ok(());
        }

        let scale = if current_norm < norm_lower {
            norm_lower / current_norm
        } else if current_norm > norm_upper {
            norm_upper / current_norm
        } else {
            1.0
        };

        if (scale - 1.0).abs() > 1e-6 {
            let mut chunks_mut = buf.chunks_exact_mut(8);
            for chunk in chunks_mut.by_ref() {
                chunk[0] *= scale;
                chunk[1] *= scale;
                chunk[2] *= scale;
                chunk[3] *= scale;
                chunk[4] *= scale;
                chunk[5] *= scale;
                chunk[6] *= scale;
                chunk[7] *= scale;
            }
            let rem = chunks_mut.into_remainder();
            for v in rem {
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
        let num_heads = self.num_heads as usize;
        if logits.is_empty() || phase_deltas.len() < num_heads || num_heads == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let l_buf = logits.as_mut_slice();
        let p_buf = phase_deltas.as_slice();

        let mut sum_phase = 0.0f32;

        for &p in p_buf.iter().take(num_heads) {
            sum_phase += (p as f32) * Q7_PHASE_SCALE;
        }

        let avg_phase_shift = sum_phase / (num_heads as f32);

        // Process 8 elements per iteration for FMA vectorization
        let mut chunks = l_buf.chunks_exact_mut(8);

        for chunk in chunks.by_ref() {
            chunk[0] += avg_phase_shift;
            chunk[1] += avg_phase_shift;
            chunk[2] += avg_phase_shift;
            chunk[3] += avg_phase_shift;
            chunk[4] += avg_phase_shift;
            chunk[5] += avg_phase_shift;
            chunk[6] += avg_phase_shift;
            chunk[7] += avg_phase_shift;
        }

        let remainder = chunks.into_remainder();
        for logit in remainder {
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
        let expected_tensor_len = head_dim
            .checked_mul(num_heads)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;

        if src_slice.len() < expected_tensor_len
            || tgt_slice.len() < expected_tensor_len
            || out_slice.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        for h in 0..num_heads {
            let src_head = &src_slice[h * head_dim..(h + 1) * head_dim];
            let tgt_head = &tgt_slice[h * head_dim..(h + 1) * head_dim];

            let mut sum_cos0 = 0.0f32;
            let mut sum_cos1 = 0.0f32;
            let mut sum_sin0 = 0.0f32;
            let mut sum_sin1 = 0.0f32;

            let mut pair_chunks = src_head.chunks_exact(8);
            let mut tgt_chunks = tgt_head.chunks_exact(8);

            for (s_chunk, t_chunk) in pair_chunks.by_ref().zip(tgt_chunks.by_ref()) {
                let u0_x = s_chunk[0];
                let u0_y = s_chunk[1];
                let v0_x = t_chunk[0];
                let v0_y = t_chunk[1];
                sum_cos0 += u0_x * v0_x + u0_y * v0_y;
                sum_sin0 += u0_x * v0_y - u0_y * v0_x;

                let u1_x = s_chunk[2];
                let u1_y = s_chunk[3];
                let v1_x = t_chunk[2];
                let v1_y = t_chunk[3];
                sum_cos1 += u1_x * v1_x + u1_y * v1_y;
                sum_sin1 += u1_x * v1_y - u1_y * v1_x;

                let u2_x = s_chunk[4];
                let u2_y = s_chunk[5];
                let v2_x = t_chunk[4];
                let v2_y = t_chunk[5];
                sum_cos0 += u2_x * v2_x + u2_y * v2_y;
                sum_sin0 += u2_x * v2_y - u2_y * v2_x;

                let u3_x = s_chunk[6];
                let u3_y = s_chunk[7];
                let v3_x = t_chunk[6];
                let v3_y = t_chunk[7];
                sum_cos1 += u3_x * v3_x + u3_y * v3_y;
                sum_sin1 += u3_x * v3_y - u3_y * v3_x;
            }

            let s_rem = pair_chunks.remainder();
            let t_rem = tgt_chunks.remainder();
            for (s_pair, t_pair) in s_rem.chunks_exact(2).zip(t_rem.chunks_exact(2)) {
                let u_x = s_pair[0];
                let u_y = s_pair[1];
                let v_x = t_pair[0];
                let v_y = t_pair[1];
                sum_cos0 += u_x * v_x + u_y * v_y;
                sum_sin0 += u_x * v_y - u_y * v_x;
            }

            let sum_cos = sum_cos0 + sum_cos1;
            let sum_sin = sum_sin0 + sum_sin1;

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

        #[repr(align(16))]
        struct AlignedPhases([gcso_q7_t; 16]);

        let mut raw_buf = AlignedPhases([1, 5, -2, 10, -128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut aligned = AlignedSliceMut16::new(&mut raw_buf.0).unwrap();

        // 5 * Q7_PHASE_SCALE is ~0.122 rad
        assert!(engine.qdps_filter_step(&mut aligned, 0.1).is_ok());

        let result = aligned.as_ref();
        assert_eq!(result[0], 0); // Filtered out (< 0.1 rad, Q7 threshold = 5)
        assert_eq!(result[1], 5); // Retained
        assert_eq!(result[2], 0); // Filtered out
        assert_eq!(result[3], 10); // Retained
        assert_eq!(result[4], -128); // Retained (-128_i8 safely handled without overflow panic)

        // Threshold exceeding max Q7 range (127) -> Should clear all elements to 0
        assert!(engine.qdps_filter_step(&mut aligned, 4.0).is_ok());
        assert_eq!(aligned.as_ref()[1], 0);
        assert_eq!(aligned.as_ref()[3], 0);
    }

    #[test]
    fn test_lazy_unwrap_override_so2_rotation() {
        let engine = DpsrEngine::new(4, 1).unwrap();

        #[repr(align(32))]
        struct AlignedQ([f32; 4]);
        #[repr(align(32))]
        struct AlignedC([f32; 1]);

        let mut raw_q = AlignedQ([1.0, 0.0, 0.0, 1.0]); // Two 2D pairs: (1,0) and (0,1)
        let raw_c = AlignedC([PI * 0.5]); // Rotate by +90 degrees

        let mut q_slice = AlignedSliceMut32::new(&mut raw_q.0).unwrap();
        let c_slice = AlignedSlice32::new(&raw_c.0).unwrap();

        assert!(engine.lazy_unwrap_override(&mut q_slice, &c_slice).is_ok());

        let res = q_slice.as_ref();
        // (1,0) rotated by 90 deg -> (0, 1)
        assert!((res[0] - 0.0).abs() < 1e-4);
        assert!((res[1] - 1.0).abs() < 1e-4);
        // (0,1) rotated by 90 deg -> (-1, 0)
        assert!((res[2] - (-1.0)).abs() < 1e-4);
        assert!((res[3] - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_apply_soft_phase_damping() {
        let engine = DpsrEngine::new(4, 1).unwrap();

        #[repr(align(32))]
        struct AlignedQ([f32; 4]);

        let mut raw_q = AlignedQ([2.0, 4.0, -6.0, 8.0]);
        let mut q_slice = AlignedSliceMut32::new(&mut raw_q.0).unwrap();

        assert!(engine.apply_soft_phase_damping(&mut q_slice, 0.5).is_ok());

        let res = q_slice.as_ref();
        assert_eq!(res[0], 1.0);
        assert_eq!(res[1], 2.0);
        assert_eq!(res[2], -3.0);
        assert_eq!(res[3], 4.0);

        assert_eq!(
            engine.apply_soft_phase_damping(&mut q_slice, -0.1),
            Err(GCSO_ERROR_INVALID_ARGUMENT)
        );
    }

    #[test]
    fn test_slerp_norm_guard_equal_bounds() {
        let engine = DpsrEngine::new(128, 1).unwrap();

        #[repr(align(32))]
        struct AlignedTensor([f32; 128]);

        let mut raw_t = AlignedTensor([2.0; 128]);
        let mut t_slice = AlignedSliceMut32::new(&mut raw_t.0).unwrap();

        // Allow norm_lower == norm_upper (unit normalization)
        assert!(engine
            .slerp_norm_guard_stable(&mut t_slice, 1.0, 1.0)
            .is_ok());

        let mut sum_sq = 0.0f32;
        for &val in t_slice.as_ref() {
            sum_sq += val * val;
        }
        let norm = sum_sq.sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sincos_f32_quadrant_folding_accuracy() {
        // Test sincos_f32 near PI, -PI, PI/2, -PI/2
        let (s_pi, c_pi) = sincos_f32(PI);
        assert!(s_pi.abs() < 1e-5);
        assert!((c_pi - (-1.0)).abs() < 1e-5);

        let (s_neg_pi, c_neg_pi) = sincos_f32(-PI);
        assert!(s_neg_pi.abs() < 1e-5);
        assert!((c_neg_pi - (-1.0)).abs() < 1e-5);

        let (s_half_pi, c_half_pi) = sincos_f32(PI * 0.5);
        assert!((s_half_pi - 1.0).abs() < 1e-5);
        assert!(c_half_pi.abs() < 1e-5);
    }

    #[test]
    fn test_fused_logit_shift() {
        let engine = DpsrEngine::new(64, 2).unwrap();

        #[repr(align(32))]
        struct AlignedLogits([f32; 8]);
        #[repr(align(16))]
        struct AlignedPhases([gcso_q7_t; 16]);

        let mut raw_logits = AlignedLogits([1.0; 8]);
        // Phase values 32, 64 -> (32 + 64)/2 = 48 Q7 steps -> 48 * PI / 128 = 0.375 * PI rad
        let raw_phases = AlignedPhases([32, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

        let mut logits_slice = AlignedSliceMut32::new(&mut raw_logits.0).unwrap();
        let phases_slice = AlignedSlice16::new(&raw_phases.0).unwrap();

        assert!(engine
            .fused_logit_shift(&mut logits_slice, &phases_slice)
            .is_ok());

        let expected_shift = (48.0 * PI) / 128.0;
        for &val in logits_slice.as_ref() {
            assert!((val - (1.0 + expected_shift)).abs() < 1e-4);
        }
    }

    #[test]
    fn test_compute_procrustes_phase_delta() {
        let engine = DpsrEngine::new(4, 1).unwrap();

        #[repr(align(32))]
        struct AlignedVec([f32; 4]);
        #[repr(align(16))]
        struct AlignedOut([gcso_q7_t; 16]);

        // Source = (1, 0, 0, 1), Target = (0, 1, -1, 0) -> 90 degree rotation (PI/2 = 64 Q7 steps)
        let src_buf = AlignedVec([1.0, 0.0, 0.0, 1.0]);
        let tgt_buf = AlignedVec([0.0, 1.0, -1.0, 0.0]);
        let mut out_buf = AlignedOut([0; 16]);

        let src_slice = AlignedSlice32::new(&src_buf.0).unwrap();
        let tgt_slice = AlignedSlice32::new(&tgt_buf.0).unwrap();
        let mut out_slice = AlignedSliceMut16::new(&mut out_buf.0).unwrap();

        assert!(engine
            .compute_procrustes_phase_delta(&src_slice, &tgt_slice, &mut out_slice)
            .is_ok());

        assert_eq!(out_slice.as_ref()[0], 64);
    }
}
