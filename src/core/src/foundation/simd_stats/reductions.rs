// name: src/core/src/foundation/simd_stats/reductions.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Scalar & AVX2 constant-time norm, reduction, scaling, and Chebyshev deg-3 kernels.

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::*;

use crate::foundation::lut_crc32::{F32_MIN_POSITIVE_NORMAL, SUBNORMAL_SQ_RISK_FLOOR};
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use crate::foundation::phase_ops::{avx2_hsum_ps, avx2_hsum2_ps};
use crate::foundation::scalar_math::{
    branchless_abs_f32, branchless_clamp_f32, branchless_flush_subnormal_f32,
    branchless_is_finite_f32, branchless_max_f32, branchless_relu_f32, branchless_sqrt_f32,
};
use crate::foundation::swar_simd::sanitize_inplace_gain_f32;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use crate::liminal::branchless_select_f32;
use crate::liminal::branchless_select_u32;

/// Evaluates $(\sum_i |x_i|, \sum_i x_i^2)$ across `slice` using a 32-float (2-pair tree-reduced)
/// AVX2 SIMD kernel that computes `a = |x|` once per vector and uses $a^2 \equiv x^2$ to eliminate
/// Port 0/1 accumulator contention.
#[inline]
pub fn sum_abs_and_sq_f32_simd8(slice: &[f32]) -> (f32, f32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let len = slice.len();
        let ptr = slice.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and all loads are bounds-checked against `len`.
        // Peak live YMM usage is capped at 5 registers (`sign_mask`, `abs_all`, `sq_all`, `a0`, `a1` <= 6)
        // to avoid non-volatile `xmm6..xmm15` stack saves on Windows x64 ABI, and remainder elements
        // use loop-free binary tail steps (`32`, `16`, `8`, `4`, `2`, `1`).
        unsafe {
            let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
            let mut abs_all = _mm256_setzero_ps();
            let mut sq_all = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 64 <= len {
                let a0 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i)), sign_mask);
                let a1 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 8)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, _mm256_add_ps(a0, a1));
                sq_all = _mm256_add_ps(
                    sq_all,
                    _mm256_add_ps(_mm256_mul_ps(a0, a0), _mm256_mul_ps(a1, a1)),
                );
                let a2 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 16)), sign_mask);
                let a3 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 24)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, _mm256_add_ps(a2, a3));
                sq_all = _mm256_add_ps(
                    sq_all,
                    _mm256_add_ps(_mm256_mul_ps(a2, a2), _mm256_mul_ps(a3, a3)),
                );
                let a4 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 32)), sign_mask);
                let a5 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 40)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, _mm256_add_ps(a4, a5));
                sq_all = _mm256_add_ps(
                    sq_all,
                    _mm256_add_ps(_mm256_mul_ps(a4, a4), _mm256_mul_ps(a5, a5)),
                );
                let a6 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 48)), sign_mask);
                let a7 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 56)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, _mm256_add_ps(a6, a7));
                sq_all = _mm256_add_ps(
                    sq_all,
                    _mm256_add_ps(_mm256_mul_ps(a6, a6), _mm256_mul_ps(a7, a7)),
                );
                i += 64;
            }
            let rem = len - i;
            if (rem & 32) != 0 {
                let a0 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i)), sign_mask);
                let a1 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 8)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, _mm256_add_ps(a0, a1));
                sq_all = _mm256_add_ps(
                    sq_all,
                    _mm256_add_ps(_mm256_mul_ps(a0, a0), _mm256_mul_ps(a1, a1)),
                );
                let a2 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 16)), sign_mask);
                let a3 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 24)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, _mm256_add_ps(a2, a3));
                sq_all = _mm256_add_ps(
                    sq_all,
                    _mm256_add_ps(_mm256_mul_ps(a2, a2), _mm256_mul_ps(a3, a3)),
                );
                i += 32;
            }
            if (rem & 16) != 0 {
                let a0 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i)), sign_mask);
                let a1 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 8)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, _mm256_add_ps(a0, a1));
                sq_all = _mm256_add_ps(
                    sq_all,
                    _mm256_add_ps(_mm256_mul_ps(a0, a0), _mm256_mul_ps(a1, a1)),
                );
                i += 16;
            }
            if (rem & 8) != 0 {
                let a0 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, a0);
                sq_all = _mm256_add_ps(sq_all, _mm256_mul_ps(a0, a0));
                i += 8;
            }
            let (mut sum_abs, mut sum_sq) = avx2_hsum2_ps(abs_all, sq_all);
            if (rem & 4) != 0 {
                let v0 = branchless_abs_f32(*ptr.add(i));
                let v1 = branchless_abs_f32(*ptr.add(i + 1));
                let v2 = branchless_abs_f32(*ptr.add(i + 2));
                let v3 = branchless_abs_f32(*ptr.add(i + 3));
                sum_abs += (v0 + v1) + (v2 + v3);
                sum_sq += (v0 * v0 + v1 * v1) + (v2 * v2 + v3 * v3);
                i += 4;
            }
            if (rem & 2) != 0 {
                let v0 = branchless_abs_f32(*ptr.add(i));
                let v1 = branchless_abs_f32(*ptr.add(i + 1));
                sum_abs += v0 + v1;
                sum_sq += v0 * v0 + v1 * v1;
                i += 2;
            }
            if (rem & 1) != 0 {
                let v0 = branchless_abs_f32(*ptr.add(i));
                sum_abs += v0;
                sum_sq += v0 * v0;
            }
            (sum_abs, sum_sq)
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut acc_abs = [0.0_f32; 4];
        let mut acc_sq = [0.0_f32; 4];
        let (chunks, rem) = slice.as_chunks::<4>();
        for c in chunks {
            for lane in 0..4 {
                let v = branchless_abs_f32(c[lane]);
                acc_abs[lane] += v;
                acc_sq[lane] += v * v;
            }
        }
        let mut sum_abs = (acc_abs[0] + acc_abs[1]) + (acc_abs[2] + acc_abs[3]);
        let mut sum_sq = (acc_sq[0] + acc_sq[1]) + (acc_sq[2] + acc_sq[3]);
        for &v in rem {
            let a = branchless_abs_f32(v);
            sum_abs += a;
            sum_sq += a * a;
        }
        (sum_abs, sum_sq)
    }
}

/// Evaluates $(\sum_i |x_i|, \sum_i x_i^2, \max_i |x_i|)$ in a **single memory pass** across `slice`
/// using 32-float unrolled AVX2 SIMD accumulators while keeping peak live YMM usage `<= 6`
/// (`0` Windows x64 `xmm6..xmm15` stack saves/restores).
#[inline]
pub fn sum_abs_sq_and_max_f32_simd8(slice: &[f32]) -> (f32, f32, f32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let len = slice.len();
        let ptr = slice.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and all loads are bounds-checked against `len`.
        // Processing `a0, a1` then `a2, a3` into 3 accumulators (`abs_all, sq_all, max_all`) keeps peak
        // live YMM registers `<= 6` (volatile `ymm0..ymm5`), eliminating 5 `vmovaps` stack spills on Windows x64.
        unsafe {
            let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
            let mut abs_all = _mm256_setzero_ps();
            let mut sq_all = _mm256_setzero_ps();
            let mut max_all = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 32 <= len {
                let a0 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i)), sign_mask);
                let a1 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 8)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, _mm256_add_ps(a0, a1));
                sq_all = _mm256_add_ps(
                    sq_all,
                    _mm256_add_ps(_mm256_mul_ps(a0, a0), _mm256_mul_ps(a1, a1)),
                );
                max_all = _mm256_max_ps(max_all, _mm256_max_ps(a0, a1));

                let a2 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 16)), sign_mask);
                let a3 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 24)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, _mm256_add_ps(a2, a3));
                sq_all = _mm256_add_ps(
                    sq_all,
                    _mm256_add_ps(_mm256_mul_ps(a2, a2), _mm256_mul_ps(a3, a3)),
                );
                max_all = _mm256_max_ps(max_all, _mm256_max_ps(a2, a3));
                i += 32;
            }
            let rem = len - i;
            if (rem & 16) != 0 {
                let a0 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i)), sign_mask);
                let a1 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 8)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, _mm256_add_ps(a0, a1));
                sq_all = _mm256_add_ps(
                    sq_all,
                    _mm256_add_ps(_mm256_mul_ps(a0, a0), _mm256_mul_ps(a1, a1)),
                );
                max_all = _mm256_max_ps(max_all, _mm256_max_ps(a0, a1));
                i += 16;
            }
            if (rem & 8) != 0 {
                let a0 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, a0);
                sq_all = _mm256_add_ps(sq_all, _mm256_mul_ps(a0, a0));
                max_all = _mm256_max_ps(max_all, a0);
                i += 8;
            }
            let (mut sum_abs, mut sum_sq) = avx2_hsum2_ps(abs_all, sq_all);
            let m128 = _mm_max_ps(
                _mm256_castps256_ps128(max_all),
                _mm256_extractf128_ps::<1>(max_all),
            );
            let m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
            let mut max_abs = _mm_cvtss_f32(_mm_max_ss(m64, _mm_shuffle_ps::<0x55>(m64, m64)));
            while i < len {
                let a = branchless_abs_f32(*ptr.add(i));
                sum_abs += a;
                sum_sq += a * a;
                max_abs = branchless_max_f32(max_abs, a);
                i += 1;
            }
            (sum_abs, sum_sq, max_abs)
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut sum_abs = 0.0_f32;
        let mut sum_sq = 0.0_f32;
        let mut max_abs = 0.0_f32;
        for &v in slice {
            let a = branchless_abs_f32(v);
            sum_abs += a;
            sum_sq += a * a;
            max_abs = branchless_max_f32(max_abs, a);
        }
        (sum_abs, sum_sq, max_abs)
    }
}

/// Evaluates $\sum_i x_i^2$ across `slice` using a 32-float (4-accumulator) unrolled AVX2 tree reduction.
#[inline]
pub fn sum_sq_f32_simd8(slice: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let len = slice.len();
        let ptr = slice.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and all loads are bounds-checked against `len`.
        // Peak live YMM usage is capped at 4 registers (`sq0`, `sq1`, `v0`, `v1` <= 6) with loop-free
        // binary tail steps (`32`, `16`, `8`, `4`, `2`, `1`).
        unsafe {
            let mut sq0 = _mm256_setzero_ps();
            let mut sq1 = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 64 <= len {
                let v0 = _mm256_loadu_ps(ptr.add(i));
                let v1 = _mm256_loadu_ps(ptr.add(i + 8));
                sq0 = _mm256_add_ps(
                    sq0,
                    _mm256_add_ps(_mm256_mul_ps(v0, v0), _mm256_mul_ps(v1, v1)),
                );
                let v2 = _mm256_loadu_ps(ptr.add(i + 16));
                let v3 = _mm256_loadu_ps(ptr.add(i + 24));
                sq1 = _mm256_add_ps(
                    sq1,
                    _mm256_add_ps(_mm256_mul_ps(v2, v2), _mm256_mul_ps(v3, v3)),
                );
                let v4 = _mm256_loadu_ps(ptr.add(i + 32));
                let v5 = _mm256_loadu_ps(ptr.add(i + 40));
                sq0 = _mm256_add_ps(
                    sq0,
                    _mm256_add_ps(_mm256_mul_ps(v4, v4), _mm256_mul_ps(v5, v5)),
                );
                let v6 = _mm256_loadu_ps(ptr.add(i + 48));
                let v7 = _mm256_loadu_ps(ptr.add(i + 56));
                sq1 = _mm256_add_ps(
                    sq1,
                    _mm256_add_ps(_mm256_mul_ps(v6, v6), _mm256_mul_ps(v7, v7)),
                );
                i += 64;
            }
            let rem = len - i;
            if (rem & 32) != 0 {
                let v0 = _mm256_loadu_ps(ptr.add(i));
                let v1 = _mm256_loadu_ps(ptr.add(i + 8));
                sq0 = _mm256_add_ps(
                    sq0,
                    _mm256_add_ps(_mm256_mul_ps(v0, v0), _mm256_mul_ps(v1, v1)),
                );
                let v2 = _mm256_loadu_ps(ptr.add(i + 16));
                let v3 = _mm256_loadu_ps(ptr.add(i + 24));
                sq1 = _mm256_add_ps(
                    sq1,
                    _mm256_add_ps(_mm256_mul_ps(v2, v2), _mm256_mul_ps(v3, v3)),
                );
                i += 32;
            }
            if (rem & 16) != 0 {
                let v0 = _mm256_loadu_ps(ptr.add(i));
                let v1 = _mm256_loadu_ps(ptr.add(i + 8));
                sq0 = _mm256_add_ps(sq0, _mm256_mul_ps(v0, v0));
                sq1 = _mm256_add_ps(sq1, _mm256_mul_ps(v1, v1));
                i += 16;
            }
            let mut sq_all = _mm256_add_ps(sq0, sq1);
            if (rem & 8) != 0 {
                let v0 = _mm256_loadu_ps(ptr.add(i));
                sq_all = _mm256_add_ps(sq_all, _mm256_mul_ps(v0, v0));
                i += 8;
            }
            let mut sum_sq = avx2_hsum_ps(sq_all);
            if (rem & 4) != 0 {
                let v0 = *ptr.add(i);
                let v1 = *ptr.add(i + 1);
                let v2 = *ptr.add(i + 2);
                let v3 = *ptr.add(i + 3);
                sum_sq += (v0 * v0 + v1 * v1) + (v2 * v2 + v3 * v3);
                i += 4;
            }
            if (rem & 2) != 0 {
                let v0 = *ptr.add(i);
                let v1 = *ptr.add(i + 1);
                sum_sq += v0 * v0 + v1 * v1;
                i += 2;
            }
            if (rem & 1) != 0 {
                let v0 = *ptr.add(i);
                sum_sq += v0 * v0;
            }
            sum_sq
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        sum_sq_scaled_f32_simd8(slice, 1.0)
    }
}

/// Evaluates $\sum_i (|x_i| \cdot 2^{126})^2$ across a subnormal-dominated `slice` (`|x_i| < F32_MIN_POSITIVE_NORMAL`)
/// by lifting the 23-bit mantissa directly in IEEE-754 integer SIMD (`(bits & 0x007F_FFFF) | 0x3F80_0000`
/// minus `1.0_f32`), completely bypassing x86 Denormal-Operand microcode assists.
#[inline(always)]
pub(super) fn sum_sq_subnormal_mantissa_f32_simd8(slice: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let len = slice.len();
        // SAFETY: `target_feature = "avx2"` is enabled, and all loads are bounds-checked against `len`.
        // For `abs_bits = r & 0x7FFF_FFFF`, `abs_bits < 0x0080_0000` (`_mm256_cmpgt_epi32`) selects
        // subnormal lanes where exponent bits 23..30 are already zero (`abs_bits == r & 0x007F_FFFF`),
        // keeping peak live YMM usage <= 6 (`0` Windows x64 `xmm6..xmm15` stack saves).
        unsafe {
            let v_one = _mm256_set1_ps(1.0_f32);
            let mut sq0 = _mm256_setzero_ps();
            let mut sq1 = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 16 <= len {
                let p = slice.as_ptr().add(i).cast::<__m256i>();
                let abs_mask = _mm256_set1_epi32(0x7FFF_FFFF);
                let a0 = _mm256_and_si256(_mm256_loadu_si256(p), abs_mask);
                let a1 = _mm256_and_si256(_mm256_loadu_si256(p.add(1)), abs_mask);
                let min_norm = _mm256_set1_epi32(0x0080_0000);
                let m0 = _mm256_and_si256(a0, _mm256_cmpgt_epi32(min_norm, a0));
                let m1 = _mm256_and_si256(a1, _mm256_cmpgt_epi32(min_norm, a1));
                let u0 = _mm256_sub_ps(_mm256_or_ps(_mm256_castsi256_ps(m0), v_one), v_one);
                let u1 = _mm256_sub_ps(_mm256_or_ps(_mm256_castsi256_ps(m1), v_one), v_one);
                sq0 = _mm256_add_ps(sq0, _mm256_mul_ps(u0, u0));
                sq1 = _mm256_add_ps(sq1, _mm256_mul_ps(u1, u1));
                i += 16;
            }
            let mut sum_sq = avx2_hsum_ps(_mm256_add_ps(sq0, sq1));
            while i < len {
                let bits = (*slice.as_ptr().add(i)).to_bits();
                let is_sub = (bits & 0x7F80_0000) == 0;
                let mant = branchless_select_u32(is_sub, bits & 0x007F_FFFF, 0);
                let u = f32::from_bits(mant | 0x3F80_0000) - 1.0_f32;
                sum_sq += u * u;
                i += 1;
            }
            sum_sq
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut sum_sq = 0.0_f32;
        for &x in slice {
            let bits = x.to_bits();
            let is_sub = (bits & 0x7F80_0000) == 0;
            let mant = branchless_select_u32(is_sub, bits & 0x007F_FFFF, 0);
            let u = f32::from_bits(mant | 0x3F80_0000) - 1.0_f32;
            sum_sq += u * u;
        }
        sum_sq
    }
}

/// Evaluates $\sum_i (x_i \cdot \mathrm{scale})^2$ across `slice` using a 64-float unrolled AVX2 tree reduction.
#[inline]
pub(crate) fn sum_sq_scaled_f32_simd8(slice: &[f32], scale: f32) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let len = slice.len();
        let ptr = slice.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and all loads are bounds-checked against `len`.
        // Note that `(x * scale)^2 = x^2 * scale^2`, but since `x^2` may underflow on near-zero inputs,
        // we scale `x` first across 64-float unrolled blocks with binary tail steps (`32`, `16`, `8`).
        unsafe {
            let v_scale = _mm256_set1_ps(scale);
            let mut sq0 = _mm256_setzero_ps();
            let mut sq1 = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 64 <= len {
                let v0 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i)), v_scale);
                let v1 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 8)), v_scale);
                sq0 = _mm256_add_ps(
                    sq0,
                    _mm256_add_ps(_mm256_mul_ps(v0, v0), _mm256_mul_ps(v1, v1)),
                );
                let v2 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 16)), v_scale);
                let v3 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 24)), v_scale);
                sq1 = _mm256_add_ps(
                    sq1,
                    _mm256_add_ps(_mm256_mul_ps(v2, v2), _mm256_mul_ps(v3, v3)),
                );
                let v4 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 32)), v_scale);
                let v5 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 40)), v_scale);
                sq0 = _mm256_add_ps(
                    sq0,
                    _mm256_add_ps(_mm256_mul_ps(v4, v4), _mm256_mul_ps(v5, v5)),
                );
                let v6 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 48)), v_scale);
                let v7 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 56)), v_scale);
                sq1 = _mm256_add_ps(
                    sq1,
                    _mm256_add_ps(_mm256_mul_ps(v6, v6), _mm256_mul_ps(v7, v7)),
                );
                i += 64;
            }
            let rem = len - i;
            if (rem & 32) != 0 {
                let v0 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i)), v_scale);
                let v1 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 8)), v_scale);
                sq0 = _mm256_add_ps(
                    sq0,
                    _mm256_add_ps(_mm256_mul_ps(v0, v0), _mm256_mul_ps(v1, v1)),
                );
                let v2 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 16)), v_scale);
                let v3 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 24)), v_scale);
                sq1 = _mm256_add_ps(
                    sq1,
                    _mm256_add_ps(_mm256_mul_ps(v2, v2), _mm256_mul_ps(v3, v3)),
                );
                i += 32;
            }
            if (rem & 16) != 0 {
                let v0 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i)), v_scale);
                let v1 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 8)), v_scale);
                sq0 = _mm256_add_ps(sq0, _mm256_mul_ps(v0, v0));
                sq1 = _mm256_add_ps(sq1, _mm256_mul_ps(v1, v1));
                i += 16;
            }
            let mut sq_all = _mm256_add_ps(sq0, sq1);
            if (rem & 8) != 0 {
                let v0 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i)), v_scale);
                sq_all = _mm256_add_ps(sq_all, _mm256_mul_ps(v0, v0));
                i += 8;
            }
            let mut sum_sq = avx2_hsum_ps(sq_all);
            while i < len {
                let bits = (*ptr.add(i)).to_bits();
                let clean_x = branchless_select_f32((bits & 0x7F80_0000) != 0, *ptr.add(i), 0.0);
                let v = clean_x * scale;
                sum_sq += v * v;
                i += 1;
            }
            sum_sq
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut acc_sq = [0.0_f32; 4];
        let (chunks, rem) = slice.as_chunks::<4>();
        for c in chunks {
            for lane in 0..4 {
                let clean_x = branchless_flush_subnormal_f32(c[lane]);
                let v = clean_x * scale;
                acc_sq[lane] += v * v;
            }
        }
        let mut sum_sq = (acc_sq[0] + acc_sq[1]) + (acc_sq[2] + acc_sq[3]);
        for &x in rem {
            let v = branchless_flush_subnormal_f32(x) * scale;
            sum_sq += v * v;
        }
        sum_sq
    }
}

/// Evaluates $\max(0, \max_i x_i)$ across `slice` using a 32-float (4-YMM) unrolled AVX2 SIMD reduction.
#[inline(always)]
pub(crate) fn max_pos_f32_unrolled4(slice: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let len = slice.len();
        let ptr = slice.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and all loads are bounds-checked against `len`.
        // Staged 2-vector pairs keep peak live YMMs <= 6 (`ymm0..ymm5`, `0` Windows x64 `xmm6..xmm15`
        // stack saves) while `_CMP_LT_OQ(v & 0x7F80_0000, +Inf)` stays 100% in the FP execution domain.
        unsafe {
            let inf_exp_ps = _mm256_castsi256_ps(_mm256_set1_epi32(0x7F80_0000));
            let mut m0 = _mm256_setzero_ps();
            let mut m1 = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 32 <= len {
                let v0 = _mm256_loadu_ps(ptr.add(i));
                let v1 = _mm256_loadu_ps(ptr.add(i + 8));
                let fin0 = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_and_ps(v0, inf_exp_ps), inf_exp_ps);
                let fin1 = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_and_ps(v1, inf_exp_ps), inf_exp_ps);
                m0 = _mm256_max_ps(m0, _mm256_and_ps(v0, fin0));
                m1 = _mm256_max_ps(m1, _mm256_and_ps(v1, fin1));

                let v2 = _mm256_loadu_ps(ptr.add(i + 16));
                let v3 = _mm256_loadu_ps(ptr.add(i + 24));
                let fin2 = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_and_ps(v2, inf_exp_ps), inf_exp_ps);
                let fin3 = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_and_ps(v3, inf_exp_ps), inf_exp_ps);
                m0 = _mm256_max_ps(m0, _mm256_and_ps(v2, fin2));
                m1 = _mm256_max_ps(m1, _mm256_and_ps(v3, fin3));
                i += 32;
            }
            while i + 8 <= len {
                let v0 = _mm256_loadu_ps(ptr.add(i));
                let fin0 = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_and_ps(v0, inf_exp_ps), inf_exp_ps);
                m0 = _mm256_max_ps(m0, _mm256_and_ps(v0, fin0));
                i += 8;
            }
            let mall = _mm256_max_ps(m0, m1);
            let m128 = _mm_max_ps(
                _mm256_castps256_ps128(mall),
                _mm256_extractf128_ps::<1>(mall),
            );
            let m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
            let mut max_val = _mm_cvtss_f32(_mm_max_ss(m64, _mm_shuffle_ps::<0x55>(m64, m64)));
            while i < len {
                max_val = branchless_max_f32(max_val, branchless_relu_f32(*ptr.add(i)));
                i += 1;
            }
            max_val
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut m = [0.0_f32; 4];
        let (chunks, rem) = slice.as_chunks::<4>();
        for c in chunks {
            m[0] = branchless_max_f32(m[0], branchless_relu_f32(c[0]));
            m[1] = branchless_max_f32(m[1], branchless_relu_f32(c[1]));
            m[2] = branchless_max_f32(m[2], branchless_relu_f32(c[2]));
            m[3] = branchless_max_f32(m[3], branchless_relu_f32(c[3]));
        }
        let mut max_val = branchless_max_f32(
            branchless_max_f32(m[0], m[1]),
            branchless_max_f32(m[2], m[3]),
        );
        for &v in rem {
            max_val = branchless_max_f32(max_val, branchless_relu_f32(v));
        }
        max_val
    }
}

/// Scales `buf` in-place by `scale` in strictly constant time (`0` data-dependent early-return
/// branches) while flushing any subnormal product (`|y| < F32_MIN_POSITIVE_NORMAL`) to `+0.0`
/// directly in the AVX2 FP execution domain (`<= 6` live YMMs, `0` FP-to-INT domain-crossing stalls,
/// `0` Windows x64 `xmm6..xmm15` stack saves).
#[inline]
pub(crate) fn scale_f32_slice_unrolled8(buf: &mut [f32], scale: f32) {
    let len = buf.len();
    if len == 0 {
        return;
    }
    let eff_scale = branchless_flush_subnormal_f32(sanitize_inplace_gain_f32(scale));
    if len <= 16 {
        // Match preceding 32-bit/128-bit stores in the CPU store buffer on short logit/score slices
        // (`<= 16` floats) to prevent a ~14-cycle 32-bit-store -> 256-bit-YMM-load STLF stall.
        for val in buf.iter_mut() {
            *val = branchless_flush_subnormal_f32(*val * eff_scale);
        }
        return;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let ptr = buf.as_mut_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and pointer offsets are bounds-checked against `len`.
        // Masking `prod` with `exp_mask_ps` (`0x7F80_0000`) clears the sign and mantissa bits so any
        // normal float (`1 <= exp <= 254`), `+/-Inf`, or `NaN` (`exp == 255 -> +Inf`) satisfies
        // `_CMP_GT_OQ(prod & exp_mask_ps, +0.0)`, while zero and subnormals (`exp == 0`) flush to `+0.0`
        // without crossing into the Integer SIMD execution domain. Staging 2 vectors (`64B` cache line)
        // at a time keeps peak live YMMs <= 6 (`ymm0..ymm5`).
        unsafe {
            let v_scale = _mm256_set1_ps(eff_scale);
            let exp_mask_ps = _mm256_castsi256_ps(_mm256_set1_epi32(0x7F80_0000));
            let zero_ps = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 64 <= len {
                let p0 = ptr.add(i);
                let p1 = ptr.add(i + 8);
                let prod0 = _mm256_mul_ps(_mm256_loadu_ps(p0), v_scale);
                let prod1 = _mm256_mul_ps(_mm256_loadu_ps(p1), v_scale);
                let m0 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod0, exp_mask_ps), zero_ps);
                let m1 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod1, exp_mask_ps), zero_ps);
                _mm256_storeu_ps(p0, _mm256_and_ps(prod0, m0));
                _mm256_storeu_ps(p1, _mm256_and_ps(prod1, m1));

                let p2 = ptr.add(i + 16);
                let p3 = ptr.add(i + 24);
                let prod2 = _mm256_mul_ps(_mm256_loadu_ps(p2), v_scale);
                let prod3 = _mm256_mul_ps(_mm256_loadu_ps(p3), v_scale);
                let m2 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod2, exp_mask_ps), zero_ps);
                let m3 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod3, exp_mask_ps), zero_ps);
                _mm256_storeu_ps(p2, _mm256_and_ps(prod2, m2));
                _mm256_storeu_ps(p3, _mm256_and_ps(prod3, m3));

                let p4 = ptr.add(i + 32);
                let p5 = ptr.add(i + 40);
                let prod4 = _mm256_mul_ps(_mm256_loadu_ps(p4), v_scale);
                let prod5 = _mm256_mul_ps(_mm256_loadu_ps(p5), v_scale);
                let m4 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod4, exp_mask_ps), zero_ps);
                let m5 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod5, exp_mask_ps), zero_ps);
                _mm256_storeu_ps(p4, _mm256_and_ps(prod4, m4));
                _mm256_storeu_ps(p5, _mm256_and_ps(prod5, m5));

                let p6 = ptr.add(i + 48);
                let p7 = ptr.add(i + 56);
                let prod6 = _mm256_mul_ps(_mm256_loadu_ps(p6), v_scale);
                let prod7 = _mm256_mul_ps(_mm256_loadu_ps(p7), v_scale);
                let m6 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod6, exp_mask_ps), zero_ps);
                let m7 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod7, exp_mask_ps), zero_ps);
                _mm256_storeu_ps(p6, _mm256_and_ps(prod6, m6));
                _mm256_storeu_ps(p7, _mm256_and_ps(prod7, m7));
                i += 64;
            }
            let rem = len - i;
            if rem != 0 {
                if (rem & 32) != 0 {
                    let p0 = ptr.add(i);
                    let p1 = ptr.add(i + 8);
                    let prod0 = _mm256_mul_ps(_mm256_loadu_ps(p0), v_scale);
                    let prod1 = _mm256_mul_ps(_mm256_loadu_ps(p1), v_scale);
                    let m0 =
                        _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod0, exp_mask_ps), zero_ps);
                    let m1 =
                        _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod1, exp_mask_ps), zero_ps);
                    _mm256_storeu_ps(p0, _mm256_and_ps(prod0, m0));
                    _mm256_storeu_ps(p1, _mm256_and_ps(prod1, m1));

                    let p2 = ptr.add(i + 16);
                    let p3 = ptr.add(i + 24);
                    let prod2 = _mm256_mul_ps(_mm256_loadu_ps(p2), v_scale);
                    let prod3 = _mm256_mul_ps(_mm256_loadu_ps(p3), v_scale);
                    let m2 =
                        _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod2, exp_mask_ps), zero_ps);
                    let m3 =
                        _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod3, exp_mask_ps), zero_ps);
                    _mm256_storeu_ps(p2, _mm256_and_ps(prod2, m2));
                    _mm256_storeu_ps(p3, _mm256_and_ps(prod3, m3));
                    i += 32;
                }
                if (rem & 16) != 0 {
                    let p0 = ptr.add(i);
                    let p1 = ptr.add(i + 8);
                    let prod0 = _mm256_mul_ps(_mm256_loadu_ps(p0), v_scale);
                    let prod1 = _mm256_mul_ps(_mm256_loadu_ps(p1), v_scale);
                    let m0 =
                        _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod0, exp_mask_ps), zero_ps);
                    let m1 =
                        _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod1, exp_mask_ps), zero_ps);
                    _mm256_storeu_ps(p0, _mm256_and_ps(prod0, m0));
                    _mm256_storeu_ps(p1, _mm256_and_ps(prod1, m1));
                    i += 16;
                }
                if (rem & 8) != 0 {
                    let p = ptr.add(i);
                    let prod = _mm256_mul_ps(_mm256_loadu_ps(p), v_scale);
                    let m = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod, exp_mask_ps), zero_ps);
                    _mm256_storeu_ps(p, _mm256_and_ps(prod, m));
                    i += 8;
                }
                while i < len {
                    *ptr.add(i) = branchless_flush_subnormal_f32(*ptr.add(i) * eff_scale);
                    i += 1;
                }
            }
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for val in buf.iter_mut() {
            *val = branchless_flush_subnormal_f32(*val * eff_scale);
        }
    }
}

/// Evaluates a 3rd-order Chebyshev Polynomial Spectral Filter $\sum_{k=0}^3 c_k T_k(\tilde{\lambda})$
/// across `eigenvalues_in` into `filtered_out` using 32-float (4-YMM) unrolled AVX2 Horner evaluation.
#[inline]
pub(crate) fn chebyshev_deg3_filter_slice_simd8(
    eigenvalues_in: &[f32],
    filtered_out: &mut [f32],
    inv_half_lmax: f32,
    k0: f32,
    k1: f32,
    k2: f32,
    k3: f32,
) {
    let len = eigenvalues_in.len().min(filtered_out.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let pin = eigenvalues_in.as_ptr();
        let pout = filtered_out.as_mut_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and all loads/stores are bounds-checked against `len`.
        unsafe {
            let v_scale = _mm256_set1_ps(inv_half_lmax);
            let v_one = _mm256_set1_ps(1.0_f32);
            let v_neg_one = _mm256_set1_ps(-1.0_f32);
            let v_k0 = _mm256_set1_ps(k0);
            let v_k1 = _mm256_set1_ps(k1);
            let v_k2 = _mm256_set1_ps(k2);
            let v_k3 = _mm256_set1_ps(k3);
            let mut i = 0usize;
            while i + 32 <= len {
                let l0 = _mm256_loadu_ps(pin.add(i));
                let l1 = _mm256_loadu_ps(pin.add(i + 8));
                let l2 = _mm256_loadu_ps(pin.add(i + 16));
                let l3 = _mm256_loadu_ps(pin.add(i + 24));
                let x0 = _mm256_min_ps(
                    _mm256_max_ps(_mm256_sub_ps(_mm256_mul_ps(l0, v_scale), v_one), v_neg_one),
                    v_one,
                );
                let x1 = _mm256_min_ps(
                    _mm256_max_ps(_mm256_sub_ps(_mm256_mul_ps(l1, v_scale), v_one), v_neg_one),
                    v_one,
                );
                let x2 = _mm256_min_ps(
                    _mm256_max_ps(_mm256_sub_ps(_mm256_mul_ps(l2, v_scale), v_one), v_neg_one),
                    v_one,
                );
                let x3 = _mm256_min_ps(
                    _mm256_max_ps(_mm256_sub_ps(_mm256_mul_ps(l3, v_scale), v_one), v_neg_one),
                    v_one,
                );
                let p0 = _mm256_add_ps(
                    v_k0,
                    _mm256_mul_ps(
                        x0,
                        _mm256_add_ps(
                            v_k1,
                            _mm256_mul_ps(x0, _mm256_add_ps(v_k2, _mm256_mul_ps(v_k3, x0))),
                        ),
                    ),
                );
                let p1 = _mm256_add_ps(
                    v_k0,
                    _mm256_mul_ps(
                        x1,
                        _mm256_add_ps(
                            v_k1,
                            _mm256_mul_ps(x1, _mm256_add_ps(v_k2, _mm256_mul_ps(v_k3, x1))),
                        ),
                    ),
                );
                let p2 = _mm256_add_ps(
                    v_k0,
                    _mm256_mul_ps(
                        x2,
                        _mm256_add_ps(
                            v_k1,
                            _mm256_mul_ps(x2, _mm256_add_ps(v_k2, _mm256_mul_ps(v_k3, x2))),
                        ),
                    ),
                );
                let p3 = _mm256_add_ps(
                    v_k0,
                    _mm256_mul_ps(
                        x3,
                        _mm256_add_ps(
                            v_k1,
                            _mm256_mul_ps(x3, _mm256_add_ps(v_k2, _mm256_mul_ps(v_k3, x3))),
                        ),
                    ),
                );
                _mm256_storeu_ps(pout.add(i), p0);
                _mm256_storeu_ps(pout.add(i + 8), p1);
                _mm256_storeu_ps(pout.add(i + 16), p2);
                _mm256_storeu_ps(pout.add(i + 24), p3);
                i += 32;
            }
            while i + 8 <= len {
                let lam = _mm256_loadu_ps(pin.add(i));
                let raw_x = _mm256_sub_ps(_mm256_mul_ps(lam, v_scale), v_one);
                let x = _mm256_min_ps(_mm256_max_ps(raw_x, v_neg_one), v_one);
                let p2 = _mm256_add_ps(v_k2, _mm256_mul_ps(v_k3, x));
                let p1 = _mm256_add_ps(v_k1, _mm256_mul_ps(x, p2));
                let p0 = _mm256_add_ps(v_k0, _mm256_mul_ps(x, p1));
                _mm256_storeu_ps(pout.add(i), p0);
                i += 8;
            }
            while i < len {
                let x = branchless_clamp_f32(*pin.add(i) * inv_half_lmax - 1.0, -1.0, 1.0);
                *pout.add(i) = k0 + x * (k1 + x * (k2 + k3 * x));
                i += 1;
            }
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut i = 0usize;
        while i + 4 <= len {
            let x0 = branchless_clamp_f32(eigenvalues_in[i] * inv_half_lmax - 1.0, -1.0, 1.0);
            let x1 = branchless_clamp_f32(eigenvalues_in[i + 1] * inv_half_lmax - 1.0, -1.0, 1.0);
            let x2 = branchless_clamp_f32(eigenvalues_in[i + 2] * inv_half_lmax - 1.0, -1.0, 1.0);
            let x3 = branchless_clamp_f32(eigenvalues_in[i + 3] * inv_half_lmax - 1.0, -1.0, 1.0);
            filtered_out[i] = k0 + x0 * (k1 + x0 * (k2 + k3 * x0));
            filtered_out[i + 1] = k0 + x1 * (k1 + x1 * (k2 + k3 * x1));
            filtered_out[i + 2] = k0 + x2 * (k1 + x2 * (k2 + k3 * x2));
            filtered_out[i + 3] = k0 + x3 * (k1 + x3 * (k2 + k3 * x3));
            i += 4;
        }
        while i < len {
            let x = branchless_clamp_f32(eigenvalues_in[i] * inv_half_lmax - 1.0, -1.0, 1.0);
            filtered_out[i] = k0 + x * (k1 + x * (k2 + k3 * x));
            i += 1;
        }
    }
}

/// Adds scalar `delta` to `buf` in-place in strictly constant time (`0` data-dependent early-return
/// branches), grouping full 64-byte cacheline loads before stores across 64-float (`256B`) chunks
/// to prevent same-cacheline Load-Store Queue RFO stalls on large 32K+ vocabulary logit slices.
#[inline(always)]
pub(crate) fn add_scalar_f32_slice_unrolled8(buf: &mut [f32], delta: f32) {
    let len = buf.len();
    if len == 0 {
        return;
    }
    let clean_delta = branchless_flush_subnormal_f32(delta);
    if len <= 16 {
        for val in buf.iter_mut() {
            *val += clean_delta;
        }
        return;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let ptr = buf.as_mut_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and pointer offsets are bounds-checked against `len`.
        unsafe {
            let v_delta = _mm256_set1_ps(clean_delta);
            let mut i = 0usize;
            while i + 64 <= len {
                let p0 = ptr.add(i);
                let p1 = ptr.add(i + 8);
                let p2 = ptr.add(i + 16);
                let p3 = ptr.add(i + 24);
                let x0 = _mm256_loadu_ps(p0);
                let x1 = _mm256_loadu_ps(p1);
                let x2 = _mm256_loadu_ps(p2);
                let x3 = _mm256_loadu_ps(p3);
                _mm256_storeu_ps(p0, _mm256_add_ps(x0, v_delta));
                _mm256_storeu_ps(p1, _mm256_add_ps(x1, v_delta));
                _mm256_storeu_ps(p2, _mm256_add_ps(x2, v_delta));
                _mm256_storeu_ps(p3, _mm256_add_ps(x3, v_delta));

                let p4 = ptr.add(i + 32);
                let p5 = ptr.add(i + 40);
                let p6 = ptr.add(i + 48);
                let p7 = ptr.add(i + 56);
                let x4 = _mm256_loadu_ps(p4);
                let x5 = _mm256_loadu_ps(p5);
                let x6 = _mm256_loadu_ps(p6);
                let x7 = _mm256_loadu_ps(p7);
                _mm256_storeu_ps(p4, _mm256_add_ps(x4, v_delta));
                _mm256_storeu_ps(p5, _mm256_add_ps(x5, v_delta));
                _mm256_storeu_ps(p6, _mm256_add_ps(x6, v_delta));
                _mm256_storeu_ps(p7, _mm256_add_ps(x7, v_delta));
                i += 64;
            }
            while i + 32 <= len {
                let p0 = ptr.add(i);
                let p1 = ptr.add(i + 8);
                let p2 = ptr.add(i + 16);
                let p3 = ptr.add(i + 24);
                let x0 = _mm256_loadu_ps(p0);
                let x1 = _mm256_loadu_ps(p1);
                let x2 = _mm256_loadu_ps(p2);
                let x3 = _mm256_loadu_ps(p3);
                _mm256_storeu_ps(p0, _mm256_add_ps(x0, v_delta));
                _mm256_storeu_ps(p1, _mm256_add_ps(x1, v_delta));
                _mm256_storeu_ps(p2, _mm256_add_ps(x2, v_delta));
                _mm256_storeu_ps(p3, _mm256_add_ps(x3, v_delta));
                i += 32;
            }
            while i + 8 <= len {
                let p = ptr.add(i);
                _mm256_storeu_ps(p, _mm256_add_ps(_mm256_loadu_ps(p), v_delta));
                i += 8;
            }
            while i < len {
                *ptr.add(i) += clean_delta;
                i += 1;
            }
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for val in buf.iter_mut() {
            *val += clean_delta;
        }
    }
}

#[inline(always)]
pub(crate) fn dot_f32_unrolled4(a: &[f32], b: &[f32], len: usize) -> f32 {
    let a_sub = &a[..len];
    let b_sub = &b[..len];
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let pa = a_sub.as_ptr();
        let pb = b_sub.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and `pa`, `pb` are valid for `len` elements.
        unsafe {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 32 <= len {
                acc0 = _mm256_add_ps(
                    acc0,
                    _mm256_mul_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i))),
                );
                acc1 = _mm256_add_ps(
                    acc1,
                    _mm256_mul_ps(
                        _mm256_loadu_ps(pa.add(i + 8)),
                        _mm256_loadu_ps(pb.add(i + 8)),
                    ),
                );
                acc2 = _mm256_add_ps(
                    acc2,
                    _mm256_mul_ps(
                        _mm256_loadu_ps(pa.add(i + 16)),
                        _mm256_loadu_ps(pb.add(i + 16)),
                    ),
                );
                acc3 = _mm256_add_ps(
                    acc3,
                    _mm256_mul_ps(
                        _mm256_loadu_ps(pa.add(i + 24)),
                        _mm256_loadu_ps(pb.add(i + 24)),
                    ),
                );
                i += 32;
            }
            let mut acc_all = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            while i + 8 <= len {
                acc_all = _mm256_add_ps(
                    acc_all,
                    _mm256_mul_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i))),
                );
                i += 8;
            }
            let mut total = avx2_hsum_ps(acc_all);
            for (&av, &bv) in a_sub[i..].iter().zip(b_sub[i..].iter()) {
                total += av * bv;
            }
            total
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut total = 0.0_f32;
        for (&av, &bv) in a_sub.iter().zip(b_sub.iter()) {
            total += av * bv;
        }
        total
    }
}

/// Evaluates the underflow-resilient L2 norm $(\|\mathbf{x}\|_2, \mathrm{is\_finite})$,
/// pre-detecting subnormal-square risk (`0.0 < |buf[0]| < 1.0e-18`) to bypass x86 subnormal
/// microcode traps and using AVX2 `sum_sq_scaled_f32_simd8` for $2^{64}$ rescue.
#[inline(always)]
pub(crate) fn norm_l2_underflow_resilient_f32(buf: &[f32]) -> (f32, bool) {
    let len = buf.len();
    if len == 0 {
        return (0.0, true);
    }
    const SCALE_UP_2P84: f32 = f32::from_bits((127 + 84) << 23);
    const SCALE_DOWN_2M84: f32 = f32::from_bits((127 - 84) << 23);
    const SUBNORMAL_SQ_RISK_BITS: u32 = SUBNORMAL_SQ_RISK_FLOOR.to_bits();

    let b0_bits = buf[0].to_bits() & 0x7FFF_FFFF;
    let bl_bits = buf[len - 1].to_bits() & 0x7FFF_FFFF;
    if (b0_bits >= 0x7F80_0000) | (bl_bits >= 0x7F80_0000) {
        return (f32::NAN, false);
    }
    let probe_bits = branchless_select_u32(b0_bits > bl_bits, b0_bits, bl_bits);
    if (probe_bits > 0) & (probe_bits < SUBNORMAL_SQ_RISK_BITS) {
        if probe_bits < 0x0080_0000 {
            let sum_mant = sum_sq_subnormal_mantissa_f32_simd8(buf);
            if !branchless_is_finite_f32(sum_mant) {
                return (f32::NAN, false);
            }
            let root = branchless_sqrt_f32(sum_mant);
            let root_bits = root.to_bits();
            let norm = if root_bits >= 0x7F00_0000 {
                f32::from_bits(root_bits - (126 << 23))
            } else {
                0.0
            };
            return (norm, true);
        }
        let sum_scaled = sum_sq_scaled_f32_simd8(buf, SCALE_UP_2P84);
        if !branchless_is_finite_f32(sum_scaled) {
            return (f32::NAN, false);
        }
        let root = branchless_sqrt_f32(sum_scaled);
        let root_bits = root.to_bits();
        let norm = if root_bits >= ((1 + 84) << 23) {
            root * SCALE_DOWN_2M84
        } else {
            0.0
        };
        return (norm, true);
    }

    let sum_sq = sum_sq_f32_simd8(buf);
    if !branchless_is_finite_f32(sum_sq) {
        return (f32::NAN, false);
    }
    if sum_sq >= F32_MIN_POSITIVE_NORMAL {
        return (branchless_sqrt_f32(sum_sq), true);
    }
    if (probe_bits == 0) & ((buf[len >> 1].to_bits() & 0x7FFF_FFFF) == 0) & (sum_sq == 0.0) {
        return (0.0, true);
    }
    let sum_scaled = sum_sq_scaled_f32_simd8(buf, SCALE_UP_2P84);
    if !branchless_is_finite_f32(sum_scaled) {
        return (f32::NAN, false);
    }
    let root = branchless_sqrt_f32(sum_scaled);
    let root_bits = root.to_bits();
    let norm = if root_bits >= ((1 + 84) << 23) {
        root * SCALE_DOWN_2M84
    } else {
        0.0
    };
    (norm, true)
}
