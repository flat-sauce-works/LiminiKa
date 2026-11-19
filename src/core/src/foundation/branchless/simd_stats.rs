// name: src/core/src/foundation/branchless/simd_stats.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Scalar & AVX2 constant-time norm/sum, RIPA clamp, Chebyshev deg-3, Procrustes, and Hodge dual kernels.

use super::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn avx2_hsum3_ps(
    v0: core::arch::x86_64::__m256,
    v1: core::arch::x86_64::__m256,
    v2: core::arch::x86_64::__m256,
) -> (f32, f32, f32) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`.
    unsafe { (avx2_hsum_ps(v0), avx2_hsum_ps(v1), avx2_hsum_ps(v2)) }
}

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
        let mut chunks = slice.chunks_exact(4);
        for c in chunks.by_ref() {
            for lane in 0..4 {
                let v = branchless_abs_f32(c[lane]);
                acc_abs[lane] += v;
                acc_sq[lane] += v * v;
            }
        }
        let mut sum_abs = (acc_abs[0] + acc_abs[1]) + (acc_abs[2] + acc_abs[3]);
        let mut sum_sq = (acc_sq[0] + acc_sq[1]) + (acc_sq[2] + acc_sq[3]);
        for &v in chunks.remainder() {
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
        let mut chunks = slice.chunks_exact(4);
        for c in chunks.by_ref() {
            for lane in 0..4 {
                let clean_x = branchless_flush_subnormal_f32(c[lane]);
                let v = clean_x * scale;
                acc_sq[lane] += v * v;
            }
        }
        let mut sum_sq = (acc_sq[0] + acc_sq[1]) + (acc_sq[2] + acc_sq[3]);
        for &x in chunks.remainder() {
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
        let mut chunks = slice.chunks_exact(4);
        for c in chunks.by_ref() {
            m[0] = branchless_max_f32(m[0], branchless_relu_f32(c[0]));
            m[1] = branchless_max_f32(m[1], branchless_relu_f32(c[1]));
            m[2] = branchless_max_f32(m[2], branchless_relu_f32(c[2]));
            m[3] = branchless_max_f32(m[3], branchless_relu_f32(c[3]));
        }
        let mut max_val = branchless_max_f32(
            branchless_max_f32(m[0], m[1]),
            branchless_max_f32(m[2], m[3]),
        );
        for &v in chunks.remainder() {
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

/// Evaluates the $SO(2)$ Procrustes inner and exterior cross products using a 32-float (4-YMM)
/// unrolled dual-accumulator AVX2 tree reduction.
#[inline(always)]
pub(crate) fn procrustes_sincos_f32_simd(src_head: &[f32], tgt_head: &[f32]) -> (f32, f32) {
    let len = src_head.len().min(tgt_head.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let ps = src_head.as_ptr();
        let pt = tgt_head.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and `ps`, `pt` are valid for `len` elements.
        // Delegates 8-float chunks to `avx2_procrustes_head_rc128` (`<= 6` live YMMs, `0` Windows x64 spills).
        unsafe {
            let vec_len = len & !7;
            let rc = avx2_procrustes_head_rc128(ps, pt, vec_len);
            let sc = _mm_addsub_ps(
                _mm_shuffle_ps::<0x88>(rc, rc),
                _mm_shuffle_ps::<0xDD>(rc, rc),
            );
            let mut sum_sin = _mm_cvtss_f32(sc);
            let mut sum_cos = _mm_cvtss_f32(_mm_shuffle_ps::<0x55>(sc, sc));
            let mut i = vec_len;
            while i + 2 <= len {
                let (ux, uy) = (*ps.add(i), *ps.add(i + 1));
                let (vx, vy) = (*pt.add(i), *pt.add(i + 1));
                sum_cos += ux * vx + uy * vy;
                sum_sin += ux * vy - uy * vx;
                i += 2;
            }
            (sum_cos, sum_sin)
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut sum_cos = 0.0_f32;
        let mut sum_sin = 0.0_f32;
        for (s_pair, t_pair) in src_head[..len]
            .chunks_exact(2)
            .zip(tgt_head[..len].chunks_exact(2))
        {
            sum_cos += s_pair[0] * t_pair[0] + s_pair[1] * t_pair[1];
            sum_sin += s_pair[0] * t_pair[1] - s_pair[1] * t_pair[0];
        }
        (sum_cos, sum_sin)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn avx2_procrustes_head_rc128(
    ps: *const f32,
    pt: *const f32,
    head_dim: usize,
) -> core::arch::x86_64::__m128 {
    // SAFETY: Caller guarantees `target_feature = "avx2"`, `(head_dim & 7) == 0`, and valid pointers.
    // Dual-accumulator `(cos0, crs0, cos1, crs1)` halves the 32-float FP add latency chain (`2x` ILP)
    // while staging 1 `(a_k, b_k)` vector pair at a time (`<= 6` live YMMs `ymm0..ymm5`, `0` Windows x64
    // `xmm6..xmm15` stack saves), then folds to `[R_even, R_odd, C_even, C_odd]` in a single XMM register.
    unsafe {
        use core::arch::x86_64::{
            _mm256_castps256_ps128, _mm256_extractf128_ps, _mm_add_ps, _mm_movehl_ps, _mm_movelh_ps,
        };
        let mut cos0 = _mm256_setzero_ps();
        let mut crs0 = _mm256_setzero_ps();
        let mut cos1 = _mm256_setzero_ps();
        let mut crs1 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 32 <= head_dim {
            let a0 = _mm256_loadu_ps(ps.add(i));
            let b0 = _mm256_loadu_ps(pt.add(i));
            cos0 = _mm256_add_ps(cos0, _mm256_mul_ps(a0, b0));
            crs0 = _mm256_add_ps(crs0, _mm256_mul_ps(a0, _mm256_permute_ps::<0xB1>(b0)));
            let a1 = _mm256_loadu_ps(ps.add(i + 8));
            let b1 = _mm256_loadu_ps(pt.add(i + 8));
            cos1 = _mm256_add_ps(cos1, _mm256_mul_ps(a1, b1));
            crs1 = _mm256_add_ps(crs1, _mm256_mul_ps(a1, _mm256_permute_ps::<0xB1>(b1)));
            let a2 = _mm256_loadu_ps(ps.add(i + 16));
            let b2 = _mm256_loadu_ps(pt.add(i + 16));
            cos0 = _mm256_add_ps(cos0, _mm256_mul_ps(a2, b2));
            crs0 = _mm256_add_ps(crs0, _mm256_mul_ps(a2, _mm256_permute_ps::<0xB1>(b2)));
            let a3 = _mm256_loadu_ps(ps.add(i + 24));
            let b3 = _mm256_loadu_ps(pt.add(i + 24));
            cos1 = _mm256_add_ps(cos1, _mm256_mul_ps(a3, b3));
            crs1 = _mm256_add_ps(crs1, _mm256_mul_ps(a3, _mm256_permute_ps::<0xB1>(b3)));
            i += 32;
        }
        cos0 = _mm256_add_ps(cos0, cos1);
        crs0 = _mm256_add_ps(crs0, crs1);
        while i + 8 <= head_dim {
            let s0 = _mm256_loadu_ps(ps.add(i));
            let t0 = _mm256_loadu_ps(pt.add(i));
            cos0 = _mm256_add_ps(cos0, _mm256_mul_ps(s0, t0));
            crs0 = _mm256_add_ps(crs0, _mm256_mul_ps(s0, _mm256_permute_ps::<0xB1>(t0)));
            i += 8;
        }
        let r4 = _mm_add_ps(
            _mm256_castps256_ps128(crs0),
            _mm256_extractf128_ps::<1>(crs0),
        );
        let c4 = _mm_add_ps(
            _mm256_castps256_ps128(cos0),
            _mm256_extractf128_ps::<1>(cos0),
        );
        _mm_add_ps(_mm_movelh_ps(r4, c4), _mm_movehl_ps(c4, r4))
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn avx2_procrustes_cell8_sincos_vec256(
    ps_base: *const f32,
    pt_base: *const f32,
    off0: usize,
    head_dim: usize,
) -> (core::arch::x86_64::__m256, core::arch::x86_64::__m256) {
    // SAFETY: Caller guarantees `target_feature = "avx2"` and valid pointer ranges for 8 heads.
    // Packs `[sin0..sin3]` and `[cos0..cos3]` directly in the XMM FP domain via `_mm_shuffle_ps::<0x88/0xDD>`,
    // stages the lower 128-bit half via exact-width 128-bit `_mm_storeu_ps` -> `_mm_loadu_ps` (`0` STLF
    // stalls, `0` XMM<->GPR domain transfers), and merges the upper 4 heads via `_mm256_insertf128_ps::<1>`.
    unsafe {
        use core::arch::x86_64::{
            _mm256_castps128_ps256, _mm256_insertf128_ps, _mm_addsub_ps, _mm_loadu_ps,
            _mm_shuffle_ps, _mm_storeu_ps,
        };
        let mut sin_lo_buf = [0.0_f32; 4];
        let mut cos_lo_buf = [0.0_f32; 4];
        let ps = ps_base.add(off0);
        let pt = pt_base.add(off0);
        let step2 = head_dim * 2;
        let step4 = head_dim * 4;

        let rc0 = avx2_procrustes_head_rc128(ps, pt, head_dim);
        let rc1 = avx2_procrustes_head_rc128(ps.add(head_dim), pt.add(head_dim), head_dim);
        let sc01 = _mm_addsub_ps(
            _mm_shuffle_ps::<0x88>(rc0, rc1),
            _mm_shuffle_ps::<0xDD>(rc0, rc1),
        );
        let ps2 = ps.add(step2);
        let pt2 = pt.add(step2);
        let rc2 = avx2_procrustes_head_rc128(ps2, pt2, head_dim);
        let rc3 = avx2_procrustes_head_rc128(ps2.add(head_dim), pt2.add(head_dim), head_dim);
        let sc23 = _mm_addsub_ps(
            _mm_shuffle_ps::<0x88>(rc2, rc3),
            _mm_shuffle_ps::<0xDD>(rc2, rc3),
        );
        _mm_storeu_ps(sin_lo_buf.as_mut_ptr(), _mm_shuffle_ps::<0x88>(sc01, sc23));
        _mm_storeu_ps(cos_lo_buf.as_mut_ptr(), _mm_shuffle_ps::<0xDD>(sc01, sc23));

        let ps4 = ps.add(step4);
        let pt4 = pt.add(step4);
        let rc4 = avx2_procrustes_head_rc128(ps4, pt4, head_dim);
        let rc5 = avx2_procrustes_head_rc128(ps4.add(head_dim), pt4.add(head_dim), head_dim);
        let sc45 = _mm_addsub_ps(
            _mm_shuffle_ps::<0x88>(rc4, rc5),
            _mm_shuffle_ps::<0xDD>(rc4, rc5),
        );
        let ps6 = ps4.add(step2);
        let pt6 = pt4.add(step2);
        let rc6 = avx2_procrustes_head_rc128(ps6, pt6, head_dim);
        let rc7 = avx2_procrustes_head_rc128(ps6.add(head_dim), pt6.add(head_dim), head_dim);
        let sc67 = _mm_addsub_ps(
            _mm_shuffle_ps::<0x88>(rc6, rc7),
            _mm_shuffle_ps::<0xDD>(rc6, rc7),
        );
        let sin_hi = _mm_shuffle_ps::<0x88>(sc45, sc67);
        let cos_hi = _mm_shuffle_ps::<0xDD>(sc45, sc67);
        (
            _mm256_insertf128_ps::<1>(
                _mm256_castps128_ps256(_mm_loadu_ps(sin_lo_buf.as_ptr())),
                sin_hi,
            ),
            _mm256_insertf128_ps::<1>(
                _mm256_castps128_ps256(_mm_loadu_ps(cos_lo_buf.as_ptr())),
                cos_hi,
            ),
        )
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn avx2_atan2_to_q7_word_x8_vec(
    vy: core::arch::x86_64::__m256,
    vx: core::arch::x86_64::__m256,
) -> u64 {
    // SAFETY: `target_feature = "avx2"` is enabled; `vy` and `vx` are in YMM registers.
    // Schedules `swap_mask`, `x_neg_mask`, `y_sign`, and `valid_mask` strictly after `r`, `r2`,
    // and `p` are dead so peak live YMMs <= 6 (`ymm0..ymm5`, `0` Windows x64 `xmm6..xmm15` stack saves).
    unsafe {
        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
        let ax = _mm256_and_ps(vx, abs_mask);
        let ay = _mm256_and_ps(vy, abs_mask);
        let min_v = _mm256_min_ps(ax, ay);
        let max_v = _mm256_max_ps(_mm256_max_ps(ax, ay), _mm256_set1_ps(1.0e-30_f32));

        let rcp0 = _mm256_rcp_ps(max_v);
        let inv_max = _mm256_mul_ps(
            rcp0,
            _mm256_sub_ps(_mm256_set1_ps(2.0_f32), _mm256_mul_ps(max_v, rcp0)),
        );
        let r = _mm256_min_ps(
            _mm256_max_ps(_mm256_mul_ps(min_v, inv_max), _mm256_setzero_ps()),
            _mm256_set1_ps(1.0_f32),
        );
        let r2 = _mm256_mul_ps(r, r);

        let mut p = _mm256_add_ps(
            _mm256_mul_ps(r2, _mm256_set1_ps(-0.477_609_f32)),
            _mm256_set1_ps(2.145_485_f32),
        );
        p = _mm256_add_ps(_mm256_mul_ps(r2, p), _mm256_set1_ps(-4.744_295_f32));
        p = _mm256_add_ps(_mm256_mul_ps(r2, p), _mm256_set1_ps(7.886_416_f32));
        p = _mm256_add_ps(_mm256_mul_ps(r2, p), _mm256_set1_ps(-13.553_533_f32));
        p = _mm256_add_ps(_mm256_mul_ps(r2, p), _mm256_set1_ps(40.742_737_f32));
        let poly_q7 = _mm256_mul_ps(r, p);

        let sq_x = _mm256_mul_ps(vx, vx);
        let sq_y = _mm256_mul_ps(vy, vy);
        let swap_mask = _mm256_cmp_ps::<_CMP_LT_OQ>(sq_x, sq_y);
        let norm_sq = _mm256_add_ps(sq_x, sq_y);
        let mut ang_q7 = _mm256_blendv_ps(
            poly_q7,
            _mm256_sub_ps(_mm256_set1_ps(64.0_f32), poly_q7),
            swap_mask,
        );
        let x_neg_mask = _mm256_cmp_ps::<_CMP_LT_OQ>(vx, _mm256_setzero_ps());
        ang_q7 = _mm256_blendv_ps(
            ang_q7,
            _mm256_sub_ps(_mm256_set1_ps(128.0_f32), ang_q7),
            x_neg_mask,
        );
        let y_sign = _mm256_and_ps(vy, _mm256_set1_ps(f32::from_bits(0x8000_0000)));
        ang_q7 = _mm256_or_ps(
            _mm256_and_ps(ang_q7, _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF))),
            y_sign,
        );
        let rounded_q7 = _mm256_add_ps(ang_q7, _mm256_or_ps(_mm256_set1_ps(0.5_f32), y_sign));
        let valid_mask = _mm256_and_ps(
            _mm256_cmp_ps::<_CMP_GT_OQ>(norm_sq, _mm256_set1_ps(DEFAULT_ENTROPY_EPS)),
            _mm256_cmp_ps::<_CMP_LT_OQ>(norm_sq, _mm256_set1_ps(f32::INFINITY)),
        );
        let v_i32 = _mm256_and_si256(
            _mm256_cvttps_epi32(rounded_q7),
            _mm256_castps_si256(valid_mask),
        );
        let lo128 = _mm256_castsi256_si128(v_i32);
        let hi128 = _mm256_extracti128_si256::<1>(v_i32);
        let p16 = _mm_packs_epi32(lo128, hi128);
        let p8 = _mm_packs_epi16(p16, p16);
        _mm_cvtsi128_si64(p8) as u64
    }
}

#[inline(always)]
pub(crate) fn atan2_to_q7_word_x8_simd(sin_arr: &[f32; 8], cos_arr: &[f32; 8]) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: `target_feature = "avx2"` is enabled; `sin_arr` and `cos_arr` hold 8 `f32`s.
        unsafe {
            let vy = _mm256_loadu_ps(sin_arr.as_ptr());
            let vx = _mm256_loadu_ps(cos_arr.as_ptr());
            return avx2_atan2_to_q7_word_x8_vec(vy, vx);
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut word = 0u64;
        let mut lane = 0usize;
        while lane < 8 {
            let sum_sin = sin_arr[lane];
            let sum_cos = cos_arr[lane];
            let norm_sq = sum_cos * sum_cos + sum_sin * sum_sin;
            let valid = (norm_sq > DEFAULT_ENTROPY_EPS) & branchless_is_finite_f32(norm_sq);
            let q7 = if valid {
                branchless_atan2_to_q7_fast(sum_sin, sum_cos)
            } else {
                0
            };
            word |= ((q7 as u8) as u64) << (lane << 3);
            lane += 1;
        }
        word
    }
}

/// Evaluates 8-lane pairwise $\operatorname{atan2}(y_k, x_k)$ directly on 16 interleaved
/// `[x_0, y_0, x_1, y_1, ..., x_7, y_7]` floats (`interleaved_16.len() >= 16`), deinterleaving
/// even (`cos`) and odd (`sin`) lanes in AVX2 registers to eliminate 16 scalar stack stores and
/// 8-to-1 Store-to-Load Forwarding (STLF) pipeline stalls.
#[inline(always)]
pub(crate) fn atan2_interleaved_pairs_to_q7_word_x8_simd(interleaved_16: &[f32]) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        if interleaved_16.len() >= 16 {
            // SAFETY: `target_feature = "avx2"` is enabled and `interleaved_16.len() >= 16`.
            unsafe {
                let ptr = interleaved_16.as_ptr();
                let v_lo = _mm256_loadu_ps(ptr);
                let v_hi = _mm256_loadu_ps(ptr.add(8));
                let cos_shuf = _mm256_shuffle_ps::<0x88>(v_lo, v_hi);
                let sin_shuf = _mm256_shuffle_ps::<0xDD>(v_lo, v_hi);
                let vx = _mm256_castsi256_ps(_mm256_permute4x64_epi64::<0xD8>(
                    _mm256_castps_si256(cos_shuf),
                ));
                let vy = _mm256_castsi256_ps(_mm256_permute4x64_epi64::<0xD8>(
                    _mm256_castps_si256(sin_shuf),
                ));
                return avx2_atan2_to_q7_word_x8_vec(vy, vx);
            }
        }
    }
    let sin_arr = [
        interleaved_16[1],
        interleaved_16[3],
        interleaved_16[5],
        interleaved_16[7],
        interleaved_16[9],
        interleaved_16[11],
        interleaved_16[13],
        interleaved_16[15],
    ];
    let cos_arr = [
        interleaved_16[0],
        interleaved_16[2],
        interleaved_16[4],
        interleaved_16[6],
        interleaved_16[8],
        interleaved_16[10],
        interleaved_16[12],
        interleaved_16[14],
    ];
    atan2_to_q7_word_x8_simd(&sin_arr, &cos_arr)
}

/// Computes the Orthogonal Procrustes $SO(2)$ phase delta for a single 8-head SWAR cell
/// (`h_base .. h_base + cell_heads`, where `cell_heads <= 8`), returning the packed 64-bit Q7 SWAR word.
#[inline(always)]
pub(crate) fn procrustes_q7_cell_word_simd(
    src_slice: &[f32],
    tgt_slice: &[f32],
    head_dim: usize,
    h_base: usize,
    cell_heads: usize,
) -> u64 {
    if cell_heads == 8 {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            if (head_dim & 7) == 0
                && head_dim >= 8
                && src_slice.len() >= (h_base + 8) * head_dim
                && tgt_slice.len() >= (h_base + 8) * head_dim
            {
                let ps_base = src_slice.as_ptr();
                let pt_base = tgt_slice.as_ptr();
                // SAFETY: `target_feature = "avx2"`, `(head_dim & 7) == 0`, and slice bounds verified above.
                return unsafe {
                    let (vy, vx) = avx2_procrustes_cell8_sincos_vec256(
                        ps_base,
                        pt_base,
                        h_base * head_dim,
                        head_dim,
                    );
                    avx2_atan2_to_q7_word_x8_vec(vy, vx)
                };
            }
        }
        let mut sin_arr = [0.0_f32; 8];
        let mut cos_arr = [0.0_f32; 8];
        let mut lane = 0usize;
        while lane < 8 {
            let base = (h_base + lane) * head_dim;
            let end = base + head_dim;
            let (sum_cos, sum_sin) =
                procrustes_sincos_f32_simd(&src_slice[base..end], &tgt_slice[base..end]);
            cos_arr[lane] = sum_cos;
            sin_arr[lane] = sum_sin;
            lane += 1;
        }
        return atan2_to_q7_word_x8_simd(&sin_arr, &cos_arr);
    }
    let mut sin_arr = [0.0_f32; 8];
    let mut cos_arr = [0.0_f32; 8];
    let mut lane = 0usize;
    while lane < cell_heads {
        let base = (h_base + lane) * head_dim;
        let end = base + head_dim;
        let (sum_cos, sum_sin) =
            procrustes_sincos_f32_simd(&src_slice[base..end], &tgt_slice[base..end]);
        cos_arr[lane] = sum_cos;
        sin_arr[lane] = sum_sin;
        lane += 1;
    }
    atan2_to_q7_word_x8_simd(&sin_arr, &cos_arr)
}

/// Computes the Orthogonal Procrustes $SO(2)$ phase delta across `num_heads` directly into
/// 8 register-resident `u64` SWAR words (`[u64; 8]`), evaluating 8-head AVX2 `atan2` in YMM
/// registers with zero scalar XMM-to-GPR roundtrips.
#[inline(always)]
pub(crate) fn procrustes_q7_words_64_simd(
    src_slice: &[f32],
    tgt_slice: &[f32],
    head_dim: usize,
    num_heads: usize,
) -> [u64; 8] {
    let mut words = [0u64; 8];
    let limit_heads = num_heads.min(MAX_ATTENTION_HEADS);
    let full_cells = limit_heads >> 3;
    let mut c = 0usize;
    while c < full_cells {
        words[c] = procrustes_q7_cell_word_simd(src_slice, tgt_slice, head_dim, c << 3, 8);
        c += 1;
    }
    let rem_start = full_cells << 3;
    if rem_start < limit_heads {
        words[full_cells] = procrustes_q7_cell_word_simd(
            src_slice,
            tgt_slice,
            head_dim,
            rem_start,
            limit_heads - rem_start,
        );
    }
    words
}

#[inline(always)]
pub(crate) fn procrustes_q7_slice_simd(
    src_slice: &[f32],
    tgt_slice: &[f32],
    head_dim: usize,
    num_heads: usize,
    phases_out: &mut [gcso_q7_t],
) {
    let limit_heads = num_heads.min(MAX_ATTENTION_HEADS).min(phases_out.len());
    let words = procrustes_q7_words_64_simd(src_slice, tgt_slice, head_dim, limit_heads);
    if limit_heads == MAX_ATTENTION_HEADS {
        // SAFETY: `phases_out.len() >= MAX_ATTENTION_HEADS` and `align_of::<gcso_q7_t>() == 1`.
        let out_arr = unsafe {
            &mut *phases_out
                .as_mut_ptr()
                .cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>()
        };
        store_q7_words_64(out_arr, &words);
        return;
    }
    let full_cells = limit_heads >> 3;
    let mut c = 0usize;
    while c < full_cells {
        store_q7_x8_slice(phases_out, c << 3, words[c]);
        c += 1;
    }
    let rem_start = full_cells << 3;
    if rem_start < limit_heads {
        let mut w = words[full_cells];
        let mut h = rem_start;
        while h < limit_heads {
            phases_out[h] = (w & 0xFF) as gcso_q7_t;
            w >>= 8;
            h += 1;
        }
    }
}

/// Evaluates range-reduced $(\sin\theta, \cos\theta)$ across 8 `f32` angles simultaneously using AVX2.
#[inline(always)]
pub(crate) fn sincos_f32_x8_simd(angles: &[f32], sin_out: &mut [f32; 8], cos_out: &mut [f32; 8]) {
    debug_assert!(angles.len() >= 8);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: `target_feature = "avx2"` is enabled, and `angles`, `sin_out`, `cos_out` hold at least 8 `f32`s.
        unsafe {
            let v_in = _mm256_loadu_ps(angles.as_ptr());
            let v_abs = _mm256_and_ps(v_in, _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF)));
            let in_fast_range = _mm256_cmp_ps::<_CMP_LE_OQ>(v_abs, _mm256_set1_ps(65_536.0_f32));

            if _mm256_movemask_ps(in_fast_range) == 0xFF {
                let k =
                    _mm256_round_ps::<0x08>(_mm256_mul_ps(v_in, _mm256_set1_ps(0.159_154_94_f32)));
                let x = _mm256_sub_ps(
                    _mm256_sub_ps(v_in, _mm256_mul_ps(k, _mm256_set1_ps(6.283_185_f32))),
                    _mm256_mul_ps(k, _mm256_set1_ps(3.071_795_9e-7_f32)),
                );

                let gt_hp = _mm256_cmp_ps::<_CMP_GT_OQ>(x, _mm256_set1_ps(FRAC_PI_2));
                let lt_nhp = _mm256_cmp_ps::<_CMP_LT_OQ>(x, _mm256_set1_ps(-FRAC_PI_2));
                let x_fold_lo = _mm256_blendv_ps(x, _mm256_sub_ps(_mm256_set1_ps(-PI), x), lt_nhp);
                let x_folded =
                    _mm256_blendv_ps(x_fold_lo, _mm256_sub_ps(_mm256_set1_ps(PI), x), gt_hp);
                let cos_neg_mask = _mm256_and_ps(
                    _mm256_or_ps(gt_hp, lt_nhp),
                    _mm256_set1_ps(f32::from_bits(0x8000_0000)),
                );

                let x2 = _mm256_mul_ps(x_folded, x_folded);
                let one_v = _mm256_set1_ps(1.0_f32);

                let mut sp = _mm256_add_ps(
                    _mm256_mul_ps(x2, _mm256_set1_ps(-0.000_000_025_052_108_f32)),
                    _mm256_set1_ps(0.000_002_755_732_f32),
                );
                sp = _mm256_add_ps(_mm256_mul_ps(x2, sp), _mm256_set1_ps(-0.000_198_412_7_f32));
                sp = _mm256_add_ps(_mm256_mul_ps(x2, sp), _mm256_set1_ps(0.008_333_334_f32));
                sp = _mm256_add_ps(_mm256_mul_ps(x2, sp), _mm256_set1_ps(-0.166_666_67_f32));
                let sin_v = _mm256_mul_ps(x_folded, _mm256_add_ps(_mm256_mul_ps(x2, sp), one_v));

                let mut cp = _mm256_add_ps(
                    _mm256_mul_ps(x2, _mm256_set1_ps(-0.000_000_275_573_19_f32)),
                    _mm256_set1_ps(0.000_024_801_587_f32),
                );
                cp = _mm256_add_ps(_mm256_mul_ps(x2, cp), _mm256_set1_ps(-0.001_388_888_9_f32));
                cp = _mm256_add_ps(_mm256_mul_ps(x2, cp), _mm256_set1_ps(0.041_666_668_f32));
                cp = _mm256_add_ps(_mm256_mul_ps(x2, cp), _mm256_set1_ps(-0.5_f32));
                let cos_v =
                    _mm256_xor_ps(_mm256_add_ps(_mm256_mul_ps(x2, cp), one_v), cos_neg_mask);

                _mm256_storeu_ps(sin_out.as_mut_ptr(), sin_v);
                _mm256_storeu_ps(cos_out.as_mut_ptr(), cos_v);
                return;
            }
        }
    }
    for lane in 0..8 {
        let (s, c) = branchless_sincos_f32(angles[lane]);
        sin_out[lane] = s;
        cos_out[lane] = c;
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_rotate_so2_block32(ptr: *mut f32, v_cos: __m256, v_sin: __m256) {
    // SAFETY: Caller guarantees `ptr` is valid for 32 floats and AVX2 is enabled.
    unsafe {
        let p0 = ptr;
        let p1 = ptr.add(8);
        let v0 = _mm256_loadu_ps(p0);
        let v1 = _mm256_loadu_ps(p1);
        _mm256_storeu_ps(
            p0,
            _mm256_addsub_ps(
                _mm256_mul_ps(v0, v_cos),
                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), v_sin),
            ),
        );
        _mm256_storeu_ps(
            p1,
            _mm256_addsub_ps(
                _mm256_mul_ps(v1, v_cos),
                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v1), v_sin),
            ),
        );
        let p2 = ptr.add(16);
        let p3 = ptr.add(24);
        let v2 = _mm256_loadu_ps(p2);
        let v3 = _mm256_loadu_ps(p3);
        _mm256_storeu_ps(
            p2,
            _mm256_addsub_ps(
                _mm256_mul_ps(v2, v_cos),
                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v2), v_sin),
            ),
        );
        _mm256_storeu_ps(
            p3,
            _mm256_addsub_ps(
                _mm256_mul_ps(v3, v_cos),
                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v3), v_sin),
            ),
        );
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
pub(crate) unsafe fn avx2_rotate_so2_dispatch(
    ptr: *mut f32,
    len: usize,
    v_cos: __m256,
    v_sin: __m256,
    sin_t: f32,
    cos_t: f32,
) {
    // SAFETY: Caller guarantees `ptr` is valid for `len` floats and AVX2 is enabled.
    // Staged 2-YMM (16-float) sub-blocks keep peak live vector register count <= 6 (`ymm0..ymm5`),
    // eliminating callee-saved `xmm6..xmm15` stack spills on Windows x64 ABI while saturating both 256-bit FMA ports.
    unsafe {
        if len == 32 {
            avx2_rotate_so2_block32(ptr, v_cos, v_sin);
            return;
        }
        if len == 64 {
            avx2_rotate_so2_block32(ptr, v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(32), v_cos, v_sin);
            return;
        }
        if len == 128 {
            avx2_rotate_so2_block32(ptr, v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(32), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(64), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(96), v_cos, v_sin);
            return;
        }
        if len == 256 {
            avx2_rotate_so2_block32(ptr, v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(32), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(64), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(96), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(128), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(160), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(192), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(224), v_cos, v_sin);
            return;
        }
        let mut b = 0usize;
        if len >= 65_536 {
            // For large streaming buffers (`>= 256 KB`), step in 4KB-page chunks (`1024` floats)
            // issuing multi-page DTLB/L2 prefetches (`+8 KB`, `+16 KB`, `+32 KB`) outside a pure
            // 4-block (`128`-float, ~400B DSB-locked) inner loop with zero inner prefetch branches.
            while b + 1024 <= len {
                let page_ptr = ptr.add(b);
                if b + 2048 < len {
                    core::arch::x86_64::_mm_prefetch(
                        page_ptr.add(2048).cast::<i8>(),
                        core::arch::x86_64::_MM_HINT_T0,
                    );
                }
                if b + 4096 < len {
                    core::arch::x86_64::_mm_prefetch(
                        page_ptr.add(4096).cast::<i8>(),
                        core::arch::x86_64::_MM_HINT_T1,
                    );
                }
                if b + 8192 < len {
                    core::arch::x86_64::_mm_prefetch(
                        page_ptr.add(8192).cast::<i8>(),
                        core::arch::x86_64::_MM_HINT_T1,
                    );
                }
                let page_end = b + 1024;
                while b < page_end {
                    let p = ptr.add(b);
                    avx2_rotate_so2_block32(p, v_cos, v_sin);
                    avx2_rotate_so2_block32(p.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(p.add(64), v_cos, v_sin);
                    avx2_rotate_so2_block32(p.add(96), v_cos, v_sin);
                    b += 128;
                }
            }
        }
        while b + 128 <= len {
            let p = ptr.add(b);
            avx2_rotate_so2_block32(p, v_cos, v_sin);
            avx2_rotate_so2_block32(p.add(32), v_cos, v_sin);
            avx2_rotate_so2_block32(p.add(64), v_cos, v_sin);
            avx2_rotate_so2_block32(p.add(96), v_cos, v_sin);
            b += 128;
        }
        while b + 32 <= len {
            avx2_rotate_so2_block32(ptr.add(b), v_cos, v_sin);
            b += 32;
        }
        if b == len {
            return;
        }
        while b + 8 <= len {
            let p0 = ptr.add(b);
            let v0 = _mm256_loadu_ps(p0);
            _mm256_storeu_ps(
                p0,
                _mm256_addsub_ps(
                    _mm256_mul_ps(v0, v_cos),
                    _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), v_sin),
                ),
            );
            b += 8;
        }
        while b + 2 <= len {
            let (r0, r1) = branchless_rotate_so2_pair(*ptr.add(b), *ptr.add(b + 1), sin_t, cos_t);
            *ptr.add(b) = r0;
            *ptr.add(b + 1) = r1;
            b += 2;
        }
    }
}

/// Bank-conflict-free 64B cache-line-staged 3-tensor (`Q`, `K`, `V`) AVX2 $SO(2)$ 32-float block rotation.
///
/// Rotates `qp`, `kp`, and `vp` sequentially in 32-float (128-byte, 2-cache-line) blocks using shared
/// register-resident `(v_cos, v_sin)` (`<= 6` live YMM registers `ymm0..ymm5`, zero Windows x64
/// `xmm6..xmm15` callee-saved stack spills, zero L1D same-bank port conflicts, and zero 3-way 4KB
/// Load-Store Queue page-offset aliasing stalls).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_rotate_so2_triple_block32(
    qp: *mut f32,
    kp: *mut f32,
    vp: *mut f32,
    v_cos: __m256,
    v_sin: __m256,
) {
    // SAFETY: Caller guarantees `qp`, `kp`, `vp` are pairwise disjoint, valid for 32 floats, and AVX2 is enabled.
    unsafe {
        avx2_rotate_so2_block32(qp, v_cos, v_sin);
        avx2_rotate_so2_block32(kp, v_cos, v_sin);
        avx2_rotate_so2_block32(vp, v_cos, v_sin);
    }
}

/// Rotates `buf` in-place by `(sin_t, cos_t)` in strictly constant time (`0` data-dependent
/// early-return branches on `sin_t == 0.0 && cos_t == 1.0`, guaranteeing `1.00x` timing parity
/// between zero phase `0` and max saturation `+127`/`-128`).
#[inline(always)]
pub(crate) fn rotate_so2_slice_unrolled16(buf: &mut [f32], sin_t: f32, cos_t: f32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: `target_feature = "avx2"` is enabled and `buf` is valid for `buf.len()` floats.
        unsafe {
            avx2_rotate_so2_dispatch(
                buf.as_mut_ptr(),
                buf.len(),
                _mm256_set1_ps(cos_t),
                _mm256_set1_ps(sin_t),
                sin_t,
                cos_t,
            );
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for pair in buf.chunks_exact_mut(2) {
            (pair[0], pair[1]) = branchless_rotate_so2_pair(pair[0], pair[1], sin_t, cos_t);
        }
    }
}

/// Computes $\sum_i (a_i - b_i)^2$ using 4-way unrolled AVX2 SIMD accumulation.
#[inline(always)]
#[must_use]
pub(crate) fn diff_norm_sq_f32_unrolled4(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled at compile time, and `i + 32 <= len` is in-bounds.
        unsafe {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 32 <= len {
                let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
                let d1 = _mm256_sub_ps(
                    _mm256_loadu_ps(pa.add(i + 8)),
                    _mm256_loadu_ps(pb.add(i + 8)),
                );
                let d2 = _mm256_sub_ps(
                    _mm256_loadu_ps(pa.add(i + 16)),
                    _mm256_loadu_ps(pb.add(i + 16)),
                );
                let d3 = _mm256_sub_ps(
                    _mm256_loadu_ps(pa.add(i + 24)),
                    _mm256_loadu_ps(pb.add(i + 24)),
                );
                acc0 = _mm256_add_ps(
                    acc0,
                    _mm256_add_ps(_mm256_mul_ps(d0, d0), _mm256_mul_ps(d1, d1)),
                );
                acc1 = _mm256_add_ps(
                    acc1,
                    _mm256_add_ps(_mm256_mul_ps(d2, d2), _mm256_mul_ps(d3, d3)),
                );
                i += 32;
            }
            while i + 8 <= len {
                let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
                i += 8;
            }
            let mut total = avx2_hsum_ps(_mm256_add_ps(acc0, acc1));
            while i < len {
                let d = *pa.add(i) - *pb.add(i);
                total += d * d;
                i += 1;
            }
            total
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut s = [0.0_f32; 4];
        let mut a_chunks = a[..len].chunks_exact(4);
        let mut b_chunks = b[..len].chunks_exact(4);
        for (ac, bc) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
            for k in 0..4 {
                let d = ac[k] - bc[k];
                s[k] += d * d;
            }
        }
        let mut total = (s[0] + s[1]) + (s[2] + s[3]);
        for (&av, &bv) in a_chunks.remainder().iter().zip(b_chunks.remainder().iter()) {
            let d = av - bv;
            total += d * d;
        }
        total
    }
}

/// Computes $(\mathbf{a} \cdot \mathbf{b}, \|\mathbf{a}\|_2^2, \|\mathbf{b}\|_2^2)$ in a single AVX2 pass
/// while keeping peak live YMM usage `<= 5` (`0` Windows x64 `xmm6..xmm15` stack saves/restores).
#[inline(always)]
#[must_use]
pub(crate) fn dot_and_norms_f32_unrolled4(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    let len = a.len().min(b.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled at compile time, and all offsets are bounds-checked against `len`.
        // Accumulating into 3 YMMs (`dot_all, na_all, nb_all`) with 2 scratch YMMs (`va, vb`) keeps peak live YMMs
        // at 5 (`ymm0..ymm4`), completely eliminating 8 `vmovaps` stack saves/restores of `xmm6..xmm13` on Windows x64.
        unsafe {
            let mut dot_all = _mm256_setzero_ps();
            let mut na_all = _mm256_setzero_ps();
            let mut nb_all = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 32 <= len {
                let va0 = _mm256_loadu_ps(pa.add(i));
                let vb0 = _mm256_loadu_ps(pb.add(i));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(va0, vb0));
                na_all = _mm256_add_ps(na_all, _mm256_mul_ps(va0, va0));
                nb_all = _mm256_add_ps(nb_all, _mm256_mul_ps(vb0, vb0));

                let va1 = _mm256_loadu_ps(pa.add(i + 8));
                let vb1 = _mm256_loadu_ps(pb.add(i + 8));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(va1, vb1));
                na_all = _mm256_add_ps(na_all, _mm256_mul_ps(va1, va1));
                nb_all = _mm256_add_ps(nb_all, _mm256_mul_ps(vb1, vb1));

                let va2 = _mm256_loadu_ps(pa.add(i + 16));
                let vb2 = _mm256_loadu_ps(pb.add(i + 16));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(va2, vb2));
                na_all = _mm256_add_ps(na_all, _mm256_mul_ps(va2, va2));
                nb_all = _mm256_add_ps(nb_all, _mm256_mul_ps(vb2, vb2));

                let va3 = _mm256_loadu_ps(pa.add(i + 24));
                let vb3 = _mm256_loadu_ps(pb.add(i + 24));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(va3, vb3));
                na_all = _mm256_add_ps(na_all, _mm256_mul_ps(va3, va3));
                nb_all = _mm256_add_ps(nb_all, _mm256_mul_ps(vb3, vb3));
                i += 32;
            }
            let rem = len - i;
            if (rem & 16) != 0 {
                let va0 = _mm256_loadu_ps(pa.add(i));
                let vb0 = _mm256_loadu_ps(pb.add(i));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(va0, vb0));
                na_all = _mm256_add_ps(na_all, _mm256_mul_ps(va0, va0));
                nb_all = _mm256_add_ps(nb_all, _mm256_mul_ps(vb0, vb0));

                let va1 = _mm256_loadu_ps(pa.add(i + 8));
                let vb1 = _mm256_loadu_ps(pb.add(i + 8));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(va1, vb1));
                na_all = _mm256_add_ps(na_all, _mm256_mul_ps(va1, va1));
                nb_all = _mm256_add_ps(nb_all, _mm256_mul_ps(vb1, vb1));
                i += 16;
            }
            if (rem & 8) != 0 {
                let va0 = _mm256_loadu_ps(pa.add(i));
                let vb0 = _mm256_loadu_ps(pb.add(i));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(va0, vb0));
                na_all = _mm256_add_ps(na_all, _mm256_mul_ps(va0, va0));
                nb_all = _mm256_add_ps(nb_all, _mm256_mul_ps(vb0, vb0));
                i += 8;
            }
            let (mut dot, mut norm_a_sq, mut norm_b_sq) = avx2_hsum3_ps(dot_all, na_all, nb_all);
            while i < len {
                let av = *pa.add(i);
                let bv = *pb.add(i);
                dot += av * bv;
                norm_a_sq += av * av;
                norm_b_sq += bv * bv;
                i += 1;
            }
            (dot, norm_a_sq, norm_b_sq)
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let (mut dot, mut na, mut nb) = (0.0_f32, 0.0_f32, 0.0_f32);
        for (&av, &bv) in a[..len].iter().zip(b[..len].iter()) {
            dot += av * bv;
            na += av * av;
            nb += bv * bv;
        }
        (dot, na, nb)
    }
}

/// Computes $(\|\mathbf{r} - \mathbf{p}\|_2^2, \langle \mathbf{r} - \mathbf{p}, \boldsymbol{\gamma} \rangle, \|\boldsymbol{\gamma}\|_2^2)$
/// in a single fused AVX2 memory pass while keeping peak live YMM usage `<= 5`
/// (`0` Windows x64 `xmm6..xmm15` stack saves/restores).
#[inline]
#[must_use]
pub(crate) fn hodge_obstruction_and_dual_pairing_f32(
    r_slice: &[f32],
    p_slice: &[f32],
    g_slice: &[f32],
) -> (f32, f32, f32) {
    let n = r_slice.len().min(p_slice.len()).min(g_slice.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let pr = r_slice.as_ptr();
        let pp = p_slice.as_ptr();
        let pg = g_slice.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled at compile time, and pointer offsets are bounds-checked against `n`.
        // Using 3 accumulator YMMs (`obs_all, dot_all, gam_all`) + 2 scratch YMMs (`d, g`) keeps peak live YMMs
        // at 5 (`ymm0..ymm4`), eliminating 8 `vmovaps` stack saves/restores of `xmm6..xmm13` on Windows x64 ABI
        // and cutting `hodge_obstruction_and_dual_homology_pairing_64` variance and latency.
        unsafe {
            let mut obs_all = _mm256_setzero_ps();
            let mut dot_all = _mm256_setzero_ps();
            let mut gam_all = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 32 <= n {
                let d0 = _mm256_sub_ps(_mm256_loadu_ps(pr.add(i)), _mm256_loadu_ps(pp.add(i)));
                let g0 = _mm256_loadu_ps(pg.add(i));
                obs_all = _mm256_add_ps(obs_all, _mm256_mul_ps(d0, d0));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(d0, g0));
                gam_all = _mm256_add_ps(gam_all, _mm256_mul_ps(g0, g0));

                let d1 = _mm256_sub_ps(
                    _mm256_loadu_ps(pr.add(i + 8)),
                    _mm256_loadu_ps(pp.add(i + 8)),
                );
                let g1 = _mm256_loadu_ps(pg.add(i + 8));
                obs_all = _mm256_add_ps(obs_all, _mm256_mul_ps(d1, d1));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(d1, g1));
                gam_all = _mm256_add_ps(gam_all, _mm256_mul_ps(g1, g1));

                let d2 = _mm256_sub_ps(
                    _mm256_loadu_ps(pr.add(i + 16)),
                    _mm256_loadu_ps(pp.add(i + 16)),
                );
                let g2 = _mm256_loadu_ps(pg.add(i + 16));
                obs_all = _mm256_add_ps(obs_all, _mm256_mul_ps(d2, d2));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(d2, g2));
                gam_all = _mm256_add_ps(gam_all, _mm256_mul_ps(g2, g2));

                let d3 = _mm256_sub_ps(
                    _mm256_loadu_ps(pr.add(i + 24)),
                    _mm256_loadu_ps(pp.add(i + 24)),
                );
                let g3 = _mm256_loadu_ps(pg.add(i + 24));
                obs_all = _mm256_add_ps(obs_all, _mm256_mul_ps(d3, d3));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(d3, g3));
                gam_all = _mm256_add_ps(gam_all, _mm256_mul_ps(g3, g3));
                i += 32;
            }
            let rem = n - i;
            if (rem & 16) != 0 {
                let d0 = _mm256_sub_ps(_mm256_loadu_ps(pr.add(i)), _mm256_loadu_ps(pp.add(i)));
                let g0 = _mm256_loadu_ps(pg.add(i));
                obs_all = _mm256_add_ps(obs_all, _mm256_mul_ps(d0, d0));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(d0, g0));
                gam_all = _mm256_add_ps(gam_all, _mm256_mul_ps(g0, g0));

                let d1 = _mm256_sub_ps(
                    _mm256_loadu_ps(pr.add(i + 8)),
                    _mm256_loadu_ps(pp.add(i + 8)),
                );
                let g1 = _mm256_loadu_ps(pg.add(i + 8));
                obs_all = _mm256_add_ps(obs_all, _mm256_mul_ps(d1, d1));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(d1, g1));
                gam_all = _mm256_add_ps(gam_all, _mm256_mul_ps(g1, g1));
                i += 16;
            }
            if (rem & 8) != 0 {
                let d = _mm256_sub_ps(_mm256_loadu_ps(pr.add(i)), _mm256_loadu_ps(pp.add(i)));
                let g = _mm256_loadu_ps(pg.add(i));
                obs_all = _mm256_add_ps(obs_all, _mm256_mul_ps(d, d));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(d, g));
                gam_all = _mm256_add_ps(gam_all, _mm256_mul_ps(g, g));
                i += 8;
            }
            let (mut obs_sq, mut pair_dot, mut gam_sq) = avx2_hsum3_ps(obs_all, dot_all, gam_all);
            while i < n {
                let r_obs = *pr.add(i) - *pp.add(i);
                let gv = *pg.add(i);
                obs_sq += r_obs * r_obs;
                pair_dot += r_obs * gv;
                gam_sq += gv * gv;
                i += 1;
            }
            (obs_sq, pair_dot, gam_sq)
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut s_obs = [0.0_f32; 4];
        let mut s_dot = [0.0_f32; 4];
        let mut s_gam = [0.0_f32; 4];

        let mut r_c = r_slice[..n].chunks_exact(4);
        let mut p_c = p_slice[..n].chunks_exact(4);
        let mut g_c = g_slice[..n].chunks_exact(4);

        for ((rc, pc), gc) in r_c.by_ref().zip(p_c.by_ref()).zip(g_c.by_ref()) {
            for k in 0..4 {
                let r_obs = rc[k] - pc[k];
                let gv = gc[k];
                s_obs[k] += r_obs * r_obs;
                s_dot[k] += r_obs * gv;
                s_gam[k] += gv * gv;
            }
        }
        let mut obs_sq = (s_obs[0] + s_obs[1]) + (s_obs[2] + s_obs[3]);
        let mut pair_dot = (s_dot[0] + s_dot[1]) + (s_dot[2] + s_dot[3]);
        let mut gam_sq = (s_gam[0] + s_gam[1]) + (s_gam[2] + s_gam[3]);
        for ((&rv, &pv), &gv) in r_c
            .remainder()
            .iter()
            .zip(p_c.remainder().iter())
            .zip(g_c.remainder().iter())
        {
            let r_obs = rv - pv;
            obs_sq += r_obs * r_obs;
            pair_dot += r_obs * gv;
            gam_sq += gv * gv;
        }
        (obs_sq, pair_dot, gam_sq)
    }
}
