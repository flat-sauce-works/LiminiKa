// name: src/core/src/foundation/phase_ops/coherence.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! 64-head SWAR/AVX2 phase-vector conversions, shortest-arc $S^1$ SLERP, phase coherence,
//! and circular resultant statistics kernels.

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::*;

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
use crate::foundation::lut_crc32::Q14_BIAS_X64_I32;
use crate::foundation::lut_crc32::{
    INV_64_Q14_F32, INV_8_Q14_F32, Q14_BIAS_X8_I32, Q7_FULL_COS_I16_REF, Q7_FULL_COS_REF,
    Q7_FULL_SINCOS_BIASED_U64_REF,
};
use crate::foundation::scalar_math::{
    branchless_atan2_to_q7_fast, branchless_clamp_f32, branchless_flush_subnormal_f32,
    branchless_relu_f32, branchless_sincos_q7, branchless_von_mises_kappa, fast_sqrt_unit_f32,
    VON_MISES_KAPPA_AT_MAX_R,
};
use crate::foundation::swar_simd::{
    is_word_array_uniform_64, load_q7_words_64, swar_q7_circular_add_x8, swar_q7_circular_diff_x8,
    swar_q7_scale_le128_with_bias_x8,
};
use crate::liminal::{branchless_select_i8, PhaseQ7, DEFAULT_ENTROPY_EPS, MAX_ATTENTION_HEADS};

/// Unchecked 8-lane SWAR shortest-arc $S^1$ geodesic interpolation for `1 <= s <= 127` and
/// precomputed `bias_restore = (128 - s) * 0x0101_0101_0101_0101` (`0` branches).
#[inline(always)]
pub const fn swar_slerp_word_unchecked_x8(a: u64, b: u64, s: u64, bias_restore: u64) -> u64 {
    swar_q7_circular_add_x8(
        a,
        swar_q7_scale_le128_with_bias_x8(swar_q7_circular_diff_x8(b, a), s, bias_restore),
    )
}

/// Interpolates two packed 8-lane Q7 words (`a` and `b`) along the shortest-arc $S^1$ geodesic
/// using a pre-hoisted `bias_restore = (128 - s) * 0x0101_0101_0101_0101` (`0` data-dependent branches).
#[inline(always)]
pub const fn swar_slerp_word_with_bias_x8(a: u64, b: u64, s: u64, bias_restore: u64) -> u64 {
    if s == 0 {
        return a;
    }
    if s >= 128 {
        return b;
    }
    swar_slerp_word_unchecked_x8(a, b, s, bias_restore)
}

/// Moves 8 `u64` words directly into two 256-bit YMM registers `(v0, v1)` via 64-bit XMM/YMM
/// register inserts (`vmovq` + `vpinsrq` + `vinserti128`), avoiding the 64-bit-store-to-256-bit-load
/// Store-to-Load Forwarding (STLF) pipeline stall caused by stack `transmute` or `_mm256_loadu_si256`
/// on recently stored 64-bit word arrays.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
// Hot-path register-passed SIMD/SWAR kernel leaf kept flat to avoid stack-frame spills.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn avx2_u64x8_to_vec64(
    w0: u64,
    w1: u64,
    w2: u64,
    w3: u64,
    w4: u64,
    w5: u64,
    w6: u64,
    w7: u64,
) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`.
    unsafe {
        use core::arch::x86_64::{
            _mm256_castsi128_si256, _mm256_inserti128_si256, _mm_cvtsi64_si128, _mm_insert_epi64,
        };
        let lo0 = _mm_insert_epi64(_mm_cvtsi64_si128(w0 as i64), w1 as i64, 1);
        let hi0 = _mm_insert_epi64(_mm_cvtsi64_si128(w2 as i64), w3 as i64, 1);
        let lo1 = _mm_insert_epi64(_mm_cvtsi64_si128(w4 as i64), w5 as i64, 1);
        let hi1 = _mm_insert_epi64(_mm_cvtsi64_si128(w6 as i64), w7 as i64, 1);
        (
            _mm256_inserti128_si256(_mm256_castsi128_si256(lo0), hi0, 1),
            _mm256_inserti128_si256(_mm256_castsi128_si256(lo1), hi1, 1),
        )
    }
}

/// Moves a `&[u64; 8]` word array into two 256-bit YMM registers `(v0, v1)` via `avx2_u64x8_to_vec64`
/// without triggering a 4-to-1 Store-to-Load Forwarding stall.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_words64_to_vec64(
    w: &[u64; 8],
) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`.
    unsafe { avx2_u64x8_to_vec64(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]) }
}

/// Moves a `&[u64; 8]` word array into two 256-bit YMM registers `(v0, v1)` using a single
/// `vpxor` or `vmovq + vpbroadcastq` when `w_is_uniform` is `true` (`1..2` instructions vs `10`
/// Port-5 inserts), or `avx2_words64_to_vec64` when heterogeneous.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_words64_to_vec64_with_uniform(
    w: &[u64; 8],
    w_is_uniform: bool,
) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`.
    unsafe {
        use core::arch::x86_64::{
            _mm256_broadcastq_epi64, _mm256_setzero_si256, _mm_cvtsi64_si128,
        };
        if w_is_uniform {
            let w0 = w[0];
            if w0 == 0 {
                let z = _mm256_setzero_si256();
                return (z, z);
            }
            let v = _mm256_broadcastq_epi64(_mm_cvtsi64_si128(w0 as i64));
            return (v, v);
        }
        avx2_words64_to_vec64(w)
    }
}

/// Extracts two 256-bit YMM registers `(v0, v1)` directly into 8 `u64` GPR words (`[u64; 8]`)
/// via `vextracti128` + `vmovq` + `vpextrq`, avoiding the 256-bit-store-to-64-bit-load
/// Store-to-Load Forwarding (STLF) pipeline stall caused by stack `transmute`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_vec64_to_words64(
    v0: core::arch::x86_64::__m256i,
    v1: core::arch::x86_64::__m256i,
) -> [u64; 8] {
    // SAFETY: Caller guarantees `target_feature = "avx2"`.
    unsafe {
        use core::arch::x86_64::{
            _mm256_castsi256_si128, _mm256_extracti128_si256, _mm_cvtsi128_si64, _mm_extract_epi64,
        };
        let lo0 = _mm256_castsi256_si128(v0);
        let hi0 = _mm256_extracti128_si256(v0, 1);
        let lo1 = _mm256_castsi256_si128(v1);
        let hi1 = _mm256_extracti128_si256(v1, 1);
        [
            _mm_cvtsi128_si64(lo0) as u64,
            _mm_extract_epi64(lo0, 1) as u64,
            _mm_cvtsi128_si64(hi0) as u64,
            _mm_extract_epi64(hi0, 1) as u64,
            _mm_cvtsi128_si64(lo1) as u64,
            _mm_extract_epi64(lo1, 1) as u64,
            _mm_cvtsi128_si64(hi1) as u64,
            _mm_extract_epi64(hi1, 1) as u64,
        ]
    }
}

/// Scales precomputed circular difference vectors `(vd0, vd1) = (vb - va)` by `s in 0..=128`
/// and adds them onto `(va0, va1)` across all 64 Q7 lanes in YMM registers (`0` redundant `vpsubb`).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_slerp_from_diff_vec64(
    va0: core::arch::x86_64::__m256i,
    va1: core::arch::x86_64::__m256i,
    vd0: core::arch::x86_64::__m256i,
    vd1: core::arch::x86_64::__m256i,
    s: u64,
) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`.
    unsafe {
        use core::arch::x86_64::{
            _mm256_add_epi8, _mm256_and_si256, _mm256_mulhrs_epi16, _mm256_or_si256,
            _mm256_set1_epi16, _mm256_slli_epi16,
        };
        let v_s = _mm256_set1_epi16(s as i16);
        let mask_ff00 = _mm256_set1_epi16(-256i16);
        let mask_00ff = _mm256_set1_epi16(0x00FFi16);

        let ev0 = _mm256_mulhrs_epi16(_mm256_slli_epi16(vd0, 8), v_s);
        let od0 = _mm256_mulhrs_epi16(_mm256_and_si256(vd0, mask_ff00), v_s);
        let sc0 = _mm256_or_si256(_mm256_and_si256(ev0, mask_00ff), _mm256_slli_epi16(od0, 8));
        let out0 = _mm256_add_epi8(va0, sc0);

        let ev1 = _mm256_mulhrs_epi16(_mm256_slli_epi16(vd1, 8), v_s);
        let od1 = _mm256_mulhrs_epi16(_mm256_and_si256(vd1, mask_ff00), v_s);
        let sc1 = _mm256_or_si256(_mm256_and_si256(ev1, mask_00ff), _mm256_slli_epi16(od1, 8));
        let out1 = _mm256_add_epi8(va1, sc1);
        (out0, out1)
    }
}

/// Interpolates two 256-bit YMM register pairs `(va0, va1)` and `(vb0, vb1)` along the shortest-arc
/// $S^1$ geodesic across all 64 Q7 lanes in YMM registers (`0` memory or GPR roundtrips).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_slerp_vec64(
    va0: core::arch::x86_64::__m256i,
    va1: core::arch::x86_64::__m256i,
    vb0: core::arch::x86_64::__m256i,
    vb1: core::arch::x86_64::__m256i,
    s: u64,
) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`.
    unsafe {
        use core::arch::x86_64::_mm256_sub_epi8;
        avx2_slerp_from_diff_vec64(
            va0,
            va1,
            _mm256_sub_epi8(vb0, va0),
            _mm256_sub_epi8(vb1, va1),
            s,
        )
    }
}

/// Scales all 64 signed Q7 phase lanes in `(v0, v1)` by `s in 0..=128` (`128 == 1.0x`) using
/// `_mm256_mulhrs_epi16`, matching `swar_q7_scale_le128_with_bias_x8` bit-for-bit across all inputs.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_scale_vec64(
    v0: core::arch::x86_64::__m256i,
    v1: core::arch::x86_64::__m256i,
    s: u64,
) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`.
    unsafe {
        use core::arch::x86_64::{
            _mm256_and_si256, _mm256_mulhrs_epi16, _mm256_or_si256, _mm256_set1_epi16,
            _mm256_slli_epi16,
        };
        let v_s = _mm256_set1_epi16(s as i16);
        let mask_ff00 = _mm256_set1_epi16(-256i16);
        let mask_00ff = _mm256_set1_epi16(0x00FFi16);
        let ev0 = _mm256_mulhrs_epi16(_mm256_slli_epi16(v0, 8), v_s);
        let od0 = _mm256_mulhrs_epi16(_mm256_and_si256(v0, mask_ff00), v_s);
        let sc0 = _mm256_or_si256(_mm256_and_si256(ev0, mask_00ff), _mm256_slli_epi16(od0, 8));
        let ev1 = _mm256_mulhrs_epi16(_mm256_slli_epi16(v1, 8), v_s);
        let od1 = _mm256_mulhrs_epi16(_mm256_and_si256(v1, mask_ff00), v_s);
        let sc1 = _mm256_or_si256(_mm256_and_si256(ev1, mask_00ff), _mm256_slli_epi16(od1, 8));
        (sc0, sc1)
    }
}

/// Interpolates two 8-word (64-head) SWAR arrays (`a` and `b`) along the shortest-arc $S^1$
/// geodesic using pre-evaluated word-uniformity flags (`a_is_uniform`, `b_is_uniform`),
/// returning `(interpolated_words, out_is_uniform)` directly in 64-bit GPRs (`0` GPR-to-YMM or
/// YMM-to-GPR Store-to-Load Forwarding stalls, `1` shared `step0` scale when `b - a` is word-uniform).
#[inline(always)]
#[must_use]
pub const fn swar_slerp_words_with_uniform_flag_64(
    a: &[u64; 8],
    a_is_uniform: bool,
    b: &[u64; 8],
    b_is_uniform: bool,
    s: u64,
    bias_restore: u64,
) -> ([u64; 8], bool) {
    if s == 0 {
        return (*a, a_is_uniform);
    }
    if s >= 128 {
        return (*b, b_is_uniform);
    }
    if a_is_uniform & b_is_uniform {
        if a[0] == b[0] {
            return (*a, true);
        }
        let w0 = swar_slerp_word_unchecked_x8(a[0], b[0], s, bias_restore);
        return ([w0; 8], true);
    }
    let d0 = swar_q7_circular_diff_x8(b[0], a[0]);
    let d1 = swar_q7_circular_diff_x8(b[1], a[1]);
    let d2 = swar_q7_circular_diff_x8(b[2], a[2]);
    let d3 = swar_q7_circular_diff_x8(b[3], a[3]);
    let d4 = swar_q7_circular_diff_x8(b[4], a[4]);
    let d5 = swar_q7_circular_diff_x8(b[5], a[5]);
    let d6 = swar_q7_circular_diff_x8(b[6], a[6]);
    let d7 = swar_q7_circular_diff_x8(b[7], a[7]);
    if (d0 == d1)
        && ((((d0 ^ d2) | (d0 ^ d3)) | ((d0 ^ d4) | (d0 ^ d5))) | ((d0 ^ d6) | (d0 ^ d7))) == 0
    {
        if d0 == 0 {
            return (*a, a_is_uniform);
        }
        let step0 = swar_q7_scale_le128_with_bias_x8(d0, s, bias_restore);
        if step0 == 0 {
            return (*a, a_is_uniform);
        }
        let out = [
            swar_q7_circular_add_x8(a[0], step0),
            swar_q7_circular_add_x8(a[1], step0),
            swar_q7_circular_add_x8(a[2], step0),
            swar_q7_circular_add_x8(a[3], step0),
            swar_q7_circular_add_x8(a[4], step0),
            swar_q7_circular_add_x8(a[5], step0),
            swar_q7_circular_add_x8(a[6], step0),
            swar_q7_circular_add_x8(a[7], step0),
        ];
        return (out, a_is_uniform);
    }
    let out = [
        swar_q7_circular_add_x8(a[0], swar_q7_scale_le128_with_bias_x8(d0, s, bias_restore)),
        swar_q7_circular_add_x8(a[1], swar_q7_scale_le128_with_bias_x8(d1, s, bias_restore)),
        swar_q7_circular_add_x8(a[2], swar_q7_scale_le128_with_bias_x8(d2, s, bias_restore)),
        swar_q7_circular_add_x8(a[3], swar_q7_scale_le128_with_bias_x8(d3, s, bias_restore)),
        swar_q7_circular_add_x8(a[4], swar_q7_scale_le128_with_bias_x8(d4, s, bias_restore)),
        swar_q7_circular_add_x8(a[5], swar_q7_scale_le128_with_bias_x8(d5, s, bias_restore)),
        swar_q7_circular_add_x8(a[6], swar_q7_scale_le128_with_bias_x8(d6, s, bias_restore)),
        swar_q7_circular_add_x8(a[7], swar_q7_scale_le128_with_bias_x8(d7, s, bias_restore)),
    ];
    (out, is_word_array_uniform_64(&out))
}

/// Interpolates two 8-word (64-head) SWAR arrays (`a` and `b`) along the shortest-arc $S^1$
/// geodesic across all 8 words (`64` heads) in strictly constant time (`0` data-dependent branches).
#[inline(always)]
#[must_use]
pub const fn swar_slerp_words_with_bias_64(
    a: &[u64; 8],
    b: &[u64; 8],
    s: u64,
    bias_restore: u64,
) -> [u64; 8] {
    if s == 0 {
        return *a;
    }
    if s >= 128 {
        return *b;
    }
    let a_uni = is_word_array_uniform_64(a);
    let b_uni = if a_uni {
        is_word_array_uniform_64(b)
    } else {
        false
    };
    swar_slerp_words_with_uniform_flag_64(a, a_uni, b, b_uni, s, bias_restore).0
}

/// Sums 8 Q14 cosine values from a single packed `u64` circular-difference word in strictly
/// constant time (`0` branches) via a balanced 4-port ILP tree.
#[inline(always)]
pub(super) const fn sum_cos_q14_from_diff_word(dw: u64) -> i32 {
    let lut = Q7_FULL_COS_I16_REF;
    let b = dw.to_le_bytes();
    let c01 = (lut[b[0] as usize] as i32) + (lut[b[1] as usize] as i32);
    let c23 = (lut[b[2] as usize] as i32) + (lut[b[3] as usize] as i32);
    let c45 = (lut[b[4] as usize] as i32) + (lut[b[5] as usize] as i32);
    let c67 = (lut[b[6] as usize] as i32) + (lut[b[7] as usize] as i32);
    (c01 + c23) + (c45 + c67)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_coherence_from_diff_vec64_with_floor(
    vd0: core::arch::x86_64::__m256i,
    vd1: core::arch::x86_64::__m256i,
    floor_coh: f32,
) -> f32 {
    // SAFETY: Guarded by `target_feature = "avx2"`. Evaluates the 6-instruction `vpabsb + vpsadbw`
    // 2-piece Lagrangian $L_1$ upper-bound screen (`min(1.052 - s_l1 / 8192, 1.215 - s_l1 / 4096)`)
    // and `s_l1 == 0` exact-match fast path BEFORE `vpbroadcastb`/`vpbroadcastq` uniformity checks,
    // cutting per-pair rejection work in `merge_and_prune_centroids_64` and $k$-Means E-steps in half.
    unsafe {
        use core::arch::x86_64::{
            _mm256_abs_epi8, _mm256_add_epi64, _mm256_broadcastb_epi8, _mm256_broadcastq_epi64,
            _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_or_si256, _mm256_sad_epu8,
            _mm256_setzero_si256, _mm256_testz_si256, _mm256_xor_si256, _mm_add_epi64,
            _mm_cvtsi128_si32, _mm_cvtsi128_si64, _mm_unpackhi_epi64,
        };
        let raw_abs0 = _mm256_abs_epi8(vd0);
        let raw_abs1 = _mm256_abs_epi8(vd1);
        let zero = _mm256_setzero_si256();
        let sad_l1 = _mm256_add_epi64(
            _mm256_sad_epu8(raw_abs0, zero),
            _mm256_sad_epu8(raw_abs1, zero),
        );
        let l1_128 = _mm_add_epi64(
            _mm256_castsi256_si128(sad_l1),
            _mm256_extracti128_si256(sad_l1, 1),
        );
        let s_l1_u64 =
            _mm_cvtsi128_si64(_mm_add_epi64(l1_128, _mm_unpackhi_epi64(l1_128, l1_128))) as u64;
        if s_l1_u64 == 0 {
            return if 1.0 < floor_coh { -2.0 } else { 1.0 };
        }
        if floor_coh > -0.95 {
            let s_l1 = s_l1_u64 as f32;
            if (1.052_f32 - s_l1 * (1.0 / 8192.0)) < floor_coh
                || (1.215_f32 - s_l1 * (1.0 / 4096.0)) < floor_coh
            {
                return -2.0;
            }
        }
        let lo128 = _mm256_castsi256_si128(vd0);
        let splat_b0 = _mm256_broadcastb_epi8(lo128);
        let diff_b0 = _mm256_or_si256(
            _mm256_xor_si256(vd0, splat_b0),
            _mm256_xor_si256(vd1, splat_b0),
        );
        if _mm256_testz_si256(diff_b0, diff_b0) != 0 {
            let b0 = (_mm_cvtsi128_si32(lo128) as u32) & 0xFF;
            let coh = Q7_FULL_COS_REF[b0 as usize];
            return if coh < floor_coh { -2.0 } else { coh };
        }
        let splat_q0 = _mm256_broadcastq_epi64(lo128);
        let diff_q0 = _mm256_or_si256(
            _mm256_xor_si256(vd0, splat_q0),
            _mm256_xor_si256(vd1, splat_q0),
        );
        if _mm256_testz_si256(diff_q0, diff_q0) != 0 {
            let d0 = _mm_cvtsi128_si64(lo128) as u64;
            let coh = branchless_clamp_f32(
                (sum_cos_q14_from_diff_word(d0) as f32) * INV_8_Q14_F32,
                -1.0,
                1.0,
            );
            return if coh < floor_coh { -2.0 } else { coh };
        }
        avx2_coherence_from_raw_abs_vec64(raw_abs0, raw_abs1)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_coherence_from_diff_vec64(
    vd0: core::arch::x86_64::__m256i,
    vd1: core::arch::x86_64::__m256i,
) -> f32 {
    // SAFETY: Guarded by `target_feature = "avx2"`. Fast-paths byte-uniform shifts in 1 LUT load
    // (`0` GPR `imul`) via `vpbroadcastb + vptest` and period-8 word-uniform shifts via
    // `vpbroadcastq + vptest` before running the 16-knot trigonometric pipeline.
    unsafe {
        use core::arch::x86_64::{
            _mm256_abs_epi8, _mm256_broadcastb_epi8, _mm256_broadcastq_epi64,
            _mm256_castsi256_si128, _mm256_or_si256, _mm256_testz_si256, _mm256_xor_si256,
            _mm_cvtsi128_si32, _mm_cvtsi128_si64,
        };
        let lo128 = _mm256_castsi256_si128(vd0);
        let splat_b0 = _mm256_broadcastb_epi8(lo128);
        let diff_b0 = _mm256_or_si256(
            _mm256_xor_si256(vd0, splat_b0),
            _mm256_xor_si256(vd1, splat_b0),
        );
        if _mm256_testz_si256(diff_b0, diff_b0) != 0 {
            let b0 = (_mm_cvtsi128_si32(lo128) as u32) & 0xFF;
            return Q7_FULL_COS_REF[b0 as usize];
        }
        let splat_q0 = _mm256_broadcastq_epi64(lo128);
        let diff_q0 = _mm256_or_si256(
            _mm256_xor_si256(vd0, splat_q0),
            _mm256_xor_si256(vd1, splat_q0),
        );
        if _mm256_testz_si256(diff_q0, diff_q0) != 0 {
            let d0 = _mm_cvtsi128_si64(lo128) as u64;
            return branchless_clamp_f32(
                (sum_cos_q14_from_diff_word(d0) as f32) * INV_8_Q14_F32,
                -1.0,
                1.0,
            );
        }
        avx2_coherence_from_raw_abs_vec64(_mm256_abs_epi8(vd0), _mm256_abs_epi8(vd1))
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
unsafe fn avx2_coherence_from_raw_abs_vec64(
    raw_abs0: core::arch::x86_64::__m256i,
    raw_abs1: core::arch::x86_64::__m256i,
) -> f32 {
    // SAFETY: Guarded by `target_feature = "avx2"`. Evaluates 64-head cosine sum in registers
    // via 16-knot `vpshufb` + `vpsadbw` + saturation-free `vpmaddubsw` trigonometric angle-addition.
    unsafe {
        use core::arch::x86_64::{
            _mm256_add_epi16, _mm256_add_epi32, _mm256_add_epi64, _mm256_and_si256,
            _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_madd_epi16,
            _mm256_maddubs_epi16, _mm256_min_epu8, _mm256_sad_epu8, _mm256_set1_epi16,
            _mm256_set1_epi8, _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8,
            _mm256_slli_epi64, _mm256_srli_epi16, _mm_add_epi32, _mm_add_epi64, _mm_cvtsi128_si32,
            _mm_cvtsi128_si64, _mm_shuffle_epi32, _mm_unpackhi_epi64,
        };
        let max_127 = _mm256_set1_epi8(127);
        let abs0 = _mm256_min_epu8(raw_abs0, max_127);
        let abs1 = _mm256_min_epu8(raw_abs1, max_127);

        let mask_15 = _mm256_set1_epi8(0x0F);
        let h0 = _mm256_and_si256(_mm256_srli_epi16(abs0, 3), mask_15);
        let h1 = _mm256_and_si256(_mm256_srli_epi16(abs1, 3), mask_15);

        let lut_c0_lo = _mm256_setr_epi8(
            0, -59, 33, 55, 65, -114, 126, 124, 0, -124, -126, 114, -65, -55, -33, 59, 0, -59, 33,
            55, 65, -114, 126, 124, 0, -124, -126, 114, -65, -55, -33, 59,
        );
        let zero = _mm256_setzero_si256();
        let sad_lo = _mm256_add_epi64(
            _mm256_sad_epu8(_mm256_shuffle_epi8(lut_c0_lo, h0), zero),
            _mm256_sad_epu8(_mm256_shuffle_epi8(lut_c0_lo, h1), zero),
        );
        let lut_c0_hi = _mm256_setr_epi8(
            -128, 126, 123, 117, 109, 99, 88, 76, 64, 51, 39, 28, 18, 10, 4, 1, -128, 126, 123,
            117, 109, 99, 88, 76, 64, 51, 39, 28, 18, 10, 4, 1,
        );
        let sad_hi = _mm256_add_epi64(
            _mm256_sad_epu8(_mm256_shuffle_epi8(lut_c0_hi, h0), zero),
            _mm256_sad_epu8(_mm256_shuffle_epi8(lut_c0_hi, h1), zero),
        );
        let sad_tot = _mm256_add_epi64(sad_lo, _mm256_slli_epi64(sad_hi, 8));
        let sad_128 = _mm_add_epi64(
            _mm256_castsi256_si128(sad_tot),
            _mm256_extracti128_si256(sad_tot, 1),
        );
        let base_sum =
            _mm_cvtsi128_si64(_mm_add_epi64(sad_128, _mm_unpackhi_epi64(sad_128, sad_128))) as i32
                - (64 * 16384);

        let lut_nsin_h = _mm256_setr_epi8(
            0, -25, -49, -71, -90, -106, -117, -125, -127, -125, -117, -106, -90, -71, -49, -25, 0,
            -25, -49, -71, -90, -106, -117, -125, -127, -125, -117, -106, -90, -71, -49, -25,
        );
        let lut_sin_r = _mm256_setr_epi8(
            0, 18, 36, 54, 72, 90, 108, 126, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 36, 54, 72, 90, 108,
            126, 0, 0, 0, 0, 0, 0, 0, 0,
        );
        let lut_ncos_h = _mm256_setr_epi8(
            -127, -125, -117, -106, -90, -71, -49, -25, 0, 25, 49, 71, 90, 106, 117, 125, -127,
            -125, -117, -106, -90, -71, -49, -25, 0, 25, 49, 71, 90, 106, 117, 125,
        );
        let lut_omc_r = _mm256_setr_epi8(
            0, 0, 1, 2, 3, 6, 8, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 6, 8, 11, 0, 0, 0, 0,
            0, 0, 0, 0,
        );
        let r0 = _mm256_and_si256(abs0, _mm256_set1_epi8(0x07));
        let corr0 = _mm256_add_epi16(
            _mm256_maddubs_epi16(
                _mm256_shuffle_epi8(lut_sin_r, r0),
                _mm256_shuffle_epi8(lut_nsin_h, h0),
            ),
            _mm256_maddubs_epi16(
                _mm256_shuffle_epi8(lut_omc_r, r0),
                _mm256_shuffle_epi8(lut_ncos_h, h0),
            ),
        );
        let corr32_0 = _mm256_madd_epi16(corr0, _mm256_set1_epi16(1));

        let r1 = _mm256_and_si256(abs1, _mm256_set1_epi8(0x07));
        let corr1 = _mm256_add_epi16(
            _mm256_maddubs_epi16(
                _mm256_shuffle_epi8(lut_sin_r, r1),
                _mm256_shuffle_epi8(lut_nsin_h, h1),
            ),
            _mm256_maddubs_epi16(
                _mm256_shuffle_epi8(lut_omc_r, r1),
                _mm256_shuffle_epi8(lut_ncos_h, h1),
            ),
        );
        let corr32 = _mm256_add_epi32(corr32_0, _mm256_madd_epi16(corr1, _mm256_set1_epi16(1)));
        let c128 = _mm_add_epi32(
            _mm256_castsi256_si128(corr32),
            _mm256_extracti128_si256(corr32, 1),
        );
        let c64 = _mm_add_epi32(c128, _mm_shuffle_epi32(c128, 0x4E));
        let c32 = _mm_add_epi32(c64, _mm_shuffle_epi32(c64, 0xB1));
        let corr_sum = _mm_cvtsi128_si32(c32);

        let total_f32 = (base_sum as f32) + 0.175_520_92_f32 * (corr_sum as f32);
        branchless_clamp_f32(total_f32 * INV_64_Q14_F32, -1.0, 1.0)
    }
}

/// Evaluates normalized 64-head phase coherence $\mathcal{C}_{\mathrm{phase}} \in [-1.0, 1.0]$
/// from a single period-8 uniform circular-difference word `d0` (`d0 == d1 == ... == d7`) in
/// 1 LUT load when byte-uniform or 8 L1D `i16` lookups when heterogeneous (`0` SIMD instructions).
#[inline(always)]
#[must_use]
pub(crate) const fn coherence_from_uniform_diff_word_x8(d0: u64) -> f32 {
    if d0 == 0 {
        return 1.0;
    }
    if d0 == d0.rotate_right(8) {
        return Q7_FULL_COS_REF[(d0 & 0xFF) as usize];
    }
    branchless_clamp_f32(
        (sum_cos_q14_from_diff_word(d0) as f32) * INV_8_Q14_F32,
        -1.0,
        1.0,
    )
}

/// Evaluates normalized 64-head phase coherence $\mathcal{C}_{\mathrm{phase}}(a, b) \in [-1.0, 1.0]$
/// when both 64-head profiles are period-8 word-uniform (`a == [wa; 8]` and `b == [wb; 8]`),
/// operating strictly on the two 64-bit representative words `(wa, wb)` without touching the
/// 64-byte phase arrays or issuing vector instructions.
#[inline(always)]
#[must_use]
pub(crate) const fn coherence_from_uniform_words_x8(wa: u64, wb: u64) -> f32 {
    if wa == wb {
        return 1.0;
    }
    coherence_from_uniform_diff_word_x8(swar_q7_circular_diff_x8(wa, wb))
}

/// Evaluates normalized 64-head phase coherence $\mathcal{C}_{\mathrm{phase}} \in [-1.0, 1.0]$
/// directly from 8 SWAR Q7 circular difference words `(d0..=d7)`.
#[inline(always)]
#[must_use]
// v0.3.0 frozen public Rust API; parameter-struct consolidation deferred to v0.4.0.
#[allow(clippy::too_many_arguments)]
pub fn coherence_from_diff_words_64(
    d0: u64,
    d1: u64,
    d2: u64,
    d3: u64,
    d4: u64,
    d5: u64,
    d6: u64,
    d7: u64,
) -> f32 {
    if (((d0 ^ d1) | (d0 ^ d2)) | ((d0 ^ d3) | (d0 ^ d4)) | ((d0 ^ d5) | (d0 ^ d6)) | (d0 ^ d7))
        == 0
    {
        return coherence_from_uniform_diff_word_x8(d0);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Guarded by `target_feature = "avx2"`. Value-level `[u64; 8] -> [__m256i; 2]`
        // transmute avoids `_mm256_setr_epi64x` 4-to-1 Store-to-Load Forwarding stalls.
        unsafe {
            let (vd0, vd1) = avx2_u64x8_to_vec64(d0, d1, d2, d3, d4, d5, d6, d7);
            avx2_coherence_from_diff_vec64(vd0, vd1)
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let s01 = sum_cos_q14_from_diff_word(d0) + sum_cos_q14_from_diff_word(d1);
        let s23 = sum_cos_q14_from_diff_word(d2) + sum_cos_q14_from_diff_word(d3);
        let s45 = sum_cos_q14_from_diff_word(d4) + sum_cos_q14_from_diff_word(d5);
        let s67 = sum_cos_q14_from_diff_word(d6) + sum_cos_q14_from_diff_word(d7);
        let total_q14 = (s01 + s23) + (s45 + s67);
        branchless_clamp_f32((total_q14 as f32) * INV_64_Q14_F32, -1.0, 1.0)
    }
}

/// Evaluates normalized 64-head phase coherence $\mathcal{C}_{\mathrm{phase}}(a, b) \in [-1.0, 1.0]$
/// from two 8-word SWAR arrays (`a` and `b`) using pre-evaluated uniformity flags
/// (`a_is_uniform`, `b_is_uniform`), eliminating redundant 8-word XOR reductions inside multi-anchor
/// and multi-centroid loops.
#[inline(always)]
#[must_use]
pub fn coherence_from_word_arrays_with_both_uniform_flags_64(
    a: &[u64; 8],
    a_is_uniform: bool,
    b: &[u64; 8],
    b_is_uniform: bool,
) -> f32 {
    if a_is_uniform & b_is_uniform {
        return coherence_from_uniform_words_x8(a[0], b[0]);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Guarded by `target_feature = "avx2"`. Value-level `[u64; 8] -> [__m256i; 2]`
        // bitcast avoids 4-to-1 GPR-to-YMM Store-to-Load Forwarding stalls on stack-resident word arrays.
        unsafe {
            use core::arch::x86_64::_mm256_sub_epi8;
            let (va0, va1) = avx2_words64_to_vec64_with_uniform(a, a_is_uniform);
            let (vb0, vb1) = avx2_words64_to_vec64_with_uniform(b, b_is_uniform);
            avx2_coherence_from_diff_vec64(_mm256_sub_epi8(va0, vb0), _mm256_sub_epi8(va1, vb1))
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        coherence_from_diff_words_64(
            swar_q7_circular_diff_x8(a[0], b[0]),
            swar_q7_circular_diff_x8(a[1], b[1]),
            swar_q7_circular_diff_x8(a[2], b[2]),
            swar_q7_circular_diff_x8(a[3], b[3]),
            swar_q7_circular_diff_x8(a[4], b[4]),
            swar_q7_circular_diff_x8(a[5], b[5]),
            swar_q7_circular_diff_x8(a[6], b[6]),
            swar_q7_circular_diff_x8(a[7], b[7]),
        )
    }
}

/// Evaluates normalized 64-head phase coherence $\mathcal{C}_{\mathrm{phase}}(a, b) \in [-1.0, 1.0]$
/// when $\mathcal{C}_{\mathrm{phase}}(a, b) \ge \text{floor\_coh}$ is possible, or returns `-2.0`
/// in 5 AVX2 instructions (`vpabsb + vpsadbw`) when the exact Lagrangian $L_1$ upper bound
/// $\mathcal{C}_{64}(a, b) \le 1.21052 - \frac{1}{4096}\sum_{h=0}^{63}|d_h|$ proves $\mathcal{C}_{64}(a, b) < \text{floor\_coh}$.
#[inline(always)]
#[must_use]
pub fn coherence_above_floor_from_word_arrays_64(
    a: &[u64; 8],
    a_is_uniform: bool,
    b: &[u64; 8],
    b_is_uniform: bool,
    floor_coh: f32,
) -> f32 {
    if a_is_uniform & b_is_uniform {
        return coherence_from_uniform_words_x8(a[0], b[0]);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Guarded by `target_feature = "avx2"`.
        unsafe {
            use core::arch::x86_64::_mm256_sub_epi8;
            let (va0, va1) = avx2_words64_to_vec64_with_uniform(a, a_is_uniform);
            let (vb0, vb1) = avx2_words64_to_vec64_with_uniform(b, b_is_uniform);
            avx2_coherence_from_diff_vec64_with_floor(
                _mm256_sub_epi8(va0, vb0),
                _mm256_sub_epi8(va1, vb1),
                floor_coh,
            )
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let d0 = swar_q7_circular_diff_x8(a[0], b[0]);
        let d1 = swar_q7_circular_diff_x8(a[1], b[1]);
        let s01 = sum_cos_q14_from_diff_word(d0) + sum_cos_q14_from_diff_word(d1);
        if floor_coh > -0.95 && ((s01 as f32 + 48.05 * 16384.0) * INV_64_Q14_F32) < floor_coh {
            return -2.0;
        }
        let s23 = sum_cos_q14_from_diff_word(swar_q7_circular_diff_x8(a[2], b[2]))
            + sum_cos_q14_from_diff_word(swar_q7_circular_diff_x8(a[3], b[3]));
        let s45 = sum_cos_q14_from_diff_word(swar_q7_circular_diff_x8(a[4], b[4]))
            + sum_cos_q14_from_diff_word(swar_q7_circular_diff_x8(a[5], b[5]));
        let s67 = sum_cos_q14_from_diff_word(swar_q7_circular_diff_x8(a[6], b[6]))
            + sum_cos_q14_from_diff_word(swar_q7_circular_diff_x8(a[7], b[7]));
        let total_q14 = (s01 + s23) + (s45 + s67);
        branchless_clamp_f32((total_q14 as f32) * INV_64_Q14_F32, -1.0, 1.0)
    }
}

/// Evaluates normalized 64-head phase coherence $\mathcal{C}_{\mathrm{phase}}(a, b) \in [-1.0, 1.0]$
/// when `a_is_uniform = is_word_array_uniform_64(a)` has already been hoisted outside a loop.
#[inline(always)]
#[must_use]
pub fn coherence_from_word_arrays_with_uniform_flag_64(
    a: &[u64; 8],
    a_is_uniform: bool,
    b: &[u64; 8],
) -> f32 {
    let b_is_uniform = if a_is_uniform {
        is_word_array_uniform_64(b)
    } else {
        false
    };
    coherence_from_word_arrays_with_both_uniform_flags_64(a, a_is_uniform, b, b_is_uniform)
}

/// Evaluates normalized 64-head phase coherence $\mathcal{C}_{\mathrm{phase}}(a, b) \in [-1.0, 1.0]$
/// from two 8-word SWAR arrays (`a` and `b`) in strictly constant time (`0` branches), fast-pathing
/// identical (`1.0`), byte-uniform (`1` LUT load), and word-uniform (`8` L1D LUT loads) arrays.
#[inline(always)]
#[must_use]
pub fn coherence_from_word_arrays_64(a: &[u64; 8], b: &[u64; 8]) -> f32 {
    coherence_from_word_arrays_with_uniform_flag_64(a, is_word_array_uniform_64(a), b)
}

/// Evaluates the mean circular phase coherence $\mathcal{C}_{\mathrm{phase}}(a, b) \in [-1.0, 1.0]$
/// using 8-word SWAR circular differences and register-shifted Q14 cosine lookups.
#[inline(always)]
pub fn branchless_q7_coherence_64(
    phases_a: &[PhaseQ7; MAX_ATTENTION_HEADS],
    phases_b: &[PhaseQ7; MAX_ATTENTION_HEADS],
) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: `phases_a` and `phases_b` are 64-byte arrays and AVX2 is enabled.
        // Direct `_mm256_loadu_si256` avoids 64-bit GPR stack copies and 4-to-1 STLF stalls.
        unsafe {
            use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_sub_epi8};
            let pa = phases_a.as_ptr().cast::<__m256i>();
            let pb = phases_b.as_ptr().cast::<__m256i>();
            let va0 = _mm256_loadu_si256(pa);
            let va1 = _mm256_loadu_si256(pa.add(1));
            let vb0 = _mm256_loadu_si256(pb);
            let vb1 = _mm256_loadu_si256(pb.add(1));
            avx2_coherence_from_diff_vec64(_mm256_sub_epi8(va0, vb0), _mm256_sub_epi8(va1, vb1))
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        coherence_from_word_arrays_64(&load_q7_words_64(phases_a), &load_q7_words_64(phases_b))
    }
}

/// Evaluates **Zero-Hyperparameter Phase-Proximity Gravity** on the unified phase manifold,
/// loading `trail_phases` and `anchor_phases` once into 64-bit registers (`0` redundant reloads)
/// and bypassing the 64-head Q14 LUT reduction when either profile is zero.
#[inline(always)]
pub fn branchless_q7_resonant_pull_64(
    trail_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    anchor_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    base_energy: f32,
) -> (f32, f32) {
    let clean_energy = branchless_flush_subnormal_f32(branchless_relu_f32(base_energy));
    if clean_energy <= DEFAULT_ENTROPY_EPS {
        return (0.0, 0.0);
    }
    let t = load_q7_words_64(trail_phases);
    let b = load_q7_words_64(anchor_phases);
    let both_nonzero = ((((t[0] | t[1]) | (t[2] | t[3])) | ((t[4] | t[5]) | (t[6] | t[7]))) != 0)
        & ((((b[0] | b[1]) | (b[2] | b[3])) | ((b[4] | b[5]) | (b[6] | b[7]))) != 0);
    let eff_coh = if both_nonzero {
        coherence_from_word_arrays_64(&t, &b)
    } else {
        0.0
    };
    let proximity_gain = branchless_clamp_f32(1.0 + eff_coh, 0.25, 2.0);
    (
        branchless_flush_subnormal_f32(clean_energy * proximity_gain),
        eff_coh,
    )
}

/// Returns `true` if any of the 8 64-bit SWAR phase words contains a non-zero Q7 lane.
#[inline(always)]
#[must_use]
pub const fn swar_words_nonzero_64(words: &[u64; 8]) -> bool {
    if words[0] != 0 {
        return true;
    }
    (((words[1] | words[2]) | (words[3] | words[4])) | ((words[5] | words[6]) | words[7])) != 0
}

/// Returns `true` if two 8-word (64-head) SWAR phase profiles are bit-for-bit identical.
#[inline(always)]
#[must_use]
pub const fn swar_words_equal_64(a: &[u64; 8], b: &[u64; 8]) -> bool {
    if a[0] != b[0] {
        return false;
    }
    ((((a[1] ^ b[1]) | (a[2] ^ b[2])) | ((a[3] ^ b[3]) | (a[4] ^ b[4])))
        | (((a[5] ^ b[5]) | (a[6] ^ b[6])) | (a[7] ^ b[7])))
        == 0
}

/// Accumulates 8 biased `(sin_q14 + 16384, cos_q14 + 16384)` pairs from a single `u64` word `w`
/// into a single `u64` register using `Q7_FULL_SINCOS_BIASED_U64_LUT` in strictly constant time
/// (`0` data-dependent branches).
#[inline(always)]
pub(super) const fn sum_sincos_biased_u64_from_word(w: u64) -> u64 {
    let lut = Q7_FULL_SINCOS_BIASED_U64_REF;
    let b = w.to_le_bytes();
    let p01 = lut[b[0] as usize].wrapping_add(lut[b[1] as usize]);
    let p23 = lut[b[2] as usize].wrapping_add(lut[b[3] as usize]);
    let p45 = lut[b[4] as usize].wrapping_add(lut[b[5] as usize]);
    let p67 = lut[b[6] as usize].wrapping_add(lut[b[7] as usize]);
    p01.wrapping_add(p23).wrapping_add(p45.wrapping_add(p67))
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline(always)]
pub(super) const fn sum_sincos_q14_from_words_scalar_64(w: &[u64; 8]) -> (f32, f32) {
    let s01 =
        sum_sincos_biased_u64_from_word(w[0]).wrapping_add(sum_sincos_biased_u64_from_word(w[1]));
    let s23 =
        sum_sincos_biased_u64_from_word(w[2]).wrapping_add(sum_sincos_biased_u64_from_word(w[3]));
    let s45 =
        sum_sincos_biased_u64_from_word(w[4]).wrapping_add(sum_sincos_biased_u64_from_word(w[5]));
    let s67 =
        sum_sincos_biased_u64_from_word(w[6]).wrapping_add(sum_sincos_biased_u64_from_word(w[7]));
    let total = s01.wrapping_add(s23).wrapping_add(s45.wrapping_add(s67));
    let sum_cos_q14 = ((total & 0xFFFF_FFFF) as i32) - Q14_BIAS_X64_I32;
    let sum_sin_q14 = ((total >> 32) as i32) - Q14_BIAS_X64_I32;
    (
        (sum_sin_q14 as f32) * INV_64_Q14_F32,
        (sum_cos_q14 as f32) * INV_64_Q14_F32,
    )
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn avx2_mean_sincos_from_words_64(w: &[u64; 8]) -> (f32, f32) {
    // SAFETY: Guarded by `target_feature = "avx2"`.
    unsafe {
        let (v0, v1) = avx2_words64_to_vec64(w);
        avx2_mean_sincos_from_vec64(v0, v1)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
pub(super) unsafe fn avx2_mean_sincos_from_vec64(
    v0: core::arch::x86_64::__m256i,
    v1: core::arch::x86_64::__m256i,
) -> (f32, f32) {
    // SAFETY: Guarded by `target_feature = "avx2"`. Evaluates 64-head (mean_sin, mean_cos) in
    // registers via 16-nibble `vpshufb` + `vpsadbw` + `vpmaddubsw` with packed horizontal reduction.
    unsafe {
        use core::arch::x86_64::{
            _mm256_add_epi16, _mm256_add_epi32, _mm256_add_epi64, _mm256_and_si256,
            _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_madd_epi16,
            _mm256_maddubs_epi16, _mm256_or_si256, _mm256_sad_epu8, _mm256_set1_epi16,
            _mm256_set1_epi8, _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8,
            _mm256_slli_epi64, _mm256_srli_epi16, _mm256_sub_epi16, _mm_add_epi32, _mm_add_epi64,
            _mm_cvtsi128_si32, _mm_cvtsi128_si64, _mm_extract_epi32, _mm_shuffle_epi32,
            _mm_unpackhi_epi64, _mm_unpacklo_epi64,
        };

        let mask_15 = _mm256_set1_epi8(0x0F);
        let h0 = _mm256_and_si256(_mm256_srli_epi16(v0, 4), mask_15);
        let h1 = _mm256_and_si256(_mm256_srli_epi16(v1, 4), mask_15);

        let lut_cos0_lo = _mm256_setr_epi8(
            0, 33, 65, 126, 0, -126, -65, -33, 0, -33, -65, -126, 0, 126, 65, 33, 0, 33, 65, 126,
            0, -126, -65, -33, 0, -33, -65, -126, 0, 126, 65, 33,
        );
        let lut_cos0_hi = _mm256_setr_epi8(
            -128, 123, 109, 88, 64, 39, 18, 4, 0, 4, 18, 39, 64, 88, 109, 123, -128, 123, 109, 88,
            64, 39, 18, 4, 0, 4, 18, 39, 64, 88, 109, 123,
        );
        let zero = _mm256_setzero_si256();
        let c_sad_lo = _mm256_add_epi64(
            _mm256_sad_epu8(_mm256_shuffle_epi8(lut_cos0_lo, h0), zero),
            _mm256_sad_epu8(_mm256_shuffle_epi8(lut_cos0_lo, h1), zero),
        );
        let c_sad_hi = _mm256_add_epi64(
            _mm256_sad_epu8(_mm256_shuffle_epi8(lut_cos0_hi, h0), zero),
            _mm256_sad_epu8(_mm256_shuffle_epi8(lut_cos0_hi, h1), zero),
        );
        let c_sad_tot = _mm256_add_epi64(c_sad_lo, _mm256_slli_epi64(c_sad_hi, 8));

        let lut_sin0_lo = _mm256_setr_epi8(
            0, 126, 65, 33, 0, 33, 65, 126, 0, -126, -65, -33, 0, -33, -65, -126, 0, 126, 65, 33,
            0, 33, 65, 126, 0, -126, -65, -33, 0, -33, -65, -126,
        );
        let lut_sin0_hi = _mm256_setr_epi8(
            64, 88, 109, 123, -128, 123, 109, 88, 64, 39, 18, 4, 0, 4, 18, 39, 64, 88, 109, 123,
            -128, 123, 109, 88, 64, 39, 18, 4, 0, 4, 18, 39,
        );
        let s_sad_lo = _mm256_add_epi64(
            _mm256_sad_epu8(_mm256_shuffle_epi8(lut_sin0_lo, h0), zero),
            _mm256_sad_epu8(_mm256_shuffle_epi8(lut_sin0_lo, h1), zero),
        );
        let s_sad_hi = _mm256_add_epi64(
            _mm256_sad_epu8(_mm256_shuffle_epi8(lut_sin0_hi, h0), zero),
            _mm256_sad_epu8(_mm256_shuffle_epi8(lut_sin0_hi, h1), zero),
        );
        let s_sad_tot = _mm256_add_epi64(s_sad_lo, _mm256_slli_epi64(s_sad_hi, 8));

        let cs_sad = _mm256_or_si256(c_sad_tot, _mm256_slli_epi64(s_sad_tot, 32));
        let cs_sad_128 = _mm_add_epi64(
            _mm256_castsi256_si128(cs_sad),
            _mm256_extracti128_si256(cs_sad, 1),
        );
        let cs_base_packed = _mm_cvtsi128_si64(_mm_add_epi64(
            cs_sad_128,
            _mm_unpackhi_epi64(cs_sad_128, cs_sad_128),
        )) as u64;
        let cos_base = (cs_base_packed as i32) - (64 * 16384);
        let sin_base = ((cs_base_packed >> 32) as i32) - (64 * 16384);

        let lut_cos_h = _mm256_setr_epi8(
            127, 117, 90, 49, 0, -49, -90, -117, -127, -117, -90, -49, 0, 49, 90, 117, 127, 117,
            90, 49, 0, -49, -90, -117, -127, -117, -90, -49, 0, 49, 90, 117,
        );
        let lut_nsin_h = _mm256_setr_epi8(
            0, -49, -90, -117, -127, -117, -90, -49, 0, 49, 90, 117, 127, 117, 90, 49, 0, -49, -90,
            -117, -127, -117, -90, -49, 0, 49, 90, 117, 127, 117, 90, 49,
        );
        let lut_sin_l = _mm256_setr_epi8(
            0, 6, 13, 19, 25, 32, 38, 44, 50, 57, 63, 69, 75, 81, 87, 93, 0, 6, 13, 19, 25, 32, 38,
            44, 50, 57, 63, 69, 75, 81, 87, 93,
        );
        let lut_omc_l = _mm256_setr_epi8(
            0, 0, 0, 1, 1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 15, 17, 0, 0, 0, 1, 1, 2, 3, 4, 5, 6, 8, 9,
            11, 13, 15, 17,
        );

        let l0 = _mm256_and_si256(v0, _mm256_set1_epi8(0x0F));
        let ch0 = _mm256_shuffle_epi8(lut_cos_h, h0);
        let nsh0 = _mm256_shuffle_epi8(lut_nsin_h, h0);
        let sl0 = _mm256_shuffle_epi8(lut_sin_l, l0);
        let ol0 = _mm256_shuffle_epi8(lut_omc_l, l0);
        let c_corr32_0 = _mm256_madd_epi16(
            _mm256_sub_epi16(
                _mm256_maddubs_epi16(sl0, nsh0),
                _mm256_maddubs_epi16(ol0, ch0),
            ),
            _mm256_set1_epi16(1),
        );
        let s_corr32_0 = _mm256_madd_epi16(
            _mm256_add_epi16(
                _mm256_maddubs_epi16(sl0, ch0),
                _mm256_maddubs_epi16(ol0, nsh0),
            ),
            _mm256_set1_epi16(1),
        );

        let l1 = _mm256_and_si256(v1, _mm256_set1_epi8(0x0F));
        let ch1 = _mm256_shuffle_epi8(lut_cos_h, h1);
        let nsh1 = _mm256_shuffle_epi8(lut_nsin_h, h1);
        let sl1 = _mm256_shuffle_epi8(lut_sin_l, l1);
        let ol1 = _mm256_shuffle_epi8(lut_omc_l, l1);
        let c_corr32 = _mm256_add_epi32(
            c_corr32_0,
            _mm256_madd_epi16(
                _mm256_sub_epi16(
                    _mm256_maddubs_epi16(sl1, nsh1),
                    _mm256_maddubs_epi16(ol1, ch1),
                ),
                _mm256_set1_epi16(1),
            ),
        );
        let s_corr32 = _mm256_add_epi32(
            s_corr32_0,
            _mm256_madd_epi16(
                _mm256_add_epi16(
                    _mm256_maddubs_epi16(sl1, ch1),
                    _mm256_maddubs_epi16(ol1, nsh1),
                ),
                _mm256_set1_epi16(1),
            ),
        );

        let cc128 = _mm_add_epi32(
            _mm256_castsi256_si128(c_corr32),
            _mm256_extracti128_si256(c_corr32, 1),
        );
        let sc128 = _mm_add_epi32(
            _mm256_castsi256_si128(s_corr32),
            _mm256_extracti128_si256(s_corr32, 1),
        );
        let pair64 = _mm_add_epi32(
            _mm_unpacklo_epi64(cc128, sc128),
            _mm_unpackhi_epi64(cc128, sc128),
        );
        let pair32 = _mm_add_epi32(pair64, _mm_shuffle_epi32(pair64, 0xB1));
        let cos_corr = _mm_cvtsi128_si32(pair32);
        let sin_corr = _mm_extract_epi32::<2>(pair32);

        let mean_sin = ((sin_base as f32) + 0.5_f32 * (sin_corr as f32)) * INV_64_Q14_F32;
        let mean_cos = ((cos_base as f32) + 0.5_f32 * (cos_corr as f32)) * INV_64_Q14_F32;
        (mean_sin, mean_cos)
    }
}

#[inline(always)]
pub(super) fn mean_sincos_from_words_with_uniform_flag_64(
    w: &[u64; 8],
    w_is_uniform: bool,
) -> (f32, f32) {
    if w_is_uniform {
        let w0 = w[0];
        if w0 == w0.rotate_right(8) {
            return branchless_sincos_q7((w0 & 0xFF) as u8 as PhaseQ7);
        }
        let p = sum_sincos_biased_u64_from_word(w0);
        let sum_cos_q14 = ((p & 0xFFFF_FFFF) as i32) - Q14_BIAS_X8_I32;
        let sum_sin_q14 = ((p >> 32) as i32) - Q14_BIAS_X8_I32;
        return (
            (sum_sin_q14 as f32) * INV_8_Q14_F32,
            (sum_cos_q14 as f32) * INV_8_Q14_F32,
        );
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Guarded by `target_feature = "avx2"`.
        unsafe { avx2_mean_sincos_from_words_64(w) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        sum_sincos_q14_from_words_scalar_64(w)
    }
}

/// Evaluates the circular resultant vector length $\bar{R}_c \in [0.0, 1.0]$ from 8 register-resident
/// `u64` SWAR words (`w`) and a pre-evaluated word-uniformity flag (`w_is_uniform`).
#[inline(always)]
#[must_use]
pub fn circular_r_bar_from_words_with_uniform_flag_64(w: &[u64; 8], w_is_uniform: bool) -> f32 {
    if w_is_uniform {
        let w0 = w[0];
        if w0 == w0.rotate_right(8) {
            return 1.0;
        }
    }
    let (mean_sin, mean_cos) = mean_sincos_from_words_with_uniform_flag_64(w, w_is_uniform);
    fast_sqrt_unit_f32(mean_sin * mean_sin + mean_cos * mean_cos)
}

/// Evaluates the circular resultant vector length
/// $\bar{R} = \sqrt{\left(\frac{1}{64}\sum_{h=0}^{63}\sin\theta_h\right)^2 + \left(\frac{1}{64}\sum_{h=0}^{63}\cos\theta_h\right)^2} \in [0.0, 1.0]$
/// directly from all 8 packed 64-bit SWAR words (`w`) in strictly constant time (`0` branches, `0 fdiv`).
#[inline(always)]
pub fn circular_r_bar_from_words_64(w: &[u64; 8]) -> f32 {
    circular_r_bar_from_words_with_uniform_flag_64(w, is_word_array_uniform_64(w))
}

/// Evaluates the 64-head circular statistics summary `(r_bar, mean_phase_q7, kappa_c)` directly
/// from 8 register-resident `u64` SWAR words (`0` intermediate stack arrays, `0` data-dependent
/// branches, `0 fdiv`).
#[inline(always)]
#[must_use]
pub fn circular_resultant_from_words_with_uniform_flag_64(
    w: &[u64; 8],
    w_is_uniform: bool,
) -> (f32, PhaseQ7, f32) {
    if w_is_uniform {
        let w0 = w[0];
        if w0 == w0.rotate_right(8) {
            let q7 = (w0 & 0xFF) as u8 as PhaseQ7;
            return (
                1.0,
                branchless_select_i8(q7 == -128, 127, q7),
                VON_MISES_KAPPA_AT_MAX_R,
            );
        }
    }
    let (mean_sin, mean_cos) = mean_sincos_from_words_with_uniform_flag_64(w, w_is_uniform);
    let r_bar = fast_sqrt_unit_f32(mean_sin * mean_sin + mean_cos * mean_cos);
    let mean_q7 = branchless_atan2_to_q7_fast(mean_sin, mean_cos);
    (r_bar, mean_q7, branchless_von_mises_kappa(r_bar))
}

/// Evaluates the 64-head circular statistics summary `(r_bar, mean_phase_q7, kappa_c)` (Section 4.2)
/// directly from pre-loaded 8-word SWAR registers `w` (`0` unaligned byte loads).
#[inline(always)]
#[must_use]
pub fn circular_resultant_from_words_64(w: &[u64; 8]) -> (f32, PhaseQ7, f32) {
    circular_resultant_from_words_with_uniform_flag_64(w, is_word_array_uniform_64(w))
}

/// Evaluates both 64-head phase coherence $\mathcal{C}_{\mathrm{phase}}(a, b) \in [-1.0, 1.0]$ and
/// circular resultant statistics $(\bar{R}_a, \bar{\theta}_{a,\mathrm{Q7}}, \kappa_a)$ in a single
/// YMM-resident pass on AVX2 (`va0, va1` loaded once via `_mm256_loadu_si256` and shared across both
/// `avx2_coherence_from_diff_vec64` and `avx2_mean_sincos_from_vec64` with `0` Port-5 `vpinsrq` instructions).
#[inline(always)]
pub fn branchless_q7_coherence_and_resultant_64(
    phases_a: &[PhaseQ7; MAX_ATTENTION_HEADS],
    phases_b: &[PhaseQ7; MAX_ATTENTION_HEADS],
) -> (f32, f32, PhaseQ7, f32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: `phases_a` and `phases_b` are 64 bytes each and `target_feature = "avx2"` is enabled.
        unsafe {
            use core::arch::x86_64::{
                __m256i, _mm256_broadcastb_epi8, _mm256_castsi256_si128, _mm256_loadu_si256,
                _mm256_or_si256, _mm256_sub_epi8, _mm256_testz_si256, _mm256_xor_si256,
            };
            let pa = phases_a.as_ptr().cast::<__m256i>();
            let pb = phases_b.as_ptr().cast::<__m256i>();
            let vd0 = _mm256_sub_epi8(_mm256_loadu_si256(pa), _mm256_loadu_si256(pb));
            let vd1 = _mm256_sub_epi8(_mm256_loadu_si256(pa.add(1)), _mm256_loadu_si256(pb.add(1)));
            let vd_or = _mm256_or_si256(vd0, vd1);
            let coherence = if _mm256_testz_si256(vd_or, vd_or) != 0 {
                1.0
            } else {
                let bcast_d = _mm256_broadcastb_epi8(_mm256_castsi256_si128(vd0));
                let d_diff = _mm256_or_si256(
                    _mm256_xor_si256(vd0, bcast_d),
                    _mm256_xor_si256(vd1, bcast_d),
                );
                if _mm256_testz_si256(d_diff, d_diff) != 0 {
                    let d0 = (phases_a[0] as u8).wrapping_sub(phases_b[0] as u8);
                    Q7_FULL_COS_REF[d0 as usize]
                } else {
                    avx2_coherence_from_diff_vec64(vd0, vd1)
                }
            };
            let va0 = _mm256_loadu_si256(pa);
            let va1 = _mm256_loadu_si256(pa.add(1));
            let bcast_a = _mm256_broadcastb_epi8(_mm256_castsi256_si128(va0));
            let a_diff = _mm256_or_si256(
                _mm256_xor_si256(va0, bcast_a),
                _mm256_xor_si256(va1, bcast_a),
            );
            if _mm256_testz_si256(a_diff, a_diff) != 0 {
                let q7 = phases_a[0];
                return (
                    coherence,
                    1.0,
                    branchless_select_i8(q7 == -128, 127, q7),
                    VON_MISES_KAPPA_AT_MAX_R,
                );
            }
            let (mean_sin, mean_cos) = avx2_mean_sincos_from_vec64(va0, va1);
            let r_bar = fast_sqrt_unit_f32(mean_sin * mean_sin + mean_cos * mean_cos);
            let mean_q7 = branchless_atan2_to_q7_fast(mean_sin, mean_cos);
            (coherence, r_bar, mean_q7, branchless_von_mises_kappa(r_bar))
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let wa = load_q7_words_64(phases_a);
        let wb = load_q7_words_64(phases_b);
        let wa_is_uniform = is_word_array_uniform_64(&wa);
        let coherence = coherence_from_word_arrays_with_uniform_flag_64(&wa, wa_is_uniform, &wb);
        let (r_bar, mean_q7, kappa) =
            circular_resultant_from_words_with_uniform_flag_64(&wa, wa_is_uniform);
        (coherence, r_bar, mean_q7, kappa)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_hsum_ps(v: __m256) -> f32 {
    // SAFETY: Caller guarantees AVX2 is enabled.
    unsafe {
        let v128 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let v64 = _mm_add_ps(v128, _mm_movehl_ps(v128, v128));
        _mm_cvtss_f32(_mm_add_ss(v64, _mm_shuffle_ps::<0x55>(v64, v64)))
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_hsum2_ps(v0: __m256, v1: __m256) -> (f32, f32) {
    // SAFETY: Caller guarantees AVX2 is enabled.
    unsafe {
        let s0 = _mm_add_ps(_mm256_castps256_ps128(v0), _mm256_extractf128_ps::<1>(v0));
        let s1 = _mm_add_ps(_mm256_castps256_ps128(v1), _mm256_extractf128_ps::<1>(v1));
        let p64 = _mm_add_ps(_mm_unpacklo_ps(s0, s1), _mm_unpackhi_ps(s0, s1));
        let p32 = _mm_add_ps(p64, _mm_movehl_ps(p64, p64));
        (
            _mm_cvtss_f32(p32),
            _mm_cvtss_f32(_mm_shuffle_ps::<0x55>(p32, p32)),
        )
    }
}
