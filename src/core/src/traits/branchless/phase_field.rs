// name: src/core/src/traits/branchless/phase_field.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! 64-head SWAR/AVX2 phase-field algebra (coherence, SLERP, unified-field, Bayesian attention, circular resultant, centroids).

use super::*;

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
        use core::arch::x86_64::{_mm256_broadcastq_epi64, _mm256_setzero_si256, _mm_cvtsi64_si128};
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

/// Filters all 64 signed Q7 phase lanes in `(v0, v1)` against `min_step_q7` in 8 AVX2 instructions
/// (`vpabsb + vpmaxub + vpcmpeqb + vpand`), matching `swar_q7_qdps_filter_x8` bit-for-bit.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_qdps_filter_vec64(
    v0: core::arch::x86_64::__m256i,
    v1: core::arch::x86_64::__m256i,
    min_step_q7: u8,
) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`.
    unsafe {
        use core::arch::x86_64::{
            _mm256_abs_epi8, _mm256_and_si256, _mm256_cmpeq_epi8, _mm256_max_epu8,
            _mm256_set1_epi8, _mm256_setzero_si256,
        };
        if min_step_q7 <= 1 {
            return (v0, v1);
        }
        if min_step_q7 > 127 {
            let z = _mm256_setzero_si256();
            return (z, z);
        }
        let v_step = _mm256_set1_epi8(min_step_q7 as i8);
        let a0 = _mm256_abs_epi8(v0);
        let a1 = _mm256_abs_epi8(v1);
        let k0 = _mm256_cmpeq_epi8(_mm256_max_epu8(a0, v_step), a0);
        let k1 = _mm256_cmpeq_epi8(_mm256_max_epu8(a1, v_step), a1);
        (_mm256_and_si256(v0, k0), _mm256_and_si256(v1, k1))
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
    let b_uni = if a_uni { is_word_array_uniform_64(b) } else { false };
    swar_slerp_words_with_uniform_flag_64(a, a_uni, b, b_uni, s, bias_restore).0
}

/// True bit-parallel SWAR 8-lane shortest-arc geodesic interpolation on $S^1$ (`t_u7 in 0..=128`).
#[inline(always)]
pub const fn swar_q7_geodesic_slerp_x8(a_packed: u64, b_packed: u64, t_u7: u8) -> u64 {
    let s = branchless_select_u8(t_u7 > 128, 128, t_u7) as u64;
    let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
    swar_slerp_word_with_bias_x8(a_packed, b_packed, s, bias_restore)
}

/// Performs shortest-arc geodesic interpolation on $S^1$ across all 64 attention heads in strictly
/// constant time, hoisting `bias_restore` once across all 8 SWAR words.
#[inline(always)]
pub fn branchless_q7_slerp_64(
    phases_a: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    phases_b: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    t_u7: u8,
) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
    let s = branchless_select_u8(t_u7 > 128, 128, t_u7) as u64;
    let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let a = load_q7_words_64(phases_a);
    let b = load_q7_words_64(phases_b);
    let out_w = swar_slerp_words_with_bias_64(&a, &b, s, bias_restore);
    let mut out = [0i8; MAX_ATTENTION_HEADS];
    store_q7_words_64(&mut out, &out_w);
    out
}

/// Folds a candidate 64-head phase vector `candidate_phases` (with weight `candidate_weight`)
/// into `running_phases` (with weight `running_weight`) along the shortest-arc $S^1$ geodesic
/// in strictly constant time (`0 fdiv`). Returns the updated total weight.
#[inline(always)]
pub fn branchless_q7_circular_weighted_step_64(
    running_phases: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    running_weight: f32,
    candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    candidate_weight: f32,
) -> f32 {
    let w_run = branchless_flush_subnormal_f32(branchless_relu_f32(running_weight));
    let w_cand = branchless_flush_subnormal_f32(branchless_relu_f32(candidate_weight));
    let total_w = w_run + w_cand;
    if w_cand <= DEFAULT_ENTROPY_EPS {
        return total_w;
    }
    if w_run <= DEFAULT_ENTROPY_EPS {
        *running_phases = *candidate_phases;
        return total_w;
    }
    let ratio_f32 = branchless_clamp_f32(
        (w_cand * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
        0.0,
        128.0,
    );
    let s = (ratio_f32 as u8) as u64;
    if s >= 128 {
        *running_phases = *candidate_phases;
    } else if s > 0 {
        let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let a = load_q7_words_64(running_phases);
        let b = load_q7_words_64(candidate_phases);
        let a_uni = is_word_array_uniform_64(&a);
        let b_uni = if a_uni {
            is_word_array_uniform_64(&b)
        } else {
            false
        };
        let (out_w, _) =
            swar_slerp_words_with_uniform_flag_64(&a, a_uni, &b, b_uni, s, bias_restore);
        store_q7_words_64(running_phases, &out_w);
    }
    total_w
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
            _mm256_slli_epi64, _mm256_srli_epi16, _mm_add_epi32, _mm_add_epi64,
            _mm_cvtsi128_si32, _mm_cvtsi128_si64, _mm_shuffle_epi32, _mm_unpackhi_epi64,
        };
        let max_127 = _mm256_set1_epi8(127);
        let abs0 = _mm256_min_epu8(raw_abs0, max_127);
        let abs1 = _mm256_min_epu8(raw_abs1, max_127);

        let mask_15 = _mm256_set1_epi8(0x0F);
        let h0 = _mm256_and_si256(_mm256_srli_epi16(abs0, 3), mask_15);
        let h1 = _mm256_and_si256(_mm256_srli_epi16(abs1, 3), mask_15);

        let lut_c0_lo = _mm256_setr_epi8(
            0, -59, 33, 55, 65, -114, 126, 124, 0, -124, -126, 114, -65, -55, -33, 59,
            0, -59, 33, 55, 65, -114, 126, 124, 0, -124, -126, 114, -65, -55, -33, 59,
        );
        let zero = _mm256_setzero_si256();
        let sad_lo = _mm256_add_epi64(
            _mm256_sad_epu8(_mm256_shuffle_epi8(lut_c0_lo, h0), zero),
            _mm256_sad_epu8(_mm256_shuffle_epi8(lut_c0_lo, h1), zero),
        );
        let lut_c0_hi = _mm256_setr_epi8(
            -128, 126, 123, 117, 109, 99, 88, 76, 64, 51, 39, 28, 18, 10, 4, 1,
            -128, 126, 123, 117, 109, 99, 88, 76, 64, 51, 39, 28, 18, 10, 4, 1,
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
            _mm_cvtsi128_si64(_mm_add_epi64(sad_128, _mm_unpackhi_epi64(sad_128, sad_128)))
            as i32
            - (64 * 16384);

        let lut_nsin_h = _mm256_setr_epi8(
            0, -25, -49, -71, -90, -106, -117, -125, -127, -125, -117, -106, -90, -71, -49, -25,
            0, -25, -49, -71, -90, -106, -117, -125, -127, -125, -117, -106, -90, -71, -49, -25,
        );
        let lut_sin_r = _mm256_setr_epi8(
            0, 18, 36, 54, 72, 90, 108, 126, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 18, 36, 54, 72, 90, 108, 126, 0, 0, 0, 0, 0, 0, 0, 0,
        );
        let lut_ncos_h = _mm256_setr_epi8(
            -127, -125, -117, -106, -90, -71, -49, -25, 0, 25, 49, 71, 90, 106, 117, 125,
            -127, -125, -117, -106, -90, -71, -49, -25, 0, 25, 49, 71, 90, 106, 117, 125,
        );
        let lut_omc_r = _mm256_setr_epi8(
            0, 0, 1, 2, 3, 6, 8, 11, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 2, 3, 6, 8, 11, 0, 0, 0, 0, 0, 0, 0, 0,
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
        return unsafe {
            let (vd0, vd1) = avx2_u64x8_to_vec64(d0, d1, d2, d3, d4, d5, d6, d7);
            avx2_coherence_from_diff_vec64(vd0, vd1)
        };
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
        return unsafe {
            use core::arch::x86_64::_mm256_sub_epi8;
            let (va0, va1) = avx2_words64_to_vec64_with_uniform(a, a_is_uniform);
            let (vb0, vb1) = avx2_words64_to_vec64_with_uniform(b, b_is_uniform);
            avx2_coherence_from_diff_vec64(_mm256_sub_epi8(va0, vb0), _mm256_sub_epi8(va1, vb1))
        };
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
        return unsafe {
            use core::arch::x86_64::_mm256_sub_epi8;
            let (va0, va1) = avx2_words64_to_vec64_with_uniform(a, a_is_uniform);
            let (vb0, vb1) = avx2_words64_to_vec64_with_uniform(b, b_is_uniform);
            avx2_coherence_from_diff_vec64_with_floor(
                _mm256_sub_epi8(va0, vb0),
                _mm256_sub_epi8(va1, vb1),
                floor_coh,
            )
        };
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
    phases_a: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    phases_b: &[gcso_q7_t; MAX_ATTENTION_HEADS],
) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: `phases_a` and `phases_b` are 64-byte arrays and AVX2 is enabled.
        // Direct `_mm256_loadu_si256` avoids 64-bit GPR stack copies and 4-to-1 STLF stalls.
        return unsafe {
            use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_sub_epi8};
            let pa = phases_a.as_ptr().cast::<__m256i>();
            let pb = phases_b.as_ptr().cast::<__m256i>();
            let va0 = _mm256_loadu_si256(pa);
            let va1 = _mm256_loadu_si256(pa.add(1));
            let vb0 = _mm256_loadu_si256(pb);
            let vb1 = _mm256_loadu_si256(pb.add(1));
            avx2_coherence_from_diff_vec64(_mm256_sub_epi8(va0, vb0), _mm256_sub_epi8(va1, vb1))
        };
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
    trail_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    base_energy: f32,
) -> (f32, f32) {
    let clean_energy = branchless_flush_subnormal_f32(branchless_relu_f32(base_energy));
    if clean_energy <= DEFAULT_ENTROPY_EPS {
        return (0.0, 0.0);
    }
    let t = load_q7_words_64(trail_phases);
    let b = load_q7_words_64(anchor_phases);
    let both_nonzero =
        ((((t[0] | t[1]) | (t[2] | t[3])) | ((t[4] | t[5]) | (t[6] | t[7]))) != 0)
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

/// Register-resident Unified-Field folder returning `([u64; 8], total_pull, mean_coherence, dominant_idx, active_count)`.
///
/// Includes a zero-division, zero-SLERP single-anchor fast path (`count == 1 && init_w <= 1.0e-12`)
/// and guards 64-head Q14 cosine LUT lookups behind `trail_nonzero & anchor_nonzero`.
/// Order-invariant (commutative) folding of an arbitrary slice of anchors
/// (`anchor_phases`, `anchor_base_energies`) directly in 8 `u64` SWAR / AVX2 registers,
/// returning `(folded_words, total_weight, mean_coherence, dominant_anchor_idx, active_anchor_count, any_resonant, any_liminal)`.
#[inline]
pub(crate) fn branchless_q7_unified_field_fold_words_with_flags_64(
    trail_words: &[u64; 8],
    trail_init_weight: f32,
    default_dominant_idx: u32,
    anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
    anchor_base_energies: &[f32],
) -> ([u64; 8], f32, f32, u32, u32, bool, bool) {
    let count = branchless_select_usize(
        anchor_phases.len() < anchor_base_energies.len(),
        anchor_phases.len(),
        anchor_base_energies.len(),
    );
    let init_w = branchless_flush_subnormal_f32(branchless_relu_f32(trail_init_weight));
    let t = *trail_words;
    if count == 0 {
        return (t, init_w, 0.0, default_dominant_idx, 0, false, false);
    }
    let trail_nonzero = (((t[0] | t[1]) | (t[2] | t[3])) | ((t[4] | t[5]) | (t[6] | t[7]))) != 0;

    if count == 1 {
        let raw_e = anchor_base_energies[0];
        let clean_energy = if branchless_is_finite_f32(raw_e) {
            branchless_flush_subnormal_f32(branchless_relu_f32(raw_e))
        } else {
            0.0
        };
        if clean_energy <= DEFAULT_ENTROPY_EPS {
            return (t, init_w, 0.0, default_dominant_idx, 0, false, false);
        }
        let b = load_q7_words_64(&anchor_phases[0]);
        let anchor_nonzero =
            (((b[0] | b[1]) | (b[2] | b[3])) | ((b[4] | b[5]) | (b[6] | b[7]))) != 0;
        let coh = if trail_nonzero & anchor_nonzero {
            coherence_from_word_arrays_64(&t, &b)
        } else {
            0.0
        };
        let res_pull =
            branchless_flush_subnormal_f32(
                clean_energy * branchless_clamp_f32(1.0 + coh, 0.25, 2.0),
            );
        let is_active = res_pull > DEFAULT_ENTROPY_EPS;
        if !is_active {
            return (t, init_w, 0.0, default_dominant_idx, 0, false, false);
        }
        let total_w = init_w + res_pull;
        let out_w = if init_w <= DEFAULT_ENTROPY_EPS {
            b
        } else {
            let ratio_f32 = branchless_clamp_f32(
                (res_pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                0.0,
                128.0,
            );
            let s = (ratio_f32 as u8) as u64;
            let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
            swar_slerp_words_with_bias_64(&t, &b, s, bias_restore)
        };
        return (
            out_w,
            branchless_flush_subnormal_f32(total_w),
            coh,
            0,
            1,
            coh > 0.25,
            coh < -0.10,
        );
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        return unsafe {
            use core::arch::x86_64::{
                __m256i, _mm256_loadu_si256, _mm256_or_si256, _mm256_set1_epi64x, _mm256_sub_epi8,
                _mm256_testz_si256,
            };
            let t_uni = is_word_array_uniform_64(&t);
            let t_w0 = t[0];
            let t_b0 = (t_w0 & 0xFF) as u8;
            let t_is_byte_uniform =
                t_uni & (t_w0 == t_w0.rotate_right(8));
            let mut vt_vecs: Option<(__m256i, __m256i)> = if t_uni {
                None
            } else {
                Some(avx2_words64_to_vec64_with_uniform(&t, false))
            };
            let mut r_uni = t_uni;
            let mut r_w0 = t_w0;
            let mut vr_vecs: Option<(__m256i, __m256i)> = vt_vecs;
            let mut running_weight = init_w;
            let mut anchor_pull_sum = 0.0_f32;
            let mut weighted_coh_sum = 0.0_f32;
            let mut best_pull = 0.0_f32;
            let mut dominant_idx = default_dominant_idx;
            let mut active_count = 0u32;
            let mut any_resonant = false;
            let mut any_liminal = false;

            let mut idx = 0usize;
            while idx < count {
                let raw_e = anchor_base_energies[idx];
                let clean_energy = if branchless_is_finite_f32(raw_e) {
                    branchless_flush_subnormal_f32(branchless_relu_f32(raw_e))
                } else {
                    0.0
                };
                if clean_energy > DEFAULT_ENTROPY_EPS {
                    let check_b_uni =
                        (running_weight <= DEFAULT_ENTROPY_EPS) | (r_uni & (t_uni | !trail_nonzero));
                    let mut b = [0u64; 8];
                    let b_uni = if check_b_uni {
                        b = load_q7_words_64(&anchor_phases[idx]);
                        is_word_array_uniform_64(&b)
                    } else {
                        false
                    };
                    if b_uni & r_uni & (t_uni | !trail_nonzero) {
                        let bw0 = b[0];
                        let coh = if !(trail_nonzero & (bw0 != 0)) {
                            0.0
                        } else if bw0 == t_w0 {
                            1.0
                        } else if t_is_byte_uniform
                            && (bw0 == bw0.rotate_right(8))
                        {
                            Q7_FULL_COS_REF[t_b0.wrapping_sub(bw0 as u8) as usize]
                        } else {
                            coherence_from_word_arrays_with_both_uniform_flags_64(
                                &t, true, &b, true,
                            )
                        };
                        let res_pull = branchless_flush_subnormal_f32(
                            clean_energy * branchless_clamp_f32(1.0 + coh, 0.25, 2.0),
                        );
                        let is_active = res_pull > DEFAULT_ENTROPY_EPS;
                        let is_better = res_pull > best_pull;
                        let total_w = running_weight + res_pull;

                        if running_weight <= DEFAULT_ENTROPY_EPS {
                            if is_active {
                                r_w0 = bw0;
                            }
                        } else if is_active {
                            let ratio_f32 = branchless_clamp_f32(
                                (res_pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                                0.0,
                                128.0,
                            );
                            let s = (ratio_f32 as u8) as u64;
                            if s >= 128 {
                                r_w0 = bw0;
                            } else if s > 0 && r_w0 != bw0 {
                                let rb0 = (r_w0 & 0xFF) as u8;
                                let bb0 = (bw0 & 0xFF) as u8;
                                if r_w0 == r_w0.rotate_right(8)
                                    && bw0 == bw0.rotate_right(8)
                                {
                                    let b_lerp = branchless_q7_circular_lerp(
                                        rb0 as gcso_q7_t,
                                        bb0 as gcso_q7_t,
                                        s as u8,
                                    );
                                    r_w0 = ((b_lerp as u8) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
                                } else {
                                    let bias_restore = (128u64.wrapping_sub(s))
                                        .wrapping_mul(SWAR_SPLAT_BYTE_64);
                                    r_w0 = swar_slerp_word_unchecked_x8(
                                        r_w0,
                                        bw0,
                                        s,
                                        bias_restore,
                                    );
                                }
                            }
                        }
                        running_weight = total_w;
                        anchor_pull_sum += branchless_select_f32(is_active, res_pull, 0.0);
                        weighted_coh_sum += branchless_select_f32(is_active, res_pull * coh, 0.0);
                        best_pull = branchless_select_f32(is_better, res_pull, best_pull);
                        dominant_idx =
                            branchless_select_u32(is_better, idx as u32, dominant_idx);
                        active_count = active_count.wrapping_add(u32::from(is_active));
                        any_resonant |= is_active & (coh > 0.25);
                        any_liminal |= is_active & (coh < -0.10);
                    } else {
                        let b_ptr = anchor_phases[idx].as_ptr().cast::<__m256i>();
                        let vb0 = _mm256_loadu_si256(b_ptr);
                        let vb1 = _mm256_loadu_si256(b_ptr.add(1));
                        let coh = if !trail_nonzero {
                            0.0
                        } else if t_uni & b_uni {
                            let bw0 = b[0];
                            if bw0 == 0 {
                                0.0
                            } else if bw0 == t_w0 {
                                1.0
                            } else {
                                coherence_from_word_arrays_with_both_uniform_flags_64(
                                    &t, true, &b, true,
                                )
                            }
                        } else {
                            let (vt0, vt1) = match vt_vecs {
                                Some(v) => v,
                                None => {
                                    let tz = _mm256_set1_epi64x(t_w0 as i64);
                                    let v = (tz, tz);
                                    vt_vecs = Some(v);
                                    v
                                }
                            };
                            let b_or = _mm256_or_si256(vb0, vb1);
                            if _mm256_testz_si256(b_or, b_or) != 0 {
                                0.0
                            } else {
                                avx2_coherence_from_diff_vec64(
                                    _mm256_sub_epi8(vt0, vb0),
                                    _mm256_sub_epi8(vt1, vb1),
                                )
                            }
                        };
                        let res_pull = branchless_flush_subnormal_f32(
                            clean_energy * branchless_clamp_f32(1.0 + coh, 0.25, 2.0),
                        );
                        let is_active = res_pull > DEFAULT_ENTROPY_EPS;
                        let is_better = res_pull > best_pull;
                        let total_w = running_weight + res_pull;

                        if running_weight <= DEFAULT_ENTROPY_EPS {
                            if is_active {
                                if b_uni {
                                    r_w0 = b[0];
                                    r_uni = true;
                                } else {
                                    vr_vecs = Some((vb0, vb1));
                                    r_uni = false;
                                }
                            }
                        } else if is_active {
                            let (mut vr0, mut vr1) = match vr_vecs {
                                Some(v) if !r_uni => v,
                                _ => {
                                    let rz = _mm256_set1_epi64x(r_w0 as i64);
                                    (rz, rz)
                                }
                            };
                            r_uni = false;
                            let ratio_f32 = branchless_clamp_f32(
                                (res_pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                                0.0,
                                128.0,
                            );
                            let s = (ratio_f32 as u8) as u64;
                            if s >= 128 {
                                if b_uni {
                                    r_w0 = b[0];
                                    r_uni = true;
                                } else {
                                    vr_vecs = Some((vb0, vb1));
                                }
                            } else {
                                if s > 0 {
                                    let diff0 = _mm256_sub_epi8(vb0, vr0);
                                    let diff1 = _mm256_sub_epi8(vb1, vr1);
                                    let diff_or = _mm256_or_si256(diff0, diff1);
                                    if _mm256_testz_si256(diff_or, diff_or) == 0 {
                                        (vr0, vr1) =
                                            avx2_slerp_from_diff_vec64(vr0, vr1, diff0, diff1, s);
                                    }
                                }
                                vr_vecs = Some((vr0, vr1));
                            }
                        }

                        running_weight = total_w;
                        anchor_pull_sum += branchless_select_f32(is_active, res_pull, 0.0);
                        weighted_coh_sum += branchless_select_f32(is_active, res_pull * coh, 0.0);
                        best_pull = branchless_select_f32(is_better, res_pull, best_pull);
                        dominant_idx =
                            branchless_select_u32(is_better, idx as u32, dominant_idx);
                        active_count = active_count.wrapping_add(u32::from(is_active));
                        any_resonant |= is_active & (coh > 0.25);
                        any_liminal |= is_active & (coh < -0.10);
                    }
                }
                idx += 1;
            }

            let has_pull = (active_count > 0) & (anchor_pull_sum > DEFAULT_ENTROPY_EPS);
            let inv_pull =
                fast_inv_pos_normal_f32(branchless_select_f32(has_pull, anchor_pull_sum, 1.0));
            let mean_coh = branchless_select_f32(
                has_pull,
                branchless_clamp_f32(weighted_coh_sum * inv_pull, -1.0, 1.0),
                0.0,
            );
            let out_w = if r_uni {
                [r_w0; 8]
            } else if let Some((vr0, vr1)) = vr_vecs {
                avx2_vec64_to_words64(vr0, vr1)
            } else {
                [r_w0; 8]
            };
            (
                out_w,
                branchless_flush_subnormal_f32(running_weight),
                mean_coh,
                dominant_idx,
                active_count,
                any_resonant,
                any_liminal,
            )
        };
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let trail_is_uniform = is_word_array_uniform_64(&t);
        let mut r = t;
        let mut r_is_uniform = trail_is_uniform;
        let mut running_weight = init_w;
        let mut anchor_pull_sum = 0.0_f32;
        let mut weighted_coh_sum = 0.0_f32;
        let mut best_pull = 0.0_f32;
        let mut dominant_idx = default_dominant_idx;
        let mut active_count = 0u32;
        let mut any_resonant = false;
        let mut any_liminal = false;

        let mut idx = 0usize;
        while idx < count {
            let raw_e = anchor_base_energies[idx];
            let clean_energy = if branchless_is_finite_f32(raw_e) {
                branchless_flush_subnormal_f32(branchless_relu_f32(raw_e))
            } else {
                0.0
            };
            if clean_energy > DEFAULT_ENTROPY_EPS {
                let b = load_q7_words_64(&anchor_phases[idx]);
                let b_is_uniform = is_word_array_uniform_64(&b);
                let anchor_nonzero = if b_is_uniform {
                    b[0] != 0
                } else {
                    (((b[0] | b[1]) | (b[2] | b[3])) | ((b[4] | b[5]) | (b[6] | b[7]))) != 0
                };
                let coh = if trail_nonzero & anchor_nonzero {
                    coherence_from_word_arrays_with_both_uniform_flags_64(
                        &t,
                        trail_is_uniform,
                        &b,
                        b_is_uniform,
                    )
                } else {
                    0.0
                };
                let res_pull = branchless_flush_subnormal_f32(
                    clean_energy * branchless_clamp_f32(1.0 + coh, 0.25, 2.0),
                );
                let is_active = res_pull > DEFAULT_ENTROPY_EPS;
                let is_better = res_pull > best_pull;
                let total_w = running_weight + res_pull;

                if running_weight <= DEFAULT_ENTROPY_EPS {
                    if is_active {
                        r = b;
                        r_is_uniform = b_is_uniform;
                    }
                } else if is_active {
                    let ratio_f32 = branchless_clamp_f32(
                        (res_pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                        0.0,
                        128.0,
                    );
                    let s = (ratio_f32 as u8) as u64;
                    let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                    let (next_r, next_uni) = swar_slerp_words_with_uniform_flag_64(
                        &r,
                        r_is_uniform,
                        &b,
                        b_is_uniform,
                        s,
                        bias_restore,
                    );
                    r = next_r;
                    r_is_uniform = next_uni;
                }

                running_weight = total_w;
                anchor_pull_sum += branchless_select_f32(is_active, res_pull, 0.0);
                weighted_coh_sum += branchless_select_f32(is_active, res_pull * coh, 0.0);
                best_pull = branchless_select_f32(is_better, res_pull, best_pull);
                dominant_idx = branchless_select_u32(is_better, idx as u32, dominant_idx);
                active_count = active_count.wrapping_add(u32::from(is_active));
                any_resonant |= is_active & (coh > 0.25);
                any_liminal |= is_active & (coh < -0.10);
            }
            idx += 1;
        }

        let has_pull = (active_count > 0) & (anchor_pull_sum > DEFAULT_ENTROPY_EPS);
        let inv_pull =
            fast_inv_pos_normal_f32(branchless_select_f32(has_pull, anchor_pull_sum, 1.0));
        let mean_coh = branchless_select_f32(
            has_pull,
            branchless_clamp_f32(weighted_coh_sum * inv_pull, -1.0, 1.0),
            0.0,
        );
        (
            r,
            branchless_flush_subnormal_f32(running_weight),
            mean_coh,
            dominant_idx,
            active_count,
            any_resonant,
            any_liminal,
        )
    }
}

/// Register-resident core for [`branchless_q7_unified_field_fold_with_init_weight_64`] that folds
/// `anchor_phases` directly in 8 `u64` SWAR registers (`trail_words`) without intermediate memory
/// stores or reloads, returning `(folded_words_64, total_weight, mean_coherence, dominant_anchor_idx, active_anchor_count)`.
#[inline(always)]
#[must_use]
pub fn branchless_q7_unified_field_fold_words_with_init_weight_64(
    trail_words: &[u64; 8],
    trail_init_weight: f32,
    anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
    anchor_base_energies: &[f32],
) -> ([u64; 8], f32, f32, u32, u32) {
    let (r, total_w, mean_coh, dom_idx, act_cnt, _any_resonant, _any_liminal) =
        branchless_q7_unified_field_fold_words_with_flags_64(
            trail_words,
            trail_init_weight,
            0,
            anchor_phases,
            anchor_base_energies,
        );
    (r, total_w, mean_coh, dom_idx, act_cnt)
}

/// Order-invariant (commutative) folding of an arbitrary slice of anchors
/// (`anchor_phases`, `anchor_base_energies`) on the **Single Unified Attractor Phase Manifold**
/// ($SO(2)^{d_{\mathrm{head}}/2}$) into `folded_phases_out` starting from `trail_phases` with
/// initial inertia weight `trail_init_weight`.
#[inline]
pub fn branchless_q7_unified_field_fold_with_init_weight_64(
    trail_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    trail_init_weight: f32,
    anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
    anchor_base_energies: &[f32],
    folded_phases_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
) -> (f32, f32, u32, u32) {
    let t = load_q7_words_64(trail_phases);
    let (r, total_pull, mean_coh, dom_idx, act_cnt) =
        branchless_q7_unified_field_fold_words_with_init_weight_64(
            &t,
            trail_init_weight,
            anchor_phases,
            anchor_base_energies,
        );
    store_q7_words_64(folded_phases_out, &r);
    (total_pull, mean_coh, dom_idx, act_cnt)
}

/// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) on the
/// **Single Unified Attractor Phase Manifold** ($SO(2)^{d_{\mathrm{head}}/2}$) into `folded_phases_out`.
#[inline(always)]
pub fn branchless_q7_unified_field_fold_64(
    trail_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
    anchor_base_energies: &[f32],
    folded_phases_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
) -> (f32, f32, u32, u32) {
    branchless_q7_unified_field_fold_with_init_weight_64(
        trail_phases,
        0.0,
        anchor_phases,
        anchor_base_energies,
        folded_phases_out,
    )
}

/// Folds an arbitrary slice of crystallized attractors (`anchor_phases`, `anchor_gravity_masses`)
/// on $SO(2)^{64}$ weighted by inverse-chord **Phase-Proximity Gravity**
/// $F_{\mathrm{grav}}(k) = \frac{M_{\mathrm{grav},k} \cdot \max(0, 1 + \mathcal{C}_{\mathrm{phase}})}{2(1 - \mathcal{C}_{\mathrm{phase}}) + \epsilon_{\mathrm{grav}}}$.
///
/// Returns `(total_gravity_pull, mean_coherence, dominant_anchor_idx, active_count)`.
#[inline]
pub fn branchless_q7_gravity_fold_64(
    query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
    anchor_gravity_masses: &[f32],
    softening_eps: f32,
    folded_phases_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
) -> (f32, f32, u32, u32) {
    let count = branchless_select_usize(
        anchor_phases.len() < anchor_gravity_masses.len(),
        anchor_phases.len(),
        anchor_gravity_masses.len(),
    );
    if count == 0 {
        *folded_phases_out = *query_phases;
        return (0.0, 0.0, 0, 0);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Guarded by `target_feature = "avx2"`. Fast-paths word-uniform anchors directly in
        // a single 64-bit GPR (`r_w0`) with zero YMM loads/extractions, and transitions to two
        // 256-bit YMM registers (`vr0, vr1`) via direct `_mm256_loadu_si256` when non-uniform.
        return unsafe {
            use core::arch::x86_64::{
                __m256i, _mm256_loadu_si256, _mm256_or_si256, _mm256_set1_epi64x, _mm256_sub_epi8,
                _mm256_testz_si256,
            };
            let q_w = load_q7_words_64(query_phases);
            let q_uni = is_word_array_uniform_64(&q_w);
            let q_w0 = q_w[0];
            let q_b0 = (q_w0 & 0xFF) as u8;
            let q_is_byte_uniform = q_uni & (q_w0 == q_w0.rotate_right(8));
            let q_nonzero = if q_uni {
                q_w0 != 0
            } else {
                (((q_w[0] | q_w[1]) | (q_w[2] | q_w[3])) | ((q_w[4] | q_w[5]) | (q_w[6] | q_w[7])))
                    != 0
            };
            let mut vq_vecs: Option<(__m256i, __m256i)> = if q_uni {
                None
            } else {
                Some(avx2_words64_to_vec64_with_uniform(&q_w, false))
            };
            let mut r_uni = q_uni;
            let mut r_w0 = q_w0;
            let mut vr_vecs: Option<(__m256i, __m256i)> = vq_vecs;
            let mut running_pull = 0.0_f32;
            let mut weighted_coh_sum = 0.0_f32;
            let mut best_pull = 0.0_f32;
            let mut dominant_idx = 0u32;
            let mut active_count = 0u32;
            let mut idx = 0usize;
            while idx < count {
                let mass =
                    branchless_flush_subnormal_f32(branchless_relu_f32(anchor_gravity_masses[idx]));
                if mass > DEFAULT_ENTROPY_EPS {
                    let check_b_uni =
                        (running_pull <= DEFAULT_ENTROPY_EPS) | (r_uni & (q_uni | !q_nonzero));
                    let mut b_w = [0u64; 8];
                    let b_uni = if check_b_uni {
                        b_w = load_q7_words_64(&anchor_phases[idx]);
                        is_word_array_uniform_64(&b_w)
                    } else {
                        false
                    };
                    if b_uni & r_uni & (q_uni | !q_nonzero) {
                        let bw0 = b_w[0];
                        let coh = if !(q_nonzero & (bw0 != 0)) {
                            0.0
                        } else if bw0 == q_w0 {
                            1.0
                        } else if q_is_byte_uniform && (bw0 == bw0.rotate_right(8)) {
                            Q7_FULL_COS_REF[q_b0.wrapping_sub(bw0 as u8) as usize]
                        } else {
                            coherence_from_word_arrays_with_both_uniform_flags_64(
                                &q_w, true, &b_w, true,
                            )
                        };
                        let grav_pull =
                            branchless_phase_proximity_gravity_f32(mass, coh, softening_eps);
                        if grav_pull > DEFAULT_ENTROPY_EPS {
                            let is_better = grav_pull > best_pull;
                            let total_w = running_pull + grav_pull;
                            if running_pull <= DEFAULT_ENTROPY_EPS {
                                r_w0 = bw0;
                            } else if r_w0 != bw0 {
                                let ratio_f32 = branchless_clamp_f32(
                                    (grav_pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                                    0.0,
                                    128.0,
                                );
                                let s = (ratio_f32 as u8) as u64;
                                if s >= 128 {
                                    r_w0 = bw0;
                                } else if s > 0 {
                                    let bias_restore =
                                        (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                                    r_w0 = swar_slerp_word_unchecked_x8(r_w0, bw0, s, bias_restore);
                                }
                            }
                            running_pull = total_w;
                            weighted_coh_sum += grav_pull * coh;
                            best_pull = branchless_select_f32(is_better, grav_pull, best_pull);
                            dominant_idx =
                                branchless_select_u32(is_better, idx as u32, dominant_idx);
                            active_count = active_count.wrapping_add(1);
                        }
                    } else {
                        let (vq0, vq1) = match vq_vecs {
                            Some(v) => v,
                            None => {
                                let z = _mm256_set1_epi64x(q_w0 as i64);
                                let v = (z, z);
                                vq_vecs = Some(v);
                                v
                            }
                        };
                        let (mut vr0, mut vr1) = match vr_vecs {
                            Some(v) if !r_uni => v,
                            _ => {
                                let rz = _mm256_set1_epi64x(r_w0 as i64);
                                (rz, rz)
                            }
                        };
                        r_uni = false;
                        let b_ptr = anchor_phases[idx].as_ptr().cast::<__m256i>();
                        let vb0 = _mm256_loadu_si256(b_ptr);
                        let vb1 = _mm256_loadu_si256(b_ptr.add(1));
                        let b_or = _mm256_or_si256(vb0, vb1);
                        let coh = if q_nonzero && _mm256_testz_si256(b_or, b_or) == 0 {
                            avx2_coherence_from_diff_vec64(
                                _mm256_sub_epi8(vq0, vb0),
                                _mm256_sub_epi8(vq1, vb1),
                            )
                        } else {
                            0.0
                        };
                        let grav_pull =
                            branchless_phase_proximity_gravity_f32(mass, coh, softening_eps);
                        if grav_pull > DEFAULT_ENTROPY_EPS {
                            let is_better = grav_pull > best_pull;
                            let total_w = running_pull + grav_pull;
                            if running_pull <= DEFAULT_ENTROPY_EPS {
                                if b_uni {
                                    r_w0 = b_w[0];
                                    r_uni = true;
                                } else {
                                    vr_vecs = Some((vb0, vb1));
                                }
                            } else {
                                let vd0 = _mm256_sub_epi8(vb0, vr0);
                                let vd1 = _mm256_sub_epi8(vb1, vr1);
                                let diff_or = _mm256_or_si256(vd0, vd1);
                                if _mm256_testz_si256(diff_or, diff_or) == 0 {
                                    let ratio_f32 = branchless_clamp_f32(
                                        (grav_pull * fast_inv_pos_normal_f32(total_w)) * 128.0
                                            + 0.5,
                                        0.0,
                                        128.0,
                                    );
                                    let s = (ratio_f32 as u8) as u64;
                                    if s >= 128 {
                                        vr0 = vb0;
                                        vr1 = vb1;
                                    } else if s > 0 {
                                        (vr0, vr1) =
                                            avx2_slerp_from_diff_vec64(vr0, vr1, vd0, vd1, s);
                                    }
                                }
                                vr_vecs = Some((vr0, vr1));
                            }
                            running_pull = total_w;
                            weighted_coh_sum += grav_pull * coh;
                            best_pull = branchless_select_f32(is_better, grav_pull, best_pull);
                            dominant_idx =
                                branchless_select_u32(is_better, idx as u32, dominant_idx);
                            active_count = active_count.wrapping_add(1);
                        } else {
                            vr_vecs = Some((vr0, vr1));
                        }
                    }
                }
                idx += 1;
            }
            let out_w = if r_uni {
                [r_w0; 8]
            } else if let Some((vr0, vr1)) = vr_vecs {
                avx2_vec64_to_words64(vr0, vr1)
            } else {
                [r_w0; 8]
            };
            store_q7_words_64(folded_phases_out, &out_w);
            let has_pull = (active_count > 0) & (running_pull > DEFAULT_ENTROPY_EPS);
            let inv_pull =
                fast_inv_pos_normal_f32(branchless_select_f32(has_pull, running_pull, 1.0));
            let mean_coh = branchless_select_f32(
                has_pull,
                branchless_clamp_f32(weighted_coh_sum * inv_pull, -1.0, 1.0),
                0.0,
            );
            (
                branchless_flush_subnormal_f32(running_pull),
                mean_coh,
                dominant_idx,
                active_count,
            )
        };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let q = load_q7_words_64(query_phases);
        let q_is_uniform = is_word_array_uniform_64(&q);
        let q_nonzero = if q_is_uniform {
            q[0] != 0
        } else {
            (((q[0] | q[1]) | (q[2] | q[3])) | ((q[4] | q[5]) | (q[6] | q[7]))) != 0
        };
        let mut r = q;
        let mut r_is_uniform = q_is_uniform;
        let mut running_pull = 0.0_f32;
        let mut weighted_coh_sum = 0.0_f32;
        let mut best_pull = 0.0_f32;
        let mut dominant_idx = 0u32;
        let mut active_count = 0u32;

        let mut idx = 0usize;
        while idx < count {
            let mass =
                branchless_flush_subnormal_f32(branchless_relu_f32(anchor_gravity_masses[idx]));
            if mass > DEFAULT_ENTROPY_EPS {
                let b = load_q7_words_64(&anchor_phases[idx]);
                let need_b_uni =
                    (q_nonzero & q_is_uniform) | (running_pull <= DEFAULT_ENTROPY_EPS) | r_is_uniform;
                let b_is_uniform = if need_b_uni {
                    is_word_array_uniform_64(&b)
                } else {
                    false
                };
                let b_nonzero = if b_is_uniform {
                    b[0] != 0
                } else {
                    (((b[0] | b[1]) | (b[2] | b[3])) | ((b[4] | b[5]) | (b[6] | b[7]))) != 0
                };
                let coh = if q_nonzero & b_nonzero {
                    coherence_from_word_arrays_with_both_uniform_flags_64(
                        &q,
                        q_is_uniform,
                        &b,
                        b_is_uniform,
                    )
                } else {
                    0.0
                };
                let grav_pull = branchless_phase_proximity_gravity_f32(mass, coh, softening_eps);
                if grav_pull > DEFAULT_ENTROPY_EPS {
                    let is_better = grav_pull > best_pull;
                    let total_w = running_pull + grav_pull;
                    if running_pull <= DEFAULT_ENTROPY_EPS {
                        r = b;
                        r_is_uniform = b_is_uniform;
                    } else {
                        let ratio_f32 = branchless_clamp_f32(
                            (grav_pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                            0.0,
                            128.0,
                        );
                        let s = (ratio_f32 as u8) as u64;
                        let bias_restore =
                            (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                        let (next_r, next_uni) = swar_slerp_words_with_uniform_flag_64(
                            &r,
                            r_is_uniform,
                            &b,
                            b_is_uniform,
                            s,
                            bias_restore,
                        );
                        r = next_r;
                        r_is_uniform = next_uni;
                    }

                    running_pull = total_w;
                    weighted_coh_sum += grav_pull * coh;
                    best_pull = branchless_select_f32(is_better, grav_pull, best_pull);
                    dominant_idx = branchless_select_u32(is_better, idx as u32, dominant_idx);
                    active_count = active_count.wrapping_add(1);
                }
            }
            idx += 1;
        }

        store_q7_words_64(folded_phases_out, &r);
        let has_pull = running_pull > DEFAULT_ENTROPY_EPS;
        let inv_pull = fast_inv_pos_normal_f32(branchless_select_f32(has_pull, running_pull, 1.0));
        let mean_coh = branchless_select_f32(
            has_pull,
            branchless_clamp_f32(weighted_coh_sum * inv_pull, -1.0, 1.0),
            0.0,
        );
        (
            branchless_flush_subnormal_f32(running_pull),
            mean_coh,
            dominant_idx,
            active_count,
        )
    }
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

/// Evaluates the conditional transition log-probability $\ln P(M_t \mid M_{t-1}) = \ln(0.5(1 + c))$
/// using the 1-lookup division-free (`0 fdiv`) identity $\ln(1 + c) - \ln 2$
/// (`fast_ln_1p_bounded_f32`, `0` divisions) whenever $c \ge 0.0$, falling back to
/// [`branchless_ln_f32`] only when $c < 0.0$.
#[inline(always)]
#[must_use]
pub const fn branchless_fast_transition_log_prob(coh: f32) -> f32 {
    if coh >= 0.999_999 {
        0.0_f32
    } else if coh >= 0.0 {
        branchless_flush_subnormal_f32(fast_ln_1p_bounded_f32(coh) - LN_2)
    } else {
        let cond_prob = branchless_clamp_f32(0.5 * (1.0 + coh), DEFAULT_ENTROPY_EPS, 1.0);
        branchless_flush_subnormal_f32(branchless_ln_f32(cond_prob))
    }
}

/// Evaluates the conditional transition coherence $\mathcal{C}_{\mathrm{phase}}(M_{t-1}, M_t) \in [-1.0, 1.0]$
/// and log-transition probability $\ln P(M_t \mid M_{t-1}) \le 0$ directly on 8 register-resident `u64` SWAR words.
#[inline(always)]
#[must_use]
pub fn branchless_q7_transition_log_prob_words_64(
    prev_words: &[u64; 8],
    curr_words: &[u64; 8],
) -> (f32, f32) {
    let coh = coherence_from_word_arrays_64(prev_words, curr_words);
    (coh, branchless_fast_transition_log_prob(coh))
}

/// Evaluates the conditional transition coherence $\mathcal{C}_{\mathrm{phase}}(M_{t-1}, M_t) \in [-1.0, 1.0]$
/// and log-transition probability $\ln P(M_t \mid M_{t-1}) = \ln(\max((1 + \mathcal{C}_{\mathrm{phase}})/2, \epsilon_{\mathrm{log}})) \le 0$.
#[inline(always)]
pub fn branchless_q7_transition_log_prob_64(
    prev_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    curr_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
) -> (f32, f32) {
    branchless_q7_transition_log_prob_words_64(
        &load_q7_words_64(prev_phases),
        &load_q7_words_64(curr_phases),
    )
}

/// Evaluates the **Attention $\times$ DPSR Phase-Interference Logit Delta**
/// $\Delta z_{t,j} = \frac{\gamma}{\sqrt{d_{\mathrm{head}}}} \mathcal{C}_{\mathrm{phase}}(\boldsymbol{\theta}_t + \boldsymbol{\Delta\theta}, \boldsymbol{\theta}_j)$.
#[inline(always)]
pub fn branchless_q7_attention_score_delta_64(
    query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    key_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    head_dim: u32,
    interference_gain: f32,
) -> (f32, f32) {
    let coh = branchless_q7_coherence_64(query_phases, key_phases);
    let safe_dim = branchless_select_u32(head_dim > 0, head_dim, 128);
    let rsqrt_d = if safe_dim <= 256 {
        POPCOUNT_RSQRT_LUT_257_REF[safe_dim as usize]
    } else {
        branchless_rsqrt_f32(safe_dim as f32)
    };
    let clean_gain = branchless_flush_subnormal_f32(branchless_relu_f32(interference_gain));
    (coh, branchless_flush_subnormal_f32(clean_gain * rsqrt_d * coh))
}

/// Evaluates the **Bayes-Attention Stigmergic Logit Delta** combining conditional phase-coherence
/// interference and stigmergic pointer-chain priors
/// in `const fn` constant time (`0` branches, `0 fdiv`), returning `(phase_coherence, bayesian_logit_delta)`.
#[inline(always)]
pub fn branchless_q7_bayesian_attention_score_delta_64(
    query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    key_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    key_stigmergic_density: f32,
    is_pointer_linked: bool,
    head_dim: u32,
    interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
) -> (f32, f32) {
    let (coh, attn_delta) = branchless_q7_attention_score_delta_64(
        query_phases,
        key_phases,
        head_dim,
        interference_gain,
    );
    let clean_rho = branchless_clamp_f32(
        branchless_flush_subnormal_f32(branchless_relu_f32(key_stigmergic_density)),
        0.0,
        MAX_BAYESIAN_STIGMERGIC_DENSITY,
    );
    let prior_delta = branchless_flush_subnormal_f32(
        branchless_relu_f32(stigmergic_prior_gain) * fast_ln_1p_bounded_f32(clean_rho),
    );
    let link_delta = branchless_select_f32(
        is_pointer_linked,
        branchless_flush_subnormal_f32(branchless_relu_f32(pointer_link_bonus)),
        0.0,
    );
    (
        coh,
        branchless_flush_subnormal_f32(attn_delta + prior_delta + link_delta),
    )
}

/// Computes the bounded **Phase-Interference Inverse-Temperature Scale**
/// $\beta_{\mathrm{eff}} = \mathrm{clamp}(1.0 + \gamma \cdot \mathcal{C}_{\mathrm{phase}}, 0.25, 4.0)$.
#[inline(always)]
pub const fn branchless_phase_coherence_inv_temp_scale(
    coherence: f32,
    interference_gain: f32,
) -> f32 {
    let clean_coh = branchless_clamp_f32(
        branchless_select_f32(branchless_is_finite_f32(coherence), coherence, 0.0),
        -1.0,
        1.0,
    );
    let clean_gain = branchless_flush_subnormal_f32(branchless_relu_f32(interference_gain));
    branchless_clamp_f32(
        1.0 + clean_gain * clean_coh,
        MIN_PHASE_INTERFERENCE_INV_TEMP,
        MAX_PHASE_INTERFERENCE_INV_TEMP,
    )
}

/// Directly modulates pre-Softmax candidate memory/key Attention scores (`attention_scores[..M]`)
/// in-place by each memory $j$'s phase interference, preloading `query_phases` into 8 `u64` registers once.
#[inline]
pub fn modulate_attention_scores_by_phase_bank_64(
    attention_scores: &mut [f32],
    query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    key_anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
    head_dim: u32,
    interference_gain: f32,
) -> f32 {
    let count = attention_scores.len().min(key_anchor_phases.len());
    if count == 0 {
        return 0.0;
    }
    let q = load_q7_words_64(query_phases);
    let q_is_uniform = is_word_array_uniform_64(&q);
    let safe_dim = branchless_select_u32(head_dim > 0, head_dim, 128);
    let rsqrt_d = if safe_dim <= 256 {
        POPCOUNT_RSQRT_LUT_257_REF[safe_dim as usize]
    } else {
        branchless_rsqrt_f32(safe_dim as f32)
    };
    let eff_scale =
        branchless_flush_subnormal_f32(branchless_relu_f32(interference_gain) * rsqrt_d);

    let q_w0 = q[0];
    let q_b0 = (q_w0 & 0xFF) as u8;
    let q_is_byte_uniform = q_is_uniform & (q_w0 == q_w0.rotate_right(8));
    let mut coh_sum = 0.0_f32;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    let mut vq_vecs: Option<(
        core::arch::x86_64::__m256i,
        core::arch::x86_64::__m256i,
    )> = if q_is_uniform {
        None
    } else {
        Some(unsafe { avx2_words64_to_vec64_with_uniform(&q, false) })
    };
    for (score_slot, key_p) in attention_scores[..count]
        .iter_mut()
        .zip(key_anchor_phases[..count].iter())
    {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let coh = {
            if q_is_uniform {
                let kw = load_q7_words_64(key_p);
                if is_word_array_uniform_64(&kw) {
                    let kw0 = kw[0];
                    if kw0 == q_w0 {
                        1.0
                    } else if q_is_byte_uniform && (kw0 == kw0.rotate_right(8)) {
                        Q7_FULL_COS_REF[q_b0.wrapping_sub(kw0 as u8) as usize]
                    } else {
                        coherence_from_word_arrays_with_both_uniform_flags_64(&q, true, &kw, true)
                    }
                } else {
                    unsafe {
                        use core::arch::x86_64::{
                            __m256i, _mm256_loadu_si256, _mm256_set1_epi64x, _mm256_sub_epi8,
                        };
                        let (vq0, vq1) = match vq_vecs {
                            Some(v) => v,
                            None => {
                                let z = _mm256_set1_epi64x(q_w0 as i64);
                                let v = (z, z);
                                vq_vecs = Some(v);
                                v
                            }
                        };
                        let kp = key_p.as_ptr().cast::<__m256i>();
                        let vk0 = _mm256_loadu_si256(kp);
                        let vk1 = _mm256_loadu_si256(kp.add(1));
                        avx2_coherence_from_diff_vec64(
                            _mm256_sub_epi8(vq0, vk0),
                            _mm256_sub_epi8(vq1, vk1),
                        )
                    }
                }
            } else {
                unsafe {
                    use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_sub_epi8};
                    let (vq0, vq1) =
                        vq_vecs.unwrap_or_else(|| avx2_words64_to_vec64_with_uniform(&q, false));
                    let kp = key_p.as_ptr().cast::<__m256i>();
                    let vk0 = _mm256_loadu_si256(kp);
                    let vk1 = _mm256_loadu_si256(kp.add(1));
                    avx2_coherence_from_diff_vec64(
                        _mm256_sub_epi8(vq0, vk0),
                        _mm256_sub_epi8(vq1, vk1),
                    )
                }
            }
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        let coh = {
            let kw = load_q7_words_64(key_p);
            if q_is_uniform && is_word_array_uniform_64(&kw) {
                let kw0 = kw[0];
                if kw0 == q_w0 {
                    1.0
                } else if q_is_byte_uniform && (kw0 == kw0.rotate_right(8)) {
                    Q7_FULL_COS_REF[q_b0.wrapping_sub(kw0 as u8) as usize]
                } else {
                    coherence_from_word_arrays_with_both_uniform_flags_64(&q, true, &kw, true)
                }
            } else {
                coherence_from_word_arrays_with_both_uniform_flags_64(&q, q_is_uniform, &kw, false)
            }
        };
        *score_slot = branchless_flush_subnormal_f32(*score_slot + eff_scale * coh);
        coh_sum += coh;
    }
    branchless_clamp_f32(coh_sum * fast_inv_len_f32(count), -1.0, 1.0)
}

/// Directly modulates pre-Softmax candidate memory/key Attention scores (`attention_scores[..M]`)
/// in-place by the full **Bayes-Attention Stigmergic Transition Score** (`0` branches, `0 fdiv`),
/// preloading `query_phases` into registers and using `fast_ln_1p_bounded_f32`.
#[inline]
pub fn modulate_attention_scores_by_bayesian_trails_64(
    attention_scores: &mut [f32],
    query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    key_anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
    key_stigmergic_densities: &[f32],
    linked_trail_id: Option<u32>,
    head_dim: u32,
    interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
) -> (f32, u32) {
    let count = attention_scores
        .len()
        .min(key_anchor_phases.len())
        .min(key_stigmergic_densities.len());
    if count == 0 {
        return (0.0, 0);
    }

    let q = load_q7_words_64(query_phases);
    let q_is_uniform = is_word_array_uniform_64(&q);
    let q_w0 = q[0];
    let q_b0 = (q_w0 & 0xFF) as u8;
    let q_is_byte_uniform = q_is_uniform & (q_w0 == q_w0.rotate_right(8));
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    let mut vq_vecs: Option<(
        core::arch::x86_64::__m256i,
        core::arch::x86_64::__m256i,
    )> = if q_is_uniform {
        None
    } else {
        Some(unsafe { avx2_words64_to_vec64_with_uniform(&q, false) })
    };
    let safe_dim = branchless_select_u32(head_dim > 0, head_dim, 128);
    let rsqrt_d = if safe_dim <= 256 {
        POPCOUNT_RSQRT_LUT_257_REF[safe_dim as usize]
    } else {
        branchless_rsqrt_f32(safe_dim as f32)
    };
    let eff_interf_scale =
        branchless_flush_subnormal_f32(branchless_relu_f32(interference_gain) * rsqrt_d);
    let clean_prior_gain = branchless_relu_f32(stigmergic_prior_gain);
    let clean_link_bonus =
        branchless_flush_subnormal_f32(branchless_relu_f32(pointer_link_bonus));

    let has_link = linked_trail_id.is_some();
    let link_target = linked_trail_id.unwrap_or(0);
    let mut coh_sum = 0.0_f32;
    let mut best_score = -1.0e30_f32;
    let mut best_idx = 0u32;

    for idx in 0..count {
        let is_linked = has_link & ((idx as u32) == link_target);
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let coh = {
            if q_is_uniform {
                let kw = load_q7_words_64(&key_anchor_phases[idx]);
                if is_word_array_uniform_64(&kw) {
                    let kw0 = kw[0];
                    if kw0 == q_w0 {
                        1.0
                    } else if q_is_byte_uniform && (kw0 == kw0.rotate_right(8)) {
                        Q7_FULL_COS_REF[q_b0.wrapping_sub(kw0 as u8) as usize]
                    } else {
                        coherence_from_word_arrays_with_both_uniform_flags_64(&q, true, &kw, true)
                    }
                } else {
                    unsafe {
                        use core::arch::x86_64::{
                            __m256i, _mm256_loadu_si256, _mm256_set1_epi64x, _mm256_sub_epi8,
                        };
                        let (vq0, vq1) = match vq_vecs {
                            Some(v) => v,
                            None => {
                                let z = _mm256_set1_epi64x(q_w0 as i64);
                                let v = (z, z);
                                vq_vecs = Some(v);
                                v
                            }
                        };
                        let kp = key_anchor_phases[idx].as_ptr().cast::<__m256i>();
                        let vk0 = _mm256_loadu_si256(kp);
                        let vk1 = _mm256_loadu_si256(kp.add(1));
                        avx2_coherence_from_diff_vec64(
                            _mm256_sub_epi8(vq0, vk0),
                            _mm256_sub_epi8(vq1, vk1),
                        )
                    }
                }
            } else {
                unsafe {
                    use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_sub_epi8};
                    let (vq0, vq1) =
                        vq_vecs.unwrap_or_else(|| avx2_words64_to_vec64_with_uniform(&q, false));
                    let kp = key_anchor_phases[idx].as_ptr().cast::<__m256i>();
                    let vk0 = _mm256_loadu_si256(kp);
                    let vk1 = _mm256_loadu_si256(kp.add(1));
                    avx2_coherence_from_diff_vec64(
                        _mm256_sub_epi8(vq0, vk0),
                        _mm256_sub_epi8(vq1, vk1),
                    )
                }
            }
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        let coh = {
            let kw = load_q7_words_64(&key_anchor_phases[idx]);
            if q_is_uniform && is_word_array_uniform_64(&kw) {
                let kw0 = kw[0];
                if kw0 == q_w0 {
                    1.0
                } else if q_is_byte_uniform && (kw0 == kw0.rotate_right(8)) {
                    Q7_FULL_COS_REF[q_b0.wrapping_sub(kw0 as u8) as usize]
                } else {
                    coherence_from_word_arrays_with_both_uniform_flags_64(&q, true, &kw, true)
                }
            } else {
                coherence_from_word_arrays_with_both_uniform_flags_64(&q, q_is_uniform, &kw, false)
            }
        };
        let attn_delta = branchless_flush_subnormal_f32(eff_interf_scale * coh);
        let clean_rho = branchless_clamp_f32(
            branchless_flush_subnormal_f32(branchless_relu_f32(key_stigmergic_densities[idx])),
            0.0,
            MAX_BAYESIAN_STIGMERGIC_DENSITY,
        );
        let prior_delta =
            branchless_flush_subnormal_f32(clean_prior_gain * fast_ln_1p_bounded_f32(clean_rho));
        let link_delta = branchless_select_f32(is_linked, clean_link_bonus, 0.0);
        let delta_z = branchless_flush_subnormal_f32(attn_delta + prior_delta + link_delta);

        let new_score = branchless_flush_subnormal_f32(attention_scores[idx] + delta_z);
        attention_scores[idx] = new_score;
        coh_sum += coh;
        let better = (idx == 0) | (new_score > best_score);
        best_score = branchless_select_f32(better, new_score, best_score);
        best_idx = branchless_select_u32(better, idx as u32, best_idx);
    }

    (
        branchless_clamp_f32(coh_sum * fast_inv_len_f32(count), -1.0, 1.0),
        best_idx,
    )
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
unsafe fn avx2_mean_sincos_from_vec64(
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
            0, 33, 65, 126, 0, -126, -65, -33, 0, -33, -65, -126, 0, 126, 65, 33,
            0, 33, 65, 126, 0, -126, -65, -33, 0, -33, -65, -126, 0, 126, 65, 33,
        );
        let lut_cos0_hi = _mm256_setr_epi8(
            -128, 123, 109, 88, 64, 39, 18, 4, 0, 4, 18, 39, 64, 88, 109, 123,
            -128, 123, 109, 88, 64, 39, 18, 4, 0, 4, 18, 39, 64, 88, 109, 123,
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
            0, 126, 65, 33, 0, 33, 65, 126, 0, -126, -65, -33, 0, -33, -65, -126,
            0, 126, 65, 33, 0, 33, 65, 126, 0, -126, -65, -33, 0, -33, -65, -126,
        );
        let lut_sin0_hi = _mm256_setr_epi8(
            64, 88, 109, 123, -128, 123, 109, 88, 64, 39, 18, 4, 0, 4, 18, 39,
            64, 88, 109, 123, -128, 123, 109, 88, 64, 39, 18, 4, 0, 4, 18, 39,
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
            127, 117, 90, 49, 0, -49, -90, -117, -127, -117, -90, -49, 0, 49, 90, 117,
            127, 117, 90, 49, 0, -49, -90, -117, -127, -117, -90, -49, 0, 49, 90, 117,
        );
        let lut_nsin_h = _mm256_setr_epi8(
            0, -49, -90, -117, -127, -117, -90, -49, 0, 49, 90, 117, 127, 117, 90, 49,
            0, -49, -90, -117, -127, -117, -90, -49, 0, 49, 90, 117, 127, 117, 90, 49,
        );
        let lut_sin_l = _mm256_setr_epi8(
            0, 6, 13, 19, 25, 32, 38, 44, 50, 57, 63, 69, 75, 81, 87, 93,
            0, 6, 13, 19, 25, 32, 38, 44, 50, 57, 63, 69, 75, 81, 87, 93,
        );
        let lut_omc_l = _mm256_setr_epi8(
            0, 0, 0, 1, 1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 15, 17,
            0, 0, 0, 1, 1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 15, 17,
        );

        let l0 = _mm256_and_si256(v0, _mm256_set1_epi8(0x0F));
        let ch0 = _mm256_shuffle_epi8(lut_cos_h, h0);
        let nsh0 = _mm256_shuffle_epi8(lut_nsin_h, h0);
        let sl0 = _mm256_shuffle_epi8(lut_sin_l, l0);
        let ol0 = _mm256_shuffle_epi8(lut_omc_l, l0);
        let c_corr32_0 = _mm256_madd_epi16(
            _mm256_sub_epi16(_mm256_maddubs_epi16(sl0, nsh0), _mm256_maddubs_epi16(ol0, ch0)),
            _mm256_set1_epi16(1),
        );
        let s_corr32_0 = _mm256_madd_epi16(
            _mm256_add_epi16(_mm256_maddubs_epi16(sl0, ch0), _mm256_maddubs_epi16(ol0, nsh0)),
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
                _mm256_sub_epi16(_mm256_maddubs_epi16(sl1, nsh1), _mm256_maddubs_epi16(ol1, ch1)),
                _mm256_set1_epi16(1),
            ),
        );
        let s_corr32 = _mm256_add_epi32(
            s_corr32_0,
            _mm256_madd_epi16(
                _mm256_add_epi16(_mm256_maddubs_epi16(sl1, ch1), _mm256_maddubs_epi16(ol1, nsh1)),
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
pub(super) fn mean_sincos_from_words_with_uniform_flag_64(w: &[u64; 8], w_is_uniform: bool) -> (f32, f32) {
    if w_is_uniform {
        let w0 = w[0];
        if w0 == w0.rotate_right(8) {
            return branchless_sincos_q7((w0 & 0xFF) as u8 as gcso_q7_t);
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
        return unsafe { avx2_mean_sincos_from_words_64(w) };
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
) -> (f32, gcso_q7_t, f32) {
    if w_is_uniform {
        let w0 = w[0];
        if w0 == w0.rotate_right(8) {
            let q7 = (w0 & 0xFF) as u8 as gcso_q7_t;
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
pub fn circular_resultant_from_words_64(w: &[u64; 8]) -> (f32, gcso_q7_t, f32) {
    circular_resultant_from_words_with_uniform_flag_64(w, is_word_array_uniform_64(w))
}

/// Evaluates the 64-head circular statistics summary `(r_bar, mean_phase_q7, kappa_c)` (Section 4.2)
/// using direct YMM vector loads on AVX2 (`0` Port-5 GPR-to-YMM inserts) or 8-word SWAR loads on scalar targets.
#[inline(always)]
pub fn branchless_q7_circular_resultant_64(
    phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
) -> (f32, gcso_q7_t, f32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: `phases` is 64 bytes and `target_feature = "avx2"` is enabled.
        return unsafe {
            use core::arch::x86_64::{
                __m256i, _mm256_broadcastb_epi8, _mm256_castsi256_si128, _mm256_loadu_si256,
                _mm256_or_si256, _mm256_testz_si256, _mm256_xor_si256,
            };
            let pa = phases.as_ptr().cast::<__m256i>();
            let va0 = _mm256_loadu_si256(pa);
            let va1 = _mm256_loadu_si256(pa.add(1));
            let bcast_a = _mm256_broadcastb_epi8(_mm256_castsi256_si128(va0));
            let a_diff = _mm256_or_si256(
                _mm256_xor_si256(va0, bcast_a),
                _mm256_xor_si256(va1, bcast_a),
            );
            if _mm256_testz_si256(a_diff, a_diff) != 0 {
                let q7 = phases[0];
                return (
                    1.0,
                    branchless_select_i8(q7 == -128, 127, q7),
                    VON_MISES_KAPPA_AT_MAX_R,
                );
            }
            let (mean_sin, mean_cos) = avx2_mean_sincos_from_vec64(va0, va1);
            let r_bar = fast_sqrt_unit_f32(mean_sin * mean_sin + mean_cos * mean_cos);
            let mean_q7 = branchless_atan2_to_q7_fast(mean_sin, mean_cos);
            (r_bar, mean_q7, branchless_von_mises_kappa(r_bar))
        };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        circular_resultant_from_words_64(&load_q7_words_64(phases))
    }
}

/// Evaluates both 64-head phase coherence $\mathcal{C}_{\mathrm{phase}}(a, b) \in [-1.0, 1.0]$ and
/// circular resultant statistics $(\bar{R}_a, \bar{\theta}_{a,\mathrm{Q7}}, \kappa_a)$ in a single
/// YMM-resident pass on AVX2 (`va0, va1` loaded once via `_mm256_loadu_si256` and shared across both
/// `avx2_coherence_from_diff_vec64` and `avx2_mean_sincos_from_vec64` with `0` Port-5 `vpinsrq` instructions).
#[inline(always)]
pub fn branchless_q7_coherence_and_resultant_64(
    phases_a: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    phases_b: &[gcso_q7_t; MAX_ATTENTION_HEADS],
) -> (f32, f32, gcso_q7_t, f32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: `phases_a` and `phases_b` are 64 bytes each and `target_feature = "avx2"` is enabled.
        return unsafe {
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
        };
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

/// Executes an **Online Toroidal $k$-Means M-Step Centroid Update** in 8 register-resident `u64`
/// SWAR words (`centroid_words`, `sample_words`) with an optional precomputed E-step phase
/// coherence `precomputed_coh_opt` and pre-evaluated word-uniformity flags `(c_uni, sample_uni)`,
/// in strictly constant time (`0` data-dependent branches on phase stationarity, normalized
/// directional cosine weighting $w_{\mathrm{coh}} = \operatorname{clamp}(0.5(1 + \mathcal{C}_{\mathrm{phase}}), 0.25, 1.0)$).
///
/// Returns `(updated_centroid_words, next_base_energy, resultant_r_bar, von_mises_kappa, words_changed)`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn branchless_toroidal_centroid_step_words_with_coh_uni_and_changed_64(
    centroid_words: &[u64; 8],
    prev_sample_count: u32,
    prev_base_energy: f32,
    prev_r_bar: f32,
    prev_kappa: f32,
    sample_words: &[u64; 8],
    precomputed_coh_opt: Option<f32>,
    sample_energy: f32,
    c_uni: bool,
    sample_uni: bool,
) -> ([u64; 8], f32, f32, f32, bool) {
    let clean_e = branchless_flush_subnormal_f32(branchless_relu_f32(sample_energy));
    if clean_e <= DEFAULT_ENTROPY_EPS {
        return (*centroid_words, prev_base_energy, prev_r_bar, prev_kappa, false);
    }
    let both_uni = c_uni & sample_uni;
    let same_words = (sample_words[0] == centroid_words[0])
        && (both_uni
            || ((((sample_words[1] ^ centroid_words[1]) | (sample_words[2] ^ centroid_words[2]))
                | ((sample_words[3] ^ centroid_words[3]) | (sample_words[4] ^ centroid_words[4]))
                | ((sample_words[5] ^ centroid_words[5]) | (sample_words[6] ^ centroid_words[6]))
                | (sample_words[7] ^ centroid_words[7]))
                == 0));
    let prev_e = branchless_flush_subnormal_f32(branchless_relu_f32(prev_base_energy));
    if prev_sample_count == 0 || prev_e <= DEFAULT_ENTROPY_EPS {
        let rb = circular_r_bar_from_words_with_uniform_flag_64(sample_words, sample_uni);
        return (
            *sample_words,
            clean_e,
            rb,
            branchless_von_mises_kappa(rb),
            !same_words,
        );
    }
    if same_words {
        let coh = precomputed_coh_opt.unwrap_or(1.0);
        let eff_sample_e = clean_e * branchless_clamp_f32(0.5 * (1.0 + coh), 0.25, 1.0);
        let next_e = prev_e + 0.5 * eff_sample_e;
        if prev_r_bar > 0.0 && prev_kappa > 0.0 {
            return (*centroid_words, next_e, prev_r_bar, prev_kappa, false);
        }
        let rb = circular_r_bar_from_words_with_uniform_flag_64(centroid_words, c_uni);
        return (*centroid_words, next_e, rb, branchless_von_mises_kappa(rb), false);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    let (d0, d1, uniform_diff_words) = if both_uni {
        (
            swar_q7_circular_diff_x8(sample_words[0], centroid_words[0]),
            0u64,
            true,
        )
    } else {
        (0u64, 0u64, false)
    };
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    let (d0, d1, uniform_diff_words) = {
        let d0_val = swar_q7_circular_diff_x8(sample_words[0], centroid_words[0]);
        if both_uni {
            (d0_val, d0_val, true)
        } else {
            let d1_val = swar_q7_circular_diff_x8(sample_words[1], centroid_words[1]);
            let is_uni = (d0_val == d1_val)
                && (swar_q7_circular_diff_x8(sample_words[2], centroid_words[2]) == d0_val)
                && (swar_q7_circular_diff_x8(sample_words[3], centroid_words[3]) == d0_val)
                && (swar_q7_circular_diff_x8(sample_words[4], centroid_words[4]) == d0_val)
                && (swar_q7_circular_diff_x8(sample_words[5], centroid_words[5]) == d0_val)
                && (swar_q7_circular_diff_x8(sample_words[6], centroid_words[6]) == d0_val)
                && (swar_q7_circular_diff_x8(sample_words[7], centroid_words[7]) == d0_val);
            (d0_val, d1_val, is_uni)
        }
    };

    if uniform_diff_words {
        let coh = match precomputed_coh_opt {
            Some(c) => c,
            None => coherence_from_uniform_diff_word_x8(d0),
        };
        let eff_sample_e = clean_e * branchless_clamp_f32(0.5 * (1.0 + coh), 0.25, 1.0);
        let next_e = prev_e + 0.5 * eff_sample_e;
        if d0 == 0 {
            if prev_r_bar > 0.0 && prev_kappa > 0.0 {
                return (*centroid_words, next_e, prev_r_bar, prev_kappa, false);
            }
            let rb = circular_r_bar_from_words_with_uniform_flag_64(centroid_words, c_uni);
            return (*centroid_words, next_e, rb, branchless_von_mises_kappa(rb), false);
        }
        let tot_e = prev_e + eff_sample_e;
        let safe_tot = branchless_select_f32(tot_e > DEFAULT_ENTROPY_EPS, tot_e, 1.0);
        let ratio = branchless_clamp_f32(
            eff_sample_e * fast_inv_pos_normal_f32(safe_tot),
            0.05,
            0.50,
        );
        let raw_s = (ratio * 128.0 + 0.5) as u8;
        let s = branchless_select_u8(raw_s > 128, 128, raw_s) as u64;
        let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let step0 = swar_q7_scale_le128_with_bias_x8(d0, s, bias_restore);
        if step0 == 0 {
            if prev_r_bar > 0.0 && prev_kappa > 0.0 {
                return (*centroid_words, next_e, prev_r_bar, prev_kappa, false);
            }
            let rb = circular_r_bar_from_words_with_uniform_flag_64(centroid_words, c_uni);
            return (*centroid_words, next_e, rb, branchless_von_mises_kappa(rb), false);
        }
        let next_w = if c_uni {
            [swar_q7_circular_add_x8(centroid_words[0], step0); 8]
        } else {
            [
                swar_q7_circular_add_x8(centroid_words[0], step0),
                swar_q7_circular_add_x8(centroid_words[1], step0),
                swar_q7_circular_add_x8(centroid_words[2], step0),
                swar_q7_circular_add_x8(centroid_words[3], step0),
                swar_q7_circular_add_x8(centroid_words[4], step0),
                swar_q7_circular_add_x8(centroid_words[5], step0),
                swar_q7_circular_add_x8(centroid_words[6], step0),
                swar_q7_circular_add_x8(centroid_words[7], step0),
            ]
        };
        if step0 == step0.rotate_right(8)
            && prev_r_bar > 0.0
            && prev_kappa > 0.0
        {
            return (next_w, next_e, prev_r_bar, prev_kappa, true);
        }
        let rb = circular_r_bar_from_words_with_uniform_flag_64(&next_w, c_uni);
        return (next_w, next_e, rb, branchless_von_mises_kappa(rb), true);
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let _ = d1;
        // SAFETY: Guarded by `target_feature = "avx2"`. Keeps `va0..vb1`, `vd0..vd1`, `sc0..sc1`,
        // and `out0..out1` 100% resident in YMM registers across coherence, SLERP, stationarity,
        // rigid SO(2) invariance, and circular resultant evaluation (`0` GPR-to-YMM or YMM-to-GPR STLF stalls).
        return unsafe {
            use core::arch::x86_64::{
                _mm256_add_epi8, _mm256_and_si256, _mm256_broadcastb_epi8,
                _mm256_castsi256_si128, _mm256_mulhrs_epi16, _mm256_or_si256,
                _mm256_permute4x64_epi64, _mm256_set1_epi16, _mm256_slli_epi16, _mm256_sub_epi8,
                _mm256_testz_si256, _mm256_xor_si256, _mm_cvtsi128_si64,
            };
            let (va0, va1) = avx2_words64_to_vec64_with_uniform(centroid_words, c_uni);
            let (vb0, vb1) = avx2_words64_to_vec64_with_uniform(sample_words, sample_uni);
            let vd0 = _mm256_sub_epi8(vb0, va0);
            let vd1 = _mm256_sub_epi8(vb1, va1);
            let coh = match precomputed_coh_opt {
                Some(c) => c,
                None => avx2_coherence_from_diff_vec64(vd0, vd1),
            };
            let eff_sample_e = clean_e * branchless_clamp_f32(0.5 * (1.0 + coh), 0.25, 1.0);
            let next_e = prev_e + 0.5 * eff_sample_e;
            let tot_e = prev_e + eff_sample_e;
            let safe_tot = branchless_select_f32(tot_e > DEFAULT_ENTROPY_EPS, tot_e, 1.0);
            let ratio = branchless_clamp_f32(
                eff_sample_e * fast_inv_pos_normal_f32(safe_tot),
                0.05,
                0.50,
            );
            let raw_s = (ratio * 128.0 + 0.5) as u8;
            let s = branchless_select_u8(raw_s > 128, 128, raw_s);
            let v_s = _mm256_set1_epi16(s as i16);
            let mask_ff00 = _mm256_set1_epi16(-256i16);
            let mask_00ff = _mm256_set1_epi16(0x00FFi16);

            let ev0 = _mm256_mulhrs_epi16(_mm256_slli_epi16(vd0, 8), v_s);
            let od0 = _mm256_mulhrs_epi16(_mm256_and_si256(vd0, mask_ff00), v_s);
            let sc0 = _mm256_or_si256(_mm256_and_si256(ev0, mask_00ff), _mm256_slli_epi16(od0, 8));

            let ev1 = _mm256_mulhrs_epi16(_mm256_slli_epi16(vd1, 8), v_s);
            let od1 = _mm256_mulhrs_epi16(_mm256_and_si256(vd1, mask_ff00), v_s);
            let sc1 = _mm256_or_si256(_mm256_and_si256(ev1, mask_00ff), _mm256_slli_epi16(od1, 8));

            let sc_or = _mm256_or_si256(sc0, sc1);
            if _mm256_testz_si256(sc_or, sc_or) != 0 {
                if prev_r_bar > 0.0 && prev_kappa > 0.0 {
                    return (*centroid_words, next_e, prev_r_bar, prev_kappa, false);
                }
                let rb = circular_r_bar_from_words_with_uniform_flag_64(centroid_words, c_uni);
                return (*centroid_words, next_e, rb, branchless_von_mises_kappa(rb), false);
            }

            let out0 = _mm256_add_epi8(va0, sc0);
            let out1 = _mm256_add_epi8(va1, sc1);
            let next_w = avx2_vec64_to_words64(out0, out1);

            let sc_splat0 = _mm256_broadcastb_epi8(_mm256_castsi256_si128(sc0));
            let sc_diff = _mm256_or_si256(
                _mm256_xor_si256(sc0, sc_splat0),
                _mm256_xor_si256(sc1, sc_splat0),
            );
            if _mm256_testz_si256(sc_diff, sc_diff) != 0 && prev_r_bar > 0.0 && prev_kappa > 0.0 {
                return (next_w, next_e, prev_r_bar, prev_kappa, true);
            }

            let out_splat0 = _mm256_broadcastb_epi8(_mm256_castsi256_si128(out0));
            let out_byte_diff = _mm256_or_si256(
                _mm256_xor_si256(out0, out_splat0),
                _mm256_xor_si256(out1, out_splat0),
            );
            if _mm256_testz_si256(out_byte_diff, out_byte_diff) != 0 {
                return (next_w, next_e, 1.0, VON_MISES_KAPPA_AT_MAX_R, true);
            }

            let diff01 = _mm256_xor_si256(out0, out1);
            let shuf0 = _mm256_permute4x64_epi64::<0x39>(out0);
            let comb = _mm256_or_si256(diff01, _mm256_xor_si256(out0, shuf0));
            let (mean_sin, mean_cos) = if _mm256_testz_si256(comb, comb) != 0 {
                let w0 = _mm_cvtsi128_si64(_mm256_castsi256_si128(out0)) as u64;
                let p = sum_sincos_biased_u64_from_word(w0);
                let sum_cos_q14 = ((p & 0xFFFF_FFFF) as i32) - Q14_BIAS_X8_I32;
                let sum_sin_q14 = ((p >> 32) as i32) - Q14_BIAS_X8_I32;
                (
                    (sum_sin_q14 as f32) * INV_8_Q14_F32,
                    (sum_cos_q14 as f32) * INV_8_Q14_F32,
                )
            } else {
                avx2_mean_sincos_from_vec64(out0, out1)
            };
            let rb = fast_sqrt_unit_f32(mean_sin * mean_sin + mean_cos * mean_cos);
            (next_w, next_e, rb, branchless_von_mises_kappa(rb), true)
        };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let coh = match precomputed_coh_opt {
            Some(c) => c,
            None => coherence_from_diff_words_64(
                d0,
                d1,
                swar_q7_circular_diff_x8(sample_words[2], centroid_words[2]),
                swar_q7_circular_diff_x8(sample_words[3], centroid_words[3]),
                swar_q7_circular_diff_x8(sample_words[4], centroid_words[4]),
                swar_q7_circular_diff_x8(sample_words[5], centroid_words[5]),
                swar_q7_circular_diff_x8(sample_words[6], centroid_words[6]),
                swar_q7_circular_diff_x8(sample_words[7], centroid_words[7]),
            ),
        };
        let eff_sample_e = clean_e * branchless_clamp_f32(0.5 * (1.0 + coh), 0.25, 1.0);
        let next_e = prev_e + 0.5 * eff_sample_e;
        let tot_e = prev_e + eff_sample_e;
        let safe_tot = branchless_select_f32(tot_e > DEFAULT_ENTROPY_EPS, tot_e, 1.0);
        let ratio = branchless_clamp_f32(
            eff_sample_e * fast_inv_pos_normal_f32(safe_tot),
            0.05,
            0.50,
        );
        let raw_s = (ratio * 128.0 + 0.5) as u8;
        let s = branchless_select_u8(raw_s > 128, 128, raw_s) as u64;
        let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let next_w = [
            swar_q7_circular_add_x8(
                centroid_words[0],
                swar_q7_scale_le128_with_bias_x8(d0, s, bias_restore),
            ),
            swar_q7_circular_add_x8(
                centroid_words[1],
                swar_q7_scale_le128_with_bias_x8(d1, s, bias_restore),
            ),
            swar_slerp_word_unchecked_x8(centroid_words[2], sample_words[2], s, bias_restore),
            swar_slerp_word_unchecked_x8(centroid_words[3], sample_words[3], s, bias_restore),
            swar_slerp_word_unchecked_x8(centroid_words[4], sample_words[4], s, bias_restore),
            swar_slerp_word_unchecked_x8(centroid_words[5], sample_words[5], s, bias_restore),
            swar_slerp_word_unchecked_x8(centroid_words[6], sample_words[6], s, bias_restore),
            swar_slerp_word_unchecked_x8(centroid_words[7], sample_words[7], s, bias_restore),
        ];
        let next_is_uniform = is_word_array_uniform_64(&next_w);
        let stationary = next_w[0] == centroid_words[0]
            && ((((next_w[1] ^ centroid_words[1]) | (next_w[2] ^ centroid_words[2]))
                | ((next_w[3] ^ centroid_words[3]) | (next_w[4] ^ centroid_words[4])))
                | (((next_w[5] ^ centroid_words[5]) | (next_w[6] ^ centroid_words[6]))
                    | (next_w[7] ^ centroid_words[7])))
                == 0;
        if stationary && prev_r_bar > 0.0 && prev_kappa > 0.0 {
            return (next_w, next_e, prev_r_bar, prev_kappa, false);
        }
        let rb = circular_r_bar_from_words_with_uniform_flag_64(&next_w, next_is_uniform);
        (next_w, next_e, rb, branchless_von_mises_kappa(rb), !stationary)
    }
}

/// Performs a single-pass 8-word SWAR toroidal centroid update with precomputed coherence
/// and word-uniform flags, returning `(next_words, next_energy, next_r_bar, next_kappa)`.
#[inline(always)]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn branchless_toroidal_centroid_step_words_with_coh_and_uni_64(
    centroid_words: &[u64; 8],
    prev_sample_count: u32,
    prev_base_energy: f32,
    prev_r_bar: f32,
    prev_kappa: f32,
    sample_words: &[u64; 8],
    precomputed_coh_opt: Option<f32>,
    sample_energy: f32,
    c_uni: bool,
    sample_uni: bool,
) -> ([u64; 8], f32, f32, f32) {
    let (next_w, next_e, rb, kappa, _changed) =
        branchless_toroidal_centroid_step_words_with_coh_uni_and_changed_64(
            centroid_words,
            prev_sample_count,
            prev_base_energy,
            prev_r_bar,
            prev_kappa,
            sample_words,
            precomputed_coh_opt,
            sample_energy,
            c_uni,
            sample_uni,
        );
    (next_w, next_e, rb, kappa)
}

/// Executes an **Online Toroidal $k$-Means M-Step Centroid Update** in 8 register-resident `u64`
/// SWAR words (`centroid_words`, `sample_words`) with an optional precomputed E-step phase
/// coherence `precomputed_coh_opt`, in strictly constant time (`0` data-dependent branches on
/// phase stationarity, normalized directional cosine weighting $w_{\mathrm{coh}} = \operatorname{clamp}(0.5(1 + \mathcal{C}_{\mathrm{phase}}), 0.25, 1.0)$).
///
/// Returns `(updated_centroid_words, next_base_energy, resultant_r_bar, von_mises_kappa)`.
#[inline(always)]
pub fn branchless_toroidal_centroid_step_words_with_coh_64(
    centroid_words: &[u64; 8],
    prev_sample_count: u32,
    prev_base_energy: f32,
    prev_r_bar: f32,
    prev_kappa: f32,
    sample_words: &[u64; 8],
    precomputed_coh_opt: Option<f32>,
    sample_energy: f32,
) -> ([u64; 8], f32, f32, f32) {
    let c_uni = is_word_array_uniform_64(centroid_words);
    let sample_uni = is_word_array_uniform_64(sample_words);
    branchless_toroidal_centroid_step_words_with_coh_and_uni_64(
        centroid_words,
        prev_sample_count,
        prev_base_energy,
        prev_r_bar,
        prev_kappa,
        sample_words,
        precomputed_coh_opt,
        sample_energy,
        c_uni,
        sample_uni,
    )
}

/// Executes an **Online Toroidal $k$-Means M-Step Centroid Update** in 8 register-resident `u64`
/// SWAR words (`centroid_words`, `sample_words`) in strictly constant time (`0` data-dependent
/// branches on phase stationarity).
///
/// Returns `(updated_centroid_words, next_base_energy, resultant_r_bar, von_mises_kappa)`.
#[inline(always)]
pub fn branchless_toroidal_centroid_step_words_64(
    centroid_words: &[u64; 8],
    prev_sample_count: u32,
    prev_base_energy: f32,
    prev_r_bar: f32,
    prev_kappa: f32,
    sample_words: &[u64; 8],
    sample_energy: f32,
) -> ([u64; 8], f32, f32, f32) {
    branchless_toroidal_centroid_step_words_with_coh_64(
        centroid_words,
        prev_sample_count,
        prev_base_energy,
        prev_r_bar,
        prev_kappa,
        sample_words,
        None,
        sample_energy,
    )
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline(always)]
pub(super) const fn swar_synergy_word_with_bias_x8(
    a: u64,
    d: u64,
    ratio_b: u64,
    bias_ratio: u64,
    beat_u7: u64,
    bias_beat: u64,
    mask: u64,
) -> u64 {
    swar_q7_circular_add_x8(
        swar_q7_circular_add_x8(
            a,
            swar_q7_scale_le128_with_bias_x8(d, ratio_b, bias_ratio),
        ),
        swar_q7_scale_le128_with_bias_x8(d, beat_u7, bias_beat),
    ) & mask
}

/// Synthesizes a non-linear synergistic 64-head Q7 phase profile directly in 8 register-resident
/// `u64` SWAR words (`0` intermediate stack arrays, `0` branches, `0 fdiv`), returning
/// `(fused_words, effective_pull, coherence)`.
#[inline]
#[must_use]
pub fn branchless_q7_synergy_fuse_words_64(
    a: &[u64; 8],
    weight_a: f32,
    b: &[u64; 8],
    weight_b: f32,
    synergy_gain: f32,
) -> ([u64; 8], f32, f32) {
    let clean_wa = branchless_relu_f32(weight_a);
    let clean_wb = branchless_relu_f32(weight_b);
    let base_sum = clean_wa + clean_wb;
    if base_sum <= 1.0e-9 {
        return ([0u64; 8], 0.0, 0.0);
    }

    let d0 = swar_q7_circular_diff_x8(b[0], a[0]);
    let d1 = swar_q7_circular_diff_x8(b[1], a[1]);
    let d2 = swar_q7_circular_diff_x8(b[2], a[2]);
    let d3 = swar_q7_circular_diff_x8(b[3], a[3]);
    let d4 = swar_q7_circular_diff_x8(b[4], a[4]);
    let d5 = swar_q7_circular_diff_x8(b[5], a[5]);
    let d6 = swar_q7_circular_diff_x8(b[6], a[6]);
    let d7 = swar_q7_circular_diff_x8(b[7], a[7]);

    let d_uni =
        ((d0 ^ d1) | (d0 ^ d2) | (d0 ^ d3) | (d0 ^ d4) | (d0 ^ d5) | (d0 ^ d6) | (d0 ^ d7)) == 0;
    let safe_gain = branchless_relu_f32(synergy_gain);

    if d_uni {
        if d0 == 0 {
            let effective_pull = base_sum + safe_gain * base_sum;
            return (*a, effective_pull, 1.0);
        }
        let coherence = coherence_from_uniform_diff_word_x8(d0);
        let pos_resonance = branchless_relu_f32(coherence);
        let liminal_beat = branchless_relu_f32(-coherence);
        let effective_pull = base_sum + safe_gain * base_sum * pos_resonance;

        let inv_sum = 128.0 * fast_inv_pos_normal_f32(base_sum);
        let ratio_b = (branchless_clamp_f32(clean_wb * inv_sum + 0.5, 0.0, 128.0) as u8) as u64;
        let beat_u7 =
            ((liminal_beat * branchless_clamp_f32(safe_gain, 0.0, 1.0) * 64.0 + 0.5) as u8) as u64;
        let bias_ratio = (128u64.wrapping_sub(ratio_b)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let step_r = swar_q7_scale_le128_with_bias_x8(d0, ratio_b, bias_ratio);
        let step0 = if beat_u7 == 0 {
            step_r
        } else {
            let bias_beat = (128u64.wrapping_sub(beat_u7)).wrapping_mul(SWAR_SPLAT_BYTE_64);
            swar_q7_circular_add_x8(
                step_r,
                swar_q7_scale_le128_with_bias_x8(d0, beat_u7, bias_beat),
            )
        };
        if step0 == 0 {
            return (*a, effective_pull, coherence);
        }
        let fused_w = [
            swar_q7_circular_add_x8(a[0], step0),
            swar_q7_circular_add_x8(a[1], step0),
            swar_q7_circular_add_x8(a[2], step0),
            swar_q7_circular_add_x8(a[3], step0),
            swar_q7_circular_add_x8(a[4], step0),
            swar_q7_circular_add_x8(a[5], step0),
            swar_q7_circular_add_x8(a[6], step0),
            swar_q7_circular_add_x8(a[7], step0),
        ];
        return (fused_w, effective_pull, coherence);
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Guarded by `target_feature = "avx2"`.
        unsafe {
            use core::arch::x86_64::_mm256_add_epi8;
            let (vd0, vd1) = avx2_u64x8_to_vec64(d0, d1, d2, d3, d4, d5, d6, d7);
            let coherence = avx2_coherence_from_diff_vec64(vd0, vd1);
            let pos_resonance = branchless_relu_f32(coherence);
            let liminal_beat = branchless_relu_f32(-coherence);
            let effective_pull = base_sum + safe_gain * base_sum * pos_resonance;

            let inv_sum = 128.0 * fast_inv_pos_normal_f32(base_sum);
            let ratio_b = (branchless_clamp_f32(clean_wb * inv_sum + 0.5, 0.0, 128.0) as u8) as u64;
            let beat_u7 = ((liminal_beat * branchless_clamp_f32(safe_gain, 0.0, 1.0) * 64.0 + 0.5)
                as u8) as u64;
            let (vr0, vr1) = avx2_scale_vec64(vd0, vd1, ratio_b);
            let (vst0, vst1) = if beat_u7 == 0 {
                (vr0, vr1)
            } else {
                let (vb0, vb1) = avx2_scale_vec64(vd0, vd1, beat_u7);
                (_mm256_add_epi8(vr0, vb0), _mm256_add_epi8(vr1, vb1))
            };
            let (va0, va1) = avx2_words64_to_vec64(a);
            let fused_w =
                avx2_vec64_to_words64(_mm256_add_epi8(va0, vst0), _mm256_add_epi8(va1, vst1));
            return (fused_w, effective_pull, coherence);
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let coherence = coherence_from_diff_words_64(d0, d1, d2, d3, d4, d5, d6, d7);
        let pos_resonance = branchless_relu_f32(coherence);
        let liminal_beat = branchless_relu_f32(-coherence);
        let effective_pull = base_sum + safe_gain * base_sum * pos_resonance;

        let inv_sum = 128.0 * fast_inv_pos_normal_f32(base_sum);
        let ratio_b = (branchless_clamp_f32(clean_wb * inv_sum + 0.5, 0.0, 128.0) as u8) as u64;
        let beat_u7 =
            ((liminal_beat * branchless_clamp_f32(safe_gain, 0.0, 1.0) * 64.0 + 0.5) as u8) as u64;
        let bias_ratio = (128u64.wrapping_sub(ratio_b)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let fused_w = if beat_u7 == 0 {
            [
                swar_q7_circular_add_x8(
                    a[0],
                    swar_q7_scale_le128_with_bias_x8(d0, ratio_b, bias_ratio),
                ),
                swar_q7_circular_add_x8(
                    a[1],
                    swar_q7_scale_le128_with_bias_x8(d1, ratio_b, bias_ratio),
                ),
                swar_q7_circular_add_x8(
                    a[2],
                    swar_q7_scale_le128_with_bias_x8(d2, ratio_b, bias_ratio),
                ),
                swar_q7_circular_add_x8(
                    a[3],
                    swar_q7_scale_le128_with_bias_x8(d3, ratio_b, bias_ratio),
                ),
                swar_q7_circular_add_x8(
                    a[4],
                    swar_q7_scale_le128_with_bias_x8(d4, ratio_b, bias_ratio),
                ),
                swar_q7_circular_add_x8(
                    a[5],
                    swar_q7_scale_le128_with_bias_x8(d5, ratio_b, bias_ratio),
                ),
                swar_q7_circular_add_x8(
                    a[6],
                    swar_q7_scale_le128_with_bias_x8(d6, ratio_b, bias_ratio),
                ),
                swar_q7_circular_add_x8(
                    a[7],
                    swar_q7_scale_le128_with_bias_x8(d7, ratio_b, bias_ratio),
                ),
            ]
        } else {
            let bias_beat = (128u64.wrapping_sub(beat_u7)).wrapping_mul(SWAR_SPLAT_BYTE_64);
            [
                swar_synergy_word_with_bias_x8(
                    a[0], d0, ratio_b, bias_ratio, beat_u7, bias_beat, !0u64,
                ),
                swar_synergy_word_with_bias_x8(
                    a[1], d1, ratio_b, bias_ratio, beat_u7, bias_beat, !0u64,
                ),
                swar_synergy_word_with_bias_x8(
                    a[2], d2, ratio_b, bias_ratio, beat_u7, bias_beat, !0u64,
                ),
                swar_synergy_word_with_bias_x8(
                    a[3], d3, ratio_b, bias_ratio, beat_u7, bias_beat, !0u64,
                ),
                swar_synergy_word_with_bias_x8(
                    a[4], d4, ratio_b, bias_ratio, beat_u7, bias_beat, !0u64,
                ),
                swar_synergy_word_with_bias_x8(
                    a[5], d5, ratio_b, bias_ratio, beat_u7, bias_beat, !0u64,
                ),
                swar_synergy_word_with_bias_x8(
                    a[6], d6, ratio_b, bias_ratio, beat_u7, bias_beat, !0u64,
                ),
                swar_synergy_word_with_bias_x8(
                    a[7], d7, ratio_b, bias_ratio, beat_u7, bias_beat, !0u64,
                ),
            ]
        };
        (fused_w, effective_pull, coherence)
    }
}

/// Synthesizes a non-linear synergistic 64-head Q7 phase profile and amplified pull force in
/// strictly constant time (`0` branches, `0 fdiv`), reusing circular-difference words `d[0..8]`
/// and hoisted SWAR bias multipliers for both coherence evaluation and geodesic + liminal beat interpolation.
#[inline]
pub fn branchless_q7_synergy_fuse_64(
    phases_a: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    weight_a: f32,
    phases_b: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    weight_b: f32,
    synergy_gain: f32,
    fused_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
) -> (f32, f32) {
    let a = load_q7_words_64(phases_a);
    let b = load_q7_words_64(phases_b);
    let (fused_w, effective_pull, coherence) =
        branchless_q7_synergy_fuse_words_64(&a, weight_a, &b, weight_b, synergy_gain);
    store_q7_words_64(fused_out, &fused_w);
    (effective_pull, coherence)
}

/// Applies branchless mutual inhibition gating directly on 8 register-resident `u64` SWAR words
/// (`target_words` vs `inhibitor_words`, `0` stack roundtrips, hoisted SWAR bias multiplier).
#[inline(always)]
pub fn branchless_q7_inhibit_words_64(
    target_words: &mut [u64; 8],
    target_weight: f32,
    inhibitor_words: &[u64; 8],
    inhibition_strength: f32,
) -> f32 {
    let clean_w = branchless_relu_f32(target_weight);
    let safe_strength = branchless_clamp_f32(branchless_relu_f32(inhibition_strength), 0.0, 1.0);
    if safe_strength <= 1.0e-9 {
        return clean_w;
    }
    let target_uni = is_word_array_uniform_64(target_words);
    let coherence =
        coherence_from_word_arrays_with_uniform_flag_64(target_words, target_uni, inhibitor_words);
    let opposition = branchless_relu_f32(-coherence);
    let retain_factor = branchless_clamp_f32(1.0 - safe_strength * opposition, 0.0, 1.0);
    let s = (((retain_factor * 128.0) + 0.5) as u8) as u64;
    if s >= 128 {
        return clean_w * retain_factor;
    }
    if s == 0 {
        *target_words = [0u64; 8];
        return 0.0;
    }
    if target_uni {
        let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let w0 = swar_q7_scale_le128_with_bias_x8(target_words[0], s, bias_restore);
        *target_words = [w0; 8];
        return clean_w * retain_factor;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Guarded by `target_feature = "avx2"`.
        unsafe {
            let (vt0, vt1) = avx2_words64_to_vec64(target_words);
            let (vs0, vs1) = avx2_scale_vec64(vt0, vt1, s);
            *target_words = avx2_vec64_to_words64(vs0, vs1);
        }
        return clean_w * retain_factor;
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        target_words[0] = swar_q7_scale_le128_with_bias_x8(target_words[0], s, bias_restore);
        target_words[1] = swar_q7_scale_le128_with_bias_x8(target_words[1], s, bias_restore);
        target_words[2] = swar_q7_scale_le128_with_bias_x8(target_words[2], s, bias_restore);
        target_words[3] = swar_q7_scale_le128_with_bias_x8(target_words[3], s, bias_restore);
        target_words[4] = swar_q7_scale_le128_with_bias_x8(target_words[4], s, bias_restore);
        target_words[5] = swar_q7_scale_le128_with_bias_x8(target_words[5], s, bias_restore);
        target_words[6] = swar_q7_scale_le128_with_bias_x8(target_words[6], s, bias_restore);
        target_words[7] = swar_q7_scale_le128_with_bias_x8(target_words[7], s, bias_restore);
        clean_w * retain_factor
    }
}

/// Applies branchless mutual inhibition gating to a target 64-head Q7 phase profile in a single
/// register-resident pass (`0` redundant reloads of `target_phases`, hoisted SWAR bias multiplier).
#[inline]
pub fn branchless_q7_inhibit_64(
    target_phases: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    target_weight: f32,
    inhibitor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    inhibition_strength: f32,
) -> f32 {
    let mut t = load_q7_words_64(target_phases);
    let b = load_q7_words_64(inhibitor_phases);
    let damped = branchless_q7_inhibit_words_64(&mut t, target_weight, &b, inhibition_strength);
    store_q7_words_64(target_phases, &t);
    damped
}

/// Projects an arbitrary `&[f32]` feature/embedding slice onto $SO(2)^{64}$ (`[gcso_q7_t; 64]`)
/// using 8-YMM (64-float) AVX2 SIMD accumulation for `values.len() >= 64` and branchless
/// sanitization of non-finite values.
#[inline]
pub fn branchless_project_f32_slice_to_q7_64(
    values: &[f32],
    gain: f32,
) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
    let mut out = [0i8; MAX_ATTENTION_HEADS];
    if values.is_empty() {
        return out;
    }
    let clean_gain = branchless_select_f32(
        branchless_is_finite_f32(gain) & (gain >= 0.0),
        gain,
        1.0,
    );
    let mut acc = [0.0_f32; MAX_ATTENTION_HEADS];
    let len = values.len();

    let folds = ((len + 63) >> 6) as f32;
    let inv_folds = clean_gain * fast_inv_pos_normal_f32(folds);

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    let rem_start = if len >= MAX_ATTENTION_HEADS {
        #[inline(always)]
        unsafe fn avx2_quantize_acc_vec_to_q7_u64(
            v_acc: core::arch::x86_64::__m256,
            v_inv_folds: core::arch::x86_64::__m256,
        ) -> u64 {
            // SAFETY: `target_feature = "avx2"` is enabled; clamps `v_acc * inv_folds` to `[-PI, PI]`
            // and quantizes 8 lanes to Q7 `[-128, 127]` in registers (`0` scalar float-to-int loop).
            unsafe {
                let rad = _mm256_min_ps(
                    _mm256_max_ps(_mm256_mul_ps(v_acc, v_inv_folds), _mm256_set1_ps(-PI)),
                    _mm256_set1_ps(PI),
                );
                let scaled = _mm256_mul_ps(rad, _mm256_set1_ps(Q7_PHASE_INV_SCALE_RAD));
                let sign_bit = _mm256_and_ps(scaled, _mm256_set1_ps(f32::from_bits(0x8000_0000)));
                let rounded = _mm256_add_ps(
                    scaled,
                    _mm256_or_ps(_mm256_set1_ps(0.5_f32), sign_bit),
                );
                let vi32 = _mm256_cvttps_epi32(rounded);
                let p16 = _mm_packs_epi32(
                    _mm256_castsi256_si128(vi32),
                    _mm256_extracti128_si256::<1>(vi32),
                );
                let p8 = _mm_packs_epi16(p16, p16);
                _mm_cvtsi128_si64(p8) as u64
            }
        }

        let ptr = values.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and `i + 64 <= len` is bounds-checked.
        unsafe {
            let inf_exp = _mm256_set1_epi32(0x7F80_0000);
            // Pass 1 (lanes 0..32): 4 YMM accumulators (`<= 6` live YMMs, `0` Windows x64 XMM stack spills)
            let mut v0 = _mm256_setzero_ps();
            let mut v1 = _mm256_setzero_ps();
            let mut v2 = _mm256_setzero_ps();
            let mut v3 = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 64 <= len {
                let x0 = _mm256_loadu_ps(ptr.add(i));
                let nf0 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
                    _mm256_and_si256(_mm256_castps_si256(x0), inf_exp),
                    inf_exp,
                ));
                v0 = _mm256_add_ps(v0, _mm256_andnot_ps(nf0, x0));

                let x1 = _mm256_loadu_ps(ptr.add(i + 8));
                let nf1 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
                    _mm256_and_si256(_mm256_castps_si256(x1), inf_exp),
                    inf_exp,
                ));
                v1 = _mm256_add_ps(v1, _mm256_andnot_ps(nf1, x1));

                let x2 = _mm256_loadu_ps(ptr.add(i + 16));
                let nf2 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
                    _mm256_and_si256(_mm256_castps_si256(x2), inf_exp),
                    inf_exp,
                ));
                v2 = _mm256_add_ps(v2, _mm256_andnot_ps(nf2, x2));

                let x3 = _mm256_loadu_ps(ptr.add(i + 24));
                let nf3 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
                    _mm256_and_si256(_mm256_castps_si256(x3), inf_exp),
                    inf_exp,
                ));
                v3 = _mm256_add_ps(v3, _mm256_andnot_ps(nf3, x3));
                i += 64;
            }
            let has_rem = i < len;
            let mut w_lo = [0u64; 4];
            if has_rem {
                let ap = acc.as_mut_ptr();
                _mm256_storeu_ps(ap, v0);
                _mm256_storeu_ps(ap.add(8), v1);
                _mm256_storeu_ps(ap.add(16), v2);
                _mm256_storeu_ps(ap.add(24), v3);
            } else {
                let v_inv = _mm256_set1_ps(inv_folds);
                w_lo[0] = avx2_quantize_acc_vec_to_q7_u64(v0, v_inv);
                w_lo[1] = avx2_quantize_acc_vec_to_q7_u64(v1, v_inv);
                w_lo[2] = avx2_quantize_acc_vec_to_q7_u64(v2, v_inv);
                w_lo[3] = avx2_quantize_acc_vec_to_q7_u64(v3, v_inv);
            }

            // Pass 2 (lanes 32..64): 4 YMM accumulators (`<= 6` live YMMs, `0` Windows x64 XMM stack spills)
            let mut v4 = _mm256_setzero_ps();
            let mut v5 = _mm256_setzero_ps();
            let mut v6 = _mm256_setzero_ps();
            let mut v7 = _mm256_setzero_ps();
            let mut j = 0usize;
            while j + 64 <= len {
                let x4 = _mm256_loadu_ps(ptr.add(j + 32));
                let nf4 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
                    _mm256_and_si256(_mm256_castps_si256(x4), inf_exp),
                    inf_exp,
                ));
                v4 = _mm256_add_ps(v4, _mm256_andnot_ps(nf4, x4));

                let x5 = _mm256_loadu_ps(ptr.add(j + 40));
                let nf5 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
                    _mm256_and_si256(_mm256_castps_si256(x5), inf_exp),
                    inf_exp,
                ));
                v5 = _mm256_add_ps(v5, _mm256_andnot_ps(nf5, x5));

                let x6 = _mm256_loadu_ps(ptr.add(j + 48));
                let nf6 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
                    _mm256_and_si256(_mm256_castps_si256(x6), inf_exp),
                    inf_exp,
                ));
                v6 = _mm256_add_ps(v6, _mm256_andnot_ps(nf6, x6));

                let x7 = _mm256_loadu_ps(ptr.add(j + 56));
                let nf7 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
                    _mm256_and_si256(_mm256_castps_si256(x7), inf_exp),
                    inf_exp,
                ));
                v7 = _mm256_add_ps(v7, _mm256_andnot_ps(nf7, x7));
                j += 64;
            }
            if !has_rem {
                let v_inv = _mm256_set1_ps(inv_folds);
                let out_words = [
                    w_lo[0],
                    w_lo[1],
                    w_lo[2],
                    w_lo[3],
                    avx2_quantize_acc_vec_to_q7_u64(v4, v_inv),
                    avx2_quantize_acc_vec_to_q7_u64(v5, v_inv),
                    avx2_quantize_acc_vec_to_q7_u64(v6, v_inv),
                    avx2_quantize_acc_vec_to_q7_u64(v7, v_inv),
                ];
                store_q7_words_64(&mut out, &out_words);
                return out;
            }
            let ap = acc.as_mut_ptr();
            _mm256_storeu_ps(ap.add(32), v4);
            _mm256_storeu_ps(ap.add(40), v5);
            _mm256_storeu_ps(ap.add(48), v6);
            _mm256_storeu_ps(ap.add(56), v7);
            i
        }
    } else {
        0usize
    };

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    let rem_start = 0usize;

    let mut i = rem_start;
    while i < len {
        let v = values[i];
        acc[i & 63] += branchless_select_f32(branchless_is_finite_f32(v), v, 0.0);
        i += 1;
    }

    let mut h = 0usize;
    while h < MAX_ATTENTION_HEADS {
        let rad = branchless_clamp_f32(acc[h] * inv_folds, -PI, PI);
        out[h] = branchless_rad_to_q7(rad);
        h += 1;
    }
    out
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

