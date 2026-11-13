// name: src/core/src/traits/branchless.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lv.0 Foundation 3: `BranchlessKernel` blanket trait, 64-head phase algebra
//! (word-uniform accelerated coherence & circular resultant, register-resident unified-field
//! folding, Phase-Proximity Gravity manifold folding, zero-hyperparameter resonant pull,
//! $S^1$ geodesic weighted folding, Online Toroidal $k$-Means M-step centroid updates,
//! SWAR $S^1$ synergy fusion, Attention $\times$ DPSR phase-interference & Bayesian
//! stigmergic score modulation, Limina transition log-probability, and AVX2 64-head
//! embedding projection), and strictly constant-time subnormal-flushed lockstep AVX2 SIMD kernels.
//! Base LUTs, IEEE 802.3 CRC32 tables, scalar selectors, and 8-lane SWAR primitives are
//! defined in [`crate::traits::primitives`].

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::inline_always)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::incompatible_msrv)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_option_as_deref)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::ifs_same_cond)]
#![allow(clippy::manual_unwrap_or)]

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::*;

use super::primitives::{
    DEFAULT_ENTROPY_EPS, F32_MIN_POSITIVE_NORMAL, INV_64_Q14_F32, INV_8_Q14_F32,
    MAX_ATTENTION_HEADS, MAX_BAYESIAN_STIGMERGIC_DENSITY, MAX_PHASE_INTERFERENCE_INV_TEMP,
    MIN_PHASE_INTERFERENCE_INV_TEMP, POPCOUNT_RSQRT_LUT_257_REF, Q14_BIAS_X8_I32,
    Q7_FULL_COS_I16_REF, Q7_FULL_COS_REF, Q7_FULL_SINCOS_BIASED_U64_REF, SUBNORMAL_SQ_RISK_FLOOR,
    SWAR_SPLAT_BYTE_64, VON_MISES_KAPPA_AT_MAX_R, branchless_abs_f32, branchless_atan2_f32,
    branchless_atan2_to_q7_fast, branchless_clamp_f32, branchless_crc32_ieee, branchless_exp_f32,
    branchless_eyring_kramers_escape, branchless_flush_subnormal_f32, branchless_inv_pos_f32,
    branchless_is_finite_f32, branchless_ln_f32, branchless_max_f32,
    branchless_phase_proximity_gravity_f32, branchless_q7_add_sat, branchless_q7_circular_diff,
    branchless_q7_circular_lerp, branchless_q7_fold_s1_geodesic, branchless_q7_neg_sat,
    branchless_q7_scale_u7, branchless_q7_sub_sat, branchless_q7_to_rad,
    branchless_qdps_filter_q7, branchless_rad_to_q7, branchless_rad_to_qdps_step_u8,
    branchless_relu_f32, branchless_ripa_clamp_rad, branchless_rotate_so2_pair,
    branchless_rotate_so2_pair_scaled, branchless_rsqrt_f32, branchless_sanitize_z_score_f32,
    branchless_select_f32, branchless_select_i32, branchless_select_i64, branchless_select_i8,
    branchless_select_q7, branchless_select_u16, branchless_select_u32, branchless_select_u64,
    branchless_select_u8, branchless_select_usize, branchless_sigmoid_f32, branchless_sincos_f32,
    branchless_sincos_half_pi_f32, branchless_sincos_q7,
    branchless_sincos_q7_geodesic_ripa_default_lut, branchless_sincos_q7_ripa,
    branchless_sincos_q7_ripa_default_lut, branchless_sincos_q7_ripa_preinv,
    branchless_splitmix64, branchless_sqrt_f32, branchless_tanh_f32, branchless_von_mises_kappa,
    fast_inv_len_f32, fast_inv_pos_normal_f32, fast_ln_1p_bounded_f32, fast_sqrt_unit_f32,
    is_word_array_uniform_64, load_q7_words_64, sanitize_inplace_gain_f32, store_q7_words_64,
    store_q7_x8_slice, swar_q7_add_sat_x4, swar_q7_add_sat_x8, swar_q7_circular_add_x8,
    swar_q7_circular_diff_x8, swar_q7_fold_s1_geodesic_x8, swar_q7_mean_i8_from_words_64,
    swar_q7_neg_sat_x8, swar_q7_qdps_filter_x8, swar_q7_scale_le128_with_bias_x8,
    swar_q7_scale_u7_x8, swar_q7_sub_sat_x4, swar_q7_sub_sat_x8, swar_u8_add_sat_x8,
    swar_u8_halve_decay_x8, swar_u8_max_x8, swar_u8_sub_sat_x8,
};
#[cfg(test)]
use super::primitives::{
    CRC32_UNIFORM56_PAD8_ZERO_SEED_REF, CRYSTAL_CRC32_KAPPA_AT_MAX_R_FINAL_CONTRIB,
    CRYSTAL_CRC32_KAPPA_MAX_FINAL_CONTRIB, VON_MISES_KAPPA_AT_MAX_R_BITS,
    branchless_crc32_128b_words, branchless_update_crc32_ieee, crc32_block64_zero_seed_words,
    crc32_slice16_dep_lo0, crc32_step_block64_words, crc32_step_l5,
    eval_fused_ripa_pspm_sincos_q7_fast, eval_ripa_sincos_q7_fast, swar_q7_pack_x8,
    swar_q7_unpack_x8,
};
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
use super::primitives::Q14_BIAS_X64_I32;
use crate::abi::gcso_q7_t;
use core::f32::consts::{LN_2, PI};
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::f32::consts::FRAC_PI_2;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use super::primitives::{Q7_PHASE_INV_SCALE_RAD, TANH_LUT_MAX_X};

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
    if w_cand <= 1.0e-12 {
        return total_w;
    }
    if w_run <= 1.0e-12 {
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
const fn sum_cos_q14_from_diff_word(dw: u64) -> i32 {
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
#[inline(always)]
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
    if clean_energy <= 1.0e-12 {
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
        if clean_energy <= 1.0e-12 {
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
        let is_active = res_pull > 1.0e-12;
        if !is_active {
            return (t, init_w, 0.0, default_dominant_idx, 0, false, false);
        }
        let total_w = init_w + res_pull;
        let out_w = if init_w <= 1.0e-12 {
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
                if clean_energy > 1.0e-12 {
                    let check_b_uni =
                        (running_weight <= 1.0e-12) | (r_uni & (t_uni | !trail_nonzero));
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
                        let is_active = res_pull > 1.0e-12;
                        let is_better = res_pull > best_pull;
                        let total_w = running_weight + res_pull;

                        if running_weight <= 1.0e-12 {
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
                        let is_active = res_pull > 1.0e-12;
                        let is_better = res_pull > best_pull;
                        let total_w = running_weight + res_pull;

                        if running_weight <= 1.0e-12 {
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

            let has_pull = (active_count > 0) & (anchor_pull_sum > 1.0e-12);
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
            if clean_energy > 1.0e-12 {
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
                let is_active = res_pull > 1.0e-12;
                let is_better = res_pull > best_pull;
                let total_w = running_weight + res_pull;

                if running_weight <= 1.0e-12 {
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

        let has_pull = (active_count > 0) & (anchor_pull_sum > 1.0e-12);
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
                if mass > 1.0e-12 {
                    let check_b_uni =
                        (running_pull <= 1.0e-12) | (r_uni & (q_uni | !q_nonzero));
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
                        if grav_pull > 1.0e-12 {
                            let is_better = grav_pull > best_pull;
                            let total_w = running_pull + grav_pull;
                            if running_pull <= 1.0e-12 {
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
                        if grav_pull > 1.0e-12 {
                            let is_better = grav_pull > best_pull;
                            let total_w = running_pull + grav_pull;
                            if running_pull <= 1.0e-12 {
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
            let has_pull = (active_count > 0) & (running_pull > 1.0e-12);
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
            if mass > 1.0e-12 {
                let b = load_q7_words_64(&anchor_phases[idx]);
                let need_b_uni =
                    (q_nonzero & q_is_uniform) | (running_pull <= 1.0e-12) | r_is_uniform;
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
                if grav_pull > 1.0e-12 {
                    let is_better = grav_pull > best_pull;
                    let total_w = running_pull + grav_pull;
                    if running_pull <= 1.0e-12 {
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
        let has_pull = running_pull > 1.0e-12;
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
const fn sum_sincos_biased_u64_from_word(w: u64) -> u64 {
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
const fn sum_sincos_q14_from_words_scalar_64(w: &[u64; 8]) -> (f32, f32) {
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
fn mean_sincos_from_words_with_uniform_flag_64(w: &[u64; 8], w_is_uniform: bool) -> (f32, f32) {
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
    if clean_e <= 1.0e-12 {
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
    if prev_sample_count == 0 || prev_e <= 1.0e-12 {
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
        let safe_tot = branchless_select_f32(tot_e > 1.0e-12, tot_e, 1.0);
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
            let safe_tot = branchless_select_f32(tot_e > 1.0e-12, tot_e, 1.0);
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
        let safe_tot = branchless_select_f32(tot_e > 1.0e-12, tot_e, 1.0);
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
const fn swar_synergy_word_with_bias_x8(
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

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_hsum3_ps(v0: __m256, v1: __m256, v2: __m256) -> (f32, f32, f32) {
    // SAFETY: Caller guarantees AVX2 is enabled.
    unsafe {
        let (s0, s1) = avx2_hsum2_ps(v0, v1);
        (s0, s1, avx2_hsum_ps(v2))
    }
}

/// Evaluates $(\sum_i |x_i|, \sum_i x_i^2)$ across `slice` using a 32-float (2-pair tree-reduced)
/// AVX2 SIMD kernel that computes `a = |x|` once per vector and uses $a^2 \equiv x^2$ to eliminate
/// Port 0/1 accumulator contention.
#[inline(always)]
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
#[inline(always)]
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
#[inline(always)]
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
fn sum_sq_subnormal_mantissa_f32_simd8(slice: &[f32]) -> f32 {
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
#[inline(always)]
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
        let mut max_val =
            branchless_max_f32(branchless_max_f32(m[0], m[1]), branchless_max_f32(m[2], m[3]));
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
#[inline(always)]
pub(crate) fn scale_f32_slice_unrolled8(buf: &mut [f32], scale: f32) {
    let len = buf.len();
    if len == 0 {
        return;
    }
    let eff_scale = branchless_flush_subnormal_f32(sanitize_inplace_gain_f32(buf[0], scale));
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
        // normal float (`1 <= exp <= 254`), `±Inf`, or `NaN` (`exp == 255 -> +Inf`) satisfies
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

/// Applies RIPA hyperbolic tangent soft-clamping ($\theta_{\mathrm{safe}} = \theta_{\max}\tanh(\theta / \theta_{\max})$)
/// in-place across `angles` using an 8-float AVX2 Padé $(5, 4)$ rational polynomial + `_mm256_rcp_ps`
/// 2-step Newton-Raphson reciprocal kernel (`<= 6` live YMM registers `ymm0..ymm5`, `0` Windows x64
/// `xmm6..xmm15` stack saves, and branchless `NaN`/`±Inf`/subnormal flushing in the FP domain).
#[inline(always)]
pub(crate) fn clamp_angles_ripa_in_place_simd8(angles: &mut [f32], max_rad: f32) {
    let len = angles.len();
    if len == 0 {
        return;
    }
    let inv_max = branchless_inv_pos_f32(max_rad);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let ptr = angles.as_mut_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and all loads/stores are bounds-checked against `len`.
        // Constants are materialized inline (`vmovaps`/`vbroadcastss`) so peak live YMM usage is capped
        // at 5 registers (`v`, `x`, `x2`, `x4`, `den`/`num`) <= 6 (`0` Windows x64 `xmm6..xmm15` stack saves).
        unsafe {
            let mut i = 0usize;
            while i + 8 <= len {
                let p = ptr.add(i);
                let v = _mm256_loadu_ps(p);
                let inf_exp = _mm256_castsi256_ps(_mm256_set1_epi32(0x7F80_0000));
                let fin_mask = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_and_ps(v, inf_exp), inf_exp);
                let clean_v = _mm256_and_ps(v, fin_mask);
                let u = _mm256_mul_ps(clean_v, _mm256_set1_ps(inv_max));
                let x = _mm256_min_ps(
                    _mm256_max_ps(u, _mm256_set1_ps(-TANH_LUT_MAX_X)),
                    _mm256_set1_ps(TANH_LUT_MAX_X),
                );
                let x2 = _mm256_mul_ps(x, x);
                let x4 = _mm256_mul_ps(x2, x2);
                let num = _mm256_mul_ps(
                    _mm256_mul_ps(x, _mm256_set1_ps(max_rad)),
                    _mm256_add_ps(
                        _mm256_add_ps(
                            _mm256_set1_ps(945.0_f32),
                            _mm256_mul_ps(_mm256_set1_ps(105.0_f32), x2),
                        ),
                        x4,
                    ),
                );
                let den = _mm256_add_ps(
                    _mm256_add_ps(
                        _mm256_set1_ps(945.0_f32),
                        _mm256_mul_ps(_mm256_set1_ps(420.0_f32), x2),
                    ),
                    _mm256_mul_ps(_mm256_set1_ps(15.0_f32), x4),
                );
                let r0 = _mm256_rcp_ps(den);
                let two = _mm256_set1_ps(2.0_f32);
                let r1 = _mm256_mul_ps(r0, _mm256_sub_ps(two, _mm256_mul_ps(den, r0)));
                let r2 = _mm256_mul_ps(r1, _mm256_sub_ps(two, _mm256_mul_ps(den, r1)));
                let prod = _mm256_mul_ps(num, r2);
                let norm_mask = _mm256_cmp_ps::<_CMP_GT_OQ>(
                    _mm256_and_ps(prod, inf_exp),
                    _mm256_setzero_ps(),
                );
                _mm256_storeu_ps(p, _mm256_and_ps(prod, norm_mask));
                i += 8;
            }
            while i < len {
                let a = *ptr.add(i);
                let clean_a = branchless_select_f32(branchless_is_finite_f32(a), a, 0.0);
                *ptr.add(i) =
                    branchless_flush_subnormal_f32(
                        max_rad * branchless_tanh_f32(clean_a * inv_max),
                    );
                i += 1;
            }
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for angle in angles.iter_mut() {
            let a = *angle;
            let clean_a = branchless_select_f32(branchless_is_finite_f32(a), a, 0.0);
            *angle =
                branchless_flush_subnormal_f32(max_rad * branchless_tanh_f32(clean_a * inv_max));
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
            _mm256_castps256_ps128, _mm256_extractf128_ps, _mm_add_ps, _mm_movehl_ps,
            _mm_movelh_ps,
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
            _mm256_cmp_ps::<_CMP_GT_OQ>(norm_sq, _mm256_set1_ps(1.0e-12_f32)),
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
            let valid = (norm_sq > 1.0e-12) & branchless_is_finite_f32(norm_sq);
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
        let out_arr =
            unsafe { &mut *phases_out.as_mut_ptr().cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>() };
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
pub(crate) fn sincos_f32_x8_simd(
    angles: &[f32],
    sin_out: &mut [f32; 8],
    cos_out: &mut [f32; 8],
) {
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
                let x_fold_lo =
                    _mm256_blendv_ps(x, _mm256_sub_ps(_mm256_set1_ps(-PI), x), lt_nhp);
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
                let sin_v =
                    _mm256_mul_ps(x_folded, _mm256_add_ps(_mm256_mul_ps(x2, sp), one_v));

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
#[inline(always)]
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

/// Bank-conflict-free 64B cache-line-staged 2-tensor (`Q`, `K`) AVX2 $SO(2)$ 32-float block rotation.
///
/// Rotates the two 64-byte cache lines of `qp` (`+0..+64B` and `+64..+128B`) via `avx2_rotate_so2_block32`
/// before rotating `kp` with the same register-resident `(v_cos, v_sin)`. Pairing `p0` (`+0B`, bit 5 = 0)
/// and `p1` (`+32B`, bit 5 = 1) inside each 64-byte cache line issues dual 256-bit loads to opposite L1D
/// 32B cache banks rather than colliding `qp` and `kp` at identical 4KB page offsets `(qp ^ kp) & 0xFFF == 0`,
/// while keeping peak live YMM usage `<= 6` (`ymm0..ymm5`, zero Windows x64 `xmm6..xmm15` stack spills).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_rotate_so2_pair_block32(
    qp: *mut f32,
    kp: *mut f32,
    v_cos: __m256,
    v_sin: __m256,
) {
    // SAFETY: Caller guarantees `qp` and `kp` are disjoint, valid for 32 floats, and AVX2 is enabled.
    unsafe {
        avx2_rotate_so2_block32(qp, v_cos, v_sin);
        avx2_rotate_so2_block32(kp, v_cos, v_sin);
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

/// Blanket trait providing branchless bitwise multiplexers, `#![no_std]` transcendentals,
/// division-free (`0 fdiv`) reciprocals, and 4/8-lane SWAR Q7 & `u8` arithmetic to all engine types.
pub trait BranchlessKernel {
    /// Branchless bitwise selection for `u8`.
    #[inline(always)]
    fn select_u8(&self, cond: bool, if_true: u8, if_false: u8) -> u8 {
        branchless_select_u8(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `i8`.
    #[inline(always)]
    fn select_i8(&self, cond: bool, if_true: i8, if_false: i8) -> i8 {
        branchless_select_i8(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `u16`.
    #[inline(always)]
    fn select_u16(&self, cond: bool, if_true: u16, if_false: u16) -> u16 {
        branchless_select_u16(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `u32`.
    #[inline(always)]
    fn select_u32(&self, cond: bool, if_true: u32, if_false: u32) -> u32 {
        branchless_select_u32(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `u64`.
    #[inline(always)]
    fn select_u64(&self, cond: bool, if_true: u64, if_false: u64) -> u64 {
        branchless_select_u64(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `usize`.
    #[inline(always)]
    fn select_usize(&self, cond: bool, if_true: usize, if_false: usize) -> usize {
        branchless_select_usize(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `i32`.
    #[inline(always)]
    fn select_i32(&self, cond: bool, if_true: i32, if_false: i32) -> i32 {
        branchless_select_i32(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `i64`.
    #[inline(always)]
    fn select_i64(&self, cond: bool, if_true: i64, if_false: i64) -> i64 {
        branchless_select_i64(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `gcso_q7_t`.
    #[inline(always)]
    fn select_q7(&self, cond: bool, if_true: gcso_q7_t, if_false: gcso_q7_t) -> gcso_q7_t {
        branchless_select_q7(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `f32`.
    #[inline(always)]
    fn select_f32(&self, cond: bool, if_true: f32, if_false: f32) -> f32 {
        branchless_select_f32(cond, if_true, if_false)
    }
    /// Branchless finiteness check for `f32`.
    #[inline(always)]
    fn is_finite_f32(&self, x: f32) -> bool {
        branchless_is_finite_f32(x)
    }
    /// Branchlessly flushes IEEE-754 subnormals and `-0.0` to `+0.0`.
    #[inline(always)]
    fn flush_subnormal_f32(&self, x: f32) -> f32 {
        branchless_flush_subnormal_f32(x)
    }
    /// Branchless IEEE-754 absolute value (`|x|`).
    #[inline(always)]
    fn abs_f32(&self, x: f32) -> f32 {
        branchless_abs_f32(x)
    }
    /// Branchless ReLU ($\max(0.0, x)$).
    #[inline(always)]
    fn relu_f32(&self, x: f32) -> f32 {
        branchless_relu_f32(x)
    }
    /// Branchless interval clamp.
    #[inline(always)]
    fn clamp_f32(&self, x: f32, lo: f32, hi: f32) -> f32 {
        branchless_clamp_f32(x, lo, hi)
    }
    /// Branchless Z-score sanitizer.
    #[inline(always)]
    fn sanitize_z_score_f32(&self, z_score: f32, max_ceiling: f32) -> f32 {
        branchless_sanitize_z_score_f32(z_score, max_ceiling)
    }
    /// Division-free (`0 fdiv`) branchless positive reciprocal ($1 / x$).
    #[inline(always)]
    fn inv_pos_f32(&self, val: f32) -> f32 {
        branchless_inv_pos_f32(val)
    }
    /// Branchless hyperbolic tangent ($\tanh(x)$).
    #[inline(always)]
    fn tanh_f32(&self, x: f32) -> f32 {
        branchless_tanh_f32(x)
    }
    /// Branchless logistic sigmoid ($\sigma(x)$).
    #[inline(always)]
    fn sigmoid_f32(&self, x: f32) -> f32 {
        branchless_sigmoid_f32(x)
    }
    /// Branchless RIPA soft-bounded phase clamp.
    #[inline(always)]
    fn ripa_clamp_rad(&self, delta_rad: f32, max_rad: f32) -> f32 {
        branchless_ripa_clamp_rad(delta_rad, max_rad)
    }
    /// Branchless natural logarithm ($\ln(x)$).
    #[inline(always)]
    fn ln_f32(&self, val: f32) -> f32 {
        branchless_ln_f32(val)
    }
    /// Branchless fast exponential ($e^x$).
    #[inline(always)]
    fn exp_f32(&self, x: f32) -> f32 {
        branchless_exp_f32(x)
    }
    /// Division-free (`0 fdiv`) branchless square root ($\sqrt{x}$).
    #[inline(always)]
    fn sqrt_f32(&self, val: f32) -> f32 {
        branchless_sqrt_f32(val)
    }
    /// Division-free (`0 fdiv`) branchless reciprocal square root ($1 / \sqrt{x}$).
    #[inline(always)]
    fn rsqrt_f32(&self, val: f32) -> f32 {
        branchless_rsqrt_f32(val)
    }
    /// Branchless fused $(\sin\theta, \cos\theta)$ for arbitrary finite angles.
    #[inline(always)]
    fn sincos_f32(&self, theta_rad: f32) -> (f32, f32) {
        branchless_sincos_f32(theta_rad)
    }
    /// Branchless direct $(\sin\theta, \cos\theta)$ for angles bounded in $[-\pi/2, \pi/2]$.
    #[inline(always)]
    fn sincos_half_pi_f32(&self, theta_rad: f32) -> (f32, f32) {
        branchless_sincos_half_pi_f32(theta_rad)
    }
    /// Division-free RIPA soft-clamped $(\sin\theta_{\mathrm{safe}}, \cos\theta_{\mathrm{safe}})$ from Q7.
    #[inline(always)]
    fn sincos_q7_ripa(&self, phase_q7: gcso_q7_t, max_rad: f32) -> (f32, f32) {
        branchless_sincos_q7_ripa(phase_q7, max_rad)
    }
    /// $\mathcal{O}(1)$ 256-entry LUT RIPA $(\sin\theta_{\mathrm{safe}}, \cos\theta_{\mathrm{safe}})$ under `DEFAULT_RIPA_CLAMP_MAX_RAD`.
    #[inline(always)]
    fn sincos_q7_ripa_default_lut(&self, phase_q7: gcso_q7_t) -> (f32, f32) {
        branchless_sincos_q7_ripa_default_lut(phase_q7)
    }
    /// $\mathcal{O}(1)$ 256-entry LUT combining $S^1$ geodesic folding and RIPA clamping.
    #[inline(always)]
    fn sincos_q7_geodesic_ripa_default_lut(&self, phase_q7: gcso_q7_t) -> (f32, f32) {
        branchless_sincos_q7_geodesic_ripa_default_lut(phase_q7)
    }
    /// Pre-inverted RIPA $(\sin\theta_{\mathrm{safe}}, \cos\theta_{\mathrm{safe}})$ from Q7.
    #[inline(always)]
    fn sincos_q7_ripa_preinv(
        &self,
        phase_q7: gcso_q7_t,
        max_rad: f32,
        q7_to_norm_scale: f32,
    ) -> (f32, f32) {
        branchless_sincos_q7_ripa_preinv(phase_q7, max_rad, q7_to_norm_scale)
    }
    /// Direct 256-entry LUT $(\sin\theta, \cos\theta)$ from Q7.
    #[inline(always)]
    fn sincos_q7(&self, phase_q7: gcso_q7_t) -> (f32, f32) {
        branchless_sincos_q7(phase_q7)
    }
    /// Branchless four-quadrant $\mathrm{atan2}(y, x) \in [-\pi, \pi]$.
    #[inline(always)]
    fn atan2_f32(&self, y: f32, x: f32) -> f32 {
        branchless_atan2_f32(y, x)
    }
    /// Branchless four-quadrant $\mathrm{atan2}(y, x)$ directly quantized into `gcso_q7_t`.
    #[inline(always)]
    fn atan2_to_q7_fast(&self, y: f32, x: f32) -> gcso_q7_t {
        branchless_atan2_to_q7_fast(y, x)
    }
    /// Branchless von Mises concentration parameter $\kappa_c$ from $\bar{R}_c \in [0, 1)$.
    #[inline(always)]
    fn von_mises_kappa(&self, r_bar: f32) -> f32 {
        branchless_von_mises_kappa(r_bar)
    }
    /// Evaluates circular resultant length $\bar{R}_c \in [0, 1]$ from 8 `u64` SWAR words.
    #[inline(always)]
    fn circular_r_bar_words_64(&self, words: &[u64; 8]) -> f32 {
        circular_r_bar_from_words_64(words)
    }
    /// Evaluates both 64-head phase coherence and circular resultant statistics in a single register pass.
    #[inline(always)]
    fn coherence_and_resultant_64(
        &self,
        phases_a: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        phases_b: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> (f32, f32, gcso_q7_t, f32) {
        branchless_q7_coherence_and_resultant_64(phases_a, phases_b)
    }
    /// Evaluates rounded arithmetic mean Q7 phase across 8 `u64` SWAR words.
    #[inline(always)]
    fn q7_mean_i8_words_64(&self, words: &[u64; 8]) -> gcso_q7_t {
        swar_q7_mean_i8_from_words_64(words)
    }
    /// Executes an Online Toroidal $k$-Means M-step centroid update in 8 `u64` SWAR words.
    #[inline(always)]
    fn toroidal_centroid_step_words_64(
        &self,
        centroid_words: &[u64; 8],
        prev_sample_count: u32,
        prev_base_energy: f32,
        prev_r_bar: f32,
        prev_kappa: f32,
        sample_words: &[u64; 8],
        sample_energy: f32,
    ) -> ([u64; 8], f32, f32, f32) {
        branchless_toroidal_centroid_step_words_64(
            centroid_words,
            prev_sample_count,
            prev_base_energy,
            prev_r_bar,
            prev_kappa,
            sample_words,
            sample_energy,
        )
    }
    /// Executes an Online Toroidal $k$-Means M-step centroid update in 8 `u64` SWAR words with an optional precomputed E-step coherence.
    #[inline(always)]
    fn toroidal_centroid_step_words_with_coh_64(
        &self,
        centroid_words: &[u64; 8],
        prev_sample_count: u32,
        prev_base_energy: f32,
        prev_r_bar: f32,
        prev_kappa: f32,
        sample_words: &[u64; 8],
        precomputed_coh_opt: Option<f32>,
        sample_energy: f32,
    ) -> ([u64; 8], f32, f32, f32) {
        branchless_toroidal_centroid_step_words_with_coh_64(
            centroid_words,
            prev_sample_count,
            prev_base_energy,
            prev_r_bar,
            prev_kappa,
            sample_words,
            precomputed_coh_opt,
            sample_energy,
        )
    }
    /// Evaluates Phase-Proximity Gravity $F_{\mathrm{grav}}$ on $SO(2)^{64}$.
    #[inline(always)]
    fn phase_proximity_gravity_f32(
        &self,
        gravity_mass: f32,
        phase_coherence: f32,
        softening_eps: f32,
    ) -> f32 {
        branchless_phase_proximity_gravity_f32(gravity_mass, phase_coherence, softening_eps)
    }
    /// Projects an `&[f32]` feature/embedding slice onto $SO(2)^{64}$ (`[gcso_q7_t; 64]`).
    #[inline(always)]
    fn project_f32_slice_to_q7_64(
        &self,
        values: &[f32],
        gain: f32,
    ) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        branchless_project_f32_slice_to_q7_64(values, gain)
    }
    /// Branchless scale-invariant Eyring-Kramers tunneling escape probability.
    #[inline(always)]
    fn eyring_kramers_escape(&self, cvoid: f32, tau_eff: f32, a: f32, b: f32, d0: f32) -> f32 {
        branchless_eyring_kramers_escape(cvoid, tau_eff, a, b, d0)
    }
    /// Branchless $SO(2)$ 2D rotation operator.
    #[inline(always)]
    fn rotate_so2_pair(&self, q0: f32, q1: f32, sin_t: f32, cos_t: f32) -> (f32, f32) {
        branchless_rotate_so2_pair(q0, q1, sin_t, cos_t)
    }
    /// Branchless composite $SO(2)$ rotation + isotropic scaling operator.
    #[inline(always)]
    fn rotate_so2_pair_scaled(
        &self,
        q0: f32,
        q1: f32,
        sin_t: f32,
        cos_t: f32,
        scale: f32,
    ) -> (f32, f32) {
        branchless_rotate_so2_pair_scaled(q0, q1, sin_t, cos_t, scale)
    }
    /// Converts Q7 phase to radians.
    #[inline(always)]
    fn q7_to_rad(&self, phase_q7: gcso_q7_t) -> f32 {
        branchless_q7_to_rad(phase_q7)
    }
    /// Quantizes radian angle to Q7 format.
    #[inline(always)]
    fn rad_to_q7(&self, rad: f32) -> gcso_q7_t {
        branchless_rad_to_q7(rad)
    }
    /// Quantizes radian QDPS threshold to ceiling `u8` step.
    #[inline(always)]
    fn rad_to_qdps_step_u8(&self, min_step_rad: f32) -> u8 {
        branchless_rad_to_qdps_step_u8(min_step_rad)
    }
    /// Branchless Q7 saturating addition.
    #[inline(always)]
    fn q7_add_sat(&self, a: gcso_q7_t, b: gcso_q7_t) -> gcso_q7_t {
        branchless_q7_add_sat(a, b)
    }
    /// Branchless Q7 saturating subtraction.
    #[inline(always)]
    fn q7_sub_sat(&self, a: gcso_q7_t, b: gcso_q7_t) -> gcso_q7_t {
        branchless_q7_sub_sat(a, b)
    }
    /// Branchless Q7 circular difference on $S^1$.
    #[inline(always)]
    fn q7_circular_diff(&self, a: gcso_q7_t, b: gcso_q7_t) -> gcso_q7_t {
        branchless_q7_circular_diff(a, b)
    }
    /// Branchless Q7 $S^1$ geodesic folding (`[-128, 127] -> [-64, 64]`).
    #[inline(always)]
    fn q7_fold_s1_geodesic(&self, phase_q7: gcso_q7_t) -> gcso_q7_t {
        branchless_q7_fold_s1_geodesic(phase_q7)
    }
    /// Branchless Q7 shortest-arc circular interpolation on $S^1$.
    #[inline(always)]
    fn q7_circular_lerp(&self, a: gcso_q7_t, b: gcso_q7_t, t_u7: u8) -> gcso_q7_t {
        branchless_q7_circular_lerp(a, b, t_u7)
    }
    /// Branchless Q7 saturating negation.
    #[inline(always)]
    fn q7_neg_sat(&self, x: gcso_q7_t) -> gcso_q7_t {
        branchless_q7_neg_sat(x)
    }
    /// Branchless Q7 fixed-point scaling.
    #[inline(always)]
    fn q7_scale_u7(&self, val: gcso_q7_t, scale_u7: u8) -> gcso_q7_t {
        branchless_q7_scale_u7(val, scale_u7)
    }
    /// SWAR 4-lane Q7 saturating addition.
    #[inline(always)]
    fn q7_add_sat_x4(&self, a_packed: u32, b_packed: u32) -> u32 {
        swar_q7_add_sat_x4(a_packed, b_packed)
    }
    /// SWAR 4-lane Q7 saturating subtraction.
    #[inline(always)]
    fn q7_sub_sat_x4(&self, a_packed: u32, b_packed: u32) -> u32 {
        swar_q7_sub_sat_x4(a_packed, b_packed)
    }
    /// SWAR 8-lane Q7 saturating addition.
    #[inline(always)]
    fn q7_add_sat_x8(&self, a_packed: u64, b_packed: u64) -> u64 {
        swar_q7_add_sat_x8(a_packed, b_packed)
    }
    /// SWAR 8-lane Q7 saturating subtraction.
    #[inline(always)]
    fn q7_sub_sat_x8(&self, a_packed: u64, b_packed: u64) -> u64 {
        swar_q7_sub_sat_x8(a_packed, b_packed)
    }
    /// SWAR 8-lane Q7 circular phase difference.
    #[inline(always)]
    fn q7_circular_diff_x8(&self, a_packed: u64, b_packed: u64) -> u64 {
        swar_q7_circular_diff_x8(a_packed, b_packed)
    }
    /// SWAR 8-lane Q7 circular phase addition.
    #[inline(always)]
    fn q7_circular_add_x8(&self, a_packed: u64, b_packed: u64) -> u64 {
        swar_q7_circular_add_x8(a_packed, b_packed)
    }
    /// SWAR 8-lane Q7 $S^1$ geodesic folding.
    #[inline(always)]
    fn q7_fold_s1_geodesic_x8(&self, packed: u64) -> u64 {
        swar_q7_fold_s1_geodesic_x8(packed)
    }
    /// SWAR 8-lane Q7 saturating negation.
    #[inline(always)]
    fn q7_neg_sat_x8(&self, packed: u64) -> u64 {
        swar_q7_neg_sat_x8(packed)
    }
    /// SWAR 8-lane Q7 shortest-arc geodesic Slerp.
    #[inline(always)]
    fn q7_geodesic_slerp_x8(&self, a_packed: u64, b_packed: u64, t_u7: u8) -> u64 {
        swar_q7_geodesic_slerp_x8(a_packed, b_packed, t_u7)
    }
    /// SWAR 8-lane Q7 fixed-point gain scaling.
    #[inline(always)]
    fn q7_scale_u7_x8(&self, packed: u64, scale_u7: u8) -> u64 {
        swar_q7_scale_u7_x8(packed, scale_u7)
    }
    /// SWAR 8-lane QDPS step filter.
    #[inline(always)]
    fn q7_qdps_filter_x8(&self, packed: u64, min_step_q7: u8) -> u64 {
        swar_q7_qdps_filter_x8(packed, min_step_q7)
    }
    /// SWAR 8-lane `u8` halving decay.
    #[inline(always)]
    fn u8_halve_decay_x8(&self, packed_u8: u64) -> u64 {
        swar_u8_halve_decay_x8(packed_u8)
    }
    /// SWAR 8-lane `u8` saturating addition.
    #[inline(always)]
    fn u8_add_sat_x8(&self, a: u64, b: u64) -> u64 {
        swar_u8_add_sat_x8(a, b)
    }
    /// SWAR 8-lane `u8` saturating subtraction.
    #[inline(always)]
    fn u8_sub_sat_x8(&self, packed_u8: u64, sub_step: u8) -> u64 {
        swar_u8_sub_sat_x8(packed_u8, sub_step)
    }
    /// SWAR 8-lane `u8` elementwise maximum.
    #[inline(always)]
    fn u8_max_x8(&self, a: u64, b: u64) -> u64 {
        swar_u8_max_x8(a, b)
    }
    /// Scalar Q7 QDPS step filter.
    #[inline(always)]
    fn qdps_filter_q7(&self, delta: gcso_q7_t, min_step_q7: i32) -> gcso_q7_t {
        branchless_qdps_filter_q7(delta, min_step_q7)
    }
    /// SplitMix64 bijective hash mixer.
    #[inline(always)]
    fn splitmix64(&self, x: u64) -> u64 {
        branchless_splitmix64(x)
    }
    /// Computes IEEE 802.3 CRC32 checksum via the 64B-aligned 4KB Slice-by-4 lookup table.
    #[inline(always)]
    fn crc32_ieee(&self, bytes: &[u8]) -> u32 {
        branchless_crc32_ieee(bytes)
    }
}

impl<T: ?Sized> BranchlessKernel for T {}

// ===================================================================
// 2. AVX2 / Scalar SIMD Kernels for 2D/3D Visual RoPE & Rank-1 SRL
// ===================================================================

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_build_so2_2d_xy_vecs(
    sin_x: f32,
    cos_x: f32,
    sin_y: f32,
    cos_y: f32,
) -> (core::arch::x86_64::__m256, core::arch::x86_64::__m256) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`. Builds `[cos_x, cos_x, cos_y, cos_y]^2`
    // and `[sin_x, sin_x, sin_y, sin_y]^2` in 3 shuffle/insert instructions without stack spills.
    unsafe {
        use core::arch::x86_64::{
            _mm256_castps128_ps256, _mm256_insertf128_ps, _mm_set_ss, _mm_unpacklo_ps,
        };
        let c_xy = _mm_unpacklo_ps(_mm_set_ss(cos_x), _mm_set_ss(cos_y));
        let c_128 = _mm_unpacklo_ps(c_xy, c_xy);
        let s_xy = _mm_unpacklo_ps(_mm_set_ss(sin_x), _mm_set_ss(sin_y));
        let s_128 = _mm_unpacklo_ps(s_xy, s_xy);
        (
            _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(c_128), c_128),
            _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(s_128), s_128),
        )
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_rotate_so2_2d_xy_slice_with_vecs(
    buf: &mut [f32],
    v_cos: core::arch::x86_64::__m256,
    v_sin: core::arch::x86_64::__m256,
) {
    let len = buf.len();
    let ptr = buf.as_mut_ptr();
    // SAFETY: Caller guarantees `target_feature = "avx2"`, and all offsets are bounds-checked against `len`.
    // Delegates 32-float blocks to `avx2_rotate_so2_block32` (`2x` paired-YMM ILP across opposite 32B L1D banks).
    unsafe {
        let mut b = 0usize;
        while b + 32 <= len {
            avx2_rotate_so2_block32(ptr.add(b), v_cos, v_sin);
            b += 32;
        }
        let rem = len - b;
        if (rem & 16) != 0 {
            let p0 = ptr.add(b);
            let p1 = ptr.add(b + 8);
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
            b += 16;
        }
        if (rem & 8) != 0 {
            let p = ptr.add(b);
            let v = _mm256_loadu_ps(p);
            _mm256_storeu_ps(
                p,
                _mm256_addsub_ps(
                    _mm256_mul_ps(v, v_cos),
                    _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v), v_sin),
                ),
            );
            b += 8;
        }
        if (rem & 4) != 0 {
            let p = ptr.add(b);
            let v = _mm_loadu_ps(p);
            let c_lo = _mm256_castps256_ps128(v_cos);
            let s_lo = _mm256_castps256_ps128(v_sin);
            _mm_storeu_ps(
                p,
                _mm_addsub_ps(
                    _mm_mul_ps(v, c_lo),
                    _mm_mul_ps(_mm_permute_ps::<0xB1>(v), s_lo),
                ),
            );
        }
    }
}

#[inline(always)]
pub(crate) fn rotate_so2_2d_xy_slice_simd(
    buf: &mut [f32],
    sin_x: f32,
    cos_x: f32,
    sin_y: f32,
    cos_y: f32,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: `target_feature = "avx2"` is enabled.
        unsafe {
            let (v_cos, v_sin) = avx2_build_so2_2d_xy_vecs(sin_x, cos_x, sin_y, cos_y);
            avx2_rotate_so2_2d_xy_slice_with_vecs(buf, v_cos, v_sin);
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for quad in buf.chunks_exact_mut(4) {
            (quad[0], quad[1]) = branchless_rotate_so2_pair(quad[0], quad[1], sin_x, cos_x);
            (quad[2], quad[3]) = branchless_rotate_so2_pair(quad[2], quad[3], sin_y, cos_y);
        }
    }
}

/// Executes 3D spatiotemporal $(\Delta\theta_t, \Delta\theta_x, \Delta\theta_y)$ rotation across `buf`
/// using a 24-float (3-YMM = 4-hexagon) unrolled AVX2 kernel with 12-float (`1 YMM + 1 XMM`) and
/// 6-float (`1 XMM + 1 pair`) vector tail handling.
#[inline(always)]
pub(crate) fn rotate_so2_3d_txy_slice_simd(
    buf: &mut [f32],
    sin_t: f32,
    cos_t: f32,
    sin_x: f32,
    cos_x: f32,
    sin_y: f32,
    cos_y: f32,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let hex_len = (buf.len() / 6) * 6;
        let ptr = buf.as_mut_ptr();
        let mut b = 0usize;
        // SAFETY: `target_feature = "avx2"` is enabled, and `b + 24 / 12 / 6 <= hex_len` are bounds-checked.
        unsafe {
            let c0 = _mm256_setr_ps(cos_t, cos_t, cos_x, cos_x, cos_y, cos_y, cos_t, cos_t);
            let s0 = _mm256_setr_ps(sin_t, sin_t, sin_x, sin_x, sin_y, sin_y, sin_t, sin_t);
            let c1 = _mm256_setr_ps(cos_x, cos_x, cos_y, cos_y, cos_t, cos_t, cos_x, cos_x);
            let s1 = _mm256_setr_ps(sin_x, sin_x, sin_y, sin_y, sin_t, sin_t, sin_x, sin_x);
            let c2 = _mm256_setr_ps(cos_y, cos_y, cos_t, cos_t, cos_x, cos_x, cos_y, cos_y);
            let s2 = _mm256_setr_ps(sin_y, sin_y, sin_t, sin_t, sin_x, sin_x, sin_y, sin_y);

            while b + 24 <= hex_len {
                let p0 = ptr.add(b);
                let p1 = ptr.add(b + 8);
                let p2 = ptr.add(b + 16);
                let v0 = _mm256_loadu_ps(p0);
                let v1 = _mm256_loadu_ps(p1);
                let v2 = _mm256_loadu_ps(p2);
                _mm256_storeu_ps(
                    p0,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v0, c0),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), s0),
                    ),
                );
                _mm256_storeu_ps(
                    p1,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v1, c1),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v1), s1),
                    ),
                );
                _mm256_storeu_ps(
                    p2,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v2, c2),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v2), s2),
                    ),
                );
                b += 24;
            }
            if b + 12 <= hex_len {
                let p0 = ptr.add(b);
                let p1 = ptr.add(b + 8);
                let v0 = _mm256_loadu_ps(p0);
                let v1_lo = _mm_loadu_ps(p1);
                let c1_lo = _mm256_castps256_ps128(c1);
                let s1_lo = _mm256_castps256_ps128(s1);
                _mm256_storeu_ps(
                    p0,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v0, c0),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), s0),
                    ),
                );
                _mm_storeu_ps(
                    p1,
                    _mm_addsub_ps(
                        _mm_mul_ps(v1_lo, c1_lo),
                        _mm_mul_ps(_mm_permute_ps::<0xB1>(v1_lo), s1_lo),
                    ),
                );
                b += 12;
            }
            if b + 6 <= hex_len {
                let p0 = ptr.add(b);
                let v0_lo = _mm_loadu_ps(p0);
                let c0_lo = _mm256_castps256_ps128(c0);
                let s0_lo = _mm256_castps256_ps128(s0);
                _mm_storeu_ps(
                    p0,
                    _mm_addsub_ps(
                        _mm_mul_ps(v0_lo, c0_lo),
                        _mm_mul_ps(_mm_permute_ps::<0xB1>(v0_lo), s0_lo),
                    ),
                );
                let (ry0, ry1) =
                    branchless_rotate_so2_pair(*ptr.add(b + 4), *ptr.add(b + 5), sin_y, cos_y);
                *ptr.add(b + 4) = ry0;
                *ptr.add(b + 5) = ry1;
            }
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for hex in buf.chunks_exact_mut(6) {
            (hex[0], hex[1]) = branchless_rotate_so2_pair(hex[0], hex[1], sin_t, cos_t);
            (hex[2], hex[3]) = branchless_rotate_so2_pair(hex[2], hex[3], sin_x, cos_x);
            (hex[4], hex[5]) = branchless_rotate_so2_pair(hex[4], hex[5], sin_y, cos_y);
        }
    }
}

/// Evaluates SRL Rank-1 outer product write `y_slice[i] = scale * u_slice[i] * g_i` using AVX2.
#[inline(always)]
pub(crate) fn srl_write_outer_product_simd(
    y_slice: &mut [f32],
    u_slice: &[f32],
    g_slice_opt: Option<&[f32]>,
    scale: f32,
) {
    let clean_scale = branchless_flush_subnormal_f32(scale);
    let len = y_slice.len().min(u_slice.len());
    if let Some(g_slice) = g_slice_opt {
        let len_g = len.min(g_slice.len());
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            let py = y_slice.as_mut_ptr();
            let pu = u_slice.as_ptr();
            let pg = g_slice.as_ptr();
            // SAFETY: `target_feature = "avx2"` is enabled, and `i + 32 <= len_g` is bounds-checked.
            unsafe {
                let vs = _mm256_set1_ps(clean_scale);
                let mut i = 0usize;
                while i + 32 <= len_g {
                    let p0 = _mm256_mul_ps(_mm256_loadu_ps(pu.add(i)), _mm256_loadu_ps(pg.add(i)));
                    let p1 = _mm256_mul_ps(
                        _mm256_loadu_ps(pu.add(i + 8)),
                        _mm256_loadu_ps(pg.add(i + 8)),
                    );
                    _mm256_storeu_ps(py.add(i), _mm256_mul_ps(p0, vs));
                    _mm256_storeu_ps(py.add(i + 8), _mm256_mul_ps(p1, vs));

                    let p2 = _mm256_mul_ps(
                        _mm256_loadu_ps(pu.add(i + 16)),
                        _mm256_loadu_ps(pg.add(i + 16)),
                    );
                    let p3 = _mm256_mul_ps(
                        _mm256_loadu_ps(pu.add(i + 24)),
                        _mm256_loadu_ps(pg.add(i + 24)),
                    );
                    _mm256_storeu_ps(py.add(i + 16), _mm256_mul_ps(p2, vs));
                    _mm256_storeu_ps(py.add(i + 24), _mm256_mul_ps(p3, vs));
                    i += 32;
                }
                while i < len_g {
                    *py.add(i) = clean_scale * *pu.add(i) * *pg.add(i);
                    i += 1;
                }
            }
            return;
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            for ((y_i, &u_i), &g_i) in y_slice[..len_g]
                .iter_mut()
                .zip(u_slice[..len_g].iter())
                .zip(g_slice[..len_g].iter())
            {
                *y_i = clean_scale * u_i * g_i;
            }
        }
    } else {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            let py = y_slice.as_mut_ptr();
            let pu = u_slice.as_ptr();
            // SAFETY: `target_feature = "avx2"` is enabled, and `i + 32 <= len` is bounds-checked.
            unsafe {
                let vs = _mm256_set1_ps(clean_scale);
                let mut i = 0usize;
                while i + 32 <= len {
                    let u0 = _mm256_loadu_ps(pu.add(i));
                    let u1 = _mm256_loadu_ps(pu.add(i + 8));
                    _mm256_storeu_ps(py.add(i), _mm256_mul_ps(u0, vs));
                    _mm256_storeu_ps(py.add(i + 8), _mm256_mul_ps(u1, vs));
                    let u2 = _mm256_loadu_ps(pu.add(i + 16));
                    let u3 = _mm256_loadu_ps(pu.add(i + 24));
                    _mm256_storeu_ps(py.add(i + 16), _mm256_mul_ps(u2, vs));
                    _mm256_storeu_ps(py.add(i + 24), _mm256_mul_ps(u3, vs));
                    i += 32;
                }
                while i < len {
                    *py.add(i) = clean_scale * *pu.add(i);
                    i += 1;
                }
            }
            return;
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            for (y_i, &u_i) in y_slice[..len].iter_mut().zip(u_slice[..len].iter()) {
                *y_i = clean_scale * u_i;
            }
        }
    }
}

/// Accumulates SRL Rank-1 outer product `y_slice[i] += scale * u_slice[i] * g_i` using AVX2.
#[inline(always)]
pub(crate) fn srl_accum_outer_product_simd(
    y_slice: &mut [f32],
    u_slice: &[f32],
    g_slice_opt: Option<&[f32]>,
    scale: f32,
) {
    let clean_scale = branchless_flush_subnormal_f32(scale);
    if (clean_scale.to_bits() & 0x7FFF_FFFF) == 0 {
        return;
    }
    let len = y_slice.len().min(u_slice.len());
    if let Some(g_slice) = g_slice_opt {
        let len_g = len.min(g_slice.len());
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            let py = y_slice.as_mut_ptr();
            let pu = u_slice.as_ptr();
            let pg = g_slice.as_ptr();
            // SAFETY: `target_feature = "avx2"` is enabled, and `i + 32 <= len_g` is bounds-checked.
            unsafe {
                let vs = _mm256_set1_ps(clean_scale);
                let mut i = 0usize;
                while i + 32 <= len_g {
                    let d0 = _mm256_mul_ps(
                        _mm256_mul_ps(_mm256_loadu_ps(pu.add(i)), _mm256_loadu_ps(pg.add(i))),
                        vs,
                    );
                    let d1 = _mm256_mul_ps(
                        _mm256_mul_ps(
                            _mm256_loadu_ps(pu.add(i + 8)),
                            _mm256_loadu_ps(pg.add(i + 8)),
                        ),
                        vs,
                    );
                    _mm256_storeu_ps(py.add(i), _mm256_add_ps(_mm256_loadu_ps(py.add(i)), d0));
                    _mm256_storeu_ps(
                        py.add(i + 8),
                        _mm256_add_ps(_mm256_loadu_ps(py.add(i + 8)), d1),
                    );

                    let d2 = _mm256_mul_ps(
                        _mm256_mul_ps(
                            _mm256_loadu_ps(pu.add(i + 16)),
                            _mm256_loadu_ps(pg.add(i + 16)),
                        ),
                        vs,
                    );
                    let d3 = _mm256_mul_ps(
                        _mm256_mul_ps(
                            _mm256_loadu_ps(pu.add(i + 24)),
                            _mm256_loadu_ps(pg.add(i + 24)),
                        ),
                        vs,
                    );
                    _mm256_storeu_ps(
                        py.add(i + 16),
                        _mm256_add_ps(_mm256_loadu_ps(py.add(i + 16)), d2),
                    );
                    _mm256_storeu_ps(
                        py.add(i + 24),
                        _mm256_add_ps(_mm256_loadu_ps(py.add(i + 24)), d3),
                    );
                    i += 32;
                }
                while i < len_g {
                    *py.add(i) += clean_scale * *pu.add(i) * *pg.add(i);
                    i += 1;
                }
            }
            return;
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            for ((y_i, &u_i), &g_i) in y_slice[..len_g]
                .iter_mut()
                .zip(u_slice[..len_g].iter())
                .zip(g_slice[..len_g].iter())
            {
                *y_i += clean_scale * u_i * g_i;
            }
        }
    } else {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            let py = y_slice.as_mut_ptr();
            let pu = u_slice.as_ptr();
            // SAFETY: `target_feature = "avx2"` is enabled, and `i + 32 <= len` is bounds-checked.
            unsafe {
                let vs = _mm256_set1_ps(clean_scale);
                let mut i = 0usize;
                while i + 32 <= len {
                    let d0 = _mm256_mul_ps(_mm256_loadu_ps(pu.add(i)), vs);
                    let d1 = _mm256_mul_ps(_mm256_loadu_ps(pu.add(i + 8)), vs);
                    _mm256_storeu_ps(py.add(i), _mm256_add_ps(_mm256_loadu_ps(py.add(i)), d0));
                    _mm256_storeu_ps(
                        py.add(i + 8),
                        _mm256_add_ps(_mm256_loadu_ps(py.add(i + 8)), d1),
                    );
                    let d2 = _mm256_mul_ps(_mm256_loadu_ps(pu.add(i + 16)), vs);
                    let d3 = _mm256_mul_ps(_mm256_loadu_ps(pu.add(i + 24)), vs);
                    _mm256_storeu_ps(
                        py.add(i + 16),
                        _mm256_add_ps(_mm256_loadu_ps(py.add(i + 16)), d2),
                    );
                    _mm256_storeu_ps(
                        py.add(i + 24),
                        _mm256_add_ps(_mm256_loadu_ps(py.add(i + 24)), d3),
                    );
                    i += 32;
                }
                while i < len {
                    *py.add(i) += clean_scale * *pu.add(i);
                    i += 1;
                }
            }
            return;
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            for (y_i, &u_i) in y_slice[..len].iter_mut().zip(u_slice[..len].iter()) {
                *y_i += clean_scale * u_i;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::primitives::Q7_PHASE_SCALE_RAD;

    struct DummyKernel;

    #[test]
    fn test_branchless_kernel_blanket_inv_pos_crc32_and_constant_time_swar() {
        let k = DummyKernel;
        assert_eq!(k.select_u32(true, 10, 20), 10);
        assert_eq!(k.select_f32(false, 1.5, -2.5), -2.5);
        assert!((k.inv_pos_f32(4.0) - 0.25).abs() < 1.0e-6);
        assert_eq!(k.inv_pos_f32(0.0), 0.0);
        assert_eq!(k.inv_pos_f32(-2.0), 0.0);
        assert_eq!(k.inv_pos_f32(f32::INFINITY), 0.0);

        // Standard IEEE 802.3 CRC32 test vector ("123456789" -> 0xCBF43926)
        assert_eq!(k.crc32_ieee(b"123456789"), 0xCBF4_3926);
        let chained = branchless_update_crc32_ieee(branchless_crc32_ieee(b"1234"), b"56789");
        assert_eq!(chained, 0xCBF4_3926);
        for b_val in [0u8, 1, 28, 127, 128, 255] {
            let unif_bytes = [b_val; 128];
            let w_unif = [u64::from_le_bytes([b_val; 8]); 16];
            assert_eq!(
                branchless_crc32_128b_words(&w_unif),
                branchless_crc32_ieee(&unif_bytes)
            );
        }
        let long_payload = [0x5Au8; 512];
        let crc_full = branchless_crc32_ieee(&long_payload);
        let crc_split = branchless_update_crc32_ieee(
            branchless_crc32_ieee(&long_payload[..256]),
            &long_payload[256..],
        );
        assert_eq!(crc_full, crc_split);

        let a = swar_q7_pack_x8(&[100, -100, 64, -64, 127, -128, 0, 12]);
        let b = swar_q7_pack_x8(&[50, -50, -64, 64, 10, -10, 0, -12]);
        let sum = swar_q7_unpack_x8(k.q7_add_sat_x8(a, b));
        assert_eq!(sum, [127, -128, 0, 0, 127, -128, 0, 0]);

        let slerp_mid = swar_q7_unpack_x8(k.q7_geodesic_slerp_x8(
            swar_q7_pack_x8(&[-120; 8]),
            swar_q7_pack_x8(&[120; 8]),
            64,
        ));
        for lane in slerp_mid {
            assert!(lane.unsigned_abs() >= 120);
        }

        for deg in -180..=180 {
            let rad = (deg as f32) * (PI / 180.0);
            let (s, c) = branchless_sincos_f32(rad);
            let q7_ref = branchless_rad_to_q7(branchless_atan2_f32(s, c));
            let q7_fast = branchless_atan2_to_q7_fast(s, c);
            assert!((i32::from(q7_ref) - i32::from(q7_fast)).abs() <= 1);
        }
    }

    #[test]
    fn test_unified_field_fold_gravity_fold_toroidal_centroid_step_and_limina_log_prob() {
        let trail = [32i8; MAX_ATTENTION_HEADS];
        let anchors = [
            [-96i8; MAX_ATTENTION_HEADS],
            [32i8; MAX_ATTENTION_HEADS],
            [28i8; MAX_ATTENTION_HEADS],
        ];
        let energies = [0.5_f32, 3.0_f32, 2.0_f32];
        let mut folded = [0i8; MAX_ATTENTION_HEADS];

        let (tot_pull, mean_coh, dom_idx, act_cnt) =
            branchless_q7_unified_field_fold_64(&trail, &anchors, &energies, &mut folded);
        assert_eq!(dom_idx, 1);
        assert_eq!(act_cnt, 3);
        assert!(tot_pull > 8.0 && mean_coh > 0.70);
        assert!((i32::from(folded[0]) - 30).abs() <= 4);

        let mut grav_folded = [0i8; MAX_ATTENTION_HEADS];
        let (grav_pull, grav_coh, grav_dom, grav_cnt) =
            branchless_q7_gravity_fold_64(&trail, &anchors, &energies, 0.25, &mut grav_folded);
        assert_eq!(grav_dom, 1);
        assert_eq!(grav_cnt, 3);
        assert!(grav_pull > 15.0 && grav_coh > 0.85);

        let (coh_ab, r_bar, mean_q7, kappa) =
            branchless_q7_coherence_and_resultant_64(&trail, &anchors[1]);
        assert!((coh_ab - 1.0).abs() < 1.0e-4);
        let mut nonunif_a = [0i8; MAX_ATTENTION_HEADS];
        let mut nonunif_b = [71i8; MAX_ATTENTION_HEADS];
        nonunif_a[0] = 1;
        nonunif_b[0] = 72;
        let coh_71 = branchless_q7_coherence_64(&nonunif_a, &nonunif_b);
        assert!((coh_71 - (-0.170_962)).abs() < 2.0e-3);
        assert!((r_bar - 1.0).abs() < 1.0e-4);
        assert_eq!(mean_q7, 32);
        assert!(kappa > 10.0);

        let w_init = load_q7_words_64(&trail);
        let (w_step1, e_step1, rb_step1, kp_step1) =
            branchless_toroidal_centroid_step_words_64(&w_init, 0, 0.0, 0.0, 0.0, &w_init, 2.5);
        assert_eq!(w_step1, w_init);
        assert!(
            (e_step1 - 2.5).abs() < 1.0e-5 && (rb_step1 - 1.0).abs() < 1.0e-4 && kp_step1 > 10.0
        );
        let (w_stat, e_stat, rb_stat, kp_stat) = branchless_toroidal_centroid_step_words_64(
            &w_step1, 1, e_step1, rb_step1, kp_step1, &w_init, 2.0,
        );
        assert_eq!(w_stat, w_init);
        assert!((e_stat - 3.5).abs() < 1.0e-5 && rb_stat == rb_step1 && kp_stat == kp_step1);

        let mut scores = [0.0_f32, 0.0_f32, 0.0_f32];
        let densities = [5.0_f32, 2.0_f32, 1.0_f32];
        let (_mean_c, best_id) = modulate_attention_scores_by_bayesian_trails_64(
            &mut scores,
            &trail,
            &anchors,
            &densities,
            Some(1),
            128,
            2.0,
            0.2,
            0.5,
        );
        assert_eq!(best_id, 1);
        assert!(scores[1] > scores[2] && scores[2] > scores[0]);

        let (coh_same, lp_same) = branchless_q7_transition_log_prob_64(&trail, &anchors[1]);
        assert!((coh_same - 1.0).abs() < 1.0e-4 && lp_same.abs() < 1.0e-4);

        let emb = [0.5_f32; 128];
        let proj = branchless_project_f32_slice_to_q7_64(&emb, 1.0);
        assert_eq!(proj[0], proj[63]);
        assert!(proj[0] > 15);
    }

    #[test]
    fn test_avx2_sincos_x8_3d_txy_lockstep_qkv_and_subnormal_norm_rescue() {
        let angles = [-3.0_f32, -1.5, -0.5, 0.0, 0.5, 1.2, 2.4, 3.1];
        let mut sin_out = [0.0_f32; 8];
        let mut cos_out = [0.0_f32; 8];
        sincos_f32_x8_simd(&angles, &mut sin_out, &mut cos_out);
        for i in 0..8 {
            let (s_ref, c_ref) = branchless_sincos_f32(angles[i]);
            assert!((sin_out[i] - s_ref).abs() < 1.0e-4);
            assert!((cos_out[i] - c_ref).abs() < 1.0e-4);
        }

        let mut grid_3d = [1.0_f32; 48];
        let (st, ct) = branchless_sincos_q7(16);
        let (sx, cx) = branchless_sincos_q7(32);
        let (sy, cy) = branchless_sincos_q7(-16);
        rotate_so2_3d_txy_slice_simd(&mut grid_3d, st, ct, sx, cx, sy, cy);
        for hex in grid_3d.chunks_exact(6) {
            assert!(((hex[0] * hex[0] + hex[1] * hex[1]) - 2.0).abs() < 1.0e-5);
            assert!(((hex[2] * hex[2] + hex[3] * hex[3]) - 2.0).abs() < 1.0e-5);
            assert!(((hex[4] * hex[4] + hex[5] * hex[5]) - 2.0).abs() < 1.0e-5);
        }

        let mut q = [1.0_f32; 64];
        let mut k = [1.0_f32; 64];
        let mut v = [1.0_f32; 64];
        rotate_so2_slice_unrolled16(&mut q, st, ct);
        rotate_so2_slice_unrolled16(&mut k, st, ct);
        rotate_so2_slice_unrolled16(&mut v, st, ct);
        assert!((q[0] - k[0]).abs() < 1.0e-6 && (q[0] - v[0]).abs() < 1.0e-6);

        // Verify constant-time identity rotation preserves exact values
        let mut q_id = [1.5_f32; 64];
        rotate_so2_slice_unrolled16(&mut q_id, 0.0, 1.0);
        assert_eq!(q_id, [1.5_f32; 64]);

        // Verify AVX2 inline subnormal flushing in `scale_f32_slice_unrolled8`
        let mut sub_buf = [2.0e-38_f32; 64];
        scale_f32_slice_unrolled8(&mut sub_buf, 0.5);
        assert_eq!(sub_buf[0], 0.0);

        let tiny = [1.0e-22_f32; 64];
        let (norm, ok) = norm_l2_underflow_resilient_f32(&tiny);
        assert!(ok && (norm - 8.0e-22).abs() < 1.0e-26);

        let sample_vec = [-2.0_f32, 1.0, 3.0, -4.0];
        let (s_abs, s_sq, m_abs) = sum_abs_sq_and_max_f32_simd8(&sample_vec);
        assert!((s_abs - 10.0).abs() < 1.0e-5);
        assert!((s_sq - 30.0).abs() < 1.0e-5);
        assert!((m_abs - 4.0).abs() < 1.0e-5);

        // Verify single-evaluation fused RIPA + PSPM angle helper
        let scale_018 = Q7_PHASE_SCALE_RAD * branchless_inv_pos_f32(0.18);
        let (s1, c1) = eval_ripa_sincos_q7_fast(28, false, 0.18, scale_018);
        let (s2, c2) = eval_ripa_sincos_q7_fast(12, false, 0.18, scale_018);
        let (sf, cf) = eval_fused_ripa_pspm_sincos_q7_fast(
            28, false, false, 0.18, scale_018, 12, false, false, 0.18, scale_018,
        );
        assert!((sf - (s1 * c2 + c1 * s2)).abs() < 1.0e-5);
        assert!((cf - (c1 * c2 - s1 * s2)).abs() < 1.0e-5);

        // Verify 1KB GF(2) CRC32_UNIFORM56_PAD8_ZERO_SEED_REF and kappa CRC32 constants
        assert_eq!(
            CRYSTAL_CRC32_KAPPA_AT_MAX_R_FINAL_CONTRIB,
            crc32_step_l5(crc32_slice16_dep_lo0(VON_MISES_KAPPA_AT_MAX_R_BITS))
        );
        assert_eq!(
            CRYSTAL_CRC32_KAPPA_MAX_FINAL_CONTRIB,
            crc32_step_l5(crc32_slice16_dep_lo0(0x4248_0000u32))
        );
        let w_trail_blk1 = [
            0x1818_1818_1818_1818u64,
            0x1818_1818_1818_1818u64,
            0x1818_1818_1818_1818u64,
            0x1818_1818_1818_1818u64, 0x1818_1818_1818_1818u64, 0x1818_1818_1818_1818u64,
            0x1818_1818_1818_1818u64, 0u64];
        let mut raw_blk1 = [0x18u8; 64];
        raw_blk1[56..64].fill(0);
        assert_eq!(
            crc32_block64_zero_seed_words(&w_trail_blk1),
            CRC32_UNIFORM56_PAD8_ZERO_SEED_REF[0x18]
        );
        assert_eq!(
            crc32_step_block64_words(0xFFFF_FFFFu32, &w_trail_blk1) ^ 0xFFFF_FFFFu32,
            branchless_crc32_ieee(&raw_blk1)
        );

        // Verify AVX2 / scalar RIPA angle clamp parity across finite, extreme, and non-finite inputs
        let mut angles_simd = [
            -12.0_f32,
            -0.5,
            -0.18,
            -1.0e-39,
            0.0,
            0.09,
            0.18,
            0.5,
            12.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        let mut angles_ref = angles_simd;
        let max_rad = 0.18_f32;
        let inv_max = branchless_inv_pos_f32(max_rad);
        for a in angles_ref.iter_mut() {
            let clean = branchless_select_f32(branchless_is_finite_f32(*a), *a, 0.0);
            *a = branchless_flush_subnormal_f32(max_rad * branchless_tanh_f32(clean * inv_max));
        }
        clamp_angles_ripa_in_place_simd8(&mut angles_simd, max_rad);
        for (got, exp) in angles_simd.iter().zip(angles_ref.iter()) {
            assert!((got - exp).abs() <= 1.0e-6, "got={} exp={}", got, exp);
        }
    }
}
