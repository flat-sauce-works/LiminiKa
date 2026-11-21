// name: src/core/src/foundation/phase_ops/toroidal_centroid.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! 64-head online toroidal centroid ($k$-Means M-step), synergy/inhibition fusion,
//! AVX2 slice projection, and P-Frame motion-delta chain seek kernels.

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::*;
use core::f32::consts::PI;

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
use super::coherence::swar_slerp_word_unchecked_x8;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use super::coherence::{
    avx2_coherence_from_diff_vec64, avx2_mean_sincos_from_vec64, avx2_scale_vec64,
    avx2_slerp_from_diff_vec64, avx2_u64x8_to_vec64, avx2_vec64_to_words64, avx2_words64_to_vec64,
    avx2_words64_to_vec64_with_uniform, sum_sincos_biased_u64_from_word,
};
use super::coherence::{
    circular_r_bar_from_words_with_uniform_flag_64, coherence_from_diff_words_64,
    coherence_from_uniform_diff_word_x8, coherence_from_word_arrays_with_uniform_flag_64,
};
use super::gravity_fold::branchless_fast_transition_log_prob;
use crate::foundation::lut_crc32::SWAR_SPLAT_BYTE_64;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use crate::foundation::lut_crc32::{INV_8_Q14_F32, Q14_BIAS_X8_I32};
use crate::foundation::pod::KvMotionDeltaPacket;
use crate::foundation::scalar_math::{
    branchless_abs_f32, branchless_clamp_f32, branchless_flush_subnormal_f32,
    branchless_is_finite_f32, branchless_rad_to_q7, branchless_relu_f32,
    branchless_von_mises_kappa, fast_inv_pos_normal_f32,
};
use crate::foundation::swar_simd::{
    is_word_array_uniform_64, load_q7_words_64, store_q7_words_64, swar_q7_circular_add_x8,
    swar_q7_circular_diff_x8, swar_q7_fold_s1_geodesic_x8, swar_q7_neg_sat_x8,
    swar_q7_phase_step_filter_words_64, swar_q7_scale_le128_with_bias_x8,
};
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use crate::liminal::Q7_PHASE_INV_SCALE_RAD;
use crate::liminal::{
    branchless_select_f32, branchless_select_u8, gcso_q7_t, GcsoResult, DEFAULT_ENTROPY_EPS,
    GCSO_ERROR_INVALID_ARGUMENT, MAX_ATTENTION_HEADS,
};

/// Executes an **Online Toroidal $k$-Means M-Step Centroid Update** in 8 register-resident `u64`
/// SWAR words (`centroid_words`, `sample_words`) with an optional precomputed E-step phase
/// coherence `precomputed_coh_opt` and pre-evaluated word-uniformity flags `(c_uni, sample_uni)`,
/// in strictly constant time (`0` data-dependent branches on phase stationarity, normalized
/// directional cosine weighting $w_{\mathrm{coh}} = \operatorname{clamp}(0.5(1 + \mathcal{C}_{\mathrm{phase}}), 0.25, 1.0)$).
///
/// Returns `(updated_centroid_words, next_base_energy, resultant_r_bar, von_mises_kappa, words_changed)`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_toroidal_centroid_words_cached_64(
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
        return (
            *centroid_words,
            prev_base_energy,
            prev_r_bar,
            prev_kappa,
            false,
        );
    }
    let both_uni = c_uni & sample_uni;
    let same_words = (sample_words[0] == centroid_words[0])
        && (both_uni
            || ((((sample_words[1] ^ centroid_words[1]) | (sample_words[2] ^ centroid_words[2]))
                | ((sample_words[3] ^ centroid_words[3])
                    | (sample_words[4] ^ centroid_words[4]))
                | ((sample_words[5] ^ centroid_words[5])
                    | (sample_words[6] ^ centroid_words[6]))
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
        return (
            *centroid_words,
            next_e,
            rb,
            branchless_von_mises_kappa(rb),
            false,
        );
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
            return (
                *centroid_words,
                next_e,
                rb,
                branchless_von_mises_kappa(rb),
                false,
            );
        }
        let tot_e = prev_e + eff_sample_e;
        let safe_tot = branchless_select_f32(tot_e > DEFAULT_ENTROPY_EPS, tot_e, 1.0);
        let ratio =
            branchless_clamp_f32(eff_sample_e * fast_inv_pos_normal_f32(safe_tot), 0.05, 0.50);
        let raw_s = (ratio * 128.0 + 0.5) as u8;
        let s = branchless_select_u8(raw_s > 128, 128, raw_s) as u64;
        let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let step0 = swar_q7_scale_le128_with_bias_x8(d0, s, bias_restore);
        if step0 == 0 {
            if prev_r_bar > 0.0 && prev_kappa > 0.0 {
                return (*centroid_words, next_e, prev_r_bar, prev_kappa, false);
            }
            let rb = circular_r_bar_from_words_with_uniform_flag_64(centroid_words, c_uni);
            return (
                *centroid_words,
                next_e,
                rb,
                branchless_von_mises_kappa(rb),
                false,
            );
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
        if step0 == step0.rotate_right(8) && prev_r_bar > 0.0 && prev_kappa > 0.0 {
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
                _mm256_add_epi8, _mm256_and_si256, _mm256_broadcastb_epi8, _mm256_castsi256_si128,
                _mm256_mulhrs_epi16, _mm256_or_si256, _mm256_permute4x64_epi64, _mm256_set1_epi16,
                _mm256_slli_epi16, _mm256_sub_epi8, _mm256_testz_si256, _mm256_xor_si256,
                _mm_cvtsi128_si64,
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
            let ratio =
                branchless_clamp_f32(eff_sample_e * fast_inv_pos_normal_f32(safe_tot), 0.05, 0.50);
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
                return (
                    *centroid_words,
                    next_e,
                    rb,
                    branchless_von_mises_kappa(rb),
                    false,
                );
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
        let ratio =
            branchless_clamp_f32(eff_sample_e * fast_inv_pos_normal_f32(safe_tot), 0.05, 0.50);
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
        (
            next_w,
            next_e,
            rb,
            branchless_von_mises_kappa(rb),
            !stationary,
        )
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
    let (next_w, next_e, rb, kappa, _changed) = step_toroidal_centroid_words_cached_64(
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
    let c_uni = is_word_array_uniform_64(centroid_words);
    let sample_uni = is_word_array_uniform_64(sample_words);
    branchless_toroidal_centroid_step_words_with_coh_and_uni_64(
        centroid_words,
        prev_sample_count,
        prev_base_energy,
        prev_r_bar,
        prev_kappa,
        sample_words,
        None,
        sample_energy,
        c_uni,
        sample_uni,
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
        swar_q7_circular_add_x8(a, swar_q7_scale_le128_with_bias_x8(d, ratio_b, bias_ratio)),
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
    let clean_gain =
        branchless_select_f32(branchless_is_finite_f32(gain) & (gain >= 0.0), gain, 1.0);
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
                let rounded =
                    _mm256_add_ps(scaled, _mm256_or_ps(_mm256_set1_ps(0.5_f32), sign_bit));
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

// ===================================================================
// Shared P-Frame Motion-Delta & Zero-Forward Seek Kernels
// ===================================================================

/// Encodes a 64-byte [`KvMotionDeltaPacket`] (P-Frame Predictive Phase-Motion Delta) between
/// `prev_phases` and `curr_phases` using 8-lane SWAR circular difference, $S^1$ geodesic folding,
/// and step filtering (`PhaseStepFilter`).
///
/// # Errors
/// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `phase_step_min_q7 > 127`.
#[inline]
pub fn encode_pframe_motion_delta_64(
    prev_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    curr_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    phase_step_min_q7: u8,
) -> GcsoResult<KvMotionDeltaPacket> {
    if phase_step_min_q7 > 127 {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    let pw = load_q7_words_64(prev_phases);
    let cw = load_q7_words_64(curr_phases);
    let diff_folded = [
        swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(cw[0], pw[0])),
        swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(cw[1], pw[1])),
        swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(cw[2], pw[2])),
        swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(cw[3], pw[3])),
        swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(cw[4], pw[4])),
        swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(cw[5], pw[5])),
        swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(cw[6], pw[6])),
        swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(cw[7], pw[7])),
    ];
    let filt_w = swar_q7_phase_step_filter_words_64(&diff_folded, phase_step_min_q7);
    let mut motion_deltas = [0i8; MAX_ATTENTION_HEADS];
    store_q7_words_64(&mut motion_deltas, &filt_w);
    Ok(KvMotionDeltaPacket::new(motion_deltas))
}

/// Applies a single 64-byte [`KvMotionDeltaPacket`] onto `cur_w` during Zero-Forward Seek,
/// negating the delta when transition coherence drops below `-safe_thresh` (contradiction guard).
///
/// Returns `(step_log_prob, repulsed)`.
#[inline(always)]
#[must_use]
pub fn apply_pframe_motion_step_words_64(
    cur_w: &mut [u64; 8],
    pframe: &KvMotionDeltaPacket,
    safe_thresh: f32,
) -> (f32, bool) {
    let dw = load_q7_words_64(&pframe.motion_deltas);
    let coh = coherence_from_diff_words_64(dw[0], dw[1], dw[2], dw[3], dw[4], dw[5], dw[6], dw[7]);
    let log_p = branchless_fast_transition_log_prob(coh);
    let repulsed = coh < -safe_thresh;
    if repulsed {
        *cur_w = [
            swar_q7_circular_add_x8(cur_w[0], swar_q7_neg_sat_x8(dw[0])),
            swar_q7_circular_add_x8(cur_w[1], swar_q7_neg_sat_x8(dw[1])),
            swar_q7_circular_add_x8(cur_w[2], swar_q7_neg_sat_x8(dw[2])),
            swar_q7_circular_add_x8(cur_w[3], swar_q7_neg_sat_x8(dw[3])),
            swar_q7_circular_add_x8(cur_w[4], swar_q7_neg_sat_x8(dw[4])),
            swar_q7_circular_add_x8(cur_w[5], swar_q7_neg_sat_x8(dw[5])),
            swar_q7_circular_add_x8(cur_w[6], swar_q7_neg_sat_x8(dw[6])),
            swar_q7_circular_add_x8(cur_w[7], swar_q7_neg_sat_x8(dw[7])),
        ];
    } else {
        *cur_w = [
            swar_q7_circular_add_x8(cur_w[0], dw[0]),
            swar_q7_circular_add_x8(cur_w[1], dw[1]),
            swar_q7_circular_add_x8(cur_w[2], dw[2]),
            swar_q7_circular_add_x8(cur_w[3], dw[3]),
            swar_q7_circular_add_x8(cur_w[4], dw[4]),
            swar_q7_circular_add_x8(cur_w[5], dw[5]),
            swar_q7_circular_add_x8(cur_w[6], dw[6]),
            swar_q7_circular_add_x8(cur_w[7], dw[7]),
        ];
    }
    (log_p, repulsed)
}

/// Applies $S^1$ geodesic folding and step filtering (`PhaseStepFilter`) to an accumulated 64-head phase word array.
#[inline(always)]
#[must_use]
pub fn finalize_pframe_chain_words_64(cur_w: &[u64; 8], phase_step_min_q7: u8) -> [u64; 8] {
    let folded_w = [
        swar_q7_fold_s1_geodesic_x8(cur_w[0]),
        swar_q7_fold_s1_geodesic_x8(cur_w[1]),
        swar_q7_fold_s1_geodesic_x8(cur_w[2]),
        swar_q7_fold_s1_geodesic_x8(cur_w[3]),
        swar_q7_fold_s1_geodesic_x8(cur_w[4]),
        swar_q7_fold_s1_geodesic_x8(cur_w[5]),
        swar_q7_fold_s1_geodesic_x8(cur_w[6]),
        swar_q7_fold_s1_geodesic_x8(cur_w[7]),
    ];
    swar_q7_phase_step_filter_words_64(&folded_w, phase_step_min_q7)
}

/// Reconstructs a 64-head phase state by replaying `pframes[..seek_steps]` onto `anchor_phases`
/// without any LLM forward pass (Zero-Forward Seek).
///
/// Returns `(reconstructed_words, joint_log_prob, repulsed_count)`.
///
/// # Errors
/// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `seek_steps > pframes.len()`,
/// `phase_step_min_q7 > 127`, or `contradiction_threshold` is non-finite.
#[inline]
pub fn seek_pframe_chain_words_64(
    anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    pframes: &[KvMotionDeltaPacket],
    seek_steps: usize,
    phase_step_min_q7: u8,
    contradiction_threshold: f32,
) -> GcsoResult<([u64; 8], f32, u32)> {
    if seek_steps > pframes.len()
        || phase_step_min_q7 > 127
        || !branchless_is_finite_f32(contradiction_threshold)
    {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    let safe_thresh = branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
    let mut cur_w = load_q7_words_64(anchor_phases);
    let mut joint_log_prob = 0.0_f32;
    let mut repulsed_count = 0u32;
    for pframe in &pframes[..seek_steps] {
        let (log_p, repulsed) = apply_pframe_motion_step_words_64(&mut cur_w, pframe, safe_thresh);
        joint_log_prob += log_p;
        if repulsed {
            repulsed_count = repulsed_count.wrapping_add(1);
        }
    }
    let filt_w = finalize_pframe_chain_words_64(&cur_w, phase_step_min_q7);
    Ok((
        filt_w,
        branchless_flush_subnormal_f32(joint_log_prob),
        repulsed_count,
    ))
}
