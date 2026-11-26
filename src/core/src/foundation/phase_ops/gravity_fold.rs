// name: src/core/src/foundation/phase_ops/gravity_fold.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! 64-head Unified-Manifold and phase-proximity gravity folding, conditional transition
//! log-probabilities, and phase-interference inverse-temperature scaling kernels.

use core::f32::consts::LN_2;

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
use super::coherence::swar_slerp_words_with_uniform_flag_64;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use super::coherence::{
    avx2_coherence_from_diff_vec64, avx2_slerp_from_diff_vec64, avx2_vec64_to_words64,
    avx2_words64_to_vec64_with_uniform, swar_slerp_word_unchecked_x8,
};
use super::coherence::{
    coherence_from_word_arrays_64, coherence_from_word_arrays_with_both_uniform_flags_64,
    swar_slerp_words_with_bias_64,
};
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use crate::foundation::lut_crc32::Q7_FULL_COS_REF;
use crate::foundation::lut_crc32::{
    MAX_PHASE_INTERFERENCE_INV_TEMP, MIN_PHASE_INTERFERENCE_INV_TEMP, SWAR_SPLAT_BYTE_64,
};
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use crate::foundation::scalar_math::branchless_q7_circular_lerp;
use crate::foundation::scalar_math::{
    branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_is_finite_f32,
    branchless_ln_f32, branchless_phase_proximity_gravity_f32, branchless_relu_f32,
    fast_inv_pos_normal_f32, fast_ln_1p_bounded_f32,
};
use crate::foundation::swar_simd::{is_word_array_uniform_64, load_q7_words_64, store_q7_words_64};
use crate::liminal::{
    DEFAULT_ENTROPY_EPS, MAX_ATTENTION_HEADS, PhaseQ7, branchless_select_f32,
    branchless_select_u32, branchless_select_usize,
};

/// Register-resident Unified-Manifold folder returning `([u64; 8], total_pull, mean_coherence, dominant_idx, active_count)`.
///
/// Includes a zero-division, zero-SLERP single-anchor fast path (`count == 1 && init_w <= 1.0e-12`)
/// and guards 64-head Q14 cosine LUT lookups behind `trail_nonzero & anchor_nonzero`.
/// Order-invariant (commutative) folding of an arbitrary slice of anchors
/// (`anchor_phases`, `anchor_base_energies`) directly in 8 `u64` SWAR / AVX2 registers,
/// returning `(folded_words, total_weight, mean_coherence, dominant_anchor_idx, active_anchor_count, any_resonant, any_liminal)`.
#[inline]
pub(crate) fn branchless_q7_unified_manifold_fold_words_with_flags_64(
    trail_words: &[u64; 8],
    trail_init_weight: f32,
    default_dominant_idx: u32,
    anchor_phases: &[[PhaseQ7; MAX_ATTENTION_HEADS]],
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
        let res_pull = branchless_flush_subnormal_f32(
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
        // SAFETY: `target_feature = "avx2"` is checked by `#[cfg]`, and `t` is a 64-byte (`8 * u64`) stack array.
        unsafe {
            use core::arch::x86_64::{
                __m256i, _mm256_loadu_si256, _mm256_or_si256, _mm256_set1_epi64x, _mm256_sub_epi8,
                _mm256_testz_si256,
            };
            let t_uni = is_word_array_uniform_64(&t);
            let t_w0 = t[0];
            let t_b0 = (t_w0 & 0xFF) as u8;
            let t_is_byte_uniform = t_uni & (t_w0 == t_w0.rotate_right(8));
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
                    let check_b_uni = (running_weight <= DEFAULT_ENTROPY_EPS)
                        | (r_uni & (t_uni | !trail_nonzero));
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
                        } else if t_is_byte_uniform && (bw0 == bw0.rotate_right(8)) {
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
                                if r_w0 == r_w0.rotate_right(8) && bw0 == bw0.rotate_right(8) {
                                    let b_lerp = branchless_q7_circular_lerp(
                                        rb0 as PhaseQ7,
                                        bb0 as PhaseQ7,
                                        s as u8,
                                    );
                                    r_w0 = ((b_lerp as u8) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
                                } else {
                                    let bias_restore =
                                        (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                                    r_w0 = swar_slerp_word_unchecked_x8(r_w0, bw0, s, bias_restore);
                                }
                            }
                        }
                        running_weight = total_w;
                        anchor_pull_sum += branchless_select_f32(is_active, res_pull, 0.0);
                        weighted_coh_sum += branchless_select_f32(is_active, res_pull * coh, 0.0);
                        best_pull = branchless_select_f32(is_better, res_pull, best_pull);
                        dominant_idx = branchless_select_u32(is_better, idx as u32, dominant_idx);
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
                        dominant_idx = branchless_select_u32(is_better, idx as u32, dominant_idx);
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
        }
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

/// Register-resident core for [`branchless_q7_unified_manifold_fold_with_init_weight_64`] that folds
/// `anchor_phases` directly in 8 `u64` SWAR registers (`trail_words`) without intermediate memory
/// stores or reloads, returning `(folded_words_64, total_weight, mean_coherence, dominant_anchor_idx, active_anchor_count)`.
#[inline(always)]
#[must_use]
pub fn branchless_q7_unified_manifold_fold_words_with_init_weight_64(
    trail_words: &[u64; 8],
    trail_init_weight: f32,
    anchor_phases: &[[PhaseQ7; MAX_ATTENTION_HEADS]],
    anchor_base_energies: &[f32],
) -> ([u64; 8], f32, f32, u32, u32) {
    let (r, total_w, mean_coh, dom_idx, act_cnt, _any_resonant, _any_liminal) =
        branchless_q7_unified_manifold_fold_words_with_flags_64(
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
pub fn branchless_q7_unified_manifold_fold_with_init_weight_64(
    trail_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    trail_init_weight: f32,
    anchor_phases: &[[PhaseQ7; MAX_ATTENTION_HEADS]],
    anchor_base_energies: &[f32],
    folded_phases_out: &mut [PhaseQ7; MAX_ATTENTION_HEADS],
) -> (f32, f32, u32, u32) {
    let t = load_q7_words_64(trail_phases);
    let (r, total_pull, mean_coh, dom_idx, act_cnt) =
        branchless_q7_unified_manifold_fold_words_with_init_weight_64(
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
pub fn branchless_q7_unified_manifold_fold_64(
    trail_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    anchor_phases: &[[PhaseQ7; MAX_ATTENTION_HEADS]],
    anchor_base_energies: &[f32],
    folded_phases_out: &mut [PhaseQ7; MAX_ATTENTION_HEADS],
) -> (f32, f32, u32, u32) {
    branchless_q7_unified_manifold_fold_with_init_weight_64(
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
    query_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    anchor_phases: &[[PhaseQ7; MAX_ATTENTION_HEADS]],
    anchor_gravity_masses: &[f32],
    softening_eps: f32,
    folded_phases_out: &mut [PhaseQ7; MAX_ATTENTION_HEADS],
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
        unsafe {
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
        }
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
                let need_b_uni = (q_nonzero & q_is_uniform)
                    | (running_pull <= DEFAULT_ENTROPY_EPS)
                    | r_is_uniform;
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
    prev_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    curr_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
) -> (f32, f32) {
    branchless_q7_transition_log_prob_words_64(
        &load_q7_words_64(prev_phases),
        &load_q7_words_64(curr_phases),
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
