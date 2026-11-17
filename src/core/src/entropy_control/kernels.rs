// name: src/core/src/entropy_control/kernels.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Constant-time branchless EDBC entropy, Pitchfork bifurcation, and register-pipelined stepping helpers.

use super::*;
#[allow(unused_imports)]
use super::{closed_loop::*, controller::*};

// ===================================================================
// 1. Standalone Constant-Time (`0 branches`, `0 fdiv`, `0 ln`) Entropy & Pitchfork Functions
// ===================================================================

/// Computes normalized Sparse Surrogate Attention Entropy from Top-2 attention probabilities
/// in `#![no_std]` **branchless mode (`0` data-dependent branches, `0 fdiv`, `0 ln`)**,
/// bounded to $[0, \ln 2]$ (Section 3.1).
///
/// # Mathematical Formulation (9-Term Even Taylor-Estrin Series with Exact Endpoint Closure)
/// Let $\delta = \frac{|p_{\max} - p_{\mathrm{2nd}}|}{p_{\max} + p_{\mathrm{2nd}}} \in [0, 1]$ and
/// $t = \delta^2 \in [0, 1]$. By expanding $H(\delta) = \ln 2 - \sum_{k=1}^{\infty} \frac{t^k}{2k(2k-1)}$
/// into a 9-term binary-tree Estrin polynomial where the 9th coefficient absorbs the residual tail
/// $R_9 = \ln 2 - \sum_{k=1}^{8} \frac{1}{2k(2k-1)}$:
/// - At $t = 0$ ($p_{\max} = p_{\mathrm{2nd}}$): $H(0) = \ln 2$ (`0 ULP` error).
/// - At $t = 1$ ($p_{\mathrm{2nd}} \le \varepsilon$): $H(1) = 0.0$ (`0 ULP` error).
/// - Across all $t \in [0, 1]$: executes a uniform instruction stream using `eps` (defaulting
///   branchlessly to `1.0e-30` if `eps` is non-finite or below `F32_MIN_POSITIVE_NORMAL`) as the
///   lower positivity floor for $p_{\mathrm{2nd}}$.
#[inline(always)]
pub const fn compute_sparse_surrogate_entropy(p_max: f32, p_2nd: f32, eps: f32) -> f32 {
    const C1: f32 = 0.5;
    const C2: f32 = 1.0 / 12.0;
    const C3: f32 = 1.0 / 30.0;
    const C4: f32 = 1.0 / 56.0;
    const C5: f32 = 1.0 / 90.0;
    const C6: f32 = 1.0 / 132.0;
    const C7: f32 = 1.0 / 182.0;
    const C8: f32 = 1.0 / 240.0;
    const C9_TAIL: f32 = MAX_SURROGATE_ENTROPY_LN2 - (C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8);

    let safe_eps = branchless_select_f32(
        branchless_is_finite_f32(eps) & (eps >= F32_MIN_POSITIVE_NORMAL),
        eps,
        1.0e-30,
    );
    let raw_sum = p_max + p_2nd;
    let raw_diff = p_max - p_2nd;
    let valid_sum = (p_max > 0.0) & (p_2nd > safe_eps) & (raw_sum < f32::INFINITY);
    let safe_sum = branchless_select_f32(valid_sum, raw_sum, 1.0);
    let inv_sum = fast_inv_pos_normal_f32(safe_sum);
    let signed_delta = raw_diff * inv_sum;
    let t = branchless_min_f32(signed_delta * signed_delta, 1.0);
    let t2 = t * t;
    let t4 = t2 * t2;
    let p01 = C1 + t * C2;
    let p23 = C3 + t * C4;
    let p45 = C5 + t * C6;
    let p67 = C7 + t * C8;
    let q03 = p01 + t2 * p23;
    let q47 = p45 + t2 * p67;
    let poly = (q03 + t4 * q47) + (t4 * t4) * C9_TAIL;
    let raw_h = MAX_SURROGATE_ENTROPY_LN2 - t * poly;
    branchless_select_f32(valid_sum & (t < 0.999_999_9), raw_h, 0.0)
}

/// Computes Moving Z-Score Normalized Attention Entropy ($\tilde{H}$) incorporating
/// surrogate entropy and inter-layer activation cosine distance variance in **strictly
/// constant-time mode (`0` branches, `0 fdiv`)** via `fast_inv_pos_normal_f32` and
/// [`tanh_lut_eval_nonneg`] (Eq. 3.1).
///
/// The inter-layer cosine modulation factor $1 + \gamma_{\mathrm{mod}} \tanh(\alpha_z \operatorname{ReLU}(z_{\mathrm{dist}}))$
/// is evaluated with zero data dependency on `h_surrogate`, allowing out-of-order CPU execution units
/// to overlap `fast_inv_pos_normal_f32(eff_sigma)` and `tanh_lut_eval_nonneg` in parallel with
/// [`compute_sparse_surrogate_entropy`].
#[inline(always)]
pub const fn compute_moving_z_score_entropy(
    h_surrogate: f32,
    cosine_dist: f32,
    mu_dist: f32,
    sigma_dist: f32,
    sigma_min: f32,
    alpha_z: f32,
    gamma_mod: f32,
) -> f32 {
    let valid_cos = branchless_is_finite_f32(cosine_dist) & branchless_is_finite_f32(mu_dist);
    let raw_diff = cosine_dist - mu_dist;
    let pos_diff = branchless_select_f32(
        valid_cos & (raw_diff > 0.0) & (raw_diff < f32::INFINITY),
        raw_diff,
        0.0,
    );

    let alpha_safe = branchless_select_f32(
        branchless_is_finite_f32(alpha_z) & (alpha_z >= 0.0),
        alpha_z,
        1.0,
    );
    let gamma_safe = branchless_select_f32(
        branchless_is_finite_f32(gamma_mod) & (gamma_mod >= 0.0),
        gamma_mod,
        0.5,
    );
    let min_floor = branchless_select_f32(
        (sigma_min >= F32_MIN_POSITIVE_NORMAL) & branchless_is_finite_f32(sigma_min),
        sigma_min,
        1.0e-6,
    );
    let eff_sigma = branchless_max_f32(
        branchless_select_f32(branchless_is_finite_f32(sigma_dist), sigma_dist, min_floor),
        min_floor,
    );

    let inv_sigma = fast_inv_pos_normal_f32(eff_sigma);
    let z_dist = pos_diff * inv_sigma;
    let relu_z = branchless_min_f32(z_dist, MAX_VALID_TOKEN_Z_SCORE);
    let d_cosine_hat = tanh_lut_eval_nonneg(alpha_safe * relu_z);
    let mod_factor = 1.0 + gamma_safe * d_cosine_hat;

    let valid_h = (h_surrogate > 0.0) & branchless_is_finite_f32(h_surrogate);
    let h_tilde = branchless_flush_subnormal_f32(h_surrogate * mod_factor);
    branchless_select_f32(
        valid_h & valid_cos & branchless_is_finite_f32(h_tilde),
        h_tilde,
        0.0,
    )
}

/// Evaluates scale-invariant Pitchfork Bifurcation potential barrier height
/// $\Delta V = \frac{\alpha^2 (\max(0, \mathcal{C}_{\mathrm{void}} - \tau_{\mathrm{eff}}))^2}{4 \beta d_{\mathrm{head}}}$
/// in division-free (`0 fdiv`) mode (Lemma 3.1).
///
/// # Errors
/// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if any parameter is `NaN`/`Inf`, `beta <= 0.0`,
/// or `head_dim` is zero or odd.
#[inline(always)]
pub const fn eval_barrier_pitchfork(
    liminal_void: f32,
    tau_eff: f32,
    alpha: f32,
    beta: f32,
    head_dim: u32,
) -> GcsoResult<f32> {
    let valid_params = branchless_is_finite_f32(liminal_void)
        & branchless_is_finite_f32(tau_eff)
        & branchless_is_finite_f32(alpha)
        & branchless_is_finite_f32(beta)
        & (beta > 0.0)
        & (head_dim != 0)
        & ((head_dim & 1) == 0);
    if !valid_params {
        return Err(GCSO_ERROR_EDBC_SINGULARITY);
    }

    let diff = branchless_relu_f32(liminal_void - tau_eff);
    let prod = alpha * diff;
    let num = prod * prod;
    let den = 4.0 * beta * (head_dim as f32);
    let safe_den = branchless_max_f32(den, F32_MIN_POSITIVE_NORMAL);
    let inv_den = fast_inv_pos_normal_f32(safe_den);
    let barrier = branchless_flush_subnormal_f32(num * inv_den);
    if !branchless_is_finite_f32(barrier) {
        return Err(GCSO_ERROR_EDBC_SINGULARITY);
    }
    Ok(barrier)
}

/// Evaluates the energy-guided dynamic sampling temperature for an explicitly resolved `mode`
/// (`0` = Normal, `1` = Pitchfork Bifurcation, `2` = Repulsion) in **strictly branchless constant
/// time (`0` branches)** across all 3 modes.
#[inline(always)]
#[must_use]
pub(super) const fn eval_temperature_for_mode(
    mode: u32,
    base_temperature: f32,
    moving_z_entropy: f32,
    void_score: f32,
    tau_eff: f32,
    alpha: f32,
    beta: f32,
    d0: f32,
) -> f32 {
    let safe_base = branchless_clamp_f32(
        branchless_select_f32(
            branchless_is_finite_f32(base_temperature) & (base_temperature > 0.0),
            base_temperature,
            DEFAULT_SAMPLE_TEMPERATURE,
        ),
        0.05,
        5.0,
    );
    if mode == EDBC_MODE_NORMAL_ENTROPY {
        return safe_base;
    }
    let norm_h = branchless_clamp_f32(branchless_relu_f32(moving_z_entropy) * INV_LN_2, 0.0, 1.0);
    let candidate = if mode == EDBC_MODE_PITCHFORK_BIFURCATION {
        let escape_p = branchless_eyring_kramers_escape(void_score, tau_eff, alpha, beta, d0);
        safe_base * (1.0 + 0.35 * escape_p * norm_h)
    } else {
        safe_base * (1.0 + 0.25 * norm_h)
    };
    branchless_clamp_f32(candidate, 0.05, 5.0)
}

// ===================================================================
// 2. Internal Register-Chained Single-Pass Helpers
// ===================================================================

/// Single-pass register-chained evaluator that simultaneously bridges all consecutive inter-memory
/// **Limina** hops (`trail_chain[0..len]`), records self-contradictory hops into `bloom_filter`,
/// and accumulates the post-bridge **Joint Probability Chain Rule Log-Likelihood**
/// $\sum_{t=1}^{T-1} \ln P(M_t \mid M_{t-1})$ and **$S^1$ Sheaf Holonomy Obstruction Energy**
/// $\mathcal{E}_{\mathrm{holonomy}} \in [0, 1]$ without data-dependent branch mispredictions.
#[inline]
pub(super) fn bridge_and_eval_liminal_trail_chain_single_pass(
    trail_chain: &mut [gcso_pointer_trail_t],
    interpolation_u7: u8,
    contradiction_threshold: f32,
    obstruction_threshold: f32,
    bloom_filter: &mut ContrastivePhaseBloomFilter,
) -> GcsoResult<(f32, f32, bool, u32, [u64; 8])> {
    let len = trail_chain.len();
    if len == 0
        || !branchless_is_finite_f32(contradiction_threshold)
        || !branchless_is_finite_f32(obstruction_threshold)
    {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    let first_w = *trail_phase_words_ref(&trail_chain[0]);
    if len == 1 {
        return Ok((0.0, 0.0, false, 0, first_w));
    }

    let safe_contradiction =
        branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
    let t_s = branchless_select_u8(interpolation_u7 > 128, 128, interpolation_u7) as u64;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Guarded by `target_feature = "avx2"`. Keeps `(v_first0, v_first1)` and
        // `(v_prev0, v_prev1)` 100% resident in YMM registers across all Limina hops `1..len`,
        // eliminating all per-hop GPR-to-YMM (`vpinsrq`) and scalar 8-lane SWAR scale/diff loops.
        return unsafe {
            use core::arch::x86_64::{
                _mm256_add_epi8, _mm256_and_si256, _mm256_broadcastb_epi8, _mm256_castsi256_si128,
                _mm256_max_epi8, _mm256_mulhrs_epi16, _mm256_or_si256, _mm256_set1_epi16,
                _mm256_set1_epi8, _mm256_setzero_si256, _mm256_slli_epi16, _mm256_sub_epi8,
                _mm256_testz_si256, _mm256_xor_si256, _mm_cvtsi128_si32,
            };
            // Load `(v_first0, v_first1)` via 64-bit word moves (`avx2_words64_to_vec64`) rather than
            // `_mm256_loadu_si256` at byte offset 56 (which straddles the 64B cache-line boundary
            // and triggers a 4-to-1 Store-to-Load Forwarding stall after 64-bit GPR trail resets).
            let (v_first0, v_first1) = crate::traits::branchless::avx2_words64_to_vec64(&first_w);
            let mut v_prev0 = v_first0;
            let mut v_prev1 = v_first1;
            let mut last_out_w = first_w;
            let mut prev_cur_ptr = trail_chain[0].current_ptr;
            let mut prev_step_count = trail_chain[0].step_count;
            let mut prev_target_anchor_id = trail_chain[0].target_anchor_id;
            let mut prev_pull_force = trail_chain[0].attractor_pull_force;
            let mut joint_log_prob = 0.0_f32;
            let mut coboundary_defect_sum = 0.0_f32;
            let mut repulsed_count = 0u32;

            let v_s = _mm256_set1_epi16(t_s as i16);
            let mask_ff00 = _mm256_set1_epi16(-256i16);
            let mask_00ff = _mm256_set1_epi16(0x00FFi16);
            let zero = _mm256_setzero_si256();
            let min_m127 = _mm256_set1_epi8(-127);

            let mut idx = 1usize;
            while idx < len {
                let curr_trail = &mut trail_chain[idx];
                let s_words = *trail_phase_words_ref(curr_trail);
                let (vs0, vs1) = crate::traits::branchless::avx2_words64_to_vec64(&s_words);
                let vd0 = _mm256_sub_epi8(vs0, v_prev0);
                let vd1 = _mm256_sub_epi8(vs1, v_prev1);
                let vd_or = _mm256_or_si256(vd0, vd1);
                let (pre_coh, contradiction, v_out0, v_out1, post_coh, store_out) =
                    if _mm256_testz_si256(vd_or, vd_or) != 0 {
                        (1.0_f32, false, vs0, vs1, 1.0_f32, false)
                    } else {
                        let lo128 = _mm256_castsi256_si128(vd0);
                        let splat_b0 = _mm256_broadcastb_epi8(lo128);
                        let diff_b0 = _mm256_or_si256(
                            _mm256_xor_si256(vd0, splat_b0),
                            _mm256_xor_si256(vd1, splat_b0),
                        );
                        let diff_is_byte_uni = _mm256_testz_si256(diff_b0, diff_b0) != 0;
                        let b0_u8 = (_mm_cvtsi128_si32(lo128) & 0xFF) as u8;
                        let pre_c = if diff_is_byte_uni {
                            Q7_FULL_COS_REF[b0_u8 as usize]
                        } else {
                            crate::traits::branchless::avx2_coherence_from_diff_vec64(vd0, vd1)
                        };
                        let contr = pre_c < -safe_contradiction;
                        if contr {
                            let nw0 = _mm256_sub_epi8(zero, _mm256_max_epi8(vs0, min_m127));
                            let nw1 = _mm256_sub_epi8(zero, _mm256_max_epi8(vs1, min_m127));
                            let pc = crate::traits::branchless::avx2_coherence_from_diff_vec64(
                                _mm256_sub_epi8(nw0, v_prev0),
                                _mm256_sub_epi8(nw1, v_prev1),
                            );
                            (pre_c, true, nw0, nw1, pc, true)
                        } else if t_s >= 128 {
                            (pre_c, false, vs0, vs1, pre_c, false)
                        } else if t_s == 0 {
                            (pre_c, false, v_prev0, v_prev1, 1.0_f32, true)
                        } else if diff_is_byte_uni {
                            let st_i8 = ((((b0_u8 as i8) as i32) * (t_s as i32) + 64) >> 7) as i8;
                            if st_i8 == 0 {
                                (pre_c, false, v_prev0, v_prev1, 1.0_f32, true)
                            } else {
                                let vst = _mm256_set1_epi8(st_i8);
                                let nw0 = _mm256_add_epi8(v_prev0, vst);
                                let nw1 = _mm256_add_epi8(v_prev1, vst);
                                let pc = Q7_FULL_COS_REF[(st_i8 as u8) as usize];
                                (pre_c, false, nw0, nw1, pc, true)
                            }
                        } else {
                            let ev0 = _mm256_mulhrs_epi16(_mm256_slli_epi16(vd0, 8), v_s);
                            let od0 = _mm256_mulhrs_epi16(_mm256_and_si256(vd0, mask_ff00), v_s);
                            let vst0 = _mm256_or_si256(
                                _mm256_and_si256(ev0, mask_00ff),
                                _mm256_slli_epi16(od0, 8),
                            );
                            let ev1 = _mm256_mulhrs_epi16(_mm256_slli_epi16(vd1, 8), v_s);
                            let od1 = _mm256_mulhrs_epi16(_mm256_and_si256(vd1, mask_ff00), v_s);
                            let vst1 = _mm256_or_si256(
                                _mm256_and_si256(ev1, mask_00ff),
                                _mm256_slli_epi16(od1, 8),
                            );
                            let vst_or = _mm256_or_si256(vst0, vst1);
                            if _mm256_testz_si256(vst_or, vst_or) != 0 {
                                (pre_c, false, v_prev0, v_prev1, 1.0_f32, true)
                            } else {
                                let nw0 = _mm256_add_epi8(v_prev0, vst0);
                                let nw1 = _mm256_add_epi8(v_prev1, vst1);
                                let pc = crate::traits::branchless::avx2_coherence_from_diff_vec64(
                                    vst0, vst1,
                                );
                                (pre_c, false, nw0, nw1, pc, true)
                            }
                        }
                    };
                let mut stored_ow = s_words;
                if store_out {
                    stored_ow = crate::traits::branchless::avx2_vec64_to_words64(v_out0, v_out1);
                    *trail_phase_words_mut(curr_trail) = stored_ow;
                }
                last_out_w = stored_ow;

                curr_trail.prev_ptr = prev_cur_ptr;
                curr_trail.step_count = prev_step_count.wrapping_add(1);
                let inherit_anchor = (curr_trail.target_anchor_id == 0) & (!contradiction);
                curr_trail.target_anchor_id = branchless_select_u32(
                    inherit_anchor,
                    prev_target_anchor_id,
                    curr_trail.target_anchor_id,
                );
                curr_trail.cluster_id = branchless_select_u32(
                    inherit_anchor & (prev_target_anchor_id != 0),
                    prev_target_anchor_id,
                    curr_trail.cluster_id,
                );

                let cur_ptr = gcso_tagged_ptr_t(curr_trail.current_ptr);
                let rep_ptr = cur_ptr
                    .with_action_and_slot(
                        ATTRACTOR_SLOT_REPULSION,
                        cur_ptr.action_mode_flags() | ACTION_PRESET_REPULSION_ESCAPE,
                    )
                    .conjugate_phase();
                curr_trail.current_ptr =
                    branchless_select_u64(contradiction, rep_ptr.raw(), cur_ptr.raw());

                let cur_rho = branchless_flush_subnormal_f32(branchless_relu_f32(
                    curr_trail.stigmergic_density,
                ));
                let pos_rho = cur_rho + 0.25 * (1.0 + branchless_relu_f32(pre_coh));
                let neg_rho = cur_rho * 0.5;
                curr_trail.stigmergic_density = branchless_min_f32(
                    branchless_flush_subnormal_f32(branchless_select_f32(
                        contradiction,
                        neg_rho,
                        pos_rho,
                    )),
                    MAX_STIGMERGIC_DENSITY,
                );

                let prev_pull =
                    branchless_flush_subnormal_f32(branchless_relu_f32(prev_pull_force));
                let cur_pull = branchless_flush_subnormal_f32(branchless_relu_f32(
                    curr_trail.attractor_pull_force,
                ));
                let bridged_pull =
                    cur_pull + 0.5 * prev_pull * branchless_clamp_f32(1.0 + pre_coh, 0.25, 2.0);
                curr_trail.attractor_pull_force = branchless_min_f32(
                    branchless_flush_subnormal_f32(branchless_select_f32(
                        contradiction,
                        cur_pull * 0.5,
                        bridged_pull,
                    )),
                    MAX_ATTRACTOR_PULL_FORCE,
                );

                let is_resonant = (!contradiction) & (pre_coh > 0.25);
                curr_trail.flags |=
                    branchless_select_u32(is_resonant, TRAIL_FLAG_SYNERGY_RESONANCE, 0)
                        | branchless_select_u32(
                            contradiction,
                            TRAIL_FLAG_REPULSION_ACTIVE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE,
                            0,
                        );

                if contradiction {
                    bloom_filter.insert_repulsive_pattern_words_weighted(&stored_ow, 1);
                    repulsed_count = repulsed_count.wrapping_add(1);
                }

                if post_coh < 0.999_999 {
                    joint_log_prob += branchless_fast_transition_log_prob(post_coh);
                    coboundary_defect_sum += branchless_relu_f32(0.5 * (1.0 - post_coh));
                }

                v_prev0 = v_out0;
                v_prev1 = v_out1;
                prev_cur_ptr = curr_trail.current_ptr;
                prev_step_count = curr_trail.step_count;
                prev_target_anchor_id = curr_trail.target_anchor_id;
                prev_pull_force = curr_trail.attractor_pull_force;
                idx += 1;
            }

            let mean_step_defect = coboundary_defect_sum * fast_inv_len_f32(len - 1);
            let l_d0 = _mm256_sub_epi8(v_first0, v_prev0);
            let l_d1 = _mm256_sub_epi8(v_first1, v_prev1);
            let l_or = _mm256_or_si256(l_d0, l_d1);
            let loop_coh = if _mm256_testz_si256(l_or, l_or) != 0 {
                1.0_f32
            } else {
                crate::traits::branchless::avx2_coherence_from_diff_vec64(l_d0, l_d1)
            };
            let loop_defect = branchless_relu_f32(0.5 * (1.0 - loop_coh));
            let holonomy_energy =
                branchless_flush_subnormal_f32(0.5 * (mean_step_defect + loop_defect));
            let is_obstructed = holonomy_energy > branchless_relu_f32(obstruction_threshold);

            Ok((
                branchless_flush_subnormal_f32(joint_log_prob),
                holonomy_energy,
                is_obstructed,
                repulsed_count,
                last_out_w,
            ))
        };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        use crate::traits::branchless::coherence_from_diff_words_64;
        use crate::traits::primitives::{
            branchless_cos_q7, swar_q7_circular_add_x8, swar_q7_circular_diff_x8,
            swar_q7_scale_le128_with_bias_x8,
        };
        let bias_restore = (128u64.wrapping_sub(t_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);

        let first_is_uniform = is_word_array_uniform_64(&first_w);
        let mut prev_w = first_w;
        let mut prev_is_uniform = first_is_uniform;
        let mut prev_cur_ptr = trail_chain[0].current_ptr;
        let mut prev_step_count = trail_chain[0].step_count;
        let mut prev_target_anchor_id = trail_chain[0].target_anchor_id;
        let mut prev_pull_force = trail_chain[0].attractor_pull_force;
        let mut joint_log_prob = 0.0_f32;
        let mut coboundary_defect_sum = 0.0_f32;
        let mut repulsed_count = 0u32;

        let mut idx = 1usize;
        while idx < len {
            let curr_trail = &mut trail_chain[idx];
            let s = *trail_phase_words_ref(curr_trail);
            let (pre_coh, contradiction, out_w, out_is_uniform, post_coh) =
                if (s[0] == prev_w[0]) && swar_words_equal_64(&s, &prev_w) {
                    (1.0_f32, false, s, prev_is_uniform, 1.0_f32)
                } else {
                    let s_is_uniform = is_word_array_uniform_64(&s);
                    let both_uni = prev_is_uniform && s_is_uniform;
                    let d0 = swar_q7_circular_diff_x8(s[0], prev_w[0]);
                    let (d1, d2, d3, d4, d5, d6, d7, diff_is_uniform) = if both_uni {
                        (d0, d0, d0, d0, d0, d0, d0, true)
                    } else {
                        let w1 = swar_q7_circular_diff_x8(s[1], prev_w[1]);
                        let w2 = swar_q7_circular_diff_x8(s[2], prev_w[2]);
                        let w3 = swar_q7_circular_diff_x8(s[3], prev_w[3]);
                        let w4 = swar_q7_circular_diff_x8(s[4], prev_w[4]);
                        let w5 = swar_q7_circular_diff_x8(s[5], prev_w[5]);
                        let w6 = swar_q7_circular_diff_x8(s[6], prev_w[6]);
                        let w7 = swar_q7_circular_diff_x8(s[7], prev_w[7]);
                        let d_uni = (d0 == w1)
                            && ((((d0 ^ w2) | (d0 ^ w3)) | ((d0 ^ w4) | (d0 ^ w5)))
                                | ((d0 ^ w6) | (d0 ^ w7)))
                                == 0;
                        (w1, w2, w3, w4, w5, w6, w7, d_uni)
                    };
                    let diff_is_byte_uniform = diff_is_uniform && (d0 == d0.rotate_right(8));
                    let pre_c = if diff_is_byte_uniform {
                        branchless_cos_q7(d0 as gcso_q7_t)
                    } else if diff_is_uniform {
                        crate::traits::branchless::coherence_from_uniform_diff_word_x8(d0)
                    } else {
                        coherence_from_diff_words_64(d0, d1, d2, d3, d4, d5, d6, d7)
                    };
                    let contr = pre_c < -safe_contradiction;
                    let (ow, ow_uni, post_c) = if contr {
                        let nw = if s_is_uniform {
                            [swar_q7_neg_sat_x8(s[0]); 8]
                        } else {
                            [
                                swar_q7_neg_sat_x8(s[0]),
                                swar_q7_neg_sat_x8(s[1]),
                                swar_q7_neg_sat_x8(s[2]),
                                swar_q7_neg_sat_x8(s[3]),
                                swar_q7_neg_sat_x8(s[4]),
                                swar_q7_neg_sat_x8(s[5]),
                                swar_q7_neg_sat_x8(s[6]),
                                swar_q7_neg_sat_x8(s[7]),
                            ]
                        };
                        let pc = coherence_from_word_arrays_with_both_uniform_flags_64(
                            &nw,
                            s_is_uniform,
                            &prev_w,
                            prev_is_uniform,
                        );
                        (nw, s_is_uniform, pc)
                    } else if t_s >= 128 {
                        (s, s_is_uniform, pre_c)
                    } else if t_s == 0 {
                        (prev_w, prev_is_uniform, 1.0_f32)
                    } else if diff_is_uniform {
                        let st0 = swar_q7_scale_le128_with_bias_x8(d0, t_s, bias_restore);
                        if st0 == 0 {
                            (prev_w, prev_is_uniform, 1.0_f32)
                        } else {
                            let nw = if prev_is_uniform {
                                [swar_q7_circular_add_x8(prev_w[0], st0); 8]
                            } else {
                                [
                                    swar_q7_circular_add_x8(prev_w[0], st0),
                                    swar_q7_circular_add_x8(prev_w[1], st0),
                                    swar_q7_circular_add_x8(prev_w[2], st0),
                                    swar_q7_circular_add_x8(prev_w[3], st0),
                                    swar_q7_circular_add_x8(prev_w[4], st0),
                                    swar_q7_circular_add_x8(prev_w[5], st0),
                                    swar_q7_circular_add_x8(prev_w[6], st0),
                                    swar_q7_circular_add_x8(prev_w[7], st0),
                                ]
                            };
                            let pc = if diff_is_byte_uniform {
                                branchless_cos_q7(st0 as gcso_q7_t)
                            } else {
                                coherence_from_diff_words_64(st0, st0, st0, st0, st0, st0, st0, st0)
                            };
                            (nw, prev_is_uniform, pc)
                        }
                    } else {
                        let st0 = swar_q7_scale_le128_with_bias_x8(d0, t_s, bias_restore);
                        let st1 = swar_q7_scale_le128_with_bias_x8(d1, t_s, bias_restore);
                        let st2 = swar_q7_scale_le128_with_bias_x8(d2, t_s, bias_restore);
                        let st3 = swar_q7_scale_le128_with_bias_x8(d3, t_s, bias_restore);
                        let st4 = swar_q7_scale_le128_with_bias_x8(d4, t_s, bias_restore);
                        let st5 = swar_q7_scale_le128_with_bias_x8(d5, t_s, bias_restore);
                        let st6 = swar_q7_scale_le128_with_bias_x8(d6, t_s, bias_restore);
                        let st7 = swar_q7_scale_le128_with_bias_x8(d7, t_s, bias_restore);
                        if (((st0 | st1) | (st2 | st3)) | ((st4 | st5) | (st6 | st7))) == 0 {
                            (prev_w, prev_is_uniform, 1.0_f32)
                        } else {
                            let nw = [
                                swar_q7_circular_add_x8(prev_w[0], st0),
                                swar_q7_circular_add_x8(prev_w[1], st1),
                                swar_q7_circular_add_x8(prev_w[2], st2),
                                swar_q7_circular_add_x8(prev_w[3], st3),
                                swar_q7_circular_add_x8(prev_w[4], st4),
                                swar_q7_circular_add_x8(prev_w[5], st5),
                                swar_q7_circular_add_x8(prev_w[6], st6),
                                swar_q7_circular_add_x8(prev_w[7], st7),
                            ];
                            let nw_uni = is_word_array_uniform_64(&nw);
                            let pc = if nw_uni && prev_is_uniform {
                                branchless_cos_q7(st0 as gcso_q7_t)
                            } else {
                                coherence_from_diff_words_64(st0, st1, st2, st3, st4, st5, st6, st7)
                            };
                            (nw, nw_uni, pc)
                        }
                    };
                    *trail_phase_words_mut(curr_trail) = ow;
                    (pre_c, contr, ow, ow_uni, post_c)
                };

            curr_trail.prev_ptr = prev_cur_ptr;
            curr_trail.step_count = prev_step_count.wrapping_add(1);
            let inherit_anchor = (curr_trail.target_anchor_id == 0) & (!contradiction);
            curr_trail.target_anchor_id = branchless_select_u32(
                inherit_anchor,
                prev_target_anchor_id,
                curr_trail.target_anchor_id,
            );
            curr_trail.cluster_id = branchless_select_u32(
                inherit_anchor & (prev_target_anchor_id != 0),
                prev_target_anchor_id,
                curr_trail.cluster_id,
            );

            let cur_ptr = gcso_tagged_ptr_t(curr_trail.current_ptr);
            let rep_ptr = cur_ptr
                .with_action_and_slot(
                    ATTRACTOR_SLOT_REPULSION,
                    cur_ptr.action_mode_flags() | ACTION_PRESET_REPULSION_ESCAPE,
                )
                .conjugate_phase();
            curr_trail.current_ptr =
                branchless_select_u64(contradiction, rep_ptr.raw(), cur_ptr.raw());

            let cur_rho =
                branchless_flush_subnormal_f32(branchless_relu_f32(curr_trail.stigmergic_density));
            let pos_rho = cur_rho + 0.25 * (1.0 + branchless_relu_f32(pre_coh));
            let neg_rho = cur_rho * 0.5;
            curr_trail.stigmergic_density = branchless_min_f32(
                branchless_flush_subnormal_f32(branchless_select_f32(
                    contradiction,
                    neg_rho,
                    pos_rho,
                )),
                MAX_STIGMERGIC_DENSITY,
            );

            let prev_pull = branchless_flush_subnormal_f32(branchless_relu_f32(prev_pull_force));
            let cur_pull = branchless_flush_subnormal_f32(branchless_relu_f32(
                curr_trail.attractor_pull_force,
            ));
            let bridged_pull =
                cur_pull + 0.5 * prev_pull * branchless_clamp_f32(1.0 + pre_coh, 0.25, 2.0);
            curr_trail.attractor_pull_force = branchless_min_f32(
                branchless_flush_subnormal_f32(branchless_select_f32(
                    contradiction,
                    cur_pull * 0.5,
                    bridged_pull,
                )),
                MAX_ATTRACTOR_PULL_FORCE,
            );

            let is_resonant = (!contradiction) & (pre_coh > 0.25);
            curr_trail.flags |= branchless_select_u32(is_resonant, TRAIL_FLAG_SYNERGY_RESONANCE, 0)
                | branchless_select_u32(
                    contradiction,
                    TRAIL_FLAG_REPULSION_ACTIVE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE,
                    0,
                );

            if contradiction {
                bloom_filter.insert_repulsive_pattern_words_weighted(&out_w, 1);
                repulsed_count = repulsed_count.wrapping_add(1);
            }

            if post_coh < 0.999_999 {
                joint_log_prob += branchless_fast_transition_log_prob(post_coh);
                coboundary_defect_sum += branchless_relu_f32(0.5 * (1.0 - post_coh));
            }

            prev_w = out_w;
            prev_is_uniform = out_is_uniform;
            prev_cur_ptr = curr_trail.current_ptr;
            prev_step_count = curr_trail.step_count;
            prev_target_anchor_id = curr_trail.target_anchor_id;
            prev_pull_force = curr_trail.attractor_pull_force;
            idx += 1;
        }

        let mean_step_defect = coboundary_defect_sum * fast_inv_len_f32(len - 1);
        let loop_coh = if (first_w[0] == prev_w[0]) && swar_words_equal_64(&first_w, &prev_w) {
            1.0_f32
        } else {
            coherence_from_word_arrays_with_both_uniform_flags_64(
                &first_w,
                first_is_uniform,
                &prev_w,
                prev_is_uniform,
            )
        };
        let loop_defect = branchless_relu_f32(0.5 * (1.0 - loop_coh));
        let holonomy_energy =
            branchless_flush_subnormal_f32(0.5 * (mean_step_defect + loop_defect));
        let is_obstructed = holonomy_energy > branchless_relu_f32(obstruction_threshold);

        Ok((
            branchless_flush_subnormal_f32(joint_log_prob),
            holonomy_energy,
            is_obstructed,
            repulsed_count,
            prev_w,
        ))
    }
}

/// Single-scan register-resident Bayesian candidate-trail selector, pointer rewirer, Limina
/// geodesic bridger, and pre-Softmax Attention score modulator that retains `best_coh` in
/// registers so the transition log-probability $\ln P(M_{j^*} \mid M_t)$ is computed with
/// **zero redundant 64-head coherence reloads**, and directly incorporates `macro_anchor_id`
/// cluster matching in a single pass.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn select_rewire_bridge_and_eval_log_prob_in_registers(
    src_trail: &mut gcso_pointer_trail_t,
    preloaded_src_w_opt: Option<[u64; NUM_STIGMERGIC_GRANULES]>,
    candidate_trails: &mut [gcso_pointer_trail_t],
    attention_scores_opt: Option<&mut [f32]>,
    head_dim: u32,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    contradiction_threshold: f32,
    interpolation_u7: u8,
    repulsion_penalty: f32,
    macro_anchor_id: u32,
    secondary_macro_anchor_id: u32,
    win_w_opt: Option<&[u64; NUM_STIGMERGIC_GRANULES]>,
    bloom_filter: &mut ContrastivePhaseBloomFilter,
) -> (u32, f32, f32, bool) {
    if candidate_trails.is_empty() {
        return (src_trail.linked_trail_id, 0.0, 0.0, false);
    }

    let src_w = match preloaded_src_w_opt {
        Some(w) => w,
        None => *trail_phase_words_ref(src_trail),
    };
    let outcome = select_and_bridge_macro_aligned_bayesian_trail_core(
        src_w,
        win_w_opt,
        macro_anchor_id,
        secondary_macro_anchor_id,
        src_trail,
        candidate_trails,
        attention_scores_opt,
        head_dim,
        phase_interference_gain,
        stigmergic_prior_gain,
        pointer_link_bonus,
        repulsion_penalty,
        interpolation_u7,
        contradiction_threshold,
        0.0,
        false,
    );

    if outcome.found_valid {
        let dst_trail = &mut candidate_trails[outcome.best_idx];
        let cur_rho =
            branchless_flush_subnormal_f32(branchless_relu_f32(src_trail.stigmergic_density));
        src_trail.stigmergic_density = branchless_min_f32(
            branchless_flush_subnormal_f32(
                cur_rho + 0.25 * (1.0 + branchless_relu_f32(outcome.best_coh)),
            ),
            MAX_STIGMERGIC_DENSITY,
        );

        let cand_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(dst_trail.attractor_pull_force));
        let self_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(src_trail.attractor_pull_force));
        src_trail.attractor_pull_force = branchless_min_f32(
            branchless_flush_subnormal_f32(
                self_pull
                    + 0.5 * cand_pull * branchless_clamp_f32(1.0 + outcome.best_coh, 0.25, 2.0),
            ),
            MAX_ATTRACTOR_PULL_FORCE,
        );

        let inherit_anchor = dst_trail.target_anchor_id != 0;
        let inherited_id = branchless_select_u32(
            inherit_anchor,
            dst_trail.target_anchor_id,
            src_trail.target_anchor_id,
        );
        src_trail.target_anchor_id = inherited_id;
        dst_trail.target_anchor_id = inherited_id;
        let resolved_cluster_id = branchless_select_u32(
            dst_trail.cluster_id != 0,
            dst_trail.cluster_id,
            branchless_select_u32(
                (macro_anchor_id != 0) & (src_trail.cluster_id != 0),
                src_trail.cluster_id,
                inherited_id,
            ),
        );
        if resolved_cluster_id != 0 {
            src_trail.cluster_id = resolved_cluster_id;
            dst_trail.cluster_id = resolved_cluster_id;
        }
        src_trail.flags =
            (src_trail.flags | TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE)
                & !TRAIL_FLAG_REPULSION_ACTIVE;
        dst_trail.flags =
            (dst_trail.flags | TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE)
                & !TRAIL_FLAG_REPULSION_ACTIVE;

        let transition_lp = branchless_fast_transition_log_prob(outcome.best_coh);
        (outcome.best_id, outcome.best_score, transition_lp, true)
    } else {
        bloom_filter.insert_repulsive_pattern_words_weighted(&src_w, 1);
        src_trail.stamp_phase_conjugate_repulsion();
        (outcome.best_id, 0.0, 0.0, false)
    }
}

/// Applies Pitchfork Metastable Tunneling phase perturbation (`Lemma 3.1`) directly to
/// register-resident `words_inout: &mut [u64; NUM_STIGMERGIC_GRANULES]` and writes only the
/// perturbed exploration cells (`explore_head_start..MAX_ATTENTION_HEADS`) back to `phases_inout`,
/// eliminating Store-to-Load Forwarding reload stalls in `step_edbc_tunneling_and_select_bayesian_core`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn inject_metastable_tunneling_perturbation_in_words_64(
    phases_inout: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    words_inout: &mut [u64; NUM_STIGMERGIC_GRANULES],
    seed_ptr: gcso_tagged_ptr_t,
    void_score: f32,
    tau_eff: f32,
    alpha: f32,
    beta: f32,
    head_dim: u32,
    explore_head_start: usize,
) -> GcsoResult<f32> {
    if explore_head_start > MAX_ATTENTION_HEADS {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    if !branchless_is_finite_f32(void_score)
        || !branchless_is_finite_f32(tau_eff)
        || !branchless_is_finite_f32(alpha)
        || !branchless_is_finite_f32(beta)
        || beta <= 0.0
        || head_dim == 0
        || (head_dim & 1) != 0
    {
        return Err(GCSO_ERROR_EDBC_SINGULARITY);
    }

    let excess = branchless_relu_f32(void_score - tau_eff);
    let safe_alpha = branchless_relu_f32(alpha);
    if excess <= 0.0 || safe_alpha <= 0.0 {
        return Ok(0.0);
    }

    let rsqrt_dim = if head_dim <= 256 {
        branchless_popcount_rsqrt_u32(head_dim)
    } else {
        branchless_rsqrt_f32(head_dim as f32)
    };
    let mu = (safe_alpha * excess) * rsqrt_dim;
    let prod = mu * beta;
    let raw_x_star = if (prod >= 1.0e-36) & (prod <= 1.0e34) {
        mu * branchless_rsqrt_f32(prod)
    } else {
        branchless_sqrt_f32(mu) * branchless_rsqrt_f32(beta)
    };
    let x_star = branchless_clamp_f32(raw_x_star, 0.0, FRAC_PI_2);
    let amp_q7 = branchless_rad_to_q7(x_star);
    if amp_q7 == 0 || explore_head_start == MAX_ATTENTION_HEADS {
        return Ok(x_star);
    }

    let hash = branchless_splitmix64(seed_ptr.raw() ^ 0x9E37_79B9_7F4A_7C15);
    let mut h = explore_head_start;
    while (h & 7) != 0 && h < MAX_ATTENTION_HEADS {
        let pos_branch = ((hash >> h) & 1) == 0;
        let signed_pulse = branchless_select_q7(pos_branch, amp_q7, -amp_q7);
        let upd = branchless_q7_add_sat(phases_inout[h], signed_pulse);
        phases_inout[h] = upd;
        let c = h >> 3;
        let shift = (h & 7) << 3;
        words_inout[c] = (words_inout[c] & !(0xFFu64 << shift)) | (u64::from(upd as u8) << shift);
        h += 1;
    }

    let pos_u8 = amp_q7 as u8;
    let neg_u8 = (-amp_q7) as u8;
    let pos_w = u64::from(pos_u8).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let xor_byte = u64::from(pos_u8 ^ neg_u8);

    let mut c = h >> 3;
    while c < NUM_STIGMERGIC_GRANULES {
        let bits_w = (hash >> c) & SWAR_SPLAT_BYTE_64;
        let pulse_w = pos_w ^ bits_w.wrapping_mul(xor_byte);
        let next_w = swar_q7_add_sat_x8(words_inout[c], pulse_w);
        words_inout[c] = next_w;
        store_q7_word_at(phases_inout, c << 3, next_w);
        c += 1;
    }
    Ok(x_star)
}

/// Unified EDBC stateful evaluation + Pitchfork metastable tunneling + single-pass register-resident
/// Bayesian candidate-trail selection & rewiring + pointer mode evolution tail shared across
/// [`EntropyClosedLoopCoordinator::step_with_bayesian_pointer_selection_and_tunneling`],
/// [`EntropyClosedLoopCoordinator::route_coarse_to_fine_and_step_bayesian_tunneling`], and
/// [`EntropyClosedLoopCoordinator::route_top2_liminal_and_step_bayesian_tunneling`].
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn step_edbc_tunneling_and_select_bayesian_core<
    E: EntropyClosedLoopCoordinator + ?Sized,
>(
    engine: &mut E,
    src_trail: &mut gcso_pointer_trail_t,
    preloaded_src_w_opt: Option<[u64; NUM_STIGMERGIC_GRANULES]>,
    candidate_trails: &mut [gcso_pointer_trail_t],
    token_z_score: f32,
    void_score: f32,
    alpha: f32,
    beta: f32,
    d0: f32,
    head_dim: u32,
    explore_head_start: usize,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    repulsion_penalty: f32,
    interpolation_u7: u8,
    contradiction_threshold: f32,
    high_density_rank1_thresh: f32,
    macro_anchor_id: u32,
    secondary_macro_anchor_id: u32,
    win_w_opt: Option<&[u64; NUM_STIGMERGIC_GRANULES]>,
    bloom_filter: &mut ContrastivePhaseBloomFilter,
    mut attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
) -> GcsoResult<(
    gcso_edbc_state_t,
    gcso_tagged_ptr_t,
    u32,
    f32,
    f32,
    f32,
    bool,
)> {
    let mut st = gcso_edbc_state_t::default();
    engine.eval_stateful(token_z_score, &mut st)?;
    let bif_thresh = branchless_select_f32(
        (st.bifurcation_threshold > 0.0) & branchless_is_finite_f32(st.bifurcation_threshold),
        st.bifurcation_threshold,
        DEFAULT_BIFURCATION_THRESHOLD,
    );
    st.bifurcation_threshold = bif_thresh;

    let resolved_mode = engine.evaluate_branch_mode_transition(
        st.active_branch_mode,
        st.moving_z_entropy,
        st.sliding_entropy_rate,
        void_score,
        bif_thresh,
        DEFAULT_TAU_FLUX,
        DEFAULT_HYSTERESIS_BAND,
    );
    st.active_branch_mode = resolved_mode;

    let (tunneling_x_star, eff_preloaded_w) = if resolved_mode == EDBC_MODE_PITCHFORK_BIFURCATION {
        let mut src_w = match preloaded_src_w_opt {
            Some(w) => w,
            None => *trail_phase_words_ref(src_trail),
        };
        let x_star = inject_metastable_tunneling_perturbation_in_words_64(
            &mut src_trail.accumulated_phase_delta,
            &mut src_w,
            gcso_tagged_ptr_t(src_trail.current_ptr),
            void_score,
            bif_thresh,
            alpha,
            beta,
            head_dim,
            explore_head_start,
        )?;
        (x_star, Some(src_w))
    } else {
        (0.0, preloaded_src_w_opt)
    };

    let mut selected_id = 0u32;
    let mut bayes_score = -1.0e30_f32;
    let mut transition_log_prob = 0.0_f32;
    let mut found_coherent = false;

    if resolved_mode != EDBC_MODE_PHASE_CONJUGATE_REPULSION {
        let (sel_id, b_score, trans_lp, found) =
            select_rewire_bridge_and_eval_log_prob_in_registers(
                src_trail,
                eff_preloaded_w,
                candidate_trails,
                attention_scores.as_deref_mut().map(|s| s.as_mut_slice()),
                head_dim,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                contradiction_threshold,
                interpolation_u7,
                repulsion_penalty,
                macro_anchor_id,
                secondary_macro_anchor_id,
                win_w_opt,
                bloom_filter,
            );
        selected_id = sel_id;
        bayes_score = b_score;
        transition_log_prob = trans_lp;
        found_coherent = found;

        if !found {
            st.active_branch_mode = EDBC_MODE_PHASE_CONJUGATE_REPULSION;
            st.repulsion_gain = LUT_REPULSION_GAIN[2];
        }
    } else {
        match eff_preloaded_w {
            Some(ref w) => bloom_filter.insert_repulsive_pattern_words_weighted(w, 1),
            None => bloom_filter.insert_repulsive_pattern(&src_trail.accumulated_phase_delta),
        }
        src_trail.stamp_phase_conjugate_repulsion();
        st.active_branch_mode = EDBC_MODE_PHASE_CONJUGATE_REPULSION;
        st.repulsion_gain = LUT_REPULSION_GAIN[2];

        if let Some(scores) = attention_scores {
            let (_mean_coh, best_id, _coh_cnt) = src_trail
                .modulate_candidate_trail_scores_in_place(
                    scores.as_mut_slice(),
                    candidate_trails,
                    head_dim,
                    phase_interference_gain,
                    stigmergic_prior_gain,
                    pointer_link_bonus,
                    contradiction_threshold,
                    repulsion_penalty,
                );
            selected_id = best_id;
        }
    }

    let repulsion_active = !found_coherent;
    let dom_lane = branchless_select_u8(
        src_trail.target_anchor_id != 0,
        (src_trail.target_anchor_id as u8) & ATTRACTOR_SLOT_ID_MASK,
        src_trail.target_attractor_slot(),
    );
    let pre_evolve_cluster_id = src_trail.cluster_id;
    let evolved_ptr = src_trail.evolve_trail_mode_stigmergically(
        st.moving_z_entropy,
        repulsion_active,
        dom_lane,
        branchless_relu_f32(bif_thresh * 0.5),
        high_density_rank1_thresh,
    );
    if (macro_anchor_id != 0) & (!repulsion_active) & (pre_evolve_cluster_id != 0) {
        src_trail.cluster_id = pre_evolve_cluster_id;
    }
    st.sample_temperature = if st.active_branch_mode == EDBC_MODE_NORMAL_ENTROPY {
        DEFAULT_SAMPLE_TEMPERATURE
    } else {
        eval_temperature_for_mode(
            st.active_branch_mode,
            DEFAULT_SAMPLE_TEMPERATURE,
            st.moving_z_entropy,
            void_score,
            bif_thresh,
            alpha,
            beta,
            d0,
        )
    };

    Ok((
        st,
        evolved_ptr,
        selected_id,
        bayes_score,
        transition_log_prob,
        tunneling_x_star,
        found_coherent,
    ))
}

/// Register-resident core helper for [`EntropyClosedLoopCoordinator::step_via_gateway_frame`],
/// [`EntropyClosedLoopCoordinator::step_via_gateway_signal`], and
/// [`EntropyClosedLoopCoordinator::step_via_gateway_kmeans_64`] that operates directly on
/// `gw_w: &[u64; 8]` without reloading `frame.phase_deltas` from the stack.
#[inline]
pub(super) fn step_via_gateway_frame_with_words_core<
    T: EdbcBranchController + SheafCohomologyEvaluator + HierarchicalTreeTable + ?Sized,
>(
    engine: &mut T,
    token_z_score: f32,
    void_score: f32,
    alpha: f32,
    beta: f32,
    d0: f32,
    high_density_rank1_thresh: f32,
    frame: &GatewayFrame,
    gw_w: &[u64; NUM_STIGMERGIC_GRANULES],
    trail: &mut gcso_pointer_trail_t,
    logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    crystal_record_opt: Option<&mut CrystallizedAttractorRecord>,
) -> GcsoResult<(
    gcso_edbc_state_t,
    gcso_tagged_ptr_t,
    CausalProvenanceReport,
    f32,
    f32,
)> {
    if !branchless_is_finite_f32(high_density_rank1_thresh) {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    if !branchless_is_finite_f32(void_score)
        || !branchless_is_finite_f32(alpha)
        || !branchless_is_finite_f32(beta)
        || !branchless_is_finite_f32(d0)
        || beta <= 0.0
        || d0 <= 0.0
    {
        return Err(GCSO_ERROR_EDBC_SINGULARITY);
    }

    let mut raw_st = gcso_edbc_state_t::default();
    engine.eval_stateful(token_z_score, &mut raw_st)?;
    let moving_z_entropy = raw_st.moving_z_entropy;
    let sliding_entropy_rate = raw_st.sliding_entropy_rate;
    let bif_thresh = branchless_select_f32(
        (raw_st.bifurcation_threshold > 0.0)
            & branchless_is_finite_f32(raw_st.bifurcation_threshold),
        raw_st.bifurcation_threshold,
        DEFAULT_BIFURCATION_THRESHOLD,
    );

    let resolved_mode = engine.evaluate_branch_mode_transition(
        raw_st.active_branch_mode,
        moving_z_entropy,
        sliding_entropy_rate,
        void_score,
        bif_thresh,
        DEFAULT_TAU_FLUX,
        DEFAULT_HYSTERESIS_BAND,
    );

    let frame_repulsion = (frame.dominant_cause_code == ROOT_CAUSE_COHOMOLOGY_REPULSION)
        | ((frame.provenance_bits & CAUSAL_CAUSE_COHOMOLOGY_REPULSION) != 0);
    let repulsion_triggered =
        (resolved_mode == EDBC_MODE_PHASE_CONJUGATE_REPULSION) | frame_repulsion;
    let final_mode = branchless_select_u32(
        repulsion_triggered,
        EDBC_MODE_PHASE_CONJUGATE_REPULSION,
        resolved_mode,
    );
    let final_rep_gain = LUT_REPULSION_GAIN[(final_mode as usize).min(2)];

    let cur_pull = branchless_flush_subnormal_f32(branchless_relu_f32(trail.attractor_pull_force));
    let gw_pull = branchless_flush_subnormal_f32(branchless_relu_f32(frame.effective_energy));
    let total_pull = cur_pull + gw_pull;
    let has_cur_pull = cur_pull > DEFAULT_ENTROPY_EPS;
    let qdps_step = frame.qdps_min_step_q7;
    #[allow(unused_mut)]
    let mut qdps_filtered_in_vec = false;
    let (mut next_w, next_eq_gw, mut next_eq_cur) = if gw_pull <= DEFAULT_ENTROPY_EPS {
        (*trail_phase_words_ref(trail), false, true)
    } else if !has_cur_pull {
        (*gw_w, true, false)
    } else {
        let cur_w_ref = trail_phase_words_ref(trail);
        if (cur_w_ref[0] == gw_w[0]) && swar_words_equal_64(cur_w_ref, gw_w) {
            (*gw_w, true, true)
        } else {
            let ratio_f32 = branchless_clamp_f32(
                (gw_pull * fast_inv_pos_normal_f32(total_pull)) * 128.0 + 0.5,
                0.0,
                128.0,
            );
            let s = (ratio_f32 as u8) as u64;
            if s == 0 {
                (*cur_w_ref, false, true)
            } else if s >= 128 {
                (*gw_w, true, false)
            } else {
                let cur_uni = is_word_array_uniform_64(cur_w_ref);
                let gw_uni = is_word_array_uniform_64(gw_w);
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                {
                    if cur_uni && gw_uni {
                        let bias_restore =
                            (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                        let w0 = crate::traits::branchless::swar_slerp_word_unchecked_x8(
                            cur_w_ref[0],
                            gw_w[0],
                            s,
                            bias_restore,
                        );
                        ([w0; NUM_STIGMERGIC_GRANULES], false, false)
                    } else {
                        // SAFETY: `target_feature = "avx2"` is enabled; loads `trail.accumulated_phase_delta`
                        // directly from L1D and converts `gw_w` via register moves (`avx2_words64_to_vec64_with_uniform`)
                        // to eliminate 64B stack copy and 64-bit-to-256-bit Store-to-Load Forwarding (STLF) stalls.
                        let nw = unsafe {
                            let (vc0, vc1) =
                                crate::traits::branchless::avx2_words64_to_vec64_with_uniform(
                                    cur_w_ref, cur_uni,
                                );
                            let (vg0, vg1) =
                                crate::traits::branchless::avx2_words64_to_vec64_with_uniform(
                                    gw_w, gw_uni,
                                );
                            let (vr0, vr1) =
                                crate::traits::branchless::avx2_slerp_vec64(vc0, vc1, vg0, vg1, s);
                            let (vf0, vf1) = if qdps_step > 1 {
                                qdps_filtered_in_vec = true;
                                crate::traits::branchless::avx2_qdps_filter_vec64(
                                    vr0, vr1, qdps_step,
                                )
                            } else {
                                (vr0, vr1)
                            };
                            crate::traits::branchless::avx2_vec64_to_words64(vf0, vf1)
                        };
                        (nw, false, false)
                    }
                }
                #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                {
                    let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                    let (nw, _) = crate::traits::swar_slerp_words_with_uniform_flag_64(
                        cur_w_ref,
                        cur_uni,
                        gw_w,
                        gw_uni,
                        s,
                        bias_restore,
                    );
                    (nw, false, false)
                }
            }
        }
    };

    if (qdps_step > 1) & (!qdps_filtered_in_vec) & has_cur_pull & (!next_eq_gw)
        && (!swar_words_equal_64(&next_w, gw_w))
    {
        next_w = swar_q7_qdps_filter_words_64(&next_w, qdps_step);
        next_eq_cur = false;
    }
    if repulsion_triggered & (!frame_repulsion) {
        next_w = [
            swar_q7_neg_sat_x8(next_w[0]),
            swar_q7_neg_sat_x8(next_w[1]),
            swar_q7_neg_sat_x8(next_w[2]),
            swar_q7_neg_sat_x8(next_w[3]),
            swar_q7_neg_sat_x8(next_w[4]),
            swar_q7_neg_sat_x8(next_w[5]),
            swar_q7_neg_sat_x8(next_w[6]),
            swar_q7_neg_sat_x8(next_w[7]),
        ];
        next_eq_cur = false;
    }
    if (!has_cur_pull) | (!next_eq_cur) {
        *trail_phase_words_mut(trail) = next_w;
    }

    trail.prev_ptr = trail.current_ptr;
    trail.step_count = trail.step_count.wrapping_add(1);
    trail.attractor_pull_force = branchless_min_f32(total_pull, MAX_ATTRACTOR_PULL_FORCE);
    let cur_rho = branchless_flush_subnormal_f32(branchless_relu_f32(trail.stigmergic_density));
    let next_rho = branchless_select_f32(
        repulsion_triggered,
        cur_rho * 0.5,
        branchless_min_f32(
            cur_rho + 0.20 * branchless_clamp_f32(1.0 + frame.fact_persona_coherence, 0.25, 2.0),
            MAX_STIGMERGIC_DENSITY,
        ),
    );
    trail.stigmergic_density = branchless_flush_subnormal_f32(next_rho);

    let has_frame_anchor = frame.target_anchor_id != 0;
    let dom_anchor = branchless_select_u32(
        has_frame_anchor,
        frame.target_anchor_id,
        trail.target_anchor_id,
    );
    trail.target_anchor_id = dom_anchor;
    let dom_lane = branchless_select_u8(
        dom_anchor != 0,
        (dom_anchor as u8) & ATTRACTOR_SLOT_ID_MASK,
        frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK,
    );

    let base_ptr = gcso_tagged_ptr_t(frame.tagged_ptr);
    let low_entropy_skip_thresh = branchless_relu_f32(bif_thresh * 0.5);
    let mut evolved_ptr = base_ptr.evolve_mode_stigmergically(
        trail.stigmergic_density,
        moving_z_entropy,
        repulsion_triggered,
        dom_lane,
        low_entropy_skip_thresh,
        high_density_rank1_thresh,
    );
    if ((frame.action_mode_flags & ACTION_MODE_LAYER_SKIP) != 0)
        & (final_mode == EDBC_MODE_NORMAL_ENTROPY)
        & (!repulsion_triggered)
    {
        evolved_ptr = evolved_ptr.with_action_and_slot(dom_lane, ACTION_PRESET_INERTIAL_SKIP);
    }
    trail.current_ptr = evolved_ptr.raw();
    trail.cluster_id = branchless_select_u32(
        (dom_anchor != 0) & (!repulsion_triggered),
        dom_anchor,
        evolved_ptr.target_attractor_slot() as u32,
    );
    trail.sync_flags_with_current_ptr();
    if repulsion_triggered {
        trail.flags |= TRAIL_FLAG_REPULSION_ACTIVE;
    }

    let gw_inv_temp = branchless_select_f32(
        branchless_is_finite_f32(frame.effective_inv_temp) & (frame.effective_inv_temp > 0.0),
        frame.effective_inv_temp,
        1.0,
    );
    let (edbc_temp, inv_edbc_temp) = if final_mode == EDBC_MODE_NORMAL_ENTROPY {
        (DEFAULT_SAMPLE_TEMPERATURE, 1.0_f32)
    } else {
        let t = eval_temperature_for_mode(
            final_mode,
            DEFAULT_SAMPLE_TEMPERATURE,
            moving_z_entropy,
            void_score,
            bif_thresh,
            alpha,
            beta,
            d0,
        );
        (t, fast_inv_pos_normal_f32(branchless_max_f32(t, 0.05)))
    };
    let eff_inv_temp = branchless_clamp_f32(gw_inv_temp * inv_edbc_temp, 0.05, 8.0);
    if let Some(logits) = logits_opt {
        scale_f32_slice_unrolled8(logits.as_mut_slice(), eff_inv_temp);
    }

    let mut report = frame.to_causal_report();
    report.dominant_slot = evolved_ptr.target_attractor_slot();
    report.action_mode_flags = evolved_ptr.action_mode_flags();
    if repulsion_triggered {
        report.provenance_bits |= CAUSAL_CAUSE_COHOMOLOGY_REPULSION;
        report.dominant_cause_code = ROOT_CAUSE_COHOMOLOGY_REPULSION;
    } else if (evolved_ptr.action_mode_flags() & ACTION_MODE_LAYER_SKIP) != 0
        && (evolved_ptr.action_mode_flags() & ACTION_MODE_DPSR_STEER) == 0
    {
        report.provenance_bits |= CAUSAL_AXIS_SYSTEM_CONSTRAINT | CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE;
        if report.dominant_cause_code == ROOT_CAUSE_NOMINAL {
            report.dominant_cause_code = ROOT_CAUSE_SYSTEM_QOS_THROTTLE;
        }
    }

    let mut crystal_mass = 0.0_f32;
    if let Some(crystal) = crystal_record_opt {
        if (!repulsion_triggered) & (gw_pull > DEFAULT_ENTROPY_EPS) {
            let new_dom = evolved_ptr.target_attractor_slot();
            let new_act = evolved_ptr.action_mode_flags();
            crystal_mass = crystal.absorb_gateway_frame_with_words_and_overrides(
                frame,
                &next_w,
                new_dom,
                new_act,
                report.provenance_bits,
            );
            if (crystal.provenance_bits & CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED) != 0 {
                trail.flags |= TRAIL_FLAG_CRYSTALLIZED_CANDIDATE;
                report.provenance_bits |= CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED;
            }
        } else {
            crystal_mass = crystal.phase_gravity_mass;
        }
    }

    raw_st.bifurcation_threshold = bif_thresh;
    raw_st.repulsion_gain = final_rep_gain;
    raw_st.sample_temperature = edbc_temp;
    raw_st.active_branch_mode = final_mode;
    Ok((raw_st, evolved_ptr, report, eff_inv_temp, crystal_mass))
}
