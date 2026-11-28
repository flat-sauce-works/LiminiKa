// name: src/core/src/stigmergy/trail/resonance.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Anchor resonance, cross-slot synergy, liminal bridging, and trail chain cohomology.

use super::{PointerTrail, trail_phase_words_mut, trail_phase_words_ref};
use crate::foundation::{
    ACTION_MODE_LAYER_SKIP, ACTION_MODE_PHASE_STEER, ACTION_PRESET_REPULSION_ESCAPE,
    AlignedSlice32, AlignedSliceMut32, PagedBitmask, SWAR_SPLAT_BYTE_64, TaggedPtr,
    branchless_abs_f32, branchless_clamp_f32, branchless_fast_transition_log_prob,
    branchless_flush_subnormal_f32, branchless_is_finite_f32, branchless_max_f32,
    branchless_min_f32, branchless_popcount_rsqrt_u32, branchless_q7_conjugate_64,
    branchless_q7_transition_log_prob_words_64, branchless_relu_f32,
    coherence_from_uniform_diff_word_x8, coherence_from_word_arrays_64,
    coherence_from_word_arrays_with_both_uniform_flags_64, evaluate_cohomological_obstruction,
    fast_inv_len_f32, is_word_array_uniform_64, load_q7_words_64, swar_q7_circular_add_x8,
    swar_q7_circular_diff_x8, swar_q7_neg_sat_x8, swar_q7_scale_le128_with_bias_x8,
    swar_slerp_words_with_uniform_flag_64, unpack_action_mode_from_tag,
};
use crate::liminal::{
    DEFAULT_ENTROPY_EPS, LMNK_ERR_INVALID_ARGUMENT, LmnkResult, MAX_ATTENTION_HEADS, PhaseQ7,
    branchless_select_f32, branchless_select_u8, branchless_select_u32, branchless_select_u64,
};
use crate::stigmergy::ledger::{ContrastivePhaseBloomFilter, eval_bitmask_dyadic_ultrametric};
use crate::stigmergy::{
    ATTRACTOR_SLOT_REPULSION, MAX_ATTRACTOR_PULL_FORCE, MAX_STIGMERGIC_DENSITY,
    TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_PHASE_STEER_ACTIVE,
    TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_FLAG_SYNERGY_RESONANCE,
};

impl PointerTrail {
    /// Applies **Zero-Hyperparameter Phase-Proximity Resonance** from a single anchor on the
    /// unified phase manifold (`anchor_phases`, `anchor_base_energy`) in strictly constant time (`0` branches).
    #[inline]
    pub fn apply_unified_manifold_anchor_resonance(
        &mut self,
        anchor_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
        anchor_base_energy: f32,
    ) -> (f32, f32) {
        let clean_energy = branchless_flush_subnormal_f32(branchless_relu_f32(anchor_base_energy));
        let cur_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(self.attractor_pull_force));
        if clean_energy <= DEFAULT_ENTROPY_EPS {
            self.attractor_pull_force = branchless_min_f32(cur_pull, MAX_ATTRACTOR_PULL_FORCE);
            return (0.0, 0.0);
        }
        let t_w = *trail_phase_words_ref(self);
        let b_w = load_q7_words_64(anchor_phases);
        let t_uni = is_word_array_uniform_64(&t_w);
        let b_uni = is_word_array_uniform_64(&b_w);
        let t_nz = if t_uni {
            t_w[0] != 0
        } else {
            (((t_w[0] | t_w[1]) | (t_w[2] | t_w[3])) | ((t_w[4] | t_w[5]) | (t_w[6] | t_w[7]))) != 0
        };
        let b_nz = if b_uni {
            b_w[0] != 0
        } else {
            (((b_w[0] | b_w[1]) | (b_w[2] | b_w[3])) | ((b_w[4] | b_w[5]) | (b_w[6] | b_w[7]))) != 0
        };
        let coherence = if t_nz & b_nz {
            coherence_from_word_arrays_with_both_uniform_flags_64(&t_w, t_uni, &b_w, b_uni)
        } else {
            0.0
        };
        let proximity_gain = branchless_clamp_f32(1.0 + coherence, 0.25, 2.0);
        let resonant_pull = branchless_flush_subnormal_f32(clean_energy * proximity_gain);
        let new_pull = cur_pull + resonant_pull;
        if resonant_pull > DEFAULT_ENTROPY_EPS {
            if cur_pull <= DEFAULT_ENTROPY_EPS {
                *trail_phase_words_mut(self) = b_w;
            } else {
                let ratio_f32 = branchless_clamp_f32(
                    (resonant_pull * crate::foundation::fast_inv_pos_normal_f32(new_pull)) * 128.0
                        + 0.5,
                    0.0,
                    128.0,
                );
                let s = (ratio_f32 as u8) as u64;
                if s >= 128 {
                    *trail_phase_words_mut(self) = b_w;
                } else if s > 0 {
                    let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                    let (out_w, _) = swar_slerp_words_with_uniform_flag_64(
                        &t_w,
                        t_uni,
                        &b_w,
                        b_uni,
                        s,
                        bias_restore,
                    );
                    *trail_phase_words_mut(self) = out_w;
                }
            }
        }
        self.attractor_pull_force = branchless_min_f32(new_pull, MAX_ATTRACTOR_PULL_FORCE);
        let is_resonant = (coherence > 0.25) & (resonant_pull > 0.0);
        let is_liminal = (coherence < -0.10) & (resonant_pull > 0.0);
        self.flags |= branchless_select_u32(is_resonant, TRAIL_FLAG_SYNERGY_RESONANCE, 0)
            | branchless_select_u32(is_liminal, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, 0);
        (resonant_pull, coherence)
    }

    /// Bridges the "unwritten space between the lines" (**Limina / Inter-Memory Hallucination**)
    /// from `prev_memory_trail` ($M_{t-1}$) to `self` ($M_t$) in a **single fused 8-word SWAR pass**
    /// (`0` branches, zero temporary array stack spills).
    #[inline]
    pub fn bridge_liminal_hallucinated_trail(
        &mut self,
        prev_memory_trail: &PointerTrail,
        interpolation_u7: u8,
        contradiction_threshold: f32,
    ) -> (f32, f32, bool) {
        let p = trail_phase_words_ref(prev_memory_trail);
        let s = trail_phase_words_ref(self);
        let p_uni = is_word_array_uniform_64(p);
        let both_uni = p_uni && is_word_array_uniform_64(s);
        let d0 = swar_q7_circular_diff_x8(s[0], p[0]);
        let safe_thresh =
            branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
        let t_s = branchless_select_u8(interpolation_u7 > 128, 128, interpolation_u7) as u64;
        let bias_restore = (128u64.wrapping_sub(t_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);

        let (coh, log_prob, contradiction, out_w, store_words) = if both_uni {
            if d0 == 0 {
                (1.0_f32, 0.0_f32, false, *s, false)
            } else {
                let c_val = coherence_from_uniform_diff_word_x8(d0);
                let lp = branchless_fast_transition_log_prob(c_val);
                let contr = c_val < -safe_thresh;
                let w0 = if contr {
                    swar_q7_neg_sat_x8(s[0])
                } else if t_s >= 128 {
                    s[0]
                } else if t_s == 0 {
                    p[0]
                } else {
                    swar_q7_circular_add_x8(
                        p[0],
                        swar_q7_scale_le128_with_bias_x8(d0, t_s, bias_restore),
                    )
                };
                (c_val, lp, contr, [w0; 8], w0 != s[0])
            }
        } else {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            {
                // SAFETY: `target_feature = "avx2"` is enabled; evaluates 64-head circular difference,
                // cosine coherence, and geodesic SLERP in two 256-bit YMM registers.
                unsafe {
                    use core::arch::x86_64::{
                        _mm256_max_epi8, _mm256_or_si256, _mm256_set1_epi8, _mm256_setzero_si256,
                        _mm256_sub_epi8, _mm256_testz_si256,
                    };
                    let (vp0, vp1) =
                        crate::foundation::avx2_words64_to_vec64_with_uniform(p, p_uni);
                    let (vs0, vs1) = crate::foundation::avx2_words64_to_vec64(s);
                    let vd0 = _mm256_sub_epi8(vs0, vp0);
                    let vd1 = _mm256_sub_epi8(vs1, vp1);
                    let vd_or = _mm256_or_si256(vd0, vd1);
                    if _mm256_testz_si256(vd_or, vd_or) != 0 {
                        (1.0_f32, 0.0_f32, false, *s, false)
                    } else {
                        let c_val = crate::foundation::avx2_coherence_from_diff_vec64(vd0, vd1);
                        let lp = branchless_fast_transition_log_prob(c_val);
                        let contr = c_val < -safe_thresh;
                        if contr {
                            let zero = _mm256_setzero_si256();
                            let min_m127 = _mm256_set1_epi8(-127);
                            let r0 = _mm256_sub_epi8(zero, _mm256_max_epi8(vs0, min_m127));
                            let r1 = _mm256_sub_epi8(zero, _mm256_max_epi8(vs1, min_m127));
                            (
                                c_val,
                                lp,
                                true,
                                crate::foundation::avx2_vec64_to_words64(r0, r1),
                                true,
                            )
                        } else if t_s >= 128 {
                            (c_val, lp, false, *s, false)
                        } else if t_s == 0 {
                            (c_val, lp, false, *p, true)
                        } else {
                            let (r0, r1) = crate::foundation::avx2_slerp_from_diff_vec64(
                                vp0, vp1, vd0, vd1, t_s,
                            );
                            (
                                c_val,
                                lp,
                                false,
                                crate::foundation::avx2_vec64_to_words64(r0, r1),
                                true,
                            )
                        }
                    }
                }
            }
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            {
                let d1 = swar_q7_circular_diff_x8(s[1], p[1]);
                let d2 = swar_q7_circular_diff_x8(s[2], p[2]);
                let d3 = swar_q7_circular_diff_x8(s[3], p[3]);
                let d4 = swar_q7_circular_diff_x8(s[4], p[4]);
                let d5 = swar_q7_circular_diff_x8(s[5], p[5]);
                let d6 = swar_q7_circular_diff_x8(s[6], p[6]);
                let d7 = swar_q7_circular_diff_x8(s[7], p[7]);

                if ((d0 | d1) | (d2 | d3) | (d4 | d5) | (d6 | d7)) == 0 {
                    (1.0_f32, 0.0_f32, false, *s, false)
                } else {
                    let c_val = crate::foundation::coherence_from_diff_words_64([
                        d0, d1, d2, d3, d4, d5, d6, d7,
                    ]);
                    let lp = branchless_fast_transition_log_prob(c_val);
                    let contr = c_val < -safe_thresh;

                    if contr {
                        let w_arr = [
                            swar_q7_neg_sat_x8(s[0]),
                            swar_q7_neg_sat_x8(s[1]),
                            swar_q7_neg_sat_x8(s[2]),
                            swar_q7_neg_sat_x8(s[3]),
                            swar_q7_neg_sat_x8(s[4]),
                            swar_q7_neg_sat_x8(s[5]),
                            swar_q7_neg_sat_x8(s[6]),
                            swar_q7_neg_sat_x8(s[7]),
                        ];
                        (c_val, lp, true, w_arr, true)
                    } else if t_s >= 128 {
                        (c_val, lp, false, *s, false)
                    } else if t_s == 0 {
                        (c_val, lp, false, *p, true)
                    } else {
                        let w_arr = [
                            swar_q7_circular_add_x8(
                                p[0],
                                swar_q7_scale_le128_with_bias_x8(d0, t_s, bias_restore),
                            ),
                            swar_q7_circular_add_x8(
                                p[1],
                                swar_q7_scale_le128_with_bias_x8(d1, t_s, bias_restore),
                            ),
                            swar_q7_circular_add_x8(
                                p[2],
                                swar_q7_scale_le128_with_bias_x8(d2, t_s, bias_restore),
                            ),
                            swar_q7_circular_add_x8(
                                p[3],
                                swar_q7_scale_le128_with_bias_x8(d3, t_s, bias_restore),
                            ),
                            swar_q7_circular_add_x8(
                                p[4],
                                swar_q7_scale_le128_with_bias_x8(d4, t_s, bias_restore),
                            ),
                            swar_q7_circular_add_x8(
                                p[5],
                                swar_q7_scale_le128_with_bias_x8(d5, t_s, bias_restore),
                            ),
                            swar_q7_circular_add_x8(
                                p[6],
                                swar_q7_scale_le128_with_bias_x8(d6, t_s, bias_restore),
                            ),
                            swar_q7_circular_add_x8(
                                p[7],
                                swar_q7_scale_le128_with_bias_x8(d7, t_s, bias_restore),
                            ),
                        ];
                        (c_val, lp, false, w_arr, true)
                    }
                }
            }
        };
        if store_words {
            *trail_phase_words_mut(self) = out_w;
        }

        self.prev_ptr = prev_memory_trail.current_ptr;
        self.step_count = prev_memory_trail.step_count.wrapping_add(1);
        let inherit_anchor = (self.target_anchor_id == 0) & (!contradiction);
        self.target_anchor_id = branchless_select_u32(
            inherit_anchor,
            prev_memory_trail.target_anchor_id,
            self.target_anchor_id,
        );
        self.cluster_id = branchless_select_u32(
            inherit_anchor & (prev_memory_trail.target_anchor_id != 0),
            prev_memory_trail.target_anchor_id,
            self.cluster_id,
        );

        let cur_ptr = TaggedPtr(self.current_ptr);
        let rep_ptr = cur_ptr
            .with_action_and_slot(
                ATTRACTOR_SLOT_REPULSION,
                cur_ptr.action_mode_flags() | ACTION_PRESET_REPULSION_ESCAPE,
            )
            .conjugate_phase();
        self.current_ptr = branchless_select_u64(contradiction, rep_ptr.raw(), cur_ptr.raw());

        let cur_rho = branchless_flush_subnormal_f32(branchless_relu_f32(self.stigmergic_density));
        let pos_rho = cur_rho + 0.25 * (1.0 + branchless_relu_f32(coh));
        let neg_rho = cur_rho * 0.5;
        self.stigmergic_density = branchless_min_f32(
            branchless_flush_subnormal_f32(branchless_select_f32(contradiction, neg_rho, pos_rho)),
            MAX_STIGMERGIC_DENSITY,
        );

        let prev_pull = branchless_flush_subnormal_f32(branchless_relu_f32(
            prev_memory_trail.attractor_pull_force,
        ));
        let cur_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(self.attractor_pull_force));
        let bridged_pull = cur_pull + 0.5 * prev_pull * branchless_clamp_f32(1.0 + coh, 0.25, 2.0);
        self.attractor_pull_force = branchless_min_f32(
            branchless_flush_subnormal_f32(branchless_select_f32(
                contradiction,
                cur_pull * 0.5,
                bridged_pull,
            )),
            MAX_ATTRACTOR_PULL_FORCE,
        );

        let is_resonant = (!contradiction) & (coh > 0.25);
        self.flags |= branchless_select_u32(is_resonant, TRAIL_FLAG_SYNERGY_RESONANCE, 0)
            | branchless_select_u32(
                contradiction,
                TRAIL_FLAG_REPULSION_ACTIVE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE,
                0,
            );
        (coh, log_prob, contradiction)
    }

    /// Links `self` ($M_{t-1}$) to `next_trail` ($M_t$, identified by `next_trail_id`) by setting
    /// `self.linked_trail_id = next_trail_id` and bridging their inter-memory **Limina** gap.
    #[inline(always)]
    pub fn link_with_next_hallucinated_trail(
        &mut self,
        next_trail_id: u32,
        next_trail: &mut PointerTrail,
        interpolation_u7: u8,
        contradiction_threshold: f32,
    ) -> (f32, f32, bool) {
        self.linked_trail_id = next_trail_id;
        next_trail.bridge_liminal_hallucinated_trail(
            self,
            interpolation_u7,
            contradiction_threshold,
        )
    }

    /// Synchronizes `self.flags` (`TRAIL_FLAG_PHASE_STEER_ACTIVE` and `TRAIL_FLAG_INERTIAL_SKIP_ACTIVE`)
    /// with the 5-bit `ACTION_MODE_*` flags embedded in `self.current_ptr` in branchless constant time.
    #[inline(always)]
    pub fn sync_flags_with_current_ptr(&mut self) {
        let mode = unpack_action_mode_from_tag(TaggedPtr(self.current_ptr).unpack_tag());
        let phase_steer_on = (mode & ACTION_MODE_PHASE_STEER) != 0;
        let skip_on = (mode & ACTION_MODE_LAYER_SKIP) != 0;
        let cleared =
            self.flags & !(TRAIL_FLAG_PHASE_STEER_ACTIVE | TRAIL_FLAG_INERTIAL_SKIP_ACTIVE);
        self.flags = cleared
            | branchless_select_u32(phase_steer_on, TRAIL_FLAG_PHASE_STEER_ACTIVE, 0)
            | branchless_select_u32(skip_on, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, 0);
    }
}

pub(crate) const BLOCK_NORM_CACHE_CAP: usize = 64;

/// Evaluates coarse supernode degrees $D_{ii} = \sum_{j \neq i} A_{ij}$ and returns
/// $\hat{\lambda}_{\max} = 2 \max_i D_{ii}$ directly from 256-bit PagedBlock bitmasks (Section 1.2 & Lemma 1.2).
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `block_masks` is empty or `degrees_out.len() < block_masks.len()`.
#[inline]
pub fn eval_stigmergic_block_degrees(
    block_masks: &[PagedBitmask],
    degrees_out: &mut AlignedSliceMut32<'_, f32>,
) -> LmnkResult<f32> {
    let n = block_masks.len();
    if n == 0 || degrees_out.len() < n {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    let deg_slice = &mut degrees_out.as_mut_slice()[..n];
    deg_slice.fill(0.0);

    let mut inv_norm_cache = [0.0_f32; BLOCK_NORM_CACHE_CAP];
    let cached_len = n.min(BLOCK_NORM_CACHE_CAP);
    for (slot, mask) in inv_norm_cache[..cached_len]
        .iter_mut()
        .zip(block_masks[..cached_len].iter())
    {
        *slot = branchless_popcount_rsqrt_u32(mask.popcount());
    }

    let mut max_deg = 0.0_f32;
    let cached_upper = cached_len;
    for i in 0..n {
        let mask_i = &block_masks[i];
        let inv_norm_i = if i < BLOCK_NORM_CACHE_CAP {
            inv_norm_cache[i]
        } else {
            branchless_popcount_rsqrt_u32(mask_i.popcount())
        };

        if inv_norm_i > 0.0 {
            let [m0, m1, m2, m3] = mask_i.bits;
            let mut acc_i0 = deg_slice[i];
            let mut acc_i1 = 0.0_f32;
            let mut j = i + 1;

            while j + 4 <= cached_upper {
                let [w0_a, w1_a, w2_a, w3_a] = block_masks[j].bits;
                let [w0_b, w1_b, w2_b, w3_b] = block_masks[j + 1].bits;
                let [w0_c, w1_c, w2_c, w3_c] = block_masks[j + 2].bits;
                let [w0_d, w1_d, w2_d, w3_d] = block_masks[j + 3].bits;

                let inter_a = ((m0 & w0_a).count_ones() + (m1 & w1_a).count_ones())
                    + ((m2 & w2_a).count_ones() + (m3 & w3_a).count_ones());
                let inter_b = ((m0 & w0_b).count_ones() + (m1 & w1_b).count_ones())
                    + ((m2 & w2_b).count_ones() + (m3 & w3_b).count_ones());
                let inter_c = ((m0 & w0_c).count_ones() + (m1 & w1_c).count_ones())
                    + ((m2 & w2_c).count_ones() + (m3 & w3_c).count_ones());
                let inter_d = ((m0 & w0_d).count_ones() + (m1 & w1_d).count_ones())
                    + ((m2 & w2_d).count_ones() + (m3 & w3_d).count_ones());

                let a_ij_a = (inter_a as f32) * (inv_norm_i * inv_norm_cache[j]);
                let a_ij_b = (inter_b as f32) * (inv_norm_i * inv_norm_cache[j + 1]);
                let a_ij_c = (inter_c as f32) * (inv_norm_i * inv_norm_cache[j + 2]);
                let a_ij_d = (inter_d as f32) * (inv_norm_i * inv_norm_cache[j + 3]);

                deg_slice[j] += a_ij_a;
                deg_slice[j + 1] += a_ij_b;
                deg_slice[j + 2] += a_ij_c;
                deg_slice[j + 3] += a_ij_d;
                acc_i0 += a_ij_a + a_ij_b;
                acc_i1 += a_ij_c + a_ij_d;
                j += 4;
            }
            while j < cached_upper {
                let [w0, w1, w2, w3] = block_masks[j].bits;
                let inter = ((m0 & w0).count_ones() + (m1 & w1).count_ones())
                    + ((m2 & w2).count_ones() + (m3 & w3).count_ones());
                let a_ij = (inter as f32) * (inv_norm_i * inv_norm_cache[j]);
                deg_slice[j] += a_ij;
                acc_i0 += a_ij;
                j += 1;
            }
            for j_tail in cached_upper.max(i + 1)..n {
                let a_ij = mask_i
                    .stigmergic_cosine_similarity_with_inv_norm(inv_norm_i, &block_masks[j_tail]);
                acc_i0 += a_ij;
                deg_slice[j_tail] += a_ij;
            }
            deg_slice[i] = acc_i0 + acc_i1;
        }
        max_deg = branchless_max_f32(max_deg, deg_slice[i]);
    }
    Ok(branchless_flush_subnormal_f32(2.0 * max_deg))
}

/// Detects cohomological obstruction $H^1(K; \mathcal{F})$ and records the offending phase
/// pattern into `bloom_filter` while returning the phase-conjugate anti-phase vector ($-\boldsymbol{\Delta\theta}$).
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if slice lengths mismatch or are empty.
/// Returns [`crate::liminal::LMNK_ERR_ENTROPY_SINGULARITY`] if residual norms are non-finite.
#[inline]
pub fn detect_and_record_obstruction_repulsion(
    cochain_residual_r: &AlignedSlice32<'_, f32>,
    exact_projection_delta0_phi: &AlignedSlice32<'_, f32>,
    obstruction_threshold: f32,
    active_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    bloom_filter: &mut ContrastivePhaseBloomFilter,
) -> LmnkResult<(f32, bool, [PhaseQ7; MAX_ATTENTION_HEADS])> {
    let (mean_obs, is_obstructed) = evaluate_cohomological_obstruction(
        cochain_residual_r,
        exact_projection_delta0_phi,
        obstruction_threshold,
    )?;
    if is_obstructed {
        bloom_filter.insert_repulsive_pattern(active_phases);
        Ok((mean_obs, true, branchless_q7_conjugate_64(active_phases)))
    } else {
        Ok((mean_obs, false, *active_phases))
    }
}

/// Evaluates the **Joint Probability Chain Rule Log-Likelihood**
/// $\ln P(M_1, \dots, M_T) = \sum_{t=1}^{T-1} \ln P(M_t \mid M_{t-1})$ and **$S^1$ Sheaf
/// Holonomy Obstruction Energy** ($H^1(K; \mathcal{F})$) across `trail_chain` in $\mathcal{O}(T)$ time.
///
/// Returns `(joint_log_prob_sum, holonomy_obstruction_energy, is_obstructed)`.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `trail_chain` is empty or `obstruction_threshold` is non-finite.
#[inline]
pub fn evaluate_trail_chain_cohomology_and_joint_log_prob(
    trail_chain: &[PointerTrail],
    obstruction_threshold: f32,
) -> LmnkResult<(f32, f32, bool)> {
    let len = trail_chain.len();
    if len == 0 || !branchless_is_finite_f32(obstruction_threshold) {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    if len == 1 {
        return Ok((0.0, 0.0, false));
    }

    let mut joint_log_prob = 0.0_f32;
    let mut coboundary_defect_sum = 0.0_f32;
    let first_w = *trail_phase_words_ref(&trail_chain[0]);
    let mut prev_w = first_w;
    for trail in &trail_chain[1..len] {
        let cur_w = *trail_phase_words_ref(trail);
        let (coh, log_p) = branchless_q7_transition_log_prob_words_64(&prev_w, &cur_w);
        joint_log_prob += log_p;
        coboundary_defect_sum += branchless_relu_f32(0.5 * (1.0 - coh));
        prev_w = cur_w;
    }

    let mean_step_defect = coboundary_defect_sum * fast_inv_len_f32(len - 1);
    let loop_coh = coherence_from_word_arrays_64(&first_w, &prev_w);
    let loop_defect = branchless_relu_f32(0.5 * (1.0 - loop_coh));
    let holonomy_energy = branchless_flush_subnormal_f32(0.5 * (mean_step_defect + loop_defect));
    let is_obstructed = holonomy_energy > branchless_relu_f32(obstruction_threshold);

    Ok((
        branchless_flush_subnormal_f32(joint_log_prob),
        holonomy_energy,
        is_obstructed,
    ))
}

/// Bridges all consecutive inter-memory gaps (**Limina**) across `trail_chain` via
/// [`PointerTrail::bridge_liminal_hallucinated_trail`], recording any self-contradictory
/// transitions ($H^1(K; \mathcal{F}) \neq 0$) into `bloom_filter`.
///
/// Returns `(total_joint_log_prob, repulsed_transition_count)`.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `trail_chain` is empty or `contradiction_threshold` is non-finite.
#[inline]
pub fn bridge_and_filter_liminal_trail_chain(
    trail_chain: &mut [PointerTrail],
    interpolation_u7: u8,
    contradiction_threshold: f32,
    bloom_filter: &mut ContrastivePhaseBloomFilter,
) -> LmnkResult<(f32, u32)> {
    let len = trail_chain.len();
    if len == 0 || !branchless_is_finite_f32(contradiction_threshold) {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    let mut total_log_prob = 0.0_f32;
    let mut repulsed_count = 0u32;

    for idx in 1..len {
        let (head, tail) = trail_chain.split_at_mut(idx);
        let prev_ref = &head[idx - 1];
        let cur_mut = &mut tail[0];
        let (_coh, log_p, repulsed) = cur_mut.bridge_liminal_hallucinated_trail(
            prev_ref,
            interpolation_u7,
            contradiction_threshold,
        );
        total_log_prob += log_p;
        if repulsed {
            bloom_filter.insert_repulsive_pattern_words_weighted(trail_phase_words_ref(cur_mut), 1);
            repulsed_count = repulsed_count.wrapping_add(1);
        }
    }
    Ok((
        branchless_flush_subnormal_f32(total_log_prob),
        repulsed_count,
    ))
}

/// Parameters for [`resolve_dyadic_qdpt_and_cascade_ptr`].
#[derive(Debug, Clone, Copy)]
pub struct QdptCascadeParams<'a> {
    /// Key weight scaling the dyadic ultrametric depth difference.
    pub w_key: f32,
    /// Previous quantized QDPT depth level.
    pub current_level: u32,
    /// Hysteresis deadband guarding depth transitions.
    pub hysteresis_band: f32,
    /// Per-level 48-bit address offsets (`len == phase_deltas.len()`).
    pub addr_offsets: &'a [u64],
    /// Per-level Q7 phase deltas (`len == addr_offsets.len()`).
    pub phase_deltas: &'a [PhaseQ7],
}

/// Evaluates 256-bit dyadic distance $d_{\mathcal{V}}(x, y) \in [0, 1)$, quantizes it into a
/// discrete QDPT level, and cascades `parent_ptr` in $\mathcal{O}(1)$.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `params.addr_offsets` is empty or its length does
/// not match `params.phase_deltas.len()`.
#[inline(always)]
pub fn resolve_dyadic_qdpt_and_cascade_ptr(
    parent_ptr: TaggedPtr,
    current_mask: &PagedBitmask,
    anchor_mask: &PagedBitmask,
    params: QdptCascadeParams<'_>,
) -> LmnkResult<(u32, f32, TaggedPtr)> {
    let QdptCascadeParams {
        w_key,
        current_level,
        hysteresis_band,
        addr_offsets: qdpt_addr_offsets,
        phase_deltas: qdpt_phase_deltas,
    } = params;
    let table_len = qdpt_addr_offsets.len();
    if table_len == 0 || table_len != qdpt_phase_deltas.len() {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    let max_level = (table_len - 1) as u32;
    let d_v = eval_bitmask_dyadic_ultrametric(current_mask, anchor_mask, w_key);
    let clamped_cur = branchless_select_u32(current_level <= max_level, current_level, max_level);
    let level =
        crate::foundation::quantize_depth_hysteresis(d_v, clamped_cur, max_level, hysteresis_band);
    let idx = (level as usize).min(table_len - 1);
    let cascaded = crate::foundation::cascade_tagged_ptr(
        parent_ptr,
        qdpt_addr_offsets[idx],
        qdpt_phase_deltas[idx],
    );
    Ok((level, d_v, cascaded))
}

#[cfg(test)]
pub(crate) use test_resonance_helpers::modulate_attention_scores_by_bayesian_trails_64;

#[cfg(test)]
mod test_resonance_helpers {
    use super::*;
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    use crate::foundation::Q7_FULL_COS_REF;
    use crate::foundation::lut_crc32::{
        MAX_BAYESIAN_STIGMERGIC_DENSITY, POPCOUNT_RSQRT_LUT_257_REF,
    };
    use crate::foundation::phase_ops::branchless_q7_unified_manifold_fold_words_with_flags_64;
    use crate::foundation::{
        ATTRACTOR_SLOT_ID_MASK, branchless_q7_inhibit_words_64,
        branchless_q7_synergy_fuse_words_64, branchless_rsqrt_f32, fast_ln_1p_bounded_f32,
        swar_or_avx2_q7_add_sat_words_64,
    };
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    use crate::foundation::{
        Q7_FULL_COS_REF, avx2_coherence_from_diff_vec64, avx2_words64_to_vec64_with_uniform,
    };
    use crate::liminal::{BitCut8, MAX_ATTRACTOR_DOMAINS, NUM_STIGMERGIC_GRANULES};
    use crate::stigmergy::coupled_slot_row_mask;

    impl PointerTrail {
        /// Applies **Order-Invariant (Commutative) Unified-Manifold Multi-Anchor Resonance** across an
        /// arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) and optional repulsive
        /// phase inhibitor (`repulsion_phases_and_strength`) on a **single unified phase manifold**
        /// using register-resident 8-word SWAR accumulators (`0` branches, `0 fdiv`).
        /// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) and optional
        /// repulsive phase inhibitor (`repulsion_phases_and_strength`) starting from pre-loaded 8-word
        /// SWAR registers `init_w` (`0` entry reload) and returning `(final_pull_force, mean_coherence,
        /// active_anchor_count, post_words)`.
        #[inline]
        pub(crate) fn apply_unified_manifold_anchor_slice_resonance_from_words(
            &mut self,
            init_w: [u64; NUM_STIGMERGIC_GRANULES],
            anchor_phases: &[[PhaseQ7; MAX_ATTENTION_HEADS]],
            anchor_base_energies: &[f32],
            repulsion_phases_and_strength: Option<(&[PhaseQ7; MAX_ATTENTION_HEADS], f32)>,
        ) -> (f32, f32, u32, [u64; NUM_STIGMERGIC_GRANULES]) {
            let (
                mut run_w,
                running_pull,
                mean_coh,
                dominant_anchor_id,
                active_anchors,
                any_resonant,
                any_liminal,
            ) = branchless_q7_unified_manifold_fold_words_with_flags_64(
                &init_w,
                self.attractor_pull_force,
                self.target_anchor_id,
                anchor_phases,
                anchor_base_energies,
            );
            let mut final_pull = branchless_min_f32(
                branchless_flush_subnormal_f32(running_pull),
                MAX_ATTRACTOR_PULL_FORCE,
            );
            let has_active = active_anchors > 0;

            self.target_anchor_id = dominant_anchor_id;
            let lane_tag = (dominant_anchor_id as u8) & ATTRACTOR_SLOT_ID_MASK;
            let cur_ptr = TaggedPtr(self.current_ptr);
            let updated_ptr = cur_ptr.with_action_and_slot(
                branchless_select_u8(has_active, lane_tag, cur_ptr.target_attractor_slot()),
                cur_ptr.action_mode_flags()
                    | branchless_select_u8(has_active, ACTION_MODE_PHASE_STEER, 0),
            );
            self.current_ptr = updated_ptr.raw();
            self.cluster_id =
                branchless_select_u32(has_active, dominant_anchor_id, self.cluster_id);
            self.flags |= branchless_select_u32(any_resonant, TRAIL_FLAG_SYNERGY_RESONANCE, 0)
                | branchless_select_u32(any_liminal, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, 0);

            if let Some((rep_phases, inh_strength)) = repulsion_phases_and_strength {
                let rep_w = load_q7_words_64(rep_phases);
                let raw_damped =
                    branchless_q7_inhibit_words_64(&mut run_w, final_pull, &rep_w, inh_strength);
                final_pull = branchless_min_f32(
                    branchless_flush_subnormal_f32(raw_damped),
                    MAX_ATTRACTOR_PULL_FORCE,
                );
            }

            *trail_phase_words_mut(self) = run_w;
            self.attractor_pull_force = final_pull;
            (final_pull, mean_coh, active_anchors, run_w)
        }

        /// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) on the single
        /// unified phase manifold and returns `(final_pull_force, mean_coherence, active_anchor_count, post_words)`
        /// directly in 8 `u64` SWAR registers (`0` exit reload).
        #[inline(always)]
        pub(crate) fn apply_unified_manifold_anchor_slice_resonance_to_words(
            &mut self,
            anchor_phases: &[[PhaseQ7; MAX_ATTENTION_HEADS]],
            anchor_base_energies: &[f32],
            repulsion_phases_and_strength: Option<(&[PhaseQ7; MAX_ATTENTION_HEADS], f32)>,
        ) -> (f32, f32, u32, [u64; NUM_STIGMERGIC_GRANULES]) {
            let init_w = *trail_phase_words_ref(self);
            self.apply_unified_manifold_anchor_slice_resonance_from_words(
                init_w,
                anchor_phases,
                anchor_base_energies,
                repulsion_phases_and_strength,
            )
        }

        /// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) and optional
        /// repulsive phase inhibitor (`repulsion_phases_and_strength`) on a **single unified phase manifold**
        /// using register-resident 8-word SWAR accumulators (`0` branches, `0 fdiv`).
        #[inline(always)]
        pub(crate) fn apply_unified_manifold_anchor_slice_resonance(
            &mut self,
            anchor_phases: &[[PhaseQ7; MAX_ATTENTION_HEADS]],
            anchor_base_energies: &[f32],
            repulsion_phases_and_strength: Option<(&[PhaseQ7; MAX_ATTENTION_HEADS], f32)>,
        ) -> (f32, f32, u32) {
            let (final_pull, mean_coh, active_anchors, _words) = self
                .apply_unified_manifold_anchor_slice_resonance_to_words(
                    anchor_phases,
                    anchor_base_energies,
                    repulsion_phases_and_strength,
                );
            (final_pull, mean_coh, active_anchors)
        }

        /// Register-resident evaluator for **8-Lane Coupled Neighborhood Synergy & Mutual Inhibition**
        /// starting from pre-loaded 8-word SWAR registers `acc_w` and returning
        /// `(damped_pull_force, mean_coherence, active_neighbor_count, post_words)`.
        #[inline]
        pub(crate) fn apply_neighborhood_synergy_from_bank_from_words(
            &mut self,
            mut acc_w: [u64; NUM_STIGMERGIC_GRANULES],
            slot_phase_bank: &[[PhaseQ7; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
            slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
            coupling_matrix_8x8: u64,
            synergy_gain: f32,
            inhibition_strength: f32,
        ) -> (f32, f32, u32, [u64; NUM_STIGMERGIC_GRANULES]) {
            let primary_slot = u32::from(self.target_attractor_slot() & ATTRACTOR_SLOT_ID_MASK);
            let p_idx = primary_slot as usize;
            let primary_phase = &slot_phase_bank[p_idx];
            let primary_w = load_q7_words_64(primary_phase);
            let primary_pull =
                branchless_flush_subnormal_f32(branchless_relu_f32(slot_weights[p_idx]));

            let exclude_mask = (1u8 << p_idx) | (1u8 << ATTRACTOR_SLOT_REPULSION);
            let mut neighbor_mask = BitCut8::split(
                coupled_slot_row_mask(coupling_matrix_8x8, primary_slot),
                exclude_mask,
            )
            .outside;

            // Keep `acc_w` and `cur_w` in 8 64-bit registers across all coupled neighbors (`0` intermediate
            // stack arrays)
            let mut cur_w = primary_w;
            let mut running_pull = primary_pull;
            let mut coh_sum = 0.0_f32;
            let mut active_neighbors = 0u32;
            let mut any_liminal = false;

            while neighbor_mask != 0 {
                let sec_idx = neighbor_mask.trailing_zeros() as usize;
                neighbor_mask &= neighbor_mask - 1;

                let sec_pull =
                    branchless_flush_subnormal_f32(branchless_relu_f32(slot_weights[sec_idx]));
                if sec_pull > 1.0e-9 {
                    let sec_w = load_q7_words_64(&slot_phase_bank[sec_idx]);
                    let (fw, syn_pull, coh) = branchless_q7_synergy_fuse_words_64(
                        &cur_w,
                        running_pull,
                        &sec_w,
                        sec_pull,
                        synergy_gain,
                    );
                    cur_w = fw;
                    running_pull = syn_pull;
                    coh_sum += coh;
                    active_neighbors += 1;
                    any_liminal |= coh < -0.10;
                }
            }

            if (active_neighbors > 0) | (primary_pull > 1.0e-9) {
                swar_or_avx2_q7_add_sat_words_64(&mut acc_w, &cur_w);
            }
            let mut total_pull =
                branchless_flush_subnormal_f32(branchless_relu_f32(self.attractor_pull_force))
                    + running_pull;

            total_pull = branchless_min_f32(
                branchless_flush_subnormal_f32(total_pull),
                MAX_ATTRACTOR_PULL_FORCE,
            );
            let safe_neighbors = branchless_select_u32(active_neighbors > 0, active_neighbors, 1);
            let mean_coh = branchless_select_f32(
                active_neighbors > 0,
                coh_sum * fast_inv_len_f32(safe_neighbors as usize),
                branchless_select_f32(primary_pull > 1.0e-9, 1.0, 0.0),
            );
            let is_resonant = (active_neighbors > 0) & (mean_coh > 0.25);
            self.flags |= branchless_select_u32(is_resonant, TRAIL_FLAG_SYNERGY_RESONANCE, 0)
                | branchless_select_u32(any_liminal, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, 0);

            if inhibition_strength > 0.0 {
                let rep_idx = usize::from(ATTRACTOR_SLOT_REPULSION);
                let rep_weight = branchless_relu_f32(slot_weights[rep_idx]);
                let eff_inh = branchless_select_f32(rep_weight > 1.0e-9, inhibition_strength, 0.0);
                let rep_words = load_q7_words_64(&slot_phase_bank[rep_idx]);
                let raw_damped =
                    branchless_q7_inhibit_words_64(&mut acc_w, total_pull, &rep_words, eff_inh);
                total_pull = branchless_min_f32(
                    branchless_flush_subnormal_f32(raw_damped),
                    MAX_ATTRACTOR_PULL_FORCE,
                );
            }

            *trail_phase_words_mut(self) = acc_w;
            self.attractor_pull_force = total_pull;
            (total_pull, mean_coh, active_neighbors, acc_w)
        }

        /// Evaluates **8-Lane Coupled Neighborhood Synergy & Mutual Inhibition** and returns both
        /// `(damped_pull_force, mean_coherence, active_neighbor_count, post_words)` in registers (`0` exit reload).
        #[inline(always)]
        pub(crate) fn apply_neighborhood_synergy_from_bank_to_words(
            &mut self,
            slot_phase_bank: &[[PhaseQ7; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
            slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
            coupling_matrix_8x8: u64,
            synergy_gain: f32,
            inhibition_strength: f32,
        ) -> (f32, f32, u32, [u64; NUM_STIGMERGIC_GRANULES]) {
            let acc_w = *trail_phase_words_ref(self);
            self.apply_neighborhood_synergy_from_bank_from_words(
                acc_w,
                slot_phase_bank,
                slot_weights,
                coupling_matrix_8x8,
                synergy_gain,
                inhibition_strength,
            )
        }

        /// Unified single-source-of-truth evaluator for **8-Lane Coupled Neighborhood Synergy & Mutual Inhibition**
        /// across `slot_phase_bank` (`[[PhaseQ7; 64]; 8]`) and `slot_weights` (`[f32; 8]`).
        #[inline(always)]
        pub(crate) fn apply_neighborhood_synergy_from_bank(
            &mut self,
            slot_phase_bank: &[[PhaseQ7; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
            slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
            coupling_matrix_8x8: u64,
            synergy_gain: f32,
            inhibition_strength: f32,
        ) -> (f32, f32, u32) {
            let (total_pull, mean_coh, active_neighbors, _words) = self
                .apply_neighborhood_synergy_from_bank_to_words(
                    slot_phase_bank,
                    slot_weights,
                    coupling_matrix_8x8,
                    synergy_gain,
                    inhibition_strength,
                );
            (total_pull, mean_coh, active_neighbors)
        }
    }

    #[cfg(test)]
    /// Directly modulates pre-Softmax candidate memory/key Attention scores (`attention_scores[..M]`)
    /// in-place by the full **Bayes-Attention Stigmergic Transition Score** (`0` branches, `0 fdiv`),
    /// accumulating coherence along the top-ranked trail's pointer chain while evaluating 64-head Q7
    /// similarity via `1`-cycle register shifts when either vector is uniform across all 64 heads.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn modulate_attention_scores_by_bayesian_trails_64(
        attention_scores: &mut [f32],
        query_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
        key_anchor_phases: &[[PhaseQ7; MAX_ATTENTION_HEADS]],
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
            // SAFETY: `target_feature = "avx2"` is checked by `#[cfg]`, and `q` has 8 `u64` words.
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
                            coherence_from_word_arrays_with_both_uniform_flags_64(
                                &q, true, &kw, true,
                            )
                        }
                    } else {
                        // SAFETY: `target_feature = "avx2"` is checked by `#[cfg]`, and
                        // `key_anchor_phases[idx]` is a 64-byte slice (`_mm256_loadu_si256` allows unaligned
                        // loads).
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
                    // SAFETY: `target_feature = "avx2"` is checked by `#[cfg]`, and `key_anchor_phases[idx]` is
                    // 64 bytes (`2 * __m256i`).
                    unsafe {
                        use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_sub_epi8};
                        let (vq0, vq1) = vq_vecs
                            .unwrap_or_else(|| avx2_words64_to_vec64_with_uniform(&q, false));
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
                    coherence_from_word_arrays_with_both_uniform_flags_64(
                        &q,
                        q_is_uniform,
                        &kw,
                        false,
                    )
                }
            };
            let attn_delta = branchless_flush_subnormal_f32(eff_interf_scale * coh);
            let clean_rho = branchless_clamp_f32(
                branchless_flush_subnormal_f32(branchless_relu_f32(key_stigmergic_densities[idx])),
                0.0,
                MAX_BAYESIAN_STIGMERGIC_DENSITY,
            );
            let prior_delta = branchless_flush_subnormal_f32(
                clean_prior_gain * fast_ln_1p_bounded_f32(clean_rho),
            );
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
}
