// name: src/core/src/stigmergy/anchor/routing.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Coarse-to-fine hierarchical routing, Top-2 liminal bifurcation routing, and ensemble
//! folding for [`PhaseAnchorRecord`].

use crate::foundation::{
    branchless_abs_f32, branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_min_f32,
    branchless_phase_proximity_gravity_f32, branchless_popcount_rsqrt_u32,
    branchless_q7_default_gravity_from_diff_u8, branchless_relu_f32, branchless_rsqrt_f32,
    coherence_above_floor_from_word_arrays_64,
    coherence_from_word_arrays_with_both_uniform_flags_64, fast_inv_len_f32,
    fast_inv_pos_normal_f32, fast_ln_1p_bounded_f32, is_word_array_uniform_64, load_q7_words_64,
    pack_action_slot_tag, store_q7_words_64, swar_q7_mean_i8_from_words_with_uniform_flag_64,
    swar_slerp_words_with_uniform_flag_64, swar_words_equal_64, TaggedPtr,
    DEFAULT_PHASE_CLAMP_MAX_RAD, DEFAULT_PHASE_GRAVITY_EPS, Q7_FULL_COS_REF, SWAR_SPLAT_BYTE_64,
};
use crate::liminal::{
    branchless_select_f32, branchless_select_u32, PhaseQ7, DEFAULT_ENTROPY_EPS,
    MAX_ATTENTION_HEADS, NUM_STIGMERGIC_GRANULES,
};
use crate::stigmergy::trail::{trail_phase_words_mut, trail_phase_words_ref, PointerTrail};
use crate::stigmergy::{
    MAX_ATTRACTOR_PULL_FORCE, MAX_STIGMERGIC_DENSITY, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE,
    TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_FLAG_SYNERGY_RESONANCE,
};

use super::PhaseAnchorRecord;

impl PhaseAnchorRecord {
    /// Evaluates macro distribution diagnostics across `centroids` in $\mathcal{O}(K^2)$ SWAR:
    /// returns `(active_clusters, total_samples, mean_resultant_r_bar, mean_von_mises_kappa, max_pairwise_coherence)`.
    #[inline]
    #[must_use]
    pub fn diagnose_macro_distribution_64(centroids: &[Self]) -> (u32, u32, f32, f32, f32) {
        let k_len = centroids.len();
        if k_len == 0 {
            return (0, 0, 0.0, 0.0, 0.0);
        }
        let mut active_cnt = 0u32;
        let mut tot_samples = 0u32;
        let mut sum_r = 0.0_f32;
        let mut sum_kappa = 0.0_f32;
        let mut max_pair_coh = -1.0_f32;
        let mut has_pair = false;

        let mut i = 0usize;
        while i < k_len {
            let rec_i = &centroids[i];
            if rec_i.sample_count > 0 && rec_i.phase_gravity_mass > DEFAULT_ENTROPY_EPS {
                active_cnt += 1;
                tot_samples = tot_samples.saturating_add(rec_i.sample_count);
                sum_r += rec_i.resultant_r_bar;
                sum_kappa += rec_i.von_mises_kappa;
                let w_i = rec_i.phase_words_ref();
                let i_uni = is_word_array_uniform_64(w_i);
                let mut j = i + 1;
                while j < k_len {
                    let rec_j = &centroids[j];
                    if rec_j.sample_count > 0 && rec_j.phase_gravity_mass > DEFAULT_ENTROPY_EPS {
                        let w_j = rec_j.phase_words_ref();
                        let j_uni = if i_uni {
                            is_word_array_uniform_64(w_j)
                        } else {
                            false
                        };
                        let floor = if has_pair { max_pair_coh } else { -2.0 };
                        let coh = coherence_above_floor_from_word_arrays_64(
                            w_i, i_uni, w_j, j_uni, floor,
                        );
                        max_pair_coh = branchless_select_f32(
                            (!has_pair) | (coh > max_pair_coh),
                            coh,
                            max_pair_coh,
                        );
                        has_pair = true;
                    }
                    j += 1;
                }
            }
            i += 1;
        }
        if active_cnt == 0 {
            return (0, 0, 0.0, 0.0, 0.0);
        }
        let inv_act = fast_inv_pos_normal_f32(active_cnt as f32);
        let mean_r = branchless_clamp_f32(sum_r * inv_act, 0.0, 1.0);
        let mean_k = branchless_flush_subnormal_f32(sum_kappa * inv_act);
        let pair_coh = branchless_select_f32(has_pair, max_pair_coh, 0.0);
        (active_cnt, tot_samples, mean_r, mean_k, pair_coh)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn apply_macro_centroid_pull_to_trail_64(
        trail: &mut PointerTrail,
        trail_w: &[u64; NUM_STIGMERGIC_GRANULES],
        trail_is_uniform: bool,
        target_w: &[u64; NUM_STIGMERGIC_GRANULES],
        target_is_uniform: bool,
        winner: &Self,
        anchor_id: u32,
        eff_coh: f32,
        tot_grav: f32,
        pull_step_u7: u8,
        extra_flags: u32,
    ) -> [u64; NUM_STIGMERGIC_GRANULES] {
        let s_u64 = pull_step_u7.min(128) as u64;
        let bias_restore = (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let (bridged_w, bridged_uni) = swar_slerp_words_with_uniform_flag_64(
            trail_w,
            trail_is_uniform,
            target_w,
            target_is_uniform,
            s_u64,
            bias_restore,
        );
        if pull_step_u7 > 0 && !swar_words_equal_64(trail_w, &bridged_w) {
            *trail_phase_words_mut(trail) = bridged_w;
        }
        trail.target_anchor_id = anchor_id;
        trail.cluster_id = anchor_id;
        trail.attractor_pull_force = branchless_min_f32(
            trail.attractor_pull_force + 0.5 * tot_grav,
            MAX_ATTRACTOR_PULL_FORCE,
        );
        trail.stigmergic_density = branchless_min_f32(
            trail.stigmergic_density + 0.20 * branchless_relu_f32(eff_coh),
            MAX_STIGMERGIC_DENSITY,
        );
        if winner.mapped_section_offset48 != 0 {
            let tag_q7 = swar_q7_mean_i8_from_words_with_uniform_flag_64(&bridged_w, bridged_uni);
            let packed_tag = pack_action_slot_tag(winner.dominant_slot, winner.action_mode_flags);
            trail.prev_ptr = trail.current_ptr;
            trail.current_ptr =
                TaggedPtr::pack(winner.mapped_section_offset48, packed_tag, tag_q7).raw();
        }
        trail.flags |=
            TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_CRYSTALLIZED_CANDIDATE | extra_flags;
        bridged_w
    }
    pub(crate) fn route_coarse_to_fine_trail_from_preloaded_words_64(
        centroids: &[Self],
        trail: &mut PointerTrail,
        trail_w: [u64; NUM_STIGMERGIC_GRANULES],
        pull_step_u7: u8,
    ) -> (
        usize,
        u32,
        f32,
        f32,
        [u64; NUM_STIGMERGIC_GRANULES],
        [u64; NUM_STIGMERGIC_GRANULES],
    ) {
        let (
            best_idx,
            best_coh,
            best_mass,
            _empty_opt,
            _weakest_idx,
            win_w,
            trail_uni,
            win_uni,
            best_byte_diff,
        ) = Self::find_nearest_toroidal_centroid_with_cache_64(centroids, &trail_w, None);
        if centroids.is_empty()
            || centroids[best_idx].sample_count == 0
            || best_mass <= DEFAULT_ENTROPY_EPS
        {
            return (
                0,
                trail.target_anchor_id,
                0.0,
                0.0,
                trail_w,
                [0u64; NUM_STIGMERGIC_GRANULES],
            );
        }

        let best_grav = match best_byte_diff {
            Some(diff_u8) => branchless_q7_default_gravity_from_diff_u8(best_mass, diff_u8),
            None => branchless_phase_proximity_gravity_f32(
                best_mass,
                best_coh,
                DEFAULT_PHASE_GRAVITY_EPS,
            ),
        };
        let winner = &centroids[best_idx];
        let anchor_id = if winner.anchor_id != 0 {
            winner.anchor_id
        } else {
            (best_idx as u32) + 1
        };
        let bridged_w = Self::apply_macro_centroid_pull_to_trail_64(
            trail,
            &trail_w,
            trail_uni,
            winner.phase_words_ref(),
            win_uni,
            winner,
            anchor_id,
            best_coh,
            best_grav,
            pull_step_u7,
            0,
        );
        (best_idx, anchor_id, best_coh, best_grav, bridged_w, win_w)
    }

    #[inline(always)]
    pub(crate) fn route_coarse_to_fine_trail_with_words_64(
        centroids: &[Self],
        trail: &mut PointerTrail,
        pull_step_u7: u8,
    ) -> (
        usize,
        u32,
        f32,
        f32,
        [u64; NUM_STIGMERGIC_GRANULES],
        [u64; NUM_STIGMERGIC_GRANULES],
    ) {
        let trail_w = *trail_phase_words_ref(trail);
        Self::route_coarse_to_fine_trail_from_preloaded_words_64(
            centroids,
            trail,
            trail_w,
            pull_step_u7,
        )
    }

    /// **Coarse-to-Fine Macro-to-Micro Routing ($\mathcal{O}(K)$)**:
    /// Returns `(selected_cluster_idx, selected_anchor_id, centroid_coherence, gravity_pull)`.
    #[inline]
    pub fn route_coarse_to_fine_trail_64(
        centroids: &[Self],
        trail: &mut PointerTrail,
        pull_step_u7: u8,
    ) -> (usize, u32, f32, f32) {
        let (best_idx, anchor_id, best_coh, best_grav, _bridged_w, _win_w) =
            Self::route_coarse_to_fine_trail_with_words_64(centroids, trail, pull_step_u7);
        (best_idx, anchor_id, best_coh, best_grav)
    }

    #[inline]
    pub(crate) fn route_top2_liminal_centroids_from_preloaded_words_64(
        centroids: &[Self],
        trail: &mut PointerTrail,
        trail_w: [u64; NUM_STIGMERGIC_GRANULES],
        pull_step_u7: u8,
        liminal_gravity_ratio_thresh: f32,
    ) -> (
        usize,
        usize,
        u32,
        u32,
        f32,
        f32,
        bool,
        [u64; NUM_STIGMERGIC_GRANULES],
        [u64; NUM_STIGMERGIC_GRANULES],
        [u64; NUM_STIGMERGIC_GRANULES],
    ) {
        let k_len = centroids.len();
        if k_len == 0 {
            return (
                0,
                0,
                trail.target_anchor_id,
                0,
                0.0,
                0.0,
                false,
                trail_w,
                trail_w,
                trail_w,
            );
        }

        let trail_is_uniform = is_word_array_uniform_64(&trail_w);
        let trail_w0 = trail_w[0];
        let trail_b0 = (trail_w0 & 0xFF) as u8;
        let trail_is_byte_uniform = trail_is_uniform & (trail_w0 == trail_w0.rotate_right(8));
        let mut idx1 = 0usize;
        let mut coh1 = -2.0_f32;
        let mut grav1 = 0.0_f32;
        let mut uni1 = false;
        let mut idx2 = 0usize;
        let mut coh2 = -2.0_f32;
        let mut grav2 = 0.0_f32;
        let mut uni2 = false;
        let mut active_found = 0u32;

        let mut i = 0usize;
        while i < k_len {
            let rec = &centroids[i];
            let mass = rec.phase_gravity_mass;
            if (rec.sample_count > 0) & (mass > DEFAULT_ENTROPY_EPS) {
                let rec_w = rec.phase_words_ref();
                let rw0 = rec_w[0];
                let rb0 = (rw0 & 0xFF) as u8;
                let rec_uni = if trail_is_uniform {
                    is_word_array_uniform_64(rec_w)
                } else {
                    false
                };
                let rec_byte_uni = rec_uni
                    & trail_is_byte_uniform
                    & ((rw0 == trail_w0) | (rw0 == rw0.rotate_right(8)));
                let (coh, grav) = if rec_uni & (rw0 == trail_w0) {
                    (1.0, branchless_q7_default_gravity_from_diff_u8(mass, 0))
                } else if rec_byte_uni {
                    let d_u8 = trail_b0.wrapping_sub(rb0);
                    (
                        Q7_FULL_COS_REF[d_u8 as usize],
                        branchless_q7_default_gravity_from_diff_u8(mass, d_u8),
                    )
                } else {
                    let floor = if (active_found >= 2) & (grav2 > mass) {
                        -0.50
                    } else {
                        -2.0
                    };
                    let c_val = coherence_above_floor_from_word_arrays_64(
                        &trail_w,
                        trail_is_uniform,
                        rec_w,
                        rec_uni,
                        floor,
                    );
                    let g_val = if c_val > -1.5 {
                        branchless_phase_proximity_gravity_f32(
                            mass,
                            c_val,
                            DEFAULT_PHASE_GRAVITY_EPS,
                        )
                    } else {
                        0.0
                    };
                    (c_val, g_val)
                };
                if coh > -1.5 {
                    let better1 =
                        (active_found == 0) | (grav > grav1) | ((grav == grav1) & (coh > coh1));
                    if better1 {
                        idx2 = idx1;
                        coh2 = coh1;
                        grav2 = grav1;
                        uni2 = uni1;
                        idx1 = i;
                        coh1 = coh;
                        grav1 = grav;
                        uni1 = rec_uni;
                    } else if (active_found == 1)
                        | (grav > grav2)
                        | ((grav == grav2) & (coh > coh2))
                    {
                        idx2 = i;
                        coh2 = coh;
                        grav2 = grav;
                        uni2 = rec_uni;
                    }
                }
                active_found += 1;
            }
            i += 1;
        }

        if active_found == 0 {
            return (
                0,
                0,
                trail.target_anchor_id,
                0,
                0.0,
                0.0,
                false,
                trail_w,
                trail_w,
                trail_w,
            );
        }

        let w1 = *centroids[idx1].phase_words_ref();
        let safe_ratio = branchless_clamp_f32(liminal_gravity_ratio_thresh, 0.10, 0.99);
        let mass_ratio_ok =
            centroids[idx2].phase_gravity_mass >= centroids[idx1].phase_gravity_mass * safe_ratio;
        let is_liminal = (active_found >= 2)
            & (grav2 > DEFAULT_ENTROPY_EPS)
            & ((grav2 >= grav1 * safe_ratio) | (mass_ratio_ok & (coh2 >= safe_ratio)))
            & (coh2 > -0.25);

        let w1_uni = if trail_is_uniform {
            uni1
        } else {
            is_word_array_uniform_64(&w1)
        };
        let (target_w, target_uni, w2_opt) = if is_liminal {
            let w2 = *centroids[idx2].phase_words_ref();
            let w2_uni = if trail_is_uniform {
                uni2
            } else if w1_uni {
                is_word_array_uniform_64(&w2)
            } else {
                false
            };
            let tot_g = grav1 + grav2;
            let safe_tot = branchless_select_f32(tot_g > DEFAULT_ENTROPY_EPS, tot_g, 1.0);
            let w2_ratio =
                branchless_clamp_f32(grav2 * fast_inv_pos_normal_f32(safe_tot), 0.10, 0.50);
            let s_lim = ((w2_ratio * 128.0 + 0.5) as u8).min(64) as u64;
            let bias_lim = (128u64.wrapping_sub(s_lim)).wrapping_mul(SWAR_SPLAT_BYTE_64);
            let (tw, t_uni) =
                swar_slerp_words_with_uniform_flag_64(&w1, w1_uni, &w2, w2_uni, s_lim, bias_lim);
            (tw, t_uni, w2)
        } else {
            (w1, w1_uni, [0u64; NUM_STIGMERGIC_GRANULES])
        };

        let win1 = &centroids[idx1];
        let aid1 = if win1.anchor_id != 0 {
            win1.anchor_id
        } else {
            (idx1 as u32) + 1
        };
        let aid2 = if is_liminal {
            let win2 = &centroids[idx2];
            if win2.anchor_id != 0 {
                win2.anchor_id
            } else {
                (idx2 as u32) + 1
            }
        } else {
            0
        };

        let tot_grav = branchless_select_f32(is_liminal, grav1 + 0.5 * grav2, grav1);
        let eff_coh = branchless_select_f32(is_liminal, 0.5 * (coh1 + coh2), coh1);
        let extra_flags = branchless_select_u32(is_liminal, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, 0);
        let target_ref = if is_liminal {
            &target_w
        } else {
            win1.phase_words_ref()
        };
        let bridged_w = Self::apply_macro_centroid_pull_to_trail_64(
            trail,
            &trail_w,
            trail_is_uniform,
            target_ref,
            target_uni,
            win1,
            aid1,
            eff_coh,
            tot_grav,
            pull_step_u7,
            extra_flags,
        );
        let out_w2 = if is_liminal { w2_opt } else { bridged_w };

        (
            idx1, idx2, aid1, aid2, eff_coh, tot_grav, is_liminal, bridged_w, w1, out_w2,
        )
    }
    pub(crate) fn route_top2_liminal_centroids_with_words_64(
        centroids: &[Self],
        trail: &mut PointerTrail,
        pull_step_u7: u8,
        liminal_gravity_ratio_thresh: f32,
    ) -> (
        usize,
        usize,
        u32,
        u32,
        f32,
        f32,
        bool,
        [u64; NUM_STIGMERGIC_GRANULES],
        [u64; NUM_STIGMERGIC_GRANULES],
        [u64; NUM_STIGMERGIC_GRANULES],
    ) {
        let trail_w = *trail_phase_words_ref(trail);
        Self::route_top2_liminal_centroids_from_preloaded_words_64(
            centroids,
            trail,
            trail_w,
            pull_step_u7,
            liminal_gravity_ratio_thresh,
        )
    }

    /// **Top-2 Liminal Bifurcation Macro Routing ($\mathcal{O}(K)$ 64-Bit SWAR)**:
    /// Evaluates the Top-2 nearest macro centroids $(k_1^*, k_2^*)$ on $SO(2)^{64}$ in a single pass.
    /// When the secondary centroid's gravitational pull satisfies $G_2 \ge \tau_{\mathrm{limina}} G_1$
    /// (e.g. [`super::DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO`]), interpolates along the shortest $S^1$
    /// geodesic between both centroids (bridging the *Limina* inter-memory boundary) and stamps
    /// `TRAIL_FLAG_LIMINAL_BEAT_ACTIVE | TRAIL_FLAG_SYNERGY_RESONANCE`.
    ///
    /// Returns `(primary_idx, secondary_idx, primary_anchor_id, secondary_anchor_id, blended_coherence, total_gravity, is_liminal_bifurcation)`.
    #[inline]
    pub fn route_top2_liminal_centroids_64(
        centroids: &[Self],
        trail: &mut PointerTrail,
        pull_step_u7: u8,
        liminal_gravity_ratio_thresh: f32,
    ) -> (usize, usize, u32, u32, f32, f32, bool) {
        let (idx1, idx2, aid1, aid2, eff_coh, tot_grav, is_liminal, _bw, _w1, _w2) =
            Self::route_top2_liminal_centroids_with_words_64(
                centroids,
                trail,
                pull_step_u7,
                liminal_gravity_ratio_thresh,
            );
        (idx1, idx2, aid1, aid2, eff_coh, tot_grav, is_liminal)
    }

    /// **Coarse-to-Fine Macro-to-Micro Routing ($\mathcal{O}(K) + \mathcal{O}(M_{\mathrm{local}})$)**:
    /// 1. Executes the $\mathcal{O}(K)$ Online Toroidal $k$-Means E-step to identify the winning
    ///    macro centroid `win_k` and stamps `src_trail.cluster_id = macro_anchor_id`.
    /// 2. Evaluates `candidate_trails` with a macro-cluster affinity bonus (`cluster_bonus_gain`)
    ///    for trails belonging to `macro_anchor_id` (or geometrically aligned with `centroids[win_k]`),
    ///    links the winning candidate into `src_trail.linked_trail_id`, and returns
    ///    `(winning_centroid_idx, winning_trail_id, best_score, best_coherence, found_valid)`.
    #[inline]
    pub fn route_coarse_to_fine_candidates_64(
        centroids: &[Self],
        src_trail: &mut PointerTrail,
        candidate_trails: &[PointerTrail],
        head_dim: u32,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        cluster_bonus_gain: f32,
        contradiction_threshold: f32,
    ) -> (usize, u32, f32, f32, bool) {
        let self_w = *trail_phase_words_ref(src_trail);
        let (
            win_k,
            _win_coh,
            _win_grav,
            _empty_opt,
            _weak_idx,
            win_w,
            self_is_uniform,
            win_uni_from_estep,
            _,
        ) = Self::find_nearest_toroidal_centroid_with_cache_64(centroids, &self_w, None);
        let has_win_centroid = !centroids.is_empty() && centroids[win_k].sample_count > 0;

        let macro_anchor_id = if has_win_centroid {
            let aid = centroids[win_k].anchor_id;
            if aid != 0 {
                aid
            } else {
                (win_k as u32) + 1
            }
        } else {
            src_trail.cluster_id
        };
        if macro_anchor_id != 0 {
            src_trail.cluster_id = macro_anchor_id;
            src_trail.target_anchor_id = macro_anchor_id;
        }

        if candidate_trails.is_empty() {
            return (win_k, src_trail.linked_trail_id, 0.0, 0.0, false);
        }

        let safe_d = branchless_select_u32(head_dim > 0, head_dim, 128);
        let inv_sqrt_d = if safe_d <= 256 {
            branchless_popcount_rsqrt_u32(safe_d)
        } else {
            branchless_rsqrt_f32(safe_d as f32)
        };
        let phase_scale =
            branchless_flush_subnormal_f32(branchless_relu_f32(phase_interference_gain))
                * inv_sqrt_d;
        let clean_eta = branchless_flush_subnormal_f32(branchless_relu_f32(stigmergic_prior_gain));
        let clean_link = branchless_flush_subnormal_f32(branchless_relu_f32(pointer_link_bonus));
        let clean_cluster_bonus =
            branchless_flush_subnormal_f32(branchless_relu_f32(cluster_bonus_gain));
        let safe_thresh =
            branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
        let cur_linked_id = src_trail.linked_trail_id;
        let self_w0 = self_w[0];
        let self_b0 = (self_w0 & 0xFF) as u8;
        let self_is_byte_uniform = self_is_uniform & (self_w0 == self_w0.rotate_right(8));
        let mut win_meta_evaluated = false;
        let mut win_eq_self = false;
        let mut win_is_uniform = false;

        let mut best_id = cur_linked_id;
        let mut best_score = -1.0e30_f32;
        let mut best_coh = 0.0_f32;
        let mut found_valid = false;

        let mut idx = 0usize;
        while idx < candidate_trails.len() {
            let cand = &candidate_trails[idx];
            if (cand.flags & TRAIL_FLAG_REPULSION_ACTIVE) != 0 {
                idx += 1;
                continue;
            }
            let cand_id = idx as u32;
            let is_linked = (cur_linked_id != 0) & (cur_linked_id == cand_id);
            let link_term = branchless_select_f32(is_linked, clean_link, 0.0);
            let in_macro_cluster = (macro_anchor_id != 0)
                & ((cand.cluster_id == macro_anchor_id)
                    | (cand.target_anchor_id == macro_anchor_id));
            let is_unstamped_for_geo = (!in_macro_cluster)
                & has_win_centroid
                & (clean_cluster_bonus > 0.0)
                & (cand.cluster_id == 0)
                & (cand.target_anchor_id == 0);
            let max_macro_bonus = branchless_select_f32(
                in_macro_cluster | is_unstamped_for_geo,
                clean_cluster_bonus,
                0.0,
            );
            let clean_rho = branchless_clamp_f32(
                branchless_relu_f32(cand.stigmergic_density),
                0.0,
                MAX_STIGMERGIC_DENSITY,
            );
            let prior_term = clean_eta * fast_ln_1p_bounded_f32(clean_rho);
            let non_phase_upper = prior_term + link_term + max_macro_bonus;
            if found_valid && (phase_scale + non_phase_upper <= best_score) {
                idx += 1;
                continue;
            }

            let cand_w = trail_phase_words_ref(cand);
            let cand_is_uniform = if self_is_uniform {
                is_word_array_uniform_64(cand_w)
            } else {
                false
            };
            let coh = if cand_is_uniform {
                let cw0 = cand_w[0];
                if cw0 == self_w0 {
                    1.0
                } else if self_is_byte_uniform && (cw0 == cw0.rotate_right(8)) {
                    Q7_FULL_COS_REF[self_b0.wrapping_sub(cw0 as u8) as usize]
                } else {
                    coherence_from_word_arrays_with_both_uniform_flags_64(
                        &self_w, true, cand_w, true,
                    )
                }
            } else {
                coherence_from_word_arrays_with_both_uniform_flags_64(
                    &self_w,
                    self_is_uniform,
                    cand_w,
                    false,
                )
            };
            if coh >= -safe_thresh {
                let base_score_no_macro = (phase_scale * coh) + prior_term + link_term;
                if !found_valid || (base_score_no_macro + max_macro_bonus > best_score) {
                    let geo_macro_affinity = if in_macro_cluster {
                        1.0
                    } else if is_unstamped_for_geo {
                        if !win_meta_evaluated {
                            win_eq_self = swar_words_equal_64(&self_w, &win_w);
                            win_is_uniform = if win_eq_self {
                                self_is_uniform
                            } else if self_is_uniform {
                                win_uni_from_estep
                            } else {
                                is_word_array_uniform_64(&win_w)
                            };
                            win_meta_evaluated = true;
                        }
                        let c_win_coh = if win_eq_self {
                            coh
                        } else {
                            let cand_uni_for_win = if self_is_uniform {
                                cand_is_uniform
                            } else if win_is_uniform {
                                is_word_array_uniform_64(cand_w)
                            } else {
                                false
                            };
                            coherence_from_word_arrays_with_both_uniform_flags_64(
                                &win_w,
                                win_is_uniform,
                                cand_w,
                                cand_uni_for_win,
                            )
                        };
                        branchless_relu_f32(c_win_coh)
                    } else {
                        0.0
                    };
                    let score = branchless_flush_subnormal_f32(
                        base_score_no_macro + (clean_cluster_bonus * geo_macro_affinity),
                    );
                    let take = (!found_valid) | (score > best_score);
                    best_id = branchless_select_u32(take, cand_id, best_id);
                    best_score = branchless_select_f32(take, score, best_score);
                    best_coh = branchless_select_f32(take, coh, best_coh);
                    found_valid = true;
                }
            }
            idx += 1;
        }

        if found_valid {
            src_trail.linked_trail_id = best_id;
            src_trail.flags |= TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_CRYSTALLIZED_CANDIDATE;
        }
        let out_score = branchless_select_f32(found_valid, best_score, 0.0);
        (win_k, best_id, out_score, best_coh, found_valid)
    }
    /// Folds a slice of `.lmnk` [`PhaseAnchorRecord`]s onto `trail_phases_inout`
    /// weighted by **Phase-Proximity Gravity** in 8 register-resident `u64` SWAR words,
    /// returning `(folded_words, total_pull, dominant_anchor_id, dominant_phase_clamp_rad, mean_coherence)`
    /// in a single pass (`0` duplicate coherence scans).
    #[inline]
    pub fn fold_crystallized_ensemble_words_and_coherence_64(
        records: &[Self],
        trail_phases_inout: &mut [PhaseQ7; MAX_ATTENTION_HEADS],
        init_pull_weight: f32,
    ) -> ([u64; NUM_STIGMERGIC_GRANULES], f32, u32, f32, f32) {
        let init_w = load_q7_words_64(trail_phases_inout);
        let init_is_uniform = is_word_array_uniform_64(&init_w);
        let init_w0 = init_w[0];
        let init_b0 = (init_w0 & 0xFF) as u8;
        let init_is_byte_uniform = init_is_uniform & (init_w0 == init_w0.rotate_right(8));
        let init_nonzero = if init_is_uniform {
            init_w0 != 0
        } else {
            (((init_w0 | init_w[1]) | (init_w[2] | init_w[3]))
                | ((init_w[4] | init_w[5]) | (init_w[6] | init_w[7])))
                != 0
        };
        let mut tot_pull = branchless_flush_subnormal_f32(branchless_relu_f32(init_pull_weight));
        if records.len() == 1 && !init_nonzero && tot_pull <= DEFAULT_ENTROPY_EPS {
            let rec = &records[0];
            if rec.sample_count > 0 && rec.phase_gravity_mass > DEFAULT_ENTROPY_EPS {
                let rec_w = *rec.phase_words_ref();
                let grav = branchless_phase_proximity_gravity_f32(
                    rec.phase_gravity_mass,
                    1.0,
                    DEFAULT_PHASE_GRAVITY_EPS,
                );
                store_q7_words_64(trail_phases_inout, &rec_w);
                return (rec_w, grav, rec.anchor_id, rec.phase_clamp_rad, 1.0);
            }
            return (init_w, 0.0, 0, DEFAULT_PHASE_CLAMP_MAX_RAD, 0.0);
        }
        let mut acc_w = init_w;
        let mut acc_is_uniform = init_is_uniform;
        let mut best_grav = 0.0_f32;
        let mut dom_id = 0u32;
        let mut dom_clamp = DEFAULT_PHASE_CLAMP_MAX_RAD;
        let mut coh_sum = 0.0_f32;
        let mut active_cnt = 0usize;

        let mut i = 0usize;
        while i < records.len() {
            let rec = &records[i];
            if rec.sample_count > 0 && rec.phase_gravity_mass > DEFAULT_ENTROPY_EPS {
                let rec_w = rec.phase_words_ref();
                let rec_is_uniform = if (init_nonzero & init_is_uniform) | acc_is_uniform {
                    is_word_array_uniform_64(rec_w)
                } else {
                    false
                };
                let rw0 = rec_w[0];
                let rb0 = (rw0 & 0xFF) as u8;
                let (coh, grav) = if (!init_nonzero)
                    | (rec_is_uniform & init_is_uniform & (rw0 == init_w0))
                {
                    (
                        1.0,
                        branchless_q7_default_gravity_from_diff_u8(rec.phase_gravity_mass, 0),
                    )
                } else if rec_is_uniform & init_is_byte_uniform & (rw0 == rw0.rotate_right(8)) {
                    let d_u8 = init_b0.wrapping_sub(rb0);
                    (
                        Q7_FULL_COS_REF[d_u8 as usize],
                        branchless_q7_default_gravity_from_diff_u8(rec.phase_gravity_mass, d_u8),
                    )
                } else {
                    let c_val = coherence_from_word_arrays_with_both_uniform_flags_64(
                        &init_w,
                        init_is_uniform,
                        rec_w,
                        rec_is_uniform,
                    );
                    (
                        c_val,
                        branchless_phase_proximity_gravity_f32(
                            rec.phase_gravity_mass,
                            c_val,
                            DEFAULT_PHASE_GRAVITY_EPS,
                        ),
                    )
                };
                coh_sum += coh;
                active_cnt += 1;
                let new_tot = tot_pull + grav;
                if grav > DEFAULT_ENTROPY_EPS {
                    if tot_pull <= DEFAULT_ENTROPY_EPS {
                        acc_w = *rec_w;
                        acc_is_uniform = rec_is_uniform;
                    } else if !swar_words_equal_64(&acc_w, rec_w) {
                        let safe_tot =
                            branchless_select_f32(new_tot > DEFAULT_ENTROPY_EPS, new_tot, 1.0);
                        let s_u8 = ((grav * fast_inv_pos_normal_f32(safe_tot)) * 128.0 + 0.5) as u8;
                        let s_u64 = s_u8.min(128) as u64;
                        if s_u64 >= 128 {
                            acc_w = *rec_w;
                            acc_is_uniform = rec_is_uniform;
                        } else if s_u64 > 0 {
                            let bias_restore =
                                (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                            let (next_acc, next_uni) = swar_slerp_words_with_uniform_flag_64(
                                &acc_w,
                                acc_is_uniform,
                                rec_w,
                                rec_is_uniform,
                                s_u64,
                                bias_restore,
                            );
                            acc_w = next_acc;
                            acc_is_uniform = next_uni;
                        }
                    }
                }

                let is_dom = grav > best_grav;
                best_grav = branchless_select_f32(is_dom, grav, best_grav);
                dom_id = branchless_select_u32(is_dom, rec.anchor_id, dom_id);
                dom_clamp = branchless_select_f32(is_dom, rec.phase_clamp_rad, dom_clamp);
                tot_pull = new_tot;
            }
            i += 1;
        }
        store_q7_words_64(trail_phases_inout, &acc_w);
        let mean_coh = if active_cnt == 0 {
            0.0
        } else {
            branchless_clamp_f32(coh_sum * fast_inv_len_f32(active_cnt), -1.0, 1.0)
        };
        (acc_w, tot_pull, dom_id, dom_clamp, mean_coh)
    }

    /// Folds a slice of `.lmnk` [`PhaseAnchorRecord`]s onto `trail_phases_inout`
    /// weighted by **Phase-Proximity Gravity** in 8 register-resident `u64` SWAR words,
    /// evaluating proximity coherence against the initial query profile (`init_w`) and
    /// fast-pathing uninitialized/zero query trails and the first active attractor (`tot_pull <= 1.0e-12`).
    #[inline(always)]
    pub fn fold_crystallized_ensemble_words_with_init_pull_64(
        records: &[Self],
        trail_phases_inout: &mut [PhaseQ7; MAX_ATTENTION_HEADS],
        init_pull_weight: f32,
    ) -> ([u64; NUM_STIGMERGIC_GRANULES], f32, u32, f32) {
        let (acc_w, tot_pull, dom_id, dom_clamp, _mean_coh) =
            Self::fold_crystallized_ensemble_words_and_coherence_64(
                records,
                trail_phases_inout,
                init_pull_weight,
            );
        (acc_w, tot_pull, dom_id, dom_clamp)
    }
}
