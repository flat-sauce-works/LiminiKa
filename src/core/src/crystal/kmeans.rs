// name: src/core/src/crystal/kmeans.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Online toroidal k-Means / DP-Means clustering, dissipative overflow scatter, and
//! serendipitous attractor nucleation for [CrystallizedAttractorRecord].

#![allow(clippy::too_many_arguments)]

use crate::abi::gcso_pointer_trail_t;
use crate::traits::branchless::{
    branchless_toroidal_centroid_step_words_with_coh_and_uni_64,
    coherence_above_floor_from_word_arrays_64,
    coherence_from_word_arrays_with_both_uniform_flags_64,
};
use crate::traits::gateway::{
    GatewayFrame, ManifoldGatewaySignal, StigmergicVennSerendipityReport,
    CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, CAUSAL_CAUSE_COHOMOLOGY_REPULSION,
    GCSO_CRYSTAL_RECORD_MAGIC, GCSO_CRYSTAL_RECORD_VERSION, ROOT_CAUSE_COHOMOLOGY_REPULSION,
};
use crate::traits::primitives::{
    branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_maturity_scale_u32,
    branchless_max_f32, branchless_min_f32, branchless_select_f32, branchless_select_u32,
    branchless_select_u64, branchless_select_usize, crc32_update_crystal_energy_and_mass_delta,
    fast_inv_pos_normal_f32, fast_ln_1p_bounded_f32, is_word_array_uniform_64,
    swar_q7_circular_diff_x8, swar_q7_mean_i8_from_words_with_uniform_flag_64,
    ACTION_MODE_PHASE_REPULSE, ACTION_PRESET_EXPRESS_PHASE, ACTION_PRESET_PASSIVE_SIDECAR,
    ATTRACTOR_SLOT_CUSTOM_7, DEFAULT_ENTROPY_EPS, DEFAULT_QDPS_MIN_STEP_RAD,
    DEFAULT_RIPA_CLAMP_MAX_RAD,
};
use crate::traits::telemetry::{MAX_ATTRACTOR_PULL_FORCE, NUM_STIGMERGIC_GRANULES};

use super::{CrystallizedAttractorRecord, DissipativeOverflowReport};

impl CrystallizedAttractorRecord {
    /// Branchless 3-stage **Dissipative Overflow -> Residual Scatter -> Autonomous Daughter Birth** step:
    /// 1. Evaluates saturation & dispersion pressure (`1 - resultant_r_bar`, `sample_count`) on the nearest attractor $A_1$.
    /// 2. If $A_1$ overflows (`effective_coh < split_coherence_thresh`), peels $A_1$'s phase from `sample_w` via SWAR circular subtraction and scans remaining active slots for a secondary basin $A_2$ without branches.
    /// 3. If the overflow residual settles on $A_2$, absorbs into $A_2$ (`scattered = true`); otherwise autonomously spawns a daughter attractor into an empty or weaker slot (`spawned_new = true`).
    #[inline]
    pub fn step_dissipative_overflow_scatter_or_spawn_64(
        centroids: &mut [Self],
        frame: &GatewayFrame,
        sample_w: &[u64; NUM_STIGMERGIC_GRANULES],
        split_coherence_thresh: f32,
        trail_opt: Option<&mut gcso_pointer_trail_t>,
    ) -> DissipativeOverflowReport {
        let k_len = centroids.len();
        if k_len == 0 {
            return DissipativeOverflowReport::default();
        }

        let is_repulsed = (frame.repulsion_severity > 0.50)
            | ((frame.action_mode_flags & ACTION_MODE_PHASE_REPULSE) != 0)
            | (frame.dominant_cause_code == ROOT_CAUSE_COHOMOLOGY_REPULSION)
            | ((frame.provenance_bits & CAUSAL_CAUSE_COHOMOLOGY_REPULSION) != 0);
        let (idx_1, coh_1, _grav_1, empty_opt, weakest_idx, w_1, sample_uni, best_1_uni, _) =
            Self::find_nearest_toroidal_centroid_with_cache_64(centroids, sample_w, None);
        let primary_aid = centroids[idx_1].anchor_id;
        let primary_count = centroids[idx_1].sample_count;
        let primary_mass = centroids[idx_1].phase_gravity_mass;
        let r_bar_1 = branchless_clamp_f32(centroids[idx_1].resultant_r_bar, 0.0, 1.0);
        let overflow_pressure = branchless_clamp_f32(
            (1.0 - r_bar_1) * 0.35 + fast_ln_1p_bounded_f32(primary_count as f32) * 0.06,
            0.0,
            0.85,
        );
        let res_w = [
            swar_q7_circular_diff_x8(sample_w[0], w_1[0]),
            swar_q7_circular_diff_x8(sample_w[1], w_1[1]),
            swar_q7_circular_diff_x8(sample_w[2], w_1[2]),
            swar_q7_circular_diff_x8(sample_w[3], w_1[3]),
            swar_q7_circular_diff_x8(sample_w[4], w_1[4]),
            swar_q7_circular_diff_x8(sample_w[5], w_1[5]),
            swar_q7_circular_diff_x8(sample_w[6], w_1[6]),
            swar_q7_circular_diff_x8(sample_w[7], w_1[7]),
        ];

        if is_repulsed {
            return DissipativeOverflowReport {
                primary_idx: idx_1,
                landed_idx: idx_1,
                primary_anchor_id: primary_aid,
                landed_anchor_id: primary_aid,
                primary_coherence: coh_1,
                scatter_coherence: 0.0,
                overflow_pressure,
                updated_gravity_mass: primary_mass,
                residual_words: res_w,
                overflowed: false,
                scattered: false,
                spawned_new: false,
            };
        }

        let safe_split = branchless_clamp_f32(split_coherence_thresh, -0.95, 0.95);
        let has_primary = primary_count > 0;
        let overflowed = has_primary & ((coh_1 - overflow_pressure) < safe_split);

        // Branchless secondary scatter scan over `res_w` across all slots `i != idx_1`.
        let res_uni = is_word_array_uniform_64(&res_w);
        let mut best_2_idx = idx_1;
        let mut best_2_coh = -2.0_f32;
        let mut best_2_uni = false;
        let mut any_2 = false;
        let mut best_2_w = [0u64; NUM_STIGMERGIC_GRANULES];
        let mut i = 0usize;
        while i < k_len {
            let rec = &centroids[i];
            let eligible = (i != idx_1)
                & (rec.sample_count > 0)
                & (rec.phase_gravity_mass > DEFAULT_ENTROPY_EPS);
            let cw = rec.phase_words_ref();
            let c_uni = is_word_array_uniform_64(cw);
            let raw_coh =
                coherence_from_word_arrays_with_both_uniform_flags_64(&res_w, res_uni, cw, c_uni);
            let eff_coh = branchless_select_f32(eligible, raw_coh, -2.0);
            let take = eligible & ((!any_2) | (eff_coh > best_2_coh));
            best_2_idx = branchless_select_usize(take, i, best_2_idx);
            best_2_coh = branchless_select_f32(take, eff_coh, best_2_coh);
            best_2_uni = (take & c_uni) | ((!take) & best_2_uni);
            best_2_w = [
                branchless_select_u64(take, cw[0], best_2_w[0]),
                branchless_select_u64(take, cw[1], best_2_w[1]),
                branchless_select_u64(take, cw[2], best_2_w[2]),
                branchless_select_u64(take, cw[3], best_2_w[3]),
                branchless_select_u64(take, cw[4], best_2_w[4]),
                branchless_select_u64(take, cw[5], best_2_w[5]),
                branchless_select_u64(take, cw[6], best_2_w[6]),
                branchless_select_u64(take, cw[7], best_2_w[7]),
            ];
            any_2 |= eligible;
            i += 1;
        }
        let clean_scatter_coh = branchless_select_f32(any_2, best_2_coh, 0.0);

        let settled_on_scatter = overflowed & any_2 & (best_2_coh >= safe_split);
        let needs_spawn = (!has_primary) | (overflowed & !settled_on_scatter);
        let (has_empty, empty_i) = match empty_opt {
            Some(idx) => (true, idx),
            None => (false, 0),
        };
        let can_evict =
            (!has_empty) & (frame.effective_energy > centroids[weakest_idx].phase_gravity_mass);
        let spawn_slot = branchless_select_usize(has_empty, empty_i, weakest_idx);
        let effective_spawn = needs_spawn & (has_empty | can_evict);

        let target_idx = branchless_select_usize(
            settled_on_scatter,
            best_2_idx,
            branchless_select_usize(effective_spawn, spawn_slot, idx_1),
        );
        let assigned_spawn_id = branchless_select_u32(
            frame.target_anchor_id != 0,
            frame.target_anchor_id,
            (target_idx as u32) + 1,
        );
        if effective_spawn {
            centroids[target_idx].init_spawned_slot_header_in_place(
                assigned_spawn_id,
                frame.dominant_slot,
                frame.mapped_section_offset48,
            );
        }

        let peel_for_daughter = effective_spawn & overflowed & (coh_1 >= safe_split);
        let use_res_words = settled_on_scatter | peel_for_daughter;
        let absorbed_w = [
            branchless_select_u64(use_res_words, res_w[0], sample_w[0]),
            branchless_select_u64(use_res_words, res_w[1], sample_w[1]),
            branchless_select_u64(use_res_words, res_w[2], sample_w[2]),
            branchless_select_u64(use_res_words, res_w[3], sample_w[3]),
            branchless_select_u64(use_res_words, res_w[4], sample_w[4]),
            branchless_select_u64(use_res_words, res_w[5], sample_w[5]),
            branchless_select_u64(use_res_words, res_w[6], sample_w[6]),
            branchless_select_u64(use_res_words, res_w[7], sample_w[7]),
        ];
        let absorbed_uni = (use_res_words & res_uni) | ((!use_res_words) & sample_uni);
        let cur_target_w = [
            branchless_select_u64(
                effective_spawn,
                0,
                branchless_select_u64(settled_on_scatter, best_2_w[0], w_1[0]),
            ),
            branchless_select_u64(
                effective_spawn,
                0,
                branchless_select_u64(settled_on_scatter, best_2_w[1], w_1[1]),
            ),
            branchless_select_u64(
                effective_spawn,
                0,
                branchless_select_u64(settled_on_scatter, best_2_w[2], w_1[2]),
            ),
            branchless_select_u64(
                effective_spawn,
                0,
                branchless_select_u64(settled_on_scatter, best_2_w[3], w_1[3]),
            ),
            branchless_select_u64(
                effective_spawn,
                0,
                branchless_select_u64(settled_on_scatter, best_2_w[4], w_1[4]),
            ),
            branchless_select_u64(
                effective_spawn,
                0,
                branchless_select_u64(settled_on_scatter, best_2_w[5], w_1[5]),
            ),
            branchless_select_u64(
                effective_spawn,
                0,
                branchless_select_u64(settled_on_scatter, best_2_w[6], w_1[6]),
            ),
            branchless_select_u64(
                effective_spawn,
                0,
                branchless_select_u64(settled_on_scatter, best_2_w[7], w_1[7]),
            ),
        ];
        let target_coh = branchless_select_f32(settled_on_scatter, best_2_coh, coh_1);
        let target_uni = (settled_on_scatter & best_2_uni) | ((!settled_on_scatter) & best_1_uni);
        let (landed_aid, _final_coh, updated_mass) = Self::commit_frame_absorption_at_index(
            centroids,
            target_idx,
            effective_spawn,
            target_coh,
            target_uni,
            absorbed_uni,
            &cur_target_w,
            frame,
            &absorbed_w,
            trail_opt,
        );

        DissipativeOverflowReport {
            primary_idx: idx_1,
            landed_idx: target_idx,
            primary_anchor_id: primary_aid,
            landed_anchor_id: landed_aid,
            primary_coherence: coh_1,
            scatter_coherence: clean_scatter_coh,
            overflow_pressure,
            updated_gravity_mass: updated_mass,
            residual_words: res_w,
            overflowed,
            scattered: settled_on_scatter,
            spawned_new: effective_spawn,
        }
    }

    /// Evaluates [`step_dissipative_overflow_scatter_or_spawn_64`](Self::step_dissipative_overflow_scatter_or_spawn_64)
    /// directly from any [`ManifoldGatewaySignal`] implementor.
    #[inline]
    pub fn step_dissipative_overflow_signal_64<S: ManifoldGatewaySignal + ?Sized>(
        centroids: &mut [Self],
        signal: &S,
        split_coherence_thresh: f32,
        trail_opt: Option<&mut gcso_pointer_trail_t>,
    ) -> (GatewayFrame, DissipativeOverflowReport) {
        let (frame, frm_w) = signal.synthesize_gateway_frame_with_words();
        let report = Self::step_dissipative_overflow_scatter_or_spawn_64(
            centroids,
            &frame,
            &frm_w,
            split_coherence_thresh,
            trail_opt,
        );
        (frame, report)
    }

    /// Autonomously nucleates a new attractor inside `centroids` when approximation noise in
    /// [`StigmergicVennSerendipityReport`] pushes a fragmented granule outside all existing
    /// attractors' Venn apertures into [`crate::traits::Liminal`] (`serendipity_triggered != 0`).
    ///
    /// Returns `(spawned_slot_idx, spawned_anchor_id, did_nucleate)` in branchless constant time
    /// relative to centroid slot contents.
    #[inline]
    pub fn nucleate_serendipitous_attractor_from_void_64(
        centroids: &mut [Self],
        venn_report: &StigmergicVennSerendipityReport,
        drifted_words: &[u64; NUM_STIGMERGIC_GRANULES],
        new_anchor_id: u32,
        trail_opt: Option<&mut gcso_pointer_trail_t>,
    ) -> (usize, u32, bool) {
        if (venn_report.serendipity_triggered == 0) || centroids.is_empty() {
            return (0, 0, false);
        }
        let mut empty_idx = 0usize;
        let mut has_empty = false;
        let mut weakest_idx = 0usize;
        let mut weakest_mass = f32::MAX;
        let mut i = 0usize;
        while i < centroids.len() {
            let rec = &centroids[i];
            let is_empty = rec.sample_count == 0;
            let take_empty = (!has_empty) & is_empty;
            empty_idx = branchless_select_usize(take_empty, i, empty_idx);
            has_empty |= is_empty;

            let take_weak = (!is_empty) & (rec.phase_gravity_mass < weakest_mass);
            weakest_idx = branchless_select_usize(take_weak, i, weakest_idx);
            weakest_mass = branchless_select_f32(take_weak, rec.phase_gravity_mass, weakest_mass);
            i += 1;
        }

        let seed_energy = branchless_max_f32(venn_report.void_excursion_energy, 0.25);
        let can_evict = (!has_empty) & (seed_energy > weakest_mass);
        if !(has_empty | can_evict) {
            return (0, 0, false);
        }
        let target_idx = branchless_select_usize(has_empty, empty_idx, weakest_idx);
        let eff_anchor_id =
            branchless_select_u32(new_anchor_id != 0, new_anchor_id, (target_idx as u32) + 1);

        let mut seed_frame = GatewayFrame {
            target_anchor_id: eff_anchor_id,
            effective_energy: seed_energy,
            ripa_clamp_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
            qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD,
            effective_inv_temp: 1.0,
            provenance_bits: CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
            dominant_slot: ATTRACTOR_SLOT_CUSTOM_7,
            action_mode_flags: ACTION_PRESET_EXPRESS_PHASE,
            active_swarm_cell_mask: venn_report.serendipity_excursion_mask | 0x01,
            ..GatewayFrame::default()
        };
        *seed_frame.phase_words_mut() = *drifted_words;
        seed_frame.mean_phase_tag_q7 =
            swar_q7_mean_i8_from_words_with_uniform_flag_64(drifted_words, false);

        centroids[target_idx].init_spawned_slot_header_in_place(
            eff_anchor_id,
            ATTRACTOR_SLOT_CUSTOM_7,
            0,
        );
        let empty_cur = [0u64; NUM_STIGMERGIC_GRANULES];
        let (final_aid, _coh, _mass) = Self::commit_frame_absorption_at_index(
            centroids,
            target_idx,
            true,
            1.0,
            true,
            is_word_array_uniform_64(drifted_words),
            &empty_cur,
            &seed_frame,
            drifted_words,
            trail_opt,
        );
        (target_idx, final_aid, true)
    }

    /// **Classical Online Toroidal $k$-Means / DP-Means Closed-Loop E+M Step ($\mathcal{O}(K)$)**:
    /// Returns `(assigned_cluster_idx, assigned_anchor_id, centroid_coherence, updated_gravity_mass, spawned_new_cluster)`.
    #[inline(always)]
    pub fn step_online_toroidal_kmeans_frame_64(
        centroids: &mut [Self],
        frame: &GatewayFrame,
        split_coherence_thresh: f32,
        trail_opt: Option<&mut gcso_pointer_trail_t>,
    ) -> (usize, u32, f32, f32, bool) {
        Self::step_online_toroidal_kmeans_frame_with_words_64(
            centroids,
            frame,
            frame.phase_words_ref(),
            split_coherence_thresh,
            trail_opt,
        )
    }

    /// Executes Online Toroidal $k$-Means / DP-Means across a batch of [`GatewayFrame`]s,
    /// returning `(last_assigned_idx, last_assigned_anchor_id, mean_coherence, spawned_clusters_count)`.
    #[inline]
    pub fn step_online_toroidal_kmeans_batch_frames_64(
        centroids: &mut [Self],
        frames: &[GatewayFrame],
        split_coherence_thresh: f32,
        mut trail_opt: Option<&mut gcso_pointer_trail_t>,
    ) -> (usize, u32, f32, u32) {
        if centroids.is_empty() || frames.is_empty() {
            return (0, 0, 0.0, 0);
        }
        let mut valid_count = 0u32;
        let mut spawned_count = 0u32;
        let mut coh_sum = 0.0_f32;
        let mut last_idx = 0usize;
        let mut last_aid = 0u32;
        for frame in frames {
            if frame.is_repulsed() {
                continue;
            }
            let (idx, aid, coh, _mass, spawned) =
                Self::step_online_toroidal_kmeans_frame_with_words_64(
                    centroids,
                    frame,
                    frame.phase_words_ref(),
                    split_coherence_thresh,
                    trail_opt.as_deref_mut(),
                );
            valid_count += 1;
            if spawned {
                spawned_count += 1;
            }
            coh_sum += coh;
            last_idx = idx;
            last_aid = aid;
        }
        let mean_coh = if valid_count > 0 {
            branchless_clamp_f32(
                coh_sum * fast_inv_pos_normal_f32(valid_count as f32),
                -1.0,
                1.0,
            )
        } else {
            0.0
        };
        (last_idx, last_aid, mean_coh, spawned_count)
    }

    /// **Online Toroidal $k$-Means / DP-Means on Any [`ManifoldGatewaySignal`]**:
    /// Synthesizes `signal` into a [`GatewayFrame`] and executes E-step + M-step over `centroids`.
    #[inline]
    pub fn step_online_toroidal_kmeans_signal_64<S: ManifoldGatewaySignal + ?Sized>(
        centroids: &mut [Self],
        signal: &S,
        split_coherence_thresh: f32,
        trail_opt: Option<&mut gcso_pointer_trail_t>,
    ) -> (GatewayFrame, usize, u32, f32, f32, bool) {
        let (frame, frm_w) = signal.synthesize_gateway_frame_with_words();
        let (cluster_idx, anchor_id, coh, mass, spawned) =
            Self::step_online_toroidal_kmeans_frame_with_words_64(
                centroids,
                &frame,
                &frm_w,
                split_coherence_thresh,
                trail_opt,
            );
        (frame, cluster_idx, anchor_id, coh, mass, spawned)
    }
    fn merge_other_centroid_words_in_place(
        &mut self,
        w_a: &[u64; NUM_STIGMERGIC_GRANULES],
        a_is_uniform: bool,
        w_b: &[u64; NUM_STIGMERGIC_GRANULES],
        b_is_uniform: bool,
        coh: f32,
        b_anchor_id: u32,
        b_sample_count: u32,
        b_zimms_offset48: u64,
        b_base_energy: f32,
        b_resultant_r_bar: f32,
        b_von_mises_kappa: f32,
        b_gravity_mass: f32,
        b_ripa_clamp_rad: f32,
        b_effective_inv_temp: f32,
        b_provenance_bits: u32,
        b_dominant_slot: u8,
        b_action_mode_flags: u8,
        b_active_swarm_mask: u8,
        b_dominant_cause_code: u8,
    ) -> [u64; NUM_STIGMERGIC_GRANULES] {
        let b_dom = b_gravity_mass > self.phase_gravity_mass;
        let (
            cur_dom_w,
            dom_uni,
            src_w,
            src_uni,
            dom_sc,
            dom_e,
            dom_r,
            dom_k,
            src_e,
            src_sc,
            src_r,
            src_k,
        ) = if b_dom {
            self.anchor_id = b_anchor_id;
            self.dominant_slot = b_dominant_slot;
            self.ripa_clamp_rad = b_ripa_clamp_rad;
            self.effective_inv_temp = b_effective_inv_temp;
            self.dominant_cause_code = b_dominant_cause_code;
            if b_zimms_offset48 != 0 {
                self.mapped_section_offset48 = b_zimms_offset48;
            }
            (
                w_b,
                b_is_uniform,
                w_a,
                a_is_uniform,
                b_sample_count,
                b_base_energy,
                b_resultant_r_bar,
                b_von_mises_kappa,
                self.base_energy,
                self.sample_count,
                self.resultant_r_bar,
                self.von_mises_kappa,
            )
        } else {
            if self.mapped_section_offset48 == 0 {
                self.mapped_section_offset48 = b_zimms_offset48;
            }
            (
                w_a,
                a_is_uniform,
                w_b,
                b_is_uniform,
                self.sample_count,
                self.base_energy,
                self.resultant_r_bar,
                self.von_mises_kappa,
                b_base_energy,
                b_sample_count,
                b_resultant_r_bar,
                b_von_mises_kappa,
            )
        };

        let _ = (src_r, src_k);
        let (next_w, raw_next_e, r_bar, kappa) =
            branchless_toroidal_centroid_step_words_with_coh_and_uni_64(
                cur_dom_w,
                dom_sc.max(1),
                dom_e,
                dom_r,
                dom_k,
                src_w,
                Some(coh),
                src_e,
                dom_uni,
                src_uni,
            );
        let next_e = branchless_min_f32(raw_next_e, MAX_ATTRACTOR_PULL_FORCE * 4.0);
        let next_count = dom_sc.saturating_add(src_sc);
        self.sample_count = next_count;
        self.base_energy = next_e;
        self.resultant_r_bar = r_bar;
        self.von_mises_kappa = kappa;
        let maturity_scale = branchless_maturity_scale_u32(next_count);
        self.phase_gravity_mass = branchless_flush_subnormal_f32(
            next_e * branchless_clamp_f32(r_bar, 0.10, 1.0) * maturity_scale,
        );
        self.provenance_bits |= b_provenance_bits | CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED;
        self.action_mode_flags |= b_action_mode_flags;
        if self.action_mode_flags == ACTION_PRESET_PASSIVE_SIDECAR {
            self.action_mode_flags = ACTION_PRESET_EXPRESS_PHASE;
        }
        self.active_swarm_mask &= b_active_swarm_mask;
        next_w
    }

    /// **Autonomous Online Toroidal Centroid Consolidation & Evaporative Pruning ($\mathcal{O}(K^2)$ SWAR)**:
    /// **Online Basin Consolidation (Centroid Merging & Evaporation Pruning on $SO(2)^{64}$)**:
    /// Applies `retention_decay`, prunes centroids below `min_gravity_mass`, and merges centroids
    /// whose pairwise phase coherence $\mathcal{C}_{\mathrm{phase}}(a, b) \ge \text{merge\_coherence\_thresh}$.
    #[inline]
    pub fn merge_and_prune_centroids_64(
        centroids: &mut [Self],
        merge_coherence_thresh: f32,
        min_gravity_mass: f32,
        retention_decay: f32,
    ) -> (u32, u32) {
        let k_len = centroids.len();
        if k_len == 0 {
            return (0, 0);
        }
        let safe_merge = branchless_clamp_f32(merge_coherence_thresh, 0.25, 0.999);
        let safe_retain = branchless_clamp_f32(retention_decay, 0.0, 1.0);
        let apply_decay = safe_retain < 0.999_999;
        let min_mass = branchless_max_f32(0.0, min_gravity_mass);

        let mut pruned_count = 0u32;
        let mut merged_count = 0u32;
        let mut active_cnt = 0usize;

        let mut i = 0usize;
        while i < k_len {
            if centroids[i].sample_count > 0 {
                if apply_decay {
                    let prev_be_bits = centroids[i].base_energy.to_bits();
                    let prev_gm_bits = centroids[i].phase_gravity_mass.to_bits();
                    let next_be =
                        branchless_flush_subnormal_f32(centroids[i].base_energy * safe_retain);
                    let next_gm = branchless_flush_subnormal_f32(
                        centroids[i].phase_gravity_mass * safe_retain,
                    );
                    centroids[i].base_energy = next_be;
                    centroids[i].phase_gravity_mass = next_gm;
                    if (centroids[i].magic == GCSO_CRYSTAL_RECORD_MAGIC)
                        & (centroids[i].version == GCSO_CRYSTAL_RECORD_VERSION)
                        & (centroids[i].checksum_crc32 != 0)
                    {
                        centroids[i].checksum_crc32 = crc32_update_crystal_energy_and_mass_delta(
                            centroids[i].checksum_crc32,
                            prev_be_bits ^ next_be.to_bits(),
                            prev_gm_bits ^ next_gm.to_bits(),
                        );
                    } else {
                        centroids[i].stamp_crc32();
                    }
                }
                if centroids[i].phase_gravity_mass < min_mass {
                    let slot = centroids[i].dominant_slot;
                    centroids[i].clear_empty_in_place(slot);
                    pruned_count += 1;
                } else {
                    active_cnt += 1;
                }
            }
            i += 1;
        }

        if active_cnt < 2 {
            return (0, pruned_count);
        }

        let mut a = 0usize;
        while a < k_len {
            if centroids[a].sample_count > 0
                && centroids[a].phase_gravity_mass > DEFAULT_ENTROPY_EPS
            {
                let mut w_a = *centroids[a].phase_words_ref();
                let mut a_uni = is_word_array_uniform_64(&w_a);
                let mut a_merged = false;
                let mut b = a + 1;
                while b < k_len {
                    if centroids[b].sample_count > 0
                        && centroids[b].phase_gravity_mass > DEFAULT_ENTROPY_EPS
                    {
                        let w_b = *centroids[b].phase_words_ref();
                        let b_uni = if a_uni {
                            is_word_array_uniform_64(&w_b)
                        } else {
                            false
                        };
                        let coh = coherence_above_floor_from_word_arrays_64(
                            &w_a, a_uni, &w_b, b_uni, safe_merge,
                        );
                        if coh >= safe_merge {
                            let (
                                b_aid,
                                b_sc,
                                b_zoff,
                                b_be,
                                b_rbar,
                                b_kap,
                                b_mass,
                                b_clamp,
                                b_temp,
                                b_prov,
                                b_slot,
                                b_act,
                                b_swm,
                                b_cause,
                            ) = (
                                centroids[b].anchor_id,
                                centroids[b].sample_count,
                                centroids[b].mapped_section_offset48,
                                centroids[b].base_energy,
                                centroids[b].resultant_r_bar,
                                centroids[b].von_mises_kappa,
                                centroids[b].phase_gravity_mass,
                                centroids[b].ripa_clamp_rad,
                                centroids[b].effective_inv_temp,
                                centroids[b].provenance_bits,
                                centroids[b].dominant_slot,
                                centroids[b].action_mode_flags,
                                centroids[b].active_swarm_mask,
                                centroids[b].dominant_cause_code,
                            );
                            w_a = centroids[a].merge_other_centroid_words_in_place(
                                &w_a, a_uni, &w_b, b_uni, coh, b_aid, b_sc, b_zoff, b_be, b_rbar,
                                b_kap, b_mass, b_clamp, b_temp, b_prov, b_slot, b_act, b_swm,
                                b_cause,
                            );
                            a_uni = if a_uni & b_uni {
                                true
                            } else {
                                is_word_array_uniform_64(&w_a)
                            };
                            a_merged = true;
                            centroids[b].clear_empty_in_place(b_slot);
                            merged_count += 1;
                        }
                    }
                    b += 1;
                }
                if a_merged {
                    *centroids[a].phase_words_mut() = w_a;
                    centroids[a].stamp_crc32_with_words(&w_a);
                }
            }
            a += 1;
        }

        (merged_count, pruned_count)
    }
}
