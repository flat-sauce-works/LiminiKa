// name: src/core/src/runtime/gateway/bridges/anchor_bridge.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`PhaseAnchorRecord`] toroidal k-Means, dissipative overflow, nucleation, and Gateway bridges.

use super::super::*;
use crate::foundation::{
    ACTION_MODE_PHASE_REPULSE, ACTION_PRESET_EXPRESS_PHASE, ACTION_PRESET_PASSIVE_SIDECAR,
    DEFAULT_PHASE_CLAMP_MAX_RAD, DEFAULT_PHASE_STEP_MIN_RAD, SWAR_SPLAT_BYTE_64, TaggedPtr,
    branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_lerp_f32, branchless_max_f32,
    branchless_min_f32, branchless_rad_to_phase_step_u8, branchless_relu_f32,
    coherence_from_word_arrays_with_both_uniform_flags_64, fast_inv_pos_normal_f32,
    fast_ln_1p_bounded_f32, fast_mean_q7_tag_from_words_64, is_word_array_uniform_64,
    pack_action_slot_tag, swar_q7_circular_diff_x8,
    swar_q7_mean_i8_from_words_with_uniform_flag_64, swar_slerp_words_with_uniform_flag_64,
    swar_words_equal_64, words_to_q7_array_64,
};
use crate::liminal::{
    DEFAULT_ENTROPY_EPS, Grounded, Indivisible, MAX_ATTENTION_HEADS, NUM_STIGMERGIC_GRANULES,
    PhaseQ7, Scoped, ScopedCut, branchless_select_f32, branchless_select_u8, branchless_select_u32,
    branchless_select_u64, branchless_select_usize,
};
use crate::stigmergy::{
    ATTRACTOR_SLOT_CRYSTALLIZED, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA,
    DissipativeOverflowReport, MAX_ATTRACTOR_PULL_FORCE, OverlapDiscoveryReport, PhaseAnchorRecord,
    PhaseWordsRef64, PointerTrail, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE,
};

#[derive(Clone, Copy)]
struct CentroidSlotCandidate {
    nearest_idx: usize,
    best_coh: f32,
    empty_opt: Option<usize>,
    weakest_idx: usize,
    best_w: [u64; NUM_STIGMERGIC_GRANULES],
}

#[derive(Clone, Copy)]
struct CentroidAbsorptionPlan<'a> {
    target_idx: usize,
    spawned_new: bool,
    best_coh: f32,
    best_is_uniform: bool,
    sample_is_uniform: bool,
    cur_target_w: &'a [u64; NUM_STIGMERGIC_GRANULES],
}

impl PhaseAnchorRecord {
    #[inline(always)]
    fn absorb_gateway_frame_with_cur_coh_and_uni_unstamped(
        &mut self,
        cur_w: &[u64; NUM_STIGMERGIC_GRANULES],
        frame: &GatewayFrame,
        frame_words: Indivisible<Grounded, PhaseWordsRef64<'_>>,
        precomputed_coh_opt: Option<f32>,
        uni_flags_opt: Option<(bool, bool)>,
    ) -> ([u64; NUM_STIGMERGIC_GRANULES], f32, bool, bool) {
        if frame.is_repulsed() {
            return (*cur_w, self.phase_gravity_mass, false, false);
        }

        if self.sample_count == 0 {
            if frame.target_anchor_id != 0 {
                self.anchor_id = frame.target_anchor_id;
            }
            if frame.mapped_section_offset48 != 0 {
                self.mapped_section_offset48 = frame.mapped_section_offset48;
            }
            self.dominant_slot = frame.dominant_slot;
            self.phase_clamp_rad = frame.phase_clamp_rad;
            self.effective_inv_temp = frame.effective_inv_temp;
        } else {
            if self.phase_clamp_rad.to_bits() != frame.phase_clamp_rad.to_bits() {
                self.phase_clamp_rad =
                    branchless_lerp_f32(self.phase_clamp_rad, frame.phase_clamp_rad, 0.25);
            }
            if self.effective_inv_temp.to_bits() != frame.effective_inv_temp.to_bits() {
                self.effective_inv_temp =
                    branchless_lerp_f32(self.effective_inv_temp, frame.effective_inv_temp, 0.25);
            }
        }
        self.action_mode_flags = frame.action_mode_flags;
        self.active_swarm_mask = frame.active_swarm_cell_mask;
        self.lineage_flags |= frame.lineage_flags;

        let (next_w, mass, words_changed) = self
            .absorb_sample_words_with_cur_coh_and_uni_unstamped(
                cur_w,
                frame_words,
                precomputed_coh_opt,
                uni_flags_opt,
                frame.effective_energy,
                1,
            );
        (next_w, mass, true, words_changed)
    }

    /// Absorbs a [`GatewayFrame`] into this [`PhaseAnchorRecord`] and returns `phase_gravity_mass`.
    #[inline]
    pub fn absorb_gateway_frame(&mut self, frame: &GatewayFrame) -> f32 {
        let grounded_w = match Scoped::from_boundary(PhaseWordsRef64(frame.phase_words_ref()))
            .cut_by_lacuna(!frame.is_repulsed(), frame.repulsion_severity, 0.50)
        {
            ScopedCut::SelfRef(aligned) => aligned.into_grounded(),
            _ => return self.phase_gravity_mass,
        };
        let prev = self.capture_header_snapshot();
        let cur_w = *self.phase_words_ref();
        let (next_w, mass, touched, words_changed) = self
            .absorb_gateway_frame_with_cur_coh_and_uni_unstamped(
                &cur_w, frame, grounded_w, None, None,
            );
        if touched {
            self.stamp_or_update_absorption_crc32(prev, &cur_w, &next_w, words_changed);
        }
        mass
    }

    #[inline(always)]
    fn select_or_spawn_centroid_slot_64(
        centroids: &mut [Self],
        frame: &GatewayFrame,
        nearest: CentroidSlotCandidate,
        safe_split: f32,
    ) -> (usize, bool, [u64; NUM_STIGMERGIC_GRANULES]) {
        let CentroidSlotCandidate {
            nearest_idx,
            best_coh,
            empty_opt,
            weakest_idx,
            best_w,
        } = nearest;
        let needs_split = (centroids[nearest_idx].sample_count == 0) | (best_coh < safe_split);
        if needs_split {
            if let Some(empty_i) = empty_opt {
                let assigned_id = if frame.target_anchor_id != 0 {
                    frame.target_anchor_id
                } else {
                    (empty_i as u32) + 1
                };
                centroids[empty_i].init_spawned_slot_header_in_place(
                    assigned_id,
                    frame.dominant_slot,
                    frame.mapped_section_offset48,
                );
                (empty_i, true, [0u64; NUM_STIGMERGIC_GRANULES])
            } else if frame.effective_energy > centroids[weakest_idx].phase_gravity_mass {
                let assigned_id = if frame.target_anchor_id != 0 {
                    frame.target_anchor_id
                } else {
                    (weakest_idx as u32) + 1
                };
                centroids[weakest_idx].init_spawned_slot_header_in_place(
                    assigned_id,
                    frame.dominant_slot,
                    frame.mapped_section_offset48,
                );
                (weakest_idx, true, [0u64; NUM_STIGMERGIC_GRANULES])
            } else {
                (nearest_idx, false, best_w)
            }
        } else {
            (nearest_idx, false, best_w)
        }
    }

    #[inline(always)]
    fn commit_frame_absorption_at_index(
        centroids: &mut [Self],
        plan: CentroidAbsorptionPlan<'_>,
        frame: &GatewayFrame,
        sample_w: Indivisible<Grounded, PhaseWordsRef64<'_>>,
        trail_opt: Option<&mut PointerTrail>,
    ) -> (u32, f32, f32) {
        let CentroidAbsorptionPlan {
            target_idx,
            spawned_new,
            best_coh,
            best_is_uniform,
            sample_is_uniform,
            cur_target_w,
        } = plan;
        let pre_coh = Some(branchless_select_f32(spawned_new, 1.0, best_coh));
        let eff_best_uni = spawned_new | best_is_uniform;
        let uni_flags = Some((eff_best_uni, sample_is_uniform));
        let fallback_aid = (target_idx as u32) + 1;
        let target = &mut centroids[target_idx];
        let mut prev = target.capture_header_snapshot();
        if spawned_new {
            prev.anchor_id = 0;
            prev.sample_count = 0;
            prev.mapped_offset48 = 0;
        }
        let (next_w, updated_mass, _modified, words_changed) = target
            .absorb_gateway_frame_with_cur_coh_and_uni_unstamped(
                cur_target_w,
                frame,
                sample_w,
                pre_coh,
                uni_flags,
            );
        let winning_anchor_id =
            branchless_select_u32(target.anchor_id != 0, target.anchor_id, fallback_aid);
        target.anchor_id = winning_anchor_id;
        target.stamp_or_update_absorption_crc32(prev, cur_target_w, &next_w, words_changed);
        let final_coh = branchless_select_f32(spawned_new, 1.0, best_coh);

        if let Some(trail) = trail_opt {
            trail.target_anchor_id = winning_anchor_id;
            trail.cluster_id = winning_anchor_id;
            trail.flags |= TRAIL_FLAG_CRYSTALLIZED_CANDIDATE;
        }
        (winning_anchor_id, final_coh, updated_mass)
    }

    /// Performs an Online Toroidal $k$-Means / DP-Means E+M step using preloaded SWAR phase words,
    /// returning `(assigned_cluster_idx, assigned_anchor_id, centroid_coherence, updated_gravity_mass, spawned_new_cluster)`.
    #[inline]
    pub fn step_online_toroidal_kmeans_frame_with_words_64(
        centroids: &mut [Self],
        frame: &GatewayFrame,
        sample_w: &[u64; NUM_STIGMERGIC_GRANULES],
        split_coherence_thresh: f32,
        trail_opt: Option<&mut PointerTrail>,
    ) -> (usize, u32, f32, f32, bool) {
        let k_len = centroids.len();
        if k_len == 0 {
            return (0, frame.target_anchor_id, 0.0, 0.0, false);
        }

        let (
            nearest_idx,
            best_coh,
            _best_grav,
            empty_opt,
            weakest_idx,
            best_w,
            sample_is_uniform,
            best_is_uniform,
            _,
        ) = Self::find_nearest_toroidal_centroid_with_cache_64(centroids, sample_w, None);

        let grounded_w = match Scoped::from_boundary(PhaseWordsRef64(sample_w)).cut_by_lacuna(
            !frame.is_repulsed(),
            frame.repulsion_severity,
            0.50,
        ) {
            ScopedCut::SelfRef(aligned) => aligned.into_grounded(),
            _ => {
                let dom_id = centroids[nearest_idx].anchor_id;
                return (
                    nearest_idx,
                    dom_id,
                    best_coh,
                    centroids[nearest_idx].phase_gravity_mass,
                    false,
                );
            }
        };

        let safe_split = branchless_clamp_f32(split_coherence_thresh, -0.95, 0.95);
        let (target_idx, spawned_new, cur_target_w) = Self::select_or_spawn_centroid_slot_64(
            centroids,
            frame,
            CentroidSlotCandidate {
                nearest_idx,
                best_coh,
                empty_opt,
                weakest_idx,
                best_w,
            },
            safe_split,
        );
        let (winning_anchor_id, final_coh, updated_mass) = Self::commit_frame_absorption_at_index(
            centroids,
            CentroidAbsorptionPlan {
                target_idx,
                spawned_new,
                best_coh,
                best_is_uniform,
                sample_is_uniform,
                cur_target_w: &cur_target_w,
            },
            frame,
            grounded_w,
            trail_opt,
        );

        (
            target_idx,
            winning_anchor_id,
            final_coh,
            updated_mass,
            spawned_new,
        )
    }
}

impl PhaseAnchorRecord {
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
        trail_opt: Option<&mut PointerTrail>,
    ) -> DissipativeOverflowReport {
        let k_len = centroids.len();
        if k_len == 0 {
            return DissipativeOverflowReport::default();
        }

        let is_repulsed = (frame.repulsion_severity > 0.50)
            | ((frame.action_mode_flags & ACTION_MODE_PHASE_REPULSE) != 0)
            | (frame.root_cause_code == ROOT_CAUSE_COHOMOLOGY_REPULSION)
            | ((frame.lineage_flags & CAUSAL_CAUSE_COHOMOLOGY_REPULSION) != 0);
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
                residual_words: res_w,
                primary_anchor_id: primary_aid,
                landed_anchor_id: primary_aid,
                primary_coherence: coh_1,
                scatter_coherence: 0.0,
                overflow_pressure,
                updated_gravity_mass: primary_mass,
                primary_idx: idx_1 as u16,
                landed_idx: idx_1 as u16,
                overflowed: false,
                scattered: false,
                spawned_new: false,
                _reserved: 0,
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
        let grounded_absorbed_w = match Scoped::from_boundary(PhaseWordsRef64(&absorbed_w))
            .cut_by_lacuna(!frame.is_repulsed(), frame.repulsion_severity, 0.50)
        {
            ScopedCut::SelfRef(aligned) => aligned.into_grounded(),
            _ => {
                return DissipativeOverflowReport {
                    residual_words: res_w,
                    primary_anchor_id: primary_aid,
                    landed_anchor_id: primary_aid,
                    primary_coherence: coh_1,
                    scatter_coherence: clean_scatter_coh,
                    overflow_pressure,
                    updated_gravity_mass: primary_mass,
                    primary_idx: idx_1 as u16,
                    landed_idx: idx_1 as u16,
                    overflowed: false,
                    scattered: false,
                    spawned_new: false,
                    _reserved: 0,
                };
            }
        };
        if effective_spawn {
            centroids[target_idx].init_spawned_slot_header_in_place(
                assigned_spawn_id,
                frame.dominant_slot,
                frame.mapped_section_offset48,
            );
        }

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
            CentroidAbsorptionPlan {
                target_idx,
                spawned_new: effective_spawn,
                best_coh: target_coh,
                best_is_uniform: target_uni,
                sample_is_uniform: absorbed_uni,
                cur_target_w: &cur_target_w,
            },
            frame,
            grounded_absorbed_w,
            trail_opt,
        );

        DissipativeOverflowReport {
            residual_words: res_w,
            primary_anchor_id: primary_aid,
            landed_anchor_id: landed_aid,
            primary_coherence: coh_1,
            scatter_coherence: clean_scatter_coh,
            overflow_pressure,
            updated_gravity_mass: updated_mass,
            primary_idx: idx_1 as u16,
            landed_idx: target_idx as u16,
            overflowed,
            scattered: settled_on_scatter,
            spawned_new: effective_spawn,
            _reserved: 0,
        }
    }

    /// Autonomously nucleates a new attractor inside `centroids` when approximation noise in
    /// [`OverlapDiscoveryReport`] pushes a fragmented granule outside all existing
    /// anchors' lacuna apertures into `Liminal` (`discovery_triggered != 0`).
    ///
    /// Returns `(spawned_slot_idx, spawned_anchor_id, did_nucleate)` in branchless constant time
    /// relative to centroid slot contents.
    #[inline]
    pub fn nucleate_discovered_anchor_from_lacuna_64(
        centroids: &mut [Self],
        overlap_report: &OverlapDiscoveryReport,
        drifted_words: &[u64; NUM_STIGMERGIC_GRANULES],
        new_anchor_id: u32,
        trail_opt: Option<&mut PointerTrail>,
    ) -> (usize, u32, bool) {
        if (overlap_report.discovery_triggered == 0) || centroids.is_empty() {
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

        let seed_energy = branchless_max_f32(overlap_report.lacuna_excursion_energy, 0.25);
        let can_evict = (!has_empty) & (seed_energy > weakest_mass);
        if !(has_empty | can_evict) {
            return (0, 0, false);
        }
        let grounded_w = match Scoped::from_delta(PhaseWordsRef64(drifted_words)).cut_by_lacuna_q7(
            overlap_report.discovery_triggered != 0,
            127,
            0,
        ) {
            ScopedCut::Exterior(spec) => {
                match spec.bridge_to_shared(overlap_report.discovery_excursion_mask | 1) {
                    Ok(bridged) => bridged.into_grounded(),
                    Err(_) => return (0, 0, false),
                }
            }
            _ => return (0, 0, false),
        };
        let target_idx = branchless_select_usize(has_empty, empty_idx, weakest_idx);
        let eff_anchor_id =
            branchless_select_u32(new_anchor_id != 0, new_anchor_id, (target_idx as u32) + 1);

        let mut seed_frame = GatewayFrame {
            target_anchor_id: eff_anchor_id,
            effective_energy: seed_energy,
            phase_clamp_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
            phase_step_min_rad: DEFAULT_PHASE_STEP_MIN_RAD,
            effective_inv_temp: 1.0,
            lineage_flags: CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
            dominant_slot: ATTRACTOR_SLOT_CRYSTALLIZED,
            action_mode_flags: ACTION_PRESET_EXPRESS_PHASE,
            active_swarm_cell_mask: overlap_report.discovery_excursion_mask | 0x01,
            ..GatewayFrame::default()
        };
        *seed_frame.phase_words_mut() = *drifted_words;
        seed_frame.mean_phase_tag_q7 =
            swar_q7_mean_i8_from_words_with_uniform_flag_64(drifted_words, false);

        centroids[target_idx].init_spawned_slot_header_in_place(
            eff_anchor_id,
            ATTRACTOR_SLOT_CRYSTALLIZED,
            0,
        );
        let empty_cur = [0u64; NUM_STIGMERGIC_GRANULES];
        let (final_aid, _coh, _mass) = Self::commit_frame_absorption_at_index(
            centroids,
            CentroidAbsorptionPlan {
                target_idx,
                spawned_new: true,
                best_coh: 1.0,
                best_is_uniform: true,
                sample_is_uniform: is_word_array_uniform_64(drifted_words),
                cur_target_w: &empty_cur,
            },
            &seed_frame,
            grounded_w,
            trail_opt,
        );
        (target_idx, final_aid, true)
    }

    /// Executes Online Toroidal $k$-Means / DP-Means across a batch of [`GatewayFrame`]s,
    /// returning `(last_assigned_idx, last_assigned_anchor_id, mean_coherence, spawned_clusters_count)`.
    #[cfg(test)]
    #[inline]
    pub(crate) fn step_online_toroidal_kmeans_batch_frames_64(
        centroids: &mut [Self],
        frames: &[GatewayFrame],
        split_coherence_thresh: f32,
        mut trail_opt: Option<&mut PointerTrail>,
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
}

impl CausalProvenanceSignal for PhaseAnchorRecord {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let is_mature = (self.sample_count >= 2) & (self.resultant_r_bar >= 0.45);
        self.lineage_flags
            | branchless_select_u32(is_mature, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, 0)
    }
}

impl CognitiveGroundingSignal for PhaseAnchorRecord {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        self.phase_profile
    }
    #[inline(always)]
    fn grounding_phases_ref_64(&self) -> Option<&[PhaseQ7; MAX_ATTENTION_HEADS]> {
        Some(&self.phase_profile)
    }
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        *self.phase_words_ref()
    }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        self.phase_gravity_mass
    }
    #[inline(always)]
    fn mapped_section_offset48(&self) -> u64 {
        self.mapped_section_offset48
    }
    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 {
        self.anchor_id
    }
}

impl ObjectiveFactSignal for PhaseAnchorRecord {
    #[inline(always)]
    fn fact_phase_clamp_rad(&self) -> f32 {
        self.phase_clamp_rad
    }
}

impl SubjectivePersonaSignal for PhaseAnchorRecord {
    #[inline(always)]
    fn persona_phase_clamp_rad(&self) -> f32 {
        self.phase_clamp_rad
    }
}

impl ManifoldGatewaySignal for PhaseAnchorRecord {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let is_active = (self.sample_count > 0) & (self.phase_gravity_mass > DEFAULT_ENTROPY_EPS);
        if !is_active {
            let packed_tag =
                pack_action_slot_tag(self.dominant_slot, ACTION_PRESET_PASSIVE_SIDECAR);
            return (
                GatewayFrame {
                    tagged_ptr: TaggedPtr::pack(self.mapped_section_offset48, packed_tag, 0).raw(),
                    mapped_section_offset48: self.mapped_section_offset48,
                    target_anchor_id: self.anchor_id,
                    dominant_slot: self.dominant_slot,
                    action_mode_flags: ACTION_PRESET_PASSIVE_SIDECAR,
                    active_swarm_cell_mask: self.active_swarm_mask,
                    root_cause_code: self.root_cause_code,
                    frame_flags: packed_tag as u16,
                    ..GatewayFrame::default()
                },
                [0u64; NUM_STIGMERGIC_GRANULES],
            );
        }
        let is_mature = (self.sample_count >= 2) & (self.resultant_r_bar >= 0.45);
        let w = *self.phase_words_ref();
        let tag_q7 = fast_mean_q7_tag_from_words_64(&w);
        let eff_action = self.action_mode_flags;
        let packed_tag = pack_action_slot_tag(self.dominant_slot, eff_action);
        let tagged_ptr = TaggedPtr::pack(self.mapped_section_offset48, packed_tag, tag_q7).raw();
        let prov_bits = self.lineage_flags
            | branchless_select_u32(is_mature, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, 0);

        let is_fact_slot = self.dominant_slot == ATTRACTOR_SLOT_KNOWLEDGE;
        let is_persona_slot = self.dominant_slot == ATTRACTOR_SLOT_PERSONA;
        let subhead_fact = branchless_select_u8(
            is_fact_slot,
            36,
            branchless_select_u8(is_persona_slot, 16, 24),
        );
        let subhead_logic = branchless_select_u8(is_fact_slot, 20, 24);
        let subhead_explore = branchless_select_u8(
            is_fact_slot,
            8,
            branchless_select_u8(is_persona_slot, 24, 16),
        );
        let fact_gain_q7 = branchless_select_u8(is_fact_slot, 147, 128);
        let explore_gain_q7 = branchless_select_u8(is_persona_slot, 154, 128);
        let logic_gain_q7 = ((fact_gain_q7 as u16 + explore_gain_q7 as u16) >> 1) as u8;

        (
            GatewayFrame {
                tagged_ptr,
                mapped_section_offset48: self.mapped_section_offset48,
                target_anchor_id: self.anchor_id,
                effective_energy: branchless_min_f32(
                    self.phase_gravity_mass,
                    MAX_ATTRACTOR_PULL_FORCE,
                ),
                phase_clamp_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
                phase_step_min_rad: DEFAULT_PHASE_STEP_MIN_RAD,
                effective_inv_temp: self.effective_inv_temp,
                fact_persona_coherence: self.resultant_r_bar,
                repulsion_severity: 0.0,
                lineage_flags: prov_bits,
                root_cause_code: self.root_cause_code,
                dominant_slot: self.dominant_slot,
                action_mode_flags: eff_action,
                active_swarm_cell_mask: self.active_swarm_mask,
                phase_step_min_q7: branchless_rad_to_phase_step_u8(DEFAULT_PHASE_STEP_MIN_RAD),
                subhead_fact_heads: subhead_fact,
                subhead_logic_heads: subhead_logic,
                subhead_explore_heads: subhead_explore,
                subhead_fact_gain_q7: fact_gain_q7,
                subhead_logic_gain_q7: logic_gain_q7,
                subhead_explore_gain_q7: explore_gain_q7,
                mean_phase_tag_q7: tag_q7,
                active_head_count: MAX_ATTENTION_HEADS as u16,
                frame_flags: packed_tag as u16,
                phase_deltas: self.phase_profile,
            },
            w,
        )
    }

    #[inline(always)]
    fn synthesize_gateway_ensemble_item(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let is_active = (self.sample_count > 0) & (self.phase_gravity_mass > DEFAULT_ENTROPY_EPS);
        if !is_active {
            return (
                GatewayFrame {
                    mapped_section_offset48: self.mapped_section_offset48,
                    target_anchor_id: self.anchor_id,
                    dominant_slot: self.dominant_slot,
                    action_mode_flags: ACTION_PRESET_PASSIVE_SIDECAR,
                    active_swarm_cell_mask: self.active_swarm_mask,
                    root_cause_code: self.root_cause_code,
                    ..GatewayFrame::default()
                },
                [0u64; NUM_STIGMERGIC_GRANULES],
            );
        }
        let is_mature = (self.sample_count >= 2) & (self.resultant_r_bar >= 0.45);
        let w = *self.phase_words_ref();
        let prov_bits = self.lineage_flags
            | branchless_select_u32(is_mature, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, 0);
        let is_fact_slot = self.dominant_slot == ATTRACTOR_SLOT_KNOWLEDGE;
        let is_persona_slot = self.dominant_slot == ATTRACTOR_SLOT_PERSONA;
        let subhead_fact = branchless_select_u8(
            is_fact_slot,
            36,
            branchless_select_u8(is_persona_slot, 16, 24),
        );
        let subhead_logic = branchless_select_u8(is_fact_slot, 20, 24);
        let subhead_explore = branchless_select_u8(
            is_fact_slot,
            8,
            branchless_select_u8(is_persona_slot, 24, 16),
        );
        let fact_gain_q7 = branchless_select_u8(is_fact_slot, 147, 128);
        let explore_gain_q7 = branchless_select_u8(is_persona_slot, 154, 128);
        let logic_gain_q7 = ((fact_gain_q7 as u16 + explore_gain_q7 as u16) >> 1) as u8;

        (
            GatewayFrame {
                tagged_ptr: 0,
                mapped_section_offset48: self.mapped_section_offset48,
                target_anchor_id: self.anchor_id,
                effective_energy: branchless_min_f32(
                    self.phase_gravity_mass,
                    MAX_ATTRACTOR_PULL_FORCE,
                ),
                phase_clamp_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
                phase_step_min_rad: DEFAULT_PHASE_STEP_MIN_RAD,
                effective_inv_temp: self.effective_inv_temp,
                fact_persona_coherence: self.resultant_r_bar,
                repulsion_severity: 0.0,
                lineage_flags: prov_bits,
                root_cause_code: self.root_cause_code,
                dominant_slot: self.dominant_slot,
                action_mode_flags: self.action_mode_flags,
                active_swarm_cell_mask: self.active_swarm_mask,
                phase_step_min_q7: branchless_rad_to_phase_step_u8(DEFAULT_PHASE_STEP_MIN_RAD),
                subhead_fact_heads: subhead_fact,
                subhead_logic_heads: subhead_logic,
                subhead_explore_heads: subhead_explore,
                subhead_fact_gain_q7: fact_gain_q7,
                subhead_logic_gain_q7: logic_gain_q7,
                subhead_explore_gain_q7: explore_gain_q7,
                mean_phase_tag_q7: 0,
                active_head_count: MAX_ATTENTION_HEADS as u16,
                frame_flags: 0,
                phase_deltas: [0i8; MAX_ATTENTION_HEADS],
            },
            w,
        )
    }

    #[inline(always)]
    fn has_active_gateway_ensemble_signal(&self) -> bool {
        (self.sample_count > 0) & (self.phase_gravity_mass > DEFAULT_ENTROPY_EPS)
    }

    #[inline]
    fn synthesize_gateway_slice_with_words(
        records: &[Self],
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let len = records.len();
        if len == 0 {
            return (GatewayFrame::default(), [0u64; NUM_STIGMERGIC_GRANULES]);
        }
        if len == 1 {
            return records[0].synthesize_gateway_frame_with_words();
        }

        let mut acc_w = [0u64; NUM_STIGMERGIC_GRANULES];
        let mut acc_is_uniform = true;
        let mut tot_e = 0.0_f32;
        let mut best_e = 0.0_f32;
        let mut weighted_inv_temp = 0.0_f32;
        let mut weighted_coh = 0.0_f32;
        let mut dom_idx = 0usize;
        let mut dom_storage_addr = 0u64;
        let mut action_flags = ACTION_PRESET_PASSIVE_SIDECAR;
        let mut swarm_mask = 0xFFu8;
        let mut prov_bits = 0u32;
        let mut dom_cause = ROOT_CAUSE_NOMINAL;
        let mut has_active = false;

        let mut i = 0usize;
        while i < len {
            let rec = &records[i];
            if (rec.sample_count > 0) & (rec.phase_gravity_mass > DEFAULT_ENTROPY_EPS) {
                let rec_w = rec.phase_words_ref();
                let rec_is_uniform = if !has_active || acc_is_uniform {
                    is_word_array_uniform_64(rec_w)
                } else {
                    false
                };
                let f_e = branchless_min_f32(
                    branchless_flush_subnormal_f32(branchless_relu_f32(rec.phase_gravity_mass)),
                    MAX_ATTRACTOR_PULL_FORCE,
                );
                let is_mature = (rec.sample_count >= 2) & (rec.resultant_r_bar >= 0.45);
                let rec_prov = rec.lineage_flags
                    | branchless_select_u32(is_mature, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, 0);

                if !has_active {
                    acc_w = *rec_w;
                    acc_is_uniform = rec_is_uniform;
                    tot_e = f_e;
                    best_e = f_e;
                    weighted_inv_temp = f_e * rec.effective_inv_temp;
                    weighted_coh = f_e * rec.resultant_r_bar;
                    dom_idx = i;
                    dom_storage_addr = rec.mapped_section_offset48;
                    action_flags = rec.action_mode_flags;
                    swarm_mask = rec.active_swarm_mask;
                    prov_bits = rec_prov;
                    dom_cause = rec.root_cause_code;
                    has_active = true;
                } else if f_e > DEFAULT_ENTROPY_EPS {
                    let new_tot = tot_e + f_e;
                    if !swar_words_equal_64(&acc_w, rec_w) {
                        let safe_tot =
                            branchless_select_f32(new_tot > DEFAULT_ENTROPY_EPS, new_tot, 1.0);
                        let w_ratio =
                            branchless_clamp_f32(f_e * fast_inv_pos_normal_f32(safe_tot), 0.0, 1.0);
                        let s_u8 = ((w_ratio * 128.0 + 0.5) as u8).min(128);
                        let s_u64 = s_u8 as u64;
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
                    weighted_inv_temp += f_e * rec.effective_inv_temp;
                    weighted_coh += f_e * rec.resultant_r_bar;
                    let take_dom = f_e > best_e;
                    best_e = branchless_select_f32(take_dom, f_e, best_e);
                    dom_idx = branchless_select_usize(take_dom, i, dom_idx);
                    dom_storage_addr = branchless_select_u64(
                        take_dom & (rec.mapped_section_offset48 != 0),
                        rec.mapped_section_offset48,
                        branchless_select_u64(
                            dom_storage_addr != 0,
                            dom_storage_addr,
                            rec.mapped_section_offset48,
                        ),
                    );
                    action_flags |= rec.action_mode_flags;
                    swarm_mask &= rec.active_swarm_mask;
                    prov_bits |= rec_prov;
                    dom_cause = branchless_select_u8(
                        rec.root_cause_code > dom_cause,
                        rec.root_cause_code,
                        dom_cause,
                    );
                    tot_e = new_tot;
                }
            }
            i += 1;
        }

        let safe_tot_e = branchless_select_f32(tot_e > DEFAULT_ENTROPY_EPS, tot_e, 1.0);
        let inv_tot_e = fast_inv_pos_normal_f32(safe_tot_e);
        let eff_inv_temp = branchless_select_f32(
            tot_e > DEFAULT_ENTROPY_EPS,
            branchless_clamp_f32(weighted_inv_temp * inv_tot_e, 0.25, 4.0),
            records[dom_idx].effective_inv_temp,
        );
        let fp_coh = branchless_select_f32(
            tot_e > DEFAULT_ENTROPY_EPS,
            branchless_clamp_f32(weighted_coh * inv_tot_e, 0.0, 1.0),
            records[dom_idx].resultant_r_bar,
        );
        let dom = &records[dom_idx];
        let dom_slot = dom.dominant_slot;
        let is_fact_slot = dom_slot == ATTRACTOR_SLOT_KNOWLEDGE;
        let is_persona_slot = dom_slot == ATTRACTOR_SLOT_PERSONA;
        let subhead_fact = branchless_select_u8(
            is_fact_slot,
            36,
            branchless_select_u8(is_persona_slot, 16, 24),
        );
        let subhead_logic = branchless_select_u8(is_fact_slot, 20, 24);
        let subhead_explore = branchless_select_u8(
            is_fact_slot,
            8,
            branchless_select_u8(is_persona_slot, 24, 16),
        );
        let fact_gain_q7 = branchless_select_u8(is_fact_slot, 147, 128);
        let explore_gain_q7 = branchless_select_u8(is_persona_slot, 154, 128);
        let logic_gain_q7 = ((fact_gain_q7 as u16 + explore_gain_q7 as u16) >> 1) as u8;

        let phase_deltas = words_to_q7_array_64(&acc_w);
        let tag_q7 = swar_q7_mean_i8_from_words_with_uniform_flag_64(&acc_w, acc_is_uniform);
        let packed_tag = pack_action_slot_tag(dom_slot, action_flags);
        let tagged_ptr = TaggedPtr::pack(dom_storage_addr, packed_tag, tag_q7).raw();

        (
            GatewayFrame {
                tagged_ptr,
                mapped_section_offset48: dom_storage_addr,
                target_anchor_id: dom.anchor_id,
                effective_energy: branchless_min_f32(tot_e, MAX_ATTRACTOR_PULL_FORCE),
                phase_clamp_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
                phase_step_min_rad: DEFAULT_PHASE_STEP_MIN_RAD,
                effective_inv_temp: eff_inv_temp,
                fact_persona_coherence: fp_coh,
                repulsion_severity: 0.0,
                lineage_flags: prov_bits,
                root_cause_code: dom_cause,
                dominant_slot: dom_slot,
                action_mode_flags: action_flags,
                active_swarm_cell_mask: swarm_mask,
                phase_step_min_q7: branchless_rad_to_phase_step_u8(DEFAULT_PHASE_STEP_MIN_RAD),
                subhead_fact_heads: subhead_fact,
                subhead_logic_heads: subhead_logic,
                subhead_explore_heads: subhead_explore,
                subhead_fact_gain_q7: fact_gain_q7,
                subhead_logic_gain_q7: logic_gain_q7,
                subhead_explore_gain_q7: explore_gain_q7,
                mean_phase_tag_q7: tag_q7,
                active_head_count: MAX_ATTENTION_HEADS as u16,
                frame_flags: packed_tag as u16,
                phase_deltas,
            },
            acc_w,
        )
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}
