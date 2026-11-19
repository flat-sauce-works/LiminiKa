// name: src/core/src/stigmergy/crystal/kmeans.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Toroidal centroid merging and evaporative pruning for [`CrystallizedAttractorRecord`].

#![allow(clippy::too_many_arguments)]

use crate::foundation::{
    branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_maturity_scale_u32,
    branchless_max_f32, branchless_min_f32,
    branchless_toroidal_centroid_step_words_with_coh_and_uni_64,
    coherence_above_floor_from_word_arrays_64, crc32_update_crystal_energy_and_mass_delta,
    is_word_array_uniform_64,
};
use crate::liminal::{
    ACTION_PRESET_EXPRESS_PHASE, ACTION_PRESET_PASSIVE_SIDECAR,
    CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, DEFAULT_ENTROPY_EPS, GCSO_CRYSTAL_RECORD_MAGIC,
    GCSO_CRYSTAL_RECORD_VERSION, NUM_STIGMERGIC_GRANULES,
};
use crate::stigmergy::MAX_ATTRACTOR_PULL_FORCE;

use super::CrystallizedAttractorRecord;

impl CrystallizedAttractorRecord {
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
