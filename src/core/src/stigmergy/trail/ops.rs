// name: src/core/src/stigmergy/trail/ops.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tagged-pointer action mode helpers and core [`PointerTrail`] updates.

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use super::eval_candidate_trail_avx2_with_macro_fast;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
use super::eval_candidate_trail_words_with_macro_fast;
use super::{
    precompute_bayesian_trail_scales, trail_phase_words_mut, trail_phase_words_ref, PointerTrail,
};
use crate::foundation::{
    branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_is_finite_f32,
    branchless_min_f32, branchless_phase_proximity_gravity_f32, branchless_relu_f32,
    circular_resultant_from_words_64, coherence_from_word_arrays_64, fast_inv_len_f32,
    is_word_array_uniform_64, load_q7_words_64, pack_action_slot_tag,
    swar_or_avx2_q7_add_sat_words_64, swar_q7_add_sat_x8, swar_q7_neg_sat_x8,
    swar_slerp_words_with_bias_64, swar_slerp_words_with_uniform_flag_64,
    unpack_action_mode_from_tag, unpack_attractor_slot_from_tag, TaggedPtr, ACTION_MODE_LAYER_SKIP,
    ACTION_MODE_PHASE_REPULSE, ACTION_MODE_PHASE_STEER, ACTION_MODE_RANK1_INJECT,
    ACTION_MODE_SUBHEAD_ROUTE, ATTRACTOR_SLOT_ID_MASK, DEFAULT_PHASE_GRAVITY_EPS,
    SWAR_SPLAT_BYTE_64,
};
use crate::liminal::{
    branchless_select_f32, branchless_select_u32, branchless_select_u64, branchless_select_u8,
    PhaseQ7, MAX_ATTENTION_HEADS, NUM_STIGMERGIC_GRANULES,
};
use crate::stigmergy::pheromone::PheromoneDensityOps;
use crate::stigmergy::{
    ATTRACTOR_SLOT_REPULSION, MAX_STIGMERGIC_DENSITY, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE,
    TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, TRAIL_FLAG_PHASE_STEER_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE,
    TRAIL_FLAG_SWARM_ESCALATED, TRAIL_FLAG_SWARM_RESOLVED, TRAIL_SWARM_QUORUM_MASK,
    TRAIL_SWARM_QUORUM_SHIFT,
};

// ===================================================================
// 3. Pointer-Driven Action Mode Orchestration (`TaggedPtr`)
// ===================================================================

impl TaggedPtr {
    /// Packs a 48-bit address, 3-bit target working lane (`0..=7`), 5-bit action mode mask, and Q7 phase.
    #[inline(always)]
    #[must_use]
    pub fn pack_action(
        raw_addr: u64,
        attractor_slot_id: u8,
        action_mode_flags: u8,
        phase_q7: PhaseQ7,
    ) -> Self {
        Self::pack(
            raw_addr,
            pack_action_slot_tag(attractor_slot_id, action_mode_flags),
            phase_q7,
        )
    }

    /// Extracts the 5-bit Action Mode execution bitmask (`ACTION_MODE_*`).
    #[inline(always)]
    #[must_use]
    pub fn action_mode_flags(self) -> u8 {
        unpack_action_mode_from_tag(self.unpack_tag())
    }

    /// Extracts the 3-bit Target Attractor Working Lane ID (`0..=7`).
    #[inline(always)]
    #[must_use]
    pub fn target_attractor_slot(self) -> u8 {
        unpack_attractor_slot_from_tag(self.unpack_tag())
    }

    /// Returns `true` if all specified `mode_mask` bits are enabled in the pointer's Action Mode.
    #[inline(always)]
    #[must_use]
    pub fn has_action_mode(self, mode_mask: u8) -> bool {
        (self.action_mode_flags() & mode_mask) == mode_mask
    }

    /// Returns a new tagged pointer with updated Target Attractor Working Lane ID and Action Mode flags.
    #[inline(always)]
    #[must_use]
    pub fn with_action_and_slot(self, attractor_slot_id: u8, action_mode_flags: u8) -> Self {
        self.with_tag(pack_action_slot_tag(attractor_slot_id, action_mode_flags))
    }

    /// Branchlessly mutates the pointer's Action Mode flags, Target Working Lane ID, and embedded
    /// Q7 phase based on bottom-up stigmergic environmental traces.
    #[inline(always)]
    #[must_use]
    pub fn evolve_mode_stigmergically(
        self,
        trail_density: f32,
        entropy_h_tilde: f32,
        repulsion_triggered: bool,
        dominant_attractor_slot: u8,
        low_entropy_skip_thresh: f32,
        high_density_rank1_thresh: f32,
    ) -> Self {
        let finite_h = branchless_is_finite_f32(entropy_h_tilde);
        let clean_h = branchless_select_f32(finite_h, entropy_h_tilde, 0.0);
        let clean_rho = branchless_relu_f32(trail_density);

        let active_signal = (!self.is_null()) | (self.unpack_phase() != 0) | (clean_rho > 0.0);
        let m_phase_steer = branchless_select_u8(active_signal, ACTION_MODE_PHASE_STEER, 0);
        let can_skip =
            active_signal & (clean_h <= low_entropy_skip_thresh) & (!repulsion_triggered);
        let m_skip = branchless_select_u8(can_skip, ACTION_MODE_LAYER_SKIP, 0);
        let m_rep = branchless_select_u8(repulsion_triggered, ACTION_MODE_PHASE_REPULSE, 0);
        let need_subhead = (clean_h > low_entropy_skip_thresh) | repulsion_triggered;
        let m_subhead = branchless_select_u8(need_subhead, ACTION_MODE_SUBHEAD_ROUTE, 0);
        let need_rank1 =
            (clean_rho >= high_density_rank1_thresh) & (high_density_rank1_thresh > 0.0);
        let m_rank1 = branchless_select_u8(need_rank1, ACTION_MODE_RANK1_INJECT, 0);

        let next_mode = m_phase_steer | m_skip | m_rep | m_subhead | m_rank1;
        let next_slot = branchless_select_u8(
            repulsion_triggered,
            ATTRACTOR_SLOT_REPULSION,
            dominant_attractor_slot & ATTRACTOR_SLOT_ID_MASK,
        );

        let updated = self.with_action_and_slot(next_slot, next_mode);
        let conjugated = updated.conjugate_phase();
        Self(branchless_select_u64(
            repulsion_triggered,
            conjugated.raw(),
            updated.raw(),
        ))
    }
}

/// Register-resident Coarse-to-Fine Bayesian candidate trail selector, rewirer, Limina geodesic
/// bridger, and Attention score modulator accepting `self_w` directly in 8 `u64` SWAR registers
/// and returning the post-bridged 8 `u64` SWAR words (`0` redundant stack/memory reloads).
#[inline]
pub(crate) fn select_rewire_bridge_and_modulate_with_preloaded_words_64(
    trail: &mut PointerTrail,
    mut self_w: [u64; NUM_STIGMERGIC_GRANULES],
    candidate_trails: &[PointerTrail],
    mut attention_scores_opt: Option<&mut [f32]>,
    head_dim: u32,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    contradiction_threshold: f32,
    liminal_interpolation_u7: u8,
    repulsion_penalty: f32,
    macro_anchor_id: u32,
    win_w_opt: Option<&[u64; NUM_STIGMERGIC_GRANULES]>,
    cluster_bonus_gain: f32,
) -> (
    u32,
    f32,
    f32,
    u32,
    bool,
    [u64; NUM_STIGMERGIC_GRANULES],
    f32,
) {
    let cand_len = candidate_trails.len();
    if cand_len == 0 {
        return (trail.linked_trail_id, 0.0, 0.0, 0, false, self_w, 0.0);
    }

    let (phase_scale, clean_eta, clean_link, safe_thresh) = precompute_bayesian_trail_scales(
        head_dim,
        phase_interference_gain,
        stigmergic_prior_gain,
        pointer_link_bonus,
        contradiction_threshold,
    );
    let clean_penalty = branchless_flush_subnormal_f32(branchless_relu_f32(repulsion_penalty));
    let clean_cluster_bonus =
        branchless_flush_subnormal_f32(branchless_relu_f32(cluster_bonus_gain));
    let self_is_uniform = is_word_array_uniform_64(&self_w);
    let self_b0 = (self_w[0] & 0xFF) as u8;
    let self_is_byte_uniform = self_is_uniform & (self_w[0] == self_w[0].rotate_right(8));
    // SAFETY: `target_feature = "avx2"` is checked by `#[cfg]`, and `self_w` has 8 `u64` words.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    let (self_v0, self_v1) =
        unsafe { crate::foundation::avx2_words64_to_vec64_with_uniform(&self_w, self_is_uniform) };
    let mut win_meta_evaluated = false;
    let mut win_eq_self = false;
    let mut win_is_uniform = false;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    let (mut win_v0, mut win_v1) = (self_v0, self_v1);
    let cur_linked_id = trail.linked_trail_id;

    let mut best_id = cur_linked_id;
    let mut best_idx = 0usize;
    let mut best_score = -1.0e30_f32;
    let mut best_coh = 0.0_f32;
    let mut best_is_uniform = self_is_uniform;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    let mut best_vec_opt: Option<(core::arch::x86_64::__m256i, core::arch::x86_64::__m256i)> = None;
    let mut found_valid = false;
    let mut coh_sum = 0.0_f32;
    let mut coherent_count = 0u32;

    let mut idx = 0usize;
    while idx < cand_len {
        let cand = &candidate_trails[idx];
        let cand_id = idx as u32;
        // SAFETY: `target_feature = "avx2"` is checked by `#[cfg]`, and all trail arguments are valid references.
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let (bayes_delta, coh, is_coherent, cand_is_uniform, cand_vec_opt) = unsafe {
            eval_candidate_trail_avx2_with_macro_fast(
                &self_w,
                self_is_uniform,
                self_is_byte_uniform,
                self_b0,
                self_v0,
                self_v1,
                cur_linked_id,
                cand_id,
                cand,
                phase_scale,
                clean_eta,
                clean_link,
                safe_thresh,
                macro_anchor_id,
                win_w_opt,
                &mut win_meta_evaluated,
                &mut win_is_uniform,
                &mut win_eq_self,
                &mut win_v0,
                &mut win_v1,
                clean_cluster_bonus,
            )
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        let (bayes_delta, coh, is_coherent, cand_is_uniform) =
            eval_candidate_trail_words_with_macro_fast(
                &self_w,
                self_is_uniform,
                self_is_byte_uniform,
                self_b0,
                cur_linked_id,
                cand_id,
                cand,
                phase_scale,
                clean_eta,
                clean_link,
                safe_thresh,
                macro_anchor_id,
                win_w_opt,
                &mut win_meta_evaluated,
                &mut win_is_uniform,
                &mut win_eq_self,
                clean_cluster_bonus,
            );

        if let Some(ref mut scores) = attention_scores_opt {
            if idx < scores.len() {
                let penalty = branchless_select_f32(is_coherent, 0.0, clean_penalty);
                scores[idx] = branchless_flush_subnormal_f32(scores[idx] + bayes_delta - penalty);
            }
        }

        let take = is_coherent & ((!found_valid) | (bayes_delta > best_score));
        if take {
            best_id = cand_id;
            best_idx = idx;
            best_score = bayes_delta;
            best_coh = coh;
            best_is_uniform = cand_is_uniform;
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            {
                best_vec_opt = cand_vec_opt;
            }
        }
        found_valid |= is_coherent;
        coh_sum += coh;
        coherent_count = coherent_count.wrapping_add(u32::from(is_coherent));
        idx += 1;
    }

    if found_valid {
        trail.linked_trail_id = best_id;
        let best_cand = &candidate_trails[best_idx];
        let eff_s = branchless_select_u8(
            liminal_interpolation_u7 > 128,
            128,
            liminal_interpolation_u7,
        ) as u64;
        if eff_s > 0 {
            let best_w = trail_phase_words_ref(best_cand);
            if !(self_is_uniform & best_is_uniform & (self_w[0] == best_w[0])) {
                if eff_s >= 128 {
                    self_w = *best_w;
                } else {
                    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                    {
                        if self_is_uniform & best_is_uniform {
                            let bias_restore =
                                (128u64.wrapping_sub(eff_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                            self_w = swar_slerp_words_with_uniform_flag_64(
                                &self_w,
                                true,
                                best_w,
                                true,
                                eff_s,
                                bias_restore,
                            )
                            .0;
                        } else {
                            // SAFETY: `(self_v0, self_v1)` and `best_vec_opt` are already hoisted in YMM
                            // registers across the candidate loop, eliminating post-loop memory reloads.
                            unsafe {
                                let (bv0, bv1) = match best_vec_opt {
                                    Some(v) => v,
                                    None => crate::foundation::avx2_words64_to_vec64_with_uniform(
                                        best_w,
                                        best_is_uniform,
                                    ),
                                };
                                let (rv0, rv1) = crate::foundation::avx2_slerp_vec64(
                                    self_v0, self_v1, bv0, bv1, eff_s,
                                );
                                self_w = crate::foundation::avx2_vec64_to_words64(rv0, rv1);
                            }
                        }
                    }
                    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                    {
                        let bias_restore =
                            (128u64.wrapping_sub(eff_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                        self_w = swar_slerp_words_with_uniform_flag_64(
                            &self_w,
                            self_is_uniform,
                            best_w,
                            best_is_uniform,
                            eff_s,
                            bias_restore,
                        )
                        .0;
                    }
                }
                *trail_phase_words_mut(trail) = self_w;
            }
        }

        let best_anchor_id = best_cand.target_anchor_id;
        let adopt_anchor = (trail.target_anchor_id == 0) & (best_anchor_id != 0);
        trail.target_anchor_id =
            branchless_select_u32(adopt_anchor, best_anchor_id, trail.target_anchor_id);
        trail.cluster_id = branchless_select_u32(
            trail.target_anchor_id != 0,
            trail.target_anchor_id,
            trail.cluster_id,
        );

        if let Some(scores) = attention_scores_opt {
            if best_idx < scores.len() && cur_linked_id != best_id {
                scores[best_idx] = branchless_flush_subnormal_f32(scores[best_idx] + clean_link);
            }
        }
    }

    let mean_coh = branchless_clamp_f32(coh_sum * fast_inv_len_f32(cand_len), -1.0, 1.0);
    let out_score = branchless_select_f32(found_valid, best_score, 0.0);
    let out_best_coh = branchless_select_f32(found_valid, best_coh, 0.0);
    (
        best_id,
        out_score,
        mean_coh,
        coherent_count,
        found_valid,
        self_w,
        out_best_coh,
    )
}

impl PointerTrail {
    /// Copies `src` and pre-loaded `phase_words` into `self` using width-matched scalar writes and
    /// 64-byte aligned 64-bit GPR stores (`*trail_phase_words_mut(self) = *phase_words`) at byte
    /// offset `64..128`, avoiding unaligned 256-bit `vmovdqu` cache-line splits, 256-bit-to-64-bit
    /// Store-to-Load Forwarding (STLF) pipeline stalls, and redundant 64-byte reloads from `src`.
    #[inline(always)]
    pub fn copy_from_with_words_fast(
        &mut self,
        src: &Self,
        phase_words: &[u64; NUM_STIGMERGIC_GRANULES],
    ) {
        self.current_ptr = src.current_ptr;
        self.prev_ptr = src.prev_ptr;
        self.user_data = src.user_data;
        self.transition_cost = src.transition_cost;
        self.step_count = src.step_count;
        self.stigmergic_density = src.stigmergic_density;
        self.target_anchor_id = src.target_anchor_id;
        self.cluster_id = src.cluster_id;
        self.linked_trail_id = src.linked_trail_id;
        self.attractor_pull_force = src.attractor_pull_force;
        self.flags = src.flags;
        self.reserved_header = src.reserved_header;
        *trail_phase_words_mut(self) = *phase_words;
    }

    /// Restores `self` from `src` using width-matched scalar writes and 64-bit GPR stores
    /// (`*trail_phase_words_mut(self) = *trail_phase_words_ref(src)`), preventing 256-bit YMM-to-GPR
    /// Store-to-Load Forwarding (STLF) pipeline stalls (`~15-20 cycles`).
    #[inline(always)]
    pub fn copy_from_fast(&mut self, src: &Self) {
        self.copy_from_with_words_fast(src, trail_phase_words_ref(src));
    }

    /// Clears `self` in-place using width-matched 64-bit GPR stores (`0` YMM stores, `0` STLF stalls).
    #[inline(always)]
    pub fn clear_fast(&mut self) {
        self.current_ptr = 0;
        self.prev_ptr = 0;
        self.user_data = 0;
        self.transition_cost = 0;
        self.step_count = 0;
        self.stigmergic_density = 0.0;
        self.target_anchor_id = 0;
        self.cluster_id = 0;
        self.linked_trail_id = 0;
        self.attractor_pull_force = 0.0;
        self.flags = 0;
        self.reserved_header = [0; 8];
        *trail_phase_words_mut(self) = [0u64; NUM_STIGMERGIC_GRANULES];
    }
}

impl PointerTrail {
    /// Records an $\mathcal{O}(1)$ tagged pointer transition and reinforces stigmergic trail density.
    #[inline(always)]
    pub fn record_pointer_transition(&mut self, next_ptr: u64, edge_cost: i32, density_boost: f32) {
        self.prev_ptr = self.current_ptr;
        self.current_ptr = next_ptr;
        self.transition_cost = self.transition_cost.saturating_add(edge_cost);
        self.step_count = self.step_count.wrapping_add(1);
        let raw_rho = branchless_flush_subnormal_f32(branchless_relu_f32(self.stigmergic_density))
            + branchless_flush_subnormal_f32(branchless_relu_f32(density_boost));
        self.stigmergic_density = branchless_min_f32(raw_rho, MAX_STIGMERGIC_DENSITY);
    }

    /// Accumulates a 64-head Q7 phase delta vector into the trail's phase state using branchless saturation.
    #[inline(always)]
    pub fn accumulate_phase_vector(&mut self, delta_phases: &[PhaseQ7; MAX_ATTENTION_HEADS]) {
        let d = load_q7_words_64(delta_phases);
        let w = trail_phase_words_mut(self);
        swar_or_avx2_q7_add_sat_words_64(w, &d);
    }

    /// Accumulates a uniform Q7 phase delta across all 64 heads in-place using 8-lane SWAR saturation.
    #[inline(always)]
    pub fn accumulate_uniform_phase(&mut self, delta_q7: PhaseQ7) {
        let splat_u64 = ((delta_q7 as u8) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let w = trail_phase_words_mut(self);
        w[0] = swar_q7_add_sat_x8(w[0], splat_u64);
        w[1] = swar_q7_add_sat_x8(w[1], splat_u64);
        w[2] = swar_q7_add_sat_x8(w[2], splat_u64);
        w[3] = swar_q7_add_sat_x8(w[3], splat_u64);
        w[4] = swar_q7_add_sat_x8(w[4], splat_u64);
        w[5] = swar_q7_add_sat_x8(w[5], splat_u64);
        w[6] = swar_q7_add_sat_x8(w[6], splat_u64);
        w[7] = swar_q7_add_sat_x8(w[7], splat_u64);
    }

    /// Smoothly blends the trail's 64-head phase trajectory toward `target_phases` along $S^1$.
    #[inline(always)]
    pub fn geodesic_blend_phase_with(
        &mut self,
        target_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
        t_u7: u8,
    ) {
        if t_u7 == 0 {
            return;
        }
        let b = load_q7_words_64(target_phases);
        if t_u7 >= 128 {
            *trail_phase_words_mut(self) = b;
            return;
        }
        let s = u64::from(t_u7);
        let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let a = *trail_phase_words_ref(self);
        *trail_phase_words_mut(self) = swar_slerp_words_with_bias_64(&a, &b, s, bias_restore);
    }

    /// Stamps a phase-conjugate repulsion marker ($-\boldsymbol{\Delta\theta}$) onto the pointer trail.
    #[inline(always)]
    pub fn stamp_phase_conjugate_repulsion(&mut self) {
        let w = trail_phase_words_mut(self);
        w[0] = swar_q7_neg_sat_x8(w[0]);
        w[1] = swar_q7_neg_sat_x8(w[1]);
        w[2] = swar_q7_neg_sat_x8(w[2]);
        w[3] = swar_q7_neg_sat_x8(w[3]);
        w[4] = swar_q7_neg_sat_x8(w[4]);
        w[5] = swar_q7_neg_sat_x8(w[5]);
        w[6] = swar_q7_neg_sat_x8(w[6]);
        w[7] = swar_q7_neg_sat_x8(w[7]);
        self.current_ptr = TaggedPtr(self.current_ptr).conjugate_phase().raw();
        self.flags |= TRAIL_FLAG_REPULSION_ACTIVE;
    }

    /// Applies subnormal-safe temporal evaporative decay (`decay_factor` $\in [0, 1]$) to trail density.
    #[inline(always)]
    pub fn apply_trail_evaporation(&mut self, decay_factor: f32) {
        let _ = self.evaporate_pheromone(decay_factor);
    }

    /// Evaluates whether the trail's stigmergic density exceeds `anchor_threshold`.
    #[inline(always)]
    pub fn should_crystallize_branchless(&self, anchor_threshold: f32) -> bool {
        (self.stigmergic_density >= anchor_threshold) & (anchor_threshold > 0.0)
    }

    /// Returns `current_ptr` wrapped as a strongly-typed [`TaggedPtr`].
    #[inline(always)]
    pub fn current_tagged_ptr(&self) -> TaggedPtr {
        TaggedPtr(self.current_ptr)
    }
    /// Returns `prev_ptr` wrapped as a strongly-typed [`TaggedPtr`].
    #[inline(always)]
    pub fn prev_tagged_ptr(&self) -> TaggedPtr {
        TaggedPtr(self.prev_ptr)
    }
    /// Extracts the active 5-bit Action Mode bitmask (`ACTION_MODE_*`) embedded in `current_ptr`.
    #[inline(always)]
    pub fn active_action_mode(&self) -> u8 {
        TaggedPtr(self.current_ptr).action_mode_flags()
    }
    /// Extracts the 3-bit Target Attractor Working Lane ID (`0..=7`) embedded in `current_ptr`.
    #[inline(always)]
    pub fn target_attractor_slot(&self) -> u8 {
        TaggedPtr(self.current_ptr).target_attractor_slot()
    }
    /// Extracts the 8-bit Micro-Swarm Quorum bitmask (`bits 8..=15`) embedded in `self.flags`.
    #[inline(always)]
    pub fn swarm_quorum_mask(&self) -> u8 {
        ((self.flags & TRAIL_SWARM_QUORUM_MASK) >> TRAIL_SWARM_QUORUM_SHIFT) as u8
    }

    /// Updates the 8-bit Micro-Swarm Quorum bitmask (`bits 8..=15`) and resolution/escalation flags.
    #[inline(always)]
    pub fn set_swarm_quorum_state(&mut self, quorum_mask: u8, hot_path_resolved: bool) {
        let cleared = self.flags
            & !(TRAIL_SWARM_QUORUM_MASK | TRAIL_FLAG_SWARM_RESOLVED | TRAIL_FLAG_SWARM_ESCALATED);
        let status_bit = branchless_select_u32(
            hot_path_resolved,
            TRAIL_FLAG_SWARM_RESOLVED,
            TRAIL_FLAG_SWARM_ESCALATED,
        );
        self.flags = cleared | (u32::from(quorum_mask) << TRAIL_SWARM_QUORUM_SHIFT) | status_bit;
    }

    /// Packs `(next_addr, attractor_slot_id, action_mode_flags, phase_q7)` and advances the trail.
    #[inline(always)]
    pub fn record_action_transition(
        &mut self,
        next_addr: u64,
        attractor_slot_id: u8,
        action_mode_flags: u8,
        phase_q7: PhaseQ7,
        edge_cost: i32,
        density_boost: f32,
    ) {
        let packed =
            TaggedPtr::pack_action(next_addr, attractor_slot_id, action_mode_flags, phase_q7);
        self.record_pointer_transition(packed.raw(), edge_cost, density_boost);
        self.cluster_id = u32::from(attractor_slot_id & ATTRACTOR_SLOT_ID_MASK);
        let phase_steer_on = (action_mode_flags & ACTION_MODE_PHASE_STEER) != 0;
        let skip_on = (action_mode_flags & ACTION_MODE_LAYER_SKIP) != 0;
        self.flags |= branchless_select_u32(phase_steer_on, TRAIL_FLAG_PHASE_STEER_ACTIVE, 0)
            | branchless_select_u32(skip_on, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, 0);
    }

    /// Evolves the trail's `current_ptr` Action Mode flags and Target Working Lane ID bottom-up,
    /// preserving 32-bit `target_anchor_id` in `cluster_id` when `target_anchor_id != 0` and
    /// `!repulsion_triggered`.
    #[inline(always)]
    pub fn evolve_trail_mode_stigmergically(
        &mut self,
        entropy_h_tilde: f32,
        repulsion_triggered: bool,
        dominant_attractor_slot: u8,
        low_entropy_skip_thresh: f32,
        high_density_rank1_thresh: f32,
    ) -> TaggedPtr {
        let evolved = TaggedPtr(self.current_ptr).evolve_mode_stigmergically(
            self.stigmergic_density,
            entropy_h_tilde,
            repulsion_triggered,
            dominant_attractor_slot,
            low_entropy_skip_thresh,
            high_density_rank1_thresh,
        );
        self.current_ptr = evolved.raw();
        let preserve_unified_anchor = (self.target_anchor_id != 0) & (!repulsion_triggered);
        self.cluster_id = branchless_select_u32(
            preserve_unified_anchor,
            self.target_anchor_id,
            u32::from(evolved.target_attractor_slot()),
        );

        let phase_steer_active = (evolved.action_mode_flags() & ACTION_MODE_PHASE_STEER) != 0;
        let skip_active = (evolved.action_mode_flags() & ACTION_MODE_LAYER_SKIP) != 0;
        self.flags = (self.flags
            & !(TRAIL_FLAG_PHASE_STEER_ACTIVE | TRAIL_FLAG_INERTIAL_SKIP_ACTIVE))
            | branchless_select_u32(phase_steer_active, TRAIL_FLAG_PHASE_STEER_ACTIVE, 0)
            | branchless_select_u32(skip_active, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, 0)
            | branchless_select_u32(repulsion_triggered, TRAIL_FLAG_REPULSION_ACTIVE, 0);
        evolved
    }

    /// Evaluates circular phase coherence $\mathcal{C}_{\mathrm{phase}} \in [-1.0, 1.0]$ with `slot_phases`.
    #[inline(always)]
    pub fn phase_coherence_with(&self, slot_phases: &[PhaseQ7; MAX_ATTENTION_HEADS]) -> f32 {
        coherence_from_word_arrays_64(trail_phase_words_ref(self), &load_q7_words_64(slot_phases))
    }

    /// Evaluates the trail's 64-head circular resultant statistics $(\bar{R}_c, \bar{\theta}_{\mathrm{Q7}}, \kappa_c)$
    /// alongside its Phase-Proximity Gravity mass $M_{\mathrm{grav}} = \rho_{\mathrm{stigmergy}} / (1 - \bar{R}_c + \epsilon)$
    /// in $\mathcal{O}(1)$ division-free (`0 fdiv`) mode.
    #[inline(always)]
    pub fn evaluate_trail_circular_stats_and_gravity(&self) -> (f32, PhaseQ7, f32, f32) {
        let (r_bar, mean_q7, kappa) = circular_resultant_from_words_64(trail_phase_words_ref(self));
        let gravity = branchless_phase_proximity_gravity_f32(
            self.stigmergic_density,
            r_bar,
            DEFAULT_PHASE_GRAVITY_EPS,
        );
        (r_bar, mean_q7, kappa, gravity)
    }
}

impl PointerTrail {
    /// Branchlessly evaluates whether a pointer trail qualifies for bottom-up macro
    /// crystallization (`stigmergic_density >= density_thresh && r_bar_t >= coherence_thresh && !repulsion_active`)
    /// and stamps or clears [`TRAIL_FLAG_CRYSTALLIZED_CANDIDATE`] on `self.flags` in $\mathcal{O}(1)$.
    #[inline(always)]
    pub fn evaluate_crystallization_candidacy(
        &mut self,
        density_crystallization_thresh: f32,
        coherence_crystallization_thresh: f32,
        r_bar_t: f32,
    ) -> bool {
        let density_ok = self.should_crystallize_branchless(density_crystallization_thresh)
            & branchless_is_finite_f32(density_crystallization_thresh);
        let coherence_ok = (r_bar_t >= coherence_crystallization_thresh)
            & (coherence_crystallization_thresh >= 0.0)
            & branchless_is_finite_f32(coherence_crystallization_thresh)
            & branchless_is_finite_f32(r_bar_t);
        let not_repulsed = (self.flags & TRAIL_FLAG_REPULSION_ACTIVE) == 0;
        let is_candidate = density_ok & coherence_ok & not_repulsed;
        self.flags = (self.flags & !TRAIL_FLAG_CRYSTALLIZED_CANDIDATE)
            | branchless_select_u32(is_candidate, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, 0);
        is_candidate
    }
}
