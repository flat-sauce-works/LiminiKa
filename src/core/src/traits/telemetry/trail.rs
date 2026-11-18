// name: src/core/src/traits/telemetry/trail.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pointer-driven action-mode routing, Bayesian trail selection, and Limina bridging.

use super::*;

// ===================================================================
// Internal Zero-Cost SWAR 64-Byte Word Helpers
// ===================================================================

#[inline(always)]
#[must_use]
pub(super) const fn precompute_bayesian_trail_scales(
    head_dim: u32,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    contradiction_threshold: f32,
) -> (f32, f32, f32, f32) {
    let safe_d = branchless_select_u32(head_dim > 0, head_dim, 128);
    let inv_sqrt_d = if safe_d <= 256 {
        branchless_popcount_rsqrt_u32(safe_d)
    } else {
        branchless_rsqrt_f32(safe_d as f32)
    };
    let phase_scale =
        branchless_flush_subnormal_f32(branchless_relu_f32(phase_interference_gain) * inv_sqrt_d);
    let clean_eta = branchless_flush_subnormal_f32(branchless_relu_f32(stigmergic_prior_gain));
    let clean_link = branchless_flush_subnormal_f32(branchless_relu_f32(pointer_link_bonus));
    let safe_thresh =
        branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
    (phase_scale, clean_eta, clean_link, safe_thresh)
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
#[must_use]
pub(super) fn eval_candidate_trail_words_with_macro_fast(
    self_w: &[u64; 8],
    self_is_uniform: bool,
    self_is_byte_uniform: bool,
    self_b0: u8,
    cur_linked_id: u32,
    cand_id: u32,
    cand: &gcso_pointer_trail_t,
    phase_scale: f32,
    clean_eta: f32,
    clean_link: f32,
    safe_thresh: f32,
    macro_anchor_id: u32,
    win_w_opt: Option<&[u64; 8]>,
    win_meta_evaluated: &mut bool,
    win_is_uniform: &mut bool,
    win_eq_self: &mut bool,
    clean_cluster_bonus: f32,
) -> (f32, f32, bool, bool) {
    let cand_w = trail_phase_words_ref(cand);
    let cand_is_uniform = if self_is_uniform {
        is_word_array_uniform_64(cand_w)
    } else {
        false
    };
    let coh = if cand_is_uniform {
        let cw0 = cand_w[0];
        if cw0 == self_w[0] {
            1.0
        } else if self_is_byte_uniform
            && (cw0 == cw0.rotate_right(8))
        {
            Q7_FULL_COS_REF[self_b0.wrapping_sub(cw0 as u8) as usize]
        } else {
            coherence_from_word_arrays_with_both_uniform_flags_64(self_w, true, cand_w, true)
        }
    } else {
        coherence_from_word_arrays_with_both_uniform_flags_64(
            self_w,
            self_is_uniform,
            cand_w,
            false,
        )
    };
    let cand_not_repulsed = (cand.flags & TRAIL_FLAG_REPULSION_ACTIVE) == 0;
    let is_coherent = (coh >= -safe_thresh) & cand_not_repulsed;
    let eff_prior_link = if is_coherent {
        let is_linked = (cur_linked_id != 0) & (cur_linked_id == cand_id);
        let link_term = branchless_select_f32(is_linked, clean_link, 0.0);
        let prior_term = if clean_eta > 0.0 {
            clean_eta * fast_ln_1p_bounded_f32(branchless_relu_f32(cand.stigmergic_density))
        } else {
            0.0
        };
        let macro_affinity = if (macro_anchor_id != 0)
            & ((cand.cluster_id == macro_anchor_id) | (cand.target_anchor_id == macro_anchor_id))
        {
            1.0
        } else if (macro_anchor_id != 0)
            & (clean_cluster_bonus > 0.0)
            & (cand.cluster_id == 0)
            & (cand.target_anchor_id == 0)
        {
            match win_w_opt {
                Some(win_w) => {
                    if !*win_meta_evaluated {
                        let eq = (((self_w[0] ^ win_w[0]) | (self_w[1] ^ win_w[1]))
                            | ((self_w[2] ^ win_w[2]) | (self_w[3] ^ win_w[3]))
                            | ((self_w[4] ^ win_w[4]) | (self_w[5] ^ win_w[5]))
                            | ((self_w[6] ^ win_w[6]) | (self_w[7] ^ win_w[7])))
                            == 0;
                        *win_eq_self = eq;
                        *win_is_uniform = if eq {
                            self_is_uniform
                        } else {
                            is_word_array_uniform_64(win_w)
                        };
                        *win_meta_evaluated = true;
                    }
                    if *win_eq_self {
                        branchless_relu_f32(coh)
                    } else {
                        let cand_uni_for_win = if self_is_uniform {
                            cand_is_uniform
                        } else if *win_is_uniform {
                            is_word_array_uniform_64(cand_w)
                        } else {
                            false
                        };
                        branchless_relu_f32(
                            coherence_from_word_arrays_with_both_uniform_flags_64(
                                win_w,
                                *win_is_uniform,
                                cand_w,
                                cand_uni_for_win,
                            ),
                        )
                    }
                }
                None => 0.0,
            }
        } else {
            0.0
        };
        prior_term + link_term + (clean_cluster_bonus * macro_affinity)
    } else {
        0.0
    };
    let bayes_delta = branchless_flush_subnormal_f32((phase_scale * coh) + eff_prior_link);
    (bayes_delta, coh, is_coherent, cand_is_uniform)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[must_use]
unsafe fn eval_candidate_trail_avx2_with_macro_fast(
    self_w: &[u64; 8],
    self_is_uniform: bool,
    self_is_byte_uniform: bool,
    self_b0: u8,
    self_v0: core::arch::x86_64::__m256i,
    self_v1: core::arch::x86_64::__m256i,
    cur_linked_id: u32,
    cand_id: u32,
    cand: &gcso_pointer_trail_t,
    phase_scale: f32,
    clean_eta: f32,
    clean_link: f32,
    safe_thresh: f32,
    macro_anchor_id: u32,
    win_w_opt: Option<&[u64; 8]>,
    win_meta_evaluated: &mut bool,
    win_is_uniform: &mut bool,
    win_eq_self: &mut bool,
    win_v0: &mut core::arch::x86_64::__m256i,
    win_v1: &mut core::arch::x86_64::__m256i,
    clean_cluster_bonus: f32,
) -> (
    f32,
    f32,
    bool,
    bool,
    Option<(core::arch::x86_64::__m256i, core::arch::x86_64::__m256i)>,
) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`. Keeps `(self_v0, self_v1)` and
    // `(win_v0, win_v1)` hoisted in YMM registers across the candidate loop and loads each
    // candidate's offset-56 phase profile via 8-byte aligned GPR loads (`trail_phase_words_ref`)
    // to eliminate 64B cache-line splits and 4-to-1 GPR-store-to-YMM-load STLF stalls.
    unsafe {
        use core::arch::x86_64::_mm256_sub_epi8;
        let cw = trail_phase_words_ref(cand);
        let c_uni = if self_is_uniform {
            is_word_array_uniform_64(cw)
        } else {
            false
        };
        let mut cv_opt: Option<(
            core::arch::x86_64::__m256i,
            core::arch::x86_64::__m256i,
        )> = None;
        let coh = if c_uni {
            let cw0 = cw[0];
            if cw0 == self_w[0] {
                1.0
            } else if self_is_byte_uniform
                && (cw0 == cw0.rotate_right(8))
            {
                Q7_FULL_COS_REF[self_b0.wrapping_sub(cw0 as u8) as usize]
            } else {
                coherence_from_word_arrays_with_both_uniform_flags_64(self_w, true, cw, true)
            }
        } else {
            let (cv0, cv1) = crate::traits::branchless::avx2_words64_to_vec64(cw);
            cv_opt = Some((cv0, cv1));
            crate::traits::branchless::avx2_coherence_from_diff_vec64(
                _mm256_sub_epi8(self_v0, cv0),
                _mm256_sub_epi8(self_v1, cv1),
            )
        };
        let cand_not_repulsed = (cand.flags & TRAIL_FLAG_REPULSION_ACTIVE) == 0;
        let is_coherent = (coh >= -safe_thresh) & cand_not_repulsed;
        let eff_prior_link = if is_coherent {
            let is_linked = (cur_linked_id != 0) & (cur_linked_id == cand_id);
            let link_term = branchless_select_f32(is_linked, clean_link, 0.0);
            let prior_term = if clean_eta > 0.0 {
                clean_eta * fast_ln_1p_bounded_f32(branchless_relu_f32(cand.stigmergic_density))
            } else {
                0.0
            };
            let macro_affinity = if (macro_anchor_id != 0)
                & ((cand.cluster_id == macro_anchor_id)
                    | (cand.target_anchor_id == macro_anchor_id))
            {
                1.0
            } else if (macro_anchor_id != 0)
                & (clean_cluster_bonus > 0.0)
                & (cand.cluster_id == 0)
                & (cand.target_anchor_id == 0)
            {
                match win_w_opt {
                    Some(win_w) => {
                        if !*win_meta_evaluated {
                            let eq = (((self_w[0] ^ win_w[0]) | (self_w[1] ^ win_w[1]))
                                | ((self_w[2] ^ win_w[2]) | (self_w[3] ^ win_w[3]))
                                | ((self_w[4] ^ win_w[4]) | (self_w[5] ^ win_w[5]))
                                | ((self_w[6] ^ win_w[6]) | (self_w[7] ^ win_w[7])))
                                == 0;
                            *win_eq_self = eq;
                            if eq {
                                *win_is_uniform = self_is_uniform;
                            } else {
                                let w_uni = is_word_array_uniform_64(win_w);
                                *win_is_uniform = w_uni;
                                (*win_v0, *win_v1) =
                                    crate::traits::branchless::avx2_words64_to_vec64_with_uniform(
                                        win_w, w_uni,
                                    );
                            }
                            *win_meta_evaluated = true;
                        }
                        if *win_eq_self {
                            branchless_relu_f32(coh)
                        } else if *win_is_uniform & c_uni {
                            branchless_relu_f32(
                                coherence_from_word_arrays_with_both_uniform_flags_64(
                                    win_w, true, cw, true,
                                ),
                            )
                        } else {
                            let (cv0, cv1) = match cv_opt {
                                Some(v) => v,
                                None => {
                                    let v = crate::traits::branchless::avx2_words64_to_vec64_with_uniform(
                                        cw, c_uni,
                                    );
                                    cv_opt = Some(v);
                                    v
                                }
                            };
                            branchless_relu_f32(crate::traits::branchless::avx2_coherence_from_diff_vec64(
                                _mm256_sub_epi8(*win_v0, cv0),
                                _mm256_sub_epi8(*win_v1, cv1),
                            ))
                        }
                    }
                    None => 0.0,
                }
            } else {
                0.0
            };
            prior_term + link_term + (clean_cluster_bonus * macro_affinity)
        } else {
            0.0
        };
        let bayes_delta = branchless_flush_subnormal_f32((phase_scale * coh) + eff_prior_link);
        (bayes_delta, coh, is_coherent, c_uni, cv_opt)
    }
}

// ===================================================================
// 3. Pointer-Driven Action Mode Orchestration (`StigmergicActionPtrOps`)
// ===================================================================

/// Branchless helper trait extending [`gcso_tagged_ptr_t`] into a self-contained
/// orchestrator micro-instruction word (48-bit Address + 5-bit Action Mode + 3-bit Working Lane ID + 8-bit Q7 Phase).
pub trait StigmergicActionPtrOps: Copy {
    /// Packs a 48-bit address, 3-bit target working lane (`0..=7`), 5-bit action mode mask, and Q7 phase.
    #[must_use]
    fn pack_action(
        raw_addr: u64,
        attractor_slot_id: u8,
        action_mode_flags: u8,
        phase_q7: gcso_q7_t,
    ) -> Self;
    /// Extracts the 5-bit Action Mode execution bitmask (`ACTION_MODE_*`).
    #[must_use]
    fn action_mode_flags(self) -> u8;
    /// Extracts the 3-bit Target Attractor Working Lane ID (`0..=7`).
    #[must_use]
    fn target_attractor_slot(self) -> u8;
    /// Returns `true` if all specified `mode_mask` bits are enabled in the pointer's Action Mode.
    #[must_use]
    fn has_action_mode(self, mode_mask: u8) -> bool;
    /// Returns a new tagged pointer with updated Target Attractor Working Lane ID and Action Mode flags.
    #[must_use]
    fn with_action_and_slot(self, attractor_slot_id: u8, action_mode_flags: u8) -> Self;
    /// Branchlessly mutates the pointer's Action Mode flags, Target Working Lane ID, and embedded
    /// Q7 phase based on bottom-up stigmergic environmental traces.
    #[must_use]
    fn evolve_mode_stigmergically(
        self,
        trail_density: f32,
        entropy_h_tilde: f32,
        repulsion_triggered: bool,
        dominant_attractor_slot: u8,
        low_entropy_skip_thresh: f32,
        high_density_rank1_thresh: f32,
    ) -> Self;
}

impl StigmergicActionPtrOps for gcso_tagged_ptr_t {
    #[inline(always)]
    fn pack_action(
        raw_addr: u64,
        attractor_slot_id: u8,
        action_mode_flags: u8,
        phase_q7: gcso_q7_t,
    ) -> Self {
        Self::pack(
            raw_addr,
            pack_action_slot_tag(attractor_slot_id, action_mode_flags),
            phase_q7,
        )
    }

    #[inline(always)]
    fn action_mode_flags(self) -> u8 {
        unpack_action_mode_from_tag(self.unpack_tag())
    }

    #[inline(always)]
    fn target_attractor_slot(self) -> u8 {
        unpack_attractor_slot_from_tag(self.unpack_tag())
    }

    #[inline(always)]
    fn has_action_mode(self, mode_mask: u8) -> bool {
        (self.action_mode_flags() & mode_mask) == mode_mask
    }

    #[inline(always)]
    fn with_action_and_slot(self, attractor_slot_id: u8, action_mode_flags: u8) -> Self {
        self.with_tag(pack_action_slot_tag(attractor_slot_id, action_mode_flags))
    }

    #[inline(always)]
    fn evolve_mode_stigmergically(
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
        let m_dpsr = branchless_select_u8(active_signal, ACTION_MODE_DPSR_STEER, 0);
        let can_skip =
            active_signal & (clean_h <= low_entropy_skip_thresh) & (!repulsion_triggered);
        let m_skip = branchless_select_u8(can_skip, ACTION_MODE_LAYER_SKIP, 0);
        let m_rep = branchless_select_u8(repulsion_triggered, ACTION_MODE_PHASE_REPULSE, 0);
        let need_pspm = (clean_h > low_entropy_skip_thresh) | repulsion_triggered;
        let m_pspm = branchless_select_u8(need_pspm, ACTION_MODE_SUBHEAD_ROUTE, 0);
        let need_srl = (clean_rho >= high_density_rank1_thresh) & (high_density_rank1_thresh > 0.0);
        let m_srl = branchless_select_u8(need_srl, ACTION_MODE_RANK1_INJECT, 0);

        let next_mode = m_dpsr | m_skip | m_rep | m_pspm | m_srl;
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

// ===================================================================
// 5. Shared Pheromone Density Laws (`PheromoneDensityOps`) & Pointer Trail
// ===================================================================

/// Shared branchless pheromone decay and coherence-weighted reinforcement law implemented by both
/// hot pointer trails ([`gcso_pointer_trail_t`]) and RAM experience cache slots (`RamPhaseKvEntry`).
pub trait PheromoneDensityOps {
    /// Returns the raw non-negative pheromone density scalar for this record.
    fn pheromone_density(&self) -> f32;

    /// Updates the stored pheromone density scalar for this record.
    fn set_pheromone_density(&mut self, density: f32);

    /// Applies subnormal-safe temporal evaporative decay ($\rho \gets \min(\rho \cdot \lambda_{\mathrm{retain}}, \rho_{\max})$).
    #[inline(always)]
    fn evaporate_pheromone(&mut self, retain_factor: f32) -> f32 {
        let clean_decay =
            branchless_select_f32(retain_factor > 1.0, 1.0, branchless_relu_f32(retain_factor));
        let decayed = branchless_relu_f32(self.pheromone_density()) * clean_decay;
        let next = branchless_min_f32(branchless_flush_subnormal_f32(decayed), MAX_STIGMERGIC_DENSITY);
        self.set_pheromone_density(next);
        next
    }

    /// Executes a single closed-loop pheromone update step: temporal evaporation, followed by
    /// coherence-weighted reinforcement (or immediate halving on contradiction repulsion).
    #[inline(always)]
    fn step_pheromone(
        &mut self,
        retain_factor: f32,
        reinforcement_gain: f32,
        phase_coherence: f32,
        contradiction_repulsed: bool,
    ) -> f32 {
        let cur_rho = self.evaporate_pheromone(retain_factor);
        let pos_coh = branchless_clamp_f32(branchless_relu_f32(phase_coherence), 0.0, 1.0);
        let deposit = branchless_relu_f32(reinforcement_gain) * (1.0 + pos_coh);
        let reinforced_rho = branchless_min_f32(cur_rho + deposit, MAX_STIGMERGIC_DENSITY);
        let damped_rho = branchless_flush_subnormal_f32(cur_rho * 0.5);
        let next_rho = branchless_select_f32(contradiction_repulsed, damped_rho, reinforced_rho);
        self.set_pheromone_density(next_rho);
        next_rho
    }
}

impl PheromoneDensityOps for gcso_pointer_trail_t {
    #[inline(always)]
    fn pheromone_density(&self) -> f32 {
        self.stigmergic_density
    }

    #[inline(always)]
    fn set_pheromone_density(&mut self, density: f32) {
        self.stigmergic_density = density;
    }
}

/// Branchless $\mathcal{O}(1)$ stigmergic trail, pointer-mode orchestration, inter-memory Limina
/// bridging, Bayesian pointer-trail transition scoring & rewiring, single-manifold unified-field
/// anchor resonance (preserving 32-bit `target_anchor_id`), and 8-lane working neighborhood
/// synergy operations on 128-byte aligned [`gcso_pointer_trail_t`].
pub trait StigmergicTrailOps {
    /// Records an $\mathcal{O}(1)$ tagged pointer transition and reinforces stigmergic trail density.
    fn record_pointer_transition(&mut self, next_ptr: u64, edge_cost: i32, density_boost: f32);
    /// Accumulates a 64-head Q7 phase delta vector into the trail's phase state using branchless saturation.
    fn accumulate_phase_vector(&mut self, delta_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]);
    /// Accumulates a uniform Q7 phase delta across all 64 heads in-place using 8-lane SWAR saturation.
    fn accumulate_uniform_phase(&mut self, delta_q7: gcso_q7_t);
    /// Smoothly blends the trail's 64-head phase trajectory toward `target_phases` along $S^1$.
    fn geodesic_blend_phase_with(
        &mut self,
        target_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        t_u7: u8,
    );
    /// Stamps a phase-conjugate repulsion marker ($-\boldsymbol{\Delta\theta}$) onto the pointer trail.
    fn stamp_phase_conjugate_repulsion(&mut self);
    /// Applies subnormal-safe temporal evaporative decay (`decay_factor` $\in [0, 1]$) to trail density.
    fn apply_trail_evaporation(&mut self, decay_factor: f32);
    /// Evaluates whether the trail's stigmergic density exceeds `crystal_threshold`.
    fn should_crystallize_branchless(&self, crystal_threshold: f32) -> bool;
    /// Returns `current_ptr` wrapped as a strongly-typed [`gcso_tagged_ptr_t`].
    fn current_tagged_ptr(&self) -> gcso_tagged_ptr_t;
    /// Returns `prev_ptr` wrapped as a strongly-typed [`gcso_tagged_ptr_t`].
    fn prev_tagged_ptr(&self) -> gcso_tagged_ptr_t;
    /// Extracts the active 5-bit Action Mode bitmask (`ACTION_MODE_*`) embedded in `current_ptr`.
    fn active_action_mode(&self) -> u8;
    /// Extracts the 3-bit Target Attractor Working Lane ID (`0..=7`) embedded in `current_ptr`.
    fn target_attractor_slot(&self) -> u8;
    /// Extracts the 8-bit Micro-Swarm Quorum bitmask (`bits 8..=15`) embedded in `self.flags`.
    fn swarm_quorum_mask(&self) -> u8;
    /// Updates the 8-bit Micro-Swarm Quorum bitmask (`bits 8..=15`) and resolution/escalation flags.
    fn set_swarm_quorum_state(&mut self, quorum_mask: u8, hot_path_resolved: bool);
    /// Packs `(next_addr, attractor_slot_id, action_mode_flags, phase_q7)` and advances the trail.
    fn record_action_transition(
        &mut self,
        next_addr: u64,
        attractor_slot_id: u8,
        action_mode_flags: u8,
        phase_q7: gcso_q7_t,
        edge_cost: i32,
        density_boost: f32,
    );
    /// Evolves the trail's `current_ptr` Action Mode flags and Target Working Lane ID bottom-up,
    /// preserving 32-bit `target_anchor_id` in `cluster_id` when `target_anchor_id != 0` and
    /// `!repulsion_triggered`.
    fn evolve_trail_mode_stigmergically(
        &mut self,
        entropy_h_tilde: f32,
        repulsion_triggered: bool,
        dominant_attractor_slot: u8,
        low_entropy_skip_thresh: f32,
        high_density_rank1_thresh: f32,
    ) -> gcso_tagged_ptr_t;
    /// Evaluates circular phase coherence $\mathcal{C}_{\mathrm{phase}} \in [-1.0, 1.0]$ with `slot_phases`.
    fn phase_coherence_with(&self, slot_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> f32;
    /// Evaluates the trail's 64-head circular resultant statistics $(\bar{R}_c, \bar{\theta}_{\mathrm{Q7}}, \kappa_c)$
    /// alongside its Phase-Proximity Gravity mass $M_{\mathrm{grav}} = \rho_{\mathrm{stigmergy}} / (1 - \bar{R}_c + \epsilon)$
    /// in $\mathcal{O}(1)$ division-free (`0 fdiv`) mode.
    fn evaluate_trail_circular_stats_and_gravity(&self) -> (f32, gcso_q7_t, f32, f32);
    /// Applies **Zero-Hyperparameter Phase-Proximity Resonance** from a single anchor on the
    /// unified phase manifold (`anchor_phases`, `anchor_base_energy`) in strictly constant time (`0` branches).
    fn apply_unified_field_anchor_resonance(
        &mut self,
        anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        anchor_base_energy: f32,
    ) -> (f32, f32);
    /// Applies **Order-Invariant (Commutative) Unified-Field Multi-Anchor Resonance** across an
    /// arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) and optional repulsive
    /// phase inhibitor (`repulsion_phases_and_strength`) on a **single unified phase manifold**
    /// using register-resident 8-word SWAR accumulators (`0` branches, `0 fdiv`).
    /// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) and optional
    /// repulsive phase inhibitor (`repulsion_phases_and_strength`) starting from pre-loaded 8-word
    /// SWAR registers `init_w` (`0` entry reload) and returning `(final_pull_force, mean_coherence, active_anchor_count, post_words)`.
    fn apply_unified_field_anchor_slice_resonance_from_words(
        &mut self,
        init_w: [u64; NUM_STIGMERGIC_GRANULES],
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
    ) -> (f32, f32, u32, [u64; NUM_STIGMERGIC_GRANULES]);

    /// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) on the single
    /// unified phase manifold and returns `(final_pull_force, mean_coherence, active_anchor_count, post_words)`
    /// directly in 8 `u64` SWAR registers (`0` exit reload).
    fn apply_unified_field_anchor_slice_resonance_to_words(
        &mut self,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
    ) -> (f32, f32, u32, [u64; NUM_STIGMERGIC_GRANULES]);

    /// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) and optional
    /// repulsive phase inhibitor (`repulsion_phases_and_strength`) on a **single unified phase manifold**
    /// using register-resident 8-word SWAR accumulators (`0` branches, `0 fdiv`).
    fn apply_unified_field_anchor_slice_resonance(
        &mut self,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
    ) -> (f32, f32, u32);
    /// Bridges the "unwritten space between the lines" (**Limina / Inter-Memory Hallucination**)
    /// from `prev_memory_trail` ($M_{t-1}$) to `self` ($M_t$) in a **single fused 8-word SWAR pass**
    /// (`0` branches, zero temporary array stack spills).
    fn bridge_liminal_hallucinated_trail(
        &mut self,
        prev_memory_trail: &gcso_pointer_trail_t,
        interpolation_u7: u8,
        contradiction_threshold: f32,
    ) -> (f32, f32, bool);
    /// Links `self` ($M_{t-1}$) to `next_trail` ($M_t$, identified by `next_trail_id`) by setting
    /// `self.linked_trail_id = next_trail_id` and bridging their inter-memory **Limina** gap.
    fn link_with_next_hallucinated_trail(
        &mut self,
        next_trail_id: u32,
        next_trail: &mut gcso_pointer_trail_t,
        interpolation_u7: u8,
        contradiction_threshold: f32,
    ) -> (f32, f32, bool);
    /// Evaluates the **Bayes-Attention Stigmergic Transition Score** from `self` ($M_t$) to a
    /// candidate memory trail `candidate_trail` ($M_j$, identified by `candidate_trail_id`) in
    /// strictly constant time (`0` branches, `0 fdiv`).
    fn evaluate_bayesian_transition_score_with(
        &self,
        candidate_trail_id: u32,
        candidate_trail: &gcso_pointer_trail_t,
        head_dim: u32,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        contradiction_threshold: f32,
    ) -> (f32, f32, bool);
    /// Synchronizes `self.flags` (`TRAIL_FLAG_DPSR_ACTIVE` and `TRAIL_FLAG_INERTIAL_SKIP_ACTIVE`)
    /// with the 5-bit `ACTION_MODE_*` flags embedded in `self.current_ptr` in branchless constant time.
    fn sync_flags_with_current_ptr(&mut self);
    /// Evaluates non-linear cross-slot synergy and liminal beat interference between two working lanes.
    fn apply_cross_slot_synergy(
        &mut self,
        slot_a: u32,
        phases_a: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        weight_a: f32,
        slot_b: u32,
        phases_b: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        weight_b: f32,
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
    ) -> (f32, f32);
    /// Applies branchless mutual inhibition gating to the trail's accumulated phase vector and pull force.
    fn apply_mutual_slot_inhibition(
        &mut self,
        inhibitor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        inhibition_strength: f32,
    ) -> f32;
    /// Register-resident evaluator for **8-Lane Coupled Neighborhood Synergy & Mutual Inhibition**
    /// starting from pre-loaded 8-word SWAR registers `acc_w` and returning
    /// `(damped_pull_force, mean_coherence, active_neighbor_count, post_words)`.
    fn apply_neighborhood_synergy_from_bank_from_words(
        &mut self,
        acc_w: [u64; NUM_STIGMERGIC_GRANULES],
        slot_phase_bank: &[[gcso_q7_t; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
        inhibition_strength: f32,
    ) -> (f32, f32, u32, [u64; NUM_STIGMERGIC_GRANULES]);

    /// Evaluates **8-Lane Coupled Neighborhood Synergy & Mutual Inhibition** and returns both
    /// `(damped_pull_force, mean_coherence, active_neighbor_count, post_words)` in registers (`0` exit reload).
    fn apply_neighborhood_synergy_from_bank_to_words(
        &mut self,
        slot_phase_bank: &[[gcso_q7_t; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
        inhibition_strength: f32,
    ) -> (f32, f32, u32, [u64; NUM_STIGMERGIC_GRANULES]);

    /// Unified single-source-of-truth evaluator for **8-Lane Coupled Neighborhood Synergy & Mutual Inhibition**
    /// across `slot_phase_bank` (`[[gcso_q7_t; 64]; 8]`) and `slot_weights` (`[f32; 8]`).
    fn apply_neighborhood_synergy_from_bank(
        &mut self,
        slot_phase_bank: &[[gcso_q7_t; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
        inhibition_strength: f32,
    ) -> (f32, f32, u32);
}

/// Register-resident Coarse-to-Fine Bayesian candidate trail selector, rewirer, Limina geodesic
/// bridger, and Attention score modulator accepting `self_w` directly in 8 `u64` SWAR registers
/// and returning the post-bridged 8 `u64` SWAR words (`0` redundant stack/memory reloads).
#[inline]
pub(crate) fn select_rewire_bridge_and_modulate_with_preloaded_words_64(
    trail: &mut gcso_pointer_trail_t,
    mut self_w: [u64; NUM_STIGMERGIC_GRANULES],
    candidate_trails: &[gcso_pointer_trail_t],
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
) -> (u32, f32, f32, u32, bool, [u64; NUM_STIGMERGIC_GRANULES], f32) {
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
    let self_is_byte_uniform =
        self_is_uniform & (self_w[0] == self_w[0].rotate_right(8));
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    let (self_v0, self_v1) = unsafe {
        crate::traits::branchless::avx2_words64_to_vec64_with_uniform(&self_w, self_is_uniform)
    };
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
    let mut best_vec_opt: Option<(
        core::arch::x86_64::__m256i,
        core::arch::x86_64::__m256i,
    )> = None;
    let mut found_valid = false;
    let mut coh_sum = 0.0_f32;
    let mut coherent_count = 0u32;

    let mut idx = 0usize;
    while idx < cand_len {
        let cand = &candidate_trails[idx];
        let cand_id = idx as u32;
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
                scores[idx] =
                    branchless_flush_subnormal_f32(scores[idx] + bayes_delta - penalty);
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
                                    None => crate::traits::branchless::avx2_words64_to_vec64_with_uniform(
                                        best_w,
                                        best_is_uniform,
                                    ),
                                };
                                let (rv0, rv1) = crate::traits::branchless::avx2_slerp_vec64(
                                    self_v0, self_v1, bv0, bv1, eff_s,
                                );
                                self_w = crate::traits::branchless::avx2_vec64_to_words64(rv0, rv1);
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
                scores[best_idx] =
                    branchless_flush_subnormal_f32(scores[best_idx] + clean_link);
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

impl gcso_pointer_trail_t {
    /// Copies `src` and pre-loaded `phase_words` into `self` using width-matched scalar writes and
    /// 8-byte aligned 64-bit GPR stores (`*trail_phase_words_mut(self) = *phase_words`) at byte
    /// offset `56..120`, avoiding unaligned 256-bit `vmovdqu` cache-line splits, 256-bit-to-64-bit
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
        *trail_phase_words_mut(self) = *phase_words;
        self.reserved_padding = src.reserved_padding;
    }

    /// Restores `self` from `src` using width-matched scalar writes and 64-bit GPR stores
    /// (`*trail_phase_words_mut(self) = *trail_phase_words_ref(src)`), preventing 256-bit YMM-to-GPR
    /// Store-to-Load Forwarding (STLF) pipeline stalls (`~15–20 cycles`).
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
        *trail_phase_words_mut(self) = [0u64; NUM_STIGMERGIC_GRANULES];
        self.reserved_padding = [0; 8];
    }
}

impl StigmergicTrailOps for gcso_pointer_trail_t {
    #[inline(always)]
    fn record_pointer_transition(&mut self, next_ptr: u64, edge_cost: i32, density_boost: f32) {
        self.prev_ptr = self.current_ptr;
        self.current_ptr = next_ptr;
        self.transition_cost = self.transition_cost.saturating_add(edge_cost);
        self.step_count = self.step_count.wrapping_add(1);
        let raw_rho = branchless_flush_subnormal_f32(branchless_relu_f32(self.stigmergic_density))
            + branchless_flush_subnormal_f32(branchless_relu_f32(density_boost));
        self.stigmergic_density = branchless_min_f32(raw_rho, MAX_STIGMERGIC_DENSITY);
    }

    #[inline(always)]
    fn accumulate_phase_vector(&mut self, delta_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) {
        let d = load_q7_words_64(delta_phases);
        let w = trail_phase_words_mut(self);
        swar_or_avx2_q7_add_sat_words_64(w, &d);
    }

    #[inline(always)]
    fn accumulate_uniform_phase(&mut self, delta_q7: gcso_q7_t) {
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

    #[inline(always)]
    fn geodesic_blend_phase_with(
        &mut self,
        target_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
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

    #[inline(always)]
    fn stamp_phase_conjugate_repulsion(&mut self) {
        let w = trail_phase_words_mut(self);
        w[0] = swar_q7_neg_sat_x8(w[0]);
        w[1] = swar_q7_neg_sat_x8(w[1]);
        w[2] = swar_q7_neg_sat_x8(w[2]);
        w[3] = swar_q7_neg_sat_x8(w[3]);
        w[4] = swar_q7_neg_sat_x8(w[4]);
        w[5] = swar_q7_neg_sat_x8(w[5]);
        w[6] = swar_q7_neg_sat_x8(w[6]);
        w[7] = swar_q7_neg_sat_x8(w[7]);
        self.current_ptr = gcso_tagged_ptr_t(self.current_ptr).conjugate_phase().raw();
        self.flags |= TRAIL_FLAG_REPULSION_ACTIVE;
    }

    #[inline(always)]
    fn apply_trail_evaporation(&mut self, decay_factor: f32) {
        let _ = self.evaporate_pheromone(decay_factor);
    }

    #[inline(always)]
    fn should_crystallize_branchless(&self, crystal_threshold: f32) -> bool {
        (self.stigmergic_density >= crystal_threshold) & (crystal_threshold > 0.0)
    }

    #[inline(always)]
    fn current_tagged_ptr(&self) -> gcso_tagged_ptr_t {
        gcso_tagged_ptr_t(self.current_ptr)
    }
    #[inline(always)]
    fn prev_tagged_ptr(&self) -> gcso_tagged_ptr_t {
        gcso_tagged_ptr_t(self.prev_ptr)
    }
    #[inline(always)]
    fn active_action_mode(&self) -> u8 {
        gcso_tagged_ptr_t(self.current_ptr).action_mode_flags()
    }
    #[inline(always)]
    fn target_attractor_slot(&self) -> u8 {
        gcso_tagged_ptr_t(self.current_ptr).target_attractor_slot()
    }
    #[inline(always)]
    fn swarm_quorum_mask(&self) -> u8 {
        ((self.flags & TRAIL_SWARM_QUORUM_MASK) >> TRAIL_SWARM_QUORUM_SHIFT) as u8
    }

    #[inline(always)]
    fn set_swarm_quorum_state(&mut self, quorum_mask: u8, hot_path_resolved: bool) {
        let cleared = self.flags
            & !(TRAIL_SWARM_QUORUM_MASK | TRAIL_FLAG_SWARM_RESOLVED | TRAIL_FLAG_SWARM_ESCALATED);
        let status_bit = branchless_select_u32(
            hot_path_resolved,
            TRAIL_FLAG_SWARM_RESOLVED,
            TRAIL_FLAG_SWARM_ESCALATED,
        );
        self.flags = cleared | (u32::from(quorum_mask) << TRAIL_SWARM_QUORUM_SHIFT) | status_bit;
    }

    #[inline(always)]
    fn record_action_transition(
        &mut self,
        next_addr: u64,
        attractor_slot_id: u8,
        action_mode_flags: u8,
        phase_q7: gcso_q7_t,
        edge_cost: i32,
        density_boost: f32,
    ) {
        let packed = gcso_tagged_ptr_t::pack_action(
            next_addr,
            attractor_slot_id,
            action_mode_flags,
            phase_q7,
        );
        self.record_pointer_transition(packed.raw(), edge_cost, density_boost);
        self.cluster_id = u32::from(attractor_slot_id & ATTRACTOR_SLOT_ID_MASK);
        let dpsr_on = (action_mode_flags & ACTION_MODE_DPSR_STEER) != 0;
        let skip_on = (action_mode_flags & ACTION_MODE_LAYER_SKIP) != 0;
        self.flags |= branchless_select_u32(dpsr_on, TRAIL_FLAG_DPSR_ACTIVE, 0)
            | branchless_select_u32(skip_on, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, 0);
    }

    #[inline(always)]
    fn evolve_trail_mode_stigmergically(
        &mut self,
        entropy_h_tilde: f32,
        repulsion_triggered: bool,
        dominant_attractor_slot: u8,
        low_entropy_skip_thresh: f32,
        high_density_rank1_thresh: f32,
    ) -> gcso_tagged_ptr_t {
        let evolved = gcso_tagged_ptr_t(self.current_ptr).evolve_mode_stigmergically(
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

        let dpsr_active = (evolved.action_mode_flags() & ACTION_MODE_DPSR_STEER) != 0;
        let skip_active = (evolved.action_mode_flags() & ACTION_MODE_LAYER_SKIP) != 0;
        self.flags = (self.flags & !(TRAIL_FLAG_DPSR_ACTIVE | TRAIL_FLAG_INERTIAL_SKIP_ACTIVE))
            | branchless_select_u32(dpsr_active, TRAIL_FLAG_DPSR_ACTIVE, 0)
            | branchless_select_u32(skip_active, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, 0)
            | branchless_select_u32(repulsion_triggered, TRAIL_FLAG_REPULSION_ACTIVE, 0);
        evolved
    }

    #[inline(always)]
    fn phase_coherence_with(&self, slot_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> f32 {
        coherence_from_word_arrays_64(trail_phase_words_ref(self), &load_q7_words_64(slot_phases))
    }

    #[inline(always)]
    fn evaluate_trail_circular_stats_and_gravity(&self) -> (f32, gcso_q7_t, f32, f32) {
        let (r_bar, mean_q7, kappa) =
            circular_resultant_from_words_64(trail_phase_words_ref(self));
        let gravity = branchless_phase_proximity_gravity_f32(
            self.stigmergic_density,
            r_bar,
            DEFAULT_PHASE_GRAVITY_EPS,
        );
        (r_bar, mean_q7, kappa, gravity)
    }

    #[inline]
    fn apply_unified_field_anchor_resonance(
        &mut self,
        anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
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
                    (resonant_pull * crate::traits::primitives::fast_inv_pos_normal_f32(new_pull)) * 128.0
                        + 0.5,
                    0.0,
                    128.0,
                );
                let s = (ratio_f32 as u8) as u64;
                if s >= 128 {
                    *trail_phase_words_mut(self) = b_w;
                } else if s > 0 {
                    let bias_restore =
                        (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
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

    #[inline]
    fn apply_unified_field_anchor_slice_resonance_from_words(
        &mut self,
        init_w: [u64; NUM_STIGMERGIC_GRANULES],
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
    ) -> (f32, f32, u32, [u64; NUM_STIGMERGIC_GRANULES]) {
        let (
            mut run_w,
            running_pull,
            mean_coh,
            dominant_anchor_id,
            active_anchors,
            any_resonant,
            any_liminal,
        ) = branchless_q7_unified_field_fold_words_with_flags_64(
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
        let cur_ptr = gcso_tagged_ptr_t(self.current_ptr);
        let updated_ptr = cur_ptr.with_action_and_slot(
            branchless_select_u8(has_active, lane_tag, cur_ptr.target_attractor_slot()),
            cur_ptr.action_mode_flags()
                | branchless_select_u8(has_active, ACTION_MODE_DPSR_STEER, 0),
        );
        self.current_ptr = updated_ptr.raw();
        self.cluster_id = branchless_select_u32(has_active, dominant_anchor_id, self.cluster_id);
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

    #[inline(always)]
    fn apply_unified_field_anchor_slice_resonance_to_words(
        &mut self,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
    ) -> (f32, f32, u32, [u64; NUM_STIGMERGIC_GRANULES]) {
        let init_w = *trail_phase_words_ref(self);
        self.apply_unified_field_anchor_slice_resonance_from_words(
            init_w,
            anchor_phases,
            anchor_base_energies,
            repulsion_phases_and_strength,
        )
    }

    #[inline(always)]
    fn apply_unified_field_anchor_slice_resonance(
        &mut self,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
    ) -> (f32, f32, u32) {
        let (final_pull, mean_coh, active_anchors, _words) = self
            .apply_unified_field_anchor_slice_resonance_to_words(
                anchor_phases,
                anchor_base_energies,
                repulsion_phases_and_strength,
            );
        (final_pull, mean_coh, active_anchors)
    }

    #[inline]
    fn bridge_liminal_hallucinated_trail(
        &mut self,
        prev_memory_trail: &gcso_pointer_trail_t,
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
                        crate::traits::branchless::avx2_words64_to_vec64_with_uniform(p, p_uni);
                    let (vs0, vs1) = crate::traits::branchless::avx2_words64_to_vec64(s);
                    let vd0 = _mm256_sub_epi8(vs0, vp0);
                    let vd1 = _mm256_sub_epi8(vs1, vp1);
                    let vd_or = _mm256_or_si256(vd0, vd1);
                    if _mm256_testz_si256(vd_or, vd_or) != 0 {
                        (1.0_f32, 0.0_f32, false, *s, false)
                    } else {
                        let c_val = crate::traits::branchless::avx2_coherence_from_diff_vec64(vd0, vd1);
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
                                crate::traits::branchless::avx2_vec64_to_words64(r0, r1),
                                true,
                            )
                        } else if t_s >= 128 {
                            (c_val, lp, false, *s, false)
                        } else if t_s == 0 {
                            (c_val, lp, false, *p, true)
                        } else {
                            let (r0, r1) = crate::traits::branchless::avx2_slerp_from_diff_vec64(
                                vp0, vp1, vd0, vd1, t_s,
                            );
                            (
                                c_val,
                                lp,
                                false,
                                crate::traits::branchless::avx2_vec64_to_words64(r0, r1),
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
                    let c_val = crate::traits::branchless::coherence_from_diff_words_64(
                        d0, d1, d2, d3, d4, d5, d6, d7,
                    );
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

        let cur_ptr = gcso_tagged_ptr_t(self.current_ptr);
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

    #[inline(always)]
    fn link_with_next_hallucinated_trail(
        &mut self,
        next_trail_id: u32,
        next_trail: &mut gcso_pointer_trail_t,
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

    #[inline(always)]
    fn evaluate_bayesian_transition_score_with(
        &self,
        candidate_trail_id: u32,
        candidate_trail: &gcso_pointer_trail_t,
        head_dim: u32,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        contradiction_threshold: f32,
    ) -> (f32, f32, bool) {
        let is_linked = (self.linked_trail_id != 0) & (self.linked_trail_id == candidate_trail_id);
        let (phase_scale, clean_eta, clean_link, safe_thresh) = precompute_bayesian_trail_scales(
            head_dim,
            phase_interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
            contradiction_threshold,
        );
        let coh = coherence_from_word_arrays_64(
            trail_phase_words_ref(self),
            trail_phase_words_ref(candidate_trail),
        );
        let attn_delta = branchless_flush_subnormal_f32(phase_scale * coh);
        let clean_rho = branchless_clamp_f32(
            branchless_flush_subnormal_f32(branchless_relu_f32(
                candidate_trail.stigmergic_density,
            )),
            0.0,
            MAX_STIGMERGIC_DENSITY,
        );
        let prior_delta =
            branchless_flush_subnormal_f32(clean_eta * fast_ln_1p_bounded_f32(clean_rho));
        let link_delta = branchless_select_f32(is_linked, clean_link, 0.0);
        let total_score = branchless_flush_subnormal_f32(attn_delta + prior_delta + link_delta);
        let cand_not_repulsed = (candidate_trail.flags & TRAIL_FLAG_REPULSION_ACTIVE) == 0;
        let is_coherent = (coh >= -safe_thresh) & cand_not_repulsed;
        (total_score, coh, is_coherent)
    }

    #[inline(always)]
    fn sync_flags_with_current_ptr(&mut self) {
        let mode = unpack_action_mode_from_tag(gcso_tagged_ptr_t(self.current_ptr).unpack_tag());
        let dpsr_on = (mode & ACTION_MODE_DPSR_STEER) != 0;
        let skip_on = (mode & ACTION_MODE_LAYER_SKIP) != 0;
        let cleared = self.flags & !(TRAIL_FLAG_DPSR_ACTIVE | TRAIL_FLAG_INERTIAL_SKIP_ACTIVE);
        self.flags = cleared
            | branchless_select_u32(dpsr_on, TRAIL_FLAG_DPSR_ACTIVE, 0)
            | branchless_select_u32(skip_on, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, 0);
    }

    #[inline]
    fn apply_cross_slot_synergy(
        &mut self,
        slot_a: u32,
        phases_a: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        weight_a: f32,
        slot_b: u32,
        phases_b: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        weight_b: f32,
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
    ) -> (f32, f32) {
        let coupled = is_slot_pair_coupled(coupling_matrix_8x8, slot_a, slot_b);
        let effective_gain = branchless_select_f32(coupled, synergy_gain, 0.0);

        let wa = load_q7_words_64(phases_a);
        let wb = load_q7_words_64(phases_b);
        let (fw, synergy_pull, coherence) =
            branchless_q7_synergy_fuse_words_64(&wa, weight_a, &wb, weight_b, effective_gain);
        let cur_w = trail_phase_words_mut(self);
        swar_or_avx2_q7_add_sat_words_64(cur_w, &fw);
        let new_pull = branchless_flush_subnormal_f32(
            branchless_relu_f32(self.attractor_pull_force) + synergy_pull,
        );
        self.attractor_pull_force = branchless_min_f32(new_pull, MAX_ATTRACTOR_PULL_FORCE);

        let dom_slot = branchless_select_u32(weight_a >= weight_b, slot_a, slot_b) as u8;
        let cur_ptr = gcso_tagged_ptr_t(self.current_ptr);
        self.current_ptr = cur_ptr
            .with_action_and_slot(dom_slot, cur_ptr.action_mode_flags() | ACTION_MODE_DPSR_STEER)
            .raw();
        self.cluster_id = u32::from(dom_slot & ATTRACTOR_SLOT_ID_MASK);

        let is_resonant = coupled & (coherence > 0.25) & (synergy_pull > 0.0);
        let is_liminal_beat = coupled & (coherence < -0.10) & (synergy_pull > 0.0);
        self.flags |= branchless_select_u32(is_resonant, TRAIL_FLAG_SYNERGY_RESONANCE, 0)
            | branchless_select_u32(is_liminal_beat, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, 0);

        (synergy_pull, coherence)
    }

    #[inline]
    fn apply_mutual_slot_inhibition(
        &mut self,
        inhibitor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        inhibition_strength: f32,
    ) -> f32 {
        let cur_pull = self.attractor_pull_force;
        let inh_w = load_q7_words_64(inhibitor_phases);
        let raw_damped = branchless_q7_inhibit_words_64(
            trail_phase_words_mut(self),
            cur_pull,
            &inh_w,
            inhibition_strength,
        );
        let damped_pull = branchless_min_f32(
            branchless_flush_subnormal_f32(raw_damped),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        self.attractor_pull_force = damped_pull;
        damped_pull
    }

    #[inline]
    fn apply_neighborhood_synergy_from_bank_from_words(
        &mut self,
        mut acc_w: [u64; NUM_STIGMERGIC_GRANULES],
        slot_phase_bank: &[[gcso_q7_t; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
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
        let mut neighbor_mask =
            coupled_slot_row_mask(coupling_matrix_8x8, primary_slot) & !exclude_mask;

        // Keep `acc_w` and `cur_w` in 8 64-bit registers across all coupled neighbors (`0` intermediate stack arrays)
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

    #[inline(always)]
    fn apply_neighborhood_synergy_from_bank_to_words(
        &mut self,
        slot_phase_bank: &[[gcso_q7_t; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
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

    #[inline(always)]
    fn apply_neighborhood_synergy_from_bank(
        &mut self,
        slot_phase_bank: &[[gcso_q7_t; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
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

