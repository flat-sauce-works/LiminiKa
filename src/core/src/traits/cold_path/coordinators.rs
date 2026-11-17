// name: src/core/src/traits/cold_path/coordinators.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cold-Path coordinators and full-stack blanket orchestrators over `ActionHub`.

use super::*;

// ===================================================================
// 7. Blanket-Implemented Lv.3 Macro Observers & Coordinators over `ActionHub`
// ===================================================================

/// Cold-path crystallization candidacy extension trait for [`gcso_pointer_trail_t`].
pub trait ColdPathTrailCrystalExt: StigmergicTrailOps {
    /// Branchlessly evaluates whether a pointer trail qualifies for bottom-up macro
    /// crystallization (`stigmergic_density >= density_thresh && r_bar_t >= coherence_thresh && !repulsion_active`)
    /// and stamps or clears [`TRAIL_FLAG_CRYSTALLIZED_CANDIDATE`] on `self.flags` in $\mathcal{O}(1)$.
    fn evaluate_crystallization_candidacy(
        &mut self,
        density_crystallization_thresh: f32,
        coherence_crystallization_thresh: f32,
        r_bar_t: f32,
    ) -> bool;

    /// Evaluates whether a trail stitched across inter-memory **Limina** transitions qualifies as
    /// a crystallized Global Section $\Gamma(K; \mathcal{F})$ (`!cohomology_obstructed` where
    /// $H^1(K; \mathcal{F}) = 0$, `stigmergic_density >= density_thresh`, and `r_bar_t >= coherence_thresh`),
    /// stamping or clearing [`TRAIL_FLAG_CRYSTALLIZED_CANDIDATE`].
    fn evaluate_liminal_global_section_candidacy(
        &mut self,
        density_crystallization_thresh: f32,
        coherence_crystallization_thresh: f32,
        r_bar_t: f32,
        cohomology_obstructed: bool,
    ) -> bool;
}

impl ColdPathTrailCrystalExt for gcso_pointer_trail_t {
    #[inline(always)]
    fn evaluate_crystallization_candidacy(
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

    #[inline(always)]
    fn evaluate_liminal_global_section_candidacy(
        &mut self,
        density_crystallization_thresh: f32,
        coherence_crystallization_thresh: f32,
        r_bar_t: f32,
        cohomology_obstructed: bool,
    ) -> bool {
        if cohomology_obstructed {
            self.flags &= !TRAIL_FLAG_CRYSTALLIZED_CANDIDATE;
            return false;
        }
        self.evaluate_crystallization_candidacy(
            density_crystallization_thresh,
            coherence_crystallization_thresh,
            r_bar_t,
        )
    }
}

/// Blanket-implemented coordinator uniting [`ActionHub`] with [`MultiDomainAttractorSpace`]
/// to project the 8-lane working bank, autonomously rebalance lane weights from phase proximity,
/// fuse coupled neighborhood synergy, and steer tensors.
pub trait AttractorSlotSynergyCoordinator: ActionHub + MultiDomainAttractorSpace {
    /// Projects all 8 Working Lanes for `trail_out`, applies coupled neighborhood
    /// synergy and mutual inhibition, and steers `query_tensor` and/or `key_tensor`.
    ///
    /// Returns `(damped_pull_force, mean_coherence, active_neighbor_count)`.
    ///
    /// # Errors
    /// Propagates any error from [`ActionHub::validate_hub_cascade_params`],
    /// [`MultiDomainAttractorSpace::steer_trail_by_coupled_neighborhood`], or
    /// [`ActionHub::filter_qdps_and_steer_qk_safe`].
    #[inline]
    fn coordinate_multi_domain_slot_synergy(
        &mut self,
        input_ptr: u64,
        synergy_gain: f32,
        inhibition_strength: f32,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32, u32)> {
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        self.step_pointer(input_ptr, trail_out)?;
        let (pull, coh, neighbors, syn_w) = self.steer_trail_by_coupled_neighborhood_to_words(
            synergy_gain,
            inhibition_strength,
            trail_out,
        )?;
        let _ = self.steer_qdps_words_prevalidated_core(
            &syn_w,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        );
        Ok((pull, coh, neighbors))
    }

    /// Executes a zero-hyperparameter **Unified-Field Auto-Rebalanced Synergy** cascade.
    ///
    /// Returns `(damped_pull_force, mean_coherence, active_neighbor_count, rebalanced_weights)`.
    ///
    /// # Errors
    /// Propagates any error from [`ActionHub::validate_hub_cascade_params`],
    /// [`MultiDomainAttractorSpace::rebalance_working_lane_weights_from_trail`], or
    /// [`ActionHub::filter_qdps_and_steer_qk_safe`].
    #[inline]
    fn coordinate_unified_field_auto_rebalanced_synergy(
        &mut self,
        input_ptr: u64,
        synergy_gain: f32,
        inhibition_strength: f32,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32, u32, [f32; MAX_ATTRACTOR_DOMAINS])> {
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        self.step_pointer(input_ptr, trail_out)?;
        let weights = self.rebalance_working_lane_weights_from_trail(trail_out)?;
        let (pull, coh, neighbors, syn_w) = self.steer_trail_by_coupled_neighborhood_to_words(
            synergy_gain,
            inhibition_strength,
            trail_out,
        )?;
        let _ = self.steer_qdps_words_prevalidated_core(
            &syn_w,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        );
        Ok((pull, coh, neighbors, weights))
    }
}

impl<T: ActionHub + MultiDomainAttractorSpace + ?Sized> AttractorSlotSynergyCoordinator for T {}

/// Blanket-implemented coordinator uniting [`ActionHub`], [`SheafCohomologyEvaluator`], and
/// [`EdbcBranchController`] to detect cohomological obstructions $H^1(K; \mathcal{F})$, bridge
/// inter-memory Limina gaps, and feed anti-phase repulsion back into the pointer trail.
pub trait StigmergicCohomologyCoordinator:
    ActionHub + SheafCohomologyEvaluator + EdbcBranchController
{
    /// Evaluates cohomological obstruction $H^1(K; \mathcal{F})$ on `cochain_residual_r` vs
    /// `exact_projection_delta0_phi`, records repulsive patterns into `bloom_filter` if obstructed,
    /// and evolves `trail_inout`'s pointer mode.
    ///
    /// Returns `(mean_obstruction_energy, is_obstructed, next_branch_mode, evolved_ptr)`.
    ///
    /// # Errors
    /// Propagates any error from [`SheafCohomologyEvaluator::detect_and_record_obstruction_repulsion`].
    #[inline]
    fn evaluate_cohomology_and_evolve_trail(
        &mut self,
        cochain_residual_r: &AlignedSlice32<'_, f32>,
        exact_projection_delta0_phi: &AlignedSlice32<'_, f32>,
        obstruction_threshold: f32,
        moving_z_entropy: f32,
        sliding_flux: f32,
        void_score: f32,
        bifurcation_threshold: f32,
        tau_flux: f32,
        hysteresis_band: f32,
        high_density_rank1_thresh: f32,
        dominant_attractor_slot: u8,
        bloom_filter: &mut ContrastivePhaseBloomFilter,
        trail_inout: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, bool, u32, gcso_tagged_ptr_t)> {
        let (mean_obs, is_obstructed) = self.evaluate_cohomological_obstruction(
            cochain_residual_r,
            exact_projection_delta0_phi,
            obstruction_threshold,
        )?;
        if is_obstructed {
            let tw = trail_phase_words_mut(trail_inout);
            bloom_filter.insert_repulsive_pattern_words_weighted(tw, 1);
            tw[0] = swar_q7_neg_sat_x8(tw[0]);
            tw[1] = swar_q7_neg_sat_x8(tw[1]);
            tw[2] = swar_q7_neg_sat_x8(tw[2]);
            tw[3] = swar_q7_neg_sat_x8(tw[3]);
            tw[4] = swar_q7_neg_sat_x8(tw[4]);
            tw[5] = swar_q7_neg_sat_x8(tw[5]);
            tw[6] = swar_q7_neg_sat_x8(tw[6]);
            tw[7] = swar_q7_neg_sat_x8(tw[7]);
            trail_inout.flags |= TRAIL_FLAG_REPULSION_ACTIVE;
        }
        let (next_mode, evolved_ptr) = self.evaluate_branch_and_evolve_pointer(
            gcso_tagged_ptr_t(trail_inout.current_ptr),
            trail_inout.stigmergic_density,
            dominant_attractor_slot,
            moving_z_entropy,
            sliding_flux,
            void_score,
            bifurcation_threshold,
            tau_flux,
            hysteresis_band,
            high_density_rank1_thresh,
        );
        trail_inout.current_ptr = evolved_ptr.raw();
        Ok((mean_obs, is_obstructed, next_mode, evolved_ptr))
    }

    /// Bridges all inter-memory **Limina** transitions across `trail_chain`, evaluates the chain's
    /// Joint Probability Log-Likelihood $\sum_t \ln P(M_t \mid M_{t-1})$ and $S^1$ Sheaf Holonomy
    /// Obstruction ($H^1(K; \mathcal{F})$), evolves the terminal trail's pointer mode via EDBC,
    /// and steers `query_tensor` and/or `key_tensor` with QDPS + RIPA soft-clamping.
    ///
    /// Returns `(joint_log_prob_sum, holonomy_obstruction_energy, is_obstructed, repulsed_count, evolved_ptr)`.
    ///
    /// # Errors
    /// Propagates any error from [`ActionHub::validate_hub_cascade_params`],
    /// [`SheafCohomologyEvaluator::bridge_and_filter_liminal_trail_chain`], or
    /// [`SheafCohomologyEvaluator::evaluate_trail_chain_cohomology_and_joint_log_prob`].
    #[inline]
    fn coordinate_liminal_trail_chain_and_steer_qk(
        &mut self,
        trail_chain: &mut [gcso_pointer_trail_t],
        interpolation_u7: u8,
        contradiction_threshold: f32,
        obstruction_threshold: f32,
        moving_z_entropy: f32,
        sliding_flux: f32,
        void_score: f32,
        bifurcation_threshold: f32,
        tau_flux: f32,
        hysteresis_band: f32,
        high_density_rank1_thresh: f32,
        dominant_attractor_slot: u8,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        bloom_filter: &mut ContrastivePhaseBloomFilter,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(f32, f32, bool, u32, gcso_tagged_ptr_t)> {
        let len = trail_chain.len();
        if len == 0
            || !branchless_is_finite_f32(contradiction_threshold)
            || !branchless_is_finite_f32(obstruction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;

        let first_w = *trail_phase_words_ref(&trail_chain[0]);
        let mut prev_w = first_w;
        let mut joint_log_prob = 0.0_f32;
        let mut coboundary_defect_sum = 0.0_f32;
        let mut repulsed_count = 0u32;

        for idx in 1..len {
            let (head, tail) = trail_chain.split_at_mut(idx);
            let prev_ref = &head[idx - 1];
            let cur_mut = &mut tail[0];
            let (pre_coh, pre_lp, repulsed) = cur_mut.bridge_liminal_hallucinated_trail(
                prev_ref,
                interpolation_u7,
                contradiction_threshold,
            );
            let cur_w = *trail_phase_words_ref(cur_mut);
            if repulsed {
                bloom_filter.insert_repulsive_pattern_words_weighted(&cur_w, 1);
                repulsed_count = repulsed_count.wrapping_add(1);
            }
            let (coh, log_p) = if cur_w == prev_w {
                (1.0_f32, 0.0_f32)
            } else if interpolation_u7 >= 128 && !repulsed {
                (pre_coh, pre_lp)
            } else {
                branchless_q7_transition_log_prob_words_64(&prev_w, &cur_w)
            };
            joint_log_prob += log_p;
            coboundary_defect_sum += branchless_relu_f32(0.5 * (1.0 - coh));
            prev_w = cur_w;
        }

        let (joint_log_prob, holonomy_obs, is_obstructed) = if len == 1 {
            (0.0_f32, 0.0_f32, false)
        } else {
            let mean_step_defect = coboundary_defect_sum * fast_inv_len_f32(len - 1);
            let loop_coh = coherence_from_word_arrays_64(&first_w, &prev_w);
            let loop_defect = branchless_relu_f32(0.5 * (1.0 - loop_coh));
            let holonomy_energy =
                branchless_flush_subnormal_f32(0.5 * (mean_step_defect + loop_defect));
            let obstructed = holonomy_energy > branchless_relu_f32(obstruction_threshold);
            (
                branchless_flush_subnormal_f32(joint_log_prob),
                holonomy_energy,
                obstructed,
            )
        };

        let last_idx = len - 1;
        if is_obstructed && (trail_chain[last_idx].flags & TRAIL_FLAG_REPULSION_ACTIVE) == 0 {
            bloom_filter.insert_repulsive_pattern_words_weighted(&prev_w, 1);
            trail_chain[last_idx].stamp_phase_conjugate_repulsion();
            prev_w = *trail_phase_words_ref(&trail_chain[last_idx]);
        }
        let (_next_mode, evolved_ptr) = self.evaluate_branch_and_evolve_pointer(
            gcso_tagged_ptr_t(trail_chain[last_idx].current_ptr),
            trail_chain[last_idx].stigmergic_density,
            dominant_attractor_slot,
            moving_z_entropy,
            sliding_flux,
            void_score,
            bifurcation_threshold,
            tau_flux,
            hysteresis_band,
            high_density_rank1_thresh,
        );
        trail_chain[last_idx].current_ptr = evolved_ptr.raw();

        let _ = self.steer_qdps_words_prevalidated_core(
            &prev_w,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        );
        Ok((
            joint_log_prob,
            holonomy_obs,
            is_obstructed,
            repulsed_count,
            evolved_ptr,
        ))
    }
}

impl<T> StigmergicCohomologyCoordinator for T where
    T: ActionHub + SheafCohomologyEvaluator + EdbcBranchController + ?Sized
{
}

/// Blanket-implemented macro coordinator uniting [`EdbcBranchController`], [`AttractorField`],
/// and [`PersonaPatcher`] for bottom-up trail crystallization, metastable tunneling perturbation,
/// and persistent topological mode quantization.
pub trait BottomUpMacroCrystalCoordinator:
    EdbcBranchController + AttractorField + PersonaPatcher
{
    /// Evaluates EDBC branch mode transition (`0` = Normal, `1` = Pitchfork, `2` = Repulsion),
    /// screens `trails` for bottom-up crystallization candidacy (`evaluate_crystallization_candidacy`),
    /// injects metastable tunneling perturbation in Pitchfork mode (`1`) or records Bloom repulsion
    /// in Repulsion mode (`2`), triggers [`AttractorField::aggregate_bottom_up`], and quantizes the
    /// persistent topological persona profile (`persona_profile_out`).
    ///
    /// Returns `(next_branch_mode, crystallized_candidate_count, new_anchors_created, persona_r_bar, persona_kappa)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `trails` is empty, or propagates errors from
    /// [`EdbcBranchController::inject_metastable_tunneling_perturbation`],
    /// [`AttractorField::aggregate_bottom_up`], or
    /// [`PersonaPatcher::quantize_persistent_topological_mode`].
    #[inline]
    fn step_macro_crystallization_with_persona_and_tunneling(
        &mut self,
        trails: &mut [gcso_pointer_trail_t],
        bloom_filter: &mut ContrastivePhaseBloomFilter,
        moving_z_entropy: f32,
        sliding_flux: f32,
        void_score: f32,
        bifurcation_threshold: f32,
        tau_flux: f32,
        hysteresis_band: f32,
        density_crystallization_thresh: f32,
        coherence_crystallization_thresh: f32,
        alpha_pitchfork: f32,
        beta_pitchfork: f32,
        head_dim: u32,
        explore_head_start: usize,
        min_persona_coherence_r_bar: f32,
        persona_profile_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<(u32, u32, u32, f32, f32)> {
        if trails.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let next_mode = self.evaluate_branch_mode_transition(
            self.active_branch_mode(),
            moving_z_entropy,
            sliding_flux,
            void_score,
            bifurcation_threshold,
            tau_flux,
            hysteresis_band,
        );

        let mut candidate_count = 0u32;
        for trail in trails.iter_mut() {
            let trail_w = *trail_phase_words_ref(trail);
            let r_bar_t = circular_r_bar_from_words_64(&trail_w);
            let is_candidate = trail.evaluate_crystallization_candidacy(
                density_crystallization_thresh,
                coherence_crystallization_thresh,
                r_bar_t,
            );
            candidate_count += u32::from(is_candidate);

            if next_mode == 1 {
                let _ = self.inject_metastable_tunneling_perturbation(
                    &mut trail.accumulated_phase_delta,
                    gcso_tagged_ptr_t(trail.current_ptr),
                    void_score,
                    bifurcation_threshold,
                    alpha_pitchfork,
                    beta_pitchfork,
                    head_dim,
                    explore_head_start,
                )?;
            } else if next_mode == 2 {
                bloom_filter.insert_repulsive_pattern_words_weighted(&trail_w, 1);
                trail.flags |= TRAIL_FLAG_REPULSION_ACTIVE;
            }
        }

        let new_anchors = if candidate_count > 0 {
            self.aggregate_bottom_up()?
        } else {
            0
        };
        let (r_bar, kappa) = self.quantize_persistent_topological_mode(
            trails,
            min_persona_coherence_r_bar,
            persona_profile_out,
        )?;
        Ok((next_mode, candidate_count, new_anchors, r_bar, kappa))
    }

    /// Bridges all inter-memory **Limina** gaps across `trail_chain` ($M_1 \to \dots \to M_T$),
    /// checks whether the stitched trajectory forms a coherent Global Section $\Gamma(K; \mathcal{F})$
    /// (`repulsed_transitions == 0`, $H^1(K; \mathcal{F}) = 0$), marks global section candidates
    /// via [`ColdPathTrailCrystalExt::evaluate_liminal_global_section_candidacy`], triggers
    /// [`AttractorField::aggregate_bottom_up`], and quantizes the persistent persona profile.
    ///
    /// Returns `(joint_log_prob, repulsed_transitions, candidate_count, new_anchors, persona_r_bar, persona_kappa)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `trail_chain` is empty or thresholds are non-finite.
    #[inline]
    fn step_liminal_chain_crystallization_with_persona_and_tunneling(
        &mut self,
        trail_chain: &mut [gcso_pointer_trail_t],
        interpolation_u7: u8,
        contradiction_threshold: f32,
        bloom_filter: &mut ContrastivePhaseBloomFilter,
        moving_z_entropy: f32,
        sliding_flux: f32,
        void_score: f32,
        bifurcation_threshold: f32,
        tau_flux: f32,
        hysteresis_band: f32,
        density_crystallization_thresh: f32,
        coherence_crystallization_thresh: f32,
        alpha_pitchfork: f32,
        beta_pitchfork: f32,
        head_dim: u32,
        explore_head_start: usize,
        min_persona_coherence_r_bar: f32,
        persona_profile_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<(f32, u32, u32, u32, f32, f32)> {
        if trail_chain.is_empty() || !branchless_is_finite_f32(contradiction_threshold) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let mut joint_log_prob = 0.0_f32;
        let mut repulsed_transitions = 0u32;
        for idx in 1..trail_chain.len() {
            let (head, tail) = trail_chain.split_at_mut(idx);
            let prev_ref = &head[idx - 1];
            let cur_mut = &mut tail[0];
            let (_coh, lp, repulsed) = cur_mut.bridge_liminal_hallucinated_trail(
                prev_ref,
                interpolation_u7,
                contradiction_threshold,
            );
            joint_log_prob += lp;
            if repulsed {
                bloom_filter.insert_repulsive_pattern_words_weighted(
                    trail_phase_words_ref(cur_mut),
                    1,
                );
                repulsed_transitions = repulsed_transitions.wrapping_add(1);
            }
        }

        let cohomology_obstructed = repulsed_transitions > 0;
        let next_mode = self.evaluate_branch_mode_transition(
            self.active_branch_mode(),
            moving_z_entropy,
            sliding_flux,
            void_score,
            bifurcation_threshold,
            tau_flux,
            hysteresis_band,
        );

        let mut candidate_count = 0u32;
        for trail in trail_chain.iter_mut() {
            let trail_w = *trail_phase_words_ref(trail);
            let r_bar_t = circular_r_bar_from_words_64(&trail_w);
            let is_candidate = trail.evaluate_liminal_global_section_candidacy(
                density_crystallization_thresh,
                coherence_crystallization_thresh,
                r_bar_t,
                cohomology_obstructed,
            );
            candidate_count += u32::from(is_candidate);

            if next_mode == 1 {
                let _ = self.inject_metastable_tunneling_perturbation(
                    &mut trail.accumulated_phase_delta,
                    gcso_tagged_ptr_t(trail.current_ptr),
                    void_score,
                    bifurcation_threshold,
                    alpha_pitchfork,
                    beta_pitchfork,
                    head_dim,
                    explore_head_start,
                )?;
            } else if next_mode == 2 {
                bloom_filter.insert_repulsive_pattern_words_weighted(&trail_w, 1);
                trail.flags |= TRAIL_FLAG_REPULSION_ACTIVE;
            }
        }

        let new_anchors = if candidate_count > 0 {
            self.aggregate_bottom_up()?
        } else {
            0
        };
        let (r_bar, kappa) = self.quantize_persistent_topological_mode(
            trail_chain,
            min_persona_coherence_r_bar,
            persona_profile_out,
        )?;
        Ok((
            branchless_flush_subnormal_f32(joint_log_prob),
            repulsed_transitions,
            candidate_count,
            new_anchors,
            r_bar,
            kappa,
        ))
    }
}

impl<T> BottomUpMacroCrystalCoordinator for T where
    T: EdbcBranchController + AttractorField + PersonaPatcher + ?Sized
{
}

/// Unified Lv.3 Macro Observer trait over [`ActionHub`] (`traits/cold_path.rs`).
///
/// Observes [`crate::traits::telemetry`] trajectories from [`ActionHub`] to evaluate EDBC pitchfork
/// bifurcations, matrix-free cellular sheaf cohomology $H^1(K; \mathcal{F})$, unified-field
/// attractor crystallization, and Online Toroidal $k$-Means / DP-Means macro distribution steering.
pub trait ColdPathObserver:
    ActionHub
    + SheafCohomologyEvaluator
    + EdbcBranchController
    + AttractorField
    + MultiDomainAttractorSpace
    + PersonaPatcher
    + AttractorSlotSynergyCoordinator
    + StigmergicCohomologyCoordinator
    + BottomUpMacroCrystalCoordinator
{
    /// Coordinates a per-token step directly on the **Single Unified Attractor Manifold**
    /// (`anchor_phases`, `anchor_base_energies`) without requiring an 8-slot phase bank:
    /// 1. Steps `input_ptr` in [`ActionHub`] to update `trail_out`.
    /// 2. Folds all unified-field anchors via [`AttractorField::steer_trail_on_unified_field_from_slices`],
    ///    binding the dominant anchor ID into `trail_out.target_anchor_id`.
    /// 3. Applies QDPS + RIPA phase steering to `query_tensor` and/or `key_tensor`.
    /// 4. Optionally modulates candidate memory Attention scores (`attention_scores`) by
    ///    Attention $\times$ DPSR phase interference.
    ///
    /// Returns `(final_pull_force, mean_coherence, active_anchor_count, dominant_anchor_id)`.
    ///
    /// # Errors
    /// Propagates any error from [`ActionHub::validate_hub_cascade_params`],
    /// [`AttractorField::steer_trail_on_unified_field_from_slices`], or
    /// [`ActionHub::filter_qdps_and_steer_qk_safe`].
    #[inline]
    fn coordinate_unified_field_step_and_steer_qk(
        &mut self,
        input_ptr: u64,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        attention_interference_gain: f32,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32, u32, u32)> {
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        self.step_pointer(input_ptr, trail_out)?;
        let (pull, mean_coh, active_cnt, uf_w) = self
            .steer_trail_on_unified_field_from_slices_to_words(
                anchor_phases,
                anchor_base_energies,
                repulsion_phases_and_strength,
                trail_out,
            )?;
        let _ = self.steer_qdps_words_prevalidated_core(
            &uf_w,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        );
        if let Some(scores) = attention_scores {
            if !anchor_phases.is_empty() {
                let _ = self.modulate_hub_attention_scores_by_unified_field(
                    scores,
                    trail_out,
                    anchor_phases,
                    attention_interference_gain,
                )?;
            }
        }
        Ok((pull, mean_coh, active_cnt, trail_out.target_anchor_id))
    }

    /// Coordinates the closed-loop **2-Origin Gateway $\to$ Online Toroidal $k$-Means / DP-Means
    /// Macro Distribution (`centroids`) $\to$ EDBC Pitchfork / Tunneling / Repulsion** pipeline:
    /// 1. Ingests `signal` via [`ActionHubGatewayExt::ingest_gateway_kmeans_and_steer_safe`],
    ///    steering `query_tensor` / `key_tensor` / `logits_opt` and updating the winning macro
    ///    centroid in `centroids`.
    /// 2. Evaluates EDBC branch mode transition (`0` = Normal, `1` = Pitchfork, `2` = Repulsion).
    /// 3. In Pitchfork mode (`1`), injects metastable tunneling perturbation onto `trail_out`;
    ///    in Repulsion mode (`2`), inserts `trail_out.accumulated_phase_delta` into `bloom_filter`.
    ///
    /// Returns `(GatewayFrame, CausalProvenanceReport, assigned_cluster_idx, assigned_anchor_id, updated_gravity_mass, next_branch_mode)`.
    ///
    /// # Errors
    /// Propagates any error from [`ActionHubGatewayExt::ingest_gateway_kmeans_and_steer_safe`] or
    /// [`EdbcBranchController::inject_metastable_tunneling_perturbation`].
    #[inline]
    fn coordinate_kmeans_macro_distribution_and_edbc_step<S: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        signal: &S,
        centroids: &mut [CrystallizedAttractorRecord],
        split_coherence_thresh: f32,
        moving_z_entropy: f32,
        sliding_flux: f32,
        void_score: f32,
        bifurcation_threshold: f32,
        tau_flux: f32,
        hysteresis_band: f32,
        alpha_pitchfork: f32,
        beta_pitchfork: f32,
        explore_head_start: usize,
        bloom_filter: &mut ContrastivePhaseBloomFilter,
        ledger_slot: &mut gcso_ledger_slot_t,
        trail_out: &mut gcso_pointer_trail_t,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(
        GatewayFrame,
        CausalProvenanceReport,
        usize,
        u32,
        f32,
        u32,
    )> {
        let (frame, report, cluster_idx, anchor_id, mass, _spawned) = self
            .ingest_gateway_kmeans_and_steer_safe(
                signal,
                centroids,
                split_coherence_thresh,
                ledger_slot,
                trail_out,
                query_tensor,
                key_tensor,
                logits_opt,
            )?;

        let next_mode = self.evaluate_branch_mode_transition(
            self.active_branch_mode(),
            moving_z_entropy,
            sliding_flux,
            void_score,
            bifurcation_threshold,
            tau_flux,
            hysteresis_band,
        );

        let head_dim = self.head_dim();
        if next_mode == 1 {
            let _ = self.inject_metastable_tunneling_perturbation(
                &mut trail_out.accumulated_phase_delta,
                gcso_tagged_ptr_t(trail_out.current_ptr),
                void_score,
                bifurcation_threshold,
                alpha_pitchfork,
                beta_pitchfork,
                head_dim,
                explore_head_start,
            )?;
        } else if next_mode == 2 {
            bloom_filter
                .insert_repulsive_pattern_words_weighted(trail_phase_words_ref(trail_out), 1);
            trail_out.flags |= TRAIL_FLAG_REPULSION_ACTIVE;
        }

        Ok((frame, report, cluster_idx, anchor_id, mass, next_mode))
    }
}

impl<T> ColdPathObserver for T where
    T: ActionHub
        + SheafCohomologyEvaluator
        + EdbcBranchController
        + AttractorField
        + MultiDomainAttractorSpace
        + PersonaPatcher
        + ?Sized
{
}

/// Composite Cold-Path Orchestrator trait uniting Macro evaluation, attractor steering, and mapped-storage persistence.
pub trait ColdPathOrchestrator:
    EntropyEvaluator + AttractorField + PersonaPatcher + L2pSvdProjector + KvKeyframeCache + MappedStorage
{
}

impl<T> ColdPathOrchestrator for T where
    T: EntropyEvaluator
        + AttractorField
        + PersonaPatcher
        + L2pSvdProjector
        + KvKeyframeCache
        + MappedStorage
        + ?Sized
{
}

/// Full-Stack Fractal GCSO Engine trait uniting Nano, Micro, Mezzo, and Macro contracts under static dispatch.
pub trait FullStackGcsoEngine:
    HotPathExecutionEngine
    + ColdPathOrchestrator
    + StigmergicChunkOps
    + GcsoRuntimeContext
    + SystemCapabilityQuery
{
}

impl<T> FullStackGcsoEngine for T where
    T: HotPathExecutionEngine
        + ColdPathOrchestrator
        + StigmergicChunkOps
        + GcsoRuntimeContext
        + SystemCapabilityQuery
        + ?Sized
{
}

