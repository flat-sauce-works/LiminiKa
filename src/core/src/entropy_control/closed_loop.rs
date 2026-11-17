// name: src/core/src/entropy_control/closed_loop.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Blanket-implemented [EntropyClosedLoopCoordinator] over EDBC, sheaf cohomology, and hierarchical trees.

use super::*;
#[allow(unused_imports)]
use super::{controller::*, kernels::*};

// ===================================================================
// 3. Blanket-Implemented Closed-Loop EDBC + Sheaf + Limina + Gateway Coordinator
// ===================================================================

/// Blanket-implemented closed-loop coordinator uniting [`EdbcBranchController`],
/// [`SheafCohomologyEvaluator`], [`HierarchicalTreeTable`], [`ManifoldGatewaySignal`],
/// Online Toroidal $k$-Means / DP-Means Coarse-to-Fine routing, and `.gcso`
/// [`CrystallizedAttractorRecord`] persistence.
///
/// Automatically implemented for all `T: EdbcBranchController + SheafCohomologyEvaluator + HierarchicalTreeTable + ?Sized`
/// (including [`EntropyController`], [`crate::traits::ColdPathObserver`], and [`crate::traits::FullStackGcsoEngine`]).
pub trait EntropyClosedLoopCoordinator:
    EdbcBranchController + SheafCohomologyEvaluator + HierarchicalTreeTable
{
    /// Computes the mean-square $\mathcal{C}_{\mathrm{void}}$ latent energy alongside the
    /// normalized $\ell_1 / \ell_2$ coherence ratio in a single AVX2 pass (`sum_abs_and_sq_f32_simd8`)
    /// with zero hardware `fdiv` instructions via `fast_inv_len_f32`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `key_vector` is empty.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if any element or accumulated norm is non-finite.
    #[inline(always)]
    fn eval_liminal_void_with_coherence_ratio(
        &self,
        key_vector: &AlignedSlice32<'_, f32>,
    ) -> GcsoResult<(f32, f32)> {
        if key_vector.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let slice = key_vector.as_slice();
        let (sum_abs, sum_sq) = sum_abs_and_sq_f32_simd8(slice);
        if !branchless_is_finite_f32(sum_abs) || !branchless_is_finite_f32(sum_sq) {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        let inv_n = fast_inv_len_f32(slice.len());
        let mean_sq = branchless_flush_subnormal_f32(sum_sq * inv_n);
        let mean_abs = sum_abs * inv_n;
        let valid_rms = mean_sq > 1.0e-24;
        let inv_rms = branchless_select_f32(
            valid_rms,
            branchless_rsqrt_f32(branchless_select_f32(valid_rms, mean_sq, 1.0)),
            0.0,
        );
        let coh_ratio =
            branchless_flush_subnormal_f32(branchless_clamp_f32(mean_abs * inv_rms, 0.0, 1.0));
        Ok((mean_sq, coh_ratio))
    }

    /// Evaluates Top-2 sparse surrogate entropy via the strictly branchless `0-ln` Taylor-Horner
    /// polynomial, combines it with inter-layer cosine distance into Moving Z-Score Normalized
    /// Attention Entropy $\tilde{H}$, and steps the EDBC controller via
    /// [`EntropyEvaluator::eval_stateful`].
    ///
    /// # Errors
    /// Propagates [`GCSO_ERROR_EDBC_SINGULARITY`] if computed entropy is non-finite.
    #[inline(always)]
    fn eval_surrogate_and_step(
        &mut self,
        p_max: f32,
        p_2nd: f32,
        cosine_dist: f32,
        mu_dist: f32,
        sigma_dist: f32,
        alpha_z: f32,
        gamma_mod: f32,
        state_out: &mut gcso_edbc_state_t,
    ) -> GcsoResult<f32> {
        let eps = branchless_select_f32(
            (state_out.singularity_eps > 0.0) & branchless_is_finite_f32(state_out.singularity_eps),
            state_out.singularity_eps,
            DEFAULT_SINGULARITY_EPS,
        );
        let h_surr = compute_sparse_surrogate_entropy(p_max, p_2nd, eps);
        let h_tilde = compute_moving_z_score_entropy(
            h_surr,
            cosine_dist,
            mu_dist,
            sigma_dist,
            eps,
            alpha_z,
            gamma_mod,
        );
        self.eval_stateful(h_tilde, state_out)?;
        Ok(state_out.moving_z_entropy)
    }

    /// Bridges all inter-memory **Limina** transitions across `trail_chain` ($M_1 \to \dots \to M_T$)
    /// and evaluates the **Joint Probability Chain Rule Log-Likelihood**
    /// $\ln P(M_1, \dots, M_T) = \sum_{t=1}^{T-1} \ln P(M_t \mid M_{t-1})$ and **$S^1$ Sheaf
    /// Holonomy Obstruction Energy** ($H^1(K; \mathcal{F})$) in a **single register-chained pass**,
    /// steps the stateful EDBC controller, and updates the terminal trail's pointer mode and
    /// Bloom repulsion memory.
    ///
    /// Returns `(edbc_state, evolved_ptr, joint_log_prob, holonomy_obstruction, repulsion_triggered)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `trail_chain` is empty or thresholds are non-finite.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if `token_z_score` or `void_score` is non-finite.
    #[inline]
    fn step_with_liminal_trail_chain_and_bloom_guard(
        &mut self,
        trail_chain: &mut [gcso_pointer_trail_t],
        interpolation_u7: u8,
        contradiction_threshold: f32,
        obstruction_threshold: f32,
        token_z_score: f32,
        void_score: f32,
        high_density_rank1_thresh: f32,
        dominant_attractor_slot: u8,
        bloom_filter: &mut ContrastivePhaseBloomFilter,
    ) -> GcsoResult<(gcso_edbc_state_t, gcso_tagged_ptr_t, f32, f32, bool)> {
        if trail_chain.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !branchless_is_finite_f32(void_score) {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        let (joint_log_prob, holonomy_obs, is_obstructed, repulsed_transitions, last_w) =
            bridge_and_eval_liminal_trail_chain_single_pass(
                trail_chain,
                interpolation_u7,
                contradiction_threshold,
                obstruction_threshold,
                bloom_filter,
            )?;

        let mut st = gcso_edbc_state_t::default();
        self.eval_stateful(token_z_score, &mut st)?;
        let bif_thresh = branchless_select_f32(
            (st.bifurcation_threshold > 0.0) & branchless_is_finite_f32(st.bifurcation_threshold),
            st.bifurcation_threshold,
            DEFAULT_BIFURCATION_THRESHOLD,
        );
        st.bifurcation_threshold = bif_thresh;

        let last_idx = trail_chain.len() - 1;
        let repulsion_triggered = is_obstructed
            | (repulsed_transitions > 0)
            | (st.active_branch_mode == EDBC_MODE_PHASE_CONJUGATE_REPULSION);

        let low_entropy_skip_thresh = branchless_relu_f32(bif_thresh * 0.5);
        let mut evolved_ptr = gcso_tagged_ptr_t(trail_chain[last_idx].current_ptr)
            .evolve_mode_stigmergically(
                trail_chain[last_idx].stigmergic_density,
                st.moving_z_entropy,
                repulsion_triggered,
                dominant_attractor_slot & ATTRACTOR_SLOT_ID_MASK,
                low_entropy_skip_thresh,
                high_density_rank1_thresh,
            );

        if repulsion_triggered {
            if (trail_chain[last_idx].flags & TRAIL_FLAG_REPULSION_ACTIVE) == 0 {
                bloom_filter.insert_repulsive_pattern_words_weighted(&last_w, 1);
                let w_dst = trail_phase_words_mut(&mut trail_chain[last_idx]);
                w_dst[0] = swar_q7_neg_sat_x8(last_w[0]);
                w_dst[1] = swar_q7_neg_sat_x8(last_w[1]);
                w_dst[2] = swar_q7_neg_sat_x8(last_w[2]);
                w_dst[3] = swar_q7_neg_sat_x8(last_w[3]);
                w_dst[4] = swar_q7_neg_sat_x8(last_w[4]);
                w_dst[5] = swar_q7_neg_sat_x8(last_w[5]);
                w_dst[6] = swar_q7_neg_sat_x8(last_w[6]);
                w_dst[7] = swar_q7_neg_sat_x8(last_w[7]);
                trail_chain[last_idx].flags |= TRAIL_FLAG_REPULSION_ACTIVE;
            }
            st.active_branch_mode = EDBC_MODE_PHASE_CONJUGATE_REPULSION;
            st.repulsion_gain = LUT_REPULSION_GAIN[2];
            st.sample_temperature =
                branchless_max_f32(st.sample_temperature, LUT_SAMPLE_TEMPERATURE[2]);
            let rep_mode = evolved_ptr.action_mode_flags() | ACTION_PRESET_REPULSION_ESCAPE;
            evolved_ptr = evolved_ptr
                .with_action_and_slot(ATTRACTOR_SLOT_REPULSION, rep_mode)
                .conjugate_phase();
            trail_chain[last_idx].cluster_id = ATTRACTOR_SLOT_REPULSION as u32;
        } else {
            let preserve_anchor = trail_chain[last_idx].target_anchor_id != 0;
            trail_chain[last_idx].cluster_id = branchless_select_u32(
                preserve_anchor,
                trail_chain[last_idx].target_anchor_id,
                evolved_ptr.target_attractor_slot() as u32,
            );
        }
        trail_chain[last_idx].current_ptr = evolved_ptr.raw();

        Ok((
            st,
            evolved_ptr,
            joint_log_prob,
            holonomy_obs,
            repulsion_triggered,
        ))
    }

    /// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) on the
    /// **Single Unified Attractor Manifold** into `trail` in register-resident order-invariant
    /// constant time via [`branchless_q7_unified_field_fold_words_with_init_weight_64`], binds the
    /// 32-bit dominant anchor index into both `trail.target_anchor_id` and `trail.cluster_id`,
    /// steps the EDBC controller, and evolves `trail.current_ptr` using
    /// `(dominant_anchor_id as u8) & 0x07` as the working lane.
    ///
    /// Returns `(edbc_state, evolved_ptr, total_resonant_pull, mean_coherence, dominant_anchor_id)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `anchor_phases.len() != anchor_base_energies.len()`.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if `token_z_score` or `void_score` is non-finite.
    #[inline(always)]
    fn step_with_unified_field_and_pointer(
        &mut self,
        token_z_score: f32,
        void_score: f32,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        high_density_rank1_thresh: f32,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(gcso_edbc_state_t, gcso_tagged_ptr_t, f32, f32, u32)> {
        if anchor_phases.len() != anchor_base_energies.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !branchless_is_finite_f32(void_score) {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        let init_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(trail.attractor_pull_force));
        let cur_w = *trail_phase_words_ref(trail);
        let (folded_w, total_weight, mean_coh, dom_idx, act_cnt) =
            branchless_q7_unified_field_fold_words_with_init_weight_64(
                &cur_w,
                init_pull,
                anchor_phases,
                anchor_base_energies,
            );
        if act_cnt > 0 {
            *trail_phase_words_mut(trail) = folded_w;
            trail.target_anchor_id = dom_idx;
            trail.cluster_id = dom_idx;
            trail.attractor_pull_force = branchless_min_f32(total_weight, MAX_ATTRACTOR_PULL_FORCE);
        }

        let mut st = gcso_edbc_state_t::default();
        self.eval_stateful(token_z_score, &mut st)?;
        let bif_thresh = branchless_select_f32(
            (st.bifurcation_threshold > 0.0) & branchless_is_finite_f32(st.bifurcation_threshold),
            st.bifurcation_threshold,
            DEFAULT_BIFURCATION_THRESHOLD,
        );
        st.bifurcation_threshold = bif_thresh;

        let dom_lane = (trail.target_anchor_id as u8) & ATTRACTOR_SLOT_ID_MASK;
        let is_repulsion = st.active_branch_mode == EDBC_MODE_PHASE_CONJUGATE_REPULSION;
        let low_entropy_skip_thresh = branchless_relu_f32(bif_thresh * 0.5);
        let evolved_ptr = gcso_tagged_ptr_t(trail.current_ptr).evolve_mode_stigmergically(
            trail.stigmergic_density,
            st.moving_z_entropy,
            is_repulsion,
            dom_lane,
            low_entropy_skip_thresh,
            high_density_rank1_thresh,
        );
        trail.current_ptr = evolved_ptr.raw();
        let preserve_anchor = (trail.target_anchor_id != 0) & (!is_repulsion);
        trail.cluster_id = branchless_select_u32(
            preserve_anchor,
            trail.target_anchor_id,
            evolved_ptr.target_attractor_slot() as u32,
        );

        Ok((
            st,
            evolved_ptr,
            total_weight,
            mean_coh,
            trail.target_anchor_id,
        ))
    }

    /// Executes the single-pass closed-loop synthesis of **EDBC Langevin Pitchfork Bifurcation / Tunneling**,
    /// **Sequential Conditional Transition Probability** ($\sum_t \ln P(M_t \mid M_{t-1})$ $\times$ Attention phase interference),
    /// and **Indirect Pointer-Chain Selection & Rewiring** in `ActionHub`.
    ///
    /// Returns `(edbc_state, evolved_ptr, selected_trail_id, bayesian_score, transition_log_prob, tunneling_x_star, found_coherent)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `candidate_trails` is empty, `explore_head_start > 64`,
    /// or gains/thresholds are non-finite.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if `token_z_score`, `void_score`, `alpha`, `beta <= 0.0`,
    /// `d0 <= 0.0`, or `head_dim` is invalid.
    #[inline(always)]
    fn step_with_bayesian_pointer_selection_and_tunneling(
        &mut self,
        src_trail: &mut gcso_pointer_trail_t,
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
        bloom_filter: &mut ContrastivePhaseBloomFilter,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(
        gcso_edbc_state_t,
        gcso_tagged_ptr_t,
        u32,
        f32,
        f32,
        f32,
        bool,
    )> {
        if candidate_trails.is_empty()
            || explore_head_start > MAX_ATTENTION_HEADS
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(repulsion_penalty)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !branchless_is_finite_f32(void_score)
            || !branchless_is_finite_f32(alpha)
            || !branchless_is_finite_f32(beta)
            || !branchless_is_finite_f32(d0)
            || beta <= 0.0
            || d0 <= 0.0
            || head_dim == 0
            || (head_dim & 1) != 0
        {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        step_edbc_tunneling_and_select_bayesian_core(
            self,
            src_trail,
            None,
            candidate_trails,
            token_z_score,
            void_score,
            alpha,
            beta,
            d0,
            head_dim,
            explore_head_start,
            phase_interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
            repulsion_penalty,
            interpolation_u7,
            contradiction_threshold,
            high_density_rank1_thresh,
            0,
            0,
            None,
            bloom_filter,
            attention_scores,
        )
    }

    /// Executes **Coarse-to-Fine Macro-to-Micro Routing ($\mathcal{O}(K) \to \mathcal{O}(M)$)**
    /// combined with **EDBC Pitchfork Metastable Tunneling & Bayesian Pointer-Trail Rewiring**:
    /// 1. **Coarse Step ($\mathcal{O}(K)$ Macro Centroid Routing)**: Routes `src_trail` to the
    ///    highest-gravity coherent macro centroid $k^*$ in `macro_centroids` via
    ///    `CrystallizedAttractorRecord::route_coarse_to_fine_trail_with_words_64`, Slerp-pulling
    ///    `src_trail` by `coarse_pull_u7`.
    /// 2. **Fine Step ($\mathcal{O}(M)$ + Pitchfork Tunneling)**: Passes `macro_anchor_id` and
    ///    register-resident `bridged_w` directly into `step_edbc_tunneling_and_select_bayesian_core`
    ///    so candidates stamped with `macro_anchor_id` receive the macro-cluster link alignment bonus
    ///    in a single pass without any separate pre-alignment scan over `candidate_trails`.
    ///
    /// Returns `(edbc_state, evolved_ptr, macro_cluster_idx, macro_anchor_id, selected_trail_id, bayesian_score, transition_log_prob, tunneling_x_star, found_coherent)`.
    ///
    /// # Errors
    /// Propagates any validation or singularity error from [`Self::step_with_bayesian_pointer_selection_and_tunneling`].
    #[inline]
    fn route_coarse_to_fine_and_step_bayesian_tunneling(
        &mut self,
        macro_centroids: &[CrystallizedAttractorRecord],
        coarse_pull_u7: u8,
        src_trail: &mut gcso_pointer_trail_t,
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
        fine_interpolation_u7: u8,
        contradiction_threshold: f32,
        high_density_rank1_thresh: f32,
        bloom_filter: &mut ContrastivePhaseBloomFilter,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(
        gcso_edbc_state_t,
        gcso_tagged_ptr_t,
        usize,
        u32,
        u32,
        f32,
        f32,
        f32,
        bool,
    )> {
        if candidate_trails.is_empty()
            || explore_head_start > MAX_ATTENTION_HEADS
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(repulsion_penalty)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !branchless_is_finite_f32(void_score)
            || !branchless_is_finite_f32(alpha)
            || !branchless_is_finite_f32(beta)
            || !branchless_is_finite_f32(d0)
            || beta <= 0.0
            || d0 <= 0.0
            || head_dim == 0
            || (head_dim & 1) != 0
        {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        let (macro_idx, macro_anchor_id, _macro_coh, _macro_grav, bridged_w, win_w) =
            CrystallizedAttractorRecord::route_coarse_to_fine_trail_with_words_64(
                macro_centroids,
                src_trail,
                coarse_pull_u7,
            );
        let has_win_centroid =
            macro_idx < macro_centroids.len() && macro_centroids[macro_idx].sample_count > 0;

        let (
            st,
            evolved_ptr,
            selected_id,
            bayes_score,
            transition_log_prob,
            tunneling_x_star,
            found_coherent,
        ) = step_edbc_tunneling_and_select_bayesian_core(
            self,
            src_trail,
            Some(bridged_w),
            candidate_trails,
            token_z_score,
            void_score,
            alpha,
            beta,
            d0,
            head_dim,
            explore_head_start,
            phase_interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
            repulsion_penalty,
            fine_interpolation_u7,
            contradiction_threshold,
            high_density_rank1_thresh,
            macro_anchor_id,
            0,
            has_win_centroid.then_some(&win_w),
            bloom_filter,
            attention_scores,
        )?;

        Ok((
            st,
            evolved_ptr,
            macro_idx,
            macro_anchor_id,
            selected_id,
            bayes_score,
            transition_log_prob,
            tunneling_x_star,
            found_coherent,
        ))
    }

    /// Executes **Top-2 Liminal Macro-to-Micro Bifurcation Routing ($\mathcal{O}(K) \to \mathcal{O}(M)$)**
    /// united with **EDBC Pitchfork Metastable Tunneling & Bayesian Pointer-Trail Rewiring**:
    /// 1. **Coarse Step ($\mathcal{O}(K)$ Top-2 Liminal Centroid Routing)**: Routes `src_trail` to
    ///    the top-2 macro centroids `(idx1, idx2, aid1, aid2)` in `macro_centroids` via
    ///    `CrystallizedAttractorRecord::route_top2_liminal_centroids_with_words_64`, interpolating
    ///    onto the Limina geodesic midpoint when `(grav1 - grav2) <= bifurcation_margin * grav1`.
    /// 2. **Fine Step ($\mathcal{O}(M)$ + Pitchfork Tunneling)**: Evaluates stateful EDBC entropy,
    ///    injects Pitchfork metastable tunneling perturbation when in Bifurcation mode, and selects/rewires
    ///    the optimal candidate trail via `step_edbc_tunneling_and_select_bayesian_core`.
    ///
    /// Returns `(edbc_state, evolved_ptr, idx1, idx2, aid1, aid2, is_liminal, selected_trail_id, bayesian_score, transition_log_prob, tunneling_x_star, found_coherent)`.
    ///
    /// # Errors
    /// Propagates any validation or singularity error from [`Self::route_coarse_to_fine_and_step_bayesian_tunneling`].
    #[inline]
    fn route_top2_liminal_and_step_bayesian_tunneling(
        &mut self,
        macro_centroids: &[CrystallizedAttractorRecord],
        coarse_pull_u7: u8,
        bifurcation_margin: f32,
        src_trail: &mut gcso_pointer_trail_t,
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
        fine_interpolation_u7: u8,
        contradiction_threshold: f32,
        high_density_rank1_thresh: f32,
        bloom_filter: &mut ContrastivePhaseBloomFilter,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(
        gcso_edbc_state_t,
        gcso_tagged_ptr_t,
        usize,
        usize,
        u32,
        u32,
        bool,
        u32,
        f32,
        f32,
        f32,
        bool,
    )> {
        if candidate_trails.is_empty()
            || explore_head_start > MAX_ATTENTION_HEADS
            || !branchless_is_finite_f32(bifurcation_margin)
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(repulsion_penalty)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !branchless_is_finite_f32(void_score)
            || !branchless_is_finite_f32(alpha)
            || !branchless_is_finite_f32(beta)
            || !branchless_is_finite_f32(d0)
            || beta <= 0.0
            || d0 <= 0.0
            || head_dim == 0
            || (head_dim & 1) != 0
        {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        let (idx1, idx2, aid1, aid2, _eff_coh, _tot_grav, is_liminal, bridged_w, w1, _w2) =
            CrystallizedAttractorRecord::route_top2_liminal_centroids_with_words_64(
                macro_centroids,
                src_trail,
                coarse_pull_u7,
                bifurcation_margin,
            );
        let has_win_centroid =
            idx1 < macro_centroids.len() && macro_centroids[idx1].sample_count > 0;

        let (
            st,
            evolved_ptr,
            selected_id,
            bayes_score,
            transition_log_prob,
            tunneling_x_star,
            found_coherent,
        ) = step_edbc_tunneling_and_select_bayesian_core(
            self,
            src_trail,
            Some(bridged_w),
            candidate_trails,
            token_z_score,
            void_score,
            alpha,
            beta,
            d0,
            head_dim,
            explore_head_start,
            phase_interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
            repulsion_penalty,
            fine_interpolation_u7,
            contradiction_threshold,
            high_density_rank1_thresh,
            aid1,
            branchless_select_u32(is_liminal, aid2, 0),
            has_win_centroid.then_some(&w1),
            bloom_filter,
            attention_scores,
        )?;

        Ok((
            st,
            evolved_ptr,
            idx1,
            idx2,
            aid1,
            aid2,
            is_liminal,
            selected_id,
            bayes_score,
            transition_log_prob,
            tunneling_x_star,
            found_coherent,
        ))
    }

    /// Modulates pre-Softmax candidate memory Attention scores (`attention_scores[..M]`) in-place
    /// in zero-copy, zero-allocation ($C_{\mathrm{alloc}} = 0$) constant time per candidate by
    /// combining:
    /// 1. Loop-hoisted candidate trail **Bayes-Attention Stigmergic Transition Score**
    ///    $\Delta z_{t \to j}$ and $H^1 \neq 0$ repulsion penalty via
    ///    [`StigmergicTrailOps::modulate_candidate_trail_scores_in_place`], and
    /// 2. EDBC's energy-guided dynamic sampling temperature $T_{\mathrm{sample}}$
    ///    fused with multiplicative **Phase-Interference Inverse-Temperature Scaling**
    ///    $\beta_{\mathrm{eff}} = \mathrm{clamp}(1.0 + \gamma \bar{\mathcal{C}}_{\mathrm{phase}}, 0.25, 4.0) / T_{\mathrm{sample}}$.
    ///
    /// Returns `(mean_coherence, effective_inv_temp, best_candidate_id, coherent_candidate_count)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `attention_scores` or `candidate_trails` is
    /// empty, `head_dim == 0`, `(head_dim & 1) != 0`, or any parameter is non-finite.
    #[inline(always)]
    fn modulate_candidate_trails_with_edbc_temperature(
        &self,
        attention_scores: &mut AlignedSliceMut32<'_, f32>,
        src_trail: &gcso_pointer_trail_t,
        candidate_trails: &[gcso_pointer_trail_t],
        head_dim: u32,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        contradiction_threshold: f32,
        repulsion_penalty: f32,
        base_temperature: f32,
        moving_z_entropy: f32,
        void_score: f32,
        tau_eff: f32,
        alpha: f32,
        beta: f32,
        d0: f32,
    ) -> GcsoResult<(f32, f32, u32, u32)> {
        let count = attention_scores.len().min(candidate_trails.len());
        if count == 0
            || head_dim == 0
            || (head_dim & 1) != 0
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(contradiction_threshold)
            || !branchless_is_finite_f32(repulsion_penalty)
            || !branchless_is_finite_f32(base_temperature)
            || !branchless_is_finite_f32(moving_z_entropy)
            || !branchless_is_finite_f32(void_score)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let edbc_temp = eval_temperature_for_mode(
            self.active_branch_mode(),
            base_temperature,
            moving_z_entropy,
            void_score,
            tau_eff,
            alpha,
            beta,
            d0,
        );
        let inv_edbc_temp = fast_inv_pos_normal_f32(branchless_max_f32(edbc_temp, 0.05));
        let scores = &mut attention_scores.as_mut_slice()[..count];
        let (mean_coh, best_id, coherent_count) = src_trail
            .modulate_candidate_trail_scores_in_place(
                scores,
                &candidate_trails[..count],
                head_dim,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                contradiction_threshold,
                repulsion_penalty,
            );
        let phase_inv_temp =
            branchless_phase_coherence_inv_temp_scale(mean_coh, phase_interference_gain);
        let eff_inv_temp = branchless_clamp_f32(phase_inv_temp * inv_edbc_temp, 0.05, 8.0);
        scale_f32_slice_unrolled8(scores, eff_inv_temp);

        Ok((mean_coh, eff_inv_temp, best_id, coherent_count))
    }

    /// Evaluates 256-bit dyadic tree distance $d_{\mathcal{V}}(x, y) \in [0, 1)$, quantizes it
    /// with hysteresis into a discrete QDPT depth level, cascades `trail.current_ptr` via
    /// [`HierarchicalTreeTable::resolve_dyadic_qdpt_and_cascade_ptr`], steps the stateful EDBC
    /// controller, and evolves the cascaded pointer mode while preserving 32-bit `trail.target_anchor_id`.
    ///
    /// Returns `(edbc_state, qdpt_level, dyadic_distance, evolved_ptr)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `qdpt_addr_offsets` is empty or mismatched with
    /// `qdpt_phase_deltas`. Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if `token_z_score` or
    /// `void_score` is non-finite.
    #[inline(always)]
    fn resolve_dyadic_tree_and_step_edbc(
        &mut self,
        current_mask: &gcso_paged_bitmask_t,
        anchor_mask: &gcso_paged_bitmask_t,
        w_key: f32,
        current_level: u32,
        qdpt_hysteresis_band: f32,
        qdpt_addr_offsets: &[u64],
        qdpt_phase_deltas: &[gcso_q7_t],
        token_z_score: f32,
        void_score: f32,
        high_density_rank1_thresh: f32,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(gcso_edbc_state_t, u32, f32, gcso_tagged_ptr_t)> {
        if !branchless_is_finite_f32(void_score) {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }
        let (qdpt_lvl, d_v, cascaded_ptr) = self.resolve_dyadic_qdpt_and_cascade_ptr(
            gcso_tagged_ptr_t(trail.current_ptr),
            current_mask,
            anchor_mask,
            w_key,
            current_level,
            qdpt_hysteresis_band,
            qdpt_addr_offsets,
            qdpt_phase_deltas,
        )?;

        let mut st = gcso_edbc_state_t::default();
        self.eval_stateful(token_z_score, &mut st)?;
        let bif_thresh = branchless_select_f32(
            (st.bifurcation_threshold > 0.0) & branchless_is_finite_f32(st.bifurcation_threshold),
            st.bifurcation_threshold,
            DEFAULT_BIFURCATION_THRESHOLD,
        );
        st.bifurcation_threshold = bif_thresh;

        let dom_lane = branchless_select_u8(
            trail.target_anchor_id != 0,
            (trail.target_anchor_id as u8) & ATTRACTOR_SLOT_ID_MASK,
            cascaded_ptr.target_attractor_slot(),
        );
        let is_repulsion = st.active_branch_mode == EDBC_MODE_PHASE_CONJUGATE_REPULSION;
        let low_entropy_skip_thresh = branchless_relu_f32(bif_thresh * 0.5);
        let evolved_ptr = cascaded_ptr.evolve_mode_stigmergically(
            trail.stigmergic_density,
            st.moving_z_entropy,
            is_repulsion,
            dom_lane,
            low_entropy_skip_thresh,
            high_density_rank1_thresh,
        );
        trail.prev_ptr = trail.current_ptr;
        trail.current_ptr = evolved_ptr.raw();
        let preserve_anchor = (trail.target_anchor_id != 0) & (!is_repulsion);
        trail.cluster_id = branchless_select_u32(
            preserve_anchor,
            trail.target_anchor_id,
            evolved_ptr.target_attractor_slot() as u32,
        );

        Ok((st, qdpt_lvl, d_v, evolved_ptr))
    }

    /// Executes a closed-loop EDBC step driven by a synthesized 128-byte [`GatewayFrame`]
    /// (**2-Origin Composite Manifold Gateway**):
    /// 1. Steps [`EntropyEvaluator::eval_stateful`] (`token_z_score`) and evaluates EDBC's
    ///    Pitchfork/Repulsion branch mode against `void_score` and `frame.dominant_cause_code`.
    /// 2. Ingests `frame.phase_deltas` into `trail` in 8 register-resident `u64` SWAR words
    ///    (`next_w`), applying `frame.qdps_min_step_q7` step filtering and binding 32-bit
    ///    `frame.target_anchor_id` into `trail.target_anchor_id` and `trail.cluster_id`.
    /// 3. Evolves `trail.current_ptr` preserving `frame.mapped_section_offset48` (`48-bit` `.gcso`
    ///    direct RAG/Fact address) and reconciling `ACTION_MODE_LAYER_SKIP` with EDBC's
    ///    entropy state.
    /// 4. Combines `frame.effective_inv_temp` with EDBC's mode-aware temperature $T_{\mathrm{sample}}$
    ///    into $\beta_{\mathrm{eff}} = \mathrm{clamp}(\beta_{\mathrm{gateway}} / T_{\mathrm{sample}}, 0.05, 8.0)$
    ///    and optionally scales `logits_opt` in-place via AVX2 (`scale_f32_slice_unrolled8`).
    /// 5. Produces the unified $\mathcal{O}(1)$ [`CausalProvenanceReport`] (16B `Pod`) and, when
    ///    `crystal_record_opt` is supplied and the trajectory is coherent (`!repulsion_triggered`),
    ///    absorbs the register-resident `next_w` directly via
    ///    [`CrystallizedAttractorRecord::absorb_gateway_frame_with_words`] without reloading
    ///    `trail.accumulated_phase_delta` from memory!
    ///
    /// Returns `(edbc_state, evolved_ptr, causal_report, effective_inv_temp, crystallized_gravity_mass)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `high_density_rank1_thresh` is non-finite.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if `token_z_score`, `void_score`, `alpha`,
    /// `beta <= 0.0`, or `d0 <= 0.0` is invalid.
    #[inline(always)]
    fn step_via_gateway_frame(
        &mut self,
        token_z_score: f32,
        void_score: f32,
        alpha: f32,
        beta: f32,
        d0: f32,
        high_density_rank1_thresh: f32,
        frame: &GatewayFrame,
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
        step_via_gateway_frame_with_words_core(
            self,
            token_z_score,
            void_score,
            alpha,
            beta,
            d0,
            high_density_rank1_thresh,
            frame,
            frame.phase_words_ref(),
            trail,
            logits_opt,
            crystal_record_opt,
        )
    }

    /// Synthesizes a [`GatewayFrame`] and its register-resident 8-word phase profile (`gw_w`)
    /// from any composite [`ManifoldGatewaySignal`] (2/3/4-tuples, slice ensembles, `Option<T>`,
    /// or [`CrystallizedAttractorRecord`]) via [`ManifoldGatewaySignal::synthesize_gateway_frame_with_words`]
    /// and executes [`Self::step_via_gateway_frame`] without reloading `frame.phase_deltas` from memory.
    ///
    /// Returns `(edbc_state, evolved_ptr, causal_report, effective_inv_temp, crystallized_gravity_mass)`.
    ///
    /// # Errors
    /// Propagates any validation or singularity error from [`Self::step_via_gateway_frame`].
    #[inline(always)]
    fn step_via_gateway_signal<G: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        token_z_score: f32,
        void_score: f32,
        alpha: f32,
        beta: f32,
        d0: f32,
        high_density_rank1_thresh: f32,
        gateway_signal: &G,
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
        let (frame, gw_w) = gateway_signal.synthesize_gateway_frame_with_words();
        step_via_gateway_frame_with_words_core(
            self,
            token_z_score,
            void_score,
            alpha,
            beta,
            d0,
            high_density_rank1_thresh,
            &frame,
            &gw_w,
            trail,
            logits_opt,
            crystal_record_opt,
        )
    }

    /// Executes the closed-loop **2-Origin Composite Manifold Gateway $\to$ EDBC Step $\to$
    /// Classical Online Toroidal $k$-Means / DP-Means Macro Clustering** pipeline on
    /// `centroids: &mut [CrystallizedAttractorRecord]` in a single zero-allocation call,
    /// keeping `gw_w` in registers across both EDBC and Toroidal $k$-Means.
    ///
    /// Returns `(edbc_state, evolved_ptr, causal_report, effective_inv_temp, assigned_cluster_idx, assigned_anchor_id, updated_gravity_mass, spawned_new_cluster)`.
    ///
    /// # Errors
    /// Propagates any validation or singularity error from [`Self::step_via_gateway_frame`].
    #[inline(always)]
    fn step_via_gateway_kmeans_64<G: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        token_z_score: f32,
        void_score: f32,
        alpha: f32,
        beta: f32,
        d0: f32,
        high_density_rank1_thresh: f32,
        gateway_signal: &G,
        centroids: &mut [CrystallizedAttractorRecord],
        split_coherence_thresh: f32,
        trail: &mut gcso_pointer_trail_t,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(
        gcso_edbc_state_t,
        gcso_tagged_ptr_t,
        CausalProvenanceReport,
        f32,
        usize,
        u32,
        f32,
        bool,
    )> {
        let (frame, gw_w) = gateway_signal.synthesize_gateway_frame_with_words();
        let (st, evolved_ptr, mut report, eff_inv_temp, _single_mass) =
            step_via_gateway_frame_with_words_core(
                self,
                token_z_score,
                void_score,
                alpha,
                beta,
                d0,
                high_density_rank1_thresh,
                &frame,
                &gw_w,
                trail,
                logits_opt,
                None,
            )?;

        let (cluster_idx, anchor_id, _coh, mass, spawned) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_frame_with_words_64(
                centroids,
                &frame,
                &gw_w,
                split_coherence_thresh,
                Some(trail),
            );
        if !centroids.is_empty()
            && (centroids[cluster_idx].provenance_bits & CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED) != 0
        {
            report.provenance_bits |= CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED;
        }

        Ok((
            st,
            evolved_ptr,
            report,
            eff_inv_temp,
            cluster_idx,
            anchor_id,
            mass,
            spawned,
        ))
    }

    /// Folds a slice of `.gcso`-persisted [`CrystallizedAttractorRecord`]s (`records`) onto `trail`
    /// via **Phase-Proximity Gravity** ([`CrystallizedAttractorRecord::fold_crystallized_ensemble_words_with_init_pull_64`]),
    /// blending directly in 8 register-resident SWAR words when prior pull inertia is present,
    /// binds the dominant 32-bit `anchor_id` into `trail.target_anchor_id` and `trail.cluster_id`,
    /// steps the stateful EDBC controller, and evolves `trail.current_ptr`.
    ///
    /// Returns `(edbc_state, evolved_ptr, total_gravity_pull, mean_coherence, dominant_anchor_id)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `high_density_rank1_thresh` is non-finite.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if `token_z_score` or `void_score` is non-finite.
    #[inline(always)]
    fn step_with_crystallized_ensemble_and_pointer(
        &mut self,
        token_z_score: f32,
        void_score: f32,
        records: &[CrystallizedAttractorRecord],
        high_density_rank1_thresh: f32,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(gcso_edbc_state_t, gcso_tagged_ptr_t, f32, f32, u32)> {
        if !branchless_is_finite_f32(high_density_rank1_thresh) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !branchless_is_finite_f32(void_score) {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        let init_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(trail.attractor_pull_force));
        let (_after_w, raw_total_pull, dom_id, _dom_ripa_rad, mean_coh) =
            CrystallizedAttractorRecord::fold_crystallized_ensemble_words_and_coherence_64(
                records,
                &mut trail.accumulated_phase_delta,
                init_pull,
            );
        let total_pull = branchless_min_f32(raw_total_pull, MAX_ATTRACTOR_PULL_FORCE);
        if raw_total_pull > init_pull + 1.0e-6 {
            if dom_id != 0 {
                trail.target_anchor_id = dom_id;
                trail.cluster_id = dom_id;
            }
            trail.attractor_pull_force = total_pull;
            trail.flags |= TRAIL_FLAG_CRYSTALLIZED_CANDIDATE;
        }

        let mut st = gcso_edbc_state_t::default();
        self.eval_stateful(token_z_score, &mut st)?;
        let bif_thresh = branchless_select_f32(
            (st.bifurcation_threshold > 0.0) & branchless_is_finite_f32(st.bifurcation_threshold),
            st.bifurcation_threshold,
            DEFAULT_BIFURCATION_THRESHOLD,
        );
        st.bifurcation_threshold = bif_thresh;

        let dom_lane = (trail.target_anchor_id as u8) & ATTRACTOR_SLOT_ID_MASK;
        let is_repulsion = st.active_branch_mode == EDBC_MODE_PHASE_CONJUGATE_REPULSION;
        let low_entropy_skip_thresh = branchless_relu_f32(bif_thresh * 0.5);
        let evolved_ptr = gcso_tagged_ptr_t(trail.current_ptr).evolve_mode_stigmergically(
            trail.stigmergic_density,
            st.moving_z_entropy,
            is_repulsion,
            dom_lane,
            low_entropy_skip_thresh,
            high_density_rank1_thresh,
        );
        trail.current_ptr = evolved_ptr.raw();
        let preserve_anchor = (trail.target_anchor_id != 0) & (!is_repulsion);
        trail.cluster_id = branchless_select_u32(
            preserve_anchor,
            trail.target_anchor_id,
            evolved_ptr.target_attractor_slot() as u32,
        );

        Ok((
            st,
            evolved_ptr,
            total_pull,
            mean_coh,
            trail.target_anchor_id,
        ))
    }
}

impl<T> EntropyClosedLoopCoordinator for T where
    T: EdbcBranchController + SheafCohomologyEvaluator + HierarchicalTreeTable + ?Sized
{
}
