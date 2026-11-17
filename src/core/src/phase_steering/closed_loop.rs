// name: src/core/src/phase_steering/closed_loop.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Blanket-implemented [PhaseSteeringClosedLoop] over [PhaseSteering] and [SubHeadRouter].

#[allow(unused_imports)]
use super::engine::*;
use super::*;

// ===================================================================
// 1. Blanket-Implemented Closed-Loop DPSR + Swarm / Unified-Field / Limina / Gateway Trait
// ===================================================================

/// Blanket-implemented closed-loop steering trait uniting [`PhaseSteering`] and [`SubHeadRouter`]
/// with [`StigmergicReconstitutionOps`], Single-Manifold Unified-Field resonance, Bayesian Pointer-Trail
/// selection & Attention score modulation, Orthogonal Procrustes trail feedback, Limina
/// inter-memory bridging, Online Toroidal $k$-Means / DP-Means Coarse-to-Fine routing, and
/// 2-Origin Composite Manifold Gateway execution.
///
/// Automatically implemented for all `T: PhaseSteering + SubHeadRouter + ?Sized` (including
/// [`PhaseSteeringEngine`], [`crate::traits::ActionHub`], [`crate::traits::HotPathObserver`],
/// [`crate::traits::ColdPathObserver`], and [`crate::traits::FullStackGcsoEngine`]).
pub trait PhaseSteeringClosedLoop: PhaseSteering + SubHeadRouter {
    /// Executes closed-loop 8-Cell SWAR Micro-Swarm consensus on `trail` against `target_phases`
    /// via [`StigmergicReconstitutionOps::try_resolve_action_in_swarm_to_words`] and steers `query_tensor`
    /// (and optionally `key_tensor`) directly from the 8 returned `u64` SWAR registers via
    /// `steer_from_swar_words_core_safe` without reloading `trail.accumulated_phase_delta`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_heads > MAX_ATTENTION_HEADS`,
    /// `target_phases.len() != MAX_ATTENTION_HEADS`, `min_step_rad < 0.0`, `max_rad <= 0.0`,
    /// or tensor bounds are invalid.
    #[inline(always)]
    fn resolve_and_apply_swarm_phase_steering_safe(
        &self,
        trail: &mut gcso_pointer_trail_t,
        target_phases: &AlignedSlice16<'_, gcso_q7_t>,
        step_gain_u7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        unresolved_damping_u7: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        min_step_rad: f32,
        max_rad: f32,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<StigmergicReconstitutionReport> {
        if min_step_rad < 0.0 || !branchless_is_finite_f32(min_step_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads) =
            validate_closed_loop_tensors(self, max_rad, query_tensor, key_tensor.as_deref())?;
        let tgt_arr = target_phases.as_array::<MAX_ATTENTION_HEADS>()?;
        let qdps_min_step_q7 = branchless_rad_to_qdps_step_u8(min_step_rad);
        let action_ptr = trail.current_tagged_ptr();

        let (report, swar_words) = trail.try_resolve_action_in_swarm_to_words(
            action_ptr,
            tgt_arr,
            0xFF,
            step_gain_u7,
            qdps_min_step_q7,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            bloom_filter,
            repulsion_threshold,
            repulsion_scale_u7,
        );

        finish_constant_time_swarm_qdps_and_steer(
            report.evolved_ptr,
            report.quorum_mask,
            &swar_words,
            qdps_min_step_q7,
            unresolved_damping_u7,
            head_dim,
            num_heads,
            max_rad,
            trail,
            query_tensor,
            key_tensor,
        );
        Ok(report)
    }

    /// Resolves `trail` in the 8-Cell SWAR Micro-Swarm directly on the **Single Unified Attractor
    /// Phase Manifold** (`anchor_phases`, `anchor_base_energies`) without requiring a 512-byte
    /// 8-slot phase bank or $8 \times 8$ coupling matrix.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `anchor_phases.len() != anchor_base_energies.len()`,
    /// `min_step_rad < 0.0`, `max_rad <= 0.0`, or tensor dimensions are invalid.
    #[inline(always)]
    fn resolve_unified_field_swarm_and_steer_safe(
        &self,
        trail: &mut gcso_pointer_trail_t,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        active_paged_mask: Option<&gcso_paged_bitmask_t>,
        step_gain_u7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        unresolved_damping_u7: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        min_step_rad: f32,
        max_rad: f32,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<StigmergicReconstitutionReport> {
        if anchor_phases.len() != anchor_base_energies.len()
            || min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads) =
            validate_closed_loop_tensors(self, max_rad, query_tensor, key_tensor.as_deref())?;

        let active_cell_mask = match active_paged_mask {
            Some(m) => {
                let mask8 = m.subswarm_activity_mask_8();
                branchless_select_u8(mask8 != 0, mask8, 0xFF)
            }
            None => 0xFF,
        };
        let qdps_min_step_q7 = branchless_rad_to_qdps_step_u8(min_step_rad);
        let action_ptr = trail.current_tagged_ptr();

        let (report, swar_words) = trail.resolve_unified_field_in_swarm_to_words(
            action_ptr,
            anchor_phases,
            anchor_base_energies,
            active_cell_mask,
            step_gain_u7,
            qdps_min_step_q7,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            bloom_filter,
            repulsion_threshold,
            repulsion_scale_u7,
        );

        finish_constant_time_swarm_qdps_and_steer(
            report.evolved_ptr,
            report.quorum_mask,
            &swar_words,
            qdps_min_step_q7,
            unresolved_damping_u7,
            head_dim,
            num_heads,
            max_rad,
            trail,
            query_tensor,
            key_tensor,
        );
        Ok(report)
    }

    /// Resolves `trail` in the 8-Cell Micro-Swarm against the optional 8-Lane Working Phase Bank
    /// (`slot_phase_bank`, `>= 512` bytes), recruits coupled neighbor lanes via
    /// [`StigmergicReconstitutionOps::resolve_slot_bank_in_swarm_to_words`] on partial quorum,
    /// and steers `query_tensor` (and optionally `key_tensor`) directly from the register-resident
    /// `swar_words` with geodesic RIPA soft-clamping.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] or [`crate::abi::GCSO_ERROR_MISALIGNED_POINTER`]
    /// if `slot_phase_bank.len() < 512`, `min_step_rad < 0.0`, `max_rad <= 0.0`, or tensor bounds fail.
    #[inline(always)]
    fn resolve_slot_bank_swarm_and_steer_safe(
        &self,
        trail: &mut gcso_pointer_trail_t,
        slot_phase_bank: &AlignedSlice16<'_, gcso_q7_t>,
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        active_paged_mask: Option<&gcso_paged_bitmask_t>,
        step_gain_u7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        unresolved_damping_u7: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        min_step_rad: f32,
        max_rad: f32,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<StigmergicReconstitutionReport> {
        if min_step_rad < 0.0 || !branchless_is_finite_f32(min_step_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads) =
            validate_closed_loop_tensors(self, max_rad, query_tensor, key_tensor.as_deref())?;
        let bank_2d =
            slot_phase_bank.as_slot_bank::<MAX_ATTRACTOR_DOMAINS, MAX_ATTENTION_HEADS>()?;

        let active_cell_mask = match active_paged_mask {
            Some(m) => {
                let mask8 = m.subswarm_activity_mask_8();
                branchless_select_u8(mask8 != 0, mask8, 0xFF)
            }
            None => 0xFF,
        };
        let qdps_min_step_q7 = branchless_rad_to_qdps_step_u8(min_step_rad);
        let action_ptr = trail.current_tagged_ptr();

        let (report, swar_words) = trail.resolve_slot_bank_in_swarm_to_words(
            action_ptr,
            bank_2d,
            slot_weights,
            coupling_matrix_8x8,
            active_cell_mask,
            step_gain_u7,
            qdps_min_step_q7,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            bloom_filter,
            repulsion_threshold,
            repulsion_scale_u7,
        );

        finish_constant_time_swarm_qdps_and_steer(
            report.evolved_ptr,
            report.quorum_mask,
            &swar_words,
            qdps_min_step_q7,
            unresolved_damping_u7,
            head_dim,
            num_heads,
            max_rad,
            trail,
            query_tensor,
            key_tensor,
        );
        Ok(report)
    }

    /// Directly modulates pre-Softmax candidate memory Attention scores (`attention_scores[..M]`)
    /// in-place in zero-copy, zero-allocation ($C_{\mathrm{alloc}} = 0$) constant time per
    /// candidate directly from `candidate_trails: &[gcso_pointer_trail_t]` using
    /// [`StigmergicTrailOps::modulate_candidate_trail_scores_in_place`].
    ///
    /// Returns `(mean_phase_coherence, best_post_modulated_candidate_id, coherent_candidate_count)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `attention_scores` or `candidate_trails` is
    /// empty, `head_dim()` is zero or odd, or any gain/threshold parameter is non-finite.
    #[inline(always)]
    fn modulate_candidate_trail_attention_scores_zero_copy(
        &self,
        attention_scores: &mut AlignedSliceMut32<'_, f32>,
        src_trail: &gcso_pointer_trail_t,
        candidate_trails: &[gcso_pointer_trail_t],
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        contradiction_threshold: f32,
        repulsion_penalty: f32,
    ) -> GcsoResult<(f32, u32, u32)> {
        let head_dim = self.head_dim();
        let count = attention_scores.len().min(candidate_trails.len());
        if count == 0
            || head_dim == 0
            || (head_dim & 1) != 0
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(contradiction_threshold)
            || !branchless_is_finite_f32(repulsion_penalty)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        Ok(src_trail.modulate_candidate_trail_scores_in_place(
            attention_scores.as_mut_slice(),
            candidate_trails,
            head_dim,
            phase_interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
            contradiction_threshold,
            repulsion_penalty,
        ))
    }

    /// Combines **Sequential Conditional Transition Probability** ($P(M_1, \dots, M_T) = \prod_t P(M_t \mid M_{<t})$
    /// $\times$ Attention phase interference) and **Indirect Pointer-Chain Graph Routing** in `ActionHub`
    /// directly on any `T: PhaseSteering + SubHeadRouter` in zero-copy, zero-allocation ($C_{\mathrm{alloc}} = 0$) mode.
    ///
    /// Returns `(selected_trail_id, bayesian_score, transition_coherence, transition_log_prob, found_coherent_transition)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `candidate_trails` is empty, parameters are
    /// non-finite, or tensor dimensions are invalid.
    #[inline(always)]
    fn select_rewire_and_steer_bayesian_pointer_trail_safe(
        &self,
        src_trail: &mut gcso_pointer_trail_t,
        candidate_trails: &mut [gcso_pointer_trail_t],
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        repulsion_penalty: f32,
        interpolation_u7: u8,
        contradiction_threshold: f32,
        min_step_rad: f32,
        max_rad: f32,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(u32, f32, f32, f32, bool)> {
        if candidate_trails.is_empty()
            || min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(repulsion_penalty)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads) =
            validate_closed_loop_tensors(self, max_rad, query_tensor, key_tensor.as_deref())?;
        let self_w = *trail_phase_words_ref(src_trail);
        Ok(route_macro_aligned_bayesian_trails_and_steer_core(
            self_w,
            &[0u64; NUM_STIGMERGIC_GRANULES],
            false,
            0,
            0,
            src_trail,
            candidate_trails,
            head_dim,
            num_heads,
            phase_interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
            repulsion_penalty,
            interpolation_u7,
            contradiction_threshold,
            min_step_rad,
            max_rad,
            attention_scores,
            query_tensor,
            key_tensor,
        ))
    }

    /// Combines **Order-Invariant Single-Manifold Unified-Field Resonance**, **RIPA Q/K Phase
    /// Steering**, optional **Register-Resident $SO(2)$-Fused sub-head router Sub-Head Pseudo-MoE**
    /// (`dispatch_fused_ripa_and_pspm_words_core`, eliminating the 64-byte `phase_buf`
    /// stack spill), and optional **Multiplicative Inverse-Temperature Logit Scaling**
    /// ($\beta_{\mathrm{eff}} = \mathrm{clamp}(1.0 + \gamma \mathcal{C}_{\mathrm{phase}}, 0.25, 4.0)$)
    /// without requiring a 512-byte 8-slot phase bank.
    ///
    /// Returns `(final_pull_force, mean_coherence, applied_inv_temp_scale, dominant_anchor_id)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `anchor_phases.len() != anchor_base_energies.len()`,
    /// `min_step_rad < 0.0`, `max_rad <= 0.0`, or tensor/sub-head router parameters are invalid.
    #[inline]
    fn steer_unified_field_pspm_and_inv_temperature_safe(
        &self,
        trail: &mut gcso_pointer_trail_t,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
        subhead_config_and_explore_pulse: Option<(&gcso_subhead_config_t, gcso_q7_t)>,
        logits_and_inv_temp_gain: Option<(&mut AlignedSliceMut32<'_, f32>, f32)>,
        min_step_rad: f32,
        max_rad: f32,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        mut key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(f32, f32, f32, u32)> {
        if anchor_phases.len() != anchor_base_energies.len()
            || min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if let Some((_, strength)) = repulsion_phases_and_strength {
            if !branchless_is_finite_f32(strength) {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
        }
        let (head_dim, num_heads) =
            validate_closed_loop_tensors(self, max_rad, query_tensor, key_tensor.as_deref())?;

        let init_pull = trail.attractor_pull_force;
        let (final_pull, mean_coh, active_cnt, raw_w) = trail
            .apply_unified_field_anchor_slice_resonance_to_words(
                anchor_phases,
                anchor_base_energies,
                repulsion_phases_and_strength,
            );

        let need_dom_w =
            logits_and_inv_temp_gain.is_some() || subhead_config_and_explore_pulse.is_some();
        let raw_eq_dom = (!need_dom_w)
            | ((active_cnt <= 1)
                & ((anchor_phases.len() <= 1) | (init_pull <= DEFAULT_ENTROPY_EPS))
                & repulsion_phases_and_strength.is_none());
        let dom_w = if raw_eq_dom || anchor_phases.is_empty() {
            raw_w
        } else {
            let dom_idx_usize = (trail.target_anchor_id as usize).min(anchor_phases.len() - 1);
            load_q7_words_64(&anchor_phases[dom_idx_usize])
        };
        let dom_eq_raw =
            raw_eq_dom || anchor_phases.len() <= 1 || swar_words_equal_64(&raw_w, &dom_w);

        let mut inv_temp = 1.0_f32;
        if let Some((logits, temp_gain)) = logits_and_inv_temp_gain {
            if logits.is_empty() || !branchless_is_finite_f32(temp_gain) {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            let coh_dom = if anchor_phases.len() <= 1 {
                mean_coh
            } else if dom_eq_raw {
                1.0_f32
            } else {
                let raw_uni = is_word_array_uniform_64(&raw_w);
                coherence_from_word_arrays_with_both_uniform_flags_64(
                    &raw_w,
                    raw_uni,
                    &dom_w,
                    raw_uni && is_word_array_uniform_64(&dom_w),
                )
            };
            inv_temp = branchless_phase_coherence_inv_temp_scale(coh_dom, temp_gain);
            scale_f32_slice_unrolled8(logits.as_mut_slice(), inv_temp);
        }

        let step_u8 = branchless_rad_to_qdps_step_u8(min_step_rad);
        let filt_w = swar_q7_qdps_filter_words_64(&raw_w, step_u8);
        let filt_nonzero = swar_words_nonzero_64(&filt_w);

        if let Some((subhead_cfg, explore_pulse_q7)) = subhead_config_and_explore_pulse {
            let total_pspm_heads = (subhead_cfg.num_fact_heads as usize)
                .checked_add(subhead_cfg.num_logic_heads as usize)
                .and_then(|v| v.checked_add(subhead_cfg.num_explore_heads as usize))
                .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
            if total_pspm_heads > num_heads {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }

            let filt_eq_dom = if step_u8 <= 1 {
                dom_eq_raw
            } else {
                dom_eq_raw || swar_words_equal_64(&filt_w, &dom_w)
            };
            let is_default_ripa_pspm = filt_eq_dom
                & (explore_pulse_q7 == 0)
                & (total_pspm_heads == num_heads)
                & (subhead_cfg.fact_phase_gain.to_bits() == 1.0_f32.to_bits())
                & (subhead_cfg.logic_phase_gain.to_bits() == 1.0_f32.to_bits())
                & (subhead_cfg.explore_phase_gain.to_bits() == 1.0_f32.to_bits())
                & (max_rad.to_bits() == DEFAULT_RIPA_CLAMP_MAX_RAD.to_bits())
                & (num_heads > 0)
                & (num_heads <= MAX_ATTENTION_HEADS);
            if is_default_ripa_pspm {
                let rot_dim = ripa_upper_rot_dim(head_dim);
                if let Some(k) = key_tensor.as_deref_mut() {
                    if filt_nonzero {
                        crate::traits::rope::rotate_fused_self_ripa_pspm_q_and_ripa_k_default_words(
                            query_tensor.as_mut_slice(),
                            k.as_mut_slice(),
                            &filt_w,
                            num_heads,
                            head_dim,
                            rot_dim,
                        );
                    }
                } else if filt_nonzero {
                    crate::traits::rope::rotate_fused_self_ripa_and_pspm_default_words(
                        query_tensor.as_mut_slice(),
                        &filt_w,
                        num_heads,
                        head_dim,
                        rot_dim,
                        true,
                    );
                }
            } else {
                let (fact_w_ref, logic_w_ref, explore_w_ref) = if filt_eq_dom {
                    (&filt_w, &filt_w, &filt_w)
                } else {
                    (&dom_w, &filt_w, &filt_w)
                };
                dispatch_fused_ripa_and_pspm_words_core(
                    query_tensor,
                    &filt_w,
                    max_rad,
                    subhead_cfg,
                    fact_w_ref,
                    logic_w_ref,
                    explore_w_ref,
                    max_rad,
                    explore_pulse_q7,
                    head_dim,
                    true,
                )?;
                if let Some(k) = key_tensor.as_deref_mut() {
                    if filt_nonzero {
                        steer_from_swar_words_core_safe(
                            &filt_w, 0xFF, 128, head_dim, num_heads, false, max_rad, k, None,
                        );
                    }
                }
            }
            if total_pspm_heads < num_heads && filt_nonzero {
                let tail_elems = (num_heads - total_pspm_heads) * head_dim;
                let q_tail = &mut query_tensor.as_mut_slice()
                    [total_pspm_heads * head_dim..total_pspm_heads * head_dim + tail_elems];
                let mut tail_words = [0u64; NUM_STIGMERGIC_GRANULES];
                let mut h = total_pspm_heads;
                while h < num_heads {
                    let th = h - total_pspm_heads;
                    let byte_val = (filt_w[h >> 3] >> ((h & 7) << 3)) & 0xFF;
                    tail_words[th >> 3] |= byte_val << ((th & 7) << 3);
                    h += 1;
                }
                if swar_words_nonzero_64(&tail_words) {
                    // SAFETY: `total_pspm_heads * head_dim` is a multiple of `head_dim * 4` bytes;
                    // for residual heads we rotate each upper `d_head / 4` slice directly.
                    let mut tail_aligned = unsafe { AlignedSliceMut32::new_unchecked(q_tail) };
                    steer_from_swar_words_core_safe(
                        &tail_words,
                        0xFF,
                        128,
                        head_dim,
                        num_heads - total_pspm_heads,
                        false,
                        max_rad,
                        &mut tail_aligned,
                        None,
                    );
                }
            }
        } else if filt_nonzero {
            steer_from_swar_words_core_safe(
                &filt_w,
                0xFF,
                128,
                head_dim,
                num_heads,
                false,
                max_rad,
                query_tensor,
                key_tensor,
            );
        }

        Ok((final_pull, mean_coh, inv_temp, trail.target_anchor_id))
    }

    /// Computes the closed-form Orthogonal Procrustes $SO(2)$ phase shift between `query_tensor`
    /// and `target_key_tensor` via [`PhaseSteering::compute_procrustes_phase_delta`], applies
    /// constant-time 8-lane SWAR gain scaling (`swar_q7_scale_u7_x8`) and QDPS step filtering
    /// (`swar_q7_qdps_filter_x8`), accumulates circularly into `trail.accumulated_phase_delta`
    /// (`swar_q7_circular_add_x8`), and steers `query_tensor` in-place via
    /// `steer_from_swar_words_core_safe`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_heads > MAX_ATTENTION_HEADS`,
    /// `min_step_rad < 0.0`, `max_rad <= 0.0`, or tensor lengths are insufficient.
    #[inline]
    fn steer_trail_with_procrustes_feedback(
        &self,
        trail: &mut gcso_pointer_trail_t,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        target_key_tensor: &AlignedSlice32<'_, f32>,
        feedback_gain_u7: u8,
        min_step_rad: f32,
        max_rad: f32,
    ) -> GcsoResult<AlignedBuffer16<gcso_q7_t, MAX_ATTENTION_HEADS>> {
        let step_bits = min_step_rad.to_bits();
        let step_abs_bits = step_bits & 0x7FFF_FFFF;
        if ((step_bits >> 31) != 0 && step_abs_bits != 0)
            || (step_bits & 0x7F80_0000) == 0x7F80_0000
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads) =
            validate_closed_loop_tensors(self, max_rad, query_tensor, None)?;
        let expected_len = head_dim * num_heads;
        if target_key_tensor.len() < expected_len {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let clean_gain = branchless_select_u8(feedback_gain_u7 > 128, 128, feedback_gain_u7);
        if clean_gain == 0 {
            let nw = *trail_phase_words_ref(trail);
            if swar_words_nonzero_64(&nw) {
                steer_from_swar_words_core_safe(
                    &nw,
                    0xFF,
                    128,
                    head_dim,
                    num_heads,
                    true,
                    max_rad,
                    query_tensor,
                    None,
                );
            }
            return Ok(AlignedBuffer16::new([0i8; MAX_ATTENTION_HEADS]));
        }

        let gain_u64 = u64::from(clean_gain);
        let gain_bias = (128u64.wrapping_sub(gain_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let step_u8 = if step_abs_bits <= Q7_PHASE_SCALE_RAD.to_bits() {
            (step_abs_bits != 0) as u8
        } else {
            branchless_rad_to_qdps_step_u8(min_step_rad)
        };
        let active_cells = num_heads
            .div_ceil(HEADS_PER_STIGMERGIC_GRANULE)
            .min(NUM_STIGMERGIC_GRANULES);
        let is_unit_gain = clean_gain == 128;
        let is_bypass_step = step_u8 <= 1;

        // Two-stage DSB-locked execution:
        // Stage 1 (Read-Only SIMD Procrustes Extraction): Runs `procrustes_q7_words_64_simd` across
        // all active 8-head cells while keeping L1D cache ports 100% in read mode and preserving
        // AVX2 `atan2` polynomial constants in YMM registers without store-turnaround stalls.
        // Stage 2 (Register-Resident SWAR Trail Accum + Coalesced Geodesic RIPA Steering): Updates
        // `trail.accumulated_phase_delta` in registers and invokes `steer_from_swar_words_core_safe`
        // while the upper `d_head / 4` channels of `query_tensor` remain 100% hot in L1D cache.
        let mut raw_words = procrustes_q7_words_64_simd(
            query_tensor.as_slice(),
            target_key_tensor.as_slice(),
            head_dim,
            num_heads,
        );
        let mut steer_words = *trail_phase_words_ref(trail);
        let mut any_nonzero = 0u64;
        if is_unit_gain & is_bypass_step {
            let mut c = 0usize;
            while c < active_cells {
                let filt_w = raw_words[c];
                let nw = swar_q7_circular_add_x8(steer_words[c], filt_w);
                steer_words[c] = nw;
                any_nonzero |= nw;
                c += 1;
            }
        } else if is_bypass_step {
            let mut c = 0usize;
            while c < active_cells {
                let filt_w = swar_q7_scale_le128_with_bias_x8(raw_words[c], gain_u64, gain_bias);
                raw_words[c] = filt_w;
                let nw = swar_q7_circular_add_x8(steer_words[c], filt_w);
                steer_words[c] = nw;
                any_nonzero |= nw;
                c += 1;
            }
        } else if is_unit_gain {
            let mut c = 0usize;
            while c < active_cells {
                let filt_w = swar_q7_qdps_filter_x8(raw_words[c], step_u8);
                raw_words[c] = filt_w;
                let nw = swar_q7_circular_add_x8(steer_words[c], filt_w);
                steer_words[c] = nw;
                any_nonzero |= nw;
                c += 1;
            }
        } else {
            let mut c = 0usize;
            while c < active_cells {
                let scaled_w = swar_q7_scale_le128_with_bias_x8(raw_words[c], gain_u64, gain_bias);
                let filt_w = swar_q7_qdps_filter_x8(scaled_w, step_u8);
                raw_words[c] = filt_w;
                let nw = swar_q7_circular_add_x8(steer_words[c], filt_w);
                steer_words[c] = nw;
                any_nonzero |= nw;
                c += 1;
            }
        }
        *trail_phase_words_mut(trail) = steer_words;
        if any_nonzero != 0 {
            steer_from_swar_words_core_safe(
                &steer_words,
                0xFF,
                128,
                head_dim,
                num_heads,
                true,
                max_rad,
                query_tensor,
                None,
            );
        }
        Ok(AlignedBuffer16::new(words_to_q7_array_64(&raw_words)))
    }

    /// Bridges the inter-memory **Limina** transition from `prev_memory_trail` ($M_{t-1}$) to
    /// `curr_trail` ($M_t$) via [`StigmergicTrailOps::bridge_liminal_hallucinated_trail`],
    /// filters `curr_trail.accumulated_phase_delta` in registers via `filter_qdps_and_steer_from_phases_64`,
    /// steers `query_tensor` (and optional `key_tensor`) with RIPA soft-clamping, and optionally
    /// modulates candidate memory pre-Softmax Attention scores (`attention_scores`) by
    /// Attention $\times$ DPSR phase interference.
    ///
    /// Returns `(transition_coherence, transition_log_prob, contradiction_repulsed)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `contradiction_threshold`, `min_step_rad`, or
    /// `max_rad` is invalid, or if tensor slice dimensions are insufficient.
    #[inline]
    fn bridge_liminal_trail_and_steer_qk_safe(
        &self,
        prev_memory_trail: &gcso_pointer_trail_t,
        curr_trail: &mut gcso_pointer_trail_t,
        interpolation_u7: u8,
        contradiction_threshold: f32,
        min_step_rad: f32,
        max_rad: f32,
        attention_interference_gain: f32,
        memory_anchor_phases: Option<&[[gcso_q7_t; MAX_ATTENTION_HEADS]]>,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(f32, f32, bool)> {
        if min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads) =
            validate_closed_loop_tensors(self, max_rad, query_tensor, key_tensor.as_deref())?;

        let safe_thresh =
            branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
        let t_s = branchless_select_u8(interpolation_u7 > 128, 128, interpolation_u7) as u64;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let (coh, repulsed, out_w, log_prob, vo0, vo1) = unsafe {
            use core::arch::x86_64::{
                _mm256_max_epi8, _mm256_or_si256, _mm256_set1_epi8, _mm256_setzero_si256,
                _mm256_sub_epi8, _mm256_testz_si256,
            };
            let p_words = *trail_phase_words_ref(prev_memory_trail);
            let s_words = *trail_phase_words_ref(curr_trail);
            let p_uni = is_word_array_uniform_64(&p_words);
            let s_uni = is_word_array_uniform_64(&s_words);
            let (vp0, vp1) =
                crate::traits::branchless::avx2_words64_to_vec64_with_uniform(&p_words, p_uni);
            let (vs0, vs1) =
                crate::traits::branchless::avx2_words64_to_vec64_with_uniform(&s_words, s_uni);
            let vd0 = _mm256_sub_epi8(vs0, vp0);
            let vd1 = _mm256_sub_epi8(vs1, vp1);
            let vd_or = _mm256_or_si256(vd0, vd1);
            if _mm256_testz_si256(vd_or, vd_or) != 0 {
                (1.0_f32, false, s_words, 0.0_f32, vs0, vs1)
            } else {
                let c = crate::traits::branchless::avx2_coherence_from_diff_vec64(vd0, vd1);
                let rep = c < -safe_thresh;
                let (v_out0, v_out1, store_out) = if rep {
                    let zero = _mm256_setzero_si256();
                    let min_m127 = _mm256_set1_epi8(-127);
                    let r0 = _mm256_sub_epi8(zero, _mm256_max_epi8(vs0, min_m127));
                    let r1 = _mm256_sub_epi8(zero, _mm256_max_epi8(vs1, min_m127));
                    (r0, r1, true)
                } else if t_s >= 128 {
                    (vs0, vs1, false)
                } else if t_s == 0 {
                    (vp0, vp1, true)
                } else {
                    let (r0, r1) = crate::traits::branchless::avx2_slerp_from_diff_vec64(
                        vp0, vp1, vd0, vd1, t_s,
                    );
                    (r0, r1, true)
                };
                let ow = crate::traits::branchless::avx2_vec64_to_words64(v_out0, v_out1);
                if store_out {
                    *trail_phase_words_mut(curr_trail) = ow;
                }
                let lp = branchless_fast_transition_log_prob(c);
                (c, rep, ow, lp, v_out0, v_out1)
            }
        };

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        let (coh, repulsed, out_w, log_prob) = {
            use crate::traits::branchless::coherence_from_diff_words_64;
            use crate::traits::primitives::{branchless_cos_q7, swar_q7_circular_diff_x8};
            let p = *trail_phase_words_ref(prev_memory_trail);
            let s = *trail_phase_words_ref(curr_trail);
            let bias_restore = (128u64.wrapping_sub(t_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
            if (s[0] == p[0]) && swar_words_equal_64(&s, &p) {
                (1.0_f32, false, s, 0.0_f32)
            } else {
                let d0 = swar_q7_circular_diff_x8(s[0], p[0]);
                let d1 = swar_q7_circular_diff_x8(s[1], p[1]);
                let d2 = swar_q7_circular_diff_x8(s[2], p[2]);
                let d3 = swar_q7_circular_diff_x8(s[3], p[3]);
                let d4 = swar_q7_circular_diff_x8(s[4], p[4]);
                let d5 = swar_q7_circular_diff_x8(s[5], p[5]);
                let d6 = swar_q7_circular_diff_x8(s[6], p[6]);
                let d7 = swar_q7_circular_diff_x8(s[7], p[7]);
                let diff_is_uniform = (d0 == d1)
                    && ((((d0 ^ d2) | (d0 ^ d3)) | ((d0 ^ d4) | (d0 ^ d5)))
                        | ((d0 ^ d6) | (d0 ^ d7)))
                        == 0;
                let c = if diff_is_uniform && (d0 == d0.rotate_right(8)) {
                    branchless_cos_q7(d0 as gcso_q7_t)
                } else {
                    coherence_from_diff_words_64(d0, d1, d2, d3, d4, d5, d6, d7)
                };
                let rep = c < -safe_thresh;
                let ow = if rep {
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
                } else if t_s >= 128 {
                    s
                } else if t_s == 0 {
                    p
                } else if diff_is_uniform {
                    let st0 = swar_q7_scale_le128_with_bias_x8(d0, t_s, bias_restore);
                    [
                        swar_q7_circular_add_x8(p[0], st0),
                        swar_q7_circular_add_x8(p[1], st0),
                        swar_q7_circular_add_x8(p[2], st0),
                        swar_q7_circular_add_x8(p[3], st0),
                        swar_q7_circular_add_x8(p[4], st0),
                        swar_q7_circular_add_x8(p[5], st0),
                        swar_q7_circular_add_x8(p[6], st0),
                        swar_q7_circular_add_x8(p[7], st0),
                    ]
                } else {
                    [
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
                    ]
                };
                let lp = branchless_fast_transition_log_prob(c);
                *trail_phase_words_mut(curr_trail) = ow;
                (c, rep, ow, lp)
            }
        };

        curr_trail.prev_ptr = prev_memory_trail.current_ptr;
        curr_trail.step_count = prev_memory_trail.step_count.wrapping_add(1);
        let inherit_anchor = (curr_trail.target_anchor_id == 0) & (!repulsed);
        curr_trail.target_anchor_id = branchless_select_u32(
            inherit_anchor,
            prev_memory_trail.target_anchor_id,
            curr_trail.target_anchor_id,
        );
        curr_trail.cluster_id = branchless_select_u32(
            inherit_anchor & (prev_memory_trail.target_anchor_id != 0),
            prev_memory_trail.target_anchor_id,
            curr_trail.cluster_id,
        );
        let cur_ptr = gcso_tagged_ptr_t(curr_trail.current_ptr);
        let rep_ptr = cur_ptr
            .with_action_and_slot(
                ATTRACTOR_SLOT_REPULSION,
                cur_ptr.action_mode_flags() | ACTION_PRESET_REPULSION_ESCAPE,
            )
            .conjugate_phase();
        curr_trail.current_ptr = branchless_select_u64(repulsed, rep_ptr.raw(), cur_ptr.raw());

        let cur_rho =
            branchless_flush_subnormal_f32(branchless_relu_f32(curr_trail.stigmergic_density));
        let pos_rho = cur_rho + 0.25 * (1.0 + branchless_relu_f32(coh));
        let neg_rho = cur_rho * 0.5;
        curr_trail.stigmergic_density = branchless_min_f32(
            branchless_flush_subnormal_f32(branchless_select_f32(repulsed, neg_rho, pos_rho)),
            MAX_STIGMERGIC_DENSITY,
        );

        let prev_pull = branchless_flush_subnormal_f32(branchless_relu_f32(
            prev_memory_trail.attractor_pull_force,
        ));
        let cur_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(curr_trail.attractor_pull_force));
        let bridged_pull = cur_pull + 0.5 * prev_pull * branchless_clamp_f32(1.0 + coh, 0.25, 2.0);
        curr_trail.attractor_pull_force = branchless_min_f32(
            branchless_flush_subnormal_f32(branchless_select_f32(
                repulsed,
                cur_pull * 0.5,
                bridged_pull,
            )),
            MAX_ATTRACTOR_PULL_FORCE,
        );

        let is_resonant = (!repulsed) & (coh > 0.25);
        curr_trail.flags |= branchless_select_u32(is_resonant, TRAIL_FLAG_SYNERGY_RESONANCE, 0)
            | branchless_select_u32(
                repulsed,
                TRAIL_FLAG_REPULSION_ACTIVE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE,
                0,
            );

        if let (Some(scores), Some(anchors)) = (attention_scores, memory_anchor_phases) {
            if !anchors.is_empty() && !scores.is_empty() {
                if !branchless_is_finite_f32(attention_interference_gain) {
                    return Err(GCSO_ERROR_INVALID_ARGUMENT);
                }
                let count = scores.len().min(anchors.len());
                let safe_dim = branchless_select_u32(head_dim > 0, head_dim as u32, 128);
                let rsqrt_d = if safe_dim <= 256 {
                    branchless_popcount_rsqrt_u32(safe_dim)
                } else {
                    branchless_rsqrt_f32(safe_dim as f32)
                };
                let eff_scale = branchless_flush_subnormal_f32(
                    branchless_relu_f32(attention_interference_gain) * rsqrt_d,
                );
                let score_slice = &mut scores.as_mut_slice()[..count];
                #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                let out_is_uniform = is_word_array_uniform_64(&out_w);
                let mut i = 0usize;
                while i < count {
                    let c_j = {
                        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                        {
                            // SAFETY: `anchors[i]` is 64 bytes (`2 * __m256i`) and `vo0, vo1` hold `out_w`.
                            unsafe {
                                use core::arch::x86_64::{
                                    __m256i, _mm256_loadu_si256, _mm256_sub_epi8,
                                };
                                let ap = anchors[i].as_ptr().cast::<__m256i>();
                                let va0 = _mm256_loadu_si256(ap);
                                let va1 = _mm256_loadu_si256(ap.add(1));
                                crate::traits::branchless::avx2_coherence_from_diff_vec64(
                                    _mm256_sub_epi8(vo0, va0),
                                    _mm256_sub_epi8(vo1, va1),
                                )
                            }
                        }
                        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                        {
                            let anc_w = load_q7_words_64(&anchors[i]);
                            coherence_from_word_arrays_with_both_uniform_flags_64(
                                &out_w,
                                out_is_uniform,
                                &anc_w,
                                out_is_uniform && is_word_array_uniform_64(&anc_w),
                            )
                        }
                    };
                    score_slice[i] =
                        branchless_flush_subnormal_f32(score_slice[i] + eff_scale * c_j);
                    i += 1;
                }
            }
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            let step_u8 = branchless_rad_to_qdps_step_u8(min_step_rad);
            let steer_w = if step_u8 > 1 {
                // SAFETY: `target_feature = "avx2"` is enabled; `vo0, vo1` are already in YMM registers.
                unsafe {
                    let (vf0, vf1) =
                        crate::traits::branchless::avx2_qdps_filter_vec64(vo0, vo1, step_u8);
                    crate::traits::branchless::avx2_vec64_to_words64(vf0, vf1)
                }
            } else {
                out_w
            };
            filter_qdps_and_steer_from_words_64(
                &steer_w,
                0.0,
                max_rad,
                head_dim,
                num_heads,
                query_tensor,
                key_tensor,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        filter_qdps_and_steer_from_words_64(
            &out_w,
            min_step_rad,
            max_rad,
            head_dim,
            num_heads,
            query_tensor,
            key_tensor,
        );
        Ok((coh, log_prob, repulsed))
    }

    /// Synthesizes any [`ManifoldGatewaySignal`] (such as a 2/3/4-tuple of
    /// `(SystemConstraintSignal, ObjectiveFactSignal, SubjectivePersonaSignal, ContradictionGuardSignal)`,
    /// a [`GatewayFrame`], or a [`CrystallizedAttractorRecord`]), folds its resonant 64-head phase
    /// and 48-bit mapped-storage tagged pointer into `trail` in 8 register-resident `u64` SWAR words,
    /// executes $SO(2)$ group-closure fused RIPA + sub-head router sub-head routing on `query_tensor` when
    /// `ACTION_MODE_SUBHEAD_ROUTE` is active (or standard RIPA steering via `steer_from_swar_words_core_safe`),
    /// and optionally scales `logits_opt` by `frame.effective_inv_temp`.
    ///
    /// Returns `(GatewayFrame, CausalProvenanceReport)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if tensor dimensions or `GCSO_RESTRICT` non-aliasing
    /// invariants are violated.
    #[inline(always)]
    fn steer_gateway_signal_safe<S: ManifoldGatewaySignal + ?Sized>(
        &self,
        signal: &S,
        trail: &mut gcso_pointer_trail_t,
        unresolved_damping_u7: u8,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(GatewayFrame, CausalProvenanceReport)> {
        let (frame, report, _frm_w) = steer_gateway_signal_core_safe(
            self,
            signal,
            trail,
            unresolved_damping_u7,
            query_tensor,
            key_tensor,
            logits_opt,
        )?;
        Ok((frame, report))
    }

    /// Synthesizes `signal` via [`Self::steer_gateway_signal_safe`], steers `query_tensor` (and
    /// optional `key_tensor` / `logits_opt`), and absorbs the resulting [`GatewayFrame`] sample
    /// into `crystal_record` on $SO(2)^{64}$ using the register-resident `frm_w` words without
    /// reloading `frame.phase_deltas`, returning `(GatewayFrame, CausalProvenanceReport, updated_gravity_mass)`.
    ///
    /// # Errors
    /// Propagates any error from [`Self::steer_gateway_signal_safe`].
    #[inline(always)]
    fn steer_and_crystallize_gateway_signal_safe<S: ManifoldGatewaySignal + ?Sized>(
        &self,
        signal: &S,
        trail: &mut gcso_pointer_trail_t,
        crystal_record: &mut CrystallizedAttractorRecord,
        unresolved_damping_u7: u8,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(GatewayFrame, CausalProvenanceReport, f32)> {
        let (frame, report, frm_w) = steer_gateway_signal_core_safe(
            self,
            signal,
            trail,
            unresolved_damping_u7,
            query_tensor,
            key_tensor,
            logits_opt,
        )?;
        let grav_mass = crystal_record.absorb_gateway_frame_with_words(&frame, &frm_w);
        Ok((frame, report, grav_mass))
    }

    /// Executes the closed-loop **2-Origin Composite Manifold Gateway $\to$ $SO(2)$-Fused
    /// RIPA/sub-head router Steering $\to$ Classical Online Toroidal $k$-Means / DP-Means Macro Clustering**
    /// pipeline on `centroids: &mut [CrystallizedAttractorRecord]` in a single zero-allocation call.
    ///
    /// Returns `(GatewayFrame, CausalProvenanceReport, assigned_cluster_idx, assigned_anchor_id, centroid_coherence, updated_gravity_mass, spawned_new_cluster)`.
    ///
    /// # Errors
    /// Propagates any validation or execution error from [`Self::steer_gateway_signal_safe`].
    #[inline(always)]
    fn steer_and_kmeans_crystallize_gateway_signal_safe<S: ManifoldGatewaySignal + ?Sized>(
        &self,
        signal: &S,
        trail: &mut gcso_pointer_trail_t,
        centroids: &mut [CrystallizedAttractorRecord],
        split_coherence_thresh: f32,
        unresolved_damping_u7: u8,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(
        GatewayFrame,
        CausalProvenanceReport,
        usize,
        u32,
        f32,
        f32,
        bool,
    )> {
        let (frame, report, frm_w) = steer_gateway_signal_core_safe(
            self,
            signal,
            trail,
            unresolved_damping_u7,
            query_tensor,
            key_tensor,
            logits_opt,
        )?;
        let (cluster_idx, anchor_id, coh, mass, spawned) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_frame_with_words_64(
                centroids,
                &frame,
                &frm_w,
                split_coherence_thresh,
                Some(trail),
            );
        Ok((frame, report, cluster_idx, anchor_id, coh, mass, spawned))
    }

    /// Executes **Coarse-to-Fine Macro-to-Micro Routing ($\mathcal{O}(K) \to \mathcal{O}(M_{\mathrm{local}})$)**
    /// uniting Classical Online Toroidal $k$-Means macro centroids (`macro_centroids`) with local
    /// Bayesian pointer-trail rewiring (`candidate_trails`) and RIPA phase steering.
    ///
    /// When `attention_scores` is `None` and the Coarse step resolves a non-zero `macro_anchor_id`,
    /// the Fine step restricts 64-head coherence and logarithm evaluation to candidates belonging to
    /// `macro_anchor_id` directly on the register-resident `bridged_w`, bypassing out-of-cluster
    /// candidates and eliminating any separate pre-alignment pass over `candidate_trails`.
    ///
    /// Returns `(macro_cluster_idx, macro_anchor_id, macro_coherence, selected_trail_id, bayesian_score, transition_log_prob, found_coherent)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `candidate_trails` is empty, parameters are
    /// non-finite, or tensor dimensions are invalid.
    #[inline(always)]
    fn route_coarse_to_fine_and_steer_bayesian_trails_safe(
        &self,
        macro_centroids: &[CrystallizedAttractorRecord],
        coarse_pull_u7: u8,
        src_trail: &mut gcso_pointer_trail_t,
        candidate_trails: &mut [gcso_pointer_trail_t],
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        repulsion_penalty: f32,
        fine_interpolation_u7: u8,
        contradiction_threshold: f32,
        min_step_rad: f32,
        max_rad: f32,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(usize, u32, f32, u32, f32, f32, bool)> {
        if candidate_trails.is_empty()
            || min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(repulsion_penalty)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads) =
            validate_closed_loop_tensors(self, max_rad, query_tensor, key_tensor.as_deref())?;

        let (macro_idx, macro_anchor_id, macro_coh, _macro_grav, self_w, win_w) =
            CrystallizedAttractorRecord::route_coarse_to_fine_trail_with_words_64(
                macro_centroids,
                src_trail,
                coarse_pull_u7,
            );
        let has_win_centroid =
            macro_idx < macro_centroids.len() && macro_centroids[macro_idx].sample_count > 0;

        let (best_id, out_score, _coh_out, trans_lp, found) =
            route_macro_aligned_bayesian_trails_and_steer_core(
                self_w,
                &win_w,
                has_win_centroid,
                macro_anchor_id,
                0,
                src_trail,
                candidate_trails,
                head_dim,
                num_heads,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                repulsion_penalty,
                fine_interpolation_u7,
                contradiction_threshold,
                min_step_rad,
                max_rad,
                attention_scores,
                query_tensor,
                key_tensor,
            );

        Ok((
            macro_idx,
            macro_anchor_id,
            macro_coh,
            best_id,
            out_score,
            trans_lp,
            found,
        ))
    }

    /// Executes **Top-2 Liminal Macro-to-Micro Bifurcation Routing ($\mathcal{O}(K) \to \mathcal{O}(M_{\mathrm{local}})$)**
    /// uniting `CrystallizedAttractorRecord::route_top2_liminal_centroids_with_words_64` with dual-basin
    /// (`aid1 | aid2`) Bayesian pointer-trail rewiring and RIPA phase steering.
    ///
    /// When `is_liminal` is triggered (the top-2 macro centroids lie within `bifurcation_margin`),
    /// `src_trail` is interpolated onto the geodesic midpoint between `aid1` and `aid2` and the
    /// Fine step admits candidates from **both** competing basins (`aid1` and `aid2`).
    ///
    /// Returns `(idx1, idx2, aid1, aid2, is_liminal, selected_trail_id, bayesian_score, transition_log_prob, found_coherent)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `candidate_trails` is empty, parameters are
    /// non-finite, or tensor dimensions are invalid.
    #[inline(always)]
    fn route_top2_liminal_and_steer_bayesian_trails_safe(
        &self,
        macro_centroids: &[CrystallizedAttractorRecord],
        coarse_pull_u7: u8,
        bifurcation_margin: f32,
        src_trail: &mut gcso_pointer_trail_t,
        candidate_trails: &mut [gcso_pointer_trail_t],
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        repulsion_penalty: f32,
        fine_interpolation_u7: u8,
        contradiction_threshold: f32,
        min_step_rad: f32,
        max_rad: f32,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(usize, usize, u32, u32, bool, u32, f32, f32, bool)> {
        if candidate_trails.is_empty()
            || min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
            || !branchless_is_finite_f32(bifurcation_margin)
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(repulsion_penalty)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads) =
            validate_closed_loop_tensors(self, max_rad, query_tensor, key_tensor.as_deref())?;

        let (idx1, idx2, aid1, aid2, _eff_coh, _tot_grav, is_liminal, self_w, w1, _w2) =
            CrystallizedAttractorRecord::route_top2_liminal_centroids_with_words_64(
                macro_centroids,
                src_trail,
                coarse_pull_u7,
                bifurcation_margin,
            );
        let has_win_centroid =
            idx1 < macro_centroids.len() && macro_centroids[idx1].sample_count > 0;
        let secondary_aid = branchless_select_u32(is_liminal, aid2, 0);

        let (best_id, out_score, _coh_out, trans_lp, found) =
            route_macro_aligned_bayesian_trails_and_steer_core(
                self_w,
                &w1,
                has_win_centroid,
                aid1,
                secondary_aid,
                src_trail,
                candidate_trails,
                head_dim,
                num_heads,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                repulsion_penalty,
                fine_interpolation_u7,
                contradiction_threshold,
                min_step_rad,
                max_rad,
                attention_scores,
                query_tensor,
                key_tensor,
            );

        Ok((
            idx1, idx2, aid1, aid2, is_liminal, best_id, out_score, trans_lp, found,
        ))
    }

    /// Folds a zero-copy slice of `.gcso` [`CrystallizedAttractorRecord`]s (`records`) onto
    /// `trail.accumulated_phase_delta` in 8 register-resident `u64` SWAR words weighted by
    /// **Phase-Proximity Gravity** ([`CrystallizedAttractorRecord::fold_crystallized_ensemble_words_with_init_pull_64`]),
    /// binds the dominant 32-bit `anchor_id` into `trail.target_anchor_id` and `trail.cluster_id`,
    /// and steers `query_tensor` (and optional `key_tensor`) directly from the folded SWAR words
    /// without constructing an intermediate 128-byte [`GatewayFrame`].
    ///
    /// Returns `(total_gravity_pull, dominant_anchor_id, dominant_ripa_clamp_rad)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `min_step_rad < 0.0`, `max_rad <= 0.0`, or
    /// tensor dimensions are invalid.
    #[inline(always)]
    fn steer_crystallized_ensemble_zero_copy_safe(
        &self,
        records: &[CrystallizedAttractorRecord],
        trail: &mut gcso_pointer_trail_t,
        min_step_rad: f32,
        max_rad: f32,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(f32, u32, f32)> {
        if min_step_rad < 0.0 || !branchless_is_finite_f32(min_step_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads) =
            validate_closed_loop_tensors(self, max_rad, query_tensor, key_tensor.as_deref())?;

        let init_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(trail.attractor_pull_force));
        let (folded_w, tot_pull, dom_id, dom_clamp) =
            CrystallizedAttractorRecord::fold_crystallized_ensemble_words_with_init_pull_64(
                records,
                &mut trail.accumulated_phase_delta,
                init_pull,
            );
        if dom_id != 0 {
            trail.target_anchor_id = dom_id;
            trail.cluster_id = dom_id;
        }
        trail.attractor_pull_force = branchless_min_f32(tot_pull, MAX_ATTRACTOR_PULL_FORCE);
        trail.flags |= TRAIL_FLAG_CRYSTALLIZED_CANDIDATE | TRAIL_FLAG_SYNERGY_RESONANCE;

        let eff_ripa = branchless_select_f32(
            branchless_is_finite_f32(dom_clamp) & (dom_clamp > 0.0),
            dom_clamp,
            max_rad,
        );
        let step_u8 = branchless_rad_to_qdps_step_u8(min_step_rad);
        let eff_w = swar_q7_qdps_filter_words_64(&folded_w, step_u8);
        if swar_words_nonzero_64(&eff_w) {
            steer_from_swar_words_core_safe(
                &eff_w,
                0xFF,
                128,
                head_dim,
                num_heads,
                false,
                eff_ripa,
                query_tensor,
                key_tensor,
            );
        }

        Ok((tot_pull, dom_id, eff_ripa))
    }

    /// Executes the full end-to-end **2-Origin Composite Gateway $\to$ 8-Cell SWAR Micro-Swarm $\to$
    /// Bayesian Pointer-Chain Selection & Limina Bridging $\to$
    /// Register-Resident $SO(2)$-Fused RIPA/sub-head router Steering $\to$ `.gcso` Crystallization** pipeline.
    ///
    /// Returns `(GatewayFrame, CausalProvenanceReport, StigmergicReconstitutionReport, selected_trail_id, found_coherent_trail)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if tensor dimensions or `GCSO_RESTRICT` non-aliasing
    /// invariants are violated.
    #[inline]
    fn steer_gateway_swarm_bayesian_and_crystallize_safe<S: ManifoldGatewaySignal + ?Sized>(
        &self,
        signal: &S,
        trail: &mut gcso_pointer_trail_t,
        candidate_trails_opt: Option<&mut [gcso_pointer_trail_t]>,
        attention_scores_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
        step_gain_u7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        unresolved_damping_u7: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        repulsion_penalty: f32,
        interpolation_u7: u8,
        contradiction_threshold: f32,
        explore_pulse_q7: gcso_q7_t,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        mut key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
        crystal_record_opt: Option<&mut CrystallizedAttractorRecord>,
    ) -> GcsoResult<(
        GatewayFrame,
        CausalProvenanceReport,
        StigmergicReconstitutionReport,
        u32,
        bool,
    )> {
        let (frame, init_frm_w) = signal.synthesize_gateway_frame_with_words();
        let report = frame.to_causal_report();
        let (head_dim, num_heads) = validate_closed_loop_tensors(
            self,
            frame.ripa_clamp_rad,
            query_tensor,
            key_tensor.as_deref(),
        )?;

        let has_frame_anchor = frame.target_anchor_id != 0;
        trail.target_anchor_id = branchless_select_u32(
            has_frame_anchor,
            frame.target_anchor_id,
            trail.target_anchor_id,
        );
        trail.cluster_id = branchless_select_u32(
            has_frame_anchor,
            frame.target_anchor_id,
            frame.dominant_slot as u32,
        );

        let qdps_step_u8 = frame.qdps_min_step_q7;
        let (swarm_report, mut swar_words) = trail.try_resolve_action_in_swarm_from_words(
            gcso_tagged_ptr_t(frame.tagged_ptr),
            &init_frm_w,
            frame.active_swarm_cell_mask,
            step_gain_u7,
            qdps_step_u8,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            bloom_filter,
            repulsion_threshold,
            repulsion_scale_u7,
        );

        trail.attractor_pull_force = branchless_min_f32(
            branchless_max_f32(trail.attractor_pull_force, frame.effective_energy),
            MAX_ATTRACTOR_PULL_FORCE,
        );

        let mut selected_trail_id = trail.linked_trail_id;
        let mut found_coherent = false;
        if let Some(candidates) = candidate_trails_opt {
            if !candidates.is_empty() {
                let scores_slice_opt = attention_scores_opt
                    .filter(|s| !s.is_empty())
                    .map(AlignedSliceMut32::as_mut_slice);
                let (best_id, _bayes_score, _mean_coh, _coh_cnt, found_valid, bridged_w, _best_coh) =
                    crate::traits::select_rewire_bridge_and_modulate_with_preloaded_words_64(
                        trail,
                        swar_words,
                        candidates,
                        scores_slice_opt,
                        head_dim as u32,
                        phase_interference_gain,
                        stigmergic_prior_gain,
                        pointer_link_bonus,
                        contradiction_threshold,
                        interpolation_u7,
                        repulsion_penalty,
                        0,
                        None,
                        0.0,
                    );
                selected_trail_id = best_id;
                found_coherent = found_valid;
                swar_words = bridged_w;
                if found_valid {
                    let best_idx = best_id as usize;
                    if best_idx < candidates.len() {
                        let dst_trail = &mut candidates[best_idx];
                        if interpolation_u7 < 128 {
                            *trail_phase_words_mut(dst_trail) = swar_words;
                        }
                        let inherit_anchor = dst_trail.target_anchor_id != 0;
                        trail.target_anchor_id = branchless_select_u32(
                            inherit_anchor,
                            dst_trail.target_anchor_id,
                            trail.target_anchor_id,
                        );
                        if trail.target_anchor_id != 0 {
                            let resolved_cluster = branchless_select_u32(
                                dst_trail.cluster_id != 0,
                                dst_trail.cluster_id,
                                trail.target_anchor_id,
                            );
                            trail.cluster_id = resolved_cluster;
                            dst_trail.target_anchor_id = trail.target_anchor_id;
                            dst_trail.cluster_id = resolved_cluster;
                        }
                    }
                }
            }
        }

        if ((frame.action_mode_flags & ACTION_MODE_LAYER_SKIP) == 0)
            & swarm_report
                .evolved_ptr
                .has_action_mode(ACTION_MODE_DPSR_STEER)
        {
            let clean_damping =
                branchless_select_u8(unresolved_damping_u7 > 128, 128, unresolved_damping_u7);
            let swar_nonzero = swar_words_nonzero_64(&swar_words);
            let use_pspm = ((frame.action_mode_flags & ACTION_MODE_SUBHEAD_ROUTE) != 0)
                & (swarm_report.quorum_mask == 0xFF);
            if use_pspm {
                let is_default_ripa = (explore_pulse_q7 == 0)
                    & swar_nonzero
                    & (frame.ripa_clamp_rad.to_bits() == DEFAULT_RIPA_CLAMP_MAX_RAD.to_bits())
                    & frame.has_unit_pspm_gains()
                    & (num_heads > 0)
                    & (num_heads <= MAX_ATTENTION_HEADS);
                if is_default_ripa {
                    let rot_dim = frame.resolve_autonomous_rot_dim(head_dim, 0.0, 0.0);
                    if let Some(k) = key_tensor.as_deref_mut() {
                        crate::traits::rope::rotate_fused_self_ripa_pspm_q_and_ripa_k_default_words(
                            query_tensor.as_mut_slice(),
                            k.as_mut_slice(),
                            &swar_words,
                            num_heads,
                            head_dim,
                            rot_dim,
                        );
                    } else {
                        crate::traits::rope::rotate_fused_self_ripa_and_pspm_default_words(
                            query_tensor.as_mut_slice(),
                            &swar_words,
                            num_heads,
                            head_dim,
                            rot_dim,
                            true,
                        );
                    }
                } else {
                    let subhead_cfg = frame.to_subhead_config_for_heads(num_heads);
                    dispatch_fused_ripa_and_pspm_words_core(
                        query_tensor,
                        &swar_words,
                        frame.ripa_clamp_rad,
                        &subhead_cfg,
                        &swar_words,
                        &swar_words,
                        &swar_words,
                        frame.ripa_clamp_rad,
                        explore_pulse_q7,
                        head_dim,
                        true,
                    )?;
                    if let Some(k) = key_tensor.as_deref_mut() {
                        if swar_nonzero {
                            steer_from_swar_words_core_safe(
                                &swar_words,
                                swarm_report.quorum_mask,
                                clean_damping,
                                head_dim,
                                num_heads,
                                true,
                                frame.ripa_clamp_rad,
                                k,
                                None,
                            );
                        }
                    }
                }
            } else if swar_nonzero {
                steer_from_swar_words_core_safe(
                    &swar_words,
                    swarm_report.quorum_mask,
                    clean_damping,
                    head_dim,
                    num_heads,
                    true,
                    frame.ripa_clamp_rad,
                    query_tensor,
                    key_tensor,
                );
            }
        }

        if let Some(logits) = logits_opt {
            if !logits.is_empty() {
                scale_f32_slice_unrolled8(logits.as_mut_slice(), frame.effective_inv_temp);
            }
        }

        if let Some(crystal) = crystal_record_opt {
            let _ = crystal.absorb_gateway_frame_with_words(&frame, &swar_words);
        }

        Ok((
            frame,
            report,
            swarm_report,
            selected_trail_id,
            found_coherent,
        ))
    }

    /// Executes the full **Gateway $\to$ Micro-Swarm $\to$ Bayesian Pointer-Trail $\to$ $SO(2)$
    /// RoPE Steering $\to$ `.gcso` Crystallization** closed loop with **zero manual tuning
    /// hyperparameters**, deriving all Swarm, Bayesian, and damping constants autonomously from
    /// the trait-composed `signal: &S` ([`ManifoldGatewaySignal`]).
    ///
    /// Returns `(GatewayFrame, CausalProvenanceReport, StigmergicReconstitutionReport, selected_trail_id, found_coherent_trail)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if tensor dimensions or `GCSO_RESTRICT` non-aliasing
    /// invariants are violated.
    #[inline(always)]
    fn steer_gateway_autonomous_safe<S: ManifoldGatewaySignal + ?Sized>(
        &self,
        signal: &S,
        trail: &mut gcso_pointer_trail_t,
        candidate_trails_opt: Option<&mut [gcso_pointer_trail_t]>,
        attention_scores_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
        crystal_record_opt: Option<&mut CrystallizedAttractorRecord>,
    ) -> GcsoResult<(
        GatewayFrame,
        CausalProvenanceReport,
        StigmergicReconstitutionReport,
        u32,
        bool,
    )> {
        let preview_frame = signal.synthesize_gateway_frame();
        self.steer_gateway_swarm_bayesian_and_crystallize_safe(
            &preview_frame,
            trail,
            candidate_trails_opt,
            attention_scores_opt,
            preview_frame.autonomous_swarm_step_gain_u7(),
            preview_frame.autonomous_swarm_tolerance_q7(),
            preview_frame.autonomous_swarm_quorum_threshold(),
            preview_frame.autonomous_unresolved_damping_u7(),
            preview_frame.effective_energy,
            0.15,
            bloom_filter,
            3,
            96,
            2.0,
            0.5,
            0.35,
            1.0,
            preview_frame.autonomous_liminal_interpolation_u7(),
            0.25,
            0,
            query_tensor,
            key_tensor,
            logits_opt,
            crystal_record_opt,
        )
    }
}

impl<T: PhaseSteering + SubHeadRouter + ?Sized> PhaseSteeringClosedLoop for T {}
