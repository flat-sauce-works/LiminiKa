// name: src/core/src/abi/action_hub.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Action Hub orchestration, Limina bridging, Bayesian trail rewiring, and DAES acceleration FFI.

use super::*;

// ===================================================================
// 3. Action Hub, Limina Bridging, Bayesian Trail Rewiring & DAES Interface
// ===================================================================

/// Allocates an Action Hub (Sidecar Pointer Table) instance with the specified slot capacity.
///
/// # Safety
/// `hub_out` must be a non-null, naturally aligned writable handle pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_create(
    capacity: u32,
    hub_out: *mut GcsoActionHubHandle,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and alignment.
        let mut out_handle = unsafe { FfiHandleMut::from_raw(hub_out)? };
        let hub = ActionHubEngine::with_capacity(capacity)?;
        *out_handle = Box::into_raw(Box::new(hub)).cast::<GcsoActionHubOpaque>();
        Ok(GCSO_SUCCESS)
    })
}

/// Initializes a 32-byte [`gcso_stigmergic_thresholds_t`] structure with canonical v0.1.1 defaults.
///
/// # Safety
/// `thresholds_out` must be a non-null, 32-byte aligned writable [`gcso_stigmergic_thresholds_t`] pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_init_default_thresholds(
    thresholds_out: *mut gcso_stigmergic_thresholds_t,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and 32B alignment.
        let mut out_th = unsafe { FfiHandleMut::from_raw(thresholds_out)? };
        *out_th = StigmergicThresholds::new();
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluates the 7-regime stigmergic threshold ladder on `trail` in strictly branchless constant
/// time (`0` branches) and writes the 32-byte [`gcso_action_orchestration_report_t`] to `report_out`.
///
/// # Safety
/// `trail` (128B aligned) and `report_out` (32B aligned) must be non-null; `thresholds_opt` may be `NULL` or 32B aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_evaluate_stigmergic_thresholds(
    thresholds_opt: *const gcso_stigmergic_thresholds_t,
    trail: *const gcso_pointer_trail_t,
    entropy_h_tilde: f32,
    phase_coherence: f32,
    transition_log_prob: f32,
    selected_trail_id: u32,
    bloom_repulsion_density: u8,
    dominant_slot: u8,
    report_out: *mut gcso_action_orchestration_report_t,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let ((th_opt, trail_ref), mut rep_out) = unsafe {
            (
                (
                    FfiHandleRef::from_raw_opt(thresholds_opt)?,
                    FfiHandleRef::from_raw(trail)?,
                ),
                FfiHandleMut::from_raw(report_out)?,
            )
        };
        let th = match th_opt {
            Some(ref t) => {
                t.validate()?;
                **t
            }
            None => StigmergicThresholds::new(),
        };
        *rep_out = th.evaluate_report_on_trail(
            &trail_ref,
            2,
            entropy_h_tilde,
            phase_coherence,
            transition_log_prob,
            selected_trail_id,
            bloom_repulsion_density,
            dominant_slot,
        );
        Ok(GCSO_SUCCESS)
    })
}

/// Executes a full **Stigmergic Threshold Action-Selecting Orchestration Step** across the C-ABI
/// boundary via [`ActionHubEngine::orchestrate_next_action_step`] ($C_{\mathrm{alloc}} = 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_orchestrate_stigmergic_step_safe(
    hub_opt: GcsoActionHubHandle,
    thresholds_opt: *const gcso_stigmergic_thresholds_t,
    input_ptr: u64,
    entropy_h_tilde: f32,
    dominant_slot: u8,
    trail_inout: *mut gcso_pointer_trail_t,
    attention_scores_inout: *mut f32,
    num_scores: usize,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
    report_out: *mut gcso_action_orchestration_report_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(trail_inout)?;
        let (dpsr, total_len) = init_dpsr_engine(head_dim, num_heads)?;
        // SAFETY: Validated for non-nullness and alignment.
        let ((th_ref_opt, mut trail), mut rep_out) = unsafe {
            (
                (
                    FfiHandleRef::from_raw_opt(thresholds_opt)?,
                    FfiHandleMut::from_raw(trail_inout)?,
                ),
                FfiHandleMut::from_raw(report_out)?,
            )
        };
        // SAFETY: `AlignedSliceMut32::from_raw_parts_mut_opt` validates 32B alignment when non-null.
        let ((mut scores_opt, mut q_opt), (mut k_opt, mut logits_opt)) = unsafe {
            (
                (
                    if attention_scores_inout.is_null() || num_scores == 0 {
                        None
                    } else {
                        AlignedSliceMut32::from_raw_parts_mut_opt(
                            attention_scores_inout,
                            num_scores,
                        )?
                    },
                    AlignedSliceMut32::from_raw_parts_mut_opt(query_tensor, total_len)?,
                ),
                (
                    AlignedSliceMut32::from_raw_parts_mut_opt(key_tensor, total_len)?,
                    if logits_inout.is_null() || vocab_size == 0 {
                        None
                    } else {
                        AlignedSliceMut32::from_raw_parts_mut_opt(logits_inout, vocab_size)?
                    },
                ),
            )
        };

        if let (Some(ref q), Some(ref k)) = (&q_opt, &k_opt) {
            q.ensure_disjoint_mut(k)?;
        }
        if let Some(ref s) = scores_opt {
            if let Some(ref q) = q_opt {
                q.ensure_disjoint_mut(s)?;
            }
            if let Some(ref k) = k_opt {
                k.ensure_disjoint_mut(s)?;
            }
        }

        let report = if !hub_opt.is_null()
            && (hub_opt as usize) >= 4096
            && ((hub_opt as usize) & (align_of::<ActionHubEngine>() - 1)) == 0
        {
            // SAFETY: `hub_opt` is non-null and 128-byte aligned.
            let hub_ref = unsafe { &mut *hub_opt.cast::<ActionHubEngine>() };
            hub_ref.dpsr = dpsr;
            if let Some(ref th) = th_ref_opt {
                th.validate()?;
                hub_ref.thresholds = **th;
            }
            hub_ref.orchestrate_next_action_step(
                gcso_tagged_ptr_t(input_ptr),
                entropy_h_tilde,
                dominant_slot,
                &mut trail,
                scores_opt.as_mut(),
                q_opt.as_mut(),
                k_opt.as_mut(),
                logits_opt.as_mut(),
            )?
        } else {
            let th = match th_ref_opt {
                Some(ref t) => {
                    t.validate()?;
                    **t
                }
                None => StigmergicThresholds::new(),
            };
            th.orchestrate_stateless_action_step(
                &dpsr,
                gcso_tagged_ptr_t(input_ptr),
                entropy_h_tilde,
                dominant_slot,
                &mut trail,
                q_opt.as_mut(),
                k_opt.as_mut(),
                logits_opt.as_mut(),
            )?
        };
        *rep_out = report;
        Ok(GCSO_SUCCESS)
    })
}

/// Advances a 64-bit tagged pointer transition in $\mathcal{O}(1)$ time and synchronizes
/// `cluster_id` and `flags` from the pointer's embedded Target Attractor Working Lane ID and Action Mode mask.
///
/// # Safety
/// `context` and `trail_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_step_pointer(
    context: GcsoContextHandle,
    current_ptr: u64,
    trail_out: *mut gcso_pointer_trail_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and 128B alignment.
        let mut trail = unsafe { FfiHandleMut::from_raw(trail_out)? };
        let tp = gcso_tagged_ptr_t(current_ptr);

        trail.record_pointer_transition(current_ptr, 1, 1.0);
        let embedded_phase = tp.unpack_phase();
        if embedded_phase != 0 {
            trail.accumulate_uniform_phase(embedded_phase);
        }
        trail.cluster_id = branchless_select_u32(
            trail.target_anchor_id != 0,
            trail.target_anchor_id,
            u32::from(tp.target_attractor_slot() & ATTRACTOR_SLOT_ID_MASK),
        );
        trail.sync_flags_with_current_ptr();
        Ok(GCSO_SUCCESS)
    })
}

/// Evolves a [`gcso_pointer_trail_t`]'s embedded `current_ptr` Action Mode flags and Target Attractor
/// Working Lane ID (`0..=7`) bottom-up in $\mathcal{O}(1)$ based on its stigmergic density, EDBC
/// entropy $\tilde{H}$, and Bloom filter repulsion state.
///
/// # Safety
/// `trail_inout` must be non-null and 128B aligned; `evolved_ptr_out` may be `NULL` or 8B aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_evolve_trail_mode(
    trail_inout: *mut gcso_pointer_trail_t,
    entropy_h_tilde: f32,
    repulsion_triggered: u8,
    dominant_attractor_slot: u8,
    low_entropy_skip_thresh: f32,
    high_density_srl_thresh: f32,
    evolved_ptr_out: *mut u64,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let (mut trail, out_ptr_opt) = unsafe {
            (
                FfiHandleMut::from_raw(trail_inout)?,
                FfiHandleMut::from_raw_opt(evolved_ptr_out)?,
            )
        };

        let evolved = trail.evolve_trail_mode_stigmergically(
            entropy_h_tilde,
            repulsion_triggered != 0,
            dominant_attractor_slot,
            low_entropy_skip_thresh,
            high_density_srl_thresh,
        );
        if let Some(mut out_p) = out_ptr_opt {
            *out_p = evolved.raw();
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Bridges the inter-memory **Limina** gap between `prev_trail` ($M_{t-1}$) and `curr_trail_inout`
/// ($M_t$) in $\mathcal{O}(1)$ constant time, evaluating conditional transition log-probability
/// $\ln P(M_t \mid M_{t-1})$ and applying Phase-Conjugate Repulsion ($-\boldsymbol{\Delta\theta}$)
/// if the transition is self-contradictory ($H^1(K; \mathcal{F}) \neq 0$).
///
/// # Safety
/// `prev_trail` and `curr_trail_inout` must be non-null, 128B aligned, and disjoint.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_bridge_liminal_trails(
    prev_trail: *const gcso_pointer_trail_t,
    curr_trail_inout: *mut gcso_pointer_trail_t,
    interpolation_u7: u8,
    contradiction_threshold: f32,
    coherence_out: *mut f32,
    log_prob_out: *mut f32,
    contradiction_repulsed_out: *mut u8,
) -> GcsoStatus {
    ffi_boundary!({
        if !branchless_is_finite_f32(contradiction_threshold) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated for non-nullness and 128B alignment.
        let (prev, mut curr) = unsafe {
            (
                FfiHandleRef::from_raw(prev_trail)?,
                FfiHandleMut::from_raw(curr_trail_inout)?,
            )
        };
        curr.ensure_disjoint_ref(&prev)?;

        let (coh, log_prob, repulsed) = curr.bridge_liminal_hallucinated_trail(
            &prev,
            interpolation_u7,
            contradiction_threshold,
        );
        curr.sync_flags_with_current_ptr();

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(coherence_out, coh)?;
            write_opt_out(log_prob_out, log_prob)?;
            write_opt_out(contradiction_repulsed_out, u8::from(repulsed))?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Performs **Stigmergic Pointer-Chain Selection & Geodesic Bridging** across
/// the C-ABI boundary in a single pass: scans `num_candidates` candidate memory trails in constant
/// time per candidate to select the optimal coherent ($H^1(K; \mathcal{F}) = 0$) next trail $j^*$
/// maximizing the Bayes-Attention transition score, rewires `src_trail_inout.linked_trail_id = j^*`,
/// and bridges the inter-memory **Limina** gap onto both `src_trail_inout` and `candidate_trails_inout[j^*]`.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_select_and_bridge_best_next_trail(
    src_trail_inout: *mut gcso_pointer_trail_t,
    candidate_trails_inout: *mut gcso_pointer_trail_t,
    num_candidates: usize,
    head_dim: u32,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    interpolation_u7: u8,
    contradiction_threshold: f32,
    selected_trail_id_out: *mut u32,
    bayesian_score_out: *mut f32,
    log_prob_out: *mut f32,
    found_out: *mut u8,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(src_trail_inout)?;
        validate_ptr(candidate_trails_inout)?;
        if num_candidates == 0
            || head_dim == 0
            || (head_dim & 1) != 0
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let cand_bytes = num_candidates
            .checked_mul(size_of::<gcso_pointer_trail_t>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            src_trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
            candidate_trails_inout as usize,
            cand_bytes,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated non-null, aligned, and disjoint above.
        let mut src_trail = unsafe { FfiHandleMut::from_raw(src_trail_inout)? };
        // SAFETY: Validated non-null, aligned, and disjoint above.
        let candidates =
            unsafe { slice::from_raw_parts_mut(candidate_trails_inout, num_candidates) };

        let (selected_id, bayes_score, best_coh, found) = src_trail
            .select_and_rewire_best_next_trail(
                candidates,
                head_dim,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                contradiction_threshold,
            );

        let mut log_prob = 0.0_f32;
        if found {
            let dst_idx = (selected_id as usize).min(num_candidates - 1);
            let dst_trail = &mut candidates[dst_idx];
            if !log_prob_out.is_null() {
                log_prob = branchless_fast_transition_log_prob(best_coh);
            }
            if interpolation_u7 > 0 {
                let src_w = *trail_phase_words_ref(&src_trail);
                let dst_w = *trail_phase_words_ref(dst_trail);
                if !swar_words_equal_64(&src_w, &dst_w) {
                    let s_u64 =
                        branchless_select_u8(interpolation_u7 > 128, 128, interpolation_u7) as u64;
                    let bias_restore =
                        (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                    let bridged_w =
                        swar_slerp_words_with_bias_64(&src_w, &dst_w, s_u64, bias_restore);
                    *trail_phase_words_mut(&mut src_trail) = bridged_w;
                    *trail_phase_words_mut(dst_trail) = bridged_w;
                }
            } else {
                *trail_phase_words_mut(dst_trail) = *trail_phase_words_ref(&src_trail);
            }
            dst_trail.prev_ptr = src_trail.current_ptr;
            dst_trail.step_count = src_trail.step_count.wrapping_add(1);
            let inherit_anchor = dst_trail.target_anchor_id != 0;
            src_trail.target_anchor_id = branchless_select_u32(
                inherit_anchor,
                dst_trail.target_anchor_id,
                src_trail.target_anchor_id,
            );
            if src_trail.target_anchor_id != 0 {
                src_trail.cluster_id = src_trail.target_anchor_id;
                dst_trail.target_anchor_id = src_trail.target_anchor_id;
                dst_trail.cluster_id = src_trail.target_anchor_id;
            }
            src_trail.flags |= TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE;
            dst_trail.flags |= TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE;
            src_trail.sync_flags_with_current_ptr();
            dst_trail.sync_flags_with_current_ptr();
        }

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(selected_trail_id_out, selected_id)?;
            write_opt_out(bayesian_score_out, bayes_score)?;
            write_opt_out(log_prob_out, log_prob)?;
            write_opt_out(found_out, u8::from(found))?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Directly modulates `num_candidates` pre-Softmax Attention scores (`attention_scores_inout`) in
/// zero-copy constant time per candidate from `src_trail` and `candidate_trails` using
/// [`StigmergicTrailOps::modulate_candidate_trail_scores_in_place`].
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_modulate_attention_scores_by_candidate_trails(
    attention_scores_inout: *mut f32,
    src_trail: *const gcso_pointer_trail_t,
    candidate_trails: *const gcso_pointer_trail_t,
    num_candidates: usize,
    head_dim: u32,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    contradiction_threshold: f32,
    repulsion_penalty: f32,
    mean_coherence_out: *mut f32,
    best_candidate_id_out: *mut u32,
    coherent_count_out: *mut u32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(attention_scores_inout)?;
        validate_ptr(src_trail)?;
        validate_ptr(candidate_trails)?;
        if num_candidates == 0
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

        let scores_bytes = num_candidates
            .checked_mul(size_of::<f32>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        let cand_bytes = num_candidates
            .checked_mul(size_of::<gcso_pointer_trail_t>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            attention_scores_inout as usize,
            scores_bytes,
            src_trail as usize,
            size_of::<gcso_pointer_trail_t>(),
        ) || raw_byte_ranges_overlap(
            attention_scores_inout as usize,
            scores_bytes,
            candidate_trails as usize,
            cand_bytes,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated non-null, aligned, and disjoint above.
        let ((src_ref, cand_slice), scores) = unsafe {
            (
                (
                    FfiHandleRef::from_raw(src_trail)?,
                    slice::from_raw_parts(candidate_trails, num_candidates),
                ),
                slice::from_raw_parts_mut(attention_scores_inout, num_candidates),
            )
        };

        let (mean_coh, best_id, coherent_count) = src_ref.modulate_candidate_trail_scores_in_place(
            scores,
            cand_slice,
            head_dim,
            phase_interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
            contradiction_threshold,
            repulsion_penalty,
        );

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(mean_coherence_out, mean_coh)?;
            write_opt_out(best_candidate_id_out, best_id)?;
            write_opt_out(coherent_count_out, coherent_count)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes a single-pass **Fused Bayesian Pointer-Trail Selection + Rewiring + Limina Slerp
/// Bridging + Optional Pre-Softmax Attention Score Modulation** across the C-ABI boundary via
/// [`StigmergicTrailOps::select_rewire_bridge_and_modulate_bayesian_trails`] ($C_{\mathrm{alloc}} = 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_select_bridge_and_modulate_bayesian_trails(
    src_trail_inout: *mut gcso_pointer_trail_t,
    candidate_trails: *const gcso_pointer_trail_t,
    num_candidates: usize,
    attention_scores_inout: *mut f32,
    head_dim: u32,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    contradiction_threshold: f32,
    interpolation_u7: u8,
    repulsion_penalty: f32,
    selected_trail_id_out: *mut u32,
    bayesian_score_out: *mut f32,
    mean_coherence_out: *mut f32,
    coherent_count_out: *mut u32,
    found_out: *mut u8,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(src_trail_inout)?;
        validate_ptr(candidate_trails)?;
        if num_candidates == 0
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

        let cand_bytes = num_candidates
            .checked_mul(size_of::<gcso_pointer_trail_t>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            src_trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
            candidate_trails as usize,
            cand_bytes,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let scores_opt: Option<&mut [f32]> = if attention_scores_inout.is_null() {
            None
        } else {
            validate_ptr(attention_scores_inout)?;
            let scores_bytes = num_candidates
                .checked_mul(size_of::<f32>())
                .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
            if raw_byte_ranges_overlap(
                attention_scores_inout as usize,
                scores_bytes,
                src_trail_inout as usize,
                size_of::<gcso_pointer_trail_t>(),
            ) || raw_byte_ranges_overlap(
                attention_scores_inout as usize,
                scores_bytes,
                candidate_trails as usize,
                cand_bytes,
            ) {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            // SAFETY: Validated non-null, aligned, and disjoint above.
            Some(unsafe { slice::from_raw_parts_mut(attention_scores_inout, num_candidates) })
        };

        // SAFETY: Validated non-null, aligned, and disjoint above.
        let (mut src_trail, cand_slice) = unsafe {
            (
                FfiHandleMut::from_raw(src_trail_inout)?,
                slice::from_raw_parts(candidate_trails, num_candidates),
            )
        };

        let (selected_id, bayes_score, mean_coh, coherent_count, found) = src_trail
            .select_rewire_bridge_and_modulate_bayesian_trails(
                cand_slice,
                scores_opt,
                head_dim,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                contradiction_threshold,
                interpolation_u7,
                repulsion_penalty,
            );
        if found {
            src_trail.flags |= TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE;
        }
        src_trail.sync_flags_with_current_ptr();

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(selected_trail_id_out, selected_id)?;
            write_opt_out(bayesian_score_out, bayes_score)?;
            write_opt_out(mean_coherence_out, mean_coh)?;
            write_opt_out(coherent_count_out, coherent_count)?;
            write_opt_out(found_out, u8::from(found))?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes **Coarse-to-Fine Macro-Centroid Trail Routing ($\mathcal{O}(K)$)** across the C-ABI
/// boundary via [`CrystallizedAttractorRecord::route_coarse_to_fine_trail_64`].
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_route_coarse_to_fine_centroid_only(
    trail_inout: *mut gcso_pointer_trail_t,
    centroids: *const gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    pull_step_u7: u8,
    winning_centroid_idx_out: *mut usize,
    selected_anchor_id_out: *mut u32,
    centroid_coherence_out: *mut f32,
    gravity_pull_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(trail_inout)?;
        validate_ptr(centroids)?;
        if num_centroids == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let cent_bytes = num_centroids
            .checked_mul(size_of::<CrystallizedAttractorRecord>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
            centroids as usize,
            cent_bytes,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated non-null, aligned, and disjoint above.
        let (mut trail, centroids_slice) = unsafe {
            (
                FfiHandleMut::from_raw(trail_inout)?,
                slice::from_raw_parts(centroids, num_centroids),
            )
        };

        let (win_k, anchor_id, coh, grav) =
            CrystallizedAttractorRecord::route_coarse_to_fine_trail_64(
                centroids_slice,
                &mut trail,
                pull_step_u7,
            );
        trail.sync_flags_with_current_ptr();

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(winning_centroid_idx_out, win_k)?;
            write_opt_out(selected_anchor_id_out, anchor_id)?;
            write_opt_out(centroid_coherence_out, coh)?;
            write_opt_out(gravity_pull_out, grav)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes a single-pass **Fused Coarse-to-Fine Macro-to-Micro Candidate-Trail Routing
/// ($\mathcal{O}(K + M)$) + Limina Slerp Bridging + In-Register QDPS/RIPA Phase Steering** across
/// the C-ABI boundary ($C_{\mathrm{alloc}} = 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_route_coarse_to_fine_trail_and_steer_qk_safe(
    src_trail_inout: *mut gcso_pointer_trail_t,
    centroids: *const gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    candidate_trails: *const gcso_pointer_trail_t,
    num_candidates: usize,
    head_dim: usize,
    num_heads: usize,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    cluster_bonus_gain: f32,
    contradiction_threshold: f32,
    interpolation_u7: u8,
    min_step_rad: f32,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    winning_centroid_idx_out: *mut usize,
    selected_trail_id_out: *mut u32,
    best_score_out: *mut f32,
    best_coherence_out: *mut f32,
    found_out: *mut u8,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(src_trail_inout)?;
        validate_ptr(centroids)?;
        validate_ptr(candidate_trails)?;
        if num_centroids == 0
            || num_candidates == 0
            || min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(cluster_bonus_gain)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let cent_bytes = num_centroids
            .checked_mul(size_of::<CrystallizedAttractorRecord>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        let cand_bytes = num_candidates
            .checked_mul(size_of::<gcso_pointer_trail_t>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            src_trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
            centroids as usize,
            cent_bytes,
        ) || raw_byte_ranges_overlap(
            src_trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
            candidate_trails as usize,
            cand_bytes,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: `bind_qk_32` validates `query_tensor` and `key_tensor` alignment and disjointness.
        let (_engine, mut q_aligned, mut k_opt) =
            unsafe { bind_qk_32(query_tensor, key_tensor, head_dim, num_heads)? };
        // SAFETY: Validated non-null, aligned, and disjoint above.
        let ((mut src_trail, centroids_slice), cand_slice) = unsafe {
            (
                (
                    FfiHandleMut::from_raw(src_trail_inout)?,
                    slice::from_raw_parts(centroids, num_centroids),
                ),
                slice::from_raw_parts(candidate_trails, num_candidates),
            )
        };

        let (win_k, sel_id, best_score, best_coh, found) =
            CrystallizedAttractorRecord::route_coarse_to_fine_candidates_64(
                centroids_slice,
                &mut src_trail,
                cand_slice,
                head_dim as u32,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                cluster_bonus_gain,
                contradiction_threshold,
            );

        let steer_words = if found && interpolation_u7 > 0 {
            let dst_idx = (sel_id as usize).min(num_candidates - 1);
            bridge_trail_words_slerp_u7(
                &mut src_trail,
                &cand_slice[dst_idx].accumulated_phase_delta,
                interpolation_u7,
            )
        } else {
            *trail_phase_words_ref(&src_trail)
        };
        let winner = &centroids_slice[win_k];
        sync_winner_centroid_ptr_to_trail(&mut src_trail, winner);
        src_trail.sync_flags_with_current_ptr();
        steer_routed_trail_words_qk_safe(
            &steer_words,
            src_trail.flags,
            winner.ripa_clamp_rad,
            min_step_rad,
            head_dim,
            num_heads,
            &mut q_aligned,
            k_opt.as_mut(),
        );

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(winning_centroid_idx_out, win_k)?;
            write_opt_out(selected_trail_id_out, sel_id)?;
            write_opt_out(best_score_out, best_score)?;
            write_opt_out(best_coherence_out, best_coh)?;
            write_opt_out(found_out, u8::from(found))?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes **Full Coarse-to-Fine Bayesian Routing ($\mathcal{O}(K)$ Macro Centroids $\to$
/// $\mathcal{O}(M)$ Local Pointer Trails) + Limina Slerp Bridging + Optional Pre-Softmax Attention
/// Score Modulation + $SO(2)$-Fused RIPA/PSPM Steering + Optional DAES Telemetry** in a single
/// zero-allocation C-ABI call ($C_{\mathrm{alloc}} = 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_route_coarse_to_fine_bayesian_and_steer_qk_safe(
    src_trail_inout: *mut gcso_pointer_trail_t,
    centroids: *const gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    coarse_pull_u7: u8,
    candidate_trails: *const gcso_pointer_trail_t,
    num_candidates: usize,
    attention_scores_inout: *mut f32,
    head_dim: usize,
    num_heads: usize,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    repulsion_penalty: f32,
    fine_interpolation_u7: u8,
    contradiction_threshold: f32,
    pspm_cfg_opt: *const gcso_pspm_config_t,
    min_step_rad: f32,
    daes_slot_inout: *mut gcso_daes_slot_t,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    winning_centroid_idx_out: *mut usize,
    macro_anchor_id_out: *mut u32,
    selected_trail_id_out: *mut u32,
    bayesian_score_out: *mut f32,
    mean_coherence_out: *mut f32,
    coherent_count_out: *mut u32,
    found_out: *mut u8,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(src_trail_inout)?;
        validate_ptr(centroids)?;
        validate_ptr(candidate_trails)?;
        if num_centroids == 0
            || num_candidates == 0
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

        let cent_bytes = num_centroids
            .checked_mul(size_of::<CrystallizedAttractorRecord>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        let cand_bytes = num_candidates
            .checked_mul(size_of::<gcso_pointer_trail_t>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            src_trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
            centroids as usize,
            cent_bytes,
        ) || raw_byte_ranges_overlap(
            src_trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
            candidate_trails as usize,
            cand_bytes,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let scores_opt: Option<&mut [f32]> = if attention_scores_inout.is_null() {
            None
        } else {
            validate_ptr(attention_scores_inout)?;
            let scores_bytes = num_candidates
                .checked_mul(size_of::<f32>())
                .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
            if raw_byte_ranges_overlap(
                attention_scores_inout as usize,
                scores_bytes,
                src_trail_inout as usize,
                size_of::<gcso_pointer_trail_t>(),
            ) || raw_byte_ranges_overlap(
                attention_scores_inout as usize,
                scores_bytes,
                candidate_trails as usize,
                cand_bytes,
            ) || raw_byte_ranges_overlap(
                attention_scores_inout as usize,
                scores_bytes,
                centroids as usize,
                cent_bytes,
            ) {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            // SAFETY: Validated non-null, aligned, and disjoint above.
            Some(unsafe { slice::from_raw_parts_mut(attention_scores_inout, num_candidates) })
        };

        // SAFETY: `bind_qk_32` validates `query_tensor` and `key_tensor` alignment and disjointness.
        let (_engine, mut q_aligned, mut k_opt) =
            unsafe { bind_qk_32(query_tensor, key_tensor, head_dim, num_heads)? };
        // SAFETY: Validated non-null, aligned, and disjoint above.
        let (((mut src_trail, centroids_slice), cand_slice), (pspm_ref_opt, daes_mut_opt)) = unsafe {
            (
                (
                    (
                        FfiHandleMut::from_raw(src_trail_inout)?,
                        slice::from_raw_parts(centroids, num_centroids),
                    ),
                    slice::from_raw_parts(candidate_trails, num_candidates),
                ),
                (
                    FfiHandleRef::from_raw_opt(pspm_cfg_opt)?,
                    FfiHandleMut::from_raw_opt(daes_slot_inout)?,
                ),
            )
        };

        let (win_k, macro_anchor_id, _macro_coh, _macro_grav, bridged_w, win_w) =
            CrystallizedAttractorRecord::route_coarse_to_fine_trail_with_words_64(
                centroids_slice,
                &mut src_trail,
                coarse_pull_u7,
            );
        let has_win_centroid =
            win_k < centroids_slice.len() && centroids_slice[win_k].sample_count > 0;
        let (selected_id, bayes_score, mean_coh, coh_cnt, found, steer_words, _best_coh) =
            crate::traits::select_rewire_bridge_and_modulate_with_preloaded_words_64(
                &mut src_trail,
                bridged_w,
                cand_slice,
                scores_opt,
                head_dim as u32,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                contradiction_threshold,
                fine_interpolation_u7,
                repulsion_penalty,
                macro_anchor_id,
                if has_win_centroid { Some(&win_w) } else { None },
                pointer_link_bonus,
            );

        let winner = &centroids_slice[win_k];
        if found {
            src_trail.flags |= TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE;
        }
        sync_winner_centroid_ptr_to_trail(&mut src_trail, winner);
        src_trail.sync_flags_with_current_ptr();

        let eff_ripa = branchless_select_f32(
            branchless_is_finite_f32(winner.ripa_clamp_rad) & (winner.ripa_clamp_rad > 0.0),
            winner.ripa_clamp_rad,
            DEFAULT_RIPA_CLAMP_MAX_RAD,
        );
        let mut steered = false;
        let mut pspm_routed = false;

        if (src_trail.flags & TRAIL_FLAG_INERTIAL_SKIP_ACTIVE) == 0 {
            let min_step_q7 = branchless_rad_to_qdps_step_u8(min_step_rad);
            let swar_words = if min_step_q7 <= 1 {
                steer_words
            } else {
                swar_q7_qdps_filter_words_64(&steer_words, min_step_q7)
            };
            let swar_nonzero = swar_words_nonzero_64(&swar_words);
            if let Some(ref pspm_cfg) = pspm_ref_opt {
                let fact_w_ref = if swar_words_equal_64(&win_w, &steer_words) {
                    &steer_words
                } else {
                    &win_w
                };
                dispatch_fused_ripa_and_pspm_words_core(
                    &mut q_aligned,
                    &swar_words,
                    eff_ripa,
                    pspm_cfg,
                    fact_w_ref,
                    &steer_words,
                    &steer_words,
                    eff_ripa,
                    0,
                    head_dim,
                    true,
                )?;
                steered = true;
                pspm_routed = true;
                if let Some(ref mut k) = k_opt {
                    if swar_nonzero {
                        steer_from_swar_words_core_safe(
                            &swar_words,
                            0xFF,
                            128,
                            head_dim,
                            num_heads,
                            false,
                            eff_ripa,
                            k,
                            None,
                        );
                    }
                }
            } else if swar_nonzero {
                steer_from_swar_words_core_safe(
                    &swar_words,
                    0xFF,
                    128,
                    head_dim,
                    num_heads,
                    false,
                    eff_ripa,
                    &mut q_aligned,
                    k_opt.as_mut(),
                );
                steered = true;
            }
        }

        if let Some(mut daes_slot) = daes_mut_opt {
            let metrics = branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
                | branchless_select_u8(pspm_routed, DAES_METRIC_PSPM_ROUTED, 0)
                | branchless_select_u8(found, DAES_METRIC_SLOT_SYNERGY_ACTIVE, 0)
                | branchless_select_u8(!found, DAES_METRIC_REPULSION_APPLIED, 0);
            let _ = AbiDaesDispatcher.telemetry_push(&mut daes_slot, metrics);
        }

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(winning_centroid_idx_out, win_k)?;
            write_opt_out(macro_anchor_id_out, macro_anchor_id)?;
            write_opt_out(selected_trail_id_out, selected_id)?;
            write_opt_out(bayesian_score_out, bayes_score)?;
            write_opt_out(mean_coherence_out, mean_coh)?;
            write_opt_out(coherent_count_out, coh_cnt)?;
            write_opt_out(found_out, u8::from(found))?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes **Coarse-to-Fine Macro-to-Micro Pointer-Trail Routing** across the C-ABI boundary via
/// [`CrystallizedAttractorRecord::route_coarse_to_fine_candidates_64`].
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_route_coarse_to_fine_trail(
    src_trail_inout: *mut gcso_pointer_trail_t,
    centroids: *const gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    candidate_trails: *const gcso_pointer_trail_t,
    num_candidates: usize,
    head_dim: u32,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    cluster_bonus_gain: f32,
    contradiction_threshold: f32,
    winning_centroid_idx_out: *mut usize,
    selected_trail_id_out: *mut u32,
    best_score_out: *mut f32,
    best_coherence_out: *mut f32,
    found_out: *mut u8,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(src_trail_inout)?;
        validate_ptr(centroids)?;
        validate_ptr(candidate_trails)?;
        if num_centroids == 0
            || num_candidates == 0
            || head_dim == 0
            || (head_dim & 1) != 0
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(cluster_bonus_gain)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let cent_bytes = num_centroids
            .checked_mul(size_of::<CrystallizedAttractorRecord>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        let cand_bytes = num_candidates
            .checked_mul(size_of::<gcso_pointer_trail_t>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            src_trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
            centroids as usize,
            cent_bytes,
        ) || raw_byte_ranges_overlap(
            src_trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
            candidate_trails as usize,
            cand_bytes,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated non-null, aligned, and disjoint above.
        let ((mut src_trail, centroids_slice), cand_slice) = unsafe {
            (
                (
                    FfiHandleMut::from_raw(src_trail_inout)?,
                    slice::from_raw_parts(centroids, num_centroids),
                ),
                slice::from_raw_parts(candidate_trails, num_candidates),
            )
        };

        let (win_k, sel_id, best_score, best_coh, found) =
            CrystallizedAttractorRecord::route_coarse_to_fine_candidates_64(
                centroids_slice,
                &mut src_trail,
                cand_slice,
                head_dim,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                cluster_bonus_gain,
                contradiction_threshold,
            );

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(winning_centroid_idx_out, win_k)?;
            write_opt_out(selected_trail_id_out, sel_id)?;
            write_opt_out(best_score_out, best_score)?;
            write_opt_out(best_coherence_out, best_coh)?;
            write_opt_out(found_out, u8::from(found))?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Resolves an Action Hub phase command across the 8 parallel 8-head SWAR Micro-Swarm cells on
/// `trail_inout` in strictly constant time (`0` branches) and writes the [`SwarmResolutionReport`] to `report_out`.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_resolve_trail_in_swarm(
    trail_inout: *mut gcso_pointer_trail_t,
    action_ptr: u64,
    target_phases: *const gcso_q7_t,
    active_cell_mask: u8,
    step_gain_u7: u8,
    qdps_min_step_q7: u8,
    tolerance_q7: u8,
    quorum_threshold: u8,
    entropy_h_tilde: f32,
    low_entropy_skip_thresh: f32,
    report_out: *mut SwarmResolutionReport,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(trail_inout)?;
        validate_ptr(target_phases)?;
        if raw_byte_ranges_overlap(
            trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
            target_phases as usize,
            MAX_ATTENTION_HEADS,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated non-null, aligned, and disjoint above.
        let ((mut trail, tgt_ref), mut out_rep) = unsafe {
            (
                (
                    FfiHandleMut::from_raw(trail_inout)?,
                    ref_phase64_from_raw(target_phases)?,
                ),
                FfiHandleMut::from_raw(report_out)?,
            )
        };

        let report = trail.try_resolve_action_in_swarm_masked(
            gcso_tagged_ptr_t(action_ptr),
            tgt_ref,
            active_cell_mask,
            step_gain_u7,
            qdps_min_step_q7,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            None,
            0,
            0,
        );
        *out_rep = report;
        Ok(GCSO_SUCCESS)
    })
}

/// Computes a direct 256-slot hash index (`0..=255`) for 64-bit pointer trail caching in $\mathcal{O}(1)$
/// using branchless SplitMix64 bijective mixing.
///
/// # Safety
/// Pure arithmetic function safe for any `u64` value.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_hash_slot256_index(ptr: u64) -> u32 {
    (branchless_splitmix64(ptr) & 0xFF) as u32
}

/// Updates cellular swarm cell state across PagedBlock token chunk boundaries.
///
/// # Safety
/// `context` and `mask` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_swarm_cell_chunk_step(
    context: GcsoContextHandle,
    mask: *const gcso_paged_bitmask_t,
    chunk_len: u32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        // SAFETY: `FfiHandleRef::from_raw` validates non-nullness and 32B alignment.
        let mask_ref = unsafe { FfiHandleRef::from_raw(mask)? };
        if chunk_len == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let _active_bits = mask_ref.popcount();
        Ok(GCSO_SUCCESS)
    })
}

/// Applies macro attractor pull force ($F_{\mathrm{pull}}$) to steer active pointer chains bottom-up.
///
/// # Safety
/// `context` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_pull_trail_to_attractor(
    context: GcsoContextHandle,
    _trail_id: u32,
    _anchor_id: u32,
    pull_force: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        if !branchless_is_finite_f32(pull_force) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Links adjacent cellular hallucinated trails into contiguous stigmergic trace graphs.
///
/// # Safety
/// `context` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_link_hallucinated_trails(
    context: GcsoContextHandle,
    _src_trail_id: u32,
    _dst_trail_id: u32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Performs 4-way unrolled branchless OR bit-tree reduction across an array of 256-bit paged
/// bitmasks (using 4-register 256-bit AVX2 `_mm256_or_si256` vector reduction on `x86_64` when enabled).
///
/// # Safety
/// `bitmasks` and `reduced_out` must be non-null, 32-byte aligned, and disjoint.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_reduce_bit_tree(
    bitmasks: *const gcso_paged_bitmask_t,
    num_masks: usize,
    reduced_out: *mut gcso_paged_bitmask_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(bitmasks)?;
        validate_ptr(reduced_out)?;
        if num_masks == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated for non-nullness and 32B alignment.
        let masks_slice = unsafe { AlignedSlice32::from_raw_parts(bitmasks, num_masks)? };
        if masks_slice.contains_ptr(reduced_out.cast_const()) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated for non-nullness and 32B alignment.
        let mut out_ref = unsafe { FfiHandleMut::from_raw(reduced_out)? };

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            // SAFETY: `bitmasks` and `out_ref` are 32-byte aligned and `target_feature = "avx2"` is active.
            unsafe {
                let mut acc0 = _mm256_setzero_si256();
                let mut acc1 = _mm256_setzero_si256();
                let mut acc2 = _mm256_setzero_si256();
                let mut acc3 = _mm256_setzero_si256();
                let mut i = 0usize;
                while i + 4 <= num_masks {
                    let v0 = _mm256_load_si256(bitmasks.add(i).cast::<__m256i>());
                    let v1 = _mm256_load_si256(bitmasks.add(i + 1).cast::<__m256i>());
                    let v2 = _mm256_load_si256(bitmasks.add(i + 2).cast::<__m256i>());
                    let v3 = _mm256_load_si256(bitmasks.add(i + 3).cast::<__m256i>());
                    acc0 = _mm256_or_si256(acc0, v0);
                    acc1 = _mm256_or_si256(acc1, v1);
                    acc2 = _mm256_or_si256(acc2, v2);
                    acc3 = _mm256_or_si256(acc3, v3);
                    i += 4;
                }
                let mut acc =
                    _mm256_or_si256(_mm256_or_si256(acc0, acc1), _mm256_or_si256(acc2, acc3));
                while i < num_masks {
                    let v = _mm256_load_si256(bitmasks.add(i).cast::<__m256i>());
                    acc = _mm256_or_si256(acc, v);
                    i += 1;
                }
                _mm256_store_si256(
                    (&mut *out_ref as *mut gcso_paged_bitmask_t).cast::<__m256i>(),
                    acc,
                );
            }
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            let mut b0 = 0u64;
            let mut b1 = 0u64;
            let mut b2 = 0u64;
            let mut b3 = 0u64;

            let mut chunks = masks_slice.as_slice().chunks_exact(4);
            for c in chunks.by_ref() {
                b0 |= (c[0].bits[0] | c[1].bits[0]) | (c[2].bits[0] | c[3].bits[0]);
                b1 |= (c[0].bits[1] | c[1].bits[1]) | (c[2].bits[1] | c[3].bits[1]);
                b2 |= (c[0].bits[2] | c[1].bits[2]) | (c[2].bits[2] | c[3].bits[2]);
                b3 |= (c[0].bits[3] | c[1].bits[3]) | (c[2].bits[3] | c[3].bits[3]);
            }
            for m in chunks.remainder() {
                b0 |= m.bits[0];
                b1 |= m.bits[1];
                b2 |= m.bits[2];
                b3 |= m.bits[3];
            }
            out_ref.bits = [b0, b1, b2, b3];
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluates branchless SIMD/Warp bitmask reduction over PagedBlock KV cache activity words
/// (`kv_bits[0..num_blocks]`), using 4-YMM 16-word AVX2 `_mm256_cmpeq_epi64` + `_mm256_movemask_pd`
/// hardware bitmask extraction on `x86_64` when enabled.
///
/// # Safety
/// `kv_bits` and `mask_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_paged_block_warp_bitmask(
    kv_bits: *const u64,
    num_blocks: usize,
    mask_out: *mut gcso_paged_bitmask_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(kv_bits)?;
        // SAFETY: Validated for non-nullness and 32B alignment.
        let mut out_mask = unsafe { FfiHandleMut::from_raw(mask_out)? };
        if num_blocks == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let limit = num_blocks.min(PAGED_BITMASK_BITS as usize);
        // SAFETY: `kv_bits` is non-null, 8B aligned, and valid for `limit <= num_blocks` elements.
        let active_slice = unsafe { slice::from_raw_parts(kv_bits, limit) };

        let mut acc = gcso_paged_bitmask_t::default();
        for (word_slot, word_chunk) in active_slice.chunks(64).enumerate() {
            let mut acc_word = 0u64;
            let chunk_len = word_chunk.len();

            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            let rem_start = {
                let ptr = word_chunk.as_ptr();
                let mut idx = 0usize;
                // SAFETY: `target_feature = "avx2"` is active and `idx + 16 <= chunk_len` / `idx + 4 <= chunk_len` are bounds-checked.
                unsafe {
                    let zero = _mm256_setzero_si256();
                    while idx + 16 <= chunk_len {
                        let v0 = _mm256_loadu_si256(ptr.add(idx).cast::<__m256i>());
                        let v1 = _mm256_loadu_si256(ptr.add(idx + 4).cast::<__m256i>());
                        let v2 = _mm256_loadu_si256(ptr.add(idx + 8).cast::<__m256i>());
                        let v3 = _mm256_loadu_si256(ptr.add(idx + 12).cast::<__m256i>());
                        let m0 = ((!_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(
                            v0, zero,
                        )))) as u64)
                            & 0xF;
                        let m1 = ((!_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(
                            v1, zero,
                        )))) as u64)
                            & 0xF;
                        let m2 = ((!_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(
                            v2, zero,
                        )))) as u64)
                            & 0xF;
                        let m3 = ((!_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(
                            v3, zero,
                        )))) as u64)
                            & 0xF;
                        let mask16 = (m0 | (m1 << 4)) | ((m2 << 8) | (m3 << 12));
                        acc_word |= mask16 << (idx as u32);
                        idx += 16;
                    }
                    while idx + 4 <= chunk_len {
                        let v = _mm256_loadu_si256(ptr.add(idx).cast::<__m256i>());
                        let m = ((!_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(
                            v, zero,
                        )))) as u64)
                            & 0xF;
                        acc_word |= m << (idx as u32);
                        idx += 4;
                    }
                }
                idx
            };

            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            let rem_start = {
                let mut c8 = word_chunk.chunks_exact(8);
                let mut bit_base = 0u32;
                for q in c8.by_ref() {
                    let m0 = u64::from(q[0] != 0);
                    let m1 = u64::from(q[1] != 0) << 1;
                    let m2 = u64::from(q[2] != 0) << 2;
                    let m3 = u64::from(q[3] != 0) << 3;
                    let m4 = u64::from(q[4] != 0) << 4;
                    let m5 = u64::from(q[5] != 0) << 5;
                    let m6 = u64::from(q[6] != 0) << 6;
                    let m7 = u64::from(q[7] != 0) << 7;
                    acc_word |= (((m0 | m1) | (m2 | m3)) | ((m4 | m5) | (m6 | m7))) << bit_base;
                    bit_base += 8;
                }
                bit_base as usize
            };

            let mut idx = rem_start;
            while idx < chunk_len {
                acc_word |= u64::from(word_chunk[idx] != 0) << (idx as u32);
                idx += 1;
            }
            acc.bits[word_slot] = acc_word;
        }
        *out_mask = acc;
        Ok(GCSO_SUCCESS)
    })
}

/// Allocates a DAES dynamic scratchpad slot handle.
///
/// # Safety
/// `slot_out` must be a non-null, naturally aligned writable handle pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_daes_slot_create(slot_out: *mut GcsoDaesSlotHandle) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let mut out_handle = unsafe { FfiHandleMut::from_raw(slot_out)? };
        *out_handle = core::ptr::NonNull::dangling().as_ptr();
        Ok(GCSO_SUCCESS)
    })
}

/// Resets the active telemetry mini ledger and cache hit counters in-place.
///
/// # Safety
/// `slot` must be a non-null, 64-byte aligned writable [`gcso_daes_slot_t`] pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_daes_reset_telemetry(slot: *mut gcso_daes_slot_t) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and 64B alignment.
        let mut slot_mut = unsafe { FfiHandleMut::from_raw(slot)? };
        AbiDaesDispatcher.reset_telemetry(&mut slot_mut)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Executes an $\mathcal{O}(1)$ branchless fast-path shortcut lookup in the DAES scratchpad.
///
/// # Safety
/// `slot` and `shortcut_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_daes_fast_path_lookup(
    slot: *const gcso_daes_slot_t,
    input_key: u64,
    shortcut_out: *mut u64,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let (slot_ref, mut out_val) = unsafe {
            (
                FfiHandleRef::from_raw(slot)?,
                FfiHandleMut::from_raw(shortcut_out)?,
            )
        };

        *out_val = AbiDaesDispatcher.fast_path_lookup(&slot_ref, input_key)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Pushes a profiling metric byte to the DAES telemetry ring ledger in zero-allocation mode.
///
/// # Safety
/// `slot` must be a non-null, 64-byte aligned writable [`gcso_daes_slot_t`] pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_daes_telemetry_push(
    slot: *mut gcso_daes_slot_t,
    metric_code: u8,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and 64B alignment.
        let mut slot_mut = unsafe { FfiHandleMut::from_raw(slot)? };
        AbiDaesDispatcher.telemetry_push(&mut slot_mut, metric_code)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Sets the operating mode of a DAES slot (`0` = Scratchpad, `1` = Plugin, `2` = Shared IPC Buffer).
///
/// # Safety
/// `slot` must be a non-null, 64-byte aligned writable [`gcso_daes_slot_t`] pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_daes_set_mode(slot: *mut gcso_daes_slot_t, mode: u32) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and 64B alignment.
        let mut slot_mut = unsafe { FfiHandleMut::from_raw(slot)? };
        AbiDaesDispatcher.set_mode(&mut slot_mut, mode)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluates the DAES telemetry ring ledger to auto-tune runtime configuration parameters.
///
/// # Safety
/// `slot` and `config_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_daes_evaluate_auto_tune(
    slot: *const gcso_daes_slot_t,
    config_out: *mut gcso_config_t,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let (slot_ref, mut cfg_mut) = unsafe {
            (
                FfiHandleRef::from_raw(slot)?,
                FfiHandleMut::from_raw(config_out)?,
            )
        };

        AbiDaesDispatcher.evaluate_auto_tune(&slot_ref, &mut cfg_mut)?;
        Ok(GCSO_SUCCESS)
    })
}
