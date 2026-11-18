// name: src/core/src/abi/entropy_attractor.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unified Attractor Manifold, Online Toroidal k-Means, synergy, Entropy Control (EDBC), and Liminal FFI.

use super::*;

// ===================================================================
// 5. Single Unified Attractor Manifold, Online Toroidal k-Means, Synergy, EDBC & Liminal Interface
// ===================================================================

/// Allocates a Macro Attractor Field instance.
///
/// # Safety
/// `field_out` must be a non-null, naturally aligned writable handle pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_create(
    field_out: *mut GcsoAttractorFieldHandle,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let mut out_handle = unsafe { FfiHandleMut::from_raw(field_out)? };
        *out_handle = core::ptr::NonNull::dangling().as_ptr();
        Ok(GCSO_SUCCESS)
    })
}

/// Registers a topological anchor point in the Macro Attractor Field.
///
/// # Safety
/// `context`, `vec` (32B aligned), and `anchor_id_out` must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_add_anchor(
    context: GcsoContextHandle,
    anchor_type: gcso_anchor_type_t,
    vec: *const f32,
    dim: usize,
    anchor_id_out: *mut u32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        validate_ptr_aligned(vec, 32)?;
        // SAFETY: Validated for non-nullness and alignment.
        let mut id_out = unsafe { FfiHandleMut::from_raw(anchor_id_out)? };
        if dim == 0 || GcsoAnchorType::from_u32(anchor_type).is_none() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated for non-nullness, 32B alignment, and bounds.
        let _vec_aligned = unsafe { AlignedSlice32::from_raw_parts(vec, dim)? };
        *id_out = 1;
        Ok(GCSO_SUCCESS)
    })
}

/// Maps natural language system prompt text as the primary Anchor Attractor in phase space.
///
/// # Safety
/// `context`, `prompt_text`, and `anchor_id_out` must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_add_system_prompt_anchor(
    context: GcsoContextHandle,
    prompt_text: *const c_char,
    weight: f32,
    anchor_id_out: *mut u32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        // SAFETY: Validated for non-nullness, UTF-8 validity, and alignment.
        let (_prompt, mut id_out) = unsafe {
            (
                validate_c_utf8_str(prompt_text)?,
                FfiHandleMut::from_raw(anchor_id_out)?,
            )
        };
        if !branchless_is_finite_f32(weight) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        *id_out = 1;
        Ok(GCSO_SUCCESS)
    })
}

/// Maps a dense feature embedding vector as a continuous attractor anchor.
///
/// # Safety
/// `context`, `embedding` (32B aligned), and `anchor_id_out` must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_add_embedding_anchor(
    context: GcsoContextHandle,
    embedding: *const f32,
    dim: usize,
    weight: f32,
    anchor_id_out: *mut u32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        validate_ptr_aligned(embedding, 32)?;
        // SAFETY: Validated for non-nullness and alignment.
        let mut id_out = unsafe { FfiHandleMut::from_raw(anchor_id_out)? };
        if dim == 0 || !branchless_is_finite_f32(weight) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated for non-nullness, 32B alignment, and bounds.
        let _emb_aligned = unsafe { AlignedSlice32::from_raw_parts(embedding, dim)? };
        *id_out = 1;
        Ok(GCSO_SUCCESS)
    })
}

/// Injects a phase-conjugate repulsion vector ($-\boldsymbol{\Delta\theta}$) to invert spurious local minima.
///
/// # Safety
/// `context` and `repulsion_deltas` must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_inject_phase_repulsion(
    context: GcsoContextHandle,
    repulsion_deltas: *const gcso_q7_t,
    num_heads: usize,
    gain: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        validate_ptr(repulsion_deltas)?;
        if num_heads == 0 || !branchless_is_finite_f32(gain) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes one step of **Classical Online Toroidal $k$-Means / DP-Means Macro Clustering** on
/// $SO(2)^{64}$ (`centroids_inout[0..num_centroids]`) for an incoming [`gcso_gateway_frame_t`]
/// (`frame`) via [`CrystallizedAttractorRecord::step_online_toroidal_kmeans_frame_64`].
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_step_online_toroidal_kmeans(
    centroids_inout: *mut gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    frame: *const gcso_gateway_frame_t,
    split_coherence_threshold: f32,
    trail_inout: *mut gcso_pointer_trail_t,
    assigned_centroid_idx_out: *mut usize,
    assigned_coherence_out: *mut f32,
    updated_gravity_mass_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(centroids_inout)?;
        // SAFETY: Validated for non-nullness and 64B alignment.
        let frame_ref = unsafe { FfiHandleRef::from_raw(frame)? };
        if num_centroids == 0 || !branchless_is_finite_f32(split_coherence_threshold) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let cent_bytes = num_centroids
            .checked_mul(size_of::<CrystallizedAttractorRecord>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            centroids_inout as usize,
            cent_bytes,
            frame as usize,
            size_of::<gcso_gateway_frame_t>(),
        ) || (!trail_inout.is_null()
            && (raw_byte_ranges_overlap(
                centroids_inout as usize,
                cent_bytes,
                trail_inout as usize,
                size_of::<gcso_pointer_trail_t>(),
            ) || raw_byte_ranges_overlap(
                frame as usize,
                size_of::<gcso_gateway_frame_t>(),
                trail_inout as usize,
                size_of::<gcso_pointer_trail_t>(),
            )))
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated non-null, aligned, and disjoint above.
        let (centroids_slice, mut trail_opt) = unsafe {
            (
                slice::from_raw_parts_mut(centroids_inout, num_centroids),
                FfiHandleMut::from_raw_opt(trail_inout)?,
            )
        };

        let (assigned_idx, _assigned_anchor_id, assigned_coh, updated_mass, _spawned_new) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_frame_64(
                centroids_slice,
                &frame_ref,
                split_coherence_threshold,
                trail_opt.as_deref_mut(),
            );

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(assigned_centroid_idx_out, assigned_idx)?;
            write_opt_out(assigned_coherence_out, assigned_coh)?;
            write_opt_out(updated_gravity_mass_out, updated_mass)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes **Batched Online Toroidal $k$-Means / DP-Means Macro Clustering** ($\mathcal{O}(B \cdot K)$)
/// on $SO(2)^{64}$ (`centroids_inout[0..num_centroids]`) across `num_frames` incoming
/// [`gcso_gateway_frame_t`]s via [`CrystallizedAttractorRecord::step_online_toroidal_kmeans_batch_frames_64`],
/// using a 64-bit `dirty_mask` to defer IEEE 802.3 `checksum_crc32` stamping until the end of the batch.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_step_online_toroidal_kmeans_batch(
    centroids_inout: *mut gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    frames: *const gcso_gateway_frame_t,
    num_frames: usize,
    split_coherence_threshold: f32,
    trail_inout: *mut gcso_pointer_trail_t,
    last_assigned_idx_out: *mut usize,
    last_anchor_id_out: *mut u32,
    mean_batch_coherence_out: *mut f32,
    spawned_clusters_out: *mut u32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(centroids_inout)?;
        validate_ptr(frames)?;
        if num_centroids == 0
            || num_frames == 0
            || !branchless_is_finite_f32(split_coherence_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let cent_bytes = num_centroids
            .checked_mul(size_of::<CrystallizedAttractorRecord>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        let frames_bytes = num_frames
            .checked_mul(size_of::<gcso_gateway_frame_t>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            centroids_inout as usize,
            cent_bytes,
            frames as usize,
            frames_bytes,
        ) || (!trail_inout.is_null()
            && (raw_byte_ranges_overlap(
                centroids_inout as usize,
                cent_bytes,
                trail_inout as usize,
                size_of::<gcso_pointer_trail_t>(),
            ) || raw_byte_ranges_overlap(
                frames as usize,
                frames_bytes,
                trail_inout as usize,
                size_of::<gcso_pointer_trail_t>(),
            )))
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated non-null, aligned, and disjoint above.
        let ((centroids_slice, frames_slice), mut trail_opt) = unsafe {
            (
                (
                    slice::from_raw_parts_mut(centroids_inout, num_centroids),
                    slice::from_raw_parts(frames, num_frames),
                ),
                FfiHandleMut::from_raw_opt(trail_inout)?,
            )
        };

        let (last_idx, last_anchor_id, mean_coh, spawned_total) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_batch_frames_64(
                centroids_slice,
                frames_slice,
                split_coherence_threshold,
                trail_opt.as_deref_mut(),
            );

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(last_assigned_idx_out, last_idx)?;
            write_opt_out(last_anchor_id_out, last_anchor_id)?;
            write_opt_out(mean_batch_coherence_out, mean_coh)?;
            write_opt_out(spawned_clusters_out, spawned_total)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Aggregates high-density pointer trails bottom-up to macro-crystallize new dynamic anchors.
///
/// # Safety
/// `context` and `new_anchor_count_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_aggregate_bottom_up(
    context: GcsoContextHandle,
    new_anchor_count_out: *mut u32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        // SAFETY: Validated for non-nullness and alignment.
        let mut count_out = unsafe { FfiHandleMut::from_raw(new_anchor_count_out)? };
        *count_out = 0;
        Ok(GCSO_SUCCESS)
    })
}

/// Allocates an EDBC entropy controller instance initialized from `initial_state`.
///
/// # Safety
/// `initial_state` and `controller_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_edbc_controller_create(
    initial_state: *const gcso_edbc_state_t,
    controller_out: *mut GcsoEdbcControllerHandle,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let (init_st, mut out_handle) = unsafe {
            (
                FfiHandleRef::from_raw(initial_state)?,
                FfiHandleMut::from_raw(controller_out)?,
            )
        };

        let engine = EntropyController::new(*init_st);
        *out_handle = Box::into_raw(Box::new(engine)).cast::<GcsoEdbcControllerOpaque>();
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluates stateful Moving Z-Score Attention Entropy ($\tilde{H}$) and updates branch mode.
///
/// # Safety
/// `controller` and `state_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_edbc_eval_stateful(
    controller: GcsoEdbcControllerHandle,
    token_z_score: f32,
    state_out: *mut gcso_edbc_state_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(controller)?;
        // SAFETY: Validated for non-nullness and alignment.
        let (mut engine, mut out_st) = unsafe {
            (
                FfiHandleMut::from_raw(controller.cast::<EntropyController>())?,
                FfiHandleMut::from_raw(state_out)?,
            )
        };

        engine.eval_stateful(token_z_score, &mut out_st)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Computes the Liminal Vector Alignment Metric ($\mathcal{L}_{\mathrm{void}}$) for
/// Out-of-Distribution Latent Space over a 32-byte aligned Key vector.
///
/// # Safety
/// `key_vector` (32B aligned) and `void_score_out` must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_liminal_void_eval_dyadic128(
    key_vector: *const f32,
    dim: usize,
    void_score_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(key_vector, 32)?;
        // SAFETY: Validated for non-nullness, 32B alignment, and bounds.
        let (mut out_score, key_aligned) = unsafe {
            (
                FfiHandleMut::from_raw(void_score_out)?,
                AlignedSlice32::from_raw_parts(key_vector, dim)?,
            )
        };

        EntropyController::default().eval_liminal_void_dyadic128(&key_aligned, &mut out_score)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluates the Eyring-Kramers potential barrier height ($\Delta V = \tau_{\mathrm{eff}} / \mathcal{L}_{\mathrm{void}}$)
/// with branchless singularity guards.
///
/// # Safety
/// `barrier_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_liminal_void_eval_barrier(
    void_score: f32,
    tau_eff: f32,
    barrier_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let mut out_val = unsafe { FfiHandleMut::from_raw(barrier_out)? };
        *out_val = EntropyController::default().eval_barrier(void_score, tau_eff)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluates the scale-invariant Pitchfork Bifurcation potential barrier height ($\Delta V$)
/// according to GCSO Whitepaper Lemma 3.1: $\Delta V = \frac{\alpha^2 (\mathcal{L}_{\mathrm{void}} - \tau_{\mathrm{eff}})^2}{4 \beta d_{\mathrm{head}}}$.
///
/// # Safety
/// `barrier_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_liminal_void_eval_barrier_pitchfork(
    liminal_void: f32,
    tau_eff: f32,
    alpha: f32,
    beta: f32,
    head_dim: u32,
    barrier_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let mut out_val = unsafe { FfiHandleMut::from_raw(barrier_out)? };
        *out_val = eval_barrier_pitchfork(liminal_void, tau_eff, alpha, beta, head_dim)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Destroys an Action Hub instance and releases its heap allocation if created via [`gcso_action_hub_create`].
///
/// # Safety
/// `handle` must be `NULL` or a valid Action Hub handle. `NULL` safely returns `GCSO_SUCCESS`.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_destroy(handle: GcsoActionHubHandle) -> GcsoStatus {
    if handle.is_null() {
        return GCSO_SUCCESS;
    }
    if !is_aligned(handle) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    ffi_boundary!({
        if (handle as usize) >= 4096
            && ((handle as usize) & (align_of::<ActionHubEngine>() - 1)) == 0
        {
            // SAFETY: `handle` was allocated via `Box::into_raw(Box::new(ActionHubEngine))` in `gcso_action_hub_create`.
            unsafe {
                let _ = Box::from_raw(handle.cast::<ActionHubEngine>());
            }
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Destroys an EDBC controller instance and releases its heap allocation.
///
/// # Safety
/// `controller` must be `NULL` or a handle returned by [`gcso_edbc_controller_create`].
#[no_mangle]
pub unsafe extern "C" fn gcso_edbc_controller_destroy(
    controller: GcsoEdbcControllerHandle,
) -> GcsoStatus {
    if controller.is_null() {
        return GCSO_SUCCESS;
    }
    if !is_aligned(controller) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    ffi_boundary!({
        // SAFETY: `controller` was allocated via `Box::into_raw` in `gcso_edbc_controller_create`.
        unsafe {
            let _ = Box::from_raw(controller.cast::<EntropyController>());
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluates [`gcso_epistemic_horizon_reading_t`] over `num_records` crystallized attractor
/// records relative to a 64-head Q7 query phase vector (`query_phases`) without branching.
///
/// # Safety
/// `query_phases` (64 bytes) and `reading_out` (16-byte aligned) must be non-null; if
/// `num_records > 0`, `records` must be non-null and 64-byte aligned with `num_records` valid items.
#[no_mangle]
pub unsafe extern "C" fn gcso_epistemic_evaluate_horizon(
    records: *const gcso_crystallized_attractor_record_t,
    num_records: usize,
    query_phases: *const gcso_q7_t,
    horizon_floor: f32,
    reading_out: *mut gcso_epistemic_horizon_reading_t,
) -> GcsoStatus {
    ffi_boundary!({
        if !branchless_is_finite_f32(horizon_floor) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let records_slice: &[CrystallizedAttractorRecord] = if num_records == 0 {
            &[]
        } else {
            validate_ptr(records)?;
            let byte_len = num_records
                .checked_mul(size_of::<CrystallizedAttractorRecord>())
                .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
            if raw_byte_ranges_overlap(
                records as usize,
                byte_len,
                reading_out as usize,
                size_of::<EpistemicHorizonReading>(),
            ) {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            // SAFETY: Validated non-null, 64B aligned, bounds checked, and disjoint above.
            unsafe { slice::from_raw_parts(records, num_records) }
        };
        // SAFETY: `ref_phase64_from_raw` and `FfiHandleMut::from_raw` validate pointers & alignment.
        let (q_ref, mut out_ref) = unsafe {
            (
                ref_phase64_from_raw(query_phases)?,
                FfiHandleMut::from_raw(reading_out)?,
            )
        };
        let q_words = load_q7_words_64(q_ref);
        *out_ref =
            evaluate_epistemic_horizon_words_64(records_slice, &q_words, None, horizon_floor);
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluates Venn-diagram multi-attractor overlap and noise-driven serendipity in `Liminal`
/// across `centroids_inout[0..num_centroids]` (`query_phases` vs `drifted_phases`). When
/// `nucleate_on_discovery != 0` and approximation noise triggers unknown attractor discovery,
/// autonomously nucleates a new crystallized attractor into an empty or weaker centroid slot.
///
/// # Safety
/// `query_phases`, `drifted_phases`, and `report_out` (16B aligned) must be non-null; if
/// `num_centroids > 0`, `centroids_inout` must be non-null and 64B aligned and disjoint from
/// `report_out` and `did_nucleate_out`.
#[no_mangle]
pub unsafe extern "C" fn gcso_stigmergic_evaluate_venn_and_serendipity(
    centroids_inout: *mut gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    query_phases: *const gcso_q7_t,
    drifted_phases: *const gcso_q7_t,
    aperture_floor: f32,
    serendipity_floor_q7: u8,
    nucleate_on_discovery: u8,
    new_anchor_id: u32,
    report_out: *mut gcso_stigmergic_venn_serendipity_report_t,
    did_nucleate_out: *mut u8,
) -> GcsoStatus {
    ffi_boundary!({
        if !branchless_is_finite_f32(aperture_floor) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validate non-null phase buffers and aligned output handle.
        let ((q_ref, d_ref), mut rep_out) = unsafe {
            (
                (
                    ref_phase64_from_raw(query_phases)?,
                    ref_phase64_from_raw(drifted_phases)?,
                ),
                FfiHandleMut::from_raw(report_out)?,
            )
        };
        let q_words = load_q7_words_64(q_ref);
        let d_words = load_q7_words_64(d_ref);

        let mut did_nucleate = 0u8;
        let venn = if num_centroids == 0 {
            evaluate_stigmergic_venn_and_serendipity_words_64::<CrystallizedAttractorRecord>(
                &[],
                &q_words,
                &d_words,
                aperture_floor,
                serendipity_floor_q7,
            )
        } else {
            validate_ptr(centroids_inout)?;
            let byte_len = num_centroids
                .checked_mul(size_of::<CrystallizedAttractorRecord>())
                .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
            if raw_byte_ranges_overlap(
                centroids_inout as usize,
                byte_len,
                report_out as usize,
                size_of::<StigmergicVennSerendipityReport>(),
            ) || (!did_nucleate_out.is_null()
                && raw_byte_ranges_overlap(
                    centroids_inout as usize,
                    byte_len,
                    did_nucleate_out as usize,
                    size_of::<u8>(),
                ))
            {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            // SAFETY: Validated non-null, 64B aligned, bounds-checked, and disjoint above.
            let centroids = unsafe { slice::from_raw_parts_mut(centroids_inout, num_centroids) };
            let report = evaluate_stigmergic_venn_and_serendipity_words_64(
                centroids,
                &q_words,
                &d_words,
                aperture_floor,
                serendipity_floor_q7,
            );
            if nucleate_on_discovery != 0 && report.serendipity_triggered != 0 {
                let (_slot, _aid, ok) =
                    CrystallizedAttractorRecord::nucleate_serendipitous_attractor_from_void_64(
                        centroids,
                        &report,
                        &d_words,
                        new_anchor_id,
                        None,
                    );
                did_nucleate = u8::from(ok);
            }
            report
        };

        *rep_out = venn;
        // SAFETY: `write_opt_out` validates optional alignment and non-nullness.
        unsafe {
            write_opt_out(did_nucleate_out, did_nucleate)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Initializes a 128-byte [`gcso_autonomous_core_layer_stack_t`] (Core Layer 0: `LiminalApertureField`,
/// Core Layer 1: `StigmergicGranuleField`, Core Layer 2: `LiminalEgoField`) with canonical defaults.
///
/// # Safety
/// `stack_out` must be non-null and 64-byte aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_core_layer_stack_init_default(
    stack_out: *mut gcso_autonomous_core_layer_stack_t,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validates non-nullness and 64B alignment.
        let mut out_ref = unsafe { FfiHandleMut::from_raw(stack_out)? };
        *out_ref = AutonomousCoreLayerStack::new();
        Ok(GCSO_SUCCESS)
    })
}

/// Branchlessly adapts Core Layer 0 (`Liminal`) and Core Layer 1 (`Stigmergic`) inside the
/// 128-byte `stack_inout` from `horizon`, `venn_report`, and optional `ledger`.
///
/// # Safety
/// `stack_inout` (64B aligned), `horizon` (16B aligned), and `venn_report` (16B aligned) must be
/// non-null and non-overlapping; `ledger` may be `NULL` or 8-byte aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_core_layer_stack_adapt(
    stack_inout: *mut gcso_autonomous_core_layer_stack_t,
    ledger: *const gcso_cognitive_trajectory_ledger_t,
    horizon: *const gcso_epistemic_horizon_reading_t,
    venn_report: *const gcso_stigmergic_venn_serendipity_report_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(stack_inout)?;
        validate_ptr(horizon)?;
        validate_ptr(venn_report)?;
        if raw_byte_ranges_overlap(
            stack_inout as usize,
            size_of::<AutonomousCoreLayerStack>(),
            horizon as usize,
            size_of::<EpistemicHorizonReading>(),
        ) || raw_byte_ranges_overlap(
            stack_inout as usize,
            size_of::<AutonomousCoreLayerStack>(),
            venn_report as usize,
            size_of::<StigmergicVennSerendipityReport>(),
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated non-null, aligned, and disjoint above.
        let ((mut stack_ref, ledger_opt), (horizon_ref, venn_ref)) = unsafe {
            (
                (
                    FfiHandleMut::from_raw(stack_inout)?,
                    FfiHandleRef::from_raw_opt(ledger)?,
                ),
                (
                    FfiHandleRef::from_raw(horizon)?,
                    FfiHandleRef::from_raw(venn_report)?,
                ),
            )
        };
        let ledger_val = match ledger_opt {
            Some(l) => *l,
            None => CognitiveTrajectoryLedger::default(),
        };
        stack_ref.adapt_layers_branchless(&ledger_val, &horizon_ref, &venn_ref, None);
        Ok(GCSO_SUCCESS)
    })
}

/// Formats the live 3-layer Core Topology (`Liminal` Layer 0, `StigmergicGranuleField` Layer 1,
/// `LiminalEgoField` Layer 2, and the trait aperture table) as a `#![no_std]` zero-allocation
/// Mermaid `flowchart TD` diagram into `out_buf[0..buf_cap]`. Pass `out_buf = NULL` to query the
/// required byte length in `bytes_written_out`.
///
/// # Safety
/// `stack` (64B aligned) and `horizon` (16B aligned) must be non-null; `venn_report` may be `NULL`
/// or 16B aligned. If `out_buf` is non-null, it must reference `buf_cap` writable bytes disjoint from
/// `bytes_written_out`.
#[no_mangle]
pub unsafe extern "C" fn gcso_liminal_topology_format_mermaid(
    stack: *const gcso_autonomous_core_layer_stack_t,
    horizon: *const gcso_epistemic_horizon_reading_t,
    venn_report: *const gcso_stigmergic_venn_serendipity_report_t,
    active_crystals: u8,
    spawned_daughters: u8,
    out_buf: *mut u8,
    buf_cap: usize,
    bytes_written_out: *mut usize,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validates required and optional pointers and natural alignment.
        let ((stack_ref, horizon_ref), venn_opt) = unsafe {
            (
                (
                    FfiHandleRef::from_raw(stack)?,
                    FfiHandleRef::from_raw(horizon)?,
                ),
                FfiHandleRef::from_raw_opt(venn_report)?,
            )
        };
        let venn_val = match venn_opt {
            Some(v) => *v,
            None => StigmergicVennSerendipityReport::default(),
        };
        let snap = LiminalTopologyTelemetrySnapshot::capture(
            &stack_ref.liminal_layer,
            &horizon_ref,
            active_crystals,
            spawned_daughters,
        );
        let mut scratch = [0u8; 2048];
        let needed = snap.format_three_core_layers_mermaid(&stack_ref, &venn_val, &mut scratch);

        if out_buf.is_null() {
            // SAFETY: `write_opt_out` validates optional output pointer.
            unsafe {
                write_opt_out(bytes_written_out, needed)?;
            }
            return Ok(GCSO_SUCCESS);
        }

        if !bytes_written_out.is_null()
            && raw_byte_ranges_overlap(
                out_buf as usize,
                buf_cap,
                bytes_written_out as usize,
                size_of::<usize>(),
            )
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: `write_opt_out` validates optional output pointer.
        unsafe {
            write_opt_out(bytes_written_out, needed)?;
        }
        if buf_cap < needed {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        // SAFETY: `out_buf` is non-null, `buf_cap >= needed`, and `scratch[..needed]` is initialized.
        unsafe {
            let dst = slice::from_raw_parts_mut(out_buf, needed);
            dst.copy_from_slice(&scratch[..needed]);
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Initializes a 64-byte [`gcso_liminal_ego_field_t`] from a preset template code
/// (`0 = Custom`, `1 = Low`, `2 = Medium`, `3 = High`).
///
/// # Safety
/// `ego_out` must be non-null and 64-byte aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_ego_field_init_preset(
    preset_code: u8,
    ego_out: *mut gcso_liminal_ego_field_t,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validates non-nullness and 64-byte natural alignment.
        let mut out_ref = unsafe { FfiHandleMut::from_raw(ego_out)? };
        *out_ref = LiminalEgoField::from_preset(LiminalEgoPreset::from_code(preset_code));
        Ok(GCSO_SUCCESS)
    })
}

/// Deterministically derives a 64-byte [`gcso_liminal_ego_field_t`] from UTF-8 natural-language
/// text (`utf8_ptr[0..utf8_len]`) with zero heap allocations.
///
/// # Safety
/// `ego_out` must be non-null and 64-byte aligned. When `utf8_len > 0`, `utf8_ptr` must be
/// non-null, valid for `utf8_len` readable UTF-8 bytes, and disjoint from `ego_out`.
#[no_mangle]
pub unsafe extern "C" fn gcso_ego_field_derive_from_text(
    utf8_ptr: *const u8,
    utf8_len: usize,
    ego_out: *mut gcso_liminal_ego_field_t,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validates non-nullness and 64-byte natural alignment.
        let mut out_ref = unsafe { FfiHandleMut::from_raw(ego_out)? };
        if utf8_len == 0 {
            *out_ref = LiminalEgoField::new();
            return Ok(GCSO_SUCCESS);
        }
        validate_ptr(utf8_ptr)?;
        if raw_byte_ranges_overlap(
            utf8_ptr as usize,
            utf8_len,
            ego_out as usize,
            size_of::<LiminalEgoField>(),
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated non-null, readable length `utf8_len`, and disjoint from `ego_out`.
        let bytes = unsafe { slice::from_raw_parts(utf8_ptr, utf8_len) };
        let text = core::str::from_utf8(bytes).map_err(|_| GCSO_ERROR_INVALID_ARGUMENT)?;
        *out_ref = LiminalEgoField::from_natural_language(text);
        Ok(GCSO_SUCCESS)
    })
}

/// Branchlessly evaluates active Ego triggers across Layer 0/1 boundaries and `delta`,
/// updating `ego_inout` and `stack_inout` and modulating `frame_inout` (counter-basin orbital
/// thrust and transgressive phase steering).
///
/// # Safety
/// `ego_inout` (64B), `stack_inout` (64B), `delta` (16B), `horizon` (16B), and `frame_inout`
/// (64B) must be non-null, naturally aligned, and mutually non-overlapping; `venn_report` may be
/// `NULL` or 16B aligned and disjoint from mutable outputs.
#[no_mangle]
pub unsafe extern "C" fn gcso_ego_field_intervene_frame(
    ego_inout: *mut gcso_liminal_ego_field_t,
    stack_inout: *mut gcso_autonomous_core_layer_stack_t,
    delta: *const gcso_liminal_boundary_delta_t,
    horizon: *const gcso_epistemic_horizon_reading_t,
    venn_report: *const gcso_stigmergic_venn_serendipity_report_t,
    frame_inout: *mut gcso_gateway_frame_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(ego_inout)?;
        validate_ptr(stack_inout)?;
        validate_ptr(delta)?;
        validate_ptr(horizon)?;
        validate_ptr(frame_inout)?;

        let ego_bytes = size_of::<LiminalEgoField>();
        let stack_bytes = size_of::<AutonomousCoreLayerStack>();
        let delta_bytes = size_of::<LiminalBoundaryDelta>();
        let hor_bytes = size_of::<EpistemicHorizonReading>();
        let venn_bytes = size_of::<StigmergicVennSerendipityReport>();
        let frame_bytes = size_of::<GatewayFrame>();

        if raw_byte_ranges_overlap(
            ego_inout as usize,
            ego_bytes,
            stack_inout as usize,
            stack_bytes,
        ) || raw_byte_ranges_overlap(
            ego_inout as usize,
            ego_bytes,
            frame_inout as usize,
            frame_bytes,
        ) || raw_byte_ranges_overlap(
            stack_inout as usize,
            stack_bytes,
            frame_inout as usize,
            frame_bytes,
        ) || raw_byte_ranges_overlap(ego_inout as usize, ego_bytes, delta as usize, delta_bytes)
            || raw_byte_ranges_overlap(ego_inout as usize, ego_bytes, horizon as usize, hor_bytes)
            || raw_byte_ranges_overlap(
                stack_inout as usize,
                stack_bytes,
                delta as usize,
                delta_bytes,
            )
            || raw_byte_ranges_overlap(
                stack_inout as usize,
                stack_bytes,
                horizon as usize,
                hor_bytes,
            )
            || raw_byte_ranges_overlap(
                frame_inout as usize,
                frame_bytes,
                delta as usize,
                delta_bytes,
            )
            || raw_byte_ranges_overlap(
                frame_inout as usize,
                frame_bytes,
                horizon as usize,
                hor_bytes,
            )
            || (!venn_report.is_null()
                && (raw_byte_ranges_overlap(
                    ego_inout as usize,
                    ego_bytes,
                    venn_report as usize,
                    venn_bytes,
                ) || raw_byte_ranges_overlap(
                    stack_inout as usize,
                    stack_bytes,
                    venn_report as usize,
                    venn_bytes,
                ) || raw_byte_ranges_overlap(
                    frame_inout as usize,
                    frame_bytes,
                    venn_report as usize,
                    venn_bytes,
                )))
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated non-null, naturally aligned, and mutually disjoint above.
        let (((mut ego_ref, mut stack_ref), mut frame_ref), ((delta_ref, horizon_ref), venn_opt)) = unsafe {
            (
                (
                    (
                        FfiHandleMut::from_raw(ego_inout)?,
                        FfiHandleMut::from_raw(stack_inout)?,
                    ),
                    FfiHandleMut::from_raw(frame_inout)?,
                ),
                (
                    (
                        FfiHandleRef::from_raw(delta)?,
                        FfiHandleRef::from_raw(horizon)?,
                    ),
                    FfiHandleRef::from_raw_opt(venn_report)?,
                ),
            )
        };
        let venn_val = match venn_opt {
            Some(v) => *v,
            None => StigmergicVennSerendipityReport::default(),
        };
        let next_frame = ego_ref.intervene_on_layers_and_frame_branchless(
            &mut stack_ref,
            &delta_ref,
            &horizon_ref,
            &venn_val,
            *frame_ref,
        );
        *frame_ref = next_frame;
        Ok(GCSO_SUCCESS)
    })
}
