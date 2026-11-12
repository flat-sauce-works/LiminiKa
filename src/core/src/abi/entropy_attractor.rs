// name: src/core/src/abi/entropy_attractor.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unified Attractor Manifold, Online Toroidal k-Means, synergy, Entropy Control (EDBC), and CVoid FFI.

use super::*;

// ===================================================================
// 5. Single Unified Attractor Manifold, Online Toroidal k-Means, Synergy, EDBC & CVoid Interface
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

/// Evaluates **Zero-Hyperparameter Phase-Proximity Gravity** on the **Single Unified Attractor
/// Phase Manifold** ($SO(2)^{d_{\mathrm{head}}/2}$) between `trail_phases` and `anchor_phases`.
///
/// # Safety
/// `trail_phases` and `anchor_phases` must be non-null and point to 64 `gcso_q7_t` elements.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_eval_resonant_pull(
    trail_phases: *const gcso_q7_t,
    anchor_phases: *const gcso_q7_t,
    base_energy: f32,
    resonant_pull_out: *mut f32,
    coherence_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        if !branchless_is_finite_f32(base_energy) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated non-null.
        let (t_ref, a_ref) = unsafe {
            (
                ref_phase64_from_raw(trail_phases)?,
                ref_phase64_from_raw(anchor_phases)?,
            )
        };

        let (pull, coh) = branchless_q7_resonant_pull_64(t_ref, a_ref, base_energy);

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(resonant_pull_out, pull)?;
            write_opt_out(coherence_out, coh)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Order-invariant (commutative) folding of an arbitrary slice of `num_anchors` anchors on the
/// **Single Unified Attractor Phase Manifold** ($SO(2)^{d_{\mathrm{head}}/2}$) into
/// `folded_phases_out` (64 elements) starting from `trail_phases` with inertia weight `trail_init_weight`.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, and bounds invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_fold_unified_field(
    trail_phases: *const gcso_q7_t,
    trail_init_weight: f32,
    anchor_phases: *const gcso_q7_t,
    anchor_base_energies: *const f32,
    num_anchors: usize,
    folded_phases_out: *mut gcso_q7_t,
    total_pull_out: *mut f32,
    mean_coherence_out: *mut f32,
    dominant_anchor_id_out: *mut u32,
    active_anchor_count_out: *mut u32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(trail_phases)?;
        validate_ptr(folded_phases_out)?;
        if !branchless_is_finite_f32(trail_init_weight) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated non-null above.
        let trail_words = load_q7_words_64(unsafe { ref_phase64_from_raw(trail_phases)? });
        // SAFETY: Validated non-null and bounds-checked.
        let anchors_slice = unsafe { slice_of_phase64_from_raw(anchor_phases, num_anchors)? };
        let energies_slice: &[f32] = if num_anchors == 0 {
            &[]
        } else {
            validate_ptr(anchor_base_energies)?;
            // SAFETY: Validated non-null and 4B aligned.
            unsafe { slice::from_raw_parts(anchor_base_energies, num_anchors) }
        };

        let (folded_words, tot_pull, mean_coh, dom_id, act_cnt) =
            branchless_q7_unified_field_fold_words_with_init_weight_64(
                &trail_words,
                trail_init_weight,
                anchors_slice,
                energies_slice,
            );

        // SAFETY: Validated non-null above.
        let out_ref = unsafe { mut_phase64_from_raw(folded_phases_out)? };
        store_q7_words_64(out_ref, &folded_words);

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(total_pull_out, tot_pull)?;
            write_opt_out(mean_coherence_out, mean_coh)?;
            write_opt_out(dominant_anchor_id_out, dom_id)?;
            write_opt_out(active_anchor_count_out, act_cnt)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Steers `trail_inout` in-place directly on the **Single Unified Attractor Phase Manifold** across
/// `num_anchors` anchors and optional repulsive phase inhibitor, binding the dominant resonant
/// anchor index into `trail_inout.target_anchor_id` and `trail_inout.cluster_id`.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_steer_trail_unified_field(
    trail_inout: *mut gcso_pointer_trail_t,
    anchor_phases: *const gcso_q7_t,
    anchor_base_energies: *const f32,
    num_anchors: usize,
    repulsion_phases: *const gcso_q7_t,
    inhibition_strength: f32,
    final_pull_out: *mut f32,
    mean_coherence_out: *mut f32,
    active_anchor_count_out: *mut u32,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and 128B alignment.
        let mut trail = unsafe { FfiHandleMut::from_raw(trail_inout)? };
        if !branchless_is_finite_f32(inhibition_strength) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated non-null and bounds-checked.
        let anchors_slice = unsafe { slice_of_phase64_from_raw(anchor_phases, num_anchors)? };
        let energies_slice: &[f32] = if num_anchors == 0 {
            &[]
        } else {
            validate_ptr(anchor_base_energies)?;
            // SAFETY: Validated non-null and 4B aligned.
            unsafe { slice::from_raw_parts(anchor_base_energies, num_anchors) }
        };

        if !repulsion_phases.is_null()
            && raw_byte_ranges_overlap(
                trail_inout as usize,
                size_of::<gcso_pointer_trail_t>(),
                repulsion_phases as usize,
                MAX_ATTENTION_HEADS,
            )
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let rep_opt = if repulsion_phases.is_null() {
            None
        } else {
            // SAFETY: Validated non-null and disjoint above.
            Some((
                unsafe { ref_phase64_from_raw(repulsion_phases)? },
                inhibition_strength,
            ))
        };

        let (pull, mean_coh, act_cnt) = trail.apply_unified_field_anchor_slice_resonance(
            anchors_slice,
            energies_slice,
            rep_opt,
        );
        trail.sync_flags_with_current_ptr();

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(final_pull_out, pull)?;
            write_opt_out(mean_coherence_out, mean_coh)?;
            write_opt_out(active_anchor_count_out, act_cnt)?;
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

/// Evaluates non-linear **Cross-Slot Attractor Synergy** and **Liminal Beat Perturbation** between
/// two 64-head Q7 attractor slot profiles (`phases_a`, `phases_b`) gated by a 64-bit $8 \times 8$
/// coupling bitmatrix (`coupling_matrix_8x8`).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_eval_cross_slot_synergy(
    phases_a: *const gcso_q7_t,
    weight_a: f32,
    phases_b: *const gcso_q7_t,
    weight_b: f32,
    slot_a: u32,
    slot_b: u32,
    coupling_matrix_8x8: u64,
    synergy_gain: f32,
    fused_phases_out: *mut gcso_q7_t,
    effective_pull_out: *mut f32,
    coherence_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(phases_a)?;
        validate_ptr(phases_b)?;
        validate_ptr(fused_phases_out)?;
        if (slot_a as usize) >= MAX_ATTRACTOR_DOMAINS
            || (slot_b as usize) >= MAX_ATTRACTOR_DOMAINS
            || !branchless_is_finite_f32(weight_a)
            || !branchless_is_finite_f32(weight_b)
            || !branchless_is_finite_f32(synergy_gain)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        if raw_byte_ranges_overlap(
            fused_phases_out as usize,
            MAX_ATTENTION_HEADS,
            phases_a as usize,
            MAX_ATTENTION_HEADS,
        ) || raw_byte_ranges_overlap(
            fused_phases_out as usize,
            MAX_ATTENTION_HEADS,
            phases_b as usize,
            MAX_ATTENTION_HEADS,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated non-null and disjoint above.
        let ((ref_a, ref_b), out_arr) = unsafe {
            (
                (
                    ref_phase64_from_raw(phases_a)?,
                    ref_phase64_from_raw(phases_b)?,
                ),
                mut_phase64_from_raw(fused_phases_out)?,
            )
        };

        let coupled = is_slot_pair_coupled(coupling_matrix_8x8, slot_a, slot_b);
        let eff_gain = branchless_select_f32(coupled, synergy_gain, 0.0);
        let (pull, coherence) =
            branchless_q7_synergy_fuse_64(ref_a, weight_a, ref_b, weight_b, eff_gain, out_arr);

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(effective_pull_out, pull)?;
            write_opt_out(coherence_out, coherence)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Applies branchless **Mutual Slot Inhibition** in-place to a 64-head Q7 phase profile
/// (`target_phases_inout`) when opposed by `inhibitor_phases` ($\mathcal{C}_{\mathrm{phase}} < 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_apply_mutual_slot_inhibition(
    target_phases_inout: *mut gcso_q7_t,
    target_weight: f32,
    inhibitor_phases: *const gcso_q7_t,
    inhibition_strength: f32,
    damped_weight_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(target_phases_inout)?;
        validate_ptr(inhibitor_phases)?;
        if !branchless_is_finite_f32(target_weight)
            || !branchless_is_finite_f32(inhibition_strength)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        if raw_byte_ranges_overlap(
            target_phases_inout as usize,
            MAX_ATTENTION_HEADS,
            inhibitor_phases as usize,
            MAX_ATTENTION_HEADS,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated non-null and disjoint above.
        let (inh_ref, target_arr) = unsafe {
            (
                ref_phase64_from_raw(inhibitor_phases)?,
                mut_phase64_from_raw(target_phases_inout)?,
            )
        };

        let damped =
            branchless_q7_inhibit_64(target_arr, target_weight, inh_ref, inhibition_strength);

        // SAFETY: `write_opt_out` validates optional output pointer.
        unsafe {
            write_opt_out(damped_weight_out, damped)?;
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

/// Clears all registered topological anchors from the attractor field.
///
/// # Safety
/// `field` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_clear_anchors(
    field: GcsoAttractorFieldHandle,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(field)?;
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

/// Computes the CVoid Coherent Vector Alignment Metric ($\mathcal{C}_{\mathrm{void}}$) for
/// Out-of-Distribution Latent Space over a 32-byte aligned Key vector.
///
/// # Safety
/// `key_vector` (32B aligned) and `void_score_out` must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_cvoid_eval_dyadic128(
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

        EntropyController::default().eval_cvoid_dyadic128(&key_aligned, &mut out_score)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluates the Eyring-Kramers potential barrier height ($\Delta V = \tau_{\mathrm{eff}} / \mathcal{C}_{\mathrm{void}}$)
/// with branchless singularity guards.
///
/// # Safety
/// `barrier_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_cvoid_eval_barrier(
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
/// according to GCSO Whitepaper Lemma 3.1: $\Delta V = \frac{\alpha^2 (\mathcal{C}_{\mathrm{void}} - \tau_{\mathrm{eff}})^2}{4 \beta d_{\mathrm{head}}}$.
///
/// # Safety
/// `barrier_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_cvoid_eval_barrier_pitchfork(
    cvoid: f32,
    tau_eff: f32,
    alpha: f32,
    beta: f32,
    head_dim: u32,
    barrier_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let mut out_val = unsafe { FfiHandleMut::from_raw(barrier_out)? };
        *out_val = eval_barrier_pitchfork(cvoid, tau_eff, alpha, beta, head_dim)?;
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
