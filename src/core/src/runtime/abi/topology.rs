// name: src/core/src/runtime/abi/topology.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unified Attractor Manifold, Online Toroidal k-Means, synergy, Entropy Control (Entropy), and Liminal FFI.

use super::*;
use crate::runtime::topology::CORE_TOPOLOGY;

// ===================================================================
// 5. Single Unified Attractor Manifold, Online Toroidal k-Means, Synergy, Entropy & Liminal Interface
// ===================================================================

/// Executes one step of **Classical Online Toroidal $k$-Means / DP-Means Macro Clustering** on
/// $SO(2)^{64}$ (`centroids_inout[0..num_centroids]`) for an incoming [`lmnk_gateway_frame_t`]
/// (`frame`) via `PhaseAnchorRecord::step_online_toroidal_kmeans_frame_64`.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `LMNK_RESTRICT` invariants.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_anchor_step_online_toroidal_kmeans(
    centroids_inout: *mut lmnk_phase_anchor_record_t,
    num_centroids: usize,
    frame: *const lmnk_gateway_frame_t,
    split_coherence_threshold: f32,
    trail_inout: *mut lmnk_pointer_trail_t,
    assigned_centroid_idx_out: *mut usize,
    assigned_coherence_out: *mut f32,
    updated_gravity_mass_out: *mut f32,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr(centroids_inout)?;
        validate_ptr(frame)?;
        if num_centroids == 0 || !branchless_is_finite_f32(split_coherence_threshold) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let cent_bytes = validate_raw_slice_bounds(centroids_inout, num_centroids)?;
        let trail_bytes = if trail_inout.is_null() {
            0
        } else {
            validate_ptr(trail_inout)?;
            size_of::<lmnk_pointer_trail_t>()
        };
        check_disjoint_rw(
            &[
                (centroids_inout as usize, cent_bytes),
                (trail_inout as usize, trail_bytes),
                opt_span(assigned_centroid_idx_out)?,
                opt_span(assigned_coherence_out)?,
                opt_span(updated_gravity_mass_out)?,
            ],
            &[(frame as usize, size_of::<lmnk_gateway_frame_t>())],
        )?;

        // SAFETY: Validated non-null, aligned, bounded, and disjoint above.
        let ((centroids_slice, mut trail_opt), frame_ref) = unsafe {
            (
                (
                    checked_slice_from_raw_parts_mut(centroids_inout, num_centroids)?,
                    FfiHandleMut::from_raw_opt(trail_inout)?,
                ),
                FfiHandleRef::from_raw(frame)?,
            )
        };

        let (assigned_idx, _assigned_anchor_id, assigned_coh, updated_mass, _spawned_new) =
            PhaseAnchorRecord::step_online_toroidal_kmeans_frame_64(
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
        Ok(LMNK_OK)
    })
}

/// Executes **Batched Online Toroidal $k$-Means / DP-Means Macro Clustering** ($\mathcal{O}(B \cdot K)$)
/// on $SO(2)^{64}$ (`centroids_inout[0..num_centroids]`) across `num_frames` incoming
/// [`lmnk_gateway_frame_t`]s via `PhaseAnchorRecord::step_online_toroidal_kmeans_batch_frames_64`,
/// deferring IEEE 802.3 `checksum_crc32` restamping across modified centroids until batch completion.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `LMNK_RESTRICT` invariants.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_anchor_step_online_toroidal_kmeans_batch(
    centroids_inout: *mut lmnk_phase_anchor_record_t,
    num_centroids: usize,
    frames: *const lmnk_gateway_frame_t,
    num_frames: usize,
    split_coherence_threshold: f32,
    trail_inout: *mut lmnk_pointer_trail_t,
    last_assigned_idx_out: *mut usize,
    last_anchor_id_out: *mut u32,
    mean_batch_coherence_out: *mut f32,
    spawned_clusters_out: *mut u32,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr(centroids_inout)?;
        validate_ptr(frames)?;
        if num_centroids == 0
            || num_frames == 0
            || !branchless_is_finite_f32(split_coherence_threshold)
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let cent_bytes = validate_raw_slice_bounds(centroids_inout, num_centroids)?;
        let frames_bytes = validate_raw_slice_bounds(frames, num_frames)?;
        let trail_bytes = if trail_inout.is_null() {
            0
        } else {
            validate_ptr(trail_inout)?;
            size_of::<lmnk_pointer_trail_t>()
        };
        check_disjoint_rw(
            &[
                (centroids_inout as usize, cent_bytes),
                (trail_inout as usize, trail_bytes),
                opt_span(last_assigned_idx_out)?,
                opt_span(last_anchor_id_out)?,
                opt_span(mean_batch_coherence_out)?,
                opt_span(spawned_clusters_out)?,
            ],
            &[(frames as usize, frames_bytes)],
        )?;

        // SAFETY: Validated non-null, aligned, bounded, and disjoint above.
        let ((centroids_slice, frames_slice), mut trail_opt) = unsafe {
            (
                (
                    checked_slice_from_raw_parts_mut(centroids_inout, num_centroids)?,
                    checked_slice_from_raw_parts(frames, num_frames)?,
                ),
                FfiHandleMut::from_raw_opt(trail_inout)?,
            )
        };

        let (last_idx, last_anchor_id, mean_coh, spawned_total) =
            PhaseAnchorRecord::step_online_toroidal_kmeans_batch_frames_64(
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
        Ok(LMNK_OK)
    })
}

/// Allocates an Entropy entropy controller instance initialized from `initial_state`.
///
/// # Safety
/// `initial_state` and `controller_out` must be non-null and naturally aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_entropy_controller_create(
    initial_state: *const lmnk_entropy_state_t,
    controller_out: *mut LmnkEntropyControllerHandle,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr(initial_state)?;
        validate_ptr(controller_out)?;
        check_disjoint_rw(
            &[(
                controller_out as usize,
                size_of::<LmnkEntropyControllerHandle>(),
            )],
            &[(initial_state as usize, size_of::<lmnk_entropy_state_t>())],
        )?;
        // SAFETY: Validated for non-nullness, alignment, and disjointness above.
        let (init_st, mut out_handle) = unsafe {
            (
                FfiHandleRef::from_raw(initial_state)?,
                FfiHandleMut::from_raw(controller_out)?,
            )
        };

        let engine = EntropyController::new(*init_st);
        *out_handle = Box::into_raw(Box::new(engine)).cast::<LmnkEntropyControllerOpaque>();
        Ok(LMNK_OK)
    })
}

/// Evaluates stateful Moving Z-Score Attention Entropy ($\tilde{H}$) and updates branch mode.
///
/// # Safety
/// `controller` and `state_out` must be non-null, naturally aligned, and non-overlapping.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_entropy_eval_stateful(
    controller: LmnkEntropyControllerHandle,
    token_z_score: f32,
    state_out: *mut lmnk_entropy_state_t,
) -> LmnkStatus {
    ffi_boundary!({
        let typed = controller.cast::<EntropyController>();
        validate_ptr(typed)?;
        validate_ptr(state_out)?;
        check_disjoint_rw(
            &[
                (typed as usize, size_of::<EntropyController>()),
                (state_out as usize, size_of::<lmnk_entropy_state_t>()),
            ],
            &[],
        )?;
        // SAFETY: Validated for non-nullness, concrete alignment, and disjointness above.
        let (mut engine, mut out_st) = unsafe {
            (
                FfiHandleMut::from_raw(typed)?,
                FfiHandleMut::from_raw(state_out)?,
            )
        };

        engine.eval_stateful(token_z_score, &mut out_st)?;
        Ok(LMNK_OK)
    })
}

/// Computes the Liminal Vector Alignment Metric ($\mathcal{L}_{\mathrm{coh}}$) for
/// Out-of-Distribution Latent Space over a 32-byte aligned Key vector.
///
/// # Safety
/// `key_vector` (32B aligned) and `lacuna_score_out` must be non-null, valid, and non-overlapping.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_liminal_eval_dyadic128(
    key_vector: *const f32,
    dim: usize,
    lacuna_score_out: *mut f32,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr_aligned(key_vector, 32)?;
        validate_ptr(lacuna_score_out)?;
        let key_bytes = if dim == 0 {
            0
        } else {
            validate_raw_slice_bounds(key_vector, dim)?
        };
        check_disjoint_rw(
            &[(lacuna_score_out as usize, size_of::<f32>())],
            &[(key_vector as usize, key_bytes)],
        )?;
        // SAFETY: Validated for non-nullness, 32B alignment, bounds, and disjointness above.
        let (mut out_score, key_aligned) = unsafe {
            (
                FfiHandleMut::from_raw(lacuna_score_out)?,
                AlignedSlice32::from_raw_parts(key_vector, dim)?,
            )
        };

        EntropyController::default().eval_lacuna_energy_dyadic128(&key_aligned, &mut out_score)?;
        Ok(LMNK_OK)
    })
}

/// Evaluates the Eyring-Kramers potential barrier height ($\Delta V = \tau_{\mathrm{eff}} / \mathcal{L}_{\mathrm{coh}}$)
/// with branchless singularity guards.
///
/// # Safety
/// `barrier_out` must be non-null and naturally aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_liminal_eval_barrier(
    lacuna_score: f32,
    tau_eff: f32,
    barrier_out: *mut f32,
) -> LmnkStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let mut out_val = unsafe { FfiHandleMut::from_raw(barrier_out)? };
        *out_val = EntropyController::default().eval_barrier(lacuna_score, tau_eff)?;
        Ok(LMNK_OK)
    })
}

/// Evaluates the scale-invariant Pitchfork Bifurcation potential barrier height ($\Delta V$)
/// in division-free (`0 fdiv`) mode: $\Delta V = \frac{\alpha^2 (\mathcal{L}_{\mathrm{coh}} - \tau_{\mathrm{eff}})^2}{4 \beta d_{\mathrm{head}}}$.
///
/// # Safety
/// `barrier_out` must be non-null and naturally aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_liminal_eval_barrier_pitchfork(
    liminal_score: f32,
    tau_eff: f32,
    alpha: f32,
    beta: f32,
    head_dim: u32,
    barrier_out: *mut f32,
) -> LmnkStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let mut out_val = unsafe { FfiHandleMut::from_raw(barrier_out)? };
        *out_val = eval_barrier_pitchfork(liminal_score, tau_eff, alpha, beta, head_dim)?;
        Ok(LMNK_OK)
    })
}

/// Destroys an Entropy controller instance and releases its heap allocation.
///
/// # Safety
/// `controller` must be `NULL` or a handle returned by [`lmnk_entropy_controller_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_entropy_controller_destroy(
    controller: LmnkEntropyControllerHandle,
) -> LmnkStatus {
    if controller.is_null() {
        return LMNK_OK;
    }
    let typed = controller.cast::<EntropyController>();
    if !is_aligned(typed) {
        return LMNK_ERR_ALIGNMENT;
    }
    ffi_boundary!({
        // SAFETY: `typed` is verified non-null, naturally aligned for `EntropyController`,
        // and was allocated via `Box::into_raw` in `lmnk_entropy_controller_create`.
        unsafe {
            let _ = Box::from_raw(typed);
        }
        Ok(LMNK_OK)
    })
}

/// Evaluates [`lmnk_epistemic_horizon_t`] over `num_records` crystallized attractor
/// records relative to a 64-head Q7 query phase vector (`query_phases`) without branching.
///
/// # Safety
/// `query_phases` (64 bytes) and `reading_out` (16-byte aligned) must be non-null; if
/// `num_records > 0`, `records` must be non-null and 64-byte aligned with `num_records` valid items.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_epistemic_evaluate_horizon(
    records: *const lmnk_phase_anchor_record_t,
    num_records: usize,
    query_phases: *const lmnk_q7_t,
    horizon_floor: f32,
    reading_out: *mut lmnk_epistemic_horizon_t,
) -> LmnkStatus {
    ffi_boundary!({
        if !branchless_is_finite_f32(horizon_floor) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        validate_ptr(query_phases)?;
        validate_ptr(reading_out)?;
        let rec_bytes = if num_records == 0 {
            0
        } else {
            validate_ptr(records)?;
            validate_raw_slice_bounds(records, num_records)?
        };
        check_disjoint_rw(
            &[(reading_out as usize, size_of::<EpistemicHorizonReading>())],
            &[
                (records as usize, rec_bytes),
                (query_phases as usize, MAX_ATTENTION_HEADS),
            ],
        )?;
        // SAFETY: Validated non-null, aligned, bounds-checked, and disjoint above.
        let (records_slice, (q_ref, mut out_ref)) = unsafe {
            (
                checked_slice_from_raw_parts(records, num_records)?,
                (
                    ref_phase64_from_raw(query_phases)?,
                    FfiHandleMut::from_raw(reading_out)?,
                ),
            )
        };
        let q_words = load_q7_words_64(q_ref);
        *out_ref =
            evaluate_epistemic_horizon_words_64(records_slice, &q_words, None, horizon_floor);
        Ok(LMNK_OK)
    })
}

/// Evaluates set-overlap multi-attractor overlap and noise-driven discovery in `Liminal`
/// across `centroids_inout[0..num_centroids]` (`query_phases` vs `drifted_phases`). When
/// `nucleate_on_discovery != 0` and approximation noise triggers unknown attractor discovery,
/// autonomously nucleates a new crystallized attractor into an empty or weaker centroid slot.
///
/// # Safety
/// `query_phases`, `drifted_phases`, and `report_out` (16B aligned) must be non-null; if
/// `num_centroids > 0`, `centroids_inout` must be non-null and 64B aligned and disjoint from
/// `report_out` and `did_nucleate_out`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_stigmergic_evaluate_overlap_and_discovery(
    centroids_inout: *mut lmnk_phase_anchor_record_t,
    num_centroids: usize,
    query_phases: *const lmnk_q7_t,
    drifted_phases: *const lmnk_q7_t,
    aperture_floor: f32,
    discovery_floor_q7: u8,
    nucleate_on_discovery: u8,
    new_anchor_id: u32,
    report_out: *mut lmnk_overlap_discovery_report_t,
    did_nucleate_out: *mut u8,
) -> LmnkStatus {
    ffi_boundary!({
        if !branchless_is_finite_f32(aperture_floor) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        validate_ptr(query_phases)?;
        validate_ptr(drifted_phases)?;
        validate_ptr(report_out)?;
        let cent_bytes = if num_centroids == 0 {
            0
        } else {
            validate_ptr(centroids_inout)?;
            validate_raw_slice_bounds(centroids_inout, num_centroids)?
        };
        let nuc_bytes = if did_nucleate_out.is_null() {
            0
        } else {
            validate_ptr(did_nucleate_out)?;
            size_of::<u8>()
        };
        check_disjoint_rw(
            &[
                (centroids_inout as usize, cent_bytes),
                (report_out as usize, size_of::<OverlapDiscoveryReport>()),
                (did_nucleate_out as usize, nuc_bytes),
            ],
            &[
                (query_phases as usize, MAX_ATTENTION_HEADS),
                (drifted_phases as usize, MAX_ATTENTION_HEADS),
            ],
        )?;
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
        let overlap_report = if num_centroids == 0 {
            evaluate_stigmergic_overlap_and_discovery_words_64::<PhaseAnchorRecord>(
                &[],
                &q_words,
                &d_words,
                aperture_floor,
                discovery_floor_q7,
            )
        } else {
            // SAFETY: Validated non-null, 64B aligned, bounds-checked, and disjoint above.
            let centroids =
                unsafe { checked_slice_from_raw_parts_mut(centroids_inout, num_centroids)? };
            let report = evaluate_stigmergic_overlap_and_discovery_words_64(
                centroids,
                &q_words,
                &d_words,
                aperture_floor,
                discovery_floor_q7,
            );
            if nucleate_on_discovery != 0 && report.discovery_triggered != 0 {
                let (_slot, _aid, ok) =
                    PhaseAnchorRecord::nucleate_discovered_anchor_from_lacuna_64(
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

        *rep_out = overlap_report;
        // SAFETY: `write_opt_out` validates optional alignment and non-nullness.
        unsafe {
            write_opt_out(did_nucleate_out, did_nucleate)?;
        }
        Ok(LMNK_OK)
    })
}

/// Initializes a 128-byte [`lmnk_core_state_stack_t`] (Core Layer 0: `LiminalThresholdState`,
/// Core Layer 1: `StigmergicGranuleState`, Core Layer 2: `LiminalEgoState`) with canonical defaults.
///
/// # Safety
/// `stack_out` must be non-null and 64-byte aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_core_stack_init_default(
    stack_out: *mut lmnk_core_state_stack_t,
) -> LmnkStatus {
    ffi_boundary!({
        // SAFETY: Validates non-nullness and 64B alignment.
        let mut out_ref = unsafe { FfiHandleMut::from_raw(stack_out)? };
        *out_ref = CoreStateStack::new();
        Ok(LMNK_OK)
    })
}

/// Branchlessly adapts Core Layer 0 (`Liminal`) and Core Layer 1 (`Stigmergic`) inside the
/// 128-byte `stack_inout` from `horizon`, `overlap_report`, and optional `ledger`.
///
/// # Safety
/// `stack_inout` (64B aligned), `horizon` (16B aligned), and `overlap_report` (16B aligned) must be
/// non-null and non-overlapping; `ledger` may be `NULL` or 8-byte aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_core_stack_adapt(
    stack_inout: *mut lmnk_core_state_stack_t,
    ledger: *const lmnk_cognitive_ledger_t,
    horizon: *const lmnk_epistemic_horizon_t,
    overlap_report: *const lmnk_overlap_discovery_report_t,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr(stack_inout)?;
        validate_ptr(horizon)?;
        validate_ptr(overlap_report)?;
        check_disjoint_rw(
            &[(stack_inout as usize, size_of::<CoreStateStack>())],
            &[
                opt_span(ledger)?,
                (horizon as usize, size_of::<EpistemicHorizonReading>()),
                (overlap_report as usize, size_of::<OverlapDiscoveryReport>()),
            ],
        )?;
        // SAFETY: Validated non-null, aligned, and disjoint above.
        let ((mut stack_ref, ledger_opt), (horizon_ref, part_ref)) = unsafe {
            (
                (
                    FfiHandleMut::from_raw(stack_inout)?,
                    FfiHandleRef::from_raw_opt(ledger)?,
                ),
                (
                    FfiHandleRef::from_raw(horizon)?,
                    FfiHandleRef::from_raw(overlap_report)?,
                ),
            )
        };
        let ledger_val = match ledger_opt {
            Some(l) => *l,
            None => CognitiveTrajectoryLedger::default(),
        };
        stack_ref.adapt_layers_branchless(&ledger_val, &horizon_ref, &part_ref, None);
        Ok(LMNK_OK)
    })
}

/// Formats the live 3-layer Core Topology (`Liminal` Layer 0, `StigmergicGranuleState` Layer 1,
/// `LiminalEgoState` Layer 2, and the trait aperture table) as a `#![no_std]` zero-allocation
/// Mermaid `flowchart TD` diagram into `out_buf[0..buf_cap]`. Pass `out_buf = NULL` to query the
/// required byte length in `bytes_written_out`.
///
/// # Safety
/// `stack` (64B aligned) and `horizon` (16B aligned) must be non-null; `overlap_report` may be `NULL`
/// or 16B aligned. If `out_buf` is non-null, it must reference `buf_cap` writable bytes disjoint from
/// `bytes_written_out`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_topology_format_mermaid(
    stack: *const lmnk_core_state_stack_t,
    horizon: *const lmnk_epistemic_horizon_t,
    overlap_report: *const lmnk_overlap_discovery_report_t,
    active_anchors: u8,
    spawned_daughters: u8,
    out_buf: *mut u8,
    buf_cap: usize,
    bytes_written_out: *mut usize,
) -> LmnkStatus {
    ffi_boundary!({
        // SAFETY: Validates required and optional pointers and natural alignment.
        let ((stack_ref, horizon_ref), overlap_opt) = unsafe {
            (
                (
                    FfiHandleRef::from_raw(stack)?,
                    FfiHandleRef::from_raw(horizon)?,
                ),
                FfiHandleRef::from_raw_opt(overlap_report)?,
            )
        };
        let overlap_val = match overlap_opt {
            Some(v) => *v,
            None => OverlapDiscoveryReport::default(),
        };
        let snap = LiminalTopologyTelemetrySnapshot::capture(
            &stack_ref.liminal,
            &horizon_ref,
            active_anchors,
            spawned_daughters,
        );
        let needed = snap.format_core_topology_mermaid(&stack_ref, &overlap_val, &mut []);

        if out_buf.is_null() {
            // SAFETY: `write_opt_out` validates optional output pointer.
            unsafe {
                write_opt_out(bytes_written_out, needed)?;
            }
            return Ok(LMNK_OK);
        }

        let out_bytes = validate_raw_slice_bounds(out_buf, buf_cap)?;
        let bw_bytes = if bytes_written_out.is_null() {
            0
        } else {
            validate_ptr(bytes_written_out)?;
            size_of::<usize>()
        };
        check_disjoint_rw(
            &[
                (out_buf as usize, out_bytes),
                (bytes_written_out as usize, bw_bytes),
            ],
            &[
                (stack as usize, size_of::<lmnk_core_state_stack_t>()),
                (horizon as usize, size_of::<lmnk_epistemic_horizon_t>()),
                (
                    overlap_report as usize,
                    if overlap_report.is_null() {
                        0
                    } else {
                        size_of::<lmnk_overlap_discovery_report_t>()
                    },
                ),
            ],
        )?;
        // SAFETY: `write_opt_out` validates optional output pointer.
        unsafe {
            write_opt_out(bytes_written_out, needed)?;
        }
        if buf_cap < needed {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        // SAFETY: `out_buf` is non-null, bounded (`<= isize::MAX`), writable for `buf_cap >= needed`, and disjoint.
        unsafe {
            let dst = checked_slice_from_raw_parts_mut(out_buf, buf_cap)?;
            let _ = snap.format_core_topology_mermaid(&stack_ref, &overlap_val, dst);
        }
        Ok(LMNK_OK)
    })
}

/// Initializes a 64-byte [`lmnk_liminal_ego_state_t`] from a preset template code
/// (`0 = Custom`, `1 = Low`, `2 = Medium`, `3 = High`).
///
/// # Safety
/// `ego_out` must be non-null and 64-byte aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_ego_state_init_preset(
    preset_code: u8,
    ego_out: *mut lmnk_liminal_ego_state_t,
) -> LmnkStatus {
    ffi_boundary!({
        // SAFETY: Validates non-nullness and 64-byte natural alignment.
        let mut out_ref = unsafe { FfiHandleMut::from_raw(ego_out)? };
        *out_ref = LiminalEgoState::from_preset(LiminalEgoPreset::from_code(preset_code));
        Ok(LMNK_OK)
    })
}

/// Deterministically derives a 64-byte [`lmnk_liminal_ego_state_t`] from UTF-8 seed
/// text (`utf8_ptr[0..utf8_len]`) with zero heap allocations.
///
/// # Safety
/// `ego_out` must be non-null and 64-byte aligned. When `utf8_len > 0`, `utf8_ptr` must be
/// non-null, valid for `utf8_len` readable UTF-8 bytes, and disjoint from `ego_out`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_ego_state_derive_from_text(
    utf8_ptr: *const u8,
    utf8_len: usize,
    ego_out: *mut lmnk_liminal_ego_state_t,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr(ego_out)?;
        if utf8_len == 0 {
            // SAFETY: Validated non-null and 64-byte aligned above.
            let mut out_ref = unsafe { FfiHandleMut::from_raw(ego_out)? };
            *out_ref = LiminalEgoState::new();
            return Ok(LMNK_OK);
        }
        validate_ptr(utf8_ptr)?;
        let utf8_bytes = validate_raw_slice_bounds(utf8_ptr, utf8_len)?;
        check_disjoint_rw(
            &[(ego_out as usize, size_of::<LiminalEgoState>())],
            &[(utf8_ptr as usize, utf8_bytes)],
        )?;
        // SAFETY: Validated non-null, aligned, bounded, and disjoint above.
        let (mut out_ref, bytes) = unsafe {
            (
                FfiHandleMut::from_raw(ego_out)?,
                checked_slice_from_raw_parts(utf8_ptr, utf8_len)?,
            )
        };
        let text = core::str::from_utf8(bytes).map_err(|_| LMNK_ERR_INVALID_ARGUMENT)?;
        let trimmed = text.trim();
        if trimmed.is_empty() {
            *out_ref = LiminalEgoState::new();
            return Ok(LMNK_OK);
        }
        let axes = EgoIntentAxes::from_seed_bytes(trimmed.as_bytes());
        *out_ref = LiminalEgoState::from_intent_axes_with_seed(&axes, trimmed.as_bytes());
        Ok(LMNK_OK)
    })
}

/// Branchlessly evaluates active Ego triggers across Layer 0/1 boundaries and `delta`,
/// updating `ego_inout` and `stack_inout` and modulating `frame_inout` (counter-basin
/// thrust and transgressive phase steering).
///
/// # Safety
/// `ego_inout` (64B), `stack_inout` (64B), `delta` (16B), `horizon` (16B), and `frame_inout`
/// (64B) must be non-null, naturally aligned, and mutually non-overlapping; `overlap_report` may be
/// `NULL` or 16B aligned and disjoint from mutable outputs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_ego_state_intervene_frame(
    ego_inout: *mut lmnk_liminal_ego_state_t,
    stack_inout: *mut lmnk_core_state_stack_t,
    delta: *const lmnk_boundary_delta_t,
    horizon: *const lmnk_epistemic_horizon_t,
    overlap_report: *const lmnk_overlap_discovery_report_t,
    frame_inout: *mut lmnk_gateway_frame_t,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr(ego_inout)?;
        validate_ptr(stack_inout)?;
        validate_ptr(delta)?;
        validate_ptr(horizon)?;
        validate_ptr(frame_inout)?;

        let ego_bytes = size_of::<LiminalEgoState>();
        let stack_bytes = size_of::<CoreStateStack>();
        let delta_bytes = size_of::<LiminalBoundaryDelta>();
        let hor_bytes = size_of::<EpistemicHorizonReading>();
        let overlap_bytes = size_of::<OverlapDiscoveryReport>();
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
            || (!overlap_report.is_null()
                && (raw_byte_ranges_overlap(
                    ego_inout as usize,
                    ego_bytes,
                    overlap_report as usize,
                    overlap_bytes,
                ) || raw_byte_ranges_overlap(
                    stack_inout as usize,
                    stack_bytes,
                    overlap_report as usize,
                    overlap_bytes,
                ) || raw_byte_ranges_overlap(
                    frame_inout as usize,
                    frame_bytes,
                    overlap_report as usize,
                    overlap_bytes,
                )))
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated non-null, naturally aligned, and mutually disjoint above.
        let (
            ((mut ego_ref, mut stack_ref), mut frame_ref),
            ((delta_ref, horizon_ref), overlap_opt),
        ) = unsafe {
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
                    FfiHandleRef::from_raw_opt(overlap_report)?,
                ),
            )
        };
        let overlap_val = match overlap_opt {
            Some(v) => *v,
            None => OverlapDiscoveryReport::default(),
        };
        ego_ref.intervene_on_layers_and_frame_branchless(
            &mut stack_ref,
            &delta_ref,
            &horizon_ref,
            &overlap_val,
            &mut frame_ref,
        );
        Ok(LMNK_OK)
    })
}

/// Evaluates the branchless 4-region Overlap partition (`S & O`, `S & !O`, `!S & O` validated
/// Known Unknown, and `!(S | O)` Unobserved) across 8 SWAR granules.
///
/// # Safety
/// `other_granule_density_q7` (`8` bytes), `granule_noise_tolerance_q7` (`8` bytes), and
/// `partition_out` (`16 B` aligned) must be non-null and non-overlapping.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_liminal_evaluate_lacuna_partition(
    self_mask: u8,
    other_granule_density_q7: *const u8,
    granule_noise_tolerance_q7: *const u8,
    concentration_floor_q7: u8,
    partition_out: *mut lmnk_lacuna_partition_t,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr(other_granule_density_q7)?;
        validate_ptr(granule_noise_tolerance_q7)?;
        validate_ptr(partition_out)?;
        let part_bytes = size_of::<LacunaPartition8>();
        if raw_byte_ranges_overlap(
            partition_out as usize,
            part_bytes,
            other_granule_density_q7 as usize,
            NUM_STIGMERGIC_GRANULES,
        ) || raw_byte_ranges_overlap(
            partition_out as usize,
            part_bytes,
            granule_noise_tolerance_q7 as usize,
            NUM_STIGMERGIC_GRANULES,
        ) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        // SAFETY: Pointers validated non-null, naturally aligned, and disjoint above.
        let (other_s, tol_s, mut part_ref) = unsafe {
            (
                checked_slice_from_raw_parts(other_granule_density_q7, NUM_STIGMERGIC_GRANULES)?,
                checked_slice_from_raw_parts(granule_noise_tolerance_q7, NUM_STIGMERGIC_GRANULES)?,
                FfiHandleMut::from_raw(partition_out)?,
            )
        };
        let mut other_arr = [0u8; NUM_STIGMERGIC_GRANULES];
        other_arr.copy_from_slice(other_s);
        let mut tol_arr = [0u8; NUM_STIGMERGIC_GRANULES];
        tol_arr.copy_from_slice(tol_s);
        *part_ref = evaluate_lacuna_partition_branchless(
            self_mask,
            &other_arr,
            &tol_arr,
            concentration_floor_q7,
        );
        Ok(LMNK_OK)
    })
}

/// Executes 1 branchless Hot-Path asymmetric homeostatic tension step on `tension_inout`.
///
/// # Safety
/// `tension_inout` must be non-null and 16-byte aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_distortion_tension_step_hot_path(
    tension_inout: *mut lmnk_distortion_tension_t,
    ego_delta_norm_q7: u8,
    boundary_sensitivity_q7: u8,
    exit_entropy_q7: u8,
    guard_proximity_q7: u8,
    active_overdrive: u8,
) -> LmnkStatus {
    ffi_boundary!({
        // SAFETY: Validates non-nullness and 16-byte natural alignment.
        let mut ten_ref = unsafe { FfiHandleMut::from_raw(tension_inout)? };
        ten_ref.step_hot_path_branchless(
            ego_delta_norm_q7,
            boundary_sensitivity_q7,
            exit_entropy_q7,
            guard_proximity_q7,
            active_overdrive != 0,
        );
        Ok(LMNK_OK)
    })
}

/// Evaluates up to `max_hops` (`1..=4`) of a P-Frame [`lmnk_distortion_delta_t`] pointer chain in
/// zero-allocation mode, enforcing [`lmnk_spine_state_t`] physical `pinned` isolation.
///
/// # Safety
/// All required struct pointers (`nodes`, `ego`, `spine`, `partition`, `stigmergy`, `tension`,
/// `frame_inout`) must be non-null, naturally aligned, and `frame_inout` / optional outputs
/// (`delta_norm_out`, `hops_traversed_out`) must not overlap inputs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_distortion_delta_apply_chain(
    nodes: *const lmnk_distortion_delta_t,
    num_nodes: usize,
    ego: *const lmnk_liminal_ego_state_t,
    spine: *const lmnk_spine_state_t,
    partition: *const lmnk_lacuna_partition_t,
    stigmergy: *const lmnk_stigmergic_granule_state_t,
    tension: *const lmnk_distortion_tension_t,
    max_hops: u8,
    frame_inout: *mut lmnk_gateway_frame_t,
    delta_norm_out: *mut u8,
    hops_traversed_out: *mut u8,
) -> LmnkStatus {
    ffi_boundary!({
        if num_nodes > MAX_EGO_PFRAME_NODES {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        validate_ptr_aligned(nodes, align_of::<DistortionDelta>())?;
        validate_ptr(ego)?;
        validate_ptr(spine)?;
        validate_ptr(partition)?;
        validate_ptr(stigmergy)?;
        validate_ptr(tension)?;
        validate_ptr(frame_inout)?;
        let nodes_bytes = validate_raw_slice_bounds(nodes, num_nodes)?;
        let d_norm_bytes = if delta_norm_out.is_null() {
            0
        } else {
            validate_ptr(delta_norm_out)?;
            size_of::<u8>()
        };
        let hops_bytes = if hops_traversed_out.is_null() {
            0
        } else {
            validate_ptr(hops_traversed_out)?;
            size_of::<u8>()
        };
        check_disjoint_rw(
            &[
                (frame_inout as usize, size_of::<GatewayFrame>()),
                (delta_norm_out as usize, d_norm_bytes),
                (hops_traversed_out as usize, hops_bytes),
            ],
            &[
                (nodes as usize, nodes_bytes),
                (ego as usize, size_of::<LiminalEgoState>()),
                (spine as usize, size_of::<SystemSpineState>()),
                (partition as usize, size_of::<LacunaPartition8>()),
                (stigmergy as usize, size_of::<StigmergicGranuleState>()),
                (tension as usize, size_of::<DistortionTension>()),
            ],
        )?;
        // SAFETY: All pointers are validated non-null, naturally aligned, bounded, and disjoint above.
        let ((ego_ref, spine_ref, part_ref, stig_ref, ten_ref), (mut frm_ref, node_slice)) = unsafe {
            (
                (
                    FfiHandleRef::from_raw(ego)?,
                    FfiHandleRef::from_raw(spine)?,
                    FfiHandleRef::from_raw(partition)?,
                    FfiHandleRef::from_raw(stigmergy)?,
                    FfiHandleRef::from_raw(tension)?,
                ),
                (
                    FfiHandleMut::from_raw(frame_inout)?,
                    checked_slice_from_raw_parts(nodes, num_nodes)?,
                ),
            )
        };
        let mut net = EgoDifferentialNet::new_empty(max_hops.min(MAX_DISTORTION_CHAIN_HOPS));
        for (idx, n) in node_slice.iter().enumerate() {
            net.set_node(idx, *n);
        }
        let (d_norm, hops) = net.evaluate_pframe_chain_branchless(
            &ego_ref,
            &spine_ref,
            &part_ref,
            &stig_ref,
            &ten_ref,
            &mut frm_ref,
        );
        // SAFETY: `write_opt_out` validates nullness and alignment for optional scalar outputs.
        unsafe {
            write_opt_out(delta_norm_out, d_norm)?;
            write_opt_out(hops_traversed_out, hops)?;
        }
        Ok(LMNK_OK)
    })
}

/// Copies the compile-time 64-byte [`lmnk_topology_matrix_t`] (`CORE_TOPOLOGY`) into `out_matrix`
/// with zero heap allocation.
///
/// # Safety
/// `out_matrix` must be non-null and 64-byte aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_topology_get_matrix(
    out_matrix: *mut lmnk_topology_matrix_t,
) -> LmnkStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and 64-byte alignment.
        let mut out_ref = unsafe { FfiHandleMut::from_raw(out_matrix)? };
        *out_ref = CORE_TOPOLOGY;
        Ok(LMNK_OK)
    })
}
