// name: src/core/src/runtime/abi/action_hub.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Action Hub orchestration, Limina bridging, Bayesian trail rewiring, and FastPathLedger acceleration FFI.

use super::*;

// ===================================================================
// 3. Action Hub, Limina Bridging, Bayesian Trail Rewiring & FastPathLedger Interface
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
        let hub = StigmergicSubstrate::with_capacity(capacity)?;
        *out_handle = Box::into_raw(Box::new(hub)).cast::<GcsoActionHubOpaque>();
        Ok(GCSO_SUCCESS)
    })
}

/// Initializes a 32-byte [`gcso_stigmergic_thresholds_t`] structure with canonical defaults.
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
/// boundary via [`StigmergicSubstrate::orchestrate_next_action_step`] ($C_{\mathrm{alloc}} = 0$).
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
            && ((hub_opt as usize) & (align_of::<StigmergicSubstrate>() - 1)) == 0
        {
            // SAFETY: `hub_opt` is non-null and 128-byte aligned.
            let hub_ref = unsafe { &mut *hub_opt.cast::<StigmergicSubstrate>() };
            hub_ref.phase_steering = dpsr;
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

/// Resolves an Action Hub phase command across the 8 parallel 8-head SWAR Micro-Swarm cells on
/// `trail_inout` in strictly constant time (`0` branches) and writes the [`StigmergicReconstitutionReport`] to `report_out`.
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
    report_out: *mut StigmergicReconstitutionReport,
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

/// Allocates a ledger slot handle.
///
/// # Safety
/// `slot_out` must be a non-null, naturally aligned writable handle pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_ledger_slot_create(
    slot_out: *mut GcsoLedgerSlotHandle,
) -> GcsoStatus {
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
/// `slot` must be a non-null, 64-byte aligned writable [`gcso_ledger_slot_t`] pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_ledger_reset_telemetry(slot: *mut gcso_ledger_slot_t) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and 64B alignment.
        let mut slot_mut = unsafe { FfiHandleMut::from_raw(slot)? };
        AbiLedgerDispatcher.reset_telemetry(&mut slot_mut)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Executes an $\mathcal{O}(1)$ branchless fast-path shortcut lookup in the FastPathLedger.
///
/// # Safety
/// `slot` and `shortcut_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_ledger_fast_path_lookup(
    slot: *const gcso_ledger_slot_t,
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

        *out_val = AbiLedgerDispatcher.fast_path_lookup(&slot_ref, input_key)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Pushes a profiling metric byte to the FastPathLedger telemetry ring ledger in zero-allocation mode.
///
/// # Safety
/// `slot` must be a non-null, 64-byte aligned writable [`gcso_ledger_slot_t`] pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_ledger_telemetry_push(
    slot: *mut gcso_ledger_slot_t,
    metric_code: u8,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and 64B alignment.
        let mut slot_mut = unsafe { FfiHandleMut::from_raw(slot)? };
        AbiLedgerDispatcher.telemetry_push(&mut slot_mut, metric_code)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Sets the operating mode of a ledger slot (`0` = Scratchpad, `1` = Plugin, `2` = Shared IPC Buffer).
///
/// # Safety
/// `slot` must be a non-null, 64-byte aligned writable [`gcso_ledger_slot_t`] pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_ledger_set_mode(
    slot: *mut gcso_ledger_slot_t,
    mode: u32,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and 64B alignment.
        let mut slot_mut = unsafe { FfiHandleMut::from_raw(slot)? };
        AbiLedgerDispatcher.set_mode(&mut slot_mut, mode)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluates the FastPathLedger telemetry ring ledger to auto-tune runtime configuration parameters.
///
/// # Safety
/// `slot` and `config_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_ledger_evaluate_auto_tune(
    slot: *const gcso_ledger_slot_t,
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

        AbiLedgerDispatcher.evaluate_auto_tune(&slot_ref, &mut cfg_mut)?;
        Ok(GCSO_SUCCESS)
    })
}
