// name: src/core/src/abi/dispatch.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Internal zero-cost stateless trait dispatchers and raw-pointer validation helpers.

use super::*;

// ===================================================================
// Internal Zero-Cost Stateless Trait Dispatchers & Validation Helpers
// ===================================================================

pub(super) struct AbiSystemDispatcher;
impl SystemCapabilityQuery for AbiSystemDispatcher {}

pub(super) struct AbiDaesDispatcher;
impl DaesScratchpad for AbiDaesDispatcher {
    #[inline]
    fn evaluate_auto_tune(
        &self,
        slot: &gcso_daes_slot_t,
        config_out: &mut gcso_config_t,
    ) -> GcsoResult<()> {
        if config_out.head_dim == 0 {
            AbiSystemDispatcher.init_default_config(config_out)?;
        }
        // Single 64-bit SWAR instruction counting all 8 bytes >= 0x80 (`0` loops, `0` branches).
        let spike_count =
            (u64::from_le_bytes(slot.telemetry_mini_ledger) & 0x8080_8080_8080_8080).count_ones();
        let base_ripa = branchless_select_f32(
            config_out.ripa_clamp_max_rad > 0.0,
            config_out.ripa_clamp_max_rad,
            DEFAULT_RIPA_CLAMP_MAX_RAD,
        );
        let damping = 1.0 - 0.025 * (spike_count as f32);
        config_out.ripa_clamp_max_rad = branchless_clamp_f32(base_ripa * damping, 0.01, 0.25);
        config_out.daes_mode = (slot.mode & 0xFF) as u8;
        Ok(())
    }
}

pub(super) struct AbiPspmDispatcher;
impl PspmRouter for AbiPspmDispatcher {}

pub(super) struct AbiSrlDispatcher;
impl SrlAdapter for AbiSrlDispatcher {}

pub(super) struct AbiL2pDispatcher;
impl L2pSvdProjector for AbiL2pDispatcher {}

/// Branchlessly checks pointer non-nullness and natural alignment for type `T`.
#[inline(always)]
#[must_use]
pub fn is_aligned<T>(ptr: *const T) -> bool {
    let align = align_of::<T>();
    (!ptr.is_null()) & (((ptr as usize) & (align - 1)) == 0)
}

/// Branchlessly checks pointer non-nullness and custom power-of-two byte alignment.
#[inline(always)]
#[must_use]
pub fn is_aligned_to<T>(ptr: *const T, align: usize) -> bool {
    is_ptr_aligned_to(ptr, align)
}

/// Validates a non-null, naturally aligned pointer or opaque handle.
#[inline(always)]
pub(super) fn validate_ptr<T>(ptr: *const T) -> GcsoResult<()> {
    if ptr.is_null() {
        Err(GCSO_ERROR_NULL_POINTER)
    } else if !is_aligned(ptr) {
        Err(GCSO_ERROR_MISALIGNED_POINTER)
    } else {
        Ok(())
    }
}

/// Validates a non-null pointer against an explicit power-of-two alignment boundary `align`.
#[inline(always)]
pub(super) fn validate_ptr_aligned<T>(ptr: *const T, align: usize) -> GcsoResult<()> {
    if ptr.is_null() {
        Err(GCSO_ERROR_NULL_POINTER)
    } else if !is_aligned_to(ptr, align) {
        Err(GCSO_ERROR_MISALIGNED_POINTER)
    } else {
        Ok(())
    }
}

/// Zero-copy borrowed 64-head Q7 phase wrapper implementing [`RawPhaseProjectable`] so FFI callers
/// passing `*const gcso_q7_t` into 2-Origin Gateway synthesis load directly into 8 `u64` SWAR
/// registers without intermediate 64-byte stack array copies.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub(super) struct AbiPhaseRef64<'a>(&'a [gcso_q7_t; MAX_ATTENTION_HEADS]);

impl RawPhaseProjectable for AbiPhaseRef64<'_> {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        *self.0
    }

    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        load_q7_words_64(self.0)
    }
}

/// Zero-copy borrowed `f32` embedding slice wrapper implementing [`RawPhaseProjectable`] so FFI
/// callers passing `*const f32` into 2-Origin Gateway synthesis project directly onto $SO(2)^{64}$
/// via 8-YMM AVX2 SIMD accumulation (`gain > 0.0`) or unrolled `swar_q7_pack_x8` register packing
/// ([`project_f32_slice_to_words_64`], `gain == 0.0`).
#[derive(Clone, Copy)]
pub(super) struct AbiF32EmbeddingRef<'a> {
    slice: &'a [f32],
    gain: f32,
}

impl RawPhaseProjectable for AbiF32EmbeddingRef<'_> {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        if self.gain > 0.0 {
            branchless_project_f32_slice_to_q7_64(self.slice, self.gain)
        } else {
            project_f32_slice_to_q7_64(self.slice)
        }
    }

    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        if self.gain > 0.0 {
            let q7 = branchless_project_f32_slice_to_q7_64(self.slice, self.gain);
            load_q7_words_64(&q7)
        } else {
            project_f32_slice_to_words_64(self.slice)
        }
    }
}

/// Private zero-copy borrowed [`ContrastivePhaseBloomFilter`] guard wrapper inside `abi::dispatch`
/// implementing [`CausalProvenanceSignal`], [`CognitiveGroundingSignal`], and
/// [`ContradictionGuardSignal`] without colliding with blanket implementations on
/// [`ContradictionGuard`] in `gateway.rs`. Note: `BranchlessKernel` is automatically
/// provided via the universal blanket `impl<T: ?Sized> BranchlessKernel for T {}` in `branchless.rs`.
#[derive(Clone, Copy)]
pub(super) struct AbiBorrowedBloomGuard<'a> {
    bloom: &'a ContrastivePhaseBloomFilter,
    bloom_repulsion_threshold: u8,
}

impl<'a> AbiBorrowedBloomGuard<'a> {
    #[inline(always)]
    const fn new(
        bloom: &'a ContrastivePhaseBloomFilter,
        _obstruction_threshold: f32,
        bloom_repulsion_threshold: u8,
    ) -> Self {
        let safe_bloom =
            branchless_select_u8(bloom_repulsion_threshold > 0, bloom_repulsion_threshold, 1);
        Self {
            bloom,
            bloom_repulsion_threshold: safe_bloom,
        }
    }
}

impl CausalProvenanceSignal for AbiBorrowedBloomGuard<'_> {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        CAUSAL_AXIS_CONTRADICTION_GUARD
    }

    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        0.0
    }
}

impl CognitiveGroundingSignal for AbiBorrowedBloomGuard<'_> {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        [0i8; MAX_ATTENTION_HEADS]
    }

    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        [0u64; NUM_SWARM_MICRO_CELLS]
    }

    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        0.0
    }
}

impl ContradictionGuardSignal for AbiBorrowedBloomGuard<'_> {
    #[inline(always)]
    fn requires_phase_slice_for_repulsion(&self) -> bool {
        false
    }

    #[inline(always)]
    fn evaluate_repulsion_severity(
        &self,
        candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let words = load_q7_words_64(candidate_phases);
        self.evaluate_repulsion_severity_from_words(&words, candidate_phases)
    }

    #[inline(always)]
    fn evaluate_repulsion_severity_from_words(
        &self,
        candidate_words: &[u64; NUM_SWARM_MICRO_CELLS],
        _candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let density = if self.bloom.is_empty() {
            0
        } else {
            self.bloom
                .query_repulsive_density_from_words(candidate_words)
        };
        let bloom_hit = density >= self.bloom_repulsion_threshold;
        branchless_select_f32(
            bloom_hit,
            branchless_clamp_f32(0.60 + (density as f32) * 0.05, 0.60, 1.0),
            0.0,
        )
    }
}

/// Generic zero-copy helper synthesizing a [`GatewayFrame`] from Origin 1 (`SystemConstraint`)
/// and Origin 2 (`ObjectiveFact<F>`, `SubjectivePersona<P>`, optional `AbiBorrowedBloomGuard`).
///
/// # Safety
/// `bloom_filter_opt` must be `NULL` or a valid aligned pointer to [`gcso_contrastive_bloom_filter_t`].
#[inline]
pub(super) unsafe fn synthesize_multi_origin_generic_with_words<
    F: RawPhaseProjectable,
    P: RawPhaseProjectable,
>(
    vram_pressure: f32,
    gpu_frame_overrun: f32,
    active_swarm_cell_mask: u8,
    recommend_layer_skip: u8,
    fact_sig_opt: Option<ObjectiveFact<F>>,
    pers_sig_opt: Option<SubjectivePersona<P>>,
    bloom_filter_opt: *const gcso_contrastive_bloom_filter_t,
    contradiction_threshold: f32,
    bloom_repulsion_threshold: u8,
) -> GcsoResult<(GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS])> {
    // SAFETY: `FfiHandleRef::from_raw_opt` validates non-nullness and alignment of `bloom_filter_opt`.
    let bloom_ref_opt = unsafe { FfiHandleRef::from_raw_opt(bloom_filter_opt)? };
    let sys_sig =
        SystemConstraint::new((), vram_pressure, gpu_frame_overrun, active_swarm_cell_mask)
            .with_layer_skip(recommend_layer_skip != 0);
    Ok(match bloom_ref_opt {
        Some(b_ref) => {
            let guard = AbiBorrowedBloomGuard::new(
                &b_ref,
                contradiction_threshold,
                bloom_repulsion_threshold,
            );
            synthesize_multi_origin_gateway_frame_with_words(
                Some(&sys_sig),
                fact_sig_opt.as_ref(),
                pers_sig_opt.as_ref(),
                Some(&guard),
            )
        }
        None => {
            let guard_none: Option<&ContradictionGuard<()>> = None;
            synthesize_multi_origin_gateway_frame_with_words(
                Some(&sys_sig),
                fact_sig_opt.as_ref(),
                pers_sig_opt.as_ref(),
                guard_none,
            )
        }
    })
}

/// Zero-copy internal helper synthesizing a [`GatewayFrame`] directly from raw Q7 phase pointers.
///
/// # Safety
/// Non-null phase pointers must point to at least 64 valid `gcso_q7_t` elements.
#[inline]
pub(super) unsafe fn synthesize_multi_origin_from_raw_ptrs_with_words(
    vram_pressure: f32,
    gpu_frame_overrun: f32,
    active_swarm_cell_mask: u8,
    recommend_layer_skip: u8,
    fact_phases: *const gcso_q7_t,
    fact_confidence: f32,
    fact_zimms_offset48: u64,
    fact_anchor_id: u32,
    persona_phases: *const gcso_q7_t,
    persona_strength: f32,
    persona_zimms_offset48: u64,
    persona_anchor_id: u32,
    bloom_filter_opt: *const gcso_contrastive_bloom_filter_t,
    contradiction_threshold: f32,
    bloom_repulsion_threshold: u8,
) -> GcsoResult<(GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS])> {
    let fact_sig_opt = if fact_phases.is_null() {
        None
    } else {
        // SAFETY: `fact_phases` is non-null and `align_of::<[gcso_q7_t; 64]>() == 1`.
        let f_ref = unsafe { &*fact_phases.cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>() };
        Some(ObjectiveFact::new(
            AbiPhaseRef64(f_ref),
            fact_confidence,
            fact_zimms_offset48,
            fact_anchor_id,
        ))
    };
    let pers_sig_opt = if persona_phases.is_null() {
        None
    } else {
        // SAFETY: `persona_phases` is non-null and `align_of::<[gcso_q7_t; 64]>() == 1`.
        let p_ref = unsafe { &*persona_phases.cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>() };
        Some(SubjectivePersona::new(
            AbiPhaseRef64(p_ref),
            persona_strength,
            persona_zimms_offset48,
            persona_anchor_id,
        ))
    };
    // SAFETY: Validated above.
    unsafe {
        synthesize_multi_origin_generic_with_words(
            vram_pressure,
            gpu_frame_overrun,
            active_swarm_cell_mask,
            recommend_layer_skip,
            fact_sig_opt,
            pers_sig_opt,
            bloom_filter_opt,
            contradiction_threshold,
            bloom_repulsion_threshold,
        )
    }
}

/// Zero-copy internal helper synthesizing a [`GatewayFrame`] directly from raw byte slices.
///
/// # Safety
/// Non-null byte pointers must reference at least `fact_byte_len` / `persona_byte_len` bytes.
#[inline]
pub(super) unsafe fn synthesize_multi_origin_from_raw_bytes_with_words(
    vram_pressure: f32,
    gpu_frame_overrun: f32,
    active_swarm_cell_mask: u8,
    recommend_layer_skip: u8,
    fact_bytes: *const u8,
    fact_byte_len: usize,
    fact_confidence: f32,
    fact_zimms_offset48: u64,
    fact_anchor_id: u32,
    persona_bytes: *const u8,
    persona_byte_len: usize,
    persona_strength: f32,
    persona_zimms_offset48: u64,
    persona_anchor_id: u32,
    bloom_filter_opt: *const gcso_contrastive_bloom_filter_t,
    contradiction_threshold: f32,
    bloom_repulsion_threshold: u8,
) -> GcsoResult<(GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS])> {
    if (fact_byte_len > 0 && fact_bytes.is_null())
        || (persona_byte_len > 0 && persona_bytes.is_null())
    {
        return Err(GCSO_ERROR_NULL_POINTER);
    }
    let fact_sig_opt = if fact_bytes.is_null() {
        None
    } else {
        let f_slice: &[u8] = if fact_byte_len == 0 {
            &[]
        } else {
            // SAFETY: Non-nullness for `fact_byte_len > 0` is verified above.
            unsafe { slice::from_raw_parts(fact_bytes, fact_byte_len) }
        };
        Some(ObjectiveFact::new(
            f_slice,
            fact_confidence,
            fact_zimms_offset48,
            fact_anchor_id,
        ))
    };
    let pers_sig_opt = if persona_bytes.is_null() {
        None
    } else {
        let p_slice: &[u8] = if persona_byte_len == 0 {
            &[]
        } else {
            // SAFETY: Non-nullness for `persona_byte_len > 0` is verified above.
            unsafe { slice::from_raw_parts(persona_bytes, persona_byte_len) }
        };
        Some(SubjectivePersona::new(
            p_slice,
            persona_strength,
            persona_zimms_offset48,
            persona_anchor_id,
        ))
    };
    // SAFETY: Validated above.
    unsafe {
        synthesize_multi_origin_generic_with_words(
            vram_pressure,
            gpu_frame_overrun,
            active_swarm_cell_mask,
            recommend_layer_skip,
            fact_sig_opt,
            pers_sig_opt,
            bloom_filter_opt,
            contradiction_threshold,
            bloom_repulsion_threshold,
        )
    }
}

/// Zero-copy internal helper synthesizing a [`GatewayFrame`] directly from raw `f32` embedding slices.
///
/// # Safety
/// Non-null `f32` pointers must be 4-byte aligned and reference at least `fact_len` / `persona_len` elements.
#[inline]
pub(super) unsafe fn synthesize_multi_origin_from_raw_f32_with_words(
    vram_pressure: f32,
    gpu_frame_overrun: f32,
    active_swarm_cell_mask: u8,
    recommend_layer_skip: u8,
    fact_values: *const f32,
    fact_len: usize,
    fact_projection_gain: f32,
    fact_confidence: f32,
    fact_zimms_offset48: u64,
    fact_anchor_id: u32,
    persona_values: *const f32,
    persona_len: usize,
    persona_projection_gain: f32,
    persona_strength: f32,
    persona_zimms_offset48: u64,
    persona_anchor_id: u32,
    bloom_filter_opt: *const gcso_contrastive_bloom_filter_t,
    contradiction_threshold: f32,
    bloom_repulsion_threshold: u8,
) -> GcsoResult<(GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS])> {
    if !branchless_is_finite_f32(fact_projection_gain)
        || fact_projection_gain < 0.0
        || !branchless_is_finite_f32(persona_projection_gain)
        || persona_projection_gain < 0.0
    {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    if fact_len > 0 {
        validate_ptr(fact_values)?;
    }
    if persona_len > 0 {
        validate_ptr(persona_values)?;
    }
    let fact_sig_opt = if fact_values.is_null() {
        None
    } else {
        let f_slice: &[f32] = if fact_len == 0 {
            &[]
        } else {
            // SAFETY: Validated for non-nullness and `f32` alignment when `fact_len > 0`.
            unsafe { slice::from_raw_parts(fact_values, fact_len) }
        };
        Some(ObjectiveFact::new(
            AbiF32EmbeddingRef {
                slice: f_slice,
                gain: fact_projection_gain,
            },
            fact_confidence,
            fact_zimms_offset48,
            fact_anchor_id,
        ))
    };
    let pers_sig_opt = if persona_values.is_null() {
        None
    } else {
        let p_slice: &[f32] = if persona_len == 0 {
            &[]
        } else {
            // SAFETY: Validated for non-nullness and `f32` alignment when `persona_len > 0`.
            unsafe { slice::from_raw_parts(persona_values, persona_len) }
        };
        Some(SubjectivePersona::new(
            AbiF32EmbeddingRef {
                slice: p_slice,
                gain: persona_projection_gain,
            },
            persona_strength,
            persona_zimms_offset48,
            persona_anchor_id,
        ))
    };
    // SAFETY: Validated above.
    unsafe {
        synthesize_multi_origin_generic_with_words(
            vram_pressure,
            gpu_frame_overrun,
            active_swarm_cell_mask,
            recommend_layer_skip,
            fact_sig_opt,
            pers_sig_opt,
            bloom_filter_opt,
            contradiction_threshold,
            bloom_repulsion_threshold,
        )
    }
}

/// Unified register-resident core helper for [`GatewayFrame`] steering across FFI entry points:
/// loads `frame_ref.phase_deltas` once into 8 `u64` SWAR registers (`frm_w`), updates `trail`,
/// steers `q_aligned` (and optional `k_opt`) with $SO(2)$-fused RIPA + PSPM routing, optionally
/// scales `logits_opt`, and returns `report` without copying `GatewayFrame`.
#[inline]
pub(super) fn steer_gateway_frame_with_words_core_safe(
    frame_ref: &GatewayFrame,
    frm_w: &[u64; NUM_SWARM_MICRO_CELLS],
    trail: &mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    head_dim: usize,
    num_heads: usize,
    q_aligned: &mut AlignedSliceMut32<'_, f32>,
    k_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
) -> GcsoResult<CausalProvenanceReport> {
    if frame_ref.qdps_min_step_rad < 0.0
        || !branchless_is_finite_f32(frame_ref.qdps_min_step_rad)
        || frame_ref.ripa_clamp_rad <= 0.0
        || !branchless_is_finite_f32(frame_ref.ripa_clamp_rad)
    {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    if let Some(ref k_ref) = k_opt {
        q_aligned.ensure_disjoint_mut(k_ref)?;
    }

    let report = frame_ref.to_causal_report();
    trail.prev_ptr = trail.current_ptr;
    trail.current_ptr = frame_ref.tagged_ptr;
    trail.step_count = trail.step_count.wrapping_add(1);
    if frame_ref.target_anchor_id != 0 {
        trail.target_anchor_id = frame_ref.target_anchor_id;
        trail.cluster_id = frame_ref.target_anchor_id;
    } else {
        trail.cluster_id = u32::from(frame_ref.dominant_slot & ATTRACTOR_SLOT_ID_MASK);
    }
    trail.sync_flags_with_current_ptr();

    let _ = crate::traits::gateway::accumulate_trail_and_steer_gateway_words_core(
        frame_ref,
        frm_w,
        trail,
        unresolved_damping_u7,
        head_dim,
        num_heads,
        Some(q_aligned),
        k_opt,
        logits_opt,
    )?;

    Ok(report)
}

/// Shared FFI execution tail for [`GatewayFrame`] steering + optional Online Toroidal $k$-Means
/// / DP-Means centroid update + optional output pointer writes.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[inline]
pub(super) unsafe fn execute_gateway_frame_with_words_steer_and_optional_kmeans_ffi(
    frame: &GatewayFrame,
    frm_w: &[u64; NUM_SWARM_MICRO_CELLS],
    trail_inout: *mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
    centroids_inout: *mut gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    split_coherence_threshold: f32,
    frame_out: *mut gcso_gateway_frame_t,
    report_out: *mut gcso_causal_provenance_report_t,
    assigned_centroid_idx_out: *mut usize,
    assigned_coherence_out: *mut f32,
    updated_gravity_mass_out: *mut f32,
) -> GcsoResult<()> {
    validate_ptr(trail_inout)?;
    let has_centroids = !centroids_inout.is_null() && num_centroids > 0;
    if has_centroids {
        validate_ptr(centroids_inout)?;
        if !branchless_is_finite_f32(split_coherence_threshold) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let cent_bytes = num_centroids
            .checked_mul(size_of::<CrystallizedAttractorRecord>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            centroids_inout as usize,
            cent_bytes,
            trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
    }

    // SAFETY: `trail_inout` is validated above.
    let mut trail = unsafe { FfiHandleMut::from_raw(trail_inout)? };
    // SAFETY: `bind_qk_and_logits_32` validates alignment, bounds, and pairwise disjointness.
    let (_engine, mut q_aligned, mut k_opt, mut logits_opt) = unsafe {
        bind_qk_and_logits_32(
            query_tensor,
            key_tensor,
            head_dim,
            num_heads,
            logits_inout,
            vocab_size,
        )?
    };

    let report = steer_gateway_frame_with_words_core_safe(
        frame,
        frm_w,
        &mut trail,
        unresolved_damping_u7,
        head_dim,
        num_heads,
        &mut q_aligned,
        k_opt.as_mut(),
        logits_opt.as_mut(),
    )?;

    if has_centroids {
        // SAFETY: `centroids_inout` is validated non-null, aligned, and disjoint from `trail_inout`.
        let centroids_slice = unsafe { slice::from_raw_parts_mut(centroids_inout, num_centroids) };
        let (assigned_idx, _assigned_anchor_id, assigned_coh, updated_mass, _spawned_new) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_frame_with_words_64(
                centroids_slice,
                frame,
                frm_w,
                split_coherence_threshold,
                Some(&mut *trail),
            );
        // SAFETY: `write_opt_out` validates optional output pointers before writing.
        unsafe {
            write_opt_out(assigned_centroid_idx_out, assigned_idx)?;
            write_opt_out(assigned_coherence_out, assigned_coh)?;
            write_opt_out(updated_gravity_mass_out, updated_mass)?;
        }
    }

    // SAFETY: `write_opt_out` validates optional output pointers before writing.
    unsafe {
        write_opt_out(frame_out, *frame)?;
        write_opt_out(report_out, report)?;
    }
    Ok(())
}

/// Shared FFI execution tail when `frame` is passed by pointer across the C-ABI boundary.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[inline]
pub(super) unsafe fn execute_gateway_frame_steer_and_optional_kmeans_ffi(
    frame: &GatewayFrame,
    trail_inout: *mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
    centroids_inout: *mut gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    split_coherence_threshold: f32,
    frame_out: *mut gcso_gateway_frame_t,
    report_out: *mut gcso_causal_provenance_report_t,
    assigned_centroid_idx_out: *mut usize,
    assigned_coherence_out: *mut f32,
    updated_gravity_mass_out: *mut f32,
) -> GcsoResult<()> {
    let frm_w = frame.phase_words_ref();
    // SAFETY: Validated by `execute_gateway_frame_with_words_steer_and_optional_kmeans_ffi`.
    unsafe {
        execute_gateway_frame_with_words_steer_and_optional_kmeans_ffi(
            frame,
            frm_w,
            trail_inout,
            unresolved_damping_u7,
            query_tensor,
            key_tensor,
            head_dim,
            num_heads,
            logits_inout,
            vocab_size,
            centroids_inout,
            num_centroids,
            split_coherence_threshold,
            frame_out,
            report_out,
            assigned_centroid_idx_out,
            assigned_coherence_out,
            updated_gravity_mass_out,
        )
    }
}

/// Shared register-resident post-routing QDPS + RIPA phase steering helper for Coarse-to-Fine
/// pointer-trail routing entry points (fast-pathing uniform and all-zero filtered words).
#[inline]
pub(super) fn steer_routed_trail_words_qk_safe(
    steer_words: &[u64; NUM_SWARM_MICRO_CELLS],
    trail_flags: u32,
    win_clamp_rad: f32,
    min_step_rad: f32,
    head_dim: usize,
    num_heads: usize,
    q_aligned: &mut AlignedSliceMut32<'_, f32>,
    k_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
) {
    if (trail_flags & TRAIL_FLAG_INERTIAL_SKIP_ACTIVE) == 0 {
        let min_step_q7 = branchless_rad_to_qdps_step_u8(min_step_rad);
        let eff_ripa = branchless_select_f32(
            branchless_is_finite_f32(win_clamp_rad) & (win_clamp_rad > 0.0),
            win_clamp_rad,
            DEFAULT_RIPA_CLAMP_MAX_RAD,
        );
        if min_step_q7 <= 1 {
            if (steer_words[0] != 0) || swar_words_nonzero_64(steer_words) {
                steer_from_swar_words_core_safe(
                    steer_words,
                    0xFF,
                    128,
                    head_dim,
                    num_heads,
                    false,
                    eff_ripa,
                    q_aligned,
                    k_opt,
                );
            }
        } else {
            let swar_words = swar_q7_qdps_filter_words_64(steer_words, min_step_q7);
            if (swar_words[0] != 0) || swar_words_nonzero_64(&swar_words) {
                steer_from_swar_words_core_safe(
                    &swar_words,
                    0xFF,
                    128,
                    head_dim,
                    num_heads,
                    false,
                    eff_ripa,
                    q_aligned,
                    k_opt,
                );
            }
        }
    }
}

/// Shared register-resident crystallized ensemble folding + optional Bloom repulsion + QDPS/RIPA
/// steering helper for [`gcso_dpsr_steer_crystallized_ensemble_zero_copy_safe`] and
/// [`gcso_container_map_and_steer_four_section_zero_copy_safe`].
#[inline]
pub(super) fn steer_crystallized_records_with_optional_bloom_safe(
    records: &[CrystallizedAttractorRecord],
    bloom_opt: Option<&ContrastivePhaseBloomFilter>,
    trail_inout: &mut gcso_pointer_trail_t,
    min_step_rad: f32,
    head_dim: usize,
    num_heads: usize,
    q_aligned: &mut AlignedSliceMut32<'_, f32>,
    k_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
) -> GcsoResult<(f32, u32, f32)> {
    if min_step_rad < 0.0 || !branchless_is_finite_f32(min_step_rad) {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }

    let init_pull =
        branchless_flush_subnormal_f32(branchless_relu_f32(trail_inout.attractor_pull_force));
    let (mut folded_w, tot_pull, dom_id, dom_clamp) =
        CrystallizedAttractorRecord::fold_crystallized_ensemble_words_with_init_pull_64(
            records,
            &mut trail_inout.accumulated_phase_delta,
            init_pull,
        );

    if dom_id != 0 {
        trail_inout.target_anchor_id = dom_id;
        trail_inout.cluster_id = dom_id;
    }
    trail_inout.attractor_pull_force = branchless_min_f32(tot_pull, MAX_ATTRACTOR_PULL_FORCE);
    trail_inout.flags |= TRAIL_FLAG_CRYSTALLIZED_CANDIDATE | TRAIL_FLAG_SYNERGY_RESONANCE;

    if let Some(bloom_ref) = bloom_opt {
        if !bloom_ref.is_empty() {
            let (_density, rep_active) =
                bloom_ref.apply_phase_conjugate_repulsion_words(&mut folded_w, 4, 128);
            if rep_active {
                *trail_phase_words_mut(trail_inout) = folded_w;
                let cur_ptr = gcso_tagged_ptr_t(trail_inout.current_ptr);
                let rep_mode = cur_ptr.action_mode_flags() | ACTION_PRESET_REPULSION_ESCAPE;
                trail_inout.current_ptr = cur_ptr
                    .with_action_and_slot(ATTRACTOR_SLOT_REPULSION, rep_mode)
                    .conjugate_phase()
                    .raw();
                trail_inout.flags |= TRAIL_FLAG_REPULSION_ACTIVE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE;
            }
        }
    }
    trail_inout.sync_flags_with_current_ptr();

    let eff_ripa = branchless_select_f32(
        branchless_is_finite_f32(dom_clamp) & (dom_clamp > 0.0),
        dom_clamp,
        DEFAULT_RIPA_CLAMP_MAX_RAD,
    );
    steer_routed_trail_words_qk_safe(
        &folded_w,
        trail_inout.flags,
        eff_ripa,
        min_step_rad,
        head_dim,
        num_heads,
        q_aligned,
        k_opt,
    );

    Ok((tot_pull, dom_id, eff_ripa))
}

/// Reinterprets a non-null raw `gcso_q7_t` pointer as a zero-copy reference to a 64-head phase array.
///
/// # Safety
/// `ptr` must be non-null and point to at least 64 valid `gcso_q7_t` elements.
#[inline(always)]
pub(super) unsafe fn ref_phase64_from_raw<'a>(
    ptr: *const gcso_q7_t,
) -> GcsoResult<&'a [gcso_q7_t; MAX_ATTENTION_HEADS]> {
    if ptr.is_null() {
        return Err(GCSO_ERROR_NULL_POINTER);
    }
    // SAFETY: `ptr` is non-null and `align_of::<[gcso_q7_t; 64]>() == 1`.
    Ok(unsafe { &*ptr.cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>() })
}

/// Reinterprets a non-null raw mutable `gcso_q7_t` pointer as a zero-copy mutable reference to a 64-head phase array.
///
/// # Safety
/// `ptr` must be non-null, uniquely borrowed, and point to at least 64 valid `gcso_q7_t` elements.
#[inline(always)]
pub(super) unsafe fn mut_phase64_from_raw<'a>(
    ptr: *mut gcso_q7_t,
) -> GcsoResult<&'a mut [gcso_q7_t; MAX_ATTENTION_HEADS]> {
    if ptr.is_null() {
        return Err(GCSO_ERROR_NULL_POINTER);
    }
    // SAFETY: `ptr` is non-null and `align_of::<[gcso_q7_t; 64]>() == 1`.
    Ok(unsafe { &mut *ptr.cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>() })
}

/// Reinterprets a raw `gcso_q7_t` pointer of `num_items * 64` bytes as a zero-copy slice of 64-head phase arrays.
///
/// # Safety
/// When `num_items > 0`, `ptr` must be non-null and point to `num_items * 64` valid `gcso_q7_t` elements.
#[inline(always)]
pub(super) unsafe fn slice_of_phase64_from_raw<'a>(
    ptr: *const gcso_q7_t,
    num_items: usize,
) -> GcsoResult<&'a [[gcso_q7_t; MAX_ATTENTION_HEADS]]> {
    if num_items == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(GCSO_ERROR_NULL_POINTER);
    }
    let _total_bytes = num_items
        .checked_mul(MAX_ATTENTION_HEADS)
        .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
    // SAFETY: Validated non-null and overflow-checked above.
    Ok(unsafe { slice::from_raw_parts(ptr.cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>(), num_items) })
}

/// Validates a `PhaseSteeringEngine` configuration from `(head_dim, num_heads)` and returns `(engine, total_elements)`.
#[inline(always)]
pub(super) fn init_dpsr_engine(
    head_dim: usize,
    num_heads: usize,
) -> GcsoResult<(PhaseSteeringEngine, usize)> {
    let hd_u32 = u32::try_from(head_dim).map_err(|_| GCSO_ERROR_INVALID_ARGUMENT)?;
    let nh_u32 = u32::try_from(num_heads).map_err(|_| GCSO_ERROR_INVALID_ARGUMENT)?;
    let engine = PhaseSteeringEngine::new(hd_u32, nh_u32)?;
    let total_len = head_dim
        .checked_mul(num_heads)
        .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
    Ok((engine, total_len))
}

/// Writes `val` to an optional raw output pointer `out_ptr` after validating natural alignment.
///
/// # Safety
/// If `out_ptr` is non-null, it must be naturally aligned and writable for `T`.
#[inline(always)]
pub(super) unsafe fn write_opt_out<T: Copy>(out_ptr: *mut T, val: T) -> GcsoResult<()> {
    // SAFETY: `FfiHandleMut::from_raw_opt` validates alignment when `out_ptr` is non-null.
    if let Some(mut out_ref) = unsafe { FfiHandleMut::from_raw_opt(out_ptr)? } {
        *out_ref = val;
    }
    Ok(())
}

/// Bridges `src_trail.accumulated_phase_delta` toward `cand_phases` in 8 `u64` SWAR registers
/// by `interpolation_u7` (`0..=128`) with a 1-word fast path when both profiles are word-uniform.
#[inline(always)]
pub(super) fn bridge_trail_words_slerp_u7(
    src_trail: &mut gcso_pointer_trail_t,
    cand_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    interpolation_u7: u8,
) -> [u64; NUM_SWARM_MICRO_CELLS] {
    let mut steer_words = *trail_phase_words_ref(src_trail);
    if interpolation_u7 > 0 {
        let cand_w = load_q7_words_64(cand_phases);
        if !swar_words_equal_64(&steer_words, &cand_w) {
            let s_u64 = branchless_select_u8(interpolation_u7 > 128, 128, interpolation_u7) as u64;
            let bias_restore = (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
            steer_words = swar_slerp_words_with_bias_64(&steer_words, &cand_w, s_u64, bias_restore);
            *trail_phase_words_mut(src_trail) = steer_words;
        }
    }
    steer_words
}

/// Synchronizes `src_trail.current_ptr` with a winning macro centroid's 48-bit ZIMMS section address and tags.
#[inline(always)]
pub(super) fn sync_winner_centroid_ptr_to_trail(
    src_trail: &mut gcso_pointer_trail_t,
    winner: &CrystallizedAttractorRecord,
) {
    if winner.sample_count > 0 && winner.zimms_section_offset48 != 0 {
        let packed_tag = pack_action_slot_tag(winner.dominant_slot, winner.action_mode_flags);
        let cur_phase = gcso_tagged_ptr_t(src_trail.current_ptr).unpack_phase();
        src_trail.prev_ptr = src_trail.current_ptr;
        src_trail.current_ptr =
            gcso_tagged_ptr_t::pack(winner.zimms_section_offset48, packed_tag, cur_phase).raw();
    }
}

/// Validates and binds `query_tensor` (mandatory, 32B aligned) and `key_tensor` (optional, 32B
/// aligned, disjoint from `query_tensor`) for `(head_dim, num_heads)`.
///
/// # Safety
/// Pointers must reference valid buffers of at least `head_dim * num_heads` `f32` elements.
#[inline(always)]
pub(super) unsafe fn bind_qk_32<'a>(
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
) -> GcsoResult<(
    PhaseSteeringEngine,
    AlignedSliceMut32<'a, f32>,
    Option<AlignedSliceMut32<'a, f32>>,
)> {
    validate_ptr_aligned(query_tensor, 32)?;
    let (engine, total_len) = init_dpsr_engine(head_dim, num_heads)?;
    // SAFETY: `AlignedSliceMut32` validates non-nullness and 32-byte alignment.
    let (q_aligned, k_opt) = unsafe {
        (
            AlignedSliceMut32::from_raw_parts_mut(query_tensor, total_len)?,
            AlignedSliceMut32::from_raw_parts_mut_opt(key_tensor, total_len)?,
        )
    };
    if let Some(ref k_ref) = k_opt {
        q_aligned.ensure_disjoint_mut(k_ref)?;
    }
    Ok((engine, q_aligned, k_opt))
}

/// Validates and binds `query_tensor` (mandatory, 32B aligned), `key_tensor` (optional, 32B aligned),
/// and `logits_inout` (optional, 32B aligned, disjoint from Q/K) for `(head_dim, num_heads)`.
///
/// # Safety
/// Pointers must reference valid buffers of the specified lengths.
#[inline(always)]
pub(super) unsafe fn bind_qk_and_logits_32<'a>(
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
) -> GcsoResult<(
    PhaseSteeringEngine,
    AlignedSliceMut32<'a, f32>,
    Option<AlignedSliceMut32<'a, f32>>,
    Option<AlignedSliceMut32<'a, f32>>,
)> {
    // SAFETY: Validated by `bind_qk_32`.
    let (engine, q_aligned, k_opt) =
        unsafe { bind_qk_32(query_tensor, key_tensor, head_dim, num_heads)? };
    let logits_opt = if logits_inout.is_null() || vocab_size == 0 {
        None
    } else {
        // SAFETY: Validated by `AlignedSliceMut32::from_raw_parts_mut_opt`.
        let l_slice =
            unsafe { AlignedSliceMut32::from_raw_parts_mut_opt(logits_inout, vocab_size)? };
        if let Some(ref l_ref) = l_slice {
            q_aligned.ensure_disjoint_mut(l_ref)?;
            if let Some(ref k_ref) = k_opt {
                k_ref.ensure_disjoint_mut(l_ref)?;
            }
        }
        l_slice
    };
    Ok((engine, q_aligned, k_opt, logits_opt))
}

/// Validates a null-terminated C string pointer and ensures valid UTF-8 encoding in `#![no_std]` mode.
///
/// # Safety
/// `c_str_ptr` must be non-null and point to a valid NUL-terminated C string.
#[inline]
pub(super) unsafe fn validate_c_utf8_str<'a>(c_str_ptr: *const c_char) -> GcsoResult<&'a str> {
    if c_str_ptr.is_null() {
        return Err(GCSO_ERROR_NULL_POINTER);
    }
    // SAFETY: `c_str_ptr` is non-null and caller guarantees NUL-termination.
    let c_str = unsafe { CStr::from_ptr(c_str_ptr) };
    c_str.to_str().map_err(|_| GCSO_ERROR_INVALID_ARGUMENT)
}

/// Resolves flexible `(head_dim, num_heads)` dimensions for flat tensor C-ABI entry points.
#[inline(always)]
pub(super) const fn resolve_flat_tensor_heads(dim: usize) -> GcsoResult<(u32, u32)> {
    if dim == 0 || (dim & 1) != 0 || dim > (u32::MAX as usize) {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    if dim >= 128 && (dim % 128 == 0) {
        let heads = dim / 128;
        if heads <= MAX_ATTENTION_HEADS {
            return Ok((128, heads as u32));
        }
        let hd = dim / MAX_ATTENTION_HEADS;
        if (hd & 1) == 0 {
            return Ok((hd as u32, MAX_ATTENTION_HEADS as u32));
        }
    }
    Ok((dim as u32, 1))
}
