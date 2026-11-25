// name: src/core/src/runtime/abi/dispatch.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Internal zero-cost stateless trait dispatchers and raw-pointer validation helpers.

use super::*;

// ===================================================================
// Internal Zero-Cost Stateless Trait Dispatchers & Validation Helpers
// ===================================================================

pub(super) struct AbiLedgerDispatcher;
impl FastPathLedger for AbiLedgerDispatcher {
    #[inline]
    fn evaluate_auto_tune(
        &self,
        slot: &lmnk_ledger_slot_t,
        config_out: &mut lmnk_config_t,
    ) -> LmnkResult<()> {
        if config_out.head_dim == 0 {
            init_default_config(config_out)?;
        }
        // Single 64-bit SWAR instruction counting all 8 bytes >= 0x80 (`0` loops, `0` branches).
        let spike_count =
            (u64::from_le_bytes(slot.telemetry_mini_ledger) & 0x8080_8080_8080_8080).count_ones();
        let base_clamp = branchless_select_f32(
            config_out.phase_clamp_max_rad > 0.0,
            config_out.phase_clamp_max_rad,
            DEFAULT_PHASE_CLAMP_MAX_RAD,
        );
        let damping = 1.0 - 0.025 * (spike_count as f32);
        config_out.phase_clamp_max_rad = branchless_clamp_f32(base_clamp * damping, 0.01, 0.25);
        config_out.ledger_mode = (slot.mode & 0xFF) as u8;
        Ok(())
    }
}

/// Branchlessly checks pointer non-nullness and natural alignment for type `T`.
#[inline(always)]
#[must_use]
pub fn is_aligned<T>(ptr: *const T) -> bool {
    let align = align_of::<T>();
    (!ptr.is_null()) & (((ptr as usize) & (align - 1)) == 0)
}

/// Validates a non-null, naturally aligned pointer or opaque handle.
#[inline(always)]
pub(super) fn validate_ptr<T>(ptr: *const T) -> LmnkResult<()> {
    if ptr.is_null() {
        Err(LMNK_ERR_NULL_POINTER)
    } else if !is_aligned(ptr) {
        Err(LMNK_ERR_ALIGNMENT)
    } else {
        Ok(())
    }
}

/// Validates a non-null pointer against an explicit power-of-two alignment boundary `align`.
#[inline(always)]
pub(super) fn validate_ptr_aligned<T>(ptr: *const T, align: usize) -> LmnkResult<()> {
    if ptr.is_null() {
        Err(LMNK_ERR_NULL_POINTER)
    } else if !is_ptr_aligned_to(ptr, align) {
        Err(LMNK_ERR_ALIGNMENT)
    } else {
        Ok(())
    }
}

/// Validates an optional pointer's alignment and returns its `(addr, byte_len)` span (`(0, 0)` on `NULL`).
#[inline(always)]
pub(super) fn opt_span<T>(ptr: *const T) -> LmnkResult<(usize, usize)> {
    if ptr.is_null() {
        Ok((0, 0))
    } else if !is_aligned(ptr) {
        Err(LMNK_ERR_ALIGNMENT)
    } else {
        Ok((ptr as usize, size_of::<T>()))
    }
}

/// Validates that all mutable memory regions in `muts` are pairwise disjoint and none overlap
/// any immutable input region in `reads` (zero-length entries are ignored).
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if any mutable-mutable or mutable-read byte ranges overlap.
#[inline]
pub(super) fn check_disjoint_rw(
    muts: &[(usize, usize)],
    reads: &[(usize, usize)],
) -> LmnkResult<()> {
    let mut i = 0usize;
    while i < muts.len() {
        let (a_start, a_len) = muts[i];
        if a_len > 0 {
            let mut j = i + 1;
            while j < muts.len() {
                let (b_start, b_len) = muts[j];
                if raw_byte_ranges_overlap(a_start, a_len, b_start, b_len) {
                    return Err(LMNK_ERR_INVALID_ARGUMENT);
                }
                j += 1;
            }
            let mut k = 0usize;
            while k < reads.len() {
                let (r_start, r_len) = reads[k];
                if raw_byte_ranges_overlap(a_start, a_len, r_start, r_len) {
                    return Err(LMNK_ERR_INVALID_ARGUMENT);
                }
                k += 1;
            }
        }
        i += 1;
    }
    Ok(())
}

/// Safely creates an immutable slice `&'a [T]` after validating non-nullness, natural alignment,
/// `len * size_of::<T>() <= isize::MAX`, and non-wrapping address bounds (`#![no_std]` safe).
///
/// # Safety
/// For `len > 0`, `ptr` must point to at least `len` valid, initialized instances of `T` not
/// concurrently mutated for lifetime `'a`.
#[inline(always)]
pub(super) unsafe fn checked_slice_from_raw_parts<'a, T>(
    ptr: *const T,
    len: usize,
) -> LmnkResult<&'a [T]> {
    if len == 0 {
        return Ok(&[]);
    }
    validate_ptr(ptr)?;
    let _ = validate_raw_slice_bounds(ptr, len)?;
    // SAFETY: `ptr` is non-null, naturally aligned, within `isize::MAX` bytes without address
    // wraparound, and caller guarantees valid initialized memory for lifetime `'a`.
    Ok(unsafe { slice::from_raw_parts(ptr, len) })
}

/// Safely creates a mutable slice `&'a mut [T]` after validating non-nullness, natural alignment,
/// `len * size_of::<T>() <= isize::MAX`, and non-wrapping address bounds (`#![no_std]` safe).
///
/// # Safety
/// For `len > 0`, `ptr` must point to at least `len` valid, initialized instances of `T` with
/// exclusive mutable access for lifetime `'a`.
#[inline(always)]
pub(super) unsafe fn checked_slice_from_raw_parts_mut<'a, T>(
    ptr: *mut T,
    len: usize,
) -> LmnkResult<&'a mut [T]> {
    if len == 0 {
        return Ok(&mut []);
    }
    validate_ptr(ptr)?;
    let _ = validate_raw_slice_bounds(ptr, len)?;
    // SAFETY: `ptr` is non-null, naturally aligned, within `isize::MAX` bytes without address
    // wraparound, and caller guarantees exclusive access for lifetime `'a`.
    Ok(unsafe { slice::from_raw_parts_mut(ptr, len) })
}

/// Zero-copy borrowed 64-head Q7 phase wrapper implementing [`RawPhaseProjectable`] so FFI callers
/// passing `*const lmnk_q7_t` into 2-Origin Gateway synthesis load directly into 8 `u64` SWAR
/// registers without intermediate 64-byte stack array copies.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub(super) struct AbiPhaseRef64<'a>(&'a [lmnk_q7_t; MAX_ATTENTION_HEADS]);

impl RawPhaseProjectable for AbiPhaseRef64<'_> {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [lmnk_q7_t; MAX_ATTENTION_HEADS] {
        *self.0
    }

    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
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
    fn project_to_q7_64(&self) -> [lmnk_q7_t; MAX_ATTENTION_HEADS] {
        if self.gain > 0.0 {
            branchless_project_f32_slice_to_q7_64(self.slice, self.gain)
        } else {
            project_f32_slice_to_q7_64(self.slice)
        }
    }

    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
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
/// [`ContradictionGuard`].
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
    fn project_grounding_phases_64(&self) -> [lmnk_q7_t; MAX_ATTENTION_HEADS] {
        [0i8; MAX_ATTENTION_HEADS]
    }

    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        [0u64; NUM_STIGMERGIC_GRANULES]
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
        candidate_phases: &[lmnk_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let words = load_q7_words_64(candidate_phases);
        self.evaluate_repulsion_severity_from_words(&words, candidate_phases)
    }

    #[inline(always)]
    fn evaluate_repulsion_severity_from_words(
        &self,
        candidate_words: &[u64; NUM_STIGMERGIC_GRANULES],
        _candidate_phases: &[lmnk_q7_t; MAX_ATTENTION_HEADS],
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

/// Bundled Origin-1 system pressure and layer-skip hints passed across internal FFI helpers.
#[derive(Clone, Copy)]
pub(super) struct AbiSystemSpec {
    pub(super) vram_pressure: f32,
    pub(super) gpu_frame_overrun: f32,
    pub(super) active_swarm_cell_mask: u8,
    pub(super) recommend_layer_skip: u8,
}

/// Bundled optional Bloom contradiction-guard parameters passed across internal FFI helpers.
#[derive(Clone, Copy)]
pub(super) struct AbiBloomSpec {
    pub(super) bloom_filter_opt: *const lmnk_bloom_filter_t,
    pub(super) contradiction_threshold: f32,
    pub(super) bloom_repulsion_threshold: u8,
}

/// Bundled raw Q7 phase-pointer origin inputs for `synthesize_multi_origin_from_raw_ptrs_with_words`.
#[derive(Clone, Copy)]
pub(super) struct AbiRawPhaseOrigin {
    pub(super) phases: *const lmnk_q7_t,
    pub(super) strength: f32,
    pub(super) mapped_offset48: u64,
    pub(super) anchor_id: u32,
}

/// Bundled raw byte-slice origin inputs for `synthesize_multi_origin_from_raw_bytes_with_words`.
#[derive(Clone, Copy)]
pub(super) struct AbiRawBytesOrigin {
    pub(super) bytes: *const u8,
    pub(super) byte_len: usize,
    pub(super) strength: f32,
    pub(super) mapped_offset48: u64,
    pub(super) anchor_id: u32,
}

/// Bundled raw `f32` embedding origin inputs for `synthesize_multi_origin_from_raw_f32_with_words`.
#[derive(Clone, Copy)]
pub(super) struct AbiRawF32Origin {
    pub(super) values: *const f32,
    pub(super) len: usize,
    pub(super) projection_gain: f32,
    pub(super) strength: f32,
    pub(super) mapped_offset48: u64,
    pub(super) anchor_id: u32,
}

/// Bundled tensor-shape metadata `(head_dim, num_heads)` for internal FFI steering helpers.
#[derive(Clone, Copy)]
pub(super) struct AbiTensorShape {
    pub(super) head_dim: usize,
    pub(super) num_heads: usize,
}

/// Bundled raw Q/K/logits tensor buffer pointers and dimensions for internal FFI gateway steering.
#[derive(Clone, Copy)]
pub(super) struct AbiSteerBuffers {
    pub(super) query_tensor: *mut f32,
    pub(super) key_tensor: *mut f32,
    pub(super) head_dim: usize,
    pub(super) num_heads: usize,
    pub(super) logits_inout: *mut f32,
    pub(super) vocab_size: usize,
}

/// Bundled optional toroidal $k$-means centroid buffer and FFI output pointers.
#[derive(Clone, Copy)]
pub(super) struct AbiKmeansTail {
    pub(super) centroids_inout: *mut lmnk_phase_anchor_record_t,
    pub(super) num_centroids: usize,
    pub(super) split_coherence_threshold: f32,
    pub(super) frame_out: *mut lmnk_gateway_frame_t,
    pub(super) report_out: *mut lmnk_causal_report_t,
    pub(super) assigned_centroid_idx_out: *mut usize,
    pub(super) assigned_coherence_out: *mut f32,
    pub(super) updated_gravity_mass_out: *mut f32,
}

/// Generic zero-copy helper synthesizing a [`GatewayFrame`] from Origin 1 (`SystemConstraint`)
/// and Origin 2 (`ObjectiveFact<F>`, `SubjectivePersona<P>`, optional `AbiBorrowedBloomGuard`).
///
/// # Safety
/// `bloom.bloom_filter_opt` must be `NULL` or a valid aligned pointer to [`lmnk_bloom_filter_t`].
#[inline]
pub(super) unsafe fn synthesize_multi_origin_generic_with_words<
    F: RawPhaseProjectable,
    P: RawPhaseProjectable,
>(
    sys: AbiSystemSpec,
    fact_sig_opt: Option<ObjectiveFact<F>>,
    pers_sig_opt: Option<SubjectivePersona<P>>,
    bloom: AbiBloomSpec,
) -> LmnkResult<(GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES])> {
    // SAFETY: `FfiHandleRef::from_raw_opt` validates non-nullness and alignment of `bloom_filter_opt`.
    let bloom_ref_opt = unsafe { FfiHandleRef::from_raw_opt(bloom.bloom_filter_opt)? };
    let sys_sig = SystemConstraint::new(
        (),
        sys.vram_pressure,
        sys.gpu_frame_overrun,
        sys.active_swarm_cell_mask,
    )
    .with_layer_skip(sys.recommend_layer_skip != 0);
    Ok(match bloom_ref_opt {
        Some(b_ref) => {
            let guard = AbiBorrowedBloomGuard::new(
                &b_ref,
                bloom.contradiction_threshold,
                bloom.bloom_repulsion_threshold,
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
/// Non-null phase pointers must point to at least 64 valid `lmnk_q7_t` elements.
#[inline]
pub(super) unsafe fn synthesize_multi_origin_from_raw_ptrs_with_words(
    sys: AbiSystemSpec,
    fact: AbiRawPhaseOrigin,
    persona: AbiRawPhaseOrigin,
    bloom: AbiBloomSpec,
) -> LmnkResult<(GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES])> {
    let fact_sig_opt = if fact.phases.is_null() {
        None
    } else {
        // SAFETY: `fact.phases` is non-null and `align_of::<[lmnk_q7_t; 64]>() == 1`.
        let f_ref = unsafe { &*fact.phases.cast::<[lmnk_q7_t; MAX_ATTENTION_HEADS]>() };
        Some(ObjectiveFact::new(
            AbiPhaseRef64(f_ref),
            fact.strength,
            fact.mapped_offset48,
            fact.anchor_id,
        ))
    };
    let pers_sig_opt = if persona.phases.is_null() {
        None
    } else {
        // SAFETY: `persona.phases` is non-null and `align_of::<[lmnk_q7_t; 64]>() == 1`.
        let p_ref = unsafe { &*persona.phases.cast::<[lmnk_q7_t; MAX_ATTENTION_HEADS]>() };
        Some(SubjectivePersona::new(
            AbiPhaseRef64(p_ref),
            persona.strength,
            persona.mapped_offset48,
            persona.anchor_id,
        ))
    };
    // SAFETY: `fact.phases` and `persona.phases` were validated non-null and bound to 64-element
    // arrays above; `bloom.bloom_filter_opt` is validated non-null and aligned via
    // `FfiHandleRef::from_raw_opt` inside `synthesize_multi_origin_generic_with_words`.
    unsafe { synthesize_multi_origin_generic_with_words(sys, fact_sig_opt, pers_sig_opt, bloom) }
}

/// Zero-copy internal helper synthesizing a [`GatewayFrame`] directly from raw byte slices.
///
/// # Safety
/// Non-null byte pointers must reference at least `fact.byte_len` / `persona.byte_len` bytes.
#[inline]
pub(super) unsafe fn synthesize_multi_origin_from_raw_bytes_with_words(
    sys: AbiSystemSpec,
    fact: AbiRawBytesOrigin,
    persona: AbiRawBytesOrigin,
    bloom: AbiBloomSpec,
) -> LmnkResult<(GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES])> {
    if (fact.byte_len > 0 && fact.bytes.is_null())
        || (persona.byte_len > 0 && persona.bytes.is_null())
    {
        return Err(LMNK_ERR_NULL_POINTER);
    }
    let fact_sig_opt = if fact.bytes.is_null() {
        None
    } else {
        // SAFETY: `checked_slice_from_raw_parts` validates non-null, alignment, and `<= isize::MAX` bounds.
        let f_slice = unsafe { checked_slice_from_raw_parts(fact.bytes, fact.byte_len)? };
        Some(ObjectiveFact::new(
            f_slice,
            fact.strength,
            fact.mapped_offset48,
            fact.anchor_id,
        ))
    };
    let pers_sig_opt = if persona.bytes.is_null() {
        None
    } else {
        // SAFETY: `checked_slice_from_raw_parts` validates non-null, alignment, and `<= isize::MAX` bounds.
        let p_slice = unsafe { checked_slice_from_raw_parts(persona.bytes, persona.byte_len)? };
        Some(SubjectivePersona::new(
            p_slice,
            persona.strength,
            persona.mapped_offset48,
            persona.anchor_id,
        ))
    };
    // SAFETY: `fact.bytes` and `persona.bytes` were verified non-null for non-zero byte lengths
    // and sliced within bounds above; `bloom.bloom_filter_opt` is validated non-null and aligned via
    // `FfiHandleRef::from_raw_opt` inside `synthesize_multi_origin_generic_with_words`.
    unsafe { synthesize_multi_origin_generic_with_words(sys, fact_sig_opt, pers_sig_opt, bloom) }
}

/// Zero-copy internal helper synthesizing a [`GatewayFrame`] directly from raw `f32` embedding slices.
///
/// # Safety
/// Non-null `f32` pointers must be 4-byte aligned and reference at least `fact.len` / `persona.len` elements.
#[inline]
pub(super) unsafe fn synthesize_multi_origin_from_raw_f32_with_words(
    sys: AbiSystemSpec,
    fact: AbiRawF32Origin,
    persona: AbiRawF32Origin,
    bloom: AbiBloomSpec,
) -> LmnkResult<(GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES])> {
    if !branchless_is_finite_f32(fact.projection_gain)
        || fact.projection_gain < 0.0
        || !branchless_is_finite_f32(persona.projection_gain)
        || persona.projection_gain < 0.0
    {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    if fact.len > 0 {
        validate_ptr(fact.values)?;
    }
    if persona.len > 0 {
        validate_ptr(persona.values)?;
    }
    let fact_sig_opt = if fact.values.is_null() {
        None
    } else {
        // SAFETY: `checked_slice_from_raw_parts` validates non-null, `f32` alignment, and `<= isize::MAX` bounds.
        let f_slice = unsafe { checked_slice_from_raw_parts(fact.values, fact.len)? };
        Some(ObjectiveFact::new(
            AbiF32EmbeddingRef {
                slice: f_slice,
                gain: fact.projection_gain,
            },
            fact.strength,
            fact.mapped_offset48,
            fact.anchor_id,
        ))
    };
    let pers_sig_opt = if persona.values.is_null() {
        None
    } else {
        // SAFETY: `checked_slice_from_raw_parts` validates non-null, `f32` alignment, and `<= isize::MAX` bounds.
        let p_slice = unsafe { checked_slice_from_raw_parts(persona.values, persona.len)? };
        Some(SubjectivePersona::new(
            AbiF32EmbeddingRef {
                slice: p_slice,
                gain: persona.projection_gain,
            },
            persona.strength,
            persona.mapped_offset48,
            persona.anchor_id,
        ))
    };
    // SAFETY: `fact.values` and `persona.values` were verified non-null, `f32`-aligned, and
    // sliced within bounds above; `bloom.bloom_filter_opt` is validated non-null and aligned via
    // `FfiHandleRef::from_raw_opt` inside `synthesize_multi_origin_generic_with_words`.
    unsafe { synthesize_multi_origin_generic_with_words(sys, fact_sig_opt, pers_sig_opt, bloom) }
}

/// Unified register-resident core helper for [`GatewayFrame`] steering across FFI entry points:
/// loads `frame_ref.phase_deltas` once into 8 `u64` SWAR registers (`frm_w`), updates `trail`,
/// steers `q_aligned` (and optional `k_opt`) with $SO(2)$-fused PhaseClamp + sub-head router routing, optionally
/// scales `logits_opt`, and returns `report` without copying `GatewayFrame`.
#[inline]
pub(super) fn steer_gateway_frame_with_words_core_safe(
    frame_ref: &GatewayFrame,
    frm_w: &[u64; NUM_STIGMERGIC_GRANULES],
    trail: &mut lmnk_pointer_trail_t,
    shape: crate::runtime::gateway::orchestrator::GatewaySteerShape,
    q_aligned: &mut AlignedSliceMut32<'_, f32>,
    k_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
) -> LmnkResult<CausalProvenanceReport> {
    if frame_ref.phase_step_min_rad < 0.0
        || !branchless_is_finite_f32(frame_ref.phase_step_min_rad)
        || frame_ref.phase_clamp_rad <= 0.0
        || !branchless_is_finite_f32(frame_ref.phase_clamp_rad)
    {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
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

    let _ = accumulate_trail_and_steer_gateway_words_core(
        frame_ref,
        frm_w,
        trail,
        shape,
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
/// Pointers must satisfy non-null, alignment, bounds, and `LMNK_RESTRICT` invariants.
#[inline]
pub(super) unsafe fn execute_gateway_frame_with_words_steer_and_optional_kmeans_ffi(
    frame: &GatewayFrame,
    frm_w: &[u64; NUM_STIGMERGIC_GRANULES],
    trail_inout: *mut lmnk_pointer_trail_t,
    unresolved_damping_u7: u8,
    buffers: AbiSteerBuffers,
    tail: AbiKmeansTail,
) -> LmnkResult<()> {
    validate_ptr(trail_inout)?;
    let (_, total_len) = init_phase_engine(buffers.head_dim, buffers.num_heads)?;
    let qk_bytes = validate_raw_slice_bounds(buffers.query_tensor, total_len)?;
    let logits_bytes = if !buffers.logits_inout.is_null() && buffers.vocab_size > 0 {
        validate_raw_slice_bounds(buffers.logits_inout, buffers.vocab_size)?
    } else {
        0
    };
    if raw_byte_ranges_overlap(
        trail_inout as usize,
        size_of::<lmnk_pointer_trail_t>(),
        buffers.query_tensor as usize,
        qk_bytes,
    ) || (!buffers.key_tensor.is_null()
        && raw_byte_ranges_overlap(
            trail_inout as usize,
            size_of::<lmnk_pointer_trail_t>(),
            buffers.key_tensor as usize,
            qk_bytes,
        ))
        || (logits_bytes > 0
            && raw_byte_ranges_overlap(
                trail_inout as usize,
                size_of::<lmnk_pointer_trail_t>(),
                buffers.logits_inout as usize,
                logits_bytes,
            ))
    {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }

    let has_centroids = !tail.centroids_inout.is_null() && tail.num_centroids > 0;
    if has_centroids {
        validate_ptr(tail.centroids_inout)?;
        if !branchless_is_finite_f32(tail.split_coherence_threshold) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let cent_bytes = validate_raw_slice_bounds(tail.centroids_inout, tail.num_centroids)?;
        if raw_byte_ranges_overlap(
            tail.centroids_inout as usize,
            cent_bytes,
            trail_inout as usize,
            size_of::<lmnk_pointer_trail_t>(),
        ) || raw_byte_ranges_overlap(
            tail.centroids_inout as usize,
            cent_bytes,
            buffers.query_tensor as usize,
            qk_bytes,
        ) || (!buffers.key_tensor.is_null()
            && raw_byte_ranges_overlap(
                tail.centroids_inout as usize,
                cent_bytes,
                buffers.key_tensor as usize,
                qk_bytes,
            ))
            || (logits_bytes > 0
                && raw_byte_ranges_overlap(
                    tail.centroids_inout as usize,
                    cent_bytes,
                    buffers.logits_inout as usize,
                    logits_bytes,
                ))
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
    }

    // SAFETY: `trail_inout` and tensor pointers are validated non-null, aligned, bounds-checked,
    // and pairwise disjoint above before constructing mutable borrows.
    let (mut trail, (_engine, mut q_aligned, mut k_opt, mut logits_opt)) = unsafe {
        (
            FfiHandleMut::from_raw(trail_inout)?,
            bind_qk_and_logits_32(
                buffers.query_tensor,
                buffers.key_tensor,
                buffers.head_dim,
                buffers.num_heads,
                buffers.logits_inout,
                buffers.vocab_size,
            )?,
        )
    };

    let report = steer_gateway_frame_with_words_core_safe(
        frame,
        frm_w,
        &mut trail,
        crate::runtime::gateway::orchestrator::GatewaySteerShape {
            unresolved_damping_u7,
            head_dim: buffers.head_dim,
            num_heads: buffers.num_heads,
        },
        &mut q_aligned,
        k_opt.as_mut(),
        logits_opt.as_mut(),
    )?;

    if has_centroids {
        // SAFETY: `centroids_inout` is validated non-null, aligned, bounded, and disjoint from `trail_inout`.
        let centroids_slice =
            unsafe { checked_slice_from_raw_parts_mut(tail.centroids_inout, tail.num_centroids)? };
        let (assigned_idx, _assigned_anchor_id, assigned_coh, updated_mass, _spawned_new) =
            PhaseAnchorRecord::step_online_toroidal_kmeans_frame_with_words_64(
                centroids_slice,
                frame,
                frm_w,
                tail.split_coherence_threshold,
                Some(&mut *trail),
            );
        // SAFETY: `write_opt_out` validates optional output pointers before writing.
        unsafe {
            write_opt_out(tail.assigned_centroid_idx_out, assigned_idx)?;
            write_opt_out(tail.assigned_coherence_out, assigned_coh)?;
            write_opt_out(tail.updated_gravity_mass_out, updated_mass)?;
        }
    }

    // SAFETY: `write_opt_out` validates optional output pointers before writing.
    unsafe {
        write_opt_out(tail.frame_out, *frame)?;
        write_opt_out(tail.report_out, report)?;
    }
    Ok(())
}

/// Shared FFI execution tail when `frame` is passed by pointer across the C-ABI boundary.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `LMNK_RESTRICT` invariants.
#[inline]
pub(super) unsafe fn execute_gateway_frame_steer_and_optional_kmeans_ffi(
    frame: &GatewayFrame,
    trail_inout: *mut lmnk_pointer_trail_t,
    unresolved_damping_u7: u8,
    buffers: AbiSteerBuffers,
    tail: AbiKmeansTail,
) -> LmnkResult<()> {
    let frm_w = frame.phase_words_ref();
    // SAFETY: Validated by `execute_gateway_frame_with_words_steer_and_optional_kmeans_ffi`.
    unsafe {
        execute_gateway_frame_with_words_steer_and_optional_kmeans_ffi(
            frame,
            frm_w,
            trail_inout,
            unresolved_damping_u7,
            buffers,
            tail,
        )
    }
}

/// Shared register-resident post-routing `PhaseStepFilter` + `PhaseClamp` phase steering helper for Coarse-to-Fine
/// pointer-trail routing entry points (fast-pathing uniform and all-zero filtered words).
#[inline]
pub(super) fn steer_routed_trail_words_qk_safe(
    steer_words: &[u64; NUM_STIGMERGIC_GRANULES],
    trail_flags: u32,
    win_clamp_rad: f32,
    min_step_rad: f32,
    shape: AbiTensorShape,
    q_aligned: &mut AlignedSliceMut32<'_, f32>,
    k_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
) {
    if (trail_flags & TRAIL_FLAG_INERTIAL_SKIP_ACTIVE) == 0 {
        let min_step_q7 = branchless_rad_to_phase_step_u8(min_step_rad);
        let eff_clamp_rad = branchless_select_f32(
            branchless_is_finite_f32(win_clamp_rad) & (win_clamp_rad > 0.0),
            win_clamp_rad,
            DEFAULT_PHASE_CLAMP_MAX_RAD,
        );
        if min_step_q7 <= 1 {
            if (steer_words[0] != 0) || swar_words_nonzero_64(steer_words) {
                steer_from_swar_words_core_safe(
                    steer_words,
                    0xFF,
                    128,
                    shape.head_dim,
                    shape.num_heads,
                    false,
                    eff_clamp_rad,
                    q_aligned,
                    k_opt,
                );
            }
        } else {
            let swar_words = swar_q7_phase_step_filter_words_64(steer_words, min_step_q7);
            if (swar_words[0] != 0) || swar_words_nonzero_64(&swar_words) {
                steer_from_swar_words_core_safe(
                    &swar_words,
                    0xFF,
                    128,
                    shape.head_dim,
                    shape.num_heads,
                    false,
                    eff_clamp_rad,
                    q_aligned,
                    k_opt,
                );
            }
        }
    }
}

/// Shared register-resident crystallized ensemble folding + optional Bloom repulsion + PhaseStepFilter/PhaseClamp
/// steering helper for [`lmnk_container_map_and_steer_zero_copy_safe`].
#[inline]
pub(super) fn steer_crystallized_records_with_optional_bloom_safe(
    records: &[PhaseAnchorRecord],
    bloom_opt: Option<&ContrastivePhaseBloomFilter>,
    trail_inout: &mut lmnk_pointer_trail_t,
    min_step_rad: f32,
    shape: AbiTensorShape,
    q_aligned: &mut AlignedSliceMut32<'_, f32>,
    k_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
) -> LmnkResult<(f32, u32, f32)> {
    if min_step_rad < 0.0 || !branchless_is_finite_f32(min_step_rad) {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }

    let init_pull =
        branchless_flush_subnormal_f32(branchless_relu_f32(trail_inout.attractor_pull_force));
    let (mut folded_w, tot_pull, dom_id, dom_clamp) =
        PhaseAnchorRecord::fold_crystallized_ensemble_words_with_init_pull_64(
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
                let cur_ptr = TaggedPtr(trail_inout.current_ptr);
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

    let eff_clamp_rad = branchless_select_f32(
        branchless_is_finite_f32(dom_clamp) & (dom_clamp > 0.0),
        dom_clamp,
        DEFAULT_PHASE_CLAMP_MAX_RAD,
    );
    steer_routed_trail_words_qk_safe(
        &folded_w,
        trail_inout.flags,
        eff_clamp_rad,
        min_step_rad,
        shape,
        q_aligned,
        k_opt,
    );

    Ok((tot_pull, dom_id, eff_clamp_rad))
}

/// Reinterprets a non-null raw `lmnk_q7_t` pointer as a zero-copy reference to a 64-head phase array.
///
/// # Safety
/// `ptr` must be non-null and point to at least 64 valid `lmnk_q7_t` elements.
#[inline(always)]
pub(super) unsafe fn ref_phase64_from_raw<'a>(
    ptr: *const lmnk_q7_t,
) -> LmnkResult<&'a [lmnk_q7_t; MAX_ATTENTION_HEADS]> {
    if ptr.is_null() {
        return Err(LMNK_ERR_NULL_POINTER);
    }
    // SAFETY: `ptr` is non-null and `align_of::<[lmnk_q7_t; 64]>() == 1`.
    Ok(unsafe { &*ptr.cast::<[lmnk_q7_t; MAX_ATTENTION_HEADS]>() })
}

/// Reinterprets a non-null raw mutable `lmnk_q7_t` pointer as a zero-copy mutable reference to a 64-head phase array.
///
/// # Safety
/// `ptr` must be non-null, uniquely borrowed, and point to at least 64 valid `lmnk_q7_t` elements.
#[inline(always)]
pub(super) unsafe fn mut_phase64_from_raw<'a>(
    ptr: *mut lmnk_q7_t,
) -> LmnkResult<&'a mut [lmnk_q7_t; MAX_ATTENTION_HEADS]> {
    if ptr.is_null() {
        return Err(LMNK_ERR_NULL_POINTER);
    }
    // SAFETY: `ptr` is non-null and `align_of::<[lmnk_q7_t; 64]>() == 1`.
    Ok(unsafe { &mut *ptr.cast::<[lmnk_q7_t; MAX_ATTENTION_HEADS]>() })
}

/// Validates a `PhaseSteeringEngine` configuration from `(head_dim, num_heads)` and returns `(engine, total_elements)`.
#[inline(always)]
pub(super) fn init_phase_engine(
    head_dim: usize,
    num_heads: usize,
) -> LmnkResult<(PhaseSteeringEngine, usize)> {
    let hd_u32 = u32::try_from(head_dim).map_err(|_| LMNK_ERR_INVALID_ARGUMENT)?;
    let nh_u32 = u32::try_from(num_heads).map_err(|_| LMNK_ERR_INVALID_ARGUMENT)?;
    let engine = PhaseSteeringEngine::new(hd_u32, nh_u32)?;
    let total_len = head_dim
        .checked_mul(num_heads)
        .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
    Ok((engine, total_len))
}

/// Writes `val` to an optional raw output pointer `out_ptr` after validating natural alignment.
///
/// # Safety
/// If `out_ptr` is non-null, it must be naturally aligned and writable for `T`.
#[inline(always)]
pub(super) unsafe fn write_opt_out<T: Copy>(out_ptr: *mut T, val: T) -> LmnkResult<()> {
    // SAFETY: `FfiHandleMut::from_raw_opt` validates alignment when `out_ptr` is non-null.
    if let Some(mut out_ref) = unsafe { FfiHandleMut::from_raw_opt(out_ptr)? } {
        *out_ref = val;
    }
    Ok(())
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
) -> LmnkResult<(
    PhaseSteeringEngine,
    AlignedSliceMut32<'a, f32>,
    Option<AlignedSliceMut32<'a, f32>>,
)> {
    validate_ptr_aligned(query_tensor, 32)?;
    let (engine, total_len) = init_phase_engine(head_dim, num_heads)?;
    let qk_bytes = validate_raw_slice_bounds(query_tensor, total_len)?;
    if !key_tensor.is_null() {
        validate_ptr_aligned(key_tensor, 32)?;
        let _ = validate_raw_slice_bounds(key_tensor, total_len)?;
        if raw_byte_ranges_overlap(
            query_tensor as usize,
            qk_bytes,
            key_tensor as usize,
            qk_bytes,
        ) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
    }
    // SAFETY: `query_tensor` and optional `key_tensor` are validated aligned, bounded, and
    // mutually disjoint BEFORE creating any `AlignedSliceMut32` references.
    let (q_aligned, k_opt) = unsafe {
        (
            AlignedSliceMut32::from_raw_parts_mut(query_tensor, total_len)?,
            AlignedSliceMut32::from_raw_parts_mut_opt(key_tensor, total_len)?,
        )
    };
    Ok((engine, q_aligned, k_opt))
}

pub(super) type BoundQkAndLogits<'a> = (
    PhaseSteeringEngine,
    AlignedSliceMut32<'a, f32>,
    Option<AlignedSliceMut32<'a, f32>>,
    Option<AlignedSliceMut32<'a, f32>>,
);

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
) -> LmnkResult<BoundQkAndLogits<'a>> {
    let (_engine, total_len) = init_phase_engine(head_dim, num_heads)?;
    let qk_bytes = validate_raw_slice_bounds(query_tensor, total_len)?;
    let has_logits = !logits_inout.is_null() && vocab_size > 0;
    if has_logits {
        validate_ptr_aligned(logits_inout, 32)?;
        let logits_bytes = validate_raw_slice_bounds(logits_inout, vocab_size)?;
        if raw_byte_ranges_overlap(
            query_tensor as usize,
            qk_bytes,
            logits_inout as usize,
            logits_bytes,
        ) || (!key_tensor.is_null()
            && raw_byte_ranges_overlap(
                key_tensor as usize,
                qk_bytes,
                logits_inout as usize,
                logits_bytes,
            ))
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
    }
    // SAFETY: Pairwise disjointness across Q, K, and logits is verified BEFORE any `&mut` slice construction.
    let ((engine, q_aligned, k_opt), logits_opt) = unsafe {
        (
            bind_qk_32(query_tensor, key_tensor, head_dim, num_heads)?,
            if has_logits {
                AlignedSliceMut32::from_raw_parts_mut_opt(logits_inout, vocab_size)?
            } else {
                None
            },
        )
    };
    Ok((engine, q_aligned, k_opt, logits_opt))
}

/// Resolves flexible `(head_dim, num_heads)` dimensions for flat tensor C-ABI entry points.
#[inline(always)]
pub(super) const fn resolve_flat_tensor_heads(dim: usize) -> LmnkResult<(u32, u32)> {
    if dim == 0 || (dim & 1) != 0 || dim > (u32::MAX as usize) {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
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
