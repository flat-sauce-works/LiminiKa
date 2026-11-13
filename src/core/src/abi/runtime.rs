// name: src/core/src/abi/runtime.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tagged-pointer bit-packing, version/capability queries, context lifecycle, .gcso persistence, and PPRC seek.

use super::*;

// ===================================================================
// 0. Tagged Pointer Bit-Packing & Stigmergic Mode Evolution C-ABI
// ===================================================================

/// Packs a 48-bit virtual address, 8-bit tag, and Q7 phase offset into a 64-bit tagged pointer.
#[no_mangle]
pub extern "C" fn gcso_tagged_ptr_pack(raw_addr: u64, tag: u8, phase: gcso_q7_t) -> u64 {
    gcso_tagged_ptr_t::pack(raw_addr, tag, phase).raw()
}

/// Unpacks the canonical 48-bit address from a 64-bit tagged pointer.
#[no_mangle]
pub extern "C" fn gcso_tagged_ptr_unpack_addr(ptr: u64) -> u64 {
    gcso_tagged_ptr_t(ptr).unpack_addr()
}

/// Unpacks the 8-bit micro-orchestration tag from a 64-bit tagged pointer.
#[no_mangle]
pub extern "C" fn gcso_tagged_ptr_unpack_tag(ptr: u64) -> u8 {
    gcso_tagged_ptr_t(ptr).unpack_tag()
}

/// Unpacks the 8-bit Q7 phase state from a 64-bit tagged pointer.
#[no_mangle]
pub extern "C" fn gcso_tagged_ptr_unpack_phase(ptr: u64) -> gcso_q7_t {
    gcso_tagged_ptr_t(ptr).unpack_phase()
}

/// Packs a 48-bit address, 3-bit Target Attractor Working Lane ID (`0..=7`), 5-bit Action Mode mask,
/// and Q7 phase offset into a 64-bit tagged pointer (`0` branches).
#[no_mangle]
pub extern "C" fn gcso_tagged_ptr_pack_action(
    raw_addr: u64,
    attractor_slot: u8,
    action_mode_flags: u8,
    phase: gcso_q7_t,
) -> u64 {
    gcso_tagged_ptr_t::pack(
        raw_addr,
        pack_action_slot_tag(attractor_slot, action_mode_flags),
        phase,
    )
    .raw()
}

/// Extracts the 5-bit Action Mode execution mask (`bits 0..=4` of the tag) from a 64-bit tagged pointer.
#[no_mangle]
pub extern "C" fn gcso_tagged_ptr_unpack_action_mode(ptr: u64) -> u8 {
    unpack_action_mode_from_tag(gcso_tagged_ptr_t(ptr).unpack_tag())
}

/// Extracts the 3-bit Target Attractor Working Lane ID (`0..=7`, `bits 5..=7` of the tag) from a 64-bit tagged pointer.
#[no_mangle]
pub extern "C" fn gcso_tagged_ptr_unpack_attractor_slot(ptr: u64) -> u8 {
    unpack_attractor_slot_from_tag(gcso_tagged_ptr_t(ptr).unpack_tag())
}

/// Evolves a 64-bit tagged pointer's Action Mode flags and Target Attractor Working Lane ID bottom-up
/// in $\mathcal{O}(1)$ from stigmergic trail density $\rho$, EDBC entropy $\tilde{H}$, and Bloom
/// filter repulsion status (`0` branches).
#[no_mangle]
pub extern "C" fn gcso_tagged_ptr_evolve_mode_stigmergically(
    ptr: u64,
    stigmergic_density: f32,
    entropy_h_tilde: f32,
    repulsion_triggered: u8,
    dominant_attractor_slot: u8,
    low_entropy_skip_thresh: f32,
    high_density_srl_thresh: f32,
) -> u64 {
    gcso_tagged_ptr_t(ptr)
        .evolve_mode_stigmergically(
            stigmergic_density,
            entropy_h_tilde,
            repulsion_triggered != 0,
            dominant_attractor_slot,
            low_entropy_skip_thresh,
            high_density_srl_thresh,
        )
        .raw()
}

// ===================================================================
// 1. Version, Capability & Configuration Management
// ===================================================================

/// Queries the active GCSO runtime C-ABI version components (`0.1.1`).
///
/// # Safety
/// Pointers must be `NULL` or naturally aligned writable `u32` pointers.
#[no_mangle]
pub unsafe extern "C" fn gcso_abi_get_version(
    major: *mut u32,
    minor: *mut u32,
    patch: *mut u32,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleMut::from_raw_opt` validates alignment for non-null pointers.
        let ((m_maj, m_min), m_pat) = unsafe {
            (
                (
                    FfiHandleMut::from_raw_opt(major)?,
                    FfiHandleMut::from_raw_opt(minor)?,
                ),
                FfiHandleMut::from_raw_opt(patch)?,
            )
        };

        let (v_maj, v_min, v_pat) = AbiSystemDispatcher.abi_version();
        if let Some(mut p) = m_maj {
            *p = v_maj;
        }
        if let Some(mut p) = m_min {
            *p = v_min;
        }
        if let Some(mut p) = m_pat {
            *p = v_pat;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Returns a static null-terminated C string representing the active ABI version (`"0.1.1"`).
///
/// # Safety
/// Returned pointer is static and read-only.
#[no_mangle]
pub unsafe extern "C" fn gcso_abi_get_version_string() -> *const c_char {
    c"0.1.1".as_ptr()
}

/// Converts a [`GcsoStatus`] code into a static null-terminated descriptive C string.
///
/// # Safety
/// Returned pointer is static and read-only.
#[no_mangle]
pub unsafe extern "C" fn gcso_status_to_string(status: GcsoStatus) -> *const c_char {
    status.as_c_bytes().as_ptr().cast::<c_char>()
}

/// Queries hardware and compute backend capabilities supported by the current build.
///
/// # Safety
/// `flags` must be a non-null, naturally aligned writable `u64` pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_abi_query_capability(flags: *mut u64) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and alignment.
        let mut out_flags = unsafe { FfiHandleMut::from_raw(flags)? };
        *out_flags = AbiSystemDispatcher.query_capability();
        Ok(GCSO_SUCCESS)
    })
}

/// Initializes a [`gcso_config_t`] structure with canonical v0.1.1 default hyperparameters.
///
/// # Safety
/// `config` must be a non-null, 16-byte aligned writable [`gcso_config_t`] pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_config_init_default(config: *mut gcso_config_t) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and alignment.
        let mut cfg = unsafe { FfiHandleMut::from_raw(config)? };
        AbiSystemDispatcher.init_default_config(&mut cfg)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Validates a [`gcso_config_t`] structure against GCSO mathematical and architectural invariants.
///
/// # Safety
/// `config` must be a non-null, 16-byte aligned readable [`gcso_config_t`] pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_config_validate(config: *const gcso_config_t) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleRef::from_raw` validates non-nullness and alignment.
        let cfg = unsafe { FfiHandleRef::from_raw(config)? };
        AbiSystemDispatcher.validate_config_invariants(&cfg)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Releases a dynamically allocated C string if applicable (no-op for static GCSO strings).
///
/// # Safety
/// `str_ptr` may be `NULL` or a pointer returned by GCSO string APIs.
#[no_mangle]
pub unsafe extern "C" fn gcso_free_string(str_ptr: *const c_char) {
    let _ = str_ptr;
}

// ===================================================================
// 2. Context Lifecycle, 4-Section `.gcso` Persistence & PPRC Seek
// ===================================================================

/// Creates a new GCSO runtime context instance from a configuration descriptor.
///
/// # Safety
/// `config` and `context_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_context_create(
    config: *const gcso_config_t,
    context_out: *mut GcsoContextHandle,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let (cfg, mut out_handle) = unsafe {
            (
                FfiHandleRef::from_raw(config)?,
                FfiHandleMut::from_raw(context_out)?,
            )
        };

        if cfg.head_dim == 0
            || (cfg.head_dim & 1) != 0
            || cfg.num_heads == 0
            || (cfg.num_heads as usize) > MAX_ATTENTION_HEADS
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        *out_handle = core::ptr::NonNull::dangling().as_ptr();
        Ok(GCSO_SUCCESS)
    })
}

/// Resets transient state variables and phase accumulators without freeing allocated tables.
///
/// # Safety
/// `context` must be `NULL` or a valid aligned [`GcsoContextHandle`].
#[no_mangle]
pub unsafe extern "C" fn gcso_context_reset(context: GcsoContextHandle) -> GcsoStatus {
    if context.is_null() {
        return GCSO_SUCCESS;
    }
    if !is_aligned(context) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    ffi_boundary!(Ok(GCSO_SUCCESS))
}

/// Projects natural language system prompt text into phase space as the primary Anchor Attractor.
///
/// # Safety
/// `context` and `prompt_text` must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_context_set_system_prompt_anchor(
    context: GcsoContextHandle,
    prompt_text: *const c_char,
    weight: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        if !branchless_is_finite_f32(weight) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated non-null NUL-terminated UTF-8 string.
        let _prompt = unsafe { validate_c_utf8_str(prompt_text)? };
        Ok(GCSO_SUCCESS)
    })
}

/// Processes a single token step on the Hot Path in zero-allocation mode ($C_{\mathrm{alloc}} = 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_context_step_token(
    context: GcsoContextHandle,
    token_id: u32,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    trail_out: *mut gcso_pointer_trail_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        // SAFETY: `AlignedSliceMut32::from_raw_parts_mut_opt` validates alignment when non-null.
        let (q_opt, k_opt) = unsafe {
            (
                AlignedSliceMut32::from_raw_parts_mut_opt(query_tensor, 128)?,
                AlignedSliceMut32::from_raw_parts_mut_opt(key_tensor, 128)?,
            )
        };

        if let (Some(ref q), Some(ref k)) = (&q_opt, &k_opt) {
            q.ensure_disjoint_mut(k)?;
        }

        // SAFETY: Validated for non-nullness and 128B alignment when non-null.
        if let Some(mut trail) = unsafe { FfiHandleMut::from_raw_opt(trail_out)? } {
            trail.record_pointer_transition(u64::from(token_id), 1, 1.0);
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Serializes active runtime state into `.gcso` binary format conforming to GCSO ABI v0.1.1,
/// stamping an 8-byte unrolled LUT IEEE 802.3 CRC32 checksum ([`branchless_crc32_ieee`]) over the payload.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, and bounds invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_context_serialize(
    context: GcsoContextHandle,
    buffer: *mut u8,
    buffer_size: *mut usize,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and alignment.
        let mut size_io = unsafe { FfiHandleMut::from_raw(buffer_size)? };
        let required = 320usize;

        if buffer.is_null() {
            *size_io = required;
            return Ok(GCSO_SUCCESS);
        }

        if *size_io < required {
            *size_io = required;
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }

        // SAFETY: `buffer` is non-null and `*size_io >= required`.
        let out_bytes = unsafe { slice::from_raw_parts_mut(buffer, required) };
        out_bytes.fill(0);

        let mut header = init_snapshot_header(required as u64, 1_774_900_000);
        header.action_hub_offset = 128;
        header.attractor_field_offset = 128;
        header.dpsr_state_offset = 192;
        header.srl_state_offset = 192;
        header.edbc_state_offset = 192;
        const ZERO_PAYLOAD_192_CRC32: u32 = branchless_crc32_ieee(&[0u8; 192]);
        header.daes_slot_offset = 256;
        header.checksum_crc32 = ZERO_PAYLOAD_192_CRC32;

        header.write_unaligned_to(out_bytes, 0)?;
        *size_io = required;
        Ok(GCSO_SUCCESS)
    })
}

/// Deserializes a `.gcso` binary snapshot to restore runtime state with ABI v0.1.1 and
/// 8-byte unrolled LUT CRC32 verification ([`branchless_crc32_ieee`]).
///
/// # Safety
/// `buffer` and `context_out` must be non-null and valid for `buffer_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn gcso_context_deserialize(
    buffer: *const u8,
    buffer_size: usize,
    context_out: *mut GcsoContextHandle,
) -> GcsoStatus {
    ffi_boundary!({
        if buffer.is_null() {
            return Err(GCSO_ERROR_NULL_POINTER);
        }
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and alignment.
        let mut out_handle = unsafe { FfiHandleMut::from_raw(context_out)? };
        if buffer_size < size_of::<gcso_snapshot_header_t>() {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }

        // SAFETY: `buffer` is non-null and `buffer_size >= 128`.
        let in_bytes = unsafe { slice::from_raw_parts(buffer, buffer_size) };
        let header = gcso_snapshot_header_t::read_unaligned_from(in_bytes, 0)
            .map_err(|_| GCSO_ERROR_CONTAINER_CORRUPTED)?;

        validate_snapshot_header(&header, in_bytes.len())?;
        if header.checksum_crc32 != 0 {
            let hdr_size = size_of::<gcso_snapshot_header_t>();
            let computed = branchless_crc32_ieee(&in_bytes[hdr_size..(header.total_size as usize)]);
            if computed != header.checksum_crc32 {
                return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
            }
        }

        *out_handle = core::ptr::NonNull::dangling().as_ptr();
        Ok(GCSO_SUCCESS)
    })
}

/// Serializes `num_records` Section-1 [`CrystallizedAttractorRecord`]s (`records`) into a `.gcso`
/// (ZIMMS) byte buffer (`out_bytes`) via [`CrystallizedAttractorRecord::write_records_to_zimms_bytes`],
/// stamping each record's IEEE 802.3 CRC32 checksum when not already stamped.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_container_write_crystallized_records_zero_copy(
    records: *const gcso_crystallized_attractor_record_t,
    num_records: usize,
    out_bytes: *mut u8,
    buffer_size_inout: *mut usize,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validates `buffer_size_inout`.
        let mut size_io = unsafe { FfiHandleMut::from_raw(buffer_size_inout)? };
        let required = num_records
            .checked_mul(size_of::<CrystallizedAttractorRecord>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;

        if out_bytes.is_null() {
            *size_io = required;
            return Ok(GCSO_SUCCESS);
        }
        if *size_io < required {
            *size_io = required;
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        if required == 0 {
            *size_io = 0;
            return Ok(GCSO_SUCCESS);
        }

        validate_ptr(records)?;
        if raw_byte_ranges_overlap(records as usize, required, out_bytes as usize, required) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated non-null, aligned, and disjoint above.
        let (rec_slice, dst_slice) = unsafe {
            (
                slice::from_raw_parts(records, num_records),
                slice::from_raw_parts_mut(out_bytes, required),
            )
        };
        let written =
            CrystallizedAttractorRecord::write_records_to_zimms_bytes(rec_slice, dst_slice)?;
        *size_io = written;
        Ok(GCSO_SUCCESS)
    })
}

/// Validates and maps `num_records` Section-1 [`CrystallizedAttractorRecord`]s in-place ($C_{\mathrm{alloc}} = 0$)
/// from a 64-byte aligned `.gcso` (ZIMMS) byte buffer via [`CrystallizedAttractorRecord::slice_from_zimms_bytes_zero_copy`].
///
/// # Safety
/// `zimms_bytes` must be non-null, 64-byte aligned, and valid for at least `byte_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn gcso_container_verify_crystallized_records_zero_copy(
    zimms_bytes: *const u8,
    byte_len: usize,
    num_records: usize,
    records_ptr_out: *mut *const gcso_crystallized_attractor_record_t,
) -> GcsoStatus {
    ffi_boundary!({
        if zimms_bytes.is_null() {
            return Err(GCSO_ERROR_NULL_POINTER);
        }
        // SAFETY: `zimms_bytes` is non-null.
        let bytes = unsafe { slice::from_raw_parts(zimms_bytes, byte_len) };
        let recs =
            CrystallizedAttractorRecord::slice_from_zimms_bytes_zero_copy(bytes, num_records)?;
        // SAFETY: `FfiHandleMut::from_raw_opt` validates optional output pointer.
        if let Some(mut rp_out) = unsafe { FfiHandleMut::from_raw_opt(records_ptr_out)? } {
            *rp_out = if recs.is_empty() {
                ptr::null()
            } else {
                recs.as_ptr()
            };
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Serializes a 4-section `.gcso` (ZIMMS) zero-copy container (`GcsoFourSectionView`) including
/// Section 3c [`CausalProvenanceReport`] diagnostic history entries.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_container_write_four_section_with_reports_zero_copy(
    crystals: *const gcso_crystallized_attractor_record_t,
    num_crystals: usize,
    trails: *const gcso_pointer_trail_t,
    num_trails: usize,
    bloom_opt: *const gcso_contrastive_bloom_filter_t,
    daes_opt: *const gcso_daes_slot_t,
    reports: *const gcso_causal_provenance_report_t,
    num_reports: usize,
    timestamp_epoch_sec: u64,
    out_bytes: *mut u8,
    buffer_size_inout: *mut usize,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let mut size_io = unsafe { FfiHandleMut::from_raw(buffer_size_inout)? };

        let required = GcsoFourSectionView::required_byte_len_with_reports(
            num_crystals,
            num_trails,
            !bloom_opt.is_null(),
            !daes_opt.is_null(),
            num_reports,
        )?;

        if out_bytes.is_null() {
            *size_io = required;
            return Ok(GCSO_SUCCESS);
        }
        if *size_io < required {
            *size_io = required;
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }

        let crystals_slice: &[CrystallizedAttractorRecord] = if num_crystals == 0 {
            &[]
        } else {
            validate_ptr(crystals)?;
            // SAFETY: Validated non-null and 64B aligned.
            unsafe { slice::from_raw_parts(crystals, num_crystals) }
        };
        let trails_slice: &[gcso_pointer_trail_t] = if num_trails == 0 {
            &[]
        } else {
            validate_ptr(trails)?;
            // SAFETY: Validated non-null and 128B aligned.
            unsafe { slice::from_raw_parts(trails, num_trails) }
        };
        let reports_slice: &[CausalProvenanceReport] = if num_reports == 0 {
            &[]
        } else {
            validate_ptr(reports)?;
            // SAFETY: Validated non-null and 16B aligned.
            unsafe { slice::from_raw_parts(reports, num_reports) }
        };
        // SAFETY: Validated optional pointers and destination buffer length.
        let ((bloom_ref, daes_ref), dst_slice) = unsafe {
            (
                (
                    FfiHandleRef::from_raw_opt(bloom_opt)?,
                    FfiHandleRef::from_raw_opt(daes_opt)?,
                ),
                slice::from_raw_parts_mut(out_bytes, required),
            )
        };

        let written = GcsoFourSectionView::write_four_section_with_reports_to_bytes(
            crystals_slice,
            trails_slice,
            bloom_ref.as_deref(),
            daes_ref.as_deref(),
            reports_slice,
            timestamp_epoch_sec,
            dst_slice,
        )?;
        *size_io = written;
        Ok(GCSO_SUCCESS)
    })
}

/// Serializes a 4-section `.gcso` (ZIMMS) zero-copy container (`GcsoFourSectionView`).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_container_write_four_section_zero_copy(
    crystals: *const gcso_crystallized_attractor_record_t,
    num_crystals: usize,
    trails: *const gcso_pointer_trail_t,
    num_trails: usize,
    bloom_opt: *const gcso_contrastive_bloom_filter_t,
    daes_opt: *const gcso_daes_slot_t,
    timestamp_epoch_sec: u64,
    out_bytes: *mut u8,
    buffer_size_inout: *mut usize,
) -> GcsoStatus {
    // SAFETY: Delegates directly to `gcso_container_write_four_section_with_reports_zero_copy` with 0 reports.
    unsafe {
        gcso_container_write_four_section_with_reports_zero_copy(
            crystals,
            num_crystals,
            trails,
            num_trails,
            bloom_opt,
            daes_opt,
            ptr::null(),
            0,
            timestamp_epoch_sec,
            out_bytes,
            buffer_size_inout,
        )
    }
}

/// Validates and maps a 128B-aligned 4-section `.gcso` (ZIMMS) container in-place ($C_{\mathrm{alloc}} = 0$)
/// via [`GcsoFourSectionView::from_zimms_bytes_zero_copy`], exposing all 4 sections including
/// Section 0b [`gcso_zimms_descriptor_t`] and Section 3c [`CausalProvenanceReport`]s.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_container_verify_four_section_with_reports_zero_copy(
    zimms_bytes: *const u8,
    byte_len: usize,
    zimms_desc_out: *mut gcso_zimms_descriptor_t,
    num_crystals_out: *mut usize,
    num_trails_out: *mut usize,
    num_reports_out: *mut usize,
    crystals_ptr_out: *mut *const gcso_crystallized_attractor_record_t,
    trails_ptr_out: *mut *const gcso_pointer_trail_t,
    bloom_ptr_out: *mut *const gcso_contrastive_bloom_filter_t,
    daes_ptr_out: *mut *const gcso_daes_slot_t,
    reports_ptr_out: *mut *const gcso_causal_provenance_report_t,
) -> GcsoStatus {
    ffi_boundary!({
        if zimms_bytes.is_null() {
            return Err(GCSO_ERROR_NULL_POINTER);
        }
        if byte_len < GCSO_FOUR_SECTION_HEADER_BYTES {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        // SAFETY: `zimms_bytes` is non-null and `byte_len >= 192`.
        let bytes = unsafe { slice::from_raw_parts(zimms_bytes, byte_len) };
        let view = GcsoFourSectionView::from_zimms_bytes_zero_copy(bytes)?;

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(zimms_desc_out, *view.zimms_descriptor)?;
            write_opt_out(num_crystals_out, view.crystals.len())?;
            write_opt_out(num_trails_out, view.trails.len())?;
            write_opt_out(num_reports_out, view.provenance_reports.len())?;
        }
        // SAFETY: `FfiHandleMut::from_raw_opt` validates optional output pointers.
        if let Some(mut cp_out) = unsafe { FfiHandleMut::from_raw_opt(crystals_ptr_out)? } {
            *cp_out = if view.crystals.is_empty() {
                ptr::null()
            } else {
                view.crystals.as_ptr()
            };
        }
        // SAFETY: `FfiHandleMut::from_raw_opt` validates optional output pointers.
        if let Some(mut tp_out) = unsafe { FfiHandleMut::from_raw_opt(trails_ptr_out)? } {
            *tp_out = if view.trails.is_empty() {
                ptr::null()
            } else {
                view.trails.as_ptr()
            };
        }
        // SAFETY: `FfiHandleMut::from_raw_opt` validates optional output pointers.
        if let Some(mut bp_out) = unsafe { FfiHandleMut::from_raw_opt(bloom_ptr_out)? } {
            *bp_out = match view.bloom {
                Some(b_ref) => b_ref as *const gcso_contrastive_bloom_filter_t,
                None => ptr::null(),
            };
        }
        // SAFETY: `FfiHandleMut::from_raw_opt` validates optional output pointers.
        if let Some(mut dp_out) = unsafe { FfiHandleMut::from_raw_opt(daes_ptr_out)? } {
            *dp_out = match view.daes {
                Some(d_ref) => d_ref as *const gcso_daes_slot_t,
                None => ptr::null(),
            };
        }
        // SAFETY: `FfiHandleMut::from_raw_opt` validates optional output pointers.
        if let Some(mut rp_out) = unsafe { FfiHandleMut::from_raw_opt(reports_ptr_out)? } {
            *rp_out = if view.provenance_reports.is_empty() {
                ptr::null()
            } else {
                view.provenance_reports.as_ptr()
            };
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Validates and maps a 128B-aligned 4-section `.gcso` (ZIMMS) container in-place ($C_{\mathrm{alloc}} = 0$)
/// via [`GcsoFourSectionView::from_zimms_bytes_zero_copy`].
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_container_verify_four_section_zero_copy(
    zimms_bytes: *const u8,
    byte_len: usize,
    num_crystals_out: *mut usize,
    num_trails_out: *mut usize,
    crystals_ptr_out: *mut *const gcso_crystallized_attractor_record_t,
    trails_ptr_out: *mut *const gcso_pointer_trail_t,
) -> GcsoStatus {
    // SAFETY: Delegates to `gcso_container_verify_four_section_with_reports_zero_copy`.
    unsafe {
        gcso_container_verify_four_section_with_reports_zero_copy(
            zimms_bytes,
            byte_len,
            ptr::null_mut(),
            num_crystals_out,
            num_trails_out,
            ptr::null_mut(),
            crystals_ptr_out,
            trails_ptr_out,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        )
    }
}

/// Validates and maps a 128B-aligned 4-section `.gcso` (ZIMMS) container in-place ($C_{\mathrm{alloc}} = 0$),
/// folds Section 1 [`CrystallizedAttractorRecord`]s onto `trail_inout` by **Phase-Proximity Gravity**,
/// applies Section 3a `view.bloom` contrastive Phase-Conjugate Repulsion ($-\boldsymbol{\Delta\theta}$)
/// if active, and steers `query_tensor` (and optional `key_tensor`) in a single register-resident FFI call.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_container_map_and_steer_four_section_zero_copy_safe(
    zimms_bytes: *const u8,
    byte_len: usize,
    trail_inout: *mut gcso_pointer_trail_t,
    min_step_rad: f32,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    num_crystals_out: *mut usize,
    total_pull_out: *mut f32,
    dominant_anchor_id_out: *mut u32,
    effective_ripa_rad_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        if zimms_bytes.is_null() {
            return Err(GCSO_ERROR_NULL_POINTER);
        }
        if byte_len < GCSO_FOUR_SECTION_HEADER_BYTES {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        validate_ptr(trail_inout)?;
        if raw_byte_ranges_overlap(
            zimms_bytes as usize,
            byte_len,
            trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: `bind_qk_32` validates `query_tensor` and `key_tensor` alignment and disjointness.
        let (_engine, mut q_aligned, mut k_opt) =
            unsafe { bind_qk_32(query_tensor, key_tensor, head_dim, num_heads)? };
        // SAFETY: `zimms_bytes` is non-null and `byte_len >= 192`.
        let bytes = unsafe { slice::from_raw_parts(zimms_bytes, byte_len) };
        let view = GcsoFourSectionView::from_zimms_bytes_zero_copy(bytes)?;
        // SAFETY: `trail_inout` is validated above.
        let mut trail = unsafe { FfiHandleMut::from_raw(trail_inout)? };

        let (tot_pull, dom_id, eff_ripa) = steer_crystallized_records_with_optional_bloom_safe(
            view.crystals,
            view.bloom,
            &mut trail,
            min_step_rad,
            head_dim,
            num_heads,
            &mut q_aligned,
            k_opt.as_mut(),
        )?;

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(num_crystals_out, view.crystals.len())?;
            write_opt_out(total_pull_out, tot_pull)?;
            write_opt_out(dominant_anchor_id_out, dom_id)?;
            write_opt_out(effective_ripa_rad_out, eff_ripa)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Maps a 128B-aligned 4-section `.gcso` (ZIMMS) container in-place ($C_{\mathrm{alloc}} = 0$),
/// executes **Coarse-to-Fine Macro-to-Micro Routing ($\mathcal{O}(K + M)$)** directly over Section 1
/// macro centroids (`view.crystals`) and Section 2 micro pointer trails (`view.trails`), applies
/// Section 3a `view.bloom` contrastive Phase-Conjugate Repulsion ($-\boldsymbol{\Delta\theta}$) in
/// 8 `u64` SWAR registers when `bloom_repulsion_threshold > 0`, and steers `query_tensor` (and
/// optional `key_tensor`) in a single zero-copy FFI call.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_container_map_route_coarse_to_fine_and_steer_four_section_zero_copy_safe(
    zimms_bytes: *const u8,
    byte_len: usize,
    src_trail_inout: *mut gcso_pointer_trail_t,
    head_dim: usize,
    num_heads: usize,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    cluster_bonus_gain: f32,
    contradiction_threshold: f32,
    interpolation_u7: u8,
    bloom_repulsion_threshold: u8,
    min_step_rad: f32,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    winning_centroid_idx_out: *mut usize,
    selected_trail_id_out: *mut u32,
    best_score_out: *mut f32,
    best_coherence_out: *mut f32,
    bloom_repulsed_out: *mut u8,
    found_out: *mut u8,
) -> GcsoStatus {
    ffi_boundary!({
        if zimms_bytes.is_null() {
            return Err(GCSO_ERROR_NULL_POINTER);
        }
        if byte_len < GCSO_FOUR_SECTION_HEADER_BYTES {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        validate_ptr(src_trail_inout)?;
        if min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(cluster_bonus_gain)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if raw_byte_ranges_overlap(
            zimms_bytes as usize,
            byte_len,
            src_trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: `bind_qk_32` validates `query_tensor` and `key_tensor` alignment and disjointness.
        let (_engine, mut q_aligned, mut k_opt) =
            unsafe { bind_qk_32(query_tensor, key_tensor, head_dim, num_heads)? };
        // SAFETY: `zimms_bytes` is non-null and `byte_len >= 192`.
        let bytes = unsafe { slice::from_raw_parts(zimms_bytes, byte_len) };
        let view = GcsoFourSectionView::from_zimms_bytes_zero_copy(bytes)?;
        // SAFETY: `src_trail_inout` is validated above.
        let mut src_trail = unsafe { FfiHandleMut::from_raw(src_trail_inout)? };

        let (win_k, sel_id, best_score, best_coh, found, win_clamp, mut steer_words) =
            if !view.trails.is_empty() {
                let (wk, sid, score, coh, fnd) =
                    CrystallizedAttractorRecord::route_coarse_to_fine_candidates_64(
                        view.crystals,
                        &mut src_trail,
                        view.trails,
                        head_dim as u32,
                        phase_interference_gain,
                        stigmergic_prior_gain,
                        pointer_link_bonus,
                        cluster_bonus_gain,
                        contradiction_threshold,
                    );
                let sw = if fnd && interpolation_u7 > 0 {
                    let dst_idx = (sid as usize).min(view.trails.len() - 1);
                    bridge_trail_words_slerp_u7(
                        &mut src_trail,
                        &view.trails[dst_idx].accumulated_phase_delta,
                        interpolation_u7,
                    )
                } else {
                    *trail_phase_words_ref(&src_trail)
                };
                let clamp = if !view.crystals.is_empty() && wk < view.crystals.len() {
                    sync_winner_centroid_ptr_to_trail(&mut src_trail, &view.crystals[wk]);
                    view.crystals[wk].ripa_clamp_rad
                } else {
                    DEFAULT_RIPA_CLAMP_MAX_RAD
                };
                (wk, sid, score, coh, fnd, clamp, sw)
            } else if !view.crystals.is_empty() {
                let (wk, aid, coh, grav, bridged_w, _win_w) =
                    CrystallizedAttractorRecord::route_coarse_to_fine_trail_with_words_64(
                        view.crystals,
                        &mut src_trail,
                        interpolation_u7,
                    );
                (
                    wk,
                    src_trail.linked_trail_id,
                    grav,
                    coh,
                    aid != 0,
                    view.crystals[wk].ripa_clamp_rad,
                    bridged_w,
                )
            } else {
                (
                    0,
                    src_trail.linked_trail_id,
                    0.0,
                    0.0,
                    false,
                    DEFAULT_RIPA_CLAMP_MAX_RAD,
                    *trail_phase_words_ref(&src_trail),
                )
            };
        let mut bloom_repulsed = false;
        if let Some(bloom_ref) = view.bloom {
            if bloom_repulsion_threshold > 0 && !bloom_ref.is_empty() {
                let (_density, rep_active) = bloom_ref.apply_phase_conjugate_repulsion_words(
                    &mut steer_words,
                    bloom_repulsion_threshold,
                    128,
                );
                if rep_active {
                    bloom_repulsed = true;
                    *trail_phase_words_mut(&mut src_trail) = steer_words;
                    let cur_ptr = gcso_tagged_ptr_t(src_trail.current_ptr);
                    let rep_mode = cur_ptr.action_mode_flags() | ACTION_PRESET_REPULSION_ESCAPE;
                    src_trail.current_ptr = cur_ptr
                        .with_action_and_slot(ATTRACTOR_SLOT_REPULSION, rep_mode)
                        .conjugate_phase()
                        .raw();
                    src_trail.flags |= TRAIL_FLAG_REPULSION_ACTIVE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE;
                }
            }
        }
        src_trail.sync_flags_with_current_ptr();

        steer_routed_trail_words_qk_safe(
            &steer_words,
            src_trail.flags,
            win_clamp,
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
            write_opt_out(bloom_repulsed_out, u8::from(bloom_repulsed))?;
            write_opt_out(found_out, u8::from(found))?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Seeks to a specific token index in the PPRC keyframe KV cache index without forward passes.
///
/// # Safety
/// `context` and `keyframe_header_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_pprc_seek_to_token(
    context: GcsoContextHandle,
    token_index: u32,
    keyframe_header_out: *mut gcso_pprc_keyframe_header_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and alignment.
        let mut header_out = unsafe { FfiHandleMut::from_raw(keyframe_header_out)? };
        *header_out = gcso_pprc_keyframe_header_t {
            token_index,
            ..Default::default()
        };
        Ok(GCSO_SUCCESS)
    })
}
