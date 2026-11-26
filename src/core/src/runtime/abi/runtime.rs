// name: src/core/src/runtime/abi/runtime.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Version handshake, configuration, `RuntimeContext` lifecycle/policy/decode/state/persistence,
//! and generic `.lmnk` container C-ABI entry points (`v0.4.0`, `0x0000_0400`).

use super::*;
use crate::runtime::storage::{
    LmnkContainerView, LmnkSectionInput, required_container_byte_len,
    write_container_sections_to_bytes,
};

// ===================================================================
// 1. Two-Stage Version Handshake & Configuration (6 functions)
// ===================================================================

/// Returns the packed hexadecimal C-ABI version identifier (`0x0000_0400`).
#[unsafe(no_mangle)]
pub extern "C" fn lmnk_abi_version_hex() -> u32 {
    LMNK_ABI_VERSION_HEX
}

/// Returns a static NUL-terminated C string representing [`LMNK_ABI_VERSION_STRING`].
///
/// # Safety
/// Returned pointer is static and read-only; callers must not free it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_abi_version_string() -> *const c_char {
    LMNK_ABI_VERSION_CSTR.as_ptr()
}

/// Converts a [`LmnkStatus`] code into a static NUL-terminated descriptive C string.
///
/// # Safety
/// Returned pointer is static and read-only; callers must not free it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_status_to_string(status: LmnkStatus) -> *const c_char {
    status.as_c_bytes().as_ptr().cast::<c_char>()
}

/// Populates a versioned [`lmnk_abi_info_t`] handshake structure with ABI version,
/// `RuntimeContext` footprint, snapshot byte budget, topology CRC32, and capability flags.
///
/// Caller must initialize `info_out->struct_size = sizeof(lmnk_abi_info_t)` before calling.
///
/// # Safety
/// `info_out` must be a non-null, naturally aligned writable [`lmnk_abi_info_t`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_abi_info(info_out: *mut lmnk_abi_info_t) -> LmnkStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and alignment.
        let mut out = unsafe { FfiHandleMut::from_raw(info_out)? };
        validate_envelope_size::<lmnk_abi_info_t>(out.struct_size)?;

        let caps = query_capability();
        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        let caps = if std::is_x86_feature_detected!("avx2") {
            caps | LMNK_CAP_SIMD_AVX2
        } else {
            caps
        };

        *out = lmnk_abi_info_t {
            struct_size: size_of::<lmnk_abi_info_t>() as u32,
            abi_version_hex: LMNK_ABI_VERSION_HEX,
            context_size: size_of::<RuntimeContext>() as u32,
            context_align: align_of::<RuntimeContext>() as u32,
            snapshot_max_bytes: LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES as u32,
            topology_crc32: CORE_TOPOLOGY.topology_crc32,
            capability_flags: caps,
            reserved: [0; 32],
        };
        Ok(LMNK_OK)
    })
}

/// Initializes a [`lmnk_config_t`] structure with canonical defaults (including `struct_size = 64`).
///
/// # Safety
/// `config` must be a non-null, 64-byte aligned writable [`lmnk_config_t`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_config_init_default(config: *mut lmnk_config_t) -> LmnkStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and alignment.
        let mut cfg = unsafe { FfiHandleMut::from_raw(config)? };
        init_default_config(&mut cfg)?;
        Ok(LMNK_OK)
    })
}

/// Validates a [`lmnk_config_t`] structure (including `struct_size == 64`) against invariants.
///
/// # Safety
/// `config` must be a non-null, 64-byte aligned readable [`lmnk_config_t`] pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_config_validate(config: *const lmnk_config_t) -> LmnkStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleRef::from_raw` validates non-nullness and alignment.
        let cfg = unsafe { FfiHandleRef::from_raw(config)? };
        validate_config_invariants(&cfg)?;
        Ok(LMNK_OK)
    })
}

// ===================================================================
// 2. RuntimeContext Lifecycle, Policy, Decode & Snapshot (11 functions)
// ===================================================================

/// Initializes a `RuntimeContext` in caller-owned memory with zero heap allocation.
///
/// # Safety
/// `config` must be valid and 64-byte aligned. `mem_out` must be non-null, aligned to
/// `info.context_align` (`64`), writable for at least `info.context_size` (`mem_bytes`) bytes,
/// and disjoint from `config`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_context_init_in_place(
    config: *const lmnk_config_t,
    mem_out: *mut u8,
    mem_bytes: usize,
) -> LmnkStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleRef::from_raw` validates non-nullness and 64-byte alignment of `config`.
        let cfg = unsafe { FfiHandleRef::from_raw(config)? };
        validate_config_invariants(&cfg)?;
        let req_size = size_of::<RuntimeContext>();
        let req_align = align_of::<RuntimeContext>();
        validate_ptr_aligned(mem_out, req_align)?;
        if mem_bytes < req_size {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        if raw_byte_ranges_overlap(
            config.addr(),
            size_of::<lmnk_config_t>(),
            mem_out.addr(),
            req_size,
        ) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }

        // SAFETY: `mem_out` is verified non-null, `req_align`-aligned, writable for `req_size`,
        // and disjoint from `config` immediately above.
        unsafe {
            RuntimeContext::write_in_place(
                mem_out.cast::<RuntimeContext>(),
                Config::from_raw(*cfg),
            )?;
        }
        Ok(LMNK_OK)
    })
}

/// Deinitializes a caller-owned `in_place` `RuntimeContext` and scrubs its backing buffer.
///
/// Returns [`LMNK_ERR_INVALID_STATE`] if `context` was allocated via [`lmnk_context_create`]
/// or [`lmnk_context_deserialize`] rather than [`lmnk_context_init_in_place`].
///
/// # Safety
/// `context` must be `NULL` or a handle previously initialized via [`lmnk_context_init_in_place`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_context_deinit_in_place(context: LmnkContextHandle) -> LmnkStatus {
    if context.is_null() {
        return LMNK_OK;
    }
    let typed = context.cast::<RuntimeContext>();
    if !is_aligned(typed) {
        return LMNK_ERR_ALIGNMENT;
    }
    ffi_boundary!({
        // SAFETY: `typed` is verified non-null and aligned to `align_of::<RuntimeContext>()`.
        // Reading `allocation_mode` and scrubbing through the same raw pointer avoids invalidating
        // Stacked Borrows tags via an intermediate `&mut RuntimeContext`.
        unsafe {
            if RuntimeContext::read_allocation_mode_raw(typed) != CTX_ALLOC_IN_PLACE {
                return Err(LMNK_ERR_INVALID_STATE);
            }
            RuntimeContext::zeroize_raw(typed);
        }
        Ok(LMNK_OK)
    })
}

/// Allocates and initializes a boxed LiminiKa runtime context from a configuration descriptor.
///
/// # Safety
/// `config` (64B aligned) and `context_out` must be non-null, aligned, and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_context_create(
    config: *const lmnk_config_t,
    context_out: *mut LmnkContextHandle,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr(config)?;
        validate_ptr(context_out)?;
        check_disjoint_rw(
            &[(context_out.addr(), size_of::<LmnkContextHandle>())],
            &[(config.addr(), size_of::<lmnk_config_t>())],
        )?;
        // SAFETY: Validated for non-nullness, alignment, and disjointness above.
        let (cfg, mut out_handle) = unsafe {
            (
                FfiHandleRef::from_raw(config)?,
                FfiHandleMut::from_raw(context_out)?,
            )
        };
        validate_config_invariants(&cfg)?;

        let mut ctx = RuntimeContext::new(Config::from_raw(*cfg))?;
        ctx.set_allocation_mode(CTX_ALLOC_HEAP_BOXED);
        *out_handle = Box::into_raw(Box::new(ctx)).cast::<LmnkContextOpaque>();
        Ok(LMNK_OK)
    })
}

/// Destroys a heap-boxed LiminiKa runtime context created by [`lmnk_context_create`] or
/// [`lmnk_context_deserialize`].
///
/// Returns [`LMNK_ERR_INVALID_STATE`] if `context` was initialized via
/// [`lmnk_context_init_in_place`].
///
/// # Safety
/// `context` must be `NULL` or a heap-boxed handle. `NULL` safely returns [`LMNK_OK`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_context_destroy(context: LmnkContextHandle) -> LmnkStatus {
    if context.is_null() {
        return LMNK_OK;
    }
    let typed = context.cast::<RuntimeContext>();
    if !is_aligned(typed) {
        return LMNK_ERR_ALIGNMENT;
    }
    ffi_boundary!({
        // SAFETY: Verify provenance tag via raw pointer before transferring ownership into
        // `Box<MaybeUninit<RuntimeContext>>` and scrubbing in-place without `&mut RuntimeContext`.
        unsafe {
            if RuntimeContext::read_allocation_mode_raw(typed) != CTX_ALLOC_HEAP_BOXED {
                return Err(LMNK_ERR_INVALID_STATE);
            }
            let mut boxed = Box::from_raw(typed.cast::<core::mem::MaybeUninit<RuntimeContext>>());
            RuntimeContext::zeroize_raw(boxed.as_mut_ptr());
            drop(boxed);
        }
        Ok(LMNK_OK)
    })
}

/// Resets transient state variables and phase accumulators without releasing context memory.
///
/// # Safety
/// `context` must be `NULL` or a valid aligned [`LmnkContextHandle`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_context_reset(context: LmnkContextHandle) -> LmnkStatus {
    if context.is_null() {
        return LMNK_OK;
    }
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and concrete `RuntimeContext` alignment.
        let mut ctx = unsafe { FfiHandleMut::from_raw(context.cast::<RuntimeContext>())? };
        ctx.reset()?;
        Ok(LMNK_OK)
    })
}

/// Configures runtime spine and/or anonymous compute pool policies atomically (`0 alloc`).
///
/// Controlled by `policy->flags` (`LMNK_POLICY_FLAG_SPINE`, `LMNK_POLICY_FLAG_COMPUTE_POOL`).
/// When both flags are set, spine pin floors are applied first and compute-pool masks are clamped
/// against the updated floor, guaranteeing identical commutative behavior regardless of call order.
///
/// # Safety
/// `context` (64B aligned) and `policy` must be non-null, naturally aligned, and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_context_configure_policy(
    context: LmnkContextHandle,
    policy: *const lmnk_runtime_policy_t,
) -> LmnkStatus {
    ffi_boundary!({
        let typed = context.cast::<RuntimeContext>();
        validate_ptr(typed)?;
        validate_ptr(policy)?;
        check_disjoint_rw(
            &[(typed.addr(), size_of::<RuntimeContext>())],
            &[(policy.addr(), size_of::<lmnk_runtime_policy_t>())],
        )?;
        // SAFETY: `typed` and `policy` are verified non-null, aligned, and disjoint above.
        let (mut ctx, pol) = unsafe {
            (
                FfiHandleMut::from_raw(typed)?,
                FfiHandleRef::from_raw(policy)?,
            )
        };
        validate_envelope_size::<lmnk_runtime_policy_t>(pol.struct_size)?;
        if (pol.flags & !LMNK_POLICY_KNOWN_FLAGS) != 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }

        if (pol.flags & LMNK_POLICY_FLAG_SPINE) != 0 {
            ctx.configure_spine(
                pol.baseline_pinned_compute_mask,
                pol.guard_hold_tokens,
                pol.max_observe_stride,
                pol.pre_bypass_enabled != 0,
            );
        }
        if (pol.flags & LMNK_POLICY_FLAG_COMPUTE_POOL) != 0 {
            ctx.configure_compute_pool(
                pol.active_compute_mask,
                pol.repulsion_compute_mask,
                pol.reversible_compute_mask,
                pol.baseline_pressure_q15,
            );
        }
        Ok(LMNK_OK)
    })
}

/// Initializes per-turn mechanical spine policy at prompt / turn boundaries (`0 alloc`).
///
/// # Safety
/// `context` (64B aligned) and `params` must be non-null, naturally aligned, and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_context_begin_turn(
    context: LmnkContextHandle,
    params: *const lmnk_turn_begin_params_t,
) -> LmnkStatus {
    ffi_boundary!({
        let typed = context.cast::<RuntimeContext>();
        validate_ptr(typed)?;
        validate_ptr(params)?;
        check_disjoint_rw(
            &[(typed.addr(), size_of::<RuntimeContext>())],
            &[(params.addr(), size_of::<lmnk_turn_begin_params_t>())],
        )?;
        // SAFETY: `typed` and `params` are verified non-null, aligned, and disjoint above.
        let (mut ctx, p) = unsafe {
            (
                FfiHandleMut::from_raw(typed)?,
                FfiHandleRef::from_raw(params)?,
            )
        };
        validate_envelope_size::<lmnk_turn_begin_params_t>(p.struct_size)?;
        if (p.flags & !LMNK_TURN_KNOWN_FLAGS) != 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let guard_turn = (p.flags & LMNK_TURN_FLAG_GUARD_ACTIVE) != 0;
        let unobserved_locked = (p.flags & LMNK_TURN_FLAG_UNOBSERVED_LOCKED) != 0;
        ctx.begin_turn_spine_policy(guard_turn, unobserved_locked);
        Ok(LMNK_OK)
    })
}

/// Evaluates one 4-stage decode control step on [`RuntimeContext`] (`0 alloc`, `0 fdiv`).
///
/// Caller must initialize `params->struct_size = sizeof(lmnk_decode_step_params_t)` and
/// `report_out->struct_size = sizeof(lmnk_decode_step_report_t)`.
///
/// # Safety
/// All pointers must be non-null, naturally aligned, and pairwise disjoint in memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_context_step_decode(
    context: LmnkContextHandle,
    params: *const lmnk_decode_step_params_t,
    report_out: *mut lmnk_decode_step_report_t,
) -> LmnkStatus {
    ffi_boundary!({
        let typed = context.cast::<RuntimeContext>();
        validate_ptr(typed)?;
        validate_ptr(params)?;
        validate_ptr(report_out)?;
        check_disjoint_rw(
            &[
                (typed.addr(), size_of::<RuntimeContext>()),
                (report_out.addr(), size_of::<lmnk_decode_step_report_t>()),
            ],
            &[(params.addr(), size_of::<lmnk_decode_step_params_t>())],
        )?;
        // SAFETY: All pointers are verified non-null, aligned, and pairwise disjoint above.
        let (mut ctx, p, mut out) = unsafe {
            (
                FfiHandleMut::from_raw(typed)?,
                FfiHandleRef::from_raw(params)?,
                FfiHandleMut::from_raw(report_out)?,
            )
        };
        validate_envelope_size::<lmnk_decode_step_params_t>(p.struct_size)?;
        validate_envelope_size::<lmnk_decode_step_report_t>(out.struct_size)?;

        let plan = ctx.step_decode_control(&p.decode_input, p.external_guard_active != 0);
        *out = lmnk_decode_step_report_t {
            struct_size: size_of::<lmnk_decode_step_report_t>() as u32,
            next_observe_mode: plan.next_observe_mode,
            veto_mask: plan.veto_mask,
            event_flags: plan.event_flags,
            effective_observe_stride: plan.effective_observe_stride,
            dispatch: plan.dispatch,
            reserved: [0; 32],
        };
        Ok(LMNK_OK)
    })
}

/// Copies an abstracted 64-byte [`lmnk_runtime_state_view_t`] snapshot out of `context` (`0 alloc`).
///
/// Caller must initialize `state_out->struct_size = sizeof(lmnk_runtime_state_view_t)`.
///
/// # Safety
/// `context` (64B aligned) and `state_out` must be non-null, aligned, and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_context_get_state(
    context: LmnkContextHandle,
    state_out: *mut lmnk_runtime_state_view_t,
) -> LmnkStatus {
    ffi_boundary!({
        let typed = context.cast_const().cast::<RuntimeContext>();
        validate_ptr(typed)?;
        validate_ptr(state_out)?;
        check_disjoint_rw(
            &[(state_out.addr(), size_of::<lmnk_runtime_state_view_t>())],
            &[(typed.addr(), size_of::<RuntimeContext>())],
        )?;
        // SAFETY: `typed` and `state_out` are verified non-null, aligned, and disjoint above.
        let (ctx, mut out) = unsafe {
            (
                FfiHandleRef::from_raw(typed)?,
                FfiHandleMut::from_raw(state_out)?,
            )
        };
        validate_envelope_size::<lmnk_runtime_state_view_t>(out.struct_size)?;

        let spine = ctx.system_spine();
        let pool = ctx.compute_pool();
        *out = lmnk_runtime_state_view_t {
            struct_size: size_of::<lmnk_runtime_state_view_t>() as u32,
            control_flags: spine.control_flags,
            last_veto_mask: spine.last_veto_mask,
            guard_hold_tokens: spine.guard_hold_tokens,
            max_observe_stride: spine.max_observe_stride,
            baseline_pinned_compute_mask: spine.baseline_pinned_compute_mask,
            pinned_compute_mask: spine.pinned_compute_mask,
            last_executed_mask: spine.last_executed_mask,
            active_compute_mask: pool.active_mask,
            last_observed_entropy_q15: spine.last_observed_entropy_q15,
            baseline_pressure_q15: pool.baseline_pressure_q15,
            whole_token_bypass_active: spine.whole_token_bypass_active,
            unobserved_locked: spine.unobserved_locked,
            reserved: [0; 18],
        };
        Ok(LMNK_OK)
    })
}

/// Serializes active runtime state into a `.lmnk` binary snapshot conforming to
/// [`LMNK_ABI_VERSION_HEX`] with an IEEE 802.3 CRC32 payload checksum.
///
/// Passing `buffer == NULL` writes the required byte length into `*buffer_size` and returns [`LMNK_OK`].
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and disjointness invariants.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_context_serialize(
    context: LmnkContextHandle,
    buffer: *mut u8,
    buffer_size: *mut usize,
) -> LmnkStatus {
    ffi_boundary!({
        let typed = context.cast::<RuntimeContext>();
        validate_ptr(typed)?;
        validate_ptr(buffer_size)?;
        if raw_byte_ranges_overlap(
            typed.addr(),
            size_of::<RuntimeContext>(),
            buffer_size.addr(),
            size_of::<usize>(),
        ) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }

        // SAFETY: `typed` and `buffer_size` are non-null, aligned, and disjoint above.
        let (ctx, mut size_io) = unsafe {
            (
                FfiHandleRef::from_raw(typed)?,
                FfiHandleMut::from_raw(buffer_size)?,
            )
        };
        let mut required = 0usize;
        ctx.serialize_to_bytes(&mut [], &mut required)?;

        if buffer.is_null() {
            *size_io = required;
            return Ok(LMNK_OK);
        }

        if *size_io < required {
            *size_io = required;
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }

        let _ = validate_raw_slice_bounds(buffer, *size_io)?;
        let buf_bytes = validate_raw_slice_bounds(buffer, required)?;
        if raw_byte_ranges_overlap(
            typed.addr(),
            size_of::<RuntimeContext>(),
            buffer.addr(),
            buf_bytes,
        ) || raw_byte_ranges_overlap(
            buffer.addr(),
            buf_bytes,
            buffer_size.addr(),
            size_of::<usize>(),
        ) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }

        let mut written = *size_io;
        // SAFETY: `buffer` is non-null, bounded, and disjoint from `context` and `buffer_size`.
        let out_bytes = unsafe { checked_slice_from_raw_parts_mut(buffer, required)? };
        ctx.serialize_to_bytes(out_bytes, &mut written)?;
        *size_io = written;
        Ok(LMNK_OK)
    })
}

/// Deserializes a `.lmnk` binary snapshot to restore a boxed runtime context with
/// [`LMNK_ABI_VERSION_HEX`] and IEEE 802.3 CRC32 verification.
///
/// # Safety
/// `buffer` and `context_out` must be non-null, valid for `buffer_size` bytes, and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_context_deserialize(
    buffer: *const u8,
    buffer_size: usize,
    context_out: *mut LmnkContextHandle,
) -> LmnkStatus {
    ffi_boundary!({
        if buffer.is_null() {
            return Err(LMNK_ERR_NULL_POINTER);
        }
        validate_ptr(context_out)?;
        if buffer_size < LMNK_CONTAINER_HEADER_BYTES {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        let buf_bytes = validate_raw_slice_bounds(buffer, buffer_size)?;
        if raw_byte_ranges_overlap(
            buffer.addr(),
            buf_bytes,
            context_out.addr(),
            size_of::<LmnkContextHandle>(),
        ) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }

        // SAFETY: `buffer` and `context_out` are validated non-null, bounded, aligned, and disjoint.
        let (in_bytes, mut out_handle) = unsafe {
            (
                checked_slice_from_raw_parts(buffer, buffer_size)?,
                FfiHandleMut::from_raw(context_out)?,
            )
        };
        let mut ctx = RuntimeContext::new(Config::new())?;
        ctx.deserialize_from_bytes(in_bytes)?;
        ctx.set_allocation_mode(CTX_ALLOC_HEAP_BOXED);

        *out_handle = Box::into_raw(Box::new(ctx)).cast::<LmnkContextOpaque>();
        Ok(LMNK_OK)
    })
}

// ===================================================================
// 3. Generic `.lmnk` Container I/O (3 functions)
// ===================================================================

/// Serializes arbitrary non-empty sections into a unified 64B-aligned `.lmnk` container (`0 alloc`).
///
/// Supports two-pass sizing: setting `params->out_bytes == NULL` writes the required byte size
/// into `*params->written_bytes_out` and returns [`LMNK_OK`].
///
/// # Safety
/// All non-null pointers in `params` and its section array must be valid, aligned, and disjoint
/// from `out_bytes` and `written_bytes_out`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_container_write(
    params: *const lmnk_container_write_params_t,
) -> LmnkStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleRef::from_raw` validates non-nullness and alignment of `params`.
        let p = unsafe { FfiHandleRef::from_raw(params)? };
        validate_envelope_size::<lmnk_container_write_params_t>(p.struct_size)?;

        let sec_count = p.section_count as usize;
        if sec_count > usize::from(LMNK_MAX_CONTAINER_SECTIONS) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }

        validate_ptr(p.written_bytes_out)?;
        check_disjoint_rw(
            &[(p.written_bytes_out.addr(), size_of::<usize>())],
            &[(params.addr(), size_of::<lmnk_container_write_params_t>())],
        )?;

        let raw_sections = if sec_count == 0 {
            &[]
        } else {
            validate_ptr(p.sections)?;
            let sec_bytes = validate_raw_slice_bounds(p.sections, sec_count)?;
            check_disjoint_rw(
                &[(p.written_bytes_out.addr(), size_of::<usize>())],
                &[(p.sections.addr(), sec_bytes)],
            )?;
            // SAFETY: `p.sections` is non-null, aligned, bounded, and disjoint from outputs.
            unsafe { checked_slice_from_raw_parts(p.sections, sec_count)? }
        };

        let mut rust_sections = [LmnkSectionInput {
            kind: 0,
            flags: 0,
            payload: &[],
        }; LMNK_MAX_CONTAINER_SECTIONS as usize];

        let mut idx = 0usize;
        while idx < sec_count {
            let entry = &raw_sections[idx];
            validate_envelope_size::<lmnk_section_input_t>(entry.struct_size)?;
            if entry.flags != 0 || entry.payload_len == 0 {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            }
            if entry.payload.is_null() {
                return Err(LMNK_ERR_NULL_POINTER);
            }
            let payload_len = entry.payload_len as usize;
            let payload_bytes = validate_raw_slice_bounds(entry.payload, payload_len)?;
            if raw_byte_ranges_overlap(
                p.written_bytes_out.addr(),
                size_of::<usize>(),
                entry.payload.addr(),
                payload_bytes,
            ) {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            }
            // SAFETY: `entry.payload` is verified non-null, bounded, and disjoint from `written_bytes_out`.
            let payload_slice =
                unsafe { checked_slice_from_raw_parts(entry.payload, payload_len)? };
            rust_sections[idx] = LmnkSectionInput {
                kind: entry.kind,
                flags: entry.flags,
                payload: payload_slice,
            };
            idx += 1;
        }

        let active_sections = &rust_sections[..sec_count];
        let required = required_container_byte_len(active_sections)?;

        // SAFETY: `p.written_bytes_out` is verified non-null, aligned, and disjoint from all inputs.
        let mut written_out = unsafe { FfiHandleMut::from_raw(p.written_bytes_out)? };

        if p.out_bytes.is_null() {
            *written_out = required;
            return Ok(LMNK_OK);
        }
        if p.out_capacity < required {
            *written_out = required;
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }

        let _ = validate_raw_slice_bounds(p.out_bytes, p.out_capacity)?;
        let out_span = validate_raw_slice_bounds(p.out_bytes, required)?;
        check_disjoint_rw(
            &[
                (p.out_bytes.addr(), out_span),
                (p.written_bytes_out.addr(), size_of::<usize>()),
            ],
            &[(params.addr(), size_of::<lmnk_container_write_params_t>())],
        )?;
        if sec_count > 0 {
            let sec_bytes = sec_count * size_of::<lmnk_section_input_t>();
            if raw_byte_ranges_overlap(p.out_bytes.addr(), out_span, p.sections.addr(), sec_bytes) {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            }
            let mut j = 0usize;
            while j < sec_count {
                let entry = &raw_sections[j];
                if raw_byte_ranges_overlap(
                    p.out_bytes.addr(),
                    out_span,
                    entry.payload.addr(),
                    entry.payload_len as usize,
                ) {
                    return Err(LMNK_ERR_INVALID_ARGUMENT);
                }
                j += 1;
            }
        }

        // SAFETY: `p.out_bytes` is non-null, bounded, and verified disjoint from every input and `written_bytes_out`.
        let dst = unsafe { checked_slice_from_raw_parts_mut(p.out_bytes, required)? };
        let written = write_container_sections_to_bytes(
            p.container_flags,
            p.timestamp_epoch_sec,
            active_sections,
            dst,
        )?;
        *written_out = written;
        Ok(LMNK_OK)
    })
}

/// Verifies a `.lmnk` container buffer (magic, ABI version, header CRC32, topology CRC32,
/// section table, per-section CRC32, and record integrity) and populates `report_out` (`0 alloc`).
///
/// Caller must initialize `report_out->struct_size = sizeof(lmnk_container_verify_report_t)`.
///
/// # Safety
/// `bytes` must be valid for `byte_len` bytes, and `report_out` must be non-null, aligned, and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_container_verify(
    bytes: *const u8,
    byte_len: usize,
    report_out: *mut lmnk_container_verify_report_t,
) -> LmnkStatus {
    ffi_boundary!({
        if bytes.is_null() {
            return Err(LMNK_ERR_NULL_POINTER);
        }
        validate_ptr(report_out)?;
        let buf_bytes = validate_raw_slice_bounds(bytes, byte_len)?;
        check_disjoint_rw(
            &[(
                report_out.addr(),
                size_of::<lmnk_container_verify_report_t>(),
            )],
            &[(bytes.addr(), buf_bytes)],
        )?;
        // SAFETY: `bytes` and `report_out` are validated non-null, aligned, bounded, and disjoint.
        let (in_bytes, mut out) = unsafe {
            (
                checked_slice_from_raw_parts(bytes, byte_len)?,
                FfiHandleMut::from_raw(report_out)?,
            )
        };
        validate_envelope_size::<lmnk_container_verify_report_t>(out.struct_size)?;

        let view = LmnkContainerView::parse(in_bytes)?;
        let hdr = view.header();
        *out = lmnk_container_verify_report_t {
            struct_size: size_of::<lmnk_container_verify_report_t>() as u32,
            abi_version: hdr.version,
            topology_crc32: hdr.topology_crc32,
            container_flags: hdr.container_flags,
            total_size: hdr.total_size,
            timestamp_epoch_sec: hdr.timestamp_epoch_sec,
            section_count: hdr.section_count,
            reserved: [0; 6],
        };
        Ok(LMNK_OK)
    })
}

/// Verifies a `.lmnk` container buffer and locates section `kind`, returning a zero-copy
/// [`lmnk_container_section_view_t`] pointing directly into `bytes` (`0 alloc`).
///
/// Returns [`LMNK_ERR_ANCHOR_NOT_FOUND`] when the container is valid but section `kind` is absent.
/// Caller must initialize `view_out->struct_size = sizeof(lmnk_container_section_view_t)`.
///
/// # Safety
/// `bytes` must be valid for `byte_len` bytes, and `view_out` must be non-null, aligned, and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_container_find_section(
    bytes: *const u8,
    byte_len: usize,
    kind: u16,
    view_out: *mut lmnk_container_section_view_t,
) -> LmnkStatus {
    ffi_boundary!({
        if bytes.is_null() {
            return Err(LMNK_ERR_NULL_POINTER);
        }
        validate_ptr(view_out)?;
        let buf_bytes = validate_raw_slice_bounds(bytes, byte_len)?;
        check_disjoint_rw(
            &[(view_out.addr(), size_of::<lmnk_container_section_view_t>())],
            &[(bytes.addr(), buf_bytes)],
        )?;
        // SAFETY: `bytes` and `view_out` are validated non-null, aligned, bounded, and disjoint.
        let (in_bytes, mut out) = unsafe {
            (
                checked_slice_from_raw_parts(bytes, byte_len)?,
                FfiHandleMut::from_raw(view_out)?,
            )
        };
        validate_envelope_size::<lmnk_container_section_view_t>(out.struct_size)?;

        let view = LmnkContainerView::parse(in_bytes)?;
        let entry = view.find_section(kind).ok_or(LMNK_ERR_ANCHOR_NOT_FOUND)?;
        let offset_usize = entry.offset as usize;
        // SAFETY: `LmnkContainerView::parse` verified `entry.offset + entry.length <= byte_len`,
        // so `bytes.add(offset_usize)` stays within the same allocation with Strict Provenance.
        let payload_ptr = unsafe { bytes.add(offset_usize) };

        *out = lmnk_container_section_view_t {
            struct_size: size_of::<lmnk_container_section_view_t>() as u32,
            kind: entry.kind,
            flags: entry.flags,
            offset: entry.offset,
            length: entry.length,
            payload_crc32: entry.payload_crc32,
            reserved0: 0,
            payload_ptr,
        };
        Ok(LMNK_OK)
    })
}
