// name: src/core/src/runtime/abi/runtime.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tagged-pointer bit-packing, version/capability queries, heap & caller-buffered `in_place`
//! context lifecycle, and unified `.lmnk` zero-copy container FFI.

use super::*;
use crate::runtime::storage::{LMNK_CONTAINER_FLAG_COMPILED_CACHE, LmnkContainerView};

// ===================================================================
// 0. Tagged Pointer Bit-Packing & Mode Evolution C-ABI
// ===================================================================

/// Packs a 48-bit virtual address, 8-bit tag, and Q7 phase offset into a 64-bit tagged pointer.
#[unsafe(no_mangle)]
pub extern "C" fn lmnk_tagged_ptr_pack(raw_addr: u64, tag: u8, phase: lmnk_q7_t) -> u64 {
    lmnk_tagged_ptr_t::pack(raw_addr, tag, phase).raw()
}

/// Unpacks the canonical 48-bit address from a 64-bit tagged pointer.
#[unsafe(no_mangle)]
pub extern "C" fn lmnk_tagged_ptr_unpack_addr(ptr: u64) -> u64 {
    TaggedPtr(ptr).unpack_addr()
}

/// Unpacks the 8-bit micro-orchestration tag from a 64-bit tagged pointer.
#[unsafe(no_mangle)]
pub extern "C" fn lmnk_tagged_ptr_unpack_tag(ptr: u64) -> u8 {
    TaggedPtr(ptr).unpack_tag()
}

/// Unpacks the 8-bit Q7 phase state from a 64-bit tagged pointer.
#[unsafe(no_mangle)]
pub extern "C" fn lmnk_tagged_ptr_unpack_phase(ptr: u64) -> lmnk_q7_t {
    TaggedPtr(ptr).unpack_phase()
}

/// Packs a 48-bit address, 3-bit target working lane ID (`0..=7`), 5-bit Action Mode mask,
/// and Q7 phase offset into a 64-bit tagged pointer (`0` branches).
#[unsafe(no_mangle)]
pub extern "C" fn lmnk_tagged_ptr_pack_action(
    raw_addr: u64,
    attractor_slot: u8,
    action_mode_flags: u8,
    phase: lmnk_q7_t,
) -> u64 {
    lmnk_tagged_ptr_t::pack(
        raw_addr,
        pack_action_slot_tag(attractor_slot, action_mode_flags),
        phase,
    )
    .raw()
}

/// Extracts the 5-bit Action Mode execution mask (`bits 0..=4` of the tag) from a 64-bit tagged pointer.
#[unsafe(no_mangle)]
pub extern "C" fn lmnk_tagged_ptr_unpack_action_mode(ptr: u64) -> u8 {
    unpack_action_mode_from_tag(TaggedPtr(ptr).unpack_tag())
}

/// Extracts the 3-bit target working lane ID (`0..=7`, `bits 5..=7` of the tag) from a 64-bit tagged pointer.
#[unsafe(no_mangle)]
pub extern "C" fn lmnk_tagged_ptr_unpack_attractor_slot(ptr: u64) -> u8 {
    unpack_attractor_slot_from_tag(TaggedPtr(ptr).unpack_tag())
}

/// Evolves a 64-bit tagged pointer's Action Mode flags and target working lane ID bottom-up
/// in $\mathcal{O}(1)$ from stigmergic trail density $\rho$, entropy $\tilde{H}$, and Bloom
/// filter repulsion status (`0` branches).
#[unsafe(no_mangle)]
pub extern "C" fn lmnk_tagged_ptr_evolve_mode(
    ptr: u64,
    stigmergic_density: f32,
    entropy_h_tilde: f32,
    repulsion_triggered: u8,
    dominant_attractor_slot: u8,
    low_entropy_skip_thresh: f32,
    high_density_rank1_thresh: f32,
) -> u64 {
    TaggedPtr(ptr)
        .evolve_mode_stigmergically(
            stigmergic_density,
            entropy_h_tilde,
            repulsion_triggered != 0,
            dominant_attractor_slot,
            low_entropy_skip_thresh,
            high_density_rank1_thresh,
        )
        .raw()
}

// ===================================================================
// 1. Version, Capability & Configuration Management
// ===================================================================

/// Returns the C-ABI major version number ([`LMNK_ABI_VERSION_MAJOR`]).
#[unsafe(no_mangle)]
pub extern "C" fn lmnk_abi_version_major() -> u32 {
    LMNK_ABI_VERSION_MAJOR
}

/// Returns the C-ABI minor version number ([`LMNK_ABI_VERSION_MINOR`]).
#[unsafe(no_mangle)]
pub extern "C" fn lmnk_abi_version_minor() -> u32 {
    LMNK_ABI_VERSION_MINOR
}

/// Returns the C-ABI patch version number ([`LMNK_ABI_VERSION_PATCH`]).
#[unsafe(no_mangle)]
pub extern "C" fn lmnk_abi_version_patch() -> u32 {
    LMNK_ABI_VERSION_PATCH
}

/// Returns the packed hexadecimal C-ABI version identifier (`0x0000_0300`).
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

/// Queries core execution capabilities supported by the current build/runtime (`LMNK_CAP_*`).
///
/// # Safety
/// `flags_out` must be a non-null, naturally aligned writable `u64` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_abi_query_capability(flags_out: *mut u64) -> LmnkStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and alignment.
        let mut out_flags = unsafe { FfiHandleMut::from_raw(flags_out)? };
        let caps = query_capability();
        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        let caps = if std::is_x86_feature_detected!("avx2") {
            caps | LMNK_CAP_SIMD_AVX2
        } else {
            caps
        };
        *out_flags = caps;
        Ok(LMNK_OK)
    })
}

/// Initializes a [`lmnk_config_t`] structure with canonical default hyperparameters.
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

/// Validates a [`lmnk_config_t`] structure against LiminiKa mathematical and architectural invariants.
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
// 2. Context Lifecycle (`in_place` & Heap), Hot Step & `.lmnk` Persistence
// ===================================================================

/// Returns the exact byte size of `RuntimeContext` required for `lmnk_context_init_in_place`.
#[unsafe(no_mangle)]
pub extern "C" fn lmnk_context_size() -> usize {
    size_of::<RuntimeContext>()
}

/// Returns the required byte alignment (`64`) of `RuntimeContext` for `lmnk_context_init_in_place`.
#[unsafe(no_mangle)]
pub extern "C" fn lmnk_context_align() -> usize {
    align_of::<RuntimeContext>()
}

/// Initializes a `RuntimeContext` in caller-owned memory with zero heap allocation.
///
/// # Safety
/// `config` must be valid and 64-byte aligned. `mem_out` must be non-null, aligned to
/// [`lmnk_context_align`], writable for at least [`lmnk_context_size`] (`mem_bytes`) bytes,
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
        let req_size = size_of::<RuntimeContext>();
        let req_align = align_of::<RuntimeContext>();
        validate_ptr_aligned(mem_out, req_align)?;
        if mem_bytes < req_size {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        if raw_byte_ranges_overlap(
            config as usize,
            size_of::<lmnk_config_t>(),
            mem_out as usize,
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
/// Returns [`LMNK_ERR_INVALID_STATE`] if `context` was allocated via `lmnk_context_create`
/// or `lmnk_context_deserialize` rather than `lmnk_context_init_in_place`.
///
/// # Safety
/// `context` must be `NULL` or a handle previously initialized via `lmnk_context_init_in_place`.
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
        // Reading `allocation_mode` and scrubbing through the same raw pointer avoids forming an
        // intermediate `&mut RuntimeContext` reference whose Stacked Borrows tag would be invalidated.
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
/// `config` (64B aligned) and `context_out` must be non-null and naturally aligned.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_context_create(
    config: *const lmnk_config_t,
    context_out: *mut LmnkContextHandle,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr(config)?;
        validate_ptr(context_out)?;
        check_disjoint_rw(
            &[(context_out as usize, size_of::<LmnkContextHandle>())],
            &[(config as usize, size_of::<lmnk_config_t>())],
        )?;
        // SAFETY: Validated for non-nullness, alignment, and disjointness above.
        let (cfg, mut out_handle) = unsafe {
            (
                FfiHandleRef::from_raw(config)?,
                FfiHandleMut::from_raw(context_out)?,
            )
        };

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
/// [`lmnk_context_init_in_place`] instead of heap boxing.
///
/// # Safety
/// `context` must be `NULL` or a handle returned by [`lmnk_context_create`] or
/// [`lmnk_context_deserialize`]. `NULL` safely returns [`LMNK_OK`].
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

/// Processes a single token step on the Hot Path in zero-allocation mode ($C_{\mathrm{alloc}} = 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `LMNK_RESTRICT` invariants.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_context_step_token(
    context: LmnkContextHandle,
    token_id: u32,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    trail_out: *mut lmnk_pointer_trail_t,
) -> LmnkStatus {
    ffi_boundary!({
        let typed = context.cast::<RuntimeContext>();
        validate_ptr(typed)?;
        // SAFETY: `typed` is verified non-null and 64-byte aligned; read scalar `head_dim` before taking `&mut`.
        let tensor_len = unsafe { (*typed).head_dim() as usize };
        if tensor_len == 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let ctx_bytes = size_of::<RuntimeContext>();
        let q_bytes = if query_tensor.is_null() {
            0
        } else {
            validate_ptr_aligned(query_tensor, 32)?;
            validate_raw_slice_bounds(query_tensor, tensor_len)?
        };
        let k_bytes = if key_tensor.is_null() {
            0
        } else {
            validate_ptr_aligned(key_tensor, 32)?;
            validate_raw_slice_bounds(key_tensor, tensor_len)?
        };
        let t_bytes = if trail_out.is_null() {
            0
        } else {
            validate_ptr(trail_out)?;
            size_of::<lmnk_pointer_trail_t>()
        };
        if raw_byte_ranges_overlap(typed as usize, ctx_bytes, query_tensor as usize, q_bytes)
            || raw_byte_ranges_overlap(typed as usize, ctx_bytes, key_tensor as usize, k_bytes)
            || raw_byte_ranges_overlap(typed as usize, ctx_bytes, trail_out as usize, t_bytes)
            || raw_byte_ranges_overlap(query_tensor as usize, q_bytes, key_tensor as usize, k_bytes)
            || raw_byte_ranges_overlap(query_tensor as usize, q_bytes, trail_out as usize, t_bytes)
            || raw_byte_ranges_overlap(key_tensor as usize, k_bytes, trail_out as usize, t_bytes)
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }

        // SAFETY: All non-null pointers are aligned, bounds-checked, and pairwise disjoint above.
        let (mut ctx, mut q_opt, mut k_opt, mut trail_opt) = unsafe {
            (
                FfiHandleMut::from_raw(typed)?,
                AlignedSliceMut32::from_raw_parts_mut_opt(query_tensor, tensor_len)?,
                AlignedSliceMut32::from_raw_parts_mut_opt(key_tensor, tensor_len)?,
                FfiHandleMut::from_raw_opt(trail_out)?,
            )
        };

        ctx.step_token(
            token_id,
            q_opt.as_mut(),
            k_opt.as_mut(),
            trail_opt.as_deref_mut(),
        )?;
        Ok(LMNK_OK)
    })
}

/// Configures runtime mechanical spine policy parameters (`0 alloc`) while monotonically raising
/// the permanent host compute pin floor (`baseline_pinned_compute_mask`).
///
/// Passing `0` for `baseline_pinned_compute_mask` preserves the active host floor while enforcing
/// [`HOST_IMMUTABLE_PINNED_COMPUTE_MASK`]. `max_observe_stride == 0` is normalized to `1`.
///
/// # Safety
/// `context` must be a valid, non-null, 64-byte-aligned [`RuntimeContext`] handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_context_configure_spine(
    context: LmnkContextHandle,
    baseline_pinned_compute_mask: u64,
    guard_hold_tokens: u8,
    max_observe_stride: u8,
    pre_bypass_enabled: u8,
) -> LmnkStatus {
    ffi_boundary!({
        let typed = context.cast::<RuntimeContext>();
        validate_ptr(typed)?;
        // SAFETY: `typed` is verified non-null, 64B-aligned, and exclusively mutably accessed.
        let mut ctx = unsafe { FfiHandleMut::from_raw(typed)? };
        ctx.configure_spine(
            baseline_pinned_compute_mask,
            guard_hold_tokens,
            max_observe_stride,
            pre_bypass_enabled != 0,
        );
        Ok(LMNK_OK)
    })
}

/// Configures the embedded `AnonymousComputePool` slot bitmasks (`active`, `repulsion`,
/// `reversible`) and optional Gauge-Zero baseline pressure (`0 alloc`).
///
/// - Bits outside `active_mask` and host-pinned bits (`pinned_compute_mask |
///   baseline_pinned_compute_mask | HOST_IMMUTABLE_PINNED_COMPUTE_MASK`) are clamped out of
///   `repulsion_mask` and `reversible_mask` without stripping host pin floors.
/// - `baseline_pressure_q15 == 0` preserves the active baseline pressure (default `8192`);
///   non-zero values re-anchor via `reanchor_baseline_pressure` (clamped to `32767`).
/// - Commutes with [`lmnk_context_configure_spine`] in either call order.
///
/// # Safety
/// `context` must be a valid, non-null, 64-byte-aligned [`RuntimeContext`] handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_context_configure_compute_pool(
    context: LmnkContextHandle,
    active_mask: u64,
    repulsion_mask: u64,
    reversible_mask: u64,
    baseline_pressure_q15: u16,
) -> LmnkStatus {
    ffi_boundary!({
        let typed = context.cast::<RuntimeContext>();
        validate_ptr(typed)?;
        // SAFETY: `typed` is verified non-null, 64B-aligned, and exclusively mutably accessed.
        let mut ctx = unsafe { FfiHandleMut::from_raw(typed)? };
        ctx.configure_compute_pool(
            active_mask,
            repulsion_mask,
            reversible_mask,
            baseline_pressure_q15,
        );
        Ok(LMNK_OK)
    })
}

/// Copies a read-only 64-byte [`lmnk_spine_state_t`] snapshot out of `context` (`0 alloc`).
///
/// # Safety
/// `context` (`64 B` aligned) and `state_out` (`64 B` aligned) must be non-null and non-overlapping.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_context_get_spine_state(
    context: LmnkContextHandle,
    state_out: *mut lmnk_spine_state_t,
) -> LmnkStatus {
    ffi_boundary!({
        let typed = context.cast_const().cast::<RuntimeContext>();
        validate_ptr(typed)?;
        validate_ptr(state_out)?;
        check_disjoint_rw(
            &[(state_out as usize, size_of::<lmnk_spine_state_t>())],
            &[(typed as usize, size_of::<RuntimeContext>())],
        )?;
        // SAFETY: `typed` and `state_out` are verified non-null, 64B-aligned, and disjoint above.
        let (ctx, mut out_ref) = unsafe {
            (
                FfiHandleRef::from_raw(typed)?,
                FfiHandleMut::from_raw(state_out)?,
            )
        };
        *out_ref = *ctx.system_spine();
        Ok(LMNK_OK)
    })
}

/// Initializes per-turn mechanical spine policy at prompt / turn boundaries (`0 alloc`).
///
/// # Safety
/// `context` must be a valid, non-null, 64-byte-aligned [`RuntimeContext`] handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_context_begin_turn_spine_policy(
    context: LmnkContextHandle,
    guard_turn: u8,
    unobserved_locked: u8,
) -> LmnkStatus {
    ffi_boundary!({
        let typed = context.cast::<RuntimeContext>();
        validate_ptr(typed)?;
        // SAFETY: `typed` is verified non-null, 64B-aligned, and exclusively mutably accessed.
        let mut ctx = unsafe { FfiHandleMut::from_raw(typed)? };
        ctx.begin_turn_spine_policy(guard_turn != 0, unobserved_locked != 0);
        Ok(LMNK_OK)
    })
}

/// Evaluates one 4-stage decode control step on [`RuntimeContext`] (`0 alloc`, `0 fdiv`), reading
/// `input` (`32 B`, `16 B` aligned) and writing `plan_out` (`64 B`, `64 B` aligned).
///
/// # Safety
/// All pointers must be non-null, naturally aligned, and pairwise disjoint in memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_context_step_decode_control(
    context: LmnkContextHandle,
    input: *const lmnk_spine_decode_input_t,
    external_guard_active: u8,
    plan_out: *mut lmnk_spine_step_plan_t,
) -> LmnkStatus {
    ffi_boundary!({
        let typed = context.cast::<RuntimeContext>();
        validate_ptr(typed)?;
        validate_ptr(input)?;
        validate_ptr(plan_out)?;
        check_disjoint_rw(
            &[
                (typed as usize, size_of::<RuntimeContext>()),
                (plan_out as usize, size_of::<lmnk_spine_step_plan_t>()),
            ],
            &[(input as usize, size_of::<lmnk_spine_decode_input_t>())],
        )?;
        // SAFETY: All pointers are non-null, aligned, and verified pairwise disjoint above.
        let (mut ctx, sig_ref, mut out_ref) = unsafe {
            (
                FfiHandleMut::from_raw(typed)?,
                FfiHandleRef::from_raw(input)?,
                FfiHandleMut::from_raw(plan_out)?,
            )
        };
        *out_ref = ctx.step_decode_control(sig_ref.as_ref(), external_guard_active != 0);
        Ok(LMNK_OK)
    })
}

/// Serializes active runtime state into a `512`-byte `.lmnk` binary snapshot conforming to
/// [`LMNK_ABI_VERSION_HEX`] with an IEEE 802.3 CRC32 payload checksum.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, and bounds invariants.
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
            typed as usize,
            size_of::<RuntimeContext>(),
            buffer_size as usize,
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
            typed as usize,
            size_of::<RuntimeContext>(),
            buffer as usize,
            buf_bytes,
        ) || raw_byte_ranges_overlap(
            buffer as usize,
            buf_bytes,
            buffer_size as usize,
            size_of::<usize>(),
        ) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        // Drop `size_io` before reborrowing `*mut usize` after `serialize_to_bytes`, or copy scalar first.
        let mut written = *size_io;
        // SAFETY: `buffer` is non-null, `<= isize::MAX`, and disjoint from `context` and `buffer_size`.
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
/// `buffer` and `context_out` must be non-null and valid for `buffer_size` bytes.
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
            buffer as usize,
            buf_bytes,
            context_out as usize,
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

/// Serializes a unified 64B-aligned `.lmnk` zero-copy container ([`LmnkContainerView`]) including
/// [`CausalProvenanceReport`] diagnostic history entries.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `LMNK_RESTRICT` invariants.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_container_write_with_reports_zero_copy(
    container_flags: u32,
    anchors: *const lmnk_phase_anchor_record_t,
    num_anchors: usize,
    trails: *const lmnk_pointer_trail_t,
    num_trails: usize,
    bloom_opt: *const lmnk_bloom_filter_t,
    ledger_opt: *const lmnk_ledger_slot_t,
    reports: *const lmnk_causal_report_t,
    num_reports: usize,
    timestamp_epoch_sec: u64,
    out_bytes: *mut u8,
    buffer_size_inout: *mut usize,
) -> LmnkStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let mut size_io = unsafe { FfiHandleMut::from_raw(buffer_size_inout)? };

        let required = LmnkContainerView::required_standard_byte_len(
            num_anchors,
            num_trails,
            !bloom_opt.is_null(),
            !ledger_opt.is_null(),
            num_reports,
        )?;

        if out_bytes.is_null() {
            *size_io = required;
            return Ok(LMNK_OK);
        }
        if *size_io < required {
            *size_io = required;
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }

        let a_bytes = if num_anchors == 0 {
            0
        } else {
            validate_ptr(anchors)?;
            validate_raw_slice_bounds(anchors, num_anchors)?
        };
        let t_bytes = if num_trails == 0 {
            0
        } else {
            validate_ptr(trails)?;
            validate_raw_slice_bounds(trails, num_trails)?
        };
        let r_bytes = if num_reports == 0 {
            0
        } else {
            validate_ptr(reports)?;
            validate_raw_slice_bounds(reports, num_reports)?
        };
        let b_bytes = if bloom_opt.is_null() {
            0
        } else {
            validate_ptr(bloom_opt)?;
            size_of::<lmnk_bloom_filter_t>()
        };
        let l_bytes = if ledger_opt.is_null() {
            0
        } else {
            validate_ptr(ledger_opt)?;
            size_of::<lmnk_ledger_slot_t>()
        };
        let out_b = validate_raw_slice_bounds(out_bytes, required)?;
        if raw_byte_ranges_overlap(out_bytes as usize, out_b, anchors as usize, a_bytes)
            || raw_byte_ranges_overlap(out_bytes as usize, out_b, trails as usize, t_bytes)
            || raw_byte_ranges_overlap(out_bytes as usize, out_b, reports as usize, r_bytes)
            || raw_byte_ranges_overlap(out_bytes as usize, out_b, bloom_opt as usize, b_bytes)
            || raw_byte_ranges_overlap(out_bytes as usize, out_b, ledger_opt as usize, l_bytes)
            || raw_byte_ranges_overlap(
                out_bytes as usize,
                out_b,
                buffer_size_inout as usize,
                size_of::<usize>(),
            )
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }

        // SAFETY: All non-empty inputs and destination buffer are aligned, bounded, and disjoint.
        let ((anchors_slice, trails_slice, reports_slice), ((bloom_ref, ledger_ref), dst_slice)) = unsafe {
            (
                (
                    checked_slice_from_raw_parts(anchors, num_anchors)?,
                    checked_slice_from_raw_parts(trails, num_trails)?,
                    checked_slice_from_raw_parts(reports, num_reports)?,
                ),
                (
                    (
                        FfiHandleRef::from_raw_opt(bloom_opt)?,
                        FfiHandleRef::from_raw_opt(ledger_opt)?,
                    ),
                    checked_slice_from_raw_parts_mut(out_bytes, required)?,
                ),
            )
        };

        let written = LmnkContainerView::write_standard_sections_to_bytes(
            container_flags,
            anchors_slice,
            trails_slice,
            bloom_ref.as_deref(),
            ledger_ref.as_deref(),
            reports_slice,
            timestamp_epoch_sec,
            dst_slice,
        )?;
        *size_io = written;
        Ok(LMNK_OK)
    })
}

/// Serializes a compiled-cache `.lmnk` zero-copy container ([`LmnkContainerView`]).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `LMNK_RESTRICT` invariants.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_container_write_zero_copy(
    anchors: *const lmnk_phase_anchor_record_t,
    num_anchors: usize,
    trails: *const lmnk_pointer_trail_t,
    num_trails: usize,
    bloom_opt: *const lmnk_bloom_filter_t,
    ledger_opt: *const lmnk_ledger_slot_t,
    timestamp_epoch_sec: u64,
    out_bytes: *mut u8,
    buffer_size_inout: *mut usize,
) -> LmnkStatus {
    // SAFETY: Delegates directly to `lmnk_container_write_with_reports_zero_copy` with compiled-cache flag and 0 reports.
    unsafe {
        lmnk_container_write_with_reports_zero_copy(
            LMNK_CONTAINER_FLAG_COMPILED_CACHE,
            anchors,
            num_anchors,
            trails,
            num_trails,
            bloom_opt,
            ledger_opt,
            ptr::null(),
            0,
            timestamp_epoch_sec,
            out_bytes,
            buffer_size_inout,
        )
    }
}

/// Validates and maps a 64B-aligned unified `.lmnk` container in-place ($C_{\mathrm{alloc}} = 0$)
/// via [`LmnkContainerView::parse`], exposing typed zero-copy section pointers.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `LMNK_RESTRICT` invariants.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_container_verify_with_reports_zero_copy(
    container_bytes: *const u8,
    byte_len: usize,
    num_anchors_out: *mut usize,
    num_trails_out: *mut usize,
    num_reports_out: *mut usize,
    anchors_ptr_out: *mut *const lmnk_phase_anchor_record_t,
    trails_ptr_out: *mut *const lmnk_pointer_trail_t,
    bloom_ptr_out: *mut *const lmnk_bloom_filter_t,
    ledger_ptr_out: *mut *const lmnk_ledger_slot_t,
    reports_ptr_out: *mut *const lmnk_causal_report_t,
) -> LmnkStatus {
    ffi_boundary!({
        if container_bytes.is_null() {
            return Err(LMNK_ERR_NULL_POINTER);
        }
        if byte_len < LMNK_CONTAINER_HEADER_BYTES {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        let c_bytes = validate_raw_slice_bounds(container_bytes, byte_len)?;
        // SAFETY: `container_bytes` is non-null, `byte_len >= 64`, and within `isize::MAX`.
        let bytes = unsafe { checked_slice_from_raw_parts(container_bytes, byte_len)? };
        let view = LmnkContainerView::parse(bytes)?;
        if ((container_bytes as usize) & 63) != 0 {
            return Err(LMNK_ERR_ALIGNMENT);
        }
        check_disjoint_rw(
            &[
                opt_span(num_anchors_out)?,
                opt_span(num_trails_out)?,
                opt_span(num_reports_out)?,
                opt_span(anchors_ptr_out)?,
                opt_span(trails_ptr_out)?,
                opt_span(bloom_ptr_out)?,
                opt_span(ledger_ptr_out)?,
                opt_span(reports_ptr_out)?,
            ],
            &[(container_bytes as usize, c_bytes)],
        )?;

        let anchors = view.phase_anchors_zero_copy()?;
        let trails = view.pointer_trails_zero_copy()?;
        let bloom = view.bloom_filter_zero_copy()?;
        let ledger = view.ledger_slot_zero_copy()?;
        let reports = view.causal_provenance_zero_copy()?;

        // SAFETY: Optional output pointers are validated non-null and naturally aligned inside `write_opt_out` / `FfiHandleMut::from_raw_opt`.
        unsafe {
            write_opt_out(num_anchors_out, anchors.len())?;
            write_opt_out(num_trails_out, trails.len())?;
            write_opt_out(num_reports_out, reports.len())?;
            if let Some(mut ap_out) = FfiHandleMut::from_raw_opt(anchors_ptr_out)? {
                *ap_out = if anchors.is_empty() {
                    ptr::null()
                } else {
                    anchors.as_ptr()
                };
            }
            if let Some(mut tp_out) = FfiHandleMut::from_raw_opt(trails_ptr_out)? {
                *tp_out = if trails.is_empty() {
                    ptr::null()
                } else {
                    trails.as_ptr()
                };
            }
            if let Some(mut bp_out) = FfiHandleMut::from_raw_opt(bloom_ptr_out)? {
                *bp_out = match bloom {
                    Some(b_ref) => b_ref as *const lmnk_bloom_filter_t,
                    None => ptr::null(),
                };
            }
            if let Some(mut dp_out) = FfiHandleMut::from_raw_opt(ledger_ptr_out)? {
                *dp_out = match ledger {
                    Some(d_ref) => d_ref as *const lmnk_ledger_slot_t,
                    None => ptr::null(),
                };
            }
            if let Some(mut rp_out) = FfiHandleMut::from_raw_opt(reports_ptr_out)? {
                *rp_out = if reports.is_empty() {
                    ptr::null()
                } else {
                    reports.as_ptr()
                };
            }
        }
        Ok(LMNK_OK)
    })
}

/// Validates and maps a 64B-aligned unified `.lmnk` container in-place ($C_{\mathrm{alloc}} = 0$)
/// via [`LmnkContainerView::parse`].
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `LMNK_RESTRICT` invariants.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_container_verify_zero_copy(
    container_bytes: *const u8,
    byte_len: usize,
    num_anchors_out: *mut usize,
    num_trails_out: *mut usize,
    anchors_ptr_out: *mut *const lmnk_phase_anchor_record_t,
    trails_ptr_out: *mut *const lmnk_pointer_trail_t,
) -> LmnkStatus {
    // SAFETY: Delegates to `lmnk_container_verify_with_reports_zero_copy`.
    unsafe {
        lmnk_container_verify_with_reports_zero_copy(
            container_bytes,
            byte_len,
            num_anchors_out,
            num_trails_out,
            ptr::null_mut(),
            anchors_ptr_out,
            trails_ptr_out,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        )
    }
}

/// Validates and maps a 64B-aligned unified `.lmnk` container in-place ($C_{\mathrm{alloc}} = 0$),
/// folds `PhaseAnchors` [`PhaseAnchorRecord`]s onto `trail_inout` by **Phase-Proximity Gravity**,
/// applies `BloomFilter` contrastive Phase-Conjugate Repulsion ($-\boldsymbol{\Delta\theta}$)
/// if active, and steers `query_tensor` (and optional `key_tensor`) in a single register-resident FFI call.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `LMNK_RESTRICT` invariants.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_container_map_and_steer_zero_copy_safe(
    container_bytes: *const u8,
    byte_len: usize,
    trail_inout: *mut lmnk_pointer_trail_t,
    min_step_rad: f32,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    num_anchors_out: *mut usize,
    total_pull_out: *mut f32,
    dominant_anchor_id_out: *mut u32,
    effective_phase_clamp_rad_out: *mut f32,
) -> LmnkStatus {
    ffi_boundary!({
        if container_bytes.is_null() {
            return Err(LMNK_ERR_NULL_POINTER);
        }
        if byte_len < LMNK_CONTAINER_HEADER_BYTES {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        let c_bytes = validate_raw_slice_bounds(container_bytes, byte_len)?;
        validate_ptr(trail_inout)?;
        let (_, total_len) = init_phase_engine(head_dim, num_heads)?;
        let qk_bytes = validate_raw_slice_bounds(query_tensor, total_len)?;
        let k_bytes = if key_tensor.is_null() { 0 } else { qk_bytes };
        check_disjoint_rw(
            &[
                (trail_inout as usize, size_of::<lmnk_pointer_trail_t>()),
                (query_tensor as usize, qk_bytes),
                (key_tensor as usize, k_bytes),
                opt_span(num_anchors_out)?,
                opt_span(total_pull_out)?,
                opt_span(dominant_anchor_id_out)?,
                opt_span(effective_phase_clamp_rad_out)?,
            ],
            &[(container_bytes as usize, c_bytes)],
        )?;

        // SAFETY: `container_bytes` is non-null and bounds-checked; `bind_qk_32` validates Q/K
        // alignment and disjointness; `trail_inout` is disjoint from Q/K and `container_bytes`.
        let (bytes, (_engine, mut q_aligned, mut k_opt)) = unsafe {
            (
                checked_slice_from_raw_parts(container_bytes, byte_len)?,
                bind_qk_32(query_tensor, key_tensor, head_dim, num_heads)?,
            )
        };
        let view = LmnkContainerView::parse(bytes)?;
        if ((container_bytes as usize) & 63) != 0 {
            return Err(LMNK_ERR_ALIGNMENT);
        }
        let anchors = view.phase_anchors_zero_copy()?;
        let bloom = view.bloom_filter_zero_copy()?;
        // SAFETY: `trail_inout` is validated aligned and disjoint above.
        let mut trail = unsafe { FfiHandleMut::from_raw(trail_inout)? };

        let (tot_pull, dom_id, eff_clamp) = steer_crystallized_records_with_optional_bloom_safe(
            anchors,
            bloom,
            &mut trail,
            min_step_rad,
            AbiTensorShape {
                head_dim,
                num_heads,
            },
            &mut q_aligned,
            k_opt.as_mut(),
        )?;

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(num_anchors_out, anchors.len())?;
            write_opt_out(total_pull_out, tot_pull)?;
            write_opt_out(dominant_anchor_id_out, dom_id)?;
            write_opt_out(effective_phase_clamp_rad_out, eff_clamp)?;
        }
        Ok(LMNK_OK)
    })
}
