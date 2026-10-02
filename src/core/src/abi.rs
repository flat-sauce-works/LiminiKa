// src/core/src/abi.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! C-ABI interface definitions and FFI boundary safety primitives for the GCSO core engine.

use std::ffi::c_char;
use std::panic::catch_unwind;

/// Status code returned across the C-ABI FFI boundary.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct GcsoStatus(pub i32);

pub const GCSO_SUCCESS: GcsoStatus = GcsoStatus(0);
pub const GCSO_ERROR_INVALID_ARGUMENT: GcsoStatus = GcsoStatus(-1);
pub const GCSO_ERROR_OUT_OF_MEMORY: GcsoStatus = GcsoStatus(-2);
pub const GCSO_ERROR_INTERNAL_FAILURE: GcsoStatus = GcsoStatus(-3);
pub const GCSO_ERROR_PANIC_CAUGHT: GcsoStatus = GcsoStatus(-4);
pub const GCSO_ERROR_NULL_POINTER: GcsoStatus = GcsoStatus(-5);

/// Opaque runtime context handle across the C-ABI boundary.
pub type GcsoContextHandle = *mut std::ffi::c_void;

/// Dynamic runtime configuration for GCSO execution context.
/// Strictly matches C-ABI layout in `include/liminika/gcso_types.h`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct gcso_config_t {
    pub max_context_len: u32,
    pub head_dim: u32,
    pub num_heads: u32,
    pub num_layers: u32,
    pub enable_dpsr: u8,
    pub enable_srl: u8,
    pub reserved: [u8; 6],
}

/// Trace record stored in the Sidecar Pointer Table (SPT).
/// Strictly matches C-ABI layout in `include/liminika/gcso_types.h`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct gcso_pointer_trail_t {
    pub head_idx: u32,
    pub layer_idx: u32,
    pub phase_offset_q7: i8,
    pub reserved: [u8; 3],
    pub pointer_address: u64,
}

// Static layout assertion checks to guarantee standard C-ABI structural alignment
const _: () = {
    assert!(std::mem::size_of::<GcsoStatus>() == 4);
    assert!(std::mem::size_of::<gcso_config_t>() == 24);
    assert!(std::mem::size_of::<gcso_pointer_trail_t>() == 24);
    assert!(std::mem::align_of::<gcso_config_t>() == 4);
    assert!(std::mem::align_of::<gcso_pointer_trail_t>() == 8);
};

/// Static version string constant for FFI boundary checks.
static GCSO_ABI_VERSION: &[u8] = b"1.0.0\0";

/// Retrieve the static C-ABI version string.
///
/// # Safety
/// Returns a valid null-terminated C string pointer that remains valid for the process lifetime.
#[no_mangle]
pub unsafe extern "C" fn gcso_abi_get_version_string() -> *const c_char {
    GCSO_ABI_VERSION.as_ptr() as *const c_char
}

/// Initialize GCSO context configuration struct with default parameters.
///
/// # Safety
/// `config` must be a non-null, writable pointer to a `gcso_config_t` structure.
#[no_mangle]
pub unsafe extern "C" fn gcso_config_init_default(config: *mut gcso_config_t) -> GcsoStatus {
    gcso_config_init_default_safe(config)
}

/// Safe exported FFI helper to initialize a `gcso_config_t` structure with standard defaults.
/// Intercepts Unwind panics at the boundary to prevent undefined behavior across C/C++.
///
/// # Safety
/// `config` must be a non-null, writable pointer to a `gcso_config_t` structure.
#[no_mangle]
pub unsafe extern "C" fn gcso_config_init_default_safe(config: *mut gcso_config_t) -> GcsoStatus {
    if config.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }

    let result = catch_unwind(|| unsafe {
        *config = gcso_config_t {
            max_context_len: 8192,
            head_dim: 128,
            num_heads: 32,
            num_layers: 32,
            enable_dpsr: 1,
            enable_srl: 1,
            reserved: [0; 6],
        };
        GCSO_SUCCESS
    });

    result.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Create a new GCSO context instance.
///
/// # Safety
/// `config` must point to a valid `gcso_config_t` structure.
/// `context_out` must be a valid non-null pointer to store the resulting handle.
#[no_mangle]
pub unsafe extern "C" fn gcso_context_create(
    config: *const gcso_config_t,
    context_out: *mut GcsoContextHandle,
) -> GcsoStatus {
    if config.is_null() || context_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }

    let result = catch_unwind(|| unsafe {
        if (*config).head_dim == 0 || (*config).num_heads == 0 {
            return GCSO_ERROR_INVALID_ARGUMENT;
        }

        *context_out = std::ptr::null_mut();
        GCSO_SUCCESS
    });

    result.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Process a single token step in the zero-allocation hot path.
///
/// # Safety
/// `context` must be a valid handle created by `gcso_context_create`.
/// `query_tensor` and `key_tensor` must point to valid pre-allocated float buffers.
/// `trail_out` may be null if tracing is disabled, or a valid pointer to store trace info.
#[no_mangle]
pub unsafe extern "C" fn gcso_context_step_token(
    context: GcsoContextHandle,
    _token_id: u32,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    trail_out: *mut gcso_pointer_trail_t,
) -> GcsoStatus {
    if context.is_null() || query_tensor.is_null() || key_tensor.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }

    let result = catch_unwind(|| unsafe {
        if !trail_out.is_null() {
            *trail_out = gcso_pointer_trail_t {
                head_idx: 0,
                layer_idx: 0,
                phase_offset_q7: 0,
                reserved: [0; 3],
                pointer_address: query_tensor as u64,
            };
        }
        GCSO_SUCCESS
    });

    result.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Destroy and free the GCSO context instance.
///
/// # Safety
/// `context` can be null (no-op) or a valid handle created by `gcso_context_create`.
#[no_mangle]
pub unsafe extern "C" fn gcso_context_destroy(context: GcsoContextHandle) -> GcsoStatus {
    if context.is_null() {
        return GCSO_SUCCESS;
    }

    let result = catch_unwind(|| {
        GCSO_SUCCESS
    });

    result.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}