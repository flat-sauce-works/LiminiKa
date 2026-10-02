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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
};

extern "C" {
    /// Retrieve the static C-ABI version string.
    pub fn gcso_abi_get_version_string() -> *const c_char;

    /// Initialize GCSO context configuration struct with default parameters.
    pub fn gcso_config_init_default(config: *mut gcso_config_t) -> GcsoStatus;

    /// Create a new GCSO context instance.
    pub fn gcso_context_create(
        config: *const gcso_config_t,
        context_out: *mut GcsoContextHandle,
    ) -> GcsoStatus;

    /// Process a single token step in the zero-allocation hot path.
    pub fn gcso_context_step_token(
        context: GcsoContextHandle,
        token_id: u32,
        query_tensor: *mut f32,
        key_tensor: *mut f32,
        trail_out: *mut gcso_pointer_trail_t,
    ) -> GcsoStatus;

    /// Destroy and free the GCSO context instance.
    pub fn gcso_context_destroy(context: GcsoContextHandle) -> GcsoStatus;
}

/// Safe exported FFI helper to initialize a `gcso_config_t` structure with standard defaults.
/// Intercepts Unwind panics at the boundary to prevent undefined behavior across C/C++.
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
