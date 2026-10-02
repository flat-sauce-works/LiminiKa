// src/core/src/abi.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::ffi::c_char;

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct GcsoStatus(pub i32);

pub const GCSO_SUCCESS: GcsoStatus = GcsoStatus(0);
pub const GCSO_ERROR_INVALID_ARGUMENT: GcsoStatus = GcsoStatus(-1);
pub const GCSO_ERROR_NULL_POINTER: GcsoStatus = GcsoStatus(-5);
pub const GCSO_ERROR_PANIC_CAUGHT: GcsoStatus = GcsoStatus(-4);

pub type GcsoContextHandle = *mut std::ffi::c_void;

extern "C" {
    pub fn gcso_abi_get_version_string() -> *const c_char;
    pub fn gcso_config_init_default(config: *mut crate::abi::gcso_config_t) -> GcsoStatus;
    pub fn gcso_context_create(
        config: *const crate::abi::gcso_config_t,
        context_out: *mut GcsoContextHandle,
    ) -> GcsoStatus;
    pub fn gcso_context_step_token(
        context: GcsoContextHandle,
        token_id: u32,
        query_tensor: *mut f32,
        key_tensor: *mut f32,
        trail_out: *mut crate::abi::gcso_pointer_trail_t,
    ) -> GcsoStatus;
    pub fn gcso_context_destroy(context: GcsoContextHandle) -> GcsoStatus;
}
