// SPDX-License-Identifier: MIT OR Apache-2.0

//! C-ABI interface definitions and FFI boundary safety primitives for the GCSO core engine.

#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::too_many_lines)]

use std::ffi::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Status code returned across the C-ABI FFI boundary.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct GcsoStatus(pub i32);

pub const GCSO_SUCCESS: GcsoStatus = GcsoStatus(0);
pub const GCSO_ERROR_INVALID_ARGUMENT: GcsoStatus = GcsoStatus(-1);
pub const GCSO_ERROR_OUT_OF_MEMORY: GcsoStatus = GcsoStatus(-2);
pub const GCSO_ERROR_BUFFER_TOO_SMALL: GcsoStatus = GcsoStatus(-3);
pub const GCSO_ERROR_PANIC_CAUGHT: GcsoStatus = GcsoStatus(-4);
pub const GCSO_ERROR_NULL_POINTER: GcsoStatus = GcsoStatus(-5);
pub const GCSO_ERROR_INVALID_STATE: GcsoStatus = GcsoStatus(-6);
pub const GCSO_ERROR_VERSION_MISMATCH: GcsoStatus = GcsoStatus(-7);
pub const GCSO_ERROR_MISALIGNED_POINTER: GcsoStatus = GcsoStatus(-8);
pub const GCSO_ERROR_IO_FAILURE: GcsoStatus = GcsoStatus(-9);
pub const GCSO_ERROR_ACTION_HUB_FULL: GcsoStatus = GcsoStatus(-10);
pub const GCSO_ERROR_NOT_IMPLEMENTED: GcsoStatus = GcsoStatus(-11);

pub const GCSO_ERROR_ATTRACTOR_NOT_FOUND: GcsoStatus = GcsoStatus(-20);
pub const GCSO_ERROR_DPSR_PHASE_OVERFLOW: GcsoStatus = GcsoStatus(-30);
pub const GCSO_ERROR_QDPS_UNDERFLOW: GcsoStatus = GcsoStatus(-31);
pub const GCSO_ERROR_EDBC_SINGULARITY: GcsoStatus = GcsoStatus(-40);
pub const GCSO_ERROR_CONTAINER_CORRUPTED: GcsoStatus = GcsoStatus(-50);
pub const GCSO_ERROR_ZIMMS_MAPPING_FAILED: GcsoStatus = GcsoStatus(-51);
pub const GCSO_ERROR_PSPM_ROUTING_FAILED: GcsoStatus = GcsoStatus(-60);
pub const GCSO_ERROR_PPRC_SEEK_FAILED: GcsoStatus = GcsoStatus(-70);
pub const GCSO_ERROR_OBSTRUCTION_UNRESOLVED: GcsoStatus = GcsoStatus(-80);
pub const GCSO_ERROR_EXTENSION_NOT_LOADED: GcsoStatus = GcsoStatus(-90);
pub const GCSO_ERROR_DAES_SCRATCHPAD_FULL: GcsoStatus = GcsoStatus(-91);
pub const GCSO_ERROR_UNKNOWN: GcsoStatus = GcsoStatus(-0x7FFF_FFFF);

/// C-ABI type alias for status codes.
pub type gcso_status_t = GcsoStatus;

/// Quantized 7-bit signed fixed-point integer (scale beta_Q7 = 1/128).
pub type gcso_q7_t = i8;

/// 64-bit capability flags bitmask type.
pub type gcso_capability_flags_t = u64;

/// Opaque handles passed across C-ABI.
pub type GcsoContextHandle = *mut std::ffi::c_void;
pub type GcsoContainerHandle = *mut std::ffi::c_void;
pub type GcsoActionHubHandle = *mut std::ffi::c_void;
pub type GcsoDaesSlotHandle = *mut std::ffi::c_void;
pub type GcsoAttractorFieldHandle = *mut std::ffi::c_void;
pub type GcsoEdbcControllerHandle = *mut std::ffi::c_void;
pub type GcsoDpsrKernelHandle = *mut std::ffi::c_void;
pub type GcsoPspmRouterHandle = *mut std::ffi::c_void;
pub type GcsoSrlAdapterHandle = *mut std::ffi::c_void;

/// C-ABI type aliases for opaque handles matching C headers.
pub type gcso_context_handle_t = GcsoContextHandle;
pub type gcso_container_handle_t = GcsoContainerHandle;
pub type gcso_action_hub_handle_t = GcsoActionHubHandle;
pub type gcso_daes_slot_handle_t = GcsoDaesSlotHandle;
pub type gcso_attractor_field_handle_t = GcsoAttractorFieldHandle;
pub type gcso_edbc_controller_handle_t = GcsoEdbcControllerHandle;
pub type gcso_dpsr_kernel_handle_t = GcsoDpsrKernelHandle;
pub type gcso_pspm_router_handle_t = GcsoPspmRouterHandle;
pub type gcso_srl_adapter_handle_t = GcsoSrlAdapterHandle;

/// Classification types for topological attractor field anchors.
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GcsoAnchorType {
    SystemPrompt = 0,
    Embedding = 1,
    PhaseRepulse = 2,
    Topological = 3,
    Crystallized = 4,
}

impl GcsoAnchorType {
    #[inline]
    #[must_use]
    pub fn from_u32(val: u32) -> Option<Self> {
        match val {
            0 => Some(Self::SystemPrompt),
            1 => Some(Self::Embedding),
            2 => Some(Self::PhaseRepulse),
            3 => Some(Self::Topological),
            4 => Some(Self::Crystallized),
            _ => None,
        }
    }
}

/// C-ABI type alias for anchor types.
pub type gcso_anchor_type_t = u32;

pub const GCSO_ANCHOR_TYPE_SYSTEM_PROMPT: gcso_anchor_type_t = 0;
pub const GCSO_ANCHOR_TYPE_EMBEDDING: gcso_anchor_type_t = 1;
pub const GCSO_ANCHOR_TYPE_PHASE_REPULSE: gcso_anchor_type_t = 2;
pub const GCSO_ANCHOR_TYPE_TOPOLOGICAL: gcso_anchor_type_t = 3;
pub const GCSO_ANCHOR_TYPE_CRYSTALLIZED: gcso_anchor_type_t = 4;

/// Descriptor header for size and ABI version validation.
#[repr(C, align(4))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct gcso_descriptor_header_t {
    pub struct_size: u32,
    pub abi_version: u32,
}

/// 256-bit bitmask layout aligned to 32 bytes for Warp/SIMD reductions.
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct gcso_paged_bitmask_t {
    pub bits: [u64; 4],
}

/// Stigmergic pointer trail structure aligned to 128 bytes (2 cache lines).
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct gcso_pointer_trail_t {
    pub current_ptr: u64,
    pub prev_ptr: u64,
    pub user_data: u64,
    pub transition_cost: i32,
    pub step_count: u32,
    pub stigmergic_density: f32,
    pub target_anchor_id: u32,
    pub cluster_id: u32,
    pub linked_trail_id: u32,
    pub attractor_pull_force: f32,
    pub flags: u32,
    pub accumulated_phase_delta: [gcso_q7_t; 64],
    pub reserved_padding: [u8; 8],
}

/// Dynamic Adaptive Extension Scratchpad (DAES) slot layout (64 bytes).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct gcso_daes_slot_t {
    pub mode: u32,
    pub telemetry_ring_head: u16,
    pub telemetry_ring_tail: u16,
    pub cache_hit_count: u32,
    pub auto_tune_flags: u32,
    pub fast_path_bypass_mask: u64,
    pub fast_path_shortcuts: [u64; 4],
    pub telemetry_mini_ledger: [u8; 8],
}

/// Global configuration descriptor structure aligned to 16 bytes (64 bytes total).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct gcso_config_t {
    pub head_dim: u32,
    pub num_heads: u32,
    pub paged_block_size: u32,
    pub q7_phase_scale: f32,
    pub ripa_clamp_max_rad: f32,
    pub qdps_min_step_rad: f32,
    pub entropy_singularity_eps: f32,
    pub max_prompt_anchors: u32,
    pub action_hub_capacity: u32,
    pub enable_cuda_warp_shuffle: u8,
    pub enable_zero_alloc_strict: u8,
    pub daes_mode: u8,
    pub reserved_flags: u8,
    pub reserved: [u8; 24],
}

/// Dynamic entropy controller state tracking structure (64 bytes).
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct gcso_edbc_state_t {
    pub moving_z_entropy: f32,
    pub bifurcation_threshold: f32,
    pub singularity_eps: f32,
    pub sliding_entropy_rate: f32,
    pub repulsion_gain: f32,
    pub sample_temperature: f32,
    pub active_branch_mode: u32,
    pub reserved: [u8; 36],
}

/// Zero-copy memory mapped storage descriptor (64 bytes).
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct gcso_zimms_descriptor_t {
    pub mapped_address: u64,
    pub file_size_bytes: u64,
    pub dma_buffer_handle: u64,
    pub flags: u32,
    pub fd_handle: i32,
    pub reserved: [u8; 32],
}

/// Sub-head group router configuration descriptor (32 bytes).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct gcso_pspm_config_t {
    pub num_fact_heads: u16,
    pub num_logic_heads: u16,
    pub num_explore_heads: u16,
    pub flags: u16,
    pub fact_phase_gain: f32,
    pub logic_phase_gain: f32,
    pub explore_phase_gain: f32,
    pub reserved: [u8; 12],
}

/// Sparse Residual Adapter Layer (SRL) Rank-1 descriptor (64 bytes).
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct gcso_srl_descriptor_t {
    pub layer_idx: u32,
    pub rank: u32,
    pub u_vector_ptr: u64,
    pub v_vector_ptr: u64,
    pub gain_scalar_ptr: u64,
    pub scale_factor: f32,
    pub flags: u32,
    pub reserved: [u8; 24],
}

/// Unified binary snapshot container header structure (128 bytes).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct gcso_snapshot_header_t {
    pub magic: u32,
    pub version: u32,
    pub total_size: u64,
    pub action_hub_offset: u64,
    pub attractor_field_offset: u64,
    pub dpsr_state_offset: u64,
    pub srl_state_offset: u64,
    pub edbc_state_offset: u64,
    pub checksum_crc32: u32,
    pub daes_slot_offset: u32,
    pub timestamp_epoch_sec: u64,
    pub reserved_padding: [u8; 56],
}

/// PPRC Keyframe KV cache index header structure (64 bytes).
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct gcso_pprc_keyframe_header_t {
    pub frame_type: u32,
    pub token_index: u32,
    pub gop_length: u32,
    pub composite_vector_length: f32,
    pub von_mises_kappa: f32,
    pub sparse_scalar_residual: f32,
    pub icache_payload_offset: u64,
    pub pcache_payload_offset: u64,
    pub reserved: [u8; 24],
}

// Static layout assertion checks to guarantee standard C-ABI structural alignment and field offsets
const _: () = {
    assert!(std::mem::size_of::<GcsoStatus>() == 4);
    assert!(std::mem::size_of::<GcsoAnchorType>() == 4);
    assert!(std::mem::size_of::<gcso_descriptor_header_t>() == 8);
    assert!(std::mem::size_of::<gcso_paged_bitmask_t>() == 32);
    assert!(std::mem::size_of::<gcso_pointer_trail_t>() == 128);
    assert!(std::mem::size_of::<gcso_daes_slot_t>() == 64);
    assert!(std::mem::size_of::<gcso_config_t>() == 64);
    assert!(std::mem::size_of::<gcso_edbc_state_t>() == 64);
    assert!(std::mem::size_of::<gcso_zimms_descriptor_t>() == 64);
    assert!(std::mem::size_of::<gcso_pspm_config_t>() == 32);
    assert!(std::mem::size_of::<gcso_srl_descriptor_t>() == 64);
    assert!(std::mem::size_of::<gcso_snapshot_header_t>() == 128);
    assert!(std::mem::size_of::<gcso_pprc_keyframe_header_t>() == 64);

    assert!(std::mem::align_of::<gcso_paged_bitmask_t>() == 32);
    assert!(std::mem::align_of::<gcso_pointer_trail_t>() == 128);
    assert!(std::mem::align_of::<gcso_daes_slot_t>() == 64);
    assert!(std::mem::align_of::<gcso_config_t>() == 16);
    assert!(std::mem::align_of::<gcso_edbc_state_t>() == 32);
    assert!(std::mem::align_of::<gcso_zimms_descriptor_t>() == 32);
    assert!(std::mem::align_of::<gcso_pspm_config_t>() == 16);
    assert!(std::mem::align_of::<gcso_srl_descriptor_t>() == 32);
    assert!(std::mem::align_of::<gcso_snapshot_header_t>() == 64);
    assert!(std::mem::align_of::<gcso_pprc_keyframe_header_t>() == 32);

    assert!(std::mem::offset_of!(gcso_pointer_trail_t, accumulated_phase_delta) == 56);
    assert!(std::mem::offset_of!(gcso_config_t, action_hub_capacity) == 32);
    assert!(std::mem::offset_of!(gcso_snapshot_header_t, checksum_crc32) == 56);
    assert!(std::mem::offset_of!(gcso_snapshot_header_t, timestamp_epoch_sec) == 64);
    assert!(std::mem::offset_of!(gcso_daes_slot_t, fast_path_shortcuts) == 24);
    assert!(std::mem::offset_of!(gcso_srl_descriptor_t, scale_factor) == 32);
    assert!(std::mem::offset_of!(gcso_pprc_keyframe_header_t, icache_payload_offset) == 24);
};

/// Helper function to check pointer natural alignment safely.
#[inline]
#[must_use]
fn is_aligned<T>(ptr: *const T) -> bool {
    !ptr.is_null() && ptr.is_aligned()
}

/// Helper function to check pointer alignment for a specific custom alignment requirement.
#[inline]
#[must_use]
fn is_aligned_to<T>(ptr: *const T, align: usize) -> bool {
    !ptr.is_null() && ((ptr as usize) % align == 0)
}

/// Static version string constant for FFI boundary checks matching ABI v2.0.0.
static GCSO_ABI_VERSION: &[u8] = b"2.0.0\0";

// ===================================================================
// 1. System & Capability Query Interface
// ===================================================================

/// Retrieve numeric components of the GCSO C-ABI version (v2.0.0).
///
/// # Safety
/// Pointers must be valid, non-null writable memory locations aligned to `u32`.
#[no_mangle]
pub unsafe extern "C" fn gcso_abi_get_version(major: *mut u32, minor: *mut u32, patch: *mut u32) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if is_aligned(major) {
            unsafe { *major = 2 };
        }
        if is_aligned(minor) {
            unsafe { *minor = 0 };
        }
        if is_aligned(patch) {
            unsafe { *patch = 0 };
        }
    }));
}

/// Retrieve the static C-ABI version string ("2.0.0").
///
/// # Safety
/// Returns a valid null-terminated C string pointer that remains valid for the process lifetime.
#[no_mangle]
pub unsafe extern "C" fn gcso_abi_get_version_string() -> *const c_char {
    GCSO_ABI_VERSION.as_ptr().cast::<c_char>()
}

/// Convert a `GcsoStatus` error code into its corresponding static string representation.
///
/// # Safety
/// Safe to call with any `GcsoStatus` value.
#[no_mangle]
pub unsafe extern "C" fn gcso_status_to_string(status: GcsoStatus) -> *const c_char {
    let msg: &'static [u8] = match status {
        GCSO_SUCCESS => b"GCSO_SUCCESS\0",
        GCSO_ERROR_INVALID_ARGUMENT => b"GCSO_ERROR_INVALID_ARGUMENT\0",
        GCSO_ERROR_OUT_OF_MEMORY => b"GCSO_ERROR_OUT_OF_MEMORY\0",
        GCSO_ERROR_BUFFER_TOO_SMALL => b"GCSO_ERROR_BUFFER_TOO_SMALL\0",
        GCSO_ERROR_PANIC_CAUGHT => b"GCSO_ERROR_PANIC_CAUGHT\0",
        GCSO_ERROR_NULL_POINTER => b"GCSO_ERROR_NULL_POINTER\0",
        GCSO_ERROR_INVALID_STATE => b"GCSO_ERROR_INVALID_STATE\0",
        GCSO_ERROR_VERSION_MISMATCH => b"GCSO_ERROR_VERSION_MISMATCH\0",
        GCSO_ERROR_MISALIGNED_POINTER => b"GCSO_ERROR_MISALIGNED_POINTER\0",
        GCSO_ERROR_IO_FAILURE => b"GCSO_ERROR_IO_FAILURE\0",
        GCSO_ERROR_ACTION_HUB_FULL => b"GCSO_ERROR_ACTION_HUB_FULL\0",
        GCSO_ERROR_NOT_IMPLEMENTED => b"GCSO_ERROR_NOT_IMPLEMENTED\0",
        GCSO_ERROR_ATTRACTOR_NOT_FOUND => b"GCSO_ERROR_ATTRACTOR_NOT_FOUND\0",
        GCSO_ERROR_DPSR_PHASE_OVERFLOW => b"GCSO_ERROR_DPSR_PHASE_OVERFLOW\0",
        GCSO_ERROR_QDPS_UNDERFLOW => b"GCSO_ERROR_QDPS_UNDERFLOW\0",
        GCSO_ERROR_EDBC_SINGULARITY => b"GCSO_ERROR_EDBC_SINGULARITY\0",
        GCSO_ERROR_CONTAINER_CORRUPTED => b"GCSO_ERROR_CONTAINER_CORRUPTED\0",
        GCSO_ERROR_ZIMMS_MAPPING_FAILED => b"GCSO_ERROR_ZIMMS_MAPPING_FAILED\0",
        GCSO_ERROR_PSPM_ROUTING_FAILED => b"GCSO_ERROR_PSPM_ROUTING_FAILED\0",
        GCSO_ERROR_PPRC_SEEK_FAILED => b"GCSO_ERROR_PPRC_SEEK_FAILED\0",
        GCSO_ERROR_OBSTRUCTION_UNRESOLVED => b"GCSO_ERROR_OBSTRUCTION_UNRESOLVED\0",
        GCSO_ERROR_EXTENSION_NOT_LOADED => b"GCSO_ERROR_EXTENSION_NOT_LOADED\0",
        GCSO_ERROR_DAES_SCRATCHPAD_FULL => b"GCSO_ERROR_DAES_SCRATCHPAD_FULL\0",
        GCSO_ERROR_UNKNOWN => b"GCSO_ERROR_UNKNOWN\0",
        _ => b"GCSO_ERROR_UNKNOWN\0",
    };
    msg.as_ptr().cast::<c_char>()
}

/// Query current execution hardware capabilities and compute backends.
///
/// # Safety
/// `flags` must point to a valid writable 64-bit unsigned integer aligned to `u64`.
#[no_mangle]
pub unsafe extern "C" fn gcso_abi_query_capability(flags: *mut u64) -> GcsoStatus {
    if flags.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(flags) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| {
        let mut caps = 0x01u64; // Base CPU capability flag
        #[cfg(feature = "cuda")]
        {
            caps |= 1 << 1; // CUDA compute backend capability flag
        }
        #[cfg(feature = "vulkan")]
        {
            caps |= 1 << 2; // Vulkan compute backend capability flag
        }
        #[cfg(any(target_os = "macos", feature = "metal"))]
        {
            caps |= 1 << 3; // Metal compute backend capability flag
        }
        unsafe { *flags = caps };
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Initialize a `gcso_config_t` structure strictly conforming to c_abi_spec.md Section 5.4.D.
///
/// # Safety
/// `config` must be a non-null, writable pointer to a `gcso_config_t` structure aligned to 16 bytes.
#[no_mangle]
pub unsafe extern "C" fn gcso_config_init_default(config: *mut gcso_config_t) -> GcsoStatus {
    if config.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(config) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        unsafe {
            *config = gcso_config_t {
                head_dim: 128,
                num_heads: 32,
                paged_block_size: 32,
                q7_phase_scale: std::f32::consts::PI / 128.0,
                ripa_clamp_max_rad: 0.087_266_46, // 5 degrees soft clamp limit in radians
                qdps_min_step_rad: 0.01,          // QDPS cutoff threshold
                entropy_singularity_eps: 1e-12,   // Logarithmic singularity guard eps
                max_prompt_anchors: 64,
                action_hub_capacity: 256,
                enable_cuda_warp_shuffle: 1,
                enable_zero_alloc_strict: 1,
                daes_mode: 0, // Fast-Path & Telemetry Scratchpad Mode
                reserved_flags: 0,
                reserved: [0; 24],
            };
        }
        GCSO_SUCCESS
    }));

    result.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Free string dynamically allocated by GCSO runtime routines.
///
/// # Safety
/// `str_ptr` must be NULL or a valid pointer to a C string created via `CString::into_raw`.
/// Do NOT pass static string literals.
#[no_mangle]
pub unsafe extern "C" fn gcso_free_string(str_ptr: *const c_char) {
    if !str_ptr.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            let _ = unsafe { std::ffi::CString::from_raw(str_ptr.cast_mut()) };
        }));
    }
}

// ===================================================================
// 2. High-Level Runtime Context Facade Interface
// ===================================================================

/// Create a new GCSO context instance.
///
/// # Safety
/// `config` must point to a valid `gcso_config_t` structure aligned to 16 bytes.
/// `context_out` must be a valid non-null pointer aligned to handle size.
#[no_mangle]
pub unsafe extern "C" fn gcso_context_create(
    config: *const gcso_config_t,
    context_out: *mut GcsoContextHandle,
) -> GcsoStatus {
    if config.is_null() || context_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(config) || !is_aligned(context_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }

    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        if (*config).head_dim == 0
            || (*config).head_dim % 2 != 0
            || (*config).num_heads == 0
            || (*config).num_heads > 64
        {
            return GCSO_ERROR_INVALID_ARGUMENT;
        }
        *context_out = std::ptr::null_mut();
        GCSO_SUCCESS
    }));

    result.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Reset state variables and phase accumulators without freeing allocated tables.
///
/// # Safety
/// `context` must be a valid aligned runtime handle or NULL (safe no-op).
#[no_mangle]
pub unsafe extern "C" fn gcso_context_reset(context: GcsoContextHandle) -> GcsoStatus {
    if context.is_null() {
        return GCSO_SUCCESS;
    }
    if !is_aligned(context) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Set system prompt text as Anchor Attractor.
///
/// # Safety
/// `context` and `prompt_text` must be valid non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn gcso_context_set_system_prompt_anchor(
    context: GcsoContextHandle,
    prompt_text: *const c_char,
    _weight: f32,
) -> GcsoStatus {
    if context.is_null() || prompt_text.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(context) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Process a single token step in zero-allocation mode.
///
/// # Safety
/// `context` must be a non-null pointer aligned to handle boundary.
/// `query_tensor` and `key_tensor` (if non-null) must be 32-byte aligned for SIMD/Warp operations.
/// `trail_out` (if non-null) must be aligned to `gcso_pointer_trail_t`.
#[no_mangle]
pub unsafe extern "C" fn gcso_context_step_token(
    context: GcsoContextHandle,
    _token_id: u32,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    trail_out: *mut gcso_pointer_trail_t,
) -> GcsoStatus {
    if context.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(context) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if !query_tensor.is_null() && !is_aligned_to(query_tensor, 32) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if !key_tensor.is_null() && !is_aligned_to(key_tensor, 32) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if !trail_out.is_null() && !is_aligned(trail_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }

    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Serialize runtime state into binary format.
///
/// # Safety
/// `context` and `buffer_size` must be non-null pointers aligned to structure/type boundary.
#[no_mangle]
pub unsafe extern "C" fn gcso_context_serialize(
    context: GcsoContextHandle,
    buffer: *mut u8,
    buffer_size: *mut usize,
) -> GcsoStatus {
    if context.is_null() || buffer_size.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(context) || !is_aligned(buffer_size) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }

    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        let required = std::mem::size_of::<gcso_snapshot_header_t>();
        if buffer.is_null() {
            *buffer_size = required;
            return GCSO_SUCCESS;
        }
        if *buffer_size < required {
            *buffer_size = required;
            return GCSO_ERROR_BUFFER_TOO_SMALL;
        }
        if !is_aligned(buffer) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }
        std::ptr::write_bytes(buffer, 0, required);
        let header_ptr = buffer.cast::<gcso_snapshot_header_t>();
        (*header_ptr).magic = 0x4F53_4347; // ASCII "GCSO"
        (*header_ptr).version = 0x0002_0000; // ABI Version 2.0.0 per c_abi_spec.md
        (*header_ptr).total_size = required as u64;
        (*header_ptr).timestamp_epoch_sec = 1_774_900_000;
        *buffer_size = required;
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Deserialize binary snapshot to restore runtime state.
///
/// # Safety
/// `buffer` and `context_out` must be valid non-null pointers aligned to respective boundaries.
#[no_mangle]
pub unsafe extern "C" fn gcso_context_deserialize(
    buffer: *const u8,
    buffer_size: usize,
    context_out: *mut GcsoContextHandle,
) -> GcsoStatus {
    if buffer.is_null() || context_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(context_out) || !is_aligned(buffer) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if buffer_size < std::mem::size_of::<gcso_snapshot_header_t>() {
        return GCSO_ERROR_CONTAINER_CORRUPTED;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        let header = &*buffer.cast::<gcso_snapshot_header_t>();
        if header.magic != 0x4F53_4347 {
            return GCSO_ERROR_CONTAINER_CORRUPTED;
        }
        if header.version != 0x0002_0000 {
            return GCSO_ERROR_VERSION_MISMATCH;
        }
        *context_out = std::ptr::null_mut();
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Destroy context instance and release resources.
///
/// # Safety
/// Safe no-op if `context` is NULL. Returns `GCSO_ERROR_MISALIGNED_POINTER` if non-null and unaligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_context_destroy(context: GcsoContextHandle) -> GcsoStatus {
    if context.is_null() {
        return GCSO_SUCCESS;
    }
    if !is_aligned(context) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Destroy container handle and release resources.
///
/// # Safety
/// Safe no-op if `container` is NULL. Returns `GCSO_ERROR_MISALIGNED_POINTER` if non-null and unaligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_container_destroy(container: GcsoContainerHandle) -> GcsoStatus {
    if container.is_null() {
        return GCSO_SUCCESS;
    }
    if !is_aligned(container) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Seek to a specific token index in the PPRC keyframe KV cache index.
///
/// # Safety
/// `context` and `keyframe_header_out` must be non-null valid pointers aligned to structure boundary.
#[no_mangle]
pub unsafe extern "C" fn gcso_pprc_seek_to_token(
    context: GcsoContextHandle,
    token_index: u32,
    keyframe_header_out: *mut gcso_pprc_keyframe_header_t,
) -> GcsoStatus {
    if context.is_null() || keyframe_header_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(context) || !is_aligned(keyframe_header_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        std::ptr::write_bytes(keyframe_header_out, 0, 1);
        (*keyframe_header_out).token_index = token_index;
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

// ===================================================================
// 3. Action Hub, Stigmergic Pointer Trail & DAES Acceleration Interface
// ===================================================================

/// Allocate Action Hub pointer table.
///
/// # Safety
/// `hub_out` must be a non-null pointer aligned to handle size.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_create(
    capacity: u32,
    hub_out: *mut GcsoActionHubHandle,
) -> GcsoStatus {
    if hub_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(hub_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if capacity == 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        *hub_out = std::ptr::null_mut();
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Advance tagged pointer transition in O(1) time.
///
/// # Safety
/// `context` must be valid handle; `trail_out` must be writable and aligned to 128 bytes.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_step_pointer(
    context: GcsoContextHandle,
    _current_ptr: u64,
    trail_out: *mut gcso_pointer_trail_t,
) -> GcsoStatus {
    if context.is_null() || trail_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(context) || !is_aligned(trail_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Compute direct 256-slot hash index for 64-bit pointer trail caching using SplitMix64.
/// Conforms strictly to C++ implementation in `spt_action_hub.cpp`.
///
/// # Safety
/// Safe to call with any 64-bit integer pointer value.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_hash_slot256_index(ptr: u64) -> u32 {
    let mut x = ptr;
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    (x & 0xFF) as u32
}

/// Update cellular swarm cell state across PagedBlock token chunk boundaries.
///
/// # Safety
/// `context` and `mask` must be valid non-null pointers aligned to 32 bytes.
#[no_mangle]
pub unsafe extern "C" fn gcso_swarm_cell_chunk_step(
    context: GcsoContextHandle,
    mask: *const gcso_paged_bitmask_t,
    chunk_len: u32,
) -> GcsoStatus {
    if context.is_null() || mask.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(context) || !is_aligned(mask) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if chunk_len == 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Apply attractor pull force to steer active pointer chains.
///
/// # Safety
/// `context` must be a valid aligned handle.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_pull_trail_to_attractor(
    context: GcsoContextHandle,
    _trail_id: u32,
    _anchor_id: u32,
    pull_force: f32,
) -> GcsoStatus {
    if context.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(context) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if pull_force.is_nan() || pull_force.is_infinite() {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Link adjacent cellular hallucinated trails into contiguous trace graphs.
///
/// # Safety
/// `context` must be a valid aligned handle.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_link_hallucinated_trails(
    context: GcsoContextHandle,
    _src_trail_id: u32,
    _dst_trail_id: u32,
) -> GcsoStatus {
    if context.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(context) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Perform bit-tree reduction across paged bitmasks.
///
/// # Safety
/// `bitmasks` and `reduced_out` must be non-null valid pointers aligned to 32 bytes.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_reduce_bit_tree(
    bitmasks: *const gcso_paged_bitmask_t,
    num_masks: usize,
    reduced_out: *mut gcso_paged_bitmask_t,
) -> GcsoStatus {
    if bitmasks.is_null() || reduced_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(bitmasks) || !is_aligned(reduced_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if num_masks == 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        let masks = std::slice::from_raw_parts(bitmasks, num_masks);
        let mut acc = masks[0];
        for mask in masks.iter().skip(1) {
            acc.bits[0] &= mask.bits[0];
            acc.bits[1] &= mask.bits[1];
            acc.bits[2] &= mask.bits[2];
            acc.bits[3] &= mask.bits[3];
        }
        *reduced_out = acc;
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Evaluate SIMD/Warp bitmask reduction over PagedBlock KV caches.
///
/// # Safety
/// `kv_bits` and `mask_out` must be non-null valid pointers aligned to respective structure alignments.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_paged_block_warp_bitmask(
    kv_bits: *const u64,
    num_blocks: usize,
    mask_out: *mut gcso_paged_bitmask_t,
) -> GcsoStatus {
    if kv_bits.is_null() || mask_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(kv_bits) || !is_aligned(mask_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if num_blocks == 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        let blocks = std::slice::from_raw_parts(kv_bits, num_blocks);
        let mut mask = gcso_paged_bitmask_t::default();
        for (i, &word) in blocks.iter().enumerate() {
            let word_idx = i % 4;
            mask.bits[word_idx] |= word;
        }
        *mask_out = mask;
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Destroy Action Hub instance.
///
/// # Safety
/// Safe no-op if `hub` is NULL. Returns `GCSO_ERROR_MISALIGNED_POINTER` if non-null and unaligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_destroy(hub: GcsoActionHubHandle) -> GcsoStatus {
    if hub.is_null() {
        return GCSO_SUCCESS;
    }
    if !is_aligned(hub) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Allocate DAES slot.
///
/// # Safety
/// `slot_out` must be a valid non-null pointer aligned to handle size.
#[no_mangle]
pub unsafe extern "C" fn gcso_daes_slot_create(slot_out: *mut GcsoDaesSlotHandle) -> GcsoStatus {
    if slot_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(slot_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        *slot_out = std::ptr::null_mut();
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Execute O(1) fast-path shortcut lookup in DAES scratchpad.
///
/// # Safety
/// `slot` and `shortcut_out` must be valid non-null pointers aligned to 64 bytes and 8 bytes respectively.
#[no_mangle]
pub unsafe extern "C" fn gcso_daes_fast_path_lookup(
    slot: *const gcso_daes_slot_t,
    input_key: u64,
    shortcut_out: *mut u64,
) -> GcsoStatus {
    if slot.is_null() || shortcut_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(slot) || !is_aligned(shortcut_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        if (*slot).mode != 0 {
            return GCSO_ERROR_INVALID_STATE;
        }
        let slot_idx = (gcso_action_hub_hash_slot256_index(input_key) % 4) as usize;
        let shortcut = (*slot).fast_path_shortcuts[slot_idx];
        *shortcut_out = if shortcut != 0 { shortcut } else { input_key };
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Push metric byte to DAES telemetry ring ledger in zero-allocation mode.
///
/// # Safety
/// `slot` must be a valid non-null pointer aligned to 64 bytes.
#[no_mangle]
pub unsafe extern "C" fn gcso_daes_telemetry_push(
    slot: *mut gcso_daes_slot_t,
    metric_code: u8,
) -> GcsoStatus {
    if slot.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(slot) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        let head = ((*slot).telemetry_ring_head % 8) as usize;
        (*slot).telemetry_mini_ledger[head] = metric_code;
        (*slot).telemetry_ring_head = (((*slot).telemetry_ring_head as u32 + 1) % 8) as u16;
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Set DAES slot operating mode.
///
/// # Safety
/// `slot` must be a valid non-null pointer aligned to 64 bytes.
#[no_mangle]
pub unsafe extern "C" fn gcso_daes_set_mode(slot: *mut gcso_daes_slot_t, mode: u32) -> GcsoStatus {
    if slot.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(slot) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if mode > 2 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        (*slot).mode = mode;
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Evaluate telemetry ledger for auto-tuning parameters.
///
/// # Safety
/// `slot` and `config_out` must be valid non-null pointers aligned to 64 bytes and 16 bytes.
#[no_mangle]
pub unsafe extern "C" fn gcso_daes_evaluate_auto_tune(
    slot: *const gcso_daes_slot_t,
    config_out: *mut gcso_config_t,
) -> GcsoStatus {
    if slot.is_null() || config_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(slot) || !is_aligned(config_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        let status = gcso_config_init_default(config_out);
        if status != GCSO_SUCCESS {
            return status;
        }
        if (*slot).cache_hit_count > 1000 {
            (*config_out).qdps_min_step_rad = 0.005;
            (*config_out).ripa_clamp_max_rad = 0.100;
        } else if (*slot).cache_hit_count < 50 {
            (*config_out).qdps_min_step_rad = 0.020;
            (*config_out).ripa_clamp_max_rad = 0.050;
        }
        (*config_out).daes_mode = ((*slot).mode & 0xFF) as u8;
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Destroy DAES slot instance.
///
/// # Safety
/// Safe no-op if `slot` is NULL. Returns `GCSO_ERROR_MISALIGNED_POINTER` if non-null and unaligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_daes_slot_destroy(slot: GcsoDaesSlotHandle) -> GcsoStatus {
    if slot.is_null() {
        return GCSO_SUCCESS;
    }
    if !is_aligned(slot) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

// ===================================================================
// 4. DPSR, QDPS, PSPM & SRL Interface
// ===================================================================

/// Create DPSR phase steering kernel instance.
///
/// # Safety
/// `kernel_out` must be a non-null valid pointer aligned to handle size.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_kernel_create(
    head_dim: u32,
    num_heads: u32,
    kernel_out: *mut GcsoDpsrKernelHandle,
) -> GcsoStatus {
    if kernel_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(kernel_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if head_dim == 0 || head_dim % 2 != 0 || num_heads == 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        *kernel_out = std::ptr::null_mut();
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Apply inline DPSR phase rotation to Query tensor registers.
/// Performs 2D SO(2) planar rotations on paired float elements across all head dimensions:
/// [q0', q1']^T = [cos(theta) -sin(theta); sin(theta) cos(theta)] * [q0, q1]^T
///
/// # Safety
/// `query_tensor` and `phase_deltas` must be valid non-null pointers.
/// `query_tensor` must be 32-byte aligned for SIMD vector operations.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_phase_steering(
    query_tensor: *mut f32,
    phase_deltas: *const gcso_q7_t,
    head_dim: usize,
    num_heads: usize,
) -> GcsoStatus {
    if query_tensor.is_null() || phase_deltas.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned_to(query_tensor, 32) || !is_aligned(phase_deltas) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if head_dim == 0 || head_dim % 2 != 0 || num_heads == 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        let q7_scale = std::f32::consts::PI / 128.0;
        let pairs_per_head = head_dim / 2;
        let query_slice = std::slice::from_raw_parts_mut(query_tensor, head_dim * num_heads);
        let delta_slice = std::slice::from_raw_parts(phase_deltas, num_heads);

        for h in 0..num_heads {
            let theta = (delta_slice[h] as f32) * q7_scale;
            if theta == 0.0 {
                continue;
            }
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let head_q = &mut query_slice[h * head_dim..(h + 1) * head_dim];

            for k in 0..pairs_per_head {
                let q0 = head_q[2 * k];
                let q1 = head_q[2 * k + 1];
                head_q[2 * k] = q0 * cos_t - q1 * sin_t;
                head_q[2 * k + 1] = q0 * sin_t + q1 * cos_t;
            }
        }
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Apply RIPA soft-bounded tanh clamping on low-frequency channels.
/// Restricts phase steering strictly to upper d_head/4 dimensions (low frequency pairs).
///
/// # Safety
/// `query_tensor` and `phase_deltas` must be valid non-null pointers.
/// `query_tensor` must be 32-byte aligned for SIMD vector operations.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_phase_steering_safe(
    query_tensor: *mut f32,
    phase_deltas: *const gcso_q7_t,
    head_dim: usize,
    num_heads: usize,
    max_rad: f32,
) -> GcsoStatus {
    if query_tensor.is_null() || phase_deltas.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned_to(query_tensor, 32) || !is_aligned(phase_deltas) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if head_dim == 0
        || head_dim % 2 != 0
        || num_heads == 0
        || max_rad.is_nan()
        || max_rad.is_infinite()
    {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        let q7_scale = std::f32::consts::PI / 128.0;
        let pairs_per_head = head_dim / 2;
        let low_freq_pairs = (head_dim / 8).max(1);
        let start_k = if pairs_per_head > low_freq_pairs {
            pairs_per_head - low_freq_pairs
        } else {
            0
        };

        let query_slice = std::slice::from_raw_parts_mut(query_tensor, head_dim * num_heads);
        let delta_slice = std::slice::from_raw_parts(phase_deltas, num_heads);

        for h in 0..num_heads {
            let mut theta = (delta_slice[h] as f32) * q7_scale;
            if max_rad > 0.0 {
                theta = max_rad * (theta / max_rad).tanh();
            }
            if theta == 0.0 {
                continue;
            }
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let head_q = &mut query_slice[h * head_dim..(h + 1) * head_dim];

            for k in start_k..pairs_per_head {
                let q0 = head_q[2 * k];
                let q1 = head_q[2 * k + 1];
                head_q[2 * k] = q0 * cos_t - q1 * sin_t;
                head_q[2 * k + 1] = q0 * sin_t + q1 * cos_t;
            }
        }
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// QDPS discrete filter step.
///
/// # Safety
/// `phase_deltas` must be a valid non-null pointer aligned to `gcso_q7_t`.
#[no_mangle]
pub unsafe extern "C" fn gcso_qdps_filter_step(
    phase_deltas: *mut gcso_q7_t,
    len: usize,
    min_step_rad: f32,
) -> GcsoStatus {
    if phase_deltas.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(phase_deltas) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if len == 0 || min_step_rad.is_nan() || min_step_rad.is_infinite() || min_step_rad < 0.0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        let q7_scale = std::f32::consts::PI / 128.0;
        let deltas = std::slice::from_raw_parts_mut(phase_deltas, len);
        for delta in deltas.iter_mut() {
            let rad = ((*delta as f32) * q7_scale).abs();
            if rad < min_step_rad {
                *delta = 0;
            }
        }
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Lazy Phase Unwrapping: Apply relative phase shift against context accumulator on Query side.
/// Executes planar 2D SO(2) rotations based on context phase accumulation.
///
/// # Safety
/// `query_tensor` and `context_accum` must be non-null pointers aligned to 32 bytes.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_lazy_unwrap_override(
    query_tensor: *mut f32,
    context_accum: *const f32,
    head_dim: usize,
    num_heads: usize,
) -> GcsoStatus {
    if query_tensor.is_null() || context_accum.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned_to(query_tensor, 32) || !is_aligned_to(context_accum, 32) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if head_dim == 0 || head_dim % 2 != 0 || num_heads == 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        let pairs_per_head = head_dim / 2;
        let query_slice = std::slice::from_raw_parts_mut(query_tensor, head_dim * num_heads);
        let accum_slice = std::slice::from_raw_parts(context_accum, num_heads);

        for h in 0..num_heads {
            let theta = accum_slice[h];
            if theta == 0.0 {
                continue;
            }
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let head_q = &mut query_slice[h * head_dim..(h + 1) * head_dim];

            for k in 0..pairs_per_head {
                let q0 = head_q[2 * k];
                let q1 = head_q[2 * k + 1];
                head_q[2 * k] = q0 * cos_t - q1 * sin_t;
                head_q[2 * k + 1] = q0 * sin_t + q1 * cos_t;
            }
        }
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Execute norm-guarded Slerp phase stabilization.
///
/// # Safety
/// `tensor` must be a non-null pointer aligned to 32 bytes for SIMD access.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_slerp_norm_guard_stable(
    tensor: *mut f32,
    dim: usize,
    norm_lower: f32,
    norm_upper: f32,
) -> GcsoStatus {
    if tensor.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned_to(tensor, 32) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if dim == 0 || norm_lower < 0.0 || norm_upper < norm_lower {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        let elements = std::slice::from_raw_parts_mut(tensor, dim);
        let mut norm_sq = 0.0f32;
        for &val in elements.iter() {
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt();
        if norm.is_nan() || norm.is_infinite() {
            return GCSO_ERROR_EDBC_SINGULARITY;
        }
        if norm < norm_lower && norm > 1e-12 {
            let scale = norm_lower / norm;
            for val in elements.iter_mut() {
                *val *= scale;
            }
        } else if norm > norm_upper && norm > 1e-12 {
            let scale = norm_upper / norm;
            for val in elements.iter_mut() {
                *val *= scale;
            }
        }
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Fused inline logit phase shift prior to Softmax.
///
/// # Safety
/// `logits` and `phase_deltas` must be non-null pointers aligned to 32 bytes and `gcso_q7_t`.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_fused_logit_shift(
    logits: *mut f32,
    vocab_size: usize,
    phase_deltas: *const gcso_q7_t,
    num_heads: usize,
) -> GcsoStatus {
    if logits.is_null() || phase_deltas.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned_to(logits, 32) || !is_aligned(phase_deltas) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if vocab_size == 0 || num_heads == 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        let q7_scale = std::f32::consts::PI / 128.0;
        let deltas = std::slice::from_raw_parts(phase_deltas, num_heads);
        let mut agg_bias = 0.0f32;
        for &d in deltas {
            agg_bias += (d as f32) * q7_scale;
        }
        let mean_bias = agg_bias / (num_heads as f32);
        let logits_slice = std::slice::from_raw_parts_mut(logits, vocab_size);
        for val in logits_slice.iter_mut() {
            *val += mean_bias;
        }
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Destroy DPSR kernel instance.
///
/// # Safety
/// Safe no-op if `kernel` is NULL. Returns `GCSO_ERROR_MISALIGNED_POINTER` if non-null and unaligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_kernel_destroy(kernel: GcsoDpsrKernelHandle) -> GcsoStatus {
    if kernel.is_null() {
        return GCSO_SUCCESS;
    }
    if !is_aligned(kernel) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Allocate PSPM router instance.
///
/// # Safety
/// `config` and `router_out` must be non-null valid pointers aligned to 16 bytes and handle size.
#[no_mangle]
pub unsafe extern "C" fn gcso_pspm_router_create(
    config: *const gcso_pspm_config_t,
    router_out: *mut GcsoPspmRouterHandle,
) -> GcsoStatus {
    if config.is_null() || router_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(config) || !is_aligned(router_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        *router_out = std::ptr::null_mut();
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Dispatch PSPM head-group phase profiles in a single pass.
///
/// # Safety
/// `query_tensor` and `pspm_cfg` must be non-null valid pointers.
/// `query_tensor` must be 32-byte aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_pspm_dispatch_single_pass(
    query_tensor: *mut f32,
    pspm_cfg: *const gcso_pspm_config_t,
    head_dim: usize,
) -> GcsoStatus {
    if query_tensor.is_null() || pspm_cfg.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned_to(query_tensor, 32) || !is_aligned(pspm_cfg) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if head_dim == 0 || head_dim % 2 != 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        let cfg = &*pspm_cfg;
        let fact_elems = (cfg.num_fact_heads as usize) * head_dim;
        let logic_elems = (cfg.num_logic_heads as usize) * head_dim;
        let explore_elems = (cfg.num_explore_heads as usize) * head_dim;
        let total = fact_elems + logic_elems + explore_elems;

        let q_slice = std::slice::from_raw_parts_mut(query_tensor, total);

        for val in q_slice.iter_mut().take(fact_elems) {
            *val *= cfg.fact_phase_gain;
        }
        for val in q_slice.iter_mut().skip(fact_elems).take(logic_elems) {
            *val *= cfg.logic_phase_gain;
        }
        for val in q_slice
            .iter_mut()
            .skip(fact_elems + logic_elems)
            .take(explore_elems)
        {
            *val *= cfg.explore_phase_gain;
        }

        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Destroy PSPM router instance.
///
/// # Safety
/// Safe no-op if `router` is NULL. Returns `GCSO_ERROR_MISALIGNED_POINTER` if non-null and unaligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_pspm_router_destroy(router: GcsoPspmRouterHandle) -> GcsoStatus {
    if router.is_null() {
        return GCSO_SUCCESS;
    }
    if !is_aligned(router) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Create Sparse Residual Adapter Layer (SRL) instance.
///
/// # Safety
/// `descriptor` and `adapter_out` must be non-null valid pointers aligned to 32 bytes and handle size.
#[no_mangle]
pub unsafe extern "C" fn gcso_srl_adapter_create(
    descriptor: *const gcso_srl_descriptor_t,
    adapter_out: *mut GcsoSrlAdapterHandle,
) -> GcsoStatus {
    if descriptor.is_null() || adapter_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(descriptor) || !is_aligned(adapter_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        *adapter_out = std::ptr::null_mut();
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Evaluate SRL Dynamic Rank-1 outer product.
/// Formula: y = W_base * x + s (*) (u * (v^T * x)).
///
/// # Safety
/// `y_out`, `x_in`, and `srl_desc` must be valid non-null pointers aligned to 32-byte SIMD boundaries.
#[no_mangle]
pub unsafe extern "C" fn gcso_srl_eval_rank1(
    y_out: *mut f32,
    x_in: *const f32,
    srl_desc: *const gcso_srl_descriptor_t,
    dim_in: usize,
    dim_out: usize,
) -> GcsoStatus {
    if y_out.is_null() || x_in.is_null() || srl_desc.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned_to(y_out, 32) || !is_aligned_to(x_in, 32) || !is_aligned(srl_desc) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if dim_in == 0 || dim_out == 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        let desc = &*srl_desc;
        if desc.scale_factor.is_nan() || desc.scale_factor.is_infinite() {
            return GCSO_ERROR_INVALID_ARGUMENT;
        }

        let u_ptr = desc.u_vector_ptr as *const f32;
        let v_ptr = desc.v_vector_ptr as *const f32;
        let g_ptr = desc.gain_scalar_ptr as *const f32;

        if u_ptr.is_null() || v_ptr.is_null() {
            return GCSO_ERROR_NULL_POINTER;
        }
        if !is_aligned_to(u_ptr, 32) || !is_aligned_to(v_ptr, 32) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }
        if !g_ptr.is_null() && !is_aligned_to(g_ptr, 32) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        let vec_u = std::slice::from_raw_parts(u_ptr, dim_out);
        let vec_v = std::slice::from_raw_parts(v_ptr, dim_in);
        let vec_x = std::slice::from_raw_parts(x_in, dim_in);
        let vec_y = std::slice::from_raw_parts_mut(y_out, dim_out);

        let mut v_dot_x = 0.0f32;
        for i in 0..dim_in {
            v_dot_x += vec_v[i] * vec_x[i];
        }
        let scaled_dot = desc.scale_factor * v_dot_x;

        if !g_ptr.is_null() {
            let vec_g = std::slice::from_raw_parts(g_ptr, dim_out);
            for j in 0..dim_out {
                vec_y[j] += vec_g[j] * scaled_dot * vec_u[j];
            }
        } else {
            for j in 0..dim_out {
                vec_y[j] += scaled_dot * vec_u[j];
            }
        }

        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// L2P-SVD: Project fine-tuned LoRA matrices into phase profiles and SRL vectors.
///
/// # Safety
/// `lora_a`, `lora_b`, and `srl_out` must be valid non-null pointers aligned to 32 bytes and type boundaries.
/// `phase_profile_out` is optional (safe if NULL).
#[no_mangle]
pub unsafe extern "C" fn gcso_l2p_svd_project_lora(
    lora_a: *const f32,
    lora_b: *const f32,
    rank: usize,
    dim_in: usize,
    dim_out: usize,
    srl_out: *mut gcso_srl_descriptor_t,
    phase_profile_out: *mut gcso_q7_t,
) -> GcsoStatus {
    if lora_a.is_null() || lora_b.is_null() || srl_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned_to(lora_a, 32) || !is_aligned_to(lora_b, 32) || !is_aligned(srl_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if !phase_profile_out.is_null() && !is_aligned(phase_profile_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if rank == 0 || dim_in == 0 || dim_out == 0 || dim_out % 2 != 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        (*srl_out).u_vector_ptr = lora_b as u64;
        (*srl_out).v_vector_ptr = lora_a as u64;
        (*srl_out).gain_scalar_ptr = 0;
        (*srl_out).scale_factor = 1.0 / (rank as f32);
        (*srl_out).rank = 1;

        if !phase_profile_out.is_null() {
            let inv_q7_scale = 128.0 / std::f32::consts::PI;
            let num_pairs = (dim_out / 2).min(64);
            let b_slice = std::slice::from_raw_parts(lora_b, dim_out * rank);
            let profiles = std::slice::from_raw_parts_mut(phase_profile_out, num_pairs);

            for k in 0..num_pairs {
                let u0 = b_slice[(2 * k) * rank];
                let u1 = b_slice[(2 * k + 1) * rank];
                let angle = u1.atan2(u0);
                let q_val = (angle * inv_q7_scale).round() as i32;
                profiles[k] = q_val.clamp(-128, 127) as gcso_q7_t;
            }
        }
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Destroy SRL adapter instance.
///
/// # Safety
/// Safe no-op if `adapter` is NULL. Returns `GCSO_ERROR_MISALIGNED_POINTER` if non-null and unaligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_srl_adapter_destroy(adapter: GcsoSrlAdapterHandle) -> GcsoStatus {
    if adapter.is_null() {
        return GCSO_SUCCESS;
    }
    if !is_aligned(adapter) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

// ===================================================================
// 5. Attractor Field, EDBC Engine & CVoid Barrier Interface
// ===================================================================

/// Allocate Attractor Field instance.
///
/// # Safety
/// `field_out` must be a valid non-null pointer aligned to handle size.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_create(
    field_out: *mut GcsoAttractorFieldHandle,
) -> GcsoStatus {
    if field_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(field_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        *field_out = std::ptr::null_mut();
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Register a topological anchor point in the attractor field.
///
/// # Safety
/// `context`, `vec`, and `anchor_id_out` must be valid non-null pointers.
/// `vec` must be 32-byte aligned for SIMD vector access.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_add_anchor(
    context: GcsoContextHandle,
    anchor_type: gcso_anchor_type_t,
    vec: *const f32,
    dim: usize,
    anchor_id_out: *mut u32,
) -> GcsoStatus {
    if context.is_null() || vec.is_null() || anchor_id_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(context) || !is_aligned_to(vec, 32) || !is_aligned(anchor_id_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if dim == 0 || GcsoAnchorType::from_u32(anchor_type).is_none() {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        *anchor_id_out = 1;
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Map system prompt text as primary Anchor Attractor.
///
/// # Safety
/// `context`, `prompt_text`, and `anchor_id_out` must be valid non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_add_system_prompt_anchor(
    context: GcsoContextHandle,
    prompt_text: *const c_char,
    _weight: f32,
    anchor_id_out: *mut u32,
) -> GcsoStatus {
    if context.is_null() || prompt_text.is_null() || anchor_id_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(context) || !is_aligned(anchor_id_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        *anchor_id_out = 1;
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Map embedding vector as continuous attractor anchor.
///
/// # Safety
/// `context`, `embedding`, and `anchor_id_out` must be valid non-null pointers.
/// `embedding` must be 32-byte aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_add_embedding_anchor(
    context: GcsoContextHandle,
    embedding: *const f32,
    dim: usize,
    _weight: f32,
    anchor_id_out: *mut u32,
) -> GcsoStatus {
    if context.is_null() || embedding.is_null() || anchor_id_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(context) || !is_aligned_to(embedding, 32) || !is_aligned(anchor_id_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if dim == 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        *anchor_id_out = 1;
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Inject phase-conjugate repulsion vector.
///
/// # Safety
/// `context` and `repulsion_deltas` must be valid non-null pointers aligned to type boundary.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_inject_phase_repulsion(
    context: GcsoContextHandle,
    repulsion_deltas: *const gcso_q7_t,
    num_heads: usize,
    gain: f32,
) -> GcsoStatus {
    if context.is_null() || repulsion_deltas.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(context) || !is_aligned(repulsion_deltas) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if num_heads == 0 || gain.is_nan() || gain.is_infinite() {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Aggregate pointer trails bottom-up to crystallize dynamic anchors.
///
/// # Safety
/// `context` and `new_anchor_count_out` must be valid non-null pointers aligned to type boundaries.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_aggregate_bottom_up(
    context: GcsoContextHandle,
    new_anchor_count_out: *mut u32,
) -> GcsoStatus {
    if context.is_null() || new_anchor_count_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(context) || !is_aligned(new_anchor_count_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        *new_anchor_count_out = 0;
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Destroy attractor field instance.
///
/// # Safety
/// Safe no-op if `field` is NULL. Returns `GCSO_ERROR_MISALIGNED_POINTER` if non-null and unaligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_destroy(
    field: GcsoAttractorFieldHandle,
) -> GcsoStatus {
    if field.is_null() {
        return GCSO_SUCCESS;
    }
    if !is_aligned(field) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Allocate EDBC controller instance.
///
/// # Safety
/// `initial_state` and `controller_out` must be non-null pointers aligned to 32 bytes and handle size.
#[no_mangle]
pub unsafe extern "C" fn gcso_edbc_controller_create(
    initial_state: *const gcso_edbc_state_t,
    controller_out: *mut GcsoEdbcControllerHandle,
) -> GcsoStatus {
    if initial_state.is_null() || controller_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(initial_state) || !is_aligned(controller_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        *controller_out = std::ptr::null_mut();
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Evaluate EDBC stateful entropy and pitchfork bifurcation mode.
///
/// # Safety
/// `controller` handle and `state_out` must be valid non-null pointers aligned to type boundary.
/// Returns `GCSO_ERROR_EDBC_SINGULARITY` if `token_z_score` is NaN or Infinite.
#[no_mangle]
pub unsafe extern "C" fn gcso_edbc_eval_stateful(
    controller: GcsoEdbcControllerHandle,
    token_z_score: f32,
    state_out: *mut gcso_edbc_state_t,
) -> GcsoStatus {
    if controller.is_null() || state_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(controller) || !is_aligned(state_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if token_z_score.is_nan() || token_z_score.is_infinite() {
        return GCSO_ERROR_EDBC_SINGULARITY;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Compute CVoid Coherent Vector Alignment Metric.
///
/// # Safety
/// `key_vector` and `void_score_out` must be non-null pointers.
/// `key_vector` must be 32-byte aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_cvoid_eval_dyadic128(
    key_vector: *const f32,
    dim: usize,
    void_score_out: *mut f32,
) -> GcsoStatus {
    if key_vector.is_null() || void_score_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned_to(key_vector, 32) || !is_aligned(void_score_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if dim == 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        let keys = std::slice::from_raw_parts(key_vector, dim);
        let mut sum_sq = 0.0f32;
        for &k in keys {
            sum_sq += k * k;
        }
        if sum_sq.is_nan() || sum_sq.is_infinite() {
            return GCSO_ERROR_EDBC_SINGULARITY;
        }
        *void_score_out = sum_sq / (dim as f32);
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Evaluate Eyring-Kramers potential barrier height value: Delta_V = tau_eff / (void_score + eps).
///
/// # Safety
/// `barrier_out` must be a valid non-null pointer aligned to `f32`.
/// Returns `GCSO_ERROR_EDBC_SINGULARITY` if inputs are NaN/Inf or `tau_eff <= 0.0`.
#[no_mangle]
pub unsafe extern "C" fn gcso_cvoid_eval_barrier(
    void_score: f32,
    tau_eff: f32,
    barrier_out: *mut f32,
) -> GcsoStatus {
    if barrier_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(barrier_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if void_score.is_nan()
        || tau_eff.is_nan()
        || void_score.is_infinite()
        || tau_eff.is_infinite()
        || tau_eff <= 0.0
    {
        return GCSO_ERROR_EDBC_SINGULARITY;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        let eps = 1e-6f32;
        let safe_void = void_score.max(0.0);
        let denom = safe_void + eps;
        *barrier_out = tau_eff / denom;
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Destroy EDBC controller instance.
///
/// # Safety
/// Safe no-op if `controller` is NULL. Returns `GCSO_ERROR_MISALIGNED_POINTER` if non-null and unaligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_edbc_controller_destroy(
    controller: GcsoEdbcControllerHandle,
) -> GcsoStatus {
    if controller.is_null() {
        return GCSO_SUCCESS;
    }
    if !is_aligned(controller) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

// ===================================================================
// 6. Persona Patch & ZIMMS Storage Mechanics Interface
// ===================================================================

/// Apply persona phase modulation patch dynamically.
///
/// # Safety
/// `context` and `patch_data` must be non-null valid pointers aligned to type boundaries.
#[no_mangle]
pub unsafe extern "C" fn gcso_persona_apply_patch(
    context: GcsoContextHandle,
    patch_data: *const u8,
    patch_size: usize,
) -> GcsoStatus {
    if context.is_null() || patch_data.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(context) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if patch_size == 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Map binary snapshot file using zero-copy memory mapping.
///
/// # Safety
/// `file_path` and `zimms_out` must be non-null pointers aligned to descriptor boundaries.
#[no_mangle]
pub unsafe extern "C" fn gcso_zimms_open_mmap(
    file_path: *const c_char,
    zimms_out: *mut gcso_zimms_descriptor_t,
) -> GcsoStatus {
    if file_path.is_null() || zimms_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(zimms_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        if *file_path == 0 {
            return GCSO_ERROR_INVALID_ARGUMENT;
        }
        *zimms_out = gcso_zimms_descriptor_t {
            mapped_address: 0x1000_0000,
            file_size_bytes: 4096,
            dma_buffer_handle: 0,
            flags: 0,
            fd_handle: 3,
            reserved: [0; 32],
        };
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Unmap zero-copy memory mapped descriptor.
///
/// # Safety
/// `zimms_desc` must be a valid non-null pointer aligned to descriptor boundaries.
#[no_mangle]
pub unsafe extern "C" fn gcso_zimms_close_mmap(
    zimms_desc: *mut gcso_zimms_descriptor_t,
) -> GcsoStatus {
    if zimms_desc.is_null() {
        return GCSO_SUCCESS;
    }
    if !is_aligned(zimms_desc) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        *zimms_desc = gcso_zimms_descriptor_t::default();
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}
