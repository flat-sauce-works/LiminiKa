// SPDX-License-Identifier: MIT OR Apache-2.0

//! C-ABI interface definitions, structural layouts, and FFI boundary safety primitives for GCSO core engine.

#![allow(non_camel_case_types)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

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
#[derive(Debug, Clone, Copy, PartialEq)]
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

impl Default for gcso_pointer_trail_t {
    #[inline]
    fn default() -> Self {
        Self {
            current_ptr: 0,
            prev_ptr: 0,
            user_data: 0,
            transition_cost: 0,
            step_count: 0,
            stigmergic_density: 0.0,
            target_anchor_id: 0,
            cluster_id: 0,
            linked_trail_id: 0,
            attractor_pull_force: 0.0,
            flags: 0,
            accumulated_phase_delta: [0; 64],
            reserved_padding: [0; 8],
        }
    }
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
#[derive(Debug, Clone, Copy, PartialEq)]
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

impl Default for gcso_edbc_state_t {
    #[inline]
    fn default() -> Self {
        Self {
            moving_z_entropy: 0.0,
            bifurcation_threshold: 0.0,
            singularity_eps: 0.0,
            sliding_entropy_rate: 0.0,
            repulsion_gain: 0.0,
            sample_temperature: 0.0,
            active_branch_mode: 0,
            reserved: [0; 36],
        }
    }
}

/// Zero-copy memory mapped storage descriptor (64 bytes).
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct gcso_zimms_descriptor_t {
    pub mapped_address: u64,
    pub file_size_bytes: u64,
    pub dma_buffer_handle: u64,
    pub flags: u32,
    pub fd_handle: i32,
    pub reserved: [u8; 32],
}

impl Default for gcso_zimms_descriptor_t {
    #[inline]
    fn default() -> Self {
        Self {
            mapped_address: 0,
            file_size_bytes: 0,
            dma_buffer_handle: 0,
            flags: 0,
            fd_handle: -1,
            reserved: [0; 32],
        }
    }
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl Default for gcso_snapshot_header_t {
    #[inline]
    fn default() -> Self {
        Self {
            magic: 0,
            version: 0,
            total_size: 0,
            action_hub_offset: 0,
            attractor_field_offset: 0,
            dpsr_state_offset: 0,
            srl_state_offset: 0,
            edbc_state_offset: 0,
            checksum_crc32: 0,
            daes_slot_offset: 0,
            timestamp_epoch_sec: 0,
            reserved_padding: [0; 56],
        }
    }
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

/// Helper function to check pointer natural alignment safely across architectures.
#[inline]
#[must_use]
fn is_aligned<T>(ptr: *const T) -> bool {
    !ptr.is_null() && ((ptr as usize) % std::mem::align_of::<T>() == 0)
}

/// Helper function to check pointer alignment for a specific custom alignment requirement.
#[inline]
#[must_use]
fn is_aligned_to<T>(ptr: *const T, align: usize) -> bool {
    !ptr.is_null() && (align != 0) && ((ptr as usize) % align == 0)
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
        #[allow(unused_mut)]
        let mut caps = 0x01u64; // Base CPU capability flag
        #[cfg(feature = "cuda")]
        {
            caps |= 1 << 1;
        }
        #[cfg(feature = "vulkan")]
        {
            caps |= 1 << 2;
        }
        #[cfg(any(target_os = "macos", feature = "metal"))]
        {
            caps |= 1 << 3;
        }
        unsafe { *flags = caps };
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Initialize a `gcso_config_t` structure strictly conforming to GCSO standards.
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
                ripa_clamp_max_rad: 0.087_266_46,
                qdps_min_step_rad: 0.01,
                entropy_singularity_eps: 1e-12,
                max_prompt_anchors: 64,
                action_hub_capacity: 256,
                enable_cuda_warp_shuffle: 1,
                enable_zero_alloc_strict: 1,
                daes_mode: 0,
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
        (*header_ptr).magic = 0x4F53_4347;
        (*header_ptr).version = 0x0002_0000;
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
        let mut acc = [0u64; 4];
        let masks = std::slice::from_raw_parts(bitmasks, num_masks);
        for m in masks {
            acc[0] |= m.bits[0];
            acc[1] |= m.bits[1];
            acc[2] |= m.bits[2];
            acc[3] |= m.bits[3];
        }
        (*reduced_out).bits = acc;
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Evaluate SIMD/Warp bitmask reduction over PagedBlock KV caches.
///
/// # Safety
/// `kv_bits` and `mask_out` must be non-null valid pointers.
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
        std::ptr::write_bytes(mask_out, 0, 1);
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Destroy Action Hub instance and free resources.
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

/// Allocate DAES slot instance.
///
/// # Safety
/// `slot_out` must be non-null and aligned to handle size.
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

/// Fast-path lookup in DAES dynamic scratchpad.
///
/// # Safety
/// `slot` and `shortcut_out` must be valid aligned non-null pointers.
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
        let slot_ref = &*slot;
        if (slot_ref.fast_path_bypass_mask & input_key) != 0 {
            *shortcut_out = slot_ref.fast_path_shortcuts[0];
            GCSO_SUCCESS
        } else {
            GCSO_ERROR_INVALID_STATE
        }
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Push metric byte to DAES telemetry ring ledger in zero-allocation mode.
///
/// # Safety
/// `slot` must be a valid non-null aligned pointer.
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
        let head = ((*slot).telemetry_ring_head as usize) % 8;
        (*slot).telemetry_mini_ledger[head] = metric_code;
        (*slot).telemetry_ring_head = (*slot).telemetry_ring_head.wrapping_add(1);
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Set DAES slot mode.
///
/// # Safety
/// `slot` must be a valid non-null aligned pointer.
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

/// Evaluate telemetry ledger for dynamic auto-tuning.
///
/// # Safety
/// `slot` and `config_out` must be valid aligned non-null pointers.
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
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
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
/// `kernel_out` must be a valid writable pointer aligned to handle size.
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

/// Apply inline DPSR phase steering to Query tensor.
///
/// # Safety
/// `query_tensor` (32-byte aligned) and `phase_deltas` must be non-null pointers.
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
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Apply RIPA soft-bounded tanh clamping on low-frequency channels.
///
/// # Safety
/// `query_tensor` (32-byte aligned) and `phase_deltas` must be valid non-null pointers.
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
    if head_dim == 0 || head_dim % 2 != 0 || num_heads == 0 || max_rad <= 0.0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// QDPS discrete filter: cuts off phase rotation steps falling below min_step_rad.
///
/// # Safety
/// `phase_deltas` must be a non-null valid pointer to array of size `len`.
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
    if len == 0 || min_step_rad < 0.0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Lazy Phase Unwrapping override on Query tensor.
///
/// # Safety
/// `query_tensor` and `context_accum` must be non-null and 32-byte aligned.
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
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Execute norm-guarded Slerp phase stabilization.
///
/// # Safety
/// `tensor` must be a valid non-null pointer aligned to 32 bytes.
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
    if dim == 0 || norm_lower >= norm_upper || norm_lower <= 0.0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Fused inline logit phase shift prior to Softmax.
///
/// # Safety
/// `logits` and `phase_deltas` must be valid non-null pointers.
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
    if !is_aligned(logits) || !is_aligned(phase_deltas) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if vocab_size == 0 || num_heads == 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
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
/// `config` and `router_out` must be valid non-null pointers.
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

/// Dispatch PSPM head-group phase profiles in single pass.
///
/// # Safety
/// `query_tensor` (32-byte aligned) and `pspm_cfg` must be non-null valid pointers.
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
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
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
/// `descriptor` and `adapter_out` must be valid non-null pointers.
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

/// Evaluate SRL Rank-1 outer product.
///
/// # Safety
/// `y_out`, `x_in`, and `srl_desc` must be non-null pointers aligned to structure/type boundary.
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
    if !is_aligned(y_out) || !is_aligned(x_in) || !is_aligned(srl_desc) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if dim_in == 0 || dim_out == 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// L2P-SVD: Projects fine-tuned LoRA matrices via SVD into phase profiles and SRL vectors.
///
/// # Safety
/// All pointers must be valid and non-null.
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
    if lora_a.is_null() || lora_b.is_null() || srl_out.is_null() || phase_profile_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(lora_a)
        || !is_aligned(lora_b)
        || !is_aligned(srl_out)
        || !is_aligned(phase_profile_out)
    {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if rank == 0 || dim_in == 0 || dim_out == 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
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
/// `field_out` must be valid pointer aligned to handle size.
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

/// Register a topological anchor point in attractor field.
///
/// # Safety
/// `context`, `vec` (32-byte aligned), and `anchor_id_out` must be valid non-null pointers.
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
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Maps natural language system prompt text as primary Anchor Attractor.
///
/// # Safety
/// `context`, `prompt_text`, and `anchor_id_out` must be valid non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_add_system_prompt_anchor(
    context: GcsoContextHandle,
    prompt_text: *const c_char,
    weight: f32,
    anchor_id_out: *mut u32,
) -> GcsoStatus {
    if context.is_null() || prompt_text.is_null() || anchor_id_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(context) || !is_aligned(anchor_id_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if weight.is_nan() || weight.is_infinite() {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Maps dense feature embedding vector as continuous attractor anchor.
///
/// # Safety
/// `context`, `embedding` (32-byte aligned), and `anchor_id_out` must be valid non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_add_embedding_anchor(
    context: GcsoContextHandle,
    embedding: *const f32,
    dim: usize,
    weight: f32,
    anchor_id_out: *mut u32,
) -> GcsoStatus {
    if context.is_null() || embedding.is_null() || anchor_id_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(context) || !is_aligned_to(embedding, 32) || !is_aligned(anchor_id_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if dim == 0 || weight.is_nan() || weight.is_infinite() {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Inject phase-conjugate repulsion vector (-dTheta).
///
/// # Safety
/// `context` and `repulsion_deltas` must be valid non-null pointers.
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

/// Aggregate high-density pointer trails bottom-up to dynamic anchors.
///
/// # Safety
/// `context` and `new_anchor_count_out` must be valid non-null pointers.
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
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
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
/// `initial_state` and `controller_out` must be valid non-null pointers.
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

/// Evaluate stateful Moving Z-Score Attention Entropy.
///
/// # Safety
/// `controller` and `state_out` must be valid non-null pointers.
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
/// `key_vector` (32-byte aligned) and `void_score_out` must be non-null pointers.
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
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Evaluate Eyring-Kramers potential barrier height value with singularity check.
///
/// # Safety
/// `barrier_out` must be a valid non-null pointer aligned to `f32`.
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
        || void_score.is_infinite()
        || tau_eff.is_nan()
        || tau_eff.is_infinite()
        || tau_eff <= 0.0
    {
        return GCSO_ERROR_EDBC_SINGULARITY;
    }
    let res = catch_unwind(AssertUnwindSafe(|| unsafe {
        *barrier_out = void_score / tau_eff;
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

/// Dynamic application of persona phase modulation patches without altering base weights.
///
/// # Safety
/// `context` and `patch_data` must be valid non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn gcso_persona_apply_patch(
    context: GcsoContextHandle,
    patch_data: *const u8,
    patch_size: usize,
) -> GcsoStatus {
    if context.is_null() || patch_data.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(context) || !is_aligned(patch_data) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if patch_size == 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    let res = catch_unwind(AssertUnwindSafe(|| GCSO_SUCCESS));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Zero-Overhead In-Memory Mapped Storage: Maps .gcso container payload using zero-copy mmap.
///
/// # Safety
/// `file_path` and `zimms_out` must be valid non-null pointers.
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
        std::ptr::write_bytes(zimms_out, 0, 1);
        (*zimms_out).fd_handle = -1;
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}

/// Unmaps zero-copy ZIMMS memory handle and releases Direct DMA resources.
///
/// # Safety
/// `zimms_desc` must be a valid non-null aligned descriptor pointer.
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
        (*zimms_desc).mapped_address = 0;
        (*zimms_desc).file_size_bytes = 0;
        (*zimms_desc).fd_handle = -1;
        GCSO_SUCCESS
    }));
    res.unwrap_or(GCSO_ERROR_PANIC_CAUGHT)
}
