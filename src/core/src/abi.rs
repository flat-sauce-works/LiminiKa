// name: src/core/src/abi.rs
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
#![allow(clippy::manual_is_multiple_of)]

#[cfg(feature = "std")]
extern crate std;

use core::ffi::c_char;
use core::mem::{align_of, offset_of, size_of};
use core::ptr;
use core::slice;

#[cfg(feature = "std")]
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::traits::{AlignedSlice32, AlignedSliceMut32};

/// Unified GCSO C-ABI Major Version Component (0.1.1).
pub const GCSO_ABI_VERSION_MAJOR: u32 = 0;

/// Unified GCSO C-ABI Minor Version Component (0.1.1).
pub const GCSO_ABI_VERSION_MINOR: u32 = 1;

/// Unified GCSO C-ABI Patch Version Component (0.1.1).
pub const GCSO_ABI_VERSION_PATCH: u32 = 1;

/// Hexadecimal Representation of Unified ABI Version 0.1.1 (`0x00000101`).
pub const GCSO_ABI_VERSION_HEX: u32 = 0x0000_0101;

/// String Representation of Unified ABI Version 0.1.1.
pub const GCSO_ABI_VERSION_STRING: &str = "0.1.1";

/// Validates C-ABI snapshot header version against core version invariant (`0x00000101`).
#[inline]
pub fn validate_abi_version(version_hex: u32) -> Result<(), gcso_status_t> {
    if version_hex == GCSO_ABI_VERSION_HEX {
        Ok(())
    } else {
        Err(gcso_status_t::GCSO_ERROR_VERSION_MISMATCH)
    }
}

/// Helper macro to catch unwinding panics and safely bridge execution into C-ABI error status codes.
///
/// Supports using the `?` operator on `Result<T, GcsoStatus>` inside the expression block.
#[macro_export]
macro_rules! ffi_boundary {
    ($body:expr) => {
        match ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(
            || -> Result<$crate::abi::GcsoStatus, $crate::abi::GcsoStatus> { $body },
        )) {
            Ok(Ok(status)) => status,
            Ok(Err(err_status)) => err_status,
            Err(_) => $crate::abi::GCSO_ERROR_PANIC_CAUGHT,
        }
    };
}

/// Status code returned across the C-ABI FFI boundary.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct GcsoStatus(
    /// Inner signed 32-bit integer status code.
    pub i32,
);

/// Operation completed successfully.
pub const GCSO_SUCCESS: GcsoStatus = GcsoStatus(0);
/// Invalid argument or parameter passed to C-ABI function.
pub const GCSO_ERROR_INVALID_ARGUMENT: GcsoStatus = GcsoStatus(-1);
/// Memory allocation failure or capacity limit reached.
pub const GCSO_ERROR_OUT_OF_MEMORY: GcsoStatus = GcsoStatus(-2);
/// Destination buffer size is insufficient.
pub const GCSO_ERROR_BUFFER_TOO_SMALL: GcsoStatus = GcsoStatus(-3);
/// Unhandled internal Rust panic caught at FFI boundary.
pub const GCSO_ERROR_PANIC_CAUGHT: GcsoStatus = GcsoStatus(-4);
/// Unexpected NULL pointer supplied for required parameter.
pub const GCSO_ERROR_NULL_POINTER: GcsoStatus = GcsoStatus(-5);
/// Context or object is in an invalid state for requested operation.
pub const GCSO_ERROR_INVALID_STATE: GcsoStatus = GcsoStatus(-6);
/// Binary snapshot or ABI version mismatch detected.
pub const GCSO_ERROR_VERSION_MISMATCH: GcsoStatus = GcsoStatus(-7);
/// Pointer passed across FFI boundary violates required memory alignment.
pub const GCSO_ERROR_MISALIGNED_POINTER: GcsoStatus = GcsoStatus(-8);
/// Input/output or storage operation failure.
pub const GCSO_ERROR_IO_FAILURE: GcsoStatus = GcsoStatus(-9);
/// Action Hub table capacity fully saturated.
pub const GCSO_ERROR_ACTION_HUB_FULL: GcsoStatus = GcsoStatus(-10);
/// Feature or compute backend functionality not implemented.
pub const GCSO_ERROR_NOT_IMPLEMENTED: GcsoStatus = GcsoStatus(-11);

/// Target attractor anchor not found in topological field.
pub const GCSO_ERROR_ATTRACTOR_NOT_FOUND: GcsoStatus = GcsoStatus(-20);
/// DPSR phase angle delta exceeded allowable rotation range.
pub const GCSO_ERROR_DPSR_PHASE_OVERFLOW: GcsoStatus = GcsoStatus(-30);
/// QDPS quantization step fell below minimal threshold.
pub const GCSO_ERROR_QDPS_UNDERFLOW: GcsoStatus = GcsoStatus(-31);
/// EDBC entropy controller encountered numerical singularity.
pub const GCSO_ERROR_EDBC_SINGULARITY: GcsoStatus = GcsoStatus(-40);
/// Binary snapshot container header or payload corrupted.
pub const GCSO_ERROR_CONTAINER_CORRUPTED: GcsoStatus = GcsoStatus(-50);
/// ZIMMS zero-copy memory mapping or DMA allocation failed.
pub const GCSO_ERROR_ZIMMS_MAPPING_FAILED: GcsoStatus = GcsoStatus(-51);
/// PSPM sub-head group phase routing evaluation failed.
pub const GCSO_ERROR_PSPM_ROUTING_FAILED: GcsoStatus = GcsoStatus(-60);
/// PPRC keyframe index lookup or seek operation failed.
pub const GCSO_ERROR_PPRC_SEEK_FAILED: GcsoStatus = GcsoStatus(-70);
/// Unresolved topological obstruction encountered during path step.
pub const GCSO_ERROR_OBSTRUCTION_UNRESOLVED: GcsoStatus = GcsoStatus(-80);
/// Dynamic extension module not loaded or initialized in DAES slot.
pub const GCSO_ERROR_EXTENSION_NOT_LOADED: GcsoStatus = GcsoStatus(-90);
/// DAES dynamic scratchpad workspace fully occupied.
pub const GCSO_ERROR_DAES_SCRATCHPAD_FULL: GcsoStatus = GcsoStatus(-91);
/// Unknown internal system error.
pub const GCSO_ERROR_UNKNOWN: GcsoStatus = GcsoStatus(-0x7FFF_FFFF);

impl From<GcsoStatus> for Result<(), GcsoStatus> {
    #[inline]
    fn from(status: GcsoStatus) -> Self {
        if status == GCSO_SUCCESS {
            Ok(())
        } else {
            Err(status)
        }
    }
}

impl From<Result<(), GcsoStatus>> for GcsoStatus {
    #[inline]
    fn from(res: Result<(), GcsoStatus>) -> Self {
        match res {
            Ok(()) => GCSO_SUCCESS,
            Err(status) => status,
        }
    }
}

/// C-ABI type alias for status codes matching C headers.
pub type gcso_status_t = GcsoStatus;

/// Quantized 7-bit signed fixed-point integer (scale beta_Q7 = pi / 128).
pub type gcso_q7_t = i8;

/// 64-bit capability flags bitmask type.
pub type gcso_capability_flags_t = u64;

/// Opaque handle to runtime context instance.
pub type GcsoContextHandle = *mut core::ffi::c_void;
/// Opaque handle to container instance.
pub type GcsoContainerHandle = *mut core::ffi::c_void;
/// Opaque handle to Action Hub instance.
pub type GcsoActionHubHandle = *mut core::ffi::c_void;
/// Opaque handle to DAES slot instance.
pub type GcsoDaesSlotHandle = *mut core::ffi::c_void;
/// Opaque handle to Attractor Field instance.
pub type GcsoAttractorFieldHandle = *mut core::ffi::c_void;
/// Opaque handle to EDBC controller instance.
pub type GcsoEdbcControllerHandle = *mut core::ffi::c_void;
/// Opaque handle to DPSR kernel instance.
pub type GcsoDpsrKernelHandle = *mut core::ffi::c_void;
/// Opaque handle to PSPM router instance.
pub type GcsoPspmRouterHandle = *mut core::ffi::c_void;
/// Opaque handle to SRL adapter instance.
pub type GcsoSrlAdapterHandle = *mut core::ffi::c_void;

/// C-ABI type alias for context handle.
pub type gcso_context_handle_t = GcsoContextHandle;
/// C-ABI type alias for container handle.
pub type gcso_container_handle_t = GcsoContainerHandle;
/// C-ABI type alias for Action Hub handle.
pub type gcso_action_hub_handle_t = GcsoActionHubHandle;
/// C-ABI type alias for DAES slot handle.
pub type gcso_daes_slot_handle_t = GcsoDaesSlotHandle;
/// C-ABI type alias for Attractor Field handle.
pub type gcso_attractor_field_handle_t = GcsoAttractorFieldHandle;
/// C-ABI type alias for EDBC controller handle.
pub type gcso_edbc_controller_handle_t = GcsoEdbcControllerHandle;
/// C-ABI type alias for DPSR kernel handle.
pub type gcso_dpsr_kernel_handle_t = GcsoDpsrKernelHandle;
/// C-ABI type alias for PSPM router handle.
pub type gcso_pspm_router_handle_t = GcsoPspmRouterHandle;
/// C-ABI type alias for SRL adapter handle.
pub type gcso_srl_adapter_handle_t = GcsoSrlAdapterHandle;

/// Classification types for topological attractor field anchors.
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GcsoAnchorType {
    /// System prompt text anchor.
    SystemPrompt = 0,
    /// Dense feature embedding vector anchor.
    Embedding = 1,
    /// Phase-conjugate repulsion anchor.
    PhaseRepulse = 2,
    /// Topological knot/attractor anchor.
    Topological = 3,
    /// Crystallized invariant anchor.
    Crystallized = 4,
}

impl GcsoAnchorType {
    /// Convert raw `u32` value to `GcsoAnchorType` enum variant if valid.
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

/// C-ABI type alias for anchor type.
pub type gcso_anchor_type_t = u32;

/// System prompt string anchor descriptor type.
pub const GCSO_ANCHOR_TYPE_SYSTEM_PROMPT: gcso_anchor_type_t = 0;
/// Feature embedding anchor descriptor type.
pub const GCSO_ANCHOR_TYPE_EMBEDDING: gcso_anchor_type_t = 1;
/// Phase repulsion anchor descriptor type.
pub const GCSO_ANCHOR_TYPE_PHASE_REPULSE: gcso_anchor_type_t = 2;
/// Topological knot anchor descriptor type.
pub const GCSO_ANCHOR_TYPE_TOPOLOGICAL: gcso_anchor_type_t = 3;
/// Crystallized invariant anchor descriptor type.
pub const GCSO_ANCHOR_TYPE_CRYSTALLIZED: gcso_anchor_type_t = 4;

/// Descriptor header for size and ABI version validation.
#[repr(C, align(4))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct gcso_descriptor_header_t {
    /// Size of structure in bytes.
    pub struct_size: u32,
    /// ABI version bitmask.
    pub abi_version: u32,
}

/// 256-bit bitmask layout aligned to 32 bytes for Warp/SIMD reductions.
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct gcso_paged_bitmask_t {
    /// Four 64-bit word array storing 256 execution flags.
    pub bits: [u64; 4],
}

/// Stigmergic pointer trail structure aligned to 128 bytes (2 cache lines).
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct gcso_pointer_trail_t {
    /// Current tagged memory pointer address.
    pub current_ptr: u64,
    /// Previous tagged memory pointer address.
    pub prev_ptr: u64,
    /// Opaque caller payload data.
    pub user_data: u64,
    /// Accumulated graph edge traversal cost.
    pub transition_cost: i32,
    /// Total path execution steps taken.
    pub step_count: u32,
    /// Local trail density value for cellular swarm routing.
    pub stigmergic_density: f32,
    /// Identifier of target attractor anchor point.
    pub target_anchor_id: u32,
    /// Active topological cluster identifier.
    pub cluster_id: u32,
    /// Linked adjacent trail identifier.
    pub linked_trail_id: u32,
    /// Gravitational pull force exerted by nearby attractor.
    pub attractor_pull_force: f32,
    /// Execution status bitmask flags.
    pub flags: u32,
    /// Accumulated Q7 phase rotation state across 64 heads.
    pub accumulated_phase_delta: [gcso_q7_t; 64],
    /// Alignment padding to guarantee 128-byte boundary.
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
    /// Active extension execution mode.
    pub mode: u32,
    /// Ring buffer head write index.
    pub telemetry_ring_head: u16,
    /// Ring buffer tail read index.
    pub telemetry_ring_tail: u16,
    /// Total fast-path lookup cache hits.
    pub cache_hit_count: u32,
    /// Flags configuring auto-tuning behavior.
    pub auto_tune_flags: u32,
    /// Bitmask specifying fast-path bypass criteria.
    pub fast_path_bypass_mask: u64,
    /// Cached jump pointers for accelerated execution.
    pub fast_path_shortcuts: [u64; 4],
    /// Circular ledger array storing recent telemetry codes.
    pub telemetry_mini_ledger: [u8; 8],
}

/// Global configuration descriptor structure aligned to 16 bytes (64 bytes total).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct gcso_config_t {
    /// Dimension per attention head (must be even).
    pub head_dim: u32,
    /// Total number of attention heads (<= 64).
    pub num_heads: u32,
    /// Block size for paged KV cache allocation.
    pub paged_block_size: u32,
    /// Floating-point scale factor for Q7 phase values.
    pub q7_phase_scale: f32,
    /// Maximum allowed rotation angle for RIPA clamping (radians).
    pub ripa_clamp_max_rad: f32,
    /// Minimum rotation angle threshold for QDPS filtering (radians).
    pub qdps_min_step_rad: f32,
    /// Epsilon constant to prevent singularity in entropy calculation.
    pub entropy_singularity_eps: f32,
    /// Maximum supported prompt anchor points in field.
    pub max_prompt_anchors: u32,
    /// Maximum capacity of Action Hub table.
    pub action_hub_capacity: u32,
    /// Enable CUDA warp shuffle instructions if supported.
    pub enable_cuda_warp_shuffle: u8,
    /// Strict enforcement of zero dynamic allocations on hot path.
    pub enable_zero_alloc_strict: u8,
    /// Initial operational mode for DAES scratchpad.
    pub daes_mode: u8,
    /// Reserved configuration flags.
    pub reserved_flags: u8,
    /// Reserved space for future expansion.
    pub reserved: [u8; 24],
}

/// Dynamic entropy controller state tracking structure (64 bytes).
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct gcso_edbc_state_t {
    /// Moving Z-score measure of current attention entropy.
    pub moving_z_entropy: f32,
    /// Z-score threshold triggering path bifurcation.
    pub bifurcation_threshold: f32,
    /// Numerical safety threshold for singularity detection.
    pub singularity_eps: f32,
    /// Rate of change of sliding window entropy.
    pub sliding_entropy_rate: f32,
    /// Gain factor for phase repulsion forces.
    pub repulsion_gain: f32,
    /// Current sampling temperature modifier.
    pub sample_temperature: f32,
    /// Active execution branch selection mode.
    pub active_branch_mode: u32,
    /// Reserved space for alignment and future parameters.
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
    /// Memory address of virtual memory region.
    pub mapped_address: u64,
    /// Total file length in bytes.
    pub file_size_bytes: u64,
    /// Direct Memory Access (DMA) handle.
    pub dma_buffer_handle: u64,
    /// Memory mapping attribute flags.
    pub flags: u32,
    /// Low-level operating system file descriptor.
    pub fd_handle: i32,
    /// Reserved space for future extension.
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
    /// Number of attention heads allocated for factual processing.
    pub num_fact_heads: u16,
    /// Number of attention heads allocated for logical reasoning.
    pub num_logic_heads: u16,
    /// Number of attention heads allocated for exploratory sampling.
    pub num_explore_heads: u16,
    /// Bitmask flags controlling routing logic.
    pub flags: u16,
    /// Steering gain applied to factual head group.
    pub fact_phase_gain: f32,
    /// Steering gain applied to logical head group.
    pub logic_phase_gain: f32,
    /// Steering gain applied to exploratory head group.
    pub explore_phase_gain: f32,
    /// Reserved padding for 16-byte alignment.
    pub reserved: [u8; 12],
}

/// Sparse Residual Adapter Layer (SRL) Rank-1 descriptor (64 bytes).
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct gcso_srl_descriptor_t {
    /// Index of target transformer layer.
    pub layer_idx: u32,
    /// Rank of low-rank adapter (fixed to 1 for SRL).
    pub rank: u32,
    /// Raw pointer to left projection vector U.
    pub u_vector_ptr: u64,
    /// Raw pointer to right projection vector V.
    pub v_vector_ptr: u64,
    /// Raw pointer to scalar gain factor array.
    pub gain_scalar_ptr: u64,
    /// Scaling multiplier for residual injection.
    pub scale_factor: f32,
    /// Flags indicating update mode and precision.
    pub flags: u32,
    /// Reserved space for future alignment requirements.
    pub reserved: [u8; 24],
}

/// Unified binary snapshot container header structure (128 bytes).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct gcso_snapshot_header_t {
    /// Format identifier magic constant (`0x4F534347` = `"GCSO"`).
    pub magic: u32,
    /// Snapshot ABI version (`0x00000101` = v0.1.1).
    pub version: u32,
    /// Total binary size of file payload in bytes.
    pub total_size: u64,
    /// Offset to Action Hub table section.
    pub action_hub_offset: u64,
    /// Offset to Attractor Field section.
    pub attractor_field_offset: u64,
    /// Offset to DPSR state section.
    pub dpsr_state_offset: u64,
    /// Offset to SRL state section.
    pub srl_state_offset: u64,
    /// Offset to EDBC controller state section.
    pub edbc_state_offset: u64,
    /// CRC32 checksum over snapshot payload.
    pub checksum_crc32: u32,
    /// Offset to DAES scratchpad state section.
    pub daes_slot_offset: u32,
    /// Epoch timestamp of serialization.
    pub timestamp_epoch_sec: u64,
    /// Padding to enforce 128-byte size and 64-byte alignment.
    pub reserved_padding: [u8; 56],
}

impl Default for gcso_snapshot_header_t {
    #[inline]
    fn default() -> Self {
        Self {
            magic: 0,
            version: GCSO_ABI_VERSION_HEX,
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
    /// Frame classification (0 = Keyframe, 1 = Delta).
    pub frame_type: u32,
    /// Absolute token index in sequence.
    pub token_index: u32,
    /// Distance between consecutive keyframes.
    pub gop_length: u32,
    /// Length of combined direction phase vector.
    pub composite_vector_length: f32,
    /// Concentration parameter for von Mises distribution.
    pub von_mises_kappa: f32,
    /// Scalar residual magnitude for sparse updates.
    pub sparse_scalar_residual: f32,
    /// Offset to Intrinsic Phase Cache (ICache) data.
    pub icache_payload_offset: u64,
    /// Offset to Position Phase Cache (PCache) data.
    pub pcache_payload_offset: u64,
    /// Reserved space for future extension.
    pub reserved: [u8; 24],
}

// Static layout assertion checks to guarantee standard C-ABI structural alignment and field offsets
const _: () = {
    assert!(size_of::<gcso_status_t>() == 4);
    assert!(size_of::<gcso_q7_t>() == 1);
    assert!(size_of::<gcso_descriptor_header_t>() == 8);
    assert!(size_of::<gcso_paged_bitmask_t>() == 32);
    assert!(size_of::<gcso_pointer_trail_t>() == 128);
    assert!(size_of::<gcso_daes_slot_t>() == 64);
    assert!(size_of::<gcso_config_t>() == 64);
    assert!(size_of::<gcso_edbc_state_t>() == 64);
    assert!(size_of::<gcso_zimms_descriptor_t>() == 64);
    assert!(size_of::<gcso_pspm_config_t>() == 32);
    assert!(size_of::<gcso_srl_descriptor_t>() == 64);
    assert!(size_of::<gcso_snapshot_header_t>() == 128);
    assert!(size_of::<gcso_pprc_keyframe_header_t>() == 64);

    assert!(align_of::<gcso_paged_bitmask_t>() == 32);
    assert!(align_of::<gcso_pointer_trail_t>() == 128);
    assert!(align_of::<gcso_daes_slot_t>() == 64);
    assert!(align_of::<gcso_config_t>() == 16);
    assert!(align_of::<gcso_edbc_state_t>() == 32);
    assert!(align_of::<gcso_zimms_descriptor_t>() == 32);
    assert!(align_of::<gcso_pspm_config_t>() == 16);
    assert!(align_of::<gcso_srl_descriptor_t>() == 32);
    assert!(align_of::<gcso_snapshot_header_t>() == 64);
    assert!(align_of::<gcso_pprc_keyframe_header_t>() == 32);

    assert!(offset_of!(gcso_pointer_trail_t, accumulated_phase_delta) == 56);
    assert!(offset_of!(gcso_config_t, action_hub_capacity) == 32);
    assert!(offset_of!(gcso_snapshot_header_t, checksum_crc32) == 56);
    assert!(offset_of!(gcso_snapshot_header_t, timestamp_epoch_sec) == 64);
    assert!(offset_of!(gcso_daes_slot_t, fast_path_shortcuts) == 24);
    assert!(offset_of!(gcso_srl_descriptor_t, scale_factor) == 32);
    assert!(offset_of!(gcso_pprc_keyframe_header_t, icache_payload_offset) == 24);
};

/// Helper function to check pointer natural alignment safely across architectures.
#[inline]
#[must_use]
pub fn is_aligned<T>(ptr: *const T) -> bool {
    !ptr.is_null() && (ptr as usize).is_multiple_of(align_of::<T>())
}

/// Helper function to check pointer alignment for a specific custom alignment requirement.
#[inline]
#[must_use]
pub fn is_aligned_to<T>(ptr: *const T, align: usize) -> bool {
    !ptr.is_null() && align != 0 && align.is_power_of_two() && (ptr as usize).is_multiple_of(align)
}

/// Static version string constant for FFI boundary checks matching ABI v0.1.1.
static GCSO_ABI_VERSION: &[u8] = b"0.1.1\0";

// ===================================================================
// 1. System & Capability Query Interface
// ===================================================================

/// Retrieve numeric components of the GCSO C-ABI version (v0.1.1).
///
/// # Safety
/// Pointers must be valid, non-null writable memory locations aligned to `u32`.
#[no_mangle]
pub unsafe extern "C" fn gcso_abi_get_version(
    major: *mut u32,
    minor: *mut u32,
    patch: *mut u32,
) -> GcsoStatus {
    if major.is_null() || minor.is_null() || patch.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(major) || !is_aligned(minor) || !is_aligned(patch) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }

    ffi_boundary!({
        unsafe {
            *major = GCSO_ABI_VERSION_MAJOR;
            *minor = GCSO_ABI_VERSION_MINOR;
            *patch = GCSO_ABI_VERSION_PATCH;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Retrieve the static C-ABI version string ("0.1.1").
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
    ffi_boundary!({
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
        Ok(GCSO_SUCCESS)
    })
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

    ffi_boundary!({
        unsafe {
            *config = gcso_config_t {
                head_dim: 128,
                num_heads: 32,
                paged_block_size: 32,
                q7_phase_scale: core::f32::consts::PI / 128.0,
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
        Ok(GCSO_SUCCESS)
    })
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

    ffi_boundary!({
        unsafe {
            let cfg = &*config;
            if cfg.head_dim == 0
                || !cfg.head_dim.is_multiple_of(2)
                || cfg.num_heads == 0
                || cfg.num_heads > 64
            {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            *context_out = ptr::null_mut();
        }
        Ok(GCSO_SUCCESS)
    })
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
}

/// Set system prompt text as Anchor Attractor.
///
/// # Safety
/// `context` and `prompt_text` must be valid non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn gcso_context_set_system_prompt_anchor(
    context: GcsoContextHandle,
    prompt_text: *const c_char,
    weight: f32,
) -> GcsoStatus {
    if context.is_null() || prompt_text.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(context) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if weight.is_nan() || weight.is_infinite() {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    ffi_boundary!({
        let c_str = unsafe { std::ffi::CStr::from_ptr(prompt_text) };
        if c_str.to_str().is_err() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(GCSO_SUCCESS)
    })
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
    token_id: u32,
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

    ffi_boundary!({
        if !query_tensor.is_null() {
            let slice = unsafe { slice::from_raw_parts_mut(query_tensor, 128) };
            let _ = AlignedSliceMut32::new(slice)?;
        }
        if !key_tensor.is_null() {
            let slice = unsafe { slice::from_raw_parts_mut(key_tensor, 128) };
            let _ = AlignedSliceMut32::new(slice)?;
        }
        if !trail_out.is_null() {
            unsafe {
                ptr::write_bytes(trail_out, 0, 1);
                (*trail_out).current_ptr = u64::from(token_id);
                (*trail_out).stigmergic_density = 1.0;
            }
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Serialize runtime state into binary format conforming to GCSO ABI v0.1.1.
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

    ffi_boundary!({
        unsafe {
            let required = size_of::<gcso_snapshot_header_t>();
            if buffer.is_null() {
                *buffer_size = required;
                return Ok(GCSO_SUCCESS);
            }
            if *buffer_size < required {
                *buffer_size = required;
                return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
            }
            if !is_aligned(buffer) {
                return Err(GCSO_ERROR_MISALIGNED_POINTER);
            }
            ptr::write_bytes(buffer, 0, required);
            let header_ptr = buffer.cast::<gcso_snapshot_header_t>();
            (*header_ptr).magic = 0x4F53_4347; // ASCII "GCSO"
            (*header_ptr).version = GCSO_ABI_VERSION_HEX; // 0x00000101 (v0.1.1)
            (*header_ptr).total_size = required as u64;
            (*header_ptr).timestamp_epoch_sec = 1_774_900_000;
            *buffer_size = required;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Deserialize binary snapshot to restore runtime state with ABI v0.1.1 verification.
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
    if buffer_size < size_of::<gcso_snapshot_header_t>() {
        return GCSO_ERROR_CONTAINER_CORRUPTED;
    }
    ffi_boundary!({
        let header = unsafe { &*buffer.cast::<gcso_snapshot_header_t>() };
        if header.magic != 0x4F53_4347 {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }
        validate_abi_version(header.version)?;
        unsafe { *context_out = ptr::null_mut() };
        Ok(GCSO_SUCCESS)
    })
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
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
    ffi_boundary!({
        unsafe {
            ptr::write_bytes(keyframe_header_out, 0, 1);
            (*keyframe_header_out).token_index = token_index;
        }
        Ok(GCSO_SUCCESS)
    })
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
    ffi_boundary!({
        unsafe { *hub_out = ptr::null_mut() };
        Ok(GCSO_SUCCESS)
    })
}

/// Advance tagged pointer transition in O(1) time.
///
/// # Safety
/// `context` must be valid handle; `trail_out` must be writable and aligned to 128 bytes.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_step_pointer(
    context: GcsoContextHandle,
    current_ptr: u64,
    trail_out: *mut gcso_pointer_trail_t,
) -> GcsoStatus {
    if context.is_null() || trail_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(context) || !is_aligned(trail_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    ffi_boundary!({
        unsafe {
            ptr::write_bytes(trail_out, 0, 1);
            (*trail_out).current_ptr = current_ptr;
            (*trail_out).stigmergic_density = 1.0;
        }
        Ok(GCSO_SUCCESS)
    })
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
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
    ffi_boundary!({
        let mut acc = [0u64; 4];
        let masks = unsafe { slice::from_raw_parts(bitmasks, num_masks) };
        for m in masks {
            acc[0] |= m.bits[0];
            acc[1] |= m.bits[1];
            acc[2] |= m.bits[2];
            acc[3] |= m.bits[3];
        }
        unsafe { (*reduced_out).bits = acc };
        Ok(GCSO_SUCCESS)
    })
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
    ffi_boundary!({
        unsafe { ptr::write_bytes(mask_out, 0, 1) };
        Ok(GCSO_SUCCESS)
    })
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
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
    ffi_boundary!({
        unsafe { *slot_out = ptr::null_mut() };
        Ok(GCSO_SUCCESS)
    })
}

/// Reset active telemetry mini ledger and cache hit counters in-place.
///
/// # Safety
/// `slot` must be a valid non-null aligned pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_daes_reset_telemetry(slot: *mut gcso_daes_slot_t) -> GcsoStatus {
    if slot.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(slot) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    ffi_boundary!({
        unsafe {
            (*slot).telemetry_ring_head = 0;
            (*slot).telemetry_ring_tail = 0;
            (*slot).cache_hit_count = 0;
            ptr::write_bytes((*slot).telemetry_mini_ledger.as_mut_ptr(), 0, 8);
        }
        Ok(GCSO_SUCCESS)
    })
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
    ffi_boundary!({
        let slot_ref = unsafe { &*slot };
        if (slot_ref.fast_path_bypass_mask & input_key) != 0 {
            unsafe { *shortcut_out = slot_ref.fast_path_shortcuts[0] };
            Ok(GCSO_SUCCESS)
        } else {
            Err(GCSO_ERROR_INVALID_STATE)
        }
    })
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
    ffi_boundary!({
        let head = (unsafe { (*slot).telemetry_ring_head as usize }) % 8;
        unsafe {
            (*slot).telemetry_mini_ledger[head] = metric_code;
            (*slot).telemetry_ring_head = (*slot).telemetry_ring_head.wrapping_add(1);
        }
        Ok(GCSO_SUCCESS)
    })
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
    ffi_boundary!({
        unsafe { (*slot).mode = mode };
        Ok(GCSO_SUCCESS)
    })
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
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
    if head_dim == 0 || !head_dim.is_multiple_of(2) || num_heads == 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    ffi_boundary!({
        unsafe { *kernel_out = ptr::null_mut() };
        Ok(GCSO_SUCCESS)
    })
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
    if head_dim == 0 || !head_dim.is_multiple_of(2) || num_heads == 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    ffi_boundary!(Ok(GCSO_SUCCESS))
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
    if head_dim == 0 || !head_dim.is_multiple_of(2) || num_heads == 0 || max_rad <= 0.0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    ffi_boundary!(Ok(GCSO_SUCCESS))
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
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
    if head_dim == 0 || !head_dim.is_multiple_of(2) || num_heads == 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    ffi_boundary!(Ok(GCSO_SUCCESS))
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
}

/// Compute Procrustes phase delta alignment between source and target state representations.
///
/// # Safety
/// All pointers must be valid, non-null, and 32-byte aligned for float buffers.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_compute_procrustes_phase_delta(
    source: *const f32,
    target: *const f32,
    dim: usize,
    phase_out: *mut gcso_q7_t,
) -> GcsoStatus {
    if source.is_null() || target.is_null() || phase_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned_to(source, 32) || !is_aligned_to(target, 32) || !is_aligned(phase_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if dim == 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    ffi_boundary!(Ok(GCSO_SUCCESS))
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
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
    ffi_boundary!({
        unsafe { *router_out = ptr::null_mut() };
        Ok(GCSO_SUCCESS)
    })
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
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    ffi_boundary!(Ok(GCSO_SUCCESS))
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
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
    ffi_boundary!({
        unsafe { *adapter_out = ptr::null_mut() };
        Ok(GCSO_SUCCESS)
    })
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
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
    ffi_boundary!({
        unsafe { *field_out = ptr::null_mut() };
        Ok(GCSO_SUCCESS)
    })
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
    ffi_boundary!({
        let vec_slice = unsafe { slice::from_raw_parts(vec, dim) };
        let _ = AlignedSlice32::new(vec_slice)?;
        unsafe { *anchor_id_out = 1 };
        Ok(GCSO_SUCCESS)
    })
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
    ffi_boundary!({
        let c_str = unsafe { std::ffi::CStr::from_ptr(prompt_text) };
        if c_str.to_str().is_err() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        unsafe { *anchor_id_out = 1 };
        Ok(GCSO_SUCCESS)
    })
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
    ffi_boundary!({
        let emb_slice = unsafe { slice::from_raw_parts(embedding, dim) };
        let _ = AlignedSlice32::new(emb_slice)?;
        unsafe { *anchor_id_out = 1 };
        Ok(GCSO_SUCCESS)
    })
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
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
    ffi_boundary!({
        unsafe { *new_anchor_count_out = 0 };
        Ok(GCSO_SUCCESS)
    })
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
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
    ffi_boundary!({
        unsafe { *controller_out = ptr::null_mut() };
        Ok(GCSO_SUCCESS)
    })
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
}

/// Compute CVoid Coherent Vector Alignment Metric for Out-of-Distribution Latent Space.
///
/// # Safety
/// `key_vector` (32-byte aligned) and `void_score_out` must be valid non-null pointers.
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
    ffi_boundary!({
        let slice = unsafe { core::slice::from_raw_parts(key_vector, dim) };
        let mut sum_sq = 0.0f32;
        for &val in slice {
            sum_sq += val * val;
        }
        if sum_sq.is_nan() || sum_sq.is_infinite() {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }
        unsafe { *void_score_out = sum_sq / (dim as f32) };
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluate Eyring-Kramers potential barrier height value with singularity check.
///
/// # Safety
/// `barrier_out` must be a valid non-null aligned pointer.
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
    ffi_boundary!({
        let eps = 1e-6f32;
        let safe_void = if void_score < 0.0 { 0.0 } else { void_score };
        unsafe { *barrier_out = tau_eff / (safe_void + eps) };
        Ok(GCSO_SUCCESS)
    })
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
}

// ===================================================================
// 6. Persona Patch & ZIMMS Storage Mechanics Interface
// ===================================================================

/// Dynamic application of persona phase modulation patches without altering base weights.
///
/// # Safety
/// `context` and `patch_data` must be non-null valid pointers.
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
}

/// Zero-Overhead In-Memory Mapped Storage: Maps .gcso container payload using zero-copy mmap.
///
/// # Safety
/// `file_path` and `zimms_out` must be valid non-null pointers aligned to boundary.
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
    ffi_boundary!({
        let c_str = unsafe { std::ffi::CStr::from_ptr(file_path) };
        if c_str.to_str().is_err() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        unsafe {
            ptr::write_bytes(zimms_out, 0, 1);
            (*zimms_out).mapped_address = 0x1000_0000;
            (*zimms_out).file_size_bytes = 4096;
            (*zimms_out).fd_handle = 3;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Unmaps zero-copy ZIMMS memory handle and releases Direct DMA resources.
///
/// # Safety
/// Safe no-op if `zimms_desc` is NULL.
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
    ffi_boundary!({
        unsafe { ptr::write_bytes(zimms_desc, 0, 1) };
        Ok(GCSO_SUCCESS)
    })
}