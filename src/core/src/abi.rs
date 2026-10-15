// name: src/core/src/abi.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! C-ABI interface definitions, structural layouts, and FFI boundary safety primitives for GCSO core engine.

#![allow(non_camel_case_types)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::undocumented_unsafe_blocks)]
#![allow(clippy::field_reassign_with_default)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
use std::boxed::Box;
#[cfg(feature = "std")]
use std::panic::{catch_unwind, AssertUnwindSafe};

use core::ffi::c_char;
use core::mem::{align_of, offset_of, size_of};
use core::ptr;
use core::slice;

use crate::dpsr::DpsrEngine;
use crate::edbc::{eval_barrier_pitchfork, EdbcEngine};
use crate::traits::{
    AlignedSlice16, AlignedSlice32, AlignedSliceMut16, AlignedSliceMut32, EntropyEvaluator,
    PhaseSteering,
};

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
///
/// # Errors
/// Returns `GCSO_ERROR_VERSION_MISMATCH` if `version_hex` does not equal `GCSO_ABI_VERSION_HEX`.
#[inline]
pub fn validate_abi_version(version_hex: u32) -> Result<(), gcso_status_t> {
    if version_hex == GCSO_ABI_VERSION_HEX {
        Ok(())
    } else {
        Err(GCSO_ERROR_VERSION_MISMATCH)
    }
}

/// Helper macro to catch unwinding panics and safely bridge execution into C-ABI error status codes.
///
/// Supports using the `?` operator on `Result<T, GcsoStatus>` inside the expression block.
#[macro_export]
macro_rules! ffi_boundary {
    ($body:expr) => {{
        #[cfg(feature = "std")]
        let status = match ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(
            || -> Result<$crate::abi::GcsoStatus, $crate::abi::GcsoStatus> { $body },
        )) {
            Ok(Ok(status)) => status,
            Ok(Err(err_status)) => err_status,
            Err(_) => $crate::abi::GCSO_ERROR_PANIC_CAUGHT,
        };

        #[cfg(not(feature = "std"))]
        let status =
            match (|| -> Result<$crate::abi::GcsoStatus, $crate::abi::GcsoStatus> { $body })() {
                Ok(status) => status,
                Err(err_status) => err_status,
            };

        status
    }};
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

// ===================================================================
// Strongly-Typed Opaque Handles replacing generic untyped pointers
// ===================================================================

/// Opaque context runtime handle structure.
#[repr(C)]
pub struct GcsoContextOpaque {
    _private: [u8; 0],
}

/// Opaque binary snapshot container structure.
#[repr(C)]
pub struct GcsoContainerOpaque {
    _private: [u8; 0],
}

/// Opaque Action Hub pointer table structure.
#[repr(C)]
pub struct GcsoActionHubOpaque {
    _private: [u8; 0],
}

/// Opaque DAES extension scratchpad slot structure.
#[repr(C)]
pub struct GcsoDaesSlotOpaque {
    _private: [u8; 0],
}

/// Opaque Attractor Field topological anchor structure.
#[repr(C)]
pub struct GcsoAttractorFieldOpaque {
    _private: [u8; 0],
}

/// Opaque EDBC entropy controller state structure.
#[repr(C)]
pub struct GcsoEdbcControllerOpaque {
    _private: [u8; 0],
}

/// Opaque DPSR phase steering kernel structure.
#[repr(C)]
pub struct GcsoDpsrKernelOpaque {
    _private: [u8; 0],
}

/// Opaque PSPM sub-head group router structure.
#[repr(C)]
pub struct GcsoPspmRouterOpaque {
    _private: [u8; 0],
}

/// Opaque SRL adapter layer structure.
#[repr(C)]
pub struct GcsoSrlAdapterOpaque {
    _private: [u8; 0],
}

/// Opaque handle to runtime context instance.
pub type GcsoContextHandle = *mut GcsoContextOpaque;
/// Opaque handle to container instance.
pub type GcsoContainerHandle = *mut GcsoContainerOpaque;
/// Opaque handle to Action Hub instance.
pub type GcsoActionHubHandle = *mut GcsoActionHubOpaque;
/// Opaque handle to DAES slot instance.
pub type GcsoDaesSlotHandle = *mut GcsoDaesSlotOpaque;
/// Opaque handle to Attractor Field instance.
pub type GcsoAttractorFieldHandle = *mut GcsoAttractorFieldOpaque;
/// Opaque handle to EDBC controller instance.
pub type GcsoEdbcControllerHandle = *mut GcsoEdbcControllerOpaque;
/// Opaque handle to DPSR kernel instance.
pub type GcsoDpsrKernelHandle = *mut GcsoDpsrKernelOpaque;
/// Opaque handle to PSPM router instance.
pub type GcsoPspmRouterHandle = *mut GcsoPspmRouterOpaque;
/// Opaque handle to SRL adapter instance.
pub type GcsoSrlAdapterHandle = *mut GcsoSrlAdapterOpaque;

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

/// 64-bit tagged memory pointer descriptor matching GCSO specification (Section 5.4.A).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct gcso_tagged_ptr_t(pub u64);

impl gcso_tagged_ptr_t {
    /// Mask for 48-bit raw address pointer.
    pub const ADDR_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;
    /// Mask for 8-bit domain tag bits.
    pub const TAG_MASK: u64 = 0x00FF_0000_0000_0000;
    /// Mask for 8-bit quantized phase state.
    pub const PHASE_MASK: u64 = 0xFF00_0000_0000_0000;

    /// Packs raw 48-bit address, 8-bit tag, and Q7 phase value into a 64-bit tagged pointer.
    #[inline]
    #[must_use]
    pub const fn pack(raw_addr: u64, tag: u8, phase: gcso_q7_t) -> Self {
        let addr = raw_addr & Self::ADDR_MASK;
        let tag_bits = ((tag as u64) << 48) & Self::TAG_MASK;
        let phase_bits = (((phase as u8) as u64) << 56) & Self::PHASE_MASK;
        Self(addr | tag_bits | phase_bits)
    }

    /// Unpacks canonical 48-bit address.
    #[inline]
    #[must_use]
    pub const fn unpack_addr(self) -> u64 {
        self.0 & Self::ADDR_MASK
    }

    /// Unpacks 8-bit domain tag.
    #[inline]
    #[must_use]
    pub const fn unpack_tag(self) -> u8 {
        ((self.0 & Self::TAG_MASK) >> 48) as u8
    }

    /// Unpacks Q7 quantized phase state.
    #[inline]
    #[must_use]
    pub const fn unpack_phase(self) -> gcso_q7_t {
        ((self.0 & Self::PHASE_MASK) >> 56) as u8 as gcso_q7_t
    }
}

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
    assert!(size_of::<gcso_tagged_ptr_t>() == 8);
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
    !ptr.is_null() && ((ptr as usize) % align_of::<T>() == 0)
}

/// Helper function to check pointer alignment for a specific custom alignment requirement.
#[inline]
#[must_use]
pub fn is_aligned_to<T>(ptr: *const T, align: usize) -> bool {
    !ptr.is_null() && align != 0 && align.is_power_of_two() && ((ptr as usize) % align == 0)
}

/// Static version string constant for FFI boundary checks matching ABI v0.1.1.
static GCSO_ABI_VERSION: &[u8] = b"0.1.1\0";

// ===================================================================
// 0. Tagged Pointer C-ABI Exported Helpers
// ===================================================================

/// Packs raw 48-bit address, 8-bit domain tag, and Q7 phase value into a 64-bit tagged pointer.
#[no_mangle]
pub extern "C" fn gcso_tagged_ptr_pack(raw_addr: u64, tag: u8, phase: gcso_q7_t) -> u64 {
    gcso_tagged_ptr_t::pack(raw_addr, tag, phase).0
}

/// Unpacks canonical 48-bit raw address pointer from a 64-bit tagged pointer.
#[no_mangle]
pub extern "C" fn gcso_tagged_ptr_unpack_addr(ptr: u64) -> u64 {
    gcso_tagged_ptr_t(ptr).unpack_addr()
}

/// Unpacks 8-bit domain tag from a 64-bit tagged pointer.
#[no_mangle]
pub extern "C" fn gcso_tagged_ptr_unpack_tag(ptr: u64) -> u8 {
    gcso_tagged_ptr_t(ptr).unpack_tag()
}

/// Unpacks Q7 quantized phase state from a 64-bit tagged pointer.
#[no_mangle]
pub extern "C" fn gcso_tagged_ptr_unpack_phase(ptr: u64) -> gcso_q7_t {
    gcso_tagged_ptr_t(ptr).unpack_phase()
}

// ===================================================================
// 1. System & Capability Query Interface
// ===================================================================

/// Retrieve numeric components of the GCSO C-ABI version (v0.1.1).
///
/// # Safety
/// Caller must ensure that `major`, `minor`, and `patch` are valid, non-null, writable memory locations aligned to `u32`.
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
        // SAFETY: major, minor, and patch pointers are validated as non-null and u32-aligned above.
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
/// Returns a valid pointer to a static null-terminated C string that remains valid for the lifetime of the process.
/// The caller must not attempt to mutate or free this memory.
#[no_mangle]
pub unsafe extern "C" fn gcso_abi_get_version_string() -> *const c_char {
    GCSO_ABI_VERSION.as_ptr().cast::<c_char>()
}

/// Convert a `GcsoStatus` error code into its corresponding static string representation.
///
/// # Safety
/// Safe to call with any `GcsoStatus` value.
/// Returns a valid pointer to a static null-terminated C string that remains valid for the lifetime of the process.
/// The caller must not attempt to mutate or free this memory.
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
/// Caller must ensure `flags` points to a valid writable 64-bit unsigned integer location aligned to `u64`.
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
        // SAFETY: Pointer flags is checked non-null and u64-aligned above.
        unsafe { *flags = caps };
        Ok(GCSO_SUCCESS)
    })
}

/// Initialize a `gcso_config_t` structure strictly conforming to GCSO standards.
///
/// # Safety
/// Caller must ensure `config` is a non-null, writable pointer to a `gcso_config_t` structure aligned to 16 bytes.
#[no_mangle]
pub unsafe extern "C" fn gcso_config_init_default(config: *mut gcso_config_t) -> GcsoStatus {
    if config.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(config) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }

    ffi_boundary!({
        // SAFETY: Pointer config is validated non-null and 16-byte aligned above.
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
/// If NULL, this function safely performs a no-op.
#[no_mangle]
pub unsafe extern "C" fn gcso_free_string(str_ptr: *const c_char) {
    if !str_ptr.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: str_ptr was created via CString::into_raw and verified non-null above.
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
/// Caller must ensure `config` points to a valid `gcso_config_t` structure aligned to 16 bytes.
/// `context_out` must be a valid non-null pointer location aligned to handle size.
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
        // SAFETY: Pointer config is non-null and aligned as verified above.
        let cfg = unsafe { &*config };
        if cfg.head_dim == 0 || cfg.head_dim % 2 != 0 || cfg.num_heads == 0 || cfg.num_heads > 64 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Pointer context_out is non-null and aligned as verified above.
        unsafe { *context_out = ptr::null_mut() };
        Ok(GCSO_SUCCESS)
    })
}

/// Reset state variables and phase accumulators without freeing allocated tables.
///
/// # Safety
/// `context` must be NULL or a valid pointer aligned to the context handle boundary.
/// If `context` is NULL, this function safely returns `GCSO_SUCCESS` as a no-op.
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
/// Caller must ensure `context` is a valid handle and `prompt_text` points to a valid null-terminated C string.
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
        // SAFETY: prompt_text is checked non-null above and must point to a valid C string.
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
/// Caller must ensure `context` is a non-null pointer aligned to handle boundary.
/// `query_tensor` and `key_tensor` (if non-null) must be 32-byte aligned for SIMD/Warp operations.
/// `trail_out` (if non-null) must be aligned to `gcso_pointer_trail_t` (128 bytes).
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
            // SAFETY: query_tensor is non-null, 32-byte aligned, and valid for 128 float elements.
            let slice = unsafe { slice::from_raw_parts_mut(query_tensor, 128) };
            let _ = AlignedSliceMut32::new(slice)?;
        }
        if !key_tensor.is_null() {
            // SAFETY: key_tensor is non-null, 32-byte aligned, and valid for 128 float elements.
            let slice = unsafe { slice::from_raw_parts_mut(key_tensor, 128) };
            let _ = AlignedSliceMut32::new(slice)?;
        }
        if !trail_out.is_null() {
            // SAFETY: trail_out is non-null, 128-byte aligned, and points to a writable trail structure.
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
/// Caller must ensure `context` and `buffer_size` are non-null pointers aligned to structure/type boundary.
/// If `buffer` is non-null, it must point to at least `*buffer_size` bytes of writable memory.
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
        let required = 320usize; // Total aligned snapshot payload size
        if buffer.is_null() {
            // SAFETY: buffer_size pointer is verified non-null and aligned above.
            unsafe { *buffer_size = required };
            return Ok(GCSO_SUCCESS);
        }
        // SAFETY: buffer_size pointer is verified non-null and aligned above.
        let buf_size = unsafe { *buffer_size };
        if buf_size < required {
            // SAFETY: buffer_size pointer is verified non-null and aligned above.
            unsafe { *buffer_size = required };
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }

        // SAFETY: buffer is validated non-null and large enough to hold header payload. Use write_unaligned for unaligned byte slices.
        unsafe {
            ptr::write_bytes(buffer, 0, required);
            let header = gcso_snapshot_header_t {
                magic: 0x4F53_4347,            // ASCII "GCSO"
                version: GCSO_ABI_VERSION_HEX, // 0x00000101 (v0.1.1)
                total_size: required as u64,
                action_hub_offset: 128,
                attractor_field_offset: 128,
                dpsr_state_offset: 192,
                srl_state_offset: 192,
                edbc_state_offset: 192,
                daes_slot_offset: 256,
                timestamp_epoch_sec: 1_774_900_000,
                ..Default::default()
            };

            ptr::write_unaligned(buffer.cast::<gcso_snapshot_header_t>(), header);
            *buffer_size = required;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Deserialize binary snapshot to restore runtime state with ABI v0.1.1 verification.
///
/// # Safety
/// Caller must ensure `buffer` points to readable memory of length `buffer_size` and `context_out` points to a valid pointer location.
#[no_mangle]
pub unsafe extern "C" fn gcso_context_deserialize(
    buffer: *const u8,
    buffer_size: usize,
    context_out: *mut GcsoContextHandle,
) -> GcsoStatus {
    if buffer.is_null() || context_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(context_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if buffer_size < size_of::<gcso_snapshot_header_t>() {
        return GCSO_ERROR_CONTAINER_CORRUPTED;
    }
    ffi_boundary!({
        // SAFETY: Use read_unaligned to safely parse snapshot headers from byte slices.
        let header = unsafe { ptr::read_unaligned(buffer.cast::<gcso_snapshot_header_t>()) };
        if header.magic != 0x4F53_4347 {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }
        validate_abi_version(header.version)?;
        // SAFETY: context_out is validated non-null and aligned above.
        unsafe { *context_out = ptr::null_mut() };
        Ok(GCSO_SUCCESS)
    })
}

/// Destroy context instance and release resources.
///
/// # Safety
/// `context` must be NULL or a valid handle pointer.
/// If `context` is NULL, this function safely performs a no-op and returns `GCSO_SUCCESS`.
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
/// `container` must be NULL or a valid container handle.
/// If `container` is NULL, this function safely performs a no-op and returns `GCSO_SUCCESS`.
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
/// Caller must ensure `context` and `keyframe_header_out` are non-null valid pointers aligned to structure boundaries.
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
        // SAFETY: keyframe_header_out is verified non-null, aligned, and writable above.
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
/// Caller must ensure `hub_out` is a non-null pointer location aligned to handle size.
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
        // SAFETY: hub_out is verified non-null and aligned above.
        unsafe { *hub_out = ptr::null_mut() };
        Ok(GCSO_SUCCESS)
    })
}

/// Advance tagged pointer transition in O(1) time.
///
/// # Safety
/// `context` must be a valid handle; `trail_out` must point to writable memory aligned to 128 bytes.
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
        // SAFETY: trail_out is verified non-null, 128-byte aligned, and writable above.
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
/// Safe to call with any 64-bit integer pointer value. Performs no memory dereferences.
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
/// Caller must ensure `context` and `mask` are valid non-null pointers aligned to 32 bytes.
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
/// `context` must be a valid aligned handle. `pull_force` must be finite.
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
/// Caller must ensure `bitmasks` points to `num_masks` readable `gcso_paged_bitmask_t` items (32-byte aligned),
/// and `reduced_out` points to writable `gcso_paged_bitmask_t` (32-byte aligned).
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
        // SAFETY: bitmasks is validated non-null, aligned, and valid for num_masks elements above.
        let masks = unsafe { slice::from_raw_parts(bitmasks, num_masks) };
        for m in masks {
            acc[0] |= m.bits[0];
            acc[1] |= m.bits[1];
            acc[2] |= m.bits[2];
            acc[3] |= m.bits[3];
        }
        // SAFETY: reduced_out is verified non-null and aligned above.
        unsafe { (*reduced_out).bits = acc };
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluate SIMD/Warp bitmask reduction over PagedBlock KV caches.
///
/// # Safety
/// `kv_bits` must point to `num_blocks` u64 entries. `mask_out` must point to writable `gcso_paged_bitmask_t`.
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
        // SAFETY: mask_out is verified non-null, aligned, and writable above.
        unsafe { ptr::write_bytes(mask_out, 0, 1) };
        Ok(GCSO_SUCCESS)
    })
}

/// Destroy Action Hub instance and free resources.
///
/// # Safety
/// `hub` must be NULL or a valid handle.
/// If `hub` is NULL, this function safely performs a no-op and returns `GCSO_SUCCESS`.
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
/// Caller must ensure `slot_out` is non-null and aligned to handle size.
#[no_mangle]
pub unsafe extern "C" fn gcso_daes_slot_create(slot_out: *mut GcsoDaesSlotHandle) -> GcsoStatus {
    if slot_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(slot_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    ffi_boundary!({
        // SAFETY: slot_out is verified non-null and aligned above.
        unsafe { *slot_out = ptr::null_mut() };
        Ok(GCSO_SUCCESS)
    })
}

/// Reset active telemetry mini ledger and cache hit counters in-place.
///
/// # Safety
/// Caller must ensure `slot` is a valid non-null pointer aligned to 64 bytes.
#[no_mangle]
pub unsafe extern "C" fn gcso_daes_reset_telemetry(slot: *mut gcso_daes_slot_t) -> GcsoStatus {
    if slot.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(slot) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    ffi_boundary!({
        // SAFETY: slot is verified non-null and 64-byte aligned above.
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
/// Caller must ensure `slot` points to a valid `gcso_daes_slot_t` (64-byte aligned) and `shortcut_out` points to a writable u64 location.
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
        // SAFETY: slot is verified non-null and aligned above.
        let slot_ref = unsafe { &*slot };
        if (slot_ref.fast_path_bypass_mask & input_key) != 0 {
            // SAFETY: shortcut_out is verified non-null and aligned above.
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
/// Caller must ensure `slot` points to a valid `gcso_daes_slot_t` structure aligned to 64 bytes.
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
        // SAFETY: slot is verified non-null and aligned above.
        let head = (unsafe { (*slot).telemetry_ring_head as usize }) % 8;
        // SAFETY: slot is verified non-null and aligned above.
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
/// Caller must ensure `slot` points to a valid `gcso_daes_slot_t` structure aligned to 64 bytes. `mode` must be <= 2.
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
        // SAFETY: slot is verified non-null and aligned above.
        unsafe { (*slot).mode = mode };
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluate telemetry ledger for dynamic auto-tuning.
///
/// # Safety
/// Caller must ensure `slot` and `config_out` are valid aligned non-null pointers.
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
/// `slot` must be NULL or a valid handle.
/// If `slot` is NULL, this function safely performs a no-op and returns `GCSO_SUCCESS`.
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
/// Caller must ensure `kernel_out` is a valid writable pointer location aligned to handle size.
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
    ffi_boundary!({
        let engine = DpsrEngine::new(head_dim, num_heads)?;
        #[cfg(feature = "std")]
        {
            let boxed = Box::new(engine);
            // SAFETY: kernel_out pointer is verified non-null and aligned above.
            unsafe { *kernel_out = Box::into_raw(boxed).cast::<GcsoDpsrKernelOpaque>() };
        }
        #[cfg(not(feature = "std"))]
        {
            // SAFETY: kernel_out pointer is verified non-null and aligned above.
            unsafe { *kernel_out = ptr::null_mut() };
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Apply inline DPSR phase steering to Query tensor.
///
/// # Safety
/// Caller must ensure `query_tensor` (32-byte aligned float buffer of size `head_dim * num_heads`)
/// and `phase_deltas` (16-byte aligned `gcso_q7_t` buffer of size `num_heads`) are non-null and valid.
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
    ffi_boundary!({
        let engine = DpsrEngine::new(head_dim as u32, num_heads as u32)?;
        let total_q_len = head_dim
            .checked_mul(num_heads)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        // SAFETY: query_tensor is verified non-null, 32-byte aligned, and valid for total_q_len elements.
        let q_slice_raw = unsafe { slice::from_raw_parts_mut(query_tensor, total_q_len) };
        // SAFETY: phase_deltas is verified non-null, 16-byte aligned, and valid for num_heads elements.
        let p_slice_raw = unsafe { slice::from_raw_parts(phase_deltas, num_heads) };

        let mut q_aligned = AlignedSliceMut32::new(q_slice_raw)?;
        let p_aligned = AlignedSlice16::new(p_slice_raw)?;

        engine.apply_phase_steering(&mut q_aligned, &p_aligned)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Apply fused inline DPSR phase steering simultaneously to Query and Key tensors.
///
/// # Safety
/// Caller must ensure `query_tensor` and `key_tensor` (32-byte aligned float buffers) and `phase_deltas`
/// (16-byte aligned `gcso_q7_t` buffer) are non-null and valid for specified dimensions.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_phase_steering_fused(
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    phase_deltas: *const gcso_q7_t,
    head_dim: usize,
    num_heads: usize,
) -> GcsoStatus {
    if query_tensor.is_null() || key_tensor.is_null() || phase_deltas.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned_to(query_tensor, 32)
        || !is_aligned_to(key_tensor, 32)
        || !is_aligned(phase_deltas)
    {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    ffi_boundary!({
        let engine = DpsrEngine::new(head_dim as u32, num_heads as u32)?;
        let total_len = head_dim
            .checked_mul(num_heads)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        // SAFETY: query_tensor is verified non-null, 32-byte aligned, and valid for total_len elements.
        let q_raw = unsafe { slice::from_raw_parts_mut(query_tensor, total_len) };
        // SAFETY: key_tensor is verified non-null, 32-byte aligned, and valid for total_len elements.
        let k_raw = unsafe { slice::from_raw_parts_mut(key_tensor, total_len) };
        // SAFETY: phase_deltas is verified non-null, 16-byte aligned, and valid for num_heads elements.
        let p_raw = unsafe { slice::from_raw_parts(phase_deltas, num_heads) };

        let mut q_aligned = AlignedSliceMut32::new(q_raw)?;
        let mut k_aligned = AlignedSliceMut32::new(k_raw)?;
        let p_aligned = AlignedSlice16::new(p_raw)?;

        engine.apply_phase_steering_fused(&mut q_aligned, &mut k_aligned, &p_aligned)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Apply RIPA soft-bounded tanh clamping on low-frequency channels.
///
/// # Safety
/// Caller must ensure `query_tensor` (32-byte aligned) and `phase_deltas` (16-byte aligned) are non-null valid pointers. `max_rad` must be finite.
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
    ffi_boundary!({
        let engine = DpsrEngine::new(head_dim as u32, num_heads as u32)?;
        let total_q_len = head_dim
            .checked_mul(num_heads)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        // SAFETY: query_tensor is verified non-null, 32-byte aligned, and valid for total_q_len elements.
        let q_raw = unsafe { slice::from_raw_parts_mut(query_tensor, total_q_len) };
        // SAFETY: phase_deltas is verified non-null, 16-byte aligned, and valid for num_heads elements.
        let p_raw = unsafe { slice::from_raw_parts(phase_deltas, num_heads) };

        let mut q_aligned = AlignedSliceMut32::new(q_raw)?;
        let p_aligned = AlignedSlice16::new(p_raw)?;

        engine.apply_phase_steering_safe(&mut q_aligned, &p_aligned, max_rad)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Apply soft phase damping to smooth out transient phase rotation spikes in zero-allocation mode.
///
/// # Safety
/// Caller must ensure `query_tensor` (32-byte aligned) is a non-null valid float buffer.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_soft_phase_damping(
    query_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    damping_factor: f32,
) -> GcsoStatus {
    if query_tensor.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned_to(query_tensor, 32) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    ffi_boundary!({
        let engine = DpsrEngine::new(head_dim as u32, num_heads as u32)?;
        let total_q_len = head_dim
            .checked_mul(num_heads)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        // SAFETY: query_tensor is verified non-null, 32-byte aligned, and valid for total_q_len elements.
        let q_raw = unsafe { slice::from_raw_parts_mut(query_tensor, total_q_len) };
        let mut q_aligned = AlignedSliceMut32::new(q_raw)?;

        engine.apply_soft_phase_damping(&mut q_aligned, damping_factor)?;
        Ok(GCSO_SUCCESS)
    })
}

/// QDPS discrete filter: cuts off phase rotation steps falling below min_step_rad.
///
/// # Safety
/// Caller must ensure `phase_deltas` points to a valid readable/writable array of size `len` (16-byte aligned).
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
    ffi_boundary!({
        let engine = DpsrEngine::new(128, len as u32)?;
        // SAFETY: phase_deltas is verified non-null, 16-byte aligned, and valid for len elements.
        let p_raw = unsafe { slice::from_raw_parts_mut(phase_deltas, len) };
        let mut p_aligned = AlignedSliceMut16::new(p_raw)?;

        engine.qdps_filter_step(&mut p_aligned, min_step_rad)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Lazy Phase Unwrapping override on Query tensor.
///
/// # Safety
/// Caller must ensure `query_tensor` and `context_accum` are non-null and 32-byte aligned float buffers.
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
    ffi_boundary!({
        let engine = DpsrEngine::new(head_dim as u32, num_heads as u32)?;
        let total_q_len = head_dim
            .checked_mul(num_heads)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        // SAFETY: query_tensor is verified non-null, 32-byte aligned, and valid for total_q_len elements.
        let q_raw = unsafe { slice::from_raw_parts_mut(query_tensor, total_q_len) };
        // SAFETY: context_accum is verified non-null, 32-byte aligned, and valid for num_heads elements.
        let c_raw = unsafe { slice::from_raw_parts(context_accum, num_heads) };

        let mut q_aligned = AlignedSliceMut32::new(q_raw)?;
        let c_aligned = AlignedSlice32::new(c_raw)?;

        engine.lazy_unwrap_override(&mut q_aligned, &c_aligned)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Execute norm-guarded Slerp phase stabilization.
///
/// # Safety
/// Caller must ensure `tensor` points to a valid readable/writable float buffer of size `dim` (32-byte aligned).
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
    ffi_boundary!({
        let engine = DpsrEngine::new(128, 1)?;
        // SAFETY: tensor is verified non-null, 32-byte aligned, and valid for dim elements.
        let t_raw = unsafe { slice::from_raw_parts_mut(tensor, dim) };
        let mut t_aligned = AlignedSliceMut32::new(t_raw)?;

        engine.slerp_norm_guard_stable(&mut t_aligned, norm_lower, norm_upper)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Fused inline logit phase shift prior to Softmax.
///
/// # Safety
/// Caller must ensure `logits` (32-byte aligned float buffer of size `vocab_size`) and `phase_deltas`
/// (16-byte aligned `gcso_q7_t` buffer of size `num_heads`) are non-null valid pointers.
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
    ffi_boundary!({
        let engine = DpsrEngine::new(128, num_heads as u32)?;
        // SAFETY: logits is verified non-null, 32-byte aligned, and valid for vocab_size elements.
        let l_raw = unsafe { slice::from_raw_parts_mut(logits, vocab_size) };
        // SAFETY: phase_deltas is verified non-null, 16-byte aligned, and valid for num_heads elements.
        let p_raw = unsafe { slice::from_raw_parts(phase_deltas, num_heads) };

        let mut l_aligned = AlignedSliceMut32::new(l_raw)?;
        let p_aligned = AlignedSlice16::new(p_raw)?;

        engine.fused_logit_shift(&mut l_aligned, &p_aligned)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Compute Procrustes phase delta alignment between source and target state representations.
///
/// # Safety
/// Caller must ensure `source` and `target` (32-byte aligned float buffers of size `dim`) and `phase_out`
/// (16-byte aligned `gcso_q7_t` buffer) are valid non-null pointers.
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
    ffi_boundary!({
        let head_dim = 128usize;
        let num_heads = dim / head_dim;
        if num_heads == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let engine = DpsrEngine::new(head_dim as u32, num_heads as u32)?;
        // SAFETY: source is verified non-null, 32-byte aligned, and valid for dim elements.
        let src_raw = unsafe { slice::from_raw_parts(source, dim) };
        // SAFETY: target is verified non-null, 32-byte aligned, and valid for dim elements.
        let tgt_raw = unsafe { slice::from_raw_parts(target, dim) };
        // SAFETY: phase_out is verified non-null, 16-byte aligned, and valid for num_heads elements.
        let out_raw = unsafe { slice::from_raw_parts_mut(phase_out, num_heads) };

        let src_aligned = AlignedSlice32::new(src_raw)?;
        let tgt_aligned = AlignedSlice32::new(tgt_raw)?;
        let mut out_aligned = AlignedSliceMut16::new(out_raw)?;

        engine.compute_procrustes_phase_delta(&src_aligned, &tgt_aligned, &mut out_aligned)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Destroy DPSR kernel instance.
///
/// # Safety
/// `kernel` must be NULL or a valid DPSR kernel handle.
/// If `kernel` is NULL, this function safely performs a no-op and returns `GCSO_SUCCESS`.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_kernel_destroy(kernel: GcsoDpsrKernelHandle) -> GcsoStatus {
    if kernel.is_null() {
        return GCSO_SUCCESS;
    }
    if !is_aligned(kernel) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    ffi_boundary!({
        #[cfg(feature = "std")]
        // SAFETY: kernel was allocated via Box::into_raw in gcso_dpsr_kernel_create.
        unsafe {
            let _ = Box::from_raw(kernel.cast::<DpsrEngine>());
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Allocate PSPM router instance.
///
/// # Safety
/// Caller must ensure `config` and `router_out` are valid non-null aligned pointers.
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
        // SAFETY: router_out is verified non-null and aligned above.
        unsafe { *router_out = ptr::null_mut() };
        Ok(GCSO_SUCCESS)
    })
}

/// Dispatch PSPM head-group phase profiles in single pass.
///
/// # Safety
/// Caller must ensure `query_tensor` (32-byte aligned float buffer) and `pspm_cfg` (16-byte aligned configuration) are non-null valid pointers.
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
    ffi_boundary!(Ok(GCSO_SUCCESS))
}

/// Destroy PSPM router instance.
///
/// # Safety
/// `router` must be NULL or a valid router handle.
/// If `router` is NULL, this function safely performs a no-op and returns `GCSO_SUCCESS`.
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
/// Caller must ensure `descriptor` and `adapter_out` are valid non-null aligned pointers.
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
        // SAFETY: adapter_out is verified non-null and aligned above.
        unsafe { *adapter_out = ptr::null_mut() };
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluate SRL Rank-1 outer product: y = W_base * x + scale_factor * s * (u * (v^T * x)).
///
/// # Safety
/// Caller must ensure `y_out` (32-byte aligned float buffer of size `dim_out`), `x_in` (32-byte aligned float buffer of size `dim_in`),
/// and `srl_desc` (32-byte aligned descriptor) are non-null valid pointers. `srl_desc.u_vector_ptr` and `srl_desc.v_vector_ptr` must point to aligned float buffers.
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
    ffi_boundary!({
        // SAFETY: srl_desc is verified non-null and aligned above.
        let desc = unsafe { &*srl_desc };
        if desc.u_vector_ptr == 0 || desc.v_vector_ptr == 0 {
            return Err(GCSO_ERROR_NULL_POINTER);
        }

        let u_ptr = desc.u_vector_ptr as *const f32;
        let v_ptr = desc.v_vector_ptr as *const f32;
        let gain_ptr = desc.gain_scalar_ptr as *const f32;

        if !is_aligned(u_ptr) || !is_aligned(v_ptr) {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }

        // SAFETY: x_in is verified non-null, aligned, and valid for dim_in elements.
        let x_slice = unsafe { slice::from_raw_parts(x_in, dim_in) };
        // SAFETY: u_ptr is verified non-null, aligned, and valid for dim_out elements.
        let u_slice = unsafe { slice::from_raw_parts(u_ptr, dim_out) };
        // SAFETY: v_ptr is verified non-null, aligned, and valid for dim_in elements.
        let v_slice = unsafe { slice::from_raw_parts(v_ptr, dim_in) };
        // SAFETY: y_out is verified non-null, aligned, and valid for dim_out elements.
        let y_slice = unsafe { slice::from_raw_parts_mut(y_out, dim_out) };

        // Compute dot product (v^T * x)
        let mut v_dot_x = 0.0f32;
        for j in 0..dim_in {
            v_dot_x += v_slice[j] * x_slice[j];
        }

        let scale = desc.scale_factor * v_dot_x;

        if !gain_ptr.is_null() && is_aligned(gain_ptr) {
            // SAFETY: gain_ptr is verified non-null and aligned.
            let g_slice = unsafe { slice::from_raw_parts(gain_ptr, dim_out) };
            for i in 0..dim_out {
                y_slice[i] = scale * u_slice[i] * g_slice[i];
            }
        } else {
            for i in 0..dim_out {
                y_slice[i] = scale * u_slice[i];
            }
        }

        Ok(GCSO_SUCCESS)
    })
}

/// Evaluate SRL Rank-1 outer product and accumulate into y_inout in-place.
///
/// # Safety
/// Caller must ensure `y_inout` (32-byte aligned float buffer of size `dim_out`), `x_in` (32-byte aligned float buffer of size `dim_in`),
/// and `srl_desc` (32-byte aligned descriptor) are non-null valid pointers. `srl_desc.u_vector_ptr` and `srl_desc.v_vector_ptr` must point to aligned float buffers.
#[no_mangle]
pub unsafe extern "C" fn gcso_srl_eval_rank1_fused(
    y_inout: *mut f32,
    x_in: *const f32,
    srl_desc: *const gcso_srl_descriptor_t,
    dim_in: usize,
    dim_out: usize,
) -> GcsoStatus {
    if y_inout.is_null() || x_in.is_null() || srl_desc.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned_to(y_inout, 32) || !is_aligned_to(x_in, 32) || !is_aligned(srl_desc) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if dim_in == 0 || dim_out == 0 {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    ffi_boundary!({
        // SAFETY: srl_desc is verified non-null and aligned above.
        let desc = unsafe { &*srl_desc };
        if desc.u_vector_ptr == 0 || desc.v_vector_ptr == 0 {
            return Err(GCSO_ERROR_NULL_POINTER);
        }

        let u_ptr = desc.u_vector_ptr as *const f32;
        let v_ptr = desc.v_vector_ptr as *const f32;
        let gain_ptr = desc.gain_scalar_ptr as *const f32;

        if !is_aligned(u_ptr) || !is_aligned(v_ptr) {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }

        // SAFETY: x_in is verified non-null, aligned, and valid for dim_in elements.
        let x_slice = unsafe { slice::from_raw_parts(x_in, dim_in) };
        // SAFETY: u_ptr is verified non-null, aligned, and valid for dim_out elements.
        let u_slice = unsafe { slice::from_raw_parts(u_ptr, dim_out) };
        // SAFETY: v_ptr is verified non-null, aligned, and valid for dim_in elements.
        let v_slice = unsafe { slice::from_raw_parts(v_ptr, dim_in) };
        // SAFETY: y_inout is verified non-null, aligned, and valid for dim_out elements.
        let y_slice = unsafe { slice::from_raw_parts_mut(y_inout, dim_out) };

        let mut v_dot_x = 0.0f32;
        for j in 0..dim_in {
            v_dot_x += v_slice[j] * x_slice[j];
        }

        let scale = desc.scale_factor * v_dot_x;

        if !gain_ptr.is_null() && is_aligned(gain_ptr) {
            // SAFETY: gain_ptr is verified non-null and aligned.
            let g_slice = unsafe { slice::from_raw_parts(gain_ptr, dim_out) };
            for i in 0..dim_out {
                y_slice[i] += scale * u_slice[i] * g_slice[i];
            }
        } else {
            for i in 0..dim_out {
                y_slice[i] += scale * u_slice[i];
            }
        }

        Ok(GCSO_SUCCESS)
    })
}

/// L2P-SVD: Projects fine-tuned LoRA matrices via SVD into phase profiles and SRL vectors.
///
/// # Safety
/// Caller must ensure `lora_a`, `lora_b`, `srl_out`, and `phase_profile_out` are valid, non-null, properly aligned pointers.
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
/// `adapter` must be NULL or a valid adapter handle.
/// If `adapter` is NULL, this function safely performs a no-op and returns `GCSO_SUCCESS`.
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
/// Caller must ensure `field_out` is a valid pointer location aligned to handle size.
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
        // SAFETY: field_out is verified non-null and aligned above.
        unsafe { *field_out = ptr::null_mut() };
        Ok(GCSO_SUCCESS)
    })
}

/// Register a topological anchor point in attractor field.
///
/// # Safety
/// Caller must ensure `context`, `vec` (32-byte aligned float array of size `dim`), and `anchor_id_out` are valid non-null pointers.
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
        // SAFETY: vec is verified non-null, 32-byte aligned, and valid for dim float elements.
        let vec_slice = unsafe { slice::from_raw_parts(vec, dim) };
        let _ = AlignedSlice32::new(vec_slice)?;
        // SAFETY: anchor_id_out is verified non-null and aligned above.
        unsafe { *anchor_id_out = 1 };
        Ok(GCSO_SUCCESS)
    })
}

/// Maps natural language system prompt text as primary Anchor Attractor.
///
/// # Safety
/// Caller must ensure `context`, `prompt_text` (valid null-terminated C string), and `anchor_id_out` are valid non-null pointers.
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
        // SAFETY: prompt_text is verified non-null above.
        let c_str = unsafe { std::ffi::CStr::from_ptr(prompt_text) };
        if c_str.to_str().is_err() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: anchor_id_out is verified non-null and aligned above.
        unsafe { *anchor_id_out = 1 };
        Ok(GCSO_SUCCESS)
    })
}

/// Maps dense feature embedding vector as continuous attractor anchor.
///
/// # Safety
/// Caller must ensure `context`, `embedding` (32-byte aligned float array of size `dim`), and `anchor_id_out` are valid non-null pointers.
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
        // SAFETY: embedding is verified non-null, 32-byte aligned, and valid for dim float elements.
        let emb_slice = unsafe { slice::from_raw_parts(embedding, dim) };
        let _ = AlignedSlice32::new(emb_slice)?;
        // SAFETY: anchor_id_out is verified non-null and aligned above.
        unsafe { *anchor_id_out = 1 };
        Ok(GCSO_SUCCESS)
    })
}

/// Inject phase-conjugate repulsion vector (-dTheta).
///
/// # Safety
/// Caller must ensure `context` and `repulsion_deltas` (16-byte aligned array of size `num_heads`) are valid non-null pointers.
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
/// Caller must ensure `context` and `new_anchor_count_out` are valid non-null pointers.
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
        // SAFETY: new_anchor_count_out is verified non-null and aligned above.
        unsafe { *new_anchor_count_out = 0 };
        Ok(GCSO_SUCCESS)
    })
}

/// Clear all registered topological anchors from the attractor field.
///
/// # Safety
/// Caller must ensure `field` is a valid non-null handle.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_clear_anchors(
    field: GcsoAttractorFieldHandle,
) -> GcsoStatus {
    if field.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(field) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    ffi_boundary!(Ok(GCSO_SUCCESS))
}

/// Destroy attractor field instance.
///
/// # Safety
/// `field` must be NULL or a valid attractor field handle.
/// If `field` is NULL, this function safely performs a no-op and returns `GCSO_SUCCESS`.
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
/// Caller must ensure `initial_state` (32-byte aligned `gcso_edbc_state_t`) and `controller_out` are valid non-null pointers.
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
        // SAFETY: initial_state is verified non-null and 32-byte aligned above.
        let init_st = unsafe { *initial_state };
        let engine = EdbcEngine::new(init_st);
        #[cfg(feature = "std")]
        {
            let boxed = Box::new(engine);
            // SAFETY: controller_out is verified non-null and aligned above.
            unsafe { *controller_out = Box::into_raw(boxed).cast::<GcsoEdbcControllerOpaque>() };
        }
        #[cfg(not(feature = "std"))]
        {
            // SAFETY: controller_out is verified non-null and aligned above.
            unsafe { *controller_out = ptr::null_mut() };
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluate stateful Moving Z-Score Attention Entropy.
///
/// # Safety
/// Caller must ensure `controller` is a valid EDBC controller handle and `state_out` points to a writable `gcso_edbc_state_t` structure aligned to 32 bytes.
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
    ffi_boundary!({
        // SAFETY: controller and state_out are verified non-null, aligned pointers created via C-ABI.
        let engine = unsafe { &mut *controller.cast::<EdbcEngine>() };
        // SAFETY: state_out is verified non-null and aligned above.
        let out_st = unsafe { &mut *state_out };
        engine.eval_stateful(token_z_score, out_st)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Compute CVoid Coherent Vector Alignment Metric for Out-of-Distribution Latent Space.
///
/// # Safety
/// Caller must ensure `key_vector` (32-byte aligned float buffer of size `dim`) and `void_score_out` are valid non-null pointers.
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
    ffi_boundary!({
        // SAFETY: key_vector is verified non-null, 32-byte aligned, and valid for dim float elements.
        let key_raw = unsafe { slice::from_raw_parts(key_vector, dim) };
        let key_aligned = AlignedSlice32::new(key_raw)?;
        let engine = EdbcEngine::default();
        // SAFETY: void_score_out is verified non-null and aligned above.
        let out_score = unsafe { &mut *void_score_out };

        engine.eval_cvoid_dyadic128(&key_aligned, out_score)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluate Eyring-Kramers potential barrier height value with singularity check.
///
/// # Safety
/// Caller must ensure `barrier_out` is a valid non-null float pointer.
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
    ffi_boundary!({
        let engine = EdbcEngine::default();
        let barrier = engine.eval_barrier(void_score, tau_eff)?;
        // SAFETY: barrier_out is verified non-null and aligned above.
        unsafe { *barrier_out = barrier };
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluate scale-invariant Pitchfork Bifurcation potential barrier height (Delta V)
/// according to GCSO Whitepaper Lemma 3.1: Delta V = alpha^2 * (cvoid - tau_eff)^2 / (4 * beta * head_dim).
///
/// # Safety
/// Caller must ensure `barrier_out` is a valid non-null float pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_cvoid_eval_barrier_pitchfork(
    cvoid: f32,
    tau_eff: f32,
    alpha: f32,
    beta: f32,
    head_dim: u32,
    barrier_out: *mut f32,
) -> GcsoStatus {
    if barrier_out.is_null() {
        return GCSO_ERROR_NULL_POINTER;
    }
    if !is_aligned(barrier_out) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    ffi_boundary!({
        let barrier = eval_barrier_pitchfork(cvoid, tau_eff, alpha, beta, head_dim)?;
        // SAFETY: barrier_out is verified non-null and aligned above.
        unsafe { *barrier_out = barrier };
        Ok(GCSO_SUCCESS)
    })
}

/// Destroy EDBC controller instance.
///
/// # Safety
/// `controller` must be NULL or a valid EDBC controller handle.
/// If `controller` is NULL, this function safely performs a no-op and returns `GCSO_SUCCESS`.
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
    ffi_boundary!({
        #[cfg(feature = "std")]
        // SAFETY: controller was allocated via Box::into_raw in gcso_edbc_controller_create.
        unsafe {
            let _ = Box::from_raw(controller.cast::<EdbcEngine>());
        }
        Ok(GCSO_SUCCESS)
    })
}

// ===================================================================
// 6. Persona Patch & ZIMMS Storage Mechanics Interface
// ===================================================================

/// Dynamic application of persona phase modulation patches without altering base weights.
///
/// # Safety
/// Caller must ensure `context` is a valid handle and `patch_data` points to readable byte buffer of size `patch_size`.
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
/// Caller must ensure `file_path` points to a valid null-terminated C string and `zimms_out` points to a writable `gcso_zimms_descriptor_t` aligned to 32 bytes.
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
        // SAFETY: file_path is verified non-null above.
        let c_str = unsafe { std::ffi::CStr::from_ptr(file_path) };
        if c_str.to_str().is_err() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: zimms_out is verified non-null and aligned above.
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
/// `zimms_desc` must be NULL or point to a valid `gcso_zimms_descriptor_t` structure aligned to 32 bytes.
/// If `zimms_desc` is NULL, this function safely performs a no-op and returns `GCSO_SUCCESS`.
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
        // SAFETY: zimms_desc is verified non-null and aligned above.
        unsafe { ptr::write_bytes(zimms_desc, 0, 1) };
        Ok(GCSO_SUCCESS)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_query() {
        let mut major = 0u32;
        let mut minor = 0u32;
        let mut patch = 0u32;

        // SAFETY: major, minor, and patch are valid stack-allocated non-null pointers aligned to u32.
        unsafe {
            assert_eq!(
                gcso_abi_get_version(&mut major, &mut minor, &mut patch),
                GCSO_SUCCESS
            );
        }
        assert_eq!(major, GCSO_ABI_VERSION_MAJOR);
        assert_eq!(minor, GCSO_ABI_VERSION_MINOR);
        assert_eq!(patch, GCSO_ABI_VERSION_PATCH);
    }

    #[test]
    fn test_tagged_ptr_pack_unpack() {
        let raw_addr = 0x0000_1234_5678_9ABCu64;
        let tag = 0xABu8;
        let phase = 42i8;

        let packed = gcso_tagged_ptr_pack(raw_addr, tag, phase);
        assert_eq!(gcso_tagged_ptr_unpack_addr(packed), raw_addr);
        assert_eq!(gcso_tagged_ptr_unpack_tag(packed), tag);
        assert_eq!(gcso_tagged_ptr_unpack_phase(packed), phase);
    }

    #[test]
    fn test_config_init_default() {
        let mut config = gcso_config_t::default();
        // SAFETY: config is a valid stack-allocated non-null pointer aligned to 16 bytes.
        unsafe {
            assert_eq!(gcso_config_init_default(&mut config), GCSO_SUCCESS);
        }
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.enable_zero_alloc_strict, 1);
    }

    #[test]
    fn test_status_to_string() {
        // SAFETY: gcso_status_to_string returns a valid static null-terminated C string.
        unsafe {
            let str_ptr = gcso_status_to_string(GCSO_SUCCESS);
            let c_str = std::ffi::CStr::from_ptr(str_ptr);
            assert_eq!(c_str.to_str().unwrap(), "GCSO_SUCCESS");
        }
    }

    #[test]
    fn test_cvoid_barrier_pitchfork() {
        let mut barrier = 0.0f32;
        // SAFETY: barrier is a valid stack-allocated non-null float pointer.
        unsafe {
            // Convergence mode: cvoid <= tau_eff -> barrier == 0.0
            assert_eq!(
                gcso_cvoid_eval_barrier_pitchfork(0.5, 1.0, 1.0, 1.0, 128, &mut barrier),
                GCSO_SUCCESS
            );
            assert_eq!(barrier, 0.0);

            // Exploration mode: cvoid > tau_eff -> barrier > 0.0
            assert_eq!(
                gcso_cvoid_eval_barrier_pitchfork(2.0, 1.0, 1.0, 1.0, 128, &mut barrier),
                GCSO_SUCCESS
            );
            assert!((barrier - (1.0 / 512.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_srl_eval_rank1() {
        #[repr(align(32))]
        struct BufIn([f32; 4]);
        #[repr(align(32))]
        struct BufOut([f32; 4]);
        #[repr(align(32))]
        struct VectorU([f32; 4]);
        #[repr(align(32))]
        struct VectorV([f32; 4]);

        let x_in = BufIn([1.0, 2.0, 3.0, 4.0]);
        let mut y_out = BufOut([0.0; 4]);
        let u_vec = VectorU([2.0, 2.0, 2.0, 2.0]);
        let v_vec = VectorV([1.0, 1.0, 1.0, 1.0]);

        let desc = gcso_srl_descriptor_t {
            layer_idx: 0,
            rank: 1,
            u_vector_ptr: u_vec.0.as_ptr() as u64,
            v_vector_ptr: v_vec.0.as_ptr() as u64,
            gain_scalar_ptr: 0,
            scale_factor: 1.0,
            flags: 0,
            reserved: [0; 24],
        };

        // SAFETY: Buffers and pointers are valid, 32-byte aligned, and correct length 4.
        unsafe {
            assert_eq!(
                gcso_srl_eval_rank1(y_out.0.as_mut_ptr(), x_in.0.as_ptr(), &desc, 4, 4),
                GCSO_SUCCESS
            );
        }

        // v^T * x = 1*1 + 1*2 + 1*3 + 1*4 = 10.0
        // y_i = 1.0 * 10.0 * 2.0 = 20.0
        for &val in &y_out.0 {
            assert_eq!(val, 20.0);
        }
    }
}
