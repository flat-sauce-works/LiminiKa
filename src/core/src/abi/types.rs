// name: src/core/src/abi/types.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! C-ABI v0.1.1 constants, status codes, opaque handles, and #[repr(C)] POD structures.

use super::*;

/// Unified GCSO C-ABI Major Version Component (`0.1.1`).
pub const GCSO_ABI_VERSION_MAJOR: u32 = 0;
/// Unified GCSO C-ABI Minor Version Component (`0.1.1`).
pub const GCSO_ABI_VERSION_MINOR: u32 = 1;
/// Unified GCSO C-ABI Patch Version Component (`0.1.1`).
pub const GCSO_ABI_VERSION_PATCH: u32 = 1;
/// Hexadecimal Representation of Unified ABI Version `0.1.1` (`0x00000101`).
pub const GCSO_ABI_VERSION_HEX: u32 = 0x0000_0101;
/// String Representation of Unified ABI Version `0.1.1`.
pub const GCSO_ABI_VERSION_STRING: &str = "0.1.1";

/// Capability bitmask flag indicating baseline CPU execution support.
pub const GCSO_CAP_CPU_BASELINE: gcso_capability_flags_t = 1 << 0;
/// Capability bitmask flag indicating NVIDIA CUDA kernel backend support.
pub const GCSO_CAP_CUDA_BACKEND: gcso_capability_flags_t = 1 << 1;
/// Capability bitmask flag indicating Vulkan Compute subgroup backend support.
pub const GCSO_CAP_VULKAN_BACKEND: gcso_capability_flags_t = 1 << 2;
/// Capability bitmask flag indicating Apple Metal simdgroup backend support.
pub const GCSO_CAP_METAL_BACKEND: gcso_capability_flags_t = 1 << 3;

/// Validates a C-ABI snapshot or descriptor version against the core version invariant (`0x00000101`).
///
/// # Errors
/// Returns [`GCSO_ERROR_VERSION_MISMATCH`] if `version_hex != GCSO_ABI_VERSION_HEX`.
#[inline(always)]
pub const fn validate_abi_version(version_hex: u32) -> Result<(), gcso_status_t> {
    if version_hex == GCSO_ABI_VERSION_HEX {
        Ok(())
    } else {
        Err(GCSO_ERROR_VERSION_MISMATCH)
    }
}

macro_rules! define_status_codes {
    ($(($name:ident, $code:expr, $c_bytes:literal, $str_val:literal, $doc:expr)),* $(,)?) => {
        $(
            #[doc = $doc]
            pub const $name: GcsoStatus = GcsoStatus($code);
        )*

        impl GcsoStatus {
            /// Returns `true` if the status equals [`GCSO_SUCCESS`] (`0`).
            #[inline(always)]
            #[must_use]
            pub const fn is_ok(self) -> bool {
                self.0 == 0
            }

            /// Returns `true` if the status represents an error (`!= 0`).
            #[inline(always)]
            #[must_use]
            pub const fn is_err(self) -> bool {
                self.0 != 0
            }

            /// Returns the raw signed 32-bit integer status code.
            #[inline(always)]
            #[must_use]
            pub const fn code(self) -> i32 {
                self.0
            }

            /// Returns the static null-terminated byte slice representation of this status code.
            #[inline(always)]
            #[must_use]
            pub const fn as_c_bytes(self) -> &'static [u8] {
                match self {
                    $($name => $c_bytes,)*
                    _ => b"GCSO_ERROR_UNKNOWN\0",
                }
            }

            /// Returns the static ASCII string representation of this status code (excluding trailing NUL).
            #[inline(always)]
            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self {
                    $($name => $str_val,)*
                    _ => "GCSO_ERROR_UNKNOWN",
                }
            }
        }
    };
}

define_status_codes!(
    (GCSO_SUCCESS, 0, b"GCSO_SUCCESS\0", "GCSO_SUCCESS", "Operation completed successfully."),
    (GCSO_ERROR_INVALID_ARGUMENT, -1, b"GCSO_ERROR_INVALID_ARGUMENT\0", "GCSO_ERROR_INVALID_ARGUMENT", "Invalid argument or out-of-bounds parameter passed to C-ABI function."),
    (GCSO_ERROR_OUT_OF_MEMORY, -2, b"GCSO_ERROR_OUT_OF_MEMORY\0", "GCSO_ERROR_OUT_OF_MEMORY", "Memory allocation failure or capacity limit reached during Cold Path creation."),
    (GCSO_ERROR_BUFFER_TOO_SMALL, -3, b"GCSO_ERROR_BUFFER_TOO_SMALL\0", "GCSO_ERROR_BUFFER_TOO_SMALL", "Destination buffer capacity is insufficient for requested payload."),
    (GCSO_ERROR_PANIC_CAUGHT, -4, b"GCSO_ERROR_PANIC_CAUGHT\0", "GCSO_ERROR_PANIC_CAUGHT", "Unhandled internal Rust panic or C++ exception caught at FFI boundary."),
    (GCSO_ERROR_NULL_POINTER, -5, b"GCSO_ERROR_NULL_POINTER\0", "GCSO_ERROR_NULL_POINTER", "Unexpected `NULL` pointer supplied for a mandatory parameter."),
    (GCSO_ERROR_INVALID_STATE, -6, b"GCSO_ERROR_INVALID_STATE\0", "GCSO_ERROR_INVALID_STATE", "Context or component is in an invalid lifecycle state for requested operation."),
    (GCSO_ERROR_VERSION_MISMATCH, -7, b"GCSO_ERROR_VERSION_MISMATCH\0", "GCSO_ERROR_VERSION_MISMATCH", "Binary snapshot or descriptor ABI version mismatch detected."),
    (GCSO_ERROR_MISALIGNED_POINTER, -8, b"GCSO_ERROR_MISALIGNED_POINTER\0", "GCSO_ERROR_MISALIGNED_POINTER", "Pointer passed across FFI boundary violates required SIMD or cacheline alignment."),
    (GCSO_ERROR_IO_FAILURE, -9, b"GCSO_ERROR_IO_FAILURE\0", "GCSO_ERROR_IO_FAILURE", "Input/output or storage persistence operation failure."),
    (GCSO_ERROR_ACTION_HUB_FULL, -10, b"GCSO_ERROR_ACTION_HUB_FULL\0", "GCSO_ERROR_ACTION_HUB_FULL", "Action Hub pointer table slot capacity fully saturated."),
    (GCSO_ERROR_NOT_IMPLEMENTED, -11, b"GCSO_ERROR_NOT_IMPLEMENTED\0", "GCSO_ERROR_NOT_IMPLEMENTED", "Feature or compute backend functionality not implemented in current build."),
    (GCSO_ERROR_ATTRACTOR_NOT_FOUND, -20, b"GCSO_ERROR_ATTRACTOR_NOT_FOUND\0", "GCSO_ERROR_ATTRACTOR_NOT_FOUND", "Target attractor anchor ID not found in topological field."),
    (GCSO_ERROR_DPSR_PHASE_OVERFLOW, -30, b"GCSO_ERROR_DPSR_PHASE_OVERFLOW\0", "GCSO_ERROR_DPSR_PHASE_OVERFLOW", "DPSR phase angle accumulation exceeded safety bound."),
    (GCSO_ERROR_QDPS_UNDERFLOW, -31, b"GCSO_ERROR_QDPS_UNDERFLOW\0", "GCSO_ERROR_QDPS_UNDERFLOW", "QDPS quantization step fell below minimal threshold $\\Delta\\theta_{\\mathrm{min\\_step}}$."),
    (GCSO_ERROR_EDBC_SINGULARITY, -40, b"GCSO_ERROR_EDBC_SINGULARITY\0", "GCSO_ERROR_EDBC_SINGULARITY", "EDBC entropy controller encountered numerical or logarithmic singularity."),
    (GCSO_ERROR_CONTAINER_CORRUPTED, -50, b"GCSO_ERROR_CONTAINER_CORRUPTED\0", "GCSO_ERROR_CONTAINER_CORRUPTED", "Binary `.gcso` snapshot container header or payload checksum corrupted."),
    (GCSO_ERROR_ZIMMS_MAPPING_FAILED, -51, b"GCSO_ERROR_ZIMMS_MAPPING_FAILED\0", "GCSO_ERROR_ZIMMS_MAPPING_FAILED", "ZIMMS zero-copy memory mapping or Direct-DMA buffer mapping failed."),
    (GCSO_ERROR_PSPM_ROUTING_FAILED, -60, b"GCSO_ERROR_PSPM_ROUTING_FAILED\0", "GCSO_ERROR_PSPM_ROUTING_FAILED", "PSPM sub-head group phase routing evaluation failed."),
    (GCSO_ERROR_PPRC_SEEK_FAILED, -70, b"GCSO_ERROR_PPRC_SEEK_FAILED\0", "GCSO_ERROR_PPRC_SEEK_FAILED", "PPRC keyframe index lookup or token seek operation failed."),
    (GCSO_ERROR_OBSTRUCTION_UNRESOLVED, -80, b"GCSO_ERROR_OBSTRUCTION_UNRESOLVED\0", "GCSO_ERROR_OBSTRUCTION_UNRESOLVED", "Unresolved cohomological obstruction class $H^1(K; \\mathcal{F})$ encountered."),
    (GCSO_ERROR_EXTENSION_NOT_LOADED, -90, b"GCSO_ERROR_EXTENSION_NOT_LOADED\0", "GCSO_ERROR_EXTENSION_NOT_LOADED", "Dynamic extension module not loaded or inactive in DAES slot."),
    (GCSO_ERROR_DAES_SCRATCHPAD_FULL, -91, b"GCSO_ERROR_DAES_SCRATCHPAD_FULL\0", "GCSO_ERROR_DAES_SCRATCHPAD_FULL", "DAES dynamic scratchpad fast-path table or ring buffer fully occupied."),
    (GCSO_ERROR_UNKNOWN, -0x7FFF_FFFF, b"GCSO_ERROR_UNKNOWN\0", "GCSO_ERROR_UNKNOWN", "Unknown internal system error."),
);

impl fmt::Display for GcsoStatus {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<GcsoStatus> for Result<(), GcsoStatus> {
    #[inline(always)]
    fn from(status: GcsoStatus) -> Self {
        if status == GCSO_SUCCESS {
            Ok(())
        } else {
            Err(status)
        }
    }
}

impl From<Result<(), GcsoStatus>> for GcsoStatus {
    #[inline(always)]
    fn from(res: Result<(), GcsoStatus>) -> Self {
        match res {
            Ok(()) => GCSO_SUCCESS,
            Err(status) => status,
        }
    }
}

/// C-ABI type alias for status codes matching C headers (`gcso_status_t`).
pub type gcso_status_t = GcsoStatus;
/// Quantized 7-bit signed fixed-point integer (phase scale $\beta_{\mathrm{Q7}} = \pi / 128$).
pub type gcso_q7_t = i8;
/// 64-bit capability flags bitmask type (`gcso_capability_flags_t`).
pub type gcso_capability_flags_t = u64;
/// C-ABI type alias for the 128-byte 2-Origin Composite Manifold Gateway Frame (`GatewayFrame`).
pub type gcso_gateway_frame_t = GatewayFrame;
/// C-ABI type alias for the 16-byte Causal Provenance Report (`CausalProvenanceReport`).
pub type gcso_causal_provenance_report_t = CausalProvenanceReport;
/// C-ABI type alias for the 128-byte `.gcso`-persisted Crystallized Attractor Record (`CrystallizedAttractorRecord`).
pub type gcso_crystallized_attractor_record_t = CrystallizedAttractorRecord;
/// C-ABI type alias for the 64-byte Contrastive Phase Bloom Filter (`ContrastivePhaseBloomFilter`).
pub type gcso_contrastive_bloom_filter_t = ContrastivePhaseBloomFilter;
/// C-ABI type alias for the 32-byte Stigmergic Thresholds configuration (`StigmergicThresholds`).
pub type gcso_stigmergic_thresholds_t = StigmergicThresholds;
/// C-ABI type alias for the 32-byte Action Orchestration Report (`ActionOrchestrationReport`).
pub type gcso_action_orchestration_report_t = ActionOrchestrationReport;
/// C-ABI type alias for the 8-bit Stigmergic Action Decision discriminant (`StigmergicActionDecision`).
pub type gcso_stigmergic_action_decision_t = u8;

// ===================================================================
// Strongly-Typed Opaque Handles
// ===================================================================

macro_rules! define_opaque_handles {
    ($(($opaque:ident, $handle:ident, $c_alias:ident, $doc:expr)),* $(,)?) => {
        $(
            #[doc = $doc]
            #[repr(C)]
            pub struct $opaque {
                _private: [u8; 0],
            }

            #[doc = $doc]
            pub type $handle = *mut $opaque;

            #[doc = $doc]
            pub type $c_alias = $handle;
        )*
    };
}

define_opaque_handles!(
    (
        GcsoContextOpaque,
        GcsoContextHandle,
        gcso_context_handle_t,
        "Opaque runtime context handle (`gcso_context_handle_t`)."
    ),
    (
        GcsoContainerOpaque,
        GcsoContainerHandle,
        gcso_container_handle_t,
        "Opaque `.gcso` binary snapshot container handle (`gcso_container_handle_t`)."
    ),
    (
        GcsoActionHubOpaque,
        GcsoActionHubHandle,
        gcso_action_hub_handle_t,
        "Opaque Action Hub (Sidecar Pointer Table) handle (`gcso_action_hub_handle_t`)."
    ),
    (
        GcsoDaesSlotOpaque,
        GcsoDaesSlotHandle,
        gcso_daes_slot_handle_t,
        "Opaque DAES extension scratchpad slot handle (`gcso_daes_slot_handle_t`)."
    ),
    (
        GcsoAttractorFieldOpaque,
        GcsoAttractorFieldHandle,
        gcso_attractor_field_handle_t,
        "Opaque Macro Attractor Field handle (`gcso_attractor_field_handle_t`)."
    ),
    (
        GcsoEdbcControllerOpaque,
        GcsoEdbcControllerHandle,
        gcso_edbc_controller_handle_t,
        "Opaque EDBC entropy controller handle (`gcso_edbc_controller_handle_t`)."
    ),
    (
        GcsoDpsrKernelOpaque,
        GcsoDpsrKernelHandle,
        gcso_dpsr_kernel_handle_t,
        "Opaque DPSR phase steering kernel handle (`gcso_dpsr_kernel_handle_t`)."
    ),
    (
        GcsoPspmRouterOpaque,
        GcsoPspmRouterHandle,
        gcso_pspm_router_handle_t,
        "Opaque PSPM sub-head group router handle (`gcso_pspm_router_handle_t`)."
    ),
    (
        GcsoSrlAdapterOpaque,
        GcsoSrlAdapterHandle,
        gcso_srl_adapter_handle_t,
        "Opaque SRL Rank-1 adapter layer handle (`gcso_srl_adapter_handle_t`)."
    ),
);

/// Classification types for topological attractor field anchors (Section 5.3).
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum GcsoAnchorType {
    /// Natural language system prompt anchor projected into phase space (Primary Endpoint).
    SystemPrompt = 0,
    /// Dense feature embedding vector anchor in latent space.
    Embedding = 1,
    /// Phase-conjugate anti-phase repulsion anchor ($-\boldsymbol{\Delta\theta}$).
    PhaseRepulse = 2,
    /// Manifold topological knot/tree anchor.
    Topological = 3,
    /// Self-organized macro anchor crystallized bottom-up from high-density pointer trails.
    Crystallized = 4,
}

impl GcsoAnchorType {
    /// Converts a raw `u32` value to a [`GcsoAnchorType`] enum variant if valid.
    #[inline(always)]
    #[must_use]
    pub const fn from_u32(val: u32) -> Option<Self> {
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

/// C-ABI type alias for anchor classification (`gcso_anchor_type_t`).
pub type gcso_anchor_type_t = u32;
/// System prompt string anchor descriptor type (`0`).
pub const GCSO_ANCHOR_TYPE_SYSTEM_PROMPT: gcso_anchor_type_t = 0;
/// Feature embedding anchor descriptor type (`1`).
pub const GCSO_ANCHOR_TYPE_EMBEDDING: gcso_anchor_type_t = 1;
/// Phase repulsion anchor descriptor type (`2`).
pub const GCSO_ANCHOR_TYPE_PHASE_REPULSE: gcso_anchor_type_t = 2;
/// Topological knot anchor descriptor type (`3`).
pub const GCSO_ANCHOR_TYPE_TOPOLOGICAL: gcso_anchor_type_t = 3;
/// Crystallized invariant anchor descriptor type (`4`).
pub const GCSO_ANCHOR_TYPE_CRYSTALLIZED: gcso_anchor_type_t = 4;

// ===================================================================
// C-ABI Binary Data Structures (Section 5.4 & Section 8.1)
// ===================================================================

/// 64-bit tagged memory pointer descriptor matching GCSO specification (Section 5.4.A).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct gcso_tagged_ptr_t(
    /// Raw 64-bit packed tagged pointer word.
    pub u64,
);

impl gcso_tagged_ptr_t {
    /// Bitmask for extracting the lower 48-bit canonical raw address.
    pub const ADDR_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;
    /// Bitmask for extracting the 8-bit domain/action tag (`bits 48..=55`).
    pub const TAG_MASK: u64 = 0x00FF_0000_0000_0000;
    /// Bitmask for extracting the 8-bit Q7 phase state (`bits 56..=63`).
    pub const PHASE_MASK: u64 = 0xFF00_0000_0000_0000;
    /// Bit shift offset for the 8-bit domain/action tag.
    pub const TAG_SHIFT: u32 = 48;
    /// Bit shift offset for the 8-bit Q7 phase state.
    pub const PHASE_SHIFT: u32 = 56;
    /// Null tagged pointer constant (`0`).
    pub const NULL: Self = Self(0);

    /// Packs a raw 48-bit address, 8-bit domain tag, and Q7 phase value into a 64-bit tagged pointer.
    #[inline(always)]
    #[must_use]
    pub const fn pack(raw_addr: u64, tag: u8, phase: gcso_q7_t) -> Self {
        let addr = raw_addr & Self::ADDR_MASK;
        let tag_bits = (tag as u64) << Self::TAG_SHIFT;
        let phase_bits = ((phase as u8) as u64) << Self::PHASE_SHIFT;
        Self(addr | tag_bits | phase_bits)
    }

    /// Returns the raw 64-bit integer representation.
    #[inline(always)]
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Returns `true` if the 48-bit canonical address component is zero.
    #[inline(always)]
    #[must_use]
    pub const fn is_null(self) -> bool {
        (self.0 & Self::ADDR_MASK) == 0
    }

    /// Unpacks the canonical 48-bit raw address pointer.
    #[inline(always)]
    #[must_use]
    pub const fn unpack_addr(self) -> u64 {
        self.0 & Self::ADDR_MASK
    }

    /// Unpacks the 8-bit domain/action tag.
    #[inline(always)]
    #[must_use]
    pub const fn unpack_tag(self) -> u8 {
        ((self.0 & Self::TAG_MASK) >> Self::TAG_SHIFT) as u8
    }

    /// Unpacks the Q7 quantized phase state.
    #[inline(always)]
    #[must_use]
    pub const fn unpack_phase(self) -> gcso_q7_t {
        ((self.0 >> Self::PHASE_SHIFT) as u8) as gcso_q7_t
    }

    /// Returns a new tagged pointer with an updated 48-bit canonical address, preserving tag and phase.
    #[inline(always)]
    #[must_use]
    pub const fn with_addr(self, new_addr: u64) -> Self {
        Self((self.0 & !Self::ADDR_MASK) | (new_addr & Self::ADDR_MASK))
    }

    /// Returns a new tagged pointer with an updated 8-bit domain/action tag, preserving address and phase.
    #[inline(always)]
    #[must_use]
    pub const fn with_tag(self, new_tag: u8) -> Self {
        Self((self.0 & !Self::TAG_MASK) | ((new_tag as u64) << Self::TAG_SHIFT))
    }

    /// Returns a new tagged pointer with an updated Q7 phase value, preserving address and tag.
    #[inline(always)]
    #[must_use]
    pub const fn with_phase(self, new_phase: gcso_q7_t) -> Self {
        Self((self.0 & !Self::PHASE_MASK) | (((new_phase as u8) as u64) << Self::PHASE_SHIFT))
    }

    /// Branchlessly accumulates `delta_q7` into the embedded Q7 phase using saturating addition.
    #[inline(always)]
    #[must_use]
    pub const fn add_phase_sat(self, delta_q7: gcso_q7_t) -> Self {
        self.with_phase(branchless_q7_add_sat(self.unpack_phase(), delta_q7))
    }

    /// Branchlessly negates the embedded Q7 phase into its phase-conjugate anti-phase ($-\Delta\theta$).
    #[inline(always)]
    #[must_use]
    pub const fn conjugate_phase(self) -> Self {
        self.with_phase(branchless_q7_neg_sat(self.unpack_phase()))
    }
}

/// Descriptor header for structure byte size and ABI version validation (8 bytes, 4-byte aligned).
#[repr(C, align(4))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct gcso_descriptor_header_t {
    /// Size of the enclosing descriptor structure in bytes.
    pub struct_size: u32,
    /// ABI version bitmask (`0x00000101` for v0.1.1).
    pub abi_version: u32,
}

/// 256-bit PagedBlock bitmask layout aligned to 32 bytes for Warp/SIMD reductions (Section 5.4.B).
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct gcso_paged_bitmask_t {
    /// Four 64-bit unsigned words storing 256 discrete token/block execution flags.
    pub bits: [u64; 4],
}

/// Stigmergic pointer trail structure aligned to 128 bytes (2 cache lines, Section 5.4.C).
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct gcso_pointer_trail_t {
    /// Current 64-bit tagged memory pointer address.
    pub current_ptr: u64,
    /// Previous 64-bit tagged memory pointer address.
    pub prev_ptr: u64,
    /// Extensible caller payload data.
    pub user_data: u64,
    /// Accumulated graph edge traversal cost.
    pub transition_cost: i32,
    /// Total path execution steps taken along this trail.
    pub step_count: u32,
    /// Local stigmergic trail density ($\rho_{\mathrm{stigmergy}}$) for cellular swarm routing.
    pub stigmergic_density: f32,
    /// Full 32-bit identifier of the dominant macro attractor anchor on the unified phase manifold.
    pub target_anchor_id: u32,
    /// Active topological cluster / macro centroid identifier (or 3-bit working lane ID `0..=7`).
    pub cluster_id: u32,
    /// Linked adjacent cellular trail identifier along the pointer chain.
    pub linked_trail_id: u32,
    /// Gravitational pull force ($F_{\mathrm{pull}}$) exerted by nearby attractor(s).
    pub attractor_pull_force: f32,
    /// Execution status, synergy resonance, Limina beat, and repulsion bitmask flags.
    pub flags: u32,
    /// Accumulated Q7 phase rotation state across 64 attention heads.
    pub accumulated_phase_delta: [gcso_q7_t; 64],
    /// Alignment padding to guarantee exact 128-byte boundary.
    pub reserved_padding: [u8; 8],
}

impl Default for gcso_pointer_trail_t {
    #[inline(always)]
    fn default() -> Self {
        // SAFETY: `gcso_pointer_trail_t` is a C-ABI `Pod` struct where all-zero bytes are valid.
        unsafe { core::mem::zeroed() }
    }
}

/// Dynamic Adaptive Extension Scratchpad (DAES) slot layout (64 bytes, 1 cache line, Section 5.4.F).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct gcso_daes_slot_t {
    /// Active extension execution mode (`0` = Scratchpad, `1` = DNP Plugin, `2` = Shared IPC Buffer).
    pub mode: u32,
    /// Ring buffer head write index.
    pub telemetry_ring_head: u16,
    /// Ring buffer tail read index.
    pub telemetry_ring_tail: u16,
    /// Total fast-path lookup cache hits recorded.
    pub cache_hit_count: u32,
    /// Bitmask flags configuring autonomous auto-tuning behavior.
    pub auto_tune_flags: u32,
    /// 64-bit bitmask specifying fast-path bypass criteria.
    pub fast_path_bypass_mask: u64,
    /// Four cached 64-bit tagged jump pointers for $\mathcal{O}(1)$ accelerated execution.
    pub fast_path_shortcuts: [u64; 4],
    /// Circular 8-byte ledger array storing recent Nano/Micro telemetry codes.
    pub telemetry_mini_ledger: [u8; 8],
}

/// Global runtime configuration descriptor structure aligned to 16 bytes (64 bytes total, Section 5.4.D).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct gcso_config_t {
    /// Dimension per attention head ($d_{\mathrm{head}}$, must be positive and even).
    pub head_dim: u32,
    /// Total number of attention heads ($H \le 64$).
    pub num_heads: u32,
    /// Tokens per paged KV cache block (typically 16 or 32).
    pub paged_block_size: u32,
    /// Floating-point scale factor for Q7 phase values ($\beta_{\mathrm{Q7}} = \pi / 128$).
    pub q7_phase_scale: f32,
    /// Maximum allowed rotation angle for RIPA soft-clamping in radians ($\theta_{\max}$).
    pub ripa_clamp_max_rad: f32,
    /// Minimum rotation angle threshold for QDPS step filtering in radians ($\Delta\theta_{\mathrm{min\_step}}$).
    pub qdps_min_step_rad: f32,
    /// Epsilon clamp to prevent logarithmic singularity in entropy calculation ($\epsilon_{\mathrm{log}}$).
    pub entropy_singularity_eps: f32,
    /// Maximum supported prompt anchor points in attractor field.
    pub max_prompt_anchors: u32,
    /// Maximum slot capacity of the Action Hub (SPT) pointer table.
    pub action_hub_capacity: u32,
    /// Enable GPU warp shuffle inline evaluations (`1` = enabled, `0` = disabled).
    pub enable_cuda_warp_shuffle: u8,
    /// Strict enforcement of zero dynamic allocations on Hot Path (`1` = strict).
    pub enable_zero_alloc_strict: u8,
    /// Initial operational mode for DAES scratchpad (`0`, `1`, or `2`).
    pub daes_mode: u8,
    /// Reserved alignment and configuration flags.
    pub reserved_flags: u8,
    /// Reserved space for future ABI core extensions (24 bytes).
    pub reserved: [u8; 24],
}

/// Dynamic Entropy-Driven Decoding Branch Controller (EDBC) state structure (64 bytes, 32-byte aligned, Section 5.4.G).
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct gcso_edbc_state_t {
    /// Moving Z-score normalized attention entropy ($\tilde{H} \in [0, \ln 2]$).
    pub moving_z_entropy: f32,
    /// Z-score threshold ($\tau_{\mathrm{bifurcation}}$) triggering pitchfork branch split.
    pub bifurcation_threshold: f32,
    /// Numerical safety threshold ($\epsilon_{\mathrm{log}}$) for singularity detection.
    pub singularity_eps: f32,
    /// Sliding-window entropy rate integrator ($\Phi_M(t) = \tilde{H}(t) - \tilde{H}(t-M)$).
    pub sliding_entropy_rate: f32,
    /// Gain factor applied to phase-conjugate repulsion pulses ($-\boldsymbol{\Delta\theta}$).
    pub repulsion_gain: f32,
    /// Energy-guided dynamic sampling temperature modifier.
    pub sample_temperature: f32,
    /// Active execution branch mode (`0` = Normal, `1` = Pitchfork Bifurcation, `2` = Repulsion).
    pub active_branch_mode: u32,
    /// Reserved space for 32-byte alignment and future parameters (36 bytes).
    pub reserved: [u8; 36],
}

impl Default for gcso_edbc_state_t {
    #[inline(always)]
    fn default() -> Self {
        // SAFETY: `gcso_edbc_state_t` is a C-ABI `Pod` struct where all-zero bytes are valid.
        unsafe { core::mem::zeroed() }
    }
}

/// Zero-Overhead In-Memory Mapped Storage (ZIMMS) descriptor (64 bytes, 32-byte aligned, Section 5.4.G).
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct gcso_zimms_descriptor_t {
    /// Base virtual memory address of the zero-copy `mmap` region.
    pub mapped_address: u64,
    /// Total mapped `.gcso` file length in bytes.
    pub file_size_bytes: u64,
    /// Direct Memory Access (DMA) ring-buffer handle for async page streaming.
    pub dma_buffer_handle: u64,
    /// Memory protection and mapping attribute flags.
    pub flags: u32,
    /// Low-level operating system file descriptor (`-1` when unmapped).
    pub fd_handle: i32,
    /// Reserved space for future storage extensions (32 bytes).
    pub reserved: [u8; 32],
}

impl Default for gcso_zimms_descriptor_t {
    #[inline(always)]
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

/// Phase-Steered Parallel Multi-head (PSPM) router configuration descriptor (32 bytes, 16-byte aligned, Section 5.4.G).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct gcso_pspm_config_t {
    /// Number of attention heads allocated to the Fact sub-group.
    pub num_fact_heads: u16,
    /// Number of attention heads allocated to the Logic sub-group.
    pub num_logic_heads: u16,
    /// Number of attention heads allocated to the Explore sub-group.
    pub num_explore_heads: u16,
    /// Bitmask flags controlling single-pass routing behavior.
    pub flags: u16,
    /// Phase/amplitude steering gain applied to the Fact head group.
    pub fact_phase_gain: f32,
    /// Phase/amplitude steering gain applied to the Logic head group.
    pub logic_phase_gain: f32,
    /// Phase/amplitude steering gain applied to the Explore head group.
    pub explore_phase_gain: f32,
    /// Reserved padding for 16-byte alignment (12 bytes).
    pub reserved: [u8; 12],
}

/// Sparse Residual Adapter Layer (SRL) Dynamic Rank-1 descriptor (64 bytes, 32-byte aligned, Section 5.4.G).
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct gcso_srl_descriptor_t {
    /// Index of target Transformer layer for Rank-1 residual injection.
    pub layer_idx: u32,
    /// Rank of low-rank adapter (fixed to `1` for SRL).
    pub rank: u32,
    /// Raw 64-bit pointer to left column projection vector $\mathbf{u}$.
    pub u_vector_ptr: u64,
    /// Raw 64-bit pointer to right row projection vector $\mathbf{v}$.
    pub v_vector_ptr: u64,
    /// Raw 64-bit pointer to optional diagonal gain vector $\mathbf{s}$ (`0` if unused).
    pub gain_scalar_ptr: u64,
    /// Global scaling multiplier for Rank-1 residual injection.
    pub scale_factor: f32,
    /// Bitmask flags indicating precision format and activation state.
    pub flags: u32,
    /// Reserved space for 32-byte alignment requirements (24 bytes).
    pub reserved: [u8; 24],
}

/// Unified `.gcso` binary snapshot container header structure (128 bytes, 64-byte aligned, Section 5.4.E).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct gcso_snapshot_header_t {
    /// Format identifier magic constant (`0x4F534347` = ASCII `"GCSO"`).
    pub magic: u32,
    /// Snapshot ABI version (`0x00000101` = v0.1.1).
    pub version: u32,
    /// Total binary size of container payload in bytes.
    pub total_size: u64,
    /// Byte offset to Action Hub (SPT) section.
    pub action_hub_offset: u64,
    /// Byte offset to Attractor Field section.
    pub attractor_field_offset: u64,
    /// Byte offset to DPSR kernel state section.
    pub dpsr_state_offset: u64,
    /// Byte offset to SRL adapter state section.
    pub srl_state_offset: u64,
    /// Byte offset to EDBC controller state section.
    pub edbc_state_offset: u64,
    /// CRC32 integrity checksum over snapshot payload.
    pub checksum_crc32: u32,
    /// Byte offset to DAES scratchpad state section.
    pub daes_slot_offset: u32,
    /// Epoch timestamp of snapshot serialization (seconds).
    pub timestamp_epoch_sec: u64,
    /// Padding to enforce exact 128-byte size and 64-byte cacheline alignment (56 bytes).
    pub reserved_padding: [u8; 56],
}

impl Default for gcso_snapshot_header_t {
    #[inline(always)]
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

/// PPRC Keyframe KV cache index header structure (64 bytes, 32-byte aligned, Section 5.4.G).
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct gcso_pprc_keyframe_header_t {
    /// Frame classification (`0` = I-Frame Keyframe Anchor, `1` = P-Frame Phase Motion Delta).
    pub frame_type: u32,
    /// Absolute token position index in sequence.
    pub token_index: u32,
    /// Dynamic entropy-driven Group of Pictures (GOP) length between keyframes.
    pub gop_length: u32,
    /// Mean resultant vector length $\bar{R}_c \in [0, 1]$ in circular statistics.
    pub composite_vector_length: f32,
    /// Concentration parameter $\kappa_c$ for von Mises phase distribution.
    pub von_mises_kappa: f32,
    /// Scalar residual magnitude / resultant phase angle $\bar{\theta}_c$ for sparse updates.
    pub sparse_scalar_residual: f32,
    /// Byte offset to Intra Key-Frame Cache (I-Cache) payload.
    pub icache_payload_offset: u64,
    /// Byte offset to Predictive Phase Motion Cache (P-Cache) payload.
    pub pcache_payload_offset: u64,
    /// Reserved space for future PPRC extensions (24 bytes).
    pub reserved: [u8; 24],
}

// Compile-time layout assertions enforcing 1:1 binary parity with `c_abi_spec.md` Section 8.1
// and the 2-Origin Gateway / `.gcso` Crystallization structures.
const _: () = {
    assert!(size_of::<gcso_status_t>() == 4);
    assert!(size_of::<gcso_q7_t>() == 1);
    assert!(size_of::<gcso_tagged_ptr_t>() == 8);
    assert!(
        size_of::<gcso_descriptor_header_t>() == 8 && align_of::<gcso_descriptor_header_t>() == 4
    );
    assert!(size_of::<gcso_paged_bitmask_t>() == 32 && align_of::<gcso_paged_bitmask_t>() == 32);
    assert!(size_of::<gcso_pointer_trail_t>() == 128 && align_of::<gcso_pointer_trail_t>() == 128);
    assert!(size_of::<gcso_daes_slot_t>() == 64 && align_of::<gcso_daes_slot_t>() == 64);
    assert!(
        size_of::<gcso_stigmergic_thresholds_t>() == 32
            && align_of::<gcso_stigmergic_thresholds_t>() == 32
    );
    assert!(
        size_of::<gcso_action_orchestration_report_t>() == 32
            && align_of::<gcso_action_orchestration_report_t>() == 32
    );
    assert!(size_of::<gcso_config_t>() == 64 && align_of::<gcso_config_t>() == 16);
    assert!(size_of::<gcso_edbc_state_t>() == 64 && align_of::<gcso_edbc_state_t>() == 32);
    assert!(
        size_of::<gcso_zimms_descriptor_t>() == 64 && align_of::<gcso_zimms_descriptor_t>() == 32
    );
    assert!(size_of::<gcso_pspm_config_t>() == 32 && align_of::<gcso_pspm_config_t>() == 16);
    assert!(size_of::<gcso_srl_descriptor_t>() == 64 && align_of::<gcso_srl_descriptor_t>() == 32);
    assert!(
        size_of::<gcso_snapshot_header_t>() == 128 && align_of::<gcso_snapshot_header_t>() == 64
    );
    assert!(
        size_of::<gcso_pprc_keyframe_header_t>() == 64
            && align_of::<gcso_pprc_keyframe_header_t>() == 32
    );
    assert!(size_of::<SwarmResolutionReport>() == 16 && align_of::<SwarmResolutionReport>() == 16);
    assert!(size_of::<gcso_gateway_frame_t>() == 128 && align_of::<gcso_gateway_frame_t>() == 64);
    assert!(
        size_of::<gcso_causal_provenance_report_t>() == 16
            && align_of::<gcso_causal_provenance_report_t>() == 16
    );
    assert!(
        size_of::<gcso_crystallized_attractor_record_t>() == 128
            && align_of::<gcso_crystallized_attractor_record_t>() == 64
    );
    assert!(
        size_of::<gcso_contrastive_bloom_filter_t>() == 64
            && align_of::<gcso_contrastive_bloom_filter_t>() == 64
    );
    assert!(offset_of!(gcso_descriptor_header_t, struct_size) == 0);
    assert!(offset_of!(gcso_descriptor_header_t, abi_version) == 4);
    assert!(offset_of!(gcso_paged_bitmask_t, bits) == 0);
    assert!(offset_of!(gcso_pointer_trail_t, current_ptr) == 0);
    assert!(offset_of!(gcso_pointer_trail_t, prev_ptr) == 8);
    assert!(offset_of!(gcso_pointer_trail_t, user_data) == 16);
    assert!(offset_of!(gcso_pointer_trail_t, transition_cost) == 24);
    assert!(offset_of!(gcso_pointer_trail_t, step_count) == 28);
    assert!(offset_of!(gcso_pointer_trail_t, stigmergic_density) == 32);
    assert!(offset_of!(gcso_pointer_trail_t, target_anchor_id) == 36);
    assert!(offset_of!(gcso_pointer_trail_t, cluster_id) == 40);
    assert!(offset_of!(gcso_pointer_trail_t, linked_trail_id) == 44);
    assert!(offset_of!(gcso_pointer_trail_t, attractor_pull_force) == 48);
    assert!(offset_of!(gcso_pointer_trail_t, flags) == 52);
    assert!(offset_of!(gcso_pointer_trail_t, accumulated_phase_delta) == 56);
    assert!(offset_of!(gcso_pointer_trail_t, reserved_padding) == 120);
    assert!(offset_of!(gcso_config_t, head_dim) == 0);
    assert!(offset_of!(gcso_config_t, qdps_min_step_rad) == 20);
    assert!(offset_of!(gcso_config_t, action_hub_capacity) == 32);
    assert!(offset_of!(gcso_config_t, daes_mode) == 38);
    assert!(offset_of!(gcso_snapshot_header_t, magic) == 0);
    assert!(offset_of!(gcso_snapshot_header_t, version) == 4);
    assert!(offset_of!(gcso_snapshot_header_t, total_size) == 8);
    assert!(offset_of!(gcso_snapshot_header_t, checksum_crc32) == 56);
    assert!(offset_of!(gcso_snapshot_header_t, daes_slot_offset) == 60);
    assert!(offset_of!(gcso_snapshot_header_t, timestamp_epoch_sec) == 64);
    assert!(offset_of!(gcso_daes_slot_t, mode) == 0);
    assert!(offset_of!(gcso_daes_slot_t, fast_path_bypass_mask) == 16);
    assert!(offset_of!(gcso_daes_slot_t, fast_path_shortcuts) == 24);
    assert!(offset_of!(gcso_daes_slot_t, telemetry_mini_ledger) == 56);
    assert!(offset_of!(gcso_pspm_config_t, num_fact_heads) == 0);
    assert!(offset_of!(gcso_pspm_config_t, fact_phase_gain) == 8);
    assert!(offset_of!(gcso_edbc_state_t, moving_z_entropy) == 0);
    assert!(offset_of!(gcso_edbc_state_t, active_branch_mode) == 24);
    assert!(offset_of!(gcso_srl_descriptor_t, layer_idx) == 0);
    assert!(offset_of!(gcso_srl_descriptor_t, u_vector_ptr) == 8);
    assert!(offset_of!(gcso_srl_descriptor_t, v_vector_ptr) == 16);
    assert!(offset_of!(gcso_srl_descriptor_t, scale_factor) == 32);
    assert!(offset_of!(gcso_zimms_descriptor_t, mapped_address) == 0);
    assert!(offset_of!(gcso_zimms_descriptor_t, file_size_bytes) == 8);
    assert!(offset_of!(gcso_pprc_keyframe_header_t, frame_type) == 0);
    assert!(offset_of!(gcso_pprc_keyframe_header_t, icache_payload_offset) == 24);
    assert!(offset_of!(gcso_pprc_keyframe_header_t, pcache_payload_offset) == 32);
};
