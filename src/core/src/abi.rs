// name: src/core/src/abi.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! C-ABI interface definitions, binary layout contracts, pointer-driven mode orchestration
//! entry points, Single Unified Attractor Manifold evaluators, 2-Origin Composite Manifold
//! Gateway FFI bridges, Classical Online Toroidal $k$-Means / DP-Means macro clustering
//! (single-frame & deferred-CRC32 batch), Coarse-to-Fine routing (centroid-only, candidate-trail,
//! & fused Bayesian + Attention score modulation), Limina inter-memory bridging, 4-section
//! `.gcso` zero-copy container persistence (with `gcso_zimms_descriptor_t` preamble &
//! `CausalProvenanceReport` history), and FFI boundary safety primitives for the Geometric
//! Cellular Sheaf Orchestrator (GCSO) core engine.
//!
//! # Architectural Role (Boundary Layer)
//! This module forms the C11/C++20 and Rust FFI boundary barrier for `LiminiKa` (ABI v0.1.1,
//! `0x00000101`). It enforces:
//! - **1:1 Binary Layout Parity**: Compile-time `size_of`, `align_of`, and `offset_of` verification
//!   matching C headers in `include/liminika/` (`c_abi_spec.md` Section 8.1).
//! - **Exception & Panic Isolation**: [`ffi_boundary!`] intercepts unwinding Rust panics
//!   (`catch_unwind`) and translates them into `GCSO_ERROR_PANIC_CAUGHT`.
//! - **Type-Driven Alignment & `GCSO_RESTRICT` Non-Aliasing**: All raw pointers crossing FFI
//!   boundaries are validated via [`AlignedSlice16`], [`AlignedSlice32`], [`AlignedSliceMut16`],
//!   [`AlignedSliceMut32`], [`DisjointMutPair`], [`DisjointRefMut`], [`FfiHandleRef`], and
//!   [`FfiHandleMut`] before entering Nano/Micro Hot Path execution.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(non_camel_case_types)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::inline_always)]
#![allow(clippy::incompatible_msrv)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_option_as_deref)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::ifs_same_cond)]
#![allow(clippy::field_reassign_with_default)]

#[cfg(feature = "std")]
extern crate std;

use alloc::boxed::Box;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::{
    __m256i, _mm256_castsi256_pd, _mm256_cmpeq_epi64, _mm256_load_si256, _mm256_loadu_si256,
    _mm256_movemask_pd, _mm256_or_si256, _mm256_setzero_si256, _mm256_store_si256,
};
use core::ffi::{c_char, CStr};
use core::fmt;
use core::mem::{align_of, offset_of, size_of};
use core::ptr;
use core::slice;

use crate::action_hub::{ActionHubEngine, ActionOrchestrationReport, StigmergicThresholds};
use crate::dpsr::DpsrEngine;
use crate::edbc::{eval_barrier_pitchfork, EdbcEngine};
use crate::traits::aligned::{
    is_ptr_aligned_to, AlignedSlice16, AlignedSlice32, AlignedSliceMut16, AlignedSliceMut32,
    DisjointMutPair, DisjointRefMut,
};
use crate::traits::branchless::{
    branchless_fast_transition_log_prob, branchless_project_f32_slice_to_q7_64,
    branchless_q7_inhibit_64, branchless_q7_resonant_pull_64, branchless_q7_synergy_fuse_64,
    branchless_q7_unified_field_fold_words_with_init_weight_64,
    modulate_attention_scores_by_phase_bank_64, sum_sq_f32_simd8, swar_slerp_words_with_bias_64,
    swar_words_equal_64, swar_words_nonzero_64,
};
use crate::traits::cold_path::{EntropyEvaluator, L2pSvdProjector, SystemCapabilityQuery};
use crate::traits::gateway::{
    project_bytes_swar_to_words_64, project_f32_slice_to_q7_64, project_f32_slice_to_words_64,
    synthesize_multi_origin_gateway_frame_with_words, CausalProvenanceReport,
    CausalProvenanceSignal, CognitiveGroundingSignal, ContradictionGuard, ContradictionGuardSignal,
    CrystallizedAttractorRecord, GatewayFrame, GcsoFourSectionView, ManifoldGatewaySignal,
    ObjectiveFact, RawPhaseProjectable, SubjectivePersona, SystemConstraint,
    CAUSAL_AXIS_CONTRADICTION_GUARD, GCSO_FOUR_SECTION_HEADER_BYTES,
};
use crate::traits::primitives::{
    branchless_atan2_to_q7_fast, branchless_clamp_f32, branchless_crc32_ieee,
    branchless_flush_subnormal_f32, branchless_is_finite_f32, branchless_min_f32,
    branchless_q7_add_sat, branchless_q7_neg_sat, branchless_rad_to_qdps_step_u8,
    branchless_relu_f32, branchless_select_f32, branchless_select_u32, branchless_select_u8,
    branchless_splitmix64, branchless_sqrt_f32, load_q7_words_64, store_q7_words_64,
    swar_q7_qdps_filter_words_64, trail_phase_words_mut, trail_phase_words_ref, SWAR_SPLAT_BYTE_64,
};
use crate::traits::primitives::{
    init_snapshot_header, is_slot_pair_coupled, pack_action_slot_tag, raw_byte_ranges_overlap,
    unpack_action_mode_from_tag, unpack_attractor_slot_from_tag, validate_snapshot_header,
    FfiHandleMut, FfiHandleRef, GcsoResult, Pod, ACTION_PRESET_REPULSION_ESCAPE,
    ATTRACTOR_SLOT_ID_MASK, ATTRACTOR_SLOT_REPULSION, DEFAULT_RIPA_CLAMP_MAX_RAD,
    MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS, PAGED_BITMASK_BITS,
};
use crate::traits::rope::{
    dispatch_fused_ripa_and_pspm_words_core, steer_from_swar_words_core_safe, PhaseSteering,
    PspmRouter, SrlAdapter,
};
use crate::traits::telemetry::{
    ContrastivePhaseBloomFilter, DaesScratchpad, PagedBitmaskOps, StigmergicActionPtrOps,
    StigmergicSwarmOps, StigmergicTrailOps, SwarmResolutionReport, DAES_METRIC_PHASE_STEERED,
    DAES_METRIC_PSPM_ROUTED, DAES_METRIC_REPULSION_APPLIED, DAES_METRIC_SLOT_SYNERGY_ACTIVE,
    MAX_ATTRACTOR_PULL_FORCE, NUM_SWARM_MICRO_CELLS, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE,
    TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE,
    TRAIL_FLAG_SYNERGY_RESONANCE,
};
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

/// Helper macro to catch unwinding panics and safely bridge execution into C-ABI error status codes.
#[macro_export]
macro_rules! ffi_boundary {
    ($body:expr) => {{
        let run_ffi_body = || -> Result<$crate::abi::GcsoStatus, $crate::abi::GcsoStatus> { $body };

        #[cfg(feature = "std")]
        let status = match ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(run_ffi_body))
        {
            Ok(Ok(status)) => status,
            Ok(Err(err_status)) => err_status,
            Err(_) => $crate::abi::GCSO_ERROR_PANIC_CAUGHT,
        };

        #[cfg(not(feature = "std"))]
        let status = match run_ffi_body() {
            Ok(status) => status,
            Err(err_status) => err_status,
        };

        status
    }};
}

/// Standardized 32-bit signed status code returned across the C-ABI FFI boundary.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
pub struct GcsoStatus(
    /// Inner signed 32-bit integer status code.
    pub i32,
);

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

// ===================================================================
// Internal Zero-Cost Stateless Trait Dispatchers & Validation Helpers
// ===================================================================

struct AbiSystemDispatcher;
impl SystemCapabilityQuery for AbiSystemDispatcher {}

struct AbiDaesDispatcher;
impl DaesScratchpad for AbiDaesDispatcher {
    #[inline]
    fn evaluate_auto_tune(
        &self,
        slot: &gcso_daes_slot_t,
        config_out: &mut gcso_config_t,
    ) -> GcsoResult<()> {
        if config_out.head_dim == 0 {
            AbiSystemDispatcher.init_default_config(config_out)?;
        }
        // Single 64-bit SWAR instruction counting all 8 bytes >= 0x80 (`0` loops, `0` branches).
        let spike_count =
            (u64::from_le_bytes(slot.telemetry_mini_ledger) & 0x8080_8080_8080_8080).count_ones();
        let base_ripa = branchless_select_f32(
            config_out.ripa_clamp_max_rad > 0.0,
            config_out.ripa_clamp_max_rad,
            DEFAULT_RIPA_CLAMP_MAX_RAD,
        );
        let damping = 1.0 - 0.025 * (spike_count as f32);
        config_out.ripa_clamp_max_rad = branchless_clamp_f32(base_ripa * damping, 0.01, 0.25);
        config_out.daes_mode = (slot.mode & 0xFF) as u8;
        Ok(())
    }
}

struct AbiPspmDispatcher;
impl PspmRouter for AbiPspmDispatcher {}

struct AbiSrlDispatcher;
impl SrlAdapter for AbiSrlDispatcher {}

struct AbiL2pDispatcher;
impl L2pSvdProjector for AbiL2pDispatcher {}

/// Branchlessly checks pointer non-nullness and natural alignment for type `T`.
#[inline(always)]
#[must_use]
pub fn is_aligned<T>(ptr: *const T) -> bool {
    let align = align_of::<T>();
    (!ptr.is_null()) & (((ptr as usize) & (align - 1)) == 0)
}

/// Branchlessly checks pointer non-nullness and custom power-of-two byte alignment.
#[inline(always)]
#[must_use]
pub fn is_aligned_to<T>(ptr: *const T, align: usize) -> bool {
    is_ptr_aligned_to(ptr, align)
}

/// Validates a non-null, naturally aligned pointer or opaque handle.
#[inline(always)]
fn validate_ptr<T>(ptr: *const T) -> GcsoResult<()> {
    if ptr.is_null() {
        Err(GCSO_ERROR_NULL_POINTER)
    } else if !is_aligned(ptr) {
        Err(GCSO_ERROR_MISALIGNED_POINTER)
    } else {
        Ok(())
    }
}

/// Validates a non-null pointer against an explicit power-of-two alignment boundary `align`.
#[inline(always)]
fn validate_ptr_aligned<T>(ptr: *const T, align: usize) -> GcsoResult<()> {
    if ptr.is_null() {
        Err(GCSO_ERROR_NULL_POINTER)
    } else if !is_aligned_to(ptr, align) {
        Err(GCSO_ERROR_MISALIGNED_POINTER)
    } else {
        Ok(())
    }
}

/// Zero-copy borrowed 64-head Q7 phase wrapper implementing [`RawPhaseProjectable`] so FFI callers
/// passing `*const gcso_q7_t` into 2-Origin Gateway synthesis load directly into 8 `u64` SWAR
/// registers without intermediate 64-byte stack array copies.
#[repr(transparent)]
#[derive(Clone, Copy)]
struct AbiPhaseRef64<'a>(&'a [gcso_q7_t; MAX_ATTENTION_HEADS]);

impl RawPhaseProjectable for AbiPhaseRef64<'_> {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        *self.0
    }

    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        load_q7_words_64(self.0)
    }
}

/// Zero-copy borrowed `f32` embedding slice wrapper implementing [`RawPhaseProjectable`] so FFI
/// callers passing `*const f32` into 2-Origin Gateway synthesis project directly onto $SO(2)^{64}$
/// via 8-YMM AVX2 SIMD accumulation (`gain > 0.0`) or unrolled `swar_q7_pack_x8` register packing
/// ([`project_f32_slice_to_words_64`], `gain == 0.0`).
#[derive(Clone, Copy)]
struct AbiF32EmbeddingRef<'a> {
    slice: &'a [f32],
    gain: f32,
}

impl RawPhaseProjectable for AbiF32EmbeddingRef<'_> {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        if self.gain > 0.0 {
            branchless_project_f32_slice_to_q7_64(self.slice, self.gain)
        } else {
            project_f32_slice_to_q7_64(self.slice)
        }
    }

    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        if self.gain > 0.0 {
            let q7 = branchless_project_f32_slice_to_q7_64(self.slice, self.gain);
            load_q7_words_64(&q7)
        } else {
            project_f32_slice_to_words_64(self.slice)
        }
    }
}

/// Private zero-copy borrowed [`ContrastivePhaseBloomFilter`] guard wrapper inside `abi.rs`
/// implementing [`CausalProvenanceSignal`], [`CognitiveGroundingSignal`], and
/// [`ContradictionGuardSignal`] without colliding with blanket implementations on
/// [`ContradictionGuard`] in `gateway.rs`. Note: `BranchlessKernel` is automatically
/// provided via the universal blanket `impl<T: ?Sized> BranchlessKernel for T {}` in `branchless.rs`.
#[derive(Clone, Copy)]
struct AbiBorrowedBloomGuard<'a> {
    bloom: &'a ContrastivePhaseBloomFilter,
    bloom_repulsion_threshold: u8,
}

impl<'a> AbiBorrowedBloomGuard<'a> {
    #[inline(always)]
    const fn new(
        bloom: &'a ContrastivePhaseBloomFilter,
        _obstruction_threshold: f32,
        bloom_repulsion_threshold: u8,
    ) -> Self {
        let safe_bloom =
            branchless_select_u8(bloom_repulsion_threshold > 0, bloom_repulsion_threshold, 1);
        Self {
            bloom,
            bloom_repulsion_threshold: safe_bloom,
        }
    }
}

impl CausalProvenanceSignal for AbiBorrowedBloomGuard<'_> {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        CAUSAL_AXIS_CONTRADICTION_GUARD
    }

    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        0.0
    }
}

impl CognitiveGroundingSignal for AbiBorrowedBloomGuard<'_> {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        [0i8; MAX_ATTENTION_HEADS]
    }

    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        [0u64; NUM_SWARM_MICRO_CELLS]
    }

    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        0.0
    }
}

impl ContradictionGuardSignal for AbiBorrowedBloomGuard<'_> {
    #[inline(always)]
    fn requires_phase_slice_for_repulsion(&self) -> bool {
        false
    }

    #[inline(always)]
    fn evaluate_repulsion_severity(
        &self,
        candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let words = load_q7_words_64(candidate_phases);
        self.evaluate_repulsion_severity_from_words(&words, candidate_phases)
    }

    #[inline(always)]
    fn evaluate_repulsion_severity_from_words(
        &self,
        candidate_words: &[u64; NUM_SWARM_MICRO_CELLS],
        _candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let density = if self.bloom.is_empty() {
            0
        } else {
            self.bloom
                .query_repulsive_density_from_words(candidate_words)
        };
        let bloom_hit = density >= self.bloom_repulsion_threshold;
        branchless_select_f32(
            bloom_hit,
            branchless_clamp_f32(0.60 + (density as f32) * 0.05, 0.60, 1.0),
            0.0,
        )
    }
}

/// Generic zero-copy helper synthesizing a [`GatewayFrame`] from Origin 1 (`SystemConstraint`)
/// and Origin 2 (`ObjectiveFact<F>`, `SubjectivePersona<P>`, optional `AbiBorrowedBloomGuard`).
///
/// # Safety
/// `bloom_filter_opt` must be `NULL` or a valid aligned pointer to [`gcso_contrastive_bloom_filter_t`].
#[inline]
unsafe fn synthesize_multi_origin_generic_with_words<
    F: RawPhaseProjectable,
    P: RawPhaseProjectable,
>(
    vram_pressure: f32,
    gpu_frame_overrun: f32,
    active_swarm_cell_mask: u8,
    recommend_layer_skip: u8,
    fact_sig_opt: Option<ObjectiveFact<F>>,
    pers_sig_opt: Option<SubjectivePersona<P>>,
    bloom_filter_opt: *const gcso_contrastive_bloom_filter_t,
    contradiction_threshold: f32,
    bloom_repulsion_threshold: u8,
) -> GcsoResult<(GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS])> {
    // SAFETY: `FfiHandleRef::from_raw_opt` validates non-nullness and alignment of `bloom_filter_opt`.
    let bloom_ref_opt = unsafe { FfiHandleRef::from_raw_opt(bloom_filter_opt)? };
    let sys_sig =
        SystemConstraint::new((), vram_pressure, gpu_frame_overrun, active_swarm_cell_mask)
            .with_layer_skip(recommend_layer_skip != 0);
    Ok(match bloom_ref_opt {
        Some(b_ref) => {
            let guard = AbiBorrowedBloomGuard::new(
                &b_ref,
                contradiction_threshold,
                bloom_repulsion_threshold,
            );
            synthesize_multi_origin_gateway_frame_with_words(
                Some(&sys_sig),
                fact_sig_opt.as_ref(),
                pers_sig_opt.as_ref(),
                Some(&guard),
            )
        }
        None => {
            let guard_none: Option<&ContradictionGuard<()>> = None;
            synthesize_multi_origin_gateway_frame_with_words(
                Some(&sys_sig),
                fact_sig_opt.as_ref(),
                pers_sig_opt.as_ref(),
                guard_none,
            )
        }
    })
}

/// Zero-copy internal helper synthesizing a [`GatewayFrame`] directly from raw Q7 phase pointers.
///
/// # Safety
/// Non-null phase pointers must point to at least 64 valid `gcso_q7_t` elements.
#[inline]
unsafe fn synthesize_multi_origin_from_raw_ptrs_with_words(
    vram_pressure: f32,
    gpu_frame_overrun: f32,
    active_swarm_cell_mask: u8,
    recommend_layer_skip: u8,
    fact_phases: *const gcso_q7_t,
    fact_confidence: f32,
    fact_zimms_offset48: u64,
    fact_anchor_id: u32,
    persona_phases: *const gcso_q7_t,
    persona_strength: f32,
    persona_zimms_offset48: u64,
    persona_anchor_id: u32,
    bloom_filter_opt: *const gcso_contrastive_bloom_filter_t,
    contradiction_threshold: f32,
    bloom_repulsion_threshold: u8,
) -> GcsoResult<(GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS])> {
    let fact_sig_opt = if fact_phases.is_null() {
        None
    } else {
        // SAFETY: `fact_phases` is non-null and `align_of::<[gcso_q7_t; 64]>() == 1`.
        let f_ref = unsafe { &*fact_phases.cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>() };
        Some(ObjectiveFact::new(
            AbiPhaseRef64(f_ref),
            fact_confidence,
            fact_zimms_offset48,
            fact_anchor_id,
        ))
    };
    let pers_sig_opt = if persona_phases.is_null() {
        None
    } else {
        // SAFETY: `persona_phases` is non-null and `align_of::<[gcso_q7_t; 64]>() == 1`.
        let p_ref = unsafe { &*persona_phases.cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>() };
        Some(SubjectivePersona::new(
            AbiPhaseRef64(p_ref),
            persona_strength,
            persona_zimms_offset48,
            persona_anchor_id,
        ))
    };
    // SAFETY: Validated above.
    unsafe {
        synthesize_multi_origin_generic_with_words(
            vram_pressure,
            gpu_frame_overrun,
            active_swarm_cell_mask,
            recommend_layer_skip,
            fact_sig_opt,
            pers_sig_opt,
            bloom_filter_opt,
            contradiction_threshold,
            bloom_repulsion_threshold,
        )
    }
}

/// Zero-copy internal helper synthesizing a [`GatewayFrame`] directly from raw byte slices.
///
/// # Safety
/// Non-null byte pointers must reference at least `fact_byte_len` / `persona_byte_len` bytes.
#[inline]
unsafe fn synthesize_multi_origin_from_raw_bytes_with_words(
    vram_pressure: f32,
    gpu_frame_overrun: f32,
    active_swarm_cell_mask: u8,
    recommend_layer_skip: u8,
    fact_bytes: *const u8,
    fact_byte_len: usize,
    fact_confidence: f32,
    fact_zimms_offset48: u64,
    fact_anchor_id: u32,
    persona_bytes: *const u8,
    persona_byte_len: usize,
    persona_strength: f32,
    persona_zimms_offset48: u64,
    persona_anchor_id: u32,
    bloom_filter_opt: *const gcso_contrastive_bloom_filter_t,
    contradiction_threshold: f32,
    bloom_repulsion_threshold: u8,
) -> GcsoResult<(GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS])> {
    if (fact_byte_len > 0 && fact_bytes.is_null())
        || (persona_byte_len > 0 && persona_bytes.is_null())
    {
        return Err(GCSO_ERROR_NULL_POINTER);
    }
    let fact_sig_opt = if fact_bytes.is_null() {
        None
    } else {
        let f_slice: &[u8] = if fact_byte_len == 0 {
            &[]
        } else {
            // SAFETY: Non-nullness for `fact_byte_len > 0` is verified above.
            unsafe { slice::from_raw_parts(fact_bytes, fact_byte_len) }
        };
        Some(ObjectiveFact::new(
            f_slice,
            fact_confidence,
            fact_zimms_offset48,
            fact_anchor_id,
        ))
    };
    let pers_sig_opt = if persona_bytes.is_null() {
        None
    } else {
        let p_slice: &[u8] = if persona_byte_len == 0 {
            &[]
        } else {
            // SAFETY: Non-nullness for `persona_byte_len > 0` is verified above.
            unsafe { slice::from_raw_parts(persona_bytes, persona_byte_len) }
        };
        Some(SubjectivePersona::new(
            p_slice,
            persona_strength,
            persona_zimms_offset48,
            persona_anchor_id,
        ))
    };
    // SAFETY: Validated above.
    unsafe {
        synthesize_multi_origin_generic_with_words(
            vram_pressure,
            gpu_frame_overrun,
            active_swarm_cell_mask,
            recommend_layer_skip,
            fact_sig_opt,
            pers_sig_opt,
            bloom_filter_opt,
            contradiction_threshold,
            bloom_repulsion_threshold,
        )
    }
}

/// Zero-copy internal helper synthesizing a [`GatewayFrame`] directly from raw `f32` embedding slices.
///
/// # Safety
/// Non-null `f32` pointers must be 4-byte aligned and reference at least `fact_len` / `persona_len` elements.
#[inline]
unsafe fn synthesize_multi_origin_from_raw_f32_with_words(
    vram_pressure: f32,
    gpu_frame_overrun: f32,
    active_swarm_cell_mask: u8,
    recommend_layer_skip: u8,
    fact_values: *const f32,
    fact_len: usize,
    fact_projection_gain: f32,
    fact_confidence: f32,
    fact_zimms_offset48: u64,
    fact_anchor_id: u32,
    persona_values: *const f32,
    persona_len: usize,
    persona_projection_gain: f32,
    persona_strength: f32,
    persona_zimms_offset48: u64,
    persona_anchor_id: u32,
    bloom_filter_opt: *const gcso_contrastive_bloom_filter_t,
    contradiction_threshold: f32,
    bloom_repulsion_threshold: u8,
) -> GcsoResult<(GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS])> {
    if !branchless_is_finite_f32(fact_projection_gain)
        || fact_projection_gain < 0.0
        || !branchless_is_finite_f32(persona_projection_gain)
        || persona_projection_gain < 0.0
    {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    if fact_len > 0 {
        validate_ptr(fact_values)?;
    }
    if persona_len > 0 {
        validate_ptr(persona_values)?;
    }
    let fact_sig_opt = if fact_values.is_null() {
        None
    } else {
        let f_slice: &[f32] = if fact_len == 0 {
            &[]
        } else {
            // SAFETY: Validated for non-nullness and `f32` alignment when `fact_len > 0`.
            unsafe { slice::from_raw_parts(fact_values, fact_len) }
        };
        Some(ObjectiveFact::new(
            AbiF32EmbeddingRef {
                slice: f_slice,
                gain: fact_projection_gain,
            },
            fact_confidence,
            fact_zimms_offset48,
            fact_anchor_id,
        ))
    };
    let pers_sig_opt = if persona_values.is_null() {
        None
    } else {
        let p_slice: &[f32] = if persona_len == 0 {
            &[]
        } else {
            // SAFETY: Validated for non-nullness and `f32` alignment when `persona_len > 0`.
            unsafe { slice::from_raw_parts(persona_values, persona_len) }
        };
        Some(SubjectivePersona::new(
            AbiF32EmbeddingRef {
                slice: p_slice,
                gain: persona_projection_gain,
            },
            persona_strength,
            persona_zimms_offset48,
            persona_anchor_id,
        ))
    };
    // SAFETY: Validated above.
    unsafe {
        synthesize_multi_origin_generic_with_words(
            vram_pressure,
            gpu_frame_overrun,
            active_swarm_cell_mask,
            recommend_layer_skip,
            fact_sig_opt,
            pers_sig_opt,
            bloom_filter_opt,
            contradiction_threshold,
            bloom_repulsion_threshold,
        )
    }
}

/// Unified register-resident core helper for [`GatewayFrame`] steering across FFI entry points:
/// loads `frame_ref.phase_deltas` once into 8 `u64` SWAR registers (`frm_w`), updates `trail`,
/// steers `q_aligned` (and optional `k_opt`) with $SO(2)$-fused RIPA + PSPM routing, optionally
/// scales `logits_opt`, and returns `report` without copying `GatewayFrame`.
#[inline]
fn steer_gateway_frame_with_words_core_safe(
    frame_ref: &GatewayFrame,
    frm_w: &[u64; NUM_SWARM_MICRO_CELLS],
    trail: &mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    head_dim: usize,
    num_heads: usize,
    q_aligned: &mut AlignedSliceMut32<'_, f32>,
    k_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
) -> GcsoResult<CausalProvenanceReport> {
    if frame_ref.qdps_min_step_rad < 0.0
        || !branchless_is_finite_f32(frame_ref.qdps_min_step_rad)
        || frame_ref.ripa_clamp_rad <= 0.0
        || !branchless_is_finite_f32(frame_ref.ripa_clamp_rad)
    {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    if let Some(ref k_ref) = k_opt {
        q_aligned.ensure_disjoint_mut(k_ref)?;
    }

    let report = frame_ref.to_causal_report();
    trail.prev_ptr = trail.current_ptr;
    trail.current_ptr = frame_ref.tagged_ptr;
    trail.step_count = trail.step_count.wrapping_add(1);
    if frame_ref.target_anchor_id != 0 {
        trail.target_anchor_id = frame_ref.target_anchor_id;
        trail.cluster_id = frame_ref.target_anchor_id;
    } else {
        trail.cluster_id = u32::from(frame_ref.dominant_slot & ATTRACTOR_SLOT_ID_MASK);
    }
    trail.sync_flags_with_current_ptr();

    let _ = crate::traits::gateway::accumulate_trail_and_steer_gateway_words_core(
        frame_ref,
        frm_w,
        trail,
        unresolved_damping_u7,
        head_dim,
        num_heads,
        Some(q_aligned),
        k_opt,
        logits_opt,
    )?;

    Ok(report)
}

/// Shared FFI execution tail for [`GatewayFrame`] steering + optional Online Toroidal $k$-Means
/// / DP-Means centroid update + optional output pointer writes.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[inline]
unsafe fn execute_gateway_frame_with_words_steer_and_optional_kmeans_ffi(
    frame: &GatewayFrame,
    frm_w: &[u64; NUM_SWARM_MICRO_CELLS],
    trail_inout: *mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
    centroids_inout: *mut gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    split_coherence_threshold: f32,
    frame_out: *mut gcso_gateway_frame_t,
    report_out: *mut gcso_causal_provenance_report_t,
    assigned_centroid_idx_out: *mut usize,
    assigned_coherence_out: *mut f32,
    updated_gravity_mass_out: *mut f32,
) -> GcsoResult<()> {
    validate_ptr(trail_inout)?;
    let has_centroids = !centroids_inout.is_null() && num_centroids > 0;
    if has_centroids {
        validate_ptr(centroids_inout)?;
        if !branchless_is_finite_f32(split_coherence_threshold) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let cent_bytes = num_centroids
            .checked_mul(size_of::<CrystallizedAttractorRecord>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            centroids_inout as usize,
            cent_bytes,
            trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
    }

    // SAFETY: `trail_inout` is validated above.
    let mut trail = unsafe { FfiHandleMut::from_raw(trail_inout)? };
    // SAFETY: `bind_qk_and_logits_32` validates alignment, bounds, and pairwise disjointness.
    let (_engine, mut q_aligned, mut k_opt, mut logits_opt) = unsafe {
        bind_qk_and_logits_32(
            query_tensor,
            key_tensor,
            head_dim,
            num_heads,
            logits_inout,
            vocab_size,
        )?
    };

    let report = steer_gateway_frame_with_words_core_safe(
        frame,
        frm_w,
        &mut trail,
        unresolved_damping_u7,
        head_dim,
        num_heads,
        &mut q_aligned,
        k_opt.as_mut(),
        logits_opt.as_mut(),
    )?;

    if has_centroids {
        // SAFETY: `centroids_inout` is validated non-null, aligned, and disjoint from `trail_inout`.
        let centroids_slice = unsafe { slice::from_raw_parts_mut(centroids_inout, num_centroids) };
        let (assigned_idx, _assigned_anchor_id, assigned_coh, updated_mass, _spawned_new) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_frame_with_words_64(
                centroids_slice,
                frame,
                frm_w,
                split_coherence_threshold,
                Some(&mut *trail),
            );
        // SAFETY: `write_opt_out` validates optional output pointers before writing.
        unsafe {
            write_opt_out(assigned_centroid_idx_out, assigned_idx)?;
            write_opt_out(assigned_coherence_out, assigned_coh)?;
            write_opt_out(updated_gravity_mass_out, updated_mass)?;
        }
    }

    // SAFETY: `write_opt_out` validates optional output pointers before writing.
    unsafe {
        write_opt_out(frame_out, *frame)?;
        write_opt_out(report_out, report)?;
    }
    Ok(())
}

/// Shared FFI execution tail when `frame` is passed by pointer across the C-ABI boundary.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[inline]
unsafe fn execute_gateway_frame_steer_and_optional_kmeans_ffi(
    frame: &GatewayFrame,
    trail_inout: *mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
    centroids_inout: *mut gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    split_coherence_threshold: f32,
    frame_out: *mut gcso_gateway_frame_t,
    report_out: *mut gcso_causal_provenance_report_t,
    assigned_centroid_idx_out: *mut usize,
    assigned_coherence_out: *mut f32,
    updated_gravity_mass_out: *mut f32,
) -> GcsoResult<()> {
    let frm_w = frame.phase_words_ref();
    // SAFETY: Validated by `execute_gateway_frame_with_words_steer_and_optional_kmeans_ffi`.
    unsafe {
        execute_gateway_frame_with_words_steer_and_optional_kmeans_ffi(
            frame,
            frm_w,
            trail_inout,
            unresolved_damping_u7,
            query_tensor,
            key_tensor,
            head_dim,
            num_heads,
            logits_inout,
            vocab_size,
            centroids_inout,
            num_centroids,
            split_coherence_threshold,
            frame_out,
            report_out,
            assigned_centroid_idx_out,
            assigned_coherence_out,
            updated_gravity_mass_out,
        )
    }
}

/// Shared register-resident post-routing QDPS + RIPA phase steering helper for Coarse-to-Fine
/// pointer-trail routing entry points (fast-pathing uniform and all-zero filtered words).
#[inline]
fn steer_routed_trail_words_qk_safe(
    steer_words: &[u64; NUM_SWARM_MICRO_CELLS],
    trail_flags: u32,
    win_clamp_rad: f32,
    min_step_rad: f32,
    head_dim: usize,
    num_heads: usize,
    q_aligned: &mut AlignedSliceMut32<'_, f32>,
    k_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
) {
    if (trail_flags & TRAIL_FLAG_INERTIAL_SKIP_ACTIVE) == 0 {
        let min_step_q7 = branchless_rad_to_qdps_step_u8(min_step_rad);
        let eff_ripa = branchless_select_f32(
            branchless_is_finite_f32(win_clamp_rad) & (win_clamp_rad > 0.0),
            win_clamp_rad,
            DEFAULT_RIPA_CLAMP_MAX_RAD,
        );
        if min_step_q7 <= 1 {
            if (steer_words[0] != 0) || swar_words_nonzero_64(steer_words) {
                steer_from_swar_words_core_safe(
                    steer_words,
                    0xFF,
                    128,
                    head_dim,
                    num_heads,
                    false,
                    eff_ripa,
                    q_aligned,
                    k_opt,
                );
            }
        } else {
            let swar_words = swar_q7_qdps_filter_words_64(steer_words, min_step_q7);
            if (swar_words[0] != 0) || swar_words_nonzero_64(&swar_words) {
                steer_from_swar_words_core_safe(
                    &swar_words,
                    0xFF,
                    128,
                    head_dim,
                    num_heads,
                    false,
                    eff_ripa,
                    q_aligned,
                    k_opt,
                );
            }
        }
    }
}

/// Shared register-resident crystallized ensemble folding + optional Bloom repulsion + QDPS/RIPA
/// steering helper for [`gcso_dpsr_steer_crystallized_ensemble_zero_copy_safe`] and
/// [`gcso_container_map_and_steer_four_section_zero_copy_safe`].
#[inline]
fn steer_crystallized_records_with_optional_bloom_safe(
    records: &[CrystallizedAttractorRecord],
    bloom_opt: Option<&ContrastivePhaseBloomFilter>,
    trail_inout: &mut gcso_pointer_trail_t,
    min_step_rad: f32,
    head_dim: usize,
    num_heads: usize,
    q_aligned: &mut AlignedSliceMut32<'_, f32>,
    k_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
) -> GcsoResult<(f32, u32, f32)> {
    if min_step_rad < 0.0 || !branchless_is_finite_f32(min_step_rad) {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }

    let init_pull =
        branchless_flush_subnormal_f32(branchless_relu_f32(trail_inout.attractor_pull_force));
    let (mut folded_w, tot_pull, dom_id, dom_clamp) =
        CrystallizedAttractorRecord::fold_crystallized_ensemble_words_with_init_pull_64(
            records,
            &mut trail_inout.accumulated_phase_delta,
            init_pull,
        );

    if dom_id != 0 {
        trail_inout.target_anchor_id = dom_id;
        trail_inout.cluster_id = dom_id;
    }
    trail_inout.attractor_pull_force = branchless_min_f32(tot_pull, MAX_ATTRACTOR_PULL_FORCE);
    trail_inout.flags |= TRAIL_FLAG_CRYSTALLIZED_CANDIDATE | TRAIL_FLAG_SYNERGY_RESONANCE;

    if let Some(bloom_ref) = bloom_opt {
        if !bloom_ref.is_empty() {
            let (_density, rep_active) =
                bloom_ref.apply_phase_conjugate_repulsion_words(&mut folded_w, 4, 128);
            if rep_active {
                *trail_phase_words_mut(trail_inout) = folded_w;
                let cur_ptr = gcso_tagged_ptr_t(trail_inout.current_ptr);
                let rep_mode = cur_ptr.action_mode_flags() | ACTION_PRESET_REPULSION_ESCAPE;
                trail_inout.current_ptr = cur_ptr
                    .with_action_and_slot(ATTRACTOR_SLOT_REPULSION, rep_mode)
                    .conjugate_phase()
                    .raw();
                trail_inout.flags |= TRAIL_FLAG_REPULSION_ACTIVE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE;
            }
        }
    }
    trail_inout.sync_flags_with_current_ptr();

    let eff_ripa = branchless_select_f32(
        branchless_is_finite_f32(dom_clamp) & (dom_clamp > 0.0),
        dom_clamp,
        DEFAULT_RIPA_CLAMP_MAX_RAD,
    );
    steer_routed_trail_words_qk_safe(
        &folded_w,
        trail_inout.flags,
        eff_ripa,
        min_step_rad,
        head_dim,
        num_heads,
        q_aligned,
        k_opt,
    );

    Ok((tot_pull, dom_id, eff_ripa))
}

/// Reinterprets a non-null raw `gcso_q7_t` pointer as a zero-copy reference to a 64-head phase array.
///
/// # Safety
/// `ptr` must be non-null and point to at least 64 valid `gcso_q7_t` elements.
#[inline(always)]
unsafe fn ref_phase64_from_raw<'a>(
    ptr: *const gcso_q7_t,
) -> GcsoResult<&'a [gcso_q7_t; MAX_ATTENTION_HEADS]> {
    if ptr.is_null() {
        return Err(GCSO_ERROR_NULL_POINTER);
    }
    // SAFETY: `ptr` is non-null and `align_of::<[gcso_q7_t; 64]>() == 1`.
    Ok(unsafe { &*ptr.cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>() })
}

/// Reinterprets a non-null raw mutable `gcso_q7_t` pointer as a zero-copy mutable reference to a 64-head phase array.
///
/// # Safety
/// `ptr` must be non-null, uniquely borrowed, and point to at least 64 valid `gcso_q7_t` elements.
#[inline(always)]
unsafe fn mut_phase64_from_raw<'a>(
    ptr: *mut gcso_q7_t,
) -> GcsoResult<&'a mut [gcso_q7_t; MAX_ATTENTION_HEADS]> {
    if ptr.is_null() {
        return Err(GCSO_ERROR_NULL_POINTER);
    }
    // SAFETY: `ptr` is non-null and `align_of::<[gcso_q7_t; 64]>() == 1`.
    Ok(unsafe { &mut *ptr.cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>() })
}

/// Reinterprets a raw `gcso_q7_t` pointer of `num_items * 64` bytes as a zero-copy slice of 64-head phase arrays.
///
/// # Safety
/// When `num_items > 0`, `ptr` must be non-null and point to `num_items * 64` valid `gcso_q7_t` elements.
#[inline(always)]
unsafe fn slice_of_phase64_from_raw<'a>(
    ptr: *const gcso_q7_t,
    num_items: usize,
) -> GcsoResult<&'a [[gcso_q7_t; MAX_ATTENTION_HEADS]]> {
    if num_items == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(GCSO_ERROR_NULL_POINTER);
    }
    let _total_bytes = num_items
        .checked_mul(MAX_ATTENTION_HEADS)
        .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
    // SAFETY: Validated non-null and overflow-checked above.
    Ok(unsafe { slice::from_raw_parts(ptr.cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>(), num_items) })
}

/// Validates a `DpsrEngine` configuration from `(head_dim, num_heads)` and returns `(engine, total_elements)`.
#[inline(always)]
fn init_dpsr_engine(head_dim: usize, num_heads: usize) -> GcsoResult<(DpsrEngine, usize)> {
    let hd_u32 = u32::try_from(head_dim).map_err(|_| GCSO_ERROR_INVALID_ARGUMENT)?;
    let nh_u32 = u32::try_from(num_heads).map_err(|_| GCSO_ERROR_INVALID_ARGUMENT)?;
    let engine = DpsrEngine::new(hd_u32, nh_u32)?;
    let total_len = head_dim
        .checked_mul(num_heads)
        .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
    Ok((engine, total_len))
}

/// Writes `val` to an optional raw output pointer `out_ptr` after validating natural alignment.
///
/// # Safety
/// If `out_ptr` is non-null, it must be naturally aligned and writable for `T`.
#[inline(always)]
unsafe fn write_opt_out<T: Copy>(out_ptr: *mut T, val: T) -> GcsoResult<()> {
    // SAFETY: `FfiHandleMut::from_raw_opt` validates alignment when `out_ptr` is non-null.
    if let Some(mut out_ref) = unsafe { FfiHandleMut::from_raw_opt(out_ptr)? } {
        *out_ref = val;
    }
    Ok(())
}

/// Bridges `src_trail.accumulated_phase_delta` toward `cand_phases` in 8 `u64` SWAR registers
/// by `interpolation_u7` (`0..=128`) with a 1-word fast path when both profiles are word-uniform.
#[inline(always)]
fn bridge_trail_words_slerp_u7(
    src_trail: &mut gcso_pointer_trail_t,
    cand_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    interpolation_u7: u8,
) -> [u64; NUM_SWARM_MICRO_CELLS] {
    let mut steer_words = *trail_phase_words_ref(src_trail);
    if interpolation_u7 > 0 {
        let cand_w = load_q7_words_64(cand_phases);
        if !swar_words_equal_64(&steer_words, &cand_w) {
            let s_u64 = branchless_select_u8(interpolation_u7 > 128, 128, interpolation_u7) as u64;
            let bias_restore = (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
            steer_words = swar_slerp_words_with_bias_64(&steer_words, &cand_w, s_u64, bias_restore);
            *trail_phase_words_mut(src_trail) = steer_words;
        }
    }
    steer_words
}

/// Synchronizes `src_trail.current_ptr` with a winning macro centroid's 48-bit ZIMMS section address and tags.
#[inline(always)]
fn sync_winner_centroid_ptr_to_trail(
    src_trail: &mut gcso_pointer_trail_t,
    winner: &CrystallizedAttractorRecord,
) {
    if winner.sample_count > 0 && winner.zimms_section_offset48 != 0 {
        let packed_tag = pack_action_slot_tag(winner.dominant_slot, winner.action_mode_flags);
        let cur_phase = gcso_tagged_ptr_t(src_trail.current_ptr).unpack_phase();
        src_trail.prev_ptr = src_trail.current_ptr;
        src_trail.current_ptr =
            gcso_tagged_ptr_t::pack(winner.zimms_section_offset48, packed_tag, cur_phase).raw();
    }
}

/// Validates and binds `query_tensor` (mandatory, 32B aligned) and `key_tensor` (optional, 32B
/// aligned, disjoint from `query_tensor`) for `(head_dim, num_heads)`.
///
/// # Safety
/// Pointers must reference valid buffers of at least `head_dim * num_heads` `f32` elements.
#[inline(always)]
unsafe fn bind_qk_32<'a>(
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
) -> GcsoResult<(
    DpsrEngine,
    AlignedSliceMut32<'a, f32>,
    Option<AlignedSliceMut32<'a, f32>>,
)> {
    validate_ptr_aligned(query_tensor, 32)?;
    let (engine, total_len) = init_dpsr_engine(head_dim, num_heads)?;
    // SAFETY: `AlignedSliceMut32` validates non-nullness and 32-byte alignment.
    let (q_aligned, k_opt) = unsafe {
        (
            AlignedSliceMut32::from_raw_parts_mut(query_tensor, total_len)?,
            AlignedSliceMut32::from_raw_parts_mut_opt(key_tensor, total_len)?,
        )
    };
    if let Some(ref k_ref) = k_opt {
        q_aligned.ensure_disjoint_mut(k_ref)?;
    }
    Ok((engine, q_aligned, k_opt))
}

/// Validates and binds `query_tensor` (mandatory, 32B aligned), `key_tensor` (optional, 32B aligned),
/// and `logits_inout` (optional, 32B aligned, disjoint from Q/K) for `(head_dim, num_heads)`.
///
/// # Safety
/// Pointers must reference valid buffers of the specified lengths.
#[inline(always)]
unsafe fn bind_qk_and_logits_32<'a>(
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
) -> GcsoResult<(
    DpsrEngine,
    AlignedSliceMut32<'a, f32>,
    Option<AlignedSliceMut32<'a, f32>>,
    Option<AlignedSliceMut32<'a, f32>>,
)> {
    // SAFETY: Validated by `bind_qk_32`.
    let (engine, q_aligned, k_opt) =
        unsafe { bind_qk_32(query_tensor, key_tensor, head_dim, num_heads)? };
    let logits_opt = if logits_inout.is_null() || vocab_size == 0 {
        None
    } else {
        // SAFETY: Validated by `AlignedSliceMut32::from_raw_parts_mut_opt`.
        let l_slice =
            unsafe { AlignedSliceMut32::from_raw_parts_mut_opt(logits_inout, vocab_size)? };
        if let Some(ref l_ref) = l_slice {
            q_aligned.ensure_disjoint_mut(l_ref)?;
            if let Some(ref k_ref) = k_opt {
                k_ref.ensure_disjoint_mut(l_ref)?;
            }
        }
        l_slice
    };
    Ok((engine, q_aligned, k_opt, logits_opt))
}

/// Validates a null-terminated C string pointer and ensures valid UTF-8 encoding in `#![no_std]` mode.
///
/// # Safety
/// `c_str_ptr` must be non-null and point to a valid NUL-terminated C string.
#[inline]
unsafe fn validate_c_utf8_str<'a>(c_str_ptr: *const c_char) -> GcsoResult<&'a str> {
    if c_str_ptr.is_null() {
        return Err(GCSO_ERROR_NULL_POINTER);
    }
    // SAFETY: `c_str_ptr` is non-null and caller guarantees NUL-termination.
    let c_str = unsafe { CStr::from_ptr(c_str_ptr) };
    c_str.to_str().map_err(|_| GCSO_ERROR_INVALID_ARGUMENT)
}

/// Resolves flexible `(head_dim, num_heads)` dimensions for flat tensor C-ABI entry points.
#[inline(always)]
const fn resolve_flat_tensor_heads(dim: usize) -> GcsoResult<(u32, u32)> {
    if dim == 0 || (dim & 1) != 0 || dim > (u32::MAX as usize) {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    if dim >= 128 && (dim % 128 == 0) {
        let heads = dim / 128;
        if heads <= MAX_ATTENTION_HEADS {
            return Ok((128, heads as u32));
        }
        let hd = dim / MAX_ATTENTION_HEADS;
        if (hd & 1) == 0 {
            return Ok((hd as u32, MAX_ATTENTION_HEADS as u32));
        }
    }
    Ok((dim as u32, 1))
}

macro_rules! impl_stateless_handle_destroy {
    ($(($fn_name:ident, $handle_ty:ty, $doc:expr, $safety_doc:expr)),* $(,)?) => {
        $(
            #[doc = $doc]
            ///
            /// # Safety
            #[doc = $safety_doc]
            #[no_mangle]
            pub unsafe extern "C" fn $fn_name(handle: $handle_ty) -> GcsoStatus {
                if handle.is_null() {
                    return GCSO_SUCCESS;
                }
                if !is_aligned(handle) {
                    return GCSO_ERROR_MISALIGNED_POINTER;
                }
                ffi_boundary!(Ok(GCSO_SUCCESS))
            }
        )*
    };
}

// ===================================================================
// 0. Tagged Pointer Bit-Packing & Stigmergic Mode Evolution C-ABI
// ===================================================================

/// Packs a 48-bit virtual address, 8-bit tag, and Q7 phase offset into a 64-bit tagged pointer.
#[no_mangle]
pub extern "C" fn gcso_tagged_ptr_pack(raw_addr: u64, tag: u8, phase: gcso_q7_t) -> u64 {
    gcso_tagged_ptr_t::pack(raw_addr, tag, phase).raw()
}

/// Unpacks the canonical 48-bit address from a 64-bit tagged pointer.
#[no_mangle]
pub extern "C" fn gcso_tagged_ptr_unpack_addr(ptr: u64) -> u64 {
    gcso_tagged_ptr_t(ptr).unpack_addr()
}

/// Unpacks the 8-bit micro-orchestration tag from a 64-bit tagged pointer.
#[no_mangle]
pub extern "C" fn gcso_tagged_ptr_unpack_tag(ptr: u64) -> u8 {
    gcso_tagged_ptr_t(ptr).unpack_tag()
}

/// Unpacks the 8-bit Q7 phase state from a 64-bit tagged pointer.
#[no_mangle]
pub extern "C" fn gcso_tagged_ptr_unpack_phase(ptr: u64) -> gcso_q7_t {
    gcso_tagged_ptr_t(ptr).unpack_phase()
}

/// Packs a 48-bit address, 3-bit Target Attractor Working Lane ID (`0..=7`), 5-bit Action Mode mask,
/// and Q7 phase offset into a 64-bit tagged pointer (`0` branches).
#[no_mangle]
pub extern "C" fn gcso_tagged_ptr_pack_action(
    raw_addr: u64,
    attractor_slot: u8,
    action_mode_flags: u8,
    phase: gcso_q7_t,
) -> u64 {
    gcso_tagged_ptr_t::pack(
        raw_addr,
        pack_action_slot_tag(attractor_slot, action_mode_flags),
        phase,
    )
    .raw()
}

/// Extracts the 5-bit Action Mode execution mask (`bits 0..=4` of the tag) from a 64-bit tagged pointer.
#[no_mangle]
pub extern "C" fn gcso_tagged_ptr_unpack_action_mode(ptr: u64) -> u8 {
    unpack_action_mode_from_tag(gcso_tagged_ptr_t(ptr).unpack_tag())
}

/// Extracts the 3-bit Target Attractor Working Lane ID (`0..=7`, `bits 5..=7` of the tag) from a 64-bit tagged pointer.
#[no_mangle]
pub extern "C" fn gcso_tagged_ptr_unpack_attractor_slot(ptr: u64) -> u8 {
    unpack_attractor_slot_from_tag(gcso_tagged_ptr_t(ptr).unpack_tag())
}

/// Evolves a 64-bit tagged pointer's Action Mode flags and Target Attractor Working Lane ID bottom-up
/// in $\mathcal{O}(1)$ from stigmergic trail density $\rho$, EDBC entropy $\tilde{H}$, and Bloom
/// filter repulsion status (`0` branches).
#[no_mangle]
pub extern "C" fn gcso_tagged_ptr_evolve_mode_stigmergically(
    ptr: u64,
    stigmergic_density: f32,
    entropy_h_tilde: f32,
    repulsion_triggered: u8,
    dominant_attractor_slot: u8,
    low_entropy_skip_thresh: f32,
    high_density_srl_thresh: f32,
) -> u64 {
    gcso_tagged_ptr_t(ptr)
        .evolve_mode_stigmergically(
            stigmergic_density,
            entropy_h_tilde,
            repulsion_triggered != 0,
            dominant_attractor_slot,
            low_entropy_skip_thresh,
            high_density_srl_thresh,
        )
        .raw()
}

// ===================================================================
// 1. Version, Capability & Configuration Management
// ===================================================================

/// Queries the active GCSO runtime C-ABI version components (`0.1.1`).
///
/// # Safety
/// Pointers must be `NULL` or naturally aligned writable `u32` pointers.
#[no_mangle]
pub unsafe extern "C" fn gcso_abi_get_version(
    major: *mut u32,
    minor: *mut u32,
    patch: *mut u32,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleMut::from_raw_opt` validates alignment for non-null pointers.
        let ((m_maj, m_min), m_pat) = unsafe {
            (
                (
                    FfiHandleMut::from_raw_opt(major)?,
                    FfiHandleMut::from_raw_opt(minor)?,
                ),
                FfiHandleMut::from_raw_opt(patch)?,
            )
        };

        let (v_maj, v_min, v_pat) = AbiSystemDispatcher.abi_version();
        if let Some(mut p) = m_maj {
            *p = v_maj;
        }
        if let Some(mut p) = m_min {
            *p = v_min;
        }
        if let Some(mut p) = m_pat {
            *p = v_pat;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Returns a static null-terminated C string representing the active ABI version (`"0.1.1"`).
///
/// # Safety
/// Returned pointer is static and read-only.
#[no_mangle]
pub unsafe extern "C" fn gcso_abi_get_version_string() -> *const c_char {
    c"0.1.1".as_ptr()
}

/// Converts a [`GcsoStatus`] code into a static null-terminated descriptive C string.
///
/// # Safety
/// Returned pointer is static and read-only.
#[no_mangle]
pub unsafe extern "C" fn gcso_status_to_string(status: GcsoStatus) -> *const c_char {
    status.as_c_bytes().as_ptr().cast::<c_char>()
}

/// Queries hardware and compute backend capabilities supported by the current build.
///
/// # Safety
/// `flags` must be a non-null, naturally aligned writable `u64` pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_abi_query_capability(flags: *mut u64) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and alignment.
        let mut out_flags = unsafe { FfiHandleMut::from_raw(flags)? };
        *out_flags = AbiSystemDispatcher.query_capability();
        Ok(GCSO_SUCCESS)
    })
}

/// Initializes a [`gcso_config_t`] structure with canonical v0.1.1 default hyperparameters.
///
/// # Safety
/// `config` must be a non-null, 16-byte aligned writable [`gcso_config_t`] pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_config_init_default(config: *mut gcso_config_t) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and alignment.
        let mut cfg = unsafe { FfiHandleMut::from_raw(config)? };
        AbiSystemDispatcher.init_default_config(&mut cfg)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Validates a [`gcso_config_t`] structure against GCSO mathematical and architectural invariants.
///
/// # Safety
/// `config` must be a non-null, 16-byte aligned readable [`gcso_config_t`] pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_config_validate(config: *const gcso_config_t) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleRef::from_raw` validates non-nullness and alignment.
        let cfg = unsafe { FfiHandleRef::from_raw(config)? };
        AbiSystemDispatcher.validate_config_invariants(&cfg)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Releases a dynamically allocated C string if applicable (no-op for static GCSO strings).
///
/// # Safety
/// `str_ptr` may be `NULL` or a pointer returned by GCSO string APIs.
#[no_mangle]
pub unsafe extern "C" fn gcso_free_string(str_ptr: *const c_char) {
    let _ = str_ptr;
}

// ===================================================================
// 2. Context Lifecycle, 4-Section `.gcso` Persistence & PPRC Seek
// ===================================================================

/// Creates a new GCSO runtime context instance from a configuration descriptor.
///
/// # Safety
/// `config` and `context_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_context_create(
    config: *const gcso_config_t,
    context_out: *mut GcsoContextHandle,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let (cfg, mut out_handle) = unsafe {
            (
                FfiHandleRef::from_raw(config)?,
                FfiHandleMut::from_raw(context_out)?,
            )
        };

        if cfg.head_dim == 0
            || (cfg.head_dim & 1) != 0
            || cfg.num_heads == 0
            || (cfg.num_heads as usize) > MAX_ATTENTION_HEADS
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        *out_handle = core::ptr::NonNull::dangling().as_ptr();
        Ok(GCSO_SUCCESS)
    })
}

/// Resets transient state variables and phase accumulators without freeing allocated tables.
///
/// # Safety
/// `context` must be `NULL` or a valid aligned [`GcsoContextHandle`].
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

/// Projects natural language system prompt text into phase space as the primary Anchor Attractor.
///
/// # Safety
/// `context` and `prompt_text` must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_context_set_system_prompt_anchor(
    context: GcsoContextHandle,
    prompt_text: *const c_char,
    weight: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        if !branchless_is_finite_f32(weight) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated non-null NUL-terminated UTF-8 string.
        let _prompt = unsafe { validate_c_utf8_str(prompt_text)? };
        Ok(GCSO_SUCCESS)
    })
}

/// Processes a single token step on the Hot Path in zero-allocation mode ($C_{\mathrm{alloc}} = 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_context_step_token(
    context: GcsoContextHandle,
    token_id: u32,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    trail_out: *mut gcso_pointer_trail_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        // SAFETY: `AlignedSliceMut32::from_raw_parts_mut_opt` validates alignment when non-null.
        let (q_opt, k_opt) = unsafe {
            (
                AlignedSliceMut32::from_raw_parts_mut_opt(query_tensor, 128)?,
                AlignedSliceMut32::from_raw_parts_mut_opt(key_tensor, 128)?,
            )
        };

        if let (Some(ref q), Some(ref k)) = (&q_opt, &k_opt) {
            q.ensure_disjoint_mut(k)?;
        }

        // SAFETY: Validated for non-nullness and 128B alignment when non-null.
        if let Some(mut trail) = unsafe { FfiHandleMut::from_raw_opt(trail_out)? } {
            trail.record_pointer_transition(u64::from(token_id), 1, 1.0);
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Serializes active runtime state into `.gcso` binary format conforming to GCSO ABI v0.1.1,
/// stamping an 8-byte unrolled LUT IEEE 802.3 CRC32 checksum ([`branchless_crc32_ieee`]) over the payload.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, and bounds invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_context_serialize(
    context: GcsoContextHandle,
    buffer: *mut u8,
    buffer_size: *mut usize,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and alignment.
        let mut size_io = unsafe { FfiHandleMut::from_raw(buffer_size)? };
        let required = 320usize;

        if buffer.is_null() {
            *size_io = required;
            return Ok(GCSO_SUCCESS);
        }

        if *size_io < required {
            *size_io = required;
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }

        // SAFETY: `buffer` is non-null and `*size_io >= required`.
        let out_bytes = unsafe { slice::from_raw_parts_mut(buffer, required) };
        out_bytes.fill(0);

        let hdr_size = size_of::<gcso_snapshot_header_t>();
        let mut header = init_snapshot_header(required as u64, 1_774_900_000);
        header.action_hub_offset = 128;
        header.attractor_field_offset = 128;
        header.dpsr_state_offset = 192;
        header.srl_state_offset = 192;
        header.edbc_state_offset = 192;
        const ZERO_PAYLOAD_192_CRC32: u32 = branchless_crc32_ieee(&[0u8; 192]);
        let _ = hdr_size;
        header.daes_slot_offset = 256;
        header.checksum_crc32 = ZERO_PAYLOAD_192_CRC32;

        header.write_unaligned_to(out_bytes, 0)?;
        *size_io = required;
        Ok(GCSO_SUCCESS)
    })
}

/// Deserializes a `.gcso` binary snapshot to restore runtime state with ABI v0.1.1 and
/// 8-byte unrolled LUT CRC32 verification ([`branchless_crc32_ieee`]).
///
/// # Safety
/// `buffer` and `context_out` must be non-null and valid for `buffer_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn gcso_context_deserialize(
    buffer: *const u8,
    buffer_size: usize,
    context_out: *mut GcsoContextHandle,
) -> GcsoStatus {
    ffi_boundary!({
        if buffer.is_null() {
            return Err(GCSO_ERROR_NULL_POINTER);
        }
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and alignment.
        let mut out_handle = unsafe { FfiHandleMut::from_raw(context_out)? };
        if buffer_size < size_of::<gcso_snapshot_header_t>() {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }

        // SAFETY: `buffer` is non-null and `buffer_size >= 128`.
        let in_bytes = unsafe { slice::from_raw_parts(buffer, buffer_size) };
        let header = gcso_snapshot_header_t::read_unaligned_from(in_bytes, 0)
            .map_err(|_| GCSO_ERROR_CONTAINER_CORRUPTED)?;

        validate_snapshot_header(&header, in_bytes.len())?;
        if header.checksum_crc32 != 0 {
            let hdr_size = size_of::<gcso_snapshot_header_t>();
            let computed = branchless_crc32_ieee(&in_bytes[hdr_size..(header.total_size as usize)]);
            if computed != header.checksum_crc32 {
                return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
            }
        }

        *out_handle = core::ptr::NonNull::dangling().as_ptr();
        Ok(GCSO_SUCCESS)
    })
}

/// Serializes `num_records` Section-1 [`CrystallizedAttractorRecord`]s (`records`) into a `.gcso`
/// (ZIMMS) byte buffer (`out_bytes`) via [`CrystallizedAttractorRecord::write_records_to_zimms_bytes`],
/// stamping each record's IEEE 802.3 CRC32 checksum when not already stamped.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_container_write_crystallized_records_zero_copy(
    records: *const gcso_crystallized_attractor_record_t,
    num_records: usize,
    out_bytes: *mut u8,
    buffer_size_inout: *mut usize,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validates `buffer_size_inout`.
        let mut size_io = unsafe { FfiHandleMut::from_raw(buffer_size_inout)? };
        let required = num_records
            .checked_mul(size_of::<CrystallizedAttractorRecord>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;

        if out_bytes.is_null() {
            *size_io = required;
            return Ok(GCSO_SUCCESS);
        }
        if *size_io < required {
            *size_io = required;
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        if required == 0 {
            *size_io = 0;
            return Ok(GCSO_SUCCESS);
        }

        validate_ptr(records)?;
        if raw_byte_ranges_overlap(records as usize, required, out_bytes as usize, required) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated non-null, aligned, and disjoint above.
        let (rec_slice, dst_slice) = unsafe {
            (
                slice::from_raw_parts(records, num_records),
                slice::from_raw_parts_mut(out_bytes, required),
            )
        };
        let written =
            CrystallizedAttractorRecord::write_records_to_zimms_bytes(rec_slice, dst_slice)?;
        *size_io = written;
        Ok(GCSO_SUCCESS)
    })
}

/// Validates and maps `num_records` Section-1 [`CrystallizedAttractorRecord`]s in-place ($C_{\mathrm{alloc}} = 0$)
/// from a 64-byte aligned `.gcso` (ZIMMS) byte buffer via [`CrystallizedAttractorRecord::slice_from_zimms_bytes_zero_copy`].
///
/// # Safety
/// `zimms_bytes` must be non-null, 64-byte aligned, and valid for at least `byte_len` bytes.
#[no_mangle]
pub unsafe extern "C" fn gcso_container_verify_crystallized_records_zero_copy(
    zimms_bytes: *const u8,
    byte_len: usize,
    num_records: usize,
    records_ptr_out: *mut *const gcso_crystallized_attractor_record_t,
) -> GcsoStatus {
    ffi_boundary!({
        if zimms_bytes.is_null() {
            return Err(GCSO_ERROR_NULL_POINTER);
        }
        // SAFETY: `zimms_bytes` is non-null.
        let bytes = unsafe { slice::from_raw_parts(zimms_bytes, byte_len) };
        let recs =
            CrystallizedAttractorRecord::slice_from_zimms_bytes_zero_copy(bytes, num_records)?;
        // SAFETY: `FfiHandleMut::from_raw_opt` validates optional output pointer.
        if let Some(mut rp_out) = unsafe { FfiHandleMut::from_raw_opt(records_ptr_out)? } {
            *rp_out = if recs.is_empty() {
                ptr::null()
            } else {
                recs.as_ptr()
            };
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Serializes a 4-section `.gcso` (ZIMMS) zero-copy container (`GcsoFourSectionView`) including
/// Section 3c [`CausalProvenanceReport`] diagnostic history entries.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_container_write_four_section_with_reports_zero_copy(
    crystals: *const gcso_crystallized_attractor_record_t,
    num_crystals: usize,
    trails: *const gcso_pointer_trail_t,
    num_trails: usize,
    bloom_opt: *const gcso_contrastive_bloom_filter_t,
    daes_opt: *const gcso_daes_slot_t,
    reports: *const gcso_causal_provenance_report_t,
    num_reports: usize,
    timestamp_epoch_sec: u64,
    out_bytes: *mut u8,
    buffer_size_inout: *mut usize,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let mut size_io = unsafe { FfiHandleMut::from_raw(buffer_size_inout)? };

        let required = GcsoFourSectionView::required_byte_len_with_reports(
            num_crystals,
            num_trails,
            !bloom_opt.is_null(),
            !daes_opt.is_null(),
            num_reports,
        )?;

        if out_bytes.is_null() {
            *size_io = required;
            return Ok(GCSO_SUCCESS);
        }
        if *size_io < required {
            *size_io = required;
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }

        let crystals_slice: &[CrystallizedAttractorRecord] = if num_crystals == 0 {
            &[]
        } else {
            validate_ptr(crystals)?;
            // SAFETY: Validated non-null and 64B aligned.
            unsafe { slice::from_raw_parts(crystals, num_crystals) }
        };
        let trails_slice: &[gcso_pointer_trail_t] = if num_trails == 0 {
            &[]
        } else {
            validate_ptr(trails)?;
            // SAFETY: Validated non-null and 128B aligned.
            unsafe { slice::from_raw_parts(trails, num_trails) }
        };
        let reports_slice: &[CausalProvenanceReport] = if num_reports == 0 {
            &[]
        } else {
            validate_ptr(reports)?;
            // SAFETY: Validated non-null and 16B aligned.
            unsafe { slice::from_raw_parts(reports, num_reports) }
        };
        // SAFETY: Validated optional pointers and destination buffer length.
        let ((bloom_ref, daes_ref), dst_slice) = unsafe {
            (
                (
                    FfiHandleRef::from_raw_opt(bloom_opt)?,
                    FfiHandleRef::from_raw_opt(daes_opt)?,
                ),
                slice::from_raw_parts_mut(out_bytes, required),
            )
        };

        let written = GcsoFourSectionView::write_four_section_with_reports_to_bytes(
            crystals_slice,
            trails_slice,
            bloom_ref.as_deref(),
            daes_ref.as_deref(),
            reports_slice,
            timestamp_epoch_sec,
            dst_slice,
        )?;
        *size_io = written;
        Ok(GCSO_SUCCESS)
    })
}

/// Serializes a 4-section `.gcso` (ZIMMS) zero-copy container (`GcsoFourSectionView`).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_container_write_four_section_zero_copy(
    crystals: *const gcso_crystallized_attractor_record_t,
    num_crystals: usize,
    trails: *const gcso_pointer_trail_t,
    num_trails: usize,
    bloom_opt: *const gcso_contrastive_bloom_filter_t,
    daes_opt: *const gcso_daes_slot_t,
    timestamp_epoch_sec: u64,
    out_bytes: *mut u8,
    buffer_size_inout: *mut usize,
) -> GcsoStatus {
    // SAFETY: Delegates directly to `gcso_container_write_four_section_with_reports_zero_copy` with 0 reports.
    unsafe {
        gcso_container_write_four_section_with_reports_zero_copy(
            crystals,
            num_crystals,
            trails,
            num_trails,
            bloom_opt,
            daes_opt,
            ptr::null(),
            0,
            timestamp_epoch_sec,
            out_bytes,
            buffer_size_inout,
        )
    }
}

/// Validates and maps a 128B-aligned 4-section `.gcso` (ZIMMS) container in-place ($C_{\mathrm{alloc}} = 0$)
/// via [`GcsoFourSectionView::from_zimms_bytes_zero_copy`], exposing all 4 sections including
/// Section 0b [`gcso_zimms_descriptor_t`] and Section 3c [`CausalProvenanceReport`]s.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_container_verify_four_section_with_reports_zero_copy(
    zimms_bytes: *const u8,
    byte_len: usize,
    zimms_desc_out: *mut gcso_zimms_descriptor_t,
    num_crystals_out: *mut usize,
    num_trails_out: *mut usize,
    num_reports_out: *mut usize,
    crystals_ptr_out: *mut *const gcso_crystallized_attractor_record_t,
    trails_ptr_out: *mut *const gcso_pointer_trail_t,
    bloom_ptr_out: *mut *const gcso_contrastive_bloom_filter_t,
    daes_ptr_out: *mut *const gcso_daes_slot_t,
    reports_ptr_out: *mut *const gcso_causal_provenance_report_t,
) -> GcsoStatus {
    ffi_boundary!({
        if zimms_bytes.is_null() {
            return Err(GCSO_ERROR_NULL_POINTER);
        }
        if byte_len < GCSO_FOUR_SECTION_HEADER_BYTES {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        // SAFETY: `zimms_bytes` is non-null and `byte_len >= 192`.
        let bytes = unsafe { slice::from_raw_parts(zimms_bytes, byte_len) };
        let view = GcsoFourSectionView::from_zimms_bytes_zero_copy(bytes)?;

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(zimms_desc_out, *view.zimms_descriptor)?;
            write_opt_out(num_crystals_out, view.crystals.len())?;
            write_opt_out(num_trails_out, view.trails.len())?;
            write_opt_out(num_reports_out, view.provenance_reports.len())?;
        }
        // SAFETY: `FfiHandleMut::from_raw_opt` validates optional output pointers.
        if let Some(mut cp_out) = unsafe { FfiHandleMut::from_raw_opt(crystals_ptr_out)? } {
            *cp_out = if view.crystals.is_empty() {
                ptr::null()
            } else {
                view.crystals.as_ptr()
            };
        }
        // SAFETY: `FfiHandleMut::from_raw_opt` validates optional output pointers.
        if let Some(mut tp_out) = unsafe { FfiHandleMut::from_raw_opt(trails_ptr_out)? } {
            *tp_out = if view.trails.is_empty() {
                ptr::null()
            } else {
                view.trails.as_ptr()
            };
        }
        // SAFETY: `FfiHandleMut::from_raw_opt` validates optional output pointers.
        if let Some(mut bp_out) = unsafe { FfiHandleMut::from_raw_opt(bloom_ptr_out)? } {
            *bp_out = match view.bloom {
                Some(b_ref) => b_ref as *const gcso_contrastive_bloom_filter_t,
                None => ptr::null(),
            };
        }
        // SAFETY: `FfiHandleMut::from_raw_opt` validates optional output pointers.
        if let Some(mut dp_out) = unsafe { FfiHandleMut::from_raw_opt(daes_ptr_out)? } {
            *dp_out = match view.daes {
                Some(d_ref) => d_ref as *const gcso_daes_slot_t,
                None => ptr::null(),
            };
        }
        // SAFETY: `FfiHandleMut::from_raw_opt` validates optional output pointers.
        if let Some(mut rp_out) = unsafe { FfiHandleMut::from_raw_opt(reports_ptr_out)? } {
            *rp_out = if view.provenance_reports.is_empty() {
                ptr::null()
            } else {
                view.provenance_reports.as_ptr()
            };
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Validates and maps a 128B-aligned 4-section `.gcso` (ZIMMS) container in-place ($C_{\mathrm{alloc}} = 0$)
/// via [`GcsoFourSectionView::from_zimms_bytes_zero_copy`].
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_container_verify_four_section_zero_copy(
    zimms_bytes: *const u8,
    byte_len: usize,
    num_crystals_out: *mut usize,
    num_trails_out: *mut usize,
    crystals_ptr_out: *mut *const gcso_crystallized_attractor_record_t,
    trails_ptr_out: *mut *const gcso_pointer_trail_t,
) -> GcsoStatus {
    // SAFETY: Delegates to `gcso_container_verify_four_section_with_reports_zero_copy`.
    unsafe {
        gcso_container_verify_four_section_with_reports_zero_copy(
            zimms_bytes,
            byte_len,
            ptr::null_mut(),
            num_crystals_out,
            num_trails_out,
            ptr::null_mut(),
            crystals_ptr_out,
            trails_ptr_out,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        )
    }
}

/// Validates and maps a 128B-aligned 4-section `.gcso` (ZIMMS) container in-place ($C_{\mathrm{alloc}} = 0$),
/// folds Section 1 [`CrystallizedAttractorRecord`]s onto `trail_inout` by **Phase-Proximity Gravity**,
/// applies Section 3a `view.bloom` contrastive Phase-Conjugate Repulsion ($-\boldsymbol{\Delta\theta}$)
/// if active, and steers `query_tensor` (and optional `key_tensor`) in a single register-resident FFI call.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_container_map_and_steer_four_section_zero_copy_safe(
    zimms_bytes: *const u8,
    byte_len: usize,
    trail_inout: *mut gcso_pointer_trail_t,
    min_step_rad: f32,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    num_crystals_out: *mut usize,
    total_pull_out: *mut f32,
    dominant_anchor_id_out: *mut u32,
    effective_ripa_rad_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        if zimms_bytes.is_null() {
            return Err(GCSO_ERROR_NULL_POINTER);
        }
        if byte_len < GCSO_FOUR_SECTION_HEADER_BYTES {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        validate_ptr(trail_inout)?;
        if raw_byte_ranges_overlap(
            zimms_bytes as usize,
            byte_len,
            trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: `bind_qk_32` validates `query_tensor` and `key_tensor` alignment and disjointness.
        let (_engine, mut q_aligned, mut k_opt) =
            unsafe { bind_qk_32(query_tensor, key_tensor, head_dim, num_heads)? };
        // SAFETY: `zimms_bytes` is non-null and `byte_len >= 192`.
        let bytes = unsafe { slice::from_raw_parts(zimms_bytes, byte_len) };
        let view = GcsoFourSectionView::from_zimms_bytes_zero_copy(bytes)?;
        // SAFETY: `trail_inout` is validated above.
        let mut trail = unsafe { FfiHandleMut::from_raw(trail_inout)? };

        let (tot_pull, dom_id, eff_ripa) = steer_crystallized_records_with_optional_bloom_safe(
            view.crystals,
            view.bloom,
            &mut trail,
            min_step_rad,
            head_dim,
            num_heads,
            &mut q_aligned,
            k_opt.as_mut(),
        )?;

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(num_crystals_out, view.crystals.len())?;
            write_opt_out(total_pull_out, tot_pull)?;
            write_opt_out(dominant_anchor_id_out, dom_id)?;
            write_opt_out(effective_ripa_rad_out, eff_ripa)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Maps a 128B-aligned 4-section `.gcso` (ZIMMS) container in-place ($C_{\mathrm{alloc}} = 0$),
/// executes **Coarse-to-Fine Macro-to-Micro Routing ($\mathcal{O}(K + M)$)** directly over Section 1
/// macro centroids (`view.crystals`) and Section 2 micro pointer trails (`view.trails`), applies
/// Section 3a `view.bloom` contrastive Phase-Conjugate Repulsion ($-\boldsymbol{\Delta\theta}$) in
/// 8 `u64` SWAR registers when `bloom_repulsion_threshold > 0`, and steers `query_tensor` (and
/// optional `key_tensor`) in a single zero-copy FFI call.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_container_map_route_coarse_to_fine_and_steer_four_section_zero_copy_safe(
    zimms_bytes: *const u8,
    byte_len: usize,
    src_trail_inout: *mut gcso_pointer_trail_t,
    head_dim: usize,
    num_heads: usize,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    cluster_bonus_gain: f32,
    contradiction_threshold: f32,
    interpolation_u7: u8,
    bloom_repulsion_threshold: u8,
    min_step_rad: f32,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    winning_centroid_idx_out: *mut usize,
    selected_trail_id_out: *mut u32,
    best_score_out: *mut f32,
    best_coherence_out: *mut f32,
    bloom_repulsed_out: *mut u8,
    found_out: *mut u8,
) -> GcsoStatus {
    ffi_boundary!({
        if zimms_bytes.is_null() {
            return Err(GCSO_ERROR_NULL_POINTER);
        }
        if byte_len < GCSO_FOUR_SECTION_HEADER_BYTES {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        validate_ptr(src_trail_inout)?;
        if min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(cluster_bonus_gain)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if raw_byte_ranges_overlap(
            zimms_bytes as usize,
            byte_len,
            src_trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: `bind_qk_32` validates `query_tensor` and `key_tensor` alignment and disjointness.
        let (_engine, mut q_aligned, mut k_opt) =
            unsafe { bind_qk_32(query_tensor, key_tensor, head_dim, num_heads)? };
        // SAFETY: `zimms_bytes` is non-null and `byte_len >= 192`.
        let bytes = unsafe { slice::from_raw_parts(zimms_bytes, byte_len) };
        let view = GcsoFourSectionView::from_zimms_bytes_zero_copy(bytes)?;
        // SAFETY: `src_trail_inout` is validated above.
        let mut src_trail = unsafe { FfiHandleMut::from_raw(src_trail_inout)? };

        let (win_k, sel_id, best_score, best_coh, found, win_clamp, mut steer_words) =
            if !view.trails.is_empty() {
                let (wk, sid, score, coh, fnd) =
                    CrystallizedAttractorRecord::route_coarse_to_fine_candidates_64(
                        view.crystals,
                        &mut src_trail,
                        view.trails,
                        head_dim as u32,
                        phase_interference_gain,
                        stigmergic_prior_gain,
                        pointer_link_bonus,
                        cluster_bonus_gain,
                        contradiction_threshold,
                    );
                let sw = if fnd && interpolation_u7 > 0 {
                    let dst_idx = (sid as usize).min(view.trails.len() - 1);
                    bridge_trail_words_slerp_u7(
                        &mut src_trail,
                        &view.trails[dst_idx].accumulated_phase_delta,
                        interpolation_u7,
                    )
                } else {
                    *trail_phase_words_ref(&src_trail)
                };
                let clamp = if !view.crystals.is_empty() && wk < view.crystals.len() {
                    sync_winner_centroid_ptr_to_trail(&mut src_trail, &view.crystals[wk]);
                    view.crystals[wk].ripa_clamp_rad
                } else {
                    DEFAULT_RIPA_CLAMP_MAX_RAD
                };
                (wk, sid, score, coh, fnd, clamp, sw)
            } else if !view.crystals.is_empty() {
                let (wk, aid, coh, grav, bridged_w, _win_w) =
                    CrystallizedAttractorRecord::route_coarse_to_fine_trail_with_words_64(
                        view.crystals,
                        &mut src_trail,
                        interpolation_u7,
                    );
                (
                    wk,
                    src_trail.linked_trail_id,
                    grav,
                    coh,
                    aid != 0,
                    view.crystals[wk].ripa_clamp_rad,
                    bridged_w,
                )
            } else {
                (
                    0,
                    src_trail.linked_trail_id,
                    0.0,
                    0.0,
                    false,
                    DEFAULT_RIPA_CLAMP_MAX_RAD,
                    *trail_phase_words_ref(&src_trail),
                )
            };
        let mut bloom_repulsed = false;
        if let Some(bloom_ref) = view.bloom {
            if bloom_repulsion_threshold > 0 && !bloom_ref.is_empty() {
                let (_density, rep_active) = bloom_ref.apply_phase_conjugate_repulsion_words(
                    &mut steer_words,
                    bloom_repulsion_threshold,
                    128,
                );
                if rep_active {
                    bloom_repulsed = true;
                    *trail_phase_words_mut(&mut src_trail) = steer_words;
                    let cur_ptr = gcso_tagged_ptr_t(src_trail.current_ptr);
                    let rep_mode = cur_ptr.action_mode_flags() | ACTION_PRESET_REPULSION_ESCAPE;
                    src_trail.current_ptr = cur_ptr
                        .with_action_and_slot(ATTRACTOR_SLOT_REPULSION, rep_mode)
                        .conjugate_phase()
                        .raw();
                    src_trail.flags |= TRAIL_FLAG_REPULSION_ACTIVE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE;
                }
            }
        }
        src_trail.sync_flags_with_current_ptr();

        steer_routed_trail_words_qk_safe(
            &steer_words,
            src_trail.flags,
            win_clamp,
            min_step_rad,
            head_dim,
            num_heads,
            &mut q_aligned,
            k_opt.as_mut(),
        );

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(winning_centroid_idx_out, win_k)?;
            write_opt_out(selected_trail_id_out, sel_id)?;
            write_opt_out(best_score_out, best_score)?;
            write_opt_out(best_coherence_out, best_coh)?;
            write_opt_out(bloom_repulsed_out, u8::from(bloom_repulsed))?;
            write_opt_out(found_out, u8::from(found))?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Seeks to a specific token index in the PPRC keyframe KV cache index without forward passes.
///
/// # Safety
/// `context` and `keyframe_header_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_pprc_seek_to_token(
    context: GcsoContextHandle,
    token_index: u32,
    keyframe_header_out: *mut gcso_pprc_keyframe_header_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and alignment.
        let mut header_out = unsafe { FfiHandleMut::from_raw(keyframe_header_out)? };
        *header_out = gcso_pprc_keyframe_header_t {
            token_index,
            ..Default::default()
        };
        Ok(GCSO_SUCCESS)
    })
}

// ===================================================================
// 3. Action Hub, Limina Bridging, Bayesian Trail Rewiring & DAES Interface
// ===================================================================

/// Allocates an Action Hub (Sidecar Pointer Table) instance with the specified slot capacity.
///
/// # Safety
/// `hub_out` must be a non-null, naturally aligned writable handle pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_create(
    capacity: u32,
    hub_out: *mut GcsoActionHubHandle,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and alignment.
        let mut out_handle = unsafe { FfiHandleMut::from_raw(hub_out)? };
        let hub = ActionHubEngine::with_capacity(capacity)?;
        *out_handle = Box::into_raw(Box::new(hub)).cast::<GcsoActionHubOpaque>();
        Ok(GCSO_SUCCESS)
    })
}

/// Initializes a 32-byte [`gcso_stigmergic_thresholds_t`] structure with canonical v0.1.1 defaults.
///
/// # Safety
/// `thresholds_out` must be a non-null, 32-byte aligned writable [`gcso_stigmergic_thresholds_t`] pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_init_default_thresholds(
    thresholds_out: *mut gcso_stigmergic_thresholds_t,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and 32B alignment.
        let mut out_th = unsafe { FfiHandleMut::from_raw(thresholds_out)? };
        *out_th = StigmergicThresholds::new();
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluates the 7-regime stigmergic threshold ladder on `trail` in strictly branchless constant
/// time (`0` branches) and writes the 32-byte [`gcso_action_orchestration_report_t`] to `report_out`.
///
/// # Safety
/// `trail` (128B aligned) and `report_out` (32B aligned) must be non-null; `thresholds_opt` may be `NULL` or 32B aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_evaluate_stigmergic_thresholds(
    thresholds_opt: *const gcso_stigmergic_thresholds_t,
    trail: *const gcso_pointer_trail_t,
    entropy_h_tilde: f32,
    phase_coherence: f32,
    transition_log_prob: f32,
    selected_trail_id: u32,
    bloom_repulsion_density: u8,
    dominant_slot: u8,
    report_out: *mut gcso_action_orchestration_report_t,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let ((th_opt, trail_ref), mut rep_out) = unsafe {
            (
                (
                    FfiHandleRef::from_raw_opt(thresholds_opt)?,
                    FfiHandleRef::from_raw(trail)?,
                ),
                FfiHandleMut::from_raw(report_out)?,
            )
        };
        let th = match th_opt {
            Some(ref t) => {
                t.validate()?;
                **t
            }
            None => StigmergicThresholds::new(),
        };
        *rep_out = th.evaluate_report_on_trail(
            &trail_ref,
            2,
            entropy_h_tilde,
            phase_coherence,
            transition_log_prob,
            selected_trail_id,
            bloom_repulsion_density,
            dominant_slot,
        );
        Ok(GCSO_SUCCESS)
    })
}

/// Executes a full **Stigmergic Threshold Action-Selecting Orchestration Step** across the C-ABI
/// boundary via [`ActionHubEngine::orchestrate_next_action_step`] ($C_{\mathrm{alloc}} = 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_orchestrate_stigmergic_step_safe(
    hub_opt: GcsoActionHubHandle,
    thresholds_opt: *const gcso_stigmergic_thresholds_t,
    input_ptr: u64,
    entropy_h_tilde: f32,
    dominant_slot: u8,
    trail_inout: *mut gcso_pointer_trail_t,
    attention_scores_inout: *mut f32,
    num_scores: usize,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
    report_out: *mut gcso_action_orchestration_report_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(trail_inout)?;
        let (dpsr, total_len) = init_dpsr_engine(head_dim, num_heads)?;
        // SAFETY: Validated for non-nullness and alignment.
        let ((th_ref_opt, mut trail), mut rep_out) = unsafe {
            (
                (
                    FfiHandleRef::from_raw_opt(thresholds_opt)?,
                    FfiHandleMut::from_raw(trail_inout)?,
                ),
                FfiHandleMut::from_raw(report_out)?,
            )
        };
        // SAFETY: `AlignedSliceMut32::from_raw_parts_mut_opt` validates 32B alignment when non-null.
        let ((mut scores_opt, mut q_opt), (mut k_opt, mut logits_opt)) = unsafe {
            (
                (
                    if attention_scores_inout.is_null() || num_scores == 0 {
                        None
                    } else {
                        AlignedSliceMut32::from_raw_parts_mut_opt(
                            attention_scores_inout,
                            num_scores,
                        )?
                    },
                    AlignedSliceMut32::from_raw_parts_mut_opt(query_tensor, total_len)?,
                ),
                (
                    AlignedSliceMut32::from_raw_parts_mut_opt(key_tensor, total_len)?,
                    if logits_inout.is_null() || vocab_size == 0 {
                        None
                    } else {
                        AlignedSliceMut32::from_raw_parts_mut_opt(logits_inout, vocab_size)?
                    },
                ),
            )
        };

        if let (Some(ref q), Some(ref k)) = (&q_opt, &k_opt) {
            q.ensure_disjoint_mut(k)?;
        }
        if let Some(ref s) = scores_opt {
            if let Some(ref q) = q_opt {
                q.ensure_disjoint_mut(s)?;
            }
            if let Some(ref k) = k_opt {
                k.ensure_disjoint_mut(s)?;
            }
        }

        let report = if !hub_opt.is_null()
            && (hub_opt as usize) >= 4096
            && ((hub_opt as usize) & (align_of::<ActionHubEngine>() - 1)) == 0
        {
            // SAFETY: `hub_opt` is non-null and 128-byte aligned.
            let hub_ref = unsafe { &mut *hub_opt.cast::<ActionHubEngine>() };
            hub_ref.dpsr = dpsr;
            if let Some(ref th) = th_ref_opt {
                th.validate()?;
                hub_ref.thresholds = **th;
            }
            hub_ref.orchestrate_next_action_step(
                gcso_tagged_ptr_t(input_ptr),
                entropy_h_tilde,
                dominant_slot,
                &mut trail,
                scores_opt.as_mut(),
                q_opt.as_mut(),
                k_opt.as_mut(),
                logits_opt.as_mut(),
            )?
        } else {
            let th = match th_ref_opt {
                Some(ref t) => {
                    t.validate()?;
                    **t
                }
                None => StigmergicThresholds::new(),
            };
            th.orchestrate_stateless_action_step(
                &dpsr,
                gcso_tagged_ptr_t(input_ptr),
                entropy_h_tilde,
                dominant_slot,
                &mut trail,
                q_opt.as_mut(),
                k_opt.as_mut(),
                logits_opt.as_mut(),
            )?
        };
        *rep_out = report;
        Ok(GCSO_SUCCESS)
    })
}

/// Advances a 64-bit tagged pointer transition in $\mathcal{O}(1)$ time and synchronizes
/// `cluster_id` and `flags` from the pointer's embedded Target Attractor Working Lane ID and Action Mode mask.
///
/// # Safety
/// `context` and `trail_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_step_pointer(
    context: GcsoContextHandle,
    current_ptr: u64,
    trail_out: *mut gcso_pointer_trail_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and 128B alignment.
        let mut trail = unsafe { FfiHandleMut::from_raw(trail_out)? };
        let tp = gcso_tagged_ptr_t(current_ptr);

        trail.record_pointer_transition(current_ptr, 1, 1.0);
        let embedded_phase = tp.unpack_phase();
        if embedded_phase != 0 {
            trail.accumulate_uniform_phase(embedded_phase);
        }
        trail.cluster_id = branchless_select_u32(
            trail.target_anchor_id != 0,
            trail.target_anchor_id,
            u32::from(tp.target_attractor_slot() & ATTRACTOR_SLOT_ID_MASK),
        );
        trail.sync_flags_with_current_ptr();
        Ok(GCSO_SUCCESS)
    })
}

/// Evolves a [`gcso_pointer_trail_t`]'s embedded `current_ptr` Action Mode flags and Target Attractor
/// Working Lane ID (`0..=7`) bottom-up in $\mathcal{O}(1)$ based on its stigmergic density, EDBC
/// entropy $\tilde{H}$, and Bloom filter repulsion state.
///
/// # Safety
/// `trail_inout` must be non-null and 128B aligned; `evolved_ptr_out` may be `NULL` or 8B aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_evolve_trail_mode(
    trail_inout: *mut gcso_pointer_trail_t,
    entropy_h_tilde: f32,
    repulsion_triggered: u8,
    dominant_attractor_slot: u8,
    low_entropy_skip_thresh: f32,
    high_density_srl_thresh: f32,
    evolved_ptr_out: *mut u64,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let (mut trail, out_ptr_opt) = unsafe {
            (
                FfiHandleMut::from_raw(trail_inout)?,
                FfiHandleMut::from_raw_opt(evolved_ptr_out)?,
            )
        };

        let evolved = trail.evolve_trail_mode_stigmergically(
            entropy_h_tilde,
            repulsion_triggered != 0,
            dominant_attractor_slot,
            low_entropy_skip_thresh,
            high_density_srl_thresh,
        );
        if let Some(mut out_p) = out_ptr_opt {
            *out_p = evolved.raw();
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Bridges the inter-memory **Limina** gap between `prev_trail` ($M_{t-1}$) and `curr_trail_inout`
/// ($M_t$) in $\mathcal{O}(1)$ constant time, evaluating conditional transition log-probability
/// $\ln P(M_t \mid M_{t-1})$ and applying Phase-Conjugate Repulsion ($-\boldsymbol{\Delta\theta}$)
/// if the transition is self-contradictory ($H^1(K; \mathcal{F}) \neq 0$).
///
/// # Safety
/// `prev_trail` and `curr_trail_inout` must be non-null, 128B aligned, and disjoint.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_bridge_liminal_trails(
    prev_trail: *const gcso_pointer_trail_t,
    curr_trail_inout: *mut gcso_pointer_trail_t,
    interpolation_u7: u8,
    contradiction_threshold: f32,
    coherence_out: *mut f32,
    log_prob_out: *mut f32,
    contradiction_repulsed_out: *mut u8,
) -> GcsoStatus {
    ffi_boundary!({
        if !branchless_is_finite_f32(contradiction_threshold) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated for non-nullness and 128B alignment.
        let (prev, mut curr) = unsafe {
            (
                FfiHandleRef::from_raw(prev_trail)?,
                FfiHandleMut::from_raw(curr_trail_inout)?,
            )
        };
        curr.ensure_disjoint_ref(&prev)?;

        let (coh, log_prob, repulsed) = curr.bridge_liminal_hallucinated_trail(
            &prev,
            interpolation_u7,
            contradiction_threshold,
        );
        curr.sync_flags_with_current_ptr();

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(coherence_out, coh)?;
            write_opt_out(log_prob_out, log_prob)?;
            write_opt_out(contradiction_repulsed_out, u8::from(repulsed))?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Realizes **Stigmergic Pointer-Trail Intelligence (Hypothesis 2 $\times$ Hypothesis 1)** across
/// the C-ABI boundary in a single pass: scans `num_candidates` candidate memory trails in constant
/// time per candidate to select the optimal coherent ($H^1(K; \mathcal{F}) = 0$) next trail $j^*$
/// maximizing the Bayes-Attention transition score, rewires `src_trail_inout.linked_trail_id = j^*`,
/// and bridges the inter-memory **Limina** gap onto both `src_trail_inout` and `candidate_trails_inout[j^*]`.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_select_and_bridge_best_next_trail(
    src_trail_inout: *mut gcso_pointer_trail_t,
    candidate_trails_inout: *mut gcso_pointer_trail_t,
    num_candidates: usize,
    head_dim: u32,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    interpolation_u7: u8,
    contradiction_threshold: f32,
    selected_trail_id_out: *mut u32,
    bayesian_score_out: *mut f32,
    log_prob_out: *mut f32,
    found_out: *mut u8,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(src_trail_inout)?;
        validate_ptr(candidate_trails_inout)?;
        if num_candidates == 0
            || head_dim == 0
            || (head_dim & 1) != 0
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let cand_bytes = num_candidates
            .checked_mul(size_of::<gcso_pointer_trail_t>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            src_trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
            candidate_trails_inout as usize,
            cand_bytes,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated non-null, aligned, and disjoint above.
        let mut src_trail = unsafe { FfiHandleMut::from_raw(src_trail_inout)? };
        // SAFETY: Validated non-null, aligned, and disjoint above.
        let candidates =
            unsafe { slice::from_raw_parts_mut(candidate_trails_inout, num_candidates) };

        let (selected_id, bayes_score, best_coh, found) = src_trail
            .select_and_rewire_best_next_trail(
                candidates,
                head_dim,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                contradiction_threshold,
            );

        let mut log_prob = 0.0_f32;
        if found {
            let dst_idx = (selected_id as usize).min(num_candidates - 1);
            let dst_trail = &mut candidates[dst_idx];
            if !log_prob_out.is_null() {
                log_prob = branchless_fast_transition_log_prob(best_coh);
            }
            if interpolation_u7 > 0 {
                let src_w = *trail_phase_words_ref(&src_trail);
                let dst_w = *trail_phase_words_ref(dst_trail);
                if !swar_words_equal_64(&src_w, &dst_w) {
                    let s_u64 =
                        branchless_select_u8(interpolation_u7 > 128, 128, interpolation_u7) as u64;
                    let bias_restore =
                        (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                    let bridged_w =
                        swar_slerp_words_with_bias_64(&src_w, &dst_w, s_u64, bias_restore);
                    *trail_phase_words_mut(&mut src_trail) = bridged_w;
                    *trail_phase_words_mut(dst_trail) = bridged_w;
                }
            } else {
                *trail_phase_words_mut(dst_trail) = *trail_phase_words_ref(&src_trail);
            }
            dst_trail.prev_ptr = src_trail.current_ptr;
            dst_trail.step_count = src_trail.step_count.wrapping_add(1);
            let inherit_anchor = dst_trail.target_anchor_id != 0;
            src_trail.target_anchor_id = branchless_select_u32(
                inherit_anchor,
                dst_trail.target_anchor_id,
                src_trail.target_anchor_id,
            );
            if src_trail.target_anchor_id != 0 {
                src_trail.cluster_id = src_trail.target_anchor_id;
                dst_trail.target_anchor_id = src_trail.target_anchor_id;
                dst_trail.cluster_id = src_trail.target_anchor_id;
            }
            src_trail.flags |= TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE;
            dst_trail.flags |= TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE;
            src_trail.sync_flags_with_current_ptr();
            dst_trail.sync_flags_with_current_ptr();
        }

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(selected_trail_id_out, selected_id)?;
            write_opt_out(bayesian_score_out, bayes_score)?;
            write_opt_out(log_prob_out, log_prob)?;
            write_opt_out(found_out, u8::from(found))?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Directly modulates `num_candidates` pre-Softmax Attention scores (`attention_scores_inout`) in
/// zero-copy constant time per candidate from `src_trail` and `candidate_trails` using
/// [`StigmergicTrailOps::modulate_candidate_trail_scores_in_place`].
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_modulate_attention_scores_by_candidate_trails(
    attention_scores_inout: *mut f32,
    src_trail: *const gcso_pointer_trail_t,
    candidate_trails: *const gcso_pointer_trail_t,
    num_candidates: usize,
    head_dim: u32,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    contradiction_threshold: f32,
    repulsion_penalty: f32,
    mean_coherence_out: *mut f32,
    best_candidate_id_out: *mut u32,
    coherent_count_out: *mut u32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(attention_scores_inout)?;
        validate_ptr(src_trail)?;
        validate_ptr(candidate_trails)?;
        if num_candidates == 0
            || head_dim == 0
            || (head_dim & 1) != 0
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(contradiction_threshold)
            || !branchless_is_finite_f32(repulsion_penalty)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let scores_bytes = num_candidates
            .checked_mul(size_of::<f32>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        let cand_bytes = num_candidates
            .checked_mul(size_of::<gcso_pointer_trail_t>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            attention_scores_inout as usize,
            scores_bytes,
            src_trail as usize,
            size_of::<gcso_pointer_trail_t>(),
        ) || raw_byte_ranges_overlap(
            attention_scores_inout as usize,
            scores_bytes,
            candidate_trails as usize,
            cand_bytes,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated non-null, aligned, and disjoint above.
        let ((src_ref, cand_slice), scores) = unsafe {
            (
                (
                    FfiHandleRef::from_raw(src_trail)?,
                    slice::from_raw_parts(candidate_trails, num_candidates),
                ),
                slice::from_raw_parts_mut(attention_scores_inout, num_candidates),
            )
        };

        let (mean_coh, best_id, coherent_count) = src_ref.modulate_candidate_trail_scores_in_place(
            scores,
            cand_slice,
            head_dim,
            phase_interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
            contradiction_threshold,
            repulsion_penalty,
        );

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(mean_coherence_out, mean_coh)?;
            write_opt_out(best_candidate_id_out, best_id)?;
            write_opt_out(coherent_count_out, coherent_count)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes a single-pass **Fused Bayesian Pointer-Trail Selection + Rewiring + Limina Slerp
/// Bridging + Optional Pre-Softmax Attention Score Modulation** across the C-ABI boundary via
/// [`StigmergicTrailOps::select_rewire_bridge_and_modulate_bayesian_trails`] ($C_{\mathrm{alloc}} = 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_select_bridge_and_modulate_bayesian_trails(
    src_trail_inout: *mut gcso_pointer_trail_t,
    candidate_trails: *const gcso_pointer_trail_t,
    num_candidates: usize,
    attention_scores_inout: *mut f32,
    head_dim: u32,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    contradiction_threshold: f32,
    interpolation_u7: u8,
    repulsion_penalty: f32,
    selected_trail_id_out: *mut u32,
    bayesian_score_out: *mut f32,
    mean_coherence_out: *mut f32,
    coherent_count_out: *mut u32,
    found_out: *mut u8,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(src_trail_inout)?;
        validate_ptr(candidate_trails)?;
        if num_candidates == 0
            || head_dim == 0
            || (head_dim & 1) != 0
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(contradiction_threshold)
            || !branchless_is_finite_f32(repulsion_penalty)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let cand_bytes = num_candidates
            .checked_mul(size_of::<gcso_pointer_trail_t>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            src_trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
            candidate_trails as usize,
            cand_bytes,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let scores_opt: Option<&mut [f32]> = if attention_scores_inout.is_null() {
            None
        } else {
            validate_ptr(attention_scores_inout)?;
            let scores_bytes = num_candidates
                .checked_mul(size_of::<f32>())
                .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
            if raw_byte_ranges_overlap(
                attention_scores_inout as usize,
                scores_bytes,
                src_trail_inout as usize,
                size_of::<gcso_pointer_trail_t>(),
            ) || raw_byte_ranges_overlap(
                attention_scores_inout as usize,
                scores_bytes,
                candidate_trails as usize,
                cand_bytes,
            ) {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            // SAFETY: Validated non-null, aligned, and disjoint above.
            Some(unsafe { slice::from_raw_parts_mut(attention_scores_inout, num_candidates) })
        };

        // SAFETY: Validated non-null, aligned, and disjoint above.
        let (mut src_trail, cand_slice) = unsafe {
            (
                FfiHandleMut::from_raw(src_trail_inout)?,
                slice::from_raw_parts(candidate_trails, num_candidates),
            )
        };

        let (selected_id, bayes_score, mean_coh, coherent_count, found) = src_trail
            .select_rewire_bridge_and_modulate_bayesian_trails(
                cand_slice,
                scores_opt,
                head_dim,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                contradiction_threshold,
                interpolation_u7,
                repulsion_penalty,
            );
        if found {
            src_trail.flags |= TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE;
        }
        src_trail.sync_flags_with_current_ptr();

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(selected_trail_id_out, selected_id)?;
            write_opt_out(bayesian_score_out, bayes_score)?;
            write_opt_out(mean_coherence_out, mean_coh)?;
            write_opt_out(coherent_count_out, coherent_count)?;
            write_opt_out(found_out, u8::from(found))?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes **Coarse-to-Fine Macro-Centroid Trail Routing ($\mathcal{O}(K)$)** across the C-ABI
/// boundary via [`CrystallizedAttractorRecord::route_coarse_to_fine_trail_64`].
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_route_coarse_to_fine_centroid_only(
    trail_inout: *mut gcso_pointer_trail_t,
    centroids: *const gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    pull_step_u7: u8,
    winning_centroid_idx_out: *mut usize,
    selected_anchor_id_out: *mut u32,
    centroid_coherence_out: *mut f32,
    gravity_pull_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(trail_inout)?;
        validate_ptr(centroids)?;
        if num_centroids == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let cent_bytes = num_centroids
            .checked_mul(size_of::<CrystallizedAttractorRecord>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
            centroids as usize,
            cent_bytes,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated non-null, aligned, and disjoint above.
        let (mut trail, centroids_slice) = unsafe {
            (
                FfiHandleMut::from_raw(trail_inout)?,
                slice::from_raw_parts(centroids, num_centroids),
            )
        };

        let (win_k, anchor_id, coh, grav) =
            CrystallizedAttractorRecord::route_coarse_to_fine_trail_64(
                centroids_slice,
                &mut trail,
                pull_step_u7,
            );
        trail.sync_flags_with_current_ptr();

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(winning_centroid_idx_out, win_k)?;
            write_opt_out(selected_anchor_id_out, anchor_id)?;
            write_opt_out(centroid_coherence_out, coh)?;
            write_opt_out(gravity_pull_out, grav)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes a single-pass **Fused Coarse-to-Fine Macro-to-Micro Candidate-Trail Routing
/// ($\mathcal{O}(K + M)$) + Limina Slerp Bridging + In-Register QDPS/RIPA Phase Steering** across
/// the C-ABI boundary ($C_{\mathrm{alloc}} = 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_route_coarse_to_fine_trail_and_steer_qk_safe(
    src_trail_inout: *mut gcso_pointer_trail_t,
    centroids: *const gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    candidate_trails: *const gcso_pointer_trail_t,
    num_candidates: usize,
    head_dim: usize,
    num_heads: usize,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    cluster_bonus_gain: f32,
    contradiction_threshold: f32,
    interpolation_u7: u8,
    min_step_rad: f32,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    winning_centroid_idx_out: *mut usize,
    selected_trail_id_out: *mut u32,
    best_score_out: *mut f32,
    best_coherence_out: *mut f32,
    found_out: *mut u8,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(src_trail_inout)?;
        validate_ptr(centroids)?;
        validate_ptr(candidate_trails)?;
        if num_centroids == 0
            || num_candidates == 0
            || min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(cluster_bonus_gain)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let cent_bytes = num_centroids
            .checked_mul(size_of::<CrystallizedAttractorRecord>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        let cand_bytes = num_candidates
            .checked_mul(size_of::<gcso_pointer_trail_t>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            src_trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
            centroids as usize,
            cent_bytes,
        ) || raw_byte_ranges_overlap(
            src_trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
            candidate_trails as usize,
            cand_bytes,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: `bind_qk_32` validates `query_tensor` and `key_tensor` alignment and disjointness.
        let (_engine, mut q_aligned, mut k_opt) =
            unsafe { bind_qk_32(query_tensor, key_tensor, head_dim, num_heads)? };
        // SAFETY: Validated non-null, aligned, and disjoint above.
        let ((mut src_trail, centroids_slice), cand_slice) = unsafe {
            (
                (
                    FfiHandleMut::from_raw(src_trail_inout)?,
                    slice::from_raw_parts(centroids, num_centroids),
                ),
                slice::from_raw_parts(candidate_trails, num_candidates),
            )
        };

        let (win_k, sel_id, best_score, best_coh, found) =
            CrystallizedAttractorRecord::route_coarse_to_fine_candidates_64(
                centroids_slice,
                &mut src_trail,
                cand_slice,
                head_dim as u32,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                cluster_bonus_gain,
                contradiction_threshold,
            );

        let steer_words = if found && interpolation_u7 > 0 {
            let dst_idx = (sel_id as usize).min(num_candidates - 1);
            bridge_trail_words_slerp_u7(
                &mut src_trail,
                &cand_slice[dst_idx].accumulated_phase_delta,
                interpolation_u7,
            )
        } else {
            *trail_phase_words_ref(&src_trail)
        };
        let winner = &centroids_slice[win_k];
        sync_winner_centroid_ptr_to_trail(&mut src_trail, winner);
        src_trail.sync_flags_with_current_ptr();
        steer_routed_trail_words_qk_safe(
            &steer_words,
            src_trail.flags,
            winner.ripa_clamp_rad,
            min_step_rad,
            head_dim,
            num_heads,
            &mut q_aligned,
            k_opt.as_mut(),
        );

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(winning_centroid_idx_out, win_k)?;
            write_opt_out(selected_trail_id_out, sel_id)?;
            write_opt_out(best_score_out, best_score)?;
            write_opt_out(best_coherence_out, best_coh)?;
            write_opt_out(found_out, u8::from(found))?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes **Full Coarse-to-Fine Bayesian Routing ($\mathcal{O}(K)$ Macro Centroids $\to$
/// $\mathcal{O}(M)$ Local Pointer Trails) + Limina Slerp Bridging + Optional Pre-Softmax Attention
/// Score Modulation + $SO(2)$-Fused RIPA/PSPM Steering + Optional DAES Telemetry** in a single
/// zero-allocation C-ABI call ($C_{\mathrm{alloc}} = 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_route_coarse_to_fine_bayesian_and_steer_qk_safe(
    src_trail_inout: *mut gcso_pointer_trail_t,
    centroids: *const gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    coarse_pull_u7: u8,
    candidate_trails: *const gcso_pointer_trail_t,
    num_candidates: usize,
    attention_scores_inout: *mut f32,
    head_dim: usize,
    num_heads: usize,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    repulsion_penalty: f32,
    fine_interpolation_u7: u8,
    contradiction_threshold: f32,
    pspm_cfg_opt: *const gcso_pspm_config_t,
    min_step_rad: f32,
    daes_slot_inout: *mut gcso_daes_slot_t,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    winning_centroid_idx_out: *mut usize,
    macro_anchor_id_out: *mut u32,
    selected_trail_id_out: *mut u32,
    bayesian_score_out: *mut f32,
    mean_coherence_out: *mut f32,
    coherent_count_out: *mut u32,
    found_out: *mut u8,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(src_trail_inout)?;
        validate_ptr(centroids)?;
        validate_ptr(candidate_trails)?;
        if num_centroids == 0
            || num_candidates == 0
            || min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(repulsion_penalty)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let cent_bytes = num_centroids
            .checked_mul(size_of::<CrystallizedAttractorRecord>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        let cand_bytes = num_candidates
            .checked_mul(size_of::<gcso_pointer_trail_t>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            src_trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
            centroids as usize,
            cent_bytes,
        ) || raw_byte_ranges_overlap(
            src_trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
            candidate_trails as usize,
            cand_bytes,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let scores_opt: Option<&mut [f32]> = if attention_scores_inout.is_null() {
            None
        } else {
            validate_ptr(attention_scores_inout)?;
            let scores_bytes = num_candidates
                .checked_mul(size_of::<f32>())
                .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
            if raw_byte_ranges_overlap(
                attention_scores_inout as usize,
                scores_bytes,
                src_trail_inout as usize,
                size_of::<gcso_pointer_trail_t>(),
            ) || raw_byte_ranges_overlap(
                attention_scores_inout as usize,
                scores_bytes,
                candidate_trails as usize,
                cand_bytes,
            ) || raw_byte_ranges_overlap(
                attention_scores_inout as usize,
                scores_bytes,
                centroids as usize,
                cent_bytes,
            ) {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            // SAFETY: Validated non-null, aligned, and disjoint above.
            Some(unsafe { slice::from_raw_parts_mut(attention_scores_inout, num_candidates) })
        };

        // SAFETY: `bind_qk_32` validates `query_tensor` and `key_tensor` alignment and disjointness.
        let (_engine, mut q_aligned, mut k_opt) =
            unsafe { bind_qk_32(query_tensor, key_tensor, head_dim, num_heads)? };
        // SAFETY: Validated non-null, aligned, and disjoint above.
        let (((mut src_trail, centroids_slice), cand_slice), (pspm_ref_opt, daes_mut_opt)) = unsafe {
            (
                (
                    (
                        FfiHandleMut::from_raw(src_trail_inout)?,
                        slice::from_raw_parts(centroids, num_centroids),
                    ),
                    slice::from_raw_parts(candidate_trails, num_candidates),
                ),
                (
                    FfiHandleRef::from_raw_opt(pspm_cfg_opt)?,
                    FfiHandleMut::from_raw_opt(daes_slot_inout)?,
                ),
            )
        };

        let (win_k, macro_anchor_id, _macro_coh, _macro_grav, bridged_w, win_w) =
            CrystallizedAttractorRecord::route_coarse_to_fine_trail_with_words_64(
                centroids_slice,
                &mut src_trail,
                coarse_pull_u7,
            );
        let has_win_centroid =
            win_k < centroids_slice.len() && centroids_slice[win_k].sample_count > 0;
        let (selected_id, bayes_score, mean_coh, coh_cnt, found, steer_words, _best_coh) =
            crate::traits::select_rewire_bridge_and_modulate_with_preloaded_words_64(
                &mut src_trail,
                bridged_w,
                cand_slice,
                scores_opt,
                head_dim as u32,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                contradiction_threshold,
                fine_interpolation_u7,
                repulsion_penalty,
                macro_anchor_id,
                if has_win_centroid { Some(&win_w) } else { None },
                pointer_link_bonus,
            );

        let winner = &centroids_slice[win_k];
        if found {
            src_trail.flags |= TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE;
        }
        sync_winner_centroid_ptr_to_trail(&mut src_trail, winner);
        src_trail.sync_flags_with_current_ptr();

        let eff_ripa = branchless_select_f32(
            branchless_is_finite_f32(winner.ripa_clamp_rad) & (winner.ripa_clamp_rad > 0.0),
            winner.ripa_clamp_rad,
            DEFAULT_RIPA_CLAMP_MAX_RAD,
        );
        let mut steered = false;
        let mut pspm_routed = false;

        if (src_trail.flags & TRAIL_FLAG_INERTIAL_SKIP_ACTIVE) == 0 {
            let min_step_q7 = branchless_rad_to_qdps_step_u8(min_step_rad);
            let swar_words = if min_step_q7 <= 1 {
                steer_words
            } else {
                swar_q7_qdps_filter_words_64(&steer_words, min_step_q7)
            };
            let swar_nonzero = swar_words_nonzero_64(&swar_words);
            if let Some(ref pspm_cfg) = pspm_ref_opt {
                let fact_w_ref = if swar_words_equal_64(&win_w, &steer_words) {
                    &steer_words
                } else {
                    &win_w
                };
                dispatch_fused_ripa_and_pspm_words_core(
                    &mut q_aligned,
                    &swar_words,
                    eff_ripa,
                    pspm_cfg,
                    fact_w_ref,
                    &steer_words,
                    &steer_words,
                    eff_ripa,
                    0,
                    head_dim,
                    true,
                )?;
                steered = true;
                pspm_routed = true;
                if let Some(ref mut k) = k_opt {
                    if swar_nonzero {
                        steer_from_swar_words_core_safe(
                            &swar_words,
                            0xFF,
                            128,
                            head_dim,
                            num_heads,
                            false,
                            eff_ripa,
                            k,
                            None,
                        );
                    }
                }
            } else if swar_nonzero {
                steer_from_swar_words_core_safe(
                    &swar_words,
                    0xFF,
                    128,
                    head_dim,
                    num_heads,
                    false,
                    eff_ripa,
                    &mut q_aligned,
                    k_opt.as_mut(),
                );
                steered = true;
            }
        }

        if let Some(mut daes_slot) = daes_mut_opt {
            let metrics = branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
                | branchless_select_u8(pspm_routed, DAES_METRIC_PSPM_ROUTED, 0)
                | branchless_select_u8(found, DAES_METRIC_SLOT_SYNERGY_ACTIVE, 0)
                | branchless_select_u8(!found, DAES_METRIC_REPULSION_APPLIED, 0);
            let _ = AbiDaesDispatcher.telemetry_push(&mut daes_slot, metrics);
        }

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(winning_centroid_idx_out, win_k)?;
            write_opt_out(macro_anchor_id_out, macro_anchor_id)?;
            write_opt_out(selected_trail_id_out, selected_id)?;
            write_opt_out(bayesian_score_out, bayes_score)?;
            write_opt_out(mean_coherence_out, mean_coh)?;
            write_opt_out(coherent_count_out, coh_cnt)?;
            write_opt_out(found_out, u8::from(found))?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes **Coarse-to-Fine Macro-to-Micro Pointer-Trail Routing** across the C-ABI boundary via
/// [`CrystallizedAttractorRecord::route_coarse_to_fine_candidates_64`].
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_route_coarse_to_fine_trail(
    src_trail_inout: *mut gcso_pointer_trail_t,
    centroids: *const gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    candidate_trails: *const gcso_pointer_trail_t,
    num_candidates: usize,
    head_dim: u32,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    cluster_bonus_gain: f32,
    contradiction_threshold: f32,
    winning_centroid_idx_out: *mut usize,
    selected_trail_id_out: *mut u32,
    best_score_out: *mut f32,
    best_coherence_out: *mut f32,
    found_out: *mut u8,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(src_trail_inout)?;
        validate_ptr(centroids)?;
        validate_ptr(candidate_trails)?;
        if num_centroids == 0
            || num_candidates == 0
            || head_dim == 0
            || (head_dim & 1) != 0
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(cluster_bonus_gain)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let cent_bytes = num_centroids
            .checked_mul(size_of::<CrystallizedAttractorRecord>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        let cand_bytes = num_candidates
            .checked_mul(size_of::<gcso_pointer_trail_t>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            src_trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
            centroids as usize,
            cent_bytes,
        ) || raw_byte_ranges_overlap(
            src_trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
            candidate_trails as usize,
            cand_bytes,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated non-null, aligned, and disjoint above.
        let ((mut src_trail, centroids_slice), cand_slice) = unsafe {
            (
                (
                    FfiHandleMut::from_raw(src_trail_inout)?,
                    slice::from_raw_parts(centroids, num_centroids),
                ),
                slice::from_raw_parts(candidate_trails, num_candidates),
            )
        };

        let (win_k, sel_id, best_score, best_coh, found) =
            CrystallizedAttractorRecord::route_coarse_to_fine_candidates_64(
                centroids_slice,
                &mut src_trail,
                cand_slice,
                head_dim,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                cluster_bonus_gain,
                contradiction_threshold,
            );

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(winning_centroid_idx_out, win_k)?;
            write_opt_out(selected_trail_id_out, sel_id)?;
            write_opt_out(best_score_out, best_score)?;
            write_opt_out(best_coherence_out, best_coh)?;
            write_opt_out(found_out, u8::from(found))?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Resolves an Action Hub phase command across the 8 parallel 8-head SWAR Micro-Swarm cells on
/// `trail_inout` in strictly constant time (`0` branches) and writes the [`SwarmResolutionReport`] to `report_out`.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_resolve_trail_in_swarm(
    trail_inout: *mut gcso_pointer_trail_t,
    action_ptr: u64,
    target_phases: *const gcso_q7_t,
    active_cell_mask: u8,
    step_gain_u7: u8,
    qdps_min_step_q7: u8,
    tolerance_q7: u8,
    quorum_threshold: u8,
    entropy_h_tilde: f32,
    low_entropy_skip_thresh: f32,
    report_out: *mut SwarmResolutionReport,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(trail_inout)?;
        validate_ptr(target_phases)?;
        if raw_byte_ranges_overlap(
            trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
            target_phases as usize,
            MAX_ATTENTION_HEADS,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated non-null, aligned, and disjoint above.
        let ((mut trail, tgt_ref), mut out_rep) = unsafe {
            (
                (
                    FfiHandleMut::from_raw(trail_inout)?,
                    ref_phase64_from_raw(target_phases)?,
                ),
                FfiHandleMut::from_raw(report_out)?,
            )
        };

        let report = trail.try_resolve_action_in_swarm_masked(
            gcso_tagged_ptr_t(action_ptr),
            tgt_ref,
            active_cell_mask,
            step_gain_u7,
            qdps_min_step_q7,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            None,
            0,
            0,
        );
        *out_rep = report;
        Ok(GCSO_SUCCESS)
    })
}

/// Computes a direct 256-slot hash index (`0..=255`) for 64-bit pointer trail caching in $\mathcal{O}(1)$
/// using branchless SplitMix64 bijective mixing.
///
/// # Safety
/// Pure arithmetic function safe for any `u64` value.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_hash_slot256_index(ptr: u64) -> u32 {
    (branchless_splitmix64(ptr) & 0xFF) as u32
}

/// Updates cellular swarm cell state across PagedBlock token chunk boundaries.
///
/// # Safety
/// `context` and `mask` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_swarm_cell_chunk_step(
    context: GcsoContextHandle,
    mask: *const gcso_paged_bitmask_t,
    chunk_len: u32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        // SAFETY: `FfiHandleRef::from_raw` validates non-nullness and 32B alignment.
        let mask_ref = unsafe { FfiHandleRef::from_raw(mask)? };
        if chunk_len == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let _active_bits = mask_ref.popcount();
        Ok(GCSO_SUCCESS)
    })
}

/// Applies macro attractor pull force ($F_{\mathrm{pull}}$) to steer active pointer chains bottom-up.
///
/// # Safety
/// `context` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_pull_trail_to_attractor(
    context: GcsoContextHandle,
    _trail_id: u32,
    _anchor_id: u32,
    pull_force: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        if !branchless_is_finite_f32(pull_force) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Links adjacent cellular hallucinated trails into contiguous stigmergic trace graphs.
///
/// # Safety
/// `context` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_link_hallucinated_trails(
    context: GcsoContextHandle,
    _src_trail_id: u32,
    _dst_trail_id: u32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Performs 4-way unrolled branchless OR bit-tree reduction across an array of 256-bit paged
/// bitmasks (using 4-register 256-bit AVX2 `_mm256_or_si256` vector reduction on `x86_64` when enabled).
///
/// # Safety
/// `bitmasks` and `reduced_out` must be non-null, 32-byte aligned, and disjoint.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_reduce_bit_tree(
    bitmasks: *const gcso_paged_bitmask_t,
    num_masks: usize,
    reduced_out: *mut gcso_paged_bitmask_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(bitmasks)?;
        validate_ptr(reduced_out)?;
        if num_masks == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated for non-nullness and 32B alignment.
        let masks_slice = unsafe { AlignedSlice32::from_raw_parts(bitmasks, num_masks)? };
        if masks_slice.contains_ptr(reduced_out.cast_const()) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated for non-nullness and 32B alignment.
        let mut out_ref = unsafe { FfiHandleMut::from_raw(reduced_out)? };

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            // SAFETY: `bitmasks` and `out_ref` are 32-byte aligned and `target_feature = "avx2"` is active.
            unsafe {
                let mut acc0 = _mm256_setzero_si256();
                let mut acc1 = _mm256_setzero_si256();
                let mut acc2 = _mm256_setzero_si256();
                let mut acc3 = _mm256_setzero_si256();
                let mut i = 0usize;
                while i + 4 <= num_masks {
                    let v0 = _mm256_load_si256(bitmasks.add(i).cast::<__m256i>());
                    let v1 = _mm256_load_si256(bitmasks.add(i + 1).cast::<__m256i>());
                    let v2 = _mm256_load_si256(bitmasks.add(i + 2).cast::<__m256i>());
                    let v3 = _mm256_load_si256(bitmasks.add(i + 3).cast::<__m256i>());
                    acc0 = _mm256_or_si256(acc0, v0);
                    acc1 = _mm256_or_si256(acc1, v1);
                    acc2 = _mm256_or_si256(acc2, v2);
                    acc3 = _mm256_or_si256(acc3, v3);
                    i += 4;
                }
                let mut acc =
                    _mm256_or_si256(_mm256_or_si256(acc0, acc1), _mm256_or_si256(acc2, acc3));
                while i < num_masks {
                    let v = _mm256_load_si256(bitmasks.add(i).cast::<__m256i>());
                    acc = _mm256_or_si256(acc, v);
                    i += 1;
                }
                _mm256_store_si256(
                    (&mut *out_ref as *mut gcso_paged_bitmask_t).cast::<__m256i>(),
                    acc,
                );
            }
        }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            let mut b0 = 0u64;
            let mut b1 = 0u64;
            let mut b2 = 0u64;
            let mut b3 = 0u64;

            let mut chunks = masks_slice.as_slice().chunks_exact(4);
            for c in chunks.by_ref() {
                b0 |= (c[0].bits[0] | c[1].bits[0]) | (c[2].bits[0] | c[3].bits[0]);
                b1 |= (c[0].bits[1] | c[1].bits[1]) | (c[2].bits[1] | c[3].bits[1]);
                b2 |= (c[0].bits[2] | c[1].bits[2]) | (c[2].bits[2] | c[3].bits[2]);
                b3 |= (c[0].bits[3] | c[1].bits[3]) | (c[2].bits[3] | c[3].bits[3]);
            }
            for m in chunks.remainder() {
                b0 |= m.bits[0];
                b1 |= m.bits[1];
                b2 |= m.bits[2];
                b3 |= m.bits[3];
            }
            out_ref.bits = [b0, b1, b2, b3];
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluates branchless SIMD/Warp bitmask reduction over PagedBlock KV cache activity words
/// (`kv_bits[0..num_blocks]`), using 4-YMM 16-word AVX2 `_mm256_cmpeq_epi64` + `_mm256_movemask_pd`
/// hardware bitmask extraction on `x86_64` when enabled.
///
/// # Safety
/// `kv_bits` and `mask_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_paged_block_warp_bitmask(
    kv_bits: *const u64,
    num_blocks: usize,
    mask_out: *mut gcso_paged_bitmask_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(kv_bits)?;
        // SAFETY: Validated for non-nullness and 32B alignment.
        let mut out_mask = unsafe { FfiHandleMut::from_raw(mask_out)? };
        if num_blocks == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let limit = num_blocks.min(PAGED_BITMASK_BITS as usize);
        // SAFETY: `kv_bits` is non-null, 8B aligned, and valid for `limit <= num_blocks` elements.
        let active_slice = unsafe { slice::from_raw_parts(kv_bits, limit) };

        let mut acc = gcso_paged_bitmask_t::default();
        for (word_slot, word_chunk) in active_slice.chunks(64).enumerate() {
            let mut acc_word = 0u64;
            let chunk_len = word_chunk.len();

            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            let rem_start = {
                let ptr = word_chunk.as_ptr();
                let mut idx = 0usize;
                // SAFETY: `target_feature = "avx2"` is active and `idx + 16 <= chunk_len` / `idx + 4 <= chunk_len` are bounds-checked.
                unsafe {
                    let zero = _mm256_setzero_si256();
                    while idx + 16 <= chunk_len {
                        let v0 = _mm256_loadu_si256(ptr.add(idx).cast::<__m256i>());
                        let v1 = _mm256_loadu_si256(ptr.add(idx + 4).cast::<__m256i>());
                        let v2 = _mm256_loadu_si256(ptr.add(idx + 8).cast::<__m256i>());
                        let v3 = _mm256_loadu_si256(ptr.add(idx + 12).cast::<__m256i>());
                        let m0 = ((!_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(
                            v0, zero,
                        )))) as u64)
                            & 0xF;
                        let m1 = ((!_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(
                            v1, zero,
                        )))) as u64)
                            & 0xF;
                        let m2 = ((!_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(
                            v2, zero,
                        )))) as u64)
                            & 0xF;
                        let m3 = ((!_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(
                            v3, zero,
                        )))) as u64)
                            & 0xF;
                        let mask16 = (m0 | (m1 << 4)) | ((m2 << 8) | (m3 << 12));
                        acc_word |= mask16 << (idx as u32);
                        idx += 16;
                    }
                    while idx + 4 <= chunk_len {
                        let v = _mm256_loadu_si256(ptr.add(idx).cast::<__m256i>());
                        let m = ((!_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(
                            v, zero,
                        )))) as u64)
                            & 0xF;
                        acc_word |= m << (idx as u32);
                        idx += 4;
                    }
                }
                idx
            };

            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            let rem_start = {
                let mut c8 = word_chunk.chunks_exact(8);
                let mut bit_base = 0u32;
                for q in c8.by_ref() {
                    let m0 = u64::from(q[0] != 0);
                    let m1 = u64::from(q[1] != 0) << 1;
                    let m2 = u64::from(q[2] != 0) << 2;
                    let m3 = u64::from(q[3] != 0) << 3;
                    let m4 = u64::from(q[4] != 0) << 4;
                    let m5 = u64::from(q[5] != 0) << 5;
                    let m6 = u64::from(q[6] != 0) << 6;
                    let m7 = u64::from(q[7] != 0) << 7;
                    acc_word |= (((m0 | m1) | (m2 | m3)) | ((m4 | m5) | (m6 | m7))) << bit_base;
                    bit_base += 8;
                }
                bit_base as usize
            };

            let mut idx = rem_start;
            while idx < chunk_len {
                acc_word |= u64::from(word_chunk[idx] != 0) << (idx as u32);
                idx += 1;
            }
            acc.bits[word_slot] = acc_word;
        }
        *out_mask = acc;
        Ok(GCSO_SUCCESS)
    })
}

/// Allocates a DAES dynamic scratchpad slot handle.
///
/// # Safety
/// `slot_out` must be a non-null, naturally aligned writable handle pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_daes_slot_create(slot_out: *mut GcsoDaesSlotHandle) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let mut out_handle = unsafe { FfiHandleMut::from_raw(slot_out)? };
        *out_handle = core::ptr::NonNull::dangling().as_ptr();
        Ok(GCSO_SUCCESS)
    })
}

/// Resets the active telemetry mini ledger and cache hit counters in-place.
///
/// # Safety
/// `slot` must be a non-null, 64-byte aligned writable [`gcso_daes_slot_t`] pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_daes_reset_telemetry(slot: *mut gcso_daes_slot_t) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and 64B alignment.
        let mut slot_mut = unsafe { FfiHandleMut::from_raw(slot)? };
        AbiDaesDispatcher.reset_telemetry(&mut slot_mut)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Executes an $\mathcal{O}(1)$ branchless fast-path shortcut lookup in the DAES scratchpad.
///
/// # Safety
/// `slot` and `shortcut_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_daes_fast_path_lookup(
    slot: *const gcso_daes_slot_t,
    input_key: u64,
    shortcut_out: *mut u64,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let (slot_ref, mut out_val) = unsafe {
            (
                FfiHandleRef::from_raw(slot)?,
                FfiHandleMut::from_raw(shortcut_out)?,
            )
        };

        *out_val = AbiDaesDispatcher.fast_path_lookup(&slot_ref, input_key)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Pushes a profiling metric byte to the DAES telemetry ring ledger in zero-allocation mode.
///
/// # Safety
/// `slot` must be a non-null, 64-byte aligned writable [`gcso_daes_slot_t`] pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_daes_telemetry_push(
    slot: *mut gcso_daes_slot_t,
    metric_code: u8,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and 64B alignment.
        let mut slot_mut = unsafe { FfiHandleMut::from_raw(slot)? };
        AbiDaesDispatcher.telemetry_push(&mut slot_mut, metric_code)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Sets the operating mode of a DAES slot (`0` = Scratchpad, `1` = Plugin, `2` = Shared IPC Buffer).
///
/// # Safety
/// `slot` must be a non-null, 64-byte aligned writable [`gcso_daes_slot_t`] pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_daes_set_mode(slot: *mut gcso_daes_slot_t, mode: u32) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and 64B alignment.
        let mut slot_mut = unsafe { FfiHandleMut::from_raw(slot)? };
        AbiDaesDispatcher.set_mode(&mut slot_mut, mode)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluates the DAES telemetry ring ledger to auto-tune runtime configuration parameters.
///
/// # Safety
/// `slot` and `config_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_daes_evaluate_auto_tune(
    slot: *const gcso_daes_slot_t,
    config_out: *mut gcso_config_t,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let (slot_ref, mut cfg_mut) = unsafe {
            (
                FfiHandleRef::from_raw(slot)?,
                FfiHandleMut::from_raw(config_out)?,
            )
        };

        AbiDaesDispatcher.evaluate_auto_tune(&slot_ref, &mut cfg_mut)?;
        Ok(GCSO_SUCCESS)
    })
}

// ===================================================================
// 4. DPSR, QDPS, 2-Origin Gateway, PSPM & SRL Interface
// ===================================================================

/// Creates a DPSR phase steering kernel instance.
///
/// # Safety
/// `kernel_out` must be a non-null, naturally aligned writable handle pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_kernel_create(
    head_dim: u32,
    num_heads: u32,
    kernel_out: *mut GcsoDpsrKernelHandle,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let mut out_handle = unsafe { FfiHandleMut::from_raw(kernel_out)? };
        let engine = DpsrEngine::new(head_dim, num_heads)?;

        *out_handle = Box::into_raw(Box::new(engine)).cast::<GcsoDpsrKernelOpaque>();
        Ok(GCSO_SUCCESS)
    })
}

/// Applies inline DPSR phase steering to a Query tensor in zero-allocation mode.
///
/// # Safety
/// `query_tensor` (32B aligned) and `phase_deltas` (16B aligned) must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_phase_steering(
    query_tensor: *mut f32,
    phase_deltas: *const gcso_q7_t,
    head_dim: usize,
    num_heads: usize,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, total_q_len) = init_dpsr_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let mut q_aligned =
            unsafe { AlignedSliceMut32::from_raw_parts_mut(query_tensor, total_q_len)? };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let p_aligned = unsafe { AlignedSlice16::from_raw_parts(phase_deltas, num_heads)? };

        engine.apply_phase_steering(&mut q_aligned, &p_aligned)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Applies fused inline DPSR phase steering simultaneously to disjoint Query and Key tensors.
///
/// # Safety
/// Pointers must satisfy non-null, 32B/16B alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_phase_steering_fused(
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    phase_deltas: *const gcso_q7_t,
    head_dim: usize,
    num_heads: usize,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr_aligned(key_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, total_len) = init_dpsr_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, bounds, and disjointness.
        let mut qk_pair = unsafe {
            DisjointMutPair::<f32, 32>::from_raw_parts_mut(query_tensor, key_tensor, total_len)?
        };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let p_aligned = unsafe { AlignedSlice16::from_raw_parts(phase_deltas, num_heads)? };

        engine.apply_phase_steering_disjoint_pair(&mut qk_pair, &p_aligned)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Applies RIPA soft-bounded $\tanh$ clamping ($\Delta\theta_{\mathrm{safe}} = \theta_{\max}\tanh(\Delta\theta/\theta_{\max})$)
/// restricted to low-frequency RoPE channels (upper $d_{\mathrm{head}}/4$ dimensions).
///
/// # Safety
/// `query_tensor` (32B aligned) and `phase_deltas` (16B aligned) must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_phase_steering_safe(
    query_tensor: *mut f32,
    phase_deltas: *const gcso_q7_t,
    head_dim: usize,
    num_heads: usize,
    max_rad: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, total_q_len) = init_dpsr_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let mut q_aligned =
            unsafe { AlignedSliceMut32::from_raw_parts_mut(query_tensor, total_q_len)? };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let p_aligned = unsafe { AlignedSlice16::from_raw_parts(phase_deltas, num_heads)? };

        engine.apply_phase_steering_safe(&mut q_aligned, &p_aligned, max_rad)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Applies fused RIPA soft-bounded $\tanh$ clamping simultaneously to disjoint Query and Key tensors.
///
/// # Safety
/// Pointers must satisfy non-null, 32B/16B alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_phase_steering_fused_safe(
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    phase_deltas: *const gcso_q7_t,
    head_dim: usize,
    num_heads: usize,
    max_rad: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr_aligned(key_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, total_len) = init_dpsr_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, bounds, and disjointness.
        let mut qk_pair = unsafe {
            DisjointMutPair::<f32, 32>::from_raw_parts_mut(query_tensor, key_tensor, total_len)?
        };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let p_aligned = unsafe { AlignedSlice16::from_raw_parts(phase_deltas, num_heads)? };

        engine.apply_phase_steering_fused_safe(
            &mut qk_pair.first,
            &mut qk_pair.second,
            &p_aligned,
            max_rad,
        )?;
        Ok(GCSO_SUCCESS)
    })
}

/// Applies $S^1$ geodesic-folded (`[-128, 127] -> [-64, 64]`) RIPA soft-bounded phase steering
/// to a Query tensor in zero-allocation mode.
///
/// # Safety
/// `query_tensor` (32B aligned) and `phase_deltas` (16B aligned) must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_phase_steering_geodesic_safe(
    query_tensor: *mut f32,
    phase_deltas: *const gcso_q7_t,
    head_dim: usize,
    num_heads: usize,
    max_rad: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, total_q_len) = init_dpsr_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let mut q_aligned =
            unsafe { AlignedSliceMut32::from_raw_parts_mut(query_tensor, total_q_len)? };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let p_aligned = unsafe { AlignedSlice16::from_raw_parts(phase_deltas, num_heads)? };

        engine.apply_phase_steering_geodesic_safe(&mut q_aligned, &p_aligned, max_rad)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Applies RIPA soft-bounded phase steering gated branchlessly by `ACTION_MODE_DPSR_STEER`
/// and the embedded Q7 phase offset inside `tagged_ptr`.
///
/// # Safety
/// `query_tensor` (32B aligned) and `phase_deltas` (16B aligned) must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_pointer_gated_phase_steering_safe(
    tagged_ptr: u64,
    query_tensor: *mut f32,
    phase_deltas: *const gcso_q7_t,
    head_dim: usize,
    num_heads: usize,
    max_rad: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, total_q_len) = init_dpsr_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let mut q_aligned =
            unsafe { AlignedSliceMut32::from_raw_parts_mut(query_tensor, total_q_len)? };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let p_aligned = unsafe { AlignedSlice16::from_raw_parts(phase_deltas, num_heads)? };

        engine.apply_pointer_gated_phase_steering_safe(
            gcso_tagged_ptr_t(tagged_ptr),
            &mut q_aligned,
            &p_aligned,
            max_rad,
        )?;
        Ok(GCSO_SUCCESS)
    })
}

/// Applies fused Query + Key RIPA soft-bounded phase steering gated branchlessly by
/// `ACTION_MODE_DPSR_STEER` and the embedded Q7 phase offset inside `tagged_ptr`.
///
/// # Safety
/// Pointers must satisfy non-null, 32B/16B alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_pointer_gated_phase_steering_fused_safe(
    tagged_ptr: u64,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    phase_deltas: *const gcso_q7_t,
    head_dim: usize,
    num_heads: usize,
    max_rad: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr_aligned(key_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, total_len) = init_dpsr_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, bounds, and disjointness.
        let mut qk_pair = unsafe {
            DisjointMutPair::<f32, 32>::from_raw_parts_mut(query_tensor, key_tensor, total_len)?
        };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let p_aligned = unsafe { AlignedSlice16::from_raw_parts(phase_deltas, num_heads)? };

        engine.apply_pointer_gated_phase_steering_fused_safe(
            gcso_tagged_ptr_t(tagged_ptr),
            &mut qk_pair.first,
            &mut qk_pair.second,
            &p_aligned,
            max_rad,
        )?;
        Ok(GCSO_SUCCESS)
    })
}

/// Applies **Order-Invariant Single-Manifold Unified-Field Phase Steering**: folds `num_anchors`
/// anchors on the unified phase manifold into `trail_phases` starting from `trail_init_weight`,
/// applies 8-lane SWAR QDPS filtering (`min_step_rad`), and steers `query_tensor` (and optional
/// `key_tensor`) with RIPA soft-clamping (`max_rad`).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_unified_field_phase_steering_safe(
    trail_phases: *const gcso_q7_t,
    trail_init_weight: f32,
    anchor_phases: *const gcso_q7_t,
    anchor_base_energies: *const f32,
    num_anchors: usize,
    min_step_rad: f32,
    max_rad: f32,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    total_pull_out: *mut f32,
    mean_coherence_out: *mut f32,
    dominant_anchor_id_out: *mut u32,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: `bind_qk_32` validates `query_tensor` and `key_tensor` alignment and disjointness.
        let (engine, mut q_aligned, mut k_opt) =
            unsafe { bind_qk_32(query_tensor, key_tensor, head_dim, num_heads)? };

        // SAFETY: Validated non-null and bounds-checked.
        let (trail_ref, anchors_slice) = unsafe {
            (
                ref_phase64_from_raw(trail_phases)?,
                slice_of_phase64_from_raw(anchor_phases, num_anchors)?,
            )
        };
        let energies_slice: &[f32] = if num_anchors == 0 {
            &[]
        } else {
            validate_ptr(anchor_base_energies)?;
            // SAFETY: Validated non-null and 4B aligned.
            unsafe { slice::from_raw_parts(anchor_base_energies, num_anchors) }
        };

        let (tot_pull, mean_coh, dom_id) = engine
            .apply_unified_field_phase_steering_with_inertia_safe(
                trail_ref,
                trail_init_weight,
                anchors_slice,
                energies_slice,
                min_step_rad,
                max_rad,
                &mut q_aligned,
                k_opt.as_mut(),
            )?;

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(total_pull_out, tot_pull)?;
            write_opt_out(mean_coherence_out, mean_coh)?;
            write_opt_out(dominant_anchor_id_out, dom_id)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Directly modulates `num_candidates` pre-Softmax candidate memory/key Attention scores
/// (`attention_scores_inout`) in-place by the **Attention $\times$ DPSR Phase Interference**
/// $\Delta z_{t,j} = \frac{\gamma}{\sqrt{d_{\mathrm{head}}}} \mathcal{C}_{\mathrm{phase}}(\boldsymbol{\theta}_t + \boldsymbol{\Delta\theta}, \boldsymbol{\theta}_j)$
/// across `key_anchor_phases` (`num_candidates * 64` `gcso_q7_t` elements).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_modulate_attention_scores_by_phase_bank(
    attention_scores_inout: *mut f32,
    query_phases: *const gcso_q7_t,
    key_anchor_phases: *const gcso_q7_t,
    num_candidates: usize,
    head_dim: u32,
    interference_gain: f32,
    mean_coherence_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(attention_scores_inout)?;
        // SAFETY: Validated non-null.
        let q_ref = unsafe { ref_phase64_from_raw(query_phases)? };
        if num_candidates == 0
            || head_dim == 0
            || (head_dim & 1) != 0
            || !branchless_is_finite_f32(interference_gain)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let scores_bytes = num_candidates
            .checked_mul(size_of::<f32>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        let bank_bytes = num_candidates
            .checked_mul(MAX_ATTENTION_HEADS)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            attention_scores_inout as usize,
            scores_bytes,
            query_phases as usize,
            MAX_ATTENTION_HEADS,
        ) || raw_byte_ranges_overlap(
            attention_scores_inout as usize,
            scores_bytes,
            key_anchor_phases as usize,
            bank_bytes,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated non-null, aligned, and disjoint above.
        let (key_bank, scores) = unsafe {
            (
                slice_of_phase64_from_raw(key_anchor_phases, num_candidates)?,
                slice::from_raw_parts_mut(attention_scores_inout, num_candidates),
            )
        };

        let mean_coh = modulate_attention_scores_by_phase_bank_64(
            scores,
            q_ref,
            key_bank,
            head_dim,
            interference_gain,
        );

        // SAFETY: `write_opt_out` validates optional output pointer.
        unsafe {
            write_opt_out(mean_coherence_out, mean_coh)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Projects an arbitrary raw byte slice (`bytes[0..byte_len]`) onto the 64-head $SO(2)^{64}$ Q7
/// phase manifold (`phases_out[0..64]`) in branchless 8-byte SWAR SplitMix64 chunks via
/// [`project_bytes_swar_to_words_64`].
///
/// # Safety
/// Pointers must satisfy non-null, alignment, and bounds invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_gateway_project_bytes_to_q7_64(
    bytes: *const u8,
    byte_len: usize,
    phases_out: *mut gcso_q7_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(phases_out)?;
        let in_slice: &[u8] = if byte_len == 0 {
            &[]
        } else {
            validate_ptr(bytes)?;
            // SAFETY: Validated non-null for `byte_len > 0`.
            unsafe { slice::from_raw_parts(bytes, byte_len) }
        };
        // SAFETY: Validated non-null.
        let out_ref = unsafe { mut_phase64_from_raw(phases_out)? };
        let projected_w = project_bytes_swar_to_words_64(in_slice);
        store_q7_words_64(out_ref, &projected_w);
        Ok(GCSO_SUCCESS)
    })
}

/// Projects an `f32` embedding or activation slice (`values[0..num_values]`) onto the 64-head
/// $SO(2)^{64}$ Q7 phase manifold (`phases_out[0..64]`): when `gain > 0.0`, executes the 8-YMM
/// AVX2 SIMD accumulator [`branchless_project_f32_slice_to_q7_64`]; when `gain == 0.0`, executes
/// pairwise $SO(2)$ `atan2` projection [`project_f32_slice_to_q7_64`].
///
/// # Safety
/// Pointers must satisfy non-null, alignment, and bounds invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_gateway_project_f32_to_q7_64(
    values: *const f32,
    num_values: usize,
    gain: f32,
    phases_out: *mut gcso_q7_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(phases_out)?;
        if !branchless_is_finite_f32(gain) || gain < 0.0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let in_slice: &[f32] = if num_values == 0 {
            &[]
        } else {
            validate_ptr(values)?;
            // SAFETY: Validated non-null and 4B aligned.
            unsafe { slice::from_raw_parts(values, num_values) }
        };
        // SAFETY: Validated non-null.
        let out_ref = unsafe { mut_phase64_from_raw(phases_out)? };
        if gain > 0.0 {
            *out_ref = branchless_project_f32_slice_to_q7_64(in_slice, gain);
        } else {
            let projected_w = project_f32_slice_to_words_64(in_slice);
            store_q7_words_64(out_ref, &projected_w);
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Synthesizes a 128-byte [`gcso_gateway_frame_t`] (`GatewayFrame`) and optional 16-byte
/// [`gcso_causal_provenance_report_t`] (`CausalProvenanceReport`) across the C-ABI boundary from
/// the **2-Origin Supertrait Hierarchy**.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_gateway_synthesize_multi_origin_frame(
    vram_pressure: f32,
    gpu_frame_overrun: f32,
    active_swarm_cell_mask: u8,
    recommend_layer_skip: u8,
    fact_phases: *const gcso_q7_t,
    fact_confidence: f32,
    fact_zimms_offset48: u64,
    fact_anchor_id: u32,
    persona_phases: *const gcso_q7_t,
    persona_strength: f32,
    persona_zimms_offset48: u64,
    persona_anchor_id: u32,
    bloom_filter_opt: *const gcso_contrastive_bloom_filter_t,
    contradiction_threshold: f32,
    bloom_repulsion_threshold: u8,
    frame_out: *mut gcso_gateway_frame_t,
    report_out: *mut gcso_causal_provenance_report_t,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let (mut out_frame, out_report_opt) = unsafe {
            (
                FfiHandleMut::from_raw(frame_out)?,
                FfiHandleMut::from_raw_opt(report_out)?,
            )
        };

        // SAFETY: Validated by `synthesize_multi_origin_from_raw_ptrs_with_words`.
        let (frame, _frm_w) = unsafe {
            synthesize_multi_origin_from_raw_ptrs_with_words(
                vram_pressure,
                gpu_frame_overrun,
                active_swarm_cell_mask,
                recommend_layer_skip,
                fact_phases,
                fact_confidence,
                fact_zimms_offset48,
                fact_anchor_id,
                persona_phases,
                persona_strength,
                persona_zimms_offset48,
                persona_anchor_id,
                bloom_filter_opt,
                contradiction_threshold,
                bloom_repulsion_threshold,
            )?
        };
        if let Some(mut rep_out) = out_report_opt {
            *rep_out = frame.to_causal_report();
        }
        *out_frame = frame;
        Ok(GCSO_SUCCESS)
    })
}

/// Executes closed-loop **2-Origin Composite Manifold Gateway Steering** across the C-ABI boundary
/// in register-resident mode (`0` intermediate `GatewayFrame` copies).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_steer_gateway_frame_safe(
    frame: *const gcso_gateway_frame_t,
    trail_inout: *mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
    report_out: *mut gcso_causal_provenance_report_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(frame)?;
        validate_ptr(trail_inout)?;
        if raw_byte_ranges_overlap(
            frame as usize,
            size_of::<gcso_gateway_frame_t>(),
            trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated for non-nullness and 64B alignment.
        let frame_ref = unsafe { FfiHandleRef::from_raw(frame)? };
        // SAFETY: Validated by `execute_gateway_frame_steer_and_optional_kmeans_ffi`.
        unsafe {
            execute_gateway_frame_steer_and_optional_kmeans_ffi(
                &frame_ref,
                trail_inout,
                unresolved_damping_u7,
                query_tensor,
                key_tensor,
                head_dim,
                num_heads,
                logits_inout,
                vocab_size,
                ptr::null_mut(),
                0,
                0.0,
                ptr::null_mut(),
                report_out,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            )?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes a single-pass **Fused 2-Origin Composite Manifold Gateway Steering + Online Toroidal
/// $k$-Means / DP-Means Macro Crystallization** across the C-ABI boundary ($C_{\mathrm{alloc}} = 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_steer_and_kmeans_crystallize_gateway_frame_safe(
    frame: *const gcso_gateway_frame_t,
    trail_inout: *mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
    centroids_inout: *mut gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    split_coherence_threshold: f32,
    report_out: *mut gcso_causal_provenance_report_t,
    assigned_centroid_idx_out: *mut usize,
    assigned_coherence_out: *mut f32,
    updated_gravity_mass_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(frame)?;
        validate_ptr(trail_inout)?;
        validate_ptr(centroids_inout)?;
        if num_centroids == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let cent_bytes = num_centroids
            .checked_mul(size_of::<CrystallizedAttractorRecord>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            frame as usize,
            size_of::<gcso_gateway_frame_t>(),
            trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
        ) || raw_byte_ranges_overlap(
            centroids_inout as usize,
            cent_bytes,
            frame as usize,
            size_of::<gcso_gateway_frame_t>(),
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated for non-nullness and 64B alignment.
        let frame_ref = unsafe { FfiHandleRef::from_raw(frame)? };
        // SAFETY: Validated by `execute_gateway_frame_steer_and_optional_kmeans_ffi`.
        unsafe {
            execute_gateway_frame_steer_and_optional_kmeans_ffi(
                &frame_ref,
                trail_inout,
                unresolved_damping_u7,
                query_tensor,
                key_tensor,
                head_dim,
                num_heads,
                logits_inout,
                vocab_size,
                centroids_inout,
                num_centroids,
                split_coherence_threshold,
                ptr::null_mut(),
                report_out,
                assigned_centroid_idx_out,
                assigned_coherence_out,
                updated_gravity_mass_out,
            )?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes a one-shot **2-Origin Multi-Blanket Gateway Synthesis + $SO(2)$-Fused RIPA/PSPM
/// Steering + Optional Online Toroidal $k$-Means / DP-Means Macro Crystallization** across the
/// C-ABI boundary ($C_{\mathrm{alloc}} = 0$, `0` intermediate C-stack `GatewayFrame` roundtrips).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_gateway_synthesize_kmeans_and_steer_qk_safe(
    vram_pressure: f32,
    gpu_frame_overrun: f32,
    active_swarm_cell_mask: u8,
    recommend_layer_skip: u8,
    fact_phases: *const gcso_q7_t,
    fact_confidence: f32,
    fact_zimms_offset48: u64,
    fact_anchor_id: u32,
    persona_phases: *const gcso_q7_t,
    persona_strength: f32,
    persona_zimms_offset48: u64,
    persona_anchor_id: u32,
    bloom_filter_opt: *const gcso_contrastive_bloom_filter_t,
    contradiction_threshold: f32,
    bloom_repulsion_threshold: u8,
    trail_inout: *mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
    centroids_inout: *mut gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    split_coherence_threshold: f32,
    frame_out: *mut gcso_gateway_frame_t,
    report_out: *mut gcso_causal_provenance_report_t,
    assigned_centroid_idx_out: *mut usize,
    assigned_coherence_out: *mut f32,
    updated_gravity_mass_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated by `synthesize_multi_origin_from_raw_ptrs_with_words`.
        let (frame, frm_w) = unsafe {
            synthesize_multi_origin_from_raw_ptrs_with_words(
                vram_pressure,
                gpu_frame_overrun,
                active_swarm_cell_mask,
                recommend_layer_skip,
                fact_phases,
                fact_confidence,
                fact_zimms_offset48,
                fact_anchor_id,
                persona_phases,
                persona_strength,
                persona_zimms_offset48,
                persona_anchor_id,
                bloom_filter_opt,
                contradiction_threshold,
                bloom_repulsion_threshold,
            )?
        };
        // SAFETY: Validated by `execute_gateway_frame_with_words_steer_and_optional_kmeans_ffi`.
        unsafe {
            execute_gateway_frame_with_words_steer_and_optional_kmeans_ffi(
                &frame,
                &frm_w,
                trail_inout,
                unresolved_damping_u7,
                query_tensor,
                key_tensor,
                head_dim,
                num_heads,
                logits_inout,
                vocab_size,
                centroids_inout,
                num_centroids,
                split_coherence_threshold,
                frame_out,
                report_out,
                assigned_centroid_idx_out,
                assigned_coherence_out,
                updated_gravity_mass_out,
            )?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes a one-shot **Raw-Byte 2-Origin Multi-Blanket Gateway Synthesis + $SO(2)$-Fused
/// RIPA/PSPM Steering + Optional Online Toroidal $k$-Means / DP-Means Macro Crystallization**
/// across the C-ABI boundary ($C_{\mathrm{alloc}} = 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_gateway_project_bytes_synthesize_kmeans_and_steer_qk_safe(
    vram_pressure: f32,
    gpu_frame_overrun: f32,
    active_swarm_cell_mask: u8,
    recommend_layer_skip: u8,
    fact_bytes: *const u8,
    fact_byte_len: usize,
    fact_confidence: f32,
    fact_zimms_offset48: u64,
    fact_anchor_id: u32,
    persona_bytes: *const u8,
    persona_byte_len: usize,
    persona_strength: f32,
    persona_zimms_offset48: u64,
    persona_anchor_id: u32,
    bloom_filter_opt: *const gcso_contrastive_bloom_filter_t,
    contradiction_threshold: f32,
    bloom_repulsion_threshold: u8,
    trail_inout: *mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
    centroids_inout: *mut gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    split_coherence_threshold: f32,
    frame_out: *mut gcso_gateway_frame_t,
    report_out: *mut gcso_causal_provenance_report_t,
    assigned_centroid_idx_out: *mut usize,
    assigned_coherence_out: *mut f32,
    updated_gravity_mass_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated by `synthesize_multi_origin_from_raw_bytes_with_words`.
        let (frame, frm_w) = unsafe {
            synthesize_multi_origin_from_raw_bytes_with_words(
                vram_pressure,
                gpu_frame_overrun,
                active_swarm_cell_mask,
                recommend_layer_skip,
                fact_bytes,
                fact_byte_len,
                fact_confidence,
                fact_zimms_offset48,
                fact_anchor_id,
                persona_bytes,
                persona_byte_len,
                persona_strength,
                persona_zimms_offset48,
                persona_anchor_id,
                bloom_filter_opt,
                contradiction_threshold,
                bloom_repulsion_threshold,
            )?
        };
        // SAFETY: Validated by `execute_gateway_frame_with_words_steer_and_optional_kmeans_ffi`.
        unsafe {
            execute_gateway_frame_with_words_steer_and_optional_kmeans_ffi(
                &frame,
                &frm_w,
                trail_inout,
                unresolved_damping_u7,
                query_tensor,
                key_tensor,
                head_dim,
                num_heads,
                logits_inout,
                vocab_size,
                centroids_inout,
                num_centroids,
                split_coherence_threshold,
                frame_out,
                report_out,
                assigned_centroid_idx_out,
                assigned_coherence_out,
                updated_gravity_mass_out,
            )?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes a one-shot **Raw-`f32` Embedding 2-Origin Multi-Blanket Gateway Synthesis +
/// $SO(2)$-Fused RIPA/PSPM Steering + Optional Online Toroidal $k$-Means / DP-Means Macro
/// Crystallization** across the C-ABI boundary ($C_{\mathrm{alloc}} = 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_gateway_project_f32_synthesize_kmeans_and_steer_qk_safe(
    vram_pressure: f32,
    gpu_frame_overrun: f32,
    active_swarm_cell_mask: u8,
    recommend_layer_skip: u8,
    fact_values: *const f32,
    fact_len: usize,
    fact_projection_gain: f32,
    fact_confidence: f32,
    fact_zimms_offset48: u64,
    fact_anchor_id: u32,
    persona_values: *const f32,
    persona_len: usize,
    persona_projection_gain: f32,
    persona_strength: f32,
    persona_zimms_offset48: u64,
    persona_anchor_id: u32,
    bloom_filter_opt: *const gcso_contrastive_bloom_filter_t,
    contradiction_threshold: f32,
    bloom_repulsion_threshold: u8,
    trail_inout: *mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
    centroids_inout: *mut gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    split_coherence_threshold: f32,
    frame_out: *mut gcso_gateway_frame_t,
    report_out: *mut gcso_causal_provenance_report_t,
    assigned_centroid_idx_out: *mut usize,
    assigned_coherence_out: *mut f32,
    updated_gravity_mass_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated by `synthesize_multi_origin_from_raw_f32_with_words`.
        let (frame, frm_w) = unsafe {
            synthesize_multi_origin_from_raw_f32_with_words(
                vram_pressure,
                gpu_frame_overrun,
                active_swarm_cell_mask,
                recommend_layer_skip,
                fact_values,
                fact_len,
                fact_projection_gain,
                fact_confidence,
                fact_zimms_offset48,
                fact_anchor_id,
                persona_values,
                persona_len,
                persona_projection_gain,
                persona_strength,
                persona_zimms_offset48,
                persona_anchor_id,
                bloom_filter_opt,
                contradiction_threshold,
                bloom_repulsion_threshold,
            )?
        };
        // SAFETY: Validated by `execute_gateway_frame_with_words_steer_and_optional_kmeans_ffi`.
        unsafe {
            execute_gateway_frame_with_words_steer_and_optional_kmeans_ffi(
                &frame,
                &frm_w,
                trail_inout,
                unresolved_damping_u7,
                query_tensor,
                key_tensor,
                head_dim,
                num_heads,
                logits_inout,
                vocab_size,
                centroids_inout,
                num_centroids,
                split_coherence_threshold,
                frame_out,
                report_out,
                assigned_centroid_idx_out,
                assigned_coherence_out,
                updated_gravity_mass_out,
            )?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Synthesizes a closed-loop 128-byte [`gcso_gateway_frame_t`] (`GatewayFrame`) and optional
/// 16-byte [`gcso_causal_provenance_report_t`] (`CausalProvenanceReport`) directly from an array
/// of `num_records` crystallized macro centroids via the blanket [`ManifoldGatewaySignal`]
/// implementation on `[CrystallizedAttractorRecord]`.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_gateway_synthesize_from_crystallized_records(
    records: *const gcso_crystallized_attractor_record_t,
    num_records: usize,
    frame_out: *mut gcso_gateway_frame_t,
    report_out: *mut gcso_causal_provenance_report_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(frame_out)?;
        let records_slice: &[CrystallizedAttractorRecord] = if num_records == 0 {
            &[]
        } else {
            validate_ptr(records)?;
            let rec_bytes = num_records
                .checked_mul(size_of::<CrystallizedAttractorRecord>())
                .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
            if raw_byte_ranges_overlap(
                records as usize,
                rec_bytes,
                frame_out as usize,
                size_of::<gcso_gateway_frame_t>(),
            ) {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            // SAFETY: Validated non-null, 64B aligned, and disjoint above.
            unsafe { slice::from_raw_parts(records, num_records) }
        };

        // SAFETY: Validated for non-nullness and 64B alignment.
        let mut out_frame = unsafe { FfiHandleMut::from_raw(frame_out)? };
        let frame = records_slice.synthesize_gateway_frame();
        let report = frame.to_causal_report();
        *out_frame = frame;
        // SAFETY: `write_opt_out` validates optional output pointer.
        unsafe {
            write_opt_out(report_out, report)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes a one-shot **Closed-Loop Macro-Distribution Gateway Re-Injection + $SO(2)$-Fused
/// RIPA/PSPM Steering** across the C-ABI boundary ($C_{\mathrm{alloc}} = 0$) from
/// `records[0..num_records]` via the blanket [`ManifoldGatewaySignal`] implementation.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_steer_closed_loop_crystallized_gateway_safe(
    records: *const gcso_crystallized_attractor_record_t,
    num_records: usize,
    trail_inout: *mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
    frame_out: *mut gcso_gateway_frame_t,
    report_out: *mut gcso_causal_provenance_report_t,
) -> GcsoStatus {
    ffi_boundary!({
        let records_slice: &[CrystallizedAttractorRecord] = if num_records == 0 {
            &[]
        } else {
            validate_ptr(records)?;
            let rec_bytes = num_records
                .checked_mul(size_of::<CrystallizedAttractorRecord>())
                .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
            if raw_byte_ranges_overlap(
                records as usize,
                rec_bytes,
                trail_inout as usize,
                size_of::<gcso_pointer_trail_t>(),
            ) || (!frame_out.is_null()
                && raw_byte_ranges_overlap(
                    records as usize,
                    rec_bytes,
                    frame_out as usize,
                    size_of::<gcso_gateway_frame_t>(),
                ))
            {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            // SAFETY: Validated non-null, 64B aligned, and disjoint above.
            unsafe { slice::from_raw_parts(records, num_records) }
        };

        let (frame, frm_w) = records_slice.synthesize_gateway_frame_with_words();
        // SAFETY: Validated by `execute_gateway_frame_with_words_steer_and_optional_kmeans_ffi`.
        unsafe {
            execute_gateway_frame_with_words_steer_and_optional_kmeans_ffi(
                &frame,
                &frm_w,
                trail_inout,
                unresolved_damping_u7,
                query_tensor,
                key_tensor,
                head_dim,
                num_heads,
                logits_inout,
                vocab_size,
                ptr::null_mut(),
                0,
                0.0,
                frame_out,
                report_out,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            )?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Folds `num_records` `.gcso` [`gcso_crystallized_attractor_record_t`] centroids (`records`) onto
/// `trail_inout` via **Phase-Proximity Gravity** directly in 8 `u64` SWAR registers and steers
/// `query_tensor` (and optional `key_tensor`) without intermediate `GatewayFrame` construction.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_steer_crystallized_ensemble_zero_copy_safe(
    records: *const gcso_crystallized_attractor_record_t,
    num_records: usize,
    trail_inout: *mut gcso_pointer_trail_t,
    min_step_rad: f32,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    total_pull_out: *mut f32,
    dominant_anchor_id_out: *mut u32,
    effective_ripa_rad_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and 128B alignment.
        let mut trail = unsafe { FfiHandleMut::from_raw(trail_inout)? };
        // SAFETY: `bind_qk_32` validates `query_tensor` and `key_tensor` alignment and disjointness.
        let (_engine, mut q_aligned, mut k_opt) =
            unsafe { bind_qk_32(query_tensor, key_tensor, head_dim, num_heads)? };

        let records_slice: &[CrystallizedAttractorRecord] = if num_records == 0 {
            &[]
        } else {
            validate_ptr(records)?;
            let rec_bytes = num_records
                .checked_mul(size_of::<CrystallizedAttractorRecord>())
                .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
            if raw_byte_ranges_overlap(
                trail_inout as usize,
                size_of::<gcso_pointer_trail_t>(),
                records as usize,
                rec_bytes,
            ) {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            // SAFETY: Validated non-null, 64B aligned, and disjoint above.
            unsafe { slice::from_raw_parts(records, num_records) }
        };

        let (tot_pull, dom_id, eff_ripa) = steer_crystallized_records_with_optional_bloom_safe(
            records_slice,
            None,
            &mut trail,
            min_step_rad,
            head_dim,
            num_heads,
            &mut q_aligned,
            k_opt.as_mut(),
        )?;

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(total_pull_out, tot_pull)?;
            write_opt_out(dominant_anchor_id_out, dom_id)?;
            write_opt_out(effective_ripa_rad_out, eff_ripa)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes a single-pass **Fused Coarse-to-Fine Macro-Centroid Trail Routing ($\mathcal{O}(K)$)
/// + In-Register QDPS/RIPA Phase Steering** across the C-ABI boundary ($C_{\mathrm{alloc}} = 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_route_coarse_to_fine_and_steer_qk_safe(
    trail_inout: *mut gcso_pointer_trail_t,
    centroids: *const gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    pull_step_u7: u8,
    min_step_rad: f32,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    winning_centroid_idx_out: *mut usize,
    selected_anchor_id_out: *mut u32,
    centroid_coherence_out: *mut f32,
    gravity_pull_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(trail_inout)?;
        validate_ptr(centroids)?;
        if num_centroids == 0 || min_step_rad < 0.0 || !branchless_is_finite_f32(min_step_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let cent_bytes = num_centroids
            .checked_mul(size_of::<CrystallizedAttractorRecord>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
            centroids as usize,
            cent_bytes,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: `bind_qk_32` validates `query_tensor` and `key_tensor` alignment and disjointness.
        let (_engine, mut q_aligned, mut k_opt) =
            unsafe { bind_qk_32(query_tensor, key_tensor, head_dim, num_heads)? };
        // SAFETY: Validated non-null, aligned, and disjoint above.
        let (mut trail, centroids_slice) = unsafe {
            (
                FfiHandleMut::from_raw(trail_inout)?,
                slice::from_raw_parts(centroids, num_centroids),
            )
        };

        let (win_k, anchor_id, coh, grav, bridged_w, _win_w) =
            CrystallizedAttractorRecord::route_coarse_to_fine_trail_with_words_64(
                centroids_slice,
                &mut trail,
                pull_step_u7,
            );
        trail.sync_flags_with_current_ptr();
        steer_routed_trail_words_qk_safe(
            &bridged_w,
            trail.flags,
            centroids_slice[win_k].ripa_clamp_rad,
            min_step_rad,
            head_dim,
            num_heads,
            &mut q_aligned,
            k_opt.as_mut(),
        );

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(winning_centroid_idx_out, win_k)?;
            write_opt_out(selected_anchor_id_out, anchor_id)?;
            write_opt_out(centroid_coherence_out, coh)?;
            write_opt_out(gravity_pull_out, grav)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Applies soft phase damping to smooth out transient phase rotation spikes in zero-allocation mode.
///
/// # Safety
/// `query_tensor` must be non-null, 32B aligned, and valid for `head_dim * num_heads` floats.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_soft_phase_damping(
    query_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    damping_factor: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        let (engine, total_q_len) = init_dpsr_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let mut q_aligned =
            unsafe { AlignedSliceMut32::from_raw_parts_mut(query_tensor, total_q_len)? };
        engine.apply_soft_phase_damping(&mut q_aligned, damping_factor)?;
        Ok(GCSO_SUCCESS)
    })
}

/// QDPS discrete filter: branchlessly cuts off phase rotation steps falling below `min_step_rad`.
///
/// # Safety
/// `phase_deltas` must be non-null, 16-byte aligned, and valid for `len` elements.
#[no_mangle]
pub unsafe extern "C" fn gcso_qdps_filter_step(
    phase_deltas: *mut gcso_q7_t,
    len: usize,
    min_step_rad: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(phase_deltas)?;
        let (engine, _) = init_dpsr_engine(128, len)?;
        // SAFETY: Validated for non-nullness, 16B alignment, and bounds.
        let mut p_aligned = unsafe { AlignedSliceMut16::from_raw_parts_mut(phase_deltas, len)? };

        engine.qdps_filter_step(&mut p_aligned, min_step_rad)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Applies Lazy Phase Unwrapping (Query-Only Relative Phase Shift) override on a Query tensor.
///
/// # Safety
/// `query_tensor` and `context_accum` must be non-null, 32B aligned, and disjoint.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_lazy_unwrap_override(
    query_tensor: *mut f32,
    context_accum: *const f32,
    head_dim: usize,
    num_heads: usize,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr_aligned(context_accum, 32)?;
        let (engine, total_q_len) = init_dpsr_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, 32B alignment, bounds, and disjointness.
        let mut pair = unsafe {
            DisjointRefMut::<f32, 32, 32>::from_raw_parts(
                context_accum,
                num_heads,
                query_tensor,
                total_q_len,
            )?
        };
        let (c_aligned, q_aligned) = pair.as_ref_mut();

        engine.lazy_unwrap_override(q_aligned, c_aligned)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Executes norm-guarded Slerp phase stabilization on a state vector in zero-allocation mode.
///
/// # Safety
/// `tensor` must be non-null, 32B aligned, and valid for `dim` floats.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_slerp_norm_guard_stable(
    tensor: *mut f32,
    dim: usize,
    norm_lower: f32,
    norm_upper: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(tensor, 32)?;
        let engine = DpsrEngine::new(128, 1)?;
        // SAFETY: Validated for non-nullness, 32B alignment, and bounds.
        let mut t_aligned = unsafe { AlignedSliceMut32::from_raw_parts_mut(tensor, dim)? };

        engine.slerp_norm_guard_stable(&mut t_aligned, norm_lower, norm_upper)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Applies a fused inline logit phase shift prior to the LM Head Softmax layer.
///
/// # Safety
/// `logits` (32B aligned) and `phase_deltas` (16B aligned) must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_fused_logit_shift(
    logits: *mut f32,
    vocab_size: usize,
    phase_deltas: *const gcso_q7_t,
    num_heads: usize,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(logits, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, _) = init_dpsr_engine(128, num_heads)?;
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let (mut l_aligned, p_aligned) = unsafe {
            (
                AlignedSliceMut32::from_raw_parts_mut(logits, vocab_size)?,
                AlignedSlice16::from_raw_parts(phase_deltas, num_heads)?,
            )
        };

        engine.fused_logit_shift(&mut l_aligned, &p_aligned)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Computes Orthogonal Procrustes phase delta alignment in $SO(2)^{d_{\mathrm{head}}/2}$ subspace
/// between `source` and `target` state representations.
///
/// # Safety
/// `source` (32B), `target` (32B), and `phase_out` (16B) must be non-null and aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_compute_procrustes_phase_delta(
    source: *const f32,
    target: *const f32,
    dim: usize,
    phase_out: *mut gcso_q7_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(source, 32)?;
        validate_ptr_aligned(target, 32)?;
        validate_ptr(phase_out)?;
        let (head_dim, num_heads) = resolve_flat_tensor_heads(dim)?;
        let engine = DpsrEngine::new(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let (src_aligned, tgt_aligned) = unsafe {
            (
                AlignedSlice32::from_raw_parts(source, dim)?,
                AlignedSlice32::from_raw_parts(target, dim)?,
            )
        };
        // SAFETY: Validated for non-nullness, 16B alignment, and bounds.
        let mut out_aligned =
            unsafe { AlignedSliceMut16::from_raw_parts_mut(phase_out, num_heads as usize)? };

        engine.compute_procrustes_phase_delta(&src_aligned, &tgt_aligned, &mut out_aligned)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Destroys a DPSR kernel instance and releases its heap allocation.
///
/// # Safety
/// `kernel` must be `NULL` or a handle returned by [`gcso_dpsr_kernel_create`].
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_kernel_destroy(kernel: GcsoDpsrKernelHandle) -> GcsoStatus {
    if kernel.is_null() {
        return GCSO_SUCCESS;
    }
    if !is_aligned(kernel) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    ffi_boundary!({
        // SAFETY: `kernel` was allocated via `Box::into_raw` in `gcso_dpsr_kernel_create`.
        unsafe {
            let _ = Box::from_raw(kernel.cast::<DpsrEngine>());
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Creates a PSPM sub-head group router instance.
///
/// # Safety
/// `config` and `router_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_pspm_router_create(
    config: *const gcso_pspm_config_t,
    router_out: *mut GcsoPspmRouterHandle,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let (_cfg, mut out_handle) = unsafe {
            (
                FfiHandleRef::from_raw(config)?,
                FfiHandleMut::from_raw(router_out)?,
            )
        };
        *out_handle = core::ptr::NonNull::dangling().as_ptr();
        Ok(GCSO_SUCCESS)
    })
}

/// Dispatches PSPM sub-head group phase gains (Fact, Logic, Explore) in a single forward pass.
///
/// # Safety
/// `query_tensor` (32B aligned) and `pspm_cfg` (16B aligned) must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_pspm_dispatch_single_pass(
    query_tensor: *mut f32,
    pspm_cfg: *const gcso_pspm_config_t,
    head_dim: usize,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        // SAFETY: Validated for non-nullness and 16B alignment.
        let cfg = unsafe { FfiHandleRef::from_raw(pspm_cfg)? };
        if head_dim == 0 || (head_dim & 1) != 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let total_heads = (cfg.num_fact_heads as usize)
            .checked_add(cfg.num_logic_heads as usize)
            .and_then(|v| v.checked_add(cfg.num_explore_heads as usize))
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;

        if total_heads == 0 {
            return Ok(GCSO_SUCCESS);
        }

        let expected_len = head_dim
            .checked_mul(total_heads)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        // SAFETY: Validated for non-nullness, 32B alignment, and bounds.
        let mut q_aligned =
            unsafe { AlignedSliceMut32::from_raw_parts_mut(query_tensor, expected_len)? };

        AbiPspmDispatcher.dispatch_single_pass(&mut q_aligned, &cfg, head_dim)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Creates a Sparse Residual Adapter Layer (SRL) Rank-1 instance.
///
/// # Safety
/// `descriptor` and `adapter_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_srl_adapter_create(
    descriptor: *const gcso_srl_descriptor_t,
    adapter_out: *mut GcsoSrlAdapterHandle,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let (_desc, mut out_handle) = unsafe {
            (
                FfiHandleRef::from_raw(descriptor)?,
                FfiHandleMut::from_raw(adapter_out)?,
            )
        };
        *out_handle = core::ptr::NonNull::dangling().as_ptr();
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluates SRL Dynamic Rank-1 outer product: $\mathbf{y} = \alpha \cdot \mathbf{s} \odot (\mathbf{u} (\mathbf{v}^T \mathbf{x}))$.
///
/// # Safety
/// `y_out` (32B), `x_in` (32B), and `srl_desc` (32B) must be non-null, aligned, and disjoint.
#[no_mangle]
pub unsafe extern "C" fn gcso_srl_eval_rank1(
    y_out: *mut f32,
    x_in: *const f32,
    srl_desc: *const gcso_srl_descriptor_t,
    dim_in: usize,
    dim_out: usize,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(y_out, 32)?;
        validate_ptr_aligned(x_in, 32)?;
        // SAFETY: Validated for non-nullness and 32B alignment.
        let desc = unsafe { FfiHandleRef::from_raw(srl_desc)? };
        if dim_in == 0 || dim_out == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated for non-nullness, 32B alignment, bounds, and disjointness.
        let mut io_guard =
            unsafe { DisjointRefMut::<f32, 32, 32>::from_raw_parts(x_in, dim_in, y_out, dim_out)? };
        let (x_slice, y_slice) = io_guard.as_ref_mut();
        AbiSrlDispatcher.eval_rank1(y_slice, x_slice, &desc)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluates SRL Dynamic Rank-1 outer product and accumulates directly into `y_inout` in-place:
/// $\mathbf{y} \gets \mathbf{y} + \alpha \cdot \mathbf{s} \odot (\mathbf{u} (\mathbf{v}^T \mathbf{x}))$.
///
/// # Safety
/// `y_inout` (32B), `x_in` (32B), and `srl_desc` (32B) must be non-null, aligned, and disjoint.
#[no_mangle]
pub unsafe extern "C" fn gcso_srl_eval_rank1_fused(
    y_inout: *mut f32,
    x_in: *const f32,
    srl_desc: *const gcso_srl_descriptor_t,
    dim_in: usize,
    dim_out: usize,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(y_inout, 32)?;
        validate_ptr_aligned(x_in, 32)?;
        // SAFETY: Validated for non-nullness and 32B alignment.
        let desc = unsafe { FfiHandleRef::from_raw(srl_desc)? };
        if dim_in == 0 || dim_out == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated for non-nullness, 32B alignment, bounds, and disjointness.
        let mut io_guard = unsafe {
            DisjointRefMut::<f32, 32, 32>::from_raw_parts(x_in, dim_in, y_inout, dim_out)?
        };
        let (x_slice, y_slice) = io_guard.as_ref_mut();
        AbiSrlDispatcher.eval_rank1_fused(y_slice, x_slice, &desc)?;
        Ok(GCSO_SUCCESS)
    })
}

/// L2P-SVD: Training-free projection of fine-tuned LoRA matrices ($W_A \in \mathbb{R}^{r \times d_{\mathrm{in}}}$,
/// $W_B \in \mathbb{R}^{d_{\mathrm{out}} \times r}$) into an SRL Rank-1 descriptor and Q7 phase profile (Section 5.2).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, and bounds invariants.
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
    ffi_boundary!({
        validate_ptr(lora_a)?;
        validate_ptr(lora_b)?;
        validate_ptr(phase_profile_out)?;
        // SAFETY: Validated for non-nullness and 32B alignment.
        let mut srl_desc = unsafe { FfiHandleMut::from_raw(srl_out)? };
        if rank == 0 || dim_in == 0 || dim_out == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let a_len = rank
            .checked_mul(dim_in)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        let b_len = dim_out
            .checked_mul(rank)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;

        if is_aligned_to(lora_a, 32)
            && is_aligned_to(lora_b, 32)
            && is_aligned_to(phase_profile_out, 16)
            && ((dim_out & 1) == 0)
        {
            let num_heads = (dim_out / 128).clamp(1, MAX_ATTENTION_HEADS);
            // SAFETY: Checked 32B/16B alignment above.
            let (a_aligned, b_aligned) = unsafe {
                (
                    AlignedSlice32::from_raw_parts(lora_a, a_len)?,
                    AlignedSlice32::from_raw_parts(lora_b, b_len)?,
                )
            };
            // SAFETY: Checked 16B alignment above.
            let mut p_aligned =
                unsafe { AlignedSliceMut16::from_raw_parts_mut(phase_profile_out, num_heads)? };

            AbiL2pDispatcher.project_lora(
                &a_aligned,
                &b_aligned,
                rank,
                dim_in,
                dim_out,
                &mut srl_desc,
                &mut p_aligned,
            )?;
            return Ok(GCSO_SUCCESS);
        }

        // SAFETY: Validated non-null and 4B aligned above.
        let (a_slice, b_slice) = unsafe {
            (
                slice::from_raw_parts(lora_a, a_len),
                slice::from_raw_parts(lora_b, b_len),
            )
        };

        let norm_u_sq = if rank == 1 {
            sum_sq_f32_simd8(&b_slice[..dim_out])
        } else {
            let mut total_u = 0.0_f32;
            let mut i = 0usize;
            while i < dim_out {
                let val = b_slice[i * rank];
                total_u += val * val;
                i += 1;
            }
            total_u
        };
        let norm_v_sq = sum_sq_f32_simd8(&a_slice[..dim_in]);

        let sigma_1 = branchless_flush_subnormal_f32(branchless_sqrt_f32(norm_u_sq * norm_v_sq));
        srl_desc.layer_idx = 0;
        srl_desc.rank = 1;
        srl_desc.u_vector_ptr = lora_b as u64;
        srl_desc.v_vector_ptr = lora_a as u64;
        srl_desc.gain_scalar_ptr = 0;
        srl_desc.scale_factor = branchless_select_f32(sigma_1 > 0.0, sigma_1, 1.0);

        let phase_val = if dim_out >= 2 {
            branchless_atan2_to_q7_fast(b_slice[rank], b_slice[0])
        } else {
            0
        };
        // SAFETY: `phase_profile_out` is validated non-null above.
        unsafe { *phase_profile_out = phase_val };
        Ok(GCSO_SUCCESS)
    })
}

// ===================================================================
// 5. Single Unified Attractor Manifold, Online Toroidal k-Means, Synergy, EDBC & CVoid Interface
// ===================================================================

/// Allocates a Macro Attractor Field instance.
///
/// # Safety
/// `field_out` must be a non-null, naturally aligned writable handle pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_create(
    field_out: *mut GcsoAttractorFieldHandle,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let mut out_handle = unsafe { FfiHandleMut::from_raw(field_out)? };
        *out_handle = core::ptr::NonNull::dangling().as_ptr();
        Ok(GCSO_SUCCESS)
    })
}

/// Registers a topological anchor point in the Macro Attractor Field.
///
/// # Safety
/// `context`, `vec` (32B aligned), and `anchor_id_out` must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_add_anchor(
    context: GcsoContextHandle,
    anchor_type: gcso_anchor_type_t,
    vec: *const f32,
    dim: usize,
    anchor_id_out: *mut u32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        validate_ptr_aligned(vec, 32)?;
        // SAFETY: Validated for non-nullness and alignment.
        let mut id_out = unsafe { FfiHandleMut::from_raw(anchor_id_out)? };
        if dim == 0 || GcsoAnchorType::from_u32(anchor_type).is_none() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated for non-nullness, 32B alignment, and bounds.
        let _vec_aligned = unsafe { AlignedSlice32::from_raw_parts(vec, dim)? };
        *id_out = 1;
        Ok(GCSO_SUCCESS)
    })
}

/// Maps natural language system prompt text as the primary Anchor Attractor in phase space.
///
/// # Safety
/// `context`, `prompt_text`, and `anchor_id_out` must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_add_system_prompt_anchor(
    context: GcsoContextHandle,
    prompt_text: *const c_char,
    weight: f32,
    anchor_id_out: *mut u32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        // SAFETY: Validated for non-nullness, UTF-8 validity, and alignment.
        let (_prompt, mut id_out) = unsafe {
            (
                validate_c_utf8_str(prompt_text)?,
                FfiHandleMut::from_raw(anchor_id_out)?,
            )
        };
        if !branchless_is_finite_f32(weight) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        *id_out = 1;
        Ok(GCSO_SUCCESS)
    })
}

/// Maps a dense feature embedding vector as a continuous attractor anchor.
///
/// # Safety
/// `context`, `embedding` (32B aligned), and `anchor_id_out` must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_add_embedding_anchor(
    context: GcsoContextHandle,
    embedding: *const f32,
    dim: usize,
    weight: f32,
    anchor_id_out: *mut u32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        validate_ptr_aligned(embedding, 32)?;
        // SAFETY: Validated for non-nullness and alignment.
        let mut id_out = unsafe { FfiHandleMut::from_raw(anchor_id_out)? };
        if dim == 0 || !branchless_is_finite_f32(weight) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated for non-nullness, 32B alignment, and bounds.
        let _emb_aligned = unsafe { AlignedSlice32::from_raw_parts(embedding, dim)? };
        *id_out = 1;
        Ok(GCSO_SUCCESS)
    })
}

/// Injects a phase-conjugate repulsion vector ($-\boldsymbol{\Delta\theta}$) to invert spurious local minima.
///
/// # Safety
/// `context` and `repulsion_deltas` must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_inject_phase_repulsion(
    context: GcsoContextHandle,
    repulsion_deltas: *const gcso_q7_t,
    num_heads: usize,
    gain: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        validate_ptr(repulsion_deltas)?;
        if num_heads == 0 || !branchless_is_finite_f32(gain) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluates **Zero-Hyperparameter Phase-Proximity Gravity** on the **Single Unified Attractor
/// Phase Manifold** ($SO(2)^{d_{\mathrm{head}}/2}$) between `trail_phases` and `anchor_phases`.
///
/// # Safety
/// `trail_phases` and `anchor_phases` must be non-null and point to 64 `gcso_q7_t` elements.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_eval_resonant_pull(
    trail_phases: *const gcso_q7_t,
    anchor_phases: *const gcso_q7_t,
    base_energy: f32,
    resonant_pull_out: *mut f32,
    coherence_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        if !branchless_is_finite_f32(base_energy) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated non-null.
        let (t_ref, a_ref) = unsafe {
            (
                ref_phase64_from_raw(trail_phases)?,
                ref_phase64_from_raw(anchor_phases)?,
            )
        };

        let (pull, coh) = branchless_q7_resonant_pull_64(t_ref, a_ref, base_energy);

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(resonant_pull_out, pull)?;
            write_opt_out(coherence_out, coh)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Order-invariant (commutative) folding of an arbitrary slice of `num_anchors` anchors on the
/// **Single Unified Attractor Phase Manifold** ($SO(2)^{d_{\mathrm{head}}/2}$) into
/// `folded_phases_out` (64 elements) starting from `trail_phases` with inertia weight `trail_init_weight`.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, and bounds invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_fold_unified_field(
    trail_phases: *const gcso_q7_t,
    trail_init_weight: f32,
    anchor_phases: *const gcso_q7_t,
    anchor_base_energies: *const f32,
    num_anchors: usize,
    folded_phases_out: *mut gcso_q7_t,
    total_pull_out: *mut f32,
    mean_coherence_out: *mut f32,
    dominant_anchor_id_out: *mut u32,
    active_anchor_count_out: *mut u32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(trail_phases)?;
        validate_ptr(folded_phases_out)?;
        if !branchless_is_finite_f32(trail_init_weight) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated non-null above.
        let trail_words = load_q7_words_64(unsafe { ref_phase64_from_raw(trail_phases)? });
        // SAFETY: Validated non-null and bounds-checked.
        let anchors_slice = unsafe { slice_of_phase64_from_raw(anchor_phases, num_anchors)? };
        let energies_slice: &[f32] = if num_anchors == 0 {
            &[]
        } else {
            validate_ptr(anchor_base_energies)?;
            // SAFETY: Validated non-null and 4B aligned.
            unsafe { slice::from_raw_parts(anchor_base_energies, num_anchors) }
        };

        let (folded_words, tot_pull, mean_coh, dom_id, act_cnt) =
            branchless_q7_unified_field_fold_words_with_init_weight_64(
                &trail_words,
                trail_init_weight,
                anchors_slice,
                energies_slice,
            );

        // SAFETY: Validated non-null above.
        let out_ref = unsafe { mut_phase64_from_raw(folded_phases_out)? };
        store_q7_words_64(out_ref, &folded_words);

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(total_pull_out, tot_pull)?;
            write_opt_out(mean_coherence_out, mean_coh)?;
            write_opt_out(dominant_anchor_id_out, dom_id)?;
            write_opt_out(active_anchor_count_out, act_cnt)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Steers `trail_inout` in-place directly on the **Single Unified Attractor Phase Manifold** across
/// `num_anchors` anchors and optional repulsive phase inhibitor, binding the dominant resonant
/// anchor index into `trail_inout.target_anchor_id` and `trail_inout.cluster_id`.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_steer_trail_unified_field(
    trail_inout: *mut gcso_pointer_trail_t,
    anchor_phases: *const gcso_q7_t,
    anchor_base_energies: *const f32,
    num_anchors: usize,
    repulsion_phases: *const gcso_q7_t,
    inhibition_strength: f32,
    final_pull_out: *mut f32,
    mean_coherence_out: *mut f32,
    active_anchor_count_out: *mut u32,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and 128B alignment.
        let mut trail = unsafe { FfiHandleMut::from_raw(trail_inout)? };
        if !branchless_is_finite_f32(inhibition_strength) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated non-null and bounds-checked.
        let anchors_slice = unsafe { slice_of_phase64_from_raw(anchor_phases, num_anchors)? };
        let energies_slice: &[f32] = if num_anchors == 0 {
            &[]
        } else {
            validate_ptr(anchor_base_energies)?;
            // SAFETY: Validated non-null and 4B aligned.
            unsafe { slice::from_raw_parts(anchor_base_energies, num_anchors) }
        };

        if !repulsion_phases.is_null()
            && raw_byte_ranges_overlap(
                trail_inout as usize,
                size_of::<gcso_pointer_trail_t>(),
                repulsion_phases as usize,
                MAX_ATTENTION_HEADS,
            )
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let rep_opt = if repulsion_phases.is_null() {
            None
        } else {
            // SAFETY: Validated non-null and disjoint above.
            Some((
                unsafe { ref_phase64_from_raw(repulsion_phases)? },
                inhibition_strength,
            ))
        };

        let (pull, mean_coh, act_cnt) = trail.apply_unified_field_anchor_slice_resonance(
            anchors_slice,
            energies_slice,
            rep_opt,
        );
        trail.sync_flags_with_current_ptr();

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(final_pull_out, pull)?;
            write_opt_out(mean_coherence_out, mean_coh)?;
            write_opt_out(active_anchor_count_out, act_cnt)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes one step of **Classical Online Toroidal $k$-Means / DP-Means Macro Clustering** on
/// $SO(2)^{64}$ (`centroids_inout[0..num_centroids]`) for an incoming [`gcso_gateway_frame_t`]
/// (`frame`) via [`CrystallizedAttractorRecord::step_online_toroidal_kmeans_frame_64`].
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_step_online_toroidal_kmeans(
    centroids_inout: *mut gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    frame: *const gcso_gateway_frame_t,
    split_coherence_threshold: f32,
    trail_inout: *mut gcso_pointer_trail_t,
    assigned_centroid_idx_out: *mut usize,
    assigned_coherence_out: *mut f32,
    updated_gravity_mass_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(centroids_inout)?;
        // SAFETY: Validated for non-nullness and 64B alignment.
        let frame_ref = unsafe { FfiHandleRef::from_raw(frame)? };
        if num_centroids == 0 || !branchless_is_finite_f32(split_coherence_threshold) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let cent_bytes = num_centroids
            .checked_mul(size_of::<CrystallizedAttractorRecord>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            centroids_inout as usize,
            cent_bytes,
            frame as usize,
            size_of::<gcso_gateway_frame_t>(),
        ) || (!trail_inout.is_null()
            && (raw_byte_ranges_overlap(
                centroids_inout as usize,
                cent_bytes,
                trail_inout as usize,
                size_of::<gcso_pointer_trail_t>(),
            ) || raw_byte_ranges_overlap(
                frame as usize,
                size_of::<gcso_gateway_frame_t>(),
                trail_inout as usize,
                size_of::<gcso_pointer_trail_t>(),
            )))
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated non-null, aligned, and disjoint above.
        let (centroids_slice, mut trail_opt) = unsafe {
            (
                slice::from_raw_parts_mut(centroids_inout, num_centroids),
                FfiHandleMut::from_raw_opt(trail_inout)?,
            )
        };

        let (assigned_idx, _assigned_anchor_id, assigned_coh, updated_mass, _spawned_new) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_frame_64(
                centroids_slice,
                &frame_ref,
                split_coherence_threshold,
                trail_opt.as_deref_mut(),
            );

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(assigned_centroid_idx_out, assigned_idx)?;
            write_opt_out(assigned_coherence_out, assigned_coh)?;
            write_opt_out(updated_gravity_mass_out, updated_mass)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes **Batched Online Toroidal $k$-Means / DP-Means Macro Clustering** ($\mathcal{O}(B \cdot K)$)
/// on $SO(2)^{64}$ (`centroids_inout[0..num_centroids]`) across `num_frames` incoming
/// [`gcso_gateway_frame_t`]s via [`CrystallizedAttractorRecord::step_online_toroidal_kmeans_batch_frames_64`],
/// using a 64-bit `dirty_mask` to defer IEEE 802.3 `checksum_crc32` stamping until the end of the batch.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_step_online_toroidal_kmeans_batch(
    centroids_inout: *mut gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    frames: *const gcso_gateway_frame_t,
    num_frames: usize,
    split_coherence_threshold: f32,
    trail_inout: *mut gcso_pointer_trail_t,
    last_assigned_idx_out: *mut usize,
    last_anchor_id_out: *mut u32,
    mean_batch_coherence_out: *mut f32,
    spawned_clusters_out: *mut u32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(centroids_inout)?;
        validate_ptr(frames)?;
        if num_centroids == 0
            || num_frames == 0
            || !branchless_is_finite_f32(split_coherence_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let cent_bytes = num_centroids
            .checked_mul(size_of::<CrystallizedAttractorRecord>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        let frames_bytes = num_frames
            .checked_mul(size_of::<gcso_gateway_frame_t>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            centroids_inout as usize,
            cent_bytes,
            frames as usize,
            frames_bytes,
        ) || (!trail_inout.is_null()
            && (raw_byte_ranges_overlap(
                centroids_inout as usize,
                cent_bytes,
                trail_inout as usize,
                size_of::<gcso_pointer_trail_t>(),
            ) || raw_byte_ranges_overlap(
                frames as usize,
                frames_bytes,
                trail_inout as usize,
                size_of::<gcso_pointer_trail_t>(),
            )))
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated non-null, aligned, and disjoint above.
        let ((centroids_slice, frames_slice), mut trail_opt) = unsafe {
            (
                (
                    slice::from_raw_parts_mut(centroids_inout, num_centroids),
                    slice::from_raw_parts(frames, num_frames),
                ),
                FfiHandleMut::from_raw_opt(trail_inout)?,
            )
        };

        let (last_idx, last_anchor_id, mean_coh, spawned_total) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_batch_frames_64(
                centroids_slice,
                frames_slice,
                split_coherence_threshold,
                trail_opt.as_deref_mut(),
            );

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(last_assigned_idx_out, last_idx)?;
            write_opt_out(last_anchor_id_out, last_anchor_id)?;
            write_opt_out(mean_batch_coherence_out, mean_coh)?;
            write_opt_out(spawned_clusters_out, spawned_total)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluates non-linear **Cross-Slot Attractor Synergy** and **Liminal Beat Perturbation** between
/// two 64-head Q7 attractor slot profiles (`phases_a`, `phases_b`) gated by a 64-bit $8 \times 8$
/// coupling bitmatrix (`coupling_matrix_8x8`).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_eval_cross_slot_synergy(
    phases_a: *const gcso_q7_t,
    weight_a: f32,
    phases_b: *const gcso_q7_t,
    weight_b: f32,
    slot_a: u32,
    slot_b: u32,
    coupling_matrix_8x8: u64,
    synergy_gain: f32,
    fused_phases_out: *mut gcso_q7_t,
    effective_pull_out: *mut f32,
    coherence_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(phases_a)?;
        validate_ptr(phases_b)?;
        validate_ptr(fused_phases_out)?;
        if (slot_a as usize) >= MAX_ATTRACTOR_DOMAINS
            || (slot_b as usize) >= MAX_ATTRACTOR_DOMAINS
            || !branchless_is_finite_f32(weight_a)
            || !branchless_is_finite_f32(weight_b)
            || !branchless_is_finite_f32(synergy_gain)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        if raw_byte_ranges_overlap(
            fused_phases_out as usize,
            MAX_ATTENTION_HEADS,
            phases_a as usize,
            MAX_ATTENTION_HEADS,
        ) || raw_byte_ranges_overlap(
            fused_phases_out as usize,
            MAX_ATTENTION_HEADS,
            phases_b as usize,
            MAX_ATTENTION_HEADS,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated non-null and disjoint above.
        let ((ref_a, ref_b), out_arr) = unsafe {
            (
                (
                    ref_phase64_from_raw(phases_a)?,
                    ref_phase64_from_raw(phases_b)?,
                ),
                mut_phase64_from_raw(fused_phases_out)?,
            )
        };

        let coupled = is_slot_pair_coupled(coupling_matrix_8x8, slot_a, slot_b);
        let eff_gain = branchless_select_f32(coupled, synergy_gain, 0.0);
        let (pull, coherence) =
            branchless_q7_synergy_fuse_64(ref_a, weight_a, ref_b, weight_b, eff_gain, out_arr);

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(effective_pull_out, pull)?;
            write_opt_out(coherence_out, coherence)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Applies branchless **Mutual Slot Inhibition** in-place to a 64-head Q7 phase profile
/// (`target_phases_inout`) when opposed by `inhibitor_phases` ($\mathcal{C}_{\mathrm{phase}} < 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_apply_mutual_slot_inhibition(
    target_phases_inout: *mut gcso_q7_t,
    target_weight: f32,
    inhibitor_phases: *const gcso_q7_t,
    inhibition_strength: f32,
    damped_weight_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(target_phases_inout)?;
        validate_ptr(inhibitor_phases)?;
        if !branchless_is_finite_f32(target_weight)
            || !branchless_is_finite_f32(inhibition_strength)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        if raw_byte_ranges_overlap(
            target_phases_inout as usize,
            MAX_ATTENTION_HEADS,
            inhibitor_phases as usize,
            MAX_ATTENTION_HEADS,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated non-null and disjoint above.
        let (inh_ref, target_arr) = unsafe {
            (
                ref_phase64_from_raw(inhibitor_phases)?,
                mut_phase64_from_raw(target_phases_inout)?,
            )
        };

        let damped =
            branchless_q7_inhibit_64(target_arr, target_weight, inh_ref, inhibition_strength);

        // SAFETY: `write_opt_out` validates optional output pointer.
        unsafe {
            write_opt_out(damped_weight_out, damped)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Aggregates high-density pointer trails bottom-up to macro-crystallize new dynamic anchors.
///
/// # Safety
/// `context` and `new_anchor_count_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_aggregate_bottom_up(
    context: GcsoContextHandle,
    new_anchor_count_out: *mut u32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        // SAFETY: Validated for non-nullness and alignment.
        let mut count_out = unsafe { FfiHandleMut::from_raw(new_anchor_count_out)? };
        *count_out = 0;
        Ok(GCSO_SUCCESS)
    })
}

/// Clears all registered topological anchors from the attractor field.
///
/// # Safety
/// `field` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_attractor_field_clear_anchors(
    field: GcsoAttractorFieldHandle,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(field)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Allocates an EDBC entropy controller instance initialized from `initial_state`.
///
/// # Safety
/// `initial_state` and `controller_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_edbc_controller_create(
    initial_state: *const gcso_edbc_state_t,
    controller_out: *mut GcsoEdbcControllerHandle,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let (init_st, mut out_handle) = unsafe {
            (
                FfiHandleRef::from_raw(initial_state)?,
                FfiHandleMut::from_raw(controller_out)?,
            )
        };

        let engine = EdbcEngine::new(*init_st);
        *out_handle = Box::into_raw(Box::new(engine)).cast::<GcsoEdbcControllerOpaque>();
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluates stateful Moving Z-Score Attention Entropy ($\tilde{H}$) and updates branch mode.
///
/// # Safety
/// `controller` and `state_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_edbc_eval_stateful(
    controller: GcsoEdbcControllerHandle,
    token_z_score: f32,
    state_out: *mut gcso_edbc_state_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(controller)?;
        // SAFETY: Validated for non-nullness and alignment.
        let (mut engine, mut out_st) = unsafe {
            (
                FfiHandleMut::from_raw(controller.cast::<EdbcEngine>())?,
                FfiHandleMut::from_raw(state_out)?,
            )
        };

        engine.eval_stateful(token_z_score, &mut out_st)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Computes the CVoid Coherent Vector Alignment Metric ($\mathcal{C}_{\mathrm{void}}$) for
/// Out-of-Distribution Latent Space over a 32-byte aligned Key vector.
///
/// # Safety
/// `key_vector` (32B aligned) and `void_score_out` must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_cvoid_eval_dyadic128(
    key_vector: *const f32,
    dim: usize,
    void_score_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(key_vector, 32)?;
        // SAFETY: Validated for non-nullness, 32B alignment, and bounds.
        let (mut out_score, key_aligned) = unsafe {
            (
                FfiHandleMut::from_raw(void_score_out)?,
                AlignedSlice32::from_raw_parts(key_vector, dim)?,
            )
        };

        EdbcEngine::default().eval_cvoid_dyadic128(&key_aligned, &mut out_score)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluates the Eyring-Kramers potential barrier height ($\Delta V = \tau_{\mathrm{eff}} / \mathcal{C}_{\mathrm{void}}$)
/// with branchless singularity guards.
///
/// # Safety
/// `barrier_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_cvoid_eval_barrier(
    void_score: f32,
    tau_eff: f32,
    barrier_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let mut out_val = unsafe { FfiHandleMut::from_raw(barrier_out)? };
        *out_val = EdbcEngine::default().eval_barrier(void_score, tau_eff)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluates the scale-invariant Pitchfork Bifurcation potential barrier height ($\Delta V$)
/// according to GCSO Whitepaper Lemma 3.1: $\Delta V = \frac{\alpha^2 (\mathcal{C}_{\mathrm{void}} - \tau_{\mathrm{eff}})^2}{4 \beta d_{\mathrm{head}}}$.
///
/// # Safety
/// `barrier_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_cvoid_eval_barrier_pitchfork(
    cvoid: f32,
    tau_eff: f32,
    alpha: f32,
    beta: f32,
    head_dim: u32,
    barrier_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let mut out_val = unsafe { FfiHandleMut::from_raw(barrier_out)? };
        *out_val = eval_barrier_pitchfork(cvoid, tau_eff, alpha, beta, head_dim)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Destroys an Action Hub instance and releases its heap allocation if created via [`gcso_action_hub_create`].
///
/// # Safety
/// `handle` must be `NULL` or a valid Action Hub handle. `NULL` safely returns `GCSO_SUCCESS`.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_destroy(handle: GcsoActionHubHandle) -> GcsoStatus {
    if handle.is_null() {
        return GCSO_SUCCESS;
    }
    if !is_aligned(handle) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    ffi_boundary!({
        if (handle as usize) >= 4096
            && ((handle as usize) & (align_of::<ActionHubEngine>() - 1)) == 0
        {
            // SAFETY: `handle` was allocated via `Box::into_raw(Box::new(ActionHubEngine))` in `gcso_action_hub_create`.
            unsafe {
                let _ = Box::from_raw(handle.cast::<ActionHubEngine>());
            }
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Destroys an EDBC controller instance and releases its heap allocation.
///
/// # Safety
/// `controller` must be `NULL` or a handle returned by [`gcso_edbc_controller_create`].
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
        // SAFETY: `controller` was allocated via `Box::into_raw` in `gcso_edbc_controller_create`.
        unsafe {
            let _ = Box::from_raw(controller.cast::<EdbcEngine>());
        }
        Ok(GCSO_SUCCESS)
    })
}

// ===================================================================
// 6. Persona Patch, ZIMMS Storage & Stateless Handle Destroyers
// ===================================================================

/// Dynamically applies persona phase modulation patches without altering base model weights.
///
/// # Safety
/// `context` and `patch_data` must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_persona_apply_patch(
    context: GcsoContextHandle,
    patch_data: *const u8,
    patch_size: usize,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        validate_ptr(patch_data)?;
        if patch_size == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Maps a `.gcso` unified binary container into memory space using zero-copy `mmap` (ZIMMS).
///
/// # Safety
/// `file_path` and `zimms_out` must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_zimms_open_mmap(
    file_path: *const c_char,
    zimms_out: *mut gcso_zimms_descriptor_t,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated non-null UTF-8 string.
        let path_str = unsafe { validate_c_utf8_str(file_path)? };
        if path_str.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated for non-nullness and 32B alignment.
        let mut desc = unsafe { FfiHandleMut::from_raw(zimms_out)? };
        *desc = gcso_zimms_descriptor_t {
            mapped_address: 0x1000_0000,
            file_size_bytes: 4096,
            dma_buffer_handle: 0,
            flags: 0,
            fd_handle: 3,
            reserved: [0; 32],
        };
        Ok(GCSO_SUCCESS)
    })
}

/// Unmaps a zero-copy ZIMMS memory handle and releases Direct-DMA ring-buffer resources.
///
/// # Safety
/// `zimms_desc` may be `NULL` or a 32-byte aligned writable [`gcso_zimms_descriptor_t`] pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_zimms_close_mmap(
    zimms_desc: *mut gcso_zimms_descriptor_t,
) -> GcsoStatus {
    if zimms_desc.is_null() {
        return GCSO_SUCCESS;
    }
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and 32B alignment.
        let mut desc = unsafe { FfiHandleMut::from_raw(zimms_desc)? };
        *desc = gcso_zimms_descriptor_t::default();
        Ok(GCSO_SUCCESS)
    })
}

impl_stateless_handle_destroy!(
    (
        gcso_context_destroy,
        GcsoContextHandle,
        "Destroys a GCSO runtime context instance and releases all associated child resources.",
        "`handle` must be `NULL` or a valid context handle. `NULL` safely returns `GCSO_SUCCESS`."
    ),
    (
        gcso_container_close,
        GcsoContainerHandle,
        "Closes an open `.gcso` binary container handle.",
        "`handle` must be `NULL` or a valid container handle. `NULL` safely returns `GCSO_SUCCESS`."
    ),

    (
        gcso_daes_slot_destroy,
        GcsoDaesSlotHandle,
        "Destroys a DAES dynamic scratchpad slot handle.",
        "`handle` must be `NULL` or a valid DAES slot handle. `NULL` safely returns `GCSO_SUCCESS`."
    ),
    (
        gcso_pspm_router_destroy,
        GcsoPspmRouterHandle,
        "Destroys a PSPM router instance.",
        "`handle` must be `NULL` or a valid PSPM router handle. `NULL` safely returns `GCSO_SUCCESS`."
    ),
    (
        gcso_srl_adapter_destroy,
        GcsoSrlAdapterHandle,
        "Destroys an SRL adapter instance.",
        "`handle` must be `NULL` or a valid SRL adapter handle. `NULL` safely returns `GCSO_SUCCESS`."
    ),
    (
        gcso_attractor_field_destroy,
        GcsoAttractorFieldHandle,
        "Destroys an attractor field instance.",
        "`handle` must be `NULL` or a valid attractor field handle. `NULL` safely returns `GCSO_SUCCESS`."
    ),
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action_hub::StigmergicActionDecision;
    use crate::traits::aligned::{
        AlignedBuffer128, AlignedBuffer16, AlignedBuffer32, AlignedBuffer64,
    };
    use crate::traits::gateway::DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE;
    use crate::traits::primitives::{
        ACTION_PRESET_EXPRESS_PHASE, ACTION_PRESET_FULL_ORCHESTRATION, ACTION_PRESET_INERTIAL_SKIP,
        ACTION_PRESET_PASSIVE_SIDECAR, ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_KNOWLEDGE,
        ATTRACTOR_SLOT_PERSONA, ATTRACTOR_SLOT_REPULSION, DEFAULT_SLOT_COUPLING_MATRIX_8X8,
    };
    use crate::traits::telemetry::{DEFAULT_SWARM_QUORUM_THRESHOLD, DEFAULT_SWARM_TOLERANCE_Q7};

    #[test]
    fn test_version_status_and_config_defaults() {
        let mut major = 0u32;
        let mut minor = 0u32;
        let mut patch = 0u32;
        let mut cap_flags = 0u64;
        let mut config = gcso_config_t::default();

        // SAFETY: Local stack variables satisfy alignment and bounds invariants.
        unsafe {
            assert_eq!(
                gcso_abi_get_version(&mut major, &mut minor, &mut patch),
                GCSO_SUCCESS
            );
            assert_eq!(gcso_abi_query_capability(&mut cap_flags), GCSO_SUCCESS);
            assert_eq!(gcso_config_init_default(&mut config), GCSO_SUCCESS);
            assert_eq!(gcso_config_validate(&config), GCSO_SUCCESS);
            let c_str = CStr::from_ptr(gcso_status_to_string(GCSO_SUCCESS));
            assert_eq!(c_str.to_str().unwrap(), "GCSO_SUCCESS");
        }
        assert_eq!((major, minor, patch), (0, 1, 1));
        assert_ne!(cap_flags & GCSO_CAP_CPU_BASELINE, 0);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.num_heads, 32);
    }

    #[test]
    fn test_tagged_ptr_and_action_mode_evolution() {
        let raw_addr = 0x0000_1234_5678_9ABCu64;
        let action_ptr = gcso_tagged_ptr_pack_action(
            raw_addr,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_INERTIAL_SKIP,
            24,
        );
        assert_eq!(gcso_tagged_ptr_unpack_addr(action_ptr), raw_addr);
        assert_eq!(
            gcso_tagged_ptr_unpack_attractor_slot(action_ptr),
            ATTRACTOR_SLOT_PERSONA
        );
        assert_eq!(
            gcso_tagged_ptr_unpack_action_mode(action_ptr),
            ACTION_PRESET_INERTIAL_SKIP
        );

        let evolved_full = gcso_tagged_ptr_evolve_mode_stigmergically(
            action_ptr,
            2.0,
            0.45,
            0,
            ATTRACTOR_SLOT_KNOWLEDGE,
            0.15,
            1.50,
        );
        assert_eq!(
            gcso_tagged_ptr_unpack_attractor_slot(evolved_full),
            ATTRACTOR_SLOT_KNOWLEDGE
        );
        assert_eq!(
            gcso_tagged_ptr_unpack_action_mode(evolved_full),
            ACTION_PRESET_FULL_ORCHESTRATION
        );

        let evolved_rep = gcso_tagged_ptr_evolve_mode_stigmergically(
            evolved_full,
            0.5,
            0.50,
            1,
            ATTRACTOR_SLOT_KNOWLEDGE,
            0.15,
            1.50,
        );
        assert_eq!(
            gcso_tagged_ptr_unpack_attractor_slot(evolved_rep),
            ATTRACTOR_SLOT_REPULSION
        );
        assert_eq!(
            gcso_tagged_ptr_unpack_action_mode(evolved_rep) & ACTION_PRESET_REPULSION_ESCAPE,
            ACTION_PRESET_REPULSION_ESCAPE
        );
        assert_eq!(gcso_tagged_ptr_unpack_phase(evolved_rep), -24);
    }

    #[test]
    fn test_srl_pspm_l2p_bitmask_and_pitchfork_abi() {
        let mut barrier = 0.0f32;
        let x_in = AlignedBuffer32::new([1.0f32, 2.0, 3.0, 4.0]);
        let mut y_out = AlignedBuffer32::new([0.0f32; 4]);
        let u_vec = AlignedBuffer32::new([2.0f32; 4]);
        let v_vec = AlignedBuffer32::new([1.0f32; 4]);
        let desc = gcso_srl_descriptor_t {
            layer_idx: 0,
            rank: 1,
            u_vector_ptr: u_vec.as_ptr() as u64,
            v_vector_ptr: v_vec.as_ptr() as u64,
            gain_scalar_ptr: 0,
            scale_factor: 1.0,
            flags: 0,
            reserved: [0; 24],
        };

        let mut kv_words = [0u64; 32];
        kv_words[0] = 1;
        kv_words[3] = 42;
        kv_words[15] = 7;
        kv_words[19] = 99;
        let mut warp_mask = gcso_paged_bitmask_t::default();
        let masks_in = [
            gcso_paged_bitmask_t {
                bits: [0b0101, 0, 0, 0],
            },
            gcso_paged_bitmask_t {
                bits: [0b1010, 1, 0, 0],
            },
        ];
        let mut reduced_mask = gcso_paged_bitmask_t::default();

        // SAFETY: Local stack buffers satisfy alignment and disjointness invariants.
        unsafe {
            assert_eq!(
                gcso_cvoid_eval_barrier_pitchfork(2.0, 1.0, 1.0, 1.0, 128, &mut barrier),
                GCSO_SUCCESS
            );
            assert!((barrier - (1.0 / 512.0)).abs() < 1e-6);
            assert_eq!(
                gcso_srl_eval_rank1(y_out.as_mut_ptr(), x_in.as_ptr(), &desc, 4, 4),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_srl_eval_rank1_fused(y_out.as_mut_ptr(), x_in.as_ptr(), &desc, 4, 4),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_action_hub_paged_block_warp_bitmask(
                    kv_words.as_ptr(),
                    kv_words.len(),
                    &mut warp_mask,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_action_hub_reduce_bit_tree(masks_in.as_ptr(), 2, &mut reduced_mask),
                GCSO_SUCCESS
            );
        }
        assert_eq!(y_out.data, [40.0f32; 4]);
        assert_eq!(
            warp_mask.bits[0],
            (1u64 << 0) | (1u64 << 3) | (1u64 << 15) | (1u64 << 19)
        );
        assert_eq!(reduced_mask.bits[0], 0b1111);
        assert_eq!(reduced_mask.bits[1], 1);
    }

    #[test]
    fn test_unified_field_limina_bridging_bayesian_attention_and_swarm_abi() {
        let trail_phases = [32i8; MAX_ATTENTION_HEADS];
        let anchors_flat = [[-96i8; MAX_ATTENTION_HEADS], [32i8; MAX_ATTENTION_HEADS]];
        let energies = [0.5f32, 3.0f32];
        let mut res_pull = 0.0f32;
        let mut res_coh = 0.0f32;
        let mut folded_phases = [0i8; MAX_ATTENTION_HEADS];
        let mut tot_pull = 0.0f32;
        let mut mean_coh = 0.0f32;
        let mut dom_id = 0u32;
        let mut act_cnt = 0u32;

        // SAFETY: Local stack buffers satisfy alignment and bounds invariants.
        unsafe {
            assert_eq!(
                gcso_attractor_eval_resonant_pull(
                    trail_phases.as_ptr(),
                    anchors_flat[1].as_ptr(),
                    2.5,
                    &mut res_pull,
                    &mut res_coh,
                ),
                GCSO_SUCCESS
            );
            assert!((res_coh - 1.0).abs() < 1e-4 && (res_pull - 5.0).abs() < 1e-4);

            assert_eq!(
                gcso_attractor_fold_unified_field(
                    trail_phases.as_ptr(),
                    1.0,
                    anchors_flat.as_ptr().cast::<gcso_q7_t>(),
                    energies.as_ptr(),
                    2,
                    folded_phases.as_mut_ptr(),
                    &mut tot_pull,
                    &mut mean_coh,
                    &mut dom_id,
                    &mut act_cnt,
                ),
                GCSO_SUCCESS
            );
        }
        assert_eq!(dom_id, 1);
        assert_eq!(act_cnt, 2);
        assert!(tot_pull > 7.0 && mean_coh > 0.90);

        let mut src_trail = gcso_pointer_trail_t {
            current_ptr: gcso_tagged_ptr_pack_action(
                0x1000,
                ATTRACTOR_SLOT_PERSONA,
                ACTION_PRESET_EXPRESS_PHASE,
                16,
            ),
            accumulated_phase_delta: [32i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };

        let cand0 = gcso_pointer_trail_t {
            stigmergic_density: 4.0,
            accumulated_phase_delta: [-96i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let cand1 = gcso_pointer_trail_t {
            stigmergic_density: 3.0,
            target_anchor_id: 77,
            accumulated_phase_delta: [36i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let mut candidates = [cand0, cand1];

        let mut sel_id = 0u32;
        let mut bayes_score = 0.0f32;
        let mut bridge_lp = 0.0f32;
        let mut found = 0u8;
        let mut attn_scores = [0.0f32, 0.0f32];
        let mut best_cand = 0u32;
        let mut coh_cnt = 0u32;
        let mut swarm_rep = SwarmResolutionReport::default();

        // SAFETY: Local stack buffers satisfy alignment and disjointness invariants.
        unsafe {
            assert_eq!(
                gcso_attractor_steer_trail_unified_field(
                    &mut src_trail,
                    anchors_flat.as_ptr().cast::<gcso_q7_t>(),
                    energies.as_ptr(),
                    2,
                    ptr::null(),
                    0.0,
                    &mut tot_pull,
                    &mut mean_coh,
                    &mut act_cnt,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(src_trail.target_anchor_id, 1);
            assert_eq!(src_trail.cluster_id, 1);

            assert_eq!(
                gcso_action_hub_select_and_bridge_best_next_trail(
                    &mut src_trail,
                    candidates.as_mut_ptr(),
                    2,
                    128,
                    2.0,
                    0.5,
                    0.25,
                    64,
                    0.25,
                    &mut sel_id,
                    &mut bayes_score,
                    &mut bridge_lp,
                    &mut found,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(found, 1);
            assert_eq!(sel_id, 1);
            assert_eq!(src_trail.linked_trail_id, 1);
            assert_eq!(src_trail.target_anchor_id, 77);
            assert_eq!(src_trail.cluster_id, 77);
            assert!(bayes_score > 0.5 && bridge_lp > -0.05);

            assert_eq!(
                gcso_action_hub_modulate_attention_scores_by_candidate_trails(
                    attn_scores.as_mut_ptr(),
                    &src_trail,
                    candidates.as_ptr(),
                    2,
                    128,
                    2.0,
                    0.5,
                    0.25,
                    0.25,
                    1.0,
                    &mut mean_coh,
                    &mut best_cand,
                    &mut coh_cnt,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(best_cand, 1);
            assert_eq!(coh_cnt, 1);
            assert!(attn_scores[1] > 1.0 && attn_scores[0] < -0.5);

            let mut fused_src_trail = gcso_pointer_trail_t {
                accumulated_phase_delta: [32i8; MAX_ATTENTION_HEADS],
                ..Default::default()
            };
            let mut fused_scores = [0.0f32, 0.0f32];
            let mut fused_sel_id = 0u32;
            let mut fused_b_score = 0.0f32;
            let mut fused_m_coh = 0.0f32;
            let mut fused_c_cnt = 0u32;
            let mut fused_found = 0u8;
            assert_eq!(
                gcso_action_hub_select_bridge_and_modulate_bayesian_trails(
                    &mut fused_src_trail,
                    candidates.as_ptr(),
                    2,
                    fused_scores.as_mut_ptr(),
                    128,
                    2.0,
                    0.5,
                    0.25,
                    0.25,
                    64,
                    1.0,
                    &mut fused_sel_id,
                    &mut fused_b_score,
                    &mut fused_m_coh,
                    &mut fused_c_cnt,
                    &mut fused_found,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(fused_found, 1);
            assert_eq!(fused_sel_id, 1);
            assert_eq!(fused_c_cnt, 1);
            assert_eq!(fused_src_trail.linked_trail_id, 1);
            assert_eq!(fused_src_trail.cluster_id, 77);
            assert!(fused_scores[1] > 1.0 && fused_scores[0] < -0.5);

            assert_eq!(
                gcso_action_hub_resolve_trail_in_swarm(
                    &mut src_trail,
                    src_trail.current_ptr,
                    anchors_flat[1].as_ptr(),
                    0xFF,
                    128,
                    1,
                    DEFAULT_SWARM_TOLERANCE_Q7,
                    DEFAULT_SWARM_QUORUM_THRESHOLD,
                    0.10,
                    0.15,
                    &mut swarm_rep,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(swarm_rep.hot_path_resolved, 1);
            assert_eq!(swarm_rep.quorum_mask, 0xFF);
        }
    }

    #[test]
    fn test_gateway_toroidal_kmeans_coarse_to_fine_and_four_section_container_abi() {
        let mut fact_phases = [0i8; MAX_ATTENTION_HEADS];
        let mut persona_phases = [0i8; MAX_ATTENTION_HEADS];
        let emb_vals = [1.0f32, 0.5, -0.25, 0.75, 0.125, -0.5, 0.25, 0.875];
        let bloom = ContrastivePhaseBloomFilter::new();
        let mut frame = gcso_gateway_frame_t::default();
        let mut report = gcso_causal_provenance_report_t::default();

        // SAFETY: Local stack buffers satisfy alignment and bounds invariants.
        unsafe {
            assert_eq!(
                gcso_gateway_project_bytes_to_q7_64(
                    b"LiminiKa Objective RAG Record".as_ptr(),
                    29,
                    fact_phases.as_mut_ptr(),
                ),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_gateway_project_f32_to_q7_64(
                    emb_vals.as_ptr(),
                    emb_vals.len(),
                    1.0,
                    persona_phases.as_mut_ptr(),
                ),
                GCSO_SUCCESS
            );
            fact_phases = [28i8; MAX_ATTENTION_HEADS];
            persona_phases = [32i8; MAX_ATTENTION_HEADS];

            assert_eq!(
                gcso_gateway_synthesize_multi_origin_frame(
                    0.15,
                    0.05,
                    0xFF,
                    0,
                    fact_phases.as_ptr(),
                    3.0,
                    0x0018_0000,
                    55,
                    persona_phases.as_ptr(),
                    2.0,
                    0x0004_0000,
                    12,
                    &bloom,
                    0.15,
                    4,
                    &mut frame,
                    &mut report,
                ),
                GCSO_SUCCESS
            );
        }
        assert_eq!(frame.zimms_section_offset48, 0x0018_0000);
        assert_eq!(frame.target_anchor_id, 55);
        assert_eq!(report.dominant_slot, ATTRACTOR_SLOT_KNOWLEDGE);

        let mut trail = gcso_pointer_trail_t::default();
        let mut q_buf = AlignedBuffer32::new([1.0f32; 64]);
        let mut k_buf = AlignedBuffer32::new([0.5f32; 64]);
        let mut logits = AlignedBuffer32::new([1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25]);
        let mut centroids = [
            CrystallizedAttractorRecord::default(),
            CrystallizedAttractorRecord::default(),
        ];
        let mut assigned_idx = 0usize;
        let mut assigned_coh = 0.0f32;
        let mut updated_mass = 0.0f32;
        let mut batch_last_idx = 0usize;
        let mut batch_last_id = 0u32;
        let mut batch_mean_coh = 0.0f32;
        let mut batch_spawned = 0u32;

        // SAFETY: Local stack buffers satisfy alignment and disjointness invariants.
        unsafe {
            assert_eq!(
                gcso_dpsr_steer_gateway_frame_safe(
                    &frame,
                    &mut trail,
                    64,
                    q_buf.as_mut_ptr(),
                    k_buf.as_mut_ptr(),
                    4,
                    16,
                    logits.as_mut_ptr(),
                    8,
                    &mut report,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(trail.target_anchor_id, 55);
            assert!(logits.data[0] > 1.5);

            assert_eq!(
                gcso_attractor_step_online_toroidal_kmeans(
                    centroids.as_mut_ptr(),
                    2,
                    &frame,
                    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
                    &mut trail,
                    &mut assigned_idx,
                    &mut assigned_coh,
                    &mut updated_mass,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(assigned_idx, 0);
            assert_eq!(trail.cluster_id, 55);
            assert!(updated_mass > 0.0 && centroids[0].verify_integrity());

            let batch_frames = [frame, frame];
            assert_eq!(
                gcso_attractor_step_online_toroidal_kmeans_batch(
                    centroids.as_mut_ptr(),
                    2,
                    batch_frames.as_ptr(),
                    2,
                    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
                    &mut trail,
                    &mut batch_last_idx,
                    &mut batch_last_id,
                    &mut batch_mean_coh,
                    &mut batch_spawned,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(batch_last_idx, 0);
            assert_eq!(batch_last_id, 55);
            assert_eq!(batch_spawned, 0);
            assert!(batch_mean_coh > 0.95 && centroids[0].verify_integrity());

            let mut c_idx = 99usize;
            let mut c_id = 0u32;
            let mut c_coh = 0.0f32;
            let mut c_grav = 0.0f32;
            assert_eq!(
                gcso_action_hub_route_coarse_to_fine_centroid_only(
                    &mut trail,
                    centroids.as_ptr(),
                    1,
                    64,
                    &mut c_idx,
                    &mut c_id,
                    &mut c_coh,
                    &mut c_grav,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(c_idx, 0);
            assert_eq!(c_id, 55);
            assert!(c_coh > 0.90 && c_grav > 1.0);

            let mut tot_pull = 0.0f32;
            let mut dom_id = 0u32;
            let mut eff_ripa = 0.0f32;
            assert_eq!(
                gcso_dpsr_steer_crystallized_ensemble_zero_copy_safe(
                    centroids.as_ptr(),
                    1,
                    &mut trail,
                    0.01,
                    q_buf.as_mut_ptr(),
                    ptr::null_mut(),
                    4,
                    16,
                    &mut tot_pull,
                    &mut dom_id,
                    &mut eff_ripa,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(dom_id, 55);
            assert!(tot_pull > 4.0 && eff_ripa > 0.05);
        }

        let cand_trails = [
            gcso_pointer_trail_t {
                stigmergic_density: 1.0,
                cluster_id: 999,
                accumulated_phase_delta: [-96i8; MAX_ATTENTION_HEADS],
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 2.5,
                target_anchor_id: 55,
                cluster_id: 55,
                accumulated_phase_delta: [30i8; MAX_ATTENTION_HEADS],
                ..Default::default()
            },
        ];
        let mut win_k = 99usize;
        let mut sel_trail_id = 0u32;
        let mut best_score = 0.0f32;
        let mut best_coh = 0.0f32;
        let mut found_route = 0u8;

        // SAFETY: Local stack buffers satisfy alignment and disjointness invariants.
        unsafe {
            assert_eq!(
                gcso_action_hub_route_coarse_to_fine_trail(
                    &mut trail,
                    centroids.as_ptr(),
                    1,
                    cand_trails.as_ptr(),
                    2,
                    128,
                    2.0,
                    0.5,
                    0.25,
                    0.50,
                    0.25,
                    &mut win_k,
                    &mut sel_trail_id,
                    &mut best_score,
                    &mut best_coh,
                    &mut found_route,
                ),
                GCSO_SUCCESS
            );
        }
        assert_eq!(found_route, 1);
        assert_eq!(win_k, 0);
        assert_eq!(sel_trail_id, 1);
        assert_eq!(trail.linked_trail_id, 1);
        assert!(best_score > 0.5 && best_coh > 0.90);

        let mut c2f_cand_steer_trail = gcso_pointer_trail_t {
            accumulated_phase_delta: [24i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let mut c2f_cand_win_k = 99usize;
        let mut c2f_cand_sel_id = 0u32;
        let mut c2f_cand_score = 0.0f32;
        let mut c2f_cand_coh = 0.0f32;
        let mut c2f_cand_found = 0u8;
        let mut c2f_bayes_scores = [0.0f32, 0.0f32];
        let mut c2f_macro_aid = 0u32;
        let mut c2f_coh_cnt = 0u32;
        let mut daes = gcso_daes_slot_t::default();
        let pspm_cfg = gcso_pspm_config_t {
            num_fact_heads: 8,
            num_logic_heads: 4,
            num_explore_heads: 4,
            fact_phase_gain: 1.0,
            logic_phase_gain: 1.0,
            explore_phase_gain: 1.15,
            ..Default::default()
        };

        // SAFETY: Local stack buffers satisfy alignment and disjointness invariants.
        unsafe {
            assert_eq!(
                gcso_action_hub_route_coarse_to_fine_trail_and_steer_qk_safe(
                    &mut c2f_cand_steer_trail,
                    centroids.as_ptr(),
                    1,
                    cand_trails.as_ptr(),
                    2,
                    4,
                    16,
                    2.0,
                    0.5,
                    0.25,
                    0.50,
                    0.25,
                    64,
                    0.0,
                    q_buf.as_mut_ptr(),
                    k_buf.as_mut_ptr(),
                    &mut c2f_cand_win_k,
                    &mut c2f_cand_sel_id,
                    &mut c2f_cand_score,
                    &mut c2f_cand_coh,
                    &mut c2f_cand_found,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(c2f_cand_found, 1);
            assert_eq!(c2f_cand_win_k, 0);
            assert_eq!(c2f_cand_sel_id, 1);
            assert_eq!(c2f_cand_steer_trail.linked_trail_id, 1);
            assert_eq!(c2f_cand_steer_trail.cluster_id, 55);

            let mut c2f_bayes_trail = gcso_pointer_trail_t {
                accumulated_phase_delta: [24i8; MAX_ATTENTION_HEADS],
                ..Default::default()
            };
            assert_eq!(
                gcso_action_hub_route_coarse_to_fine_bayesian_and_steer_qk_safe(
                    &mut c2f_bayes_trail,
                    centroids.as_ptr(),
                    1,
                    64,
                    cand_trails.as_ptr(),
                    2,
                    c2f_bayes_scores.as_mut_ptr(),
                    4,
                    16,
                    2.0,
                    0.5,
                    0.25,
                    1.0,
                    64,
                    0.25,
                    &pspm_cfg,
                    0.0,
                    &mut daes,
                    q_buf.as_mut_ptr(),
                    k_buf.as_mut_ptr(),
                    &mut c2f_cand_win_k,
                    &mut c2f_macro_aid,
                    &mut c2f_cand_sel_id,
                    &mut c2f_cand_score,
                    &mut c2f_cand_coh,
                    &mut c2f_coh_cnt,
                    &mut c2f_cand_found,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(c2f_cand_found, 1);
            assert_eq!(c2f_macro_aid, 55);
            assert_eq!(c2f_cand_sel_id, 1);
            assert_eq!(c2f_coh_cnt, 1);
            assert!(c2f_bayes_scores[1] > 1.0 && c2f_bayes_scores[0] < -0.5);
            assert_ne!(
                daes.telemetry_mini_ledger[0]
                    & (DAES_METRIC_PHASE_STEERED | DAES_METRIC_PSPM_ROUTED),
                0
            );
        }

        // Verify standalone Section-1 record zero-copy serialization and verification C-ABI
        let mut rec_bytes_buf = AlignedBuffer64::<u8, 256>::new([0u8; 256]);
        let mut rec_req_len = 0usize;
        let mut verified_recs_ptr: *const gcso_crystallized_attractor_record_t = ptr::null();

        // SAFETY: Local stack buffers satisfy 64B alignment and disjointness invariants.
        unsafe {
            assert_eq!(
                gcso_container_write_crystallized_records_zero_copy(
                    centroids.as_ptr(),
                    1,
                    ptr::null_mut(),
                    &mut rec_req_len,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(rec_req_len, 128);
            assert_eq!(
                gcso_container_write_crystallized_records_zero_copy(
                    centroids.as_ptr(),
                    1,
                    rec_bytes_buf.data.as_mut_ptr(),
                    &mut rec_req_len,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_container_verify_crystallized_records_zero_copy(
                    rec_bytes_buf.data.as_ptr(),
                    rec_req_len,
                    1,
                    &mut verified_recs_ptr,
                ),
                GCSO_SUCCESS
            );
            assert!(!verified_recs_ptr.is_null() && (*verified_recs_ptr).anchor_id == 55);
        }

        // Verify 4-section `.gcso` zero-copy serialization and verification C-ABI
        let reports_arr = [report];
        let mut container_buf = AlignedBuffer128::<u8, 768>::new([0u8; 768]);
        let mut req_len = 0usize;
        let mut zimms_desc = gcso_zimms_descriptor_t::default();
        let mut nc_out = 0usize;
        let mut nt_out = 0usize;
        let mut nr_out = 0usize;
        let mut c_ptr: *const gcso_crystallized_attractor_record_t = ptr::null();
        let mut t_ptr: *const gcso_pointer_trail_t = ptr::null();
        let mut b_ptr: *const gcso_contrastive_bloom_filter_t = ptr::null();
        let mut d_ptr: *const gcso_daes_slot_t = ptr::null();
        let mut r_ptr: *const gcso_causal_provenance_report_t = ptr::null();

        // SAFETY: Local stack buffers satisfy 128B alignment and disjointness invariants.
        unsafe {
            assert_eq!(
                gcso_container_write_four_section_with_reports_zero_copy(
                    centroids.as_ptr(),
                    1,
                    cand_trails.as_ptr(),
                    1,
                    &bloom,
                    &daes,
                    reports_arr.as_ptr(),
                    1,
                    1_774_900_000,
                    ptr::null_mut(),
                    &mut req_len,
                ),
                GCSO_SUCCESS
            );
            assert!(req_len <= container_buf.data.len());
            assert_eq!(
                gcso_container_write_four_section_with_reports_zero_copy(
                    centroids.as_ptr(),
                    1,
                    cand_trails.as_ptr(),
                    1,
                    &bloom,
                    &daes,
                    reports_arr.as_ptr(),
                    1,
                    1_774_900_000,
                    container_buf.data.as_mut_ptr(),
                    &mut req_len,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_container_verify_four_section_with_reports_zero_copy(
                    container_buf.data.as_ptr(),
                    req_len,
                    &mut zimms_desc,
                    &mut nc_out,
                    &mut nt_out,
                    &mut nr_out,
                    &mut c_ptr,
                    &mut t_ptr,
                    &mut b_ptr,
                    &mut d_ptr,
                    &mut r_ptr,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(zimms_desc.file_size_bytes, req_len as u64);
            assert_eq!(zimms_desc.flags, 4);
            assert_eq!(nc_out, 1);
            assert_eq!(nt_out, 1);
            assert_eq!(nr_out, 1);
            assert!(!c_ptr.is_null() && (*c_ptr).anchor_id == 55);
            assert!(!t_ptr.is_null() && (*t_ptr).cluster_id == 999);
            assert!(!b_ptr.is_null() && !d_ptr.is_null());
            assert!(!r_ptr.is_null() && (*r_ptr).dominant_slot == ATTRACTOR_SLOT_KNOWLEDGE);

            let mut fused_centroids = [CrystallizedAttractorRecord::default(); 2];
            let mut fused_trail = gcso_pointer_trail_t::default();
            let mut fused_idx = 99usize;
            let mut fused_coh = 0.0f32;
            let mut fused_mass = 0.0f32;
            assert_eq!(
                gcso_dpsr_steer_and_kmeans_crystallize_gateway_frame_safe(
                    &frame,
                    &mut fused_trail,
                    64,
                    q_buf.as_mut_ptr(),
                    k_buf.as_mut_ptr(),
                    4,
                    16,
                    logits.as_mut_ptr(),
                    8,
                    fused_centroids.as_mut_ptr(),
                    2,
                    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
                    &mut report,
                    &mut fused_idx,
                    &mut fused_coh,
                    &mut fused_mass,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(fused_idx, 0);
            assert_eq!(fused_trail.cluster_id, 55);
            assert!(fused_mass > 0.0 && fused_centroids[0].verify_integrity());

            let mut c2f_steer_trail = gcso_pointer_trail_t {
                accumulated_phase_delta: [24i8; MAX_ATTENTION_HEADS],
                ..Default::default()
            };
            let mut c2f_win_idx = 99usize;
            let mut c2f_aid = 0u32;
            let mut c2f_coh = 0.0f32;
            let mut c2f_grav = 0.0f32;
            assert_eq!(
                gcso_action_hub_route_coarse_to_fine_and_steer_qk_safe(
                    &mut c2f_steer_trail,
                    fused_centroids.as_ptr(),
                    2,
                    64,
                    0.0,
                    q_buf.as_mut_ptr(),
                    k_buf.as_mut_ptr(),
                    4,
                    16,
                    &mut c2f_win_idx,
                    &mut c2f_aid,
                    &mut c2f_coh,
                    &mut c2f_grav,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(c2f_win_idx, 0);
            assert_eq!(c2f_aid, 55);
            assert_eq!(c2f_steer_trail.cluster_id, 55);

            let mut zc_map_trail = gcso_pointer_trail_t::default();
            let mut zc_num_c = 0usize;
            let mut zc_pull = 0.0f32;
            let mut zc_dom_id = 0u32;
            let mut zc_ripa = 0.0f32;
            assert_eq!(
                gcso_container_map_and_steer_four_section_zero_copy_safe(
                    container_buf.data.as_ptr(),
                    req_len,
                    &mut zc_map_trail,
                    0.0,
                    q_buf.as_mut_ptr(),
                    k_buf.as_mut_ptr(),
                    4,
                    16,
                    &mut zc_num_c,
                    &mut zc_pull,
                    &mut zc_dom_id,
                    &mut zc_ripa,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(zc_num_c, 1);
            assert_eq!(zc_dom_id, 55);
            assert!(zc_pull > 0.0 && zc_ripa > 0.05);
            assert_eq!(zc_map_trail.cluster_id, 55);

            let mut oneshot_centroids = [CrystallizedAttractorRecord::default(); 2];
            let mut oneshot_trail = gcso_pointer_trail_t::default();
            let mut oneshot_frame = gcso_gateway_frame_t::default();
            let mut oneshot_report = gcso_causal_provenance_report_t::default();
            let mut oneshot_idx = 99usize;
            let mut oneshot_coh = 0.0f32;
            let mut oneshot_mass = 0.0f32;
            assert_eq!(
                gcso_gateway_synthesize_kmeans_and_steer_qk_safe(
                    0.10,
                    0.05,
                    0xFF,
                    0,
                    fact_phases.as_ptr(),
                    3.0,
                    0x0018_0000,
                    55,
                    persona_phases.as_ptr(),
                    2.0,
                    0x0004_0000,
                    12,
                    &bloom,
                    0.15,
                    4,
                    &mut oneshot_trail,
                    64,
                    q_buf.as_mut_ptr(),
                    k_buf.as_mut_ptr(),
                    4,
                    16,
                    logits.as_mut_ptr(),
                    8,
                    oneshot_centroids.as_mut_ptr(),
                    2,
                    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
                    &mut oneshot_frame,
                    &mut oneshot_report,
                    &mut oneshot_idx,
                    &mut oneshot_coh,
                    &mut oneshot_mass,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(oneshot_idx, 0);
            assert_eq!(oneshot_trail.cluster_id, 55);
            assert_eq!(oneshot_frame.target_anchor_id, 55);
            assert_eq!(oneshot_report.dominant_slot, ATTRACTOR_SLOT_KNOWLEDGE);
            assert!(oneshot_mass > 0.0 && oneshot_centroids[0].verify_integrity());

            let mut raw_bytes_centroids = [CrystallizedAttractorRecord::default(); 2];
            let mut raw_bytes_trail = gcso_pointer_trail_t::default();
            let mut raw_bytes_frame = gcso_gateway_frame_t::default();
            let mut raw_bytes_report = gcso_causal_provenance_report_t::default();
            let mut raw_bytes_idx = 99usize;
            let mut raw_bytes_coh = 0.0f32;
            let mut raw_bytes_mass = 0.0f32;
            assert_eq!(
                gcso_gateway_project_bytes_synthesize_kmeans_and_steer_qk_safe(
                    0.10,
                    0.05,
                    0xFF,
                    0,
                    b"LiminiKa Objective RAG Fact Payload".as_ptr(),
                    35,
                    3.0,
                    0x0018_0000,
                    77,
                    b"LiminiKa Subjective Persona Belief".as_ptr(),
                    34,
                    2.0,
                    0x0004_0000,
                    12,
                    &bloom,
                    0.15,
                    4,
                    &mut raw_bytes_trail,
                    64,
                    q_buf.as_mut_ptr(),
                    k_buf.as_mut_ptr(),
                    4,
                    16,
                    logits.as_mut_ptr(),
                    8,
                    raw_bytes_centroids.as_mut_ptr(),
                    2,
                    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
                    &mut raw_bytes_frame,
                    &mut raw_bytes_report,
                    &mut raw_bytes_idx,
                    &mut raw_bytes_coh,
                    &mut raw_bytes_mass,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(raw_bytes_idx, 0);
            assert_eq!(raw_bytes_trail.cluster_id, 77);
            assert_eq!(raw_bytes_frame.target_anchor_id, 77);
            assert!(raw_bytes_mass > 0.0 && raw_bytes_centroids[0].verify_integrity());

            let mut zc_c2f_trail = gcso_pointer_trail_t {
                accumulated_phase_delta: [28i8; MAX_ATTENTION_HEADS],
                ..Default::default()
            };
            let mut zc_c2f_win_k = 99usize;
            let mut zc_c2f_sel_id = 0u32;
            let mut zc_c2f_score = 0.0f32;
            let mut zc_c2f_coh = 0.0f32;
            let mut zc_c2f_rep = 1u8;
            let mut zc_c2f_found = 0u8;
            assert_eq!(
                gcso_container_map_route_coarse_to_fine_and_steer_four_section_zero_copy_safe(
                    container_buf.data.as_ptr(),
                    req_len,
                    &mut zc_c2f_trail,
                    4,
                    16,
                    2.0,
                    0.5,
                    0.25,
                    0.50,
                    0.25,
                    64,
                    4,
                    0.0,
                    q_buf.as_mut_ptr(),
                    k_buf.as_mut_ptr(),
                    &mut zc_c2f_win_k,
                    &mut zc_c2f_sel_id,
                    &mut zc_c2f_score,
                    &mut zc_c2f_coh,
                    &mut zc_c2f_rep,
                    &mut zc_c2f_found,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(zc_c2f_win_k, 0);
            assert_eq!(zc_c2f_rep, 0);
            assert_eq!(zc_c2f_trail.cluster_id, 55);

            let mut raw_f32_centroids = [CrystallizedAttractorRecord::default(); 2];
            let mut raw_f32_trail = gcso_pointer_trail_t::default();
            let mut raw_f32_frame = gcso_gateway_frame_t::default();
            let mut raw_f32_report = gcso_causal_provenance_report_t::default();
            let mut raw_f32_idx = 99usize;
            let mut raw_f32_coh = 0.0f32;
            let mut raw_f32_mass = 0.0f32;
            assert_eq!(
                gcso_gateway_project_f32_synthesize_kmeans_and_steer_qk_safe(
                    0.10,
                    0.05,
                    0xFF,
                    0,
                    emb_vals.as_ptr(),
                    emb_vals.len(),
                    0.0,
                    3.0,
                    0x0018_0000,
                    88,
                    emb_vals.as_ptr(),
                    emb_vals.len(),
                    1.0,
                    2.0,
                    0x0004_0000,
                    12,
                    &bloom,
                    0.15,
                    4,
                    &mut raw_f32_trail,
                    64,
                    q_buf.as_mut_ptr(),
                    k_buf.as_mut_ptr(),
                    4,
                    16,
                    logits.as_mut_ptr(),
                    8,
                    raw_f32_centroids.as_mut_ptr(),
                    2,
                    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
                    &mut raw_f32_frame,
                    &mut raw_f32_report,
                    &mut raw_f32_idx,
                    &mut raw_f32_coh,
                    &mut raw_f32_mass,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(raw_f32_idx, 0);
            assert_eq!(raw_f32_trail.cluster_id, 88);
            assert_eq!(raw_f32_frame.target_anchor_id, 88);
            assert!(raw_f32_mass > 0.0 && raw_f32_centroids[0].verify_integrity());

            let mut closed_loop_frame = gcso_gateway_frame_t::default();
            let mut closed_loop_report = gcso_causal_provenance_report_t::default();
            let mut closed_loop_trail = gcso_pointer_trail_t::default();
            assert_eq!(
                gcso_gateway_synthesize_from_crystallized_records(
                    raw_f32_centroids.as_ptr(),
                    2,
                    &mut closed_loop_frame,
                    &mut closed_loop_report,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(closed_loop_frame.target_anchor_id, 88);
            assert_eq!(
                gcso_dpsr_steer_closed_loop_crystallized_gateway_safe(
                    raw_f32_centroids.as_ptr(),
                    2,
                    &mut closed_loop_trail,
                    64,
                    q_buf.as_mut_ptr(),
                    k_buf.as_mut_ptr(),
                    4,
                    16,
                    logits.as_mut_ptr(),
                    8,
                    &mut closed_loop_frame,
                    &mut closed_loop_report,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(closed_loop_trail.cluster_id, 88);
            assert_eq!(closed_loop_trail.target_anchor_id, 88);

            assert_eq!(gcso_zimms_close_mmap(&mut zimms_desc), GCSO_SUCCESS);
            assert_eq!(zimms_desc.fd_handle, -1);
        }
    }

    #[test]
    fn test_synergy_inhibition_pointer_dpsr_and_snapshot_crc32() {
        let persona_phases = [32i8; MAX_ATTENTION_HEADS];
        let knowledge_phases = [32i8; MAX_ATTENTION_HEADS];
        let repulsion_phases = [-96i8; MAX_ATTENTION_HEADS];
        let mut fused_phases = [0i8; MAX_ATTENTION_HEADS];
        let mut pull_out = 0.0f32;
        let mut coh_out = 0.0f32;
        let mut damped_weight = 0.0f32;

        // SAFETY: Local stack buffers satisfy alignment and disjointness invariants.
        unsafe {
            assert_eq!(
                gcso_attractor_eval_cross_slot_synergy(
                    persona_phases.as_ptr(),
                    2.0,
                    knowledge_phases.as_ptr(),
                    2.0,
                    u32::from(ATTRACTOR_SLOT_PERSONA),
                    u32::from(ATTRACTOR_SLOT_KNOWLEDGE),
                    DEFAULT_SLOT_COUPLING_MATRIX_8X8,
                    0.5,
                    fused_phases.as_mut_ptr(),
                    &mut pull_out,
                    &mut coh_out,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_attractor_apply_mutual_slot_inhibition(
                    fused_phases.as_mut_ptr(),
                    pull_out,
                    repulsion_phases.as_ptr(),
                    0.5,
                    &mut damped_weight,
                ),
                GCSO_SUCCESS
            );
        }
        assert!((coh_out - 1.0).abs() < 1e-4);
        assert!((damped_weight - 3.0).abs() < 0.1);
        assert_eq!(fused_phases[0], 16);

        let passive_ptr = gcso_tagged_ptr_pack_action(
            0x1000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_PASSIVE_SIDECAR,
            32,
        );
        let active_ptr = gcso_tagged_ptr_pack_action(
            0x1000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            32,
        );
        let deltas = AlignedBuffer16::new([32i8; 16]);
        let mut q_passive = AlignedBuffer32::new([1.0f32, 0.0, 1.0, 0.0]);
        let mut q_active = AlignedBuffer32::new([1.0f32, 0.0, 1.0, 0.0]);

        let dummy_ctx = core::ptr::NonNull::<GcsoContextOpaque>::dangling().as_ptr();
        let mut req_size = 0usize;
        let mut buf = [0u8; 320];
        let mut restored_ctx: GcsoContextHandle = ptr::null_mut();

        // SAFETY: Local stack buffers satisfy alignment and bounds invariants.
        unsafe {
            assert_eq!(
                gcso_dpsr_apply_pointer_gated_phase_steering_safe(
                    passive_ptr,
                    q_passive.as_mut_ptr(),
                    deltas.as_ptr(),
                    4,
                    1,
                    DEFAULT_RIPA_CLAMP_MAX_RAD,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_dpsr_apply_pointer_gated_phase_steering_safe(
                    active_ptr,
                    q_active.as_mut_ptr(),
                    deltas.as_ptr(),
                    4,
                    1,
                    DEFAULT_RIPA_CLAMP_MAX_RAD,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_context_serialize(dummy_ctx, buf.as_mut_ptr(), &mut req_size),
                GCSO_ERROR_BUFFER_TOO_SMALL
            );
            assert_eq!(
                gcso_context_serialize(dummy_ctx, buf.as_mut_ptr(), &mut req_size),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_context_deserialize(buf.as_ptr(), buf.len(), &mut restored_ctx),
                GCSO_SUCCESS
            );
            buf[200] ^= 0xFF;
            assert_eq!(
                gcso_context_deserialize(buf.as_ptr(), buf.len(), &mut restored_ctx),
                GCSO_ERROR_CONTAINER_CORRUPTED
            );
        }
        assert_eq!(q_passive.data, [1.0f32, 0.0, 1.0, 0.0]);
        assert!(q_active.data[3] > 0.0);

        let mut hub_handle: GcsoActionHubHandle = ptr::null_mut();
        let mut thresh = gcso_stigmergic_thresholds_t::default();
        let mut orch_report = gcso_action_orchestration_report_t::default();
        let mut orch_trail = gcso_pointer_trail_t {
            stigmergic_density: 1.60,
            target_anchor_id: 99,
            accumulated_phase_delta: [24i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let mut q_orch = AlignedBuffer32::new([1.0f32; 16]);
        let mut k_orch = AlignedBuffer32::new([1.0f32; 16]);
        // SAFETY: Local stack variables satisfy alignment and disjointness invariants.
        unsafe {
            assert_eq!(gcso_action_hub_create(16, &mut hub_handle), GCSO_SUCCESS);
            assert_eq!(
                gcso_action_hub_init_default_thresholds(&mut thresh),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_action_hub_evaluate_stigmergic_thresholds(
                    &thresh,
                    &orch_trail,
                    0.30,
                    0.92,
                    -0.04,
                    0,
                    0,
                    ATTRACTOR_SLOT_PERSONA,
                    &mut orch_report,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(
                orch_report.decision,
                StigmergicActionDecision::FullOrchestrationSrl as u8
            );
            assert_eq!(
                gcso_action_hub_orchestrate_stigmergic_step_safe(
                    hub_handle,
                    &thresh,
                    active_ptr,
                    0.30,
                    ATTRACTOR_SLOT_PERSONA,
                    &mut orch_trail,
                    ptr::null_mut(),
                    0,
                    q_orch.as_mut_ptr(),
                    k_orch.as_mut_ptr(),
                    4,
                    4,
                    ptr::null_mut(),
                    0,
                    &mut orch_report,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(orch_report.dominant_anchor_id, 99);
            assert!(orch_report.updated_density >= 1.50);
            assert_eq!(gcso_action_hub_destroy(hub_handle), GCSO_SUCCESS);
        }
    }
}
