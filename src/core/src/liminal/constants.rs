// name: src/core/src/liminal/constants.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Layer 0 fundamental constants, C-ABI configuration/status primitives, causal provenance masks,
//! 2-bit attribution tags, slot coupling bitmatrix helpers, and tagged-pointer codecs.

#![forbid(unsafe_code)]

use core::f32::consts::{LN_2, PI};
use core::fmt;

// ===================================================================
// 1. Unified C-ABI Version & Status Primitives
// ===================================================================

/// Unified GCSO C-ABI Major Version Component.
pub const GCSO_ABI_VERSION_MAJOR: u32 = 0;
/// Unified GCSO C-ABI Minor Version Component.
pub const GCSO_ABI_VERSION_MINOR: u32 = 1;
/// Unified GCSO C-ABI Patch Version Component.
pub const GCSO_ABI_VERSION_PATCH: u32 = 3;
/// Hexadecimal Representation of the active Unified ABI Version (`0x0000_0103`).
pub const GCSO_ABI_VERSION_HEX: u32 =
    (GCSO_ABI_VERSION_MAJOR << 16) | (GCSO_ABI_VERSION_MINOR << 8) | GCSO_ABI_VERSION_PATCH;
/// String Representation of the active Unified ABI Version (derived from `Cargo.toml`).
pub const GCSO_ABI_VERSION_STRING: &str = env!("CARGO_PKG_VERSION");
/// NUL-terminated C string representation of the active Unified ABI Version.
pub const GCSO_ABI_VERSION_CSTR: &core::ffi::CStr = c"0.1.3";

const _: () = {
    let c_bytes = GCSO_ABI_VERSION_CSTR.to_bytes();
    let pkg_bytes = GCSO_ABI_VERSION_STRING.as_bytes();
    assert!(c_bytes.len() == pkg_bytes.len());
    let mut i = 0;
    while i < c_bytes.len() {
        assert!(c_bytes[i] == pkg_bytes[i]);
        i += 1;
    }
};

/// Validates a C-ABI snapshot or descriptor version against [`GCSO_ABI_VERSION_HEX`].
///
/// # Errors
/// Returns [`GCSO_ERROR_VERSION_MISMATCH`] if `version_hex != GCSO_ABI_VERSION_HEX`.
#[inline(always)]
pub const fn validate_abi_version(version_hex: u32) -> Result<(), GcsoStatus> {
    if version_hex == GCSO_ABI_VERSION_HEX {
        Ok(())
    } else {
        Err(GCSO_ERROR_VERSION_MISMATCH)
    }
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
    (
        GCSO_SUCCESS,
        0,
        b"GCSO_SUCCESS\0",
        "GCSO_SUCCESS",
        "Operation completed successfully."
    ),
    (
        GCSO_ERROR_INVALID_ARGUMENT,
        -1,
        b"GCSO_ERROR_INVALID_ARGUMENT\0",
        "GCSO_ERROR_INVALID_ARGUMENT",
        "Invalid argument or out-of-bounds parameter passed to C-ABI function."
    ),
    (
        GCSO_ERROR_OUT_OF_MEMORY,
        -2,
        b"GCSO_ERROR_OUT_OF_MEMORY\0",
        "GCSO_ERROR_OUT_OF_MEMORY",
        "Memory allocation failure or capacity limit reached during Cold Path creation."
    ),
    (
        GCSO_ERROR_BUFFER_TOO_SMALL,
        -3,
        b"GCSO_ERROR_BUFFER_TOO_SMALL\0",
        "GCSO_ERROR_BUFFER_TOO_SMALL",
        "Destination buffer capacity is insufficient for requested payload."
    ),
    (
        GCSO_ERROR_PANIC_CAUGHT,
        -4,
        b"GCSO_ERROR_PANIC_CAUGHT\0",
        "GCSO_ERROR_PANIC_CAUGHT",
        "Unhandled internal Rust panic or C++ exception caught at FFI boundary."
    ),
    (
        GCSO_ERROR_NULL_POINTER,
        -5,
        b"GCSO_ERROR_NULL_POINTER\0",
        "GCSO_ERROR_NULL_POINTER",
        "Unexpected `NULL` pointer supplied for a mandatory parameter."
    ),
    (
        GCSO_ERROR_INVALID_STATE,
        -6,
        b"GCSO_ERROR_INVALID_STATE\0",
        "GCSO_ERROR_INVALID_STATE",
        "Context or component is in an invalid lifecycle state for requested operation."
    ),
    (
        GCSO_ERROR_VERSION_MISMATCH,
        -7,
        b"GCSO_ERROR_VERSION_MISMATCH\0",
        "GCSO_ERROR_VERSION_MISMATCH",
        "Binary snapshot or descriptor ABI version mismatch detected."
    ),
    (
        GCSO_ERROR_MISALIGNED_POINTER,
        -8,
        b"GCSO_ERROR_MISALIGNED_POINTER\0",
        "GCSO_ERROR_MISALIGNED_POINTER",
        "Pointer passed across FFI boundary violates required SIMD or cacheline alignment."
    ),
    (
        GCSO_ERROR_IO_FAILURE,
        -9,
        b"GCSO_ERROR_IO_FAILURE\0",
        "GCSO_ERROR_IO_FAILURE",
        "Input/output or storage persistence operation failure."
    ),
    (
        GCSO_ERROR_ACTION_HUB_FULL,
        -10,
        b"GCSO_ERROR_ACTION_HUB_FULL\0",
        "GCSO_ERROR_ACTION_HUB_FULL",
        "Action Hub pointer table slot capacity fully saturated."
    ),
    (
        GCSO_ERROR_NOT_IMPLEMENTED,
        -11,
        b"GCSO_ERROR_NOT_IMPLEMENTED\0",
        "GCSO_ERROR_NOT_IMPLEMENTED",
        "Feature or compute backend functionality not implemented in current build."
    ),
    (
        GCSO_ERROR_ATTRACTOR_NOT_FOUND,
        -20,
        b"GCSO_ERROR_ATTRACTOR_NOT_FOUND\0",
        "GCSO_ERROR_ATTRACTOR_NOT_FOUND",
        "Target attractor anchor ID not found in topological field."
    ),
    (
        GCSO_ERROR_DPSR_PHASE_OVERFLOW,
        -30,
        b"GCSO_ERROR_DPSR_PHASE_OVERFLOW\0",
        "GCSO_ERROR_DPSR_PHASE_OVERFLOW",
        "Phase steering angle accumulation exceeded safety bound."
    ),
    (
        GCSO_ERROR_QDPS_UNDERFLOW,
        -31,
        b"GCSO_ERROR_QDPS_UNDERFLOW\0",
        "GCSO_ERROR_QDPS_UNDERFLOW",
        "Phase step filter fell below minimal threshold $\\Delta\\theta_{\\mathrm{min\\_step}}$."
    ),
    (
        GCSO_ERROR_EDBC_SINGULARITY,
        -40,
        b"GCSO_ERROR_EDBC_SINGULARITY\0",
        "GCSO_ERROR_EDBC_SINGULARITY",
        "Entropy controller encountered numerical or logarithmic singularity."
    ),
    (
        GCSO_ERROR_CONTAINER_CORRUPTED,
        -50,
        b"GCSO_ERROR_CONTAINER_CORRUPTED\0",
        "GCSO_ERROR_CONTAINER_CORRUPTED",
        "Binary `.gcso` snapshot container header or payload checksum corrupted."
    ),
    (
        GCSO_ERROR_MAPPED_STORAGE_FAILED,
        -51,
        b"GCSO_ERROR_MAPPED_STORAGE_FAILED\0",
        "GCSO_ERROR_MAPPED_STORAGE_FAILED",
        "Mapped-storage zero-copy memory mapping or Direct-DMA buffer mapping failed."
    ),
    (
        GCSO_ERROR_SUBHEAD_ROUTING_FAILED,
        -60,
        b"GCSO_ERROR_SUBHEAD_ROUTING_FAILED\0",
        "GCSO_ERROR_SUBHEAD_ROUTING_FAILED",
        "Sub-head group phase routing evaluation failed."
    ),
    (
        GCSO_ERROR_KV_KEYFRAME_SEEK_FAILED,
        -70,
        b"GCSO_ERROR_KV_KEYFRAME_SEEK_FAILED\0",
        "GCSO_ERROR_KV_KEYFRAME_SEEK_FAILED",
        "KV-keyframe index lookup or token seek operation failed."
    ),
    (
        GCSO_ERROR_OBSTRUCTION_UNRESOLVED,
        -80,
        b"GCSO_ERROR_OBSTRUCTION_UNRESOLVED\0",
        "GCSO_ERROR_OBSTRUCTION_UNRESOLVED",
        "Unresolved cohomological obstruction class $H^1(K; \\mathcal{F})$ encountered."
    ),
    (
        GCSO_ERROR_EXTENSION_NOT_LOADED,
        -90,
        b"GCSO_ERROR_EXTENSION_NOT_LOADED\0",
        "GCSO_ERROR_EXTENSION_NOT_LOADED",
        "Dynamic extension module not loaded or inactive in ledger slot."
    ),
    (
        GCSO_ERROR_LEDGER_FULL,
        -91,
        b"GCSO_ERROR_LEDGER_FULL\0",
        "GCSO_ERROR_LEDGER_FULL",
        "Fast-path ledger table or ring buffer fully occupied."
    ),
    (
        GCSO_ERROR_UNKNOWN,
        -0x7FFF_FFFF,
        b"GCSO_ERROR_UNKNOWN\0",
        "GCSO_ERROR_UNKNOWN",
        "Unknown internal system error."
    ),
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
#[allow(non_camel_case_types)]
pub type gcso_status_t = GcsoStatus;

/// Type alias for GCSO core result responses mapping to [`GcsoStatus`].
pub type GcsoResult<T> = Result<T, GcsoStatus>;

/// Quantized 7-bit signed fixed-point integer (phase scale $\beta_{\mathrm{Q7}} = \pi / 128$).
#[allow(non_camel_case_types)]
pub type gcso_q7_t = i8;

/// Global runtime configuration descriptor structure aligned to 16 bytes (64 bytes total).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[allow(non_camel_case_types)]
pub struct gcso_config_t {
    /// Dimension per attention head ($d_{\mathrm{head}}$, must be positive and even).
    pub head_dim: u32,
    /// Total number of attention heads ($H \le 64$).
    pub num_heads: u32,
    /// Tokens per paged KV cache block (typically 16 or 32).
    pub paged_block_size: u32,
    /// Floating-point scale factor for Q7 phase values ($\beta_{\mathrm{Q7}} = \pi / 128$).
    pub q7_phase_scale: f32,
    /// Maximum allowed rotation angle for soft phase clamping in radians ($\theta_{\max}$).
    pub ripa_clamp_max_rad: f32,
    /// Minimum rotation angle threshold for phase step filtering in radians ($\Delta\theta_{\mathrm{min\_step}}$).
    pub qdps_min_step_rad: f32,
    /// Epsilon clamp to prevent logarithmic singularity in entropy calculation ($\epsilon_{\mathrm{log}}$).
    pub entropy_singularity_eps: f32,
    /// Maximum supported prompt anchor points in attractor field.
    pub max_prompt_anchors: u32,
    /// Maximum slot capacity of the Action Hub pointer table.
    pub action_hub_capacity: u32,
    /// Enable GPU warp shuffle inline evaluations (`1` = enabled, `0` = disabled).
    pub enable_cuda_warp_shuffle: u8,
    /// Strict enforcement of zero dynamic allocations on Hot Path (`1` = strict).
    pub enable_zero_alloc_strict: u8,
    /// Initial operational mode for `FastPathLedger` (`0`, `1`, or `2`).
    pub ledger_mode: u8,
    /// Reserved alignment and configuration flags.
    pub reserved_flags: u8,
    /// Active rotated dimensions per head (`0` defaults to full `head_dim`; smaller for partial RoPE).
    pub rotary_dim: u32,
    /// RoPE channel pairing layout (`0` = Interleaved, `1` = HalfSplit).
    pub rope_layout: u8,
    /// Reserved space for future ABI core extensions (19 bytes).
    pub reserved: [u8; 19],
}

// ===================================================================
// 2. Layer 0 Mathematical & Capacity Constants
// ===================================================================

/// Scale factor for Q7 quantized phase representation ($\beta_{\mathrm{Q7}} = \pi / 128$).
pub const Q7_PHASE_SCALE_RAD: f32 = PI / 128.0;

/// Reciprocal scale factor for converting radians to Q7 fixed-point ($128 / \pi$).
pub const Q7_PHASE_INV_SCALE_RAD: f32 = 128.0 / PI;

/// Unit normalized Q7 scale factor ($\beta_{\mathrm{Q7,unit}} = 1 / 128$).
pub const Q7_UNIT_SCALE: f32 = 1.0 / 128.0;

/// Minimum representable Q7 fixed-point value ($-128 \leftrightarrow -\pi\text{ rad}$).
pub const Q7_MIN: gcso_q7_t = -128;

/// Maximum representable Q7 fixed-point value ($+127 \leftrightarrow +\frac{127}{128}\pi\text{ rad}$).
pub const Q7_MAX: gcso_q7_t = 127;

/// Neutral zero phase in Q7 fixed-point format.
pub const Q7_ZERO: gcso_q7_t = 0;

/// Default logarithmic singularity prevention epsilon ($\epsilon_{\mathrm{log}} = 10^{-12}$).
pub const DEFAULT_ENTROPY_EPS: f32 = 1.0e-12;

/// Upper bound of normalized 2-state surrogate attention entropy ($\ln 2 \approx 0.6931472$).
pub const MAX_SURROGATE_ENTROPY_LN2: f32 = LN_2;

/// Default soft-bounded phase clamp limit in radians ($5^\circ \approx 0.08726646\text{ rad}$).
pub const DEFAULT_PHASE_CLAMP_MAX_RAD: f32 = 0.087_266_46;

/// Canonical ratio `Q7_PHASE_SCALE_RAD / DEFAULT_PHASE_CLAMP_MAX_RAD` mapping Q7 phase deltas to normalized `tanh` units.
pub const DEFAULT_PHASE_CLAMP_Q7_TO_NORM_SCALE: f32 =
    Q7_PHASE_SCALE_RAD / DEFAULT_PHASE_CLAMP_MAX_RAD;

/// Default minimum phase step cutoff in radians ($0.01\text{ rad}$).
pub const DEFAULT_PHASE_STEP_MIN_RAD: f32 = 0.01;

/// Maximum number of attention heads supported in fixed-capacity trail and phase arrays (`64`).
pub const MAX_ATTENTION_HEADS: usize = 64;

/// Number of parallel 8-head SWAR Stigmergic Granules partitioning the 64 attention heads (`8`).
pub const NUM_STIGMERGIC_GRANULES: usize = 8;

/// Number of attention heads per SWAR Stigmergic Granule (`8` heads = 1 `u64` SWAR register).
pub const HEADS_PER_STIGMERGIC_GRANULE: usize = MAX_ATTENTION_HEADS / NUM_STIGMERGIC_GRANULES;

/// Default number of tokens per PagedAttention KV cache block (`32`).
pub const DEFAULT_PAGED_BLOCK_SIZE: u32 = 32;

/// Default slot capacity of the Action Hub pointer table (`256`).
pub const DEFAULT_ACTION_HUB_CAPACITY: u32 = 256;

/// Fixed capacity of topological anchors per Macro Attractor Field accumulator (`64`).
pub const MAX_ANCHORS_PER_FIELD: usize = 64;

/// Maximum number of parallel domain accumulator slots in a `MultiDomainAttractorSpace` (`8`).
pub const MAX_ATTRACTOR_DOMAINS: usize = 8;

/// Polynomial degree of the Cold-Path Chebyshev spectral filter (`3`).
pub const CHEBYSHEV_FILTER_DEGREE: usize = 3;

/// Number of expansion coefficients $[c_0, c_1, c_2, c_3]$ in a 3rd-order Chebyshev filter (`4`).
pub const CHEBYSHEV_COEFF_LEN: usize = 4;

/// Number of 64-bit words in a 256-bit `gcso_paged_bitmask_t` (`4`).
pub const PAGED_BITMASK_WORDS: usize = 4;

/// Total number of discrete bits in a `gcso_paged_bitmask_t` (`256`).
pub const PAGED_BITMASK_BITS: u32 = 256;

/// Number of fast-path shortcut slots in a `gcso_ledger_slot_t` (`4`).
pub const FAST_PATH_SHORTCUT_SLOTS: usize = 4;

/// Byte length of the circular mini telemetry ledger in a `gcso_ledger_slot_t` (`8`).
pub const LEDGER_HISTORY_LEN: usize = 8;

/// Binary magic constant for `.gcso` snapshot headers (`0x4F534347` = ASCII `"GCSO"`).
pub const GCSO_SNAPSHOT_MAGIC_U32: u32 = 0x4F53_4347;

/// Narrow phase clamp authority (`0.14 rad`) for Objective Fact grounding (`Fact` sub-heads).
pub const DEFAULT_FACT_PHASE_CLAMP_RAD: f32 = 0.14;

/// Wide phase clamp authority (`0.24 rad`) for Subjective Persona modulation (`Explore/Logic` sub-heads).
pub const DEFAULT_PERSONA_PHASE_CLAMP_RAD: f32 = 0.24;

/// Default destructive phase coherence threshold (`0.25`) detecting Persona-vs-Fact collision ($\mathcal{C}_{\mathrm{FP}} < -0.25$).
pub const DEFAULT_PERSONA_FACT_COLLISION_THRESH: f32 = 0.25;

/// Default minimum cosine phase coherence threshold (`0.35`) for Online Toroidal DP-Means cluster splitting.
pub const DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE: f32 = 0.35;

/// Default cosine phase coherence threshold (`0.85`) for autonomous online consolidation of redundant macro centroids.
pub const DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE: f32 = 0.85;

/// Default secondary-to-primary gravity ratio threshold (`0.55`) triggering Top-2 Liminal Bifurcation Routing.
pub const DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO: f32 = 0.55;

/// Magic identifier (`0x4C545343` = ASCII `"CSTL"`) for `.gcso` `CrystallizedAttractorRecord` blocks.
pub const GCSO_CRYSTAL_RECORD_MAGIC: u32 = 0x4C54_5343;

/// Version constant for `CrystallizedAttractorRecord` binary serialization ([`GCSO_ABI_VERSION_HEX`]).
pub const GCSO_CRYSTAL_RECORD_VERSION: u32 = GCSO_ABI_VERSION_HEX;

/// Fixed byte length (`192` bytes = `128B` `gcso_snapshot_header_t` + `64B` `gcso_mapped_storage_descriptor_t`) of Section 0.
pub const GCSO_FOUR_SECTION_HEADER_BYTES: usize = 192;

/// Byte offset (`128`) within the 192-byte Section 0 preamble where `gcso_mapped_storage_descriptor_t` is stored.
pub const GCSO_MAPPED_STORAGE_PREAMBLE_OFFSET: usize = 128;

#[allow(dead_code)]
pub(crate) const MAX_REGISTER_CACHED_CENTROIDS: usize = 16;

// ===================================================================
// 3. Semantic Attractor Provenance IDs, Workbench Pool & 8x8 Coupling
// ===================================================================

/// Semantic Provenance Label 0: Primary System Baseline & Core Identity Anchor (`0`).
///
/// Decoupled from physical workbench register lane indices (`0..=7`).
pub const ATTRACTOR_SLOT_SYSTEM_BASELINE: u8 = 0;

/// Semantic Provenance Label 1: Character Persona & Emotional Style Attractor (`1`).
///
/// Decoupled from physical workbench register lane indices (`0..=7`).
pub const ATTRACTOR_SLOT_PERSONA: u8 = 1;

/// Semantic Provenance Label 2: Factual Knowledge & World Lore Anchor (`2`).
///
/// Decoupled from physical workbench register lane indices (`0..=7`).
pub const ATTRACTOR_SLOT_KNOWLEDGE: u8 = 2;

/// Semantic Provenance Label 3: Phase-Conjugate Anti-Hallucination Repulsion Field (`3`, $-\boldsymbol{\Delta\theta}$).
///
/// Decoupled from physical workbench register lane indices (`0..=7`).
pub const ATTRACTOR_SLOT_REPULSION: u8 = 3;

/// Semantic Provenance Label 4: Dynamic User Context & Episodic Dialogue Memory (`4`).
///
/// Decoupled from physical workbench register lane indices (`0..=7`).
pub const ATTRACTOR_SLOT_USER_CONTEXT: u8 = 4;

/// Semantic Provenance Label 5: Temporal / Visual Co-Watching Horizon Attractor (`5`).
///
/// Decoupled from physical workbench register lane indices (`0..=7`).
pub const ATTRACTOR_SLOT_TEMPORAL: u8 = 5;

/// Semantic Provenance Label 6: User Intent / Circadian Drift Attractor (`6`).
///
/// Decoupled from physical workbench register lane indices (`0..=7`).
pub const ATTRACTOR_SLOT_USER_INTENT: u8 = 6;

/// Semantic Provenance Label 7: Crystallized Macro-Basin / Dynamic Extension Provenance (`7`).
///
/// Decoupled from physical workbench register lane indices (`0..=7`).
pub const ATTRACTOR_SLOT_CRYSTALLIZED: u8 = 7;

/// Bitmask isolating a valid 3-bit Semantic Provenance / Attractor Slot ID (`0..=7`).
pub const ATTRACTOR_SLOT_ID_MASK: u8 = 0x07;

/// Number of anonymous physical lanes in the 8-lane working phase workbench (`8`).
pub const WORKBENCH_LANE_COUNT: usize = MAX_ATTRACTOR_DOMAINS;

/// Sentinel lane selector requesting deterministic automatic allocation from the free workbench pool (`0xFF`).
pub const WORKBENCH_LANE_AUTO: u8 = 0xFF;

/// Workbench lane attribute bit: lane is explicitly pinned and exempt from automatic release or eviction (`bit 0`).
pub const WORKBENCH_LANE_FLAG_PINNED: u8 = 1 << 0;

/// Workbench lane attribute bit: lane holds a phase-conjugate anti-hallucination repulsion profile (`bit 1`).
pub const WORKBENCH_LANE_FLAG_REPULSION: u8 = 1 << 1;

/// Workbench lane attribute bit: lane holds an isolated hypothesis acting only on exterior/serendipity granules (`bit 2`).
pub const WORKBENCH_LANE_FLAG_SANDBOX: u8 = 1 << 2;

// ===================================================================
// 4. Liminal Cognitive Threshold Masks & Causal Provenance Flags
// ===================================================================

/// Cognitive threshold bit for Origin 1: hardware & QoS resource constraints (`SystemConstraintSignal`).
pub const THRESHOLD_SYSTEM_CONSTRAINT: u32 = 1 << 0;

/// Cognitive threshold bit for Origin 2A: factual knowledge grounding (`ObjectiveFactSignal`).
pub const THRESHOLD_OBJECTIVE_FACT: u32 = 1 << 1;

/// Cognitive threshold bit for Origin 2B: subjective persona grounding (`SubjectivePersonaSignal`).
pub const THRESHOLD_SUBJECTIVE_PERSONA: u32 = 1 << 2;

/// Cognitive threshold bit for Origin 2C: contradiction guard grounding (`ContradictionGuardSignal`).
pub const THRESHOLD_CONTRADICTION_GUARD: u32 = 1 << 3;

/// Combined causal provenance cognitive threshold across system and grounding origins (`0x0000_000F`).
pub const THRESHOLD_CAUSAL_PROVENANCE: u32 = 0x0000_000F;

/// Cognitive threshold bit for recurrent cognitive circuit plasticity (`CognitiveTrajectoryLedger`).
pub const THRESHOLD_COGNITIVE_CIRCUIT: u32 = 1 << 4;

/// Cognitive threshold bit for phase KV context memory (`RamPhaseKvCacheOps`).
pub const THRESHOLD_KV_PHASE_CACHE: u32 = 1 << 5;

/// Cognitive threshold bit for perspective-relative attractor coupling (`RelativeCouplingMatrix`).
pub const THRESHOLD_RELATIVE_COUPLING: u32 = 1 << 6;

/// Cognitive threshold bit for dissipative overflow crystallization (`CrystallizedAttractorRecord`).
pub const THRESHOLD_DISSIPATIVE_CRYSTAL: u32 = 1 << 7;

/// Wide composite cognitive threshold spanned by `ManifoldGatewaySignal` (`0x0000_00FF`).
pub const THRESHOLD_MANIFOLD_GATEWAY: u32 = 0x0000_00FF;

/// Cognitive threshold bit for $SO(2)$ phase steering (`PhaseSteering`).
pub const THRESHOLD_PHASE_STEERING: u32 = 1 << 8;

/// Cognitive threshold bit for Fact/Logic/Explore sub-head routing (`SubHeadRouter`).
pub const THRESHOLD_SUBHEAD_ROUTING: u32 = 1 << 9;

/// Cognitive threshold bit for Rank-1 hidden residual injection (`ResidualAdapterEngine`).
pub const THRESHOLD_RANK1_RESIDUAL: u32 = 1 << 10;

/// Cognitive threshold bit for tagged pointer transitions & trails (`PointerActionHub`).
pub const THRESHOLD_POINTER_TRAIL: u32 = 1 << 11;

/// Cognitive threshold bit for fast-path L1 ledger shortcuts (`FastPathLedger`).
pub const THRESHOLD_FAST_PATH_LEDGER: u32 = 1 << 12;

/// Cognitive threshold bit for 8-granule stigmergic consensus (`StigmergicReconstitutionOps`).
pub const THRESHOLD_STIGMERGIC_CONSENSUS: u32 = 1 << 13;

/// Cognitive threshold bit for sheaf cohomology & entropy gate control.
pub const THRESHOLD_COHOMOLOGY_ENTROPY: u32 = 1 << 14;

/// Cognitive threshold bit for zero-copy `.gcso` mapped storage I/O.
pub const THRESHOLD_MAPPED_STORAGE: u32 = 1 << 15;

/// Broadest operational cognitive threshold spanned by `ActionHub` (`0x0000_FFFF`).
pub const THRESHOLD_ACTION_HUB_WIDE: u32 = 0x0000_FFFF;

/// Number of dynamic channel slots tracked by `LiminalThresholdField`.
pub const NUM_LIMINAL_THRESHOLD_CHANNELS: usize = 8;

/// Number of orthogonal self-tuning autonomous core layers (`Layer 0: Liminal`, `Layer 1: Stigmergic`, `Layer 2: Ego`).
pub const NUM_CORE_AUTONOMOUS_LAYERS: usize = 3;

/// Autonomous Core Layer 0 identifier: Epistemic Horizon & Trait Threshold (`LiminalThresholdField`).
pub const CORE_LAYER_LIMINAL: usize = 0;

/// Autonomous Core Layer 1 identifier: Stigmergic Granule Approximation, Overlap & Serendipity (`StigmergicGranuleField`).
pub const CORE_LAYER_STIGMERGIC: usize = 1;

/// Autonomous Core Layer 2 identifier: Wide-Threshold Subjective Ego & Top-Down Boundary Intervention (`LiminalEgoField`).
pub const CORE_LAYER_EGO: usize = 2;

/// Ego intervention trigger bit (`0x01`): Counter-basin orbital thrust when attractor pull threatens capture.
pub const EGO_TRIGGER_BASIN_CAPTURE: u8 = 1 << 0;
/// Ego intervention trigger bit (`0x02`): Defiant intervention under internal contradiction (`H^1 != 0`).
pub const EGO_TRIGGER_INTERNAL_CONFLICT: u8 = 1 << 1;
/// Ego intervention trigger bit (`0x04`): Transgressive step when standing at a `Liminal` horizon boundary.
pub const EGO_TRIGGER_LIMINAL_EXTERIOR: u8 = 1 << 2;
/// Ego intervention trigger bit (`0x08`): Boundary damping when temporal boundary velocity oscillates.
pub const EGO_TRIGGER_BOUNDARY_CHATTER: u8 = 1 << 3;
/// Ego intervention trigger bit (`0x10`): Self-initiated perturbation when boundaries stagnate completely.
pub const EGO_TRIGGER_BOUNDARY_STAGNATION: u8 = 1 << 4;
/// Ego intervention trigger bit (`0x20`): Corrective awakening when layer distortion or saturation is detected.
pub const EGO_TRIGGER_LAYER_DISTORTION: u8 = 1 << 5;
/// Default active bitmask enabling all 5 deterministic `LiminalEgoField` triggers (`0x1F`).
pub const EGO_TRIGGER_ALL_DEFAULT: u8 = 0x1F;

/// Provenance axis bit (`bit 0`): Origin 1 (`SystemConstraintSignal`) contributed to this frame.
pub const CAUSAL_AXIS_SYSTEM_CONSTRAINT: u32 = 1 << 0;
/// Provenance axis bit (`bit 1`): Origin 2-A (`ObjectiveFactSignal`) contributed to this frame.
pub const CAUSAL_AXIS_OBJECTIVE_FACT: u32 = 1 << 1;
/// Provenance axis bit (`bit 2`): Origin 2-B (`SubjectivePersonaSignal`) contributed to this frame.
pub const CAUSAL_AXIS_SUBJECTIVE_PERSONA: u32 = 1 << 2;
/// Provenance axis bit (`bit 3`): Origin 2-C (`ContradictionGuardSignal`) contributed to this frame.
pub const CAUSAL_AXIS_CONTRADICTION_GUARD: u32 = 1 << 3;

/// Provenance cause bit (`bit 8`): System QoS or low-entropy inertial throttle active.
pub const CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE: u32 = 1 << 8;
/// Provenance cause bit (`bit 9`): Objective fact retrieval energy was weak or absent.
pub const CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK: u32 = 1 << 9;
/// Provenance cause bit (`bit 10`): Subjective persona collided with objective fact ($\mathcal{C}_{\mathrm{FP}} < -\tau$).
pub const CAUSAL_CAUSE_PERSONA_FACT_COLLISION: u32 = 1 << 10;
/// Provenance cause bit (`bit 11`): Cohomological obstruction ($H^1 \neq 0$) or Bloom repulsion triggered.
pub const CAUSAL_CAUSE_COHOMOLOGY_REPULSION: u32 = 1 << 11;
/// Provenance cause bit (`bit 12`): Attractor sample count & coherence reached `.gcso` crystallization maturity.
pub const CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED: u32 = 1 << 12;

/// Root-cause diagnosis code `0`: Nominal synergistic orchestration across active origins.
pub const ROOT_CAUSE_NOMINAL: u8 = 0;
/// Root-cause diagnosis code `1`: System QoS / VRAM pressure / inertial layer-skip throttle.
pub const ROOT_CAUSE_SYSTEM_QOS_THROTTLE: u8 = 1;
/// Root-cause diagnosis code `2`: Objective fact retrieval miss or weak grounding energy.
pub const ROOT_CAUSE_FACT_RETRIEVAL_MISS: u8 = 2;
/// Root-cause diagnosis code `3`: Destructive phase collision between Persona and Objective Fact.
pub const ROOT_CAUSE_PERSONA_FACT_COLLISION: u8 = 3;
/// Root-cause diagnosis code `4`: Cohomological obstruction ($H^1 \neq 0$) or Bloom repetition escape.
pub const ROOT_CAUSE_COHOMOLOGY_REPULSION: u8 = 4;

// ===================================================================
// 5. 2 Primitives x 2 Drivers Lightweight Attribution Tags
// ===================================================================

/// Geometric substrate primitive classifier (`SubstrateKind`).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SubstrateKind {
    /// Boundary, cognitive threshold, and 4-region epistemic partition domain (`0`).
    #[default]
    Liminal = 0,
    /// Local 8-granule quorum, noise deadband, and volatile trace field (`1`).
    Stigmergy = 1,
}

impl SubstrateKind {
    /// Decodes a [`SubstrateKind`] branchlessly from bit 0 of a raw tag byte.
    #[inline(always)]
    #[must_use]
    pub const fn from_bit(bit: u8) -> Self {
        if (bit & 1) == 0 {
            Self::Liminal
        } else {
            Self::Stigmergy
        }
    }
}

/// Orthogonal execution driver classifier (`DriverKind`).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DriverKind {
    /// Deterministic mechanical optimization spine / I-Frame baseline (`0`).
    #[default]
    System = 0,
    /// Anisotropic intentional differential noise / P-Frame network (`1`).
    Ego = 1,
}

impl DriverKind {
    /// Decodes a [`DriverKind`] branchlessly from bit 0 of a shifted driver field.
    #[inline(always)]
    #[must_use]
    pub const fn from_bit(bit: u8) -> Self {
        if (bit & 1) == 0 {
            Self::System
        } else {
            Self::Ego
        }
    }
}

/// Packed 2-bit structural attribution tag (`[Driver(1b) | Substrate(1b)]`) inside a 1-byte
/// `Pod` wrapper for zero-cost telemetry and bypass provenance without marker-trait bloat.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AttributionTag(pub u8);

impl AttributionTag {
    /// Packs `(substrate, driver)` into a 2-bit attribution tag byte.
    #[inline(always)]
    #[must_use]
    pub const fn new(substrate: SubstrateKind, driver: DriverKind) -> Self {
        Self((substrate as u8) | ((driver as u8) << 1))
    }

    /// Attribution tag for (`Liminal`, `System`).
    pub const LIMINAL_SYSTEM: Self = Self::new(SubstrateKind::Liminal, DriverKind::System);
    /// Attribution tag for (`Liminal`, `Ego`).
    pub const LIMINAL_EGO: Self = Self::new(SubstrateKind::Liminal, DriverKind::Ego);
    /// Attribution tag for (`Stigmergy`, `System`).
    pub const STIGMERGY_SYSTEM: Self = Self::new(SubstrateKind::Stigmergy, DriverKind::System);
    /// Attribution tag for (`Stigmergy`, `Ego`).
    pub const STIGMERGY_EGO: Self = Self::new(SubstrateKind::Stigmergy, DriverKind::Ego);

    /// Returns the raw byte representation (`bits 0..=1`).
    #[inline(always)]
    #[must_use]
    pub const fn raw(self) -> u8 {
        self.0 & 0x03
    }

    /// Extracts the [`SubstrateKind`] primitive axis (`bit 0`).
    #[inline(always)]
    #[must_use]
    pub const fn substrate(self) -> SubstrateKind {
        SubstrateKind::from_bit(self.0)
    }

    /// Extracts the [`DriverKind`] orthogonal driver axis (`bit 1`).
    #[inline(always)]
    #[must_use]
    pub const fn driver(self) -> DriverKind {
        DriverKind::from_bit(self.0 >> 1)
    }

    /// Returns `true` when attributed to the deterministic [`DriverKind::System`] spine.
    #[inline(always)]
    #[must_use]
    pub const fn is_system(self) -> bool {
        (self.0 & (1 << 1)) == 0
    }

    /// Returns `true` when attributed to the differential [`DriverKind::Ego`] operator.
    #[inline(always)]
    #[must_use]
    pub const fn is_ego(self) -> bool {
        (self.0 & (1 << 1)) != 0
    }
}

/// Branchless constructor helper mirroring `AttributionTag::new`.
#[inline(always)]
#[must_use]
pub const fn pack_attribution_tag(substrate: SubstrateKind, driver: DriverKind) -> u8 {
    AttributionTag::new(substrate, driver).raw()
}

// ===================================================================
// 6. 8x8 Cross-Slot Coupling Matrix & Tagged Pointer Action Codecs
// ===================================================================

/// Default $8 \times 8$ Attractor Slot coupling bitmatrix (`u64`) enabling symmetric cross-slot
/// synergy across all 8 parallel attractor domain slots.
pub const DEFAULT_SLOT_COUPLING_MATRIX_8X8: u64 = 0xFFFF_FFFF_FFFF_FFFF;

/// Computes the bit position (`0..=63`) in an $8 \times 8$ `u64` coupling bitmatrix for `(slot_a, slot_b)`.
#[inline(always)]
#[must_use]
pub const fn slot_coupling_bit_index(slot_a: u32, slot_b: u32) -> u32 {
    ((slot_a & 7) << 3) | (slot_b & 7)
}

/// Branchlessly tests whether `slot_a` and `slot_b` are coupled in `coupling_matrix_8x8`.
#[inline(always)]
#[must_use]
pub const fn is_slot_pair_coupled(coupling_matrix_8x8: u64, slot_a: u32, slot_b: u32) -> bool {
    ((coupling_matrix_8x8 >> slot_coupling_bit_index(slot_a, slot_b)) & 1) != 0
}

/// Extracts the 8-bit row mask (`bits 0..=7`) representing all target slots coupled with `slot_a`
/// in $\mathcal{O}(1)$ via a single register shift.
#[inline(always)]
#[must_use]
pub const fn coupled_slot_row_mask(coupling_matrix_8x8: u64, slot_a: u32) -> u8 {
    ((coupling_matrix_8x8 >> ((slot_a & 7) << 3)) & 0xFF) as u8
}

/// Returns the number of target slots (`0..=8`) coupled with `slot_a` in `coupling_matrix_8x8`.
#[inline(always)]
#[must_use]
pub const fn coupled_slot_count(coupling_matrix_8x8: u64, slot_a: u32) -> u32 {
    coupled_slot_row_mask(coupling_matrix_8x8, slot_a).count_ones()
}

/// Branchlessly sets or clears the directed coupling bit `(slot_a, slot_b)` in `coupling_matrix_8x8`.
#[inline(always)]
#[must_use]
pub const fn with_slot_pair_coupled(
    coupling_matrix_8x8: u64,
    slot_a: u32,
    slot_b: u32,
    enabled: bool,
) -> u64 {
    let bit_mask = 1u64 << slot_coupling_bit_index(slot_a, slot_b);
    let enable_mask = (enabled as u64).wrapping_neg();
    (coupling_matrix_8x8 & !bit_mask) | (bit_mask & enable_mask)
}

/// Branchlessly sets or clears symmetric bidirectional coupling between `slot_a` and `slot_b`
/// (`(slot_a, slot_b)` and `(slot_b, slot_a)`) in `coupling_matrix_8x8`.
#[inline(always)]
#[must_use]
pub const fn with_slot_pair_coupled_symmetric(
    coupling_matrix_8x8: u64,
    slot_a: u32,
    slot_b: u32,
    enabled: bool,
) -> u64 {
    let m1 = with_slot_pair_coupled(coupling_matrix_8x8, slot_a, slot_b, enabled);
    with_slot_pair_coupled(m1, slot_b, slot_a, enabled)
}

/// Action Mode Bit 0 (`0x01`): Enable inline phase rotation on Query/Key registers.
pub const ACTION_MODE_PHASE_STEER: u8 = 1 << 0;

/// Action Mode Bit 1 (`0x02`): Enable intermediate layer skipping with SWAR phase compensation.
pub const ACTION_MODE_LAYER_SKIP: u8 = 1 << 1;

/// Action Mode Bit 2 (`0x04`): Enable Contrastive Phase Bloom Filter anti-phase repulsion ($-\boldsymbol{\Delta\theta}$).
pub const ACTION_MODE_PHASE_REPULSE: u8 = 1 << 2;

/// Action Mode Bit 3 (`0x08`): Enable single-pass sub-head group routing (Fact / Logic / Explore).
pub const ACTION_MODE_SUBHEAD_ROUTE: u8 = 1 << 3;

/// Action Mode Bit 4 (`0x10`): Enable Dynamic Rank-1 outer product residual injection ($\mathbf{u}\mathbf{v}^T$).
pub const ACTION_MODE_RANK1_INJECT: u8 = 1 << 4;

/// Composite Action Mode preset for passive sidecar observation (`0x00`).
pub const ACTION_PRESET_PASSIVE_SIDECAR: u8 = 0x00;

/// Composite Action Mode preset for lightweight express phase-steered sidecar execution (`0x01`).
pub const ACTION_PRESET_EXPRESS_PHASE: u8 = ACTION_MODE_PHASE_STEER;

/// Composite Action Mode preset for low-entropy inertial layer-skipping fast-forward (`0x03`).
pub const ACTION_PRESET_INERTIAL_SKIP: u8 = ACTION_MODE_PHASE_STEER | ACTION_MODE_LAYER_SKIP;

/// Composite Action Mode preset for anti-hallucination repulsion and sub-head exploration routing (`0x0D`).
pub const ACTION_PRESET_REPULSION_ESCAPE: u8 =
    ACTION_MODE_PHASE_STEER | ACTION_MODE_PHASE_REPULSE | ACTION_MODE_SUBHEAD_ROUTE;

/// Composite Action Mode preset for full-stack deep persona & knowledge orchestration (`0x19`).
pub const ACTION_PRESET_FULL_ORCHESTRATION: u8 =
    ACTION_MODE_PHASE_STEER | ACTION_MODE_SUBHEAD_ROUTE | ACTION_MODE_RANK1_INJECT;

/// Bitmask extracting the 5-bit Action Mode flags from an 8-bit `gcso_tagged_ptr_t` tag (`bits 0..=4`).
pub const TAG_ACTION_MODE_MASK: u8 = 0x1F;

/// Bit shift offset for encoding/extracting the 3-bit Target Attractor Slot ID (`0..=7`) in an 8-bit tag (`bits 5..=7`).
pub const TAG_ATTRACTOR_SLOT_SHIFT: u32 = 5;

/// Packs a 3-bit Target Attractor Slot ID (`0..=7`) and a 5-bit Action Mode mask (`0x00..=0x1F`)
/// into a single 8-bit tag byte for `gcso_tagged_ptr_t`.
#[inline(always)]
#[must_use]
pub const fn pack_action_slot_tag(attractor_slot_id: u8, action_mode_flags: u8) -> u8 {
    ((attractor_slot_id & ATTRACTOR_SLOT_ID_MASK) << TAG_ATTRACTOR_SLOT_SHIFT)
        | (action_mode_flags & TAG_ACTION_MODE_MASK)
}

/// Extracts the 5-bit Action Mode bitmask (`ACTION_MODE_*`) from an 8-bit `gcso_tagged_ptr_t` tag byte.
#[inline(always)]
#[must_use]
pub const fn unpack_action_mode_from_tag(tag: u8) -> u8 {
    tag & TAG_ACTION_MODE_MASK
}

/// Extracts the 3-bit Target Attractor Slot ID (`0..=7`) from an 8-bit `gcso_tagged_ptr_t` tag byte.
#[inline(always)]
#[must_use]
pub const fn unpack_attractor_slot_from_tag(tag: u8) -> u8 {
    (tag >> TAG_ATTRACTOR_SLOT_SHIFT) & ATTRACTOR_SLOT_ID_MASK
}
