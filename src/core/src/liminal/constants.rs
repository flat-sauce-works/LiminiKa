// name: src/core/src/liminal/constants.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Layer 0 fundamental constants, C-ABI configuration/status primitives, cognitive threshold
//! masks, and 2-bit structural attribution tags.

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
///
/// Intentionally retained under the `0.1.3` C-ABI firewall; scheduled for plain-English
/// renaming (`lmnk_*`) at `v0.2.0`.
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
        "Frozen `0.1.3` C-ABI compatibility status code: substrate pointer table capacity saturated."
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
///
/// Intentionally retained under the `0.1.3` C-ABI firewall; scheduled for plain-English
/// renaming (`lmnk_*`) at `v0.2.0`.
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

/// Maximum number of attention heads supported in fixed-capacity trail and phase arrays (`64`).
pub const MAX_ATTENTION_HEADS: usize = 64;

/// Number of parallel 8-head SWAR Stigmergic Granules partitioning the 64 attention heads (`8`).
pub const NUM_STIGMERGIC_GRANULES: usize = 8;

/// Number of attention heads per SWAR Stigmergic Granule (`8` heads = 1 `u64` SWAR register).
pub const HEADS_PER_STIGMERGIC_GRANULE: usize = MAX_ATTENTION_HEADS / NUM_STIGMERGIC_GRANULES;

/// Maximum number of parallel domain accumulator slots in a `MultiDomainAttractorSpace` (`8`).
pub const MAX_ATTRACTOR_DOMAINS: usize = 8;

// ===================================================================
// 3. Liminal Cognitive Threshold Masks
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

/// Cognitive threshold bit for tagged pointer transitions & trails (`PointerSubstrate`).
pub const THRESHOLD_POINTER_TRAIL: u32 = 1 << 11;

/// Cognitive threshold bit for fast-path L1 ledger shortcuts (`FastPathLedger`).
pub const THRESHOLD_FAST_PATH_LEDGER: u32 = 1 << 12;

/// Cognitive threshold bit for 8-granule stigmergic consensus (`StigmergicReconstitutionOps`).
pub const THRESHOLD_STIGMERGIC_CONSENSUS: u32 = 1 << 13;

/// Cognitive threshold bit for sheaf cohomology & entropy gate control.
pub const THRESHOLD_COHOMOLOGY_ENTROPY: u32 = 1 << 14;

/// Cognitive threshold bit for zero-copy `.gcso` mapped storage I/O.
pub const THRESHOLD_MAPPED_STORAGE: u32 = 1 << 15;

/// Broadest operational cognitive threshold spanned by `StigmergicSubstrateOps` (`0x0000_FFFF`).
pub const THRESHOLD_STIGMERGY_WIDE: u32 = 0x0000_FFFF;

/// Number of dynamic channel slots tracked by `LiminalThresholdState`.
pub const NUM_LIMINAL_THRESHOLD_CHANNELS: usize = 8;

// ===================================================================
// 4. 2 Primitives x 2 Drivers Lightweight Attribution Tags
// ===================================================================

/// Geometric primitive classifier (`PrimitiveKind`).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrimitiveKind {
    /// Boundary, cognitive threshold, and 4-region epistemic partition domain (`0`).
    #[default]
    Liminal = 0,
    /// Local 8-granule quorum, noise deadband, and volatile trace carrier (`1`).
    Stigmergy = 1,
}

impl PrimitiveKind {
    /// Decodes a [`PrimitiveKind`] branchlessly from bit 0 of a raw tag byte.
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

/// Packed 2-bit structural attribution tag (`[Driver(1b) | Primitive(1b)]`) inside a 1-byte
/// `Pod` wrapper for zero-cost telemetry and bypass provenance without marker-trait bloat.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AttributionTag(pub u8);

impl AttributionTag {
    /// Packs `(primitive, driver)` into a 2-bit attribution tag byte.
    #[inline(always)]
    #[must_use]
    pub const fn new(primitive: PrimitiveKind, driver: DriverKind) -> Self {
        Self((primitive as u8) | ((driver as u8) << 1))
    }

    /// Attribution tag for (`Liminal`, `System`).
    pub const LIMINAL_SYSTEM: Self = Self::new(PrimitiveKind::Liminal, DriverKind::System);
    /// Attribution tag for (`Liminal`, `Ego`).
    pub const LIMINAL_EGO: Self = Self::new(PrimitiveKind::Liminal, DriverKind::Ego);
    /// Attribution tag for (`Stigmergy`, `System`).
    pub const STIGMERGY_SYSTEM: Self = Self::new(PrimitiveKind::Stigmergy, DriverKind::System);
    /// Attribution tag for (`Stigmergy`, `Ego`).
    pub const STIGMERGY_EGO: Self = Self::new(PrimitiveKind::Stigmergy, DriverKind::Ego);

    /// Returns the raw byte representation (`bits 0..=1`).
    #[inline(always)]
    #[must_use]
    pub const fn raw(self) -> u8 {
        self.0 & 0x03
    }

    /// Extracts the [`PrimitiveKind`] primitive axis (`bit 0`).
    #[inline(always)]
    #[must_use]
    pub const fn primitive(self) -> PrimitiveKind {
        PrimitiveKind::from_bit(self.0)
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
pub const fn pack_attribution_tag(primitive: PrimitiveKind, driver: DriverKind) -> u8 {
    AttributionTag::new(primitive, driver).raw()
}
