// name: src/core/src/liminal/constants.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Layer 0 fundamental constants, C-ABI configuration/status primitives, cognitive threshold
//! masks, and 3-bit structural attribution tags (`2 Primitives x 3 Drivers`).

#![forbid(unsafe_code)]

use core::f32::consts::{LN_2, PI};
use core::fmt;
use core::mem::{align_of, offset_of, size_of};

// ===================================================================
// 1. Unified C-ABI Version, Capability & Status Primitives
// ===================================================================

/// Unified LiminiKa C-ABI Major Version Component.
pub const LMNK_ABI_VERSION_MAJOR: u32 = 0;
/// Unified LiminiKa C-ABI Minor Version Component.
pub const LMNK_ABI_VERSION_MINOR: u32 = 4;
/// Unified LiminiKa C-ABI Patch Version Component.
pub const LMNK_ABI_VERSION_PATCH: u32 = 0;
/// Hexadecimal Representation of the active Unified ABI Version (`0x0000_0400`).
pub const LMNK_ABI_VERSION_HEX: u32 =
    (LMNK_ABI_VERSION_MAJOR << 16) | (LMNK_ABI_VERSION_MINOR << 8) | LMNK_ABI_VERSION_PATCH;
/// Bitmask extracting `(MAJOR << 16) | (MINOR << 8)` for Pre-1.0 ABI compatibility checks.
pub const LMNK_ABI_COMPAT_MASK: u32 = 0xFFFF_FF00;
/// Packed hexadecimal `MAJOR.MINOR` compatibility prefix (`LMNK_ABI_VERSION_HEX & LMNK_ABI_COMPAT_MASK`).
pub const LMNK_ABI_COMPAT_VERSION_HEX: u32 = LMNK_ABI_VERSION_HEX & LMNK_ABI_COMPAT_MASK;
/// String Representation of the active Unified ABI Version (derived from `Cargo.toml`).
pub const LMNK_ABI_VERSION_STRING: &str = env!("CARGO_PKG_VERSION");
/// NUL-terminated C string representation of the active Unified ABI Version.
pub const LMNK_ABI_VERSION_CSTR: &core::ffi::CStr = c"0.4.0";

const _: () = {
    let c_bytes = LMNK_ABI_VERSION_CSTR.to_bytes();
    let pkg_bytes = LMNK_ABI_VERSION_STRING.as_bytes();
    assert!(c_bytes.len() == pkg_bytes.len());
    let mut i = 0;
    while i < c_bytes.len() {
        assert!(c_bytes[i] == pkg_bytes[i]);
        i += 1;
    }
};

/// Capability bitmask flag indicating portable Little-Endian SWAR baseline execution support.
pub const LMNK_CAP_SWAR_BASELINE: u64 = 1 << 0;
/// Capability bitmask flag indicating x86_64 AVX2 vector acceleration support.
pub const LMNK_CAP_SIMD_AVX2: u64 = 1 << 1;
/// Capability bitmask flag indicating AArch64 NEON vector acceleration support.
pub const LMNK_CAP_SIMD_NEON: u64 = 1 << 2;
/// Capability bitmask flag indicating WebAssembly 128-bit SIMD acceleration support.
pub const LMNK_CAP_SIMD_WASM128: u64 = 1 << 3;
/// Capability bitmask flag indicating strict zero-allocation `#![no_std]` core execution.
pub const LMNK_CAP_ZERO_ALLOC_CORE: u64 = 1 << 4;
/// Capability bitmask flag indicating zero-heap caller-buffered `in_place` context lifecycle.
pub const LMNK_CAP_INPLACE_CONTEXT_INIT: u64 = 1 << 5;

/// Returns `true` when `file_version` is ABI-compatible with `runtime_version`:
/// non-zero, identical `MAJOR.MINOR` prefix (`& LMNK_ABI_COMPAT_MASK`), and
/// `file_patch <= runtime_patch` (Pre-1.0 strict `MAJOR.MINOR` + `PATCH` backward compatibility).
#[inline(always)]
#[must_use]
pub const fn is_compatible_abi_version(file_version: u32, runtime_version: u32) -> bool {
    (file_version != 0)
        & ((file_version & LMNK_ABI_COMPAT_MASK) == (runtime_version & LMNK_ABI_COMPAT_MASK))
        & ((file_version & 0xFF) <= (runtime_version & 0xFF))
}

/// Validates a C-ABI snapshot or container version against [`LMNK_ABI_VERSION_HEX`] using
/// [`is_compatible_abi_version`] (`MAJOR.MINOR` strict match and `file_patch <= runtime_patch`).
///
/// # Errors
/// Returns [`LMNK_ERR_VERSION_MISMATCH`] if `version_hex` is incompatible with [`LMNK_ABI_VERSION_HEX`].
#[inline(always)]
pub const fn validate_abi_version(version_hex: u32) -> Result<(), LmnkStatus> {
    if is_compatible_abi_version(version_hex, LMNK_ABI_VERSION_HEX) {
        Ok(())
    } else {
        Err(LMNK_ERR_VERSION_MISMATCH)
    }
}

/// Standardized 32-bit signed status code returned across the C-ABI FFI boundary.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
pub struct LmnkStatus(
    /// Inner signed 32-bit integer status code.
    pub i32,
);

macro_rules! define_status_codes {
    ($(($name:ident, $code:expr, $c_bytes:literal, $str_val:literal, $doc:expr)),* $(,)?) => {
        $(
            #[doc = $doc]
            pub const $name: LmnkStatus = LmnkStatus($code);
        )*

        impl LmnkStatus {
            /// Returns `true` if the status equals [`LMNK_OK`] (`0`).
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

            /// Alias for [`LMNK_ERR_TOPOLOGY_MISMATCH`].
            pub const TOPOLOGY_MISMATCH: Self = LMNK_ERR_TOPOLOGY_MISMATCH;

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
                    _ => b"LMNK_ERR_UNKNOWN\0",
                }
            }

            /// Returns the static ASCII string representation of this status code (excluding trailing NUL).
            #[inline(always)]
            #[must_use]
            pub const fn as_str(self) -> &'static str {
                match self {
                    $($name => $str_val,)*
                    _ => "LMNK_ERR_UNKNOWN",
                }
            }
        }
    };
}

define_status_codes!(
    (
        LMNK_OK,
        0,
        b"LMNK_OK\0",
        "LMNK_OK",
        "Operation completed successfully."
    ),
    (
        LMNK_ERR_INVALID_ARGUMENT,
        -1,
        b"LMNK_ERR_INVALID_ARGUMENT\0",
        "LMNK_ERR_INVALID_ARGUMENT",
        "Invalid argument or out-of-bounds parameter passed to C-ABI function."
    ),
    (
        LMNK_ERR_OUT_OF_MEMORY,
        -2,
        b"LMNK_ERR_OUT_OF_MEMORY\0",
        "LMNK_ERR_OUT_OF_MEMORY",
        "Memory allocation failure or capacity limit reached during Cold Path creation."
    ),
    (
        LMNK_ERR_BUFFER_TOO_SMALL,
        -3,
        b"LMNK_ERR_BUFFER_TOO_SMALL\0",
        "LMNK_ERR_BUFFER_TOO_SMALL",
        "Destination buffer capacity is insufficient for requested payload."
    ),
    (
        LMNK_ERR_PANIC_CAUGHT,
        -4,
        b"LMNK_ERR_PANIC_CAUGHT\0",
        "LMNK_ERR_PANIC_CAUGHT",
        "Unhandled internal Rust panic caught at FFI boundary."
    ),
    (
        LMNK_ERR_NULL_POINTER,
        -5,
        b"LMNK_ERR_NULL_POINTER\0",
        "LMNK_ERR_NULL_POINTER",
        "Unexpected `NULL` pointer supplied for a mandatory parameter."
    ),
    (
        LMNK_ERR_INVALID_STATE,
        -6,
        b"LMNK_ERR_INVALID_STATE\0",
        "LMNK_ERR_INVALID_STATE",
        "Context or component is in an invalid lifecycle state for requested operation."
    ),
    (
        LMNK_ERR_VERSION_MISMATCH,
        -7,
        b"LMNK_ERR_VERSION_MISMATCH\0",
        "LMNK_ERR_VERSION_MISMATCH",
        "Binary snapshot or descriptor ABI version mismatch detected."
    ),
    (
        LMNK_ERR_ALIGNMENT,
        -8,
        b"LMNK_ERR_ALIGNMENT\0",
        "LMNK_ERR_ALIGNMENT",
        "Pointer passed across FFI boundary violates required SIMD or cacheline alignment."
    ),
    (
        LMNK_ERR_IO_FAILURE,
        -9,
        b"LMNK_ERR_IO_FAILURE\0",
        "LMNK_ERR_IO_FAILURE",
        "Input/output or storage persistence operation failure."
    ),
    (
        LMNK_ERR_SUBSTRATE_FULL,
        -10,
        b"LMNK_ERR_SUBSTRATE_FULL\0",
        "LMNK_ERR_SUBSTRATE_FULL",
        "Substrate working-lane table or slot capacity saturated."
    ),
    (
        LMNK_ERR_NOT_IMPLEMENTED,
        -11,
        b"LMNK_ERR_NOT_IMPLEMENTED\0",
        "LMNK_ERR_NOT_IMPLEMENTED",
        "Feature or capability not implemented in current build."
    ),
    (
        LMNK_ERR_ANCHOR_NOT_FOUND,
        -20,
        b"LMNK_ERR_ANCHOR_NOT_FOUND\0",
        "LMNK_ERR_ANCHOR_NOT_FOUND",
        "Target phase anchor ID not found."
    ),
    (
        LMNK_ERR_PHASE_OVERFLOW,
        -30,
        b"LMNK_ERR_PHASE_OVERFLOW\0",
        "LMNK_ERR_PHASE_OVERFLOW",
        "Phase steering angle accumulation exceeded safety bound."
    ),
    (
        LMNK_ERR_PHASE_STEP_UNDERFLOW,
        -31,
        b"LMNK_ERR_PHASE_STEP_UNDERFLOW\0",
        "LMNK_ERR_PHASE_STEP_UNDERFLOW",
        "Phase step filter fell below minimal threshold $\\Delta\\theta_{\\mathrm{min\\_step}}$."
    ),
    (
        LMNK_ERR_ENTROPY_SINGULARITY,
        -40,
        b"LMNK_ERR_ENTROPY_SINGULARITY\0",
        "LMNK_ERR_ENTROPY_SINGULARITY",
        "Entropy controller encountered numerical or logarithmic singularity."
    ),
    (
        LMNK_ERR_CONTAINER_CORRUPTED,
        -50,
        b"LMNK_ERR_CONTAINER_CORRUPTED\0",
        "LMNK_ERR_CONTAINER_CORRUPTED",
        "Binary snapshot or container header or payload checksum corrupted."
    ),
    (
        LMNK_ERR_STORAGE_MAP_FAILED,
        -51,
        b"LMNK_ERR_STORAGE_MAP_FAILED\0",
        "LMNK_ERR_STORAGE_MAP_FAILED",
        "Zero-copy memory mapping or storage buffer mapping failed."
    ),
    (
        LMNK_ERR_TOPOLOGY_MISMATCH,
        -52,
        b"LMNK_ERR_TOPOLOGY_MISMATCH\0",
        "LMNK_ERR_TOPOLOGY_MISMATCH",
        "Container topology CRC32 does not match the active core topology matrix."
    ),
    (
        LMNK_ERR_SUBHEAD_ROUTING_FAILED,
        -60,
        b"LMNK_ERR_SUBHEAD_ROUTING_FAILED\0",
        "LMNK_ERR_SUBHEAD_ROUTING_FAILED",
        "Sub-head group phase routing evaluation failed."
    ),
    (
        LMNK_ERR_KV_KEYFRAME_SEEK_FAILED,
        -70,
        b"LMNK_ERR_KV_KEYFRAME_SEEK_FAILED\0",
        "LMNK_ERR_KV_KEYFRAME_SEEK_FAILED",
        "KV-keyframe index lookup or token seek operation failed."
    ),
    (
        LMNK_ERR_OBSTRUCTION_UNRESOLVED,
        -80,
        b"LMNK_ERR_OBSTRUCTION_UNRESOLVED\0",
        "LMNK_ERR_OBSTRUCTION_UNRESOLVED",
        "Unresolved cohomological obstruction class $H^1(K; \\mathcal{F})$ encountered."
    ),
    (
        LMNK_ERR_EXTENSION_NOT_LOADED,
        -90,
        b"LMNK_ERR_EXTENSION_NOT_LOADED\0",
        "LMNK_ERR_EXTENSION_NOT_LOADED",
        "Dynamic extension module not loaded or inactive in ledger slot."
    ),
    (
        LMNK_ERR_LEDGER_FULL,
        -91,
        b"LMNK_ERR_LEDGER_FULL\0",
        "LMNK_ERR_LEDGER_FULL",
        "Fast-path ledger table or ring buffer fully occupied."
    ),
    (
        LMNK_ERR_UNKNOWN,
        -0x7FFF_FFFF,
        b"LMNK_ERR_UNKNOWN\0",
        "LMNK_ERR_UNKNOWN",
        "Unknown internal system error."
    ),
);

impl fmt::Display for LmnkStatus {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<LmnkStatus> for Result<(), LmnkStatus> {
    #[inline(always)]
    fn from(status: LmnkStatus) -> Self {
        if status == LMNK_OK {
            Ok(())
        } else {
            Err(status)
        }
    }
}

impl From<Result<(), LmnkStatus>> for LmnkStatus {
    #[inline(always)]
    fn from(res: Result<(), LmnkStatus>) -> Self {
        match res {
            Ok(()) => LMNK_OK,
            Err(status) => status,
        }
    }
}

/// Core result response mapping to [`LmnkStatus`].
pub type LmnkResult<T> = Result<T, LmnkStatus>;

/// Quantized 7-bit signed fixed-point integer (phase scale $\beta_{\mathrm{Q7}} = \pi / 128$).
pub type PhaseQ7 = i8;

/// Portable environment-agnostic runtime configuration descriptor (64 bytes, `#[repr(C, align(64))]`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RuntimeConfig {
    /// Envelope byte size (`sizeof(lmnk_config_t)` == `64`, `offset 0..4`).
    pub struct_size: u32,
    /// Dimension per attention head ($d_{\mathrm{head}}$, must be positive and even, `offset 4..8`).
    pub head_dim: u32,
    /// Total number of attention heads ($1 \le H \le 64$, `offset 8..12`).
    pub num_heads: u32,
    /// Tokens per paged KV cache block (`16` or `32`, `offset 12..16`).
    pub paged_block_size: u32,
    /// Floating-point scale factor for Q7 phase values ($\beta_{\mathrm{Q7}} = \pi / 128$, `offset 16..20`).
    pub q7_phase_scale: f32,
    /// Maximum allowed rotation angle for soft phase clamping in radians ($\theta_{\max}$, `offset 20..24`).
    pub phase_clamp_max_rad: f32,
    /// Minimum rotation angle threshold for phase step filtering in radians
    /// ($\Delta\theta_{\mathrm{min\_step}}$, `offset 24..28`).
    pub phase_step_min_rad: f32,
    /// Epsilon clamp to prevent logarithmic singularity in entropy calculation ($\epsilon_{\mathrm{log}}$,
    /// `offset 28..32`).
    pub entropy_singularity_eps: f32,
    /// Maximum supported prompt anchor points (`offset 32..36`).
    pub max_prompt_anchors: u32,
    /// Maximum logical trail capacity of the stigmergic substrate (`offset 36..40`).
    pub substrate_capacity: u32,
    /// Active rotated dimensions per head (`0` defaults to full `head_dim`; smaller for partial RoPE, `offset 40..44`).
    pub rotary_dim: u32,
    /// Vendor-neutral execution policy bitflags (`offset 44`).
    pub execution_flags: u8,
    /// Strict enforcement of zero dynamic allocations on Hot Path (`1` = strict, `offset 45`).
    pub enable_zero_alloc_strict: u8,
    /// Initial operational mode for `FastPathLedger` (`0`, `1`, or `2`, `offset 46`).
    pub ledger_mode: u8,
    /// RoPE channel pairing layout (`0` = Interleaved, `1` = HalfSplit, `offset 47`).
    pub rope_layout: u8,
    /// Opening decode token hold window for transient contradiction-guard pins (`0` = full turn, `offset 48`).
    pub guard_hold_tokens: u8,
    /// Maximum dynamic logits D2H observation stride (`0` or `1` = observe every token, `offset 49`).
    pub max_observe_stride: u8,
    /// Explicit zero-padded space for future ABI extensions (`offset 50..64`).
    pub reserved: [u8; 14],
}

impl Default for RuntimeConfig {
    #[inline(always)]
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            head_dim: 0,
            num_heads: 0,
            paged_block_size: 0,
            q7_phase_scale: 0.0,
            phase_clamp_max_rad: 0.0,
            phase_step_min_rad: 0.0,
            entropy_singularity_eps: 0.0,
            max_prompt_anchors: 0,
            substrate_capacity: 0,
            rotary_dim: 0,
            execution_flags: 0,
            enable_zero_alloc_strict: 0,
            ledger_mode: 0,
            rope_layout: 0,
            guard_hold_tokens: 0,
            max_observe_stride: 0,
            reserved: [0; 14],
        }
    }
}

const _: () = {
    assert!(size_of::<RuntimeConfig>() == 64);
    assert!(align_of::<RuntimeConfig>() == 64);
    assert!(offset_of!(RuntimeConfig, struct_size) == 0);
    assert!(offset_of!(RuntimeConfig, head_dim) == 4);
    assert!(offset_of!(RuntimeConfig, num_heads) == 8);
    assert!(offset_of!(RuntimeConfig, paged_block_size) == 12);
    assert!(offset_of!(RuntimeConfig, q7_phase_scale) == 16);
    assert!(offset_of!(RuntimeConfig, phase_clamp_max_rad) == 20);
    assert!(offset_of!(RuntimeConfig, phase_step_min_rad) == 24);
    assert!(offset_of!(RuntimeConfig, entropy_singularity_eps) == 28);
    assert!(offset_of!(RuntimeConfig, max_prompt_anchors) == 32);
    assert!(offset_of!(RuntimeConfig, substrate_capacity) == 36);
    assert!(offset_of!(RuntimeConfig, rotary_dim) == 40);
    assert!(offset_of!(RuntimeConfig, execution_flags) == 44);
    assert!(offset_of!(RuntimeConfig, enable_zero_alloc_strict) == 45);
    assert!(offset_of!(RuntimeConfig, ledger_mode) == 46);
    assert!(offset_of!(RuntimeConfig, rope_layout) == 47);
    assert!(offset_of!(RuntimeConfig, guard_hold_tokens) == 48);
    assert!(offset_of!(RuntimeConfig, max_observe_stride) == 49);
    assert!(offset_of!(RuntimeConfig, reserved) == 50);
};

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
pub const Q7_MIN: PhaseQ7 = -128;

/// Maximum representable Q7 fixed-point value ($+127 \leftrightarrow +\frac{127}{128}\pi\text{ rad}$).
pub const Q7_MAX: PhaseQ7 = 127;

/// Neutral zero phase in Q7 fixed-point format.
pub const Q7_ZERO: PhaseQ7 = 0;

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

/// Maximum number of parallel physical working lanes (`8`).
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

/// Cognitive threshold bit for perspective-relative coupling (`RelativeCouplingMatrix`).
pub const THRESHOLD_RELATIVE_COUPLING: u32 = 1 << 6;

/// Cognitive threshold bit for dissipative overflow anchor crystallization (`PhaseAnchorRecord`).
pub const THRESHOLD_DISSIPATIVE_ANCHOR: u32 = 1 << 7;

/// Wide composite cognitive threshold spanned by `ManifoldGatewaySignal` (`0x0000_00FF`).
pub const THRESHOLD_MANIFOLD_GATEWAY: u32 = 0x0000_00FF;

/// Cognitive threshold bit for $SO(2)$ phase steering (`PhaseSteering`).
pub const THRESHOLD_PHASE_STEERING: u32 = 1 << 8;

/// Cognitive threshold bit for Fact/Logic/Explore sub-head routing (`SubheadRouter`).
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

/// Cognitive threshold bit for zero-copy `.lmnk` mapped storage I/O.
pub const THRESHOLD_MAPPED_STORAGE: u32 = 1 << 15;

/// Broadest operational cognitive threshold spanned by `StigmergicSubstrateOps` (`0x0000_FFFF`).
pub const THRESHOLD_STIGMERGY_WIDE: u32 = 0x0000_FFFF;

/// Number of dynamic channel slots tracked by `LiminalThresholdState`.
pub const NUM_LIMINAL_THRESHOLD_CHANNELS: usize = 8;

/// Minimum host-immutable compute slot bitmask (`bit 0`) preserved across untrusted snapshot restores
/// and shared across isolated Lv.3 drivers (`spine`, `envelope`).
pub const HOST_IMMUTABLE_PINNED_COMPUTE_MASK: u64 = 1u64 << 0;

// ===================================================================
// 4. 2 Primitives x 3 Drivers Lightweight Attribution Tags
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
    /// Bounded safety-envelope constraint and pin-floor driver (`2`).
    Envelope = 2,
}

impl DriverKind {
    /// Decodes a [`DriverKind`] branchlessly from a 2-bit driver field (`0 => System`, `1 => Ego`,
    /// `2 => Envelope`, reserved `3 => System`).
    #[inline(always)]
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        match bits & 0x03 {
            1 => Self::Ego,
            2 => Self::Envelope,
            _ => Self::System,
        }
    }
}

/// Packed 3-bit structural attribution tag (`[Driver(2b) | Primitive(1b)]`) inside a 1-byte
/// wrapper for zero-cost telemetry and bypass provenance across `2 Primitives x 3 Drivers`
/// (`0..=5`) without marker-trait bloat.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AttributionTag(pub u8);

impl AttributionTag {
    /// Packs `(primitive, driver)` into a 3-bit attribution tag byte (`0..=5`).
    #[inline(always)]
    #[must_use]
    pub const fn new(primitive: PrimitiveKind, driver: DriverKind) -> Self {
        Self(((primitive as u8) & 0x01) | (((driver as u8) & 0x03) << 1))
    }

    /// Attribution tag for (`Liminal`, `System`) = `0`.
    pub const LIMINAL_SYSTEM: Self = Self::new(PrimitiveKind::Liminal, DriverKind::System);
    /// Attribution tag for (`Stigmergy`, `System`) = `1`.
    pub const STIGMERGY_SYSTEM: Self = Self::new(PrimitiveKind::Stigmergy, DriverKind::System);
    /// Attribution tag for (`Liminal`, `Ego`) = `2`.
    pub const LIMINAL_EGO: Self = Self::new(PrimitiveKind::Liminal, DriverKind::Ego);
    /// Attribution tag for (`Stigmergy`, `Ego`) = `3`.
    pub const STIGMERGY_EGO: Self = Self::new(PrimitiveKind::Stigmergy, DriverKind::Ego);
    /// Attribution tag for (`Liminal`, `Envelope`) = `4`.
    pub const LIMINAL_ENVELOPE: Self = Self::new(PrimitiveKind::Liminal, DriverKind::Envelope);
    /// Attribution tag for (`Stigmergy`, `Envelope`) = `5`.
    pub const STIGMERGY_ENVELOPE: Self = Self::new(PrimitiveKind::Stigmergy, DriverKind::Envelope);

    /// Returns the raw 3-bit tag representation (`bits 0..=2`).
    #[inline(always)]
    #[must_use]
    pub const fn raw(self) -> u8 {
        self.0 & 0x07
    }

    /// Extracts the [`PrimitiveKind`] primitive axis (`bit 0`).
    #[inline(always)]
    #[must_use]
    pub const fn primitive(self) -> PrimitiveKind {
        PrimitiveKind::from_bit(self.0)
    }

    /// Extracts the [`DriverKind`] orthogonal driver axis (`bits 1..=2`).
    #[inline(always)]
    #[must_use]
    pub const fn driver(self) -> DriverKind {
        DriverKind::from_bits(self.0 >> 1)
    }

    /// Returns `true` when attributed to the deterministic [`DriverKind::System`] spine.
    #[inline(always)]
    #[must_use]
    pub const fn is_system(self) -> bool {
        ((self.0 >> 1) & 0x03) == DriverKind::System as u8
    }

    /// Returns `true` when attributed to the differential [`DriverKind::Ego`] operator.
    #[inline(always)]
    #[must_use]
    pub const fn is_ego(self) -> bool {
        ((self.0 >> 1) & 0x03) == DriverKind::Ego as u8
    }

    /// Returns `true` when attributed to the bounded [`DriverKind::Envelope`] constraint driver.
    #[inline(always)]
    #[must_use]
    pub const fn is_envelope(self) -> bool {
        ((self.0 >> 1) & 0x03) == DriverKind::Envelope as u8
    }
}

/// Branchless constructor helper mirroring `AttributionTag::new`.
#[inline(always)]
#[must_use]
pub const fn pack_attribution_tag(primitive: PrimitiveKind, driver: DriverKind) -> u8 {
    AttributionTag::new(primitive, driver).raw()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attribution_tag_two_by_three_mutual_exclusivity() {
        let cases = [
            (
                AttributionTag::LIMINAL_SYSTEM,
                0u8,
                PrimitiveKind::Liminal,
                DriverKind::System,
                (true, false, false),
            ),
            (
                AttributionTag::STIGMERGY_SYSTEM,
                1u8,
                PrimitiveKind::Stigmergy,
                DriverKind::System,
                (true, false, false),
            ),
            (
                AttributionTag::LIMINAL_EGO,
                2u8,
                PrimitiveKind::Liminal,
                DriverKind::Ego,
                (false, true, false),
            ),
            (
                AttributionTag::STIGMERGY_EGO,
                3u8,
                PrimitiveKind::Stigmergy,
                DriverKind::Ego,
                (false, true, false),
            ),
            (
                AttributionTag::LIMINAL_ENVELOPE,
                4u8,
                PrimitiveKind::Liminal,
                DriverKind::Envelope,
                (false, false, true),
            ),
            (
                AttributionTag::STIGMERGY_ENVELOPE,
                5u8,
                PrimitiveKind::Stigmergy,
                DriverKind::Envelope,
                (false, false, true),
            ),
        ];

        for (tag, expected_raw, expected_prim, expected_drv, (sys, ego, env)) in cases {
            assert_eq!(tag.raw(), expected_raw);
            assert_eq!(
                pack_attribution_tag(expected_prim, expected_drv),
                expected_raw
            );
            assert_eq!(tag.primitive(), expected_prim);
            assert_eq!(tag.driver(), expected_drv);
            assert_eq!(tag.is_system(), sys);
            assert_eq!(tag.is_ego(), ego);
            assert_eq!(tag.is_envelope(), env);
            let active_count = usize::from(tag.is_system())
                + usize::from(tag.is_ego())
                + usize::from(tag.is_envelope());
            assert_eq!(active_count, 1);
        }

        // Reserved driver code 3 falls back fail-safely to System.
        assert_eq!(DriverKind::from_bits(3), DriverKind::System);
        assert_eq!(DriverKind::from_bits(0b1111_1110), DriverKind::Envelope);
    }
}
