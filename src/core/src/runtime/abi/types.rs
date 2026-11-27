// name: src/core/src/runtime/abi/types.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! C-ABI constants, status codes, `#[repr(C)]` POD aliases (`lmnk_*_t`),
//! versioned `struct_size` parameter structs, and compile-time layout assertions
//! for `LiminiKa` `v0.4.0` (`0x0000_0400`).
//!
//! # Portable Wire & Device Contract
//! - **Byte Order**: All multi-byte scalars and `u64` SWAR words use **Canonical Little-Endian**.
//! - **Portable Device-Ready PODs**: Power-of-two `8 B..=128 B` fixed-width structs with zero
//!   implicit padding (`ArenaSpan`, `Rank1Descriptor`, `PointerTrail`, `GatewayFrame`,
//!   `PhaseAnchorRecord`, `LacunaPartition8`, `BoundarySignal`, `EnvelopeBounds`,
//!   `EnvelopeFeedback`, `EnvelopeState`, `LmnkContainerHeader`, etc.). These contain zero
//!   raw pointers and do NOT carry `struct_size`.
//! - **Versioned Parameter & Report Structs (`struct_size: u32`)**: All `*_params_t`, `*_config_t`,
//!   `*_view_t`, `*_report_t`, `lmnk_abi_info_t`, and `lmnk_runtime_policy_t` descriptors carry
//!   a leading `struct_size: u32` field validated fail-closed across the C-ABI firewall.
//! - **Transient Host Parameter Structs**: Pointer-carrying call descriptors
//!   (`lmnk_rope_steer_params_t`, `lmnk_rope_procrustes_params_t`, `lmnk_rank1_eval_params_t`,
//!   `lmnk_gateway_steer_params_t`, `lmnk_section_input_t`, `lmnk_container_write_params_t`,
//!   `lmnk_container_section_view_t`) are strictly transient in-memory descriptors and are
//!   NEVER persisted or hashed.

use super::*;

pub use crate::drivers::{
    ComputeDispatchMask, DistortionDelta, DistortionTension, ENVELOPE_BREACH_CHATTER,
    ENVELOPE_BREACH_EXTERIOR, ENVELOPE_BREACH_KNOWN_MASK, ENVELOPE_BREACH_NONE,
    ENVELOPE_BREACH_PRESSURE, ENVELOPE_BREACH_UNDEFINED_FAULT, ENVELOPE_FLAG_CONTRACTED,
    ENVELOPE_FLAG_KNOWN_MASK, ENVELOPE_FLAG_RECOVERY_ACTIVE, ENVELOPE_LOCK_FAULT_ISOLATED,
    ENVELOPE_LOCK_IRREVERSIBLE, ENVELOPE_LOCK_UNOBSERVED, EgoIntentAxes, EntropyController,
    EntropyState, EnvelopeBounds, EnvelopeFeedback, EnvelopeState,
    HOST_IMMUTABLE_PINNED_COMPUTE_MASK, LiminalEgoState, SPINE_BYPASS_VETO_GUARD_ACTIVE,
    SPINE_BYPASS_VETO_KNOWN_MASK, SPINE_BYPASS_VETO_NONE, SPINE_BYPASS_VETO_PINNED_LOCK,
    SPINE_BYPASS_VETO_REPULSION_ACTIVE, SPINE_CONTROL_FLAG_PRE_BYPASS_ENABLED,
    SPINE_CONTROL_FLAG_PREV_RESIDENCY_VALID, SPINE_CONTROL_FLAG_PREV_WAS_ABSTRACT,
    SPINE_CONTROL_FLAG_TRANSIENT_GUARD_RELAXED, SPINE_CONTROL_FLAG_TURN_GUARD_ACTIVE,
    SPINE_CONTROL_KNOWN_FLAGS, SPINE_DECODE_FLAG_REWARD_OVERRIDE,
    SPINE_DECODE_INPUT_FLAG_GREEDY_SAMPLING, SPINE_DECODE_INPUT_FLAG_PRE_BYPASS_ENABLED,
    SPINE_DECODE_INPUT_FLAG_REPULSION_ACTIVE, SPINE_DECODE_INPUT_FLAG_REWARD_OVERRIDE,
    SPINE_ENTROPY_CALM_THRESH_Q15, SPINE_ENTROPY_GUARD_RELAX_MAX_Q15, SPINE_ENTROPY_Q15_ONE,
    SPINE_ENTROPY_REWARD_LOW_THRESH_Q15, SPINE_ENTROPY_SURGE_THRESH_Q15,
    SPINE_ENTROPY_TENSION_FLOOR_Q15, SPINE_MIN_DYNAMIC_OBSERVE_STRIDE, SPINE_SENTINEL_AUTO_U8,
    SPINE_STEP_EVENT_BOUNDARY_CHURN, SPINE_STEP_EVENT_GUARD_REESCALATED,
    SPINE_STEP_EVENT_INSTABILITY_RECOVERY, SPINE_STEP_EVENT_LOGITS_OBSERVE_SKIPPED,
    SPINE_STEP_EVENT_NONE, SPINE_STEP_EVENT_TRANSIENT_GUARD_RELAXED, SpineDecodeInput,
    SpineStepPlan, SystemPressureReading,
};
pub use crate::foundation::{
    AlignedSlice, AlignedSliceMut, ArenaSpan, DisjointMutTriple, KvMotionDeltaPacket, PagedBitmask,
    RANK1_GAIN_UNUSED, Rank1Descriptor, SubheadConfig, TaggedPtr,
};
pub use crate::liminal::{
    AttributionTag, BOUNDARY_SIGNAL_FLAG_EXTERIOR_ALERT, BOUNDARY_SIGNAL_FLAG_FAULT,
    BOUNDARY_SIGNAL_FLAG_IRREVERSIBLE_INTENT, BOUNDARY_SIGNAL_FLAG_OBSERVED,
    BOUNDARY_SIGNAL_FLAG_STATIONARY_READY, BOUNDARY_SIGNAL_KNOWN_FLAGS, BoundarySignal,
    LMNK_ABI_COMPAT_MASK, LMNK_ABI_COMPAT_VERSION_HEX, LMNK_ABI_VERSION_CSTR, LMNK_ABI_VERSION_HEX,
    LMNK_ABI_VERSION_MAJOR, LMNK_ABI_VERSION_MINOR, LMNK_ABI_VERSION_PATCH,
    LMNK_ABI_VERSION_STRING, LMNK_CAP_INPLACE_CONTEXT_INIT, LMNK_CAP_SIMD_AVX2, LMNK_CAP_SIMD_NEON,
    LMNK_CAP_SIMD_WASM128, LMNK_CAP_SWAR_BASELINE, LMNK_CAP_ZERO_ALLOC_CORE, LMNK_ERR_ALIGNMENT,
    LMNK_ERR_ANCHOR_NOT_FOUND, LMNK_ERR_BUFFER_TOO_SMALL, LMNK_ERR_CONTAINER_CORRUPTED,
    LMNK_ERR_ENTROPY_SINGULARITY, LMNK_ERR_EXTENSION_NOT_LOADED, LMNK_ERR_INVALID_ARGUMENT,
    LMNK_ERR_INVALID_STATE, LMNK_ERR_IO_FAILURE, LMNK_ERR_KV_KEYFRAME_SEEK_FAILED,
    LMNK_ERR_LEDGER_FULL, LMNK_ERR_NOT_IMPLEMENTED, LMNK_ERR_NULL_POINTER,
    LMNK_ERR_OBSTRUCTION_UNRESOLVED, LMNK_ERR_OUT_OF_MEMORY, LMNK_ERR_PANIC_CAUGHT,
    LMNK_ERR_PHASE_OVERFLOW, LMNK_ERR_PHASE_STEP_UNDERFLOW, LMNK_ERR_STORAGE_MAP_FAILED,
    LMNK_ERR_SUBHEAD_ROUTING_FAILED, LMNK_ERR_SUBSTRATE_FULL, LMNK_ERR_TOPOLOGY_MISMATCH,
    LMNK_ERR_UNKNOWN, LMNK_ERR_VERSION_MISMATCH, LMNK_OK, LacunaPartition8, LiminalThresholdState,
    LmnkStatus, PhaseQ7, RuntimeConfig, is_compatible_abi_version, validate_abi_version,
};
pub use crate::runtime::boundary_delta::LiminalBoundaryDelta;
pub use crate::runtime::context::{CoreStateStack, LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES};
pub use crate::runtime::gateway::{CausalProvenanceReport, GatewayFrame};
pub use crate::runtime::horizon::{EpistemicHorizonReading, RamPhaseKvEntry};
pub use crate::runtime::storage::{
    KvKeyframeHeader, KvKeyframePacket, LMNK_CONTAINER_FLAG_COMPILED_CACHE,
    LMNK_CONTAINER_FLAG_PORTABLE_PACKAGE, LMNK_CONTAINER_FLAG_RUNTIME_SNAPSHOT,
    LMNK_CONTAINER_HEADER_BYTES, LMNK_CONTAINER_MAGIC, LMNK_MAX_CONTAINER_BYTES,
    LMNK_MAX_CONTAINER_SECTIONS, LMNK_SECTION_BLOOM_FILTER, LMNK_SECTION_CAUSAL_PROVENANCE,
    LMNK_SECTION_CONTENT_HASH, LMNK_SECTION_ENTRY_BYTES, LMNK_SECTION_LEDGER_SLOT,
    LMNK_SECTION_LOGICAL_PFRAMES, LMNK_SECTION_PHASE_ANCHORS, LMNK_SECTION_POINTER_TRAILS,
    LMNK_SECTION_RECALL_METADATA, LMNK_SECTION_RUNTIME_SNAPSHOT, LMNK_SECTION_SIGNATURE,
    LmnkContainerHeader, LmnkSectionEntry, LmnkSectionInput, LmnkSectionKind,
};
pub use crate::runtime::telemetry::LiminalTopologyTelemetrySnapshot;
pub use crate::runtime::topology::{
    CORE_TOPOLOGY, LMNK_DAG_FLAG_AXIOM_CUT_LACUNA_ORTHOGONAL, LMNK_DAG_FLAG_NO_STD_CORE,
    LMNK_DAG_FLAG_SPINE_EGO_ISOLATED, LMNK_DAG_FLAG_THREE_DRIVERS_ISOLATED,
    LMNK_DAG_FLAG_Y_SHAPED_PARALLEL_ROOTS, LMNK_TOPOLOGY_CHANNEL_COUNT,
    LMNK_TOPOLOGY_DAG_EDGE_MATRIX, LMNK_TOPOLOGY_DAG_FLAGS_CANONICAL, LMNK_TOPOLOGY_LAYER_COUNT,
    LMNK_TOPOLOGY_MAGIC, LMNK_TOPOLOGY_PINNED_CHANNEL_MASK, LMNK_TOPOLOGY_SANDBOX_CHANNEL_MASK,
    LiminalTopologyMatrix,
};
pub use crate::stigmergy::{
    ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA, CognitiveCircuitActivation, CognitiveStepOp,
    CognitiveTrajectoryLedger, ContrastivePhaseBloomFilter, CorePrimitiveState, LEDGER_HISTORY_LEN,
    LMNK_PHASE_ANCHOR_MAGIC, LMNK_PHASE_ANCHOR_VERSION, LedgerSlot, MAX_PHASE_ANCHORS,
    OverlapDiscoveryReport, PhaseAnchorRecord, PointerTrail, StigmergicFragmentEnsembleReport,
    StigmergicGranuleState, StigmergicReconstitutionReport, StigmergicThresholds,
};

// ===================================================================
// Versioned Parameter Struct Option Flags (v0.4.0)
// ===================================================================

/// Apply mechanical spine policy fields in [`lmnk_runtime_policy_t`].
pub const LMNK_POLICY_FLAG_SPINE: u32 = 1 << 0;
/// Apply compute-pool mask and baseline-pressure fields in [`lmnk_runtime_policy_t`].
pub const LMNK_POLICY_FLAG_COMPUTE_POOL: u32 = 1 << 1;
/// Valid bitmask for [`lmnk_runtime_policy_t::flags`].
pub const LMNK_POLICY_KNOWN_FLAGS: u32 = LMNK_POLICY_FLAG_SPINE | LMNK_POLICY_FLAG_COMPUTE_POOL;

/// Marks the turn as contradiction-guard active in [`lmnk_turn_begin_params_t::flags`].
pub const LMNK_TURN_FLAG_GUARD_ACTIVE: u32 = 1 << 0;
/// Marks the turn as unobserved-locked in [`lmnk_turn_begin_params_t::flags`].
pub const LMNK_TURN_FLAG_UNOBSERVED_LOCKED: u32 = 1 << 1;
/// Valid bitmask for [`lmnk_turn_begin_params_t::flags`].
pub const LMNK_TURN_KNOWN_FLAGS: u32 =
    LMNK_TURN_FLAG_GUARD_ACTIVE | LMNK_TURN_FLAG_UNOBSERVED_LOCKED;

/// Marks external contradiction guard as active for this step in [`lmnk_decode_step_params_t::flags`].
pub const LMNK_DECODE_STEP_FLAG_EXTERNAL_GUARD: u32 = 1 << 0;
/// Valid bitmask for [`lmnk_decode_step_params_t::flags`].
pub const LMNK_DECODE_STEP_KNOWN_FLAGS: u32 = LMNK_DECODE_STEP_FLAG_EXTERNAL_GUARD;

/// Disable soft PhaseClamp and apply direct full-head Interleaved rotation in [`lmnk_rope_steer_params_t::flags`].
pub const LMNK_ROPE_STEER_FLAG_UNCLAMPED: u32 = 1 << 0;
/// Valid bitmask for [`lmnk_rope_steer_params_t::flags`].
pub const LMNK_ROPE_STEER_KNOWN_FLAGS: u32 = LMNK_ROPE_STEER_FLAG_UNCLAMPED;

/// Accumulate `y_inout += delta` in-place instead of overwriting `y_inout = delta` in [`lmnk_rank1_eval_params_t::flags`].
pub const LMNK_RANK1_FLAG_FUSED: u32 = 1 << 0;
/// Valid bitmask for [`lmnk_rank1_eval_params_t::flags`].
pub const LMNK_RANK1_KNOWN_FLAGS: u32 = LMNK_RANK1_FLAG_FUSED;

/// Ignore `params->bounds` and apply canonical [`EnvelopeBounds::new()`] in [`lmnk_envelope_step_params_t::flags`].
pub const LMNK_ENVELOPE_STEP_FLAG_DEFAULT_BOUNDS: u32 = 1 << 0;
/// Valid bitmask for [`lmnk_envelope_step_params_t::flags`].
pub const LMNK_ENVELOPE_STEP_KNOWN_FLAGS: u32 = LMNK_ENVELOPE_STEP_FLAG_DEFAULT_BOUNDS;

/// Stamp irreversible transition intent (`BOUNDARY_SIGNAL_FLAG_IRREVERSIBLE_INTENT`) in
/// [`lmnk_boundary_emit_params_t::flags`].
pub const LMNK_BOUNDARY_EMIT_FLAG_IRREVERSIBLE_INTENT: u32 = 1 << 0;
/// Valid bitmask for [`lmnk_boundary_emit_params_t::flags`].
pub const LMNK_BOUNDARY_EMIT_KNOWN_FLAGS: u32 = LMNK_BOUNDARY_EMIT_FLAG_IRREVERSIBLE_INTENT;

// ===================================================================
// Portable POD Type Aliases (`lmnk_*_t`)
// ===================================================================

/// 64-bit capability flags bitmask type (`lmnk_capability_flags_t`).
pub type lmnk_capability_flags_t = u64;
/// Standardized 32-bit signed status code (`lmnk_status_t`).
pub type lmnk_status_t = LmnkStatus;
/// Signed Q0.7 phase element (`lmnk_q7_t`).
pub type lmnk_q7_t = PhaseQ7;
/// 64-bit tagged pointer (`lmnk_tagged_ptr_t`).
pub type lmnk_tagged_ptr_t = TaggedPtr;
/// 8-byte portable arena span `(offset, len)` (`ArenaSpan`).
pub type lmnk_arena_span_t = ArenaSpan;
/// 64-byte versioned C-ABI runtime configuration descriptor (`RuntimeConfig`).
pub type lmnk_config_t = RuntimeConfig;
/// 64-byte `.lmnk` unified container header (`LmnkContainerHeader`).
pub type lmnk_container_header_t = LmnkContainerHeader;
/// 16-byte `.lmnk` section directory entry (`LmnkSectionEntry`).
pub type lmnk_section_entry_t = LmnkSectionEntry;
/// 64-byte static architectural topology matrix (`LiminalTopologyMatrix`).
pub type lmnk_topology_matrix_t = LiminalTopologyMatrix;
/// 64-byte KV-keyframe header (`KvKeyframeHeader`).
pub type lmnk_kv_keyframe_header_t = KvKeyframeHeader;
/// 32-byte 256-bit PagedBlock bitmask (`PagedBitmask`).
pub type lmnk_paged_bitmask_t = PagedBitmask;
/// 32-byte parallel multi-head sub-head routing config (`SubheadConfig`).
pub type lmnk_subhead_config_t = SubheadConfig;
/// 32-byte portable arena-indexed descriptor for Rank-1 residual injection (`Rank1Descriptor`).
pub type lmnk_rank1_descriptor_t = Rank1Descriptor;
/// 64-byte FastPathLedger telemetry and shortcut slot (`LedgerSlot`).
pub type lmnk_ledger_slot_t = LedgerSlot;
/// 128-byte dual-cacheline Stigmergic Pointer Trail (`PointerTrail`).
pub type lmnk_pointer_trail_t = PointerTrail;
/// 32-byte Entropy Controller state descriptor (`EntropyState`).
pub type lmnk_entropy_state_t = EntropyState;
/// 128-byte Gateway Frame (`GatewayFrame`).
pub type lmnk_gateway_frame_t = GatewayFrame;
/// 16-byte Causal Provenance Report (`CausalProvenanceReport`).
pub type lmnk_causal_report_t = CausalProvenanceReport;
/// 128-byte crystallized Phase Anchor Record (`PhaseAnchorRecord`, `"LNKR"`).
pub type lmnk_phase_anchor_record_t = PhaseAnchorRecord;
/// 64-byte Contrastive Phase Bloom Filter (`ContrastivePhaseBloomFilter`).
pub type lmnk_bloom_filter_t = ContrastivePhaseBloomFilter;
/// 32-byte Stigmergic Thresholds configuration (`StigmergicThresholds`).
pub type lmnk_stigmergic_thresholds_t = StigmergicThresholds;
/// 8-bit Stigmergic Action Decision discriminant (`StigmergicActionDecision`).
pub type lmnk_action_decision_t = u8;
/// 16-byte Stigmergic Reconstitution Report (`StigmergicReconstitutionReport`).
pub type lmnk_reconstitution_report_t = StigmergicReconstitutionReport;
/// 8-byte Cognitive Trajectory Ledger (`CognitiveTrajectoryLedger`).
pub type lmnk_cognitive_ledger_t = CognitiveTrajectoryLedger;
/// 32-byte Epistemic Horizon Reading (`EpistemicHorizonReading`).
pub type lmnk_epistemic_horizon_t = EpistemicHorizonReading;
/// 32-byte Liminal Threshold State (`LiminalThresholdState`).
pub type lmnk_liminal_threshold_state_t = LiminalThresholdState;
/// 32-byte Stigmergic Granule State (`StigmergicGranuleState`).
pub type lmnk_stigmergic_granule_state_t = StigmergicGranuleState;
/// 32-byte Overlap & Discovery Report (`OverlapDiscoveryReport`).
pub type lmnk_overlap_discovery_report_t = OverlapDiscoveryReport;
/// 128-byte Core State Stack (`CoreStateStack`).
pub type lmnk_core_state_stack_t = CoreStateStack;
/// 32-byte Self-Introspecting Topology Telemetry Snapshot (`LiminalTopologyTelemetrySnapshot`).
pub type lmnk_topology_snapshot_t = LiminalTopologyTelemetrySnapshot;
/// 16-byte Temporal Boundary Velocity Analyzer (`LiminalBoundaryDelta`).
pub type lmnk_boundary_delta_t = LiminalBoundaryDelta;
/// 16-byte Ego Intent Axes (`EgoIntentAxes`).
pub type lmnk_ego_intent_axes_t = EgoIntentAxes;
/// 16-byte Fragmented Micro-Ensemble Report (`StigmergicFragmentEnsembleReport`).
pub type lmnk_fragment_ensemble_report_t = StigmergicFragmentEnsembleReport;
/// 64-byte Wide-Aperture Subjective Ego State (`LiminalEgoState`).
pub type lmnk_liminal_ego_state_t = LiminalEgoState;
/// 16-byte 4-State Lacuna Partition (`LacunaPartition8`).
pub type lmnk_lacuna_partition_t = LacunaPartition8;
/// 1-byte Structural Attribution Tag (`AttributionTag`).
pub type lmnk_attribution_tag_t = AttributionTag;
/// 64-byte Unified Primitive State (`CorePrimitiveState`).
pub type lmnk_core_primitive_state_t = CorePrimitiveState;
/// 128-byte P-Frame Distortion Delta (`DistortionDelta`).
pub type lmnk_distortion_delta_t = DistortionDelta;
/// 32-byte Asymmetric Homeostatic Tension (`DistortionTension`).
pub type lmnk_distortion_tension_t = DistortionTension;
/// 8-byte Inline Sub-Record: External System Gauge-Zero Pressure Reading (`SystemPressureReading`).
pub type lmnk_system_pressure_reading_t = SystemPressureReading;
/// 40-byte Inline Sub-Record: Pre-Execution Anonymous Compute Dispatch Mask (`ComputeDispatchMask`).
pub type lmnk_compute_dispatch_mask_t = ComputeDispatchMask;
/// 32-byte Spine Decode Signal Input (`SpineDecodeInput`).
pub type lmnk_spine_decode_input_t = SpineDecodeInput;
/// 32-byte Domain-Agnostic Boundary Signal (`BoundarySignal`).
pub type lmnk_boundary_signal_t = BoundarySignal;
/// 32-byte Safety-Envelope Constraint Bounds (`EnvelopeBounds`).
pub type lmnk_envelope_bounds_t = EnvelopeBounds;
/// 16-byte Safety-Envelope Step Feedback (`EnvelopeFeedback`).
pub type lmnk_envelope_feedback_t = EnvelopeFeedback;
/// 64-byte Persistent Safety-Envelope Driver State (`EnvelopeState`).
pub type lmnk_envelope_state_t = EnvelopeState;

// ===================================================================
// Versioned v0.4.0 Fixed-Width Parameter & View Structs (Zero Padding)
// ===================================================================

/// 64-byte versioned ABI handshake descriptor returned by [`crate::runtime::abi::lmnk_abi_info`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct lmnk_abi_info_t {
    /// Parameter struct byte size (`sizeof(lmnk_abi_info_t)` == `64`, `offset 0..4`).
    pub struct_size: u32,
    /// Packed hexadecimal ABI version (`0x0000_0400`, `offset 4..8`).
    pub abi_version_hex: u32,
    /// Required byte size of `RuntimeContext` for `lmnk_context_init_in_place` (`offset 8..12`).
    pub context_size: u32,
    /// Required byte alignment (`64`) of `RuntimeContext` (`offset 12..16`).
    pub context_align: u32,
    /// Byte size of a serialized runtime snapshot container (`offset 16..20`).
    pub snapshot_max_bytes: u32,
    /// IEEE 802.3 CRC32 of the compiled core topology matrix (`offset 20..24`).
    pub topology_crc32: u32,
    /// Active runtime capability flags (`LMNK_CAP_*`, `offset 24..32`).
    pub capability_flags: u64,
    /// Reserved zero-padded bytes for future minor extensions (`offset 32..64`).
    pub reserved: [u8; 32],
}

impl Default for lmnk_abi_info_t {
    #[inline(always)]
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version_hex: 0,
            context_size: 0,
            context_align: 0,
            snapshot_max_bytes: 0,
            topology_crc32: 0,
            capability_flags: 0,
            reserved: [0; 32],
        }
    }
}

/// 64-byte (1 cacheline) versioned policy parameter struct for [`crate::runtime::abi::lmnk_context_configure_policy`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct lmnk_runtime_policy_t {
    /// Parameter struct byte size (`sizeof(lmnk_runtime_policy_t)` == `64`, `offset 0..4`).
    pub struct_size: u32,
    /// Policy section bitflags (`LMNK_POLICY_FLAG_SPINE | LMNK_POLICY_FLAG_COMPUTE_POOL`, `offset 4..8`).
    pub flags: u32,
    /// Monotonic host compute pin floor (`0` preserves active floor, `offset 8..16`).
    pub baseline_pinned_compute_mask: u64,
    /// Active anonymous compute slot mask (`offset 16..24`).
    pub active_compute_mask: u64,
    /// Repulsion-eligible compute slot mask (`offset 24..32`).
    pub repulsion_compute_mask: u64,
    /// Reversible bypass-eligible compute slot mask (`offset 32..40`).
    pub reversible_compute_mask: u64,
    /// Gauge-Zero baseline pressure in Q0.15 (`0` preserves current baseline, `offset 40..42`).
    pub baseline_pressure_q15: u16,
    /// Opening decode token hold window for transient guard pins (`0` = full turn, `offset 42`).
    pub guard_hold_tokens: u8,
    /// Maximum dynamic logits D2H observation stride (`0` normalized to `1`, `offset 43`).
    pub max_observe_stride: u8,
    /// Non-zero enables pre-bypass scheduling (`offset 44`).
    pub pre_bypass_enabled: u8,
    /// Explicit zero-initialized extension space (`offset 45..64`).
    pub reserved: [u8; 19],
}

impl Default for lmnk_runtime_policy_t {
    #[inline(always)]
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            flags: 0,
            baseline_pinned_compute_mask: 0,
            active_compute_mask: 0,
            repulsion_compute_mask: 0,
            reversible_compute_mask: 0,
            baseline_pressure_q15: 0,
            guard_hold_tokens: 0,
            max_observe_stride: 1,
            pre_bypass_enabled: 0,
            reserved: [0; 19],
        }
    }
}

/// 16-byte versioned turn-boundary parameter struct for [`crate::runtime::abi::lmnk_context_begin_turn`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct lmnk_turn_begin_params_t {
    /// Parameter struct byte size (`sizeof(lmnk_turn_begin_params_t)` == `16`, `offset 0..4`).
    pub struct_size: u32,
    /// Turn policy bitflags (`LMNK_TURN_FLAG_*`, `offset 4..8`).
    pub flags: u32,
    /// Explicit zero padding (`offset 8..16`).
    pub reserved: [u8; 8],
}

impl Default for lmnk_turn_begin_params_t {
    #[inline(always)]
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            flags: 0,
            reserved: [0; 8],
        }
    }
}

/// 48-byte versioned decode-step input parameter struct for [`crate::runtime::abi::lmnk_context_step_decode`].
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct lmnk_decode_step_params_t {
    /// Parameter struct byte size (`sizeof(lmnk_decode_step_params_t)` == `48`, `offset 0..4`).
    pub struct_size: u32,
    /// Step control bitflags (`LMNK_DECODE_STEP_FLAG_*`, unknown bits fail-closed, `offset 4..8`).
    pub flags: u32,
    /// Explicit zero-initialized alignment padding (`offset 8..16`).
    pub reserved: [u8; 8],
    /// 32-byte decode signal input (`offset 16..48`).
    pub decode_input: lmnk_spine_decode_input_t,
}

impl Default for lmnk_decode_step_params_t {
    #[inline(always)]
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            flags: 0,
            reserved: [0; 8],
            decode_input: lmnk_spine_decode_input_t::default(),
        }
    }
}

/// 64-byte (1 cacheline) versioned decode-step report emitted by [`crate::runtime::abi::lmnk_context_step_decode`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct lmnk_decode_step_report_t {
    /// Parameter struct byte size (`sizeof(lmnk_decode_step_report_t)` == `64`, `offset 0..4`).
    pub struct_size: u32,
    /// Next logits observation mode (`0` = Full, `1` = Skip, `offset 4`).
    pub next_observe_mode: u8,
    /// Active bypass veto bitmask (`SPINE_BYPASS_VETO_*`, `offset 5`).
    pub veto_mask: u8,
    /// Step event bitflags (`SPINE_STEP_EVENT_*`, `offset 6`).
    pub event_flags: u8,
    /// Active dynamic observation stride (`offset 7`).
    pub effective_observe_stride: u8,
    /// 40-byte pre-execution anonymous compute dispatch mask (`offset 8..48`).
    pub dispatch: lmnk_compute_dispatch_mask_t,
    /// Reserved zero-padded bytes for future extensions (`offset 48..64`).
    pub reserved: [u8; 16],
}

impl Default for lmnk_decode_step_report_t {
    #[inline(always)]
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            next_observe_mode: 0,
            veto_mask: 0,
            event_flags: 0,
            effective_observe_stride: 1,
            dispatch: lmnk_compute_dispatch_mask_t::default(),
            reserved: [0; 16],
        }
    }
}

/// 64-byte abstracted read-only runtime state view emitted by [`crate::runtime::abi::lmnk_context_get_state`].
///
/// Decouples external callers from internal `SystemSpineState`, `EnvelopeState`, and
/// `AnonymousComputePool` layouts.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct lmnk_runtime_state_view_t {
    /// Parameter struct byte size (`sizeof(lmnk_runtime_state_view_t)` == `64`, `offset 0..4`).
    pub struct_size: u32,
    /// Active spine control flags (`SPINE_CONTROL_FLAG_*`, `offset 4`).
    pub control_flags: u8,
    /// Most recent bypass veto bitmask (`SPINE_BYPASS_VETO_*`, `offset 5`).
    pub last_veto_mask: u8,
    /// Configured opening-token transient guard hold window (`offset 6`).
    pub guard_hold_tokens: u8,
    /// Configured maximum dynamic logits observation stride (`offset 7`).
    pub max_observe_stride: u8,
    /// Permanent monotonic host-pinned compute mask floor owned by `EnvelopeState` (`offset 8..16`).
    pub baseline_pinned_compute_mask: u64,
    /// Active effective pinned compute mask owned by `EnvelopeState` (`offset 16..24`).
    pub pinned_compute_mask: u64,
    /// Last executed anonymous compute slot mask (`offset 24..32`).
    pub last_executed_mask: u64,
    /// Active anonymous compute pool slot mask (`offset 32..40`).
    pub active_compute_mask: u64,
    /// Most recently observed exit entropy in Q0.15 (`offset 40..42`).
    pub last_observed_entropy_q15: u16,
    /// Current `AnonymousComputePool` Gauge-Zero baseline pressure in Q0.15 (`offset 42..44`).
    pub baseline_pressure_q15: u16,
    /// Non-zero when whole-token steering bypass is active (`offset 44`).
    pub whole_token_bypass_active: u8,
    /// Non-zero when unobserved-lock is engaged on `EnvelopeState` (`offset 45`).
    pub unobserved_locked: u8,
    /// Current `EnvelopeState` aperture contraction scale in Q0.7 (`0..=128`, `offset 46`).
    pub envelope_contraction_q7: u8,
    /// Most recent `EnvelopeState` breach bitmask (`ENVELOPE_BREACH_*`, `offset 47`).
    pub envelope_breach_mask: u8,
    /// Cumulative `EnvelopeState` violation epoch counter (`offset 48..52`).
    pub envelope_violation_epochs: u32,
    /// Current `EnvelopeState` adaptive baseline pressure in Q0.15
    /// (`offset 52..54`, distinct from `baseline_pressure_q15`).
    pub envelope_baseline_pressure_q15: u16,
    /// Packed `EnvelopeState` lock bitflags (`ENVELOPE_LOCK_*`, `offset 54`).
    pub envelope_lock_flags: u8,
    /// Consecutive calm steps accumulated toward `EnvelopeState` baseline EMA resumption (`offset 55`).
    pub envelope_ema_resume_dwell: u8,
    /// Reserved zero-padded bytes (`offset 56..64`).
    pub reserved: [u8; 8],
}

impl Default for lmnk_runtime_state_view_t {
    #[inline(always)]
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            control_flags: 0,
            last_veto_mask: 0,
            guard_hold_tokens: 0,
            max_observe_stride: 1,
            baseline_pinned_compute_mask: 0,
            pinned_compute_mask: 0,
            last_executed_mask: 0,
            active_compute_mask: 0,
            last_observed_entropy_q15: 0,
            baseline_pressure_q15: 0,
            whole_token_bypass_active: 0,
            unobserved_locked: 0,
            envelope_contraction_q7: 128,
            envelope_breach_mask: 0,
            envelope_violation_epochs: 0,
            envelope_baseline_pressure_q15: 0,
            envelope_lock_flags: 0,
            envelope_ema_resume_dwell: 0,
            reserved: [0; 8],
        }
    }
}

/// 96-byte (`#[repr(C, align(32))]`) versioned parameter struct for Tier-1 standalone
/// [`crate::runtime::abi::lmnk_envelope_step`].
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct lmnk_envelope_step_params_t {
    /// Parameter struct byte size (`sizeof(lmnk_envelope_step_params_t)` == `96`, `offset 0..4`).
    pub struct_size: u32,
    /// Option bitflags (`LMNK_ENVELOPE_STEP_FLAG_DEFAULT_BOUNDS`, unknown bits fail-closed, `offset 4..8`).
    pub flags: u32,
    /// Explicit zero-initialized extension space (`offset 8..32`).
    pub reserved: [u8; 24],
    /// 32-byte constraint bounds (ignored when `LMNK_ENVELOPE_STEP_FLAG_DEFAULT_BOUNDS` is set, `offset 32..64`).
    pub bounds: lmnk_envelope_bounds_t,
    /// 32-byte boundary observation signal (`offset 64..96`).
    pub signal: lmnk_boundary_signal_t,
}

impl Default for lmnk_envelope_step_params_t {
    #[inline(always)]
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            flags: 0,
            reserved: [0; 24],
            bounds: EnvelopeBounds::new(),
            signal: BoundarySignal::new(),
        }
    }
}

/// 16-byte (`#[repr(C)]`) versioned parameter struct for
/// [`crate::runtime::abi::lmnk_context_emit_boundary_signal`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct lmnk_boundary_emit_params_t {
    /// Parameter struct byte size (`sizeof(lmnk_boundary_emit_params_t)` == `16`, `offset 0..4`).
    pub struct_size: u32,
    /// Option bitflags (`LMNK_BOUNDARY_EMIT_FLAG_IRREVERSIBLE_INTENT`, unknown bits fail-closed, `offset 4..8`).
    pub flags: u32,
    /// Normalized signal pressure in Q0.15 (`0..=32768`, `offset 8..10`).
    pub pressure_q15: u16,
    /// Observed boundary chatter intensity in Q0.7 (`0..=128`, `offset 10`).
    pub chatter_q7: u8,
    /// Explicit zero padding (`offset 11`).
    pub reserved: u8,
    /// Monotonic logical step/epoch identifier (`offset 12..16`).
    pub epoch_id: u32,
}

impl Default for lmnk_boundary_emit_params_t {
    #[inline(always)]
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            flags: 0,
            pressure_q15: 0,
            chatter_q7: 0,
            reserved: 0,
            epoch_id: 0,
        }
    }
}

/// 24-byte value envelope for [`crate::runtime::abi::lmnk_lacuna_partition_eval`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct lmnk_lacuna_eval_params_t {
    /// Envelope byte size (`sizeof(lmnk_lacuna_eval_params_t)` == `24`, `offset 0..4`).
    pub struct_size: u32,
    /// Bitmask of granules currently inside stationary self-reference (`offset 4`).
    pub self_ref_mask: u8,
    /// Q0.7 observation deadband threshold (`offset 5`).
    pub deadband_q7: u8,
    /// Explicit zero padding (`offset 6..8`).
    pub reserved: [u8; 2],
    /// Per-granule observed exterior density in Q0.7 (`offset 8..16`).
    pub other_density_q7: [u8; 8],
    /// Per-granule noise tolerance floor in Q0.7 (`offset 16..24`).
    pub noise_tolerance_q7: [u8; 8],
}

impl Default for lmnk_lacuna_eval_params_t {
    #[inline(always)]
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            self_ref_mask: 0,
            deadband_q7: 0,
            reserved: [0; 2],
            other_density_q7: [0; 8],
            noise_tolerance_q7: [0; 8],
        }
    }
}

/// 40-byte versioned container verification summary emitted by [`crate::runtime::abi::lmnk_container_verify`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct lmnk_container_verify_report_t {
    /// Envelope byte size (`sizeof(lmnk_container_verify_report_t)` == `40`, `offset 0..4`).
    pub struct_size: u32,
    /// Verified container ABI version hex (`offset 4..8`).
    pub abi_version: u32,
    /// Verified topology CRC32 (`offset 8..12`).
    pub topology_crc32: u32,
    /// Container classification bitflags (`LMNK_CONTAINER_FLAG_*`, `offset 12..16`).
    pub container_flags: u32,
    /// Verified total container size in bytes (`offset 16..24`).
    pub total_size: u64,
    /// Container Unix epoch timestamp in seconds (`offset 24..32`).
    pub timestamp_epoch_sec: u64,
    /// Number of validated sections in the section directory (`offset 32..34`).
    pub section_count: u16,
    /// Explicit zero padding (`offset 34..40`).
    pub reserved: [u8; 6],
}

impl Default for lmnk_container_verify_report_t {
    #[inline(always)]
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            abi_version: 0,
            topology_crc32: 0,
            container_flags: 0,
            total_size: 0,
            timestamp_epoch_sec: 0,
            section_count: 0,
            reserved: [0; 6],
        }
    }
}

// ===================================================================
// Transient Host Call Envelopes (Pointer-Carrying, Never Persisted)
// ===================================================================

/// Transient host call envelope for [`crate::runtime::abi::lmnk_rope_steer_q`].
///
/// Strictly in-memory; never persisted or hashed. All tensor buffers are strongly typed `float*`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct lmnk_rope_steer_params_t {
    /// Envelope byte size (`sizeof(lmnk_rope_steer_params_t)`, `offset 0..4`).
    pub struct_size: u32,
    /// Steering mode flags (`0` = soft-clamped `PhaseClamp`, `LMNK_ROPE_STEER_FLAG_UNCLAMPED`, `offset 4..8`).
    pub flags: u32,
    /// Attention head dimension (`d_head`, positive even, `offset 8..12`).
    pub head_dim: u32,
    /// Rotated dimensions per head (`0` defaults to `head_dim`, even, `offset 12..16`).
    pub rotary_dim: u32,
    /// Number of attention heads (`1..=64`, `offset 16..20`).
    pub num_heads: u32,
    /// RoPE channel pairing layout (`0` = Interleaved, `1` = HalfSplit, `offset 20`).
    pub rope_layout: u8,
    /// Explicit zero padding (`offset 21..24`).
    pub reserved0: [u8; 3],
    /// Soft PhaseClamp ceiling in radians (`0.0` uses `DEFAULT_PHASE_CLAMP_MAX_RAD`, `offset 24..28`).
    pub max_rad: f32,
    /// Batch token count (`v0.4.0` supports `1` only; `!= 1` returns `LMNK_ERR_NOT_IMPLEMENTED`, `offset 28..32`).
    pub num_tokens: u32,
    /// Stride in `f32` elements between consecutive tokens (`0` = dense `head_dim * num_heads`, `offset 32..36`).
    pub token_stride_elems: u32,
    /// Explicit alignment padding (`offset 36..40`).
    pub reserved1: u32,
    /// 32B-aligned mutable Query tensor (`head_dim * num_heads` `f32` elements, `offset 40..`).
    pub query_tensor: *mut f32,
    /// 16B-aligned per-head Q0.7 phase deltas (`num_heads` elements).
    pub phase_deltas: *const lmnk_q7_t,
}

impl Default for lmnk_rope_steer_params_t {
    #[inline(always)]
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            flags: 0,
            head_dim: 0,
            rotary_dim: 0,
            num_heads: 0,
            rope_layout: 0,
            reserved0: [0; 3],
            max_rad: 0.0,
            num_tokens: 1,
            token_stride_elems: 0,
            reserved1: 0,
            query_tensor: ptr::null_mut(),
            phase_deltas: ptr::null(),
        }
    }
}

/// Transient host call envelope for [`crate::runtime::abi::lmnk_rope_procrustes_q7`].
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct lmnk_rope_procrustes_params_t {
    /// Envelope byte size (`sizeof(lmnk_rope_procrustes_params_t)`, `offset 0..4`).
    pub struct_size: u32,
    /// Reserved flags (must be `0` in `v0.4.0`, `offset 4..8`).
    pub flags: u32,
    /// Attention head dimension (`offset 8..12`).
    pub head_dim: u32,
    /// Rotated dimensions per head (`0` defaults to `head_dim`, `offset 12..16`).
    pub rotary_dim: u32,
    /// Number of attention heads (`1..=64`, `offset 16..20`).
    pub num_heads: u32,
    /// RoPE channel pairing layout (`0` = Interleaved, `1` = HalfSplit, `offset 20`).
    pub rope_layout: u8,
    /// Explicit zero padding (`offset 21..24`).
    pub reserved0: [u8; 3],
    /// Batch token count (`v0.4.0` supports `1` only; `!= 1` returns `LMNK_ERR_NOT_IMPLEMENTED`, `offset 24..28`).
    pub num_tokens: u32,
    /// Stride in `f32` elements between consecutive tokens (`0` = dense, `offset 28..32`).
    pub token_stride_elems: u32,
    /// 32B-aligned reference Query tensor before steering (`offset 32..`).
    pub q_before: *const f32,
    /// 32B-aligned steered Query tensor after steering.
    pub q_after: *const f32,
    /// 16B-aligned output array of `num_heads` recovered Q0.7 phase deltas.
    pub q7_phase_out: *mut lmnk_q7_t,
}

impl Default for lmnk_rope_procrustes_params_t {
    #[inline(always)]
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            flags: 0,
            head_dim: 0,
            rotary_dim: 0,
            num_heads: 0,
            rope_layout: 0,
            reserved0: [0; 3],
            num_tokens: 1,
            token_stride_elems: 0,
            q_before: ptr::null(),
            q_after: ptr::null(),
            q7_phase_out: ptr::null_mut(),
        }
    }
}

/// Transient host call envelope for [`crate::runtime::abi::lmnk_rank1_eval`].
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct lmnk_rank1_eval_params_t {
    /// Envelope byte size (`sizeof(lmnk_rank1_eval_params_t)`, `offset 0..4`).
    pub struct_size: u32,
    /// Evaluation mode flags (`0` = overwrite `y_inout`, `LMNK_RANK1_FLAG_FUSED` = add-accumulate, `offset 4..8`).
    pub flags: u32,
    /// Input vector dimension (`offset 8..12`).
    pub dim_in: u32,
    /// Output vector dimension (`offset 12..16`).
    pub dim_out: u32,
    /// Total element count of the caller-owned `arena` buffer (`offset 16..20`).
    pub arena_len: u32,
    /// Batch token count (`v0.4.0` supports `1` only; `!= 1` returns `LMNK_ERR_NOT_IMPLEMENTED`, `offset 20..24`).
    pub num_tokens: u32,
    /// Stride in `f32` elements between consecutive tokens (`0` = dense, `offset 24..28`).
    pub token_stride_elems: u32,
    /// Explicit alignment padding (`offset 28..32`).
    pub reserved0: u32,
    /// 32B-aligned portable arena-indexed [`lmnk_rank1_descriptor_t`] (`offset 32..`).
    pub desc: *const lmnk_rank1_descriptor_t,
    /// Caller-owned `f32` arena slice referenced by `desc->u_span`, `desc->v_span`, and `desc->gain_offset`.
    pub arena: *const f32,
    /// 32B-aligned input vector (`dim_in` `f32` elements).
    pub x_in: *const f32,
    /// 32B-aligned output/accumulator vector (`dim_out` `f32` elements).
    pub y_inout: *mut f32,
}

impl Default for lmnk_rank1_eval_params_t {
    #[inline(always)]
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            flags: 0,
            dim_in: 0,
            dim_out: 0,
            arena_len: 0,
            num_tokens: 1,
            token_stride_elems: 0,
            reserved0: 0,
            desc: ptr::null(),
            arena: ptr::null(),
            x_in: ptr::null(),
            y_inout: ptr::null_mut(),
        }
    }
}

/// Transient host call envelope for [`crate::runtime::abi::lmnk_gateway_steer`].
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct lmnk_gateway_steer_params_t {
    /// Envelope byte size (`sizeof(lmnk_gateway_steer_params_t)`, `offset 0..4`).
    pub struct_size: u32,
    /// Reserved flags (must be `0` in `v0.4.0`, `offset 4..8`).
    pub flags: u32,
    /// Attention head dimension (`offset 8..12`).
    pub head_dim: u32,
    /// Rotated dimensions per head (`0` defaults to `head_dim`, `offset 12..16`).
    pub rotary_dim: u32,
    /// Number of attention heads (`1..=64`, `offset 16..20`).
    pub num_heads: u32,
    /// RoPE channel pairing layout (`0` = Interleaved, `1` = HalfSplit, `offset 20`).
    pub rope_layout: u8,
    /// Unresolved damping scale in U0.7 (`0` defaults to `128` = `1.0x`, `offset 21`).
    pub unresolved_damping_u7: u8,
    /// Explicit zero padding (`offset 22..24`).
    pub reserved0: [u8; 2],
    /// Objective-fact signal energy (`offset 24..28`).
    pub fact_energy: f32,
    /// Subjective-persona signal energy (`offset 28..32`).
    pub persona_energy: f32,
    /// Batch token count (`v0.4.0` supports `1` only; `!= 1` returns `LMNK_ERR_NOT_IMPLEMENTED`, `offset 32..36`).
    pub num_tokens: u32,
    /// Stride in `f32` elements between consecutive tokens (`0` = dense, `offset 36..40`).
    pub token_stride_elems: u32,
    /// 32B-aligned mutable Query tensor (`offset 40..`).
    pub query_tensor: *mut f32,
    /// 16B-aligned per-head objective-fact Q0.7 phases (`num_heads` elements).
    pub fact_phases: *const lmnk_q7_t,
    /// 16B-aligned per-head subjective-persona Q0.7 phases (`num_heads` elements).
    pub persona_phases: *const lmnk_q7_t,
    /// 64B-aligned mutable stigmergic pointer trail.
    pub trail_inout: *mut lmnk_pointer_trail_t,
    /// Optional 64B-aligned output for the synthesized [`lmnk_gateway_frame_t`] (`NULL` allowed).
    pub frame_out: *mut lmnk_gateway_frame_t,
    /// Optional 16B-aligned output for the [`lmnk_causal_report_t`] (`NULL` allowed).
    pub report_out: *mut lmnk_causal_report_t,
}

impl Default for lmnk_gateway_steer_params_t {
    #[inline(always)]
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            flags: 0,
            head_dim: 0,
            rotary_dim: 0,
            num_heads: 0,
            rope_layout: 0,
            unresolved_damping_u7: 128,
            reserved0: [0; 2],
            fact_energy: 0.0,
            persona_energy: 0.0,
            num_tokens: 1,
            token_stride_elems: 0,
            query_tensor: ptr::null_mut(),
            fact_phases: ptr::null(),
            persona_phases: ptr::null(),
            trail_inout: ptr::null_mut(),
            frame_out: ptr::null_mut(),
            report_out: ptr::null_mut(),
        }
    }
}

/// Transient host section input descriptor consumed by [`lmnk_container_write_params_t`].
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct lmnk_section_input_t {
    /// Envelope byte size (`sizeof(lmnk_section_input_t)`, `offset 0..4`).
    pub struct_size: u32,
    /// Section kind ID (`LMNK_SECTION_*` or custom `u16`, `offset 4..6`).
    pub kind: u16,
    /// Per-section flags (must be `0` in `v0.4.0`, `offset 6..8`).
    pub flags: u16,
    /// Byte length of `payload` (must be `> 0`, `offset 8..12`).
    pub payload_len: u32,
    /// Explicit alignment padding (`offset 12..16`).
    pub reserved0: u32,
    /// Immutable pointer to `payload_len` payload bytes (`offset 16..`).
    pub payload: *const u8,
}

impl Default for lmnk_section_input_t {
    #[inline(always)]
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            kind: 0,
            flags: 0,
            payload_len: 0,
            reserved0: 0,
            payload: ptr::null(),
        }
    }
}

/// Transient host call envelope for [`crate::runtime::abi::lmnk_container_write`].
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct lmnk_container_write_params_t {
    /// Envelope byte size (`sizeof(lmnk_container_write_params_t)`, `offset 0..4`).
    pub struct_size: u32,
    /// Container classification bitflags (`LMNK_CONTAINER_FLAG_*`, `offset 4..8`).
    pub container_flags: u32,
    /// Unix epoch timestamp in seconds stamped into the container header (`offset 8..16`).
    pub timestamp_epoch_sec: u64,
    /// Number of elements in `sections` (`0..=32`, `offset 16..20`).
    pub section_count: u32,
    /// Explicit alignment padding (`offset 20..24`).
    pub reserved0: u32,
    /// Array of `section_count` [`lmnk_section_input_t`] descriptors (`NULL` permitted iff `section_count == 0`, `offset 24..`).
    pub sections: *const lmnk_section_input_t,
    /// Destination byte buffer (`NULL` requests two-pass required byte-length query).
    pub out_bytes: *mut u8,
    /// Capacity of `out_bytes` in bytes.
    pub out_capacity: usize,
    /// Non-null output receiving required or written container byte length.
    pub written_bytes_out: *mut usize,
}

impl Default for lmnk_container_write_params_t {
    #[inline(always)]
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            container_flags: 0,
            timestamp_epoch_sec: 0,
            section_count: 0,
            reserved0: 0,
            sections: ptr::null(),
            out_bytes: ptr::null_mut(),
            out_capacity: 0,
            written_bytes_out: ptr::null_mut(),
        }
    }
}

/// Transient host zero-copy section view emitted by [`crate::runtime::abi::lmnk_container_find_section`].
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct lmnk_container_section_view_t {
    /// Envelope byte size (`sizeof(lmnk_container_section_view_t)`, `offset 0..4`).
    pub struct_size: u32,
    /// Matched section kind ID (`offset 4..6`).
    pub kind: u16,
    /// Section flags (`offset 6..8`).
    pub flags: u16,
    /// Byte offset of section payload from container start (`offset 8..12`).
    pub offset: u32,
    /// Byte length of section payload (`offset 12..16`).
    pub length: u32,
    /// Verified IEEE 802.3 CRC32 of section payload (`offset 16..20`).
    pub payload_crc32: u32,
    /// Explicit alignment padding (`offset 20..24`).
    pub reserved0: u32,
    /// Zero-copy pointer into the caller's container buffer (`bytes + offset`, `offset 24..`).
    pub payload_ptr: *const u8,
}

impl Default for lmnk_container_section_view_t {
    #[inline(always)]
    fn default() -> Self {
        Self {
            struct_size: size_of::<Self>() as u32,
            kind: 0,
            flags: 0,
            offset: 0,
            length: 0,
            payload_crc32: 0,
            reserved0: 0,
            payload_ptr: ptr::null(),
        }
    }
}

// Compile-time layout assertions enforcing Portable Device-Ready POD guarantees
// and 1:1 C11/C++20 header parity for v0.4.0 (`0x0000_0400`).
const _: () = {
    assert!(LMNK_ABI_VERSION_MAJOR == 0);
    assert!(LMNK_ABI_VERSION_HEX == 0x0000_0400);
    assert!(!LMNK_ABI_VERSION_CSTR.to_bytes().is_empty());
    assert!(LMNK_CONTAINER_MAGIC == 0x4B4E_4D4C);
    assert!(LMNK_TOPOLOGY_MAGIC == 0x504F_544C);
    assert!(LMNK_PHASE_ANCHOR_MAGIC == 0x524B_4E4C);
    assert!(LMNK_PHASE_ANCHOR_VERSION == 0x0000_0400);

    assert!(size_of::<lmnk_status_t>() == 4);
    assert!(size_of::<lmnk_q7_t>() == 1);
    assert!(size_of::<lmnk_tagged_ptr_t>() == 8 && align_of::<lmnk_tagged_ptr_t>() == 8);
    assert!(size_of::<lmnk_attribution_tag_t>() == 1);
    assert!(size_of::<lmnk_arena_span_t>() == 8 && align_of::<lmnk_arena_span_t>() == 4);
    assert!(offset_of!(lmnk_arena_span_t, offset) == 0);
    assert!(offset_of!(lmnk_arena_span_t, len) == 4);

    assert!(size_of::<lmnk_config_t>() == 64 && align_of::<lmnk_config_t>() == 64);
    assert!(offset_of!(lmnk_config_t, struct_size) == 0);
    assert!(offset_of!(lmnk_config_t, head_dim) == 4);
    assert!(offset_of!(lmnk_config_t, rotary_dim) == 40);
    assert!(offset_of!(lmnk_config_t, execution_flags) == 44);
    assert!(offset_of!(lmnk_config_t, guard_hold_tokens) == 48);
    assert!(offset_of!(lmnk_config_t, max_observe_stride) == 49);
    assert!(offset_of!(lmnk_config_t, reserved) == 50);

    assert!(
        size_of::<lmnk_rank1_descriptor_t>() == 32 && align_of::<lmnk_rank1_descriptor_t>() == 32
    );
    assert!(offset_of!(lmnk_rank1_descriptor_t, layer_idx) == 0);
    assert!(offset_of!(lmnk_rank1_descriptor_t, rank) == 4);
    assert!(offset_of!(lmnk_rank1_descriptor_t, u_span) == 8);
    assert!(offset_of!(lmnk_rank1_descriptor_t, v_span) == 16);
    assert!(offset_of!(lmnk_rank1_descriptor_t, gain_offset) == 24);
    assert!(offset_of!(lmnk_rank1_descriptor_t, scale_factor) == 28);

    assert!(size_of::<lmnk_abi_info_t>() == 64 && align_of::<lmnk_abi_info_t>() == 8);
    assert!(offset_of!(lmnk_abi_info_t, struct_size) == 0);
    assert!(offset_of!(lmnk_abi_info_t, capability_flags) == 24);
    assert!(offset_of!(lmnk_abi_info_t, reserved) == 32);

    assert!(size_of::<lmnk_runtime_policy_t>() == 64 && align_of::<lmnk_runtime_policy_t>() == 8);
    assert!(offset_of!(lmnk_runtime_policy_t, struct_size) == 0);
    assert!(offset_of!(lmnk_runtime_policy_t, baseline_pinned_compute_mask) == 8);
    assert!(offset_of!(lmnk_runtime_policy_t, baseline_pressure_q15) == 40);
    assert!(offset_of!(lmnk_runtime_policy_t, reserved) == 45);

    assert!(
        size_of::<lmnk_turn_begin_params_t>() == 16 && align_of::<lmnk_turn_begin_params_t>() == 4
    );
    assert!(
        size_of::<lmnk_decode_step_params_t>() == 48
            && align_of::<lmnk_decode_step_params_t>() == 16
    );
    assert!(offset_of!(lmnk_decode_step_params_t, flags) == 4);
    assert!(offset_of!(lmnk_decode_step_params_t, reserved) == 8);
    assert!(offset_of!(lmnk_decode_step_params_t, decode_input) == 16);

    assert!(
        size_of::<lmnk_decode_step_report_t>() == 64
            && align_of::<lmnk_decode_step_report_t>() == 8
    );
    assert!(offset_of!(lmnk_decode_step_report_t, dispatch) == 8);
    assert!(offset_of!(lmnk_decode_step_report_t, reserved) == 48);

    assert!(
        size_of::<lmnk_runtime_state_view_t>() == 64
            && align_of::<lmnk_runtime_state_view_t>() == 8
    );
    assert!(offset_of!(lmnk_runtime_state_view_t, baseline_pinned_compute_mask) == 8);
    assert!(offset_of!(lmnk_runtime_state_view_t, active_compute_mask) == 32);
    assert!(offset_of!(lmnk_runtime_state_view_t, envelope_contraction_q7) == 46);
    assert!(offset_of!(lmnk_runtime_state_view_t, envelope_breach_mask) == 47);
    assert!(offset_of!(lmnk_runtime_state_view_t, envelope_violation_epochs) == 48);
    assert!(offset_of!(lmnk_runtime_state_view_t, envelope_baseline_pressure_q15) == 52);
    assert!(offset_of!(lmnk_runtime_state_view_t, envelope_lock_flags) == 54);
    assert!(offset_of!(lmnk_runtime_state_view_t, envelope_ema_resume_dwell) == 55);
    assert!(offset_of!(lmnk_runtime_state_view_t, reserved) == 56);

    assert!(
        size_of::<lmnk_system_pressure_reading_t>() == 8
            && align_of::<lmnk_system_pressure_reading_t>() == 2
    );
    assert!(
        size_of::<lmnk_compute_dispatch_mask_t>() == 40
            && align_of::<lmnk_compute_dispatch_mask_t>() == 8
    );
    assert!(
        size_of::<lmnk_spine_decode_input_t>() == 32
            && align_of::<lmnk_spine_decode_input_t>() == 16
    );

    assert!(
        size_of::<lmnk_boundary_signal_t>() == 32 && align_of::<lmnk_boundary_signal_t>() == 32
    );
    assert!(offset_of!(lmnk_boundary_signal_t, reserved) == 26);
    assert!(offset_of!(lmnk_boundary_signal_t, epoch_id) == 28);
    assert!(
        size_of::<lmnk_envelope_bounds_t>() == 32 && align_of::<lmnk_envelope_bounds_t>() == 32
    );
    assert!(
        size_of::<lmnk_envelope_feedback_t>() == 16 && align_of::<lmnk_envelope_feedback_t>() == 16
    );
    assert!(size_of::<lmnk_envelope_state_t>() == 64 && align_of::<lmnk_envelope_state_t>() == 64);
    assert!(
        size_of::<lmnk_envelope_step_params_t>() == 96
            && align_of::<lmnk_envelope_step_params_t>() == 32
    );
    assert!(offset_of!(lmnk_envelope_step_params_t, flags) == 4);
    assert!(offset_of!(lmnk_envelope_step_params_t, bounds) == 32);
    assert!(offset_of!(lmnk_envelope_step_params_t, signal) == 64);
    assert!(
        size_of::<lmnk_boundary_emit_params_t>() == 16
            && align_of::<lmnk_boundary_emit_params_t>() == 4
    );
    assert!(offset_of!(lmnk_boundary_emit_params_t, pressure_q15) == 8);
    assert!(offset_of!(lmnk_boundary_emit_params_t, chatter_q7) == 10);
    assert!(offset_of!(lmnk_boundary_emit_params_t, epoch_id) == 12);

    assert!(
        size_of::<lmnk_lacuna_eval_params_t>() == 24
            && align_of::<lmnk_lacuna_eval_params_t>() == 4
    );
    assert!(offset_of!(lmnk_lacuna_eval_params_t, other_density_q7) == 8);
    assert!(offset_of!(lmnk_lacuna_eval_params_t, noise_tolerance_q7) == 16);

    assert!(
        size_of::<lmnk_container_verify_report_t>() == 40
            && align_of::<lmnk_container_verify_report_t>() == 8
    );
    assert!(offset_of!(lmnk_container_verify_report_t, total_size) == 16);
    assert!(offset_of!(lmnk_container_verify_report_t, section_count) == 32);

    assert!(offset_of!(lmnk_rope_steer_params_t, query_tensor) == 40);
    assert!(offset_of!(lmnk_rope_procrustes_params_t, q_before) == 32);
    assert!(offset_of!(lmnk_rank1_eval_params_t, desc) == 32);
    assert!(offset_of!(lmnk_gateway_steer_params_t, query_tensor) == 40);
    assert!(offset_of!(lmnk_section_input_t, payload) == 16);
    assert!(offset_of!(lmnk_container_write_params_t, sections) == 24);
    assert!(offset_of!(lmnk_container_section_view_t, payload_ptr) == 24);

    assert!(size_of::<lmnk_pointer_trail_t>() == 128 && align_of::<lmnk_pointer_trail_t>() == 64);
    assert!(size_of::<lmnk_gateway_frame_t>() == 128 && align_of::<lmnk_gateway_frame_t>() == 64);
    assert!(offset_of!(lmnk_gateway_frame_t, root_cause_code) == 48);
    assert!(size_of::<lmnk_causal_report_t>() == 16 && align_of::<lmnk_causal_report_t>() == 16);
    assert!(offset_of!(lmnk_causal_report_t, root_cause_code) == 12);
    assert!(
        size_of::<lmnk_phase_anchor_record_t>() == 128
            && align_of::<lmnk_phase_anchor_record_t>() == 64
    );
    assert!(offset_of!(lmnk_phase_anchor_record_t, root_cause_code) == 59);
    assert!(
        size_of::<lmnk_liminal_ego_state_t>() == 64 && align_of::<lmnk_liminal_ego_state_t>() == 64
    );
    assert!(offset_of!(lmnk_liminal_ego_state_t, basin_resistance_q7) == 40);
    assert!(offset_of!(lmnk_liminal_ego_state_t, escape_floor_q7) == 41);
    assert!(offset_of!(lmnk_topology_snapshot_t, spawned_anchors) == 31);
    assert!(CORE_TOPOLOGY.verify_crc32());
    assert!(PhaseAnchorRecord::new_empty(0, ATTRACTOR_SLOT_PERSONA, 0).verify_integrity());
    assert!(PhaseAnchorRecord::new_empty(0, ATTRACTOR_SLOT_KNOWLEDGE, 0).verify_integrity());
};
