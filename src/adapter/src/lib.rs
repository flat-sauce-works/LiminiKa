// name: src/adapter/src/lib.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model connection, RoPE geometry mapping, and optimization adapter over [`liminika_core`].
//!
//! Keeps [`liminika_core`] strictly `#![no_std]` and model-agnostic while providing a
//! unified, zero-external-dependency bridge for external LLM runtimes (Candle, `llama.cpp`,
//! and CLI evaluation harnesses).

use liminika_core::abi::{lmnk_entropy_state_t, lmnk_pointer_trail_t, lmnk_q7_t};
use liminika_core::drivers::{
    DISTORTION_DELTA_FLAG_ACTIVE, DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID,
    EGO_TRIGGER_BASIN_CAPTURE, EGO_TRIGGER_BOUNDARY_STAGNATION, EGO_TRIGGER_INTERNAL_CONFLICT,
    EGO_TRIGGER_LAYER_DISTORTION, EntropyController, LiminalEgoPreset, LiminalEgoState,
};
use liminika_core::foundation::{
    ACTION_MODE_PHASE_REPULSE, AlignedSlice16, AlignedSlice32, AlignedSliceMut16,
    AlignedSliceMut32, PhaseSteeringGeometryExt, RopeGeometryDescriptor, RopePairLayout,
    RopeSteeringTarget, SWAR_SPLAT_BYTE_64, load_q7_words_64, swar_q7_circular_add_x8,
};
use liminika_core::liminal::{
    BitCut8, EpistemicRegime, LMNK_ABI_VERSION_STRING, LMNK_ERR_ALIGNMENT,
    LMNK_ERR_TOPOLOGY_MISMATCH, LiminalThresholdState, LmnkResult, LmnkStatus, MAX_ATTENTION_HEADS,
    NUM_STIGMERGIC_GRANULES,
};
use liminika_core::runtime::{
    AdaptiveThresholdPolicy, CAUSAL_AXIS_CONTRADICTION_GUARD, CAUSAL_AXIS_OBJECTIVE_FACT,
    CAUSAL_AXIS_SUBJECTIVE_PERSONA, CORE_TOPOLOGY, CausalProvenanceReport, CausalProvenanceSignal,
    Config, CoreStateStack, CoreTelemetryBank, CoreTelemetrySnapshot, DefaultResidualSpilloverSlot,
    EpistemicHorizonReading, GatewayFrame, GatewayPhaseSteeringExt, GuardPhaseAnchor,
    HorizonDwellState, HorizonEscalationProvider, HorizonEscalationRequest, HorizonTriggerInput,
    LMNK_CONTAINER_FLAG_COMPILED_CACHE, LMNK_CONTAINER_FLAG_PORTABLE_PACKAGE,
    LMNK_CONTAINER_FLAG_RUNTIME_SNAPSHOT, LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES,
    LMNK_SECTION_BLOOM_FILTER, LMNK_SECTION_CAUSAL_PROVENANCE, LMNK_SECTION_LEDGER_SLOT,
    LMNK_SECTION_LOGICAL_PFRAMES, LMNK_SECTION_PHASE_ANCHORS, LMNK_SECTION_POINTER_TRAILS,
    LMNK_SECTION_RECALL_METADATA, LMNK_SECTION_RUNTIME_SNAPSHOT, LatentHorizonSlot,
    LayerDistortionReport, LiminalBoundaryDelta, LiminalTopologyTelemetrySnapshot,
    LmnkContainerHeader, LmnkContainerView, LmnkSectionInput, ManifoldGatewaySignal,
    ManifoldProbeBaseline, ObjectiveFact, ROOT_CAUSE_NOMINAL, ROOT_CAUSE_SYSTEM_QOS_THROTTLE,
    RuntimeContext, SlowLoopTelemetryAdjustment, StigmergicThresholdPolicy, SubjectivePersona,
    SubstrateGatewayExt, SystemConstraint, TelemetryCapturePolicy, apply_latent_horizon_slots,
    required_container_byte_len, write_container_sections_to_bytes,
};
use liminika_core::stigmergy::{
    ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA, ATTRACTOR_SLOT_REPULSION,
    CognitiveCircuitActivation, CognitiveCircuitBankOps, CognitiveTrajectoryLedger,
    DissipativeOverflowReport, LEDGER_FLAG_COGNITIVE_TRAJECTORY, OverlapDiscoveryReport,
    PhaseAnchorRecord, PhaseDeltaSlice, RelativeAttractorNode, ResidualAdapterEngine,
    ResidualScale, SandboxLifecycleEvent, StigmergicActionDecision,
    StigmergicFragmentEnsembleReport, StigmergicGranuleState, StigmergicReconstitutionReport,
    StigmergicScaleGranularity, StigmergicSubstrate, StigmergicThresholds,
    WORKBENCH_LANE_FLAG_REPULSION, evaluate_stigmergic_overlap_and_discovery_words_64,
};

pub mod host;
pub mod pooling;
pub mod reflection;
pub mod stratum;

pub use host::*;
pub use liminika_core::drivers::{
    AnonymousComputePool, ComputeDispatchMask, ComputePoolConfig, ComputePressureRegime,
    DistortionDelta, DistortionTension, DistortionTensionStateCode, EgoDifferentialNet,
    EgoIntentAxes, HOST_IMMUTABLE_PINNED_COMPUTE_MASK, SPINE_BYPASS_VETO_GUARD_ACTIVE,
    SPINE_BYPASS_VETO_KNOWN_MASK, SPINE_BYPASS_VETO_NONE, SPINE_BYPASS_VETO_PINNED_LOCK,
    SPINE_BYPASS_VETO_REPULSION_ACTIVE, SPINE_CONTROL_FLAG_PRE_BYPASS_ENABLED,
    SPINE_CONTROL_FLAG_PREV_RESIDENCY_VALID, SPINE_CONTROL_FLAG_PREV_WAS_ABSTRACT,
    SPINE_CONTROL_FLAG_TRANSIENT_GUARD_RELAXED, SPINE_CONTROL_FLAG_TURN_GUARD_ACTIVE,
    SPINE_CONTROL_KNOWN_FLAGS, SPINE_DECODE_INPUT_FLAG_GREEDY_SAMPLING,
    SPINE_DECODE_INPUT_FLAG_PRE_BYPASS_ENABLED, SPINE_DECODE_INPUT_FLAG_REPULSION_ACTIVE,
    SPINE_MIN_DYNAMIC_OBSERVE_STRIDE, SPINE_SENTINEL_AUTO_U8, SPINE_STEP_EVENT_BOUNDARY_CHURN,
    SPINE_STEP_EVENT_GUARD_REESCALATED, SPINE_STEP_EVENT_INSTABILITY_RECOVERY,
    SPINE_STEP_EVENT_LOGITS_OBSERVE_SKIPPED, SPINE_STEP_EVENT_NONE,
    SPINE_STEP_EVENT_TRANSIENT_GUARD_RELAXED, SpineDecodeInput, SpineDecodeSignal,
    SpineLogitsObserveMode, SpineStepPlan, SystemPressureReading, SystemSpineState,
    TokenBypassObservation,
};
pub use liminika_core::liminal::{
    AttributionTag, DriverKind, Lacuna, LacunaPartition8, PrimitiveKind, SCOPE_REGION_EXTERIOR,
    SCOPE_REGION_INTERIOR, SCOPE_REGION_SELF_REF, evaluate_lacuna_partition_branchless,
};
pub use liminika_core::stigmergy::{
    CorePrimitiveState, WorkbenchLanePlacement, WorkbenchPreset, WorkbenchSlotSpec,
};
pub use pooling::*;
pub use reflection::*;
pub use stratum::*;

pub mod corpus_recall;
mod corpus_recall_cache;
pub mod ego_intent;
pub mod model_adapter;
pub mod package;
pub mod timeline;
pub mod turn_orchestrator;

pub use corpus_recall::*;
pub use ego_intent::*;
pub use model_adapter::*;
pub use package::*;
pub use timeline::*;
pub use turn_orchestrator::*;

#[cfg(test)]
mod tests;
