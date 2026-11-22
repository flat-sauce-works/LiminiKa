// name: src/adapter/src/lib.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model connection, RoPE geometry mapping, and optimization adapter over [`liminika_core`].
//!
//! Keeps [`liminika_core`] strictly `#![no_std]` and model-agnostic while providing a
//! unified, zero-external-dependency bridge for external LLM runtimes (Candle, `llama.cpp`,
//! and CLI evaluation harnesses).

use liminika_core::abi::{
    lmnk_entropy_state_t, lmnk_ledger_slot_t, lmnk_pointer_trail_t, lmnk_q7_t,
};
use liminika_core::drivers::{
    EntropyController, LiminalEgoPreset, LiminalEgoState, DISTORTION_DELTA_FLAG_ACTIVE,
    DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID, EGO_TRIGGER_BASIN_CAPTURE,
    EGO_TRIGGER_BOUNDARY_STAGNATION, EGO_TRIGGER_INTERNAL_CONFLICT, EGO_TRIGGER_LAYER_DISTORTION,
};
use liminika_core::foundation::{
    load_q7_words_64, swar_q7_circular_add_x8, AlignedSlice16, AlignedSlice32, AlignedSliceMut16,
    AlignedSliceMut32, PhaseSteeringGeometryExt, RopeGeometryDescriptor, RopePairLayout,
    RopeSteeringTarget, ACTION_MODE_PHASE_REPULSE, SWAR_SPLAT_BYTE_64,
};
use liminika_core::liminal::{
    BitCut8, EpistemicRegime, LiminalThresholdState, LmnkResult, MAX_ATTENTION_HEADS,
    NUM_STIGMERGIC_GRANULES,
};
use liminika_core::runtime::{
    apply_latent_horizon_slots, AdaptiveThresholdPolicy, CausalProvenanceReport,
    CausalProvenanceSignal, CoreLayerEvalContext, CoreStateStack, CoreTelemetryBank,
    CoreTelemetrySnapshot, DefaultResidualSpilloverSlot, EpistemicHorizonReading, GatewayFrame,
    GatewayPhaseSteeringExt, GuardPhaseAnchor, HorizonDwellState, HorizonEscalationProvider,
    HorizonEscalationRequest, HorizonTriggerInput, LatentHorizonSlot, LayerDistortionReport,
    LiminalBoundaryDelta, LiminalTopologyTelemetrySnapshot, LmnkFourSectionView,
    ManifoldGatewaySignal, ManifoldProbeBaseline, ObjectiveFact, SlowLoopTelemetryAdjustment,
    StigmergicThresholdPolicy, SubjectivePersona, SubstrateGatewayExt, SystemConstraint,
    TelemetryCapturePolicy, CAUSAL_AXIS_CONTRADICTION_GUARD, CAUSAL_AXIS_OBJECTIVE_FACT,
    CAUSAL_AXIS_SUBJECTIVE_PERSONA, ROOT_CAUSE_NOMINAL, ROOT_CAUSE_SYSTEM_QOS_THROTTLE,
};
use liminika_core::stigmergy::{
    evaluate_stigmergic_overlap_and_discovery_words_64, CognitiveCircuitActivation,
    CognitiveCircuitBankOps, CognitiveTrajectoryLedger, DissipativeOverflowReport,
    OverlapDiscoveryReport, PhaseAnchorRecord, PhaseDeltaSlice, PhaseSteeringEngine,
    RelativeAttractorNode, ResidualAdapterEngine, ResidualScale, SandboxLifecycleEvent,
    StigmergicActionDecision, StigmergicFragmentEnsembleReport, StigmergicGranuleState,
    StigmergicReconstitutionReport, StigmergicScaleGranularity, StigmergicSubstrate,
    StigmergicThresholds, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA,
    ATTRACTOR_SLOT_REPULSION, LEDGER_FLAG_COGNITIVE_TRAJECTORY, WORKBENCH_LANE_FLAG_REPULSION,
};

pub mod pooling;
pub mod reflection;
pub mod stratum;

pub use liminika_core::drivers::{
    AnonymousComputePool, ComputeDispatchMask, ComputePoolConfig, ComputePressureRegime,
    DistortionDelta, DistortionTension, DistortionTensionStateCode, EgoDifferentialNet,
    EgoIntentAxes, SystemPressureReading, SystemSpineState, TokenBypassObservation,
    SPINE_BYPASS_VETO_GUARD_ACTIVE, SPINE_BYPASS_VETO_NONE, SPINE_BYPASS_VETO_PINNED_LOCK,
    SPINE_BYPASS_VETO_REPULSION_ACTIVE,
};
pub use liminika_core::liminal::{
    evaluate_lacuna_partition_branchless, AttributionTag, DriverKind, LacunaPartition8,
    PrimitiveKind, SCOPE_REGION_EXTERIOR, SCOPE_REGION_INTERIOR, SCOPE_REGION_SELF_REF,
};
pub use liminika_core::stigmergy::{
    CorePrimitiveState, WorkbenchLanePlacement, WorkbenchPreset, WorkbenchSlotSpec,
};
pub use pooling::*;
pub use reflection::*;
pub use stratum::*;

pub mod corpus_recall;
pub mod ego_intent;
pub mod model_adapter;
pub mod timeline;

pub use corpus_recall::*;
pub use ego_intent::*;
pub use model_adapter::*;
pub use timeline::*;

#[cfg(test)]
mod tests;
