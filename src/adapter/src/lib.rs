// name: src/adapter/src/lib.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model connection, RoPE geometry mapping, and optimization adapter over [`liminika_core`].
//!
//! Keeps [`liminika_core`] strictly `#![no_std]` and model-agnostic while providing a
//! unified, zero-external-dependency bridge for external LLM runtimes (Candle, `llama.cpp`,
//! and CLI evaluation harnesses).

use liminika_core::abi::{gcso_edbc_state_t, gcso_ledger_slot_t, gcso_pointer_trail_t, gcso_q7_t};
use liminika_core::drivers::{
    EntropyController, LayerDistortionReport, LiminalEgoField, LiminalEgoPreset,
    DISTORTION_DELTA_FLAG_ACTIVE, DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID,
};
use liminika_core::foundation::{
    load_q7_words_64, swar_q7_circular_add_x8, AlignedSlice16, AlignedSlice32, AlignedSliceMut16,
    AlignedSliceMut32, PhaseSteeringEngine, PhaseSteeringGeometryExt, ResidualAdapterEngine,
    RopeGeometryDescriptor, RopePairLayout, RopeSteeringTarget, SWAR_SPLAT_BYTE_64,
};
use liminika_core::liminal::{
    EpistemicRegime, GcsoResult, LiminalBoundaryDelta, LiminalThresholdField,
    ACTION_MODE_PHASE_REPULSE, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA,
    ATTRACTOR_SLOT_REPULSION, CAUSAL_AXIS_CONTRADICTION_GUARD, CAUSAL_AXIS_OBJECTIVE_FACT,
    CAUSAL_AXIS_SUBJECTIVE_PERSONA, EGO_TRIGGER_BASIN_CAPTURE, EGO_TRIGGER_BOUNDARY_STAGNATION,
    EGO_TRIGGER_INTERNAL_CONFLICT, EGO_TRIGGER_LAYER_DISTORTION, MAX_ATTENTION_HEADS,
    NUM_STIGMERGIC_GRANULES, ROOT_CAUSE_NOMINAL, ROOT_CAUSE_SYSTEM_QOS_THROTTLE,
    WORKBENCH_LANE_FLAG_REPULSION,
};
use liminika_core::runtime::{
    apply_latent_horizon_slots, ActionHubGatewayExt, AdaptiveThresholdPolicy,
    AutonomousCoreLayerStack, CausalProvenanceReport, CausalProvenanceSignal, CoreLayerEvalContext,
    CoreTelemetryBank, CoreTelemetrySnapshot, DefaultResidualSpilloverSlot,
    EpistemicHorizonReading, GatewayFrame, GatewayPhaseSteeringExt, GcsoFourSectionView,
    GuardPhaseAnchor, HorizonDwellState, HorizonEscalationProvider, HorizonEscalationRequest,
    HorizonTriggerInput, LatentHorizonSlot, LiminalTopologyTelemetrySnapshot,
    ManifoldGatewaySignal, ManifoldProbeBaseline, ObjectiveFact, SlowLoopTelemetryAdjustment,
    StigmergicThresholdPolicy, SubjectivePersona, SystemConstraint, TelemetryCapturePolicy,
};
use liminika_core::stigmergy::{
    evaluate_stigmergic_overlap_and_serendipity_words_64, ActionHubEngine,
    CognitiveCircuitActivation, CognitiveCircuitBankOps, CognitiveCircuitPlasticity,
    CognitiveTrajectoryLedger, CrystallizedAttractorRecord, DissipativeOverflowReport,
    RelativeAttractorNode, StigmergicActionDecision, StigmergicFragmentEnsembleReport,
    StigmergicGranuleField, StigmergicOverlapSerendipityReport, StigmergicReconstitutionReport,
    StigmergicScaleGranularity, StigmergicThresholds, LEDGER_FLAG_COGNITIVE_TRAJECTORY,
};

pub mod pooling;
pub mod reflection;
pub mod stratum;

pub use liminika_core::drivers::{
    AnonymousComputePool, ComputeDispatchMask, ComputePoolConfig, ComputePressureRegime,
    DistortionDelta, DistortionTension, DistortionTensionStateCode, EgoDifferentialNet,
    EgoIntentAxes, SystemPressureReading, SystemSpineState, TokenBypassObservation,
};
pub use liminika_core::liminal::{
    evaluate_epistemic_partition_branchless, AttributionTag, DriverKind, EpistemicPartition,
    SubstrateKind, SCOPE_REGION_KNOWN_UNKNOWN, SCOPE_REGION_SELF_EXCLUSIVE,
    SCOPE_REGION_SHARED_CORE,
};
pub use liminika_core::stigmergy::{
    CognitiveSubstrate, WorkbenchLanePlacement, WorkbenchPreset, WorkbenchSlotSpec,
};
pub use pooling::*;
pub use reflection::*;
pub use stratum::*;

pub mod corpus_recall;
pub mod model_adapter;
pub mod timeline;

pub use corpus_recall::*;
pub use model_adapter::*;
pub use timeline::*;

#[cfg(test)]
mod tests;
