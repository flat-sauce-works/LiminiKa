// name: src/adapter/src/lib.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model connection, RoPE geometry mapping, and optimization adapter over [`liminika_core`].
//!
//! Keeps [`liminika_core`] strictly `#![no_std]` and model-agnostic while providing a
//! unified, zero-external-dependency bridge for external LLM runtimes (Candle, `llama.cpp`,
//! and CLI evaluation harnesses).

use liminika_core::abi::{gcso_edbc_state_t, gcso_ledger_slot_t, gcso_pointer_trail_t, gcso_q7_t};
use liminika_core::action_hub::{ActionHubEngine, StigmergicActionDecision, StigmergicThresholds};
use liminika_core::crystal::CrystallizedAttractorRecord;
use liminika_core::entropy_control::EntropyController;
use liminika_core::phase_steering::PhaseSteeringEngine;
use liminika_core::residual_adapter::ResidualAdapterEngine;
use liminika_core::storage::GcsoFourSectionView;
use liminika_core::traits::aligned::{
    AlignedSlice16, AlignedSlice32, AlignedSliceMut16, AlignedSliceMut32,
};
use liminika_core::traits::gateway::{
    ActionHubGatewayExt, AdaptiveThresholdPolicy, AutonomousCoreLayerStack, CausalProvenanceReport,
    CausalProvenanceSignal, CognitiveCircuitActivation, CognitiveCircuitBankOps,
    CognitiveCircuitPlasticity, CognitiveTrajectoryLedger, DissipativeOverflowReport,
    EgoIntentAxes, EpistemicHorizonReading, GatewayFrame, GatewayPhaseSteeringExt,
    GatewayRank1SteeringExt, GuardPhaseAnchor, LiminalApertureField, LiminalBoundaryDelta,
    LiminalEgoField, LiminalEgoPreset, LiminalTopologyTelemetrySnapshot, ManifoldGatewaySignal,
    ManifoldProbeBaseline, ObjectiveFact, StigmergicFragmentEnsembleReport, StigmergicGranuleField,
    StigmergicScaleGranularity, StigmergicThresholdPolicy, StigmergicVennSerendipityReport,
    SubjectivePersona, SystemConstraint, CAUSAL_AXIS_CONTRADICTION_GUARD,
    CAUSAL_AXIS_OBJECTIVE_FACT, CAUSAL_AXIS_SUBJECTIVE_PERSONA, EGO_TRIGGER_BOUNDARY_STAGNATION,
};
use liminika_core::traits::primitives::{
    ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA, ATTRACTOR_SLOT_REPULSION, MAX_ATTENTION_HEADS,
};
use liminika_core::traits::rope::{
    PhaseSteeringGeometryExt, RopeGeometryDescriptor, RopePairLayout, RopeSteeringTarget,
};
use liminika_core::traits::telemetry::{
    StigmergicReconstitutionReport, LEDGER_FLAG_COGNITIVE_TRAJECTORY, NUM_STIGMERGIC_GRANULES,
};
use liminika_core::traits::Rank1ResidualAdapter;

pub mod pooling;
pub mod reflection;
pub mod stratum;

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
