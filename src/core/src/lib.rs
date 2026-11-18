// name: src/core/src/lib.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core execution engine library for the Geometric Cellular Sheaf Orchestrator (GCSO).
//!
//! This crate serves as the primary low-level runtime engine for `LiminiKa`, implementing the
//! Geometric Cellular Sheaf Orchestrator (GCSO) architecture targeting 2–4 GB VRAM environments.
//!
//! # Architecture: Two-Layer Hybrid Trait Topology (`Liminal` + `ActionHub` + Gateway)
//! - **`Liminal` ([`crate::traits::Liminal`])**: Root epistemic supertrait defining the
//!   unknowable exterior beyond any trait's cognitive aperture (`!APERTURE_MASK`), gating
//!   out-of-horizon steering cleanly to zero without data-dependent branches.
//! - **`ActionHub` (Core Sidecar State & Capability Ledger, [`crate::traits::ActionHub`])**: The
//!   widest-aperture composite ledger (`HUB_APERTURE_MASK = 0x0000_FFFF`) combining
//!   `BranchlessKernel + PhaseSteering + PointerActionHub + FastPathLedger`, completely agnostic
//!   to Hot vs. Cold Path execution.
//! - **Concrete Core Kernels**: Deterministic constant-time $\mathcal{O}(1)$ execution (`PhaseSteeringEngine`,
//!   `ActionHubEngine`, `EntropyController`, `MultiAttractorSpace`, `MappedStorageEngine`, plus pure
//!   sub-head routing, Rank-1 residual, visual steering, sheaf cohomology, L2P-SVD, and DNP functions)
//!   with zero dynamic allocations ($C_{\mathrm{alloc}} = 0$).
//! - **Latent Horizon Traits (`ManifoldGatewaySignal`, `StigmergicActionReceptor`)**: Thin boundary
//!   contracts uniting `SystemConstraintSignal`, `ObjectiveFactSignal`, `SubjectivePersonaSignal`,
//!   `ContradictionGuardSignal`, and external action/storage slots around the 3-layer core stack.
//!
//! # Core Mathematical & Systems Design Principles
//! 1. **Sequential Conditional Transition Probability $\times$ Attention Phase Interference**:
//!    Long-horizon persona and context consistency $P(M_1, \dots, M_T) = \prod_{t=1}^T P(M_t \mid M_{<t})$
//!    is factored into local conditional transitions modulated in $\mathcal{O}(1)$ via RoPE phase
//!    steering without modifying host model weights $W_Q, W_K$.
//! 2. **Fractal Self-Similarity & Indirect Pointer-Chain Routing in `ActionHub`**:
//!    From 128-byte pointer trails (`gcso_pointer_trail_t`), 128-byte `GatewayFrame` /
//!    `CrystallizedAttractorRecord` basins, up to nested composites `[Node; K]` and `(A, B)`,
//!    every layer implements [`crate::traits::RelativeAttractorNode`] (`Liminal`) with recursive
//!    $SO(2)^{64}$ resultant folding (`fold_fractal_nodes_words_64`), dissipative overflow scatter /
//!    autonomous daughter crystallization (`DissipativeOverflowReport`), and perspective-flipped signed
//!    coupling (`RelativeCouplingMatrix`).
//! 3. **`Liminal`, Dynamic `LiminalApertureField`, & Liminal Bifurcation Routing**:
//!    Queries inside carved memory horizons separate coherent grounding ($H^0$) from internal
//!    opposition ($H^1 \neq 0$), while out-of-horizon queries enter `EpistemicRegime::Liminal`.
//!    [`crate::traits::LiminalApertureField`] shifts per-channel cognitive horizons branchlessly in
//!    response to cognitive trajectory habits and dissipative overflow, exposing live Mermaid topology
//!    self-introspection via [`crate::traits::LiminalTopologyTelemetrySnapshot`].
//! 4. **Single Unified Attractor Manifold**:
//!    All anchors coexist on a single unified phase manifold ($SO(2)^{d_{\mathrm{head}}/2}$)
//!    governed by zero-hyperparameter phase-proximity gravity (`branchless_q7_resonant_pull_64`),
//!    with an optional 8-lane hardware working cache window (`slot_phase_bank`) for $\mathcal{O}(1)$
//!    tagged-pointer addressing.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::inline_always)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
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
#![allow(clippy::undocumented_unsafe_blocks)]
#![allow(clippy::needless_return)]

#[cfg(any(feature = "std", test))]
extern crate std;

extern crate alloc;

// ===================================================================
// Core Architecture Module Declarations
// ===================================================================

/// C-ABI bindings, FFI safety barriers, binary layout contracts, and status codes.
pub mod abi;

/// Stigmergic Threshold Action-Selecting Orchestrator (`ActionHubEngine`) and Sidecar Pointer Table.
pub mod action_hub;

/// Dynamic Attractor Field Engine, Unified Phase Manifold, and Accumulator Pressure Dynamics.
pub mod attractor;

/// Hot-Path RoPE Phase Steering, RIPA soft-clamping, and sub-head routing routines.
pub mod phase_steering;

/// Cold-Path Entropy Evaluation, Branch Control, and Langevin Pitchfork stability routines.
pub mod entropy_control;

/// Dynamic Rank-1 Sparse Residual Adapter (`u * v^T`) and persona patching routines.
pub mod residual_adapter;

/// Cellular stigmergic coordination, Sidecar Pointer Table (SPT), and reconstitution engine.
pub mod stigmergy;

/// Core system contracts, `1 + 8` trait hierarchy, branchless primitives, and alignment guards.
pub mod traits;

/// Unified `.gcso` container storage and zero-copy memory-mapped persistence.
pub mod storage;

/// `.gcso` Crystallized Attractor Record (`CrystallizedAttractorRecord`) and Online Toroidal $k$-Means / DP-Means.
pub mod crystal;

/// High-level runtime configuration builder (`Config`) and full-stack `RuntimeContext`.
pub mod runtime;

// ===================================================================
// Primary Re-exports for Ergonomics & Ground Truth Alignment
// ===================================================================

pub use crate::abi::{
    gcso_anchor_type_t, gcso_capability_flags_t, gcso_config_t, gcso_descriptor_header_t,
    gcso_edbc_state_t, gcso_kv_keyframe_header_t, gcso_ledger_slot_t,
    gcso_mapped_storage_descriptor_t, gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_q7_t,
    gcso_rank1_residual_descriptor_t, gcso_snapshot_header_t, gcso_status_t, gcso_subhead_config_t,
    gcso_tagged_ptr_t, GcsoAnchorType, GcsoStatus, GCSO_CAP_CPU_BASELINE, GCSO_CAP_CUDA_BACKEND,
    GCSO_CAP_METAL_BACKEND, GCSO_CAP_VULKAN_BACKEND, GCSO_SUCCESS,
};
pub use crate::action_hub::{
    ActionHubEngine, ActionOrchestrationReport, StigmergicActionDecision, StigmergicActionReceptor,
    StigmergicThresholds, ACTION_HUB_INLINE_TRAILS, DEFAULT_STIGMERGIC_CONTRADICTION_THRESH,
    DEFAULT_STIGMERGIC_CRYSTALLIZE_THRESH, DEFAULT_STIGMERGIC_EVAPORATION_RETAIN,
    DEFAULT_STIGMERGIC_EXPRESS_STEER_THRESH, DEFAULT_STIGMERGIC_INERTIAL_SKIP_ENTROPY,
    DEFAULT_STIGMERGIC_RANK1_INJECT_THRESH, DEFAULT_STIGMERGIC_REINFORCEMENT_GAIN,
    DEFAULT_STIGMERGIC_SUBHEAD_ROUTE_THRESH,
};

pub use crate::attractor::{
    AttractorAnchor, AttractorFieldEngine, AttractorGrowthCoordinator, AttractorGrowthReport,
    AttractorGrowthSnapshot, CrystallizedAnchorRecord, DomainAccumulatorSummary,
    EpisodicPhaseSample, EpisodicSampleSteering, MultiAttractorSpace, DOMAIN_CUSTOM_5,
    DOMAIN_CUSTOM_6, DOMAIN_CUSTOM_7, DOMAIN_KNOWLEDGE, DOMAIN_PERSONA, DOMAIN_PHASE_REPULSION,
    DOMAIN_SYSTEM_BASELINE, DOMAIN_USER_INTERACTION, MAX_ANCHORS_PER_FIELD, MAX_ATTRACTOR_DOMAINS,
    MAX_EMBEDDING_DIM,
};
pub use crate::entropy_control::{EntropyClosedLoopCoordinator, EntropyController};
pub use crate::phase_steering::{PhaseSteeringClosedLoop, PhaseSteeringEngine};
pub use crate::residual_adapter::ResidualAdapterEngine;
pub use crate::stigmergy::StigmergicGranuleEngine;
pub use crate::storage::MappedStorageEngine;
pub use crate::traits::{
    abi_version_string, bridge_and_filter_liminal_trail_chain, cascade_action_slot_ptr,
    cascade_tagged_ptr, compute_tikhonov_chebyshev_coeffs_deg3,
    decode_and_apply_dnp_packet_to_trail, decode_and_verify_dnp_packet,
    decode_and_verify_dnp_words, detect_and_record_obstruction_repulsion,
    detect_frame_delta_bitmask, detect_frame_mse_bitmask, dispatch_dnp_packet,
    dispatch_pointer_gated, dispatch_pointer_gated_with_phases,
    dispatch_ripa_and_subhead_slot_ensemble_fused, dispatch_ripa_and_subhead_unified_anchors_fused,
    dispatch_single_pass, dispatch_single_pass_with_phases, dispatch_subhead_slot_ensemble,
    dispatch_subhead_slot_ensemble_geodesic, dispatch_subhead_unified_anchors,
    encode_dnp_phase_packet, eval_chebyshev_spectral_filter_deg3, eval_gershgorin_lambda_max,
    eval_pseudo_ultrametric_distance, eval_rank1, eval_rank1_fused, eval_rank1_pointer_gated_fused,
    eval_rank1_slices, eval_rank1_slices_fused, eval_stigmergic_block_degrees,
    evaluate_cohomological_obstruction, evaluate_epistemic_horizon_words_64,
    evaluate_hodge_obstruction_and_dual_pairing, evaluate_stigmergic_venn_and_serendipity_words_64,
    evaluate_trail_chain_cohomology_and_joint_log_prob, fold_fractal_nodes_words_64,
    init_default_config, project_lora, project_lora_with_block_scales, quantize_depth_hysteresis,
    query_capability, resolve_autonomous_swarm_rot_dim, resolve_dyadic_qdpt_and_cascade_ptr,
    ripa_upper_rot_dim, steer_visual_grid_2d_bitmask_gated, steer_visual_grid_3d_bitmask_gated,
    steer_visual_patch_2d, steer_visual_patch_3d, validate_config_invariants, validate_pspm_layout,
    ActionHub, ActionHubGatewayExt, AlignedBuffer128, AlignedBuffer16, AlignedBuffer256,
    AlignedBuffer32, AlignedBuffer512, AlignedBuffer64, AlignedSlice, AlignedSlice128,
    AlignedSlice16, AlignedSlice256, AlignedSlice32, AlignedSlice512, AlignedSlice64,
    AlignedSliceMut, AlignedSliceMut128, AlignedSliceMut16, AlignedSliceMut256, AlignedSliceMut32,
    AlignedSliceMut512, AlignedSliceMut64, AttractorField, AutonomousCoreLayerStack,
    BranchlessKernel, CausalProvenanceReport, CausalProvenanceSignal, CognitiveGroundingSignal,
    ColdPathTrailCrystalExt, ContradictionGuard, ContradictionGuardSignal,
    ContrastivePhaseBloomFilter, CoreTelemetryBank, CoreTelemetrySnapshot,
    CrystallizedAttractorRecord, DefaultResidualSpilloverSlot, DetailedLayerTelemetry,
    DisjointMutPair, DisjointMutTriple, DisjointRefMut, DissipativeOverflowReport,
    EdbcBranchController, EgoIntentAxes, EgoLayerTelemetry, EntropyEvaluator,
    EpistemicHorizonReading, EpistemicManifoldOps, EpistemicRegime, FastPathLedger, FfiHandleMut,
    FfiHandleRef, GatewayFrame, GatewayPhaseSteeringExt, GcsoFourSectionView, GcsoResult,
    HorizonDwellState, HorizonEscalationProvider, HorizonEscalationRequest, HorizonTriggerInput,
    KvKeyframeCache, KvKeyframePacket, KvMotionDeltaPacket, LatentHorizonSlot, LayerDistortionKind,
    LayerDistortionReport, Liminal, LiminalApertureField, LiminalBoundaryDelta, LiminalEgoField,
    LiminalEgoPreset, LiminalHorizonNodeDescriptor, LiminalLayerTelemetry,
    LiminalTopologyTelemetrySnapshot, ManifoldGatewaySignal, MappedStorage,
    MultiDomainAttractorSpace, ObjectiveFact, ObjectiveFactSignal, PagedBitmaskOps, PersonaPatcher,
    PhaseSteering, PheromoneDensityOps, Pod, PointerActionHub, RawPhaseProjectable,
    RelativeAttractorNode, RelativeCouplingMatrix, ResidualSpilloverHorizonSlot,
    SlowLoopTelemetryAdjustment, StigmergicActionPtrOps, StigmergicChunkOps,
    StigmergicFragmentEnsembleReport, StigmergicGranuleField, StigmergicLayerTelemetry,
    StigmergicReconstitutionOps, StigmergicReconstitutionReport, StigmergicScaleGranularity,
    StigmergicTrailOps, StigmergicVennSerendipityReport, SubjectivePersona,
    SubjectivePersonaSignal, SystemConstraint, SystemConstraintSignal, SystemSpineTelemetry,
    TelemetryCapturePolicy, TelemetryVerbosityMode, ACTION_MODE_DPSR_STEER, ACTION_MODE_LAYER_SKIP,
    ACTION_MODE_PHASE_REPULSE, ACTION_MODE_RANK1_INJECT, ACTION_MODE_SUBHEAD_ROUTE,
    ACTION_PRESET_EXPRESS_PHASE, ACTION_PRESET_FULL_ORCHESTRATION, ACTION_PRESET_INERTIAL_SKIP,
    ACTION_PRESET_PASSIVE_SIDECAR, ACTION_PRESET_REPULSION_ESCAPE, APERTURE_ACTION_HUB_WIDE,
    APERTURE_CAUSAL_PROVENANCE, APERTURE_COGNITIVE_CIRCUIT, APERTURE_COHOMOLOGY_EDBC,
    APERTURE_CONTRADICTION_GUARD, APERTURE_DISSIPATIVE_CRYSTAL, APERTURE_FAST_PATH_LEDGER,
    APERTURE_KV_PHASE_CACHE, APERTURE_MANIFOLD_GATEWAY, APERTURE_MAPPED_STORAGE,
    APERTURE_OBJECTIVE_FACT, APERTURE_PHASE_STEERING, APERTURE_POINTER_TRAIL,
    APERTURE_RANK1_RESIDUAL, APERTURE_RELATIVE_COUPLING, APERTURE_STIGMERGIC_CONSENSUS,
    APERTURE_SUBHEAD_ROUTING, APERTURE_SUBJECTIVE_PERSONA, APERTURE_SYSTEM_CONSTRAINT,
    ATTRACTOR_SLOT_CUSTOM_5, ATTRACTOR_SLOT_CUSTOM_6, ATTRACTOR_SLOT_CUSTOM_7,
    ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA, ATTRACTOR_SLOT_REPULSION,
    ATTRACTOR_SLOT_SYSTEM_BASELINE, ATTRACTOR_SLOT_USER_CONTEXT, CAUSAL_AXIS_CONTRADICTION_GUARD,
    CAUSAL_AXIS_OBJECTIVE_FACT, CAUSAL_AXIS_SUBJECTIVE_PERSONA, CAUSAL_AXIS_SYSTEM_CONSTRAINT,
    CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, CAUSAL_CAUSE_COHOMOLOGY_REPULSION,
    CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK, CAUSAL_CAUSE_PERSONA_FACT_COLLISION,
    CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE, CORE_LAYER_EGO, CORE_LAYER_LIMINAL, CORE_LAYER_STIGMERGIC,
    DEFAULT_FACT_RIPA_CLAMP_RAD, DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO,
    DEFAULT_PERSONA_FACT_COLLISION_THRESH, DEFAULT_PERSONA_RIPA_CLAMP_RAD,
    DEFAULT_SLOT_COUPLING_MATRIX_8X8, DEFAULT_STIGMERGIC_QUORUM_THRESHOLD,
    DEFAULT_STIGMERGIC_TOLERANCE_Q7, DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE,
    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE, EGO_TRIGGER_ALL_DEFAULT, EGO_TRIGGER_BASIN_CAPTURE,
    EGO_TRIGGER_BOUNDARY_CHATTER, EGO_TRIGGER_BOUNDARY_STAGNATION, EGO_TRIGGER_INTERNAL_CONFLICT,
    EGO_TRIGGER_LAYER_DISTORTION, EGO_TRIGGER_LIMINAL_EXTERIOR, GCSO_CRYSTAL_RECORD_MAGIC,
    GCSO_CRYSTAL_RECORD_VERSION, GCSO_FOUR_SECTION_HEADER_BYTES,
    GCSO_MAPPED_STORAGE_PREAMBLE_OFFSET, HEADS_PER_STIGMERGIC_GRANULE, NUM_CORE_AUTONOMOUS_LAYERS,
    NUM_LIMINAL_APERTURE_CHANNELS, NUM_STIGMERGIC_GRANULES, ROOT_CAUSE_COHOMOLOGY_REPULSION,
    ROOT_CAUSE_FACT_RETRIEVAL_MISS, ROOT_CAUSE_NOMINAL, ROOT_CAUSE_PERSONA_FACT_COLLISION,
    ROOT_CAUSE_SYSTEM_QOS_THROTTLE, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, TRAIL_FLAG_DPSR_ACTIVE,
    TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE,
    TRAIL_FLAG_SWARM_ESCALATED, TRAIL_FLAG_SWARM_RESOLVED, TRAIL_FLAG_SYNERGY_RESONANCE,
};

pub use crate::runtime::{
    Config, OrchestratorStatusSnapshot, Result, RuntimeContext, DEFAULT_BLOOM_REPULSION_SCALE_U7,
    DEFAULT_BLOOM_REPULSION_THRESHOLD, DEFAULT_EDBC_BIFURCATION_THRESHOLD,
    DEFAULT_EDBC_HYSTERESIS_BAND, DEFAULT_EDBC_TAU_FLUX, DEFAULT_SLOT_WEIGHTS, SLOT_PHASE_BANK_LEN,
};

// ===================================================================
// Convenient Prelude Module
// ===================================================================

/// Convenient prelude module importing essential GCSO traits, `ActionHub`, Hot/Cold observers,
/// 2-Origin Composite Manifold Gateway, Online Toroidal $k$-Means / DP-Means, `.gcso` 4-Section View,
/// Unified-Field & Bayesian-Attention kernels, types, and alignment wrappers.
pub mod prelude {
    pub use crate::abi::{
        gcso_anchor_type_t, gcso_capability_flags_t, gcso_config_t, gcso_descriptor_header_t,
        gcso_edbc_state_t, gcso_kv_keyframe_header_t, gcso_ledger_slot_t,
        gcso_mapped_storage_descriptor_t, gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_q7_t,
        gcso_rank1_residual_descriptor_t, gcso_snapshot_header_t, gcso_status_t,
        gcso_subhead_config_t, gcso_tagged_ptr_t, GcsoAnchorType, GcsoStatus, GCSO_ABI_VERSION_HEX,
        GCSO_ABI_VERSION_STRING, GCSO_SUCCESS,
    };
    pub use crate::action_hub::{
        ActionHubEngine, ActionOrchestrationReport, StigmergicActionDecision,
        StigmergicActionReceptor, StigmergicThresholds, ACTION_HUB_INLINE_TRAILS,
    };
    pub use crate::attractor::{
        AttractorAnchor, AttractorFieldEngine, AttractorGrowthCoordinator, AttractorGrowthReport,
        AttractorGrowthSnapshot, CrystallizedAnchorRecord, DomainAccumulatorSummary,
        EpisodicPhaseSample, EpisodicSampleSteering, MultiAttractorSpace, DOMAIN_CUSTOM_5,
        DOMAIN_CUSTOM_6, DOMAIN_CUSTOM_7, DOMAIN_KNOWLEDGE, DOMAIN_PERSONA, DOMAIN_PHASE_REPULSION,
        DOMAIN_SYSTEM_BASELINE, DOMAIN_USER_INTERACTION, MAX_ANCHORS_PER_FIELD,
        MAX_ATTRACTOR_DOMAINS, MAX_EMBEDDING_DIM,
    };
    pub use crate::entropy_control::{EntropyClosedLoopCoordinator, EntropyController};
    pub use crate::phase_steering::{PhaseSteeringClosedLoop, PhaseSteeringEngine};
    pub use crate::residual_adapter::ResidualAdapterEngine;
    pub use crate::stigmergy::StigmergicGranuleEngine;
    pub use crate::traits::prelude::*;
    pub use crate::{Config, OrchestratorStatusSnapshot, Result, RuntimeContext};
}

/// Static package version string constant obtained from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns the static package version string defined in `Cargo.toml`.
#[must_use]
#[inline(always)]
pub const fn version() -> &'static str {
    VERSION
}

/// Returns the unified C-ABI version tuple `(major, minor, patch)`.
#[inline(always)]
#[must_use]
pub const fn abi_version() -> (u32, u32, u32) {
    (
        crate::abi::GCSO_ABI_VERSION_MAJOR,
        crate::abi::GCSO_ABI_VERSION_MINOR,
        crate::abi::GCSO_ABI_VERSION_PATCH,
    )
}

/// Returns the hexadecimal ABI version identifier ([`crate::abi::GCSO_ABI_VERSION_HEX`]).
#[inline(always)]
#[must_use]
pub const fn abi_version_hex() -> u32 {
    crate::abi::GCSO_ABI_VERSION_HEX
}
