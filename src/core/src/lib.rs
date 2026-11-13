// name: src/core/src/lib.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core execution engine library for the Geometric Cellular Sheaf Orchestrator (GCSO).
//!
//! This crate serves as the primary low-level runtime engine for `LiminiKa`, implementing the
//! Geometric Cellular Sheaf Orchestrator (GCSO) architecture targeting 2–4 GB VRAM environments.
//!
//! # Architecture: `1 + 8` Flat Trait Topology (`ActionHub` + 2 Observers + Gateway)
//! - **`ActionHub` (Core Sidecar State & Capability Ledger, [`crate::traits::ActionHub`])**: The
//!   foundational micro composite ledger of state and branchless capabilities (`BranchlessKernel +
//!   PhaseSteering + PointerActionHub + PspmRouter + SrlAdapter + DaesScratchpad`), completely
//!   agnostic to Hot vs. Cold Path execution.
//! - **Lv.2 Micro Observer ([`crate::traits::HotPathObserver`])**: Drives constant-time $\mathcal{O}(1)$
//!   per-token cascades (`Pointer -> DAES -> Bloom Repulsion -> Unified Field / Limina / Swarm ->
//!   QDPS/RIPA/RoPE -> PSPM/SRL`) with zero dynamic allocations ($C_{\mathrm{alloc}} = 0$).
//! - **Lv.3 Macro Observer ([`crate::traits::ColdPathObserver`])**: Evaluates trajectory telemetry,
//!   moving Z-score entropy ($\tilde{H} \in [0, \ln 2]$), sliding-window flux ($\Phi_M(t)$),
//!   scale-invariant Langevin pitchfork bifurcations, matrix-free Gershgorin/Chebyshev sheaf
//!   cohomology $H^1(K; \mathcal{F})$, Single Unified Attractor Manifold superposition, L2P-SVD
//!   projection, PPRC keyframe replay, and ZIMMS zero-copy `.gcso` persistence.
//!
//! # Core Mathematical & Systems Design Principles
//! 1. **Sequential Conditional Transition Probability $\times$ Attention Phase Interference**:
//!    Long-horizon persona and context consistency $P(M_1, \dots, M_T) = \prod_{t=1}^T P(M_t \mid M_{<t})$
//!    is factored into local conditional transitions modulated in $\mathcal{O}(1)$ via RoPE phase
//!    steering without modifying host model weights $W_Q, W_K$.
//! 2. **Indirect Pointer-Chain Graph Routing in `ActionHub`**:
//!    Rather than holding all knowledge inside active model parameters, the runtime maintains compact
//!    64-bit tagged pointers ([`gcso_tagged_ptr_t`]) and 128-byte pointer trails ([`gcso_pointer_trail_t`])
//!    that reference external `.gcso` storage offsets (`zimms_section_offset48`) and chain to subsequent
//!    trail nodes (`linked_trail_id`). Traversing and reinforcing these pointer chains forms a sparse,
//!    associative routing graph that bridges inter-basin **Limina** thresholds while repulsing
//!    topological contradictions ($H^1(K; \mathcal{F}) \neq 0$) via Phase-Conjugate Attractor Repulsion ($-\boldsymbol{\Delta\theta}$).
//! 3. **Coherence Void ($\mathcal{C}_{\mathrm{void}}$) & Liminal Bifurcation Routing**:
//!    Representation-space sparsity and inter-basin dispersion are quantified via the dyadic
//!    **Coherence Void** functional ($\mathcal{C}_{\mathrm{void}}$) and $\ell_1 / \ell_2$ coherence
//!    ratios. When a trajectory enters the threshold boundary (*Limina*) between competing attractor
//!    basins ($G_2 / G_1 \ge \tau_{\mathrm{liminal}}$), Top-2 geodesic midpoint interpolation bridges
//!    both basins without collapsing into winner-take-all truncation or incoherent drift.
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

/// Cellular swarm coordination, Sidecar Pointer Table (SPT), and stigmergic routines.
pub mod swarm;

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
    gcso_anchor_type_t, gcso_capability_flags_t, gcso_config_t, gcso_daes_slot_t,
    gcso_descriptor_header_t, gcso_edbc_state_t, gcso_paged_bitmask_t, gcso_pointer_trail_t,
    gcso_pprc_keyframe_header_t, gcso_pspm_config_t, gcso_q7_t, gcso_snapshot_header_t,
    gcso_srl_descriptor_t, gcso_status_t, gcso_tagged_ptr_t, gcso_zimms_descriptor_t,
    GcsoAnchorType, GcsoStatus, GCSO_CAP_CPU_BASELINE, GCSO_CAP_CUDA_BACKEND,
    GCSO_CAP_METAL_BACKEND, GCSO_CAP_VULKAN_BACKEND, GCSO_SUCCESS,
};
pub use crate::action_hub::{
    ActionHubEngine, ActionOrchestrationReport, StigmergicActionDecision, StigmergicActionReceptor,
    StigmergicThresholds, ACTION_HUB_INLINE_TRAILS, DEFAULT_STIGMERGIC_CONTRADICTION_THRESH,
    DEFAULT_STIGMERGIC_CRYSTALLIZE_THRESH, DEFAULT_STIGMERGIC_EVAPORATION_RETAIN,
    DEFAULT_STIGMERGIC_EXPRESS_STEER_THRESH, DEFAULT_STIGMERGIC_INERTIAL_SKIP_ENTROPY,
    DEFAULT_STIGMERGIC_PSPM_ROUTE_THRESH, DEFAULT_STIGMERGIC_REINFORCEMENT_GAIN,
    DEFAULT_STIGMERGIC_SRL_INJECT_THRESH,
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
pub use crate::storage::MappedStorageEngine;
pub use crate::swarm::SwarmEngine;
pub use crate::traits::{
    resolve_autonomous_swarm_rot_dim, ripa_upper_rot_dim, ActionHub, ActionHubGatewayExt,
    AlignedBuffer128, AlignedBuffer16, AlignedBuffer256, AlignedBuffer32, AlignedBuffer512,
    AlignedBuffer64, AlignedSlice, AlignedSlice128, AlignedSlice16, AlignedSlice256,
    AlignedSlice32, AlignedSlice512, AlignedSlice64, AlignedSliceMut, AlignedSliceMut128,
    AlignedSliceMut16, AlignedSliceMut256, AlignedSliceMut32, AlignedSliceMut512,
    AlignedSliceMut64, AttractorField, AttractorSlotSynergyCoordinator,
    BottomUpMacroCrystalCoordinator, BranchlessKernel, CausalProvenanceReport,
    CausalProvenanceSignal, CognitiveGroundingSignal, ColdPathObserver, ColdPathOrchestrator,
    ColdPathTrailCrystalExt, ContradictionGuard, ContradictionGuardSignal,
    ContrastivePhaseBloomFilter, CrystallizedAttractorRecord, DaesScratchpad, DisjointMutPair,
    DisjointMutTriple, DisjointRefMut, DnpPacketDispatcher, DynamicLayerSkipper,
    EdbcBranchController, EntropyEvaluator, EntropyGatedPhaseLookup, FfiHandleMut, FfiHandleRef,
    FractalCascadeCoordinator, FullStackGcsoEngine, GatewayFrame, GatewayPhaseSteeringExt,
    GcsoFourSectionView, GcsoResult, GcsoRuntimeContext, HierarchicalTreeTable,
    HotPathExecutionEngine, HotPathObserver, L2pSvdProjector, ManifoldGatewaySignal,
    MultiDomainAttractorSpace, ObjectiveFact, ObjectiveFactSignal, PagedBitmaskOps, PersonaPatcher,
    PhaseSteering, Pod, PointerActionHub, PprcCache, PprcKeyframePacket, PprcMotionDeltaPacket,
    PspmRouter, RawPhaseProjectable, SheafCohomologyEvaluator, SrlAdapter, StigmergicActionPtrOps,
    StigmergicCohomologyCoordinator, StigmergicSwarmOps, StigmergicTrailOps, SubjectivePersona,
    SubjectivePersonaSignal, SwarmCellChunk, SwarmResolutionReport, SystemCapabilityQuery,
    SystemConstraint, SystemConstraintSignal, VisualAttentionSteering, ZimmsStorage,
    ACTION_MODE_DPSR_STEER, ACTION_MODE_LAYER_SKIP, ACTION_MODE_PHASE_REPULSE,
    ACTION_MODE_PSPM_ROUTE, ACTION_MODE_SRL_INJECT, ACTION_PRESET_EXPRESS_PHASE,
    ACTION_PRESET_FULL_ORCHESTRATION, ACTION_PRESET_INERTIAL_SKIP, ACTION_PRESET_PASSIVE_SIDECAR,
    ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_CUSTOM_5, ATTRACTOR_SLOT_CUSTOM_6,
    ATTRACTOR_SLOT_CUSTOM_7, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA,
    ATTRACTOR_SLOT_REPULSION, ATTRACTOR_SLOT_SYSTEM_BASELINE, ATTRACTOR_SLOT_USER_CONTEXT,
    CAUSAL_AXIS_CONTRADICTION_GUARD, CAUSAL_AXIS_OBJECTIVE_FACT, CAUSAL_AXIS_SUBJECTIVE_PERSONA,
    CAUSAL_AXIS_SYSTEM_CONSTRAINT, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
    CAUSAL_CAUSE_COHOMOLOGY_REPULSION, CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK,
    CAUSAL_CAUSE_PERSONA_FACT_COLLISION, CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE,
    DEFAULT_FACT_RIPA_CLAMP_RAD, DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO,
    DEFAULT_PERSONA_FACT_COLLISION_THRESH, DEFAULT_PERSONA_RIPA_CLAMP_RAD,
    DEFAULT_SLOT_COUPLING_MATRIX_8X8, DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE,
    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE, GCSO_CRYSTAL_RECORD_MAGIC,
    GCSO_CRYSTAL_RECORD_VERSION, GCSO_FOUR_SECTION_HEADER_BYTES,
    GCSO_ZIMMS_DESCRIPTOR_PREAMBLE_OFFSET, ROOT_CAUSE_COHOMOLOGY_REPULSION,
    ROOT_CAUSE_FACT_RETRIEVAL_MISS, ROOT_CAUSE_NOMINAL, ROOT_CAUSE_PERSONA_FACT_COLLISION,
    ROOT_CAUSE_SYSTEM_QOS_THROTTLE, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, TRAIL_FLAG_DPSR_ACTIVE,
    TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE,
    TRAIL_FLAG_SWARM_ESCALATED, TRAIL_FLAG_SWARM_RESOLVED, TRAIL_FLAG_SYNERGY_RESONANCE,
};

pub use crate::crystal::CrystallizedAttractorRecord as CrystalRecord;
pub use crate::runtime::{
    Config, Result, RuntimeContext, DEFAULT_BLOOM_REPULSION_SCALE_U7,
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
        gcso_anchor_type_t, gcso_capability_flags_t, gcso_config_t, gcso_daes_slot_t,
        gcso_descriptor_header_t, gcso_edbc_state_t, gcso_paged_bitmask_t, gcso_pointer_trail_t,
        gcso_pprc_keyframe_header_t, gcso_pspm_config_t, gcso_q7_t, gcso_snapshot_header_t,
        gcso_srl_descriptor_t, gcso_status_t, gcso_tagged_ptr_t, gcso_zimms_descriptor_t,
        GcsoAnchorType, GcsoStatus, GCSO_ABI_VERSION_HEX, GCSO_ABI_VERSION_STRING, GCSO_SUCCESS,
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
    pub use crate::storage::MappedStorageEngine;
    pub use crate::swarm::SwarmEngine;
    pub use crate::traits::prelude::*;
    pub use crate::{Config, Result, RuntimeContext};
}

/// Static package version string constant obtained from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns the static package version string defined in `Cargo.toml`.
#[must_use]
#[inline(always)]
pub const fn version() -> &'static str {
    VERSION
}

/// Returns the unified C-ABI version tuple `(major, minor, patch)` (`(0, 1, 1)`).
#[inline(always)]
#[must_use]
pub const fn abi_version() -> (u32, u32, u32) {
    (
        crate::abi::GCSO_ABI_VERSION_MAJOR,
        crate::abi::GCSO_ABI_VERSION_MINOR,
        crate::abi::GCSO_ABI_VERSION_PATCH,
    )
}

/// Returns the hexadecimal ABI version identifier (`0x00000101`).
#[inline(always)]
#[must_use]
pub const fn abi_version_hex() -> u32 {
    crate::abi::GCSO_ABI_VERSION_HEX
}
