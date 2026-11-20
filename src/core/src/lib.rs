// name: src/core/src/lib.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core execution engine library for the Geometric Cellular Sheaf Orchestrator (GCSO).
//!
//! This crate serves as the primary low-level runtime engine for `LiminiKa`, implementing the
//! Geometric Cellular Sheaf Orchestrator (GCSO) architecture targeting 2-4 GB VRAM environments.
//!
//! # Architecture: Two-Layer Hybrid Trait Topology (`Liminal` + `ActionHub` + Gateway)
//! - **`Liminal` ([`crate::liminal::Liminal`])**: Root epistemic supertrait defining the
//!   unknowable exterior beyond any trait's cognitive aperture (`!APERTURE_MASK`), gating
//!   out-of-horizon steering cleanly to zero without data-dependent branches.
//! - **`ActionHub` (Core Sidecar State & Capability Ledger, [`crate::stigmergy::ActionHub`])**: The
//!   widest-aperture composite ledger (`HUB_APERTURE_MASK = 0x0000_FFFF`) combining
//!   `Liminal + PhaseSteering + PointerActionHub + FastPathLedger`, completely agnostic
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
//!    every layer implements [`crate::stigmergy::RelativeAttractorNode`] (`Liminal`) with recursive
//!    $SO(2)^{64}$ resultant folding (`fold_fractal_nodes_words_64`), dissipative overflow scatter /
//!    autonomous daughter crystallization (`DissipativeOverflowReport`), and perspective-flipped signed
//!    coupling (`RelativeCouplingMatrix`).
//! 3. **`Liminal`, Dynamic `LiminalThresholdField`, & Liminal Bifurcation Routing**:
//!    Queries inside carved memory horizons separate coherent grounding ($H^0$) from internal
//!    opposition ($H^1 \neq 0$), while out-of-horizon queries enter `EpistemicRegime::Liminal`.
//!    [`crate::liminal::LiminalThresholdField`] shifts per-channel cognitive horizons branchlessly in
//!    response to cognitive trajectory habits and dissipative overflow, exposing live Mermaid topology
//!    self-introspection via [`crate::runtime::LiminalTopologyTelemetrySnapshot`].
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
// 5-Layer Unidirectional DAG Module Declarations (Lv.0 -> Lv.4)
// ===================================================================

/// Level 0 (`Liminal`): Foundational boundary axioms, zero-sized scope markers, orthogonal
/// 4-state `Lacuna`, threshold sensitivity, and C-ABI invariant constants (`core` only).
pub mod liminal;

/// Level 1 (`Foundation`): Pure geometric, POD, aligned-slice, SWAR/AVX2 SIMD, lookup-table,
/// RoPE phase-steering, Rank-1 residual, LoRA SVD, and matrix-free cohomology kernels.
pub mod foundation;

/// Level 2 (`Stigmergy`): 8-granule SWAR stigmergic trails, pheromone density laws, bitmasks,
/// contrastive Bloom filters, fast-path ledgers, 8-lane [`ActionHubEngine`], cognitive circuits,
/// `.gcso` crystallization, and macro attractor fields.
pub mod stigmergy;

/// Level 3 (`Drivers`): Dual execution axes (`spine` vs. `ego`) isolated behind `distortion`.
pub mod drivers;

/// Level 4 (`Runtime`): Stateful orchestration (`RuntimeContext`), telemetry, Manifold Gateway,
/// Latent Horizon, `.gcso` storage, and C-ABI translation firewall (`abi`).
pub mod runtime;

/// C-ABI `v0.1.3` bindings, FFI safety barriers, binary layout contracts, and status codes.
pub use crate::runtime::abi;

// ===================================================================
// Primary Crate-Root Re-exports Across All 5 Layers
// ===================================================================

pub use crate::liminal::{
    branchless_select_f32, branchless_select_i32, branchless_select_i64, branchless_select_i8,
    branchless_select_q7, branchless_select_u16, branchless_select_u32, branchless_select_u64,
    branchless_select_u8, branchless_select_usize, cut_observed_granules,
    cut_self_vs_exterior_granules, evaluate_epistemic_partition_branchless, pack_attribution_tag,
    sensitivity_4p1mp_f32, sensitivity_4p1mp_q7, Active, ActiveScope, AttributionTag, BitCut32,
    BitCut8, Bottom, BoundaryViolation, BridgedDiscovery, Cuttable, Definite, DriverKind,
    EpistemicPartition, EpistemicRegime, Exterior, FromBoundary, FromDelta, GcsoResult, GcsoStatus,
    Grounded, GroundedOrigin, GroundedScope, Interior, Lacuna, Liminal, LiminalThresholdField,
    PackedLacunaField, Refine, Scoped, ScopedCut, Shared, SpeculativeDiscovery, SubstrateKind,
    Unmarked, VerifiedAligned, DEFAULT_ENTROPY_EPS, GCSO_SUCCESS, HEADS_PER_STIGMERGIC_GRANULE,
    MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS, MAX_SURROGATE_ENTROPY_LN2,
    NUM_LIMINAL_THRESHOLD_CHANNELS, NUM_STIGMERGIC_GRANULES, Q7_PHASE_INV_SCALE_RAD,
    Q7_PHASE_SCALE_RAD, SCOPE_REGION_KNOWN_UNKNOWN, SCOPE_REGION_SELF_EXCLUSIVE,
    SCOPE_REGION_SHARED_CORE, THRESHOLD_ACTION_HUB_WIDE, THRESHOLD_CAUSAL_PROVENANCE,
    THRESHOLD_COGNITIVE_CIRCUIT, THRESHOLD_COHOMOLOGY_ENTROPY, THRESHOLD_CONTRADICTION_GUARD,
    THRESHOLD_DISSIPATIVE_CRYSTAL, THRESHOLD_FAST_PATH_LEDGER, THRESHOLD_KV_PHASE_CACHE,
    THRESHOLD_MANIFOLD_GATEWAY, THRESHOLD_MAPPED_STORAGE, THRESHOLD_OBJECTIVE_FACT,
    THRESHOLD_PHASE_STEERING, THRESHOLD_POINTER_TRAIL, THRESHOLD_RANK1_RESIDUAL,
    THRESHOLD_RELATIVE_COUPLING, THRESHOLD_STIGMERGIC_CONSENSUS, THRESHOLD_SUBHEAD_ROUTING,
    THRESHOLD_SUBJECTIVE_PERSONA, THRESHOLD_SYSTEM_CONSTRAINT,
};

pub use crate::foundation::{
    branchless_abs_f32, branchless_argmin_f32_x8_masked, branchless_atan2_f32,
    branchless_atan2_to_q7_fast, branchless_clamp_f32, branchless_copysign_f32, branchless_cos_q7,
    branchless_crc32_128b_words, branchless_crc32_ieee, branchless_exp_f32,
    branchless_eyring_kramers_escape, branchless_fast_transition_log_prob,
    branchless_flush_subnormal_f32, branchless_hysteresis_quantize_depth, branchless_inv_pos_f32,
    branchless_is_finite_f32, branchless_is_nan_f32, branchless_lerp_f32, branchless_ln_f32,
    branchless_mask8_min_delta, branchless_max_f32, branchless_min_f32, branchless_phase_clamp_rad,
    branchless_phase_coherence_inv_temp_scale, branchless_phase_proximity_gravity_f32,
    branchless_phase_step_filter_q7, branchless_project_f32_slice_to_q7_64, branchless_q7_abs_u8,
    branchless_q7_accumulate_64, branchless_q7_add_sat, branchless_q7_circular_diff,
    branchless_q7_circular_lerp, branchless_q7_coherence_64,
    branchless_q7_coherence_and_resultant_64, branchless_q7_conjugate_64,
    branchless_q7_fold_s1_geodesic, branchless_q7_gravity_fold_64, branchless_q7_inhibit_64,
    branchless_q7_inhibit_words_64, branchless_q7_neg_sat, branchless_q7_phase_step_filter_64,
    branchless_q7_resonant_pull_64, branchless_q7_scale_64, branchless_q7_scale_u7,
    branchless_q7_sub_64, branchless_q7_sub_sat, branchless_q7_synergy_fuse_64,
    branchless_q7_synergy_fuse_words_64, branchless_q7_to_rad,
    branchless_q7_transition_log_prob_64, branchless_q7_transition_log_prob_words_64,
    branchless_q7_unified_field_fold_64, branchless_q7_unified_field_fold_with_init_weight_64,
    branchless_q7_unified_field_fold_words_with_init_weight_64, branchless_rad_to_phase_step_u8,
    branchless_rad_to_q7, branchless_relu_f32, branchless_rotate_so2_pair,
    branchless_rotate_so2_pair_scaled, branchless_rsqrt_f32, branchless_sanitize_z_score_f32,
    branchless_sigmoid_f32, branchless_signum_f32, branchless_sin_q7, branchless_sincos_f32,
    branchless_sincos_half_pi_f32, branchless_sincos_q7,
    branchless_sincos_q7_geodesic_phase_clamp_default_lut, branchless_sincos_q7_phase_clamp,
    branchless_sincos_q7_phase_clamp_default_lut, branchless_sincos_q7_phase_clamp_preinv,
    branchless_smoothstep_f32, branchless_splitmix64, branchless_sqrt_f32, branchless_tanh_f32,
    branchless_toroidal_centroid_step_words_64, branchless_update_crc32_ieee,
    branchless_von_mises_kappa, cascade_tagged_ptr, circular_r_bar_from_words_64,
    circular_resultant_from_words_64, coherence_from_diff_words_64, coherence_from_word_arrays_64,
    coherence_from_word_arrays_with_both_uniform_flags_64,
    coherence_from_word_arrays_with_uniform_flag_64, compute_crc32_ieee,
    compute_tikhonov_chebyshev_coeffs_deg3, detect_frame_delta_bitmask, detect_frame_mse_bitmask,
    dispatch_phase_clamp_and_subhead_slot_ensemble_fused, dispatch_single_pass,
    dispatch_subhead_slot_ensemble, dispatch_subhead_slot_ensemble_geodesic,
    eval_chebyshev_spectral_filter_deg3, eval_gershgorin_lambda_max,
    eval_pseudo_ultrametric_distance, eval_rank1, eval_rank1_fused, eval_rank1_pointer_gated_fused,
    eval_rank1_slices, eval_rank1_slices_fused, evaluate_cohomological_obstruction,
    evaluate_hodge_obstruction_and_dual_pairing, is_aligned_to, is_phase_profile_nonzero_64,
    is_ptr_aligned_to, is_word_array_uniform_64, load_q7_words_64, pack_action_slot_tag,
    phase_clamp_upper_rot_dim, project_bytes_swar_to_q7_64, project_bytes_swar_to_words_64,
    project_f32_slice_to_q7_64, project_f32_slice_to_words_64, project_lora,
    project_lora_with_block_scales, quantize_depth_hysteresis, raw_byte_ranges_overlap,
    resolve_autonomous_swarm_rot_dim, steer_visual_grid_2d_bitmask_gated,
    steer_visual_grid_3d_bitmask_gated, store_q7_words_64, sum_abs_and_sq_f32_simd8,
    sum_abs_sq_and_max_f32_simd8, sum_sq_f32_simd8, swar_q7_abs_u8_x8, swar_q7_add_sat_x4,
    swar_q7_add_sat_x8, swar_q7_avg_round_x8, swar_q7_circular_add_x8, swar_q7_circular_diff_x8,
    swar_q7_fold_s1_geodesic_x8, swar_q7_mean_i8_from_words_64, swar_q7_neg_sat_x4,
    swar_q7_neg_sat_x8, swar_q7_pack_x8, swar_q7_phase_step_filter_words_64,
    swar_q7_phase_step_filter_x8, swar_q7_scale_le128_with_bias_x8, swar_q7_scale_u7_x8,
    swar_q7_sub_sat_x4, swar_q7_sub_sat_x8, swar_q7_unpack_x8, swar_slerp_word_unchecked_x8,
    swar_slerp_word_with_bias_x8, swar_slerp_words_with_bias_64,
    swar_slerp_words_with_uniform_flag_64, swar_u8_add_sat_x8, swar_u8_halve_decay_x8,
    swar_u8_max_x8, swar_u8_mean_round_x8, swar_u8_nonzero_count_x8, swar_u8_sub_sat_x8,
    swar_u8_sum_x8, swar_words_equal_64, swar_words_nonzero_64, unpack_action_mode_from_tag,
    unpack_attractor_slot_from_tag, update_crc32_ieee, validate_subhead_layout, AlignedBuffer128,
    AlignedBuffer16, AlignedBuffer256, AlignedBuffer32, AlignedBuffer512, AlignedBuffer64,
    AlignedSlice, AlignedSlice128, AlignedSlice16, AlignedSlice256, AlignedSlice32,
    AlignedSlice512, AlignedSlice64, AlignedSliceMut, AlignedSliceMut128, AlignedSliceMut16,
    AlignedSliceMut256, AlignedSliceMut32, AlignedSliceMut512, AlignedSliceMut64, DisjointMutPair,
    DisjointMutTriple, DisjointRefMut, KvMotionDeltaPacket, PhaseSteering, Pod,
    RawPhaseProjectable, RopeGeometryDescriptor, RopePairLayout, RopeSteeringTarget,
    ACTION_MODE_LAYER_SKIP, ACTION_MODE_PHASE_REPULSE, ACTION_MODE_PHASE_STEER,
    ACTION_MODE_RANK1_INJECT, ACTION_MODE_SUBHEAD_ROUTE, ACTION_PRESET_EXPRESS_PHASE,
    ACTION_PRESET_FULL_ORCHESTRATION, ACTION_PRESET_INERTIAL_SKIP, ACTION_PRESET_PASSIVE_SIDECAR,
    ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_ID_MASK, CHEBYSHEV_COEFF_LEN,
    CHEBYSHEV_FILTER_DEGREE, DEFAULT_MAX_VALID_Z_SCORE, DEFAULT_PHASE_CLAMP_MAX_RAD,
    DEFAULT_PHASE_CLAMP_Q7_TO_NORM_SCALE, DEFAULT_PHASE_GRAVITY_EPS, DEFAULT_PHASE_STEP_MIN_RAD,
    F32_MIN_POSITIVE_NORMAL, MAX_PHASE_INTERFERENCE_INV_TEMP, MIN_PHASE_INTERFERENCE_INV_TEMP,
    PAGED_BITMASK_BITS, PAGED_BITMASK_WORDS, SWAR_EVEN_BYTES_64, SWAR_LOW7_MASK_64,
    SWAR_ODD_BYTES_64, SWAR_SIGN_MASK_64, SWAR_SPLAT_BYTE_64, SWAR_SUM_U16_64,
    TAG_ACTION_MODE_MASK, TAG_ATTRACTOR_SLOT_SHIFT,
};

pub use crate::stigmergy::{
    bridge_and_filter_liminal_trail_chain, coupled_slot_count, coupled_slot_row_mask,
    detect_and_record_obstruction_repulsion, eval_stigmergic_block_degrees,
    evaluate_stigmergic_overlap_and_serendipity_words_64,
    evaluate_trail_chain_cohomology_and_joint_log_prob, fold_fractal_nodes_words_64,
    is_slot_pair_coupled, resolve_dyadic_qdpt_and_cascade_ptr, slot_coupling_bit_index,
    trail_phase_words_mut, trail_phase_words_ref, with_slot_pair_coupled,
    with_slot_pair_coupled_symmetric, ActionHub, ActionHubEngine, ActionOrchestrationReport,
    AttractorAnchor, AttractorFieldEngine, AttractorGrowthReport, AttractorGrowthSnapshot,
    CognitiveCircuitActivation, CognitiveCircuitBank, CognitiveCircuitBankOps, CognitiveOpKind,
    CognitiveStepOp, CognitiveSubstrate, CognitiveTrajectoryLedger, ContrastivePhaseBloomFilter,
    CrystallizedAnchorRecord, CrystallizedAttractorRecord, DissipativeOverflowReport,
    DomainAccumulatorSummary, EpisodicPhaseSample, EpisodicSampleSteering, FastPathLedger,
    HighCapCognitiveBank, LowSpecCognitiveBank, MultiAttractorSpace, PhaseSteeringEngine,
    PheromoneDensityOps, PointerActionHub, RelativeAttractorNode, RelativeCouplingMatrix,
    ResidualAdapterEngine, StandardCognitiveBank, StigmergicActionDecision,
    StigmergicFragmentEnsembleReport, StigmergicGranuleEngine, StigmergicGranuleField,
    StigmergicOverlapSerendipityReport, StigmergicReconstitutionReport, StigmergicScaleGranularity,
    StigmergicThresholds, WorkbenchLanePlacement, WorkbenchPreset, WorkbenchSlotSpec,
    ACTION_HUB_INLINE_TRAILS, ATTRACTOR_SLOT_CRYSTALLIZED, ATTRACTOR_SLOT_KNOWLEDGE,
    ATTRACTOR_SLOT_PERSONA, ATTRACTOR_SLOT_REPULSION, ATTRACTOR_SLOT_SYSTEM_BASELINE,
    ATTRACTOR_SLOT_TEMPORAL, ATTRACTOR_SLOT_USER_CONTEXT, ATTRACTOR_SLOT_USER_INTENT,
    CAUSAL_AXIS_CONTRADICTION_GUARD, CAUSAL_AXIS_OBJECTIVE_FACT, CAUSAL_AXIS_SUBJECTIVE_PERSONA,
    CAUSAL_AXIS_SYSTEM_CONSTRAINT, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
    CAUSAL_CAUSE_COHOMOLOGY_REPULSION, CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK,
    CAUSAL_CAUSE_PERSONA_FACT_COLLISION, CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE,
    DEFAULT_ACTION_HUB_CAPACITY, DEFAULT_FACT_PHASE_CLAMP_RAD,
    DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO, DEFAULT_PERSONA_PHASE_CLAMP_RAD,
    DEFAULT_SLOT_COUPLING_MATRIX_8X8, DEFAULT_STIGMERGIC_CONTRADICTION_THRESH,
    DEFAULT_STIGMERGIC_CRYSTALLIZE_THRESH, DEFAULT_STIGMERGIC_EVAPORATION_RETAIN,
    DEFAULT_STIGMERGIC_EXPRESS_STEER_THRESH, DEFAULT_STIGMERGIC_INERTIAL_SKIP_ENTROPY,
    DEFAULT_STIGMERGIC_QUORUM_THRESHOLD, DEFAULT_STIGMERGIC_RANK1_INJECT_THRESH,
    DEFAULT_STIGMERGIC_REINFORCEMENT_GAIN, DEFAULT_STIGMERGIC_SUBHEAD_ROUTE_THRESH,
    DEFAULT_STIGMERGIC_TOLERANCE_Q7, DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE,
    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE, FAST_PATH_SHORTCUT_SLOTS, GCSO_CRYSTAL_RECORD_MAGIC,
    GCSO_CRYSTAL_RECORD_VERSION, GCSO_SNAPSHOT_MAGIC_U32, LEDGER_FLAG_COGNITIVE_TRAJECTORY,
    LEDGER_HISTORY_LEN, MAX_ANCHORS_PER_FIELD,
    MAX_ATTRACTOR_DOMAINS as STIGMERGY_MAX_ATTRACTOR_DOMAINS, MAX_EMBEDDING_DIM,
    ROOT_CAUSE_COHOMOLOGY_REPULSION, ROOT_CAUSE_FACT_RETRIEVAL_MISS, ROOT_CAUSE_NOMINAL,
    ROOT_CAUSE_PERSONA_FACT_COLLISION, ROOT_CAUSE_SYSTEM_QOS_THROTTLE,
    TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE,
    TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_PHASE_STEER_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE,
    TRAIL_FLAG_SWARM_ESCALATED, TRAIL_FLAG_SWARM_RESOLVED, TRAIL_FLAG_SYNERGY_RESONANCE,
    WORKBENCH_LANE_AUTO, WORKBENCH_LANE_COUNT, WORKBENCH_LANE_FLAG_PINNED,
    WORKBENCH_LANE_FLAG_REPULSION, WORKBENCH_LANE_FLAG_SANDBOX,
};

pub use crate::drivers::{
    AnonymousComputePool, ComputeDispatchMask, ComputePoolConfig, ComputePressureRegime,
    DecodingGate, DistortionDelta, DistortionTension, DistortionTensionStateCode,
    EgoDifferentialNet, EgoIntentAxes, EntropyController, EntropyEvaluator, LayerDistortionKind,
    LayerDistortionReport, LiminalEgoField, LiminalEgoPreset, SystemPressureReading,
    SystemSpineState, TokenBypassObservation, EGO_TRIGGER_ALL_DEFAULT, EGO_TRIGGER_BASIN_CAPTURE,
    EGO_TRIGGER_BOUNDARY_CHATTER, EGO_TRIGGER_BOUNDARY_STAGNATION, EGO_TRIGGER_INTERNAL_CONFLICT,
    EGO_TRIGGER_LAYER_DISTORTION, EGO_TRIGGER_LIMINAL_EXTERIOR, SPINE_BYPASS_VETO_GUARD_ACTIVE,
    SPINE_BYPASS_VETO_NONE, SPINE_BYPASS_VETO_PINNED_LOCK, SPINE_BYPASS_VETO_REPULSION_ACTIVE,
};

pub use crate::runtime::{
    abi_version_string, apply_gateway_frame_rank1_hidden_fused, apply_latent_horizon_slots,
    decode_and_apply_dnp_packet_to_trail, decode_and_verify_dnp_packet,
    decode_and_verify_dnp_words, dispatch_dnp_packet, encode_dnp_phase_packet,
    evaluate_epistemic_horizon_words_64, finalize_snapshot_header_crc32, init_default_config,
    init_descriptor_header, init_snapshot_header, project_gateway_signal_at_scale,
    query_capability, synthesize_multi_origin_gateway_frame,
    synthesize_multi_origin_gateway_frame_with_words,
    synthesize_relative_gateway_from_perspective_words_64, validate_config_invariants,
    validate_descriptor_header, validate_snapshot_header, verify_snapshot_payload_crc32,
    ActionHubGatewayExt, ActionHubScaleOrchestrator, AdaptiveThresholdPolicy,
    AutonomousCoreLayerStack, CausalProvenanceReport, CausalProvenanceSignal,
    CognitiveCircuitThresholdPolicy, CognitiveGroundingSignal, Config, ContradictionGuard,
    ContradictionGuardSignal, CoreLayerEvalContext, CoreTelemetryBank, CoreTelemetrySnapshot,
    DefaultResidualSpilloverSlot, DetailedLayerTelemetry, EgoLayerTelemetry,
    EntropyClosedLoopCoordinator, EpistemicHorizonReading, EpistemicManifoldOps, GatewayFrame,
    GatewayPhaseSteeringExt, GcsoFourSectionView, GuardPhaseAnchor, HighCapPhaseKvCache,
    HighCapPhaseKvGopCache, HorizonDwellState, HorizonEscalationProvider, HorizonEscalationRequest,
    HorizonTriggerInput, HybridScaleCircuit, KvKeyframeCache, KvKeyframePacket, LatentHorizonSlot,
    LiminalBoundaryDelta, LiminalHorizonNodeDescriptor, LiminalLayerTelemetry,
    LiminalTopologyTelemetrySnapshot, LowSpecPhaseKvCache, LowSpecPhaseKvGopCache,
    ManifoldGatewaySignal, ManifoldProbeBaseline, MappedStorage, MappedStorageEngine,
    ObjectiveFact, ObjectiveFactSignal, OrchestratorStatusSnapshot, PhaseSteeringClosedLoop,
    RamPhaseKvCache, RamPhaseKvCacheOps, RamPhaseKvEntry, RamPhaseKvGopCache,
    RamPhaseKvStepOutcome, ResidualSpilloverHorizonSlot, ResonantSignalSlice, Result,
    RuntimeContext, ScaleLayerNode, ScaleLayerProjection, SlowLoopTelemetryAdjustment,
    StandardPhaseKvCache, StandardPhaseKvGopCache, StigmergicActionReceptor,
    StigmergicLayerTelemetry, StigmergicScaleLayer, StigmergicThresholdPolicy, SubjectivePersona,
    SubjectivePersonaSignal, SystemConstraint, SystemConstraintSignal, SystemSpineTelemetry,
    TelemetryCapturePolicy, TelemetryVerbosityMode, CHATTER_FIRE_THRESHOLD_U8, CORE_LAYER_EGO,
    CORE_LAYER_LIMINAL, CORE_LAYER_STIGMERGIC, DEFAULT_BLOOM_REPULSION_SCALE_U7,
    DEFAULT_BLOOM_REPULSION_THRESHOLD, DEFAULT_ENTROPY_GATE_BIFURCATION_THRESHOLD,
    DEFAULT_ENTROPY_GATE_HYSTERESIS_BAND, DEFAULT_ENTROPY_GATE_TAU_FLUX, DEFAULT_PAGED_BLOCK_SIZE,
    DEFAULT_PERSONA_FACT_COLLISION_THRESH, DEFAULT_SLOT_WEIGHTS, GCSO_FOUR_SECTION_HEADER_BYTES,
    GCSO_LIMINAL_HORIZON_TOPOLOGY, GCSO_MAPPED_STORAGE_PREAMBLE_OFFSET, NUM_CORE_AUTONOMOUS_LAYERS,
    SLOT_PHASE_BANK_LEN,
};

pub use crate::runtime::abi::{
    gcso_anchor_type_t, gcso_attribution_tag_t, gcso_capability_flags_t,
    gcso_cognitive_substrate_t, gcso_config_t, gcso_descriptor_header_t, gcso_distortion_delta_t,
    gcso_distortion_tension_t, gcso_edbc_state_t, gcso_kv_keyframe_header_t, gcso_ledger_slot_t,
    gcso_liminal_venn_partition_t, gcso_mapped_storage_descriptor_t, gcso_paged_bitmask_t,
    gcso_pointer_trail_t, gcso_q7_t, gcso_rank1_residual_descriptor_t, gcso_snapshot_header_t,
    gcso_status_t, gcso_subhead_config_t, gcso_system_spine_state_t, gcso_tagged_ptr_t,
    FfiHandleMut, FfiHandleRef, GcsoAnchorType, GCSO_CAP_CPU_BASELINE, GCSO_CAP_CUDA_BACKEND,
    GCSO_CAP_METAL_BACKEND, GCSO_CAP_VULKAN_BACKEND,
};

// ===================================================================
// Convenient Prelude Module
// ===================================================================

/// Convenient prelude importing essential types, traits, engines, kernels, and FFI wrappers
/// across all 5 architectural layers (`liminal`, `foundation`, `stigmergy`, `drivers`, `runtime`).
pub mod prelude {
    pub use crate::drivers::*;
    pub use crate::foundation::{
        branchless_atan2_to_q7_fast, branchless_crc32_ieee, branchless_flush_subnormal_f32,
        branchless_inv_pos_f32, branchless_phase_coherence_inv_temp_scale,
        branchless_phase_proximity_gravity_f32, branchless_project_f32_slice_to_q7_64,
        branchless_q7_coherence_64, branchless_q7_coherence_and_resultant_64,
        branchless_q7_fold_s1_geodesic, branchless_q7_gravity_fold_64, branchless_q7_inhibit_64,
        branchless_q7_inhibit_words_64, branchless_q7_resonant_pull_64,
        branchless_q7_synergy_fuse_64, branchless_q7_synergy_fuse_words_64,
        branchless_q7_transition_log_prob_64, branchless_q7_transition_log_prob_words_64,
        branchless_q7_unified_field_fold_64, branchless_q7_unified_field_fold_with_init_weight_64,
        branchless_q7_unified_field_fold_words_with_init_weight_64,
        branchless_rad_to_phase_step_u8, branchless_sanitize_z_score_f32,
        branchless_sincos_half_pi_f32, branchless_sincos_q7_geodesic_phase_clamp_default_lut,
        branchless_sincos_q7_phase_clamp, branchless_sincos_q7_phase_clamp_default_lut,
        branchless_sincos_q7_phase_clamp_preinv, branchless_toroidal_centroid_step_words_64,
        branchless_update_crc32_ieee, branchless_von_mises_kappa, circular_r_bar_from_words_64,
        circular_resultant_from_words_64, coherence_from_diff_words_64,
        coherence_from_word_arrays_64, compute_crc32_ieee, compute_tikhonov_chebyshev_coeffs_deg3,
        detect_frame_delta_bitmask, detect_frame_mse_bitmask,
        dispatch_phase_clamp_and_subhead_slot_ensemble_fused, dispatch_single_pass,
        dispatch_subhead_slot_ensemble, dispatch_subhead_slot_ensemble_geodesic,
        eval_chebyshev_spectral_filter_deg3, eval_gershgorin_lambda_max,
        eval_pseudo_ultrametric_distance, eval_rank1, eval_rank1_fused,
        eval_rank1_pointer_gated_fused, eval_rank1_slices, eval_rank1_slices_fused,
        evaluate_cohomological_obstruction, evaluate_hodge_obstruction_and_dual_pairing,
        is_phase_profile_nonzero_64, is_word_array_uniform_64, load_q7_words_64,
        pack_action_slot_tag, phase_clamp_upper_rot_dim, project_bytes_swar_to_q7_64,
        project_bytes_swar_to_words_64, project_f32_slice_to_q7_64, project_f32_slice_to_words_64,
        project_lora, project_lora_with_block_scales, raw_byte_ranges_overlap,
        resolve_autonomous_swarm_rot_dim, steer_visual_grid_2d_bitmask_gated,
        steer_visual_grid_3d_bitmask_gated, store_q7_words_64, sum_abs_and_sq_f32_simd8,
        sum_abs_sq_and_max_f32_simd8, sum_sq_f32_simd8, swar_q7_circular_add_x8,
        swar_q7_fold_s1_geodesic_x8, swar_q7_mean_i8_from_words_64,
        swar_q7_phase_step_filter_words_64, swar_q7_scale_le128_with_bias_x8,
        swar_slerp_word_unchecked_x8, swar_slerp_word_with_bias_x8, swar_slerp_words_with_bias_64,
        swar_u8_add_sat_x8, swar_u8_halve_decay_x8, swar_u8_max_x8, swar_u8_mean_round_x8,
        swar_u8_nonzero_count_x8, swar_u8_sub_sat_x8, swar_u8_sum_x8, unpack_action_mode_from_tag,
        unpack_attractor_slot_from_tag, update_crc32_ieee, validate_subhead_layout,
        AlignedBuffer128, AlignedBuffer16, AlignedBuffer256, AlignedBuffer32, AlignedBuffer512,
        AlignedBuffer64, AlignedSlice, AlignedSlice128, AlignedSlice16, AlignedSlice256,
        AlignedSlice32, AlignedSlice512, AlignedSlice64, AlignedSliceMut, AlignedSliceMut128,
        AlignedSliceMut16, AlignedSliceMut256, AlignedSliceMut32, AlignedSliceMut512,
        AlignedSliceMut64, DisjointMutPair, DisjointMutTriple, DisjointRefMut, KvMotionDeltaPacket,
        PhaseSteering, Pod, RawPhaseProjectable, RopeGeometryDescriptor, RopePairLayout,
        RopeSteeringTarget, ACTION_MODE_LAYER_SKIP, ACTION_MODE_PHASE_REPULSE,
        ACTION_MODE_PHASE_STEER, ACTION_MODE_RANK1_INJECT, ACTION_MODE_SUBHEAD_ROUTE,
        ACTION_PRESET_EXPRESS_PHASE, ACTION_PRESET_FULL_ORCHESTRATION, ACTION_PRESET_INERTIAL_SKIP,
        ACTION_PRESET_PASSIVE_SIDECAR, ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_ID_MASK,
        CHEBYSHEV_COEFF_LEN, DEFAULT_MAX_VALID_Z_SCORE, DEFAULT_PHASE_CLAMP_MAX_RAD,
        DEFAULT_PHASE_CLAMP_Q7_TO_NORM_SCALE, DEFAULT_PHASE_GRAVITY_EPS,
        DEFAULT_PHASE_STEP_MIN_RAD, F32_MIN_POSITIVE_NORMAL, MAX_PHASE_INTERFERENCE_INV_TEMP,
        MIN_PHASE_INTERFERENCE_INV_TEMP, PAGED_BITMASK_BITS, TAG_ACTION_MODE_MASK,
    };
    pub use crate::liminal::*;
    pub use crate::runtime::abi::{
        gcso_anchor_type_t, gcso_capability_flags_t, gcso_config_t, gcso_descriptor_header_t,
        gcso_edbc_state_t, gcso_kv_keyframe_header_t, gcso_ledger_slot_t,
        gcso_mapped_storage_descriptor_t, gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_q7_t,
        gcso_rank1_residual_descriptor_t, gcso_snapshot_header_t, gcso_status_t,
        gcso_subhead_config_t, gcso_tagged_ptr_t, FfiHandleMut, FfiHandleRef, GcsoAnchorType,
        GCSO_ABI_VERSION_HEX, GCSO_ABI_VERSION_STRING,
    };
    pub use crate::runtime::{
        abi_version_string, apply_gateway_frame_rank1_hidden_fused, apply_latent_horizon_slots,
        decode_and_apply_dnp_packet_to_trail, decode_and_verify_dnp_packet,
        decode_and_verify_dnp_words, dispatch_dnp_packet, encode_dnp_phase_packet,
        evaluate_epistemic_horizon_words_64, finalize_snapshot_header_crc32, init_default_config,
        init_descriptor_header, init_snapshot_header, project_gateway_signal_at_scale,
        query_capability, synthesize_multi_origin_gateway_frame,
        synthesize_multi_origin_gateway_frame_with_words,
        synthesize_relative_gateway_from_perspective_words_64, validate_config_invariants,
        validate_descriptor_header, validate_snapshot_header, verify_snapshot_payload_crc32,
        ActionHubGatewayExt, ActionHubScaleOrchestrator, AdaptiveThresholdPolicy,
        AutonomousCoreLayerStack, CausalProvenanceReport, CausalProvenanceSignal,
        CognitiveCircuitThresholdPolicy, CognitiveGroundingSignal, Config, ContradictionGuard,
        ContradictionGuardSignal, CoreLayerEvalContext, CoreTelemetryBank, CoreTelemetrySnapshot,
        DefaultResidualSpilloverSlot, DetailedLayerTelemetry, EgoLayerTelemetry,
        EntropyClosedLoopCoordinator, EpistemicHorizonReading, EpistemicManifoldOps, GatewayFrame,
        GatewayPhaseSteeringExt, GcsoFourSectionView, GuardPhaseAnchor, HighCapPhaseKvCache,
        HighCapPhaseKvGopCache, HorizonDwellState, HorizonEscalationProvider,
        HorizonEscalationRequest, HorizonTriggerInput, HybridScaleCircuit, KvKeyframeCache,
        KvKeyframePacket, LatentHorizonSlot, LiminalBoundaryDelta, LiminalHorizonNodeDescriptor,
        LiminalLayerTelemetry, LiminalTopologyTelemetrySnapshot, LowSpecPhaseKvCache,
        LowSpecPhaseKvGopCache, ManifoldGatewaySignal, ManifoldProbeBaseline, MappedStorage,
        MappedStorageEngine, ObjectiveFact, ObjectiveFactSignal, OrchestratorStatusSnapshot,
        PhaseSteeringClosedLoop, RamPhaseKvCache, RamPhaseKvCacheOps, RamPhaseKvEntry,
        RamPhaseKvGopCache, RamPhaseKvStepOutcome, ResidualSpilloverHorizonSlot,
        ResonantSignalSlice, Result, RuntimeContext, ScaleLayerNode, ScaleLayerProjection,
        SlowLoopTelemetryAdjustment, StandardPhaseKvCache, StandardPhaseKvGopCache,
        StigmergicActionReceptor, StigmergicLayerTelemetry, StigmergicScaleLayer,
        StigmergicThresholdPolicy, SubjectivePersona, SubjectivePersonaSignal, SystemConstraint,
        SystemConstraintSignal, SystemSpineTelemetry, TelemetryCapturePolicy,
        TelemetryVerbosityMode, CHATTER_FIRE_THRESHOLD_U8, CORE_LAYER_EGO, CORE_LAYER_LIMINAL,
        CORE_LAYER_STIGMERGIC, DEFAULT_PAGED_BLOCK_SIZE, DEFAULT_PERSONA_FACT_COLLISION_THRESH,
        GCSO_FOUR_SECTION_HEADER_BYTES, GCSO_LIMINAL_HORIZON_TOPOLOGY,
        GCSO_MAPPED_STORAGE_PREAMBLE_OFFSET, NUM_CORE_AUTONOMOUS_LAYERS,
    };
    pub use crate::stigmergy::*;
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
        crate::runtime::abi::GCSO_ABI_VERSION_MAJOR,
        crate::runtime::abi::GCSO_ABI_VERSION_MINOR,
        crate::runtime::abi::GCSO_ABI_VERSION_PATCH,
    )
}

/// Returns the hexadecimal ABI version identifier ([`crate::runtime::abi::GCSO_ABI_VERSION_HEX`]).
#[inline(always)]
#[must_use]
pub const fn abi_version_hex() -> u32 {
    crate::runtime::abi::GCSO_ABI_VERSION_HEX
}
