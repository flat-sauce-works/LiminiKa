// name: src/core/src/lib.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core execution engine library for `LiminiKa` (`v0.3.0`, `0x0000_0300`).
//!
//! This crate serves as the primary `#![no_std]` runtime engine for `LiminiKa`, implementing
//! non-invasive LLM steering and low-VRAM (2-4 GB) orchestration across a Y-shaped 5-layer
//! unidirectional DAG (`liminal` || `foundation` -> `stigmergy` -> `drivers/{spine,ego}` -> `runtime`).
//!
//! # Architecture: `2 Primitives x 2 Drivers` over the 5-Layer DAG
//! - **Level 0 — `liminal` ([`crate::liminal`])**: Boundary axiom (`liminal::axiom`),
//!   4-state `Lacuna` classification (`LacunaPartition8`), `BitCut<T>`, and
//!   `LiminalThresholdState` gating out-of-horizon steering cleanly to zero without data-dependent branches.
//! - **Level 1 — `foundation` ([`crate::foundation`])**: Environment-agnostic pure geometry,
//!   Portable Little-Endian POD layouts (`0` implicit padding, `16 B..=128 B`, `u32x16`/`u64x8`
//!   device-ready phase profiles), aligned-slice wrappers, SWAR baseline and AVX2 leaf kernels,
//!   CRC32/trig lookup tables, RoPE phase steering, Rank-1 residual injection, LoRA phase
//!   projection, and matrix-free sheaf cohomology kernels.
//! - **Level 2 — `stigmergy` ([`crate::stigmergy`])**: Passive carrier medium (`StigmergicSubstrate`)
//!   with 8-lane working bank, 8-granule quorum, contrastive Bloom filters, `.lmnk` phase anchor
//!   crystallization (`stigmergy::anchor`, `"LNKR"`), concrete `PhaseSteeringEngine` /
//!   `ResidualAdapterEngine`, and [`crate::stigmergy::StigmergicSubstrateOps`] spanning the
//!   widest threshold mask (`THRESHOLD_STIGMERGY_WIDE`).
//! - **Level 3 — `drivers` ([`crate::drivers`])**: Orthogonal `spine` (`SystemSpineState`,
//!   `EntropyController`, `AnonymousComputePool`) and `ego` (`EgoDifferentialNet`, `LiminalEgoState`)
//!   forces isolated behind dual `DistortionTension` / `DistortionDelta`.
//! - **Level 4 — `runtime` ([`crate::runtime`])**: Single orchestrator facade (`RuntimeContext`,
//!   `#[repr(C, align(64))]`, caller-owned `in_place` initialization), Manifold Gateway
//!   (`ManifoldGatewaySignal`, `GatewayFrame`), Latent Horizon (`LatentHorizonSlot`),
//!   `.lmnk` snapshot storage, telemetry, and the C11/C++20 `v0.3.0` (`lmnk_*`) C-ABI firewall (`abi`).
//!
//! # Core Mathematical & Systems Design Principles
//! 1. **Sequential Conditional Transition Probability x Attention Phase Interference**:
//!    Long-horizon persona and context consistency $P(M_1, \dots, M_T) = \prod_{t=1}^T P(M_t \mid M_{<t})$
//!    is factored into local conditional transitions modulated in $\mathcal{O}(1)$ via RoPE phase
//!    steering without modifying host model weights $W_Q, W_K$.
//! 2. **Fractal Self-Similarity & Indirect Pointer-Chain Routing on `StigmergicSubstrate`**:
//!    From 128-byte pointer trails (`PointerTrail`), 128-byte `GatewayFrame` /
//!    `PhaseAnchorRecord` basins, up to nested composites `[Node; K]` and `(A, B)`,
//!    every node implements [`crate::stigmergy::RelativeAttractorNode`] with recursive
//!    $SO(2)^{64}$ resultant folding (`fold_fractal_nodes_words_64`), dissipative overflow scatter /
//!    autonomous daughter crystallization (`DissipativeOverflowReport`), and perspective-flipped signed
//!    coupling (`RelativeCouplingMatrix`).
//! 3. **`Liminal`, Dynamic `LiminalThresholdState`, & Epistemic Horizon Routing**:
//!    Queries inside carved memory horizons separate coherent grounding ($H^0$) from internal
//!    opposition ($H^1 \neq 0$), while out-of-horizon queries enter `EpistemicRegime::Liminal`.
//!    [`crate::liminal::LiminalThresholdState`] shifts per-channel thresholds branchlessly and
//!    exposes live topology introspection via [`crate::runtime::LiminalTopologyTelemetrySnapshot`].
//! 4. **Single Unified Attractor Manifold**:
//!    All anchors coexist on a single unified phase manifold ($SO(2)^{d_{\mathrm{head}}/2}$)
//!    governed by phase-proximity gravity (`branchless_q7_resonant_pull_64`), with an optional
//!    8-lane working cache (`slot_phase_bank`) for $\mathcal{O}(1)$ tagged-pointer addressing.

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
#![warn(clippy::undocumented_unsafe_blocks)]
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
/// contrastive Bloom filters, fast-path ledgers, 8-lane [`StigmergicSubstrate`], cognitive circuits,
/// `.lmnk` phase anchor records, and manifold attraction.
pub mod stigmergy;

/// Level 3 (`Drivers`): Dual execution axes (`spine` vs. `ego`) isolated behind `distortion`.
pub mod drivers;

/// Level 4 (`Runtime`): Stateful orchestration (`RuntimeContext`), telemetry, Manifold Gateway,
/// Latent Horizon, `.lmnk` storage, and C-ABI translation firewall (`abi`).
pub mod runtime;

/// C-ABI `v0.3.0` (`0x0000_0300`) bindings, FFI safety barriers, binary layout contracts, and status codes.
pub use crate::runtime::abi;

// ===================================================================
// Primary Crate-Root Re-exports Across All 5 Layers
// ===================================================================

pub use crate::liminal::{
    branchless_select_f32, branchless_select_i32, branchless_select_i64, branchless_select_i8,
    branchless_select_q7, branchless_select_u16, branchless_select_u32, branchless_select_u64,
    branchless_select_u8, branchless_select_usize, evaluate_lacuna_partition_branchless,
    is_compatible_abi_version, pack_attribution_tag, sensitivity_4p1mp_f32, sensitivity_4p1mp_q7,
    validate_abi_version, AbstractDomain, Activatable, Active, AttributionTag, BitCut, BitCut32,
    BitCut64, BitCut8, Bottom, BoundaryViolation, BridgedDiscovery, Cuttable, Definite, DriverKind,
    EpistemicRegime, Exterior, FromBoundary, FromDelta, Groundable, Grounded, GroundedOrigin,
    Indivisible, IndivisiblePhase, Interior, Lacuna, LacunaPartition8, LiminalThresholdState,
    LmnkResult, LmnkStatus, PhaseQ7, PrimitiveKind, Refine, RuntimeConfig, Scoped, ScopedCut,
    Shared, SpeculativeDiscovery, Unmarked, VerifiedAligned, DEFAULT_ENTROPY_EPS,
    HEADS_PER_STIGMERGIC_GRANULE, LMNK_ABI_COMPAT_MASK, LMNK_ABI_COMPAT_VERSION_HEX,
    LMNK_ABI_VERSION_CSTR, LMNK_ABI_VERSION_HEX, LMNK_ABI_VERSION_MAJOR, LMNK_ABI_VERSION_MINOR,
    LMNK_ABI_VERSION_PATCH, LMNK_ABI_VERSION_STRING, LMNK_CAP_INPLACE_CONTEXT_INIT,
    LMNK_CAP_SIMD_AVX2, LMNK_CAP_SIMD_NEON, LMNK_CAP_SIMD_WASM128, LMNK_CAP_SWAR_BASELINE,
    LMNK_CAP_ZERO_ALLOC_CORE, LMNK_ERR_TOPOLOGY_MISMATCH, LMNK_OK, MAX_ATTENTION_HEADS,
    MAX_ATTRACTOR_DOMAINS, MAX_SURROGATE_ENTROPY_LN2, NUM_LIMINAL_THRESHOLD_CHANNELS,
    NUM_STIGMERGIC_GRANULES, Q7_PHASE_INV_SCALE_RAD, Q7_PHASE_SCALE_RAD, SCOPE_REGION_EXTERIOR,
    SCOPE_REGION_INTERIOR, SCOPE_REGION_SELF_REF, THRESHOLD_CAUSAL_PROVENANCE,
    THRESHOLD_COGNITIVE_CIRCUIT, THRESHOLD_COHOMOLOGY_ENTROPY, THRESHOLD_CONTRADICTION_GUARD,
    THRESHOLD_DISSIPATIVE_ANCHOR, THRESHOLD_FAST_PATH_LEDGER, THRESHOLD_KV_PHASE_CACHE,
    THRESHOLD_MANIFOLD_GATEWAY, THRESHOLD_MAPPED_STORAGE, THRESHOLD_OBJECTIVE_FACT,
    THRESHOLD_PHASE_STEERING, THRESHOLD_POINTER_TRAIL, THRESHOLD_RANK1_RESIDUAL,
    THRESHOLD_RELATIVE_COUPLING, THRESHOLD_STIGMERGIC_CONSENSUS, THRESHOLD_STIGMERGY_WIDE,
    THRESHOLD_SUBHEAD_ROUTING, THRESHOLD_SUBJECTIVE_PERSONA, THRESHOLD_SYSTEM_CONSTRAINT,
};

pub use crate::foundation::{
    branchless_abs_f32, branchless_argmin_f32_x8_masked, branchless_argmin_u8_x8_masked,
    branchless_atan2_f32, branchless_atan2_to_q7_fast, branchless_clamp_f32,
    branchless_copysign_f32, branchless_cos_q7, branchless_crc32_128b_words, branchless_crc32_ieee,
    branchless_exp_f32, branchless_eyring_kramers_escape, branchless_fast_transition_log_prob,
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
    branchless_q7_unified_manifold_fold_64,
    branchless_q7_unified_manifold_fold_with_init_weight_64,
    branchless_q7_unified_manifold_fold_words_with_init_weight_64,
    branchless_q7_weighted_ensemble_contiguous_words_64, branchless_q7_weighted_ensemble_words_64,
    branchless_rad_to_phase_step_u8, branchless_rad_to_q7, branchless_relu_f32,
    branchless_rotate_so2_pair, branchless_rotate_so2_pair_scaled, branchless_rsqrt_f32,
    branchless_sanitize_z_score_f32, branchless_sigmoid_f32, branchless_signum_f32,
    branchless_sin_q7, branchless_sincos_f32, branchless_sincos_half_pi_f32, branchless_sincos_q7,
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
    unpack_attractor_slot_from_tag, update_crc32_ieee, validate_raw_slice_bounds,
    validate_subhead_layout, AlignedBuffer128, AlignedBuffer16, AlignedBuffer256, AlignedBuffer32,
    AlignedBuffer512, AlignedBuffer64, AlignedSlice, AlignedSlice128, AlignedSlice16,
    AlignedSlice256, AlignedSlice32, AlignedSlice512, AlignedSlice64, AlignedSliceMut,
    AlignedSliceMut128, AlignedSliceMut16, AlignedSliceMut256, AlignedSliceMut32,
    AlignedSliceMut512, AlignedSliceMut64, DisjointMutPair, DisjointMutTriple, DisjointRefMut,
    KvMotionDeltaPacket, PagedBitmask, PhaseSteering, Pod, Rank1Descriptor, RawPhaseProjectable,
    RopeGeometryDescriptor, RopePairLayout, RopeSteeringTarget, SubheadConfig, TaggedPtr,
    ACTION_MODE_LAYER_SKIP, ACTION_MODE_PHASE_REPULSE, ACTION_MODE_PHASE_STEER,
    ACTION_MODE_RANK1_INJECT, ACTION_MODE_SUBHEAD_ROUTE, ACTION_PRESET_EXPRESS_PHASE,
    ACTION_PRESET_FULL_ORCHESTRATION, ACTION_PRESET_INERTIAL_SKIP, ACTION_PRESET_PASSIVE_SIDECAR,
    ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_ID_MASK, CHEBYSHEV_COEFF_LEN,
    CHEBYSHEV_FILTER_DEGREE, DEFAULT_MAX_VALID_Z_SCORE, DEFAULT_PHASE_CLAMP_MAX_RAD,
    DEFAULT_PHASE_CLAMP_Q7_TO_NORM_SCALE, DEFAULT_PHASE_GRAVITY_EPS, DEFAULT_PHASE_STEER_HEAD_DIM,
    DEFAULT_PHASE_STEER_NUM_HEADS, DEFAULT_PHASE_STEP_MIN_RAD, F32_MIN_POSITIVE_NORMAL,
    INV_PARALLEL_COUNT_Q7, MAX_PHASE_INTERFERENCE_INV_TEMP, MIN_PHASE_INTERFERENCE_INV_TEMP,
    PAGED_BITMASK_BITS, PAGED_BITMASK_WORDS, SWAR_EVEN_BYTES_64, SWAR_LOW7_MASK_64,
    SWAR_ODD_BYTES_64, SWAR_SIGN_MASK_64, SWAR_SPLAT_BYTE_64, SWAR_SUM_U16_64,
    TAG_ACTION_MODE_MASK, TAG_ATTRACTOR_SLOT_SHIFT,
};

pub use crate::stigmergy::{
    bridge_and_filter_liminal_trail_chain, coupled_slot_count, coupled_slot_row_mask,
    detect_and_record_obstruction_repulsion, eval_stigmergic_block_degrees,
    evaluate_stigmergic_overlap_and_discovery_words_64,
    evaluate_trail_chain_cohomology_and_joint_log_prob, fold_fractal_nodes_words_64,
    is_slot_pair_coupled, quantize_persistent_topological_mode,
    resolve_dyadic_qdpt_and_cascade_ptr, slot_coupling_bit_index, trail_phase_words_mut,
    trail_phase_words_ref, with_slot_pair_coupled, with_slot_pair_coupled_symmetric,
    AttractorGrowthReport, CognitiveCircuitActivation, CognitiveCircuitBank,
    CognitiveCircuitBankOps, CognitiveOpKind, CognitiveStepOp, CognitiveTrajectoryLedger,
    ContrastivePhaseBloomFilter, CorePrimitiveState, DissipativeOverflowReport,
    EpisodicPhaseSample, EpisodicSampleSteering, FastPathLedger, HighCapCognitiveBank, LedgerSlot,
    LowSpecCognitiveBank, OverlapDiscoveryReport, PhaseAnchorRecord, PhaseDeltaSlice,
    PhaseProfile64, PhaseProfileRef64, PhaseSteeringActiveExt, PhaseSteeringEngine,
    PhaseWordsRef64, PheromoneDensityOps, PointerSubstrate, PointerTrail, RelativeAttractorNode,
    RelativeCouplingMatrix, ResidualAdapterEngine, ResidualScale, StandardCognitiveBank,
    StigmergicFragmentEnsembleReport, StigmergicGranuleEngine, StigmergicGranuleState,
    StigmergicReconstitutionReport, StigmergicScaleGranularity, StigmergicSubstrate,
    StigmergicSubstrateOps, StigmergicThresholds, SubstrateStepReport, WorkbenchLanePlacement,
    WorkbenchPreset, WorkbenchSlotSpec, ANCHOR_ATTRIBUTION_NOMINAL, ANCHOR_LINEAGE_MATURE_BIT,
    ATTRACTOR_SLOT_CRYSTALLIZED, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA,
    ATTRACTOR_SLOT_REPULSION, ATTRACTOR_SLOT_SYSTEM_BASELINE, ATTRACTOR_SLOT_TEMPORAL,
    ATTRACTOR_SLOT_USER_CONTEXT, ATTRACTOR_SLOT_USER_INTENT, DEFAULT_FACT_PHASE_CLAMP_RAD,
    DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO, DEFAULT_PERSONA_PHASE_CLAMP_RAD,
    DEFAULT_SLOT_COUPLING_MATRIX_8X8, DEFAULT_STIGMERGIC_CONTRADICTION_THRESH,
    DEFAULT_STIGMERGIC_CRYSTALLIZE_THRESH, DEFAULT_STIGMERGIC_EVAPORATION_RETAIN,
    DEFAULT_STIGMERGIC_EXPRESS_STEER_THRESH, DEFAULT_STIGMERGIC_INERTIAL_SKIP_ENTROPY,
    DEFAULT_STIGMERGIC_QUORUM_THRESHOLD, DEFAULT_STIGMERGIC_RANK1_INJECT_THRESH,
    DEFAULT_STIGMERGIC_REINFORCEMENT_GAIN, DEFAULT_STIGMERGIC_SUBHEAD_ROUTE_THRESH,
    DEFAULT_STIGMERGIC_TOLERANCE_Q7, DEFAULT_SUBSTRATE_CAPACITY,
    DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE, DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
    FAST_PATH_SHORTCUT_SLOTS, LEDGER_FLAG_COGNITIVE_TRAJECTORY, LEDGER_HISTORY_LEN,
    LMNK_PHASE_ANCHOR_MAGIC, LMNK_PHASE_ANCHOR_VERSION, MAX_EMBEDDING_DIM, MAX_PHASE_ANCHORS,
    SUBSTRATE_INLINE_TRAILS, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE,
    TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_PHASE_STEER_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE,
    TRAIL_FLAG_SWARM_ESCALATED, TRAIL_FLAG_SWARM_RESOLVED, TRAIL_FLAG_SYNERGY_RESONANCE,
    WORKBENCH_LANE_AUTO, WORKBENCH_LANE_COUNT, WORKBENCH_LANE_FLAG_PINNED,
    WORKBENCH_LANE_FLAG_REPULSION, WORKBENCH_LANE_FLAG_SANDBOX,
};

pub use crate::drivers::{
    AnonymousComputePool, ComputeDispatchMask, ComputePoolConfig, ComputePressureRegime,
    DistortionDelta, DistortionTension, DistortionTensionStateCode, EgoDifferentialNet,
    EgoIntentAxes, EntropyController, EntropyState, FrameDirective, FrameKind, LiminalEgoPreset,
    LiminalEgoState, PFrameColdMeta, PFrameRoutingMeta, SpineDecodeInput, SpineDecodeSignal,
    SpineFrameTransitionInput, SpineLogitsObserveMode, SpineStepPlan, SystemPressureReading,
    SystemSpineState, TokenBypassObservation, EGO_TRIGGER_ALL_DEFAULT, EGO_TRIGGER_BASIN_CAPTURE,
    EGO_TRIGGER_BOUNDARY_CHATTER, EGO_TRIGGER_BOUNDARY_STAGNATION, EGO_TRIGGER_INTERNAL_CONFLICT,
    EGO_TRIGGER_LAYER_DISTORTION, EGO_TRIGGER_LIMINAL_EXTERIOR, HOST_IMMUTABLE_PINNED_COMPUTE_MASK,
    SPINE_BYPASS_VETO_GUARD_ACTIVE, SPINE_BYPASS_VETO_KNOWN_MASK, SPINE_BYPASS_VETO_NONE,
    SPINE_BYPASS_VETO_PINNED_LOCK, SPINE_BYPASS_VETO_REPULSION_ACTIVE,
    SPINE_CONTROL_FLAG_PREV_RESIDENCY_VALID, SPINE_CONTROL_FLAG_PREV_WAS_ABSTRACT,
    SPINE_CONTROL_FLAG_PRE_BYPASS_ENABLED, SPINE_CONTROL_FLAG_TRANSIENT_GUARD_RELAXED,
    SPINE_CONTROL_FLAG_TURN_GUARD_ACTIVE, SPINE_CONTROL_KNOWN_FLAGS,
    SPINE_DECODE_FLAG_REWARD_OVERRIDE, SPINE_DECODE_INPUT_FLAG_GREEDY_SAMPLING,
    SPINE_DECODE_INPUT_FLAG_PRE_BYPASS_ENABLED, SPINE_DECODE_INPUT_FLAG_REPULSION_ACTIVE,
    SPINE_DECODE_INPUT_FLAG_REWARD_OVERRIDE, SPINE_ENTROPY_CALM_THRESH_Q15,
    SPINE_ENTROPY_GUARD_RELAX_MAX_Q15, SPINE_ENTROPY_Q15_ONE, SPINE_ENTROPY_REWARD_LOW_THRESH_Q15,
    SPINE_ENTROPY_SURGE_THRESH_Q15, SPINE_ENTROPY_TENSION_FLOOR_Q15,
    SPINE_MIN_DYNAMIC_OBSERVE_STRIDE, SPINE_SENTINEL_AUTO_U8, SPINE_STEP_EVENT_BOUNDARY_CHURN,
    SPINE_STEP_EVENT_GUARD_REESCALATED, SPINE_STEP_EVENT_INSTABILITY_RECOVERY,
    SPINE_STEP_EVENT_LOGITS_OBSERVE_SKIPPED, SPINE_STEP_EVENT_NONE,
    SPINE_STEP_EVENT_TRANSIENT_GUARD_RELAXED, SPINE_TRANSITION_FLAG_GUARD_REPULSION,
    SPINE_TRANSITION_FLAG_IRREVERSIBLE, SPINE_TRANSITION_FLAG_NEW_BASIN_SPAWNED,
    SPINE_TRANSITION_FLAG_STATIONARY_BYPASS_READY,
};

pub use crate::runtime::{
    abi_version_string, apply_gateway_frame_rank1_hidden_fused, apply_latent_horizon_slots,
    decode_and_apply_phase_packet_to_trail, decode_and_verify_phase_packet,
    decode_and_verify_phase_packet_words, dispatch_phase_packet, encode_phase_packet,
    evaluate_epistemic_horizon_words_64, init_container_header, init_default_config,
    init_descriptor_header, project_gateway_signal_at_scale, query_capability,
    required_container_byte_len, synthesize_multi_origin_gateway_frame,
    synthesize_multi_origin_gateway_frame_with_words,
    synthesize_relative_gateway_from_perspective_words_64, validate_config_invariants,
    validate_descriptor_header, write_container_sections_to_bytes, AdaptiveThresholdPolicy,
    CausalProvenanceReport, CausalProvenanceSignal, CognitiveCircuitThresholdPolicy,
    CognitiveGroundingSignal, Config, ContradictionGuard, ContradictionGuardSignal,
    CoreLayerEvalContext, CoreStateStack, CoreTelemetryBank, CoreTelemetrySnapshot,
    DefaultResidualSpilloverSlot, DescriptorHeader, DetailedLayerTelemetry, EgoTelemetry,
    EpistemicHorizonReading, EpistemicManifoldOps, GatewayFrame, GatewayPhaseSteeringExt,
    GuardPhaseAnchor, HighCapPhaseKvCache, HighCapPhaseKvGopCache, HorizonDwellState,
    HorizonEscalationProvider, HorizonEscalationRequest, HorizonTriggerInput, HybridScaleCircuit,
    KvKeyframeCache, KvKeyframeHeader, KvKeyframePacket, LatentHorizonSlot, LayerDistortionKind,
    LayerDistortionReport, LiminalBoundaryDelta, LiminalHorizonNodeDescriptor, LiminalTelemetry,
    LiminalTopologyMatrix, LiminalTopologyTelemetrySnapshot, LmnkContainerHeader,
    LmnkContainerView, LmnkSectionEntry, LmnkSectionInput, LmnkSectionKind, LowSpecPhaseKvCache,
    LowSpecPhaseKvGopCache, ManifoldGatewaySignal, ManifoldProbeBaseline, MappedStorage,
    MappedStorageEngine, ObjectiveFact, ObjectiveFactSignal, OrchestratorStatusSnapshot,
    RamPhaseKvCache, RamPhaseKvCacheOps, RamPhaseKvEntry, RamPhaseKvGopCache,
    RamPhaseKvStepOutcome, ResidualSpilloverHorizonSlot, ResonantSignalSlice, Result,
    RuntimeContext, ScaleLayerNode, ScaleLayerOrchestrator, ScaleLayerProjection,
    SlowLoopTelemetryAdjustment, StandardPhaseKvCache, StandardPhaseKvGopCache, StigmergicReceptor,
    StigmergicScaleLayer, StigmergicTelemetry, StigmergicThresholdPolicy, SubjectivePersona,
    SubjectivePersonaSignal, SubstrateGatewayExt, SystemConstraint, SystemConstraintSignal,
    SystemSpineTelemetry, TelemetryCapturePolicy, TelemetryVerbosityMode,
    CAUSAL_AXIS_CONTRADICTION_GUARD, CAUSAL_AXIS_OBJECTIVE_FACT, CAUSAL_AXIS_SUBJECTIVE_PERSONA,
    CAUSAL_AXIS_SYSTEM_CONSTRAINT, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
    CAUSAL_CAUSE_COHOMOLOGY_REPULSION, CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK,
    CAUSAL_CAUSE_PERSONA_FACT_COLLISION, CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE,
    CHATTER_FIRE_THRESHOLD_U8, CORE_TOPOLOGY, DEFAULT_BLOOM_REPULSION_SCALE_U7,
    DEFAULT_BLOOM_REPULSION_THRESHOLD, DEFAULT_ENTROPY_GATE_BIFURCATION_THRESHOLD,
    DEFAULT_ENTROPY_GATE_HYSTERESIS_BAND, DEFAULT_ENTROPY_GATE_TAU_FLUX, DEFAULT_PAGED_BLOCK_SIZE,
    DEFAULT_PERSONA_FACT_COLLISION_THRESH, DEFAULT_SLOT_WEIGHTS,
    LMNK_CONTAINER_FLAG_COMPILED_CACHE, LMNK_CONTAINER_FLAG_PORTABLE_PACKAGE,
    LMNK_CONTAINER_FLAG_RUNTIME_SNAPSHOT, LMNK_CONTAINER_HEADER_BYTES, LMNK_CONTAINER_KNOWN_FLAGS,
    LMNK_CONTAINER_MAGIC, LMNK_DAG_FLAG_AXIOM_CUT_LACUNA_ORTHOGONAL, LMNK_DAG_FLAG_NO_STD_CORE,
    LMNK_DAG_FLAG_SPINE_EGO_ISOLATED, LMNK_DAG_FLAG_Y_SHAPED_PARALLEL_ROOTS,
    LMNK_LIMINAL_HORIZON_TOPOLOGY, LMNK_MAX_CONTAINER_BYTES, LMNK_MAX_CONTAINER_SECTIONS,
    LMNK_RUNTIME_SNAPSHOT_PAYLOAD_BYTES, LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES,
    LMNK_SECTION_BLOOM_FILTER, LMNK_SECTION_CAUSAL_PROVENANCE, LMNK_SECTION_CONTENT_HASH,
    LMNK_SECTION_ENTRY_BYTES, LMNK_SECTION_LEDGER_SLOT, LMNK_SECTION_LOGICAL_PFRAMES,
    LMNK_SECTION_PHASE_ANCHORS, LMNK_SECTION_POINTER_TRAILS, LMNK_SECTION_RECALL_METADATA,
    LMNK_SECTION_RUNTIME_SNAPSHOT, LMNK_SECTION_SIGNATURE, LMNK_TOPOLOGY_CHANNEL_COUNT,
    LMNK_TOPOLOGY_DAG_EDGE_MATRIX, LMNK_TOPOLOGY_DAG_FLAGS_CANONICAL, LMNK_TOPOLOGY_LAYER_COUNT,
    LMNK_TOPOLOGY_MAGIC, LMNK_TOPOLOGY_PINNED_CHANNEL_MASK, LMNK_TOPOLOGY_SANDBOX_CHANNEL_MASK,
    ROOT_CAUSE_COHOMOLOGY_REPULSION, ROOT_CAUSE_FACT_RETRIEVAL_MISS, ROOT_CAUSE_NOMINAL,
    ROOT_CAUSE_PERSONA_FACT_COLLISION, ROOT_CAUSE_SYSTEM_QOS_THROTTLE, SLOT_PHASE_BANK_LEN,
};

pub use crate::runtime::abi::{
    lmnk_action_decision_t, lmnk_attribution_tag_t, lmnk_bloom_filter_t, lmnk_boundary_delta_t,
    lmnk_capability_flags_t, lmnk_causal_report_t, lmnk_cognitive_ledger_t,
    lmnk_compute_dispatch_mask_t, lmnk_config_t, lmnk_container_header_t,
    lmnk_core_primitive_state_t, lmnk_core_state_stack_t, lmnk_descriptor_header_t,
    lmnk_distortion_delta_t, lmnk_distortion_tension_t, lmnk_ego_intent_axes_t,
    lmnk_entropy_state_t, lmnk_epistemic_horizon_t, lmnk_fragment_ensemble_report_t,
    lmnk_gateway_frame_t, lmnk_kv_keyframe_header_t, lmnk_lacuna_partition_t, lmnk_ledger_slot_t,
    lmnk_liminal_ego_state_t, lmnk_liminal_threshold_state_t, lmnk_overlap_discovery_report_t,
    lmnk_paged_bitmask_t, lmnk_phase_anchor_record_t, lmnk_pointer_trail_t, lmnk_q7_t,
    lmnk_rank1_descriptor_t, lmnk_reconstitution_report_t, lmnk_section_entry_t,
    lmnk_spine_decode_input_t, lmnk_spine_state_t, lmnk_spine_step_plan_t, lmnk_status_t,
    lmnk_stigmergic_granule_state_t, lmnk_stigmergic_thresholds_t, lmnk_subhead_config_t,
    lmnk_substrate_step_report_t, lmnk_system_pressure_reading_t, lmnk_tagged_ptr_t,
    lmnk_topology_matrix_t, lmnk_topology_snapshot_t, FfiHandleMut, FfiHandleRef,
    LmnkContextHandle, LmnkContextOpaque, LmnkEntropyControllerHandle, LmnkEntropyControllerOpaque,
    LmnkPhaseSteeringHandle, LmnkPhaseSteeringOpaque, LmnkSubstrateHandle, LmnkSubstrateOpaque,
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
        branchless_q7_unified_manifold_fold_64,
        branchless_q7_unified_manifold_fold_with_init_weight_64,
        branchless_q7_unified_manifold_fold_words_with_init_weight_64,
        branchless_q7_weighted_ensemble_contiguous_words_64,
        branchless_q7_weighted_ensemble_words_64, branchless_rad_to_phase_step_u8,
        branchless_sanitize_z_score_f32, branchless_sincos_half_pi_f32,
        branchless_sincos_q7_geodesic_phase_clamp_default_lut, branchless_sincos_q7_phase_clamp,
        branchless_sincos_q7_phase_clamp_default_lut, branchless_sincos_q7_phase_clamp_preinv,
        branchless_toroidal_centroid_step_words_64, branchless_update_crc32_ieee,
        branchless_von_mises_kappa, circular_r_bar_from_words_64, circular_resultant_from_words_64,
        coherence_from_diff_words_64, coherence_from_word_arrays_64, compute_crc32_ieee,
        compute_tikhonov_chebyshev_coeffs_deg3, detect_frame_delta_bitmask,
        detect_frame_mse_bitmask, dispatch_phase_clamp_and_subhead_slot_ensemble_fused,
        dispatch_single_pass, dispatch_subhead_slot_ensemble,
        dispatch_subhead_slot_ensemble_geodesic, eval_chebyshev_spectral_filter_deg3,
        eval_gershgorin_lambda_max, eval_pseudo_ultrametric_distance, eval_rank1, eval_rank1_fused,
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
        PagedBitmask, PhaseSteering, Pod, Rank1Descriptor, RawPhaseProjectable,
        RopeGeometryDescriptor, RopePairLayout, RopeSteeringTarget, SubheadConfig, TaggedPtr,
        ACTION_MODE_LAYER_SKIP, ACTION_MODE_PHASE_REPULSE, ACTION_MODE_PHASE_STEER,
        ACTION_MODE_RANK1_INJECT, ACTION_MODE_SUBHEAD_ROUTE, ACTION_PRESET_EXPRESS_PHASE,
        ACTION_PRESET_FULL_ORCHESTRATION, ACTION_PRESET_INERTIAL_SKIP,
        ACTION_PRESET_PASSIVE_SIDECAR, ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_ID_MASK,
        CHEBYSHEV_COEFF_LEN, DEFAULT_MAX_VALID_Z_SCORE, DEFAULT_PHASE_CLAMP_MAX_RAD,
        DEFAULT_PHASE_CLAMP_Q7_TO_NORM_SCALE, DEFAULT_PHASE_GRAVITY_EPS,
        DEFAULT_PHASE_STEP_MIN_RAD, F32_MIN_POSITIVE_NORMAL, INV_PARALLEL_COUNT_Q7,
        MAX_PHASE_INTERFERENCE_INV_TEMP, MIN_PHASE_INTERFERENCE_INV_TEMP, PAGED_BITMASK_BITS,
        TAG_ACTION_MODE_MASK,
    };
    pub use crate::liminal::*;
    pub use crate::runtime::abi::{
        lmnk_action_decision_t, lmnk_attribution_tag_t, lmnk_bloom_filter_t, lmnk_boundary_delta_t,
        lmnk_capability_flags_t, lmnk_causal_report_t, lmnk_cognitive_ledger_t,
        lmnk_compute_dispatch_mask_t, lmnk_config_t, lmnk_container_header_t,
        lmnk_core_primitive_state_t, lmnk_core_state_stack_t, lmnk_descriptor_header_t,
        lmnk_distortion_delta_t, lmnk_distortion_tension_t, lmnk_ego_intent_axes_t,
        lmnk_entropy_state_t, lmnk_epistemic_horizon_t, lmnk_fragment_ensemble_report_t,
        lmnk_gateway_frame_t, lmnk_kv_keyframe_header_t, lmnk_lacuna_partition_t,
        lmnk_ledger_slot_t, lmnk_liminal_ego_state_t, lmnk_liminal_threshold_state_t,
        lmnk_overlap_discovery_report_t, lmnk_paged_bitmask_t, lmnk_phase_anchor_record_t,
        lmnk_pointer_trail_t, lmnk_q7_t, lmnk_rank1_descriptor_t, lmnk_reconstitution_report_t,
        lmnk_section_entry_t, lmnk_spine_decode_input_t, lmnk_spine_state_t,
        lmnk_spine_step_plan_t, lmnk_status_t, lmnk_stigmergic_granule_state_t,
        lmnk_stigmergic_thresholds_t, lmnk_subhead_config_t, lmnk_substrate_step_report_t,
        lmnk_system_pressure_reading_t, lmnk_tagged_ptr_t, lmnk_topology_matrix_t,
        lmnk_topology_snapshot_t, FfiHandleMut, FfiHandleRef, LmnkContextHandle, LmnkContextOpaque,
        LmnkEntropyControllerHandle, LmnkEntropyControllerOpaque, LmnkPhaseSteeringHandle,
        LmnkPhaseSteeringOpaque, LmnkSubstrateHandle, LmnkSubstrateOpaque,
    };
    pub use crate::runtime::{
        abi_version_string, apply_gateway_frame_rank1_hidden_fused, apply_latent_horizon_slots,
        decode_and_apply_phase_packet_to_trail, decode_and_verify_phase_packet,
        decode_and_verify_phase_packet_words, dispatch_phase_packet, encode_phase_packet,
        evaluate_epistemic_horizon_words_64, init_container_header, init_default_config,
        init_descriptor_header, project_gateway_signal_at_scale, query_capability,
        required_container_byte_len, synthesize_multi_origin_gateway_frame,
        synthesize_multi_origin_gateway_frame_with_words,
        synthesize_relative_gateway_from_perspective_words_64, validate_config_invariants,
        validate_descriptor_header, write_container_sections_to_bytes, AdaptiveThresholdPolicy,
        CausalProvenanceReport, CausalProvenanceSignal, CognitiveCircuitThresholdPolicy,
        CognitiveGroundingSignal, Config, ContradictionGuard, ContradictionGuardSignal,
        CoreLayerEvalContext, CoreStateStack, CoreTelemetryBank, CoreTelemetrySnapshot,
        DefaultResidualSpilloverSlot, DescriptorHeader, DetailedLayerTelemetry, EgoTelemetry,
        EpistemicHorizonReading, EpistemicManifoldOps, GatewayFrame, GatewayPhaseSteeringExt,
        GuardPhaseAnchor, HighCapPhaseKvCache, HighCapPhaseKvGopCache, HorizonDwellState,
        HorizonEscalationProvider, HorizonEscalationRequest, HorizonTriggerInput,
        HybridScaleCircuit, KvKeyframeCache, KvKeyframeHeader, KvKeyframePacket, LatentHorizonSlot,
        LayerDistortionKind, LayerDistortionReport, LiminalBoundaryDelta,
        LiminalHorizonNodeDescriptor, LiminalTelemetry, LiminalTopologyMatrix,
        LiminalTopologyTelemetrySnapshot, LmnkContainerHeader, LmnkContainerView, LmnkSectionEntry,
        LmnkSectionInput, LmnkSectionKind, LowSpecPhaseKvCache, LowSpecPhaseKvGopCache,
        ManifoldGatewaySignal, ManifoldProbeBaseline, MappedStorage, MappedStorageEngine,
        ObjectiveFact, ObjectiveFactSignal, OrchestratorStatusSnapshot, RamPhaseKvCache,
        RamPhaseKvCacheOps, RamPhaseKvEntry, RamPhaseKvGopCache, RamPhaseKvStepOutcome,
        ResidualSpilloverHorizonSlot, ResonantSignalSlice, Result, RuntimeContext, ScaleLayerNode,
        ScaleLayerOrchestrator, ScaleLayerProjection, SlowLoopTelemetryAdjustment,
        StandardPhaseKvCache, StandardPhaseKvGopCache, StigmergicReceptor, StigmergicScaleLayer,
        StigmergicTelemetry, StigmergicThresholdPolicy, SubjectivePersona, SubjectivePersonaSignal,
        SubstrateGatewayExt, SystemConstraint, SystemConstraintSignal, SystemSpineTelemetry,
        TelemetryCapturePolicy, TelemetryVerbosityMode, CAUSAL_AXIS_CONTRADICTION_GUARD,
        CAUSAL_AXIS_OBJECTIVE_FACT, CAUSAL_AXIS_SUBJECTIVE_PERSONA, CAUSAL_AXIS_SYSTEM_CONSTRAINT,
        CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, CAUSAL_CAUSE_COHOMOLOGY_REPULSION,
        CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK, CAUSAL_CAUSE_PERSONA_FACT_COLLISION,
        CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE, CHATTER_FIRE_THRESHOLD_U8, CORE_TOPOLOGY,
        DEFAULT_PAGED_BLOCK_SIZE, DEFAULT_PERSONA_FACT_COLLISION_THRESH,
        LMNK_CONTAINER_FLAG_COMPILED_CACHE, LMNK_CONTAINER_FLAG_PORTABLE_PACKAGE,
        LMNK_CONTAINER_FLAG_RUNTIME_SNAPSHOT, LMNK_CONTAINER_HEADER_BYTES, LMNK_CONTAINER_MAGIC,
        LMNK_DAG_FLAG_AXIOM_CUT_LACUNA_ORTHOGONAL, LMNK_DAG_FLAG_NO_STD_CORE,
        LMNK_DAG_FLAG_SPINE_EGO_ISOLATED, LMNK_DAG_FLAG_Y_SHAPED_PARALLEL_ROOTS,
        LMNK_LIMINAL_HORIZON_TOPOLOGY, LMNK_MAX_CONTAINER_BYTES, LMNK_MAX_CONTAINER_SECTIONS,
        LMNK_RUNTIME_SNAPSHOT_PAYLOAD_BYTES, LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES,
        LMNK_SECTION_BLOOM_FILTER, LMNK_SECTION_CAUSAL_PROVENANCE, LMNK_SECTION_CONTENT_HASH,
        LMNK_SECTION_ENTRY_BYTES, LMNK_SECTION_LEDGER_SLOT, LMNK_SECTION_LOGICAL_PFRAMES,
        LMNK_SECTION_PHASE_ANCHORS, LMNK_SECTION_POINTER_TRAILS, LMNK_SECTION_RECALL_METADATA,
        LMNK_SECTION_RUNTIME_SNAPSHOT, LMNK_SECTION_SIGNATURE, LMNK_TOPOLOGY_CHANNEL_COUNT,
        LMNK_TOPOLOGY_DAG_EDGE_MATRIX, LMNK_TOPOLOGY_DAG_FLAGS_CANONICAL,
        LMNK_TOPOLOGY_LAYER_COUNT, LMNK_TOPOLOGY_MAGIC, LMNK_TOPOLOGY_PINNED_CHANNEL_MASK,
        LMNK_TOPOLOGY_SANDBOX_CHANNEL_MASK, ROOT_CAUSE_COHOMOLOGY_REPULSION,
        ROOT_CAUSE_FACT_RETRIEVAL_MISS, ROOT_CAUSE_NOMINAL, ROOT_CAUSE_PERSONA_FACT_COLLISION,
        ROOT_CAUSE_SYSTEM_QOS_THROTTLE,
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
        LMNK_ABI_VERSION_MAJOR,
        LMNK_ABI_VERSION_MINOR,
        LMNK_ABI_VERSION_PATCH,
    )
}

/// Returns the hexadecimal ABI version identifier ([`LMNK_ABI_VERSION_HEX`]).
#[inline(always)]
#[must_use]
pub const fn abi_version_hex() -> u32 {
    LMNK_ABI_VERSION_HEX
}
