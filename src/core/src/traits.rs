// name: src/core/src/traits.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Root trait definitions, `ActionHub` micro-capability composite ledger, zero-copy alignment
//! wrappers, non-aliasing (`GCSO_RESTRICT`) type guards, branchless mathematical/SIMD primitives,
//! 2-Origin Composite Manifold Gateway (`gateway`), Online Toroidal $k$-Means / DP-Means
//! Coarse-to-Fine & Top-2 Liminal Bifurcation routing, and Ground Truth C-ABI contract assertions
//! for the Geometric Cellular Sheaf Orchestrator (GCSO) engine.
//!
//! # Architectural Role (Two-Layer Hybrid Topology: Concrete Core + Slide-In Horizon Traits)
//! The `crate::traits` hierarchy separates **concrete deterministic core kernels (`pub fn` / struct
//! methods)** from **latent slide-in horizon trait contracts (`Liminal`, `ActionHub`, `ManifoldGatewaySignal`)**
//! across a flat 8-submodule DAG:
//! - **Lv.0 Foundation**:
//!   - `primitives`: Base C-ABI constants ([`crate::abi::GCSO_ABI_VERSION_HEX`]), [`Pod`] safety contract, branchless
//!     IEEE 802.3 CRC32, 5-bit `ACTION_MODE_*` & 3-bit working-lane `ATTRACTOR_SLOT_*` tag codecs,
//!     $8 \times 8$ slot coupling bitmatrix helpers, and lifetime-bound [`FfiHandleRef`] / [`FfiHandleMut`].
//!   - `aligned`: Const-generic [`AlignedSlice`] / [`AlignedSliceMut`] (`16..512`B), $\mathcal{O}(1)$
//!     2D slot bank & Micro-Swarm chunk accessors, [`DisjointRefMut`] / [`DisjointMutPair`] /
//!     [`DisjointMutTriple`] (`GCSO_RESTRICT`), and stack-allocated `AlignedBuffer16..512`.
//!   - `branchless`: [`BranchlessKernel`] constant-time marker trait, `#![no_std]` bitwise
//!     multiplexers, 64B-aligned 256-entry LUT trig, `0-fdiv` `branchless_inv_pos_f32` /
//!     `branchless_sqrt_f32` / `branchless_rsqrt_f32`, 4/8-lane SWAR Q7 & `u8` algebra,
//!     64-head Single Unified Attractor Manifold & Bayesian-Attention phase algebra, and
//!     unified AVX2/SSE2 SIMD kernels ($SO(2)$ rotation, 8-head `sincos`, Procrustes, tree reductions).
//! - **Lv.1 Capabilities (Registered into [`ActionHub`] + Concrete Pure Kernels)**:
//!   - `rope`: [`PhaseSteering`] trait alongside pure sub-head routing, Rank-1 residual, and
//!     2D/3D visual attention steering kernels.
//!   - `telemetry`: [`PagedBitmaskOps`], [`ContrastivePhaseBloomFilter`],
//!     [`StigmergicActionPtrOps`], [`StigmergicTrailOps`], [`StigmergicReconstitutionOps`],
//!     [`FastPathLedger`], and [`StigmergicReconstitutionReport`].
//! - **Lv.2 Micro Capabilities (`hot_path`)**:
//!   - [`PointerActionHub`] and [`StigmergicChunkOps`] feeding [`ActionHub::execute_stigmergic_token_cascade`].
//! - **Lv.3 Macro Capabilities & Composite Gateway (`cold_path` & `gateway`)**:
//!   - `cold_path`: [`EntropyEvaluator`], [`EdbcBranchController`], [`AttractorField`],
//!     [`MultiDomainAttractorSpace`], [`PersonaPatcher`], [`KvKeyframeCache`], [`MappedStorage`],
//!     [`ColdPathTrailCrystalExt`], and pure matrix-free cohomology, QDPT tree, L2P-SVD, DNP, and
//!     capability query helpers.
//!   - `gateway`: 2-Origin Supertrait Hierarchy ([`CausalProvenanceSignal`] ->
//!     [`SystemConstraintSignal`] x [`CognitiveGroundingSignal`] \[[`ObjectiveFactSignal`],
//!     [`SubjectivePersonaSignal`], [`ContradictionGuardSignal`]\]), Single Composite Gateway
//!     ([`ManifoldGatewaySignal`], [`GatewayFrame`], [`GatewayPhaseSteeringExt`],
//!     [`ActionHubGatewayExt`]), and re-exports of [`CrystallizedAttractorRecord`] (`crate::crystal`)
//!     and [`GcsoFourSectionView`] (`crate::storage`).

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

// Preserve strict architectural bottom-up dependency ordering (Lv.0 -> Lv.3) across rustfmt runs.
#[rustfmt::skip]
/// Lv.0 Foundation 1: Base C-ABI constants, `Pod` safety contract, IEEE 802.3 CRC32,
/// 5-bit ActionMode & 3-bit working-lane AttractorSlot tag codecs, $8 \times 8$ coupling helpers, and FFI handle guards.
pub mod primitives;
#[rustfmt::skip]
/// Lv.0 Foundation 2: Hardware-aligned slice wrappers, $\mathcal{O}(1)$ 2D slot bank & Micro-Swarm
/// chunk accessors, `GCSO_RESTRICT` non-aliasing guards, and stack buffers.
pub mod aligned;
#[rustfmt::skip]
/// Lv.0 Foundation 3: `BranchlessKernel` marker trait, `#![no_std]` mathematical primitives,
/// 64B-aligned 256-entry LUT trig, 4/8-lane SWAR Q7/`u8` operations, Unified-Field & Bayesian-Attention
/// phase algebra, and unified AVX2/SSE2 SIMD kernels.
pub mod branchless;
#[rustfmt::skip]
/// Lv.1 Capability 1: RoPE & inline tensor modulation (`PhaseSteering` trait plus pure
/// sub-head routing, Rank-1 residual, and visual attention steering kernels).
pub mod rope;
#[rustfmt::skip]
/// Lv.1 Capability 2: Stigmergic trails, internal telemetry & swarm intelligence (`PagedBitmaskOps`,
/// `ContrastivePhaseBloomFilter`, `StigmergicActionPtrOps`, `StigmergicTrailOps`, `StigmergicReconstitutionOps`, `FastPathLedger`).
pub mod telemetry;
#[rustfmt::skip]
/// Lv.2 Micro Capabilities: `PointerActionHub` and `StigmergicChunkOps`.
pub mod hot_path;
#[rustfmt::skip]
/// Lv.3 Macro Observer: Trajectory analysis, EDBC pitchfork control, matrix-free cellular sheaf
/// cohomology $H^1$, hierarchical trees, Single Unified Attractor Manifold, persona, L2P-SVD, KV-keyframe, DNP, and mapped-storage storage.
pub mod cold_path;
#[rustfmt::skip]
/// Lv.3 Composite Manifold Gateway: 2-Origin Supertrait Hierarchy, $\mathcal{O}(1)$ Causal Provenance,
/// `ManifoldGatewaySignal` (`GatewayFrame` 128B `Pod`), Online Toroidal $k$-Means / DP-Means,
/// Top-2 Liminal Bifurcation Routing, and `.gcso` `CrystallizedAttractorRecord` / `GcsoFourSectionView` persistence.
pub mod gateway;

#[rustfmt::skip]
/// Core Composite Capability Ledger: `ActionHub` and register-resident SWAR/QDPS helpers.
pub mod action_hub;

// ===================================================================
// Flat Single-Source-of-Truth Re-Exports
// ===================================================================

pub use self::aligned::{
    is_aligned_to, is_ptr_aligned_to, AlignedBuffer128, AlignedBuffer16, AlignedBuffer256,
    AlignedBuffer32, AlignedBuffer512, AlignedBuffer64, AlignedSlice, AlignedSlice128,
    AlignedSlice16, AlignedSlice256, AlignedSlice32, AlignedSlice512, AlignedSlice64,
    AlignedSliceMut, AlignedSliceMut128, AlignedSliceMut16, AlignedSliceMut256, AlignedSliceMut32,
    AlignedSliceMut512, AlignedSliceMut64, DisjointMutPair, DisjointMutTriple, DisjointRefMut,
};
pub use self::branchless::{
    branchless_fast_transition_log_prob, branchless_phase_coherence_inv_temp_scale,
    branchless_project_f32_slice_to_q7_64, branchless_q7_coherence_64,
    branchless_q7_coherence_and_resultant_64, branchless_q7_gravity_fold_64,
    branchless_q7_inhibit_64, branchless_q7_inhibit_words_64, branchless_q7_resonant_pull_64,
    branchless_q7_synergy_fuse_64, branchless_q7_synergy_fuse_words_64,
    branchless_q7_transition_log_prob_64, branchless_q7_transition_log_prob_words_64,
    branchless_q7_unified_field_fold_64, branchless_q7_unified_field_fold_with_init_weight_64,
    branchless_q7_unified_field_fold_words_with_init_weight_64,
    branchless_toroidal_centroid_step_words_64, circular_r_bar_from_words_64,
    circular_r_bar_from_words_with_uniform_flag_64, circular_resultant_from_words_64,
    coherence_from_diff_words_64, coherence_from_word_arrays_64,
    coherence_from_word_arrays_with_both_uniform_flags_64,
    coherence_from_word_arrays_with_uniform_flag_64, sum_abs_and_sq_f32_simd8,
    sum_abs_sq_and_max_f32_simd8, sum_sq_f32_simd8, swar_slerp_word_unchecked_x8,
    swar_slerp_word_with_bias_x8, swar_slerp_words_with_bias_64,
    swar_slerp_words_with_uniform_flag_64, swar_words_equal_64, swar_words_nonzero_64,
    BranchlessKernel, Liminal,
};
pub use self::cold_path::{
    abi_version, abi_version_string, bridge_and_filter_liminal_trail_chain,
    cascade_action_slot_ptr, cascade_tagged_ptr, compute_tikhonov_chebyshev_coeffs_deg3,
    decode_and_apply_dnp_packet_to_trail, decode_and_verify_dnp_packet,
    decode_and_verify_dnp_words, detect_and_record_obstruction_repulsion, dispatch_dnp_packet,
    encode_dnp_phase_packet, eval_chebyshev_spectral_filter_deg3, eval_gershgorin_lambda_max,
    eval_pseudo_ultrametric_distance, eval_stigmergic_block_degrees,
    evaluate_cohomological_obstruction, evaluate_hodge_obstruction_and_dual_pairing,
    evaluate_trail_chain_cohomology_and_joint_log_prob, init_default_config, project_lora,
    project_lora_with_block_scales, quantize_depth_hysteresis, query_capability,
    resolve_dyadic_qdpt_and_cascade_ptr, validate_config_invariants, AttractorField,
    ColdPathTrailCrystalExt, EdbcBranchController, EntropyEvaluator, KvKeyframeCache,
    MappedStorage, MultiDomainAttractorSpace, PersonaPatcher,
};
pub use self::gateway::{
    apply_gateway_frame_rank1_hidden_fused, apply_latent_horizon_slots,
    evaluate_epistemic_horizon_words_64, evaluate_stigmergic_venn_and_serendipity_words_64,
    fold_fractal_nodes_words_64, project_bytes_swar_to_q7_64, project_bytes_swar_to_words_64,
    project_f32_slice_to_q7_64, project_f32_slice_to_words_64, project_gateway_signal_at_scale,
    synthesize_multi_origin_gateway_frame, synthesize_multi_origin_gateway_frame_with_words,
    synthesize_relative_gateway_from_perspective_words_64, ActionHubGatewayExt,
    ActionHubScaleOrchestrator, AnonymousComputePool, AutonomousCoreLayerStack,
    CausalProvenanceReport, CausalProvenanceSignal, CognitiveCircuitActivation,
    CognitiveCircuitBank, CognitiveCircuitBankOps, CognitiveCircuitPlasticity,
    CognitiveCircuitThresholdPolicy, CognitiveGroundingSignal, CognitiveOpKind, CognitiveStepOp,
    CognitiveTrajectoryLedger, ComputeDispatchMask, ComputePoolConfig, ComputePressureRegime,
    ContradictionGuard, ContradictionGuardSignal, CoreLayerEvalContext, CoreTelemetryBank,
    CoreTelemetrySnapshot, CrystallizedAttractorRecord, DefaultResidualSpilloverSlot,
    DetailedLayerTelemetry, DissipativeOverflowReport, EgoIntentAxes, EgoLayerTelemetry,
    EpistemicHorizonReading, EpistemicManifoldOps, EpistemicRegime, GatewayFrame,
    GatewayPhaseSteeringExt, GcsoFourSectionView, HighCapCognitiveBank, HighCapPhaseKvCache,
    HighCapPhaseKvGopCache, HorizonDwellState, HorizonEscalationProvider, HorizonEscalationRequest,
    HorizonTriggerInput, HybridScaleCircuit, LatentHorizonSlot, LayerDistortionKind,
    LayerDistortionReport, LiminalApertureField, LiminalBoundaryDelta, LiminalEgoField,
    LiminalEgoPreset, LiminalHorizonNodeDescriptor, LiminalLayerTelemetry,
    LiminalTopologyTelemetrySnapshot, LowSpecCognitiveBank, LowSpecPhaseKvCache,
    LowSpecPhaseKvGopCache, ManifoldGatewaySignal, ObjectiveFact, ObjectiveFactSignal,
    RamPhaseKvCache, RamPhaseKvCacheOps, RamPhaseKvEntry, RamPhaseKvGopCache,
    RamPhaseKvStepOutcome, RawPhaseProjectable, RelativeAttractorNode, RelativeCouplingMatrix,
    ResidualSpilloverHorizonSlot, ScaleLayerNode, ScaleLayerProjection,
    SlowLoopTelemetryAdjustment, StandardCognitiveBank, StandardPhaseKvCache,
    StandardPhaseKvGopCache, StigmergicFragmentEnsembleReport, StigmergicGranuleField,
    StigmergicLayerTelemetry, StigmergicLayeredSwarmExt, StigmergicScaleGranularity,
    StigmergicScaleLayer, StigmergicVennSerendipityReport, SubjectivePersona,
    SubjectivePersonaSignal, SystemConstraint, SystemConstraintSignal, SystemPressureReading,
    SystemSpineTelemetry, TelemetryCapturePolicy, TelemetryVerbosityMode, APERTURE_ACTION_HUB_WIDE,
    APERTURE_CAUSAL_PROVENANCE, APERTURE_COGNITIVE_CIRCUIT, APERTURE_COHOMOLOGY_EDBC,
    APERTURE_CONTRADICTION_GUARD, APERTURE_DISSIPATIVE_CRYSTAL, APERTURE_FAST_PATH_LEDGER,
    APERTURE_KV_PHASE_CACHE, APERTURE_MANIFOLD_GATEWAY, APERTURE_MAPPED_STORAGE,
    APERTURE_OBJECTIVE_FACT, APERTURE_PHASE_STEERING, APERTURE_POINTER_TRAIL,
    APERTURE_RANK1_RESIDUAL, APERTURE_RELATIVE_COUPLING, APERTURE_STIGMERGIC_CONSENSUS,
    APERTURE_SUBHEAD_ROUTING, APERTURE_SUBJECTIVE_PERSONA, APERTURE_SYSTEM_CONSTRAINT,
    CAUSAL_AXIS_CONTRADICTION_GUARD, CAUSAL_AXIS_OBJECTIVE_FACT, CAUSAL_AXIS_SUBJECTIVE_PERSONA,
    CAUSAL_AXIS_SYSTEM_CONSTRAINT, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
    CAUSAL_CAUSE_COHOMOLOGY_REPULSION, CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK,
    CAUSAL_CAUSE_PERSONA_FACT_COLLISION, CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE, CORE_LAYER_EGO,
    CORE_LAYER_LIMINAL, CORE_LAYER_STIGMERGIC, DEFAULT_FACT_RIPA_CLAMP_RAD,
    DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO, DEFAULT_PERSONA_FACT_COLLISION_THRESH,
    DEFAULT_PERSONA_RIPA_CLAMP_RAD, DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE,
    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE, EGO_TRIGGER_ALL_DEFAULT, EGO_TRIGGER_BASIN_CAPTURE,
    EGO_TRIGGER_BOUNDARY_CHATTER, EGO_TRIGGER_BOUNDARY_STAGNATION, EGO_TRIGGER_INTERNAL_CONFLICT,
    EGO_TRIGGER_LAYER_DISTORTION, EGO_TRIGGER_LIMINAL_EXTERIOR, GCSO_CRYSTAL_RECORD_MAGIC,
    GCSO_CRYSTAL_RECORD_VERSION, GCSO_FOUR_SECTION_HEADER_BYTES, GCSO_LIMINAL_HORIZON_TOPOLOGY,
    GCSO_MAPPED_STORAGE_PREAMBLE_OFFSET, NUM_CORE_AUTONOMOUS_LAYERS, NUM_LIMINAL_APERTURE_CHANNELS,
    ROOT_CAUSE_COHOMOLOGY_REPULSION, ROOT_CAUSE_FACT_RETRIEVAL_MISS, ROOT_CAUSE_NOMINAL,
    ROOT_CAUSE_PERSONA_FACT_COLLISION, ROOT_CAUSE_SYSTEM_QOS_THROTTLE,
};
pub use self::hot_path::{PointerActionHub, StigmergicChunkOps};
pub use self::primitives::{
    branchless_abs_f32, branchless_atan2_f32, branchless_atan2_to_q7_fast, branchless_clamp_f32,
    branchless_copysign_f32, branchless_cos_q7, branchless_crc32_128b_words, branchless_crc32_ieee,
    branchless_exp_f32, branchless_eyring_kramers_escape, branchless_flush_subnormal_f32,
    branchless_hysteresis_quantize_depth, branchless_inv_pos_f32, branchless_is_finite_f32,
    branchless_is_nan_f32, branchless_lerp_f32, branchless_ln_f32, branchless_max_f32,
    branchless_min_f32, branchless_phase_proximity_gravity_f32, branchless_q7_abs_u8,
    branchless_q7_accumulate_64, branchless_q7_add_sat, branchless_q7_circular_diff,
    branchless_q7_circular_lerp, branchless_q7_conjugate_64, branchless_q7_fold_s1_geodesic,
    branchless_q7_neg_sat, branchless_q7_qdps_filter_64, branchless_q7_scale_64,
    branchless_q7_scale_u7, branchless_q7_sub_64, branchless_q7_sub_sat, branchless_q7_to_rad,
    branchless_qdps_filter_q7, branchless_rad_to_q7, branchless_rad_to_qdps_step_u8,
    branchless_relu_f32, branchless_ripa_clamp_rad, branchless_rotate_so2_pair,
    branchless_rotate_so2_pair_scaled, branchless_rsqrt_f32, branchless_sanitize_z_score_f32,
    branchless_select_f32, branchless_select_i32, branchless_select_i64, branchless_select_i8,
    branchless_select_q7, branchless_select_u16, branchless_select_u32, branchless_select_u64,
    branchless_select_u8, branchless_select_usize, branchless_sigmoid_f32, branchless_signum_f32,
    branchless_sin_q7, branchless_sincos_f32, branchless_sincos_half_pi_f32, branchless_sincos_q7,
    branchless_sincos_q7_geodesic_ripa_default_lut, branchless_sincos_q7_ripa,
    branchless_sincos_q7_ripa_default_lut, branchless_sincos_q7_ripa_preinv,
    branchless_smoothstep_f32, branchless_splitmix64, branchless_sqrt_f32, branchless_tanh_f32,
    branchless_update_crc32_ieee, branchless_von_mises_kappa, is_phase_profile_nonzero_64,
    is_word_array_uniform_64, load_q7_words_64, store_q7_words_64, swar_q7_abs_u8_x8,
    swar_q7_add_sat_x4, swar_q7_add_sat_x8, swar_q7_avg_round_x8, swar_q7_circular_add_x8,
    swar_q7_circular_diff_x8, swar_q7_fold_s1_geodesic_x8, swar_q7_mean_i8_from_words_64,
    swar_q7_neg_sat_x4, swar_q7_neg_sat_x8, swar_q7_pack_x8, swar_q7_qdps_filter_words_64,
    swar_q7_qdps_filter_x8, swar_q7_scale_le128_with_bias_x8, swar_q7_scale_u7_x8,
    swar_q7_sub_sat_x4, swar_q7_sub_sat_x8, swar_q7_unpack_x8, swar_u8_add_sat_x8,
    swar_u8_halve_decay_x8, swar_u8_max_x8, swar_u8_mean_round_x8, swar_u8_nonzero_count_x8,
    swar_u8_sub_sat_x8, swar_u8_sum_x8, trail_phase_words_mut, trail_phase_words_ref,
    DEFAULT_MAX_VALID_Z_SCORE, DEFAULT_PHASE_GRAVITY_EPS, DEFAULT_RIPA_Q7_TO_NORM_SCALE,
    F32_MIN_POSITIVE_NORMAL, MAX_PHASE_INTERFERENCE_INV_TEMP, MIN_PHASE_INTERFERENCE_INV_TEMP,
    SWAR_EVEN_BYTES_64, SWAR_LOW7_MASK_64, SWAR_ODD_BYTES_64, SWAR_SIGN_MASK_64,
    SWAR_SPLAT_BYTE_64, SWAR_SUM_U16_64,
};
pub use self::primitives::{
    compute_crc32_ieee, coupled_slot_count, coupled_slot_row_mask, finalize_snapshot_header_crc32,
    init_descriptor_header, init_snapshot_header, is_slot_pair_coupled, pack_action_slot_tag,
    raw_byte_ranges_overlap, slot_coupling_bit_index, unpack_action_mode_from_tag,
    unpack_attractor_slot_from_tag, update_crc32_ieee, validate_descriptor_header,
    validate_snapshot_header, verify_snapshot_payload_crc32, with_slot_pair_coupled,
    with_slot_pair_coupled_symmetric, FfiHandleMut, FfiHandleRef, GcsoActionHubHandle,
    GcsoActionHubOpaque, GcsoAttractorFieldHandle, GcsoAttractorFieldOpaque, GcsoContainerHandle,
    GcsoContainerOpaque, GcsoContextHandle, GcsoContextOpaque, GcsoDpsrKernelHandle,
    GcsoDpsrKernelOpaque, GcsoEdbcControllerHandle, GcsoEdbcControllerOpaque, GcsoLedgerSlotHandle,
    GcsoLedgerSlotOpaque, GcsoRank1ResidualHandle, GcsoRank1ResidualOpaque, GcsoResult,
    GcsoSubHeadRouterHandle, GcsoSubHeadRouterOpaque, KvKeyframePacket, KvMotionDeltaPacket, Pod,
    ACTION_MODE_DPSR_STEER, ACTION_MODE_LAYER_SKIP, ACTION_MODE_PHASE_REPULSE,
    ACTION_MODE_RANK1_INJECT, ACTION_MODE_SUBHEAD_ROUTE, ACTION_PRESET_EXPRESS_PHASE,
    ACTION_PRESET_FULL_ORCHESTRATION, ACTION_PRESET_INERTIAL_SKIP, ACTION_PRESET_PASSIVE_SIDECAR,
    ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_CRYSTALLIZED, ATTRACTOR_SLOT_ID_MASK,
    ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA, ATTRACTOR_SLOT_REPULSION,
    ATTRACTOR_SLOT_SYSTEM_BASELINE, ATTRACTOR_SLOT_TEMPORAL, ATTRACTOR_SLOT_USER_CONTEXT,
    ATTRACTOR_SLOT_USER_INTENT, CHEBYSHEV_COEFF_LEN, CHEBYSHEV_FILTER_DEGREE,
    DEFAULT_ACTION_HUB_CAPACITY, DEFAULT_ENTROPY_EPS, DEFAULT_PAGED_BLOCK_SIZE,
    DEFAULT_QDPS_MIN_STEP_RAD, DEFAULT_RIPA_CLAMP_MAX_RAD, DEFAULT_SLOT_COUPLING_MATRIX_8X8,
    FAST_PATH_SHORTCUT_SLOTS, GCSO_SNAPSHOT_MAGIC_U32, LEDGER_HISTORY_LEN, MAX_ANCHORS_PER_FIELD,
    MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS, MAX_SURROGATE_ENTROPY_LN2, PAGED_BITMASK_BITS,
    PAGED_BITMASK_WORDS, Q7_MAX, Q7_MIN, Q7_PHASE_INV_SCALE_RAD, Q7_PHASE_SCALE_RAD, Q7_UNIT_SCALE,
    Q7_ZERO, TAG_ACTION_MODE_MASK, TAG_ATTRACTOR_SLOT_SHIFT, WORKBENCH_LANE_AUTO,
    WORKBENCH_LANE_COUNT, WORKBENCH_LANE_FLAG_PINNED, WORKBENCH_LANE_FLAG_REPULSION,
    WORKBENCH_LANE_FLAG_SANDBOX,
};
pub use self::rope::{
    detect_frame_delta_bitmask, detect_frame_mse_bitmask, dispatch_pointer_gated,
    dispatch_pointer_gated_with_phases, dispatch_ripa_and_subhead_slot_ensemble_fused,
    dispatch_ripa_and_subhead_unified_anchors_fused, dispatch_single_pass,
    dispatch_single_pass_with_phases, dispatch_subhead_slot_ensemble,
    dispatch_subhead_slot_ensemble_geodesic, dispatch_subhead_unified_anchors, eval_rank1,
    eval_rank1_fused, eval_rank1_pointer_gated_fused, eval_rank1_slices, eval_rank1_slices_fused,
    resolve_autonomous_swarm_rot_dim, ripa_upper_rot_dim, steer_visual_grid_2d_bitmask_gated,
    steer_visual_grid_3d_bitmask_gated, steer_visual_patch_2d, steer_visual_patch_3d,
    validate_pspm_layout, PhaseSteering,
};
pub use self::telemetry::{
    eval_bitmask_dyadic_ultrametric, swar_subswarm_resolve_x8, ContrastivePhaseBloomFilter,
    FastPathLedger, PagedBitmaskOps, PheromoneDensityOps, StigmergicActionPtrOps,
    StigmergicReconstitutionOps, StigmergicReconstitutionReport, StigmergicTrailOps,
    DEFAULT_BLOOM_MAX_ACTIVE_BUCKETS, DEFAULT_BLOOM_MAX_MASS_THRESHOLD,
    DEFAULT_STIGMERGIC_QUORUM_THRESHOLD, DEFAULT_STIGMERGIC_TOLERANCE_Q7,
    HEADS_PER_STIGMERGIC_GRANULE, LEDGER_FLAG_COGNITIVE_TRAJECTORY, LEDGER_METRIC_FAST_PATH_HIT,
    LEDGER_METRIC_LAYER_SKIP_COMPENSATED, LEDGER_METRIC_LAZY_UNWRAP_APPLIED,
    LEDGER_METRIC_PHASE_STEERED, LEDGER_METRIC_RANK1_INJECTED, LEDGER_METRIC_REPULSION_APPLIED,
    LEDGER_METRIC_SLOT_SYNERGY_ACTIVE, LEDGER_METRIC_SUBHEAD_ROUTED, MAX_ATTRACTOR_PULL_FORCE,
    MAX_STIGMERGIC_DENSITY, NUM_STIGMERGIC_GRANULES, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE,
    TRAIL_FLAG_DPSR_ACTIVE, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE,
    TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_FLAG_SWARM_ESCALATED, TRAIL_FLAG_SWARM_RESOLVED,
    TRAIL_FLAG_SYNERGY_RESONANCE, TRAIL_SWARM_QUORUM_MASK, TRAIL_SWARM_QUORUM_SHIFT,
};

pub(crate) use self::action_hub::validate_optional_qk_tensors;
pub use self::action_hub::ActionHub;
pub use crate::entropy_control::EntropyClosedLoopCoordinator;
pub use crate::phase_steering::PhaseSteeringClosedLoop;

// ===================================================================
// Ergonomic Prelude Module
// ===================================================================

/// Ergonomic prelude re-exporting core domain traits, `ActionHub`, Hot/Cold observers,
/// Manifold Gateway traits & Online Toroidal $k$-Means / DP-Means records, Top-2 Liminal
/// Bifurcation routing constants, 4-section `.gcso` zero-copy container views, alignment guards,
/// Unified-Field & Bayesian-Attention kernels, Micro-Swarm ops, and FFI wrappers.
pub mod prelude {
    pub use super::aligned::{
        AlignedBuffer128, AlignedBuffer16, AlignedBuffer256, AlignedBuffer32, AlignedBuffer512,
        AlignedBuffer64, AlignedSlice, AlignedSlice128, AlignedSlice16, AlignedSlice256,
        AlignedSlice32, AlignedSlice512, AlignedSlice64, AlignedSliceMut, AlignedSliceMut128,
        AlignedSliceMut16, AlignedSliceMut256, AlignedSliceMut32, AlignedSliceMut512,
        AlignedSliceMut64, DisjointMutPair, DisjointMutTriple, DisjointRefMut,
    };
    pub use super::branchless::{
        branchless_phase_coherence_inv_temp_scale, branchless_project_f32_slice_to_q7_64,
        branchless_q7_coherence_64, branchless_q7_coherence_and_resultant_64,
        branchless_q7_gravity_fold_64, branchless_q7_inhibit_64, branchless_q7_inhibit_words_64,
        branchless_q7_resonant_pull_64, branchless_q7_synergy_fuse_64,
        branchless_q7_synergy_fuse_words_64, branchless_q7_transition_log_prob_64,
        branchless_q7_transition_log_prob_words_64, branchless_q7_unified_field_fold_64,
        branchless_q7_unified_field_fold_with_init_weight_64,
        branchless_q7_unified_field_fold_words_with_init_weight_64,
        branchless_toroidal_centroid_step_words_64, circular_r_bar_from_words_64,
        circular_resultant_from_words_64, coherence_from_diff_words_64,
        coherence_from_word_arrays_64, sum_abs_and_sq_f32_simd8, sum_abs_sq_and_max_f32_simd8,
        sum_sq_f32_simd8, swar_slerp_word_unchecked_x8, swar_slerp_word_with_bias_x8,
        swar_slerp_words_with_bias_64, BranchlessKernel, Liminal,
    };
    pub use super::cold_path::{
        abi_version, abi_version_string, bridge_and_filter_liminal_trail_chain,
        cascade_action_slot_ptr, cascade_tagged_ptr, compute_tikhonov_chebyshev_coeffs_deg3,
        decode_and_apply_dnp_packet_to_trail, decode_and_verify_dnp_packet,
        decode_and_verify_dnp_words, detect_and_record_obstruction_repulsion, dispatch_dnp_packet,
        encode_dnp_phase_packet, eval_chebyshev_spectral_filter_deg3, eval_gershgorin_lambda_max,
        eval_pseudo_ultrametric_distance, eval_stigmergic_block_degrees,
        evaluate_cohomological_obstruction, evaluate_hodge_obstruction_and_dual_pairing,
        evaluate_trail_chain_cohomology_and_joint_log_prob, init_default_config, project_lora,
        project_lora_with_block_scales, quantize_depth_hysteresis, query_capability,
        resolve_dyadic_qdpt_and_cascade_ptr, validate_config_invariants, AttractorField,
        ColdPathTrailCrystalExt, EdbcBranchController, EntropyEvaluator, KvKeyframeCache,
        MappedStorage, MultiDomainAttractorSpace, PersonaPatcher,
    };
    pub use super::gateway::{
        apply_gateway_frame_rank1_hidden_fused, apply_latent_horizon_slots,
        evaluate_epistemic_horizon_words_64, evaluate_stigmergic_venn_and_serendipity_words_64,
        fold_fractal_nodes_words_64, project_bytes_swar_to_q7_64, project_bytes_swar_to_words_64,
        project_f32_slice_to_q7_64, project_f32_slice_to_words_64,
        synthesize_multi_origin_gateway_frame, synthesize_multi_origin_gateway_frame_with_words,
        synthesize_relative_gateway_from_perspective_words_64, ActionHubGatewayExt,
        AnonymousComputePool, AutonomousCoreLayerStack, CausalProvenanceReport,
        CausalProvenanceSignal, CognitiveGroundingSignal, ComputeDispatchMask, ComputePoolConfig,
        ComputePressureRegime, ContradictionGuard, ContradictionGuardSignal, CoreLayerEvalContext,
        CoreTelemetryBank, CoreTelemetrySnapshot, CrystallizedAttractorRecord,
        DefaultResidualSpilloverSlot, DetailedLayerTelemetry, DissipativeOverflowReport,
        EgoIntentAxes, EgoLayerTelemetry, EpistemicHorizonReading, EpistemicManifoldOps,
        EpistemicRegime, GatewayFrame, GatewayPhaseSteeringExt, GcsoFourSectionView,
        HorizonDwellState, HorizonEscalationProvider, HorizonEscalationRequest,
        HorizonTriggerInput, LatentHorizonSlot, LayerDistortionKind, LayerDistortionReport,
        LiminalApertureField, LiminalBoundaryDelta, LiminalEgoField, LiminalEgoPreset,
        LiminalHorizonNodeDescriptor, LiminalLayerTelemetry, LiminalTopologyTelemetrySnapshot,
        ManifoldGatewaySignal, ObjectiveFact, ObjectiveFactSignal, RawPhaseProjectable,
        RelativeAttractorNode, RelativeCouplingMatrix, ResidualSpilloverHorizonSlot,
        SlowLoopTelemetryAdjustment, StigmergicFragmentEnsembleReport, StigmergicGranuleField,
        StigmergicLayerTelemetry, StigmergicScaleGranularity, StigmergicVennSerendipityReport,
        SubjectivePersona, SubjectivePersonaSignal, SystemConstraint, SystemConstraintSignal,
        SystemPressureReading, SystemSpineTelemetry, TelemetryCapturePolicy,
        TelemetryVerbosityMode, APERTURE_ACTION_HUB_WIDE, APERTURE_CAUSAL_PROVENANCE,
        APERTURE_COGNITIVE_CIRCUIT, APERTURE_COHOMOLOGY_EDBC, APERTURE_CONTRADICTION_GUARD,
        APERTURE_DISSIPATIVE_CRYSTAL, APERTURE_FAST_PATH_LEDGER, APERTURE_KV_PHASE_CACHE,
        APERTURE_MANIFOLD_GATEWAY, APERTURE_MAPPED_STORAGE, APERTURE_OBJECTIVE_FACT,
        APERTURE_PHASE_STEERING, APERTURE_POINTER_TRAIL, APERTURE_RANK1_RESIDUAL,
        APERTURE_RELATIVE_COUPLING, APERTURE_STIGMERGIC_CONSENSUS, APERTURE_SUBHEAD_ROUTING,
        APERTURE_SUBJECTIVE_PERSONA, APERTURE_SYSTEM_CONSTRAINT, CAUSAL_AXIS_CONTRADICTION_GUARD,
        CAUSAL_AXIS_OBJECTIVE_FACT, CAUSAL_AXIS_SUBJECTIVE_PERSONA, CAUSAL_AXIS_SYSTEM_CONSTRAINT,
        CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, CAUSAL_CAUSE_COHOMOLOGY_REPULSION,
        CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK, CAUSAL_CAUSE_PERSONA_FACT_COLLISION,
        CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE, CORE_LAYER_EGO, CORE_LAYER_LIMINAL,
        CORE_LAYER_STIGMERGIC, DEFAULT_FACT_RIPA_CLAMP_RAD,
        DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO, DEFAULT_PERSONA_FACT_COLLISION_THRESH,
        DEFAULT_PERSONA_RIPA_CLAMP_RAD, DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE,
        DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE, EGO_TRIGGER_ALL_DEFAULT,
        EGO_TRIGGER_BASIN_CAPTURE, EGO_TRIGGER_BOUNDARY_CHATTER, EGO_TRIGGER_BOUNDARY_STAGNATION,
        EGO_TRIGGER_INTERNAL_CONFLICT, EGO_TRIGGER_LAYER_DISTORTION, EGO_TRIGGER_LIMINAL_EXTERIOR,
        GCSO_CRYSTAL_RECORD_MAGIC, GCSO_CRYSTAL_RECORD_VERSION, GCSO_FOUR_SECTION_HEADER_BYTES,
        GCSO_LIMINAL_HORIZON_TOPOLOGY, GCSO_MAPPED_STORAGE_PREAMBLE_OFFSET,
        NUM_CORE_AUTONOMOUS_LAYERS, NUM_LIMINAL_APERTURE_CHANNELS, ROOT_CAUSE_COHOMOLOGY_REPULSION,
        ROOT_CAUSE_FACT_RETRIEVAL_MISS, ROOT_CAUSE_NOMINAL, ROOT_CAUSE_PERSONA_FACT_COLLISION,
        ROOT_CAUSE_SYSTEM_QOS_THROTTLE,
    };
    pub use super::hot_path::{PointerActionHub, StigmergicChunkOps};
    pub use super::primitives::{
        branchless_atan2_to_q7_fast, branchless_crc32_ieee, branchless_flush_subnormal_f32,
        branchless_inv_pos_f32, branchless_phase_proximity_gravity_f32,
        branchless_q7_fold_s1_geodesic, branchless_rad_to_qdps_step_u8,
        branchless_sanitize_z_score_f32, branchless_sincos_half_pi_f32,
        branchless_sincos_q7_geodesic_ripa_default_lut, branchless_sincos_q7_ripa,
        branchless_sincos_q7_ripa_default_lut, branchless_sincos_q7_ripa_preinv,
        branchless_update_crc32_ieee, branchless_von_mises_kappa, is_phase_profile_nonzero_64,
        is_word_array_uniform_64, load_q7_words_64, store_q7_words_64, swar_q7_circular_add_x8,
        swar_q7_fold_s1_geodesic_x8, swar_q7_mean_i8_from_words_64, swar_q7_qdps_filter_words_64,
        swar_q7_scale_le128_with_bias_x8, swar_u8_add_sat_x8, swar_u8_halve_decay_x8,
        swar_u8_max_x8, swar_u8_mean_round_x8, swar_u8_nonzero_count_x8, swar_u8_sub_sat_x8,
        swar_u8_sum_x8, trail_phase_words_mut, trail_phase_words_ref, DEFAULT_MAX_VALID_Z_SCORE,
        DEFAULT_PHASE_GRAVITY_EPS, DEFAULT_RIPA_Q7_TO_NORM_SCALE, F32_MIN_POSITIVE_NORMAL,
        MAX_PHASE_INTERFERENCE_INV_TEMP, MIN_PHASE_INTERFERENCE_INV_TEMP,
    };
    pub use super::primitives::{
        compute_crc32_ieee, coupled_slot_count, coupled_slot_row_mask,
        finalize_snapshot_header_crc32, init_descriptor_header, init_snapshot_header,
        is_slot_pair_coupled, pack_action_slot_tag, raw_byte_ranges_overlap,
        unpack_action_mode_from_tag, unpack_attractor_slot_from_tag, update_crc32_ieee,
        validate_descriptor_header, validate_snapshot_header, verify_snapshot_payload_crc32,
        with_slot_pair_coupled, with_slot_pair_coupled_symmetric, FfiHandleMut, FfiHandleRef,
        GcsoResult, KvKeyframePacket, KvMotionDeltaPacket, Pod, ACTION_MODE_DPSR_STEER,
        ACTION_MODE_LAYER_SKIP, ACTION_MODE_PHASE_REPULSE, ACTION_MODE_RANK1_INJECT,
        ACTION_MODE_SUBHEAD_ROUTE, ACTION_PRESET_EXPRESS_PHASE, ACTION_PRESET_FULL_ORCHESTRATION,
        ACTION_PRESET_INERTIAL_SKIP, ACTION_PRESET_PASSIVE_SIDECAR, ACTION_PRESET_REPULSION_ESCAPE,
        ATTRACTOR_SLOT_CRYSTALLIZED, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA,
        ATTRACTOR_SLOT_REPULSION, ATTRACTOR_SLOT_SYSTEM_BASELINE, ATTRACTOR_SLOT_TEMPORAL,
        ATTRACTOR_SLOT_USER_CONTEXT, ATTRACTOR_SLOT_USER_INTENT, DEFAULT_SLOT_COUPLING_MATRIX_8X8,
        WORKBENCH_LANE_AUTO, WORKBENCH_LANE_COUNT, WORKBENCH_LANE_FLAG_PINNED,
        WORKBENCH_LANE_FLAG_REPULSION, WORKBENCH_LANE_FLAG_SANDBOX,
    };
    pub use super::rope::{
        detect_frame_delta_bitmask, detect_frame_mse_bitmask, dispatch_pointer_gated,
        dispatch_pointer_gated_with_phases, dispatch_ripa_and_subhead_slot_ensemble_fused,
        dispatch_ripa_and_subhead_unified_anchors_fused, dispatch_single_pass,
        dispatch_single_pass_with_phases, dispatch_subhead_slot_ensemble,
        dispatch_subhead_slot_ensemble_geodesic, dispatch_subhead_unified_anchors, eval_rank1,
        eval_rank1_fused, eval_rank1_pointer_gated_fused, eval_rank1_slices,
        eval_rank1_slices_fused, resolve_autonomous_swarm_rot_dim, ripa_upper_rot_dim,
        steer_visual_grid_2d_bitmask_gated, steer_visual_grid_3d_bitmask_gated,
        steer_visual_patch_2d, steer_visual_patch_3d, validate_pspm_layout, PhaseSteering,
    };
    pub use super::telemetry::{
        eval_bitmask_dyadic_ultrametric, swar_subswarm_resolve_x8, ContrastivePhaseBloomFilter,
        FastPathLedger, PagedBitmaskOps, PheromoneDensityOps, StigmergicActionPtrOps,
        StigmergicReconstitutionOps, StigmergicReconstitutionReport, StigmergicTrailOps,
        DEFAULT_BLOOM_MAX_ACTIVE_BUCKETS, DEFAULT_BLOOM_MAX_MASS_THRESHOLD,
        DEFAULT_STIGMERGIC_QUORUM_THRESHOLD, DEFAULT_STIGMERGIC_TOLERANCE_Q7,
        HEADS_PER_STIGMERGIC_GRANULE, LEDGER_METRIC_FAST_PATH_HIT,
        LEDGER_METRIC_LAYER_SKIP_COMPENSATED, LEDGER_METRIC_LAZY_UNWRAP_APPLIED,
        LEDGER_METRIC_PHASE_STEERED, LEDGER_METRIC_RANK1_INJECTED, LEDGER_METRIC_REPULSION_APPLIED,
        LEDGER_METRIC_SLOT_SYNERGY_ACTIVE, LEDGER_METRIC_SUBHEAD_ROUTED, MAX_ATTRACTOR_PULL_FORCE,
        MAX_STIGMERGIC_DENSITY, NUM_STIGMERGIC_GRANULES, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE,
        TRAIL_FLAG_DPSR_ACTIVE, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE,
        TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_FLAG_SWARM_ESCALATED, TRAIL_FLAG_SWARM_RESOLVED,
        TRAIL_FLAG_SYNERGY_RESONANCE,
    };
    pub use super::{ActionHub, EntropyClosedLoopCoordinator, PhaseSteeringClosedLoop};
}

// ===================================================================
// Compile-Time Ground Truth C-ABI Layout & Structural Parity
// ===================================================================

const _: () = {
    use crate::abi::{
        gcso_config_t, gcso_descriptor_header_t, gcso_edbc_state_t, gcso_kv_keyframe_header_t,
        gcso_ledger_slot_t, gcso_mapped_storage_descriptor_t, gcso_paged_bitmask_t,
        gcso_pointer_trail_t, gcso_rank1_residual_descriptor_t, gcso_snapshot_header_t,
        gcso_subhead_config_t, gcso_tagged_ptr_t, GCSO_ABI_VERSION_HEX, GCSO_ABI_VERSION_MAJOR,
        GCSO_ABI_VERSION_MINOR, GCSO_ABI_VERSION_PATCH,
    };
    use core::mem::{align_of, offset_of, size_of};

    assert!(
        GCSO_ABI_VERSION_HEX
            == ((GCSO_ABI_VERSION_MAJOR << 16)
                | (GCSO_ABI_VERSION_MINOR << 8)
                | GCSO_ABI_VERSION_PATCH)
    );
    assert!(GCSO_CRYSTAL_RECORD_VERSION == GCSO_ABI_VERSION_HEX);
    assert!(MAX_ATTENTION_HEADS == 64);
    assert!(MAX_ANCHORS_PER_FIELD == 64);
    assert!(MAX_ATTRACTOR_DOMAINS == 8);
    assert!(NUM_STIGMERGIC_GRANULES == 8);
    assert!(HEADS_PER_STIGMERGIC_GRANULE == 8);
    assert!(NUM_STIGMERGIC_GRANULES * HEADS_PER_STIGMERGIC_GRANULE == MAX_ATTENTION_HEADS);
    assert!(CHEBYSHEV_FILTER_DEGREE == 3);
    assert!(CHEBYSHEV_COEFF_LEN == 4);
    assert!(PAGED_BITMASK_WORDS == 4);
    assert!(PAGED_BITMASK_BITS == 256);
    assert!(GCSO_FOUR_SECTION_HEADER_BYTES == 192);
    assert!(GCSO_MAPPED_STORAGE_PREAMBLE_OFFSET == 128);

    assert!((TAG_ACTION_MODE_MASK as u32) == 0x1F);
    assert!(TAG_ATTRACTOR_SLOT_SHIFT == 5);
    assert!(((7u8 << TAG_ATTRACTOR_SLOT_SHIFT) | TAG_ACTION_MODE_MASK) == 0xFF);

    assert!(size_of::<gcso_descriptor_header_t>() == 8);
    assert!(align_of::<gcso_descriptor_header_t>() == 4);
    assert!(offset_of!(gcso_descriptor_header_t, struct_size) == 0);
    assert!(offset_of!(gcso_descriptor_header_t, abi_version) == 4);

    assert!(size_of::<gcso_tagged_ptr_t>() == 8);
    assert!(align_of::<gcso_tagged_ptr_t>() == 8);

    assert!(size_of::<gcso_paged_bitmask_t>() == 32);
    assert!(align_of::<gcso_paged_bitmask_t>() == 32);
    assert!(offset_of!(gcso_paged_bitmask_t, bits) == 0);

    assert!(size_of::<gcso_pointer_trail_t>() == 128);
    assert!(align_of::<gcso_pointer_trail_t>() == 128);
    assert!(offset_of!(gcso_pointer_trail_t, current_ptr) == 0);
    assert!(offset_of!(gcso_pointer_trail_t, user_data) == 16);
    assert!(offset_of!(gcso_pointer_trail_t, accumulated_phase_delta) == 56);

    assert!(size_of::<gcso_config_t>() == 64);
    assert!(align_of::<gcso_config_t>() == 16);
    assert!(offset_of!(gcso_config_t, head_dim) == 0);
    assert!(offset_of!(gcso_config_t, qdps_min_step_rad) == 20);
    assert!(offset_of!(gcso_config_t, ledger_mode) == 38);

    assert!(size_of::<gcso_snapshot_header_t>() == 128);
    assert!(align_of::<gcso_snapshot_header_t>() == 64);
    assert!(offset_of!(gcso_snapshot_header_t, magic) == 0);
    assert!(offset_of!(gcso_snapshot_header_t, checksum_crc32) == 56);
    assert!(offset_of!(gcso_snapshot_header_t, timestamp_epoch_sec) == 64);

    assert!(size_of::<gcso_ledger_slot_t>() == 64);
    assert!(align_of::<gcso_ledger_slot_t>() == 64);
    assert!(offset_of!(gcso_ledger_slot_t, mode) == 0);
    assert!(offset_of!(gcso_ledger_slot_t, fast_path_shortcuts) == 24);
    assert!(offset_of!(gcso_ledger_slot_t, telemetry_mini_ledger) == 56);

    assert!(size_of::<gcso_kv_keyframe_header_t>() == 64);
    assert!(align_of::<gcso_kv_keyframe_header_t>() == 32);
    assert!(offset_of!(gcso_kv_keyframe_header_t, frame_type) == 0);
    assert!(offset_of!(gcso_kv_keyframe_header_t, icache_payload_offset) == 24);
    assert!(offset_of!(gcso_kv_keyframe_header_t, pcache_payload_offset) == 32);

    assert!(size_of::<KvKeyframePacket>() == 128);
    assert!(align_of::<KvKeyframePacket>() == 64);
    assert!(offset_of!(KvKeyframePacket, header) == 0);
    assert!(offset_of!(KvKeyframePacket, anchor_phases) == 64);
    assert!(size_of::<KvMotionDeltaPacket>() == 64);
    assert!(align_of::<KvMotionDeltaPacket>() == 64);
    assert!(offset_of!(KvMotionDeltaPacket, motion_deltas) == 0);

    assert!(size_of::<gcso_subhead_config_t>() == 32);
    assert!(align_of::<gcso_subhead_config_t>() == 16);
    assert!(offset_of!(gcso_subhead_config_t, num_fact_heads) == 0);
    assert!(offset_of!(gcso_subhead_config_t, fact_phase_gain) == 8);

    assert!(size_of::<gcso_edbc_state_t>() == 64);
    assert!(align_of::<gcso_edbc_state_t>() == 32);
    assert!(offset_of!(gcso_edbc_state_t, moving_z_entropy) == 0);
    assert!(offset_of!(gcso_edbc_state_t, active_branch_mode) == 24);

    assert!(size_of::<gcso_rank1_residual_descriptor_t>() == 64);
    assert!(align_of::<gcso_rank1_residual_descriptor_t>() == 32);
    assert!(offset_of!(gcso_rank1_residual_descriptor_t, layer_idx) == 0);
    assert!(offset_of!(gcso_rank1_residual_descriptor_t, u_vector_ptr) == 8);
    assert!(offset_of!(gcso_rank1_residual_descriptor_t, v_vector_ptr) == 16);

    assert!(size_of::<gcso_mapped_storage_descriptor_t>() == 64);
    assert!(align_of::<gcso_mapped_storage_descriptor_t>() == 32);
    assert!(offset_of!(gcso_mapped_storage_descriptor_t, mapped_address) == 0);
    assert!(offset_of!(gcso_mapped_storage_descriptor_t, file_size_bytes) == 8);

    assert!(size_of::<ContrastivePhaseBloomFilter>() == 64);
    assert!(align_of::<ContrastivePhaseBloomFilter>() == 64);
    assert!(size_of::<StigmergicReconstitutionReport>() == 16);
    assert!(align_of::<StigmergicReconstitutionReport>() == 16);
    assert!(size_of::<CausalProvenanceReport>() == 16);
    assert!(align_of::<CausalProvenanceReport>() == 16);

    assert!(size_of::<GatewayFrame>() == 128);
    assert!(align_of::<GatewayFrame>() == 64);
    assert!(offset_of!(GatewayFrame, subhead_fact_gain_q7) == 56);
    assert!(offset_of!(GatewayFrame, mean_phase_tag_q7) == 59);
    assert!(offset_of!(GatewayFrame, active_head_count) == 60);
    assert!(offset_of!(GatewayFrame, frame_flags) == 62);
    assert!(offset_of!(GatewayFrame, phase_deltas) == 64);

    assert!(size_of::<CrystallizedAttractorRecord>() == 128);
    assert!(align_of::<CrystallizedAttractorRecord>() == 64);
    assert!(offset_of!(CrystallizedAttractorRecord, checksum_crc32) == 52);
    assert!(offset_of!(CrystallizedAttractorRecord, dominant_slot) == 56);
    assert!(offset_of!(CrystallizedAttractorRecord, reserved_align64) == 60);
    assert!(offset_of!(CrystallizedAttractorRecord, phase_profile) == 64);

    assert!(size_of::<AlignedSlice<'static, f32, 32>>() == size_of::<&'static [f32]>());
    assert!(align_of::<AlignedSlice<'static, f32, 32>>() == align_of::<&'static [f32]>());
    assert!(size_of::<AlignedSliceMut<'static, f32, 32>>() == size_of::<&'static mut [f32]>());
    assert!(align_of::<AlignedSliceMut<'static, f32, 32>>() == align_of::<&'static mut [f32]>());

    assert!(size_of::<DisjointRefMut<'static, f32, 32, 32>>() == 2 * size_of::<&'static [f32]>());
    assert!(size_of::<DisjointMutPair<'static, f32, 32>>() == 2 * size_of::<&'static mut [f32]>());
    assert!(
        size_of::<DisjointMutTriple<'static, f32, 32>>() == 3 * size_of::<&'static mut [f32]>()
    );

    assert!(size_of::<FfiHandleRef<'static, u8>>() == size_of::<*const u8>());
    assert!(size_of::<FfiHandleMut<'static, u8>>() == size_of::<*mut u8>());
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::{
        gcso_anchor_type_t, gcso_config_t, gcso_edbc_state_t, gcso_ledger_slot_t,
        gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_q7_t, gcso_tagged_ptr_t,
    };

    struct MockActionHubComposite {
        head_dim: u32,
        num_heads: u32,
        default_phase: gcso_q7_t,
        branch_mode: u32,
        crystallized_count: u32,
    }

    impl PhaseSteering for MockActionHubComposite {
        fn head_dim(&self) -> u32 {
            self.head_dim
        }

        fn num_heads(&self) -> u32 {
            self.num_heads
        }
    }

    impl PointerActionHub for MockActionHubComposite {
        fn reset(&mut self) -> GcsoResult<()> {
            Ok(())
        }

        fn step_pointer(
            &mut self,
            current_ptr: u64,
            trail_out: &mut gcso_pointer_trail_t,
        ) -> GcsoResult<()> {
            trail_out.record_pointer_transition(current_ptr, 1, 0.25);
            trail_out.accumulate_uniform_phase(self.default_phase);
            Ok(())
        }

        fn update_cellular_state_express(
            &mut self,
            _trail_id: u32,
            _phase_delta: gcso_q7_t,
        ) -> GcsoResult<()> {
            Ok(())
        }

        fn pull_trail_to_attractor(
            &mut self,
            _trail_id: u32,
            _anchor_id: u32,
            _pull_force: f32,
        ) -> GcsoResult<()> {
            Ok(())
        }

        fn link_hallucinated_trails(
            &mut self,
            _src_trail_id: u32,
            _dst_trail_id: u32,
        ) -> GcsoResult<()> {
            Ok(())
        }

        fn capacity(&self) -> u32 {
            256
        }

        fn active_count(&self) -> u32 {
            1
        }
    }

    impl FastPathLedger for MockActionHubComposite {
        fn evaluate_auto_tune(
            &self,
            _slot: &gcso_ledger_slot_t,
            _config_out: &mut gcso_config_t,
        ) -> GcsoResult<()> {
            Ok(())
        }
    }

    impl EntropyEvaluator for MockActionHubComposite {
        fn eval_stateful(
            &mut self,
            token_z_score: f32,
            state_out: &mut gcso_edbc_state_t,
        ) -> GcsoResult<()> {
            state_out.moving_z_entropy = token_z_score;
            state_out.active_branch_mode = self.branch_mode;
            Ok(())
        }
    }

    impl EdbcBranchController for MockActionHubComposite {
        fn active_branch_mode(&self) -> u32 {
            self.branch_mode
        }

        fn set_bifurcation_threshold(&mut self, _threshold: f32) {}
        fn set_hysteresis_band(&mut self, _band: f32) {}
        fn set_tau_flux(&mut self, _tau_flux: f32) {}
    }

    impl PersonaPatcher for MockActionHubComposite {
        fn apply_patch(&mut self, _patch_data: &[u8]) -> GcsoResult<()> {
            Ok(())
        }
    }

    impl AttractorField for MockActionHubComposite {
        fn anchor_count(&self) -> u32 {
            self.crystallized_count
        }

        fn apply_evaporative_decay(&mut self) {}
        fn total_potential_energy(&self) -> f32 {
            1.0
        }
        fn pressure_ratio(&self) -> f32 {
            0.25
        }
        fn evaporate_weakest_anchor(&mut self) -> Option<u32> {
            None
        }

        fn add_system_prompt_anchor(&mut self, _prompt: &str, _weight: f32) -> GcsoResult<u32> {
            Ok(1)
        }

        fn add_embedding_anchor(
            &mut self,
            _embedding: &AlignedSlice32<'_, f32>,
            _anchor_type: gcso_anchor_type_t,
            _weight: f32,
        ) -> GcsoResult<u32> {
            Ok(1)
        }

        fn add_anchor(
            &mut self,
            _anchor_type: gcso_anchor_type_t,
            _vec: &AlignedSlice32<'_, f32>,
            _dim: usize,
        ) -> GcsoResult<u32> {
            Ok(1)
        }

        fn inject_phase_repulsion(
            &mut self,
            _repulsion_deltas: &AlignedSlice16<'_, gcso_q7_t>,
            _gain: f32,
        ) -> GcsoResult<()> {
            Ok(())
        }

        fn aggregate_bottom_up(&mut self) -> GcsoResult<u32> {
            self.crystallized_count += 1;
            Ok(1)
        }

        fn clear_anchors(&mut self) -> GcsoResult<()> {
            self.crystallized_count = 0;
            Ok(())
        }
    }

    impl MultiDomainAttractorSpace for MockActionHubComposite {
        fn advance_step(&mut self) {}

        fn set_domain_weight(&mut self, _domain_id: u32, _weight: f32) -> GcsoResult<()> {
            Ok(())
        }

        fn set_domain_enabled(&mut self, _domain_id: u32, _enabled: bool) -> GcsoResult<()> {
            Ok(())
        }

        fn find_dominant_domain(&self) -> (u32, f32) {
            (1, 2.0)
        }

        fn evaluate_multi_domain_superposition(
            &self,
            trail: &gcso_pointer_trail_t,
            composite_phase_out: &mut [gcso_q7_t; 64],
        ) -> f32 {
            *composite_phase_out = trail.accumulated_phase_delta;
            1.0
        }

        fn clear_all_domains(&mut self) -> GcsoResult<()> {
            Ok(())
        }
    }

    #[test]
    fn test_canonical_modules_and_action_hub_composite() {
        fn assert_action_hub<T: ActionHub + ActionHubGatewayExt>(hub: &T) -> (u32, u32) {
            (
                <T as rope::PhaseSteering>::head_dim(hub),
                <T as rope::PhaseSteering>::num_heads(hub),
            )
        }

        let mut hub = MockActionHubComposite {
            head_dim: 8,
            num_heads: 2,
            default_phase: 24,
            branch_mode: 0,
            crystallized_count: 0,
        };
        assert_eq!(assert_action_hub(&hub), (8, 2));

        let mut mask = gcso_paged_bitmask_t::default();
        <gcso_paged_bitmask_t as telemetry::PagedBitmaskOps>::set_bit_branchless(&mut mask, 7);
        assert!(
            <gcso_paged_bitmask_t as telemetry::PagedBitmaskOps>::test_bit_branchless(&mask, 7)
        );

        let deg = AlignedBuffer32::new([1.0f32, 3.0, 2.0, 0.5]);
        let lmax = eval_gershgorin_lambda_max(&deg.as_aligned_slice()).unwrap();
        assert!((lmax - 6.0).abs() < 1e-5);

        let mut ledger = gcso_ledger_slot_t::default();
        let bloom = ContrastivePhaseBloomFilter::new();
        let mut q_buf = AlignedBuffer32::new([1.0f32; 16]);
        let mut trail = gcso_pointer_trail_t::default();
        let input_ptr = gcso_tagged_ptr_t::pack_action(
            0x0400,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            12,
        );

        let density = hub
            .execute_stigmergic_token_cascade(
                input_ptr.raw(),
                &mut ledger,
                &bloom,
                2,
                128,
                DEFAULT_QDPS_MIN_STEP_RAD,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                Some(&mut q_buf.as_aligned_slice_mut()),
                None,
                &mut trail,
            )
            .unwrap();
        assert_eq!(density, 0);
        assert_ne!(
            ledger.telemetry_mini_ledger[0] & LEDGER_METRIC_PHASE_STEERED,
            0
        );
    }
}
