// name: src/core/src/traits.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Root trait definitions, `ActionHub` micro-capability composite ledger, zero-copy alignment
//! wrappers, non-aliasing (`GCSO_RESTRICT`) type guards, branchless mathematical/SIMD primitives,
//! 2-Origin Composite Manifold Gateway (`gateway`), Online Toroidal $k$-Means / DP-Means
//! Coarse-to-Fine & Top-2 Liminal Bifurcation routing, and Ground Truth C-ABI contract assertions
//! for the Geometric Cellular Sheaf Orchestrator (GCSO) engine.
//!
//! # Architectural Role (1 + 8 Flat Module Topology: `ActionHub` + 2 Observers + Gateway)
//! The `crate::traits` hierarchy separates **the core sidecar state ledger (`ActionHub`)** from **two observers
//! (`HotPathObserver` and `ColdPathObserver`)** and the **single composite gateway (`ManifoldGatewaySignal`)**
//! across a flat 8-submodule DAG:
//! - **Lv.0 Foundation**:
//!   - `primitives`: C-ABI v0.1.1 (`0x00000101`) constants, [`Pod`] safety contract, branchless
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
//! - **Lv.1 Pure Capability Lists (Registered into [`ActionHub`])**:
//!   - `rope`: [`PhaseSteering`], [`EntropyGatedPhaseLookup`], [`DynamicLayerSkipper`],
//!     [`PspmRouter`], [`SrlAdapter`], and [`VisualAttentionSteering`].
//!   - `telemetry`: [`PagedBitmaskOps`], [`ContrastivePhaseBloomFilter`],
//!     [`StigmergicActionPtrOps`], [`StigmergicTrailOps`], [`StigmergicSwarmOps`],
//!     [`DaesScratchpad`], and [`SwarmResolutionReport`].
//! - **Lv.2 Micro Observer (`hot_path`)**:
//!   - [`PointerActionHub`], [`SwarmCellChunk`], and blanket-implemented [`HotPathExecutionEngine`] /
//!     [`HotPathObserver`] / [`FractalCascadeCoordinator`] observing [`ActionHub`] as a constant-time
//!     $\mathcal{O}(1)$ pointer-to-RoPE execution pipeline over the Single Unified Attractor Manifold.
//! - **Lv.3 Macro Observer & Composite Gateway (`cold_path` & `gateway`)**:
//!   - `cold_path`: [`EntropyEvaluator`], [`EdbcBranchController`], [`SheafCohomologyEvaluator`],
//!     [`HierarchicalTreeTable`], [`AttractorField`], [`MultiDomainAttractorSpace`],
//!     [`PersonaPatcher`], [`L2pSvdProjector`], [`PprcCache`], [`DnpPacketDispatcher`],
//!     [`ZimmsStorage`], [`ColdPathTrailCrystalExt`], and blanket-implemented macro coordinators
//!     ([`ColdPathOrchestrator`], [`ColdPathObserver`], [`AttractorSlotSynergyCoordinator`],
//!     [`StigmergicCohomologyCoordinator`], [`BottomUpMacroCrystalCoordinator`],
//!     [`FullStackGcsoEngine`]).
//!   - `gateway`: 2-Origin Supertrait Hierarchy ([`CausalProvenanceSignal`] ->
//!     [`SystemConstraintSignal`] x [`CognitiveGroundingSignal`] \[[`ObjectiveFactSignal`],
//!     [`SubjectivePersonaSignal`], [`ContradictionGuardSignal`]\]), Single Composite Gateway
//!     ([`ManifoldGatewaySignal`], [`GatewayFrame`], [`GatewayPhaseSteeringExt`],
//!     [`ActionHubGatewayExt`]), and re-exports of [`CrystallizedAttractorRecord`] (`crate::crystal`)
//!     and [`GcsoFourSectionView`] (`crate::zimms`).

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
/// Lv.0 Foundation 1: Base C-ABI v0.1.1 constants, `Pod` safety contract, IEEE 802.3 CRC32,
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
/// Lv.1 Capability 1: RoPE & inline tensor modulation (`PhaseSteering`, `EntropyGatedPhaseLookup`,
/// `DynamicLayerSkipper`, `PspmRouter`, `SrlAdapter`, `VisualAttentionSteering`).
pub mod rope;
#[rustfmt::skip]
/// Lv.1 Capability 2: Stigmergic trails, internal telemetry & swarm intelligence (`PagedBitmaskOps`,
/// `ContrastivePhaseBloomFilter`, `StigmergicActionPtrOps`, `StigmergicTrailOps`, `StigmergicSwarmOps`, `DaesScratchpad`).
pub mod telemetry;
#[rustfmt::skip]
/// Lv.2 Micro Observer: `PointerActionHub`, `SwarmCellChunk`, and constant-time `ActionHub` -> RoPE
/// execution engines (`HotPathExecutionEngine`, `HotPathObserver`, `FractalCascadeCoordinator`).
pub mod hot_path;
#[rustfmt::skip]
/// Lv.3 Macro Observer: Trajectory analysis, EDBC pitchfork control, matrix-free cellular sheaf
/// cohomology $H^1$, hierarchical trees, Single Unified Attractor Manifold, persona, L2P-SVD, PPRC, DNP, and ZIMMS storage.
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
    branchless_project_f32_slice_to_q7_64, branchless_q7_attention_score_delta_64,
    branchless_q7_bayesian_attention_score_delta_64, branchless_q7_circular_resultant_64,
    branchless_q7_circular_weighted_step_64, branchless_q7_coherence_64,
    branchless_q7_coherence_and_resultant_64, branchless_q7_gravity_fold_64,
    branchless_q7_inhibit_64, branchless_q7_inhibit_words_64, branchless_q7_resonant_pull_64,
    branchless_q7_slerp_64, branchless_q7_synergy_fuse_64, branchless_q7_synergy_fuse_words_64,
    branchless_q7_transition_log_prob_64, branchless_q7_transition_log_prob_words_64,
    branchless_q7_unified_field_fold_64, branchless_q7_unified_field_fold_with_init_weight_64,
    branchless_q7_unified_field_fold_words_with_init_weight_64,
    branchless_toroidal_centroid_step_words_64,
    branchless_toroidal_centroid_step_words_with_coh_64, circular_r_bar_from_words_64,
    circular_r_bar_from_words_with_uniform_flag_64, circular_resultant_from_words_64,
    coherence_from_diff_words_64, coherence_from_word_arrays_64,
    coherence_from_word_arrays_with_both_uniform_flags_64,
    coherence_from_word_arrays_with_uniform_flag_64,
    modulate_attention_scores_by_bayesian_trails_64, modulate_attention_scores_by_phase_bank_64,
    sum_abs_and_sq_f32_simd8, sum_abs_sq_and_max_f32_simd8, sum_sq_f32_simd8,
    swar_q7_geodesic_slerp_x8, swar_slerp_word_unchecked_x8, swar_slerp_word_with_bias_x8,
    swar_slerp_words_with_bias_64, swar_slerp_words_with_uniform_flag_64, swar_words_equal_64,
    swar_words_nonzero_64, BranchlessKernel,
};
pub use self::cold_path::{
    AttractorField, AttractorSlotSynergyCoordinator, BottomUpMacroCrystalCoordinator,
    ColdPathObserver, ColdPathOrchestrator, ColdPathTrailCrystalExt, DnpPacketDispatcher,
    EdbcBranchController, EntropyEvaluator, FullStackGcsoEngine, GcsoRuntimeContext,
    HierarchicalTreeTable, L2pSvdProjector, MultiDomainAttractorSpace, PersonaPatcher, PprcCache,
    SheafCohomologyEvaluator, StigmergicCohomologyCoordinator, SystemCapabilityQuery, ZimmsStorage,
};
pub use self::gateway::{
    mean_q7_phase_tag_64, project_bytes_swar_to_q7_64, project_bytes_swar_to_words_64,
    project_f32_slice_to_q7_64, project_f32_slice_to_words_64,
    synthesize_multi_origin_gateway_frame, synthesize_multi_origin_gateway_frame_with_words,
    ActionHubGatewayExt, CausalProvenanceReport, CausalProvenanceSignal, CognitiveGroundingSignal,
    ContradictionGuard, ContradictionGuardSignal, CrystallizedAttractorRecord, GatewayFrame,
    GatewayPhaseSteeringExt, GcsoFourSectionView, ManifoldGatewaySignal, ObjectiveFact,
    ObjectiveFactSignal, RawPhaseProjectable, SubjectivePersona, SubjectivePersonaSignal,
    SystemConstraint, SystemConstraintSignal, CAUSAL_AXIS_CONTRADICTION_GUARD,
    CAUSAL_AXIS_OBJECTIVE_FACT, CAUSAL_AXIS_SUBJECTIVE_PERSONA, CAUSAL_AXIS_SYSTEM_CONSTRAINT,
    CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, CAUSAL_CAUSE_COHOMOLOGY_REPULSION,
    CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK, CAUSAL_CAUSE_PERSONA_FACT_COLLISION,
    CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE, DEFAULT_FACT_RIPA_CLAMP_RAD,
    DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO, DEFAULT_PERSONA_FACT_COLLISION_THRESH,
    DEFAULT_PERSONA_RIPA_CLAMP_RAD, DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE,
    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE, GCSO_CRYSTAL_RECORD_MAGIC,
    GCSO_CRYSTAL_RECORD_VERSION, GCSO_FOUR_SECTION_HEADER_BYTES,
    GCSO_ZIMMS_DESCRIPTOR_PREAMBLE_OFFSET, ROOT_CAUSE_COHOMOLOGY_REPULSION,
    ROOT_CAUSE_FACT_RETRIEVAL_MISS, ROOT_CAUSE_NOMINAL, ROOT_CAUSE_PERSONA_FACT_COLLISION,
    ROOT_CAUSE_SYSTEM_QOS_THROTTLE,
};
pub use self::hot_path::{
    FractalCascadeCoordinator, HotPathExecutionEngine, HotPathObserver, PointerActionHub,
    SwarmCellChunk,
};
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
    GcsoContainerOpaque, GcsoContextHandle, GcsoContextOpaque, GcsoDaesSlotHandle,
    GcsoDaesSlotOpaque, GcsoDpsrKernelHandle, GcsoDpsrKernelOpaque, GcsoEdbcControllerHandle,
    GcsoEdbcControllerOpaque, GcsoPspmRouterHandle, GcsoPspmRouterOpaque, GcsoResult,
    GcsoSrlAdapterHandle, GcsoSrlAdapterOpaque, Pod, PprcKeyframePacket, PprcMotionDeltaPacket,
    ACTION_MODE_DPSR_STEER, ACTION_MODE_LAYER_SKIP, ACTION_MODE_PHASE_REPULSE,
    ACTION_MODE_PSPM_ROUTE, ACTION_MODE_SRL_INJECT, ACTION_PRESET_EXPRESS_PHASE,
    ACTION_PRESET_FULL_ORCHESTRATION, ACTION_PRESET_INERTIAL_SKIP, ACTION_PRESET_PASSIVE_SIDECAR,
    ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_CUSTOM_5, ATTRACTOR_SLOT_CUSTOM_6,
    ATTRACTOR_SLOT_CUSTOM_7, ATTRACTOR_SLOT_ID_MASK, ATTRACTOR_SLOT_KNOWLEDGE,
    ATTRACTOR_SLOT_PERSONA, ATTRACTOR_SLOT_REPULSION, ATTRACTOR_SLOT_SYSTEM_BASELINE,
    ATTRACTOR_SLOT_USER_CONTEXT, CHEBYSHEV_COEFF_LEN, CHEBYSHEV_FILTER_DEGREE,
    DAES_MINI_LEDGER_LEN, DAES_SHORTCUT_SLOTS, DEFAULT_ACTION_HUB_CAPACITY, DEFAULT_ENTROPY_EPS,
    DEFAULT_PAGED_BLOCK_SIZE, DEFAULT_QDPS_MIN_STEP_RAD, DEFAULT_RIPA_CLAMP_MAX_RAD,
    DEFAULT_SLOT_COUPLING_MATRIX_8X8, GCSO_SNAPSHOT_MAGIC_U32, MAX_ANCHORS_PER_FIELD,
    MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS, MAX_SURROGATE_ENTROPY_LN2, PAGED_BITMASK_BITS,
    PAGED_BITMASK_WORDS, Q7_MAX, Q7_MIN, Q7_PHASE_INV_SCALE_RAD, Q7_PHASE_SCALE_RAD, Q7_UNIT_SCALE,
    Q7_ZERO, TAG_ACTION_MODE_MASK, TAG_ATTRACTOR_SLOT_SHIFT,
};
pub use self::rope::{
    resolve_autonomous_swarm_rot_dim, ripa_upper_rot_dim, DynamicLayerSkipper,
    EntropyGatedPhaseLookup, PhaseSteering, PspmRouter, SrlAdapter, VisualAttentionSteering,
};
pub use self::telemetry::{
    eval_bitmask_dyadic_ultrametric, swar_subswarm_resolve_x8, ContrastivePhaseBloomFilter,
    DaesScratchpad, PagedBitmaskOps, StigmergicActionPtrOps, StigmergicSwarmOps,
    StigmergicTrailOps, SwarmResolutionReport, DAES_METRIC_FAST_PATH_HIT,
    DAES_METRIC_LAYER_SKIP_COMPENSATED, DAES_METRIC_LAZY_UNWRAP_APPLIED, DAES_METRIC_PHASE_STEERED,
    DAES_METRIC_PSPM_ROUTED, DAES_METRIC_REPULSION_APPLIED, DAES_METRIC_SLOT_SYNERGY_ACTIVE,
    DAES_METRIC_SRL_INJECTED, DEFAULT_BLOOM_MAX_ACTIVE_BUCKETS, DEFAULT_BLOOM_MAX_MASS_THRESHOLD,
    DEFAULT_SWARM_QUORUM_THRESHOLD, DEFAULT_SWARM_TOLERANCE_Q7, HEADS_PER_SWARM_CELL,
    MAX_ATTRACTOR_PULL_FORCE, MAX_STIGMERGIC_DENSITY, NUM_SWARM_MICRO_CELLS,
    TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, TRAIL_FLAG_DPSR_ACTIVE, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE,
    TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_FLAG_SWARM_ESCALATED,
    TRAIL_FLAG_SWARM_RESOLVED, TRAIL_FLAG_SYNERGY_RESONANCE, TRAIL_SWARM_QUORUM_MASK,
    TRAIL_SWARM_QUORUM_SHIFT,
};

pub(crate) use self::action_hub::{
    accumulate_q7_slice_into_words_64, fuse_lazy_unwrap_and_qdps_from_words_64,
    validate_optional_qk_tensors,
};
pub use self::action_hub::{fuse_lazy_unwrap_and_qdps_into_aligned_64, ActionHub};
pub(crate) use self::telemetry::select_rewire_bridge_and_modulate_with_preloaded_words_64;

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
        branchless_q7_attention_score_delta_64, branchless_q7_bayesian_attention_score_delta_64,
        branchless_q7_circular_resultant_64, branchless_q7_circular_weighted_step_64,
        branchless_q7_coherence_64, branchless_q7_coherence_and_resultant_64,
        branchless_q7_gravity_fold_64, branchless_q7_inhibit_64, branchless_q7_inhibit_words_64,
        branchless_q7_resonant_pull_64, branchless_q7_slerp_64, branchless_q7_synergy_fuse_64,
        branchless_q7_synergy_fuse_words_64, branchless_q7_transition_log_prob_64,
        branchless_q7_transition_log_prob_words_64, branchless_q7_unified_field_fold_64,
        branchless_q7_unified_field_fold_with_init_weight_64,
        branchless_q7_unified_field_fold_words_with_init_weight_64,
        branchless_toroidal_centroid_step_words_64,
        branchless_toroidal_centroid_step_words_with_coh_64, circular_r_bar_from_words_64,
        circular_resultant_from_words_64, coherence_from_diff_words_64,
        coherence_from_word_arrays_64, modulate_attention_scores_by_bayesian_trails_64,
        modulate_attention_scores_by_phase_bank_64, sum_abs_and_sq_f32_simd8,
        sum_abs_sq_and_max_f32_simd8, sum_sq_f32_simd8, swar_q7_geodesic_slerp_x8,
        swar_slerp_word_unchecked_x8, swar_slerp_word_with_bias_x8, swar_slerp_words_with_bias_64,
        BranchlessKernel,
    };
    pub use super::cold_path::{
        AttractorField, AttractorSlotSynergyCoordinator, BottomUpMacroCrystalCoordinator,
        ColdPathObserver, ColdPathOrchestrator, ColdPathTrailCrystalExt, DnpPacketDispatcher,
        EdbcBranchController, EntropyEvaluator, FullStackGcsoEngine, GcsoRuntimeContext,
        HierarchicalTreeTable, L2pSvdProjector, MultiDomainAttractorSpace, PersonaPatcher,
        PprcCache, SheafCohomologyEvaluator, StigmergicCohomologyCoordinator,
        SystemCapabilityQuery, ZimmsStorage,
    };
    pub use super::gateway::{
        mean_q7_phase_tag_64, project_bytes_swar_to_q7_64, project_bytes_swar_to_words_64,
        project_f32_slice_to_q7_64, project_f32_slice_to_words_64,
        synthesize_multi_origin_gateway_frame, synthesize_multi_origin_gateway_frame_with_words,
        ActionHubGatewayExt, CausalProvenanceReport, CausalProvenanceSignal,
        CognitiveGroundingSignal, ContradictionGuard, ContradictionGuardSignal,
        CrystallizedAttractorRecord, GatewayFrame, GatewayPhaseSteeringExt, GcsoFourSectionView,
        ManifoldGatewaySignal, ObjectiveFact, ObjectiveFactSignal, RawPhaseProjectable,
        SubjectivePersona, SubjectivePersonaSignal, SystemConstraint, SystemConstraintSignal,
        CAUSAL_AXIS_CONTRADICTION_GUARD, CAUSAL_AXIS_OBJECTIVE_FACT,
        CAUSAL_AXIS_SUBJECTIVE_PERSONA, CAUSAL_AXIS_SYSTEM_CONSTRAINT,
        CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, CAUSAL_CAUSE_COHOMOLOGY_REPULSION,
        CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK, CAUSAL_CAUSE_PERSONA_FACT_COLLISION,
        CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE, DEFAULT_FACT_RIPA_CLAMP_RAD,
        DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO, DEFAULT_PERSONA_FACT_COLLISION_THRESH,
        DEFAULT_PERSONA_RIPA_CLAMP_RAD, DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE,
        DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE, GCSO_CRYSTAL_RECORD_MAGIC,
        GCSO_CRYSTAL_RECORD_VERSION, GCSO_FOUR_SECTION_HEADER_BYTES,
        GCSO_ZIMMS_DESCRIPTOR_PREAMBLE_OFFSET, ROOT_CAUSE_COHOMOLOGY_REPULSION,
        ROOT_CAUSE_FACT_RETRIEVAL_MISS, ROOT_CAUSE_NOMINAL, ROOT_CAUSE_PERSONA_FACT_COLLISION,
        ROOT_CAUSE_SYSTEM_QOS_THROTTLE,
    };
    pub use super::hot_path::{
        FractalCascadeCoordinator, HotPathExecutionEngine, HotPathObserver, PointerActionHub,
        SwarmCellChunk,
    };
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
        GcsoResult, Pod, PprcKeyframePacket, PprcMotionDeltaPacket, ACTION_MODE_DPSR_STEER,
        ACTION_MODE_LAYER_SKIP, ACTION_MODE_PHASE_REPULSE, ACTION_MODE_PSPM_ROUTE,
        ACTION_MODE_SRL_INJECT, ACTION_PRESET_EXPRESS_PHASE, ACTION_PRESET_FULL_ORCHESTRATION,
        ACTION_PRESET_INERTIAL_SKIP, ACTION_PRESET_PASSIVE_SIDECAR, ACTION_PRESET_REPULSION_ESCAPE,
        ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA, ATTRACTOR_SLOT_REPULSION,
        ATTRACTOR_SLOT_SYSTEM_BASELINE, ATTRACTOR_SLOT_USER_CONTEXT,
        DEFAULT_SLOT_COUPLING_MATRIX_8X8,
    };
    pub use super::rope::{
        resolve_autonomous_swarm_rot_dim, ripa_upper_rot_dim, DynamicLayerSkipper,
        EntropyGatedPhaseLookup, PhaseSteering, PspmRouter, SrlAdapter, VisualAttentionSteering,
    };
    pub use super::telemetry::{
        eval_bitmask_dyadic_ultrametric, swar_subswarm_resolve_x8, ContrastivePhaseBloomFilter,
        DaesScratchpad, PagedBitmaskOps, StigmergicActionPtrOps, StigmergicSwarmOps,
        StigmergicTrailOps, SwarmResolutionReport, DAES_METRIC_FAST_PATH_HIT,
        DAES_METRIC_LAYER_SKIP_COMPENSATED, DAES_METRIC_LAZY_UNWRAP_APPLIED,
        DAES_METRIC_PHASE_STEERED, DAES_METRIC_PSPM_ROUTED, DAES_METRIC_REPULSION_APPLIED,
        DAES_METRIC_SLOT_SYNERGY_ACTIVE, DAES_METRIC_SRL_INJECTED,
        DEFAULT_BLOOM_MAX_ACTIVE_BUCKETS, DEFAULT_BLOOM_MAX_MASS_THRESHOLD,
        DEFAULT_SWARM_QUORUM_THRESHOLD, DEFAULT_SWARM_TOLERANCE_Q7, HEADS_PER_SWARM_CELL,
        MAX_ATTRACTOR_PULL_FORCE, MAX_STIGMERGIC_DENSITY, NUM_SWARM_MICRO_CELLS,
        TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, TRAIL_FLAG_DPSR_ACTIVE, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE,
        TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_FLAG_SWARM_ESCALATED,
        TRAIL_FLAG_SWARM_RESOLVED, TRAIL_FLAG_SYNERGY_RESONANCE,
    };
    pub use super::ActionHub;
}

// ===================================================================
// Compile-Time Ground Truth C-ABI v0.1.1 Layout & Structural Parity
// ===================================================================

const _: () = {
    use crate::abi::{
        gcso_config_t, gcso_daes_slot_t, gcso_descriptor_header_t, gcso_edbc_state_t,
        gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_pprc_keyframe_header_t,
        gcso_pspm_config_t, gcso_snapshot_header_t, gcso_srl_descriptor_t, gcso_tagged_ptr_t,
        gcso_zimms_descriptor_t,
    };
    use core::mem::{align_of, offset_of, size_of};

    assert!(crate::abi::GCSO_ABI_VERSION_HEX == 0x0000_0101);
    assert!(MAX_ATTENTION_HEADS == 64);
    assert!(MAX_ANCHORS_PER_FIELD == 64);
    assert!(MAX_ATTRACTOR_DOMAINS == 8);
    assert!(NUM_SWARM_MICRO_CELLS == 8);
    assert!(HEADS_PER_SWARM_CELL == 8);
    assert!(NUM_SWARM_MICRO_CELLS * HEADS_PER_SWARM_CELL == MAX_ATTENTION_HEADS);
    assert!(CHEBYSHEV_FILTER_DEGREE == 3);
    assert!(CHEBYSHEV_COEFF_LEN == 4);
    assert!(PAGED_BITMASK_WORDS == 4);
    assert!(PAGED_BITMASK_BITS == 256);
    assert!(GCSO_FOUR_SECTION_HEADER_BYTES == 192);
    assert!(GCSO_ZIMMS_DESCRIPTOR_PREAMBLE_OFFSET == 128);

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
    assert!(offset_of!(gcso_config_t, daes_mode) == 38);

    assert!(size_of::<gcso_snapshot_header_t>() == 128);
    assert!(align_of::<gcso_snapshot_header_t>() == 64);
    assert!(offset_of!(gcso_snapshot_header_t, magic) == 0);
    assert!(offset_of!(gcso_snapshot_header_t, checksum_crc32) == 56);
    assert!(offset_of!(gcso_snapshot_header_t, timestamp_epoch_sec) == 64);

    assert!(size_of::<gcso_daes_slot_t>() == 64);
    assert!(align_of::<gcso_daes_slot_t>() == 64);
    assert!(offset_of!(gcso_daes_slot_t, mode) == 0);
    assert!(offset_of!(gcso_daes_slot_t, fast_path_shortcuts) == 24);
    assert!(offset_of!(gcso_daes_slot_t, telemetry_mini_ledger) == 56);

    assert!(size_of::<gcso_pprc_keyframe_header_t>() == 64);
    assert!(align_of::<gcso_pprc_keyframe_header_t>() == 32);
    assert!(offset_of!(gcso_pprc_keyframe_header_t, frame_type) == 0);
    assert!(offset_of!(gcso_pprc_keyframe_header_t, icache_payload_offset) == 24);
    assert!(offset_of!(gcso_pprc_keyframe_header_t, pcache_payload_offset) == 32);

    assert!(size_of::<PprcKeyframePacket>() == 128);
    assert!(align_of::<PprcKeyframePacket>() == 64);
    assert!(offset_of!(PprcKeyframePacket, header) == 0);
    assert!(offset_of!(PprcKeyframePacket, anchor_phases) == 64);
    assert!(size_of::<PprcMotionDeltaPacket>() == 64);
    assert!(align_of::<PprcMotionDeltaPacket>() == 64);
    assert!(offset_of!(PprcMotionDeltaPacket, motion_deltas) == 0);

    assert!(size_of::<gcso_pspm_config_t>() == 32);
    assert!(align_of::<gcso_pspm_config_t>() == 16);
    assert!(offset_of!(gcso_pspm_config_t, num_fact_heads) == 0);
    assert!(offset_of!(gcso_pspm_config_t, fact_phase_gain) == 8);

    assert!(size_of::<gcso_edbc_state_t>() == 64);
    assert!(align_of::<gcso_edbc_state_t>() == 32);
    assert!(offset_of!(gcso_edbc_state_t, moving_z_entropy) == 0);
    assert!(offset_of!(gcso_edbc_state_t, active_branch_mode) == 24);

    assert!(size_of::<gcso_srl_descriptor_t>() == 64);
    assert!(align_of::<gcso_srl_descriptor_t>() == 32);
    assert!(offset_of!(gcso_srl_descriptor_t, layer_idx) == 0);
    assert!(offset_of!(gcso_srl_descriptor_t, u_vector_ptr) == 8);
    assert!(offset_of!(gcso_srl_descriptor_t, v_vector_ptr) == 16);

    assert!(size_of::<gcso_zimms_descriptor_t>() == 64);
    assert!(align_of::<gcso_zimms_descriptor_t>() == 32);
    assert!(offset_of!(gcso_zimms_descriptor_t, mapped_address) == 0);
    assert!(offset_of!(gcso_zimms_descriptor_t, file_size_bytes) == 8);

    assert!(size_of::<ContrastivePhaseBloomFilter>() == 64);
    assert!(align_of::<ContrastivePhaseBloomFilter>() == 64);
    assert!(size_of::<SwarmResolutionReport>() == 16);
    assert!(align_of::<SwarmResolutionReport>() == 16);
    assert!(size_of::<CausalProvenanceReport>() == 16);
    assert!(align_of::<CausalProvenanceReport>() == 16);

    assert!(size_of::<GatewayFrame>() == 128);
    assert!(align_of::<GatewayFrame>() == 64);
    assert!(offset_of!(GatewayFrame, pspm_fact_gain_q7) == 56);
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
        gcso_anchor_type_t, gcso_config_t, gcso_daes_slot_t, gcso_edbc_state_t,
        gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_pspm_config_t, gcso_q7_t,
        gcso_tagged_ptr_t,
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

    impl PspmRouter for MockActionHubComposite {}
    impl SrlAdapter for MockActionHubComposite {}

    impl DaesScratchpad for MockActionHubComposite {
        fn evaluate_auto_tune(
            &self,
            _slot: &gcso_daes_slot_t,
            _config_out: &mut gcso_config_t,
        ) -> GcsoResult<()> {
            Ok(())
        }
    }

    impl SheafCohomologyEvaluator for MockActionHubComposite {}

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
        fn assert_action_hub<T: ActionHub + HotPathObserver + ActionHubGatewayExt>(
            hub: &T,
        ) -> (u32, u32) {
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
        let lmax =
            <MockActionHubComposite as SheafCohomologyEvaluator>::eval_gershgorin_lambda_max(
                &hub,
                &deg.as_aligned_slice(),
            )
            .unwrap();
        assert!((lmax - 6.0).abs() < 1e-5);

        let mut daes = gcso_daes_slot_t::default();
        let mut bloom = ContrastivePhaseBloomFilter::new();
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
                &mut daes,
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
        assert_ne!(daes.telemetry_mini_ledger[0] & DAES_METRIC_PHASE_STEERED, 0);

        trail.stigmergic_density = 1.5;
        let mut trails = [trail];
        let mut persona_out = [0i8; MAX_ATTENTION_HEADS];
        let (mode, candidates, new_anchors, r_bar, kappa) = hub
            .step_macro_crystallization_with_persona_and_tunneling(
                &mut trails,
                &mut bloom,
                0.60,
                0.25,
                0.75,
                0.45,
                0.15,
                0.02,
                0.9,
                0.8,
                1.5,
                0.5,
                128,
                48,
                0.8,
                &mut persona_out,
            )
            .unwrap();
        assert_eq!(mode, 1);
        assert_eq!(candidates, 1);
        assert_eq!(new_anchors, 1);
        assert!(r_bar > 0.85);
        assert!(kappa > 3.0);
    }

    #[test]
    fn test_action_hub_unified_field_kmeans_top2_liminal_and_bayesian_attention_modulation() {
        let mut hub = MockActionHubComposite {
            head_dim: 8,
            num_heads: 2,
            default_phase: 24,
            branch_mode: 0,
            crystallized_count: 1,
        };

        let mut trail = gcso_pointer_trail_t {
            accumulated_phase_delta: [20i8; MAX_ATTENTION_HEADS],
            attractor_pull_force: 3.2,
            stigmergic_density: 2.4,
            linked_trail_id: 1,
            target_anchor_id: 42,
            cluster_id: 42,
            ..Default::default()
        };

        let anchors = [
            [-60i8; MAX_ATTENTION_HEADS],
            [20i8; MAX_ATTENTION_HEADS],
            [0i8; MAX_ATTENTION_HEADS],
        ];
        let energies = [0.1f32, 3.5, 0.5];
        let mut q_buf = AlignedBuffer32::new([1.0f32; 16]);
        let mut k_buf = AlignedBuffer32::new([0.5f32; 16]);
        let q_norm_before = sum_sq_f32_simd8(&q_buf.data);

        let (steered, tot_pull, mean_coh, dom_idx) = hub
            .filter_qdps_and_steer_qk_unified_field_safe(
                &trail,
                &anchors,
                &energies,
                DEFAULT_QDPS_MIN_STEP_RAD,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                Some(&mut q_buf.as_aligned_slice_mut()),
                Some(&mut k_buf.as_aligned_slice_mut()),
            )
            .unwrap();
        assert!(steered);
        assert_eq!(dom_idx, 1);
        assert!(tot_pull > 9.0 && mean_coh > 0.80);
        let q_norm_after = sum_sq_f32_simd8(&q_buf.data);
        assert!((q_norm_before - q_norm_after).abs() < 1e-4);

        let pspm_cfg = gcso_pspm_config_t {
            num_fact_heads: 1,
            num_logic_heads: 0,
            num_explore_heads: 1,
            fact_phase_gain: 1.0,
            logic_phase_gain: 1.0,
            explore_phase_gain: 1.25,
            ..Default::default()
        };
        let (steered_pspm, routed_pspm) = hub
            .filter_qdps_and_steer_qk_with_unified_pspm_safe(
                &trail.accumulated_phase_delta,
                1,
                &anchors,
                Some((&pspm_cfg, 8)),
                DEFAULT_QDPS_MIN_STEP_RAD,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                Some(&mut q_buf.as_aligned_slice_mut()),
                Some(&mut k_buf.as_aligned_slice_mut()),
            )
            .unwrap();
        assert!(steered_pspm && routed_pspm);

        let mut centroids = [
            CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_KNOWLEDGE, 0),
            CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_PERSONA, 0),
        ];
        let fact_sig = ObjectiveFact::new([22i8; MAX_ATTENTION_HEADS], 3.0, 0x0010_0000, 303);
        let mut daes = gcso_daes_slot_t::default();
        let (_frame, _rep, c_idx, c_id, c_mass, spawned) = hub
            .ingest_gateway_kmeans_and_steer_safe(
                &fact_sig,
                &mut centroids,
                DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
                &mut daes,
                &mut trail,
                Some(&mut q_buf.as_aligned_slice_mut()),
                Some(&mut k_buf.as_aligned_slice_mut()),
                None,
            )
            .unwrap();
        assert!(spawned);
        assert_eq!(c_idx, 0);
        assert_eq!(c_id, 303);
        assert!(c_mass > 0.0);

        let (routed_idx, routed_id, routed_coh, routed_grav, routed_steered) = hub
            .route_coarse_to_fine_and_steer_qk_safe(
                &centroids,
                64,
                DEFAULT_QDPS_MIN_STEP_RAD,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                &mut trail,
                Some(&mut q_buf.as_aligned_slice_mut()),
                None,
            )
            .unwrap();
        assert_eq!(routed_idx, 0);
        assert_eq!(routed_id, 303);
        assert!(routed_coh > 0.90 && routed_grav > 1.0 && routed_steered);

        // Populate second centroid to verify Top-2 Liminal Bifurcation Routing & Steering
        let _ = centroids[1].absorb_sample_point_64(&[46i8; MAX_ATTENTION_HEADS], 3.0);
        centroids[1].anchor_id = 404;
        centroids[1].stamp_crc32();
        trail.accumulated_phase_delta = [34i8; MAX_ATTENTION_HEADS];
        let (idx1, idx2, aid1, aid2, lim_coh, lim_grav, is_liminal, lim_steered) = hub
            .route_top2_liminal_and_steer_qk_safe(
                &centroids,
                64,
                DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO,
                Some((&pspm_cfg, 4)),
                DEFAULT_QDPS_MIN_STEP_RAD,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                &mut trail,
                Some(&mut q_buf.as_aligned_slice_mut()),
                Some(&mut k_buf.as_aligned_slice_mut()),
            )
            .unwrap();
        assert!(is_liminal && lim_steered);
        assert!(idx1 != idx2 && aid1 != 0 && aid2 != 0);
        assert!(lim_coh > 0.80 && lim_grav > 1.0);
        assert_ne!(trail.flags & TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, 0);

        let mut uf_scores = AlignedBuffer32::new([0.0f32; 3]);
        trail.accumulated_phase_delta = [20i8; MAX_ATTENTION_HEADS];
        let uf_coh = hub
            .modulate_hub_attention_scores_by_unified_field(
                &mut uf_scores.as_aligned_slice_mut(),
                &trail,
                &anchors,
                1.2,
            )
            .unwrap();
        assert!(uf_coh.is_finite());
        assert!(uf_scores.data[1] > uf_scores.data[2]);
        assert!(uf_scores.data[2] > uf_scores.data[0]);

        let mut bayes_scores = AlignedBuffer32::new([0.0f32; 3]);
        trail.linked_trail_id = 1;
        let (mean_coh_b, best_cand) = hub
            .modulate_hub_attention_scores_by_bayesian_trails(
                &mut bayes_scores.as_aligned_slice_mut(),
                &trail,
                &anchors,
                &energies,
                1.2,
                0.4,
                0.75,
            )
            .unwrap();
        assert!(mean_coh_b.is_finite());
        assert_eq!(best_cand, 1);
        assert!(bayes_scores.data[1] > uf_scores.data[1] + 1.0);

        // Verify unstamped candidate trail (cluster_id == 0) is linked via continuous geodesic affinity
        let cand0 = gcso_pointer_trail_t {
            stigmergic_density: 3.0,
            accumulated_phase_delta: [-96i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let cand1 = gcso_pointer_trail_t {
            stigmergic_density: 3.5,
            target_anchor_id: 0,
            cluster_id: 0,
            accumulated_phase_delta: [22i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let mut cand_trails = [cand0, cand1];
        let mut cand_scores = AlignedBuffer32::new([0.0f32; 2]);
        let (m_idx, m_id, sel_id, sel_score, found, c2f_steered) = hub
            .route_coarse_to_fine_bayesian_and_steer_qk_safe(
                0,
                &centroids,
                64,
                &mut trail,
                &mut cand_trails,
                Some(&mut cand_scores.as_aligned_slice_mut()),
                1.5,
                0.5,
                0.25,
                1.0,
                64,
                0.25,
                Some((&pspm_cfg, 4)),
                DEFAULT_QDPS_MIN_STEP_RAD,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                &mut daes,
                Some(&mut q_buf.as_aligned_slice_mut()),
                Some(&mut k_buf.as_aligned_slice_mut()),
            )
            .unwrap();
        assert_eq!(m_idx, 0);
        assert_eq!(m_id, 303);
        assert!(found && sel_id == 1 && sel_score > 0.5 && c2f_steered);
        assert_eq!(trail.target_anchor_id, 303);
        assert!(cand_scores.data[1] > 1.0 && cand_scores.data[0] < -0.5);

        let (ens_pull, ens_id, ens_steered) = hub
            .steer_crystallized_ensemble_on_hub_safe(
                &centroids,
                DEFAULT_QDPS_MIN_STEP_RAD,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                &mut daes,
                &mut trail,
                Some(&mut q_buf.as_aligned_slice_mut()),
                None,
            )
            .unwrap();
        assert_eq!(ens_id, 303);
        assert!(ens_pull > 1.0 && ens_steered);

        // Verify 4-Section `.gcso` zero-copy in-place evolution, Coarse-to-Fine, Top-2 Liminal routing & Q/K steering on ActionHub
        let mut container_buf = AlignedBuffer128::new([0u8; 1024]);
        let written = GcsoFourSectionView::write_four_section_with_reports_to_bytes(
            &centroids,
            &cand_trails,
            None,
            Some(&daes),
            &[],
            999,
            &mut container_buf.data,
        )
        .unwrap();
        let (_ev_frame, ev_idx, ev_aid, ev_mass, _ev_spawned, _ev_merged, ev_steered) = hub
            .evolve_four_section_kmeans_and_steer_qk_safe(
                &mut container_buf.data[..written],
                &fact_sig,
                DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
                DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE,
                1,
                Some(&mut q_buf.as_aligned_slice_mut()),
                None,
            )
            .unwrap();
        assert_eq!(ev_idx, 0);
        assert_eq!(ev_aid, 303);
        assert!(ev_mass > 1.0 && ev_steered);

        let (fs_idx, fs_aid, fs_coh, fs_grav, fs_steered) = hub
            .route_four_section_coarse_to_fine_and_steer_qk_safe(
                &mut container_buf.data[..written],
                1,
                64,
                DEFAULT_QDPS_MIN_STEP_RAD,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                Some(&mut q_buf.as_aligned_slice_mut()),
                Some(&mut k_buf.as_aligned_slice_mut()),
            )
            .unwrap();
        assert_eq!(fs_idx, 0);
        assert_eq!(fs_aid, 303);
        assert!(fs_coh > 0.90 && fs_grav > 1.0 && fs_steered);

        let (
            lim_idx1,
            lim_idx2,
            lim_aid1,
            lim_aid2,
            lim_fs_coh,
            lim_fs_grav,
            lim_fs_bif,
            lim_fs_steered,
        ) = hub
            .route_four_section_top2_liminal_and_steer_qk_safe(
                &mut container_buf.data[..written],
                1,
                64,
                DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO,
                Some((&pspm_cfg, 4)),
                DEFAULT_QDPS_MIN_STEP_RAD,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                Some(&mut q_buf.as_aligned_slice_mut()),
                Some(&mut k_buf.as_aligned_slice_mut()),
            )
            .unwrap();
        assert!(lim_fs_bif && lim_fs_steered);
        assert!(lim_idx1 != lim_idx2 && lim_aid1 != 0 && lim_aid2 != 0);
        assert!(lim_fs_coh > 0.80 && lim_fs_grav > 1.0);

        let mapped_view =
            GcsoFourSectionView::from_zimms_bytes_zero_copy(&container_buf.data[..written])
                .unwrap();
        assert_eq!(mapped_view.trails[1].cluster_id, lim_aid1);
        assert_ne!(
            mapped_view.trails[1].flags & TRAIL_FLAG_LIMINAL_BEAT_ACTIVE,
            0
        );

        let mut logits = AlignedBuffer32::new([1.0f32, -1.0, 2.0, -2.0]);
        let (t_coh, inv_temp) = hub
            .modulate_hub_logit_inv_temperature(
                &mut logits.as_aligned_slice_mut(),
                &trail,
                &trail.accumulated_phase_delta,
                1.0,
            )
            .unwrap();
        assert!((t_coh - 1.0).abs() < 1e-5 && (inv_temp - 2.0).abs() < 1e-5);
        assert!((logits.data[0] - 2.0).abs() < 1e-5);
    }
}
