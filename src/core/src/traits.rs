// name: src/core/src/traits.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Root trait definitions, `ActionHub` micro-capability composite ledger, zero-copy alignment
//! wrappers, non-aliasing (`GCSO_RESTRICT`) type guards, branchless mathematical/SIMD primitives,
//! 2-Origin Composite Manifold Gateway (`gateway`), Online Toroidal $k$-Means / DP-Means
//! Coarse-to-Fine & Top-2 Liminal Bifurcation routing, and Ground Truth C-ABI contract assertions
//! for the Geometric Cellular Sheaf Orchestrator (GCSO) engine.
//!
//! # Architectural Role (1 + 8 Flat Module Topology: `ActionHub` + 2 Observers + Gateway)
//! The `crate::traits` hierarchy separates **the thing itself (`ActionHub`)** from **two observers
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
//!     [`ActionHubGatewayExt`]), Classical Online Toroidal $k$-Means / DP-Means Macro Clustering,
//!     Top-2 Liminal Bifurcation Routing, and `.gcso` 4-section zero-copy crystallized attractor
//!     persistence ([`CrystallizedAttractorRecord`], [`GcsoFourSectionView`]).

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

use core::mem::size_of;

use self::primitives::load_q7_x8_slice;
use self::rope::{
    dispatch_fused_ripa_and_pspm_words_core,
    rotate_fused_self_ripa_pspm_q_and_ripa_k_default_words, steer_from_swar_words_core_safe,
};
pub(crate) use self::telemetry::select_rewire_bridge_and_modulate_with_preloaded_words_64;
use crate::abi::{
    gcso_daes_slot_t, gcso_pointer_trail_t, gcso_pspm_config_t, gcso_q7_t, gcso_snapshot_header_t,
    gcso_srl_descriptor_t, gcso_tagged_ptr_t, GCSO_ERROR_INVALID_ARGUMENT,
};

// ===================================================================
// Core Composite Ledger: `ActionHub` ("The Thing Itself")
// ===================================================================

/// Loads the 8 register-resident `u64` SWAR phase words of `trails[trail_idx]` directly from an
/// already-validated 4-section `.gcso` byte slice (`zimms_bytes`) without re-running a second
/// full-container IEEE 802.3 CRC32 verification pass.
#[inline(always)]
#[must_use]
pub(crate) fn load_validated_four_section_trail_words_64(
    zimms_bytes: &[u8],
    trail_idx: usize,
    fallback_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
) -> [u64; NUM_SWARM_MICRO_CELLS] {
    if let Ok(header) = gcso_snapshot_header_t::from_bytes_at(zimms_bytes, 0) {
        let off_trails = header.action_hub_offset as usize;
        let off_bloom = header.edbc_state_offset as usize;
        let trail_size = size_of::<gcso_pointer_trail_t>();
        if off_trails <= off_bloom && off_bloom <= zimms_bytes.len() {
            let num_trails = (off_bloom - off_trails) / trail_size;
            if trail_idx < num_trails {
                let trail_off = off_trails + trail_idx * trail_size;
                if let Ok(trail_ref) = gcso_pointer_trail_t::from_bytes_at(zimms_bytes, trail_off) {
                    let trail_w = trail_phase_words_ref(trail_ref);
                    if swar_words_nonzero_64(trail_w) {
                        return *trail_w;
                    }
                }
            }
        }
    }
    load_q7_words_64(fallback_phases)
}

/// Accumulates up to 64 Q7 phase deltas from `delta_slice` directly into an 8-word (`[u64; 8]`)
/// register-resident SWAR phase array using 8-lane saturating addition (`swar_q7_add_sat_x8`),
/// eliminating 64-byte stack buffer spills and Store-to-Load Forwarding (STLF) reload stalls.
#[inline(always)]
pub(crate) fn accumulate_q7_slice_into_words_64(
    acc_words: &mut [u64; NUM_SWARM_MICRO_CELLS],
    delta_slice: &[gcso_q7_t],
) {
    let limit = delta_slice.len().min(MAX_ATTENTION_HEADS);
    if limit == MAX_ATTENTION_HEADS {
        acc_words[0] = swar_q7_add_sat_x8(acc_words[0], load_q7_x8_slice(delta_slice, 0));
        acc_words[1] = swar_q7_add_sat_x8(acc_words[1], load_q7_x8_slice(delta_slice, 8));
        acc_words[2] = swar_q7_add_sat_x8(acc_words[2], load_q7_x8_slice(delta_slice, 16));
        acc_words[3] = swar_q7_add_sat_x8(acc_words[3], load_q7_x8_slice(delta_slice, 24));
        acc_words[4] = swar_q7_add_sat_x8(acc_words[4], load_q7_x8_slice(delta_slice, 32));
        acc_words[5] = swar_q7_add_sat_x8(acc_words[5], load_q7_x8_slice(delta_slice, 40));
        acc_words[6] = swar_q7_add_sat_x8(acc_words[6], load_q7_x8_slice(delta_slice, 48));
        acc_words[7] = swar_q7_add_sat_x8(acc_words[7], load_q7_x8_slice(delta_slice, 56));
        return;
    }

    let full_words = limit >> 3;
    let mut w_idx = 0usize;
    while w_idx < full_words {
        acc_words[w_idx] =
            swar_q7_add_sat_x8(acc_words[w_idx], load_q7_x8_slice(delta_slice, w_idx << 3));
        w_idx += 1;
    }
    let rem = limit & 7;
    if rem != 0 {
        let base_h = full_words << 3;
        let mut tail_bytes = [0u8; 8];
        let mut lane = 0usize;
        while lane < rem {
            tail_bytes[lane] = delta_slice[base_h + lane] as u8;
            lane += 1;
        }
        let delta_w = u64::from_le_bytes(tail_bytes);
        acc_words[full_words] = swar_q7_add_sat_x8(acc_words[full_words], delta_w);
    }
}

/// Validates optional Query/Key tensor slice lengths and `GCSO_RESTRICT` pairwise disjointness.
#[inline(always)]
pub(crate) fn validate_optional_qk_tensors(
    query_tensor: Option<&AlignedSliceMut32<'_, f32>>,
    key_tensor: Option<&AlignedSliceMut32<'_, f32>>,
    expected_len: usize,
) -> GcsoResult<()> {
    match (query_tensor, key_tensor) {
        (Some(q), Some(k)) => {
            if q.len() < expected_len || k.len() < expected_len {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            q.ensure_disjoint_mut(k)
        }
        (Some(q), None) => {
            if q.len() < expected_len {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            Ok(())
        }
        (None, Some(k)) => {
            if k.len() < expected_len {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            Ok(())
        }
        (None, None) => Ok(()),
    }
}

/// Computes register-resident 8-word (`[u64; 8]`) Lazy Phase Unwrap (`query - context`, either
/// saturating or $S^1$ geodesic-folded) + 8-lane SWAR QDPS discrete step filtering directly in
/// 64-bit registers (`0` stack buffer allocation when consumed by `steer_from_swar_words_core_safe`).
#[inline(always)]
#[must_use]
pub(crate) fn fuse_lazy_unwrap_and_qdps_from_words_64(
    q_w: &[u64; NUM_SWARM_MICRO_CELLS],
    query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    context_slice: &[gcso_q7_t],
    num_heads: usize,
    qdps_min_step_rad: f32,
    fold_geodesic: bool,
) -> [u64; NUM_SWARM_MICRO_CELLS] {
    let min_step_q7 = branchless_rad_to_qdps_step_u8(qdps_min_step_rad);
    let limit = num_heads.min(MAX_ATTENTION_HEADS).min(context_slice.len());

    if limit == MAX_ATTENTION_HEADS {
        if let Ok(c_arr) =
            <&[gcso_q7_t; MAX_ATTENTION_HEADS]>::try_from(&context_slice[..MAX_ATTENTION_HEADS])
        {
            let c_w = load_q7_words_64(c_arr);
            if is_word_array_uniform_64(q_w) && is_word_array_uniform_64(&c_w) {
                let raw0 = if fold_geodesic {
                    swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(q_w[0], c_w[0]))
                } else {
                    swar_q7_sub_sat_x8(q_w[0], c_w[0])
                };
                let filt0 = if min_step_q7 <= 1 {
                    raw0
                } else {
                    swar_q7_qdps_filter_x8(raw0, min_step_q7)
                };
                return [filt0; NUM_SWARM_MICRO_CELLS];
            }
            let raw_w = if fold_geodesic {
                [
                    swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(q_w[0], c_w[0])),
                    swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(q_w[1], c_w[1])),
                    swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(q_w[2], c_w[2])),
                    swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(q_w[3], c_w[3])),
                    swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(q_w[4], c_w[4])),
                    swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(q_w[5], c_w[5])),
                    swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(q_w[6], c_w[6])),
                    swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(q_w[7], c_w[7])),
                ]
            } else {
                [
                    swar_q7_sub_sat_x8(q_w[0], c_w[0]),
                    swar_q7_sub_sat_x8(q_w[1], c_w[1]),
                    swar_q7_sub_sat_x8(q_w[2], c_w[2]),
                    swar_q7_sub_sat_x8(q_w[3], c_w[3]),
                    swar_q7_sub_sat_x8(q_w[4], c_w[4]),
                    swar_q7_sub_sat_x8(q_w[5], c_w[5]),
                    swar_q7_sub_sat_x8(q_w[6], c_w[6]),
                    swar_q7_sub_sat_x8(q_w[7], c_w[7]),
                ]
            };
            return swar_q7_qdps_filter_words_64(&raw_w, min_step_q7);
        }
    }

    let mut out_words = [0u64; NUM_SWARM_MICRO_CELLS];
    let mut h = 0usize;
    while h + 8 <= limit {
        let qw = load_q7_x8_slice(query_phases, h);
        let cw = load_q7_x8_slice(context_slice, h);
        let raw_w = if fold_geodesic {
            swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(qw, cw))
        } else {
            swar_q7_sub_sat_x8(qw, cw)
        };
        out_words[h >> 3] = if min_step_q7 <= 1 {
            raw_w
        } else {
            swar_q7_qdps_filter_x8(raw_w, min_step_q7)
        };
        h += 8;
    }
    if h < limit {
        let word_idx = h >> 3;
        let mut tail_bytes = [0u8; 8];
        let mut lane = 0usize;
        while h + lane < limit {
            let idx = h + lane;
            let raw_diff = if fold_geodesic {
                branchless_q7_fold_s1_geodesic(branchless_q7_circular_diff(
                    query_phases[idx],
                    context_slice[idx],
                ))
            } else {
                branchless_q7_sub_sat(query_phases[idx], context_slice[idx])
            };
            let filtered = if min_step_q7 <= 1 {
                raw_diff
            } else {
                branchless_qdps_filter_q7(raw_diff, i32::from(min_step_q7))
            };
            tail_bytes[lane] = filtered as u8;
            lane += 1;
        }
        out_words[word_idx] = u64::from_le_bytes(tail_bytes);
    }
    out_words
}

/// Computes register-resident 8-word (`[u64; 8]`) Lazy Phase Unwrap (`query - context`, either
/// saturating or $S^1$ geodesic-folded) + 8-lane SWAR QDPS discrete step filtering directly in
/// 64-bit registers (`0` stack buffer allocation when consumed by `steer_from_swar_words_core_safe`).
#[inline(always)]
#[must_use]
pub(crate) fn fuse_lazy_unwrap_and_qdps_words_64(
    query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    context_slice: &[gcso_q7_t],
    num_heads: usize,
    qdps_min_step_rad: f32,
    fold_geodesic: bool,
) -> [u64; NUM_SWARM_MICRO_CELLS] {
    let q_w = load_q7_words_64(query_phases);
    fuse_lazy_unwrap_and_qdps_from_words_64(
        &q_w,
        query_phases,
        context_slice,
        num_heads,
        qdps_min_step_rad,
        fold_geodesic,
    )
}

/// Single-pass register-fused Lazy Phase Unwrap (`query - context`, either saturating or
/// $S^1$ geodesic-folded) + 8-lane SWAR QDPS discrete step filtering directly into `dst_aligned`.
#[inline(always)]
pub fn fuse_lazy_unwrap_and_qdps_into_aligned_64(
    query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    context_slice: &[gcso_q7_t],
    num_heads: usize,
    qdps_min_step_rad: f32,
    fold_geodesic: bool,
    dst_aligned: &mut AlignedBuffer16<gcso_q7_t, MAX_ATTENTION_HEADS>,
) {
    let words = fuse_lazy_unwrap_and_qdps_words_64(
        query_phases,
        context_slice,
        num_heads,
        qdps_min_step_rad,
        fold_geodesic,
    );
    store_q7_words_64(&mut dst_aligned.data, &words);
}

/// The bottom-most micro composite ledger of state and branchless capabilities (**"The Thing Itself"**).
///
/// `ActionHub` bundles the fundamental constant-time capabilities registered on the engine:
/// - [`BranchlessKernel`]: Type-level constant-time execution guarantee.
/// - [`PointerActionHub`]: $\mathcal{O}(1)$ tagged-pointer transition table and stigmergic trail state.
/// - [`DaesScratchpad`]: 64-byte fast-path shortcut table and 8-byte ring telemetry ledger.
/// - [`PhaseSteering`]: Inline $SO(2)^{d_{\mathrm{head}}/2}$ DPSR/RIPA/QDPS phase & Bayesian-Attention modulation.
/// - [`PspmRouter`]: Single-pass Fact/Logic/Explore sub-head and Unified-Field Pseudo-MoE routing.
/// - [`SrlAdapter`]: Dynamic Rank-1 outer product residual injection ($\mathbf{u}\mathbf{v}^T$).
///
/// Both [`HotPathObserver`] (`traits/hot_path.rs`), [`ColdPathObserver`] (`traits/cold_path.rs`),
/// and [`ActionHubGatewayExt`] (`traits/gateway.rs`) observe and mutate `ActionHub` via blanket implementations.
pub trait ActionHub:
    BranchlessKernel + PhaseSteering + PointerActionHub + PspmRouter + SrlAdapter + DaesScratchpad
{
    /// Validates `head_dim`, `num_heads <= MAX_ATTENTION_HEADS`, `qdps_min_step_rad >= 0.0`,
    /// `ripa_max_rad > 0.0`, and optional Query/Key tensor lengths and `GCSO_RESTRICT` disjointness.
    ///
    /// Returns `(head_dim, num_heads, expected_tensor_len)` on success.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any dimension, radian threshold, or tensor slice
    /// invariant is violated.
    #[inline(always)]
    fn validate_hub_cascade_params(
        &self,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(usize, usize, usize)> {
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if num_heads > MAX_ATTENTION_HEADS
            || qdps_min_step_rad < 0.0
            || !branchless_is_finite_f32(qdps_min_step_rad)
            || ripa_max_rad <= 0.0
            || !branchless_is_finite_f32(ripa_max_rad)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        validate_optional_qk_tensors(query_tensor, key_tensor, expected_len)?;
        Ok((head_dim, num_heads, expected_len))
    }

    /// Resolves `input_ptr` against `daes_slot`'s $\mathcal{O}(1)$ shortcut table, advances the
    /// pointer transition in `trail_out`, evaluates `bloom_filter` anti-phase repulsion in-place,
    /// and returns `(bypass_hit, effective_ptr, repulsion_density, repulsion_active, post_repulsion_words)`
    /// directly in 8 `u64` SWAR registers (`0` stack reload for downstream QDPS/RIPA or unified-field kernels).
    ///
    /// # Errors
    /// Propagates any error returned by [`PointerActionHub::step_pointer`].
    #[inline(always)]
    fn step_hub_with_bloom_repulsion_to_words(
        &mut self,
        input_ptr: gcso_tagged_ptr_t,
        daes_slot: &mut gcso_daes_slot_t,
        bloom_filter: &ContrastivePhaseBloomFilter,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(
        bool,
        gcso_tagged_ptr_t,
        u8,
        bool,
        [u64; NUM_SWARM_MICRO_CELLS],
    )> {
        let (bypass_hit, effective_ptr) = self.resolve_or_pass_tagged_ptr(daes_slot, input_ptr);
        self.step_pointer(effective_ptr.raw(), trail_out)?;

        let (density, repulsion_active, words) = bloom_filter
            .apply_phase_conjugate_repulsion_to_words(
                &mut trail_out.accumulated_phase_delta,
                repulsion_threshold,
                repulsion_scale_u7,
            );
        trail_out.flags |= branchless_select_u32(repulsion_active, TRAIL_FLAG_REPULSION_ACTIVE, 0);
        Ok((bypass_hit, effective_ptr, density, repulsion_active, words))
    }

    /// Resolves `input_ptr` against `daes_slot`'s $\mathcal{O}(1)$ shortcut table, advances the
    /// pointer transition in `trail_out`, and evaluates `bloom_filter` anti-phase repulsion in-place.
    ///
    /// Returns `(bypass_hit, effective_ptr, repulsion_density, repulsion_active)`.
    ///
    /// # Errors
    /// Propagates any error returned by [`PointerActionHub::step_pointer`].
    #[inline(always)]
    fn step_hub_with_bloom_repulsion(
        &mut self,
        input_ptr: gcso_tagged_ptr_t,
        daes_slot: &mut gcso_daes_slot_t,
        bloom_filter: &ContrastivePhaseBloomFilter,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(bool, gcso_tagged_ptr_t, u8, bool)> {
        let (bypass_hit, effective_ptr, density, repulsion_active, _words) = self
            .step_hub_with_bloom_repulsion_to_words(
                input_ptr,
                daes_slot,
                bloom_filter,
                repulsion_threshold,
                repulsion_scale_u7,
                trail_out,
            )?;
        Ok((bypass_hit, effective_ptr, density, repulsion_active))
    }

    /// Pre-validated internal core for [`Self::filter_qdps_words_and_steer_qk_safe`]: filters
    /// `raw_words` via 8-lane SWAR QDPS and steers `query_tensor` / `key_tensor` using caller-validated
    /// `(head_dim, num_heads)` (`0` redundant `validate_hub_cascade_params` checks inside cascades).
    #[inline(always)]
    fn steer_qdps_words_prevalidated_core(
        &self,
        raw_words: &[u64; NUM_SWARM_MICRO_CELLS],
        head_dim: usize,
        num_heads: usize,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> bool {
        let steered = query_tensor.is_some() | key_tensor.is_some();
        if !steered {
            return false;
        }
        let min_step_q7 = if qdps_min_step_rad.to_bits() == 0 {
            0
        } else {
            branchless_rad_to_qdps_step_u8(qdps_min_step_rad)
        };
        let filtered_buf;
        let swar_words = if min_step_q7 <= 1 {
            raw_words
        } else {
            filtered_buf = swar_q7_qdps_filter_words_64(raw_words, min_step_q7);
            &filtered_buf
        };
        if !swar_words_nonzero_64(swar_words) {
            return true;
        }

        match (query_tensor, key_tensor) {
            (Some(q), k_opt) => {
                steer_from_swar_words_core_safe(
                    swar_words,
                    0xFF,
                    128,
                    head_dim,
                    num_heads,
                    false,
                    ripa_max_rad,
                    q,
                    k_opt,
                );
            }
            (None, Some(k)) => {
                steer_from_swar_words_core_safe(
                    swar_words,
                    0xFF,
                    128,
                    head_dim,
                    num_heads,
                    false,
                    ripa_max_rad,
                    k,
                    None,
                );
            }
            (None, None) => {}
        }
        true
    }

    /// Filters 8 register-resident `u64` SWAR words (`raw_words`) via 8-lane SWAR QDPS
    /// (`swar_q7_qdps_filter_words_64`) and steers `query_tensor` and/or `key_tensor`
    /// with RIPA soft-clamping (`ripa_max_rad`) directly from registers (`0` stack store/reload).
    ///
    /// When all 64 heads are cut off to `0` by QDPS (`!swar_words_nonzero_64(&swar_words)`),
    /// skips the tensor memory pass because $\mathbf{R}_2(0) = \mathbf{I}_2$.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if dimensions, thresholds, or tensor slices are invalid.
    #[inline(always)]
    fn filter_qdps_words_and_steer_qk_safe(
        &self,
        raw_words: &[u64; NUM_SWARM_MICRO_CELLS],
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<bool> {
        if !(query_tensor.is_some() | key_tensor.is_some()) {
            return Ok(false);
        }
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        Ok(self.steer_qdps_words_prevalidated_core(
            raw_words,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        ))
    }

    /// Loads `raw_phases` into 8 `u64` SWAR registers, applies 8-lane SWAR QDPS step filtering in
    /// registers (`swar_q7_qdps_filter_words_64`), and steers `query_tensor` and/or `key_tensor`
    /// with RIPA soft-clamping (`ripa_max_rad`) without writing/reloading an intermediate stack buffer.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if dimensions, thresholds, or tensor slices are invalid.
    #[inline(always)]
    fn filter_qdps_and_steer_qk_safe(
        &self,
        raw_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<bool> {
        let raw_words = load_q7_words_64(raw_phases);
        self.filter_qdps_words_and_steer_qk_safe(
            &raw_words,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        )
    }

    /// Pre-validated internal core for [`Self::filter_qdps_and_steer_qk_with_pspm_words_safe`],
    /// accepting caller-validated `(head_dim, num_heads)` (`0` redundant validation checks).
    ///
    /// # Errors
    /// Propagates any PSPM configuration error from `dispatch_fused_ripa_and_pspm_words_core`.
    #[inline(always)]
    fn steer_qdps_pspm_words_prevalidated_core(
        &self,
        raw_words: &[u64; NUM_SWARM_MICRO_CELLS],
        fact_words: &[u64; NUM_SWARM_MICRO_CELLS],
        logic_words: &[u64; NUM_SWARM_MICRO_CELLS],
        pspm_config_and_pulse: Option<(&gcso_pspm_config_t, gcso_q7_t)>,
        head_dim: usize,
        num_heads: usize,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(bool, bool)> {
        let Some((pspm_cfg, pulse_q7)) = pspm_config_and_pulse else {
            let steered = self.steer_qdps_words_prevalidated_core(
                raw_words,
                head_dim,
                num_heads,
                qdps_min_step_rad,
                ripa_max_rad,
                query_tensor,
                key_tensor,
            );
            return Ok((steered, false));
        };

        let has_q = query_tensor.is_some();
        let steered = has_q | key_tensor.is_some();
        if !steered {
            return Ok((false, false));
        }
        let min_step_q7 = if qdps_min_step_rad.to_bits() == 0 {
            0
        } else {
            branchless_rad_to_qdps_step_u8(qdps_min_step_rad)
        };
        let filtered_buf;
        let swar_words = if min_step_q7 <= 1 {
            raw_words
        } else {
            filtered_buf = swar_q7_qdps_filter_words_64(raw_words, min_step_q7);
            &filtered_buf
        };

        match (query_tensor, key_tensor) {
            (Some(q), Some(k)) => {
                let total_pspm_heads = (pspm_cfg.num_fact_heads as usize)
                    .wrapping_add(pspm_cfg.num_logic_heads as usize)
                    .wrapping_add(pspm_cfg.num_explore_heads as usize);
                let is_default_self_fast = ripa_max_rad.to_bits()
                    == DEFAULT_RIPA_CLAMP_MAX_RAD.to_bits()
                    && pulse_q7 == 0
                    && total_pspm_heads == num_heads
                    && pspm_cfg.fact_phase_gain.to_bits() == 1.0_f32.to_bits()
                    && pspm_cfg.logic_phase_gain.to_bits() == 1.0_f32.to_bits()
                    && pspm_cfg.explore_phase_gain.to_bits() == 1.0_f32.to_bits()
                    && (core::ptr::eq(fact_words, swar_words) || fact_words == swar_words)
                    && (core::ptr::eq(logic_words, swar_words) || logic_words == swar_words)
                    && (core::ptr::eq(raw_words, swar_words) || raw_words == swar_words);
                if is_default_self_fast {
                    if swar_words_nonzero_64(swar_words) {
                        rotate_fused_self_ripa_pspm_q_and_ripa_k_default_words(
                            q.as_mut_slice(),
                            k.as_mut_slice(),
                            swar_words,
                            num_heads,
                            head_dim,
                            ripa_upper_rot_dim(head_dim),
                        );
                    }
                    return Ok((true, true));
                }
                dispatch_fused_ripa_and_pspm_words_core(
                    q,
                    swar_words,
                    ripa_max_rad,
                    pspm_cfg,
                    fact_words,
                    logic_words,
                    raw_words,
                    ripa_max_rad,
                    pulse_q7,
                    head_dim,
                    true,
                )?;
                if swar_words_nonzero_64(swar_words) {
                    steer_from_swar_words_core_safe(
                        swar_words,
                        0xFF,
                        128,
                        head_dim,
                        num_heads,
                        false,
                        ripa_max_rad,
                        k,
                        None,
                    );
                }
                return Ok((true, true));
            }
            (Some(q), None) => {
                dispatch_fused_ripa_and_pspm_words_core(
                    q,
                    swar_words,
                    ripa_max_rad,
                    pspm_cfg,
                    fact_words,
                    logic_words,
                    raw_words,
                    ripa_max_rad,
                    pulse_q7,
                    head_dim,
                    true,
                )?;
            }
            (None, Some(k)) => {
                if swar_words_nonzero_64(swar_words) {
                    steer_from_swar_words_core_safe(
                        swar_words,
                        0xFF,
                        128,
                        head_dim,
                        num_heads,
                        false,
                        ripa_max_rad,
                        k,
                        None,
                    );
                }
            }
            (None, None) => {}
        }
        Ok((true, has_q))
    }

    /// Register-direct fused QDPS + $SO(2)$ Group-Closure RIPA + PSPM Sub-Head Routing over
    /// pre-loaded 8-word SWAR registers (`raw_words`, `fact_words`, `logic_words`) without
    /// intermediate stack array copies or memory reloads.
    ///
    /// Returns `(steered, pspm_routed)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if dimensions, thresholds, or tensor slices are invalid.
    #[inline(always)]
    fn filter_qdps_and_steer_qk_with_pspm_words_safe(
        &self,
        raw_words: &[u64; NUM_SWARM_MICRO_CELLS],
        fact_words: &[u64; NUM_SWARM_MICRO_CELLS],
        logic_words: &[u64; NUM_SWARM_MICRO_CELLS],
        pspm_config_and_pulse: Option<(&gcso_pspm_config_t, gcso_q7_t)>,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(bool, bool)> {
        if !(query_tensor.is_some() | key_tensor.is_some()) {
            return Ok((false, false));
        }
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        self.steer_qdps_pspm_words_prevalidated_core(
            raw_words,
            fact_words,
            logic_words,
            pspm_config_and_pulse,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        )
    }

    /// Filters `raw_phases` via 8-lane SWAR QDPS directly in 64-bit registers (`swar_words`) and
    /// executes **Single-Pass $SO(2)$ Group-Closure RIPA + PSPM Sub-Head Routing** on `query_tensor`
    /// when `pspm_config_and_pulse` is `Some` via [`Self::filter_qdps_and_steer_qk_with_pspm_words_safe`]
    /// (`0` intermediate stack buffer copies or reloads).
    ///
    /// Returns `(steered, pspm_routed)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if dimensions, thresholds, or tensor slices are invalid.
    #[inline]
    fn filter_qdps_and_steer_qk_with_unified_pspm_safe(
        &self,
        raw_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        dominant_anchor_id: u32,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        pspm_config_and_pulse: Option<(&gcso_pspm_config_t, gcso_q7_t)>,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(bool, bool)> {
        if pspm_config_and_pulse.is_none() {
            let steered = self.filter_qdps_and_steer_qk_safe(
                raw_phases,
                qdps_min_step_rad,
                ripa_max_rad,
                query_tensor,
                key_tensor,
            )?;
            return Ok((steered, false));
        }
        let raw_words = load_q7_words_64(raw_phases);
        let dom_idx = dominant_anchor_id as usize;
        let n_anc = anchor_phases.len();
        let f_idx = if n_anc > 0 { dom_idx.min(n_anc - 1) } else { 0 };
        let fact_w = if n_anc > 0 {
            if core::ptr::eq(&anchor_phases[f_idx], raw_phases) {
                raw_words
            } else {
                load_q7_words_64(&anchor_phases[f_idx])
            }
        } else {
            [0u64; NUM_SWARM_MICRO_CELLS]
        };
        let logic_w = if n_anc > 1 {
            let l_idx = (dom_idx + 1) % n_anc;
            if l_idx == f_idx {
                fact_w
            } else if core::ptr::eq(&anchor_phases[l_idx], raw_phases) {
                raw_words
            } else {
                load_q7_words_64(&anchor_phases[l_idx])
            }
        } else {
            raw_words
        };
        self.filter_qdps_and_steer_qk_with_pspm_words_safe(
            &raw_words,
            &fact_w,
            &logic_w,
            pspm_config_and_pulse,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        )
    }

    /// Executes **Coarse-to-Fine Macro-to-Micro Routing & Steering ($\mathcal{O}(K)$)** over a
    /// slice of Online Toroidal $k$-Means / DP-Means macro centroids (`centroids: &[CrystallizedAttractorRecord]`):
    /// 1. Validates optional `query_tensor` / `key_tensor` parameters upfront for strong error atomicity.
    /// 2. Routes `trail` toward the nearest high-gravity macro centroid $k^*$ on $SO(2)^{64}$ via
    ///    [`CrystallizedAttractorRecord::route_coarse_to_fine_trail_64`] (binding `trail.cluster_id`
    ///    and `trail.target_anchor_id`).
    /// 3. Steers `query_tensor` and/or `key_tensor` using register-resident 8-lane SWAR QDPS + RIPA
    ///    soft-clamping (`ripa_max_rad`).
    ///
    /// Returns `(selected_cluster_idx, selected_anchor_id, centroid_coherence, gravity_pull, steered)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if dimensions, thresholds, or tensor slices are invalid.
    #[inline]
    fn route_coarse_to_fine_and_steer_qk_safe(
        &self,
        centroids: &[CrystallizedAttractorRecord],
        pull_step_u7: u8,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        trail: &mut gcso_pointer_trail_t,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(usize, u32, f32, f32, bool)> {
        let want_steer = query_tensor.is_some() | key_tensor.is_some();
        let (head_dim, num_heads) = if want_steer {
            let (hd, nh, _) = self.validate_hub_cascade_params(
                qdps_min_step_rad,
                ripa_max_rad,
                query_tensor.as_deref(),
                key_tensor.as_deref(),
            )?;
            (hd, nh)
        } else {
            (0, 0)
        };
        let (cluster_idx, anchor_id, coh, grav, bridged_w, _win_w) =
            CrystallizedAttractorRecord::route_coarse_to_fine_trail_with_words_64(
                centroids,
                trail,
                pull_step_u7,
            );
        let steered = if want_steer {
            self.steer_qdps_words_prevalidated_core(
                &bridged_w,
                head_dim,
                num_heads,
                qdps_min_step_rad,
                ripa_max_rad,
                query_tensor,
                key_tensor,
            )
        } else {
            false
        };
        Ok((cluster_idx, anchor_id, coh, grav, steered))
    }

    /// Executes **Top-2 Liminal Bifurcation Macro Routing & $SO(2)$-Fused RIPA/PSPM Steering ($\mathcal{O}(K)$)**
    /// over a slice of Online Toroidal $k$-Means / DP-Means macro centroids (`centroids: &[CrystallizedAttractorRecord]`):
    /// 1. Validates optional `query_tensor` / `key_tensor` parameters upfront for strong error atomicity.
    /// 2. Identifies the Top-2 macro centroids $(k_1^*, k_2^*)$ via
    ///    [`CrystallizedAttractorRecord::route_top2_liminal_centroids_64`]; when $G_2 \ge \tau_{\mathrm{limina}} G_1$
    ///    (e.g. [`DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO`]), bridges the inter-memory *Limina* boundary
    ///    along the shortest $S^1$ geodesic between both centroids and stamps
    ///    `TRAIL_FLAG_LIMINAL_BEAT_ACTIVE | TRAIL_FLAG_SYNERGY_RESONANCE`.
    /// 3. Steers `query_tensor` and/or `key_tensor` via [`Self::filter_qdps_and_steer_qk_with_pspm_words_safe`],
    ///    binding `Fact` sub-heads to the primary centroid $k_1^*$ and `Logic` sub-heads to the
    ///    secondary centroid $k_2^*$ directly in 8 `u64` SWAR registers (`0` stack array copies).
    ///
    /// Returns `(primary_idx, secondary_idx, primary_anchor_id, secondary_anchor_id, blended_coherence, total_gravity, is_liminal_bifurcation, steered)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if dimensions, thresholds, or tensor slices are invalid.
    #[inline]
    fn route_top2_liminal_and_steer_qk_safe(
        &self,
        centroids: &[CrystallizedAttractorRecord],
        pull_step_u7: u8,
        liminal_gravity_ratio_thresh: f32,
        pspm_config_and_pulse: Option<(&gcso_pspm_config_t, gcso_q7_t)>,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        trail: &mut gcso_pointer_trail_t,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(usize, usize, u32, u32, f32, f32, bool, bool)> {
        let want_steer = query_tensor.is_some() | key_tensor.is_some();
        let (head_dim, num_heads) = if want_steer {
            let (hd, nh, _) = self.validate_hub_cascade_params(
                qdps_min_step_rad,
                ripa_max_rad,
                query_tensor.as_deref(),
                key_tensor.as_deref(),
            )?;
            (hd, nh)
        } else {
            (0, 0)
        };
        let (idx1, idx2, aid1, aid2, eff_coh, tot_grav, is_liminal, raw_w, fact_w, logic_w) =
            CrystallizedAttractorRecord::route_top2_liminal_centroids_with_words_64(
                centroids,
                trail,
                pull_step_u7,
                liminal_gravity_ratio_thresh,
            );
        let steered = if want_steer {
            self.steer_qdps_pspm_words_prevalidated_core(
                &raw_w,
                &fact_w,
                &logic_w,
                pspm_config_and_pulse,
                head_dim,
                num_heads,
                qdps_min_step_rad,
                ripa_max_rad,
                query_tensor,
                key_tensor,
            )?
            .0
        } else {
            false
        };
        Ok((
            idx1, idx2, aid1, aid2, eff_coh, tot_grav, is_liminal, steered,
        ))
    }

    /// Executes **Full Coarse-to-Fine Macro-to-Micro Routing ($\mathcal{O}(K)$ Macro Centroids $\to$
    /// $\mathcal{O}(M)$ Local Pointer Trails) + Continuous Geodesic Affinity Prior + Bayesian Pointer
    /// Rewiring + $SO(2)$-Fused RIPA/PSPM Steering + DAES Telemetry** directly on `ActionHub`:
    /// 1. **Coarse Step ($\mathcal{O}(K)$)**: Routes `src_trail` to the highest-gravity coherent
    ///    macro centroid $k^*$ in `centroids` via
    ///    [`CrystallizedAttractorRecord::route_coarse_to_fine_trail_64`] and pre-aligns
    ///    `src_trail.linked_trail_id` with the best candidate trail in `candidate_trails` via
    ///    `select_rewire_bridge_and_modulate_hub_with_macro_centroid_64` (using both hard `cluster_id == macro_anchor_id`
    ///    matching AND hoisted uniform-word geodesic affinity $\mathcal{C}(\boldsymbol{\theta}_i, \boldsymbol{\mu}_{k^*}) > 0.50$
    ///    for unstamped candidate trails).
    /// 2. **Fine Step ($\mathcal{O}(M)$)**: Selects and rewires the optimal coherent local pointer
    ///    trail $j^*$ in `candidate_trails` via
    ///    [`Self::select_rewire_bridge_and_modulate_hub_bayesian_trails`], calling
    ///    `self.link_hallucinated_trails(src_trail_id, j^*)` and modulating `attention_scores`.
    /// 3. **Execution & Telemetry Step**: Steers `query_tensor` and/or `key_tensor` via
    ///    [`Self::filter_qdps_and_steer_qk_with_pspm_words_safe`] and records DAES telemetry in `daes_slot`.
    ///
    /// Returns `(macro_cluster_idx, macro_anchor_id, selected_trail_id, bayesian_score, found_coherent, steered)`.
    ///
    /// # Errors
    /// Propagates any validation or execution error from `ActionHub`.
    #[inline]
    fn route_coarse_to_fine_bayesian_and_steer_qk_safe(
        &mut self,
        src_trail_id: u32,
        centroids: &[CrystallizedAttractorRecord],
        coarse_pull_u7: u8,
        src_trail: &mut gcso_pointer_trail_t,
        candidate_trails: &mut [gcso_pointer_trail_t],
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        repulsion_penalty: f32,
        fine_interpolation_u7: u8,
        contradiction_threshold: f32,
        pspm_config_and_pulse: Option<(&gcso_pspm_config_t, gcso_q7_t)>,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        daes_slot: &mut gcso_daes_slot_t,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(usize, u32, u32, f32, bool, bool)> {
        let want_steer = query_tensor.is_some() | key_tensor.is_some();
        let (head_dim, num_heads) = if want_steer {
            let (hd, nh, _) = self.validate_hub_cascade_params(
                qdps_min_step_rad,
                ripa_max_rad,
                query_tensor.as_deref(),
                key_tensor.as_deref(),
            )?;
            (hd, nh)
        } else {
            (0, 0)
        };

        let (macro_idx, macro_anchor_id, _macro_coh, _macro_grav, bridged_w, dom_w) =
            CrystallizedAttractorRecord::route_coarse_to_fine_trail_with_words_64(
                centroids,
                src_trail,
                coarse_pull_u7,
            );
        let has_win_centroid = macro_idx < centroids.len() && centroids[macro_idx].sample_count > 0;

        let (selected_id, bayes_score, _mean_coh, _coh_cnt, found, raw_w) = self
            .select_rewire_bridge_and_modulate_hub_with_preloaded_words_64(
                src_trail_id,
                src_trail,
                bridged_w,
                candidate_trails,
                attention_scores,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                repulsion_penalty,
                fine_interpolation_u7,
                contradiction_threshold,
                macro_anchor_id,
                if has_win_centroid { Some(&dom_w) } else { None },
                pointer_link_bonus,
            )?;
        let fact_w = if has_win_centroid { &dom_w } else { &raw_w };
        let (steered, pspm_routed) = if want_steer {
            self.steer_qdps_pspm_words_prevalidated_core(
                &raw_w,
                fact_w,
                &raw_w,
                pspm_config_and_pulse,
                head_dim,
                num_heads,
                qdps_min_step_rad,
                ripa_max_rad,
                query_tensor,
                key_tensor,
            )?
        } else {
            (false, false)
        };

        let metrics = branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(pspm_routed, DAES_METRIC_PSPM_ROUTED, 0)
            | branchless_select_u8(found, DAES_METRIC_SLOT_SYNERGY_ACTIVE, 0)
            | branchless_select_u8(!found, DAES_METRIC_REPULSION_APPLIED, 0);
        let _ = self.telemetry_push(daes_slot, metrics);

        Ok((
            macro_idx,
            macro_anchor_id,
            selected_id,
            bayes_score,
            found,
            steered,
        ))
    }

    /// Executes **In-Place 4-Section `.gcso` Zero-Copy Online Toroidal $k$-Means / DP-Means
    /// Evolution + Single-CRC32 Register-Direct QDPS/RIPA Q/K Steering** on `ActionHub`:
    /// 1. Validates optional `query_tensor` / `key_tensor` parameters upfront for strong error atomicity
    ///    (ensuring `zimms_bytes` is never mutated if tensor layout or `GCSO_RESTRICT` invariants fail).
    /// 2. Synthesizes `signal` and evolves Section 1 (`crystals`), Section 2 (`trails[trail_idx]`),
    ///    Section 3a (`bloom`), Section 3b (`daes`), Section 3c (`provenance_reports`), and the
    ///    Section 0 IEEE 802.3 CRC32 checksum in-place via
    ///    [`GcsoFourSectionView::evolve_four_section_kmeans_in_place`].
    /// 3. Extracts the updated trail's 8 SWAR phase words directly via
    ///    `load_validated_four_section_trail_words_64` (without re-running a second full-container
    ///    CRC32 verification pass) and steers `query_tensor` and/or `key_tensor` using
    ///    `frame.qdps_min_step_rad` and `frame.ripa_clamp_rad` unless `ACTION_MODE_LAYER_SKIP` is active.
    ///
    /// Returns `(GatewayFrame, assigned_cluster_idx, assigned_anchor_id, updated_gravity_mass, spawned_new_cluster, merged_count, steered)`.
    ///
    /// # Errors
    /// Propagates any container validation or tensor steering error.
    #[inline]
    fn evolve_four_section_kmeans_and_steer_qk_safe<S: ManifoldGatewaySignal + ?Sized>(
        &self,
        zimms_bytes: &mut [u8],
        signal: &S,
        split_coherence_thresh: f32,
        merge_coherence_thresh: f32,
        trail_idx: usize,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(GatewayFrame, usize, u32, f32, bool, u32, bool)> {
        let want_steer = query_tensor.is_some() | key_tensor.is_some();
        let (head_dim, num_heads) = if want_steer {
            let (hd, nh, expected_len) = self.validate_head_layout()?;
            if nh > MAX_ATTENTION_HEADS {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            validate_optional_qk_tensors(
                query_tensor.as_deref(),
                key_tensor.as_deref(),
                expected_len,
            )?;
            (hd, nh)
        } else {
            (0, 0)
        };

        let (frame, c_idx, c_id, mass, spawned, merged) =
            GcsoFourSectionView::evolve_four_section_kmeans_in_place(
                zimms_bytes,
                signal,
                split_coherence_thresh,
                merge_coherence_thresh,
                trail_idx,
            )?;
        if !want_steer || (frame.action_mode_flags & ACTION_MODE_LAYER_SKIP) != 0 {
            return Ok((frame, c_idx, c_id, mass, spawned, merged, false));
        }
        if frame.qdps_min_step_rad < 0.0
            || !branchless_is_finite_f32(frame.qdps_min_step_rad)
            || frame.ripa_clamp_rad <= 0.0
            || !branchless_is_finite_f32(frame.ripa_clamp_rad)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let steer_w =
            load_validated_four_section_trail_words_64(zimms_bytes, trail_idx, &frame.phase_deltas);
        let steered = self.steer_qdps_words_prevalidated_core(
            &steer_w,
            head_dim,
            num_heads,
            frame.qdps_min_step_rad,
            frame.ripa_clamp_rad,
            query_tensor,
            key_tensor,
        );
        Ok((frame, c_idx, c_id, mass, spawned, merged, steered))
    }

    /// Executes **In-Place 4-Section `.gcso` Zero-Copy Coarse-to-Fine Routing + Single-CRC32
    /// Register-Direct QDPS/RIPA Q/K Steering** on `ActionHub`:
    /// 1. Validates optional `query_tensor` / `key_tensor` parameters upfront for strong error atomicity.
    /// 2. Routes `trail_idx` in Section 2 of `zimms_bytes` against the macro centroids in Section 1
    ///    and restamps the IEEE 802.3 CRC32 checksum once via
    ///    [`GcsoFourSectionView::route_four_section_coarse_to_fine_in_place`].
    /// 3. Uses the bridged 8 SWAR phase words returned directly in registers to steer
    ///    `query_tensor` and/or `key_tensor` (`0` post-routing container reloads).
    ///
    /// Returns `(selected_cluster_idx, selected_anchor_id, centroid_coherence, gravity_pull, steered)`.
    ///
    /// # Errors
    /// Propagates any container validation or tensor steering error.
    #[inline]
    fn route_four_section_coarse_to_fine_and_steer_qk_safe(
        &self,
        zimms_bytes: &mut [u8],
        trail_idx: usize,
        pull_step_u7: u8,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(usize, u32, f32, f32, bool)> {
        let want_steer = query_tensor.is_some() | key_tensor.is_some();
        let (head_dim, num_heads) = if want_steer {
            let (hd, nh, _) = self.validate_hub_cascade_params(
                qdps_min_step_rad,
                ripa_max_rad,
                query_tensor.as_deref(),
                key_tensor.as_deref(),
            )?;
            (hd, nh)
        } else {
            (0, 0)
        };
        let (cluster_idx, anchor_id, coh, grav, steer_w) =
            GcsoFourSectionView::route_four_section_coarse_to_fine_with_words_in_place(
                zimms_bytes,
                trail_idx,
                pull_step_u7,
            )?;
        if !want_steer {
            return Ok((cluster_idx, anchor_id, coh, grav, false));
        }
        let steered = self.steer_qdps_words_prevalidated_core(
            &steer_w,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        );
        Ok((cluster_idx, anchor_id, coh, grav, steered))
    }

    /// Executes **In-Place 4-Section `.gcso` Zero-Copy Top-2 Liminal Bifurcation Routing +
    /// Dual-Centroid ($k_1^*, k_2^*$) PSPM Sub-Head Steering + Section-3b DAES Telemetry +
    /// Single-Pass IEEE 802.3 CRC32 Refresh** on `ActionHub` using 100% safe disjoint slice
    /// splitting (`zimms_bytes.split_at_mut(off_trails)` + [`Pod::slice_from_bytes_at`] /
    /// [`Pod::slice_from_bytes_mut_at`]):
    /// 1. Validates optional `query_tensor` / `key_tensor` parameters upfront for strong error atomicity.
    /// 2. Validates the 128B-aligned 4-section `.gcso` container layout (`zimms_bytes`) and executes
    ///    [`CrystallizedAttractorRecord::route_top2_liminal_centroids_64`] in-place on Section 1
    ///    (`crystals`) and Section 2 (`trails[trail_idx]`), bridging inter-memory *Limina* boundaries
    ///    along the shortest $S^1$ geodesic when $G_2 \ge \tau_{\mathrm{limina}} G_1$.
    /// 3. Uses the bridged trail phase words (`raw_w`), primary centroid words (`fact_w`, $k_1^*$),
    ///    and secondary centroid words (`logic_w`, $k_2^*$) returned directly in 8-word SWAR registers
    ///    to steer `query_tensor` and/or `key_tensor` via
    ///    [`Self::filter_qdps_and_steer_qk_with_pspm_words_safe`].
    /// 4. Updates Section 3b (`gcso_daes_slot_t`) telemetry in-place if present and restamps the
    ///    Section 0 IEEE 802.3 CRC32 checksum in a single pass.
    ///
    /// Returns `(primary_idx, secondary_idx, primary_anchor_id, secondary_anchor_id, blended_coherence, total_gravity, is_liminal_bifurcation, steered)`.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_BUFFER_TOO_SMALL`], [`crate::abi::GCSO_ERROR_MISALIGNED_POINTER`],
    /// [`GCSO_ERROR_INVALID_ARGUMENT`], or [`crate::abi::GCSO_ERROR_CONTAINER_CORRUPTED`] on validation failure.
    #[inline]
    fn route_four_section_top2_liminal_and_steer_qk_safe(
        &self,
        zimms_bytes: &mut [u8],
        trail_idx: usize,
        pull_step_u7: u8,
        liminal_gravity_ratio_thresh: f32,
        pspm_config_and_pulse: Option<(&gcso_pspm_config_t, gcso_q7_t)>,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(usize, usize, u32, u32, f32, f32, bool, bool)> {
        let want_steer = query_tensor.is_some() | key_tensor.is_some();
        let (head_dim, num_heads) = if want_steer {
            let (hd, nh, _) = self.validate_hub_cascade_params(
                qdps_min_step_rad,
                ripa_max_rad,
                query_tensor.as_deref(),
                key_tensor.as_deref(),
            )?;
            (hd, nh)
        } else {
            (0, 0)
        };
        let (
            idx1,
            idx2,
            aid1,
            aid2,
            eff_coh,
            tot_grav,
            is_liminal,
            raw_w,
            fact_w,
            logic_w,
            mut header,
            total_len,
            off_daes,
            updated_ptr,
        ) = GcsoFourSectionView::route_four_section_top2_liminal_core_in_place(
            zimms_bytes,
            trail_idx,
            pull_step_u7,
            liminal_gravity_ratio_thresh,
        )?;

        let (steered, pspm_routed) = if want_steer {
            self.steer_qdps_pspm_words_prevalidated_core(
                &raw_w,
                &fact_w,
                &logic_w,
                pspm_config_and_pulse,
                head_dim,
                num_heads,
                qdps_min_step_rad,
                ripa_max_rad,
                query_tensor,
                key_tensor,
            )?
        } else {
            (false, false)
        };

        let metric_code = branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(pspm_routed, DAES_METRIC_PSPM_ROUTED, 0)
            | branchless_select_u8((aid1 != 0) | is_liminal, DAES_METRIC_SLOT_SYNERGY_ACTIVE, 0);
        let _ = GcsoFourSectionView::update_four_section_daes_in_place(
            zimms_bytes,
            &header,
            total_len,
            off_daes,
            metric_code,
            updated_ptr,
        )?;
        GcsoFourSectionView::finalize_four_section_header_in_place(
            zimms_bytes,
            &mut header,
            total_len,
        )?;

        Ok((
            idx1, idx2, aid1, aid2, eff_coh, tot_grav, is_liminal, steered,
        ))
    }

    /// Folds a zero-copy slice of `.gcso` [`CrystallizedAttractorRecord`]s (`records`) onto
    /// `trail.accumulated_phase_delta` in 8 register-resident `u64` SWAR words weighted by
    /// **Phase-Proximity Gravity**, steers `query_tensor` and/or `key_tensor` directly from the
    /// folded SWAR words via [`Self::filter_qdps_words_and_steer_qk_safe`], and pushes DAES
    /// telemetry to `daes_slot`.
    ///
    /// Returns `(total_gravity_pull, dominant_anchor_id, steered)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if dimensions, thresholds, or tensor slices are invalid.
    #[inline]
    fn steer_crystallized_ensemble_on_hub_safe(
        &mut self,
        records: &[CrystallizedAttractorRecord],
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        daes_slot: &mut gcso_daes_slot_t,
        trail: &mut gcso_pointer_trail_t,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(f32, u32, bool)> {
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;

        let init_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(trail.attractor_pull_force));
        let (folded_w, tot_pull, dom_id, _dom_clamp) =
            CrystallizedAttractorRecord::fold_crystallized_ensemble_words_with_init_pull_64(
                records,
                &mut trail.accumulated_phase_delta,
                init_pull,
            );
        if dom_id != 0 {
            trail.target_anchor_id = dom_id;
            trail.cluster_id = dom_id;
        }
        trail.attractor_pull_force = branchless_min_f32(tot_pull, MAX_ATTRACTOR_PULL_FORCE);
        trail.flags |= TRAIL_FLAG_CRYSTALLIZED_CANDIDATE | TRAIL_FLAG_SYNERGY_RESONANCE;

        let steered = self.steer_qdps_words_prevalidated_core(
            &folded_w,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        );

        let metrics = branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(dom_id != 0, DAES_METRIC_SLOT_SYNERGY_ACTIVE, 0);
        let _ = self.telemetry_push(daes_slot, metrics);
        Ok((tot_pull, dom_id, steered))
    }

    /// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) on the
    /// Single Unified Attractor Manifold into `trail.accumulated_phase_delta` starting from
    /// `trail.attractor_pull_force` inertia weight, applies 8-lane SWAR QDPS step filtering, and
    /// steers `query_tensor` and/or `key_tensor` with RIPA soft-clamping (`ripa_max_rad`) via
    /// [`PhaseSteering::apply_unified_field_phase_steering_with_inertia_safe`].
    ///
    /// Returns `(steered, total_resonant_pull, mean_coherence, dominant_anchor_idx)`.
    ///
    /// # Errors
    /// Propagates any error from [`PhaseSteering::apply_unified_field_phase_steering_with_inertia_safe`].
    #[inline(always)]
    fn filter_qdps_and_steer_qk_unified_field_safe(
        &self,
        trail: &gcso_pointer_trail_t,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(bool, f32, f32, u32)> {
        match (query_tensor, key_tensor) {
            (Some(q), k_opt) => {
                let (pull, coh, dom_idx) = self
                    .apply_unified_field_phase_steering_with_inertia_safe(
                        &trail.accumulated_phase_delta,
                        trail.attractor_pull_force,
                        anchor_phases,
                        anchor_base_energies,
                        qdps_min_step_rad,
                        ripa_max_rad,
                        q,
                        k_opt,
                    )?;
                Ok((true, pull, coh, dom_idx))
            }
            (None, Some(k)) => {
                let (pull, coh, dom_idx) = self
                    .apply_unified_field_phase_steering_with_inertia_safe(
                        &trail.accumulated_phase_delta,
                        trail.attractor_pull_force,
                        anchor_phases,
                        anchor_base_energies,
                        qdps_min_step_rad,
                        ripa_max_rad,
                        k,
                        None,
                    )?;
                Ok((true, pull, coh, dom_idx))
            }
            (None, None) => Ok((false, 0.0, 0.0, trail.target_anchor_id)),
        }
    }

    /// Modulates a candidate memory bank's pre-Softmax attention scores (`attention_scores`)
    /// directly by the **Attention $\times$ DPSR Phase Interference**
    /// $\Delta z_{t,j} = \frac{\gamma}{\sqrt{d_{\mathrm{head}}}} \mathcal{C}_{\mathrm{phase}}(\boldsymbol{\theta}_t + \boldsymbol{\Delta\theta}, \boldsymbol{\theta}_j)$
    /// across `anchor_phases` on the Single Unified Attractor Manifold.
    ///
    /// Returns the mean phase coherence across the modulated candidate memories.
    ///
    /// # Errors
    /// Propagates any validation error from [`PhaseSteering::modulate_attention_scores_by_phase_interference`].
    #[inline(always)]
    fn modulate_hub_attention_scores_by_unified_field(
        &self,
        attention_scores: &mut AlignedSliceMut32<'_, f32>,
        trail: &gcso_pointer_trail_t,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        interference_gain: f32,
    ) -> GcsoResult<f32> {
        self.modulate_attention_scores_by_phase_interference(
            attention_scores,
            &trail.accumulated_phase_delta,
            anchor_phases,
            interference_gain,
        )
    }

    /// Modulates a candidate memory bank's pre-Softmax attention scores (`attention_scores`)
    /// directly from `trail`'s accumulated phase delta and `linked_trail_id` pointer link
    /// (unifying Hypothesis 1 Joint Probability $\times$ Attention and Hypothesis 2 `ActionHub`
    /// pointer-trail manipulation).
    ///
    /// Automatically maps `trail.linked_trail_id != 0` to `Some(trail.linked_trail_id)`.
    ///
    /// Returns `(mean_phase_coherence, best_post_modulated_candidate_idx)`.
    ///
    /// # Errors
    /// Propagates any validation error from [`PhaseSteering::modulate_attention_scores_by_bayesian_trails`].
    #[inline(always)]
    fn modulate_hub_attention_scores_by_bayesian_trails(
        &self,
        attention_scores: &mut AlignedSliceMut32<'_, f32>,
        trail: &gcso_pointer_trail_t,
        candidate_key_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        candidate_stigmergic_densities: &[f32],
        interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
    ) -> GcsoResult<(f32, u32)> {
        let linked_trail_id = if trail.linked_trail_id != 0 {
            Some(trail.linked_trail_id)
        } else {
            None
        };
        self.modulate_attention_scores_by_bayesian_trails(
            attention_scores,
            &trail.accumulated_phase_delta,
            candidate_key_phases,
            candidate_stigmergic_densities,
            linked_trail_id,
            interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
        )
    }

    /// Executes **Single-Pass Bayesian Pointer-Trail Selection, Rewiring, Limina Bridging, and
    /// Pre-Softmax Attention Score Modulation** directly on `ActionHub` (uniting Hypothesis 1
    /// and Hypothesis 2 in a single scan over `candidate_trails`).
    ///
    /// Returns `(selected_trail_id, bayesian_score, mean_coherence, coherent_candidate_count, found_coherent_transition)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `candidate_trails` is empty, `head_dim()` is invalid,
    /// or any gain/threshold parameter is non-finite, or propagates errors from [`PointerActionHub::link_hallucinated_trails`].
    #[inline(always)]
    fn select_rewire_bridge_and_modulate_hub_bayesian_trails(
        &mut self,
        src_trail_id: u32,
        src_trail: &mut gcso_pointer_trail_t,
        candidate_trails: &mut [gcso_pointer_trail_t],
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        repulsion_penalty: f32,
        interpolation_u7: u8,
        contradiction_threshold: f32,
    ) -> GcsoResult<(u32, f32, f32, u32, bool)> {
        self.select_rewire_bridge_and_modulate_hub_with_macro_centroid_64(
            src_trail_id,
            src_trail,
            candidate_trails,
            attention_scores,
            phase_interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
            repulsion_penalty,
            interpolation_u7,
            contradiction_threshold,
            0,
            None,
            0.0,
        )
    }

    /// Executes **Single-Pass Coarse-to-Fine Macro-Centroid Affinity + Bayesian Pointer-Trail
    /// Selection, Rewiring, Limina Bridging, and Pre-Softmax Attention Score Modulation**
    /// directly on `ActionHub` in a single scan over `candidate_trails` (`0` duplicate candidate scans).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `candidate_trails` is empty, `head_dim()` is invalid,
    /// or any gain/threshold parameter is non-finite, or propagates errors from [`PointerActionHub::link_hallucinated_trails`].
    #[inline(always)]
    fn select_rewire_bridge_and_modulate_hub_with_macro_centroid_64(
        &mut self,
        src_trail_id: u32,
        src_trail: &mut gcso_pointer_trail_t,
        candidate_trails: &mut [gcso_pointer_trail_t],
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        repulsion_penalty: f32,
        interpolation_u7: u8,
        contradiction_threshold: f32,
        macro_anchor_id: u32,
        win_w_opt: Option<&[u64; NUM_SWARM_MICRO_CELLS]>,
        cluster_bonus_gain: f32,
    ) -> GcsoResult<(u32, f32, f32, u32, bool)> {
        let init_w = *trail_phase_words_ref(src_trail);
        let (selected_id, bayes_score, mean_coh, coh_cnt, found, _post_w) = self
            .select_rewire_bridge_and_modulate_hub_with_preloaded_words_64(
                src_trail_id,
                src_trail,
                init_w,
                candidate_trails,
                attention_scores,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                repulsion_penalty,
                interpolation_u7,
                contradiction_threshold,
                macro_anchor_id,
                win_w_opt,
                cluster_bonus_gain,
            )?;
        Ok((selected_id, bayes_score, mean_coh, coh_cnt, found))
    }

    /// Register-resident variant of [`Self::select_rewire_bridge_and_modulate_hub_with_macro_centroid_64`]
    /// accepting `src_trail_words` directly in 8 `u64` SWAR registers and returning the post-bridged
    /// 8 `u64` SWAR words (`0` redundant 64-byte stack/memory reloads or Store-to-Load Forwarding stalls).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `candidate_trails` is empty, `head_dim()` is invalid,
    /// or any gain/threshold parameter is non-finite, or propagates errors from [`PointerActionHub::link_hallucinated_trails`].
    #[inline]
    fn select_rewire_bridge_and_modulate_hub_with_preloaded_words_64(
        &mut self,
        src_trail_id: u32,
        src_trail: &mut gcso_pointer_trail_t,
        src_trail_words: [u64; NUM_SWARM_MICRO_CELLS],
        candidate_trails: &mut [gcso_pointer_trail_t],
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        repulsion_penalty: f32,
        interpolation_u7: u8,
        contradiction_threshold: f32,
        macro_anchor_id: u32,
        win_w_opt: Option<&[u64; NUM_SWARM_MICRO_CELLS]>,
        cluster_bonus_gain: f32,
    ) -> GcsoResult<(u32, f32, f32, u32, bool, [u64; NUM_SWARM_MICRO_CELLS])> {
        let (head_dim, _num_heads, _expected_len) = self.validate_head_layout()?;
        if candidate_trails.is_empty()
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(repulsion_penalty)
            || !branchless_is_finite_f32(contradiction_threshold)
            || !branchless_is_finite_f32(cluster_bonus_gain)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let scores_slice_opt = attention_scores
            .filter(|s| !s.is_empty())
            .map(|s| s.as_mut_slice());

        let (selected_id, bayes_score, mean_coh, coh_cnt, found, post_w, _best_coh) =
            select_rewire_bridge_and_modulate_with_preloaded_words_64(
                src_trail,
                src_trail_words,
                candidate_trails,
                scores_slice_opt,
                head_dim as u32,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                contradiction_threshold,
                interpolation_u7,
                repulsion_penalty,
                macro_anchor_id,
                win_w_opt,
                cluster_bonus_gain,
            );

        if found {
            self.link_hallucinated_trails(src_trail_id, selected_id)?;
            let dst_idx = (selected_id as usize).min(candidate_trails.len() - 1);
            let dst_trail = &mut candidate_trails[dst_idx];
            *trail_phase_words_mut(dst_trail) = post_w;
            let dst_has_anchor = dst_trail.target_anchor_id != 0;
            let inherited_anchor = branchless_select_u32(
                dst_has_anchor,
                dst_trail.target_anchor_id,
                src_trail.target_anchor_id,
            );
            src_trail.target_anchor_id = inherited_anchor;
            dst_trail.target_anchor_id = inherited_anchor;
            if inherited_anchor != 0 {
                src_trail.cluster_id = inherited_anchor;
                dst_trail.cluster_id = inherited_anchor;
            }
            src_trail.flags |= TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE;
            dst_trail.flags |= TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE;
            dst_trail.sync_flags_with_current_ptr();
        }
        src_trail.sync_flags_with_current_ptr();
        Ok((selected_id, bayes_score, mean_coh, coh_cnt, found, post_w))
    }

    /// Modulates a vocabulary or branch `logits` slice by **Phase-Interference Inverse-Temperature
    /// Scaling** $\beta_{\mathrm{eff}} = \mathrm{clamp}(1.0 + \gamma \mathcal{C}_{\mathrm{phase}}, 0.25, 4.0)$
    /// (`logits[i] *= beta_eff`) between `trail` and `memory_anchor_phases` in zero-copy mode.
    ///
    /// Returns `(phase_coherence, applied_inv_temp_scale)`.
    ///
    /// # Errors
    /// Propagates any validation error from [`PhaseSteering::apply_phase_interference_inv_temperature_scaling_64`].
    #[inline(always)]
    fn modulate_hub_logit_inv_temperature(
        &self,
        logits: &mut AlignedSliceMut32<'_, f32>,
        trail: &gcso_pointer_trail_t,
        memory_anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        interference_gain: f32,
    ) -> GcsoResult<(f32, f32)> {
        self.apply_phase_interference_inv_temperature_scaling_64(
            logits,
            &trail.accumulated_phase_delta,
            memory_anchor_phases,
            interference_gain,
        )
    }

    /// Executes optional single-pass PSPM sub-head routing (`pspm_config`) and SRL Rank-1 outer
    /// product residual injection (`srl_activation_and_desc`) on `query_tensor`, returning the
    /// accumulated DAES telemetry metric bitmask (`DAES_METRIC_PSPM_ROUTED | DAES_METRIC_SRL_INJECTED`).
    ///
    /// # Errors
    /// Propagates any error from [`PspmRouter::dispatch_single_pass`] or [`SrlAdapter::eval_rank1_fused`].
    #[inline(always)]
    fn apply_optional_pspm_and_srl(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        head_dim: usize,
        pspm_config: Option<&gcso_pspm_config_t>,
        srl_activation_and_desc: Option<(&AlignedSlice32<'_, f32>, &gcso_srl_descriptor_t)>,
    ) -> GcsoResult<u8> {
        let mut extra_metrics = 0u8;
        if let Some(cfg) = pspm_config {
            self.dispatch_single_pass(query_tensor, cfg, head_dim)?;
            extra_metrics |= DAES_METRIC_PSPM_ROUTED;
        }
        if let Some((x_in, desc)) = srl_activation_and_desc {
            self.eval_rank1_fused(query_tensor, x_in, desc)?;
            extra_metrics |= DAES_METRIC_SRL_INJECTED;
        }
        Ok(extra_metrics)
    }
}

impl<T> ActionHub for T where
    T: BranchlessKernel
        + PhaseSteering
        + PointerActionHub
        + PspmRouter
        + SrlAdapter
        + DaesScratchpad
        + ?Sized
{
}

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
    use super::prelude::*;
    use super::*;
    use crate::abi::{gcso_anchor_type_t, gcso_config_t, gcso_edbc_state_t, gcso_paged_bitmask_t};

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
