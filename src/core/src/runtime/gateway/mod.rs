// name: src/core/src/runtime/gateway/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Level 4 Gateway (`signals`, `frame`, `orchestrator`, `closed_loop`, and distributed
//! `bridges` implementing Gateway/Signal integration over Lv.1-Lv.3 types).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
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

#[allow(unused_imports)]
use core::mem::{align_of, offset_of, size_of};

#[allow(unused_imports)]
use crate::drivers::{
    gcso_edbc_state_t, DecodingGate, DistortionDelta, DistortionTension, EgoDifferentialNet,
    EgoIntentAxes, EntropyController, EntropyEvaluator, LayerDistortionKind, LiminalEgoField,
    LiminalEgoPreset, SystemSpineState, ENTROPY_GATE_MODE_NORMAL_ENTROPY,
    ENTROPY_GATE_MODE_PHASE_CONJUGATE_REPULSION, ENTROPY_GATE_MODE_PITCHFORK_BIFURCATION,
    MAX_DISTORTION_CHAIN_HOPS, MAX_EGO_PFRAME_NODES,
};
#[allow(unused_imports)]
use crate::drivers::{
    EGO_TRIGGER_BASIN_CAPTURE, EGO_TRIGGER_BOUNDARY_CHATTER, EGO_TRIGGER_BOUNDARY_STAGNATION,
    EGO_TRIGGER_INTERNAL_CONFLICT, EGO_TRIGGER_LAYER_DISTORTION, EGO_TRIGGER_LIMINAL_EXTERIOR,
};
#[allow(unused_imports)]
use crate::foundation::{
    atan2_interleaved_pairs_to_q7_word_x8_simd, atan2_to_q7_word_x8_simd, branchless_abs_f32,
    branchless_atan2_to_q7_fast, branchless_clamp_f32, branchless_crc32_ieee,
    branchless_eyring_kramers_escape, branchless_flush_subnormal_f32, branchless_inv_pos_f32,
    branchless_is_finite_f32, branchless_lerp_f32, branchless_maturity_scale_u32,
    branchless_max_f32, branchless_min_f32, branchless_phase_clamp_rad,
    branchless_phase_coherence_inv_temp_scale, branchless_phase_proximity_gravity_f32,
    branchless_popcount_rsqrt_u32, branchless_q7_abs_u8, branchless_q7_add_sat,
    branchless_q7_circular_diff, branchless_q7_conjugate_64, branchless_q7_inhibit_words_64,
    branchless_q7_phase_step_filter_64, branchless_q7_resonant_pull_64,
    branchless_q7_synergy_fuse_words_64,
    branchless_q7_unified_field_fold_words_with_init_weight_64, branchless_rad_to_phase_step_u8,
    branchless_relu_f32, branchless_rotate_so2_pair, branchless_rsqrt_f32, branchless_sincos_q7,
    branchless_sincos_q7_phase_clamp, branchless_splitmix64, branchless_sqrt_f32,
    circular_r_bar_from_words_64, circular_resultant_from_words_64,
    coherence_from_uniform_words_x8, coherence_from_word_arrays_64,
    coherence_from_word_arrays_with_both_uniform_flags_64, crc32_block64_zero_seed_words,
    crc32_step_block64_words, crc32_step_l16, crc32_step_l4, crc32_step_l8,
    dispatch_fused_phase_clamp_and_subhead_words_core, encode_pframe_motion_delta_64,
    eval_rank1_slices_fused, fast_inv_pos_normal_f32, fast_ln_1p_bounded_f32,
    fast_mean_q7_tag_from_words_64, gcso_subhead_config_t, gcso_tagged_ptr_t, is_aligned_to,
    is_word_array_uniform_64, load_q7_word_at, load_q7_words_64, pack_action_slot_tag,
    phase_clamp_upper_rot_dim, project_bytes_swar_to_q7_64, project_bytes_swar_to_words_64,
    project_f32_slice_to_q7_64, project_f32_slice_to_words_64, resolve_autonomous_swarm_rot_dim,
    rotate_fused_self_phase_clamp_and_subhead_default_words,
    rotate_fused_self_phase_clamp_subhead_q_and_clamp_k_default_words, scale_f32_slice_unrolled8,
    seek_pframe_chain_words_64, steer_from_swar_words_core_safe, store_q7_words_64,
    swar_or_avx2_q7_add_sat_words_64, swar_q7_abs_u8_x8, swar_q7_add_sat_x8, swar_q7_avg_round_x8,
    swar_q7_circular_add_x8, swar_q7_circular_diff_x8, swar_q7_mean_i8_from_words_64,
    swar_q7_mean_i8_from_words_with_uniform_flag_64, swar_q7_neg_sat_x8,
    swar_q7_phase_step_filter_words_64, swar_q7_phase_step_filter_words_with_uniform_flag_64,
    swar_q7_scale_le128_with_bias_x8, swar_slerp_words_with_uniform_flag_64, swar_words_equal_64,
    swar_words_nonzero_64, words_to_q7_array_64, AlignedBuffer16, AlignedSlice, AlignedSlice16,
    AlignedSlice32, AlignedSliceMut32, PhaseSteering, PhaseSteeringGeometryExt, Pod,
    RawPhaseProjectable, RopePairLayout, RopeSteeringTarget, ACTION_MODE_LAYER_SKIP,
    ACTION_MODE_PHASE_REPULSE, ACTION_MODE_PHASE_STEER, ACTION_MODE_RANK1_INJECT,
    ACTION_MODE_SUBHEAD_ROUTE, ACTION_PRESET_EXPRESS_PHASE, ACTION_PRESET_FULL_ORCHESTRATION,
    ACTION_PRESET_INERTIAL_SKIP, ACTION_PRESET_PASSIVE_SIDECAR, ACTION_PRESET_REPULSION_ESCAPE,
    ATTRACTOR_SLOT_ID_MASK, DEFAULT_PHASE_CLAMP_MAX_RAD, DEFAULT_PHASE_STEP_MIN_RAD,
    SWAR_LOW7_MASK_64, SWAR_SIGN_MASK_64, SWAR_SPLAT_BYTE_64, TAG_ACTION_MODE_MASK,
};
#[allow(unused_imports)]
use crate::liminal::{
    branchless_select_f32, branchless_select_i32, branchless_select_i8, branchless_select_q7,
    branchless_select_u32, branchless_select_u64, branchless_select_u8, branchless_select_usize,
    gcso_config_t, gcso_q7_t, EpistemicPartition, GcsoResult, Liminal, LiminalThresholdField,
    DEFAULT_ENTROPY_EPS, GCSO_ABI_VERSION_HEX, GCSO_ERROR_ACTION_HUB_FULL,
    GCSO_ERROR_INVALID_ARGUMENT, MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS,
    NUM_LIMINAL_THRESHOLD_CHANNELS, NUM_STIGMERGIC_GRANULES, SCOPE_REGION_KNOWN_UNKNOWN,
    SCOPE_REGION_SELF_EXCLUSIVE, SCOPE_REGION_SHARED_CORE, THRESHOLD_ACTION_HUB_WIDE,
    THRESHOLD_CAUSAL_PROVENANCE, THRESHOLD_COGNITIVE_CIRCUIT, THRESHOLD_COHOMOLOGY_ENTROPY,
    THRESHOLD_CONTRADICTION_GUARD, THRESHOLD_DISSIPATIVE_CRYSTAL, THRESHOLD_FAST_PATH_LEDGER,
    THRESHOLD_KV_PHASE_CACHE, THRESHOLD_MANIFOLD_GATEWAY, THRESHOLD_MAPPED_STORAGE,
    THRESHOLD_OBJECTIVE_FACT, THRESHOLD_PHASE_STEERING, THRESHOLD_POINTER_TRAIL,
    THRESHOLD_RANK1_RESIDUAL, THRESHOLD_RELATIVE_COUPLING, THRESHOLD_STIGMERGIC_CONSENSUS,
    THRESHOLD_SUBHEAD_ROUTING, THRESHOLD_SUBJECTIVE_PERSONA, THRESHOLD_SYSTEM_CONSTRAINT,
};
#[allow(unused_imports)]
use crate::runtime::boundary_delta::LiminalBoundaryDelta;
#[allow(unused_imports)]
use crate::runtime::context::{
    AutonomousCoreLayerStack, CORE_LAYER_EGO, CORE_LAYER_LIMINAL, CORE_LAYER_STIGMERGIC,
    NUM_CORE_AUTONOMOUS_LAYERS,
};
#[allow(unused_imports)]
use crate::runtime::storage::{
    GCSO_FOUR_SECTION_HEADER_BYTES, GCSO_MAPPED_STORAGE_PREAMBLE_OFFSET,
};
#[allow(unused_imports)]
use crate::stigmergy::{
    ATTRACTOR_SLOT_CRYSTALLIZED, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA,
    ATTRACTOR_SLOT_REPULSION, ATTRACTOR_SLOT_SYSTEM_BASELINE, DEFAULT_ACTION_HUB_CAPACITY,
    DEFAULT_FACT_PHASE_CLAMP_RAD, DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO,
    DEFAULT_PERSONA_PHASE_CLAMP_RAD, DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE,
    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE, GCSO_CRYSTAL_RECORD_MAGIC,
    GCSO_CRYSTAL_RECORD_VERSION, LEDGER_HISTORY_LEN,
};
pub use crate::stigmergy::{
    CAUSAL_AXIS_CONTRADICTION_GUARD, CAUSAL_AXIS_OBJECTIVE_FACT, CAUSAL_AXIS_SUBJECTIVE_PERSONA,
    CAUSAL_AXIS_SYSTEM_CONSTRAINT, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
    CAUSAL_CAUSE_COHOMOLOGY_REPULSION, CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK,
    CAUSAL_CAUSE_PERSONA_FACT_COLLISION, CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE,
    ROOT_CAUSE_COHOMOLOGY_REPULSION, ROOT_CAUSE_FACT_RETRIEVAL_MISS, ROOT_CAUSE_NOMINAL,
    ROOT_CAUSE_PERSONA_FACT_COLLISION, ROOT_CAUSE_SYSTEM_QOS_THROTTLE,
};

/// Default destructive phase coherence threshold (`0.25`) detecting Persona-vs-Fact collision ($\mathcal{C}_{\mathrm{FP}} < -0.25$).
pub const DEFAULT_PERSONA_FACT_COLLISION_THRESH: f32 = 0.25;
#[allow(unused_imports)]
use crate::runtime::horizon::{EpistemicHorizonReading, EpistemicManifoldOps};
#[allow(unused_imports)]
use crate::runtime::telemetry::CoreTelemetryBank;
#[allow(unused_imports)]
use crate::stigmergy::attractor::hash_prompt_to_q7_phases;
#[allow(unused_imports)]
use crate::stigmergy::{
    gcso_anchor_type_t, gcso_ledger_slot_t, gcso_pointer_trail_t, swar_mean_u8_le128_x8,
    trail_phase_words_mut, trail_phase_words_ref, validate_optional_qk_tensors, ActionHub,
    ActionHubEngine, ActionOrchestrationReport, AttractorAnchor, AttractorGrowthReport,
    CognitiveCircuitActivation, CognitiveCircuitBank, CognitiveCircuitBankOps, CognitiveOpKind,
    CognitiveStepOp, CognitiveTrajectoryLedger, ContrastivePhaseBloomFilter,
    CrystallizedAnchorRecord, CrystallizedAttractorRecord, DissipativeOverflowReport,
    EpisodicPhaseSample, FastPathLedger, PhaseSteeringEngine, RelativeAttractorNode,
    ResidualAdapterEngine, SandboxLifecycleEvent, StigmergicActionDecision,
    StigmergicFragmentEnsembleReport, StigmergicGranuleField, StigmergicOverlapSerendipityReport,
    StigmergicReconstitutionReport, StigmergicScaleGranularity, StigmergicThresholds,
    WorkbenchSlotSpec, ACTION_HUB_INLINE_TRAILS, DEFAULT_STIGMERGIC_QUORUM_THRESHOLD,
    DEFAULT_STIGMERGIC_TOLERANCE_Q7, GCSO_ANCHOR_TYPE_CRYSTALLIZED,
    LEDGER_FLAG_COGNITIVE_TRAJECTORY, LEDGER_METRIC_FAST_PATH_HIT,
    LEDGER_METRIC_LAYER_SKIP_COMPENSATED, LEDGER_METRIC_PHASE_STEERED,
    LEDGER_METRIC_RANK1_INJECTED, LEDGER_METRIC_REPULSION_APPLIED,
    LEDGER_METRIC_SLOT_SYNERGY_ACTIVE, MAX_ATTRACTOR_PULL_FORCE, MAX_STIGMERGIC_DENSITY,
    TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE,
    TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_PHASE_STEER_ACTIVE, TRAIL_FLAG_PHEROMONE_RISING,
    TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_FLAG_SYNERGY_RESONANCE, TRAIL_SWARM_QUORUM_MASK,
    TRAIL_SWARM_QUORUM_SHIFT,
};

pub mod bridges;
pub mod closed_loop;
pub mod frame;
pub mod orchestrator;
pub mod signals;

pub(crate) use self::frame::tangent_cosine_active_heads_64;
pub use self::{bridges::*, closed_loop::*, frame::*, orchestrator::*, signals::*};
