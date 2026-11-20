// name: src/core/src/runtime/horizon/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Level 4 Latent Horizon (`epistemic`, `kv_cache`, and dynamic slide-in `slots`).

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
use crate::drivers::{
    ENTROPY_GATE_MODE_NORMAL_ENTROPY, ENTROPY_GATE_MODE_PHASE_CONJUGATE_REPULSION,
};
#[allow(unused_imports)]
use crate::foundation::{
    branchless_clamp_f32, branchless_inv_pos_f32, branchless_is_finite_f32, branchless_max_f32,
    branchless_min_f32, branchless_q7_inhibit_words_64, branchless_rad_to_phase_step_u8,
    branchless_relu_f32, branchless_sincos_q7, branchless_splitmix64,
    circular_resultant_from_words_64, coherence_from_word_arrays_64,
    coherence_from_word_arrays_with_both_uniform_flags_64, encode_pframe_motion_delta_64,
    fast_mean_q7_tag_from_words_64, gcso_tagged_ptr_t, is_word_array_uniform_64, load_q7_words_64,
    pack_action_slot_tag, seek_pframe_chain_words_64, store_q7_words_64, swar_q7_abs_u8_x8,
    swar_q7_circular_add_x8, swar_q7_circular_diff_x8, swar_q7_mean_i8_from_words_64,
    swar_q7_scale_le128_with_bias_x8, swar_words_nonzero_64, words_to_q7_array_64,
    KvMotionDeltaPacket, RawPhaseProjectable, ACTION_MODE_PHASE_REPULSE, ACTION_MODE_RANK1_INJECT,
    ACTION_MODE_SUBHEAD_ROUTE, ACTION_PRESET_EXPRESS_PHASE, ACTION_PRESET_PASSIVE_SIDECAR,
    ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_ID_MASK, DEFAULT_PHASE_CLAMP_MAX_RAD,
    DEFAULT_PHASE_STEP_MIN_RAD, SWAR_SIGN_MASK_64, SWAR_SPLAT_BYTE_64,
};
#[allow(unused_imports)]
use crate::liminal::{
    branchless_select_f32, branchless_select_i8, branchless_select_u32, branchless_select_u64,
    branchless_select_u8, gcso_q7_t, EpistemicRegime, GcsoResult, LiminalThresholdField,
    DEFAULT_ENTROPY_EPS, GCSO_ERROR_INVALID_ARGUMENT, MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS,
    NUM_LIMINAL_THRESHOLD_CHANNELS, NUM_STIGMERGIC_GRANULES, THRESHOLD_ACTION_HUB_WIDE,
};
#[allow(unused_imports)]
use crate::runtime::context::AutonomousCoreLayerStack;
#[allow(unused_imports)]
use crate::runtime::gateway::{
    tangent_cosine_active_heads_64, CausalProvenanceSignal, CognitiveCircuitThresholdPolicy,
    CognitiveGroundingSignal, GatewayFrame, ManifoldGatewaySignal, ObjectiveFactSignal,
};
#[allow(unused_imports)]
use crate::runtime::storage::KvKeyframePacket;
#[allow(unused_imports)]
use crate::stigmergy::{
    evaluate_stigmergic_overlap_and_serendipity_words_64, fold_fractal_nodes_words_64,
    fold_sincos64_to_words, swar_mean_u8_le128_x8, ActionHubEngine, CognitiveCircuitPlasticity,
    CognitiveOpKind, CognitiveTrajectoryLedger, DissipativeOverflowReport, PheromoneDensityOps,
    RelativeAttractorNode, StigmergicGranuleField, StigmergicOverlapSerendipityReport,
    StigmergicTrailOps, WorkbenchLanePlacement, WorkbenchSlotSpec, MAX_ATTRACTOR_PULL_FORCE,
    TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE,
};
#[allow(unused_imports)]
use crate::stigmergy::{
    ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_REPULSION, CAUSAL_AXIS_CONTRADICTION_GUARD,
    CAUSAL_AXIS_OBJECTIVE_FACT, CAUSAL_CAUSE_COHOMOLOGY_REPULSION,
    CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK, DEFAULT_FACT_PHASE_CLAMP_RAD,
    ROOT_CAUSE_COHOMOLOGY_REPULSION, ROOT_CAUSE_FACT_RETRIEVAL_MISS, ROOT_CAUSE_NOMINAL,
    WORKBENCH_LANE_FLAG_PINNED,
};

pub mod epistemic;
pub mod kv_cache;
pub mod slots;

pub use self::{epistemic::*, kv_cache::*, slots::*};
