// name: src/core/src/traits/gateway/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lv.3 2-Origin Supertrait Hierarchy, Causal Provenance, Single Composite Manifold Gateway,
//! and autonomous Gateway-derived physical parameter synthesis.
//!
//! Macro-basin clustering ([`CrystallizedAttractorRecord`]) and 4-section `.gcso` zero-copy
//! container persistence ([`GcsoFourSectionView`]) live in [`crate::crystal`] and
//! [`crate::storage`] respectively and are re-exported here for backward compatibility.

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
use core::mem::{align_of, offset_of, size_of};
use crate::action_hub::{StigmergicActionDecision, StigmergicThresholds};
use crate::prelude::MAX_STIGMERGIC_DENSITY;
use crate::TRAIL_FLAG_REPULSION_ACTIVE;
use crate::traits::aligned::{
    AlignedBuffer16, AlignedSlice, AlignedSlice16, AlignedSlice32, AlignedSliceMut32,
    is_aligned_to,
};
use crate::traits::branchless::{
    BranchlessKernel, atan2_interleaved_pairs_to_q7_word_x8_simd, atan2_to_q7_word_x8_simd,
    branchless_phase_coherence_inv_temp_scale, branchless_q7_circular_resultant_64,
    circular_resultant_from_words_64, coherence_from_uniform_words_x8,
    coherence_from_word_arrays_with_both_uniform_flags_64, encode_pframe_motion_delta_64,
    scale_f32_slice_unrolled8, seek_pframe_chain_words_64,
    swar_slerp_words_with_uniform_flag_64, swar_words_equal_64, swar_words_nonzero_64,
};
use crate::traits::primitives::{
    SWAR_LOW7_MASK_64, SWAR_SIGN_MASK_64, SWAR_SPLAT_BYTE_64, branchless_abs_f32,
    branchless_atan2_to_q7_fast, branchless_clamp_f32, branchless_crc32_ieee,
    branchless_flush_subnormal_f32, branchless_inv_pos_f32, branchless_is_finite_f32,
    branchless_lerp_f32, branchless_max_f32, branchless_min_f32, branchless_rad_to_qdps_step_u8,
    branchless_relu_f32, branchless_select_f32, branchless_select_i8, branchless_select_u32,
    branchless_select_u64, branchless_select_u8, branchless_select_usize,
    branchless_sincos_q7_ripa, branchless_splitmix64, crc32_block64_zero_seed_words,
    crc32_step_block64_words, crc32_step_l16, crc32_step_l4, crc32_step_l8,
    fast_inv_pos_normal_f32, is_word_array_uniform_64, load_q7_words_64, store_q7_words_64,
    swar_q7_abs_u8_x8, swar_q7_circular_add_x8, swar_q7_circular_diff_x8,
    swar_q7_mean_i8_from_words_64, swar_q7_mean_i8_from_words_with_uniform_flag_64,
    swar_q7_neg_sat_x8, swar_q7_qdps_filter_words_64,
    swar_q7_qdps_filter_words_with_uniform_flag_64, swar_q7_scale_le128_with_bias_x8,
    trail_phase_words_mut, trail_phase_words_ref,
};
use crate::traits::primitives::{
    ACTION_MODE_DPSR_STEER, ACTION_MODE_LAYER_SKIP, ACTION_MODE_PHASE_REPULSE,
    ACTION_MODE_SUBHEAD_ROUTE, ACTION_MODE_RANK1_INJECT, ACTION_PRESET_EXPRESS_PHASE,
    ACTION_PRESET_INERTIAL_SKIP, ACTION_PRESET_PASSIVE_SIDECAR, ACTION_PRESET_REPULSION_ESCAPE,
    ATTRACTOR_SLOT_ID_MASK, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA,
    ATTRACTOR_SLOT_REPULSION, ATTRACTOR_SLOT_SYSTEM_BASELINE, LEDGER_HISTORY_LEN,
    DEFAULT_ENTROPY_EPS, DEFAULT_QDPS_MIN_STEP_RAD, DEFAULT_RIPA_CLAMP_MAX_RAD, GcsoResult,
    KvKeyframePacket, KvMotionDeltaPacket, MAX_ATTENTION_HEADS, Pod, pack_action_slot_tag,
};
use crate::traits::rope::{
    PhaseSteering, PhaseSteeringGeometryExt, SubHeadRouter, RopePairLayout, RopeSteeringTarget,
    Rank1ResidualAdapter, dispatch_fused_ripa_and_pspm_words_core, resolve_autonomous_swarm_rot_dim,
    rotate_fused_self_ripa_and_pspm_default_words,
    rotate_fused_self_ripa_pspm_q_and_ripa_k_default_words, steer_from_swar_words_core_safe,
};
use crate::traits::telemetry::{
    ContrastivePhaseBloomFilter, DEFAULT_SWARM_QUORUM_THRESHOLD, DEFAULT_SWARM_TOLERANCE_Q7,
    LEDGER_FLAG_COGNITIVE_TRAJECTORY, LEDGER_METRIC_FAST_PATH_HIT,
    LEDGER_METRIC_LAYER_SKIP_COMPENSATED, LEDGER_METRIC_PHASE_STEERED, LEDGER_METRIC_SUBHEAD_ROUTED,
    LEDGER_METRIC_REPULSION_APPLIED, LEDGER_METRIC_SLOT_SYNERGY_ACTIVE, LEDGER_METRIC_RANK1_INJECTED,
    MAX_ATTRACTOR_PULL_FORCE, NUM_SWARM_MICRO_CELLS, StigmergicSwarmOps, SwarmResolutionReport,
    TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE,
    TRAIL_FLAG_SYNERGY_RESONANCE,
};
use crate::traits::{ActionHub, validate_optional_qk_tensors};
use crate::abi::{
    GCSO_ABI_VERSION_HEX, GCSO_ERROR_INVALID_ARGUMENT, gcso_config_t, gcso_ledger_slot_t,
    gcso_edbc_state_t, gcso_pointer_trail_t, gcso_subhead_config_t, gcso_q7_t, gcso_rank1_residual_descriptor_t,
    gcso_tagged_ptr_t,
};


pub use crate::crystal::CrystallizedAttractorRecord;
pub use crate::storage::GcsoFourSectionView;

pub mod circuit;
pub mod kv_cache;
pub mod orchestrator;
pub mod signals;

pub use self::circuit::*;
pub use self::kv_cache::*;
pub use self::orchestrator::*;
pub use self::signals::*;

// ===================================================================
// 12. Compile-Time Layout Verification & Unit Tests
// ===================================================================

const _: () = {
    assert!(size_of::<CausalProvenanceReport>() == 16);
    assert!(align_of::<CausalProvenanceReport>() == 16);
    assert!(offset_of!(CausalProvenanceReport, provenance_bits) == 0);
    assert!(offset_of!(CausalProvenanceReport, fact_persona_coherence) == 4);
    assert!(offset_of!(CausalProvenanceReport, anomaly_severity) == 8);
    assert!(offset_of!(CausalProvenanceReport, dominant_cause_code) == 12);

    assert!(size_of::<GatewayFrame>() == 128);
    assert!(align_of::<GatewayFrame>() == 64);
    assert!(offset_of!(GatewayFrame, tagged_ptr) == 0);
    assert!(offset_of!(GatewayFrame, mapped_section_offset48) == 8);
    assert!(offset_of!(GatewayFrame, target_anchor_id) == 16);
    assert!(offset_of!(GatewayFrame, effective_energy) == 20);
    assert!(offset_of!(GatewayFrame, subhead_fact_gain_q7) == 56);
    assert!(offset_of!(GatewayFrame, mean_phase_tag_q7) == 59);
    assert!(offset_of!(GatewayFrame, active_head_count) == 60);
    assert!(offset_of!(GatewayFrame, frame_flags) == 62);
    assert!(offset_of!(GatewayFrame, phase_deltas) == 64);

    assert!(size_of::<CrystallizedAttractorRecord>() == 128);
    assert!(align_of::<CrystallizedAttractorRecord>() == 64);
    assert!(offset_of!(CrystallizedAttractorRecord, magic) == 0);
    assert!(offset_of!(CrystallizedAttractorRecord, version) == 4);
    assert!(offset_of!(CrystallizedAttractorRecord, anchor_id) == 8);
    assert!(offset_of!(CrystallizedAttractorRecord, sample_count) == 12);
    assert!(offset_of!(CrystallizedAttractorRecord, mapped_section_offset48) == 16);
    assert!(offset_of!(CrystallizedAttractorRecord, checksum_crc32) == 52);
    assert!(offset_of!(CrystallizedAttractorRecord, dominant_slot) == 56);
    assert!(offset_of!(CrystallizedAttractorRecord, reserved_align64) == 60);
    assert!(offset_of!(CrystallizedAttractorRecord, phase_profile) == 64);
    assert!(GCSO_FOUR_SECTION_HEADER_BYTES == 192);
    assert!(GCSO_MAPPED_STORAGE_PREAMBLE_OFFSET == 128);
    assert!(
        CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_PERSONA, 0).verify_integrity()
    );
    assert!(
        CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_KNOWLEDGE, 0).verify_integrity()
    );
    assert!(size_of::<CognitiveStepOp>() == 1);
    assert!(align_of::<CognitiveStepOp>() == 1);
    assert!(size_of::<CognitiveCircuitActivation>() == 8);
    assert!(align_of::<CognitiveCircuitActivation>() == 8);
    assert!(size_of::<CognitiveTrajectoryLedger>() == LEDGER_HISTORY_LEN);
    assert!(align_of::<CognitiveTrajectoryLedger>() == 8);
    assert!(size_of::<RamPhaseKvEntry>() == 192);
    assert!(align_of::<RamPhaseKvEntry>() == 64);
    assert!(offset_of!(RamPhaseKvEntry, keyframe) == 0);
    assert!(offset_of!(RamPhaseKvEntry, circuit) == 128);
    assert!(offset_of!(RamPhaseKvEntry, last_activation) == 136);
    assert!(offset_of!(RamPhaseKvEntry, context_hash) == 144);
    assert!(offset_of!(RamPhaseKvEntry, reuse_count) == 152);
    assert!(offset_of!(RamPhaseKvEntry, last_used_epoch) == 156);
};


#[cfg(test)]
mod tests;
