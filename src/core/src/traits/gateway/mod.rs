// name: src/core/src/traits/gateway/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lv.3 2-Origin Supertrait Hierarchy, Causal Provenance, Single Composite Manifold Gateway,
//! and autonomous Gateway-derived physical parameter synthesis.
//!
//! Macro-basin clustering ([`CrystallizedAttractorRecord`]) and 4-section `.gcso` zero-copy
//! container persistence ([`GcsoFourSectionView`]) live in [`crate::crystal`] and
//! [`crate::storage`] respectively.

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
    BranchlessKernel, Liminal, atan2_interleaved_pairs_to_q7_word_x8_simd,
    atan2_to_q7_word_x8_simd,
    branchless_phase_coherence_inv_temp_scale,
    circular_resultant_from_words_64, coherence_from_uniform_words_x8,
    coherence_from_word_arrays_with_both_uniform_flags_64, encode_pframe_motion_delta_64,
    scale_f32_slice_unrolled8, seek_pframe_chain_words_64,
    swar_slerp_words_with_uniform_flag_64, swar_words_equal_64, swar_words_nonzero_64,
};
use crate::traits::primitives::{
    SWAR_LOW7_MASK_64, SWAR_SIGN_MASK_64, SWAR_SPLAT_BYTE_64,
    branchless_atan2_to_q7_fast, branchless_clamp_f32, branchless_crc32_ieee,
    branchless_flush_subnormal_f32, branchless_inv_pos_f32, branchless_is_finite_f32,
    branchless_lerp_f32, branchless_max_f32, branchless_min_f32, branchless_rad_to_qdps_step_u8,
    branchless_relu_f32, branchless_select_f32, branchless_select_i32, branchless_select_i8,
    branchless_select_u32,
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
    ContrastivePhaseBloomFilter, DEFAULT_STIGMERGIC_QUORUM_THRESHOLD,
    DEFAULT_STIGMERGIC_TOLERANCE_Q7, LEDGER_FLAG_COGNITIVE_TRAJECTORY, LEDGER_METRIC_FAST_PATH_HIT,
    LEDGER_METRIC_LAYER_SKIP_COMPENSATED, LEDGER_METRIC_PHASE_STEERED,
    LEDGER_METRIC_RANK1_INJECTED, LEDGER_METRIC_REPULSION_APPLIED,
    LEDGER_METRIC_SLOT_SYNERGY_ACTIVE, MAX_ATTRACTOR_PULL_FORCE,
    NUM_STIGMERGIC_GRANULES, StigmergicReconstitutionOps, StigmergicReconstitutionReport,
    TRAIL_FLAG_INERTIAL_SKIP_ACTIVE,
};
use crate::traits::{ActionHub, validate_optional_qk_tensors};
use crate::abi::{
    GCSO_ABI_VERSION_HEX, GCSO_ERROR_INVALID_ARGUMENT, gcso_config_t, gcso_ledger_slot_t,
    gcso_edbc_state_t, gcso_pointer_trail_t, gcso_subhead_config_t, gcso_q7_t,
    gcso_tagged_ptr_t,
};


pub use crate::crystal::{CrystallizedAttractorRecord, DissipativeOverflowReport};
pub use crate::storage::GcsoFourSectionView;
pub use crate::traits::primitives::{
    APERTURE_ACTION_HUB_WIDE, APERTURE_CAUSAL_PROVENANCE, APERTURE_COGNITIVE_CIRCUIT,
    APERTURE_COHOMOLOGY_EDBC, APERTURE_CONTRADICTION_GUARD, APERTURE_DISSIPATIVE_CRYSTAL,
    APERTURE_FAST_PATH_LEDGER, APERTURE_KV_PHASE_CACHE, APERTURE_MANIFOLD_GATEWAY,
    APERTURE_MAPPED_STORAGE, APERTURE_OBJECTIVE_FACT, APERTURE_PHASE_STEERING,
    APERTURE_POINTER_TRAIL, APERTURE_RANK1_RESIDUAL, APERTURE_RELATIVE_COUPLING,
    APERTURE_STIGMERGIC_CONSENSUS, APERTURE_SUBHEAD_ROUTING, APERTURE_SUBJECTIVE_PERSONA,
    APERTURE_SYSTEM_CONSTRAINT, CORE_LAYER_EGO, CORE_LAYER_LIMINAL, CORE_LAYER_STIGMERGIC,
    EGO_TRIGGER_ALL_DEFAULT, EGO_TRIGGER_BASIN_CAPTURE, EGO_TRIGGER_BOUNDARY_CHATTER,
    EGO_TRIGGER_BOUNDARY_STAGNATION, EGO_TRIGGER_INTERNAL_CONFLICT, EGO_TRIGGER_LIMINAL_EXTERIOR,
    NUM_CORE_AUTONOMOUS_LAYERS, NUM_LIMINAL_APERTURE_CHANNELS,
};

pub mod circuit;
pub mod core_layers;
pub mod epistemic;
pub mod frame;
pub mod kv_cache;
pub mod orchestrator;
pub mod signals;

pub use self::circuit::{
    project_gateway_signal_at_scale, CognitiveCircuitActivation, CognitiveCircuitBank,
    CognitiveCircuitBankOps, CognitiveCircuitPlasticity, CognitiveOpKind, CognitiveStepOp,
    CognitiveTrajectoryLedger, HighCapCognitiveBank, HybridScaleCircuit,
    LowSpecCognitiveBank, ScaleLayerNode, ScaleLayerProjection, StandardCognitiveBank,
    StigmergicScaleGranularity, StigmergicScaleLayer,
};
pub use self::core_layers::*;
pub use self::epistemic::*;
pub use self::frame::*;
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
    assert!(size_of::<EpistemicHorizonReading>() == 32);
    assert!(align_of::<EpistemicHorizonReading>() == 16);
    assert!(offset_of!(EpistemicHorizonReading, horizon_gate) == 12);
    assert!(offset_of!(EpistemicHorizonReading, regime_code) == 28);
    assert!(size_of::<LiminalApertureField>() == 32);
    assert!(align_of::<LiminalApertureField>() == 16);
    assert!(offset_of!(LiminalApertureField, active_aperture_mask) == 24);
    assert!(offset_of!(LiminalApertureField, adaptation_epochs) == 28);
    assert!(size_of::<LiminalTopologyTelemetrySnapshot>() == 32);
    assert!(align_of::<LiminalTopologyTelemetrySnapshot>() == 16);
    assert!(offset_of!(LiminalTopologyTelemetrySnapshot, active_aperture_mask) == 24);
    assert!(offset_of!(LiminalTopologyTelemetrySnapshot, spawned_daughters) == 31);
    assert!(size_of::<StigmergicVennSerendipityReport>() == 32);
    assert!(align_of::<StigmergicVennSerendipityReport>() == 16);
    assert!(offset_of!(StigmergicVennSerendipityReport, dominant_overlap_anchor_id) == 12);
    assert!(offset_of!(StigmergicVennSerendipityReport, serendipitous_seed_word) == 16);
    assert!(offset_of!(StigmergicVennSerendipityReport, venn_core_mask) == 24);
    assert!(offset_of!(StigmergicVennSerendipityReport, serendipity_triggered) == 30);
    assert!(size_of::<StigmergicGranuleField>() == 32);
    assert!(align_of::<StigmergicGranuleField>() == 16);
    assert!(offset_of!(StigmergicGranuleField, venn_overlap_mask) == 24);
    assert!(offset_of!(StigmergicGranuleField, adaptation_epochs) == 28);
    assert!(size_of::<AutonomousCoreLayerStack>() == 128);
    assert!(align_of::<AutonomousCoreLayerStack>() == 64);
    assert!(offset_of!(AutonomousCoreLayerStack, liminal_layer) == 0);
    assert!(offset_of!(AutonomousCoreLayerStack, stigmergic_layer) == 32);
    assert!(offset_of!(AutonomousCoreLayerStack, ego_layer) == 64);
    assert!(size_of::<LiminalBoundaryDelta>() == 16);
    assert!(align_of::<LiminalBoundaryDelta>() == 16);
    assert!(offset_of!(LiminalBoundaryDelta, channel_floor_velocity_q7) == 0);
    assert!(offset_of!(LiminalBoundaryDelta, granule_transition_mask) == 8);
    assert!(offset_of!(LiminalBoundaryDelta, chattering_channel_mask) == 9);
    assert!(offset_of!(LiminalBoundaryDelta, kinetic_energy_q7) == 10);
    assert!(offset_of!(LiminalBoundaryDelta, stationary_shortcut_ready) == 11);
    assert!(offset_of!(LiminalBoundaryDelta, expansion_dominant) == 12);
    assert!(size_of::<EgoIntentAxes>() == 16);
    assert!(align_of::<EgoIntentAxes>() == 16);
    assert!(offset_of!(EgoIntentAxes, autonomy_q7) == 0);
    assert!(offset_of!(EgoIntentAxes, preferred_slot_mask) == 8);
    assert!(offset_of!(EgoIntentAxes, active_granule_bias_mask) == 9);
    assert!(offset_of!(EgoIntentAxes, confidence_q7) == 10);
    assert!(offset_of!(EgoIntentAxes, flags) == 11);
    assert!(offset_of!(EgoIntentAxes, signature_hash) == 12);
    assert!(size_of::<StigmergicFragmentEnsembleReport>() == 16);
    assert!(align_of::<StigmergicFragmentEnsembleReport>() == 16);
    assert!(offset_of!(StigmergicFragmentEnsembleReport, quorum_agreed_granule_mask) == 8);
    assert!(offset_of!(StigmergicFragmentEnsembleReport, split_contested_granule_mask) == 9);
    assert!(offset_of!(StigmergicFragmentEnsembleReport, ego_tiebreak_granule_mask) == 10);
    assert!(offset_of!(StigmergicFragmentEnsembleReport, active_fragment_count) == 11);
    assert!(offset_of!(StigmergicFragmentEnsembleReport, dominant_fragment_slot) == 12);
    assert!(size_of::<LiminalEgoField>() == 64);
    assert!(align_of::<LiminalEgoField>() == 64);
    assert!(offset_of!(LiminalEgoField, sovereign_granule_phase_q7) == 0);
    assert!(offset_of!(LiminalEgoField, trait_floor_bias_q7) == 8);
    assert!(offset_of!(LiminalEgoField, trait_gain_bias_q7) == 16);
    assert!(offset_of!(LiminalEgoField, slot_affinity_bias_q7) == 24);
    assert!(offset_of!(LiminalEgoField, granule_autonomy_q7) == 32);
    assert!(offset_of!(LiminalEgoField, counter_basin_thrust_q7) == 40);
    assert!(offset_of!(LiminalEgoField, basin_escape_floor_q7) == 41);
    assert!(offset_of!(LiminalEgoField, transgression_bias_q7) == 42);
    assert!(offset_of!(LiminalEgoField, irrational_defiance_q7) == 43);
    assert!(offset_of!(LiminalEgoField, trigger_mask) == 44);
    assert!(offset_of!(LiminalEgoField, preset_code) == 47);
    assert!(offset_of!(LiminalEgoField, ego_signature_hash) == 48);
    assert!(offset_of!(LiminalEgoField, intervention_epochs) == 52);
    assert!(offset_of!(LiminalEgoField, last_intervened_mask) == 56);
};

// SAFETY: `EpistemicHorizonReading` is `#[repr(C, align(16))]`, exact size 32 bytes with zero padding.
unsafe impl Pod for EpistemicHorizonReading {}
// SAFETY: `LiminalApertureField` is `#[repr(C, align(16))]`, exact size 32 bytes with zero padding.
unsafe impl Pod for LiminalApertureField {}
// SAFETY: `LiminalTopologyTelemetrySnapshot` is `#[repr(C, align(16))]`, exact size 32 bytes with zero padding.
unsafe impl Pod for LiminalTopologyTelemetrySnapshot {}
// SAFETY: `StigmergicVennSerendipityReport` is `#[repr(C, align(16))]`, exact size 32 bytes with zero padding.
unsafe impl Pod for StigmergicVennSerendipityReport {}
// SAFETY: `StigmergicGranuleField` is `#[repr(C, align(16))]`, exact size 32 bytes with zero padding.
unsafe impl Pod for StigmergicGranuleField {}
// SAFETY: `AutonomousCoreLayerStack` is `#[repr(C, align(64))]`, exact size 128 bytes with zero padding.
unsafe impl Pod for AutonomousCoreLayerStack {}
// SAFETY: `LiminalBoundaryDelta` is `#[repr(C, align(16))]`, exact size 16 bytes with zero padding.
unsafe impl Pod for LiminalBoundaryDelta {}
// SAFETY: `EgoIntentAxes` is `#[repr(C, align(16))]`, exact size 16 bytes with zero padding.
unsafe impl Pod for EgoIntentAxes {}
// SAFETY: `StigmergicFragmentEnsembleReport` is `#[repr(C, align(16))]`, exact size 16 bytes with zero padding.
unsafe impl Pod for StigmergicFragmentEnsembleReport {}
// SAFETY: `LiminalEgoField` is `#[repr(C, align(64))]`, exact size 64 bytes with zero padding.
unsafe impl Pod for LiminalEgoField {}


#[cfg(test)]
mod tests;
