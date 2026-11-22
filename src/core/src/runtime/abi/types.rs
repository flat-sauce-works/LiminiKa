// name: src/core/src/runtime/abi/types.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! C-ABI constants, status codes, `#[repr(C)]` POD aliases (`lmnk_*_t`),
//! Portable Device-Ready POD vs Host FFI Descriptor wire contract, and compile-time
//! layout assertions for `LiminiKa` `v0.2.0` (`0x0000_0200`).
//!
//! # Portable Wire & Device Contract
//! - **Byte Order**: All multi-byte scalars and `u64` SWAR words use **Canonical Little-Endian**.
//! - **Portable Device-Ready PODs**: Power-of-two `16 B..=128 B` structures using fixed-width
//!   types with `0` implicit padding and 64 B-aligned `[PhaseQ7; 64]` arrays directly consumable
//!   as `u32x16` / `u64x8` (WGSL, Vulkan `std430`, Metal, CUDA, x86_64 64 B CL, ARM64 128 B CL).
//! - **Host FFI Descriptors**: Host-only pointer-carrying structs (`Rank1Descriptor`, `Config`
//!   wrappers) are classified separately from cross-device buffer PODs.

use super::*;

pub use crate::drivers::{
    DistortionDelta, DistortionTension, EgoIntentAxes, EntropyController, EntropyState,
    LiminalEgoState, SystemSpineState,
};
pub use crate::foundation::{
    AlignedSlice, AlignedSliceMut, DisjointMutTriple, KvMotionDeltaPacket, PagedBitmask,
    Rank1Descriptor, SubheadConfig, TaggedPtr,
};
pub use crate::liminal::{
    validate_abi_version, AttributionTag, LacunaPartition8, LiminalThresholdState, LmnkStatus,
    PhaseQ7, RuntimeConfig, LMNK_ABI_VERSION_CSTR, LMNK_ABI_VERSION_HEX, LMNK_ABI_VERSION_MAJOR,
    LMNK_ABI_VERSION_MINOR, LMNK_ABI_VERSION_PATCH, LMNK_ABI_VERSION_STRING,
    LMNK_CAP_INPLACE_CONTEXT_INIT, LMNK_CAP_SIMD_AVX2, LMNK_CAP_SIMD_NEON, LMNK_CAP_SIMD_WASM128,
    LMNK_CAP_SWAR_BASELINE, LMNK_CAP_ZERO_ALLOC_CORE, LMNK_ERR_ALIGNMENT,
    LMNK_ERR_ANCHOR_NOT_FOUND, LMNK_ERR_BUFFER_TOO_SMALL, LMNK_ERR_CONTAINER_CORRUPTED,
    LMNK_ERR_ENTROPY_SINGULARITY, LMNK_ERR_EXTENSION_NOT_LOADED, LMNK_ERR_INVALID_ARGUMENT,
    LMNK_ERR_INVALID_STATE, LMNK_ERR_IO_FAILURE, LMNK_ERR_KV_KEYFRAME_SEEK_FAILED,
    LMNK_ERR_LEDGER_FULL, LMNK_ERR_NOT_IMPLEMENTED, LMNK_ERR_NULL_POINTER,
    LMNK_ERR_OBSTRUCTION_UNRESOLVED, LMNK_ERR_OUT_OF_MEMORY, LMNK_ERR_PANIC_CAUGHT,
    LMNK_ERR_PHASE_OVERFLOW, LMNK_ERR_PHASE_STEP_UNDERFLOW, LMNK_ERR_STORAGE_MAP_FAILED,
    LMNK_ERR_SUBHEAD_ROUTING_FAILED, LMNK_ERR_SUBSTRATE_FULL, LMNK_ERR_UNKNOWN,
    LMNK_ERR_VERSION_MISMATCH, LMNK_OK,
};
pub use crate::runtime::boundary_delta::LiminalBoundaryDelta;
pub use crate::runtime::context::CoreStateStack;
pub use crate::runtime::gateway::{CausalProvenanceReport, GatewayFrame};
pub use crate::runtime::horizon::{EpistemicHorizonReading, RamPhaseKvEntry};
pub use crate::runtime::storage::{
    DescriptorHeader, KvKeyframeHeader, KvKeyframePacket, LmnkSnapshotHeader,
    LMNK_FOUR_SECTION_HEADER_BYTES, LMNK_SNAPSHOT_MAGIC_U32, LMNK_STORAGE_PREAMBLE_OFFSET,
};
pub use crate::runtime::telemetry::LiminalTopologyTelemetrySnapshot;
pub use crate::stigmergy::{
    CognitiveCircuitActivation, CognitiveStepOp, CognitiveTrajectoryLedger,
    ContrastivePhaseBloomFilter, CorePrimitiveState, LedgerSlot, OverlapDiscoveryReport,
    PhaseAnchorRecord, PointerTrail, StigmergicFragmentEnsembleReport, StigmergicGranuleState,
    StigmergicReconstitutionReport, StigmergicThresholds, SubstrateStepReport,
    ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA, LEDGER_HISTORY_LEN, LMNK_PHASE_ANCHOR_MAGIC,
    LMNK_PHASE_ANCHOR_VERSION, MAX_PHASE_ANCHORS,
};

/// 64-bit capability flags bitmask type (`lmnk_capability_flags_t`).
pub type lmnk_capability_flags_t = u64;
/// Standardized 32-bit signed status code (`lmnk_status_t`).
pub type lmnk_status_t = LmnkStatus;
/// Signed Q0.7 phase element (`lmnk_q7_t`).
pub type lmnk_q7_t = PhaseQ7;
/// 64-bit tagged pointer (`lmnk_tagged_ptr_t`).
pub type lmnk_tagged_ptr_t = TaggedPtr;
/// 64-byte C-ABI runtime configuration descriptor (`RuntimeConfig`).
pub type lmnk_config_t = RuntimeConfig;
/// 8-byte C-ABI structure descriptor header (`DescriptorHeader`).
pub type lmnk_descriptor_header_t = DescriptorHeader;
/// 128-byte `.lmnk` snapshot container header (`LmnkSnapshotHeader`).
pub type lmnk_snapshot_header_t = LmnkSnapshotHeader;
/// 64-byte KV-keyframe header (`KvKeyframeHeader`).
pub type lmnk_kv_keyframe_header_t = KvKeyframeHeader;
/// 32-byte 256-bit PagedBlock bitmask (`PagedBitmask`).
pub type lmnk_paged_bitmask_t = PagedBitmask;
/// 32-byte parallel multi-head sub-head routing config (`SubheadConfig`).
pub type lmnk_subhead_config_t = SubheadConfig;
/// 64-byte Host FFI descriptor for Rank-1 residual injection (`Rank1Descriptor`).
pub type lmnk_rank1_descriptor_t = Rank1Descriptor;
/// 64-byte FastPathLedger telemetry and shortcut slot (`LedgerSlot`).
pub type lmnk_ledger_slot_t = LedgerSlot;
/// 128-byte dual-cacheline Stigmergic Pointer Trail (`PointerTrail`).
pub type lmnk_pointer_trail_t = PointerTrail;
/// 32-byte Entropy Controller state descriptor (`EntropyState`).
pub type lmnk_entropy_state_t = EntropyState;
/// 128-byte Gateway Frame (`GatewayFrame`).
pub type lmnk_gateway_frame_t = GatewayFrame;
/// 16-byte Causal Provenance Report (`CausalProvenanceReport`).
pub type lmnk_causal_report_t = CausalProvenanceReport;
/// 128-byte crystallized Phase Anchor Record (`PhaseAnchorRecord`, `"LNKR"`).
pub type lmnk_phase_anchor_record_t = PhaseAnchorRecord;
/// 64-byte Contrastive Phase Bloom Filter (`ContrastivePhaseBloomFilter`).
pub type lmnk_bloom_filter_t = ContrastivePhaseBloomFilter;
/// 32-byte Stigmergic Thresholds configuration (`StigmergicThresholds`).
pub type lmnk_stigmergic_thresholds_t = StigmergicThresholds;
/// 32-byte Substrate Step Report (`SubstrateStepReport`).
pub type lmnk_substrate_step_report_t = SubstrateStepReport;
/// 8-bit Stigmergic Action Decision discriminant (`StigmergicActionDecision`).
pub type lmnk_action_decision_t = u8;
/// 16-byte Stigmergic Reconstitution Report (`StigmergicReconstitutionReport`).
pub type lmnk_reconstitution_report_t = StigmergicReconstitutionReport;
/// 8-byte Cognitive Trajectory Ledger (`CognitiveTrajectoryLedger`).
pub type lmnk_cognitive_ledger_t = CognitiveTrajectoryLedger;
/// 32-byte Epistemic Horizon Reading (`EpistemicHorizonReading`).
pub type lmnk_epistemic_horizon_t = EpistemicHorizonReading;
/// 32-byte Liminal Threshold State (`LiminalThresholdState`).
pub type lmnk_liminal_threshold_state_t = LiminalThresholdState;
/// 32-byte Stigmergic Granule State (`StigmergicGranuleState`).
pub type lmnk_stigmergic_granule_state_t = StigmergicGranuleState;
/// 32-byte Overlap & Discovery Report (`OverlapDiscoveryReport`).
pub type lmnk_overlap_discovery_report_t = OverlapDiscoveryReport;
/// 128-byte Core State Stack (`CoreStateStack`).
pub type lmnk_core_state_stack_t = CoreStateStack;
/// 32-byte Self-Introspecting Topology Telemetry Snapshot (`LiminalTopologyTelemetrySnapshot`).
pub type lmnk_topology_snapshot_t = LiminalTopologyTelemetrySnapshot;
/// 16-byte Temporal Boundary Velocity Analyzer (`LiminalBoundaryDelta`).
pub type lmnk_boundary_delta_t = LiminalBoundaryDelta;
/// 16-byte Ego Intent Axes (`EgoIntentAxes`).
pub type lmnk_ego_intent_axes_t = EgoIntentAxes;
/// 16-byte Fragmented Micro-Ensemble Report (`StigmergicFragmentEnsembleReport`).
pub type lmnk_fragment_ensemble_report_t = StigmergicFragmentEnsembleReport;
/// 64-byte Wide-Aperture Subjective Ego State (`LiminalEgoState`).
pub type lmnk_liminal_ego_state_t = LiminalEgoState;
/// 16-byte 4-State Lacuna Partition (`LacunaPartition8`).
pub type lmnk_lacuna_partition_t = LacunaPartition8;
/// 1-byte Structural Attribution Tag (`AttributionTag`).
pub type lmnk_attribution_tag_t = AttributionTag;
/// 64-byte Unified Primitive State (`CorePrimitiveState`).
pub type lmnk_core_primitive_state_t = CorePrimitiveState;
/// 128-byte P-Frame Distortion Delta (`DistortionDelta`).
pub type lmnk_distortion_delta_t = DistortionDelta;
/// 32-byte Asymmetric Homeostatic Tension (`DistortionTension`).
pub type lmnk_distortion_tension_t = DistortionTension;
/// 32-byte Mechanical Execution Spine State (`SystemSpineState`).
pub type lmnk_spine_state_t = SystemSpineState;

// Compile-time layout assertions enforcing Portable Device-Ready POD guarantees
// and 1:1 C11/C++20 header parity for v0.2.0 (`0x0000_0200`).
const _: () = {
    assert!(LMNK_ABI_VERSION_HEX == 0x0000_0200);
    assert!(LMNK_SNAPSHOT_MAGIC_U32 == 0x4B4E_4D4C);
    assert!(LMNK_PHASE_ANCHOR_MAGIC == 0x524B_4E4C);
    assert!(LMNK_PHASE_ANCHOR_VERSION == 0x0000_0200);

    assert!(size_of::<lmnk_status_t>() == 4);
    assert!(size_of::<lmnk_q7_t>() == 1);
    assert!(size_of::<lmnk_tagged_ptr_t>() == 8 && align_of::<lmnk_tagged_ptr_t>() == 8);
    assert!(size_of::<lmnk_attribution_tag_t>() == 1);
    assert!(
        size_of::<lmnk_descriptor_header_t>() == 8 && align_of::<lmnk_descriptor_header_t>() == 4
    );
    assert!(offset_of!(lmnk_descriptor_header_t, struct_size) == 0);
    assert!(offset_of!(lmnk_descriptor_header_t, abi_version) == 4);

    assert!(size_of::<lmnk_paged_bitmask_t>() == 32 && align_of::<lmnk_paged_bitmask_t>() == 32);
    assert!(offset_of!(lmnk_paged_bitmask_t, bits) == 0);

    assert!(size_of::<lmnk_pointer_trail_t>() == 128 && align_of::<lmnk_pointer_trail_t>() == 64);
    assert!(offset_of!(lmnk_pointer_trail_t, current_ptr) == 0);
    assert!(offset_of!(lmnk_pointer_trail_t, user_data) == 16);
    assert!(offset_of!(lmnk_pointer_trail_t, reserved_header) == 56);
    assert!(offset_of!(lmnk_pointer_trail_t, accumulated_phase_delta) == 64);

    assert!(size_of::<lmnk_ledger_slot_t>() == 64 && align_of::<lmnk_ledger_slot_t>() == 64);
    assert!(offset_of!(lmnk_ledger_slot_t, mode) == 0);
    assert!(offset_of!(lmnk_ledger_slot_t, fast_path_shortcuts) == 24);
    assert!(offset_of!(lmnk_ledger_slot_t, telemetry_mini_ledger) == 56);

    assert!(
        size_of::<lmnk_stigmergic_thresholds_t>() == 32
            && align_of::<lmnk_stigmergic_thresholds_t>() == 32
    );
    assert!(
        size_of::<lmnk_substrate_step_report_t>() == 32
            && align_of::<lmnk_substrate_step_report_t>() == 32
    );

    assert!(size_of::<lmnk_config_t>() == 64 && align_of::<lmnk_config_t>() == 64);
    assert!(offset_of!(lmnk_config_t, head_dim) == 0);
    assert!(offset_of!(lmnk_config_t, phase_step_min_rad) == 20);
    assert!(offset_of!(lmnk_config_t, substrate_capacity) == 32);
    assert!(offset_of!(lmnk_config_t, rotary_dim) == 36);
    assert!(offset_of!(lmnk_config_t, execution_flags) == 40);
    assert!(offset_of!(lmnk_config_t, ledger_mode) == 42);
    assert!(offset_of!(lmnk_config_t, rope_layout) == 43);

    assert!(size_of::<lmnk_entropy_state_t>() == 32 && align_of::<lmnk_entropy_state_t>() == 32);
    assert!(offset_of!(lmnk_entropy_state_t, moving_z_entropy) == 0);
    assert!(offset_of!(lmnk_entropy_state_t, active_branch_mode) == 24);
    assert!(offset_of!(lmnk_entropy_state_t, flags) == 28);

    assert!(
        size_of::<lmnk_snapshot_header_t>() == 128 && align_of::<lmnk_snapshot_header_t>() == 64
    );
    assert!(offset_of!(lmnk_snapshot_header_t, magic) == 0);
    assert!(offset_of!(lmnk_snapshot_header_t, version) == 4);
    assert!(offset_of!(lmnk_snapshot_header_t, checksum_crc32) == 56);
    assert!(offset_of!(lmnk_snapshot_header_t, timestamp_epoch_sec) == 64);

    assert!(
        size_of::<lmnk_kv_keyframe_header_t>() == 64
            && align_of::<lmnk_kv_keyframe_header_t>() == 32
    );
    assert!(offset_of!(lmnk_kv_keyframe_header_t, frame_type) == 0);
    assert!(offset_of!(lmnk_kv_keyframe_header_t, icache_payload_offset) == 24);
    assert!(offset_of!(lmnk_kv_keyframe_header_t, pcache_payload_offset) == 32);

    assert!(size_of::<KvKeyframePacket>() == 128 && align_of::<KvKeyframePacket>() == 64);
    assert!(offset_of!(KvKeyframePacket, header) == 0);
    assert!(offset_of!(KvKeyframePacket, anchor_phases) == 64);
    assert!(size_of::<KvMotionDeltaPacket>() == 64 && align_of::<KvMotionDeltaPacket>() == 64);

    assert!(size_of::<lmnk_subhead_config_t>() == 32 && align_of::<lmnk_subhead_config_t>() == 16);
    assert!(offset_of!(lmnk_subhead_config_t, num_fact_heads) == 0);
    assert!(offset_of!(lmnk_subhead_config_t, fact_phase_gain) == 8);

    assert!(
        size_of::<lmnk_rank1_descriptor_t>() == 64 && align_of::<lmnk_rank1_descriptor_t>() == 32
    );
    assert!(offset_of!(lmnk_rank1_descriptor_t, layer_idx) == 0);
    assert!(offset_of!(lmnk_rank1_descriptor_t, u_vector_ptr) == 8);
    assert!(offset_of!(lmnk_rank1_descriptor_t, v_vector_ptr) == 16);

    assert!(size_of::<lmnk_bloom_filter_t>() == 64 && align_of::<lmnk_bloom_filter_t>() == 64);
    assert!(
        size_of::<lmnk_reconstitution_report_t>() == 16
            && align_of::<lmnk_reconstitution_report_t>() == 16
    );
    assert!(size_of::<lmnk_causal_report_t>() == 16 && align_of::<lmnk_causal_report_t>() == 16);
    assert!(offset_of!(lmnk_causal_report_t, lineage_flags) == 0);
    assert!(offset_of!(lmnk_causal_report_t, fact_persona_coherence) == 4);
    assert!(offset_of!(lmnk_causal_report_t, anomaly_severity) == 8);
    assert!(offset_of!(lmnk_causal_report_t, attribution_tag) == 12);

    assert!(size_of::<lmnk_gateway_frame_t>() == 128 && align_of::<lmnk_gateway_frame_t>() == 64);
    assert!(offset_of!(lmnk_gateway_frame_t, tagged_ptr) == 0);
    assert!(offset_of!(lmnk_gateway_frame_t, mapped_section_offset48) == 8);
    assert!(offset_of!(lmnk_gateway_frame_t, target_anchor_id) == 16);
    assert!(offset_of!(lmnk_gateway_frame_t, effective_energy) == 20);
    assert!(offset_of!(lmnk_gateway_frame_t, phase_clamp_rad) == 24);
    assert!(offset_of!(lmnk_gateway_frame_t, phase_step_min_rad) == 28);
    assert!(offset_of!(lmnk_gateway_frame_t, subhead_fact_gain_q7) == 56);
    assert!(offset_of!(lmnk_gateway_frame_t, mean_phase_tag_q7) == 59);
    assert!(offset_of!(lmnk_gateway_frame_t, active_head_count) == 60);
    assert!(offset_of!(lmnk_gateway_frame_t, frame_flags) == 62);
    assert!(offset_of!(lmnk_gateway_frame_t, phase_deltas) == 64);

    assert!(
        size_of::<lmnk_phase_anchor_record_t>() == 128
            && align_of::<lmnk_phase_anchor_record_t>() == 64
    );
    assert!(offset_of!(lmnk_phase_anchor_record_t, magic) == 0);
    assert!(offset_of!(lmnk_phase_anchor_record_t, version) == 4);
    assert!(offset_of!(lmnk_phase_anchor_record_t, anchor_id) == 8);
    assert!(offset_of!(lmnk_phase_anchor_record_t, sample_count) == 12);
    assert!(offset_of!(lmnk_phase_anchor_record_t, mapped_section_offset48) == 16);
    assert!(offset_of!(lmnk_phase_anchor_record_t, phase_clamp_rad) == 40);
    assert!(offset_of!(lmnk_phase_anchor_record_t, lineage_flags) == 48);
    assert!(offset_of!(lmnk_phase_anchor_record_t, checksum_crc32) == 52);
    assert!(offset_of!(lmnk_phase_anchor_record_t, dominant_slot) == 56);
    assert!(offset_of!(lmnk_phase_anchor_record_t, attribution_tag) == 59);
    assert!(offset_of!(lmnk_phase_anchor_record_t, reserved_align64) == 60);
    assert!(offset_of!(lmnk_phase_anchor_record_t, phase_profile) == 64);
    assert!(LMNK_FOUR_SECTION_HEADER_BYTES == 192);
    assert!(LMNK_STORAGE_PREAMBLE_OFFSET == 128);
    assert!(PhaseAnchorRecord::new_empty(0, ATTRACTOR_SLOT_PERSONA, 0).verify_integrity());
    assert!(PhaseAnchorRecord::new_empty(0, ATTRACTOR_SLOT_KNOWLEDGE, 0).verify_integrity());

    assert!(size_of::<CognitiveStepOp>() == 1 && align_of::<CognitiveStepOp>() == 1);
    assert!(
        size_of::<CognitiveCircuitActivation>() == 8
            && align_of::<CognitiveCircuitActivation>() == 8
    );
    assert!(
        size_of::<lmnk_cognitive_ledger_t>() == LEDGER_HISTORY_LEN
            && align_of::<lmnk_cognitive_ledger_t>() == 8
    );
    assert!(size_of::<RamPhaseKvEntry>() == 192 && align_of::<RamPhaseKvEntry>() == 64);
    assert!(offset_of!(RamPhaseKvEntry, keyframe) == 0);
    assert!(offset_of!(RamPhaseKvEntry, circuit) == 128);

    assert!(
        size_of::<lmnk_epistemic_horizon_t>() == 32 && align_of::<lmnk_epistemic_horizon_t>() == 16
    );
    assert!(offset_of!(lmnk_epistemic_horizon_t, horizon_gate) == 12);
    assert!(offset_of!(lmnk_epistemic_horizon_t, regime_code) == 28);

    assert!(
        size_of::<lmnk_liminal_threshold_state_t>() == 32
            && align_of::<lmnk_liminal_threshold_state_t>() == 16
    );
    assert!(offset_of!(lmnk_liminal_threshold_state_t, active_aperture_mask) == 24);
    assert!(offset_of!(lmnk_liminal_threshold_state_t, adaptation_epochs) == 28);

    assert!(
        size_of::<lmnk_topology_snapshot_t>() == 32 && align_of::<lmnk_topology_snapshot_t>() == 16
    );
    assert!(offset_of!(lmnk_topology_snapshot_t, active_aperture_mask) == 24);
    assert!(offset_of!(lmnk_topology_snapshot_t, spawned_daughters) == 31);

    assert!(
        size_of::<lmnk_overlap_discovery_report_t>() == 32
            && align_of::<lmnk_overlap_discovery_report_t>() == 16
    );
    assert!(offset_of!(lmnk_overlap_discovery_report_t, dominant_overlap_anchor_id) == 12);
    assert!(offset_of!(lmnk_overlap_discovery_report_t, discovery_seed_word) == 16);
    assert!(offset_of!(lmnk_overlap_discovery_report_t, overlap_core_mask) == 24);
    assert!(offset_of!(lmnk_overlap_discovery_report_t, discovery_triggered) == 30);

    assert!(
        size_of::<lmnk_stigmergic_granule_state_t>() == 32
            && align_of::<lmnk_stigmergic_granule_state_t>() == 16
    );
    assert!(offset_of!(lmnk_stigmergic_granule_state_t, overlap_mask) == 24);
    assert!(offset_of!(lmnk_stigmergic_granule_state_t, adaptation_epochs) == 28);

    assert!(
        size_of::<lmnk_core_state_stack_t>() == 128 && align_of::<lmnk_core_state_stack_t>() == 64
    );
    assert!(offset_of!(lmnk_core_state_stack_t, liminal) == 0);
    assert!(offset_of!(lmnk_core_state_stack_t, stigmergy) == 32);
    assert!(offset_of!(lmnk_core_state_stack_t, ego) == 64);

    assert!(size_of::<lmnk_boundary_delta_t>() == 16 && align_of::<lmnk_boundary_delta_t>() == 16);
    assert!(offset_of!(lmnk_boundary_delta_t, kinetic_energy_q7) == 10);
    assert!(offset_of!(lmnk_boundary_delta_t, stationary_shortcut_ready) == 11);

    assert!(
        size_of::<lmnk_ego_intent_axes_t>() == 16 && align_of::<lmnk_ego_intent_axes_t>() == 16
    );
    assert!(offset_of!(lmnk_ego_intent_axes_t, preferred_slot_mask) == 8);
    assert!(offset_of!(lmnk_ego_intent_axes_t, active_granule_bias_mask) == 9);
    assert!(offset_of!(lmnk_ego_intent_axes_t, flags) == 11);
    assert!(offset_of!(lmnk_ego_intent_axes_t, signature_hash) == 12);

    assert!(
        size_of::<lmnk_fragment_ensemble_report_t>() == 16
            && align_of::<lmnk_fragment_ensemble_report_t>() == 16
    );
    assert!(offset_of!(lmnk_fragment_ensemble_report_t, quorum_agreed_granule_mask) == 8);
    assert!(offset_of!(lmnk_fragment_ensemble_report_t, ego_tiebreak_granule_mask) == 10);

    assert!(
        size_of::<lmnk_liminal_ego_state_t>() == 64 && align_of::<lmnk_liminal_ego_state_t>() == 64
    );
    assert!(offset_of!(lmnk_liminal_ego_state_t, counter_basin_thrust_q7) == 40);
    assert!(offset_of!(lmnk_liminal_ego_state_t, basin_escape_floor_q7) == 41);
    assert!(offset_of!(lmnk_liminal_ego_state_t, trigger_mask) == 44);
    assert!(offset_of!(lmnk_liminal_ego_state_t, ego_signature_hash) == 48);
    assert!(offset_of!(lmnk_liminal_ego_state_t, last_intervened_mask) == 56);

    assert!(
        size_of::<lmnk_core_primitive_state_t>() == 64
            && align_of::<lmnk_core_primitive_state_t>() == 64
    );
    assert!(offset_of!(lmnk_core_primitive_state_t, liminal) == 0);
    assert!(offset_of!(lmnk_core_primitive_state_t, stigmergy) == 32);

    assert!(
        size_of::<lmnk_lacuna_partition_t>() == 16 && align_of::<lmnk_lacuna_partition_t>() == 16
    );
    assert!(offset_of!(lmnk_lacuna_partition_t, self_ref_mask) == 2);
    assert!(offset_of!(lmnk_lacuna_partition_t, interior_mask) == 3);
    assert!(offset_of!(lmnk_lacuna_partition_t, exterior_mask) == 4);
    assert!(offset_of!(lmnk_lacuna_partition_t, unobserved_mask) == 5);
    assert!(offset_of!(lmnk_lacuna_partition_t, undefined_mask) == 6);
    assert!(offset_of!(lmnk_lacuna_partition_t, boundary_sensitivity_q7) == 8);
    assert!(offset_of!(lmnk_lacuna_partition_t, packed_lacuna_u16) == 12);
    assert!(offset_of!(lmnk_lacuna_partition_t, dominant_lacuna) == 14);
    assert!(offset_of!(lmnk_lacuna_partition_t, attribution_tag) == 15);

    assert!(size_of::<lmnk_spine_state_t>() == 32 && align_of::<lmnk_spine_state_t>() == 16);
    assert!(offset_of!(lmnk_spine_state_t, pinned_aperture_mask) == 0);
    assert!(offset_of!(lmnk_spine_state_t, pinned_slot_mask) == 4);
    assert!(offset_of!(lmnk_spine_state_t, pinned_action_mode_mask) == 5);
    assert!(offset_of!(lmnk_spine_state_t, unobserved_locked) == 6);
    assert!(offset_of!(lmnk_spine_state_t, irreversible_step_lock) == 7);
    assert!(offset_of!(lmnk_spine_state_t, pinned_compute_mask) == 8);
    assert!(offset_of!(lmnk_spine_state_t, whole_token_bypass_active) == 24);

    assert!(
        size_of::<lmnk_distortion_delta_t>() == 128 && align_of::<lmnk_distortion_delta_t>() == 64
    );
    assert!(offset_of!(lmnk_distortion_delta_t, node_id) == 0);
    assert!(offset_of!(lmnk_distortion_delta_t, next_hop_idx) == 8);
    assert!(offset_of!(lmnk_distortion_delta_t, intent_delta) == 16);
    assert!(offset_of!(lmnk_distortion_delta_t, flags) == 44);
    assert!(offset_of!(lmnk_distortion_delta_t, delta_phase_q7) == 64);

    assert!(
        size_of::<lmnk_distortion_tension_t>() == 32
            && align_of::<lmnk_distortion_tension_t>() == 16
    );
    assert!(offset_of!(lmnk_distortion_tension_t, tension_q7) == 0);
    assert!(offset_of!(lmnk_distortion_tension_t, effective_ego_scale_q7) == 3);
    assert!(offset_of!(lmnk_distortion_tension_t, state_code) == 7);
    assert!(offset_of!(lmnk_distortion_tension_t, immediate_damping_events) == 8);
    assert!(offset_of!(lmnk_distortion_tension_t, slow_excitation_events) == 12);
    assert!(offset_of!(lmnk_distortion_tension_t, step_epochs) == 24);

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
