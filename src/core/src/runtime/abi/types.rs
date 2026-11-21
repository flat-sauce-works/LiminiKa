// name: src/core/src/runtime/abi/types.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! C-ABI constants, status codes, opaque handles, `#[repr(C)]` POD structures,
//! legacy `gcso_*_t` compatibility type aliases, and compile-time layout assertions.

use super::*;

pub use crate::drivers::{
    gcso_edbc_state_t, DistortionDelta as gcso_distortion_delta_t,
    DistortionTension as gcso_distortion_tension_t, SystemSpineState as gcso_system_spine_state_t,
};
pub use crate::foundation::{
    gcso_paged_bitmask_t, gcso_rank1_residual_descriptor_t, gcso_subhead_config_t,
    gcso_tagged_ptr_t, AlignedSlice, AlignedSliceMut, DisjointMutTriple, KvMotionDeltaPacket,
};
pub use crate::liminal::{
    gcso_config_t, gcso_q7_t, gcso_status_t, validate_abi_version, AttributionTag, GcsoStatus,
    GCSO_ABI_VERSION_CSTR, GCSO_ABI_VERSION_HEX, GCSO_ABI_VERSION_MAJOR, GCSO_ABI_VERSION_MINOR,
    GCSO_ABI_VERSION_PATCH, GCSO_ABI_VERSION_STRING, GCSO_ERROR_ACTION_HUB_FULL,
    GCSO_ERROR_ATTRACTOR_NOT_FOUND, GCSO_ERROR_BUFFER_TOO_SMALL, GCSO_ERROR_CONTAINER_CORRUPTED,
    GCSO_ERROR_DPSR_PHASE_OVERFLOW, GCSO_ERROR_EDBC_SINGULARITY, GCSO_ERROR_EXTENSION_NOT_LOADED,
    GCSO_ERROR_INVALID_ARGUMENT, GCSO_ERROR_INVALID_STATE, GCSO_ERROR_IO_FAILURE,
    GCSO_ERROR_KV_KEYFRAME_SEEK_FAILED, GCSO_ERROR_LEDGER_FULL, GCSO_ERROR_MAPPED_STORAGE_FAILED,
    GCSO_ERROR_MISALIGNED_POINTER, GCSO_ERROR_NOT_IMPLEMENTED, GCSO_ERROR_NULL_POINTER,
    GCSO_ERROR_OBSTRUCTION_UNRESOLVED, GCSO_ERROR_OUT_OF_MEMORY, GCSO_ERROR_PANIC_CAUGHT,
    GCSO_ERROR_QDPS_UNDERFLOW, GCSO_ERROR_SUBHEAD_ROUTING_FAILED, GCSO_ERROR_UNKNOWN,
    GCSO_ERROR_VERSION_MISMATCH, GCSO_SUCCESS,
};
pub use crate::runtime::horizon::RamPhaseKvEntry;
pub use crate::runtime::storage::{KvKeyframePacket, GCSO_MAPPED_STORAGE_PREAMBLE_OFFSET};
pub use crate::stigmergy::{
    gcso_anchor_type_t, gcso_ledger_slot_t, gcso_pointer_trail_t, CognitiveCircuitActivation,
    CognitiveStepOp, CognitiveSubstrate, GcsoAnchorType, ATTRACTOR_SLOT_KNOWLEDGE,
    ATTRACTOR_SLOT_PERSONA, GCSO_ANCHOR_TYPE_CRYSTALLIZED, GCSO_ANCHOR_TYPE_EMBEDDING,
    GCSO_ANCHOR_TYPE_PHASE_REPULSE, GCSO_ANCHOR_TYPE_SYSTEM_PROMPT, GCSO_ANCHOR_TYPE_TOPOLOGICAL,
    LEDGER_HISTORY_LEN,
};

/// 64-bit capability flags bitmask type (`gcso_capability_flags_t`).
pub type gcso_capability_flags_t = u64;
/// Capability bitmask flag indicating baseline CPU execution support.
pub const GCSO_CAP_CPU_BASELINE: gcso_capability_flags_t = 1 << 0;
/// Capability bitmask flag indicating NVIDIA CUDA kernel backend support.
pub const GCSO_CAP_CUDA_BACKEND: gcso_capability_flags_t = 1 << 1;
/// Capability bitmask flag indicating Vulkan Compute subgroup backend support.
pub const GCSO_CAP_VULKAN_BACKEND: gcso_capability_flags_t = 1 << 2;
/// Capability bitmask flag indicating Apple Metal simdgroup backend support.
pub const GCSO_CAP_METAL_BACKEND: gcso_capability_flags_t = 1 << 3;

/// C-ABI type alias for the 128-byte 2-Origin Composite Manifold Gateway Frame (`GatewayFrame`).
pub type gcso_gateway_frame_t = GatewayFrame;
/// C-ABI type alias for the 16-byte Causal Provenance Report (`CausalProvenanceReport`).
pub type gcso_causal_provenance_report_t = CausalProvenanceReport;
/// C-ABI type alias for the 128-byte `.gcso`-persisted Crystallized Attractor Record (`CrystallizedAttractorRecord`).
pub type gcso_crystallized_attractor_record_t = CrystallizedAttractorRecord;
/// C-ABI type alias for the 64-byte Contrastive Phase Bloom Filter (`ContrastivePhaseBloomFilter`).
pub type gcso_contrastive_bloom_filter_t = ContrastivePhaseBloomFilter;
/// C-ABI type alias for the 32-byte Stigmergic Thresholds configuration (`StigmergicThresholds`).
pub type gcso_stigmergic_thresholds_t = StigmergicThresholds;
/// C-ABI type alias for the 32-byte Substrate Step Report (`SubstrateStepReport`).
pub type gcso_action_orchestration_report_t = SubstrateStepReport;
/// C-ABI type alias for the 8-bit Stigmergic Action Decision discriminant (`StigmergicActionDecision`).
pub type gcso_stigmergic_action_decision_t = u8;
/// C-ABI type alias for the 16-byte Stigmergic Reconstitution Report (`StigmergicReconstitutionReport`).
pub type gcso_reconstitution_report_t = StigmergicReconstitutionReport;
/// C-ABI type alias for the 8-byte Cognitive Trajectory Ledger (`CognitiveTrajectoryLedger`).
pub type gcso_cognitive_trajectory_ledger_t = CognitiveTrajectoryLedger;
/// C-ABI type alias for the 32-byte Epistemic Horizon Reading (`EpistemicHorizonReading`).
pub type gcso_epistemic_horizon_reading_t = EpistemicHorizonReading;
/// C-ABI type alias for the 32-byte Core Layer 0 `LiminalThresholdField`.
pub type gcso_liminal_aperture_field_t = LiminalThresholdField;
/// C-ABI type alias for the 32-byte Core Layer 1 `StigmergicGranuleField`.
pub type gcso_stigmergic_granule_field_t = StigmergicGranuleField;
/// C-ABI type alias for the 32-byte Venn-Overlap & Serendipity Report (`StigmergicOverlapSerendipityReport`).
pub type gcso_stigmergic_venn_serendipity_report_t = StigmergicOverlapSerendipityReport;
/// C-ABI type alias for the 128-byte Unified 3-Layer Autonomous Core Stack (`AutonomousCoreLayerStack`).
pub type gcso_autonomous_core_layer_stack_t = AutonomousCoreLayerStack;
/// C-ABI type alias for the 32-byte Self-Introspecting Topology Telemetry Snapshot.
pub type gcso_liminal_topology_snapshot_t = LiminalTopologyTelemetrySnapshot;
/// C-ABI type alias for the 16-byte Temporal Boundary Velocity Analyzer (`LiminalBoundaryDelta`).
pub type gcso_liminal_boundary_delta_t = LiminalBoundaryDelta;
/// C-ABI type alias for the 16-byte Normalized Intermediate Ego Intent Axes (`EgoIntentAxes`).
pub type gcso_ego_intent_axes_t = EgoIntentAxes;
/// C-ABI type alias for the 16-byte Fragmented Micro-Ensemble Report (`StigmergicFragmentEnsembleReport`).
pub type gcso_stigmergic_fragment_ensemble_report_t = StigmergicFragmentEnsembleReport;
/// C-ABI type alias for the 64-byte Wide-Aperture Subjective Ego Field (`LiminalEgoField`).
pub type gcso_liminal_ego_field_t = LiminalEgoField;
/// C-ABI type alias for the 16-byte 4-Region Branchless Venn Partition (`EpistemicPartition`).
pub type gcso_liminal_venn_partition_t = EpistemicPartition;

// ===================================================================
// Strongly-Typed Opaque Handles
// ===================================================================

macro_rules! define_opaque_handles {
    ($(($opaque:ident, $handle:ident, $c_alias:ident, $doc:expr)),* $(,)?) => {
        $(
            #[doc = $doc]
            #[repr(C)]
            pub struct $opaque {
                _private: [u8; 0],
            }

            #[doc = $doc]
            pub type $handle = *mut $opaque;

            #[doc = $doc]
            pub type $c_alias = $handle;
        )*
    };
}

define_opaque_handles!(
    (
        GcsoContextOpaque,
        GcsoContextHandle,
        gcso_context_handle_t,
        "Opaque runtime context handle (`gcso_context_handle_t`)."
    ),
    (
        GcsoContainerOpaque,
        GcsoContainerHandle,
        gcso_container_handle_t,
        "Opaque `.gcso` binary snapshot container handle (`gcso_container_handle_t`)."
    ),
    (
        GcsoActionHubOpaque,
        GcsoActionHubHandle,
        gcso_action_hub_handle_t,
        "Opaque Action Hub (Sidecar Pointer Table) handle (`gcso_action_hub_handle_t`)."
    ),
    (
        GcsoLedgerSlotOpaque,
        GcsoLedgerSlotHandle,
        gcso_ledger_slot_handle_t,
        "Opaque FastPathLedger extension scratchpad slot handle (`gcso_ledger_slot_handle_t`)."
    ),
    (
        GcsoAttractorFieldOpaque,
        GcsoAttractorFieldHandle,
        gcso_attractor_field_handle_t,
        "Opaque Macro Attractor Field handle (`gcso_attractor_field_handle_t`)."
    ),
    (
        GcsoEdbcControllerOpaque,
        GcsoEdbcControllerHandle,
        gcso_edbc_controller_handle_t,
        "Opaque EDBC entropy controller handle (`gcso_edbc_controller_handle_t`)."
    ),
    (
        GcsoDpsrKernelOpaque,
        GcsoDpsrKernelHandle,
        gcso_dpsr_kernel_handle_t,
        "Opaque DPSR phase steering kernel handle (`gcso_dpsr_kernel_handle_t`)."
    ),
    (
        GcsoSubHeadRouterOpaque,
        GcsoSubHeadRouterHandle,
        gcso_subhead_router_handle_t,
        "Opaque sub-head router sub-head group router handle (`gcso_subhead_router_handle_t`)."
    ),
    (
        GcsoRank1ResidualOpaque,
        GcsoRank1ResidualHandle,
        gcso_rank1_residual_handle_t,
        "Opaque Rank-1 residual Rank-1 adapter layer handle (`gcso_rank1_residual_handle_t`)."
    ),
);

/// C-ABI type alias for the 1-byte Attribution Tag (`AttributionTag`).
pub type gcso_attribution_tag_t = AttributionTag;
/// C-ABI type alias for the 64-byte Unified Substrate (`CognitiveSubstrate`).
pub type gcso_cognitive_substrate_t = CognitiveSubstrate;

/// Descriptor header for structure byte size and ABI version validation (8 bytes, 4-byte aligned).
#[repr(C, align(4))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct gcso_descriptor_header_t {
    /// Size of the enclosing descriptor structure in bytes.
    pub struct_size: u32,
    /// ABI version bitmask matching [`GCSO_ABI_VERSION_HEX`].
    pub abi_version: u32,
}

/// Zero-Overhead In-Memory Mapped Storage (mapped-storage) descriptor (64 bytes, 32-byte aligned, Section 5.4.G).
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct gcso_mapped_storage_descriptor_t {
    /// Base virtual memory address of the zero-copy `mmap` region.
    pub mapped_address: u64,
    /// Total mapped `.gcso` file length in bytes.
    pub file_size_bytes: u64,
    /// Direct Memory Access (DMA) ring-buffer handle for async page streaming.
    pub dma_buffer_handle: u64,
    /// Memory protection and mapping attribute flags.
    pub flags: u32,
    /// Low-level operating system file descriptor (`-1` when unmapped).
    pub fd_handle: i32,
    /// Reserved space for future storage extensions (32 bytes).
    pub reserved: [u8; 32],
}

impl Default for gcso_mapped_storage_descriptor_t {
    #[inline(always)]
    fn default() -> Self {
        Self {
            mapped_address: 0,
            file_size_bytes: 0,
            dma_buffer_handle: 0,
            flags: 0,
            fd_handle: -1,
            reserved: [0; 32],
        }
    }
}

/// Unified `.gcso` binary snapshot container header structure (128 bytes, 64-byte aligned, Section 5.4.E).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct gcso_snapshot_header_t {
    /// Format identifier magic constant (`0x4F534347` = ASCII `"GCSO"`).
    pub magic: u32,
    /// Snapshot ABI version matching [`GCSO_ABI_VERSION_HEX`].
    pub version: u32,
    /// Total binary size of container payload in bytes.
    pub total_size: u64,
    /// Byte offset to Action Hub (SPT) section.
    pub action_hub_offset: u64,
    /// Byte offset to Attractor Field section.
    pub attractor_field_offset: u64,
    /// Byte offset to DPSR kernel state section.
    pub dpsr_state_offset: u64,
    /// Byte offset to Rank-1 residual adapter state section.
    pub rank1_state_offset: u64,
    /// Byte offset to EDBC controller state section.
    pub edbc_state_offset: u64,
    /// CRC32 integrity checksum over snapshot payload.
    pub checksum_crc32: u32,
    /// Byte offset to FastPathLedger state section.
    pub ledger_slot_offset: u32,
    /// Epoch timestamp of snapshot serialization (seconds).
    pub timestamp_epoch_sec: u64,
    /// Padding to enforce exact 128-byte size and 64-byte cacheline alignment (56 bytes).
    pub reserved_padding: [u8; 56],
}

impl Default for gcso_snapshot_header_t {
    #[inline(always)]
    fn default() -> Self {
        Self {
            magic: 0,
            version: GCSO_ABI_VERSION_HEX,
            total_size: 0,
            action_hub_offset: 0,
            attractor_field_offset: 0,
            dpsr_state_offset: 0,
            rank1_state_offset: 0,
            edbc_state_offset: 0,
            checksum_crc32: 0,
            ledger_slot_offset: 0,
            timestamp_epoch_sec: 0,
            reserved_padding: [0; 56],
        }
    }
}

/// KV-keyframe Keyframe KV cache index header structure (64 bytes, 32-byte aligned, Section 5.4.G).
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct gcso_kv_keyframe_header_t {
    /// Frame classification (`0` = I-Frame Keyframe Anchor, `1` = P-Frame Phase Motion Delta).
    pub frame_type: u32,
    /// Absolute token position index in sequence.
    pub token_index: u32,
    /// Dynamic entropy-driven Group of Pictures (GOP) length between keyframes.
    pub gop_length: u32,
    /// Mean resultant vector length $\bar{R}_c \in [0, 1]$ in circular statistics.
    pub composite_vector_length: f32,
    /// Concentration parameter $\kappa_c$ for von Mises phase distribution.
    pub von_mises_kappa: f32,
    /// Scalar residual magnitude / resultant phase angle $\bar{\theta}_c$ for sparse updates.
    pub sparse_scalar_residual: f32,
    /// Byte offset to Intra Key-Frame Cache (I-Cache) payload.
    pub icache_payload_offset: u64,
    /// Byte offset to Predictive Phase Motion Cache (P-Cache) payload.
    pub pcache_payload_offset: u64,
    /// Reserved space for future KV-keyframe extensions (24 bytes).
    pub reserved: [u8; 24],
}

// Compile-time layout assertions enforcing 1:1 binary parity with `c_abi_spec.md` Section 8.1
// and the 2-Origin Gateway / `.gcso` Crystallization structures.
const _: () = {
    assert!(size_of::<gcso_status_t>() == 4);
    assert!(size_of::<gcso_q7_t>() == 1);
    assert!(size_of::<gcso_tagged_ptr_t>() == 8);
    assert!(
        size_of::<gcso_descriptor_header_t>() == 8 && align_of::<gcso_descriptor_header_t>() == 4
    );
    assert!(size_of::<gcso_paged_bitmask_t>() == 32 && align_of::<gcso_paged_bitmask_t>() == 32);
    assert!(size_of::<gcso_pointer_trail_t>() == 128 && align_of::<gcso_pointer_trail_t>() == 128);
    assert!(size_of::<gcso_ledger_slot_t>() == 64 && align_of::<gcso_ledger_slot_t>() == 64);
    assert!(
        size_of::<gcso_stigmergic_thresholds_t>() == 32
            && align_of::<gcso_stigmergic_thresholds_t>() == 32
    );
    assert!(
        size_of::<gcso_action_orchestration_report_t>() == 32
            && align_of::<gcso_action_orchestration_report_t>() == 32
    );
    assert!(size_of::<gcso_config_t>() == 64 && align_of::<gcso_config_t>() == 16);
    assert!(size_of::<gcso_edbc_state_t>() == 64 && align_of::<gcso_edbc_state_t>() == 32);
    assert!(
        size_of::<gcso_mapped_storage_descriptor_t>() == 64
            && align_of::<gcso_mapped_storage_descriptor_t>() == 32
    );
    assert!(
        size_of::<gcso_reconstitution_report_t>() == 16
            && align_of::<gcso_reconstitution_report_t>() == 16
    );
    assert!(
        size_of::<gcso_epistemic_horizon_reading_t>() == 32
            && align_of::<gcso_epistemic_horizon_reading_t>() == 16
    );
    assert!(
        size_of::<gcso_liminal_aperture_field_t>() == 32
            && align_of::<gcso_liminal_aperture_field_t>() == 16
    );
    assert!(
        size_of::<gcso_stigmergic_granule_field_t>() == 32
            && align_of::<gcso_stigmergic_granule_field_t>() == 16
    );
    assert!(
        size_of::<gcso_stigmergic_venn_serendipity_report_t>() == 32
            && align_of::<gcso_stigmergic_venn_serendipity_report_t>() == 16
    );
    assert!(
        size_of::<gcso_autonomous_core_layer_stack_t>() == 128
            && align_of::<gcso_autonomous_core_layer_stack_t>() == 64
    );
    assert!(
        size_of::<gcso_liminal_topology_snapshot_t>() == 32
            && align_of::<gcso_liminal_topology_snapshot_t>() == 16
    );
    assert!(offset_of!(gcso_autonomous_core_layer_stack_t, liminal_layer) == 0);
    assert!(offset_of!(gcso_autonomous_core_layer_stack_t, stigmergic_layer) == 32);
    assert!(offset_of!(gcso_autonomous_core_layer_stack_t, ego_layer) == 64);
    assert!(
        size_of::<gcso_liminal_boundary_delta_t>() == 16
            && align_of::<gcso_liminal_boundary_delta_t>() == 16
    );
    assert!(offset_of!(gcso_liminal_boundary_delta_t, kinetic_energy_q7) == 10);
    assert!(offset_of!(gcso_liminal_boundary_delta_t, stationary_shortcut_ready) == 11);
    assert!(
        size_of::<gcso_ego_intent_axes_t>() == 16 && align_of::<gcso_ego_intent_axes_t>() == 16
    );
    assert!(offset_of!(gcso_ego_intent_axes_t, preferred_slot_mask) == 8);
    assert!(offset_of!(gcso_ego_intent_axes_t, active_granule_bias_mask) == 9);
    assert!(offset_of!(gcso_ego_intent_axes_t, flags) == 11);
    assert!(offset_of!(gcso_ego_intent_axes_t, signature_hash) == 12);
    assert!(
        size_of::<gcso_stigmergic_fragment_ensemble_report_t>() == 16
            && align_of::<gcso_stigmergic_fragment_ensemble_report_t>() == 16
    );
    assert!(
        offset_of!(
            gcso_stigmergic_fragment_ensemble_report_t,
            quorum_agreed_granule_mask
        ) == 8
    );
    assert!(
        offset_of!(
            gcso_stigmergic_fragment_ensemble_report_t,
            ego_tiebreak_granule_mask
        ) == 10
    );
    assert!(
        size_of::<gcso_liminal_ego_field_t>() == 64 && align_of::<gcso_liminal_ego_field_t>() == 64
    );
    assert!(offset_of!(gcso_liminal_ego_field_t, counter_basin_thrust_q7) == 40);
    assert!(offset_of!(gcso_liminal_ego_field_t, basin_escape_floor_q7) == 41);
    assert!(offset_of!(gcso_liminal_ego_field_t, trigger_mask) == 44);
    assert!(offset_of!(gcso_liminal_ego_field_t, ego_signature_hash) == 48);
    assert!(offset_of!(gcso_liminal_ego_field_t, last_intervened_mask) == 56);
    assert!(size_of::<gcso_subhead_config_t>() == 32 && align_of::<gcso_subhead_config_t>() == 16);
    assert!(
        size_of::<gcso_rank1_residual_descriptor_t>() == 64
            && align_of::<gcso_rank1_residual_descriptor_t>() == 32
    );
    assert!(
        size_of::<gcso_snapshot_header_t>() == 128 && align_of::<gcso_snapshot_header_t>() == 64
    );
    assert!(
        size_of::<gcso_kv_keyframe_header_t>() == 64
            && align_of::<gcso_kv_keyframe_header_t>() == 32
    );
    assert!(
        size_of::<StigmergicReconstitutionReport>() == 16
            && align_of::<StigmergicReconstitutionReport>() == 16
    );
    assert!(size_of::<gcso_gateway_frame_t>() == 128 && align_of::<gcso_gateway_frame_t>() == 64);
    assert!(
        size_of::<gcso_causal_provenance_report_t>() == 16
            && align_of::<gcso_causal_provenance_report_t>() == 16
    );
    assert!(
        size_of::<gcso_crystallized_attractor_record_t>() == 128
            && align_of::<gcso_crystallized_attractor_record_t>() == 64
    );
    assert!(
        size_of::<gcso_contrastive_bloom_filter_t>() == 64
            && align_of::<gcso_contrastive_bloom_filter_t>() == 64
    );
    assert!(offset_of!(gcso_descriptor_header_t, struct_size) == 0);
    assert!(offset_of!(gcso_descriptor_header_t, abi_version) == 4);
    assert!(offset_of!(gcso_paged_bitmask_t, bits) == 0);
    assert!(offset_of!(gcso_pointer_trail_t, current_ptr) == 0);
    assert!(offset_of!(gcso_pointer_trail_t, prev_ptr) == 8);
    assert!(offset_of!(gcso_pointer_trail_t, user_data) == 16);
    assert!(offset_of!(gcso_pointer_trail_t, transition_cost) == 24);
    assert!(offset_of!(gcso_pointer_trail_t, step_count) == 28);
    assert!(offset_of!(gcso_pointer_trail_t, stigmergic_density) == 32);
    assert!(offset_of!(gcso_pointer_trail_t, target_anchor_id) == 36);
    assert!(offset_of!(gcso_pointer_trail_t, cluster_id) == 40);
    assert!(offset_of!(gcso_pointer_trail_t, linked_trail_id) == 44);
    assert!(offset_of!(gcso_pointer_trail_t, attractor_pull_force) == 48);
    assert!(offset_of!(gcso_pointer_trail_t, flags) == 52);
    assert!(offset_of!(gcso_pointer_trail_t, accumulated_phase_delta) == 56);
    assert!(offset_of!(gcso_pointer_trail_t, reserved_padding) == 120);
    assert!(offset_of!(gcso_config_t, head_dim) == 0);
    assert!(offset_of!(gcso_config_t, qdps_min_step_rad) == 20);
    assert!(offset_of!(gcso_config_t, action_hub_capacity) == 32);
    assert!(offset_of!(gcso_config_t, ledger_mode) == 38);
    assert!(offset_of!(gcso_snapshot_header_t, magic) == 0);
    assert!(offset_of!(gcso_snapshot_header_t, version) == 4);
    assert!(offset_of!(gcso_snapshot_header_t, total_size) == 8);
    assert!(offset_of!(gcso_snapshot_header_t, checksum_crc32) == 56);
    assert!(offset_of!(gcso_snapshot_header_t, ledger_slot_offset) == 60);
    assert!(offset_of!(gcso_snapshot_header_t, timestamp_epoch_sec) == 64);
    assert!(offset_of!(gcso_ledger_slot_t, mode) == 0);
    assert!(offset_of!(gcso_ledger_slot_t, fast_path_bypass_mask) == 16);
    assert!(offset_of!(gcso_ledger_slot_t, fast_path_shortcuts) == 24);
    assert!(offset_of!(gcso_ledger_slot_t, telemetry_mini_ledger) == 56);
    assert!(offset_of!(gcso_subhead_config_t, num_fact_heads) == 0);
    assert!(offset_of!(gcso_subhead_config_t, fact_phase_gain) == 8);
    assert!(offset_of!(gcso_edbc_state_t, moving_z_entropy) == 0);
    assert!(offset_of!(gcso_edbc_state_t, active_branch_mode) == 24);
    assert!(offset_of!(gcso_rank1_residual_descriptor_t, layer_idx) == 0);
    assert!(offset_of!(gcso_rank1_residual_descriptor_t, u_vector_ptr) == 8);
    assert!(offset_of!(gcso_rank1_residual_descriptor_t, v_vector_ptr) == 16);
    assert!(offset_of!(gcso_rank1_residual_descriptor_t, scale_factor) == 32);
    assert!(offset_of!(gcso_mapped_storage_descriptor_t, mapped_address) == 0);
    assert!(offset_of!(gcso_mapped_storage_descriptor_t, file_size_bytes) == 8);
    assert!(offset_of!(gcso_kv_keyframe_header_t, frame_type) == 0);
    assert!(offset_of!(gcso_kv_keyframe_header_t, icache_payload_offset) == 24);
    assert!(offset_of!(gcso_kv_keyframe_header_t, pcache_payload_offset) == 32);
    assert!(size_of::<gcso_attribution_tag_t>() == 1 && align_of::<gcso_attribution_tag_t>() == 1);
    assert!(
        size_of::<gcso_cognitive_substrate_t>() == 64
            && align_of::<gcso_cognitive_substrate_t>() == 64
    );
    assert!(offset_of!(gcso_cognitive_substrate_t, liminal) == 0);
    assert!(offset_of!(gcso_cognitive_substrate_t, stigmergy) == 32);
    assert!(
        size_of::<gcso_liminal_venn_partition_t>() == 16
            && align_of::<gcso_liminal_venn_partition_t>() == 16
    );
    assert!(offset_of!(gcso_liminal_venn_partition_t, known_unknown_mask) == 4);
    assert!(offset_of!(gcso_liminal_venn_partition_t, boundary_sensitivity_q7) == 8);
    assert!(
        size_of::<gcso_system_spine_state_t>() == 32
            && align_of::<gcso_system_spine_state_t>() == 16
    );
    assert!(offset_of!(gcso_system_spine_state_t, pinned_compute_mask) == 8);
    assert!(offset_of!(gcso_system_spine_state_t, whole_token_bypass_active) == 24);
    assert!(
        size_of::<gcso_distortion_delta_t>() == 128 && align_of::<gcso_distortion_delta_t>() == 64
    );
    assert!(offset_of!(gcso_distortion_delta_t, intent_delta) == 16);
    assert!(offset_of!(gcso_distortion_delta_t, delta_phase_q7) == 64);
    assert!(
        size_of::<gcso_distortion_tension_t>() == 32
            && align_of::<gcso_distortion_tension_t>() == 16
    );
    assert!(offset_of!(gcso_distortion_tension_t, immediate_damping_events) == 8);
    assert!(offset_of!(gcso_distortion_tension_t, slow_excitation_events) == 12);
};

// SAFETY: `gcso_descriptor_header_t` is `#[repr(C)]` with primitives only and zero padding traps.
unsafe impl Pod for gcso_descriptor_header_t {}
// SAFETY: `gcso_mapped_storage_descriptor_t` is `#[repr(C)]` with primitives only and zero padding traps.
unsafe impl Pod for gcso_mapped_storage_descriptor_t {}
// SAFETY: `gcso_snapshot_header_t` is `#[repr(C)]` with primitives only and zero padding traps.
unsafe impl Pod for gcso_snapshot_header_t {}
// SAFETY: `gcso_kv_keyframe_header_t` is `#[repr(C)]` with primitives only and zero padding traps.
unsafe impl Pod for gcso_kv_keyframe_header_t {}

const _: () = {
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
    assert!(size_of::<LiminalThresholdField>() == 32);
    assert!(align_of::<LiminalThresholdField>() == 16);
    assert!(offset_of!(LiminalThresholdField, active_aperture_mask) == 24);
    assert!(offset_of!(LiminalThresholdField, adaptation_epochs) == 28);
    assert!(size_of::<LiminalTopologyTelemetrySnapshot>() == 32);
    assert!(align_of::<LiminalTopologyTelemetrySnapshot>() == 16);
    assert!(offset_of!(LiminalTopologyTelemetrySnapshot, active_aperture_mask) == 24);
    assert!(offset_of!(LiminalTopologyTelemetrySnapshot, spawned_daughters) == 31);
    assert!(size_of::<StigmergicOverlapSerendipityReport>() == 32);
    assert!(align_of::<StigmergicOverlapSerendipityReport>() == 16);
    assert!(
        offset_of!(
            StigmergicOverlapSerendipityReport,
            dominant_overlap_anchor_id
        ) == 12
    );
    assert!(offset_of!(StigmergicOverlapSerendipityReport, serendipitous_seed_word) == 16);
    assert!(offset_of!(StigmergicOverlapSerendipityReport, venn_core_mask) == 24);
    assert!(offset_of!(StigmergicOverlapSerendipityReport, serendipity_triggered) == 30);
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
    assert!(offset_of!(LiminalBoundaryDelta, chatter_potential_u8) == 13);
    assert!(offset_of!(LiminalBoundaryDelta, phase_transition_pulse) == 14);
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
    assert!(
        offset_of!(
            StigmergicFragmentEnsembleReport,
            split_contested_granule_mask
        ) == 9
    );
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

    // Unified v0.1.3 2 Primitives x 2 Drivers & Distortion ABI invariants
    assert!(size_of::<CognitiveSubstrate>() == 64 && align_of::<CognitiveSubstrate>() == 64);
    assert!(offset_of!(CognitiveSubstrate, liminal) == 0);
    assert!(offset_of!(CognitiveSubstrate, stigmergy) == 32);

    assert!(size_of::<EpistemicPartition>() == 16 && align_of::<EpistemicPartition>() == 16);
    assert!(offset_of!(EpistemicPartition, shared_core_mask) == 2);
    assert!(offset_of!(EpistemicPartition, self_exclusive_mask) == 3);
    assert!(offset_of!(EpistemicPartition, known_unknown_mask) == 4);
    assert!(offset_of!(EpistemicPartition, true_void_mask) == 5);
    assert!(offset_of!(EpistemicPartition, boundary_sensitivity_q7) == 8);
    assert!(offset_of!(EpistemicPartition, sandbox_eligible) == 10);
    assert!(offset_of!(EpistemicPartition, true_void_aperture_zero) == 11);

    assert!(size_of::<SystemSpineState>() == 32 && align_of::<SystemSpineState>() == 16);
    assert!(offset_of!(SystemSpineState, pinned_aperture_mask) == 0);
    assert!(offset_of!(SystemSpineState, pinned_slot_mask) == 4);
    assert!(offset_of!(SystemSpineState, pinned_action_mode_mask) == 5);
    assert!(offset_of!(SystemSpineState, ood_void_locked) == 6);
    assert!(offset_of!(SystemSpineState, irreversible_step_lock) == 7);
    assert!(offset_of!(SystemSpineState, pinned_compute_mask) == 8);
    assert!(offset_of!(SystemSpineState, whole_token_bypass_active) == 24);

    assert!(size_of::<DistortionDelta>() == 128 && align_of::<DistortionDelta>() == 64);
    assert!(offset_of!(DistortionDelta, node_id) == 0);
    assert!(offset_of!(DistortionDelta, next_hop_idx) == 8);
    assert!(offset_of!(DistortionDelta, intent_delta) == 16);
    assert!(offset_of!(DistortionDelta, flags) == 44);
    assert!(offset_of!(DistortionDelta, delta_phase_q7) == 64);

    assert!(size_of::<DistortionTension>() == 32 && align_of::<DistortionTension>() == 16);
    assert!(offset_of!(DistortionTension, tension_q7) == 0);
    assert!(offset_of!(DistortionTension, effective_ego_scale_q7) == 3);
    assert!(offset_of!(DistortionTension, state_code) == 7);
    assert!(offset_of!(DistortionTension, immediate_damping_events) == 8);
    assert!(offset_of!(DistortionTension, slow_excitation_events) == 12);
    assert!(offset_of!(DistortionTension, step_epochs) == 24);
};
