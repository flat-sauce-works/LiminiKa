// name: src/core/src/runtime/abi/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! C-ABI interface definitions, FFI safety barriers, binary layout contracts, pointer-driven
//! orchestration entry points, and handle management for the Geometric Cellular Sheaf Orchestrator.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(non_camel_case_types)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::inline_always)]
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

#[cfg(feature = "std")]
extern crate std;

use alloc::boxed::Box;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::{
    __m256i, _mm256_castsi256_pd, _mm256_cmpeq_epi64, _mm256_load_si256, _mm256_loadu_si256,
    _mm256_movemask_pd, _mm256_or_si256, _mm256_setzero_si256, _mm256_store_si256,
};
use core::ffi::{c_char, CStr};
use core::mem::{align_of, offset_of, size_of};
use core::ptr;
use core::slice;

#[allow(unused_imports)]
use crate::drivers::{
    eval_barrier_pitchfork, DistortionDelta, DistortionTension, EgoDifferentialNet, EgoIntentAxes,
    EntropyController, LiminalEgoField, LiminalEgoPreset, SystemSpineState,
    MAX_DISTORTION_CHAIN_HOPS, MAX_EGO_PFRAME_NODES,
};
#[allow(unused_imports)]
use crate::foundation::{
    branchless_atan2_to_q7_fast, branchless_clamp_f32, branchless_crc32_ieee,
    branchless_flush_subnormal_f32, branchless_is_finite_f32, branchless_min_f32,
    branchless_project_f32_slice_to_q7_64, branchless_q7_add_sat, branchless_q7_neg_sat,
    branchless_rad_to_phase_step_u8, branchless_relu_f32, branchless_splitmix64,
    branchless_sqrt_f32, dispatch_single_pass, eval_rank1, eval_rank1_fused, is_ptr_aligned_to,
    load_q7_words_64, pack_action_slot_tag, project_bytes_swar_to_words_64,
    project_f32_slice_to_q7_64, project_f32_slice_to_words_64, project_lora,
    raw_byte_ranges_overlap, steer_from_swar_words_core_safe, store_q7_words_64, sum_sq_f32_simd8,
    swar_q7_phase_step_filter_words_64, swar_words_nonzero_64, unpack_action_mode_from_tag,
    unpack_attractor_slot_from_tag, AlignedSlice16, AlignedSlice32, AlignedSliceMut16,
    AlignedSliceMut32, DisjointMutPair, DisjointRefMut, PhaseSteering, PhaseSteeringGeometryExt,
    Pod, RawPhaseProjectable, RopeGeometryDescriptor, RopePairLayout,
    ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_ID_MASK, DEFAULT_PHASE_CLAMP_MAX_RAD,
    PAGED_BITMASK_BITS,
};
#[allow(unused_imports)]
use crate::liminal::{
    branchless_select_f32, branchless_select_u32, branchless_select_u8,
    evaluate_epistemic_partition_branchless, EpistemicPartition, GcsoResult, LiminalThresholdField,
    MAX_ATTENTION_HEADS,
};
#[allow(unused_imports)]
use crate::runtime::boundary_delta::LiminalBoundaryDelta;
#[allow(unused_imports)]
use crate::runtime::context::{
    init_default_config, query_capability, validate_config_invariants, AutonomousCoreLayerStack,
};
#[allow(unused_imports)]
use crate::runtime::gateway::{
    accumulate_trail_and_steer_gateway_words_core,
    synthesize_multi_origin_gateway_frame_with_words, CausalProvenanceReport,
    CausalProvenanceSignal, CognitiveGroundingSignal, ContradictionGuard, ContradictionGuardSignal,
    GatewayFrame, GatewayPhaseSteeringExt, ManifoldGatewaySignal, ObjectiveFact, SubjectivePersona,
    SystemConstraint,
};
#[allow(unused_imports)]
use crate::runtime::horizon::{evaluate_epistemic_horizon_words_64, EpistemicHorizonReading};
#[allow(unused_imports)]
use crate::runtime::storage::{
    finalize_snapshot_header_crc32, init_descriptor_header, init_snapshot_header,
    validate_descriptor_header, validate_snapshot_header, verify_snapshot_payload_crc32,
    GcsoFourSectionView, GCSO_FOUR_SECTION_HEADER_BYTES,
};
#[allow(unused_imports)]
use crate::runtime::telemetry::LiminalTopologyTelemetrySnapshot;
#[allow(unused_imports)]
use crate::stigmergy::{
    evaluate_stigmergic_overlap_and_serendipity_words_64, trail_phase_words_mut,
    trail_phase_words_ref, ActionHubEngine, ActionOrchestrationReport, CognitiveTrajectoryLedger,
    ContrastivePhaseBloomFilter, CrystallizedAttractorRecord, FastPathLedger, PhaseSteeringEngine,
    StigmergicFragmentEnsembleReport, StigmergicGranuleField, StigmergicOverlapSerendipityReport,
    StigmergicReconstitutionReport, StigmergicThresholds, ATTRACTOR_SLOT_REPULSION,
    CAUSAL_AXIS_CONTRADICTION_GUARD, CAUSAL_CAUSE_COHOMOLOGY_REPULSION, MAX_ATTRACTOR_PULL_FORCE,
    NUM_STIGMERGIC_GRANULES, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE,
    TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_FLAG_SYNERGY_RESONANCE,
};

/// Helper macro to catch unwinding panics and safely bridge execution into C-ABI error status codes.
#[macro_export]
macro_rules! ffi_boundary {
    ($body:expr) => {{
        let run_ffi_body =
            || -> Result<$crate::runtime::abi::GcsoStatus, $crate::runtime::abi::GcsoStatus> {
                $body
            };

        #[cfg(feature = "std")]
        let status = match ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(run_ffi_body))
        {
            Ok(Ok(status)) => status,
            Ok(Err(err_status)) => err_status,
            Err(_) => $crate::runtime::abi::GCSO_ERROR_PANIC_CAUGHT,
        };

        #[cfg(not(feature = "std"))]
        let status = match run_ffi_body() {
            Ok(status) => status,
            Err(err_status) => err_status,
        };

        status
    }};
}

macro_rules! impl_stateless_handle_destroy {
    ($(($fn_name:ident, $handle_ty:ty, $doc:expr, $safety_doc:expr)),* $(,)?) => {
        $(
            #[doc = $doc]
            ///
            /// # Safety
            #[doc = $safety_doc]
            #[no_mangle]
            pub unsafe extern "C" fn $fn_name(handle: $handle_ty) -> GcsoStatus {
                if handle.is_null() {
                    return GCSO_SUCCESS;
                }
                if !is_aligned(handle) {
                    return GCSO_ERROR_MISALIGNED_POINTER;
                }
                ffi_boundary!(Ok(GCSO_SUCCESS))
            }
        )*
    };
}

pub mod handles;
pub use handles::*;

pub mod types;
pub use types::*;

mod dispatch;
use dispatch::*;

pub mod runtime;
pub use runtime::*;

pub mod action_hub;
pub use action_hub::*;

pub mod phase_steering;
pub use phase_steering::*;

pub mod entropy_attractor;
pub use entropy_attractor::*;

pub mod storage;
pub use storage::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::{
        AlignedBuffer128, AlignedBuffer16, AlignedBuffer32, ACTION_PRESET_EXPRESS_PHASE,
        ACTION_PRESET_FULL_ORCHESTRATION, ACTION_PRESET_INERTIAL_SKIP,
        ACTION_PRESET_PASSIVE_SIDECAR, ACTION_PRESET_REPULSION_ESCAPE,
    };
    use crate::stigmergy::{
        StigmergicActionDecision, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA,
        ATTRACTOR_SLOT_REPULSION, DEFAULT_STIGMERGIC_QUORUM_THRESHOLD,
        DEFAULT_STIGMERGIC_TOLERANCE_Q7, DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
    };

    #[test]
    fn test_version_status_and_config_defaults() {
        let mut major = 0u32;
        let mut minor = 0u32;
        let mut patch = 0u32;
        let mut cap_flags = 0u64;
        let mut config = gcso_config_t::default();

        // SAFETY: Local stack variables satisfy alignment and bounds invariants.
        unsafe {
            assert_eq!(
                gcso_abi_get_version(&mut major, &mut minor, &mut patch),
                GCSO_SUCCESS
            );
            assert_eq!(gcso_abi_query_capability(&mut cap_flags), GCSO_SUCCESS);
            assert_eq!(gcso_config_init_default(&mut config), GCSO_SUCCESS);
            assert_eq!(gcso_config_validate(&config), GCSO_SUCCESS);
            let c_str = CStr::from_ptr(gcso_status_to_string(GCSO_SUCCESS));
            assert_eq!(c_str.to_str().unwrap(), "GCSO_SUCCESS");
        }
        assert_eq!(
            (major, minor, patch),
            (
                GCSO_ABI_VERSION_MAJOR,
                GCSO_ABI_VERSION_MINOR,
                GCSO_ABI_VERSION_PATCH
            )
        );
        assert_ne!(cap_flags & GCSO_CAP_CPU_BASELINE, 0);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.num_heads, 32);
    }

    #[test]
    fn test_tagged_ptr_and_action_mode_evolution() {
        let raw_addr = 0x0000_1234_5678_9ABCu64;
        let action_ptr = gcso_tagged_ptr_pack_action(
            raw_addr,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_INERTIAL_SKIP,
            24,
        );
        assert_eq!(gcso_tagged_ptr_unpack_addr(action_ptr), raw_addr);
        assert_eq!(
            gcso_tagged_ptr_unpack_attractor_slot(action_ptr),
            ATTRACTOR_SLOT_PERSONA
        );
        assert_eq!(
            gcso_tagged_ptr_unpack_action_mode(action_ptr),
            ACTION_PRESET_INERTIAL_SKIP
        );

        let evolved_full = gcso_tagged_ptr_evolve_mode_stigmergically(
            action_ptr,
            2.0,
            0.45,
            0,
            ATTRACTOR_SLOT_KNOWLEDGE,
            0.15,
            1.50,
        );
        assert_eq!(
            gcso_tagged_ptr_unpack_attractor_slot(evolved_full),
            ATTRACTOR_SLOT_KNOWLEDGE
        );
        assert_eq!(
            gcso_tagged_ptr_unpack_action_mode(evolved_full),
            ACTION_PRESET_FULL_ORCHESTRATION
        );

        let evolved_rep = gcso_tagged_ptr_evolve_mode_stigmergically(
            evolved_full,
            0.5,
            0.50,
            1,
            ATTRACTOR_SLOT_KNOWLEDGE,
            0.15,
            1.50,
        );
        assert_eq!(
            gcso_tagged_ptr_unpack_attractor_slot(evolved_rep),
            ATTRACTOR_SLOT_REPULSION
        );
        assert_eq!(
            gcso_tagged_ptr_unpack_action_mode(evolved_rep) & ACTION_PRESET_REPULSION_ESCAPE,
            ACTION_PRESET_REPULSION_ESCAPE
        );
        assert_eq!(gcso_tagged_ptr_unpack_phase(evolved_rep), -24);
    }

    #[test]
    fn test_srl_pspm_l2p_bitmask_and_pitchfork_abi() {
        let mut barrier = 0.0f32;
        let x_in = AlignedBuffer32::new([1.0f32, 2.0, 3.0, 4.0]);
        let mut y_out = AlignedBuffer32::new([0.0f32; 4]);
        let u_vec = AlignedBuffer32::new([2.0f32; 4]);
        let v_vec = AlignedBuffer32::new([1.0f32; 4]);
        let desc = gcso_rank1_residual_descriptor_t {
            layer_idx: 0,
            rank: 1,
            u_vector_ptr: u_vec.as_ptr() as u64,
            v_vector_ptr: v_vec.as_ptr() as u64,
            gain_scalar_ptr: 0,
            scale_factor: 1.0,
            flags: 0,
            reserved: [0; 24],
        };

        let mut kv_words = [0u64; 32];
        kv_words[0] = 1;
        kv_words[3] = 42;
        kv_words[15] = 7;
        kv_words[19] = 99;
        let mut warp_mask = gcso_paged_bitmask_t::default();
        let masks_in = [
            gcso_paged_bitmask_t {
                bits: [0b0101, 0, 0, 0],
            },
            gcso_paged_bitmask_t {
                bits: [0b1010, 1, 0, 0],
            },
        ];
        let mut reduced_mask = gcso_paged_bitmask_t::default();

        // SAFETY: Local stack buffers satisfy alignment and disjointness invariants.
        unsafe {
            assert_eq!(
                gcso_liminal_void_eval_barrier_pitchfork(2.0, 1.0, 1.0, 1.0, 128, &mut barrier),
                GCSO_SUCCESS
            );
            assert!((barrier - (1.0 / 512.0)).abs() < 1e-6);
            assert_eq!(
                gcso_rank1_eval(y_out.as_mut_ptr(), x_in.as_ptr(), &desc, 4, 4),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_rank1_eval_fused(y_out.as_mut_ptr(), x_in.as_ptr(), &desc, 4, 4),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_action_hub_paged_block_warp_bitmask(
                    kv_words.as_ptr(),
                    kv_words.len(),
                    &mut warp_mask,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_action_hub_reduce_bit_tree(masks_in.as_ptr(), 2, &mut reduced_mask),
                GCSO_SUCCESS
            );
        }
        assert_eq!(y_out.data, [40.0f32; 4]);
        assert_eq!(
            warp_mask.bits[0],
            (1u64 << 0) | (1u64 << 3) | (1u64 << 15) | (1u64 << 19)
        );
        assert_eq!(reduced_mask.bits[0], 0b1111);
        assert_eq!(reduced_mask.bits[1], 1);
    }

    #[test]
    fn test_action_hub_swarm_resolution_abi() {
        let anchors_flat = [[-96i8; MAX_ATTENTION_HEADS], [32i8; MAX_ATTENTION_HEADS]];
        let mut src_trail = gcso_pointer_trail_t {
            current_ptr: gcso_tagged_ptr_pack_action(
                0x1000,
                ATTRACTOR_SLOT_PERSONA,
                ACTION_PRESET_EXPRESS_PHASE,
                16,
            ),
            accumulated_phase_delta: [32i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let mut swarm_rep = StigmergicReconstitutionReport::default();

        // SAFETY: Local stack buffers satisfy alignment and disjointness invariants.
        unsafe {
            assert_eq!(
                gcso_action_hub_resolve_trail_in_swarm(
                    &mut src_trail,
                    src_trail.current_ptr,
                    anchors_flat[1].as_ptr(),
                    0xFF,
                    128,
                    1,
                    DEFAULT_STIGMERGIC_TOLERANCE_Q7,
                    DEFAULT_STIGMERGIC_QUORUM_THRESHOLD,
                    0.10,
                    0.15,
                    &mut swarm_rep,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(swarm_rep.hot_path_resolved, 1);
            assert_eq!(swarm_rep.quorum_mask, 0xFF);
        }
    }

    #[test]
    fn test_gateway_toroidal_kmeans_and_four_section_container_abi() {
        let mut fact_phases = [0i8; MAX_ATTENTION_HEADS];
        let mut persona_phases = [0i8; MAX_ATTENTION_HEADS];
        let emb_vals = [1.0f32, 0.5, -0.25, 0.75, 0.125, -0.5, 0.25, 0.875];
        let bloom = ContrastivePhaseBloomFilter::new();
        let mut frame = gcso_gateway_frame_t::default();
        let mut report = gcso_causal_provenance_report_t::default();

        // SAFETY: Local stack buffers satisfy alignment and bounds invariants.
        unsafe {
            assert_eq!(
                gcso_gateway_project_bytes_to_q7_64(
                    b"LiminiKa Objective RAG Record".as_ptr(),
                    29,
                    fact_phases.as_mut_ptr(),
                ),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_gateway_project_f32_to_q7_64(
                    emb_vals.as_ptr(),
                    emb_vals.len(),
                    1.0,
                    persona_phases.as_mut_ptr(),
                ),
                GCSO_SUCCESS
            );
            fact_phases = [28i8; MAX_ATTENTION_HEADS];
            persona_phases = [32i8; MAX_ATTENTION_HEADS];

            assert_eq!(
                gcso_gateway_synthesize_multi_origin_frame(
                    0.15,
                    0.05,
                    0xFF,
                    0,
                    fact_phases.as_ptr(),
                    3.0,
                    0x0018_0000,
                    55,
                    persona_phases.as_ptr(),
                    2.0,
                    0x0004_0000,
                    12,
                    &bloom,
                    0.15,
                    4,
                    &mut frame,
                    &mut report,
                ),
                GCSO_SUCCESS
            );
        }
        assert_eq!(frame.mapped_section_offset48, 0x0018_0000);
        assert_eq!(frame.target_anchor_id, 55);
        assert_eq!(report.dominant_slot, ATTRACTOR_SLOT_KNOWLEDGE);

        let mut trail = gcso_pointer_trail_t::default();
        let mut q_buf = AlignedBuffer32::new([1.0f32; 64]);
        let mut k_buf = AlignedBuffer32::new([0.5f32; 64]);
        let mut logits = AlignedBuffer32::new([1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25]);
        let mut centroids = [
            CrystallizedAttractorRecord::default(),
            CrystallizedAttractorRecord::default(),
        ];
        let mut assigned_idx = 0usize;
        let mut assigned_coh = 0.0f32;
        let mut updated_mass = 0.0f32;
        let mut batch_last_idx = 0usize;
        let mut batch_last_id = 0u32;
        let mut batch_mean_coh = 0.0f32;
        let mut batch_spawned = 0u32;

        // SAFETY: Local stack buffers satisfy alignment and disjointness invariants.
        unsafe {
            assert_eq!(
                gcso_dpsr_steer_gateway_frame_safe(
                    &frame,
                    &mut trail,
                    64,
                    q_buf.as_mut_ptr(),
                    k_buf.as_mut_ptr(),
                    4,
                    16,
                    logits.as_mut_ptr(),
                    8,
                    &mut report,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(trail.target_anchor_id, 55);
            assert!(logits.data[0] > 1.5);

            assert_eq!(
                gcso_attractor_step_online_toroidal_kmeans(
                    centroids.as_mut_ptr(),
                    2,
                    &frame,
                    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
                    &mut trail,
                    &mut assigned_idx,
                    &mut assigned_coh,
                    &mut updated_mass,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(assigned_idx, 0);
            assert_eq!(trail.cluster_id, 55);
            assert!(updated_mass > 0.0 && centroids[0].verify_integrity());

            let batch_frames = [frame, frame];
            assert_eq!(
                gcso_attractor_step_online_toroidal_kmeans_batch(
                    centroids.as_mut_ptr(),
                    2,
                    batch_frames.as_ptr(),
                    2,
                    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
                    &mut trail,
                    &mut batch_last_idx,
                    &mut batch_last_id,
                    &mut batch_mean_coh,
                    &mut batch_spawned,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(batch_last_idx, 0);
            assert_eq!(batch_last_id, 55);
            assert_eq!(batch_spawned, 0);
            assert!(batch_mean_coh > 0.95 && centroids[0].verify_integrity());
        }

        let cand_trails = [
            gcso_pointer_trail_t {
                stigmergic_density: 1.0,
                cluster_id: 999,
                accumulated_phase_delta: [-96i8; MAX_ATTENTION_HEADS],
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 2.5,
                target_anchor_id: 55,
                cluster_id: 55,
                accumulated_phase_delta: [30i8; MAX_ATTENTION_HEADS],
                ..Default::default()
            },
        ];
        let ledger = gcso_ledger_slot_t::default();

        // Verify 4-section `.gcso` zero-copy serialization and verification C-ABI
        let reports_arr = [report];
        let mut container_buf = AlignedBuffer128::<u8, 768>::new([0u8; 768]);
        let mut req_len = 0usize;
        let mut storage_desc = gcso_mapped_storage_descriptor_t::default();
        let mut nc_out = 0usize;
        let mut nt_out = 0usize;
        let mut nr_out = 0usize;
        let mut c_ptr: *const gcso_crystallized_attractor_record_t = ptr::null();
        let mut t_ptr: *const gcso_pointer_trail_t = ptr::null();
        let mut b_ptr: *const gcso_contrastive_bloom_filter_t = ptr::null();
        let mut d_ptr: *const gcso_ledger_slot_t = ptr::null();
        let mut r_ptr: *const gcso_causal_provenance_report_t = ptr::null();

        // SAFETY: Local stack buffers satisfy 128B alignment and disjointness invariants.
        unsafe {
            assert_eq!(
                gcso_container_write_four_section_with_reports_zero_copy(
                    centroids.as_ptr(),
                    1,
                    cand_trails.as_ptr(),
                    1,
                    &bloom,
                    &ledger,
                    reports_arr.as_ptr(),
                    1,
                    1_774_900_000,
                    ptr::null_mut(),
                    &mut req_len,
                ),
                GCSO_SUCCESS
            );
            assert!(req_len <= container_buf.data.len());
            assert_eq!(
                gcso_container_write_four_section_with_reports_zero_copy(
                    centroids.as_ptr(),
                    1,
                    cand_trails.as_ptr(),
                    1,
                    &bloom,
                    &ledger,
                    reports_arr.as_ptr(),
                    1,
                    1_774_900_000,
                    container_buf.data.as_mut_ptr(),
                    &mut req_len,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_container_verify_four_section_with_reports_zero_copy(
                    container_buf.data.as_ptr(),
                    req_len,
                    &mut storage_desc,
                    &mut nc_out,
                    &mut nt_out,
                    &mut nr_out,
                    &mut c_ptr,
                    &mut t_ptr,
                    &mut b_ptr,
                    &mut d_ptr,
                    &mut r_ptr,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(storage_desc.file_size_bytes, req_len as u64);
            assert_eq!(storage_desc.flags, 4);
            assert_eq!(nc_out, 1);
            assert_eq!(nt_out, 1);
            assert_eq!(nr_out, 1);
            assert!(!c_ptr.is_null() && (*c_ptr).anchor_id == 55);
            assert!(!t_ptr.is_null() && (*t_ptr).cluster_id == 999);
            assert!(!b_ptr.is_null() && !d_ptr.is_null());
            assert!(!r_ptr.is_null() && (*r_ptr).dominant_slot == ATTRACTOR_SLOT_KNOWLEDGE);

            let mut zc_map_trail = gcso_pointer_trail_t::default();
            let mut zc_num_c = 0usize;
            let mut zc_pull = 0.0f32;
            let mut zc_dom_id = 0u32;
            let mut zc_ripa = 0.0f32;
            assert_eq!(
                gcso_container_map_and_steer_four_section_zero_copy_safe(
                    container_buf.data.as_ptr(),
                    req_len,
                    &mut zc_map_trail,
                    0.0,
                    q_buf.as_mut_ptr(),
                    k_buf.as_mut_ptr(),
                    4,
                    16,
                    &mut zc_num_c,
                    &mut zc_pull,
                    &mut zc_dom_id,
                    &mut zc_ripa,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(zc_num_c, 1);
            assert_eq!(zc_dom_id, 55);
            assert!(zc_pull > 0.0 && zc_ripa > 0.05);
            assert_eq!(zc_map_trail.cluster_id, 55);

            let mut raw_bytes_centroids = [CrystallizedAttractorRecord::default(); 2];
            let mut raw_bytes_trail = gcso_pointer_trail_t::default();
            let mut raw_bytes_frame = gcso_gateway_frame_t::default();
            let mut raw_bytes_report = gcso_causal_provenance_report_t::default();
            let mut raw_bytes_idx = 99usize;
            let mut raw_bytes_coh = 0.0f32;
            let mut raw_bytes_mass = 0.0f32;
            assert_eq!(
                gcso_gateway_project_bytes_synthesize_kmeans_and_steer_qk_safe(
                    0.10,
                    0.05,
                    0xFF,
                    0,
                    b"LiminiKa Objective RAG Fact Payload".as_ptr(),
                    35,
                    3.0,
                    0x0018_0000,
                    77,
                    b"LiminiKa Subjective Persona Belief".as_ptr(),
                    34,
                    2.0,
                    0x0004_0000,
                    12,
                    &bloom,
                    0.15,
                    4,
                    &mut raw_bytes_trail,
                    64,
                    q_buf.as_mut_ptr(),
                    k_buf.as_mut_ptr(),
                    4,
                    16,
                    logits.as_mut_ptr(),
                    8,
                    raw_bytes_centroids.as_mut_ptr(),
                    2,
                    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
                    &mut raw_bytes_frame,
                    &mut raw_bytes_report,
                    &mut raw_bytes_idx,
                    &mut raw_bytes_coh,
                    &mut raw_bytes_mass,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(raw_bytes_idx, 0);
            assert_eq!(raw_bytes_trail.cluster_id, 77);
            assert_eq!(raw_bytes_frame.target_anchor_id, 77);
            assert!(raw_bytes_mass > 0.0 && raw_bytes_centroids[0].verify_integrity());

            let mut raw_f32_centroids = [CrystallizedAttractorRecord::default(); 2];
            let mut raw_f32_trail = gcso_pointer_trail_t::default();
            let mut raw_f32_frame = gcso_gateway_frame_t::default();
            let mut raw_f32_report = gcso_causal_provenance_report_t::default();
            let mut raw_f32_idx = 99usize;
            let mut raw_f32_coh = 0.0f32;
            let mut raw_f32_mass = 0.0f32;
            assert_eq!(
                gcso_gateway_project_f32_synthesize_kmeans_and_steer_qk_safe(
                    0.10,
                    0.05,
                    0xFF,
                    0,
                    emb_vals.as_ptr(),
                    emb_vals.len(),
                    0.0,
                    3.0,
                    0x0018_0000,
                    88,
                    emb_vals.as_ptr(),
                    emb_vals.len(),
                    1.0,
                    2.0,
                    0x0004_0000,
                    12,
                    &bloom,
                    0.15,
                    4,
                    &mut raw_f32_trail,
                    64,
                    q_buf.as_mut_ptr(),
                    k_buf.as_mut_ptr(),
                    4,
                    16,
                    logits.as_mut_ptr(),
                    8,
                    raw_f32_centroids.as_mut_ptr(),
                    2,
                    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
                    &mut raw_f32_frame,
                    &mut raw_f32_report,
                    &mut raw_f32_idx,
                    &mut raw_f32_coh,
                    &mut raw_f32_mass,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(raw_f32_idx, 0);
            assert_eq!(raw_f32_trail.cluster_id, 88);
            assert_eq!(raw_f32_frame.target_anchor_id, 88);
            assert!(raw_f32_mass > 0.0 && raw_f32_centroids[0].verify_integrity());

            assert_eq!(
                gcso_mapped_storage_close_mmap(&mut storage_desc),
                GCSO_SUCCESS
            );
            assert_eq!(storage_desc.fd_handle, -1);
        }
    }

    #[test]
    fn test_pointer_dpsr_and_snapshot_crc32() {
        let passive_ptr = gcso_tagged_ptr_pack_action(
            0x1000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_PASSIVE_SIDECAR,
            32,
        );
        let active_ptr = gcso_tagged_ptr_pack_action(
            0x1000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            32,
        );
        let deltas = AlignedBuffer16::new([32i8; 16]);
        let mut q_passive = AlignedBuffer32::new([1.0f32, 0.0, 1.0, 0.0]);
        let mut q_active = AlignedBuffer32::new([1.0f32, 0.0, 1.0, 0.0]);

        let dummy_ctx = core::ptr::NonNull::<GcsoContextOpaque>::dangling().as_ptr();
        let mut req_size = 0usize;
        let mut buf = [0u8; 320];
        let mut restored_ctx: GcsoContextHandle = ptr::null_mut();

        // SAFETY: Local stack buffers satisfy alignment and bounds invariants.
        unsafe {
            assert_eq!(
                gcso_dpsr_apply_pointer_gated_phase_steering_safe(
                    passive_ptr,
                    q_passive.as_mut_ptr(),
                    deltas.as_ptr(),
                    4,
                    1,
                    DEFAULT_PHASE_CLAMP_MAX_RAD,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_dpsr_apply_pointer_gated_phase_steering_safe(
                    active_ptr,
                    q_active.as_mut_ptr(),
                    deltas.as_ptr(),
                    4,
                    1,
                    DEFAULT_PHASE_CLAMP_MAX_RAD,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_context_serialize(dummy_ctx, buf.as_mut_ptr(), &mut req_size),
                GCSO_ERROR_BUFFER_TOO_SMALL
            );
            assert_eq!(
                gcso_context_serialize(dummy_ctx, buf.as_mut_ptr(), &mut req_size),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_context_deserialize(buf.as_ptr(), buf.len(), &mut restored_ctx),
                GCSO_SUCCESS
            );
            buf[200] ^= 0xFF;
            assert_eq!(
                gcso_context_deserialize(buf.as_ptr(), buf.len(), &mut restored_ctx),
                GCSO_ERROR_CONTAINER_CORRUPTED
            );
        }
        assert_eq!(q_passive.data, [1.0f32, 0.0, 1.0, 0.0]);
        assert!(q_active.data[3] > 0.0);

        let mut hub_handle: GcsoActionHubHandle = ptr::null_mut();
        let mut thresh = gcso_stigmergic_thresholds_t::default();
        let mut orch_report = gcso_action_orchestration_report_t::default();
        let mut orch_trail = gcso_pointer_trail_t {
            stigmergic_density: 1.60,
            target_anchor_id: 99,
            accumulated_phase_delta: [24i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let mut q_orch = AlignedBuffer32::new([1.0f32; 16]);
        let mut k_orch = AlignedBuffer32::new([1.0f32; 16]);
        // SAFETY: Local stack variables satisfy alignment and disjointness invariants.
        unsafe {
            assert_eq!(gcso_action_hub_create(16, &mut hub_handle), GCSO_SUCCESS);
            assert_eq!(
                gcso_action_hub_init_default_thresholds(&mut thresh),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_action_hub_evaluate_stigmergic_thresholds(
                    &thresh,
                    &orch_trail,
                    0.30,
                    0.92,
                    -0.04,
                    0,
                    0,
                    ATTRACTOR_SLOT_PERSONA,
                    &mut orch_report,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(
                orch_report.decision,
                StigmergicActionDecision::FullOrchestration as u8
            );
            assert_eq!(
                gcso_action_hub_orchestrate_stigmergic_step_safe(
                    hub_handle,
                    &thresh,
                    active_ptr,
                    0.30,
                    ATTRACTOR_SLOT_PERSONA,
                    &mut orch_trail,
                    ptr::null_mut(),
                    0,
                    q_orch.as_mut_ptr(),
                    k_orch.as_mut_ptr(),
                    4,
                    4,
                    ptr::null_mut(),
                    0,
                    &mut orch_report,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(orch_report.dominant_anchor_id, 99);
            assert!(orch_report.updated_density >= 1.50);
            assert_eq!(gcso_action_hub_destroy(hub_handle), GCSO_SUCCESS);
        }
    }

    #[test]
    fn test_epistemic_venn_core_layers_and_mermaid_abi() {
        let mut crystals = [
            CrystallizedAttractorRecord::new_empty(11, ATTRACTOR_SLOT_KNOWLEDGE, 0),
            CrystallizedAttractorRecord::new_empty(12, ATTRACTOR_SLOT_KNOWLEDGE, 0),
            CrystallizedAttractorRecord::default(),
        ];
        let f0 = GatewayFrame {
            target_anchor_id: 11,
            effective_energy: 1.5,
            dominant_slot: ATTRACTOR_SLOT_KNOWLEDGE,
            phase_deltas: [16i8; MAX_ATTENTION_HEADS],
            ..GatewayFrame::default()
        };
        let mut p1 = [16i8; MAX_ATTENTION_HEADS];
        for h in 48..56 {
            p1[h] = -100;
        }
        let f1 = GatewayFrame {
            target_anchor_id: 12,
            effective_energy: 1.4,
            dominant_slot: ATTRACTOR_SLOT_KNOWLEDGE,
            phase_deltas: p1,
            ..GatewayFrame::default()
        };
        let _ = crystals[0].absorb_gateway_frame(&f0);
        let _ = crystals[1].absorb_gateway_frame(&f1);

        let query_phases = [16i8; MAX_ATTENTION_HEADS];
        let mut drifted_phases = [16i8; MAX_ATTENTION_HEADS];
        for h in 48..56 {
            drifted_phases[h] = 100;
        }

        let mut horizon = gcso_epistemic_horizon_reading_t::default();
        let mut venn_report = gcso_stigmergic_venn_serendipity_report_t::default();
        let mut did_nucleate = 0u8;
        let mut stack = gcso_autonomous_core_layer_stack_t::default();
        let ledger = gcso_cognitive_trajectory_ledger_t::default();
        let mut mermaid_buf = [0u8; 2048];
        let mut written = 0usize;

        // SAFETY: Local stack buffers satisfy all alignment and disjointness invariants.
        unsafe {
            assert_eq!(
                gcso_epistemic_evaluate_horizon(
                    crystals.as_ptr(),
                    crystals.len(),
                    query_phases.as_ptr(),
                    0.35,
                    &mut horizon,
                ),
                GCSO_SUCCESS
            );
            assert_ne!(horizon.horizon_gate, 0.0);

            assert_eq!(
                gcso_stigmergic_evaluate_venn_and_serendipity(
                    crystals.as_mut_ptr(),
                    crystals.len(),
                    query_phases.as_ptr(),
                    drifted_phases.as_ptr(),
                    0.70,
                    16,
                    1,
                    701,
                    &mut venn_report,
                    &mut did_nucleate,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(venn_report.serendipity_triggered, 1);
            assert_eq!(did_nucleate, 1);
            assert!(crystals[2].verify_integrity());

            assert_eq!(gcso_core_layer_stack_init_default(&mut stack), GCSO_SUCCESS);
            assert_eq!(
                gcso_core_layer_stack_adapt(&mut stack, &ledger, &horizon, &venn_report),
                GCSO_SUCCESS
            );
            assert_eq!(stack.stigmergic_layer.adaptation_epochs, 1);

            assert_eq!(
                gcso_liminal_topology_format_mermaid(
                    &stack,
                    &horizon,
                    &venn_report,
                    3,
                    1,
                    mermaid_buf.as_mut_ptr(),
                    mermaid_buf.len(),
                    &mut written,
                ),
                GCSO_SUCCESS
            );
        }
        let s = core::str::from_utf8(&mermaid_buf[..written]).expect("valid utf8");
        assert!(s.contains("Core Layer 0: Liminal"));
        assert!(s.contains("Core Layer 1: StigmergicGranuleField"));
        assert!(s.contains("Core Layer 2: LiminalEgoField"));

        let mut ego_preset = gcso_liminal_ego_field_t::default();
        let mut ego_text = gcso_liminal_ego_field_t::default();
        let desc = b"Independent rebel explorer stepping beyond limits";
        let delta = gcso_liminal_boundary_delta_t::default();
        let mut frame = gcso_gateway_frame_t {
            effective_energy: 8.0,
            ..Default::default()
        };
        *frame.phase_words_mut() = [0x2424_2424_2424_2424u64; NUM_STIGMERGIC_GRANULES];
        let orig_words = *frame.phase_words_ref();

        // SAFETY: Stack-allocated buffers satisfy 64B/16B alignment and disjointness invariants.
        unsafe {
            assert_eq!(gcso_ego_field_init_preset(3, &mut ego_preset), GCSO_SUCCESS);
            assert!(ego_preset.counter_basin_thrust_q7 >= 90);
            assert_eq!(
                gcso_ego_field_derive_from_text(desc.as_ptr(), desc.len(), &mut ego_text),
                GCSO_SUCCESS
            );
            assert!(ego_text.transgression_bias_q7 > 0);
            assert_eq!(
                gcso_ego_field_intervene_frame(
                    &mut ego_text,
                    &mut stack,
                    &delta,
                    &horizon,
                    &venn_report,
                    &mut frame,
                ),
                GCSO_SUCCESS
            );
        }
        assert!(ego_text.intervention_epochs >= 1);
        assert_ne!(frame.phase_words_ref(), &orig_words);
    }

    #[test]
    fn test_opaque_handles_type_safety() {
        assert_eq!(size_of::<GcsoContextHandle>(), size_of::<*mut ()>());
        assert_eq!(size_of::<GcsoContainerHandle>(), size_of::<*mut ()>());
        assert_eq!(size_of::<GcsoActionHubHandle>(), size_of::<*mut ()>());
        assert_eq!(size_of::<GcsoLedgerSlotHandle>(), size_of::<*mut ()>());
        assert_eq!(size_of::<GcsoAttractorFieldHandle>(), size_of::<*mut ()>());
        assert_eq!(size_of::<GcsoEdbcControllerHandle>(), size_of::<*mut ()>());
        assert_eq!(size_of::<GcsoDpsrKernelHandle>(), size_of::<*mut ()>());
        assert_eq!(size_of::<GcsoSubHeadRouterHandle>(), size_of::<*mut ()>());
        assert_eq!(size_of::<GcsoRank1ResidualHandle>(), size_of::<*mut ()>());
    }

    #[test]
    fn test_ffi_handle_ref_mut_cast_and_disjoint_validation() {
        let mut config = gcso_config_t::default();
        let raw_ptr: *mut gcso_config_t = &mut config;

        // SAFETY: `raw_ptr` points to a valid stack-allocated `gcso_config_t`.
        let handle_ref = unsafe { FfiHandleRef::from_raw(raw_ptr) }.unwrap();
        let cloned_ref = handle_ref;
        assert_eq!(cloned_ref.read_copy().head_dim, 0);

        let cast_u32 = handle_ref.try_cast::<u32>().unwrap();
        assert_eq!(*cast_u32, 0);

        // SAFETY: `raw_ptr` is uniquely borrowed here.
        let mut handle_mut = unsafe { FfiHandleMut::from_raw(raw_ptr) }.unwrap();
        handle_mut.head_dim = 128;
        assert_eq!(handle_mut.into_mut().head_dim, 128);
        assert_eq!(config.head_dim, 128);

        let mut a = 1u32;
        let mut b = 2u32;
        let mut c = 3u32;
        // SAFETY: `a`, `b`, and `c` are distinct stack variables.
        unsafe {
            let (mut ha, mut hb, mut hc) =
                FfiHandleMut::from_raw_disjoint_triple(&mut a, &mut b, &mut c).unwrap();
            ha.write(10);
            hb.write(20);
            hc.write(30);
            assert!(FfiHandleMut::from_raw_disjoint_pair(&mut a, &mut a).is_err());
        }
        assert_eq!((a, b, c), (10, 20, 30));

        let null_ptr: *const gcso_config_t = core::ptr::null();
        // SAFETY: Testing null pointer rejection and optional null handling.
        unsafe {
            assert_eq!(
                FfiHandleRef::from_raw(null_ptr),
                Err(GCSO_ERROR_NULL_POINTER)
            );
            assert!(FfiHandleRef::from_raw_opt(null_ptr).unwrap().is_none());
        }
    }

    #[test]
    fn test_pod_unaligned_bulk_io_and_incremental_crc32() {
        let hdr = init_descriptor_header::<gcso_config_t>();
        assert_eq!(validate_descriptor_header::<gcso_config_t>(&hdr), Ok(()));

        let mut buf = [0u8; 32];
        let src_arr = [hdr, hdr];
        assert_eq!(
            gcso_descriptor_header_t::write_unaligned_slice_to(&src_arr, &mut buf, 3),
            Ok(())
        );
        let mut dst_arr = [gcso_descriptor_header_t::default(); 2];
        assert_eq!(
            gcso_descriptor_header_t::read_unaligned_slice_from(&buf, 3, &mut dst_arr),
            Ok(())
        );
        assert_eq!(dst_arr, src_arr);

        let full_crc = crate::foundation::compute_crc32_ieee(b"123456789");
        assert_eq!(full_crc, 0xCBF4_3926);
        let part1 = crate::foundation::update_crc32_ieee(0, b"1234");
        let part2 = crate::foundation::update_crc32_ieee(part1, b"56789");
        assert_eq!(part2, full_crc);

        let mut container = [0u8; 256];
        container[128..137].copy_from_slice(b"123456789");
        let mut snap = init_snapshot_header(256, 1_774_900_000);
        let stamped_crc = finalize_snapshot_header_crc32(&mut snap, &container).unwrap();
        assert_eq!(stamped_crc, snap.checksum_crc32);
        assert_eq!(verify_snapshot_payload_crc32(&snap, &container), Ok(()));

        container[128] ^= 0xFF;
        assert_eq!(
            verify_snapshot_payload_crc32(&snap, &container),
            Err(GCSO_ERROR_CONTAINER_CORRUPTED)
        );
    }
}
