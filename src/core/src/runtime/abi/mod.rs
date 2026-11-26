// name: src/core/src/runtime/abi/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! C-ABI interface definitions, FFI safety barriers, binary layout contracts, pointer-driven
//! orchestration entry points, and opaque handle management for `LiminiKa` `v0.3.0` (`0x0000_0300`).

// C-ABI typedef aliases keep their `lmnk_*_t` spelling to mirror `include/liminika/lmnk_types.h`.
#![allow(non_camel_case_types)]

#[cfg(feature = "std")]
extern crate std;

use alloc::boxed::Box;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::{
    __m256i, _mm256_castsi256_pd, _mm256_cmpeq_epi64, _mm256_load_si256, _mm256_loadu_si256,
    _mm256_movemask_pd, _mm256_or_si256, _mm256_setzero_si256, _mm256_store_si256,
};
use core::ffi::c_char;
use core::mem::{align_of, offset_of, size_of};
use core::ptr;
use core::slice;

use crate::drivers::{
    EgoDifferentialNet, LiminalEgoPreset, MAX_DISTORTION_CHAIN_HOPS, MAX_EGO_PFRAME_NODES,
    eval_barrier_pitchfork,
};
use crate::foundation::{
    ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_ID_MASK, AlignedSlice16, AlignedSlice32,
    AlignedSliceMut16, AlignedSliceMut32, DEFAULT_PHASE_CLAMP_MAX_RAD, DisjointMutPair,
    DisjointRefMut, PAGED_BITMASK_BITS, PhaseSteering, PhaseSteeringGeometryExt,
    RawPhaseProjectable, RopeGeometryDescriptor, RopePairLayout, branchless_atan2_to_q7_fast,
    branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_is_finite_f32,
    branchless_min_f32, branchless_project_f32_slice_to_q7_64, branchless_rad_to_phase_step_u8,
    branchless_relu_f32, branchless_splitmix64, branchless_sqrt_f32, dispatch_single_pass,
    eval_rank1, eval_rank1_fused, is_ptr_aligned_to, load_q7_words_64, pack_action_slot_tag,
    project_bytes_swar_to_words_64, project_f32_slice_to_q7_64, project_f32_slice_to_words_64,
    project_lora, raw_byte_ranges_overlap, steer_from_swar_words_core_safe, store_q7_words_64,
    sum_sq_f32_simd8, swar_q7_phase_step_filter_words_64, swar_words_nonzero_64,
    unpack_action_mode_from_tag, unpack_attractor_slot_from_tag, validate_raw_slice_bounds,
};
use crate::liminal::{
    LmnkResult, MAX_ATTENTION_HEADS, branchless_select_f32, branchless_select_u8,
    branchless_select_u32, evaluate_lacuna_partition_branchless,
};
use crate::runtime::context::{
    CTX_ALLOC_HEAP_BOXED, CTX_ALLOC_IN_PLACE, Config, RuntimeContext, init_default_config,
    query_capability, validate_config_invariants,
};
use crate::runtime::gateway::{
    CAUSAL_AXIS_CONTRADICTION_GUARD, CausalProvenanceSignal, CognitiveGroundingSignal,
    ContradictionGuard, ContradictionGuardSignal, ObjectiveFact, SubjectivePersona,
    SystemConstraint, accumulate_trail_and_steer_gateway_words_core,
    synthesize_multi_origin_gateway_frame_with_words,
};
use crate::runtime::horizon::evaluate_epistemic_horizon_words_64;
use crate::stigmergy::{
    ATTRACTOR_SLOT_REPULSION, FastPathLedger, MAX_ATTRACTOR_PULL_FORCE, NUM_STIGMERGIC_GRANULES,
    PhaseSteeringEngine, StigmergicSubstrate, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE,
    TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE,
    TRAIL_FLAG_SYNERGY_RESONANCE, evaluate_stigmergic_overlap_and_discovery_words_64,
    trail_phase_words_mut,
};

/// Helper macro to catch unwinding panics and safely bridge execution into C-ABI status codes.
#[macro_export]
macro_rules! ffi_boundary {
    ($body:expr) => {{
        let run_ffi_body =
            || -> Result<$crate::runtime::abi::LmnkStatus, $crate::runtime::abi::LmnkStatus> {
                $body
            };

        #[cfg(feature = "std")]
        let status = match ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(run_ffi_body))
        {
            Ok(Ok(status)) => status,
            Ok(Err(err_status)) => err_status,
            Err(_) => $crate::runtime::abi::LMNK_ERR_PANIC_CAUGHT,
        };

        #[cfg(not(feature = "std"))]
        let status = match run_ffi_body() {
            Ok(status) => status,
            Err(err_status) => err_status,
        };

        status
    }};
}

pub mod handles;
pub use handles::*;

pub mod types;
pub use types::*;

mod dispatch;
use dispatch::*;

pub mod runtime;
pub use runtime::*;

pub mod substrate;
pub use substrate::*;

pub mod phase_steering;
pub use phase_steering::*;

pub mod topology;
pub use topology::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::pod::Pod;
    use crate::foundation::{
        ACTION_PRESET_EXPRESS_PHASE, ACTION_PRESET_FULL_ORCHESTRATION, ACTION_PRESET_INERTIAL_SKIP,
        ACTION_PRESET_PASSIVE_SIDECAR, ACTION_PRESET_REPULSION_ESCAPE, AlignedBuffer16,
        AlignedBuffer32, AlignedBuffer128,
    };
    use crate::runtime::context::LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES;
    use crate::runtime::storage::{
        LmnkContainerView, init_container_header, init_descriptor_header,
        validate_descriptor_header, write_container_sections_to_bytes,
    };
    use crate::stigmergy::{
        ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA, ATTRACTOR_SLOT_REPULSION,
        DEFAULT_STIGMERGIC_QUORUM_THRESHOLD, DEFAULT_STIGMERGIC_TOLERANCE_Q7,
        DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE, StigmergicActionDecision,
    };
    use core::ffi::CStr;

    #[test]
    fn test_version_status_and_config_defaults() {
        let major = lmnk_abi_version_major();
        let minor = lmnk_abi_version_minor();
        let patch = lmnk_abi_version_patch();
        let hex = lmnk_abi_version_hex();
        let mut cap_flags = 0u64;
        let mut config = lmnk_config_t::default();

        // SAFETY: Local stack variables satisfy alignment and bounds invariants.
        unsafe {
            assert_eq!(lmnk_abi_query_capability(&mut cap_flags), LMNK_OK);
            assert_eq!(lmnk_config_init_default(&mut config), LMNK_OK);
            assert_eq!(lmnk_config_validate(&config), LMNK_OK);
            let ver_c = CStr::from_ptr(lmnk_abi_version_string());
            assert_eq!(ver_c.to_str().unwrap(), "0.3.0");
            let c_str = CStr::from_ptr(lmnk_status_to_string(LMNK_OK));
            assert_eq!(c_str.to_str().unwrap(), "LMNK_OK");
        }
        assert_eq!(
            (major, minor, patch, hex),
            (
                LMNK_ABI_VERSION_MAJOR,
                LMNK_ABI_VERSION_MINOR,
                LMNK_ABI_VERSION_PATCH,
                0x0000_0300,
            )
        );
        assert_ne!(cap_flags & LMNK_CAP_SWAR_BASELINE, 0);
        assert_ne!(cap_flags & LMNK_CAP_ZERO_ALLOC_CORE, 0);
        assert_ne!(cap_flags & LMNK_CAP_INPLACE_CONTEXT_INIT, 0);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.num_heads, 32);
    }

    #[test]
    fn test_tagged_ptr_and_action_mode_evolution() {
        let raw_addr = 0x0000_1234_5678_9ABCu64;
        let action_ptr = lmnk_tagged_ptr_pack_action(
            raw_addr,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_INERTIAL_SKIP,
            24,
        );
        assert_eq!(lmnk_tagged_ptr_unpack_addr(action_ptr), raw_addr);
        assert_eq!(
            lmnk_tagged_ptr_unpack_attractor_slot(action_ptr),
            ATTRACTOR_SLOT_PERSONA
        );
        assert_eq!(
            lmnk_tagged_ptr_unpack_action_mode(action_ptr),
            ACTION_PRESET_INERTIAL_SKIP
        );

        let evolved_full = lmnk_tagged_ptr_evolve_mode(
            action_ptr,
            2.0,
            0.45,
            0,
            ATTRACTOR_SLOT_KNOWLEDGE,
            0.15,
            1.50,
        );
        assert_eq!(
            lmnk_tagged_ptr_unpack_attractor_slot(evolved_full),
            ATTRACTOR_SLOT_KNOWLEDGE
        );
        assert_eq!(
            lmnk_tagged_ptr_unpack_action_mode(evolved_full),
            ACTION_PRESET_FULL_ORCHESTRATION
        );

        let evolved_rep = lmnk_tagged_ptr_evolve_mode(
            evolved_full,
            0.5,
            0.50,
            1,
            ATTRACTOR_SLOT_KNOWLEDGE,
            0.15,
            1.50,
        );
        assert_eq!(
            lmnk_tagged_ptr_unpack_attractor_slot(evolved_rep),
            ATTRACTOR_SLOT_REPULSION
        );
        assert_eq!(
            lmnk_tagged_ptr_unpack_action_mode(evolved_rep) & ACTION_PRESET_REPULSION_ESCAPE,
            ACTION_PRESET_REPULSION_ESCAPE
        );
        assert_eq!(lmnk_tagged_ptr_unpack_phase(evolved_rep), -24);
    }

    #[test]
    fn test_rank1_lora_bitmask_and_pitchfork_abi() {
        let mut barrier = 0.0f32;
        let x_in = AlignedBuffer32::new([1.0f32, 2.0, 3.0, 4.0]);
        let mut y_out = AlignedBuffer32::new([0.0f32; 4]);
        let u_vec = AlignedBuffer32::new([2.0f32; 4]);
        let v_vec = AlignedBuffer32::new([1.0f32; 4]);
        let desc = lmnk_rank1_descriptor_t {
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
        let mut warp_mask = lmnk_paged_bitmask_t::default();
        let masks_in = [
            lmnk_paged_bitmask_t {
                bits: [0b0101, 0, 0, 0],
            },
            lmnk_paged_bitmask_t {
                bits: [0b1010, 1, 0, 0],
            },
        ];
        let mut reduced_mask = lmnk_paged_bitmask_t::default();

        // SAFETY: Local stack buffers satisfy alignment and disjointness invariants.
        unsafe {
            assert_eq!(
                lmnk_liminal_eval_barrier_pitchfork(2.0, 1.0, 1.0, 1.0, 128, &mut barrier),
                LMNK_OK
            );
            assert!((barrier - (1.0 / 512.0)).abs() < 1e-6);
            assert_eq!(
                lmnk_rank1_eval(y_out.as_mut_ptr(), x_in.as_ptr(), &desc, 4, 4),
                LMNK_OK
            );
            assert_eq!(
                lmnk_rank1_eval_fused(y_out.as_mut_ptr(), x_in.as_ptr(), &desc, 4, 4),
                LMNK_OK
            );
            assert_eq!(
                lmnk_substrate_paged_block_warp_bitmask(
                    kv_words.as_ptr(),
                    kv_words.len(),
                    &mut warp_mask,
                ),
                LMNK_OK
            );
            assert_eq!(
                lmnk_substrate_reduce_bit_tree(masks_in.as_ptr(), 2, &mut reduced_mask),
                LMNK_OK
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
    fn test_substrate_swarm_resolution_abi() {
        let anchors_flat = [[-96i8; MAX_ATTENTION_HEADS], [32i8; MAX_ATTENTION_HEADS]];
        let mut src_trail = lmnk_pointer_trail_t {
            current_ptr: lmnk_tagged_ptr_pack_action(
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
                lmnk_substrate_resolve_trail_in_swarm(
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
                LMNK_OK
            );
            assert_eq!(swarm_rep.hot_path_resolved, 1);
            assert_eq!(swarm_rep.quorum_mask, 0xFF);
        }
    }

    #[test]
    fn test_gateway_toroidal_kmeans_and_unified_container_abi() {
        let mut fact_phases = [0i8; MAX_ATTENTION_HEADS];
        let mut persona_phases = [0i8; MAX_ATTENTION_HEADS];
        let emb_vals = [1.0f32, 0.5, -0.25, 0.75, 0.125, -0.5, 0.25, 0.875];
        let bloom = ContrastivePhaseBloomFilter::new();
        let mut frame = lmnk_gateway_frame_t::default();
        let mut report = lmnk_causal_report_t::default();

        // SAFETY: Local stack buffers satisfy alignment and bounds invariants.
        unsafe {
            assert_eq!(
                lmnk_gateway_project_bytes_to_q7_64(
                    b"LiminiKa Objective RAG Record".as_ptr(),
                    29,
                    fact_phases.as_mut_ptr(),
                ),
                LMNK_OK
            );
            assert_eq!(
                lmnk_gateway_project_f32_to_q7_64(
                    emb_vals.as_ptr(),
                    emb_vals.len(),
                    1.0,
                    persona_phases.as_mut_ptr(),
                ),
                LMNK_OK
            );
            fact_phases = [28i8; MAX_ATTENTION_HEADS];
            persona_phases = [32i8; MAX_ATTENTION_HEADS];

            assert_eq!(
                lmnk_gateway_synthesize_multi_origin_frame(
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
                LMNK_OK
            );
        }
        assert_eq!(frame.mapped_section_offset48, 0x0018_0000);
        assert_eq!(frame.target_anchor_id, 55);
        assert_eq!(report.dominant_slot, ATTRACTOR_SLOT_KNOWLEDGE);

        let mut trail = lmnk_pointer_trail_t::default();
        let mut q_buf = AlignedBuffer32::new([1.0f32; 64]);
        let mut k_buf = AlignedBuffer32::new([0.5f32; 64]);
        let mut logits = AlignedBuffer32::new([1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25]);
        let mut centroids = [PhaseAnchorRecord::default(), PhaseAnchorRecord::default()];
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
                lmnk_phase_steering_steer_gateway_frame_safe(
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
                LMNK_OK
            );
            assert_eq!(trail.target_anchor_id, 55);
            assert!(logits.data[0] > 1.5);

            assert_eq!(
                lmnk_anchor_step_online_toroidal_kmeans(
                    centroids.as_mut_ptr(),
                    2,
                    &frame,
                    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
                    &mut trail,
                    &mut assigned_idx,
                    &mut assigned_coh,
                    &mut updated_mass,
                ),
                LMNK_OK
            );
            assert_eq!(assigned_idx, 0);
            assert_eq!(trail.cluster_id, 55);
            assert!(updated_mass > 0.0 && centroids[0].verify_integrity());

            let batch_frames = [frame, frame];
            assert_eq!(
                lmnk_anchor_step_online_toroidal_kmeans_batch(
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
                LMNK_OK
            );
            assert_eq!(batch_last_idx, 0);
            assert_eq!(batch_last_id, 55);
            assert_eq!(batch_spawned, 0);
            assert!(batch_mean_coh > 0.95 && centroids[0].verify_integrity());
        }

        let cand_trails = [
            lmnk_pointer_trail_t {
                stigmergic_density: 1.0,
                cluster_id: 999,
                accumulated_phase_delta: [-96i8; MAX_ATTENTION_HEADS],
                ..Default::default()
            },
            lmnk_pointer_trail_t {
                stigmergic_density: 2.5,
                target_anchor_id: 55,
                cluster_id: 55,
                accumulated_phase_delta: [30i8; MAX_ATTENTION_HEADS],
                ..Default::default()
            },
        ];
        let ledger = lmnk_ledger_slot_t::default();

        // Verify unified `.lmnk` zero-copy serialization and verification C-ABI
        let reports_arr = [report];
        let mut container_buf = AlignedBuffer128::<u8, 768>::new([0u8; 768]);
        let mut req_len = 0usize;
        let mut nc_out = 0usize;
        let mut nt_out = 0usize;
        let mut nr_out = 0usize;
        let mut c_ptr: *const lmnk_phase_anchor_record_t = ptr::null();
        let mut t_ptr: *const lmnk_pointer_trail_t = ptr::null();
        let mut b_ptr: *const lmnk_bloom_filter_t = ptr::null();
        let mut d_ptr: *const lmnk_ledger_slot_t = ptr::null();
        let mut r_ptr: *const lmnk_causal_report_t = ptr::null();

        // SAFETY: Local stack buffers satisfy 128B alignment and disjointness invariants.
        unsafe {
            assert_eq!(
                lmnk_container_write_with_reports_zero_copy(
                    LMNK_CONTAINER_FLAG_COMPILED_CACHE,
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
                LMNK_OK
            );
            assert!(req_len <= container_buf.data.len());
            assert_eq!(
                lmnk_container_write_with_reports_zero_copy(
                    LMNK_CONTAINER_FLAG_COMPILED_CACHE,
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
                LMNK_OK
            );
            assert_eq!(
                lmnk_container_verify_with_reports_zero_copy(
                    container_buf.data.as_ptr(),
                    req_len,
                    &mut nc_out,
                    &mut nt_out,
                    &mut nr_out,
                    &mut c_ptr,
                    &mut t_ptr,
                    &mut b_ptr,
                    &mut d_ptr,
                    &mut r_ptr,
                ),
                LMNK_OK
            );
            assert_eq!(nc_out, 1);
            assert_eq!(nt_out, 1);
            assert_eq!(nr_out, 1);
            assert!(!c_ptr.is_null() && (*c_ptr).anchor_id == 55);
            assert!(!t_ptr.is_null() && (*t_ptr).cluster_id == 999);
            assert!(!b_ptr.is_null() && !d_ptr.is_null());
            assert!(!r_ptr.is_null() && (*r_ptr).dominant_slot == ATTRACTOR_SLOT_KNOWLEDGE);

            let mut zc_map_trail = lmnk_pointer_trail_t::default();
            let mut zc_num_c = 0usize;
            let mut zc_pull = 0.0f32;
            let mut zc_dom_id = 0u32;
            let mut zc_clamp_rad = 0.0f32;
            assert_eq!(
                lmnk_container_map_and_steer_zero_copy_safe(
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
                    &mut zc_clamp_rad,
                ),
                LMNK_OK
            );
            assert_eq!(zc_num_c, 1);
            assert_eq!(zc_dom_id, 55);
            assert!(zc_pull > 0.0 && zc_clamp_rad > 0.05);
            assert_eq!(zc_map_trail.cluster_id, 55);

            let mut raw_bytes_centroids = [PhaseAnchorRecord::default(); 2];
            let mut raw_bytes_trail = lmnk_pointer_trail_t::default();
            let mut raw_bytes_frame = lmnk_gateway_frame_t::default();
            let mut raw_bytes_report = lmnk_causal_report_t::default();
            let mut raw_bytes_idx = 99usize;
            let mut raw_bytes_coh = 0.0f32;
            let mut raw_bytes_mass = 0.0f32;
            assert_eq!(
                lmnk_gateway_project_bytes_synthesize_kmeans_and_steer_qk_safe(
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
                LMNK_OK
            );
            assert_eq!(raw_bytes_idx, 0);
            assert_eq!(raw_bytes_trail.cluster_id, 77);
            assert_eq!(raw_bytes_frame.target_anchor_id, 77);
            assert!(raw_bytes_mass > 0.0 && raw_bytes_centroids[0].verify_integrity());

            let mut raw_f32_centroids = [PhaseAnchorRecord::default(); 2];
            let mut raw_f32_trail = lmnk_pointer_trail_t::default();
            let mut raw_f32_frame = lmnk_gateway_frame_t::default();
            let mut raw_f32_report = lmnk_causal_report_t::default();
            let mut raw_f32_idx = 99usize;
            let mut raw_f32_coh = 0.0f32;
            let mut raw_f32_mass = 0.0f32;
            assert_eq!(
                lmnk_gateway_project_f32_synthesize_kmeans_and_steer_qk_safe(
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
                LMNK_OK
            );
            assert_eq!(raw_f32_idx, 0);
            assert_eq!(raw_f32_trail.cluster_id, 88);
            assert_eq!(raw_f32_frame.target_anchor_id, 88);
            assert!(raw_f32_mass > 0.0 && raw_f32_centroids[0].verify_integrity());
        }
    }

    #[test]
    fn test_pointer_steering_in_place_and_snapshot_crc32() {
        let passive_ptr = lmnk_tagged_ptr_pack_action(
            0x1000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_PASSIVE_SIDECAR,
            32,
        );
        let active_ptr = lmnk_tagged_ptr_pack_action(
            0x1000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            32,
        );
        let deltas = AlignedBuffer16::new([32i8; 16]);
        let mut q_passive = AlignedBuffer32::new([1.0f32, 0.0, 1.0, 0.0]);
        let mut q_active = AlignedBuffer32::new([1.0f32, 0.0, 1.0, 0.0]);

        let mut ctx_cfg = lmnk_config_t::default();
        let mut ctx: LmnkContextHandle = ptr::null_mut();
        let mut req_size = 0usize;
        let mut buf = [0u8; LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES];
        let mut restored_ctx: LmnkContextHandle = ptr::null_mut();

        // SAFETY: Local stack buffers satisfy alignment and bounds invariants.
        unsafe {
            assert_eq!(lmnk_config_init_default(&mut ctx_cfg), LMNK_OK);
            assert_eq!(lmnk_context_create(&ctx_cfg, &mut ctx), LMNK_OK);
            assert!(!ctx.is_null());
            assert_eq!(
                lmnk_phase_steering_apply_pointer_gated_safe(
                    passive_ptr,
                    q_passive.as_mut_ptr(),
                    deltas.as_ptr(),
                    4,
                    1,
                    DEFAULT_PHASE_CLAMP_MAX_RAD,
                ),
                LMNK_OK
            );
            assert_eq!(
                lmnk_phase_steering_apply_pointer_gated_safe(
                    active_ptr,
                    q_active.as_mut_ptr(),
                    deltas.as_ptr(),
                    4,
                    1,
                    DEFAULT_PHASE_CLAMP_MAX_RAD,
                ),
                LMNK_OK
            );
            assert_eq!(
                lmnk_context_serialize(ctx, buf.as_mut_ptr(), &mut req_size),
                LMNK_ERR_BUFFER_TOO_SMALL
            );
            assert_eq!(req_size, LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES);
            assert_eq!(
                lmnk_context_serialize(ctx, buf.as_mut_ptr(), &mut req_size),
                LMNK_OK
            );
            assert_eq!(
                lmnk_context_deserialize(buf.as_ptr(), buf.len(), &mut restored_ctx),
                LMNK_OK
            );
            assert!(!restored_ctx.is_null());
            assert_eq!(lmnk_context_destroy(restored_ctx), LMNK_OK);

            let mut mismatch_buf = buf;
            mismatch_buf[8] ^= 0x01;
            let mismatch_hdr_crc = crate::foundation::compute_crc32_ieee(&mismatch_buf[..60]);
            mismatch_buf[60..64].copy_from_slice(&mismatch_hdr_crc.to_le_bytes());
            let mut mismatch_ctx: LmnkContextHandle = ptr::null_mut();
            assert_eq!(
                lmnk_context_deserialize(
                    mismatch_buf.as_ptr(),
                    mismatch_buf.len(),
                    &mut mismatch_ctx
                ),
                LMNK_ERR_TOPOLOGY_MISMATCH
            );
            assert!(mismatch_ctx.is_null());

            buf[200] ^= 0xFF;
            let mut failed_restore: LmnkContextHandle = ptr::null_mut();
            assert_eq!(
                lmnk_context_deserialize(buf.as_ptr(), buf.len(), &mut failed_restore),
                LMNK_ERR_CONTAINER_CORRUPTED
            );
            assert!(failed_restore.is_null());
            assert_eq!(lmnk_context_destroy(ctx), LMNK_OK);

            // Verify topology query and caller-managed in_place full zeroization + double-deinit guard.
            let mut topo = lmnk_topology_matrix_t::default();
            assert_eq!(lmnk_topology_get_matrix(&mut topo), LMNK_OK);
            assert_eq!(topo.magic, LMNK_TOPOLOGY_MAGIC);
            assert_eq!(topo, CORE_TOPOLOGY);
            assert!(topo.verify_crc32());
            assert_eq!(
                lmnk_topology_get_matrix(ptr::null_mut()),
                LMNK_ERR_NULL_POINTER
            );

            let needed_ctx_bytes = lmnk_context_size();
            assert!(needed_ctx_bytes <= 8192);
            let mut inplace_storage = AlignedBuffer128::<u8, 8192>::new([0xA5u8; 8192]);
            assert_eq!(
                lmnk_context_init_in_place(
                    &ctx_cfg,
                    inplace_storage.data.as_mut_ptr(),
                    needed_ctx_bytes,
                ),
                LMNK_OK
            );
            let inplace_ctx = inplace_storage
                .data
                .as_mut_ptr()
                .cast::<LmnkContextOpaque>();
            assert_eq!(lmnk_context_deinit_in_place(inplace_ctx), LMNK_OK);
            assert!(
                inplace_storage.data[..needed_ctx_bytes]
                    .iter()
                    .all(|&b| b == 0)
            );
            assert_eq!(
                lmnk_context_deinit_in_place(inplace_ctx),
                LMNK_ERR_INVALID_STATE
            );
        }
        assert_eq!(q_passive.data, [1.0f32, 0.0, 1.0, 0.0]);
        assert!(q_active.data[3] > 0.0);

        let mut substrate_handle: LmnkSubstrateHandle = ptr::null_mut();
        let mut thresh = lmnk_stigmergic_thresholds_t::default();
        let mut orch_report = lmnk_substrate_step_report_t::default();
        let mut orch_trail = lmnk_pointer_trail_t {
            stigmergic_density: 1.60,
            target_anchor_id: 99,
            accumulated_phase_delta: [24i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let mut q_orch = AlignedBuffer32::new([1.0f32; 16]);
        let mut k_orch = AlignedBuffer32::new([1.0f32; 16]);
        // SAFETY: Local stack variables satisfy alignment and disjointness invariants.
        unsafe {
            assert_eq!(lmnk_substrate_create(16, &mut substrate_handle), LMNK_OK);
            assert_eq!(lmnk_substrate_init_default_thresholds(&mut thresh), LMNK_OK);
            assert_eq!(
                lmnk_substrate_evaluate_thresholds(
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
                LMNK_OK
            );
            assert_eq!(
                orch_report.decision,
                StigmergicActionDecision::FullOrchestration as u8
            );
            assert_eq!(
                lmnk_substrate_step_safe(
                    substrate_handle,
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
                LMNK_OK
            );
            assert_eq!(orch_report.dominant_anchor_id, 99);
            assert!(orch_report.updated_density >= 1.50);
            assert_eq!(lmnk_substrate_destroy(substrate_handle), LMNK_OK);
        }
    }

    #[test]
    fn test_epistemic_overlap_core_stack_and_mermaid_abi() {
        let mut anchors = [
            PhaseAnchorRecord::new_empty(11, ATTRACTOR_SLOT_KNOWLEDGE, 0),
            PhaseAnchorRecord::new_empty(12, ATTRACTOR_SLOT_KNOWLEDGE, 0),
            PhaseAnchorRecord::default(),
        ];
        let f0 = GatewayFrame {
            target_anchor_id: 11,
            effective_energy: 1.5,
            dominant_slot: ATTRACTOR_SLOT_KNOWLEDGE,
            phase_deltas: [16i8; MAX_ATTENTION_HEADS],
            ..GatewayFrame::default()
        };
        let mut p1 = [16i8; MAX_ATTENTION_HEADS];
        p1[48..56].fill(-100);
        let f1 = GatewayFrame {
            target_anchor_id: 12,
            effective_energy: 1.4,
            dominant_slot: ATTRACTOR_SLOT_KNOWLEDGE,
            phase_deltas: p1,
            ..GatewayFrame::default()
        };
        let _ = anchors[0].absorb_gateway_frame(&f0);
        let _ = anchors[1].absorb_gateway_frame(&f1);

        let query_phases = [16i8; MAX_ATTENTION_HEADS];
        let mut drifted_phases = [16i8; MAX_ATTENTION_HEADS];
        drifted_phases[48..56].fill(100);

        let mut horizon = lmnk_epistemic_horizon_t::default();
        let mut overlap_report = lmnk_overlap_discovery_report_t::default();
        let mut did_nucleate = 0u8;
        let mut stack = lmnk_core_state_stack_t::default();
        let ledger = lmnk_cognitive_ledger_t::default();
        let mut mermaid_buf = [0u8; 2048];
        let mut written = 0usize;

        // SAFETY: Local stack buffers satisfy all alignment and disjointness invariants.
        unsafe {
            assert_eq!(
                lmnk_epistemic_evaluate_horizon(
                    anchors.as_ptr(),
                    anchors.len(),
                    query_phases.as_ptr(),
                    0.35,
                    &mut horizon,
                ),
                LMNK_OK
            );
            assert_ne!(horizon.horizon_gate, 0.0);

            assert_eq!(
                lmnk_stigmergic_evaluate_overlap_and_discovery(
                    anchors.as_mut_ptr(),
                    anchors.len(),
                    query_phases.as_ptr(),
                    drifted_phases.as_ptr(),
                    0.70,
                    16,
                    1,
                    701,
                    &mut overlap_report,
                    &mut did_nucleate,
                ),
                LMNK_OK
            );
            assert_eq!(overlap_report.discovery_triggered, 1);
            assert_eq!(did_nucleate, 1);
            assert!(anchors[2].verify_integrity());

            assert_eq!(lmnk_core_stack_init_default(&mut stack), LMNK_OK);
            assert_eq!(
                lmnk_core_stack_adapt(&mut stack, &ledger, &horizon, &overlap_report),
                LMNK_OK
            );
            assert_eq!(stack.stigmergy.adaptation_epochs, 1);

            let mut needed = 0usize;
            assert_eq!(
                lmnk_topology_format_mermaid(
                    &stack,
                    &horizon,
                    &overlap_report,
                    3,
                    1,
                    ptr::null_mut(),
                    0,
                    &mut needed,
                ),
                LMNK_OK
            );
            assert!(needed > 0 && needed <= mermaid_buf.len());
            assert_eq!(
                lmnk_topology_format_mermaid(
                    &stack,
                    &horizon,
                    &overlap_report,
                    3,
                    1,
                    mermaid_buf.as_mut_ptr(),
                    mermaid_buf.len(),
                    &mut written,
                ),
                LMNK_OK
            );
            assert_eq!(written, needed);
        }
        let s = core::str::from_utf8(&mermaid_buf[..written]).expect("valid utf8");
        assert!(s.contains("Liminal"));
        assert!(s.contains("StigmergicGranuleState"));
        assert!(s.contains("LiminalEgoState"));

        let mut ego_preset = lmnk_liminal_ego_state_t::default();
        let mut ego_text = lmnk_liminal_ego_state_t::default();
        let desc = b"Independent rebel explorer stepping beyond limits";
        let delta = lmnk_boundary_delta_t::default();
        let mut frame = lmnk_gateway_frame_t {
            effective_energy: 8.0,
            ..Default::default()
        };
        *frame.phase_words_mut() = [0x2424_2424_2424_2424u64; NUM_STIGMERGIC_GRANULES];
        let orig_words = *frame.phase_words_ref();

        // SAFETY: Stack-allocated buffers satisfy 64B/16B alignment and disjointness invariants.
        unsafe {
            assert_eq!(lmnk_ego_state_init_preset(3, &mut ego_preset), LMNK_OK);
            assert!(ego_preset.counter_basin_thrust_q7 >= 90);
            assert_eq!(
                lmnk_ego_state_derive_from_text(desc.as_ptr(), desc.len(), &mut ego_text),
                LMNK_OK
            );
            assert!(ego_text.is_active());
            assert_ne!(ego_text.ego_signature_hash, 0);
            assert_eq!(
                lmnk_ego_state_intervene_frame(
                    &mut ego_text,
                    &mut stack,
                    &delta,
                    &horizon,
                    &overlap_report,
                    &mut frame,
                ),
                LMNK_OK
            );
        }
        assert!(ego_text.intervention_epochs >= 1);
        assert_ne!(frame.phase_words_ref(), &orig_words);
    }

    #[test]
    fn test_context_in_place_lifecycle_and_cross_guards() {
        #[repr(C, align(64))]
        struct AlignedCtxBuf([u8; 8192]);

        let sz = lmnk_context_size();
        let al = lmnk_context_align();
        assert_eq!(al, 64);
        assert!(sz > 0 && sz <= 8192 && sz.is_multiple_of(64));

        let mut cfg = lmnk_config_t::default();
        let mut backing = AlignedCtxBuf([0u8; 8192]);
        let mut boxed_ctx: LmnkContextHandle = ptr::null_mut();

        // SAFETY: Buffers and stack locals satisfy alignment, size, and disjointness.
        unsafe {
            assert_eq!(lmnk_config_init_default(&mut cfg), LMNK_OK);
            assert_eq!(
                lmnk_context_init_in_place(&cfg, backing.0.as_mut_ptr(), sz - 1),
                LMNK_ERR_BUFFER_TOO_SMALL
            );
            assert_eq!(
                lmnk_context_init_in_place(&cfg, backing.0.as_mut_ptr(), sz),
                LMNK_OK
            );
            let in_place_handle = backing.0.as_mut_ptr().cast::<LmnkContextOpaque>();
            // Cross-guard: boxed destroy must reject in_place handle
            assert_eq!(
                lmnk_context_destroy(in_place_handle),
                LMNK_ERR_INVALID_STATE
            );
            assert_eq!(lmnk_context_reset(in_place_handle), LMNK_OK);
            assert_eq!(lmnk_context_deinit_in_place(in_place_handle), LMNK_OK);
            // Second deinit must reject scrubbed buffer
            assert_eq!(
                lmnk_context_deinit_in_place(in_place_handle),
                LMNK_ERR_INVALID_STATE
            );

            // Reverse cross-guard: in_place deinit must reject heap-boxed handle
            assert_eq!(lmnk_context_create(&cfg, &mut boxed_ctx), LMNK_OK);
            assert_eq!(
                lmnk_context_deinit_in_place(boxed_ctx),
                LMNK_ERR_INVALID_STATE
            );
            assert_eq!(lmnk_context_destroy(boxed_ctx), LMNK_OK);
        }
    }

    #[test]
    fn test_opaque_handles_type_safety() {
        assert_eq!(size_of::<LmnkContextHandle>(), size_of::<*mut ()>());
        assert_eq!(size_of::<LmnkSubstrateHandle>(), size_of::<*mut ()>());
        assert_eq!(
            size_of::<LmnkEntropyControllerHandle>(),
            size_of::<*mut ()>()
        );
        assert_eq!(size_of::<LmnkPhaseSteeringHandle>(), size_of::<*mut ()>());
    }

    #[test]
    fn test_ffi_handle_ref_mut_cast_and_disjoint_validation() {
        let mut config = lmnk_config_t::default();
        let raw_ptr: *mut lmnk_config_t = &mut config;

        // SAFETY: `raw_ptr` points to a valid stack-allocated `lmnk_config_t`.
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

        let null_ptr: *const lmnk_config_t = core::ptr::null();
        // SAFETY: Testing null pointer rejection and optional null handling.
        unsafe {
            assert_eq!(FfiHandleRef::from_raw(null_ptr), Err(LMNK_ERR_NULL_POINTER));
            assert!(FfiHandleRef::from_raw_opt(null_ptr).unwrap().is_none());
        }
    }

    #[test]
    fn test_pod_unaligned_bulk_io_and_incremental_crc32() {
        let hdr = init_descriptor_header::<lmnk_config_t>();
        assert_eq!(validate_descriptor_header::<lmnk_config_t>(&hdr), Ok(()));

        let mut buf = [0u8; 32];
        let src_arr = [hdr, hdr];
        assert_eq!(
            lmnk_descriptor_header_t::write_unaligned_slice_to(&src_arr, &mut buf, 3),
            Ok(())
        );
        let mut dst_arr = [lmnk_descriptor_header_t::default(); 2];
        assert_eq!(
            lmnk_descriptor_header_t::read_unaligned_slice_from(&buf, 3, &mut dst_arr),
            Ok(())
        );
        assert_eq!(dst_arr, src_arr);

        let full_crc = crate::foundation::compute_crc32_ieee(b"123456789");
        assert_eq!(full_crc, 0xCBF4_3926);
        let part1 = crate::foundation::update_crc32_ieee(0, b"1234");
        let part2 = crate::foundation::update_crc32_ieee(part1, b"56789");
        assert_eq!(part2, full_crc);

        let payload = *b"123456789";
        let sections = [LmnkSectionInput {
            kind: LMNK_SECTION_RUNTIME_SNAPSHOT,
            flags: 0,
            payload: &payload,
        }];
        let mut container = [0u8; 256];
        let written = write_container_sections_to_bytes(
            LMNK_CONTAINER_FLAG_RUNTIME_SNAPSHOT,
            1_774_900_000,
            &sections,
            &mut container,
        )
        .unwrap();
        let parsed = LmnkContainerView::parse(&container[..written]).unwrap();
        assert_eq!(
            parsed.section_bytes(LMNK_SECTION_RUNTIME_SNAPSHOT),
            Some(&payload[..])
        );

        let mut hdr = init_container_header(
            LMNK_CONTAINER_FLAG_RUNTIME_SNAPSHOT,
            written as u64,
            1_774_900_000,
            1,
        );
        let stamped_crc = hdr.finalize_crc32();
        assert_eq!(hdr.header_crc32, hdr.compute_header_crc32());
        assert_eq!(hdr.total_size, written as u64);
        assert_eq!(hdr.section_count, 1);
        assert_eq!(hdr.header_crc32, stamped_crc);

        container[128] ^= 0xFF;
        assert_eq!(
            LmnkContainerView::parse(&container[..written]).err(),
            Some(LMNK_ERR_CONTAINER_CORRUPTED)
        );
    }

    #[test]
    fn test_context_spine_policy_and_step_decode_control_c_abi() {
        let mut cfg = lmnk_config_t::default();
        let mut ctx_handle: LmnkContextHandle = ptr::null_mut();
        // SAFETY: Testing C-ABI creation, spine turn policy init, and decode step execution.
        unsafe {
            assert_eq!(lmnk_config_init_default(&mut cfg), LMNK_OK);
            assert_eq!(lmnk_context_create(&cfg, &mut ctx_handle), LMNK_OK);
            assert!(!ctx_handle.is_null());

            assert_eq!(
                lmnk_context_begin_turn_spine_policy(ptr::null_mut(), 1, 0),
                LMNK_ERR_NULL_POINTER
            );
            assert_eq!(
                lmnk_context_begin_turn_spine_policy(ctx_handle, 0, 0),
                LMNK_OK
            );

            let input = lmnk_spine_decode_input_t::from_observed_entropy_f32(
                lmnk_system_pressure_reading_t::default(),
                0,
                0.35,
                8,
                false,
                true,
            );
            let mut plan = lmnk_spine_step_plan_t::default();
            assert_eq!(
                lmnk_context_step_decode_control(ctx_handle, ptr::null(), 0, &mut plan),
                LMNK_ERR_NULL_POINTER
            );
            assert_eq!(
                lmnk_context_step_decode_control(ctx_handle, &input, 0, ptr::null_mut()),
                LMNK_ERR_NULL_POINTER
            );
            assert_eq!(
                lmnk_context_step_decode_control(ctx_handle, &input, 0, &mut plan),
                LMNK_OK
            );
            assert_eq!(plan.next_observe_mode, 0);
            assert_eq!(plan.effective_observe_stride, 1);

            assert_eq!(lmnk_context_destroy(ctx_handle), LMNK_OK);
        }
    }
}
