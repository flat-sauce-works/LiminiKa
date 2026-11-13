// name: src/core/src/abi/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! C-ABI interface definitions, binary layout contracts, pointer-driven mode orchestration
//! entry points, Single Unified Attractor Manifold evaluators, 2-Origin Composite Manifold
//! Gateway FFI bridges, Classical Online Toroidal $k$-Means / DP-Means macro clustering
//! (single-frame & deferred-CRC32 batch), Coarse-to-Fine routing (centroid-only, candidate-trail,
//! & fused Bayesian + Attention score modulation), Limina inter-memory bridging, 4-section
//! `.gcso` zero-copy container persistence (with `gcso_zimms_descriptor_t` preamble &
//! `CausalProvenanceReport` history), and FFI boundary safety primitives for the Geometric
//! Cellular Sheaf Orchestrator (GCSO) core engine.
//!
//! # Architectural Role (Boundary Layer)
//! This module forms the C11/C++20 and Rust FFI boundary barrier for `LiminiKa` (ABI v0.1.1,
//! `0x00000101`). It enforces:
//! - **1:1 Binary Layout Parity**: Compile-time `size_of`, `align_of`, and `offset_of` verification
//!   matching C headers in `include/liminika/` (`c_abi_spec.md` Section 8.1).
//! - **Exception & Panic Isolation**: [`ffi_boundary!`] intercepts unwinding Rust panics
//!   (`catch_unwind`) and translates them into `GCSO_ERROR_PANIC_CAUGHT`.
//! - **Type-Driven Alignment & `GCSO_RESTRICT` Non-Aliasing**: All raw pointers crossing FFI
//!   boundaries are validated via [`AlignedSlice16`], [`AlignedSlice32`], [`AlignedSliceMut16`],
//!   [`AlignedSliceMut32`], [`DisjointMutPair`], [`DisjointRefMut`], [`FfiHandleRef`], and
//!   [`FfiHandleMut`] before entering Nano/Micro Hot Path execution.

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
use core::fmt;
use core::mem::{align_of, offset_of, size_of};
use core::ptr;
use core::slice;

use crate::action_hub::{ActionHubEngine, ActionOrchestrationReport, StigmergicThresholds};
use crate::entropy_control::{eval_barrier_pitchfork, EntropyController};
use crate::phase_steering::PhaseSteeringEngine;
use crate::traits::aligned::{
    is_ptr_aligned_to, AlignedSlice16, AlignedSlice32, AlignedSliceMut16, AlignedSliceMut32,
    DisjointMutPair, DisjointRefMut,
};
use crate::traits::branchless::{
    branchless_fast_transition_log_prob, branchless_project_f32_slice_to_q7_64,
    branchless_q7_inhibit_64, branchless_q7_resonant_pull_64, branchless_q7_synergy_fuse_64,
    branchless_q7_unified_field_fold_words_with_init_weight_64,
    modulate_attention_scores_by_phase_bank_64, sum_sq_f32_simd8, swar_slerp_words_with_bias_64,
    swar_words_equal_64, swar_words_nonzero_64,
};
use crate::traits::cold_path::{EntropyEvaluator, L2pSvdProjector, SystemCapabilityQuery};
use crate::traits::gateway::{
    project_bytes_swar_to_words_64, project_f32_slice_to_q7_64, project_f32_slice_to_words_64,
    synthesize_multi_origin_gateway_frame_with_words, CausalProvenanceReport,
    CausalProvenanceSignal, CognitiveGroundingSignal, ContradictionGuard, ContradictionGuardSignal,
    CrystallizedAttractorRecord, GatewayFrame, GcsoFourSectionView, ManifoldGatewaySignal,
    ObjectiveFact, RawPhaseProjectable, SubjectivePersona, SystemConstraint,
    CAUSAL_AXIS_CONTRADICTION_GUARD, GCSO_FOUR_SECTION_HEADER_BYTES,
};
use crate::traits::primitives::{
    branchless_atan2_to_q7_fast, branchless_clamp_f32, branchless_crc32_ieee,
    branchless_flush_subnormal_f32, branchless_is_finite_f32, branchless_min_f32,
    branchless_q7_add_sat, branchless_q7_neg_sat, branchless_rad_to_qdps_step_u8,
    branchless_relu_f32, branchless_select_f32, branchless_select_u32, branchless_select_u8,
    branchless_splitmix64, branchless_sqrt_f32, load_q7_words_64, store_q7_words_64,
    swar_q7_qdps_filter_words_64, trail_phase_words_mut, trail_phase_words_ref, SWAR_SPLAT_BYTE_64,
};
use crate::traits::primitives::{
    init_snapshot_header, is_slot_pair_coupled, pack_action_slot_tag, raw_byte_ranges_overlap,
    unpack_action_mode_from_tag, unpack_attractor_slot_from_tag, validate_snapshot_header,
    FfiHandleMut, FfiHandleRef, GcsoResult, Pod, ACTION_PRESET_REPULSION_ESCAPE,
    ATTRACTOR_SLOT_ID_MASK, ATTRACTOR_SLOT_REPULSION, DEFAULT_RIPA_CLAMP_MAX_RAD,
    MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS, PAGED_BITMASK_BITS,
};
use crate::traits::rope::{
    dispatch_fused_ripa_and_pspm_words_core, steer_from_swar_words_core_safe, PhaseSteering,
    PspmRouter, SrlAdapter,
};
use crate::traits::telemetry::{
    ContrastivePhaseBloomFilter, DaesScratchpad, PagedBitmaskOps, StigmergicActionPtrOps,
    StigmergicSwarmOps, StigmergicTrailOps, SwarmResolutionReport, DAES_METRIC_PHASE_STEERED,
    DAES_METRIC_PSPM_ROUTED, DAES_METRIC_REPULSION_APPLIED, DAES_METRIC_SLOT_SYNERGY_ACTIVE,
    MAX_ATTRACTOR_PULL_FORCE, NUM_SWARM_MICRO_CELLS, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE,
    TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE,
    TRAIL_FLAG_SYNERGY_RESONANCE,
};

/// Helper macro to catch unwinding panics and safely bridge execution into C-ABI error status codes.
#[macro_export]
macro_rules! ffi_boundary {
    ($body:expr) => {{
        let run_ffi_body = || -> Result<$crate::abi::GcsoStatus, $crate::abi::GcsoStatus> { $body };

        #[cfg(feature = "std")]
        let status = match ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(run_ffi_body))
        {
            Ok(Ok(status)) => status,
            Ok(Err(err_status)) => err_status,
            Err(_) => $crate::abi::GCSO_ERROR_PANIC_CAUGHT,
        };

        #[cfg(not(feature = "std"))]
        let status = match run_ffi_body() {
            Ok(status) => status,
            Err(err_status) => err_status,
        };

        status
    }};
}

/// Standardized 32-bit signed status code returned across the C-ABI FFI boundary.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
pub struct GcsoStatus(
    /// Inner signed 32-bit integer status code.
    pub i32,
);

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

mod types;
pub use types::*;

mod dispatch;
use dispatch::*;

mod runtime;
pub use runtime::*;

mod action_hub;
pub use action_hub::*;

mod phase_steering;
pub use phase_steering::*;

mod entropy_attractor;
pub use entropy_attractor::*;

mod storage;
pub use storage::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action_hub::StigmergicActionDecision;
    use crate::traits::aligned::{
        AlignedBuffer128, AlignedBuffer16, AlignedBuffer32, AlignedBuffer64,
    };
    use crate::traits::gateway::DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE;
    use crate::traits::primitives::{
        ACTION_PRESET_EXPRESS_PHASE, ACTION_PRESET_FULL_ORCHESTRATION, ACTION_PRESET_INERTIAL_SKIP,
        ACTION_PRESET_PASSIVE_SIDECAR, ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_KNOWLEDGE,
        ATTRACTOR_SLOT_PERSONA, ATTRACTOR_SLOT_REPULSION, DEFAULT_SLOT_COUPLING_MATRIX_8X8,
    };
    use crate::traits::telemetry::{DEFAULT_SWARM_QUORUM_THRESHOLD, DEFAULT_SWARM_TOLERANCE_Q7};

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
        assert_eq!((major, minor, patch), (0, 1, 1));
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
        let desc = gcso_srl_descriptor_t {
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
                gcso_cvoid_eval_barrier_pitchfork(2.0, 1.0, 1.0, 1.0, 128, &mut barrier),
                GCSO_SUCCESS
            );
            assert!((barrier - (1.0 / 512.0)).abs() < 1e-6);
            assert_eq!(
                gcso_srl_eval_rank1(y_out.as_mut_ptr(), x_in.as_ptr(), &desc, 4, 4),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_srl_eval_rank1_fused(y_out.as_mut_ptr(), x_in.as_ptr(), &desc, 4, 4),
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
    fn test_unified_field_limina_bridging_bayesian_attention_and_swarm_abi() {
        let trail_phases = [32i8; MAX_ATTENTION_HEADS];
        let anchors_flat = [[-96i8; MAX_ATTENTION_HEADS], [32i8; MAX_ATTENTION_HEADS]];
        let energies = [0.5f32, 3.0f32];
        let mut res_pull = 0.0f32;
        let mut res_coh = 0.0f32;
        let mut folded_phases = [0i8; MAX_ATTENTION_HEADS];
        let mut tot_pull = 0.0f32;
        let mut mean_coh = 0.0f32;
        let mut dom_id = 0u32;
        let mut act_cnt = 0u32;

        // SAFETY: Local stack buffers satisfy alignment and bounds invariants.
        unsafe {
            assert_eq!(
                gcso_attractor_eval_resonant_pull(
                    trail_phases.as_ptr(),
                    anchors_flat[1].as_ptr(),
                    2.5,
                    &mut res_pull,
                    &mut res_coh,
                ),
                GCSO_SUCCESS
            );
            assert!((res_coh - 1.0).abs() < 1e-4 && (res_pull - 5.0).abs() < 1e-4);

            assert_eq!(
                gcso_attractor_fold_unified_field(
                    trail_phases.as_ptr(),
                    1.0,
                    anchors_flat.as_ptr().cast::<gcso_q7_t>(),
                    energies.as_ptr(),
                    2,
                    folded_phases.as_mut_ptr(),
                    &mut tot_pull,
                    &mut mean_coh,
                    &mut dom_id,
                    &mut act_cnt,
                ),
                GCSO_SUCCESS
            );
        }
        assert_eq!(dom_id, 1);
        assert_eq!(act_cnt, 2);
        assert!(tot_pull > 7.0 && mean_coh > 0.90);

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

        let cand0 = gcso_pointer_trail_t {
            stigmergic_density: 4.0,
            accumulated_phase_delta: [-96i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let cand1 = gcso_pointer_trail_t {
            stigmergic_density: 3.0,
            target_anchor_id: 77,
            accumulated_phase_delta: [36i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let mut candidates = [cand0, cand1];

        let mut sel_id = 0u32;
        let mut bayes_score = 0.0f32;
        let mut bridge_lp = 0.0f32;
        let mut found = 0u8;
        let mut attn_scores = [0.0f32, 0.0f32];
        let mut best_cand = 0u32;
        let mut coh_cnt = 0u32;
        let mut swarm_rep = SwarmResolutionReport::default();

        // SAFETY: Local stack buffers satisfy alignment and disjointness invariants.
        unsafe {
            assert_eq!(
                gcso_attractor_steer_trail_unified_field(
                    &mut src_trail,
                    anchors_flat.as_ptr().cast::<gcso_q7_t>(),
                    energies.as_ptr(),
                    2,
                    ptr::null(),
                    0.0,
                    &mut tot_pull,
                    &mut mean_coh,
                    &mut act_cnt,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(src_trail.target_anchor_id, 1);
            assert_eq!(src_trail.cluster_id, 1);

            assert_eq!(
                gcso_action_hub_select_and_bridge_best_next_trail(
                    &mut src_trail,
                    candidates.as_mut_ptr(),
                    2,
                    128,
                    2.0,
                    0.5,
                    0.25,
                    64,
                    0.25,
                    &mut sel_id,
                    &mut bayes_score,
                    &mut bridge_lp,
                    &mut found,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(found, 1);
            assert_eq!(sel_id, 1);
            assert_eq!(src_trail.linked_trail_id, 1);
            assert_eq!(src_trail.target_anchor_id, 77);
            assert_eq!(src_trail.cluster_id, 77);
            assert!(bayes_score > 0.5 && bridge_lp > -0.05);

            assert_eq!(
                gcso_action_hub_modulate_attention_scores_by_candidate_trails(
                    attn_scores.as_mut_ptr(),
                    &src_trail,
                    candidates.as_ptr(),
                    2,
                    128,
                    2.0,
                    0.5,
                    0.25,
                    0.25,
                    1.0,
                    &mut mean_coh,
                    &mut best_cand,
                    &mut coh_cnt,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(best_cand, 1);
            assert_eq!(coh_cnt, 1);
            assert!(attn_scores[1] > 1.0 && attn_scores[0] < -0.5);

            let mut fused_src_trail = gcso_pointer_trail_t {
                accumulated_phase_delta: [32i8; MAX_ATTENTION_HEADS],
                ..Default::default()
            };
            let mut fused_scores = [0.0f32, 0.0f32];
            let mut fused_sel_id = 0u32;
            let mut fused_b_score = 0.0f32;
            let mut fused_m_coh = 0.0f32;
            let mut fused_c_cnt = 0u32;
            let mut fused_found = 0u8;
            assert_eq!(
                gcso_action_hub_select_bridge_and_modulate_bayesian_trails(
                    &mut fused_src_trail,
                    candidates.as_ptr(),
                    2,
                    fused_scores.as_mut_ptr(),
                    128,
                    2.0,
                    0.5,
                    0.25,
                    0.25,
                    64,
                    1.0,
                    &mut fused_sel_id,
                    &mut fused_b_score,
                    &mut fused_m_coh,
                    &mut fused_c_cnt,
                    &mut fused_found,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(fused_found, 1);
            assert_eq!(fused_sel_id, 1);
            assert_eq!(fused_c_cnt, 1);
            assert_eq!(fused_src_trail.linked_trail_id, 1);
            assert_eq!(fused_src_trail.cluster_id, 77);
            assert!(fused_scores[1] > 1.0 && fused_scores[0] < -0.5);

            assert_eq!(
                gcso_action_hub_resolve_trail_in_swarm(
                    &mut src_trail,
                    src_trail.current_ptr,
                    anchors_flat[1].as_ptr(),
                    0xFF,
                    128,
                    1,
                    DEFAULT_SWARM_TOLERANCE_Q7,
                    DEFAULT_SWARM_QUORUM_THRESHOLD,
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
    fn test_gateway_toroidal_kmeans_coarse_to_fine_and_four_section_container_abi() {
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
        assert_eq!(frame.zimms_section_offset48, 0x0018_0000);
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

            let mut c_idx = 99usize;
            let mut c_id = 0u32;
            let mut c_coh = 0.0f32;
            let mut c_grav = 0.0f32;
            assert_eq!(
                gcso_action_hub_route_coarse_to_fine_centroid_only(
                    &mut trail,
                    centroids.as_ptr(),
                    1,
                    64,
                    &mut c_idx,
                    &mut c_id,
                    &mut c_coh,
                    &mut c_grav,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(c_idx, 0);
            assert_eq!(c_id, 55);
            assert!(c_coh > 0.90 && c_grav > 1.0);

            let mut tot_pull = 0.0f32;
            let mut dom_id = 0u32;
            let mut eff_ripa = 0.0f32;
            assert_eq!(
                gcso_dpsr_steer_crystallized_ensemble_zero_copy_safe(
                    centroids.as_ptr(),
                    1,
                    &mut trail,
                    0.01,
                    q_buf.as_mut_ptr(),
                    ptr::null_mut(),
                    4,
                    16,
                    &mut tot_pull,
                    &mut dom_id,
                    &mut eff_ripa,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(dom_id, 55);
            assert!(tot_pull > 4.0 && eff_ripa > 0.05);
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
        let mut win_k = 99usize;
        let mut sel_trail_id = 0u32;
        let mut best_score = 0.0f32;
        let mut best_coh = 0.0f32;
        let mut found_route = 0u8;

        // SAFETY: Local stack buffers satisfy alignment and disjointness invariants.
        unsafe {
            assert_eq!(
                gcso_action_hub_route_coarse_to_fine_trail(
                    &mut trail,
                    centroids.as_ptr(),
                    1,
                    cand_trails.as_ptr(),
                    2,
                    128,
                    2.0,
                    0.5,
                    0.25,
                    0.50,
                    0.25,
                    &mut win_k,
                    &mut sel_trail_id,
                    &mut best_score,
                    &mut best_coh,
                    &mut found_route,
                ),
                GCSO_SUCCESS
            );
        }
        assert_eq!(found_route, 1);
        assert_eq!(win_k, 0);
        assert_eq!(sel_trail_id, 1);
        assert_eq!(trail.linked_trail_id, 1);
        assert!(best_score > 0.5 && best_coh > 0.90);

        let mut c2f_cand_steer_trail = gcso_pointer_trail_t {
            accumulated_phase_delta: [24i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let mut c2f_cand_win_k = 99usize;
        let mut c2f_cand_sel_id = 0u32;
        let mut c2f_cand_score = 0.0f32;
        let mut c2f_cand_coh = 0.0f32;
        let mut c2f_cand_found = 0u8;
        let mut c2f_bayes_scores = [0.0f32, 0.0f32];
        let mut c2f_macro_aid = 0u32;
        let mut c2f_coh_cnt = 0u32;
        let mut daes = gcso_daes_slot_t::default();
        let pspm_cfg = gcso_pspm_config_t {
            num_fact_heads: 8,
            num_logic_heads: 4,
            num_explore_heads: 4,
            fact_phase_gain: 1.0,
            logic_phase_gain: 1.0,
            explore_phase_gain: 1.15,
            ..Default::default()
        };

        // SAFETY: Local stack buffers satisfy alignment and disjointness invariants.
        unsafe {
            assert_eq!(
                gcso_action_hub_route_coarse_to_fine_trail_and_steer_qk_safe(
                    &mut c2f_cand_steer_trail,
                    centroids.as_ptr(),
                    1,
                    cand_trails.as_ptr(),
                    2,
                    4,
                    16,
                    2.0,
                    0.5,
                    0.25,
                    0.50,
                    0.25,
                    64,
                    0.0,
                    q_buf.as_mut_ptr(),
                    k_buf.as_mut_ptr(),
                    &mut c2f_cand_win_k,
                    &mut c2f_cand_sel_id,
                    &mut c2f_cand_score,
                    &mut c2f_cand_coh,
                    &mut c2f_cand_found,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(c2f_cand_found, 1);
            assert_eq!(c2f_cand_win_k, 0);
            assert_eq!(c2f_cand_sel_id, 1);
            assert_eq!(c2f_cand_steer_trail.linked_trail_id, 1);
            assert_eq!(c2f_cand_steer_trail.cluster_id, 55);

            let mut c2f_bayes_trail = gcso_pointer_trail_t {
                accumulated_phase_delta: [24i8; MAX_ATTENTION_HEADS],
                ..Default::default()
            };
            assert_eq!(
                gcso_action_hub_route_coarse_to_fine_bayesian_and_steer_qk_safe(
                    &mut c2f_bayes_trail,
                    centroids.as_ptr(),
                    1,
                    64,
                    cand_trails.as_ptr(),
                    2,
                    c2f_bayes_scores.as_mut_ptr(),
                    4,
                    16,
                    2.0,
                    0.5,
                    0.25,
                    1.0,
                    64,
                    0.25,
                    &pspm_cfg,
                    0.0,
                    &mut daes,
                    q_buf.as_mut_ptr(),
                    k_buf.as_mut_ptr(),
                    &mut c2f_cand_win_k,
                    &mut c2f_macro_aid,
                    &mut c2f_cand_sel_id,
                    &mut c2f_cand_score,
                    &mut c2f_cand_coh,
                    &mut c2f_coh_cnt,
                    &mut c2f_cand_found,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(c2f_cand_found, 1);
            assert_eq!(c2f_macro_aid, 55);
            assert_eq!(c2f_cand_sel_id, 1);
            assert_eq!(c2f_coh_cnt, 1);
            assert!(c2f_bayes_scores[1] > 1.0 && c2f_bayes_scores[0] < -0.5);
            assert_ne!(
                daes.telemetry_mini_ledger[0]
                    & (DAES_METRIC_PHASE_STEERED | DAES_METRIC_PSPM_ROUTED),
                0
            );
        }

        // Verify standalone Section-1 record zero-copy serialization and verification C-ABI
        let mut rec_bytes_buf = AlignedBuffer64::<u8, 256>::new([0u8; 256]);
        let mut rec_req_len = 0usize;
        let mut verified_recs_ptr: *const gcso_crystallized_attractor_record_t = ptr::null();

        // SAFETY: Local stack buffers satisfy 64B alignment and disjointness invariants.
        unsafe {
            assert_eq!(
                gcso_container_write_crystallized_records_zero_copy(
                    centroids.as_ptr(),
                    1,
                    ptr::null_mut(),
                    &mut rec_req_len,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(rec_req_len, 128);
            assert_eq!(
                gcso_container_write_crystallized_records_zero_copy(
                    centroids.as_ptr(),
                    1,
                    rec_bytes_buf.data.as_mut_ptr(),
                    &mut rec_req_len,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_container_verify_crystallized_records_zero_copy(
                    rec_bytes_buf.data.as_ptr(),
                    rec_req_len,
                    1,
                    &mut verified_recs_ptr,
                ),
                GCSO_SUCCESS
            );
            assert!(!verified_recs_ptr.is_null() && (*verified_recs_ptr).anchor_id == 55);
        }

        // Verify 4-section `.gcso` zero-copy serialization and verification C-ABI
        let reports_arr = [report];
        let mut container_buf = AlignedBuffer128::<u8, 768>::new([0u8; 768]);
        let mut req_len = 0usize;
        let mut zimms_desc = gcso_zimms_descriptor_t::default();
        let mut nc_out = 0usize;
        let mut nt_out = 0usize;
        let mut nr_out = 0usize;
        let mut c_ptr: *const gcso_crystallized_attractor_record_t = ptr::null();
        let mut t_ptr: *const gcso_pointer_trail_t = ptr::null();
        let mut b_ptr: *const gcso_contrastive_bloom_filter_t = ptr::null();
        let mut d_ptr: *const gcso_daes_slot_t = ptr::null();
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
                    &daes,
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
                    &daes,
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
                    &mut zimms_desc,
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
            assert_eq!(zimms_desc.file_size_bytes, req_len as u64);
            assert_eq!(zimms_desc.flags, 4);
            assert_eq!(nc_out, 1);
            assert_eq!(nt_out, 1);
            assert_eq!(nr_out, 1);
            assert!(!c_ptr.is_null() && (*c_ptr).anchor_id == 55);
            assert!(!t_ptr.is_null() && (*t_ptr).cluster_id == 999);
            assert!(!b_ptr.is_null() && !d_ptr.is_null());
            assert!(!r_ptr.is_null() && (*r_ptr).dominant_slot == ATTRACTOR_SLOT_KNOWLEDGE);

            let mut fused_centroids = [CrystallizedAttractorRecord::default(); 2];
            let mut fused_trail = gcso_pointer_trail_t::default();
            let mut fused_idx = 99usize;
            let mut fused_coh = 0.0f32;
            let mut fused_mass = 0.0f32;
            assert_eq!(
                gcso_dpsr_steer_and_kmeans_crystallize_gateway_frame_safe(
                    &frame,
                    &mut fused_trail,
                    64,
                    q_buf.as_mut_ptr(),
                    k_buf.as_mut_ptr(),
                    4,
                    16,
                    logits.as_mut_ptr(),
                    8,
                    fused_centroids.as_mut_ptr(),
                    2,
                    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
                    &mut report,
                    &mut fused_idx,
                    &mut fused_coh,
                    &mut fused_mass,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(fused_idx, 0);
            assert_eq!(fused_trail.cluster_id, 55);
            assert!(fused_mass > 0.0 && fused_centroids[0].verify_integrity());

            let mut c2f_steer_trail = gcso_pointer_trail_t {
                accumulated_phase_delta: [24i8; MAX_ATTENTION_HEADS],
                ..Default::default()
            };
            let mut c2f_win_idx = 99usize;
            let mut c2f_aid = 0u32;
            let mut c2f_coh = 0.0f32;
            let mut c2f_grav = 0.0f32;
            assert_eq!(
                gcso_action_hub_route_coarse_to_fine_and_steer_qk_safe(
                    &mut c2f_steer_trail,
                    fused_centroids.as_ptr(),
                    2,
                    64,
                    0.0,
                    q_buf.as_mut_ptr(),
                    k_buf.as_mut_ptr(),
                    4,
                    16,
                    &mut c2f_win_idx,
                    &mut c2f_aid,
                    &mut c2f_coh,
                    &mut c2f_grav,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(c2f_win_idx, 0);
            assert_eq!(c2f_aid, 55);
            assert_eq!(c2f_steer_trail.cluster_id, 55);

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

            let mut oneshot_centroids = [CrystallizedAttractorRecord::default(); 2];
            let mut oneshot_trail = gcso_pointer_trail_t::default();
            let mut oneshot_frame = gcso_gateway_frame_t::default();
            let mut oneshot_report = gcso_causal_provenance_report_t::default();
            let mut oneshot_idx = 99usize;
            let mut oneshot_coh = 0.0f32;
            let mut oneshot_mass = 0.0f32;
            assert_eq!(
                gcso_gateway_synthesize_kmeans_and_steer_qk_safe(
                    0.10,
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
                    &mut oneshot_trail,
                    64,
                    q_buf.as_mut_ptr(),
                    k_buf.as_mut_ptr(),
                    4,
                    16,
                    logits.as_mut_ptr(),
                    8,
                    oneshot_centroids.as_mut_ptr(),
                    2,
                    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
                    &mut oneshot_frame,
                    &mut oneshot_report,
                    &mut oneshot_idx,
                    &mut oneshot_coh,
                    &mut oneshot_mass,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(oneshot_idx, 0);
            assert_eq!(oneshot_trail.cluster_id, 55);
            assert_eq!(oneshot_frame.target_anchor_id, 55);
            assert_eq!(oneshot_report.dominant_slot, ATTRACTOR_SLOT_KNOWLEDGE);
            assert!(oneshot_mass > 0.0 && oneshot_centroids[0].verify_integrity());

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

            let mut zc_c2f_trail = gcso_pointer_trail_t {
                accumulated_phase_delta: [28i8; MAX_ATTENTION_HEADS],
                ..Default::default()
            };
            let mut zc_c2f_win_k = 99usize;
            let mut zc_c2f_sel_id = 0u32;
            let mut zc_c2f_score = 0.0f32;
            let mut zc_c2f_coh = 0.0f32;
            let mut zc_c2f_rep = 1u8;
            let mut zc_c2f_found = 0u8;
            assert_eq!(
                gcso_container_map_route_coarse_to_fine_and_steer_four_section_zero_copy_safe(
                    container_buf.data.as_ptr(),
                    req_len,
                    &mut zc_c2f_trail,
                    4,
                    16,
                    2.0,
                    0.5,
                    0.25,
                    0.50,
                    0.25,
                    64,
                    4,
                    0.0,
                    q_buf.as_mut_ptr(),
                    k_buf.as_mut_ptr(),
                    &mut zc_c2f_win_k,
                    &mut zc_c2f_sel_id,
                    &mut zc_c2f_score,
                    &mut zc_c2f_coh,
                    &mut zc_c2f_rep,
                    &mut zc_c2f_found,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(zc_c2f_win_k, 0);
            assert_eq!(zc_c2f_rep, 0);
            assert_eq!(zc_c2f_trail.cluster_id, 55);

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

            let mut closed_loop_frame = gcso_gateway_frame_t::default();
            let mut closed_loop_report = gcso_causal_provenance_report_t::default();
            let mut closed_loop_trail = gcso_pointer_trail_t::default();
            assert_eq!(
                gcso_gateway_synthesize_from_crystallized_records(
                    raw_f32_centroids.as_ptr(),
                    2,
                    &mut closed_loop_frame,
                    &mut closed_loop_report,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(closed_loop_frame.target_anchor_id, 88);
            assert_eq!(
                gcso_dpsr_steer_closed_loop_crystallized_gateway_safe(
                    raw_f32_centroids.as_ptr(),
                    2,
                    &mut closed_loop_trail,
                    64,
                    q_buf.as_mut_ptr(),
                    k_buf.as_mut_ptr(),
                    4,
                    16,
                    logits.as_mut_ptr(),
                    8,
                    &mut closed_loop_frame,
                    &mut closed_loop_report,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(closed_loop_trail.cluster_id, 88);
            assert_eq!(closed_loop_trail.target_anchor_id, 88);

            assert_eq!(gcso_zimms_close_mmap(&mut zimms_desc), GCSO_SUCCESS);
            assert_eq!(zimms_desc.fd_handle, -1);
        }
    }

    #[test]
    fn test_synergy_inhibition_pointer_dpsr_and_snapshot_crc32() {
        let persona_phases = [32i8; MAX_ATTENTION_HEADS];
        let knowledge_phases = [32i8; MAX_ATTENTION_HEADS];
        let repulsion_phases = [-96i8; MAX_ATTENTION_HEADS];
        let mut fused_phases = [0i8; MAX_ATTENTION_HEADS];
        let mut pull_out = 0.0f32;
        let mut coh_out = 0.0f32;
        let mut damped_weight = 0.0f32;

        // SAFETY: Local stack buffers satisfy alignment and disjointness invariants.
        unsafe {
            assert_eq!(
                gcso_attractor_eval_cross_slot_synergy(
                    persona_phases.as_ptr(),
                    2.0,
                    knowledge_phases.as_ptr(),
                    2.0,
                    u32::from(ATTRACTOR_SLOT_PERSONA),
                    u32::from(ATTRACTOR_SLOT_KNOWLEDGE),
                    DEFAULT_SLOT_COUPLING_MATRIX_8X8,
                    0.5,
                    fused_phases.as_mut_ptr(),
                    &mut pull_out,
                    &mut coh_out,
                ),
                GCSO_SUCCESS
            );
            assert_eq!(
                gcso_attractor_apply_mutual_slot_inhibition(
                    fused_phases.as_mut_ptr(),
                    pull_out,
                    repulsion_phases.as_ptr(),
                    0.5,
                    &mut damped_weight,
                ),
                GCSO_SUCCESS
            );
        }
        assert!((coh_out - 1.0).abs() < 1e-4);
        assert!((damped_weight - 3.0).abs() < 0.1);
        assert_eq!(fused_phases[0], 16);

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
                    DEFAULT_RIPA_CLAMP_MAX_RAD,
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
                    DEFAULT_RIPA_CLAMP_MAX_RAD,
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
                StigmergicActionDecision::FullOrchestrationSrl as u8
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
}
