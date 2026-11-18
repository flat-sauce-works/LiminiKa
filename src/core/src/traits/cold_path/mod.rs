// name: src/core/src/traits/cold_path/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lv.3 Macro Observer traits (`traits/cold_path.rs`) for the
//! Geometric Cellular Sheaf Orchestrator (GCSO):
//! - **[`SheafCohomologyEvaluator`] & [`HierarchicalTreeTable`]**: Matrix-free $\mathcal{O}(N)$
//!   Gershgorin upper-bound evaluation ($\hat{\lambda}_{\max} = 2 \max_i D_{ii}$), `0-fdiv` 3rd-order
//!   Chebyshev spectral filtering, dual homology-cohomology pairing
//!   $\langle \mathbf{r}_{\mathrm{obs}}, \boldsymbol{\gamma} \rangle$, trail-chain joint
//!   log-probability $\sum_t \ln P(M_t \mid M_{t-1})$ & $S^1$ holonomy obstruction evaluation,
//!   and 256-bit dyadic QDPT/CPT pointer cascading.
//! - **[`EntropyEvaluator`] & [`EdbcBranchController`]**: `0-fdiv` sparse surrogate entropy,
//!   moving Z-score entropy $\tilde{H} \in [0, \ln 2]$, sliding-window flux $\Phi_M(t)$,
//!   $\mathcal{C}_{\mathrm{void}}$, scale-invariant Langevin pitchfork barriers, energy-guided
//!   temperature modulation, and 16-head SWAR metastable tunneling perturbation.
//! - **[`AttractorField`] & [`MultiDomainAttractorSpace`]**: Single unified phase manifold
//!   ($SO(2)^{d_{\mathrm{head}}/2}$) with zero-hyperparameter phase-proximity gravity, $S^1$
//!   geodesic anchor folding (`branchless_q7_unified_field_fold_64`), dynamic 8-lane hardware
//!   working cache projection, autonomous lane weight rebalancing, Product of Lie Group Rotations,
//!   cross-slot synergy, and mutual inhibition.
//! - **[`PersonaPatcher`], [`L2pSvdProjector`], [`KvKeyframeCache`], [`DnpPacketDispatcher`], & [`MappedStorage`]**:
//!   Persistent topological mode quantization, training-free LoRA-to-Phase SVD projection,
//!   keyframe KV-keyframe trajectory & Limina joint-probability replay, 64B DNP packet codecs, and
//!   zero-copy `.gcso` persistence.
//! - **[`ColdPathTrailCrystalExt`], [`ColdPathObserver`], [`ColdPathOrchestrator`], & [`FullStackGcsoEngine`]**:
//!   Blanket-implemented macro observers and full-stack capability markers over [`crate::traits::ActionHub`].

#![forbid(unsafe_code)]
#![deny(missing_docs)]
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

pub use crate::traits::branchless::{sum_abs_and_sq_f32_simd8, sum_sq_f32_simd8};
use core::f32::consts::{FRAC_PI_2, LN_2};
use crate::traits::aligned::{
    AlignedBuffer16, AlignedSlice, AlignedSlice16, AlignedSlice32, AlignedSlice64,
    AlignedSliceMut16, AlignedSliceMut32, AlignedSliceMut64,
};
use crate::traits::branchless::{BranchlessKernel, apply_pframe_motion_step_words_64, atan2_interleaved_pairs_to_q7_word_x8_simd, atan2_to_q7_word_x8_simd, branchless_fast_transition_log_prob, branchless_q7_inhibit_64, branchless_q7_inhibit_words_64, branchless_q7_synergy_fuse_64, branchless_q7_synergy_fuse_words_64, branchless_q7_transition_log_prob_words_64, branchless_q7_unified_field_fold_words_with_init_weight_64, chebyshev_deg3_filter_slice_simd8, circular_r_bar_from_words_64, circular_resultant_from_words_64, coherence_from_diff_words_64, coherence_from_word_arrays_64, coherence_from_word_arrays_with_uniform_flag_64, diff_norm_sq_f32_unrolled4, dot_and_norms_f32_unrolled4, encode_pframe_motion_delta_64, finalize_pframe_chain_words_64, hodge_obstruction_and_dual_pairing_f32, max_pos_f32_unrolled4, seek_pframe_chain_words_64};
use crate::traits::primitives::{DEFAULT_MAX_VALID_Z_SCORE, branchless_abs_f32, branchless_atan2_f32, branchless_atan2_to_q7_fast, branchless_clamp_f32, branchless_copysign_f32, branchless_eyring_kramers_escape, branchless_flush_subnormal_f32, branchless_hysteresis_quantize_depth, branchless_inv_pos_f32, branchless_is_finite_f32, branchless_is_nan_f32, branchless_max_f32, branchless_min_f32, branchless_popcount_rsqrt_u32, branchless_q7_accumulate_64, branchless_q7_add_sat, branchless_q7_conjugate_64, branchless_q7_qdps_filter_64, branchless_rad_to_q7, branchless_relu_f32, branchless_rsqrt_f32, branchless_sanitize_z_score_f32, branchless_select_f32, branchless_select_q7, branchless_select_u32, branchless_select_u8, branchless_sincos_q7, branchless_splitmix64, branchless_sqrt_f32, branchless_tanh_f32, branchless_von_mises_kappa, crc32_block64_zero_seed_words, fast_inv_len_f32, fast_inv_pos_normal_f32, is_word_array_uniform_64, load_q7_word_at, load_q7_words_64, store_q7_word_at, store_q7_words_64, swar_q7_add_sat_x8, swar_q7_circular_add_x8, swar_q7_fold_s1_geodesic_x8, swar_q7_neg_sat_x8, swar_q7_qdps_filter_words_64, swar_q7_scale_le128_with_bias_x8, swar_q7_sub_sat_x8, trail_phase_words_mut, trail_phase_words_ref};
use crate::traits::primitives::{
    ACTION_MODE_PHASE_REPULSE, ATTRACTOR_SLOT_ID_MASK, ATTRACTOR_SLOT_REPULSION,
    CHEBYSHEV_COEFF_LEN, DEFAULT_ACTION_HUB_CAPACITY, DEFAULT_ENTROPY_EPS,
    DEFAULT_PAGED_BLOCK_SIZE, DEFAULT_QDPS_MIN_STEP_RAD, DEFAULT_RIPA_CLAMP_MAX_RAD,
    DEFAULT_SLOT_COUPLING_MATRIX_8X8, GcsoResult, MAX_ANCHORS_PER_FIELD, MAX_ATTENTION_HEADS,
    MAX_ATTRACTOR_DOMAINS, Pod, KvKeyframePacket, KvMotionDeltaPacket, Q7_PHASE_SCALE_RAD,
    finalize_snapshot_header_crc32,
    is_slot_pair_coupled, pack_action_slot_tag, unpack_action_mode_from_tag,
    unpack_attractor_slot_from_tag, verify_snapshot_payload_crc32,
};
use crate::traits::telemetry::{
    ContrastivePhaseBloomFilter, PagedBitmaskOps, MAX_ATTRACTOR_PULL_FORCE,
    MAX_STIGMERGIC_DENSITY, NUM_STIGMERGIC_GRANULES,
    StigmergicActionPtrOps, StigmergicTrailOps, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE,
    TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_FLAG_SYNERGY_RESONANCE,
    eval_bitmask_dyadic_ultrametric,
};
use crate::traits::ActionHub;
use crate::abi::{
    GCSO_ABI_VERSION_MAJOR, GCSO_ABI_VERSION_MINOR, GCSO_ABI_VERSION_PATCH, GCSO_CAP_CPU_BASELINE,
    GCSO_CAP_CUDA_BACKEND, GCSO_CAP_METAL_BACKEND, GCSO_CAP_VULKAN_BACKEND,
    GCSO_ERROR_BUFFER_TOO_SMALL, GCSO_ERROR_CONTAINER_CORRUPTED, GCSO_ERROR_EDBC_SINGULARITY,
    GCSO_ERROR_INVALID_ARGUMENT, GCSO_ERROR_INVALID_STATE, gcso_anchor_type_t,
    gcso_capability_flags_t, gcso_config_t, gcso_ledger_slot_t, gcso_edbc_state_t,
    gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_kv_keyframe_header_t, gcso_q7_t,
    gcso_snapshot_header_t, gcso_rank1_residual_descriptor_t, gcso_tagged_ptr_t, gcso_mapped_storage_descriptor_t,
};
use crate::traits::hot_path::{HotPathExecutionEngine, StigmergicChunkOps};

pub(super) const BLOCK_NORM_CACHE_CAP: usize = 64;


pub mod attractor;
pub mod cohomology;
pub mod coordinators;
pub mod entropy;
pub mod storage_pprc;

pub use self::attractor::*;
pub use self::cohomology::*;
pub use self::coordinators::*;
pub use self::entropy::*;
pub use self::storage_pprc::*;

#[cfg(test)]
mod tests;
