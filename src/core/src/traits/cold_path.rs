// name: src/core/src/traits/cold_path.rs
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
//! - **[`PersonaPatcher`], [`L2pSvdProjector`], [`PprcCache`], [`DnpPacketDispatcher`], & [`ZimmsStorage`]**:
//!   Persistent topological mode quantization, training-free LoRA-to-Phase SVD projection,
//!   keyframe PPRC trajectory & Limina joint-probability replay, 64B DNP packet codecs, and
//!   zero-copy `.gcso` persistence.
//! - **[`ColdPathTrailCrystalExt`], [`AttractorSlotSynergyCoordinator`],
//!   [`StigmergicCohomologyCoordinator`], [`BottomUpMacroCrystalCoordinator`],
//!   [`ColdPathObserver`], [`ColdPathOrchestrator`], & [`FullStackGcsoEngine`]**:
//!   Blanket-implemented macro observers and coordinators over [`super::ActionHub`].

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

use core::f32::consts::{FRAC_PI_2, LN_2};

pub use super::branchless::{sum_abs_and_sq_f32_simd8, sum_sq_f32_simd8};

use super::aligned::{
    AlignedBuffer16, AlignedSlice, AlignedSlice16, AlignedSlice32, AlignedSlice64,
    AlignedSliceMut16, AlignedSliceMut32, AlignedSliceMut64,
};
use super::branchless::{
    branchless_abs_f32, branchless_atan2_f32, branchless_atan2_to_q7_fast, branchless_clamp_f32,
    branchless_copysign_f32, branchless_eyring_kramers_escape, branchless_flush_subnormal_f32,
    branchless_hysteresis_quantize_depth, branchless_inv_pos_f32, branchless_is_finite_f32,
    branchless_is_nan_f32, branchless_ln_f32, branchless_max_f32, branchless_min_f32,
    branchless_popcount_rsqrt_u32, branchless_q7_accumulate_64, branchless_q7_add_sat,
    branchless_q7_conjugate_64, branchless_q7_inhibit_64, branchless_q7_inhibit_words_64,
    branchless_q7_qdps_filter_64, branchless_q7_synergy_fuse_64,
    branchless_q7_synergy_fuse_words_64, branchless_q7_transition_log_prob_words_64,
    branchless_q7_unified_field_fold_words_with_init_weight_64, branchless_rad_to_q7,
    branchless_relu_f32, branchless_rsqrt_f32, branchless_sanitize_z_score_f32,
    branchless_select_f32, branchless_select_q7, branchless_select_u32, branchless_select_u8,
    branchless_sincos_q7, branchless_splitmix64, branchless_sqrt_f32, branchless_tanh_f32,
    branchless_von_mises_kappa, chebyshev_deg3_filter_slice_simd8, circular_r_bar_from_words_64,
    coherence_from_word_arrays_64, coherence_from_word_arrays_with_uniform_flag_64,
    crc32_block64_zero_seed_words, diff_norm_sq_f32_unrolled4, dot_and_norms_f32_unrolled4,
    fast_inv_len_f32, hodge_obstruction_and_dual_pairing_f32, is_word_array_uniform_64,
    load_q7_word_at, load_q7_words_64, max_pos_f32_unrolled4, store_q7_word_at,
    store_q7_words_64, swar_q7_add_sat_x8, swar_q7_circular_add_x8, swar_q7_fold_s1_geodesic_x8,
    trail_phase_words_ref,
    swar_q7_neg_sat_x8, swar_q7_qdps_filter_words_64, swar_q7_scale_le128_with_bias_x8,
    swar_q7_sub_sat_x8, BranchlessKernel, DEFAULT_MAX_VALID_Z_SCORE,
};
use super::gateway::{
    ActionHubGatewayExt, CausalProvenanceReport, CrystallizedAttractorRecord, GatewayFrame,
    ManifoldGatewaySignal,
};
use super::hot_path::{HotPathExecutionEngine, SwarmCellChunk};
use super::primitives::{
    finalize_snapshot_header_crc32, is_slot_pair_coupled, pack_action_slot_tag,
    unpack_action_mode_from_tag, unpack_attractor_slot_from_tag, verify_snapshot_payload_crc32,
    GcsoResult, Pod, ACTION_MODE_PHASE_REPULSE, ATTRACTOR_SLOT_ID_MASK, ATTRACTOR_SLOT_REPULSION,
    CHEBYSHEV_COEFF_LEN, DEFAULT_ACTION_HUB_CAPACITY, DEFAULT_ENTROPY_EPS,
    DEFAULT_PAGED_BLOCK_SIZE, DEFAULT_QDPS_MIN_STEP_RAD, DEFAULT_RIPA_CLAMP_MAX_RAD,
    DEFAULT_SLOT_COUPLING_MATRIX_8X8, MAX_ANCHORS_PER_FIELD, MAX_ATTENTION_HEADS,
    MAX_ATTRACTOR_DOMAINS, Q7_PHASE_SCALE_RAD,
};
use super::telemetry::{
    eval_bitmask_dyadic_ultrametric, ContrastivePhaseBloomFilter, PagedBitmaskOps,
    StigmergicActionPtrOps, StigmergicTrailOps, MAX_ATTRACTOR_PULL_FORCE, MAX_STIGMERGIC_DENSITY,
    NUM_SWARM_MICRO_CELLS,
    TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE,
    TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_FLAG_SYNERGY_RESONANCE,
};
use super::ActionHub;
use crate::abi::{
    gcso_anchor_type_t, gcso_capability_flags_t, gcso_config_t, gcso_daes_slot_t,
    gcso_edbc_state_t, gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_pprc_keyframe_header_t,
    gcso_q7_t, gcso_snapshot_header_t, gcso_srl_descriptor_t, gcso_tagged_ptr_t,
    gcso_zimms_descriptor_t, GCSO_ABI_VERSION_MAJOR, GCSO_ABI_VERSION_MINOR,
    GCSO_ABI_VERSION_PATCH, GCSO_CAP_CPU_BASELINE, GCSO_CAP_CUDA_BACKEND, GCSO_CAP_METAL_BACKEND,
    GCSO_CAP_VULKAN_BACKEND, GCSO_ERROR_BUFFER_TOO_SMALL, GCSO_ERROR_CONTAINER_CORRUPTED,
    GCSO_ERROR_EDBC_SINGULARITY, GCSO_ERROR_INVALID_ARGUMENT, GCSO_ERROR_INVALID_STATE,
};

const BLOCK_NORM_CACHE_CAP: usize = 64;

// ===================================================================
// 1. Cellular Sheaf Cohomology & Matrix-Free Spectral Filtering
// ===================================================================

/// Cold-path trait for Cellular Sheaf Cohomology $H^1(K; \mathcal{F})$ evaluation, matrix-free
/// spectral filtering, and Joint Probability Chain Rule / Limina trail-chain verification (**Macro Level**).
pub trait SheafCohomologyEvaluator: BranchlessKernel + Send + Sync {
    /// Evaluates the Gershgorin upper bound $\hat{\lambda}_{\max} = 2 \cdot \max_i D_{ii}$ in $\mathcal{O}(\vert V \vert)$ time (Section 1.2).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `node_degrees` is empty.
    #[inline]
    fn eval_gershgorin_lambda_max(
        &self,
        node_degrees: &AlignedSlice32<'_, f32>,
    ) -> GcsoResult<f32> {
        if node_degrees.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(branchless_flush_subnormal_f32(
            2.0 * max_pos_f32_unrolled4(node_degrees.as_slice()),
        ))
    }

    /// Evaluates coarse supernode degrees $D_{ii} = \sum_{j \neq i} A_{ij}$ and returns
    /// $\hat{\lambda}_{\max} = 2 \max_i D_{ii}$ directly from 256-bit PagedBlock bitmasks (Section 1.2 & Lemma 1.2).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `block_masks` is empty or `degrees_out.len() < block_masks.len()`.
    #[inline]
    fn eval_stigmergic_block_degrees(
        &self,
        block_masks: &[gcso_paged_bitmask_t],
        degrees_out: &mut AlignedSliceMut32<'_, f32>,
    ) -> GcsoResult<f32> {
        let n = block_masks.len();
        if n == 0 || degrees_out.len() < n {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let deg_slice = &mut degrees_out.as_mut_slice()[..n];
        deg_slice.fill(0.0);

        let mut inv_norm_cache = [0.0_f32; BLOCK_NORM_CACHE_CAP];
        let cached_len = n.min(BLOCK_NORM_CACHE_CAP);
        for (slot, mask) in inv_norm_cache[..cached_len]
            .iter_mut()
            .zip(block_masks[..cached_len].iter())
        {
            *slot = branchless_popcount_rsqrt_u32(mask.popcount());
        }

        let mut max_deg = 0.0_f32;
        if n <= BLOCK_NORM_CACHE_CAP {
            let norms = &inv_norm_cache[..n];
            let masks = &block_masks[..n];
            for i in 0..n {
                let inv_norm_i = norms[i];
                if inv_norm_i > 0.0 {
                    let [m0, m1, m2, m3] = masks[i].bits;
                    let mut acc_i0 = deg_slice[i];
                    let mut acc_i1 = 0.0_f32;
                    let mut j = i + 1;

                    while j + 4 <= n {
                        let [w0_a, w1_a, w2_a, w3_a] = masks[j].bits;
                        let [w0_b, w1_b, w2_b, w3_b] = masks[j + 1].bits;
                        let [w0_c, w1_c, w2_c, w3_c] = masks[j + 2].bits;
                        let [w0_d, w1_d, w2_d, w3_d] = masks[j + 3].bits;

                        let inter_a = ((m0 & w0_a).count_ones() + (m1 & w1_a).count_ones())
                            + ((m2 & w2_a).count_ones() + (m3 & w3_a).count_ones());
                        let inter_b = ((m0 & w0_b).count_ones() + (m1 & w1_b).count_ones())
                            + ((m2 & w2_b).count_ones() + (m3 & w3_b).count_ones());
                        let inter_c = ((m0 & w0_c).count_ones() + (m1 & w1_c).count_ones())
                            + ((m2 & w2_c).count_ones() + (m3 & w3_c).count_ones());
                        let inter_d = ((m0 & w0_d).count_ones() + (m1 & w1_d).count_ones())
                            + ((m2 & w2_d).count_ones() + (m3 & w3_d).count_ones());

                        let a_ij_a = (inter_a as f32) * (inv_norm_i * norms[j]);
                        let a_ij_b = (inter_b as f32) * (inv_norm_i * norms[j + 1]);
                        let a_ij_c = (inter_c as f32) * (inv_norm_i * norms[j + 2]);
                        let a_ij_d = (inter_d as f32) * (inv_norm_i * norms[j + 3]);

                        deg_slice[j] += a_ij_a;
                        deg_slice[j + 1] += a_ij_b;
                        deg_slice[j + 2] += a_ij_c;
                        deg_slice[j + 3] += a_ij_d;
                        acc_i0 += a_ij_a + a_ij_b;
                        acc_i1 += a_ij_c + a_ij_d;
                        j += 4;
                    }
                    while j < n {
                        let [w0, w1, w2, w3] = masks[j].bits;
                        let inter = ((m0 & w0).count_ones() + (m1 & w1).count_ones())
                            + ((m2 & w2).count_ones() + (m3 & w3).count_ones());
                        let a_ij = (inter as f32) * (inv_norm_i * norms[j]);
                        deg_slice[j] += a_ij;
                        acc_i0 += a_ij;
                        j += 1;
                    }
                    deg_slice[i] = acc_i0 + acc_i1;
                }
                max_deg = branchless_max_f32(max_deg, deg_slice[i]);
            }
            return Ok(branchless_flush_subnormal_f32(2.0 * max_deg));
        }

        let cached_upper = BLOCK_NORM_CACHE_CAP;
        for i in 0..n {
            let mask_i = &block_masks[i];
            let inv_norm_i = if i < BLOCK_NORM_CACHE_CAP {
                inv_norm_cache[i]
            } else {
                branchless_popcount_rsqrt_u32(mask_i.popcount())
            };

            if inv_norm_i > 0.0 {
                let [m0, m1, m2, m3] = mask_i.bits;
                let mut acc_i0 = deg_slice[i];
                let mut acc_i1 = 0.0_f32;
                let mut j = i + 1;

                while j + 4 <= cached_upper {
                    let [w0_a, w1_a, w2_a, w3_a] = block_masks[j].bits;
                    let [w0_b, w1_b, w2_b, w3_b] = block_masks[j + 1].bits;
                    let [w0_c, w1_c, w2_c, w3_c] = block_masks[j + 2].bits;
                    let [w0_d, w1_d, w2_d, w3_d] = block_masks[j + 3].bits;

                    let inter_a = ((m0 & w0_a).count_ones() + (m1 & w1_a).count_ones())
                        + ((m2 & w2_a).count_ones() + (m3 & w3_a).count_ones());
                    let inter_b = ((m0 & w0_b).count_ones() + (m1 & w1_b).count_ones())
                        + ((m2 & w2_b).count_ones() + (m3 & w3_b).count_ones());
                    let inter_c = ((m0 & w0_c).count_ones() + (m1 & w1_c).count_ones())
                        + ((m2 & w2_c).count_ones() + (m3 & w3_c).count_ones());
                    let inter_d = ((m0 & w0_d).count_ones() + (m1 & w1_d).count_ones())
                        + ((m2 & w2_d).count_ones() + (m3 & w3_d).count_ones());

                    let a_ij_a = (inter_a as f32) * (inv_norm_i * inv_norm_cache[j]);
                    let a_ij_b = (inter_b as f32) * (inv_norm_i * inv_norm_cache[j + 1]);
                    let a_ij_c = (inter_c as f32) * (inv_norm_i * inv_norm_cache[j + 2]);
                    let a_ij_d = (inter_d as f32) * (inv_norm_i * inv_norm_cache[j + 3]);

                    deg_slice[j] += a_ij_a;
                    deg_slice[j + 1] += a_ij_b;
                    deg_slice[j + 2] += a_ij_c;
                    deg_slice[j + 3] += a_ij_d;
                    acc_i0 += a_ij_a + a_ij_b;
                    acc_i1 += a_ij_c + a_ij_d;
                    j += 4;
                }
                while j < cached_upper {
                    let [w0, w1, w2, w3] = block_masks[j].bits;
                    let inter = ((m0 & w0).count_ones() + (m1 & w1).count_ones())
                        + ((m2 & w2).count_ones() + (m3 & w3).count_ones());
                    let a_ij = (inter as f32) * (inv_norm_i * inv_norm_cache[j]);
                    deg_slice[j] += a_ij;
                    acc_i0 += a_ij;
                    j += 1;
                }
                for j_tail in cached_upper.max(i + 1)..n {
                    let a_ij = mask_i.stigmergic_cosine_similarity_with_inv_norm(
                        inv_norm_i,
                        &block_masks[j_tail],
                    );
                    acc_i0 += a_ij;
                    deg_slice[j_tail] += a_ij;
                }
                deg_slice[i] = acc_i0 + acc_i1;
            }
            max_deg = branchless_max_f32(max_deg, deg_slice[i]);
        }
        Ok(branchless_flush_subnormal_f32(2.0 * max_deg))
    }

    /// Computes 3rd-order Chebyshev expansion coefficients $[c_0, c_1, c_2, c_3]$ in $\mathcal{O}(1)$
    /// using Gauss-Chebyshev nodes and division-free (`0 fdiv`) 4-way Montgomery inversion (Section 1.2).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `lambda_max <= 0.0`, `epsilon <= 0.0`, or either is non-finite.
    #[inline]
    fn compute_tikhonov_chebyshev_coeffs_deg3(
        &self,
        lambda_max: f32,
        epsilon: f32,
    ) -> GcsoResult<[f32; CHEBYSHEV_COEFF_LEN]> {
        if !branchless_is_finite_f32(lambda_max)
            || !branchless_is_finite_f32(epsilon)
            || lambda_max <= 0.0
            || epsilon <= 0.0
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        const T1: [f32; 4] = [0.923_879_5, 0.382_683_43, -0.382_683_43, -0.923_879_5];
        const NODES_P1: [f32; 4] = [1.923_879_5, 1.382_683_4, 0.617_316_6, 0.076_120_496];
        const T2: [f32; 4] = [0.707_106_65, -0.707_106_77, -0.707_106_77, 0.707_106_65];
        const T3: [f32; 4] = [0.382_683_22, -0.923_879_5, 0.923_879_5, -0.382_683_22];

        let half_lmax = 0.5 * lambda_max;
        let d0 = half_lmax * NODES_P1[0] + epsilon;
        let d1 = half_lmax * NODES_P1[1] + epsilon;
        let d2 = half_lmax * NODES_P1[2] + epsilon;
        let d3 = half_lmax * NODES_P1[3] + epsilon;

        let d01 = d0 * d1;
        let d23 = d2 * d3;
        let prod_all = d01 * d23;
        let (g0, g1, g2, g3) = if branchless_is_finite_f32(prod_all) && prod_all > 1.0e-30 {
            let inv_all = branchless_inv_pos_f32(prod_all);
            let inv01 = inv_all * d23;
            let inv23 = inv_all * d01;
            (inv01 * d1, inv01 * d0, inv23 * d3, inv23 * d2)
        } else {
            (
                branchless_inv_pos_f32(d0),
                branchless_inv_pos_f32(d1),
                branchless_inv_pos_f32(d2),
                branchless_inv_pos_f32(d3),
            )
        };

        let c0 = (g0 + g1) + (g2 + g3);
        let c1 = (g0 * T1[0] + g1 * T1[1]) + (g2 * T1[2] + g3 * T1[3]);
        let c2 = (g0 * T2[0] + g1 * T2[1]) + (g2 * T2[2] + g3 * T2[3]);
        let c3 = (g0 * T3[0] + g1 * T3[1]) + (g2 * T3[2] + g3 * T3[3]);

        Ok([0.25 * c0, 0.5 * c1, 0.5 * c2, 0.5 * c3])
    }

    /// Evaluates a 3rd-order Chebyshev Polynomial Spectral Filter $\sum_{k=0}^3 c_k T_k(\tilde{\lambda})$
    /// using `0-fdiv` affine scaling and Horner-nested evaluation.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `lambda_max <= 0.0`, `lambda_max` is non-finite,
    /// `eigenvalues_in` is empty, or `eigenvalues_in.len() != filtered_out.len()`.
    #[inline]
    fn eval_chebyshev_spectral_filter_deg3(
        &self,
        eigenvalues_in: &AlignedSlice32<'_, f32>,
        filtered_out: &mut AlignedSliceMut32<'_, f32>,
        lambda_max: f32,
        cheb_coeffs: &[f32; CHEBYSHEV_COEFF_LEN],
    ) -> GcsoResult<()> {
        if !branchless_is_finite_f32(lambda_max)
            || lambda_max <= 0.0
            || eigenvalues_in.is_empty()
            || eigenvalues_in.len() != filtered_out.len()
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let inv_half_lmax = 2.0 * branchless_inv_pos_f32(lambda_max);
        let k0 = cheb_coeffs[0] - cheb_coeffs[2];
        let k1 = cheb_coeffs[1] - 3.0 * cheb_coeffs[3];
        let k2 = 2.0 * cheb_coeffs[2];
        let k3 = 4.0 * cheb_coeffs[3];

        chebyshev_deg3_filter_slice_simd8(
            eigenvalues_in.as_slice(),
            filtered_out.as_mut_slice(),
            inv_half_lmax,
            k0,
            k1,
            k2,
            k3,
        );
        Ok(())
    }

    /// Evaluates the Hierarchical Tree Pseudo-ultrametric distance $d_{\mathcal{V}}(x, y) = \tanh(w_{\mathrm{key}} (l_{\max} - l_{\mathrm{lca}}))$ (Lemma 1.1).
    #[inline(always)]
    #[must_use]
    fn eval_pseudo_ultrametric_distance(&self, l_lca: u32, l_max: u32, w_key: f32) -> f32 {
        let diff = l_max.saturating_sub(l_lca) as f32;
        let safe_w =
            branchless_select_f32(branchless_is_finite_f32(w_key) & (w_key > 0.0), w_key, 0.0);
        branchless_tanh_f32(safe_w * diff)
    }

    /// Evaluates the cohomological obstruction residual $\mathbf{r}_{\mathrm{obs}} = \mathbf{r} - \delta_0 \boldsymbol{\phi} \in \mathrm{im}(\delta_0)^\perp$ (Section 1.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `cochain_residual_r` is empty or slice lengths differ.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if the squared residual norm overflows or is non-finite.
    #[inline]
    fn evaluate_cohomological_obstruction(
        &self,
        cochain_residual_r: &AlignedSlice32<'_, f32>,
        exact_projection_delta0_phi: &AlignedSlice32<'_, f32>,
        obstruction_threshold: f32,
    ) -> GcsoResult<(f32, bool)> {
        let n = cochain_residual_r.len();
        if n == 0 || n != exact_projection_delta0_phi.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let norm_sq = diff_norm_sq_f32_unrolled4(
            cochain_residual_r.as_slice(),
            exact_projection_delta0_phi.as_slice(),
        );
        if !branchless_is_finite_f32(norm_sq) {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }
        let mean_obs = branchless_flush_subnormal_f32(norm_sq * fast_inv_len_f32(n));
        Ok((mean_obs, mean_obs > obstruction_threshold))
    }

    /// Evaluates the Hodge obstruction residual $\mathbf{r}_{\mathrm{obs}} \in H^1(K; \mathcal{F})$
    /// and normalized dual Cellular Sheaf Homology pairing $\frac{\vert\langle \mathbf{r}_{\mathrm{obs}}, \boldsymbol{\gamma} \rangle\vert}{\|\mathbf{r}_{\mathrm{obs}}\|_2 \|\boldsymbol{\gamma}\|_2} \in [0, 1]$ (Section 1.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any slice is empty or lengths differ.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if any accumulated norm or pairing inner product is non-finite.
    #[inline]
    fn evaluate_hodge_obstruction_and_dual_pairing(
        &self,
        cochain_residual_r: &AlignedSlice32<'_, f32>,
        exact_projection_delta0_phi: &AlignedSlice32<'_, f32>,
        dual_homology_cycle_gamma: &AlignedSlice32<'_, f32>,
        obstruction_threshold: f32,
    ) -> GcsoResult<(f32, f32, bool)> {
        let n = cochain_residual_r.len();
        if n == 0 || exact_projection_delta0_phi.len() != n || dual_homology_cycle_gamma.len() != n
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let (obs_sq, pair_dot, gam_sq) = hodge_obstruction_and_dual_pairing_f32(
            cochain_residual_r.as_slice(),
            exact_projection_delta0_phi.as_slice(),
            dual_homology_cycle_gamma.as_slice(),
        );

        if !branchless_is_finite_f32(obs_sq)
            || !branchless_is_finite_f32(pair_dot)
            || !branchless_is_finite_f32(gam_sq)
        {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        let mean_obs = branchless_flush_subnormal_f32(obs_sq * fast_inv_len_f32(n));
        let prod_sq = obs_sq * gam_sq;
        let valid_pair = prod_sq > 1.0e-24;
        let inv_denom = branchless_rsqrt_f32(branchless_select_f32(valid_pair, prod_sq, 1.0));
        let norm_pairing = branchless_select_f32(
            valid_pair,
            branchless_clamp_f32(branchless_abs_f32(pair_dot) * inv_denom, 0.0, 1.0),
            0.0,
        );

        Ok((mean_obs, norm_pairing, mean_obs > obstruction_threshold))
    }

    /// Detects cohomological obstruction $H^1(K; \mathcal{F})$ and records the offending phase
    /// pattern into `bloom_filter` while returning the phase-conjugate anti-phase vector ($-\boldsymbol{\Delta\theta}$).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if slice lengths mismatch or are empty.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if residual norms are non-finite.
    #[inline]
    fn detect_and_record_obstruction_repulsion(
        &self,
        cochain_residual_r: &AlignedSlice32<'_, f32>,
        exact_projection_delta0_phi: &AlignedSlice32<'_, f32>,
        obstruction_threshold: f32,
        active_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        bloom_filter: &mut ContrastivePhaseBloomFilter,
    ) -> GcsoResult<(f32, bool, [gcso_q7_t; MAX_ATTENTION_HEADS])> {
        let (mean_obs, is_obstructed) = self.evaluate_cohomological_obstruction(
            cochain_residual_r,
            exact_projection_delta0_phi,
            obstruction_threshold,
        )?;
        if is_obstructed {
            bloom_filter.insert_repulsive_pattern(active_phases);
            Ok((mean_obs, true, branchless_q7_conjugate_64(active_phases)))
        } else {
            Ok((mean_obs, false, *active_phases))
        }
    }

    /// Evaluates the **Joint Probability Chain Rule Log-Likelihood**
    /// $\ln P(M_1, \dots, M_T) = \sum_{t=1}^{T-1} \ln P(M_t \mid M_{t-1})$ and **$S^1$ Sheaf
    /// Holonomy Obstruction Energy** ($H^1(K; \mathcal{F})$) across `trail_chain` in $\mathcal{O}(T)$ time.
    ///
    /// Returns `(joint_log_prob_sum, holonomy_obstruction_energy, is_obstructed)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `trail_chain` is empty or `obstruction_threshold` is non-finite.
    #[inline]
    fn evaluate_trail_chain_cohomology_and_joint_log_prob(
        &self,
        trail_chain: &[gcso_pointer_trail_t],
        obstruction_threshold: f32,
    ) -> GcsoResult<(f32, f32, bool)> {
        let len = trail_chain.len();
        if len == 0 || !branchless_is_finite_f32(obstruction_threshold) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if len == 1 {
            return Ok((0.0, 0.0, false));
        }

        let mut joint_log_prob = 0.0_f32;
        let mut coboundary_defect_sum = 0.0_f32;
        let first_w = *trail_phase_words_ref(&trail_chain[0]);
        let mut prev_w = first_w;
        for trail in &trail_chain[1..len] {
            let cur_w = *trail_phase_words_ref(trail);
            let (coh, log_p) = branchless_q7_transition_log_prob_words_64(&prev_w, &cur_w);
            joint_log_prob += log_p;
            coboundary_defect_sum += branchless_relu_f32(0.5 * (1.0 - coh));
            prev_w = cur_w;
        }

        let mean_step_defect = coboundary_defect_sum * fast_inv_len_f32(len - 1);
        let loop_coh = coherence_from_word_arrays_64(&first_w, &prev_w);
        let loop_defect = branchless_relu_f32(0.5 * (1.0 - loop_coh));
        let holonomy_energy =
            branchless_flush_subnormal_f32(0.5 * (mean_step_defect + loop_defect));
        let is_obstructed = holonomy_energy > branchless_relu_f32(obstruction_threshold);

        Ok((
            branchless_flush_subnormal_f32(joint_log_prob),
            holonomy_energy,
            is_obstructed,
        ))
    }

    /// Bridges all consecutive inter-memory gaps (**Limina**) across `trail_chain` via
    /// [`StigmergicTrailOps::bridge_liminal_hallucinated_trail`], recording any self-contradictory
    /// transitions ($H^1(K; \mathcal{F}) \neq 0$) into `bloom_filter`.
    ///
    /// Returns `(total_joint_log_prob, repulsed_transition_count)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `trail_chain` is empty or `contradiction_threshold` is non-finite.
    #[inline]
    fn bridge_and_filter_liminal_trail_chain(
        &self,
        trail_chain: &mut [gcso_pointer_trail_t],
        interpolation_u7: u8,
        contradiction_threshold: f32,
        bloom_filter: &mut ContrastivePhaseBloomFilter,
    ) -> GcsoResult<(f32, u32)> {
        let len = trail_chain.len();
        if len == 0 || !branchless_is_finite_f32(contradiction_threshold) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let mut total_log_prob = 0.0_f32;
        let mut repulsed_count = 0u32;

        for idx in 1..len {
            let (head, tail) = trail_chain.split_at_mut(idx);
            let prev_ref = &head[idx - 1];
            let cur_mut = &mut tail[0];
            let (_coh, log_p, repulsed) = cur_mut.bridge_liminal_hallucinated_trail(
                prev_ref,
                interpolation_u7,
                contradiction_threshold,
            );
            total_log_prob += log_p;
            if repulsed {
                bloom_filter.insert_repulsive_pattern_words_weighted(
                    trail_phase_words_ref(cur_mut),
                    1,
                );
                repulsed_count = repulsed_count.wrapping_add(1);
            }
        }
        Ok((branchless_flush_subnormal_f32(total_log_prob), repulsed_count))
    }
}

// ===================================================================
// 2. Hierarchical Tree Table (QDPT / CPT) & Pointer Mode Cascading
// ===================================================================

/// Trait for Quantized Depth Pointer Table (QDPT) and Cascade Pointer Table (CPT) (**Micro / Mezzo Level**).
pub trait HierarchicalTreeTable: BranchlessKernel + Send + Sync {
    /// Quantizes continuous pseudo-ultrametric depth $d_{\mathcal{V}} \in [0, 1)$ into discrete QDPT level
    /// $\bar{D}_{\mathcal{V},\mathrm{quantized}}$ with hysteresis band $\delta_{\mathrm{hyst}}$ (Section 1.3).
    #[inline(always)]
    #[must_use]
    fn quantize_depth_hysteresis(
        &self,
        d_v_distance: f32,
        current_level: u32,
        max_level: u32,
        hysteresis_band: f32,
    ) -> u32 {
        let scaled = branchless_clamp_f32(d_v_distance, 0.0, 1.0) * (max_level as f32);
        branchless_hysteresis_quantize_depth(scaled, current_level, hysteresis_band, max_level)
    }

    /// Executes an $\mathcal{O}(1)$ cascade pointer lookup combining parent and child phase deltas.
    #[inline(always)]
    #[must_use]
    fn cascade_tagged_ptr(
        &self,
        parent_ptr: gcso_tagged_ptr_t,
        child_addr_offset: u64,
        child_phase_delta: gcso_q7_t,
    ) -> gcso_tagged_ptr_t {
        let new_addr =
            parent_ptr.unpack_addr().wrapping_add(child_addr_offset) & gcso_tagged_ptr_t::ADDR_MASK;
        let combined_phase = branchless_q7_add_sat(parent_ptr.unpack_phase(), child_phase_delta);
        gcso_tagged_ptr_t::pack(new_addr, parent_ptr.unpack_tag(), combined_phase)
    }

    /// Executes an $\mathcal{O}(1)$ cascade pointer transition merging `child_action_mode_or_mask`
    /// and binding to `child_attractor_slot` (`0..=7`).
    #[inline(always)]
    #[must_use]
    fn cascade_action_slot_ptr(
        &self,
        parent_ptr: gcso_tagged_ptr_t,
        child_addr_offset: u64,
        child_attractor_slot: u8,
        child_action_mode_or_mask: u8,
        child_phase_delta: gcso_q7_t,
    ) -> gcso_tagged_ptr_t {
        let new_addr =
            parent_ptr.unpack_addr().wrapping_add(child_addr_offset) & gcso_tagged_ptr_t::ADDR_MASK;
        let merged_mode = parent_ptr.action_mode_flags() | child_action_mode_or_mask;
        let tag = pack_action_slot_tag(child_attractor_slot, merged_mode);
        let combined_phase = branchless_q7_add_sat(parent_ptr.unpack_phase(), child_phase_delta);
        gcso_tagged_ptr_t::pack(new_addr, tag, combined_phase)
    }

    /// Evaluates 256-bit dyadic distance $d_{\mathcal{V}}(x, y) \in [0, 1)$, quantizes it into a
    /// discrete QDPT level, and cascades `parent_ptr` in $\mathcal{O}(1)$.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `qdpt_addr_offsets` is empty or its length does
    /// not match `qdpt_phase_deltas.len()`.
    #[inline(always)]
    fn resolve_dyadic_qdpt_and_cascade_ptr(
        &self,
        parent_ptr: gcso_tagged_ptr_t,
        current_mask: &gcso_paged_bitmask_t,
        anchor_mask: &gcso_paged_bitmask_t,
        w_key: f32,
        current_level: u32,
        hysteresis_band: f32,
        qdpt_addr_offsets: &[u64],
        qdpt_phase_deltas: &[gcso_q7_t],
    ) -> GcsoResult<(u32, f32, gcso_tagged_ptr_t)> {
        let table_len = qdpt_addr_offsets.len();
        if table_len == 0 || table_len != qdpt_phase_deltas.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let max_level = (table_len - 1) as u32;
        let d_v = eval_bitmask_dyadic_ultrametric(current_mask, anchor_mask, w_key);
        let clamped_cur =
            branchless_select_u32(current_level <= max_level, current_level, max_level);
        let level = self.quantize_depth_hysteresis(d_v, clamped_cur, max_level, hysteresis_band);
        let idx = (level as usize).min(table_len - 1);
        let cascaded =
            self.cascade_tagged_ptr(parent_ptr, qdpt_addr_offsets[idx], qdpt_phase_deltas[idx]);
        Ok((level, d_v, cascaded))
    }
}

// ===================================================================
// 3. Entropy Evaluation & EDBC Pitchfork Bifurcation Control
// ===================================================================

/// Cold-path trait for entropy-driven branch evaluation and bifurcation tracking (**Macro Level**).
pub trait EntropyEvaluator: BranchlessKernel + Send + Sync {
    /// Evaluates token Z-score activation entropy and updates dynamic branch mode in `state_out`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if `token_z_score` is `NaN` or infinite.
    fn eval_stateful(
        &mut self,
        token_z_score: f32,
        state_out: &mut gcso_edbc_state_t,
    ) -> GcsoResult<()>;

    /// Evaluates normalized Sparse Surrogate Attention Entropy $H_{\mathrm{sparse\text{-}surrogate}} \in [0, \ln 2]$
    /// in division-free (`0 fdiv`) branchless `#![no_std]` mode (Section 3.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `p_max <= 0.0` or either probability is `NaN`.
    #[inline]
    fn eval_sparse_surrogate_entropy(&self, p_max: f32, p_2nd: f32, eps: f32) -> GcsoResult<f32> {
        if p_max <= 0.0 || branchless_is_nan_f32(p_max) || branchless_is_nan_f32(p_2nd) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let p_2nd_clean = branchless_relu_f32(p_2nd);
        let sum = p_max + p_2nd_clean;
        let inv_sum = branchless_inv_pos_f32(sum);

        let p_max_norm = branchless_clamp_f32(p_max * inv_sum, 0.0, 1.0);
        let p_2nd_norm = branchless_clamp_f32(p_2nd_clean * inv_sum, 0.0, 1.0);
        let eps_log = branchless_select_f32(
            (eps > 0.0) & branchless_is_finite_f32(eps),
            eps,
            DEFAULT_ENTROPY_EPS,
        );

        let term1 = -p_max_norm * branchless_ln_f32(p_max_norm + eps_log);
        let term2 = -p_2nd_norm * branchless_ln_f32(p_2nd_norm + eps_log);
        Ok(branchless_clamp_f32(term1 + term2, 0.0, LN_2))
    }

    /// Evaluates 1st-order inter-layer activation cosine distance
    /// $d_{\mathrm{cosine}}(\mathbf{h}_{\mathrm{mid,srl}}, \mathbf{h}_{\mathrm{deep,srl}}) \in [0, 2]$ (Section 1.5 & 3.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if slices are empty or differ in length.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if inner products overflow or are non-finite.
    #[inline]
    fn eval_activation_cosine_distance(
        &self,
        h_mid_srl: &AlignedSlice32<'_, f32>,
        h_deep_srl: &AlignedSlice32<'_, f32>,
    ) -> GcsoResult<f32> {
        if h_mid_srl.is_empty() || h_mid_srl.len() != h_deep_srl.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (dot, norm_a_sq, norm_b_sq) =
            dot_and_norms_f32_unrolled4(h_mid_srl.as_slice(), h_deep_srl.as_slice());

        if !branchless_is_finite_f32(dot)
            || !branchless_is_finite_f32(norm_a_sq)
            || !branchless_is_finite_f32(norm_b_sq)
        {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }
        let prod_sq = norm_a_sq * norm_b_sq;
        let valid = prod_sq > 1.0e-24;
        let inv_denom = branchless_rsqrt_f32(branchless_select_f32(valid, prod_sq, 1.0));
        let cos_sim = branchless_clamp_f32(dot * inv_denom, -1.0, 1.0);
        Ok(branchless_select_f32(valid, 1.0 - cos_sim, 1.0))
    }

    /// Computes Moving Z-Score Normalized Attention Entropy ($\tilde{H}$) in division-free (`0 fdiv`)
    /// branchless `#![no_std]` mode (Section 3.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `h_surrogate`, `cosine_dist`, or `mu_dist` is `NaN`.
    #[inline]
    fn eval_moving_z_score_entropy(
        &self,
        h_surrogate: f32,
        cosine_dist: f32,
        mu_dist: f32,
        sigma_dist: f32,
        sigma_min: f32,
        alpha_z: f32,
        gamma_mod: f32,
    ) -> GcsoResult<f32> {
        if branchless_is_nan_f32(h_surrogate)
            || branchless_is_nan_f32(cosine_dist)
            || branchless_is_nan_f32(mu_dist)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let min_floor = branchless_max_f32(sigma_min, 1.0e-6);
        let eff_sigma = branchless_max_f32(sigma_dist, min_floor);
        let z_dist = (cosine_dist - mu_dist) * branchless_inv_pos_f32(eff_sigma);
        let relu_z = branchless_sanitize_z_score_f32(z_dist, DEFAULT_MAX_VALID_Z_SCORE);
        let safe_alpha = branchless_relu_f32(alpha_z);
        let safe_gamma = branchless_relu_f32(gamma_mod);

        let d_hat = branchless_tanh_f32(safe_alpha * relu_z);
        let h_tilde = branchless_flush_subnormal_f32(h_surrogate * (1.0 + safe_gamma * d_hat));
        Ok(branchless_select_f32(
            branchless_is_finite_f32(h_tilde),
            h_tilde,
            0.0,
        ))
    }

    /// Evaluates Sliding-Window Entropy Rate ($\Phi_M(t) = \tilde{H}(t) - \tilde{H}(t-M)$) (Section 3.2).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if either entropy input is `NaN`.
    #[inline]
    fn eval_sliding_entropy_rate(
        &self,
        current_entropy: f32,
        past_entropy_m: f32,
    ) -> GcsoResult<f32> {
        if branchless_is_nan_f32(current_entropy) || branchless_is_nan_f32(past_entropy_m) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(branchless_flush_subnormal_f32(
            current_entropy - past_entropy_m,
        ))
    }

    /// Computes the CVoid Coherent Vector Alignment Metric ($\mathcal{C}_{\mathrm{void}} \in [0, 1]$)
    /// in division-free (`0 fdiv`) SIMD mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `key_vector` is empty.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if any element or accumulated norm of `key_vector` is non-finite.
    #[inline]
    fn eval_cvoid_dyadic128(
        &self,
        key_vector: &AlignedSlice32<'_, f32>,
        void_score_out: &mut f32,
    ) -> GcsoResult<()> {
        if key_vector.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let slice = key_vector.as_slice();
        let (sum_abs, sum_sq) = sum_abs_and_sq_f32_simd8(slice);

        if !branchless_is_finite_f32(sum_abs) || !branchless_is_finite_f32(sum_sq) {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        let inv_n = fast_inv_len_f32(slice.len());
        let mean_sq = branchless_flush_subnormal_f32(sum_sq * inv_n);
        let mean_abs = branchless_flush_subnormal_f32(sum_abs * inv_n);
        let valid_rms = mean_sq > 1.0e-24;
        let inv_rms = branchless_select_f32(
            valid_rms,
            branchless_rsqrt_f32(branchless_select_f32(valid_rms, mean_sq, 1.0)),
            0.0,
        );
        *void_score_out = branchless_clamp_f32(mean_abs * inv_rms, 0.0, 1.0);
        Ok(())
    }

    /// Calculates the Eyring-Kramers potential barrier height ($\Delta V = \tau_{\mathrm{eff}} / \mathcal{C}_{\mathrm{void}}$)
    /// in division-free (`0 fdiv`) mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if `void_score <= 1.0e-12`, `tau_eff < 0.0`, or either is non-finite.
    #[inline]
    fn eval_barrier(&self, void_score: f32, tau_eff: f32) -> GcsoResult<f32> {
        if !branchless_is_finite_f32(void_score)
            || !branchless_is_finite_f32(tau_eff)
            || void_score <= 1.0e-12
            || tau_eff < 0.0
        {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }
        Ok(branchless_flush_subnormal_f32(
            tau_eff * branchless_inv_pos_f32(void_score),
        ))
    }

    /// Evaluates the scale-invariant Pitchfork Bifurcation potential barrier height
    /// $\Delta V = \frac{\alpha^2 (\mathcal{C}_{\mathrm{void}} - \tau_{\mathrm{eff}})^2}{4 \beta d_{\mathrm{head}}}$
    /// in division-free (`0 fdiv`) mode (Lemma 3.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if any parameter is non-finite, `beta <= 0.0`,
    /// or `head_dim` is zero or odd.
    #[inline]
    fn eval_barrier_pitchfork(
        &self,
        cvoid: f32,
        tau_eff: f32,
        alpha: f32,
        beta: f32,
        head_dim: u32,
    ) -> GcsoResult<f32> {
        if !branchless_is_finite_f32(cvoid)
            || !branchless_is_finite_f32(tau_eff)
            || !branchless_is_finite_f32(alpha)
            || !branchless_is_finite_f32(beta)
            || beta <= 0.0
            || head_dim == 0
            || (head_dim & 1) != 0
        {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }
        let excess = branchless_relu_f32(cvoid - tau_eff);
        let num = alpha * alpha * excess * excess;
        let den = 4.0 * beta * (head_dim as f32);
        Ok(branchless_flush_subnormal_f32(
            num * branchless_inv_pos_f32(den),
        ))
    }

    /// Evaluates the scale-invariant Eyring-Kramers tunneling escape factor $\exp(-\Delta V / D)$ (Lemma 3.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if any parameter is non-finite, `beta <= 0.0`, or `d0 <= 0.0`.
    #[inline]
    fn eval_eyring_kramers_escape_prob(
        &self,
        cvoid: f32,
        tau_eff: f32,
        alpha: f32,
        beta: f32,
        d0: f32,
    ) -> GcsoResult<f32> {
        if !branchless_is_finite_f32(cvoid)
            || !branchless_is_finite_f32(tau_eff)
            || !branchless_is_finite_f32(alpha)
            || !branchless_is_finite_f32(beta)
            || !branchless_is_finite_f32(d0)
            || beta <= 0.0
            || d0 <= 0.0
        {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }
        Ok(branchless_eyring_kramers_escape(
            cvoid, tau_eff, alpha, beta, d0,
        ))
    }
}

/// Cold-path trait for stateful EDBC parameters and hysteresis management (**Macro Level**).
pub trait EdbcBranchController: EntropyEvaluator {
    /// Returns the currently active decoding branch mode (`0` = Normal, `1` = Pitchfork, `2` = Repulsion).
    fn active_branch_mode(&self) -> u32;

    /// Sets the bifurcation threshold ($\tau_{\mathrm{bifurcation}}$).
    fn set_bifurcation_threshold(&mut self, threshold: f32);

    /// Sets the anti-chattering hysteresis band width ($\delta_{\mathrm{hyst}}$).
    fn set_hysteresis_band(&mut self, band: f32);

    /// Sets the sliding window entropy rate flux threshold ($\tau_{\mathrm{flux}}$).
    fn set_tau_flux(&mut self, tau_flux: f32);

    /// Returns `true` if operating in Normal Convergence Mode (`0`).
    #[inline(always)]
    fn is_normal_mode(&self) -> bool {
        self.active_branch_mode() == 0
    }

    /// Returns `true` if operating in Pitchfork Bifurcation Exploration Mode (`1`).
    #[inline(always)]
    fn is_bifurcation_mode(&self) -> bool {
        self.active_branch_mode() == 1
    }

    /// Returns `true` if operating in Phase-Conjugate Repulsion Mode (`2`).
    #[inline(always)]
    fn is_repulsion_mode(&self) -> bool {
        self.active_branch_mode() == 2
    }

    /// Branchlessly evaluates the next EDBC branch mode (`0`, `1`, `2`).
    #[inline(always)]
    #[must_use]
    fn evaluate_branch_mode_transition(
        &self,
        current_mode: u32,
        moving_z_entropy: f32,
        sliding_flux: f32,
        cvoid_score: f32,
        bifurcation_threshold: f32,
        tau_flux: f32,
        hysteresis_band: f32,
    ) -> u32 {
        let hyst = branchless_relu_f32(hysteresis_band);
        let upper_thresh = bifurcation_threshold + hyst;
        let lower_thresh = branchless_relu_f32(bifurcation_threshold - hyst);

        let surge_triggered = (moving_z_entropy > upper_thresh)
            | (sliding_flux > (tau_flux + hyst))
            | ((moving_z_entropy > bifurcation_threshold) & (cvoid_score >= upper_thresh));
        let stay_active = (current_mode != 0) & (moving_z_entropy > lower_thresh);
        let active_non_normal = surge_triggered | stay_active;

        let active_target = branchless_select_u32(cvoid_score >= bifurcation_threshold, 1, 2);
        branchless_select_u32(active_non_normal, active_target, 0)
    }

    /// Evaluates the next EDBC branch mode (`0`, `1`, `2`) and evolves `tagged_ptr` in $\mathcal{O}(1)$.
    #[inline(always)]
    #[must_use]
    fn evaluate_branch_and_evolve_pointer(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        trail_density: f32,
        dominant_attractor_slot: u8,
        moving_z_entropy: f32,
        sliding_flux: f32,
        cvoid_score: f32,
        bifurcation_threshold: f32,
        tau_flux: f32,
        hysteresis_band: f32,
        high_density_srl_thresh: f32,
    ) -> (u32, gcso_tagged_ptr_t) {
        let next_mode = self.evaluate_branch_mode_transition(
            self.active_branch_mode(),
            moving_z_entropy,
            sliding_flux,
            cvoid_score,
            bifurcation_threshold,
            tau_flux,
            hysteresis_band,
        );
        let is_repulsion = next_mode == 2;
        let low_entropy_skip_thresh = branchless_relu_f32(bifurcation_threshold * 0.5);
        let evolved_ptr = tagged_ptr.evolve_mode_stigmergically(
            trail_density,
            moving_z_entropy,
            is_repulsion,
            dominant_attractor_slot,
            low_entropy_skip_thresh,
            high_density_srl_thresh,
        );
        (next_mode, evolved_ptr)
    }

    /// Evaluates the energy-guided dynamic sampling temperature modifier (`sample_temperature`, Section 3.3).
    #[inline(always)]
    #[must_use]
    fn eval_energy_guided_temperature(
        &self,
        base_temperature: f32,
        moving_z_entropy: f32,
        cvoid_score: f32,
        tau_eff: f32,
        alpha: f32,
        beta: f32,
        d0: f32,
    ) -> f32 {
        let safe_base = branchless_clamp_f32(
            branchless_select_f32(
                branchless_is_finite_f32(base_temperature) & (base_temperature > 0.0),
                base_temperature,
                1.0,
            ),
            0.05,
            5.0,
        );
        let mode = self.active_branch_mode();
        if mode == 0 {
            return safe_base;
        }
        let norm_h =
            branchless_clamp_f32(branchless_relu_f32(moving_z_entropy) * (1.0 / LN_2), 0.0, 1.0);
        if mode == 2 {
            return branchless_clamp_f32(safe_base * (1.0 + 0.25 * norm_h), 0.05, 5.0);
        }
        let escape_p = branchless_eyring_kramers_escape(cvoid_score, tau_eff, alpha, beta, d0);
        let explore_temp = safe_base * (1.0 + 0.35 * escape_p * norm_h);
        let candidate = branchless_select_f32(mode == 1, explore_temp, safe_base);
        branchless_clamp_f32(candidate, 0.05, 5.0)
    }

    /// Injects a deterministic **Metastable Transition Phase Perturbation** ($\boldsymbol{\Delta\theta}_{\mathrm{tunnelling}}$)
    /// across exploration sub-heads (`explore_head_start..MAX_ATTENTION_HEADS`) in 16-head unrolled SWAR mode (Lemma 3.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `explore_head_start > MAX_ATTENTION_HEADS`.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if parameters are non-finite, `beta <= 0.0`,
    /// or `head_dim` is zero or odd.
    #[inline]
    fn inject_metastable_tunneling_perturbation(
        &self,
        phases_inout: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
        seed_ptr: gcso_tagged_ptr_t,
        cvoid_score: f32,
        tau_eff: f32,
        alpha: f32,
        beta: f32,
        head_dim: u32,
        explore_head_start: usize,
    ) -> GcsoResult<f32> {
        if explore_head_start > MAX_ATTENTION_HEADS {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !branchless_is_finite_f32(cvoid_score)
            || !branchless_is_finite_f32(tau_eff)
            || !branchless_is_finite_f32(alpha)
            || !branchless_is_finite_f32(beta)
            || beta <= 0.0
            || head_dim == 0
            || (head_dim & 1) != 0
        {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        let excess = branchless_relu_f32(cvoid_score - tau_eff);
        let safe_alpha = branchless_relu_f32(alpha);
        if excess <= 0.0 || safe_alpha <= 0.0 {
            return Ok(0.0);
        }

        let rsqrt_dim = if head_dim <= 256 {
            branchless_popcount_rsqrt_u32(head_dim)
        } else {
            branchless_rsqrt_f32(head_dim as f32)
        };
        let mu = (safe_alpha * excess) * rsqrt_dim;
        let prod = mu * beta;
        let raw_x_star = if (prod >= 1.0e-36) & (prod <= 1.0e34) {
            mu * branchless_rsqrt_f32(prod)
        } else {
            branchless_sqrt_f32(mu) * branchless_rsqrt_f32(beta)
        };
        let x_star = branchless_clamp_f32(raw_x_star, 0.0, FRAC_PI_2);
        let amp_q7 = branchless_rad_to_q7(x_star);
        if amp_q7 == 0 || explore_head_start == MAX_ATTENTION_HEADS {
            return Ok(x_star);
        }

        let hash = branchless_splitmix64(seed_ptr.raw() ^ 0x9E37_79B9_7F4A_7C15);
        let mut h = explore_head_start;
        while (h & 7) != 0 && h < MAX_ATTENTION_HEADS {
            let pos_branch = ((hash >> h) & 1) == 0;
            let signed_pulse = branchless_select_q7(pos_branch, amp_q7, -amp_q7);
            phases_inout[h] = branchless_q7_add_sat(phases_inout[h], signed_pulse);
            h += 1;
        }

        let pos_u8 = amp_q7 as u8;
        let neg_u8 = (-amp_q7) as u8;
        let pos_w = u64::from(pos_u8).wrapping_mul(0x0101_0101_0101_0101);
        let xor_byte = u64::from(pos_u8 ^ neg_u8);

        while h + 16 <= MAX_ATTENTION_HEADS {
            let shift0 = h >> 3;
            let shift1 = shift0 + 1;
            let bits0 = (hash >> shift0) & 0x0101_0101_0101_0101;
            let bits1 = (hash >> shift1) & 0x0101_0101_0101_0101;
            let pulse0 = pos_w ^ bits0.wrapping_mul(xor_byte);
            let pulse1 = pos_w ^ bits1.wrapping_mul(xor_byte);
            let cur0 = load_q7_word_at(phases_inout, h);
            let cur1 = load_q7_word_at(phases_inout, h + 8);
            store_q7_word_at(phases_inout, h, swar_q7_add_sat_x8(cur0, pulse0));
            store_q7_word_at(phases_inout, h + 8, swar_q7_add_sat_x8(cur1, pulse1));
            h += 16;
        }
        while h + 8 <= MAX_ATTENTION_HEADS {
            let cell_shift = h >> 3;
            let bits_w = (hash >> cell_shift) & 0x0101_0101_0101_0101;
            let pulse_w = pos_w ^ bits_w.wrapping_mul(xor_byte);
            let cur_w = load_q7_word_at(phases_inout, h);
            store_q7_word_at(phases_inout, h, swar_q7_add_sat_x8(cur_w, pulse_w));
            h += 8;
        }
        Ok(x_star)
    }
}

// ===================================================================
// 4. System Capability & Runtime Context Facade Traits
// ===================================================================

/// Trait for querying system ABI versions and runtime compute capabilities (**Boundary / System Layer**).
pub trait SystemCapabilityQuery: BranchlessKernel + Send + Sync {
    /// Returns the semantic ABI version tuple `(major, minor, patch)` (`(0, 1, 1)`).
    #[inline(always)]
    fn abi_version(&self) -> (u32, u32, u32) {
        (
            GCSO_ABI_VERSION_MAJOR,
            GCSO_ABI_VERSION_MINOR,
            GCSO_ABI_VERSION_PATCH,
        )
    }

    /// Returns the static ABI version string (`"0.1.1"`).
    #[inline(always)]
    fn abi_version_string(&self) -> &'static str {
        crate::abi::GCSO_ABI_VERSION_STRING
    }

    /// Queries active compute capability bitmask flags without branching.
    #[inline(always)]
    fn query_capability(&self) -> gcso_capability_flags_t {
        GCSO_CAP_CPU_BASELINE
            | ((cfg!(feature = "cuda") as u64) * GCSO_CAP_CUDA_BACKEND)
            | ((cfg!(feature = "vulkan") as u64) * GCSO_CAP_VULKAN_BACKEND)
            | (((cfg!(target_os = "macos") || cfg!(feature = "metal")) as u64)
                * GCSO_CAP_METAL_BACKEND)
    }

    /// Populates default runtime configuration parameters conforming to GCSO v0.1.1 invariants.
    ///
    /// # Errors
    /// Returns an error status only if a custom implementor rejects the target configuration state.
    #[inline(always)]
    fn init_default_config(&self, config: &mut gcso_config_t) -> GcsoResult<()> {
        *config = gcso_config_t {
            head_dim: 128,
            num_heads: 32,
            paged_block_size: DEFAULT_PAGED_BLOCK_SIZE,
            q7_phase_scale: Q7_PHASE_SCALE_RAD,
            ripa_clamp_max_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
            qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD,
            entropy_singularity_eps: DEFAULT_ENTROPY_EPS,
            max_prompt_anchors: MAX_ANCHORS_PER_FIELD as u32,
            action_hub_capacity: DEFAULT_ACTION_HUB_CAPACITY,
            enable_cuda_warp_shuffle: 1,
            enable_zero_alloc_strict: 1,
            daes_mode: 0,
            reserved_flags: 0,
            reserved: [0; 24],
        };
        Ok(())
    }

    /// Branchlessly validates a [`gcso_config_t`] descriptor against GCSO v0.1.1 architectural invariants.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any dimension, scale, or capacity invariant is violated.
    #[inline(always)]
    fn validate_config_invariants(&self, config: &gcso_config_t) -> GcsoResult<()> {
        let valid_dims = (config.head_dim != 0)
            & ((config.head_dim & 1) == 0)
            & (config.num_heads != 0)
            & ((config.num_heads as usize) <= MAX_ATTENTION_HEADS)
            & ((config.paged_block_size == 16) | (config.paged_block_size == 32));
        let valid_scales = branchless_is_finite_f32(config.q7_phase_scale)
            & (config.q7_phase_scale > 0.0)
            & branchless_is_finite_f32(config.ripa_clamp_max_rad)
            & (config.ripa_clamp_max_rad > 0.0)
            & branchless_is_finite_f32(config.qdps_min_step_rad)
            & (config.qdps_min_step_rad >= 0.0)
            & branchless_is_finite_f32(config.entropy_singularity_eps)
            & (config.entropy_singularity_eps > 0.0);
        let valid_caps = (config.max_prompt_anchors != 0)
            & (config.action_hub_capacity != 0)
            & (config.daes_mode <= 2);

        if valid_dims & valid_scales & valid_caps {
            Ok(())
        } else {
            Err(GCSO_ERROR_INVALID_ARGUMENT)
        }
    }
}

/// Unified Hot-Path Runtime Context trait for token-by-token inference loops (**Macro / Boundary Layer**).
pub trait GcsoRuntimeContext: BranchlessKernel + Send + Sync {
    /// Resets transient phase accumulators and step counters without freeing pre-allocated tables.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_STATE`] if the context is uninitialized or locked.
    fn reset(&mut self) -> GcsoResult<()>;

    /// Registers a natural language system prompt text anchor in phase space as the primary baseline attractor.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `prompt` is empty or `weight` is non-finite.
    fn set_system_prompt_anchor(&mut self, prompt: &str, weight: f32) -> GcsoResult<()>;

    /// Executes a per-token Hot Path inference step in zero-allocation mode ($C_{\mathrm{alloc}} = 0$).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `query_tensor` and `key_tensor` alias or have
    /// lengths smaller than `head_dim * num_heads`.
    fn step_token(
        &mut self,
        token_id: u32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: Option<&mut gcso_pointer_trail_t>,
    ) -> GcsoResult<()>;

    /// Serializes active runtime context state into a `.gcso` binary snapshot buffer.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `buffer.len() < *required_size`.
    fn serialize(&self, buffer: &mut [u8], required_size: &mut usize) -> GcsoResult<()>;

    /// Deserializes a `.gcso` binary snapshot buffer to restore context state in-place.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] or [`crate::abi::GCSO_ERROR_VERSION_MISMATCH`]
    /// if the snapshot header magic, version (`0x00000101`), or payload checksum is invalid.
    fn deserialize(&mut self, buffer: &[u8]) -> GcsoResult<()>;

    /// Seeks to a specific token position in the PPRC keyframe KV cache without forward passes.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_PPRC_SEEK_FAILED`] or [`GCSO_ERROR_INVALID_ARGUMENT`]
    /// if `token_index` is out of bounds.
    fn seek_to_token(
        &mut self,
        token_index: u32,
        header_out: &mut gcso_pprc_keyframe_header_t,
    ) -> GcsoResult<()>;
}

// ===================================================================
// 5. Unified Macro Attractor Field & 8-Lane Working Space Traits
// ===================================================================

/// Cold-path trait for macro attractor field steering, zero-hyperparameter phase-proximity gravity,
/// and repulsion control on a **Single Unified Phase Manifold ($SO(2)^{d_{\mathrm{head}}/2}$)**
/// without an architectural 8-slot domain restriction (**Macro Level**).
pub trait AttractorField: BranchlessKernel + Send + Sync {
    /// Returns the total number of registered topological anchors (`<= MAX_ANCHORS_PER_FIELD`).
    fn anchor_count(&self) -> u32;

    /// Evaluates evaporative pressure decay across active anchors.
    fn apply_evaporative_decay(&mut self);

    /// Returns the total potential energy accumulated across active anchors in this field.
    fn total_potential_energy(&self) -> f32;

    /// Returns the current accumulator pressure ratio (`active_count / MAX_ANCHORS_PER_FIELD`).
    fn pressure_ratio(&self) -> f32;

    /// Evaporates and removes the weakest non-crystallized anchor slot under accumulator pressure.
    fn evaporate_weakest_anchor(&mut self) -> Option<u32>;

    /// Registers a system prompt anchor in phase space.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `prompt` is empty or `weight` is non-finite.
    /// Returns [`crate::abi::GCSO_ERROR_OUT_OF_MEMORY`] if all 64 slots are occupied by crystallized anchors.
    fn add_system_prompt_anchor(&mut self, prompt: &str, weight: f32) -> GcsoResult<u32>;

    /// Registers a dense feature embedding vector as a continuous attractor anchor.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `embedding` is empty, `anchor_type` is invalid, or `weight` is non-finite.
    fn add_embedding_anchor(
        &mut self,
        embedding: &AlignedSlice32<'_, f32>,
        anchor_type: gcso_anchor_type_t,
        weight: f32,
    ) -> GcsoResult<u32>;

    /// Registers a raw topological anchor point in the attractor field.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `dim == 0`, `vec.len() < dim`, or `anchor_type` is invalid.
    fn add_anchor(
        &mut self,
        anchor_type: gcso_anchor_type_t,
        vec: &AlignedSlice32<'_, f32>,
        dim: usize,
    ) -> GcsoResult<u32>;

    /// Injects a phase-conjugate repulsion vector ($-\boldsymbol{\Delta\theta}$) into the field.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `repulsion_deltas` is empty or `gain` is non-finite.
    fn inject_phase_repulsion(
        &mut self,
        repulsion_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        gain: f32,
    ) -> GcsoResult<()>;

    /// Deterministically projects a UTF-8 prompt string into a 64-head Q7 phase profile using
    /// 8-byte SplitMix64 bijective hashing in $\mathcal{O}(L)$ with zero heap allocation.
    #[inline]
    #[must_use]
    fn hash_prompt_to_phase_profile(
        &self,
        prompt: &str,
        domain_salt: u32,
    ) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        let bytes = prompt.as_bytes();
        let mut seed = 0xCBF2_9CE4_8422_2325_u64 ^ (u64::from(domain_salt) << 32);
        let mut i = 0usize;
        while i < bytes.len() {
            seed = branchless_splitmix64(
                seed ^ u64::from(bytes[i]).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            i += 1;
        }
        let mut out = [0i8; MAX_ATTENTION_HEADS];
        let mut w = 0usize;
        while w < 8 {
            seed = branchless_splitmix64(seed.wrapping_add((w as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)));
            let shifted = (seed >> 1) & 0x7F7F_7F7F_7F7F_7F7F;
            let signed_w = shifted ^ (seed & 0x8080_8080_8080_8080);
            store_q7_word_at(&mut out, w * 8, signed_w);
            w += 1;
        }
        out
    }

    /// Evaluates **Zero-Hyperparameter Unified-Field Superposition** across an arbitrary slice of
    /// anchors (`anchor_phases`, `anchor_base_energies`) on the single phase manifold in strictly
    /// constant time (`0` data-dependent branches on `base_e > 0.0`, `0 fdiv`) via
    /// [`super::branchless_q7_unified_field_fold_64`], followed by branchless 8-lane SWAR repulsion subtraction.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `anchor_phases.len() != anchor_base_energies.len()`.
    #[inline]
    fn evaluate_unified_field_superposition_from_slices(
        &self,
        trail: &gcso_pointer_trail_t,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_gain: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
        composite_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<f32> {
        if anchor_phases.len() != anchor_base_energies.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let ref_w = load_q7_words_64(&trail.accumulated_phase_delta);
        let (mut comp_w, total_pull, _mean_coh, _dom_idx, _act_cnt) =
            branchless_q7_unified_field_fold_words_with_init_weight_64(
                &ref_w,
                0.0,
                anchor_phases,
                anchor_base_energies,
            );

        if let Some((rep_phases, rep_gain)) = repulsion_phases_and_gain {
            let clean_gain = branchless_clamp_f32(branchless_relu_f32(rep_gain), 0.0, 1.0);
            let scale_u7 = (clean_gain * 128.0 + 0.5) as u8;
            if scale_u7 > 0 {
                let s = branchless_select_u8(scale_u7 > 128, 128, scale_u7) as u64;
                let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(0x0101_0101_0101_0101);
                let rep_w = load_q7_words_64(rep_phases);
                let mut c = 0usize;
                while c < 8 {
                    let sr = swar_q7_scale_le128_with_bias_x8(rep_w[c], s, bias_restore);
                    comp_w[c] = swar_q7_sub_sat_x8(comp_w[c], sr);
                    c += 1;
                }
            }
        }
        store_q7_words_64(composite_phase_out, &comp_w);

        Ok(branchless_min_f32(
            branchless_flush_subnormal_f32(total_pull),
            MAX_ATTRACTOR_PULL_FORCE,
        ))
    }

    /// Steers `trail` in-place directly on the **Single Unified Attractor Manifold** and returns
    /// `(final_pull_force, mean_coherence, active_anchor_count, post_words)` in 8 `u64` SWAR registers (`0` reload).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `anchor_phases.len() != anchor_base_energies.len()`
    /// or `repulsion_phases_and_strength` contains a non-finite strength.
    #[inline]
    fn steer_trail_on_unified_field_from_slices_to_words(
        &self,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32, u32, [u64; NUM_SWARM_MICRO_CELLS])> {
        if anchor_phases.len() != anchor_base_energies.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if let Some((_, strength)) = repulsion_phases_and_strength {
            if !branchless_is_finite_f32(strength) {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
        }
        Ok(trail.apply_unified_field_anchor_slice_resonance_to_words(
            anchor_phases,
            anchor_base_energies,
            repulsion_phases_and_strength,
        ))
    }

    /// Steers `trail` in-place directly on the **Single Unified Attractor Manifold** across an
    /// arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) and optional repulsive
    /// phase inhibitor (`repulsion_phases_and_strength`), automatically binding the dominant
    /// resonant anchor index into `trail.target_anchor_id`.
    ///
    /// Returns `(final_pull_force, mean_coherence, active_anchor_count)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `anchor_phases.len() != anchor_base_energies.len()`
    /// or `repulsion_phases_and_strength` contains a non-finite strength.
    #[inline(always)]
    fn steer_trail_on_unified_field_from_slices(
        &self,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32, u32)> {
        let (pull, coh, cnt, _words) = self.steer_trail_on_unified_field_from_slices_to_words(
            anchor_phases,
            anchor_base_energies,
            repulsion_phases_and_strength,
            trail,
        )?;
        Ok((pull, coh, cnt))
    }

    /// Evaluates composite potential force pull and combines 64-head phase shifts for a given trail.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_STATE`] if a custom implementor encounters an uninitialized field.
    #[inline]
    fn evaluate_field_force(
        &self,
        trail: &gcso_pointer_trail_t,
        composite_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<f32> {
        composite_phase_out.copy_from_slice(&trail.accumulated_phase_delta);
        let safe_pull = branchless_min_f32(
            branchless_flush_subnormal_f32(branchless_relu_f32(trail.attractor_pull_force)),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        Ok(safe_pull)
    }

    /// Synthesizes a Product of Lie Group Rotations $\mathbf{R}_{\mathrm{total}} = \prod_{l \in \mathrm{Path}} \mathbf{R}_l(\boldsymbol{\Delta\theta}_l)$
    /// across a hierarchical topic phase tree in $\mathcal{O}(L \cdot H)$ using 8-lane SWAR saturating additions (Section 5.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `hierarchy_levels` is empty.
    #[inline]
    fn compose_lie_group_hierarchy_rotations(
        &self,
        hierarchy_levels: &[&[gcso_q7_t; MAX_ATTENTION_HEADS]],
        total_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<()> {
        if hierarchy_levels.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let mut acc_w = [0u64; 8];
        for &level_deltas in hierarchy_levels {
            let lw = load_q7_words_64(level_deltas);
            acc_w[0] = swar_q7_add_sat_x8(acc_w[0], lw[0]);
            acc_w[1] = swar_q7_add_sat_x8(acc_w[1], lw[1]);
            acc_w[2] = swar_q7_add_sat_x8(acc_w[2], lw[2]);
            acc_w[3] = swar_q7_add_sat_x8(acc_w[3], lw[3]);
            acc_w[4] = swar_q7_add_sat_x8(acc_w[4], lw[4]);
            acc_w[5] = swar_q7_add_sat_x8(acc_w[5], lw[5]);
            acc_w[6] = swar_q7_add_sat_x8(acc_w[6], lw[6]);
            acc_w[7] = swar_q7_add_sat_x8(acc_w[7], lw[7]);
        }
        store_q7_words_64(total_phase_out, &acc_w);
        Ok(())
    }

    /// Synthesizes a depth-weighted Product of Lie Group Rotations
    /// $\mathbf{R}_{\mathrm{total}} = \prod_{l \in \mathrm{Path}} \mathbf{R}_l(s_l \cdot \boldsymbol{\Delta\theta}_l)$ (Section 5.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `hierarchy_levels` is empty or its length does
    /// not match `level_scales_u7.len()`.
    #[inline]
    fn compose_weighted_lie_group_hierarchy(
        &self,
        hierarchy_levels: &[&[gcso_q7_t; MAX_ATTENTION_HEADS]],
        level_scales_u7: &[u8],
        total_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<()> {
        if hierarchy_levels.is_empty() || hierarchy_levels.len() != level_scales_u7.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let mut acc_w = [0u64; 8];
        for (&level_deltas, &scale_u7) in hierarchy_levels.iter().zip(level_scales_u7.iter()) {
            if scale_u7 == 0 {
                continue;
            }
            let lw = load_q7_words_64(level_deltas);
            let gt_128 = scale_u7 > 128;
            let s_sub = branchless_select_u8(gt_128, scale_u7.wrapping_sub(128), scale_u7) as u64;
            let bias_restore = (128u64.wrapping_sub(s_sub)).wrapping_mul(0x0101_0101_0101_0101);
            let extra_mask = (gt_128 as u64).wrapping_neg();
            let mut c = 0usize;
            while c < 8 {
                let w = lw[c];
                let sw = swar_q7_add_sat_x8(
                    swar_q7_scale_le128_with_bias_x8(w, s_sub, bias_restore),
                    w & extra_mask,
                );
                acc_w[c] = swar_q7_add_sat_x8(acc_w[c], sw);
                c += 1;
            }
        }
        store_q7_words_64(total_phase_out, &acc_w);
        Ok(())
    }

    /// Aggregates high-density pointer trails bottom-up to macro-crystallize new dynamic anchors.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_STATE`] if crystallization cannot proceed.
    fn aggregate_bottom_up(&mut self) -> GcsoResult<u32>;

    /// Clears all registered topological anchors from the attractor field.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_STATE`] if the field is locked.
    fn clear_anchors(&mut self) -> GcsoResult<()>;
}

/// Cold-path & Mezzo-path trait managing the optional **8-Lane Hardware Working Cache Window
/// (`MAX_ATTRACTOR_DOMAINS = 8`)** projected from the unified attractor manifold with
/// Zero-Hyperparameter Autonomous Weight Rebalancing, Cross-Lane Synergy, Liminal Beat
/// Perturbation, and Mutual Inhibition (**Macro Space Level**).
pub trait MultiDomainAttractorSpace: BranchlessKernel + Send + Sync {
    /// Advances global step counters across all working lanes and applies evaporative pressure decay.
    fn advance_step(&mut self);

    /// Sets the relative weighting multiplier scalar for a specific working lane (`0..=7`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `domain_id >= 8` or `weight` is negative/non-finite.
    fn set_domain_weight(&mut self, domain_id: u32, weight: f32) -> GcsoResult<()>;

    /// Enables or disables a specific working lane (`0..=7`) via bitmask.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `domain_id >= 8`.
    fn set_domain_enabled(&mut self, domain_id: u32, enabled: bool) -> GcsoResult<()>;

    /// Identifies the dominant working lane (`0..=7`) currently holding maximum potential energy.
    fn find_dominant_domain(&self) -> (u32, f32);

    /// Evaluates unified-field superposition of gravitational forces and composite phase shifts in zero-allocation mode.
    fn evaluate_multi_domain_superposition(
        &self,
        trail: &gcso_pointer_trail_t,
        composite_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32;

    /// Clears all registered topological anchors across all 8 working lanes on the unified field.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_STATE`] if any lane accumulator cannot be cleared.
    fn clear_all_domains(&mut self) -> GcsoResult<()>;

    /// Returns the active $8 \times 8$ cross-lane coupling bitmatrix (`u64`).
    #[inline(always)]
    fn slot_coupling_matrix(&self) -> u64 {
        DEFAULT_SLOT_COUPLING_MATRIX_8X8
    }

    /// Evaluates the gravitational pull force and 64-head Q7 phase profile of a single working
    /// lane `domain_id` (`0..=7`) for `trail`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `domain_id >= MAX_ATTRACTOR_DOMAINS` (`8`).
    #[inline]
    fn evaluate_slot_force(
        &self,
        domain_id: u32,
        trail: &gcso_pointer_trail_t,
        slot_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<f32> {
        if (domain_id as usize) >= MAX_ATTRACTOR_DOMAINS {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(self.evaluate_multi_domain_superposition(trail, slot_phase_out))
    }

    /// Projects all 8 Working Lanes (`0..=7`) for `reference_trail` into a contiguous
    /// 512-byte (`[[gcso_q7_t; 64]; 8]`) phase bank (`bank_out`) and bounded weight array (`weights_out`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `bank_out.len() < 512`.
    #[inline]
    fn project_slot_phase_bank(
        &self,
        reference_trail: &gcso_pointer_trail_t,
        bank_out: &mut AlignedSliceMut16<'_, gcso_q7_t>,
        weights_out: &mut [f32; MAX_ATTRACTOR_DOMAINS],
    ) -> GcsoResult<()> {
        let bank_2d = bank_out.as_slot_bank_mut::<MAX_ATTRACTOR_DOMAINS, MAX_ATTENTION_HEADS>()?;
        for slot_id in 0..MAX_ATTRACTOR_DOMAINS {
            let raw_pull =
                self.evaluate_slot_force(slot_id as u32, reference_trail, &mut bank_2d[slot_id])?;
            weights_out[slot_id] = branchless_min_f32(
                branchless_flush_subnormal_f32(branchless_relu_f32(raw_pull)),
                MAX_ATTRACTOR_PULL_FORCE,
            );
        }
        Ok(())
    }

    /// Autonomously rebalances all 8 working lane weights (`0..=7`) from each lane's phase
    /// proximity resonance with `reference_trail` (`branchless_q7_resonant_pull_64`) in
    /// division-free (`0 fdiv`) mode, eliminating manual per-domain hyperparameter tuning.
    ///
    /// # Errors
    /// Propagates any error from [`Self::evaluate_slot_force`] or [`Self::set_domain_weight`].
    #[inline]
    fn rebalance_working_lane_weights_from_trail(
        &mut self,
        reference_trail: &gcso_pointer_trail_t,
    ) -> GcsoResult<[f32; MAX_ATTRACTOR_DOMAINS]> {
        let mut pulls = [0.0_f32; MAX_ATTRACTOR_DOMAINS];
        let mut max_pull = 0.0_f32;
        let mut scratch_phase = [0i8; MAX_ATTENTION_HEADS];
        let ref_w = load_q7_words_64(&reference_trail.accumulated_phase_delta);
        let ref_is_uniform = is_word_array_uniform_64(&ref_w);
        let ref_nonzero = if ref_is_uniform {
            ref_w[0] != 0
        } else {
            ((ref_w[0] | ref_w[1]) | (ref_w[2] | ref_w[3]) | (ref_w[4] | ref_w[5]) | (ref_w[6] | ref_w[7])) != 0
        };

        for (slot_id, pull_slot) in pulls.iter_mut().enumerate() {
            let raw_pull =
                self.evaluate_slot_force(slot_id as u32, reference_trail, &mut scratch_phase)?;
            let clean_base = branchless_flush_subnormal_f32(branchless_relu_f32(raw_pull));
            let res_pull = if clean_base > 0.0 {
                let coh = if ref_nonzero {
                    let lane_w = load_q7_words_64(&scratch_phase);
                    coherence_from_word_arrays_with_uniform_flag_64(&ref_w, ref_is_uniform, &lane_w)
                } else {
                    0.0
                };
                let proximity_gain = branchless_clamp_f32(1.0 + coh, 0.25, 2.0);
                branchless_flush_subnormal_f32(clean_base * proximity_gain)
            } else {
                0.0
            };
            *pull_slot = res_pull;
            max_pull = branchless_select_f32(res_pull > max_pull, res_pull, max_pull);
        }

        let mut weights = [1.0_f32; MAX_ATTRACTOR_DOMAINS];
        let has_max = max_pull > 1.0e-6;
        let inv_max = branchless_inv_pos_f32(branchless_select_f32(has_max, max_pull, 1.0));

        for slot_id in 0..MAX_ATTRACTOR_DOMAINS {
            let p = pulls[slot_id];
            let norm_w = branchless_clamp_f32(0.25 + 1.75 * (p * inv_max), 0.25, 2.0);
            let w = branchless_select_f32(has_max & (p > 0.0), norm_w, 1.0);
            self.set_domain_weight(slot_id as u32, w)?;
            weights[slot_id] = w;
        }
        Ok(weights)
    }

    /// Evaluates unified-field superposition across the 8 working lanes by projecting the 8-lane
    /// phase bank, applying mutual inhibition from Lane 3 (`ATTRACTOR_SLOT_REPULSION`), and
    /// folding all coupled lanes via [`branchless_q7_synergy_fuse_64`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `synergy_gain` or `inhibition_strength` is non-finite.
    #[inline]
    fn evaluate_unified_superposition_from_working_bank(
        &self,
        trail: &gcso_pointer_trail_t,
        synergy_gain: f32,
        inhibition_strength: f32,
        composite_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<f32> {
        if !branchless_is_finite_f32(synergy_gain) || !branchless_is_finite_f32(inhibition_strength)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let mut bank =
            AlignedBuffer16::<gcso_q7_t, { MAX_ATTRACTOR_DOMAINS * MAX_ATTENTION_HEADS }>::zeroed();
        let mut pulls = [0.0_f32; MAX_ATTRACTOR_DOMAINS];
        self.project_slot_phase_bank(trail, &mut bank.as_aligned_slice_mut(), &mut pulls)?;
        let bank_2d = bank
            .as_aligned_slice_mut()
            .into_slot_bank_mut::<MAX_ATTRACTOR_DOMAINS, MAX_ATTENTION_HEADS>()?;

        let rep_idx = usize::from(ATTRACTOR_SLOT_REPULSION);
        let rep_pull = pulls[rep_idx];
        let rep_w = load_q7_words_64(&bank_2d[rep_idx]);
        let apply_rep = (rep_pull > 1.0e-6) & (inhibition_strength > 0.0);

        let mut running_w = [0u64; 8];
        let mut running_pull = 0.0_f32;
        let mut primary_slot = 0u32;
        let coupling = self.slot_coupling_matrix();

        for idx in 0..MAX_ATTRACTOR_DOMAINS {
            if idx == rep_idx || pulls[idx] <= 0.0 {
                continue;
            }
            let mut lane_w = load_q7_words_64(&bank_2d[idx]);
            let mut lane_pull = pulls[idx];
            if apply_rep {
                lane_pull = branchless_q7_inhibit_words_64(
                    &mut lane_w,
                    lane_pull,
                    &rep_w,
                    inhibition_strength,
                );
            }
            if running_pull <= 0.0 {
                running_w = lane_w;
                running_pull = lane_pull;
                primary_slot = idx as u32;
            } else {
                let coupled = is_slot_pair_coupled(coupling, primary_slot, idx as u32);
                let eff_gain = branchless_select_f32(coupled, synergy_gain, 0.0);
                let (fused_w, new_pull, _coh) = branchless_q7_synergy_fuse_words_64(
                    &running_w,
                    running_pull,
                    &lane_w,
                    lane_pull,
                    eff_gain,
                );
                running_w = fused_w;
                running_pull = new_pull;
            }
        }

        if running_pull <= 0.0 {
            *composite_phase_out = trail.accumulated_phase_delta;
            return Ok(0.0);
        }

        store_q7_words_64(composite_phase_out, &running_w);
        Ok(branchless_min_f32(
            branchless_flush_subnormal_f32(running_pull),
            MAX_ATTRACTOR_PULL_FORCE,
        ))
    }

    /// Evaluates non-linear **Cross-Slot Synergy** and **Liminal Beat Interference** between
    /// two working lanes (`slot_a`, `slot_b`) gated by `coupling_matrix_8x8` (`u64`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `slot_a >= 8`, `slot_b >= 8`, or `synergy_gain` is non-finite.
    #[inline]
    fn evaluate_cross_slot_synergy(
        &self,
        slot_a: u32,
        slot_b: u32,
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
        trail: &gcso_pointer_trail_t,
        synergy_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<(f32, f32)> {
        if (slot_a as usize) >= MAX_ATTRACTOR_DOMAINS
            || (slot_b as usize) >= MAX_ATTRACTOR_DOMAINS
            || !branchless_is_finite_f32(synergy_gain)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let mut phase_a = [0i8; MAX_ATTENTION_HEADS];
        let mut phase_b = [0i8; MAX_ATTENTION_HEADS];
        let pull_a = self.evaluate_slot_force(slot_a, trail, &mut phase_a)?;
        let pull_b = self.evaluate_slot_force(slot_b, trail, &mut phase_b)?;

        let coupled = is_slot_pair_coupled(coupling_matrix_8x8, slot_a, slot_b);
        let effective_gain = branchless_select_f32(coupled, synergy_gain, 0.0);

        let (raw_pull, coherence) = branchless_q7_synergy_fuse_64(
            &phase_a,
            pull_a,
            &phase_b,
            pull_b,
            effective_gain,
            synergy_phase_out,
        );
        let bounded_pull = branchless_min_f32(
            branchless_flush_subnormal_f32(raw_pull),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        Ok((bounded_pull, coherence))
    }

    /// Evaluates branchless **Mutual Inhibition** on `target_slot` by an opposing `inhibitor_slot`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `target_slot >= 8`, `inhibitor_slot >= 8`,
    /// or `inhibition_strength` is non-finite.
    #[inline]
    fn evaluate_mutual_slot_inhibition(
        &self,
        target_slot: u32,
        inhibitor_slot: u32,
        inhibition_strength: f32,
        trail: &gcso_pointer_trail_t,
        inhibited_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<f32> {
        if (target_slot as usize) >= MAX_ATTRACTOR_DOMAINS
            || (inhibitor_slot as usize) >= MAX_ATTRACTOR_DOMAINS
            || !branchless_is_finite_f32(inhibition_strength)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let pull_target = self.evaluate_slot_force(target_slot, trail, inhibited_phase_out)?;
        let mut inhibitor_phase = [0i8; MAX_ATTENTION_HEADS];
        let pull_inhibitor =
            self.evaluate_slot_force(inhibitor_slot, trail, &mut inhibitor_phase)?;

        let eff_inhibition =
            branchless_select_f32(pull_inhibitor > 1.0e-9, inhibition_strength, 0.0);
        let damped = branchless_q7_inhibit_64(
            inhibited_phase_out,
            pull_target,
            &inhibitor_phase,
            eff_inhibition,
        );
        Ok(branchless_min_f32(
            branchless_flush_subnormal_f32(damped),
            MAX_ATTRACTOR_PULL_FORCE,
        ))
    }

    /// Steers `trail` in-place by extracting the primary Target Working Lane ID (`0..=7`)
    /// from `trail.current_ptr`, fusing it synergistically with `secondary_slot`, and applying
    /// mutual inhibition from [`ATTRACTOR_SLOT_REPULSION`] (`lane 3`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `secondary_slot >= 8` or gains are non-finite.
    #[inline]
    fn steer_trail_by_pointer_slot_synergy(
        &self,
        secondary_slot: u32,
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
        inhibition_strength: f32,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32)> {
        if (secondary_slot as usize) >= MAX_ATTRACTOR_DOMAINS
            || !branchless_is_finite_f32(synergy_gain)
            || !branchless_is_finite_f32(inhibition_strength)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let primary_slot = u32::from(trail.target_attractor_slot() & ATTRACTOR_SLOT_ID_MASK);
        let mut syn_phase = [0i8; MAX_ATTENTION_HEADS];
        let (syn_pull, coherence) = self.evaluate_cross_slot_synergy(
            primary_slot,
            secondary_slot,
            coupling_matrix_8x8,
            synergy_gain,
            trail,
            &mut syn_phase,
        )?;

        let coupled = is_slot_pair_coupled(coupling_matrix_8x8, primary_slot, secondary_slot);
        branchless_q7_accumulate_64(&mut trail.accumulated_phase_delta, &syn_phase);
        let raw_pull = branchless_flush_subnormal_f32(
            branchless_relu_f32(trail.attractor_pull_force) + syn_pull,
        );
        trail.attractor_pull_force = branchless_min_f32(raw_pull, MAX_ATTRACTOR_PULL_FORCE);

        let is_resonant = coupled & (coherence > 0.25) & (syn_pull > 0.0);
        let is_liminal = coupled & (coherence < -0.10) & (syn_pull > 0.0);
        trail.flags |= branchless_select_u32(is_resonant, TRAIL_FLAG_SYNERGY_RESONANCE, 0)
            | branchless_select_u32(is_liminal, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, 0);

        if inhibition_strength > 0.0 {
            let mut rep_phase = [0i8; MAX_ATTENTION_HEADS];
            let rep_pull = self.evaluate_slot_force(
                u32::from(ATTRACTOR_SLOT_REPULSION),
                trail,
                &mut rep_phase,
            )?;
            let eff_inh = branchless_select_f32(rep_pull > 1.0e-9, inhibition_strength, 0.0);
            let damped = trail.apply_mutual_slot_inhibition(&rep_phase, eff_inh);
            return Ok((damped, coherence));
        }

        Ok((trail.attractor_pull_force, coherence))
    }

    /// Steers `trail` in-place across the entire coupled neighborhood of its embedded primary
    /// Target Working Lane ID (`0..=7`) and returns
    /// `(damped_pull_force, mean_coherence, active_neighbor_count, post_words)` in 8 `u64` SWAR registers (`0` reload).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `synergy_gain` or `inhibition_strength` is non-finite.
    #[inline]
    fn steer_trail_by_coupled_neighborhood_to_words(
        &self,
        synergy_gain: f32,
        inhibition_strength: f32,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32, u32, [u64; NUM_SWARM_MICRO_CELLS])> {
        if !branchless_is_finite_f32(synergy_gain) || !branchless_is_finite_f32(inhibition_strength)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let mut bank =
            AlignedBuffer16::<gcso_q7_t, { MAX_ATTRACTOR_DOMAINS * MAX_ATTENTION_HEADS }>::zeroed();
        let mut weights = [0.0_f32; MAX_ATTRACTOR_DOMAINS];
        self.project_slot_phase_bank(trail, &mut bank.as_aligned_slice_mut(), &mut weights)?;
        let bank_2d = bank
            .as_aligned_slice()
            .as_slot_bank::<MAX_ATTRACTOR_DOMAINS, MAX_ATTENTION_HEADS>()?;
        Ok(trail.apply_neighborhood_synergy_from_bank_to_words(
            bank_2d,
            &weights,
            self.slot_coupling_matrix(),
            synergy_gain,
            inhibition_strength,
        ))
    }

    /// Steers `trail` in-place across the entire coupled neighborhood of its embedded primary
    /// Target Working Lane ID (`0..=7`) by projecting all 8 lanes and delegating to
    /// [`StigmergicTrailOps::apply_neighborhood_synergy_from_bank`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `synergy_gain` or `inhibition_strength` is non-finite.
    #[inline(always)]
    fn steer_trail_by_coupled_neighborhood(
        &self,
        synergy_gain: f32,
        inhibition_strength: f32,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32, u32)> {
        let (pull, coh, neighbors, _words) =
            self.steer_trail_by_coupled_neighborhood_to_words(synergy_gain, inhibition_strength, trail)?;
        Ok((pull, coh, neighbors))
    }
}

// ===================================================================
// 6. Persona Patch, L2P-SVD, PPRC Keyframe Cache, DNP & ZIMMS Storage
// ===================================================================

/// Cold-path trait for dynamic persona phase modulation patches without altering base weights (**Macro Level**).
pub trait PersonaPatcher: BranchlessKernel + Send + Sync {
    /// Applies a binary persona phase modulation patch to the runtime context.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `patch_data` is empty or malformed.
    fn apply_patch(&mut self, patch_data: &[u8]) -> GcsoResult<()>;

    /// Executes **Persistent Topological Mode Quantization** (Section 3.3) in row-major order,
    /// returning `Ok((0.0, 0.0))` with zeroed `profile_out` if total trail weight is zero.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `coherent_trails` is empty or `min_coherence_r_bar` is non-finite.
    #[inline]
    fn quantize_persistent_topological_mode(
        &self,
        coherent_trails: &[gcso_pointer_trail_t],
        min_coherence_r_bar: f32,
        profile_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<(f32, f32)> {
        if coherent_trails.is_empty() || !branchless_is_finite_f32(min_coherence_r_bar) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let mut candidate = [0i8; MAX_ATTENTION_HEADS];
        let candidate_words: [u64; 8];
        if coherent_trails.len() == 1 {
            let trail = &coherent_trails[0];
            let d_bounded =
                branchless_clamp_f32(trail.stigmergic_density, 0.0, MAX_STIGMERGIC_DENSITY);
            let f_bounded =
                branchless_clamp_f32(trail.attractor_pull_force, 1.0, MAX_ATTRACTOR_PULL_FORCE);
            let w = branchless_flush_subnormal_f32(d_bounded * f_bounded);
            if w <= 1.0e-10 {
                *profile_out = [0i8; MAX_ATTENTION_HEADS];
                return Ok((0.0, 0.0));
            }
            candidate = trail.accumulated_phase_delta;
            candidate_words = *trail_phase_words_ref(trail);
        } else {
            let mut sum_s = [0.0_f32; MAX_ATTENTION_HEADS];
            let mut sum_c = [0.0_f32; MAX_ATTENTION_HEADS];
            let mut word_s = [0.0_f32; 8];
            let mut word_c = [0.0_f32; 8];
            let mut total_w = 0.0_f32;
            let mut all_byte_uniform = true;
            let mut all_word_uniform = true;
            let mut uniform_s = 0.0_f32;
            let mut uniform_c = 0.0_f32;

            for trail in coherent_trails {
                let d_bounded =
                    branchless_clamp_f32(trail.stigmergic_density, 0.0, MAX_STIGMERGIC_DENSITY);
                let f_bounded =
                    branchless_clamp_f32(trail.attractor_pull_force, 1.0, MAX_ATTRACTOR_PULL_FORCE);
                let w = branchless_flush_subnormal_f32(d_bounded * f_bounded);
                if w > 0.0 {
                    total_w += w;
                    let phases = &trail.accumulated_phase_delta;
                    let tw = *trail_phase_words_ref(trail);
                    let w0 = tw[0];
                    if is_word_array_uniform_64(&tw) {
                        if w0 == (w0 & 0xFF).wrapping_mul(0x0101_0101_0101_0101) {
                            let (s, c) = branchless_sincos_q7(w0 as gcso_q7_t);
                            uniform_s += w * s;
                            uniform_c += w * c;
                        } else {
                            all_byte_uniform = false;
                            let b = w0.to_le_bytes();
                            let mut lane = 0usize;
                            while lane < 8 {
                                let (s, c) = branchless_sincos_q7(b[lane] as gcso_q7_t);
                                word_s[lane] += w * s;
                                word_c[lane] += w * c;
                                lane += 1;
                            }
                        }
                    } else {
                        all_byte_uniform = false;
                        all_word_uniform = false;
                        for h in 0..MAX_ATTENTION_HEADS {
                            let (s, c) = branchless_sincos_q7(phases[h]);
                            sum_s[h] += w * s;
                            sum_c[h] += w * c;
                        }
                    }
                }
            }
            if total_w <= 1.0e-10 {
                *profile_out = [0i8; MAX_ATTENTION_HEADS];
                return Ok((0.0, 0.0));
            }
            if all_byte_uniform {
                let mag_sq = uniform_s * uniform_s + uniform_c * uniform_c;
                let has_mag = mag_sq > 1.0e-20;
                let q7 = branchless_select_q7(
                    has_mag,
                    branchless_atan2_to_q7_fast(uniform_s, uniform_c),
                    0,
                );
                let r_bar = branchless_select_f32(has_mag, 1.0, 0.0);
                let kappa_c = branchless_von_mises_kappa(r_bar);
                let keep = r_bar >= branchless_clamp_f32(min_coherence_r_bar, 0.0, 1.0);
                *profile_out = [branchless_select_q7(keep, q7, 0); MAX_ATTENTION_HEADS];
                return Ok((r_bar, kappa_c));
            } else if all_word_uniform {
                let mut w_bytes = [0u8; 8];
                let mut lane = 0usize;
                while lane < 8 {
                    let s = word_s[lane] + uniform_s;
                    let c = word_c[lane] + uniform_c;
                    let mag_sq = s * s + c * c;
                    let q7 = branchless_select_q7(
                        mag_sq > 1.0e-20,
                        branchless_atan2_to_q7_fast(s, c),
                        0,
                    );
                    w_bytes[lane] = q7 as u8;
                    lane += 1;
                }
                let word0 = u64::from_le_bytes(w_bytes);
                candidate_words = [word0; 8];
                store_q7_words_64(&mut candidate, &candidate_words);
            } else {
                for h in 0..MAX_ATTENTION_HEADS {
                    let lane = h & 7;
                    let s = sum_s[h] + word_s[lane] + uniform_s;
                    let c = sum_c[h] + word_c[lane] + uniform_c;
                    let mag_sq = s * s + c * c;
                    candidate[h] = branchless_select_q7(
                        mag_sq > 1.0e-20,
                        branchless_atan2_to_q7_fast(s, c),
                        0,
                    );
                }
                candidate_words = load_q7_words_64(&candidate);
            }
        }

        let r_bar = circular_r_bar_from_words_64(&candidate_words);
        let kappa_c = branchless_von_mises_kappa(r_bar);
        let keep = r_bar >= branchless_clamp_f32(min_coherence_r_bar, 0.0, 1.0);
        if keep {
            *profile_out = candidate;
        } else {
            *profile_out = [0i8; MAX_ATTENTION_HEADS];
        }
        Ok((r_bar, kappa_c))
    }
}

/// Cold-path trait for Training-Free LoRA-to-Phase SVD Projection (L2P-SVD) (**Macro / Offline Level**).
pub trait L2pSvdProjector: BranchlessKernel + Send + Sync {
    /// Projects fine-tuned LoRA matrices ($W_A \in \mathbb{R}^{r \times d_{\mathrm{in}}}, W_B \in \mathbb{R}^{d_{\mathrm{out}} \times r}$)
    /// via first-order SVD approximation into phase profiles and Rank-1 SRL vectors (Section 5.2).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `rank == 0`, `dim_in == 0`, `dim_out == 0`,
    /// `dim_out` is odd, matrix slices are smaller than `rank * dim_in` / `dim_out * rank`,
    /// or `phase_profile_out` is empty.
    #[inline]
    fn project_lora(
        &self,
        lora_a: &AlignedSlice32<'_, f32>,
        lora_b: &AlignedSlice32<'_, f32>,
        rank: usize,
        dim_in: usize,
        dim_out: usize,
        srl_out: &mut gcso_srl_descriptor_t,
        phase_profile_out: &mut AlignedSliceMut16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        if rank == 0
            || dim_in == 0
            || dim_out == 0
            || (dim_out & 1) != 0
            || lora_a.len()
                < rank
                    .checked_mul(dim_in)
                    .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?
            || lora_b.len()
                < dim_out
                    .checked_mul(rank)
                    .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?
            || phase_profile_out.is_empty()
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let b_slice = lora_b.as_slice();
        let a_slice = lora_a.as_slice();

        let norm_u_sq = if rank == 1 {
            sum_sq_f32_simd8(&b_slice[..dim_out])
        } else {
            let mut total_u = 0.0_f32;
            for i in 0..dim_out {
                let v = b_slice[i * rank];
                total_u += v * v;
            }
            total_u
        };
        let norm_v_sq = sum_sq_f32_simd8(&a_slice[..dim_in]);

        let sigma_1 = branchless_flush_subnormal_f32(branchless_sqrt_f32(norm_u_sq * norm_v_sq));
        srl_out.layer_idx = 0;
        srl_out.rank = 1;
        srl_out.u_vector_ptr = lora_b.as_ptr() as u64;
        srl_out.v_vector_ptr = lora_a.as_ptr() as u64;
        srl_out.gain_scalar_ptr = 0;
        srl_out.scale_factor = branchless_select_f32(sigma_1 > 0.0, sigma_1, 1.0);

        let out_phases = phase_profile_out.as_mut_slice();
        let num_heads = out_phases.len();
        let pairs_per_head = (dim_out / 2).max(1) / num_heads.max(1);

        for (h, phase_slot) in out_phases.iter_mut().enumerate() {
            let pair_idx = (h * pairs_per_head.max(1)) % (dim_out / 2);
            let y_comp = b_slice[(pair_idx * 2 + 1) * rank];
            let x_comp = b_slice[(pair_idx * 2) * rank];
            *phase_slot = branchless_atan2_to_q7_fast(y_comp, x_comp);
        }

        Ok(())
    }

    /// Projects fine-tuned LoRA matrices into an SRL Rank-1 descriptor, a Q7 rotational phase
    /// profile, and an **Isotropic Block-Diagonal Scaling** vector $\mathbf{D}$ (Theorem 1.1 & Section 5.2).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `rank == 0`, `dim_in == 0`, `dim_out == 0`,
    /// `dim_out` is odd, matrix slices are too small, `phase_profile_out` is empty, or
    /// `block_scales_out.len() < dim_out / 2`.
    #[inline]
    fn project_lora_with_block_scales(
        &self,
        lora_a: &AlignedSlice32<'_, f32>,
        lora_b: &AlignedSlice32<'_, f32>,
        rank: usize,
        dim_in: usize,
        dim_out: usize,
        srl_out: &mut gcso_srl_descriptor_t,
        phase_profile_out: &mut AlignedSliceMut16<'_, gcso_q7_t>,
        block_scales_out: &mut AlignedSliceMut32<'_, f32>,
    ) -> GcsoResult<()> {
        let num_pairs = dim_out / 2;
        if num_pairs == 0 || block_scales_out.len() < num_pairs {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.project_lora(
            lora_a,
            lora_b,
            rank,
            dim_in,
            dim_out,
            srl_out,
            phase_profile_out,
        )?;

        let b_slice = lora_b.as_slice();
        let pair_stride = 2 * rank;
        let scales_slice = &mut block_scales_out.as_mut_slice()[..num_pairs];

        for (p_idx, scale_slot) in scales_slice.iter_mut().enumerate() {
            let row0 = p_idx * pair_stride;
            let pair_energy = sum_sq_f32_simd8(&b_slice[row0..row0 + pair_stride]);
            let mag = branchless_flush_subnormal_f32(branchless_sqrt_f32(pair_energy));
            *scale_slot = branchless_select_f32(mag > 1.0e-9, mag, 1.0);
        }
        srl_out.gain_scalar_ptr = block_scales_out.as_ptr() as u64;
        Ok(())
    }
}

/// Mezzo/Cold-path trait for Predictive Phase-Motion & Residual Compensation (PPRC) (**Mezzo / Storage Level**).
pub trait PprcCache: BranchlessKernel + Send + Sync {
    /// Seeks to a specific token index in the keyframe KV cache without forward passes.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_PPRC_SEEK_FAILED`] or [`GCSO_ERROR_INVALID_ARGUMENT`]
    /// if `token_index` is out of range.
    fn seek_to_token(
        &mut self,
        token_index: u32,
        header_out: &mut gcso_pprc_keyframe_header_t,
    ) -> GcsoResult<()>;

    /// Computes Circular Statistics resultant vector length $\bar{R}_c \in [0, 1]$ and von Mises
    /// concentration $\kappa_c$ using $\mathcal{O}(1)$ 256-entry Q7 lookup (`branchless_sincos_q7`)
    /// and division-free (`0 fdiv`) `branchless_inv_pos_f32` (Section 4.2).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `pruned_phases` is empty or its length does not
    /// match `token_weights.len()`.
    #[inline]
    fn aggregate_pruned_kv_circular_stats(
        &self,
        pruned_phases: &AlignedSlice16<'_, gcso_q7_t>,
        token_weights: &AlignedSlice32<'_, f32>,
        header_out: &mut gcso_pprc_keyframe_header_t,
    ) -> GcsoResult<(f32, f32)> {
        if pruned_phases.is_empty() || pruned_phases.len() != token_weights.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let mut sum_c = 0.0_f32;
        let mut sum_s = 0.0_f32;
        let mut sum_w = 0.0_f32;

        for (&q7_theta, &w) in pruned_phases
            .as_slice()
            .iter()
            .zip(token_weights.as_slice().iter())
        {
            let clean_w = branchless_relu_f32(w);
            let (sin_t, cos_t) = branchless_sincos_q7(q7_theta);
            sum_c += clean_w * cos_t;
            sum_s += clean_w * sin_t;
            sum_w += clean_w;
        }

        let norm = branchless_sqrt_f32(sum_c * sum_c + sum_s * sum_s);
        let r_bar = branchless_flush_subnormal_f32(branchless_select_f32(
            sum_w > 1.0e-12,
            norm * branchless_inv_pos_f32(sum_w),
            0.0,
        ));
        let r_bar_clamped = branchless_clamp_f32(r_bar, 0.0, 1.0);
        let kappa_c = branchless_von_mises_kappa(r_bar_clamped);
        let theta_bar = branchless_atan2_f32(sum_s, sum_c);

        header_out.composite_vector_length = r_bar_clamped;
        header_out.von_mises_kappa = kappa_c;
        header_out.sparse_scalar_residual = theta_bar;
        Ok((r_bar_clamped, kappa_c))
    }

    /// Computes Multi-Head Virtual Position Shift $\Delta m^{(h)}$ in division-free (`0 fdiv`)
    /// branchless mode (Section 4.3).
    #[inline(always)]
    #[must_use]
    fn compute_virtual_position_shift(
        &self,
        accumulated_phase_rad: f32,
        mean_base_rope_freq: f32,
    ) -> f32 {
        let abs_freq = branchless_abs_f32(mean_base_rope_freq);
        let valid_freq = (abs_freq > 1.0e-9) & branchless_is_finite_f32(mean_base_rope_freq);
        let inv_freq = branchless_copysign_f32(
            branchless_inv_pos_f32(branchless_select_f32(valid_freq, abs_freq, 1.0)),
            mean_base_rope_freq,
        );
        branchless_select_f32(
            valid_freq,
            branchless_flush_subnormal_f32(accumulated_phase_rad * inv_freq),
            0.0,
        )
    }

    /// Evaluates the **Variable Group of Pictures (GOP) Structure** keyframe classification
    /// (`0` = I-Frame Keyframe Anchor, `1` = P-Frame Phase Motion Delta) and effective GOP length (Section 4.3).
    #[inline(always)]
    #[must_use]
    fn evaluate_variable_gop_trigger(
        &self,
        moving_z_entropy: f32,
        sliding_flux: f32,
        steps_since_keyframe: u32,
        min_gop: u32,
        max_gop: u32,
        entropy_surge_thresh: f32,
        flux_surge_thresh: f32,
    ) -> (u32, u32) {
        let safe_min = branchless_select_u32(min_gop > 0, min_gop, 1);
        let safe_max = branchless_select_u32(max_gop >= safe_min, max_gop, safe_min);
        let clean_h = branchless_select_f32(
            branchless_is_finite_f32(moving_z_entropy),
            moving_z_entropy,
            0.0,
        );
        let clean_flux =
            branchless_select_f32(branchless_is_finite_f32(sliding_flux), sliding_flux, 0.0);

        let surge = (clean_h > entropy_surge_thresh) | (clean_flux > flux_surge_thresh);
        let is_iframe =
            (steps_since_keyframe >= safe_max) | ((steps_since_keyframe >= safe_min) & surge);
        let frame_type = branchless_select_u32(is_iframe, 0, 1);
        let gop_len = branchless_select_u32(is_iframe, steps_since_keyframe.max(1), safe_max);
        (frame_type, gop_len)
    }

    /// Executes **Zero-Forward Latency Seek (Instantaneous Replay)** phase reconstruction (Section 4.3)
    /// using 8-lane SWAR saturating addition and SWAR QDPS filtering without forward passes.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `qdps_min_step_q7 > 127`.
    #[inline]
    fn reconstruct_pframe_phase_trajectory(
        &self,
        icache_anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        pcache_motion_deltas: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        qdps_min_step_q7: u8,
        reconstructed_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<()> {
        if qdps_min_step_q7 > 127 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        *reconstructed_out = *icache_anchor_phases;
        for delta in pcache_motion_deltas {
            branchless_q7_accumulate_64(reconstructed_out, delta);
        }
        branchless_q7_qdps_filter_64(reconstructed_out, qdps_min_step_q7);
        Ok(())
    }

    /// Executes **Seam-Free $S^1$ Geodesic P-Frame Trajectory Replay** (Section 4.3):
    /// accumulates P-Frame motion deltas on the circle group $S^1 \cong \mathbb{Z}/256\mathbb{Z}$,
    /// folds onto `[-64, +64]`, and applies 8-lane SWAR QDPS step filtering.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `qdps_min_step_q7 > 127`.
    #[inline]
    fn reconstruct_pframe_geodesic_trajectory(
        &self,
        icache_anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        pcache_motion_deltas: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        qdps_min_step_q7: u8,
        reconstructed_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<()> {
        if qdps_min_step_q7 > 127 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let mut cur_w = load_q7_words_64(icache_anchor_phases);
        for delta in pcache_motion_deltas {
            let dw = load_q7_words_64(delta);
            cur_w[0] = swar_q7_circular_add_x8(cur_w[0], dw[0]);
            cur_w[1] = swar_q7_circular_add_x8(cur_w[1], dw[1]);
            cur_w[2] = swar_q7_circular_add_x8(cur_w[2], dw[2]);
            cur_w[3] = swar_q7_circular_add_x8(cur_w[3], dw[3]);
            cur_w[4] = swar_q7_circular_add_x8(cur_w[4], dw[4]);
            cur_w[5] = swar_q7_circular_add_x8(cur_w[5], dw[5]);
            cur_w[6] = swar_q7_circular_add_x8(cur_w[6], dw[6]);
            cur_w[7] = swar_q7_circular_add_x8(cur_w[7], dw[7]);
        }
        let folded_w = [
            swar_q7_fold_s1_geodesic_x8(cur_w[0]),
            swar_q7_fold_s1_geodesic_x8(cur_w[1]),
            swar_q7_fold_s1_geodesic_x8(cur_w[2]),
            swar_q7_fold_s1_geodesic_x8(cur_w[3]),
            swar_q7_fold_s1_geodesic_x8(cur_w[4]),
            swar_q7_fold_s1_geodesic_x8(cur_w[5]),
            swar_q7_fold_s1_geodesic_x8(cur_w[6]),
            swar_q7_fold_s1_geodesic_x8(cur_w[7]),
        ];
        let filt_w = swar_q7_qdps_filter_words_64(&folded_w, qdps_min_step_q7);
        store_q7_words_64(reconstructed_out, &filt_w);
        Ok(())
    }

    /// Replays a **Limina P-Frame Motion Chain** ($\boldsymbol{\Delta\theta}_1, \dots, \boldsymbol{\Delta\theta}_T$)
    /// from an **I-Frame Key Anchor** (`icache_anchor_phases`), evaluating the **Joint Probability
    /// Chain Rule Log-Likelihood** $\ln P(M_1, \dots, M_T) = \sum_{t=1}^T \ln P(M_t \mid M_{t-1})$
    /// and inverting any self-contradictory ($H^1(K; \mathcal{F}) \neq 0$) P-Frame step via
    /// Phase-Conjugate Repulsion ($-\boldsymbol{\Delta\theta}_t$).
    ///
    /// Populates `header_out` (`gop_length`, `composite_vector_length = r_bar`,
    /// `von_mises_kappa = kappa_c`, `sparse_scalar_residual = joint_log_prob`) and returns
    /// `(joint_log_prob, r_bar, kappa_c, repulsed_pframe_count)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `qdps_min_step_q7 > 127` or `contradiction_threshold` is non-finite.
    #[inline]
    fn replay_liminal_pframe_chain_with_joint_log_prob(
        &self,
        icache_anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        pcache_motion_deltas: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        qdps_min_step_q7: u8,
        contradiction_threshold: f32,
        reconstructed_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
        header_out: &mut gcso_pprc_keyframe_header_t,
    ) -> GcsoResult<(f32, f32, f32, u32)> {
        if qdps_min_step_q7 > 127 || !branchless_is_finite_f32(contradiction_threshold) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let safe_thresh =
            branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
        let mut cur_w = load_q7_words_64(icache_anchor_phases);
        let mut joint_log_prob = 0.0_f32;
        let mut repulsed_count = 0u32;

        for delta in pcache_motion_deltas {
            let dw = load_q7_words_64(delta);
            let next_w = [
                swar_q7_circular_add_x8(cur_w[0], dw[0]),
                swar_q7_circular_add_x8(cur_w[1], dw[1]),
                swar_q7_circular_add_x8(cur_w[2], dw[2]),
                swar_q7_circular_add_x8(cur_w[3], dw[3]),
                swar_q7_circular_add_x8(cur_w[4], dw[4]),
                swar_q7_circular_add_x8(cur_w[5], dw[5]),
                swar_q7_circular_add_x8(cur_w[6], dw[6]),
                swar_q7_circular_add_x8(cur_w[7], dw[7]),
            ];
            let (coh, log_p) = branchless_q7_transition_log_prob_words_64(&cur_w, &next_w);
            if coh < -safe_thresh {
                cur_w = [
                    swar_q7_circular_add_x8(cur_w[0], swar_q7_neg_sat_x8(dw[0])),
                    swar_q7_circular_add_x8(cur_w[1], swar_q7_neg_sat_x8(dw[1])),
                    swar_q7_circular_add_x8(cur_w[2], swar_q7_neg_sat_x8(dw[2])),
                    swar_q7_circular_add_x8(cur_w[3], swar_q7_neg_sat_x8(dw[3])),
                    swar_q7_circular_add_x8(cur_w[4], swar_q7_neg_sat_x8(dw[4])),
                    swar_q7_circular_add_x8(cur_w[5], swar_q7_neg_sat_x8(dw[5])),
                    swar_q7_circular_add_x8(cur_w[6], swar_q7_neg_sat_x8(dw[6])),
                    swar_q7_circular_add_x8(cur_w[7], swar_q7_neg_sat_x8(dw[7])),
                ];
                repulsed_count = repulsed_count.wrapping_add(1);
            } else {
                cur_w = next_w;
            }
            joint_log_prob += log_p;
        }

        let folded_w = [
            swar_q7_fold_s1_geodesic_x8(cur_w[0]),
            swar_q7_fold_s1_geodesic_x8(cur_w[1]),
            swar_q7_fold_s1_geodesic_x8(cur_w[2]),
            swar_q7_fold_s1_geodesic_x8(cur_w[3]),
            swar_q7_fold_s1_geodesic_x8(cur_w[4]),
            swar_q7_fold_s1_geodesic_x8(cur_w[5]),
            swar_q7_fold_s1_geodesic_x8(cur_w[6]),
            swar_q7_fold_s1_geodesic_x8(cur_w[7]),
        ];
        let filt_w = swar_q7_qdps_filter_words_64(&folded_w, qdps_min_step_q7);
        store_q7_words_64(reconstructed_out, &filt_w);

        let r_bar = circular_r_bar_from_words_64(&filt_w);
        let kappa_c = branchless_von_mises_kappa(r_bar);
        let clean_lp = branchless_flush_subnormal_f32(joint_log_prob);
        header_out.gop_length = (pcache_motion_deltas.len() as u32).max(1);
        header_out.composite_vector_length = r_bar;
        header_out.von_mises_kappa = kappa_c;
        header_out.sparse_scalar_residual = clean_lp;

        Ok((clean_lp, r_bar, kappa_c, repulsed_count))
    }
}

/// Trait for Dynamic Node Protocol (GCSO-DNP) distributed phase synchronization (**DAES Mode 1 / Boundary Layer**).
pub trait DnpPacketDispatcher: BranchlessKernel + Send + Sync {
    /// Dispatches a compact 64-byte GCSO-DNP phase/repulsion synchronization packet in zero-allocation mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_STATE`] if `daes_slot.mode != 1` (Network Plugin Mode).
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `packet_payload.len() < 64`.
    #[inline]
    fn dispatch_dnp_packet(
        &mut self,
        daes_slot: &gcso_daes_slot_t,
        packet_payload: &AlignedSlice64<'_, u8>,
    ) -> GcsoResult<()> {
        if daes_slot.mode != 1 {
            return Err(GCSO_ERROR_INVALID_STATE);
        }
        if packet_payload.len() < 64 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(())
    }

    /// Encodes a 64-head Q7 phase/repulsion vector into a 64-byte cacheline-aligned GCSO-DNP
    /// packet (`packet_out`) using bulk `Pod` copy and returns its branchless IEEE 802.3 CRC32 integrity tag.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_STATE`] if `daes_slot.mode != 1`.
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `packet_out.len() < 64` or `source_slot_id >= 8`.
    #[inline]
    fn encode_dnp_phase_packet(
        &self,
        daes_slot: &gcso_daes_slot_t,
        source_slot_id: u8,
        action_mode_flags: u8,
        phase_deltas: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        packet_out: &mut AlignedSliceMut64<'_, u8>,
    ) -> GcsoResult<u32> {
        if daes_slot.mode != 1 {
            return Err(GCSO_ERROR_INVALID_STATE);
        }
        if packet_out.len() < MAX_ATTENTION_HEADS
            || (source_slot_id as usize) >= MAX_ATTRACTOR_DOMAINS
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let phase_w = load_q7_words_64(phase_deltas);
        let dst = &mut packet_out.as_mut_slice()[..MAX_ATTENTION_HEADS];
        dst.copy_from_slice(gcso_q7_t::slice_as_bytes(phase_deltas));
        let tag_byte = pack_action_slot_tag(source_slot_id, action_mode_flags);
        let crc = crc32_block64_zero_seed_words(&phase_w) ^ u32::from(tag_byte);
        Ok(crc)
    }

    /// Verifies and decodes a 64-byte cacheline-aligned GCSO-DNP packet (`packet_payload`) into
    /// a 64-head Q7 phase/repulsion vector (`phases_out`), returning the recovered 8-bit tag byte.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_STATE`] if `daes_slot.mode != 1`.
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `packet_payload.len() < 64`.
    /// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] if the upper 24 bits of the IEEE 802.3 CRC32
    /// checksum do not match `expected_crc32`.
    #[inline]
    fn decode_and_verify_dnp_packet(
        &self,
        daes_slot: &gcso_daes_slot_t,
        packet_payload: &AlignedSlice64<'_, u8>,
        expected_crc32: u32,
        phases_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<u8> {
        if daes_slot.mode != 1 {
            return Err(GCSO_ERROR_INVALID_STATE);
        }
        if packet_payload.len() < MAX_ATTENTION_HEADS {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let src = &packet_payload.as_slice()[..MAX_ATTENTION_HEADS];
        gcso_q7_t::read_unaligned_slice_from(src, 0, phases_out)?;
        let raw_crc = crc32_block64_zero_seed_words(&load_q7_words_64(phases_out));
        let diff = raw_crc ^ expected_crc32;
        if (diff & 0xFFFF_FF00) != 0 {
            *phases_out = [0i8; MAX_ATTENTION_HEADS];
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }
        Ok((diff & 0xFF) as u8)
    }

    /// Verifies a 64-byte GCSO-DNP packet and applies its decoded 64-head phase/repulsion vector
    /// directly to `trail` (and `bloom_filter` if repulsive).
    ///
    /// # Errors
    /// Propagates any error returned by [`Self::decode_and_verify_dnp_packet`].
    #[inline]
    fn decode_and_apply_dnp_packet_to_trail(
        &self,
        daes_slot: &gcso_daes_slot_t,
        packet_payload: &AlignedSlice64<'_, u8>,
        expected_crc32: u32,
        inhibition_strength: f32,
        bloom_filter: Option<&mut ContrastivePhaseBloomFilter>,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(u8, u8, bool)> {
        let mut decoded = [0i8; MAX_ATTENTION_HEADS];
        let tag_byte = self.decode_and_verify_dnp_packet(
            daes_slot,
            packet_payload,
            expected_crc32,
            &mut decoded,
        )?;
        let slot_id = unpack_attractor_slot_from_tag(tag_byte);
        let mode_flags = unpack_action_mode_from_tag(tag_byte);
        let is_repulsion = (slot_id == ATTRACTOR_SLOT_REPULSION)
            | ((mode_flags & ACTION_MODE_PHASE_REPULSE) != 0);

        if is_repulsion {
            if let Some(bf) = bloom_filter {
                bf.insert_repulsive_pattern(&decoded);
            }
            let _ = trail.apply_mutual_slot_inhibition(&decoded, inhibition_strength);
            trail.flags |= TRAIL_FLAG_REPULSION_ACTIVE;
        } else {
            trail.accumulate_phase_vector(&decoded);
        }
        Ok((slot_id, mode_flags, is_repulsion))
    }
}

/// Cold-path trait for Zero-Overhead In-Memory Mapped Storage (ZIMMS) (**Storage / Persistence Level**).
pub trait ZimmsStorage: BranchlessKernel + Send + Sync {
    /// Serializes active runtime state into a `.gcso` binary snapshot buffer.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `buffer.len() < *required_size`.
    fn serialize_snapshot(&self, buffer: &mut [u8], required_size: &mut usize) -> GcsoResult<()>;

    /// Deserializes a `.gcso` binary snapshot buffer to restore context state.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] or [`crate::abi::GCSO_ERROR_VERSION_MISMATCH`]
    /// if the snapshot header or checksum is invalid.
    fn deserialize_snapshot(
        &mut self,
        buffer: &[u8],
        config_out: &mut gcso_config_t,
    ) -> GcsoResult<()>;

    /// Opens a memory-mapped zero-copy handle for a `.gcso` container payload.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `file_path` is empty, or
    /// [`crate::abi::GCSO_ERROR_ZIMMS_MAPPING_FAILED`] if OS `mmap` fails.
    fn open_mmap(
        &mut self,
        file_path: &str,
        descriptor_out: &mut gcso_zimms_descriptor_t,
    ) -> GcsoResult<()>;

    /// Unmaps a zero-copy ZIMMS memory handle and releases Direct-DMA resources.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_IO_FAILURE`] if unmapping fails.
    fn close_mmap(&mut self, descriptor: &mut gcso_zimms_descriptor_t) -> GcsoResult<()>;

    /// Retrieves the byte offset for a specific track in a `.gcso` container
    /// (`0` = `CORE`, `1` = `I-CACHE`, `2` = `P-CACHE`, `3` = `SNAPSHOT`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `track_id > 3`.
    #[inline(always)]
    fn get_track_offset(&self, track_id: u32) -> GcsoResult<u64> {
        let header_size = core::mem::size_of::<gcso_snapshot_header_t>() as u64;
        match track_id {
            0 => Ok(header_size),
            1 => Ok(header_size + 1024),
            2 => Ok(header_size + 2048),
            3 => Ok(header_size + 3072),
            _ => Err(GCSO_ERROR_INVALID_ARGUMENT),
        }
    }

    /// Reads and validates a `.gcso` container's 128-byte [`gcso_snapshot_header_t`] via [`verify_snapshot_payload_crc32`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] if `container_bytes` is truncated, magic/offsets
    /// are invalid, or CRC32 verification fails.
    /// Returns [`crate::abi::GCSO_ERROR_VERSION_MISMATCH`] if `version != 0x00000101`.
    #[inline]
    fn inspect_and_verify_container(
        &self,
        container_bytes: &[u8],
    ) -> GcsoResult<gcso_snapshot_header_t> {
        let header = gcso_snapshot_header_t::read_unaligned_from(container_bytes, 0)
            .map_err(|_| GCSO_ERROR_CONTAINER_CORRUPTED)?;
        verify_snapshot_payload_crc32(&header, container_bytes)?;
        Ok(header)
    }

    /// Finalizes the IEEE 802.3 CRC32 payload checksum on `header` and writes the 128-byte header
    /// into the beginning of `container_bytes`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] or [`crate::abi::GCSO_ERROR_VERSION_MISMATCH`]
    /// if `header` or `container_bytes` bounds are invalid.
    #[inline]
    fn stamp_and_write_snapshot_header(
        &self,
        header: &mut gcso_snapshot_header_t,
        container_bytes: &mut [u8],
    ) -> GcsoResult<u32> {
        let crc = finalize_snapshot_header_crc32(header, container_bytes)?;
        header
            .write_unaligned_to(container_bytes, 0)
            .map_err(|_| GCSO_ERROR_CONTAINER_CORRUPTED)?;
        Ok(crc)
    }

    /// Extracts a zero-copy byte slice `&'a [u8]` of length `track_len` for `track_id` (`0..=3`) from `container_bytes`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `track_id > 3`.
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `container_bytes` cannot hold `offset + track_len`.
    #[inline]
    fn extract_track_slice<'a>(
        &self,
        container_bytes: &'a [u8],
        track_id: u32,
        track_len: usize,
    ) -> GcsoResult<&'a [u8]> {
        let start = usize::try_from(self.get_track_offset(track_id)?)
            .map_err(|_| GCSO_ERROR_BUFFER_TOO_SMALL)?;
        let end = start
            .checked_add(track_len)
            .ok_or(GCSO_ERROR_BUFFER_TOO_SMALL)?;
        if container_bytes.len() < end {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        Ok(&container_bytes[start..end])
    }

    /// Extracts a hardware-aligned zero-copy [`AlignedSlice`] (`AlignedSlice<'a, T, ALIGN>`) of `elem_count` elements
    /// for `track_id` (`0..=3`) directly from a ZIMMS memory-mapped `.gcso` container buffer.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `track_id > 3` or `T` is zero-sized.
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `container_bytes` is too small.
    /// Returns [`crate::abi::GCSO_ERROR_MISALIGNED_POINTER`] if the track start address violates `ALIGN`.
    #[inline]
    fn extract_aligned_track_slice<'a, T: Pod, const ALIGN: usize>(
        &self,
        container_bytes: &'a [u8],
        track_id: u32,
        elem_count: usize,
    ) -> GcsoResult<AlignedSlice<'a, T, ALIGN>> {
        let start = usize::try_from(self.get_track_offset(track_id)?)
            .map_err(|_| GCSO_ERROR_BUFFER_TOO_SMALL)?;
        let typed_slice = T::slice_from_bytes_at(container_bytes, start, elem_count)?;
        AlignedSlice::<T, ALIGN>::new(typed_slice)
    }
}

// ===================================================================
// 7. Blanket-Implemented Lv.3 Macro Observers & Coordinators over `ActionHub`
// ===================================================================

/// Cold-path crystallization candidacy extension trait for [`gcso_pointer_trail_t`].
pub trait ColdPathTrailCrystalExt: StigmergicTrailOps {
    /// Branchlessly evaluates whether a pointer trail qualifies for bottom-up macro
    /// crystallization (`stigmergic_density >= density_thresh && r_bar_t >= coherence_thresh && !repulsion_active`)
    /// and stamps or clears [`TRAIL_FLAG_CRYSTALLIZED_CANDIDATE`] on `self.flags` in $\mathcal{O}(1)$.
    fn evaluate_crystallization_candidacy(
        &mut self,
        density_crystallization_thresh: f32,
        coherence_crystallization_thresh: f32,
        r_bar_t: f32,
    ) -> bool;

    /// Evaluates whether a trail stitched across inter-memory **Limina** transitions qualifies as
    /// a crystallized Global Section $\Gamma(K; \mathcal{F})$ (`!cohomology_obstructed` where
    /// $H^1(K; \mathcal{F}) = 0$, `stigmergic_density >= density_thresh`, and `r_bar_t >= coherence_thresh`),
    /// stamping or clearing [`TRAIL_FLAG_CRYSTALLIZED_CANDIDATE`].
    fn evaluate_liminal_global_section_candidacy(
        &mut self,
        density_crystallization_thresh: f32,
        coherence_crystallization_thresh: f32,
        r_bar_t: f32,
        cohomology_obstructed: bool,
    ) -> bool;
}

impl ColdPathTrailCrystalExt for gcso_pointer_trail_t {
    #[inline(always)]
    fn evaluate_crystallization_candidacy(
        &mut self,
        density_crystallization_thresh: f32,
        coherence_crystallization_thresh: f32,
        r_bar_t: f32,
    ) -> bool {
        let density_ok = self.should_crystallize_branchless(density_crystallization_thresh)
            & branchless_is_finite_f32(density_crystallization_thresh);
        let coherence_ok = (r_bar_t >= coherence_crystallization_thresh)
            & (coherence_crystallization_thresh >= 0.0)
            & branchless_is_finite_f32(coherence_crystallization_thresh)
            & branchless_is_finite_f32(r_bar_t);
        let not_repulsed = (self.flags & TRAIL_FLAG_REPULSION_ACTIVE) == 0;
        let is_candidate = density_ok & coherence_ok & not_repulsed;
        self.flags = (self.flags & !TRAIL_FLAG_CRYSTALLIZED_CANDIDATE)
            | branchless_select_u32(is_candidate, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, 0);
        is_candidate
    }

    #[inline(always)]
    fn evaluate_liminal_global_section_candidacy(
        &mut self,
        density_crystallization_thresh: f32,
        coherence_crystallization_thresh: f32,
        r_bar_t: f32,
        cohomology_obstructed: bool,
    ) -> bool {
        if cohomology_obstructed {
            self.flags &= !TRAIL_FLAG_CRYSTALLIZED_CANDIDATE;
            return false;
        }
        self.evaluate_crystallization_candidacy(
            density_crystallization_thresh,
            coherence_crystallization_thresh,
            r_bar_t,
        )
    }
}

/// Blanket-implemented coordinator uniting [`ActionHub`] with [`MultiDomainAttractorSpace`]
/// to project the 8-lane working bank, autonomously rebalance lane weights from phase proximity,
/// fuse coupled neighborhood synergy, and steer tensors.
pub trait AttractorSlotSynergyCoordinator: ActionHub + MultiDomainAttractorSpace {
    /// Projects all 8 Working Lanes for `trail_out`, applies coupled neighborhood
    /// synergy and mutual inhibition, and steers `query_tensor` and/or `key_tensor`.
    ///
    /// Returns `(damped_pull_force, mean_coherence, active_neighbor_count)`.
    ///
    /// # Errors
    /// Propagates any error from [`ActionHub::validate_hub_cascade_params`],
    /// [`MultiDomainAttractorSpace::steer_trail_by_coupled_neighborhood`], or
    /// [`ActionHub::filter_qdps_and_steer_qk_safe`].
    #[inline]
    fn coordinate_multi_domain_slot_synergy(
        &mut self,
        input_ptr: u64,
        synergy_gain: f32,
        inhibition_strength: f32,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32, u32)> {
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        self.step_pointer(input_ptr, trail_out)?;
        let (pull, coh, neighbors, syn_w) = self.steer_trail_by_coupled_neighborhood_to_words(
            synergy_gain,
            inhibition_strength,
            trail_out,
        )?;
        let _ = self.steer_qdps_words_prevalidated_core(
            &syn_w,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        );
        Ok((pull, coh, neighbors))
    }

    /// Executes a zero-hyperparameter **Unified-Field Auto-Rebalanced Synergy** cascade.
    ///
    /// Returns `(damped_pull_force, mean_coherence, active_neighbor_count, rebalanced_weights)`.
    ///
    /// # Errors
    /// Propagates any error from [`ActionHub::validate_hub_cascade_params`],
    /// [`MultiDomainAttractorSpace::rebalance_working_lane_weights_from_trail`], or
    /// [`ActionHub::filter_qdps_and_steer_qk_safe`].
    #[inline]
    fn coordinate_unified_field_auto_rebalanced_synergy(
        &mut self,
        input_ptr: u64,
        synergy_gain: f32,
        inhibition_strength: f32,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32, u32, [f32; MAX_ATTRACTOR_DOMAINS])> {
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        self.step_pointer(input_ptr, trail_out)?;
        let weights = self.rebalance_working_lane_weights_from_trail(trail_out)?;
        let (pull, coh, neighbors, syn_w) = self.steer_trail_by_coupled_neighborhood_to_words(
            synergy_gain,
            inhibition_strength,
            trail_out,
        )?;
        let _ = self.steer_qdps_words_prevalidated_core(
            &syn_w,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        );
        Ok((pull, coh, neighbors, weights))
    }
}

impl<T: ActionHub + MultiDomainAttractorSpace + ?Sized> AttractorSlotSynergyCoordinator for T {}

/// Blanket-implemented coordinator uniting [`ActionHub`], [`SheafCohomologyEvaluator`], and
/// [`EdbcBranchController`] to detect cohomological obstructions $H^1(K; \mathcal{F})$, bridge
/// inter-memory Limina gaps, and feed anti-phase repulsion back into the pointer trail.
pub trait StigmergicCohomologyCoordinator:
    ActionHub + SheafCohomologyEvaluator + EdbcBranchController
{
    /// Evaluates cohomological obstruction $H^1(K; \mathcal{F})$ on `cochain_residual_r` vs
    /// `exact_projection_delta0_phi`, records repulsive patterns into `bloom_filter` if obstructed,
    /// and evolves `trail_inout`'s pointer mode.
    ///
    /// Returns `(mean_obstruction_energy, is_obstructed, next_branch_mode, evolved_ptr)`.
    ///
    /// # Errors
    /// Propagates any error from [`SheafCohomologyEvaluator::detect_and_record_obstruction_repulsion`].
    #[inline]
    fn evaluate_cohomology_and_evolve_trail(
        &mut self,
        cochain_residual_r: &AlignedSlice32<'_, f32>,
        exact_projection_delta0_phi: &AlignedSlice32<'_, f32>,
        obstruction_threshold: f32,
        moving_z_entropy: f32,
        sliding_flux: f32,
        cvoid_score: f32,
        bifurcation_threshold: f32,
        tau_flux: f32,
        hysteresis_band: f32,
        high_density_srl_thresh: f32,
        dominant_attractor_slot: u8,
        bloom_filter: &mut ContrastivePhaseBloomFilter,
        trail_inout: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, bool, u32, gcso_tagged_ptr_t)> {
        let (mean_obs, is_obstructed, post_phases) = self.detect_and_record_obstruction_repulsion(
            cochain_residual_r,
            exact_projection_delta0_phi,
            obstruction_threshold,
            &trail_inout.accumulated_phase_delta,
            bloom_filter,
        )?;
        trail_inout.accumulated_phase_delta = post_phases;
        if is_obstructed {
            trail_inout.flags |= TRAIL_FLAG_REPULSION_ACTIVE;
        }
        let (next_mode, evolved_ptr) = self.evaluate_branch_and_evolve_pointer(
            gcso_tagged_ptr_t(trail_inout.current_ptr),
            trail_inout.stigmergic_density,
            dominant_attractor_slot,
            moving_z_entropy,
            sliding_flux,
            cvoid_score,
            bifurcation_threshold,
            tau_flux,
            hysteresis_band,
            high_density_srl_thresh,
        );
        trail_inout.current_ptr = evolved_ptr.raw();
        Ok((mean_obs, is_obstructed, next_mode, evolved_ptr))
    }

    /// Bridges all inter-memory **Limina** transitions across `trail_chain`, evaluates the chain's
    /// Joint Probability Log-Likelihood $\sum_t \ln P(M_t \mid M_{t-1})$ and $S^1$ Sheaf Holonomy
    /// Obstruction ($H^1(K; \mathcal{F})$), evolves the terminal trail's pointer mode via EDBC,
    /// and steers `query_tensor` and/or `key_tensor` with QDPS + RIPA soft-clamping.
    ///
    /// Returns `(joint_log_prob_sum, holonomy_obstruction_energy, is_obstructed, repulsed_count, evolved_ptr)`.
    ///
    /// # Errors
    /// Propagates any error from [`ActionHub::validate_hub_cascade_params`],
    /// [`SheafCohomologyEvaluator::bridge_and_filter_liminal_trail_chain`], or
    /// [`SheafCohomologyEvaluator::evaluate_trail_chain_cohomology_and_joint_log_prob`].
    #[inline]
    fn coordinate_liminal_trail_chain_and_steer_qk(
        &mut self,
        trail_chain: &mut [gcso_pointer_trail_t],
        interpolation_u7: u8,
        contradiction_threshold: f32,
        obstruction_threshold: f32,
        moving_z_entropy: f32,
        sliding_flux: f32,
        cvoid_score: f32,
        bifurcation_threshold: f32,
        tau_flux: f32,
        hysteresis_band: f32,
        high_density_srl_thresh: f32,
        dominant_attractor_slot: u8,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        bloom_filter: &mut ContrastivePhaseBloomFilter,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(f32, f32, bool, u32, gcso_tagged_ptr_t)> {
        let len = trail_chain.len();
        if len == 0
            || !branchless_is_finite_f32(contradiction_threshold)
            || !branchless_is_finite_f32(obstruction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;

        let first_w = *trail_phase_words_ref(&trail_chain[0]);
        let mut prev_w = first_w;
        let mut joint_log_prob = 0.0_f32;
        let mut coboundary_defect_sum = 0.0_f32;
        let mut repulsed_count = 0u32;

        for idx in 1..len {
            let (head, tail) = trail_chain.split_at_mut(idx);
            let prev_ref = &head[idx - 1];
            let cur_mut = &mut tail[0];
            let (pre_coh, pre_lp, repulsed) = cur_mut.bridge_liminal_hallucinated_trail(
                prev_ref,
                interpolation_u7,
                contradiction_threshold,
            );
            let cur_w = *trail_phase_words_ref(cur_mut);
            if repulsed {
                bloom_filter.insert_repulsive_pattern_words_weighted(&cur_w, 1);
                repulsed_count = repulsed_count.wrapping_add(1);
            }
            let (coh, log_p) = if cur_w == prev_w {
                (1.0_f32, 0.0_f32)
            } else if interpolation_u7 >= 128 && !repulsed {
                (pre_coh, pre_lp)
            } else {
                branchless_q7_transition_log_prob_words_64(&prev_w, &cur_w)
            };
            joint_log_prob += log_p;
            coboundary_defect_sum += branchless_relu_f32(0.5 * (1.0 - coh));
            prev_w = cur_w;
        }

        let (joint_log_prob, holonomy_obs, is_obstructed) = if len == 1 {
            (0.0_f32, 0.0_f32, false)
        } else {
            let mean_step_defect = coboundary_defect_sum * fast_inv_len_f32(len - 1);
            let loop_coh = coherence_from_word_arrays_64(&first_w, &prev_w);
            let loop_defect = branchless_relu_f32(0.5 * (1.0 - loop_coh));
            let holonomy_energy =
                branchless_flush_subnormal_f32(0.5 * (mean_step_defect + loop_defect));
            let obstructed = holonomy_energy > branchless_relu_f32(obstruction_threshold);
            (
                branchless_flush_subnormal_f32(joint_log_prob),
                holonomy_energy,
                obstructed,
            )
        };

        let last_idx = len - 1;
        if is_obstructed && (trail_chain[last_idx].flags & TRAIL_FLAG_REPULSION_ACTIVE) == 0 {
            bloom_filter.insert_repulsive_pattern_words_weighted(&prev_w, 1);
            trail_chain[last_idx].stamp_phase_conjugate_repulsion();
            prev_w = *trail_phase_words_ref(&trail_chain[last_idx]);
        }
        let (_next_mode, evolved_ptr) = self.evaluate_branch_and_evolve_pointer(
            gcso_tagged_ptr_t(trail_chain[last_idx].current_ptr),
            trail_chain[last_idx].stigmergic_density,
            dominant_attractor_slot,
            moving_z_entropy,
            sliding_flux,
            cvoid_score,
            bifurcation_threshold,
            tau_flux,
            hysteresis_band,
            high_density_srl_thresh,
        );
        trail_chain[last_idx].current_ptr = evolved_ptr.raw();

        let _ = self.steer_qdps_words_prevalidated_core(
            &prev_w,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        );
        Ok((
            joint_log_prob,
            holonomy_obs,
            is_obstructed,
            repulsed_count,
            evolved_ptr,
        ))
    }
}

impl<T> StigmergicCohomologyCoordinator for T where
    T: ActionHub + SheafCohomologyEvaluator + EdbcBranchController + ?Sized
{
}

/// Blanket-implemented macro coordinator uniting [`EdbcBranchController`], [`AttractorField`],
/// and [`PersonaPatcher`] for bottom-up trail crystallization, metastable tunneling perturbation,
/// and persistent topological mode quantization.
pub trait BottomUpMacroCrystalCoordinator:
    EdbcBranchController + AttractorField + PersonaPatcher
{
    /// Evaluates EDBC branch mode transition (`0` = Normal, `1` = Pitchfork, `2` = Repulsion),
    /// screens `trails` for bottom-up crystallization candidacy (`evaluate_crystallization_candidacy`),
    /// injects metastable tunneling perturbation in Pitchfork mode (`1`) or records Bloom repulsion
    /// in Repulsion mode (`2`), triggers [`AttractorField::aggregate_bottom_up`], and quantizes the
    /// persistent topological persona profile (`persona_profile_out`).
    ///
    /// Returns `(next_branch_mode, crystallized_candidate_count, new_anchors_created, persona_r_bar, persona_kappa)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `trails` is empty, or propagates errors from
    /// [`EdbcBranchController::inject_metastable_tunneling_perturbation`],
    /// [`AttractorField::aggregate_bottom_up`], or
    /// [`PersonaPatcher::quantize_persistent_topological_mode`].
    #[inline]
    fn step_macro_crystallization_with_persona_and_tunneling(
        &mut self,
        trails: &mut [gcso_pointer_trail_t],
        bloom_filter: &mut ContrastivePhaseBloomFilter,
        moving_z_entropy: f32,
        sliding_flux: f32,
        cvoid_score: f32,
        bifurcation_threshold: f32,
        tau_flux: f32,
        hysteresis_band: f32,
        density_crystallization_thresh: f32,
        coherence_crystallization_thresh: f32,
        alpha_pitchfork: f32,
        beta_pitchfork: f32,
        head_dim: u32,
        explore_head_start: usize,
        min_persona_coherence_r_bar: f32,
        persona_profile_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<(u32, u32, u32, f32, f32)> {
        if trails.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let next_mode = self.evaluate_branch_mode_transition(
            self.active_branch_mode(),
            moving_z_entropy,
            sliding_flux,
            cvoid_score,
            bifurcation_threshold,
            tau_flux,
            hysteresis_band,
        );

        let mut candidate_count = 0u32;
        for trail in trails.iter_mut() {
            let trail_w = *trail_phase_words_ref(trail);
            let r_bar_t = circular_r_bar_from_words_64(&trail_w);
            let is_candidate = trail.evaluate_crystallization_candidacy(
                density_crystallization_thresh,
                coherence_crystallization_thresh,
                r_bar_t,
            );
            candidate_count += u32::from(is_candidate);

            if next_mode == 1 {
                let _ = self.inject_metastable_tunneling_perturbation(
                    &mut trail.accumulated_phase_delta,
                    gcso_tagged_ptr_t(trail.current_ptr),
                    cvoid_score,
                    bifurcation_threshold,
                    alpha_pitchfork,
                    beta_pitchfork,
                    head_dim,
                    explore_head_start,
                )?;
            } else if next_mode == 2 {
                bloom_filter.insert_repulsive_pattern_words_weighted(&trail_w, 1);
                trail.flags |= TRAIL_FLAG_REPULSION_ACTIVE;
            }
        }

        let new_anchors = if candidate_count > 0 {
            self.aggregate_bottom_up()?
        } else {
            0
        };
        let (r_bar, kappa) = self.quantize_persistent_topological_mode(
            trails,
            min_persona_coherence_r_bar,
            persona_profile_out,
        )?;
        Ok((next_mode, candidate_count, new_anchors, r_bar, kappa))
    }

    /// Bridges all inter-memory **Limina** gaps across `trail_chain` ($M_1 \to \dots \to M_T$),
    /// checks whether the stitched trajectory forms a coherent Global Section $\Gamma(K; \mathcal{F})$
    /// (`repulsed_transitions == 0`, $H^1(K; \mathcal{F}) = 0$), marks global section candidates
    /// via [`ColdPathTrailCrystalExt::evaluate_liminal_global_section_candidacy`], triggers
    /// [`AttractorField::aggregate_bottom_up`], and quantizes the persistent persona profile.
    ///
    /// Returns `(joint_log_prob, repulsed_transitions, candidate_count, new_anchors, persona_r_bar, persona_kappa)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `trail_chain` is empty or thresholds are non-finite.
    #[inline]
    fn step_liminal_chain_crystallization_with_persona_and_tunneling(
        &mut self,
        trail_chain: &mut [gcso_pointer_trail_t],
        interpolation_u7: u8,
        contradiction_threshold: f32,
        bloom_filter: &mut ContrastivePhaseBloomFilter,
        moving_z_entropy: f32,
        sliding_flux: f32,
        cvoid_score: f32,
        bifurcation_threshold: f32,
        tau_flux: f32,
        hysteresis_band: f32,
        density_crystallization_thresh: f32,
        coherence_crystallization_thresh: f32,
        alpha_pitchfork: f32,
        beta_pitchfork: f32,
        head_dim: u32,
        explore_head_start: usize,
        min_persona_coherence_r_bar: f32,
        persona_profile_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<(f32, u32, u32, u32, f32, f32)> {
        if trail_chain.is_empty() || !branchless_is_finite_f32(contradiction_threshold) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let mut joint_log_prob = 0.0_f32;
        let mut repulsed_transitions = 0u32;
        for idx in 1..trail_chain.len() {
            let (head, tail) = trail_chain.split_at_mut(idx);
            let prev_ref = &head[idx - 1];
            let cur_mut = &mut tail[0];
            let (_coh, lp, repulsed) = cur_mut.bridge_liminal_hallucinated_trail(
                prev_ref,
                interpolation_u7,
                contradiction_threshold,
            );
            joint_log_prob += lp;
            if repulsed {
                bloom_filter.insert_repulsive_pattern_words_weighted(
                    trail_phase_words_ref(cur_mut),
                    1,
                );
                repulsed_transitions = repulsed_transitions.wrapping_add(1);
            }
        }

        let cohomology_obstructed = repulsed_transitions > 0;
        let next_mode = self.evaluate_branch_mode_transition(
            self.active_branch_mode(),
            moving_z_entropy,
            sliding_flux,
            cvoid_score,
            bifurcation_threshold,
            tau_flux,
            hysteresis_band,
        );

        let mut candidate_count = 0u32;
        for trail in trail_chain.iter_mut() {
            let trail_w = *trail_phase_words_ref(trail);
            let r_bar_t = circular_r_bar_from_words_64(&trail_w);
            let is_candidate = trail.evaluate_liminal_global_section_candidacy(
                density_crystallization_thresh,
                coherence_crystallization_thresh,
                r_bar_t,
                cohomology_obstructed,
            );
            candidate_count += u32::from(is_candidate);

            if next_mode == 1 {
                let _ = self.inject_metastable_tunneling_perturbation(
                    &mut trail.accumulated_phase_delta,
                    gcso_tagged_ptr_t(trail.current_ptr),
                    cvoid_score,
                    bifurcation_threshold,
                    alpha_pitchfork,
                    beta_pitchfork,
                    head_dim,
                    explore_head_start,
                )?;
            } else if next_mode == 2 {
                bloom_filter.insert_repulsive_pattern_words_weighted(&trail_w, 1);
                trail.flags |= TRAIL_FLAG_REPULSION_ACTIVE;
            }
        }

        let new_anchors = if candidate_count > 0 {
            self.aggregate_bottom_up()?
        } else {
            0
        };
        let (r_bar, kappa) = self.quantize_persistent_topological_mode(
            trail_chain,
            min_persona_coherence_r_bar,
            persona_profile_out,
        )?;
        Ok((
            branchless_flush_subnormal_f32(joint_log_prob),
            repulsed_transitions,
            candidate_count,
            new_anchors,
            r_bar,
            kappa,
        ))
    }
}

impl<T> BottomUpMacroCrystalCoordinator for T where
    T: EdbcBranchController + AttractorField + PersonaPatcher + ?Sized
{
}

/// Unified Lv.3 Macro Observer trait over [`ActionHub`] (`traits/cold_path.rs`).
///
/// Observes [`super::telemetry`] trajectories from [`ActionHub`] to evaluate EDBC pitchfork
/// bifurcations, matrix-free cellular sheaf cohomology $H^1(K; \mathcal{F})$, unified-field
/// attractor crystallization, and Online Toroidal $k$-Means / DP-Means macro distribution steering.
pub trait ColdPathObserver:
    ActionHub
    + SheafCohomologyEvaluator
    + EdbcBranchController
    + AttractorField
    + MultiDomainAttractorSpace
    + PersonaPatcher
    + AttractorSlotSynergyCoordinator
    + StigmergicCohomologyCoordinator
    + BottomUpMacroCrystalCoordinator
{
    /// Coordinates a per-token step directly on the **Single Unified Attractor Manifold**
    /// (`anchor_phases`, `anchor_base_energies`) without requiring an 8-slot phase bank:
    /// 1. Steps `input_ptr` in [`ActionHub`] to update `trail_out`.
    /// 2. Folds all unified-field anchors via [`AttractorField::steer_trail_on_unified_field_from_slices`],
    ///    binding the dominant anchor ID into `trail_out.target_anchor_id`.
    /// 3. Applies QDPS + RIPA phase steering to `query_tensor` and/or `key_tensor`.
    /// 4. Optionally modulates candidate memory Attention scores (`attention_scores`) by
    ///    Attention $\times$ DPSR phase interference.
    ///
    /// Returns `(final_pull_force, mean_coherence, active_anchor_count, dominant_anchor_id)`.
    ///
    /// # Errors
    /// Propagates any error from [`ActionHub::validate_hub_cascade_params`],
    /// [`AttractorField::steer_trail_on_unified_field_from_slices`], or
    /// [`ActionHub::filter_qdps_and_steer_qk_safe`].
    #[inline]
    fn coordinate_unified_field_step_and_steer_qk(
        &mut self,
        input_ptr: u64,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        attention_interference_gain: f32,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32, u32, u32)> {
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        self.step_pointer(input_ptr, trail_out)?;
        let (pull, mean_coh, active_cnt, uf_w) = self
            .steer_trail_on_unified_field_from_slices_to_words(
                anchor_phases,
                anchor_base_energies,
                repulsion_phases_and_strength,
                trail_out,
            )?;
        let _ = self.steer_qdps_words_prevalidated_core(
            &uf_w,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        );
        if let Some(scores) = attention_scores {
            if !anchor_phases.is_empty() {
                let _ = self.modulate_hub_attention_scores_by_unified_field(
                    scores,
                    trail_out,
                    anchor_phases,
                    attention_interference_gain,
                )?;
            }
        }
        Ok((pull, mean_coh, active_cnt, trail_out.target_anchor_id))
    }

    /// Coordinates the closed-loop **2-Origin Gateway $\to$ Online Toroidal $k$-Means / DP-Means
    /// Macro Distribution (`centroids`) $\to$ EDBC Pitchfork / Tunneling / Repulsion** pipeline:
    /// 1. Ingests `signal` via [`ActionHubGatewayExt::ingest_gateway_kmeans_and_steer_safe`],
    ///    steering `query_tensor` / `key_tensor` / `logits_opt` and updating the winning macro
    ///    centroid in `centroids`.
    /// 2. Evaluates EDBC branch mode transition (`0` = Normal, `1` = Pitchfork, `2` = Repulsion).
    /// 3. In Pitchfork mode (`1`), injects metastable tunneling perturbation onto `trail_out`;
    ///    in Repulsion mode (`2`), inserts `trail_out.accumulated_phase_delta` into `bloom_filter`.
    ///
    /// Returns `(GatewayFrame, CausalProvenanceReport, assigned_cluster_idx, assigned_anchor_id, updated_gravity_mass, next_branch_mode)`.
    ///
    /// # Errors
    /// Propagates any error from [`ActionHubGatewayExt::ingest_gateway_kmeans_and_steer_safe`] or
    /// [`EdbcBranchController::inject_metastable_tunneling_perturbation`].
    #[inline]
    fn coordinate_kmeans_macro_distribution_and_edbc_step<S: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        signal: &S,
        centroids: &mut [CrystallizedAttractorRecord],
        split_coherence_thresh: f32,
        moving_z_entropy: f32,
        sliding_flux: f32,
        cvoid_score: f32,
        bifurcation_threshold: f32,
        tau_flux: f32,
        hysteresis_band: f32,
        alpha_pitchfork: f32,
        beta_pitchfork: f32,
        explore_head_start: usize,
        bloom_filter: &mut ContrastivePhaseBloomFilter,
        daes_slot: &mut gcso_daes_slot_t,
        trail_out: &mut gcso_pointer_trail_t,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(
        GatewayFrame,
        CausalProvenanceReport,
        usize,
        u32,
        f32,
        u32,
    )> {
        let (frame, report, cluster_idx, anchor_id, mass, _spawned) = self
            .ingest_gateway_kmeans_and_steer_safe(
                signal,
                centroids,
                split_coherence_thresh,
                daes_slot,
                trail_out,
                query_tensor,
                key_tensor,
                logits_opt,
            )?;

        let next_mode = self.evaluate_branch_mode_transition(
            self.active_branch_mode(),
            moving_z_entropy,
            sliding_flux,
            cvoid_score,
            bifurcation_threshold,
            tau_flux,
            hysteresis_band,
        );

        let head_dim = self.head_dim();
        if next_mode == 1 {
            let _ = self.inject_metastable_tunneling_perturbation(
                &mut trail_out.accumulated_phase_delta,
                gcso_tagged_ptr_t(trail_out.current_ptr),
                cvoid_score,
                bifurcation_threshold,
                alpha_pitchfork,
                beta_pitchfork,
                head_dim,
                explore_head_start,
            )?;
        } else if next_mode == 2 {
            bloom_filter.insert_repulsive_pattern(&trail_out.accumulated_phase_delta);
            trail_out.flags |= TRAIL_FLAG_REPULSION_ACTIVE;
        }

        Ok((frame, report, cluster_idx, anchor_id, mass, next_mode))
    }
}

impl<T> ColdPathObserver for T where
    T: ActionHub
        + SheafCohomologyEvaluator
        + EdbcBranchController
        + AttractorField
        + MultiDomainAttractorSpace
        + PersonaPatcher
        + ?Sized
{
}

/// Composite Cold-Path Orchestrator trait uniting Macro evaluation, attractor steering, and ZIMMS persistence.
pub trait ColdPathOrchestrator:
    EntropyEvaluator + AttractorField + PersonaPatcher + L2pSvdProjector + PprcCache + ZimmsStorage
{
}

impl<T> ColdPathOrchestrator for T where
    T: EntropyEvaluator
        + AttractorField
        + PersonaPatcher
        + L2pSvdProjector
        + PprcCache
        + ZimmsStorage
        + ?Sized
{
}

/// Full-Stack Fractal GCSO Engine trait uniting Nano, Micro, Mezzo, and Macro contracts under static dispatch.
pub trait FullStackGcsoEngine:
    HotPathExecutionEngine
    + ColdPathOrchestrator
    + SwarmCellChunk
    + GcsoRuntimeContext
    + SystemCapabilityQuery
{
}

impl<T> FullStackGcsoEngine for T where
    T: HotPathExecutionEngine
        + ColdPathOrchestrator
        + SwarmCellChunk
        + GcsoRuntimeContext
        + SystemCapabilityQuery
        + ?Sized
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::primitives::{
        with_slot_pair_coupled_symmetric, ACTION_PRESET_EXPRESS_PHASE,
        ACTION_PRESET_FULL_ORCHESTRATION, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA,
        ATTRACTOR_SLOT_USER_CONTEXT,
    };
    use crate::traits::{AlignedBuffer16, AlignedBuffer32, AlignedBuffer64};
    use core::f32::consts::PI;

    #[test]
    fn test_sheaf_cohomology_symmetric_degrees_dual_pairing_and_qdpt() {
        struct MockSheafTree;
        impl SheafCohomologyEvaluator for MockSheafTree {}
        impl HierarchicalTreeTable for MockSheafTree {}

        let evaluator = MockSheafTree;
        let degrees = AlignedBuffer32::new([2.0f32, 5.5, 3.0, 1.0, 4.0, 2.5, 1.5, 0.5]);
        let lambda_max = evaluator
            .eval_gershgorin_lambda_max(&degrees.as_aligned_slice())
            .unwrap();
        assert!((lambda_max - 11.0).abs() < 1e-5);

        let mut m0 = gcso_paged_bitmask_t::default();
        let mut m1 = gcso_paged_bitmask_t::default();
        let mut m2 = gcso_paged_bitmask_t::default();
        m0.set_bit_branchless(1);
        m0.set_bit_branchless(2);
        m1.set_bit_branchless(1);
        m1.set_bit_branchless(2);
        m2.set_bit_branchless(10);

        let masks = [m0, m1, m2];
        let mut deg_out = AlignedBuffer32::new([0.0f32; 4]);
        let lmax_blocks = evaluator
            .eval_stigmergic_block_degrees(&masks, &mut deg_out.as_aligned_slice_mut())
            .unwrap();
        assert!((deg_out.data[0] - 1.0).abs() < 1e-4);
        assert!((deg_out.data[1] - 1.0).abs() < 1e-4);
        assert_eq!(deg_out.data[2], 0.0);
        assert!((lmax_blocks - 2.0).abs() < 1e-4);

        let coeffs = evaluator
            .compute_tikhonov_chebyshev_coeffs_deg3(lambda_max, 0.1)
            .unwrap();
        let eig_in = AlignedBuffer32::new([0.0f32, 5.5, 11.0, 2.75, 1.0, 2.0, 3.0, 4.0, 8.0, 9.0]);
        let mut filt_out = AlignedBuffer32::new([0.0f32; 10]);
        evaluator
            .eval_chebyshev_spectral_filter_deg3(
                &eig_in.as_aligned_slice(),
                &mut filt_out.as_aligned_slice_mut(),
                11.0,
                &coeffs,
            )
            .unwrap();
        assert!(filt_out.data[0] > filt_out.data[2]);
        assert!(filt_out.data[4] > filt_out.data[9]);

        let r = AlignedBuffer32::new([2.0f32, 2.0, 2.0, 2.0]);
        let p = AlignedBuffer32::new([0.5f32, 0.5, 0.5, 0.5]);
        let gamma = AlignedBuffer32::new([1.0f32, 1.0, 1.0, 1.0]);
        let (mean_obs, dual_pair, obstructed) = evaluator
            .evaluate_hodge_obstruction_and_dual_pairing(
                &r.as_aligned_slice(),
                &p.as_aligned_slice(),
                &gamma.as_aligned_slice(),
                1.0,
            )
            .unwrap();
        assert!(obstructed);
        assert!((mean_obs - 2.25).abs() < 1e-5);
        assert!((dual_pair - 1.0).abs() < 1e-5);

        let base_ptr = gcso_tagged_ptr_t::pack_action(
            0x1000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            10,
        );
        let (qdpt_lvl, _dv, cascaded_ptr) = evaluator
            .resolve_dyadic_qdpt_and_cascade_ptr(
                base_ptr,
                &m0,
                &m1,
                0.05,
                0,
                0.1,
                &[0x100, 0x200, 0x300, 0x400],
                &[4, 8, 12, 16],
            )
            .unwrap();
        assert_eq!(qdpt_lvl, 0);
        assert_eq!(cascaded_ptr.unpack_addr(), 0x1100);
        assert_eq!(cascaded_ptr.unpack_phase(), 14);

        let t0 = gcso_pointer_trail_t {
            accumulated_phase_delta: [24i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let t1 = gcso_pointer_trail_t {
            accumulated_phase_delta: [28i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let t2 = gcso_pointer_trail_t {
            accumulated_phase_delta: [32i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let mut chain = [t0, t1, t2];
        let mut bloom = ContrastivePhaseBloomFilter::new();

        let (bridged_lp, rep_cnt) = evaluator
            .bridge_and_filter_liminal_trail_chain(&mut chain, 64, 0.25, &mut bloom)
            .unwrap();
        assert_eq!(rep_cnt, 0);
        assert!(bridged_lp > -0.10);

        let (joint_lp, hol_obs, is_obs) = evaluator
            .evaluate_trail_chain_cohomology_and_joint_log_prob(&chain, 0.20)
            .unwrap();
        assert!(!is_obs && hol_obs < 0.05 && joint_lp > -0.05);
    }

    #[test]
    fn test_l2p_svd_pprc_limina_replay_unified_field_and_edbc_tunneling() {
        struct MockColdSystem;
        impl SystemCapabilityQuery for MockColdSystem {}
        impl L2pSvdProjector for MockColdSystem {}
        impl DnpPacketDispatcher for MockColdSystem {}
        impl PprcCache for MockColdSystem {
            fn seek_to_token(
                &mut self,
                token_index: u32,
                header_out: &mut gcso_pprc_keyframe_header_t,
            ) -> GcsoResult<()> {
                header_out.token_index = token_index;
                Ok(())
            }
        }
        impl PersonaPatcher for MockColdSystem {
            fn apply_patch(&mut self, _patch_data: &[u8]) -> GcsoResult<()> {
                Ok(())
            }
        }
        impl EntropyEvaluator for MockColdSystem {
            fn eval_stateful(
                &mut self,
                token_z_score: f32,
                state_out: &mut gcso_edbc_state_t,
            ) -> GcsoResult<()> {
                state_out.moving_z_entropy = token_z_score;
                Ok(())
            }
        }
        impl EdbcBranchController for MockColdSystem {
            fn active_branch_mode(&self) -> u32 {
                1
            }
            fn set_bifurcation_threshold(&mut self, _threshold: f32) {}
            fn set_hysteresis_band(&mut self, _band: f32) {}
            fn set_tau_flux(&mut self, _tau_flux: f32) {}
        }
        impl AttractorField for MockColdSystem {
            fn anchor_count(&self) -> u32 {
                1
            }
            fn apply_evaporative_decay(&mut self) {}
            fn total_potential_energy(&self) -> f32 {
                2.0
            }
            fn pressure_ratio(&self) -> f32 {
                0.1
            }
            fn evaporate_weakest_anchor(&mut self) -> Option<u32> {
                None
            }
            fn add_system_prompt_anchor(&mut self, _prompt: &str, _weight: f32) -> GcsoResult<u32> {
                Ok(1)
            }
            fn add_embedding_anchor(
                &mut self,
                _embedding: &AlignedSlice32<'_, f32>,
                _anchor_type: gcso_anchor_type_t,
                _weight: f32,
            ) -> GcsoResult<u32> {
                Ok(1)
            }
            fn add_anchor(
                &mut self,
                _anchor_type: gcso_anchor_type_t,
                _vec: &AlignedSlice32<'_, f32>,
                _dim: usize,
            ) -> GcsoResult<u32> {
                Ok(1)
            }
            fn inject_phase_repulsion(
                &mut self,
                _repulsion_deltas: &AlignedSlice16<'_, gcso_q7_t>,
                _gain: f32,
            ) -> GcsoResult<()> {
                Ok(())
            }
            fn aggregate_bottom_up(&mut self) -> GcsoResult<u32> {
                Ok(1)
            }
            fn clear_anchors(&mut self) -> GcsoResult<()> {
                Ok(())
            }
        }
        impl MultiDomainAttractorSpace for MockColdSystem {
            fn advance_step(&mut self) {}
            fn set_domain_weight(&mut self, domain_id: u32, _weight: f32) -> GcsoResult<()> {
                if (domain_id as usize) >= MAX_ATTRACTOR_DOMAINS {
                    return Err(GCSO_ERROR_INVALID_ARGUMENT);
                }
                Ok(())
            }
            fn set_domain_enabled(&mut self, domain_id: u32, _enabled: bool) -> GcsoResult<()> {
                if (domain_id as usize) >= MAX_ATTRACTOR_DOMAINS {
                    return Err(GCSO_ERROR_INVALID_ARGUMENT);
                }
                Ok(())
            }
            fn find_dominant_domain(&self) -> (u32, f32) {
                (u32::from(ATTRACTOR_SLOT_PERSONA), 2.0)
            }
            fn slot_coupling_matrix(&self) -> u64 {
                let m = with_slot_pair_coupled_symmetric(
                    0,
                    u32::from(ATTRACTOR_SLOT_PERSONA),
                    u32::from(ATTRACTOR_SLOT_KNOWLEDGE),
                    true,
                );
                with_slot_pair_coupled_symmetric(
                    m,
                    u32::from(ATTRACTOR_SLOT_PERSONA),
                    u32::from(ATTRACTOR_SLOT_USER_CONTEXT),
                    true,
                )
            }
            fn evaluate_slot_force(
                &self,
                domain_id: u32,
                _trail: &gcso_pointer_trail_t,
                slot_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
            ) -> GcsoResult<f32> {
                match domain_id {
                    1 | 2 | 4 => {
                        *slot_phase_out = [32i8; MAX_ATTENTION_HEADS];
                        Ok(2.0)
                    }
                    3 => {
                        *slot_phase_out = [-96i8; MAX_ATTENTION_HEADS];
                        Ok(1.5)
                    }
                    0..=7 => {
                        *slot_phase_out = [0i8; MAX_ATTENTION_HEADS];
                        Ok(0.0)
                    }
                    _ => Err(GCSO_ERROR_INVALID_ARGUMENT),
                }
            }
            fn evaluate_multi_domain_superposition(
                &self,
                trail: &gcso_pointer_trail_t,
                composite_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
            ) -> f32 {
                self.evaluate_unified_superposition_from_working_bank(
                    trail,
                    0.35,
                    0.50,
                    composite_phase_out,
                )
                .unwrap_or(0.0)
            }
            fn clear_all_domains(&mut self) -> GcsoResult<()> {
                Ok(())
            }
        }

        let mut sys = MockColdSystem;
        let mut cfg = gcso_config_t::default();
        sys.init_default_config(&mut cfg).unwrap();
        assert_eq!(sys.validate_config_invariants(&cfg), Ok(()));
        assert_eq!(sys.abi_version(), (0, 1, 1));

        let hashed_p = sys.hash_prompt_to_phase_profile("LiminiKa Persona Anchor", 1);
        assert!(hashed_p.iter().any(|&x| x != 0));

        let lora_a = AlignedBuffer32::new([1.0f32; 16]);
        let lora_b = AlignedBuffer32::new([1.0f32; 16]);
        let mut srl_desc = gcso_srl_descriptor_t::default();
        let mut prof_out = AlignedBuffer16::new([0i8; 16]);
        let mut scales_out = AlignedBuffer32::new([0.0f32; 4]);
        sys.project_lora_with_block_scales(
            &lora_a.as_aligned_slice(),
            &lora_b.as_aligned_slice(),
            2,
            8,
            8,
            &mut srl_desc,
            &mut prof_out.as_aligned_slice_mut(),
            &mut scales_out.as_aligned_slice_mut(),
        )
        .unwrap();
        assert!((scales_out.data[0] - 2.0).abs() < 1e-5);

        let phases = AlignedBuffer16::new([32i8; 16]);
        let weights = AlignedBuffer32::new([1.0f32; 16]);
        let mut header = gcso_pprc_keyframe_header_t::default();
        let (r_bar, kappa) = sys
            .aggregate_pruned_kv_circular_stats(
                &phases.as_aligned_slice(),
                &weights.as_aligned_slice(),
                &mut header,
            )
            .unwrap();
        assert!((r_bar - 1.0).abs() < 1e-4);
        assert!(kappa > 10.0);
        assert!((header.sparse_scalar_residual - (PI * 0.25)).abs() < 1e-3);

        // Verify Limina PPRC P-Frame Chain Replay & Joint Probability Log-Likelihood
        let icache_anchor = [24i8; MAX_ATTENTION_HEADS];
        let pframes = [[4i8; MAX_ATTENTION_HEADS], [4i8; MAX_ATTENTION_HEADS]];
        let mut replayed = [0i8; MAX_ATTENTION_HEADS];
        let mut gop_hdr = gcso_pprc_keyframe_header_t::default();
        let (gop_lp, gop_r, gop_k, gop_rep) = sys
            .replay_liminal_pframe_chain_with_joint_log_prob(
                &icache_anchor,
                &pframes,
                1,
                0.25,
                &mut replayed,
                &mut gop_hdr,
            )
            .unwrap();
        assert_eq!(gop_rep, 0);
        assert_eq!(replayed[0], 32);
        assert!(gop_lp > -0.05 && (gop_r - 1.0).abs() < 1e-4 && gop_k > 10.0);

        let daes_net = gcso_daes_slot_t {
            mode: 1,
            ..Default::default()
        };
        let src_phases = [28i8; MAX_ATTENTION_HEADS];
        let mut pkt_buf = AlignedBuffer64::<u8, 64>::zeroed();
        let crc_tag = sys
            .encode_dnp_phase_packet(
                &daes_net,
                ATTRACTOR_SLOT_PERSONA,
                ACTION_PRESET_FULL_ORCHESTRATION,
                &src_phases,
                &mut pkt_buf.as_aligned_slice_mut(),
            )
            .unwrap();
        let mut decoded_phases = [0i8; MAX_ATTENTION_HEADS];
        let recovered_tag = sys
            .decode_and_verify_dnp_packet(
                &daes_net,
                &pkt_buf.as_aligned_slice(),
                crc_tag,
                &mut decoded_phases,
            )
            .unwrap();
        assert_eq!(decoded_phases[0], 28);
        assert_eq!(
            recovered_tag,
            pack_action_slot_tag(ATTRACTOR_SLOT_PERSONA, ACTION_PRESET_FULL_ORCHESTRATION)
        );

        let mut trail = gcso_pointer_trail_t {
            current_ptr: gcso_tagged_ptr_t::pack_action(
                0x1000,
                ATTRACTOR_SLOT_PERSONA,
                ACTION_PRESET_EXPRESS_PHASE,
                16,
            )
            .raw(),
            ..Default::default()
        };

        let rebalanced = sys
            .rebalance_working_lane_weights_from_trail(&trail)
            .unwrap();
        assert!(rebalanced[usize::from(ATTRACTOR_SLOT_PERSONA)] >= 1.0);

        let (final_pull, mean_coh, neighbors) = sys
            .steer_trail_by_coupled_neighborhood(0.5, 0.5, &mut trail)
            .unwrap();
        assert_eq!(neighbors, 2);
        assert!((mean_coh - 1.0).abs() < 1e-4);
        assert!((final_pull - 6.0).abs() < 0.2);
        assert_ne!(trail.flags & TRAIL_FLAG_SYNERGY_RESONANCE, 0);

        trail.stigmergic_density = 1.5;
        let mut trails = [trail, trail];
        let mut bloom = ContrastivePhaseBloomFilter::new();
        let mut persona_out = [0i8; MAX_ATTENTION_HEADS];
        let (chain_lp, rep_trans, candidates, new_anchors, p_r_bar, p_kappa) = sys
            .step_liminal_chain_crystallization_with_persona_and_tunneling(
                &mut trails,
                64,
                0.25,
                &mut bloom,
                0.60,
                0.25,
                0.75,
                0.45,
                0.15,
                0.02,
                0.9,
                0.8,
                1.5,
                0.5,
                128,
                48,
                0.8,
                &mut persona_out,
            )
            .unwrap();
        assert_eq!(rep_trans, 0);
        assert!(chain_lp > -0.05);
        assert_eq!(candidates, 2);
        assert_eq!(new_anchors, 1);
        assert_ne!(trails[1].flags & TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, 0);
        assert!(p_r_bar > 0.85 && p_kappa > 3.0);
    }
}