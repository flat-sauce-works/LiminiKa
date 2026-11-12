// name: src/core/src/edbc.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Entropy-Driven Decoding Branch Controller (EDBC) core engine implementation.
//!
//! # Architectural Role (Thin `Pod` State Composite + Blanket Closed-Loop & Gateway Coordinator)
//! - **[`EdbcEngine`]**: Compact 128-byte (2-cacheline, `#[repr(C, align(32))]`) [`Pod`] stateful
//!   controller managing Moving Z-Score Normalized Attention Entropy $\tilde{H}$, the 8-slot
//!   Sliding-Window Entropy Rate Integrator $\Phi_M(t)$ ($M = 5$), anti-chattering dwell-step
//!   inertia, and Langevin pitchfork branch mode transitions (`Mode 0/1/2`). Directly implements
//!   **Origin 1 (`CausalProvenanceSignal` + `SystemConstraintSignal`)** and [`ManifoldGatewaySignal`]
//!   so `&EdbcEngine` can be used both standalone and inside `(S, F, P)` / `(S, F, P, G)` tuples.
//! - **[`EdbcClosedLoopCoordinator`]**: Blanket-implemented extension trait
//!   (`impl<T: EdbcBranchController + SheafCohomologyEvaluator + HierarchicalTreeTable + ?Sized> EdbcClosedLoopCoordinator for T {}`)
//!   that equips [`EdbcEngine`], [`crate::traits::ColdPathObserver`], and
//!   [`crate::traits::FullStackGcsoEngine`] with:
//!   1. Single-pass AVX2 $\mathcal{C}_{\mathrm{void}}$ + $\ell_1 / \ell_2$ coherence-ratio evaluation
//!      (`eval_cvoid_with_coherence_ratio`) and strictly branchless `0-ln` Taylor-Estrin
//!      surrogate-to-Z-score stepping (`eval_surrogate_and_step`).
//!   2. Single-pass register-chained Limina inter-memory trail-chain cohomology verification under
//!      the Joint Probability Chain Rule $\sum_t \ln P(M_t \mid M_{t-1})$ with deterministic
//!      repulsion pointer override (`step_with_liminal_trail_chain_and_bloom_guard`).
//!   3. Register-resident Single-Manifold Unified-Field EDBC pointer evolution
//!      (`step_with_unified_field_and_pointer`), preserving 32-bit `target_anchor_id` in `cluster_id`.
//!   4. Single-scan closed-loop **Pitchfork Metastable Tunneling + Bayesian Pointer-Trail selection,
//!      rewiring, Limina bridging & Attention score modulation**
//!      (`step_with_bayesian_pointer_selection_and_tunneling`,
//!      `route_coarse_to_fine_and_step_bayesian_tunneling`), combining sequential conditional
//!      transition probability ($\sum_t \ln P(M_t \mid M_{t-1})$ $\times$ Attention phase interference)
//!      with indirect pointer-chain routing in `ActionHub`.
//!   5. Loop-hoisted zero-copy candidate-trail Bayes-Attention score modulation coupled with EDBC
//!      energy-guided inverse-temperature scaling (`modulate_candidate_trails_with_edbc_temperature`).
//!   6. 256-bit dyadic tree QDPT depth resolution and pointer cascading
//!      (`resolve_dyadic_tree_and_step_edbc`).
//!   7. **2-Origin Composite Manifold Gateway, $\mathcal{O}(1)$ Causal Provenance, Online Toroidal
//!      $k$-Means / DP-Means Macro Clustering & `.gcso` Sample-Driven Attractor Crystallization**
//!      (`step_via_gateway_frame`, `step_via_gateway_signal`, `step_via_gateway_kmeans_64`,
//!      `step_with_crystallized_ensemble_and_pointer`).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::inline_always)]
#![allow(clippy::too_many_arguments)]
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

use core::f32::consts::{FRAC_PI_2, LN_2};
use core::mem::{align_of, offset_of, size_of};

use crate::abi::{
    gcso_edbc_state_t, gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_q7_t, gcso_tagged_ptr_t,
    GCSO_ERROR_EDBC_SINGULARITY, GCSO_ERROR_INVALID_ARGUMENT,
};
use crate::traits::aligned::{AlignedSlice32, AlignedSliceMut32};
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
use crate::traits::branchless::coherence_from_word_arrays_with_both_uniform_flags_64;
use crate::traits::branchless::{
    branchless_fast_transition_log_prob, branchless_phase_coherence_inv_temp_scale,
    branchless_q7_unified_field_fold_words_with_init_weight_64, scale_f32_slice_unrolled8,
    sum_abs_and_sq_f32_simd8, sum_sq_f32_simd8, swar_words_equal_64,
};
use crate::traits::cold_path::{
    EdbcBranchController, EntropyEvaluator, HierarchicalTreeTable, SheafCohomologyEvaluator,
};
use crate::traits::gateway::{
    synthesize_multi_origin_gateway_frame, synthesize_multi_origin_gateway_frame_with_words,
    CausalProvenanceReport, CausalProvenanceSignal, ContradictionGuard,
    CrystallizedAttractorRecord, GatewayFrame, ManifoldGatewaySignal, ObjectiveFact,
    SubjectivePersona, SystemConstraintSignal, CAUSAL_AXIS_SYSTEM_CONSTRAINT,
    CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, CAUSAL_CAUSE_COHOMOLOGY_REPULSION,
    CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE, DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
    ROOT_CAUSE_COHOMOLOGY_REPULSION, ROOT_CAUSE_NOMINAL, ROOT_CAUSE_SYSTEM_QOS_THROTTLE,
};
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use crate::traits::primitives::Q7_FULL_COS_REF;
use crate::traits::primitives::{
    branchless_abs_f32, branchless_clamp_f32, branchless_eyring_kramers_escape,
    branchless_flush_subnormal_f32, branchless_is_finite_f32, branchless_max_f32,
    branchless_min_f32, branchless_popcount_rsqrt_u32, branchless_q7_add_sat, branchless_rad_to_q7,
    branchless_relu_f32, branchless_rsqrt_f32, branchless_sanitize_z_score_f32,
    branchless_select_f32, branchless_select_q7, branchless_select_u32, branchless_select_u64,
    branchless_select_u8, branchless_splitmix64, branchless_sqrt_f32, fast_inv_len_f32,
    fast_inv_pos_normal_f32, is_word_array_uniform_64, load_q7_words_64, store_q7_word_at,
    store_q7_words_64, swar_q7_add_sat_x8, swar_q7_neg_sat_x8, swar_q7_qdps_filter_words_64,
    tanh_lut_eval_nonneg, trail_phase_words_mut, trail_phase_words_ref, DEFAULT_MAX_VALID_Z_SCORE,
    F32_MIN_POSITIVE_NORMAL, SWAR_SPLAT_BYTE_64,
};
use crate::traits::primitives::{
    GcsoResult, Pod, ACTION_MODE_DPSR_STEER, ACTION_MODE_LAYER_SKIP, ACTION_PRESET_INERTIAL_SKIP,
    ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_ID_MASK, ATTRACTOR_SLOT_REPULSION,
    DEFAULT_ENTROPY_EPS, MAX_ATTENTION_HEADS, MAX_SURROGATE_ENTROPY_LN2,
};
use crate::traits::telemetry::{
    select_and_bridge_macro_aligned_bayesian_trail_core, ContrastivePhaseBloomFilter,
    StigmergicActionPtrOps, StigmergicTrailOps, MAX_ATTRACTOR_PULL_FORCE, MAX_STIGMERGIC_DENSITY,
    NUM_SWARM_MICRO_CELLS, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE,
    TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_FLAG_SYNERGY_RESONANCE,
};
/// Normal entropy convergence mode (`Mode 0`: single stable attractor at origin $x^* = 0$).
pub const EDBC_MODE_NORMAL_ENTROPY: u32 = 0;

/// Pitchfork bifurcation exploration mode (`Mode 1`: supercritical double-well $\pm x^*_\pm$).
pub const EDBC_MODE_PITCHFORK_BIFURCATION: u32 = 1;

/// Phase-conjugate repulsion mode (`Mode 2`: anti-phase $-\boldsymbol{\Delta\theta}$ escape).
pub const EDBC_MODE_PHASE_CONJUGATE_REPULSION: u32 = 2;

/// Default logarithmic singularity epsilon fallback constant ($10^{-12}$, `c_abi_spec.md`).
pub const DEFAULT_SINGULARITY_EPS: f32 = DEFAULT_ENTROPY_EPS;

/// Default pitchfork bifurcation Z-score threshold constant ($\tau_{\mathrm{bifurcation}} = 1.5$).
pub const DEFAULT_BIFURCATION_THRESHOLD: f32 = 1.5;

/// Default sliding-window entropy rate flux threshold constant ($\tau_{\mathrm{flux}} = 0.5$, Section 3.2).
pub const DEFAULT_TAU_FLUX: f32 = 0.5;

/// Default initial sampling temperature modifier (`1.0`).
pub const DEFAULT_SAMPLE_TEMPERATURE: f32 = 1.0;

/// Default hysteresis deadband width for anti-chattering mode stabilization ($\delta_{\mathrm{hyst}} = 0.15$).
pub const DEFAULT_HYSTERESIS_BAND: f32 = 0.15;

/// Sliding-window lookback horizon $M = 5$ for the Entropy Rate Integrator $\Phi_M(t) = \tilde{H}(t) - \tilde{H}(t-M)$.
pub const EDBC_SLIDING_WINDOW_HORIZON: usize = 5;

/// Capacity of the power-of-two ring buffer storing recent $\tilde{H}$ samples (`8`).
pub const EDBC_WINDOW_RING_CAPACITY: usize = 8;

/// Maximum barrier safety ceiling for numerical stabilization near zero void scores ($10^9$).
pub const MAX_BARRIER_CEILING: f32 = 1.0e9;

/// Exponential moving average smoothing factor for Z-score entropy tracking ($\alpha_{\mathrm{EMA}} = 0.1$).
pub const EMA_ALPHA_ENTROPY: f32 = 0.1;

/// Upper ceiling for single-token Z-score inputs ($10.0\sigma$) preventing single-token outlier
/// spikes from locking `moving_z_entropy` into prolonged repulsion mode over subsequent tokens.
pub const MAX_VALID_TOKEN_Z_SCORE: f32 = DEFAULT_MAX_VALID_Z_SCORE;

/// Precomputed reciprocal of $\ln 2$ ($1 / \ln 2 \approx 1.44269504$) eliminating hardware `fdiv`
/// instructions during normalized entropy scaling.
const INV_LN_2: f32 = 1.0 / LN_2;

/// Constant Look-Up Table (LUT) for branch mode sample temperatures `[Mode 0, Mode 1, Mode 2]`.
const LUT_SAMPLE_TEMPERATURE: [f32; 3] = [1.0, 1.2, 1.5];

/// Constant Look-Up Table (LUT) for branch mode repulsion gain factors `[Mode 0, Mode 1, Mode 2]`.
const LUT_REPULSION_GAIN: [f32; 3] = [0.0, 0.5, 1.0];

// ===================================================================
// 1. Standalone Constant-Time (`0 branches`, `0 fdiv`, `0 ln`) Entropy & Pitchfork Functions
// ===================================================================

/// Computes normalized Sparse Surrogate Attention Entropy from Top-2 attention probabilities
/// in `#![no_std]` **branchless mode (`0` data-dependent branches, `0 fdiv`, `0 ln`)**,
/// bounded to $[0, \ln 2]$ (Section 3.1).
///
/// # Mathematical Formulation (9-Term Even Taylor-Estrin Series with Exact Endpoint Closure)
/// Let $\delta = \frac{|p_{\max} - p_{\mathrm{2nd}}|}{p_{\max} + p_{\mathrm{2nd}}} \in [0, 1]$ and
/// $t = \delta^2 \in [0, 1]$. By expanding $H(\delta) = \ln 2 - \sum_{k=1}^{\infty} \frac{t^k}{2k(2k-1)}$
/// into a 9-term binary-tree Estrin polynomial where the 9th coefficient absorbs the residual tail
/// $R_9 = \ln 2 - \sum_{k=1}^{8} \frac{1}{2k(2k-1)}$:
/// - At $t = 0$ ($p_{\max} = p_{\mathrm{2nd}}$): $H(0) = \ln 2$ (`0 ULP` error).
/// - At $t = 1$ ($p_{\mathrm{2nd}} \le \varepsilon$): $H(1) = 0.0$ (`0 ULP` error).
/// - Across all $t \in [0, 1]$: executes a uniform instruction stream using `eps` (defaulting
///   branchlessly to `1.0e-30` if `eps` is non-finite or below `F32_MIN_POSITIVE_NORMAL`) as the
///   lower positivity floor for $p_{\mathrm{2nd}}$.
#[inline(always)]
pub const fn compute_sparse_surrogate_entropy(p_max: f32, p_2nd: f32, eps: f32) -> f32 {
    const C1: f32 = 0.5;
    const C2: f32 = 1.0 / 12.0;
    const C3: f32 = 1.0 / 30.0;
    const C4: f32 = 1.0 / 56.0;
    const C5: f32 = 1.0 / 90.0;
    const C6: f32 = 1.0 / 132.0;
    const C7: f32 = 1.0 / 182.0;
    const C8: f32 = 1.0 / 240.0;
    const C9_TAIL: f32 = MAX_SURROGATE_ENTROPY_LN2 - (C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8);

    let safe_eps = branchless_select_f32(
        branchless_is_finite_f32(eps) & (eps >= F32_MIN_POSITIVE_NORMAL),
        eps,
        1.0e-30,
    );
    let raw_sum = p_max + p_2nd;
    let raw_diff = p_max - p_2nd;
    let valid_sum = (p_max > 0.0) & (p_2nd > safe_eps) & (raw_sum < f32::INFINITY);
    let safe_sum = branchless_select_f32(valid_sum, raw_sum, 1.0);
    let inv_sum = fast_inv_pos_normal_f32(safe_sum);
    let signed_delta = raw_diff * inv_sum;
    let t = branchless_min_f32(signed_delta * signed_delta, 1.0);
    let t2 = t * t;
    let t4 = t2 * t2;
    let p01 = C1 + t * C2;
    let p23 = C3 + t * C4;
    let p45 = C5 + t * C6;
    let p67 = C7 + t * C8;
    let q03 = p01 + t2 * p23;
    let q47 = p45 + t2 * p67;
    let poly = (q03 + t4 * q47) + (t4 * t4) * C9_TAIL;
    let raw_h = MAX_SURROGATE_ENTROPY_LN2 - t * poly;
    branchless_select_f32(valid_sum & (t < 0.999_999_9), raw_h, 0.0)
}

/// Computes Moving Z-Score Normalized Attention Entropy ($\tilde{H}$) incorporating
/// surrogate entropy and inter-layer activation cosine distance variance in **strictly
/// constant-time mode (`0` branches, `0 fdiv`)** via `fast_inv_pos_normal_f32` and
/// [`tanh_lut_eval_nonneg`] (Eq. 3.1).
///
/// The inter-layer cosine modulation factor $1 + \gamma_{\mathrm{mod}} \tanh(\alpha_z \operatorname{ReLU}(z_{\mathrm{dist}}))$
/// is evaluated with zero data dependency on `h_surrogate`, allowing out-of-order CPU execution units
/// to overlap `fast_inv_pos_normal_f32(eff_sigma)` and `tanh_lut_eval_nonneg` in parallel with
/// [`compute_sparse_surrogate_entropy`].
#[inline(always)]
pub const fn compute_moving_z_score_entropy(
    h_surrogate: f32,
    cosine_dist: f32,
    mu_dist: f32,
    sigma_dist: f32,
    sigma_min: f32,
    alpha_z: f32,
    gamma_mod: f32,
) -> f32 {
    let valid_cos = branchless_is_finite_f32(cosine_dist) & branchless_is_finite_f32(mu_dist);
    let raw_diff = cosine_dist - mu_dist;
    let pos_diff = branchless_select_f32(
        valid_cos & (raw_diff > 0.0) & (raw_diff < f32::INFINITY),
        raw_diff,
        0.0,
    );

    let alpha_safe = branchless_select_f32(
        branchless_is_finite_f32(alpha_z) & (alpha_z >= 0.0),
        alpha_z,
        1.0,
    );
    let gamma_safe = branchless_select_f32(
        branchless_is_finite_f32(gamma_mod) & (gamma_mod >= 0.0),
        gamma_mod,
        0.5,
    );
    let min_floor = branchless_select_f32(
        (sigma_min >= F32_MIN_POSITIVE_NORMAL) & branchless_is_finite_f32(sigma_min),
        sigma_min,
        1.0e-6,
    );
    let eff_sigma = branchless_max_f32(
        branchless_select_f32(branchless_is_finite_f32(sigma_dist), sigma_dist, min_floor),
        min_floor,
    );

    let inv_sigma = fast_inv_pos_normal_f32(eff_sigma);
    let z_dist = pos_diff * inv_sigma;
    let relu_z = branchless_min_f32(z_dist, MAX_VALID_TOKEN_Z_SCORE);
    let d_cosine_hat = tanh_lut_eval_nonneg(alpha_safe * relu_z);
    let mod_factor = 1.0 + gamma_safe * d_cosine_hat;

    let valid_h = (h_surrogate > 0.0) & branchless_is_finite_f32(h_surrogate);
    let h_tilde = branchless_flush_subnormal_f32(h_surrogate * mod_factor);
    branchless_select_f32(
        valid_h & valid_cos & branchless_is_finite_f32(h_tilde),
        h_tilde,
        0.0,
    )
}

/// Evaluates scale-invariant Pitchfork Bifurcation potential barrier height
/// $\Delta V = \frac{\alpha^2 (\max(0, \mathcal{C}_{\mathrm{void}} - \tau_{\mathrm{eff}}))^2}{4 \beta d_{\mathrm{head}}}$
/// in division-free (`0 fdiv`) mode (Lemma 3.1).
///
/// # Errors
/// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if any parameter is `NaN`/`Inf`, `beta <= 0.0`,
/// or `head_dim` is zero or odd.
#[inline(always)]
pub const fn eval_barrier_pitchfork(
    cvoid: f32,
    tau_eff: f32,
    alpha: f32,
    beta: f32,
    head_dim: u32,
) -> GcsoResult<f32> {
    let valid_params = branchless_is_finite_f32(cvoid)
        & branchless_is_finite_f32(tau_eff)
        & branchless_is_finite_f32(alpha)
        & branchless_is_finite_f32(beta)
        & (beta > 0.0)
        & (head_dim != 0)
        & ((head_dim & 1) == 0);
    if !valid_params {
        return Err(GCSO_ERROR_EDBC_SINGULARITY);
    }

    let diff = branchless_relu_f32(cvoid - tau_eff);
    let prod = alpha * diff;
    let num = prod * prod;
    let den = 4.0 * beta * (head_dim as f32);
    let safe_den = branchless_max_f32(den, F32_MIN_POSITIVE_NORMAL);
    let inv_den = fast_inv_pos_normal_f32(safe_den);
    let barrier = branchless_flush_subnormal_f32(num * inv_den);
    if !branchless_is_finite_f32(barrier) {
        return Err(GCSO_ERROR_EDBC_SINGULARITY);
    }
    Ok(barrier)
}

/// Evaluates the energy-guided dynamic sampling temperature for an explicitly resolved `mode`
/// (`0` = Normal, `1` = Pitchfork Bifurcation, `2` = Repulsion) in **strictly branchless constant
/// time (`0` branches)** across all 3 modes.
#[inline(always)]
#[must_use]
const fn eval_temperature_for_mode(
    mode: u32,
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
            DEFAULT_SAMPLE_TEMPERATURE,
        ),
        0.05,
        5.0,
    );
    if mode == EDBC_MODE_NORMAL_ENTROPY {
        return safe_base;
    }
    let norm_h = branchless_clamp_f32(branchless_relu_f32(moving_z_entropy) * INV_LN_2, 0.0, 1.0);
    let candidate = if mode == EDBC_MODE_PITCHFORK_BIFURCATION {
        let escape_p = branchless_eyring_kramers_escape(cvoid_score, tau_eff, alpha, beta, d0);
        safe_base * (1.0 + 0.35 * escape_p * norm_h)
    } else {
        safe_base * (1.0 + 0.25 * norm_h)
    };
    branchless_clamp_f32(candidate, 0.05, 5.0)
}

// ===================================================================
// 2. Internal Register-Chained Single-Pass Helpers
// ===================================================================

/// Single-pass register-chained evaluator that simultaneously bridges all consecutive inter-memory
/// **Limina** hops (`trail_chain[0..len]`), records self-contradictory hops into `bloom_filter`,
/// and accumulates the post-bridge **Joint Probability Chain Rule Log-Likelihood**
/// $\sum_{t=1}^{T-1} \ln P(M_t \mid M_{t-1})$ and **$S^1$ Sheaf Holonomy Obstruction Energy**
/// $\mathcal{E}_{\mathrm{holonomy}} \in [0, 1]$ without data-dependent branch mispredictions.
#[inline(always)]
fn bridge_and_eval_liminal_trail_chain_single_pass(
    trail_chain: &mut [gcso_pointer_trail_t],
    interpolation_u7: u8,
    contradiction_threshold: f32,
    obstruction_threshold: f32,
    bloom_filter: &mut ContrastivePhaseBloomFilter,
) -> GcsoResult<(f32, f32, bool, u32, [u64; 8])> {
    let len = trail_chain.len();
    if len == 0
        || !branchless_is_finite_f32(contradiction_threshold)
        || !branchless_is_finite_f32(obstruction_threshold)
    {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    let first_w = *trail_phase_words_ref(&trail_chain[0]);
    if len == 1 {
        return Ok((0.0, 0.0, false, 0, first_w));
    }

    let safe_contradiction =
        branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
    let t_s = branchless_select_u8(interpolation_u7 > 128, 128, interpolation_u7) as u64;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Guarded by `target_feature = "avx2"`. Keeps `(v_first0, v_first1)` and
        // `(v_prev0, v_prev1)` 100% resident in YMM registers across all Limina hops `1..len`,
        // eliminating all per-hop GPR-to-YMM (`vpinsrq`) and scalar 8-lane SWAR scale/diff loops.
        return unsafe {
            use core::arch::x86_64::{
                _mm256_add_epi8, _mm256_and_si256, _mm256_broadcastb_epi8, _mm256_castsi256_si128,
                _mm256_max_epi8, _mm256_mulhrs_epi16, _mm256_or_si256, _mm256_set1_epi16,
                _mm256_set1_epi8, _mm256_setzero_si256, _mm256_slli_epi16, _mm256_sub_epi8,
                _mm256_testz_si256, _mm256_xor_si256, _mm_cvtsi128_si32,
            };
            // Load `(v_first0, v_first1)` via 64-bit word moves (`avx2_words64_to_vec64`) rather than
            // `_mm256_loadu_si256` at byte offset 56 (which straddles the 64B cache-line boundary
            // and triggers a 4-to-1 Store-to-Load Forwarding stall after 64-bit GPR trail resets).
            let (v_first0, v_first1) = crate::traits::branchless::avx2_words64_to_vec64(&first_w);
            let mut v_prev0 = v_first0;
            let mut v_prev1 = v_first1;
            let mut last_out_w = first_w;
            let mut prev_cur_ptr = trail_chain[0].current_ptr;
            let mut prev_step_count = trail_chain[0].step_count;
            let mut prev_target_anchor_id = trail_chain[0].target_anchor_id;
            let mut prev_pull_force = trail_chain[0].attractor_pull_force;
            let mut joint_log_prob = 0.0_f32;
            let mut coboundary_defect_sum = 0.0_f32;
            let mut repulsed_count = 0u32;

            let v_s = _mm256_set1_epi16(t_s as i16);
            let mask_ff00 = _mm256_set1_epi16(-256i16);
            let mask_00ff = _mm256_set1_epi16(0x00FFi16);
            let zero = _mm256_setzero_si256();
            let min_m127 = _mm256_set1_epi8(-127);

            let mut idx = 1usize;
            while idx < len {
                let curr_trail = &mut trail_chain[idx];
                let s_words = *trail_phase_words_ref(curr_trail);
                let (vs0, vs1) = crate::traits::branchless::avx2_words64_to_vec64(&s_words);
                let vd0 = _mm256_sub_epi8(vs0, v_prev0);
                let vd1 = _mm256_sub_epi8(vs1, v_prev1);
                let vd_or = _mm256_or_si256(vd0, vd1);
                let (pre_coh, contradiction, v_out0, v_out1, post_coh, store_out) =
                    if _mm256_testz_si256(vd_or, vd_or) != 0 {
                        (1.0_f32, false, vs0, vs1, 1.0_f32, false)
                    } else {
                        let lo128 = _mm256_castsi256_si128(vd0);
                        let splat_b0 = _mm256_broadcastb_epi8(lo128);
                        let diff_b0 = _mm256_or_si256(
                            _mm256_xor_si256(vd0, splat_b0),
                            _mm256_xor_si256(vd1, splat_b0),
                        );
                        let diff_is_byte_uni = _mm256_testz_si256(diff_b0, diff_b0) != 0;
                        let b0_u8 = (_mm_cvtsi128_si32(lo128) & 0xFF) as u8;
                        let pre_c = if diff_is_byte_uni {
                            Q7_FULL_COS_REF[b0_u8 as usize]
                        } else {
                            crate::traits::branchless::avx2_coherence_from_diff_vec64(vd0, vd1)
                        };
                        let contr = pre_c < -safe_contradiction;
                        if contr {
                            let nw0 = _mm256_sub_epi8(zero, _mm256_max_epi8(vs0, min_m127));
                            let nw1 = _mm256_sub_epi8(zero, _mm256_max_epi8(vs1, min_m127));
                            let pc = crate::traits::branchless::avx2_coherence_from_diff_vec64(
                                _mm256_sub_epi8(nw0, v_prev0),
                                _mm256_sub_epi8(nw1, v_prev1),
                            );
                            (pre_c, true, nw0, nw1, pc, true)
                        } else if t_s >= 128 {
                            (pre_c, false, vs0, vs1, pre_c, false)
                        } else if t_s == 0 {
                            (pre_c, false, v_prev0, v_prev1, 1.0_f32, true)
                        } else if diff_is_byte_uni {
                            let st_i8 = ((((b0_u8 as i8) as i32) * (t_s as i32) + 64) >> 7) as i8;
                            if st_i8 == 0 {
                                (pre_c, false, v_prev0, v_prev1, 1.0_f32, true)
                            } else {
                                let vst = _mm256_set1_epi8(st_i8);
                                let nw0 = _mm256_add_epi8(v_prev0, vst);
                                let nw1 = _mm256_add_epi8(v_prev1, vst);
                                let pc = Q7_FULL_COS_REF[(st_i8 as u8) as usize];
                                (pre_c, false, nw0, nw1, pc, true)
                            }
                        } else {
                            let ev0 = _mm256_mulhrs_epi16(_mm256_slli_epi16(vd0, 8), v_s);
                            let od0 = _mm256_mulhrs_epi16(_mm256_and_si256(vd0, mask_ff00), v_s);
                            let vst0 = _mm256_or_si256(
                                _mm256_and_si256(ev0, mask_00ff),
                                _mm256_slli_epi16(od0, 8),
                            );
                            let ev1 = _mm256_mulhrs_epi16(_mm256_slli_epi16(vd1, 8), v_s);
                            let od1 = _mm256_mulhrs_epi16(_mm256_and_si256(vd1, mask_ff00), v_s);
                            let vst1 = _mm256_or_si256(
                                _mm256_and_si256(ev1, mask_00ff),
                                _mm256_slli_epi16(od1, 8),
                            );
                            let vst_or = _mm256_or_si256(vst0, vst1);
                            if _mm256_testz_si256(vst_or, vst_or) != 0 {
                                (pre_c, false, v_prev0, v_prev1, 1.0_f32, true)
                            } else {
                                let nw0 = _mm256_add_epi8(v_prev0, vst0);
                                let nw1 = _mm256_add_epi8(v_prev1, vst1);
                                let pc = crate::traits::branchless::avx2_coherence_from_diff_vec64(
                                    vst0, vst1,
                                );
                                (pre_c, false, nw0, nw1, pc, true)
                            }
                        }
                    };
                let mut stored_ow = s_words;
                if store_out {
                    stored_ow = crate::traits::branchless::avx2_vec64_to_words64(v_out0, v_out1);
                    *trail_phase_words_mut(curr_trail) = stored_ow;
                }
                last_out_w = stored_ow;

                curr_trail.prev_ptr = prev_cur_ptr;
                curr_trail.step_count = prev_step_count.wrapping_add(1);
                let inherit_anchor = (curr_trail.target_anchor_id == 0) & (!contradiction);
                curr_trail.target_anchor_id = branchless_select_u32(
                    inherit_anchor,
                    prev_target_anchor_id,
                    curr_trail.target_anchor_id,
                );
                curr_trail.cluster_id = branchless_select_u32(
                    inherit_anchor & (prev_target_anchor_id != 0),
                    prev_target_anchor_id,
                    curr_trail.cluster_id,
                );

                let cur_ptr = gcso_tagged_ptr_t(curr_trail.current_ptr);
                let rep_ptr = cur_ptr
                    .with_action_and_slot(
                        ATTRACTOR_SLOT_REPULSION,
                        cur_ptr.action_mode_flags() | ACTION_PRESET_REPULSION_ESCAPE,
                    )
                    .conjugate_phase();
                curr_trail.current_ptr =
                    branchless_select_u64(contradiction, rep_ptr.raw(), cur_ptr.raw());

                let cur_rho = branchless_flush_subnormal_f32(branchless_relu_f32(
                    curr_trail.stigmergic_density,
                ));
                let pos_rho = cur_rho + 0.25 * (1.0 + branchless_relu_f32(pre_coh));
                let neg_rho = cur_rho * 0.5;
                curr_trail.stigmergic_density = branchless_min_f32(
                    branchless_flush_subnormal_f32(branchless_select_f32(
                        contradiction,
                        neg_rho,
                        pos_rho,
                    )),
                    MAX_STIGMERGIC_DENSITY,
                );

                let prev_pull =
                    branchless_flush_subnormal_f32(branchless_relu_f32(prev_pull_force));
                let cur_pull = branchless_flush_subnormal_f32(branchless_relu_f32(
                    curr_trail.attractor_pull_force,
                ));
                let bridged_pull =
                    cur_pull + 0.5 * prev_pull * branchless_clamp_f32(1.0 + pre_coh, 0.25, 2.0);
                curr_trail.attractor_pull_force = branchless_min_f32(
                    branchless_flush_subnormal_f32(branchless_select_f32(
                        contradiction,
                        cur_pull * 0.5,
                        bridged_pull,
                    )),
                    MAX_ATTRACTOR_PULL_FORCE,
                );

                let is_resonant = (!contradiction) & (pre_coh > 0.25);
                curr_trail.flags |=
                    branchless_select_u32(is_resonant, TRAIL_FLAG_SYNERGY_RESONANCE, 0)
                        | branchless_select_u32(
                            contradiction,
                            TRAIL_FLAG_REPULSION_ACTIVE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE,
                            0,
                        );

                if contradiction {
                    bloom_filter.insert_repulsive_pattern_words_weighted(&stored_ow, 1);
                    repulsed_count = repulsed_count.wrapping_add(1);
                }

                if post_coh < 0.999_999 {
                    joint_log_prob += branchless_fast_transition_log_prob(post_coh);
                    coboundary_defect_sum += branchless_relu_f32(0.5 * (1.0 - post_coh));
                }

                v_prev0 = v_out0;
                v_prev1 = v_out1;
                prev_cur_ptr = curr_trail.current_ptr;
                prev_step_count = curr_trail.step_count;
                prev_target_anchor_id = curr_trail.target_anchor_id;
                prev_pull_force = curr_trail.attractor_pull_force;
                idx += 1;
            }

            let mean_step_defect = coboundary_defect_sum * fast_inv_len_f32(len - 1);
            let l_d0 = _mm256_sub_epi8(v_first0, v_prev0);
            let l_d1 = _mm256_sub_epi8(v_first1, v_prev1);
            let l_or = _mm256_or_si256(l_d0, l_d1);
            let loop_coh = if _mm256_testz_si256(l_or, l_or) != 0 {
                1.0_f32
            } else {
                crate::traits::branchless::avx2_coherence_from_diff_vec64(l_d0, l_d1)
            };
            let loop_defect = branchless_relu_f32(0.5 * (1.0 - loop_coh));
            let holonomy_energy =
                branchless_flush_subnormal_f32(0.5 * (mean_step_defect + loop_defect));
            let is_obstructed = holonomy_energy > branchless_relu_f32(obstruction_threshold);

            Ok((
                branchless_flush_subnormal_f32(joint_log_prob),
                holonomy_energy,
                is_obstructed,
                repulsed_count,
                last_out_w,
            ))
        };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        use crate::traits::branchless::coherence_from_diff_words_64;
        use crate::traits::primitives::{
            branchless_cos_q7, swar_q7_circular_add_x8, swar_q7_circular_diff_x8,
            swar_q7_scale_le128_with_bias_x8,
        };
        let bias_restore = (128u64.wrapping_sub(t_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);

        let first_is_uniform = is_word_array_uniform_64(&first_w);
        let mut prev_w = first_w;
        let mut prev_is_uniform = first_is_uniform;
        let mut prev_cur_ptr = trail_chain[0].current_ptr;
        let mut prev_step_count = trail_chain[0].step_count;
        let mut prev_target_anchor_id = trail_chain[0].target_anchor_id;
        let mut prev_pull_force = trail_chain[0].attractor_pull_force;
        let mut joint_log_prob = 0.0_f32;
        let mut coboundary_defect_sum = 0.0_f32;
        let mut repulsed_count = 0u32;

        let mut idx = 1usize;
        while idx < len {
            let curr_trail = &mut trail_chain[idx];
            let s = *trail_phase_words_ref(curr_trail);
            let (pre_coh, contradiction, out_w, out_is_uniform, post_coh) =
                if (s[0] == prev_w[0]) && swar_words_equal_64(&s, &prev_w) {
                    (1.0_f32, false, s, prev_is_uniform, 1.0_f32)
                } else {
                    let s_is_uniform = is_word_array_uniform_64(&s);
                    let both_uni = prev_is_uniform && s_is_uniform;
                    let d0 = swar_q7_circular_diff_x8(s[0], prev_w[0]);
                    let (d1, d2, d3, d4, d5, d6, d7, diff_is_uniform) = if both_uni {
                        (d0, d0, d0, d0, d0, d0, d0, true)
                    } else {
                        let w1 = swar_q7_circular_diff_x8(s[1], prev_w[1]);
                        let w2 = swar_q7_circular_diff_x8(s[2], prev_w[2]);
                        let w3 = swar_q7_circular_diff_x8(s[3], prev_w[3]);
                        let w4 = swar_q7_circular_diff_x8(s[4], prev_w[4]);
                        let w5 = swar_q7_circular_diff_x8(s[5], prev_w[5]);
                        let w6 = swar_q7_circular_diff_x8(s[6], prev_w[6]);
                        let w7 = swar_q7_circular_diff_x8(s[7], prev_w[7]);
                        let d_uni = (d0 == w1)
                            && ((((d0 ^ w2) | (d0 ^ w3)) | ((d0 ^ w4) | (d0 ^ w5)))
                                | ((d0 ^ w6) | (d0 ^ w7)))
                                == 0;
                        (w1, w2, w3, w4, w5, w6, w7, d_uni)
                    };
                    let diff_is_byte_uniform = diff_is_uniform && (d0 == d0.rotate_right(8));
                    let pre_c = if diff_is_byte_uniform {
                        branchless_cos_q7(d0 as gcso_q7_t)
                    } else if diff_is_uniform {
                        crate::traits::branchless::coherence_from_uniform_diff_word_x8(d0)
                    } else {
                        coherence_from_diff_words_64(d0, d1, d2, d3, d4, d5, d6, d7)
                    };
                    let contr = pre_c < -safe_contradiction;
                    let (ow, ow_uni, post_c) = if contr {
                        let nw = if s_is_uniform {
                            [swar_q7_neg_sat_x8(s[0]); 8]
                        } else {
                            [
                                swar_q7_neg_sat_x8(s[0]),
                                swar_q7_neg_sat_x8(s[1]),
                                swar_q7_neg_sat_x8(s[2]),
                                swar_q7_neg_sat_x8(s[3]),
                                swar_q7_neg_sat_x8(s[4]),
                                swar_q7_neg_sat_x8(s[5]),
                                swar_q7_neg_sat_x8(s[6]),
                                swar_q7_neg_sat_x8(s[7]),
                            ]
                        };
                        let pc = coherence_from_word_arrays_with_both_uniform_flags_64(
                            &nw,
                            s_is_uniform,
                            &prev_w,
                            prev_is_uniform,
                        );
                        (nw, s_is_uniform, pc)
                    } else if t_s >= 128 {
                        (s, s_is_uniform, pre_c)
                    } else if t_s == 0 {
                        (prev_w, prev_is_uniform, 1.0_f32)
                    } else if diff_is_uniform {
                        let st0 = swar_q7_scale_le128_with_bias_x8(d0, t_s, bias_restore);
                        if st0 == 0 {
                            (prev_w, prev_is_uniform, 1.0_f32)
                        } else {
                            let nw = if prev_is_uniform {
                                [swar_q7_circular_add_x8(prev_w[0], st0); 8]
                            } else {
                                [
                                    swar_q7_circular_add_x8(prev_w[0], st0),
                                    swar_q7_circular_add_x8(prev_w[1], st0),
                                    swar_q7_circular_add_x8(prev_w[2], st0),
                                    swar_q7_circular_add_x8(prev_w[3], st0),
                                    swar_q7_circular_add_x8(prev_w[4], st0),
                                    swar_q7_circular_add_x8(prev_w[5], st0),
                                    swar_q7_circular_add_x8(prev_w[6], st0),
                                    swar_q7_circular_add_x8(prev_w[7], st0),
                                ]
                            };
                            let pc = if diff_is_byte_uniform {
                                branchless_cos_q7(st0 as gcso_q7_t)
                            } else {
                                coherence_from_diff_words_64(st0, st0, st0, st0, st0, st0, st0, st0)
                            };
                            (nw, prev_is_uniform, pc)
                        }
                    } else {
                        let st0 = swar_q7_scale_le128_with_bias_x8(d0, t_s, bias_restore);
                        let st1 = swar_q7_scale_le128_with_bias_x8(d1, t_s, bias_restore);
                        let st2 = swar_q7_scale_le128_with_bias_x8(d2, t_s, bias_restore);
                        let st3 = swar_q7_scale_le128_with_bias_x8(d3, t_s, bias_restore);
                        let st4 = swar_q7_scale_le128_with_bias_x8(d4, t_s, bias_restore);
                        let st5 = swar_q7_scale_le128_with_bias_x8(d5, t_s, bias_restore);
                        let st6 = swar_q7_scale_le128_with_bias_x8(d6, t_s, bias_restore);
                        let st7 = swar_q7_scale_le128_with_bias_x8(d7, t_s, bias_restore);
                        if (((st0 | st1) | (st2 | st3)) | ((st4 | st5) | (st6 | st7))) == 0 {
                            (prev_w, prev_is_uniform, 1.0_f32)
                        } else {
                            let nw = [
                                swar_q7_circular_add_x8(prev_w[0], st0),
                                swar_q7_circular_add_x8(prev_w[1], st1),
                                swar_q7_circular_add_x8(prev_w[2], st2),
                                swar_q7_circular_add_x8(prev_w[3], st3),
                                swar_q7_circular_add_x8(prev_w[4], st4),
                                swar_q7_circular_add_x8(prev_w[5], st5),
                                swar_q7_circular_add_x8(prev_w[6], st6),
                                swar_q7_circular_add_x8(prev_w[7], st7),
                            ];
                            let nw_uni = is_word_array_uniform_64(&nw);
                            let pc = if nw_uni && prev_is_uniform {
                                branchless_cos_q7(st0 as gcso_q7_t)
                            } else {
                                coherence_from_diff_words_64(st0, st1, st2, st3, st4, st5, st6, st7)
                            };
                            (nw, nw_uni, pc)
                        }
                    };
                    *trail_phase_words_mut(curr_trail) = ow;
                    (pre_c, contr, ow, ow_uni, post_c)
                };

            curr_trail.prev_ptr = prev_cur_ptr;
            curr_trail.step_count = prev_step_count.wrapping_add(1);
            let inherit_anchor = (curr_trail.target_anchor_id == 0) & (!contradiction);
            curr_trail.target_anchor_id = branchless_select_u32(
                inherit_anchor,
                prev_target_anchor_id,
                curr_trail.target_anchor_id,
            );
            curr_trail.cluster_id = branchless_select_u32(
                inherit_anchor & (prev_target_anchor_id != 0),
                prev_target_anchor_id,
                curr_trail.cluster_id,
            );

            let cur_ptr = gcso_tagged_ptr_t(curr_trail.current_ptr);
            let rep_ptr = cur_ptr
                .with_action_and_slot(
                    ATTRACTOR_SLOT_REPULSION,
                    cur_ptr.action_mode_flags() | ACTION_PRESET_REPULSION_ESCAPE,
                )
                .conjugate_phase();
            curr_trail.current_ptr =
                branchless_select_u64(contradiction, rep_ptr.raw(), cur_ptr.raw());

            let cur_rho =
                branchless_flush_subnormal_f32(branchless_relu_f32(curr_trail.stigmergic_density));
            let pos_rho = cur_rho + 0.25 * (1.0 + branchless_relu_f32(pre_coh));
            let neg_rho = cur_rho * 0.5;
            curr_trail.stigmergic_density = branchless_min_f32(
                branchless_flush_subnormal_f32(branchless_select_f32(
                    contradiction,
                    neg_rho,
                    pos_rho,
                )),
                MAX_STIGMERGIC_DENSITY,
            );

            let prev_pull = branchless_flush_subnormal_f32(branchless_relu_f32(prev_pull_force));
            let cur_pull = branchless_flush_subnormal_f32(branchless_relu_f32(
                curr_trail.attractor_pull_force,
            ));
            let bridged_pull =
                cur_pull + 0.5 * prev_pull * branchless_clamp_f32(1.0 + pre_coh, 0.25, 2.0);
            curr_trail.attractor_pull_force = branchless_min_f32(
                branchless_flush_subnormal_f32(branchless_select_f32(
                    contradiction,
                    cur_pull * 0.5,
                    bridged_pull,
                )),
                MAX_ATTRACTOR_PULL_FORCE,
            );

            let is_resonant = (!contradiction) & (pre_coh > 0.25);
            curr_trail.flags |= branchless_select_u32(is_resonant, TRAIL_FLAG_SYNERGY_RESONANCE, 0)
                | branchless_select_u32(
                    contradiction,
                    TRAIL_FLAG_REPULSION_ACTIVE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE,
                    0,
                );

            if contradiction {
                bloom_filter.insert_repulsive_pattern_words_weighted(&out_w, 1);
                repulsed_count = repulsed_count.wrapping_add(1);
            }

            if post_coh < 0.999_999 {
                joint_log_prob += branchless_fast_transition_log_prob(post_coh);
                coboundary_defect_sum += branchless_relu_f32(0.5 * (1.0 - post_coh));
            }

            prev_w = out_w;
            prev_is_uniform = out_is_uniform;
            prev_cur_ptr = curr_trail.current_ptr;
            prev_step_count = curr_trail.step_count;
            prev_target_anchor_id = curr_trail.target_anchor_id;
            prev_pull_force = curr_trail.attractor_pull_force;
            idx += 1;
        }

        let mean_step_defect = coboundary_defect_sum * fast_inv_len_f32(len - 1);
        let loop_coh = if (first_w[0] == prev_w[0]) && swar_words_equal_64(&first_w, &prev_w) {
            1.0_f32
        } else {
            coherence_from_word_arrays_with_both_uniform_flags_64(
                &first_w,
                first_is_uniform,
                &prev_w,
                prev_is_uniform,
            )
        };
        let loop_defect = branchless_relu_f32(0.5 * (1.0 - loop_coh));
        let holonomy_energy =
            branchless_flush_subnormal_f32(0.5 * (mean_step_defect + loop_defect));
        let is_obstructed = holonomy_energy > branchless_relu_f32(obstruction_threshold);

        Ok((
            branchless_flush_subnormal_f32(joint_log_prob),
            holonomy_energy,
            is_obstructed,
            repulsed_count,
            prev_w,
        ))
    }
}

/// Single-scan register-resident Bayesian candidate-trail selector, pointer rewirer, Limina
/// geodesic bridger, and pre-Softmax Attention score modulator that retains `best_coh` in
/// registers so the transition log-probability $\ln P(M_{j^*} \mid M_t)$ is computed with
/// **zero redundant 64-head coherence reloads**, and directly incorporates `macro_anchor_id`
/// cluster matching in a single pass.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn select_rewire_bridge_and_eval_log_prob_in_registers(
    src_trail: &mut gcso_pointer_trail_t,
    preloaded_src_w_opt: Option<[u64; NUM_SWARM_MICRO_CELLS]>,
    candidate_trails: &mut [gcso_pointer_trail_t],
    attention_scores_opt: Option<&mut [f32]>,
    head_dim: u32,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    contradiction_threshold: f32,
    interpolation_u7: u8,
    repulsion_penalty: f32,
    macro_anchor_id: u32,
    secondary_macro_anchor_id: u32,
    win_w_opt: Option<&[u64; NUM_SWARM_MICRO_CELLS]>,
    bloom_filter: &mut ContrastivePhaseBloomFilter,
) -> (u32, f32, f32, bool) {
    if candidate_trails.is_empty() {
        return (src_trail.linked_trail_id, 0.0, 0.0, false);
    }

    let src_w = match preloaded_src_w_opt {
        Some(w) => w,
        None => *trail_phase_words_ref(src_trail),
    };
    let outcome = select_and_bridge_macro_aligned_bayesian_trail_core(
        src_w,
        win_w_opt,
        macro_anchor_id,
        secondary_macro_anchor_id,
        src_trail,
        candidate_trails,
        attention_scores_opt,
        head_dim,
        phase_interference_gain,
        stigmergic_prior_gain,
        pointer_link_bonus,
        repulsion_penalty,
        interpolation_u7,
        contradiction_threshold,
        0.0,
        false,
    );

    if outcome.found_valid {
        let dst_trail = &mut candidate_trails[outcome.best_idx];
        let cur_rho =
            branchless_flush_subnormal_f32(branchless_relu_f32(src_trail.stigmergic_density));
        src_trail.stigmergic_density = branchless_min_f32(
            branchless_flush_subnormal_f32(
                cur_rho + 0.25 * (1.0 + branchless_relu_f32(outcome.best_coh)),
            ),
            MAX_STIGMERGIC_DENSITY,
        );

        let cand_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(dst_trail.attractor_pull_force));
        let self_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(src_trail.attractor_pull_force));
        src_trail.attractor_pull_force = branchless_min_f32(
            branchless_flush_subnormal_f32(
                self_pull
                    + 0.5 * cand_pull * branchless_clamp_f32(1.0 + outcome.best_coh, 0.25, 2.0),
            ),
            MAX_ATTRACTOR_PULL_FORCE,
        );

        let inherit_anchor = dst_trail.target_anchor_id != 0;
        let inherited_id = branchless_select_u32(
            inherit_anchor,
            dst_trail.target_anchor_id,
            src_trail.target_anchor_id,
        );
        src_trail.target_anchor_id = inherited_id;
        dst_trail.target_anchor_id = inherited_id;
        let resolved_cluster_id = branchless_select_u32(
            dst_trail.cluster_id != 0,
            dst_trail.cluster_id,
            branchless_select_u32(
                (macro_anchor_id != 0) & (src_trail.cluster_id != 0),
                src_trail.cluster_id,
                inherited_id,
            ),
        );
        if resolved_cluster_id != 0 {
            src_trail.cluster_id = resolved_cluster_id;
            dst_trail.cluster_id = resolved_cluster_id;
        }
        src_trail.flags =
            (src_trail.flags | TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE)
                & !TRAIL_FLAG_REPULSION_ACTIVE;
        dst_trail.flags =
            (dst_trail.flags | TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE)
                & !TRAIL_FLAG_REPULSION_ACTIVE;

        let transition_lp = branchless_fast_transition_log_prob(outcome.best_coh);
        (outcome.best_id, outcome.best_score, transition_lp, true)
    } else {
        bloom_filter.insert_repulsive_pattern_words_weighted(&src_w, 1);
        src_trail.stamp_phase_conjugate_repulsion();
        (outcome.best_id, 0.0, 0.0, false)
    }
}

/// Applies Pitchfork Metastable Tunneling phase perturbation (`Lemma 3.1`) directly to
/// register-resident `words_inout: &mut [u64; NUM_SWARM_MICRO_CELLS]` and writes only the
/// perturbed exploration cells (`explore_head_start..MAX_ATTENTION_HEADS`) back to `phases_inout`,
/// eliminating Store-to-Load Forwarding reload stalls in `step_edbc_tunneling_and_select_bayesian_core`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn inject_metastable_tunneling_perturbation_in_words_64(
    phases_inout: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    words_inout: &mut [u64; NUM_SWARM_MICRO_CELLS],
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
        let upd = branchless_q7_add_sat(phases_inout[h], signed_pulse);
        phases_inout[h] = upd;
        let c = h >> 3;
        let shift = (h & 7) << 3;
        words_inout[c] = (words_inout[c] & !(0xFFu64 << shift)) | (u64::from(upd as u8) << shift);
        h += 1;
    }

    let pos_u8 = amp_q7 as u8;
    let neg_u8 = (-amp_q7) as u8;
    let pos_w = u64::from(pos_u8).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let xor_byte = u64::from(pos_u8 ^ neg_u8);

    let mut c = h >> 3;
    while c < NUM_SWARM_MICRO_CELLS {
        let bits_w = (hash >> c) & SWAR_SPLAT_BYTE_64;
        let pulse_w = pos_w ^ bits_w.wrapping_mul(xor_byte);
        let next_w = swar_q7_add_sat_x8(words_inout[c], pulse_w);
        words_inout[c] = next_w;
        store_q7_word_at(phases_inout, c << 3, next_w);
        c += 1;
    }
    Ok(x_star)
}

/// Unified EDBC stateful evaluation + Pitchfork metastable tunneling + single-pass register-resident
/// Bayesian candidate-trail selection & rewiring + pointer mode evolution tail shared across
/// [`EdbcClosedLoopCoordinator::step_with_bayesian_pointer_selection_and_tunneling`],
/// [`EdbcClosedLoopCoordinator::route_coarse_to_fine_and_step_bayesian_tunneling`], and
/// [`EdbcClosedLoopCoordinator::route_top2_liminal_and_step_bayesian_tunneling`].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn step_edbc_tunneling_and_select_bayesian_core<E: EdbcClosedLoopCoordinator + ?Sized>(
    engine: &mut E,
    src_trail: &mut gcso_pointer_trail_t,
    preloaded_src_w_opt: Option<[u64; NUM_SWARM_MICRO_CELLS]>,
    candidate_trails: &mut [gcso_pointer_trail_t],
    token_z_score: f32,
    cvoid_score: f32,
    alpha: f32,
    beta: f32,
    d0: f32,
    head_dim: u32,
    explore_head_start: usize,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    repulsion_penalty: f32,
    interpolation_u7: u8,
    contradiction_threshold: f32,
    high_density_srl_thresh: f32,
    macro_anchor_id: u32,
    secondary_macro_anchor_id: u32,
    win_w_opt: Option<&[u64; NUM_SWARM_MICRO_CELLS]>,
    bloom_filter: &mut ContrastivePhaseBloomFilter,
    mut attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
) -> GcsoResult<(
    gcso_edbc_state_t,
    gcso_tagged_ptr_t,
    u32,
    f32,
    f32,
    f32,
    bool,
)> {
    let mut st = gcso_edbc_state_t::default();
    engine.eval_stateful(token_z_score, &mut st)?;
    let bif_thresh = branchless_select_f32(
        (st.bifurcation_threshold > 0.0) & branchless_is_finite_f32(st.bifurcation_threshold),
        st.bifurcation_threshold,
        DEFAULT_BIFURCATION_THRESHOLD,
    );
    st.bifurcation_threshold = bif_thresh;

    let resolved_mode = engine.evaluate_branch_mode_transition(
        st.active_branch_mode,
        st.moving_z_entropy,
        st.sliding_entropy_rate,
        cvoid_score,
        bif_thresh,
        DEFAULT_TAU_FLUX,
        DEFAULT_HYSTERESIS_BAND,
    );
    st.active_branch_mode = resolved_mode;

    let (tunneling_x_star, eff_preloaded_w) = if resolved_mode == EDBC_MODE_PITCHFORK_BIFURCATION {
        let mut src_w = match preloaded_src_w_opt {
            Some(w) => w,
            None => *trail_phase_words_ref(src_trail),
        };
        let x_star = inject_metastable_tunneling_perturbation_in_words_64(
            &mut src_trail.accumulated_phase_delta,
            &mut src_w,
            gcso_tagged_ptr_t(src_trail.current_ptr),
            cvoid_score,
            bif_thresh,
            alpha,
            beta,
            head_dim,
            explore_head_start,
        )?;
        (x_star, Some(src_w))
    } else {
        (0.0, preloaded_src_w_opt)
    };

    let mut selected_id = 0u32;
    let mut bayes_score = -1.0e30_f32;
    let mut transition_log_prob = 0.0_f32;
    let mut found_coherent = false;

    if resolved_mode != EDBC_MODE_PHASE_CONJUGATE_REPULSION {
        let (sel_id, b_score, trans_lp, found) =
            select_rewire_bridge_and_eval_log_prob_in_registers(
                src_trail,
                eff_preloaded_w,
                candidate_trails,
                attention_scores.as_deref_mut().map(|s| s.as_mut_slice()),
                head_dim,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                contradiction_threshold,
                interpolation_u7,
                repulsion_penalty,
                macro_anchor_id,
                secondary_macro_anchor_id,
                win_w_opt,
                bloom_filter,
            );
        selected_id = sel_id;
        bayes_score = b_score;
        transition_log_prob = trans_lp;
        found_coherent = found;

        if !found {
            st.active_branch_mode = EDBC_MODE_PHASE_CONJUGATE_REPULSION;
            st.repulsion_gain = LUT_REPULSION_GAIN[2];
        }
    } else {
        match eff_preloaded_w {
            Some(ref w) => bloom_filter.insert_repulsive_pattern_words_weighted(w, 1),
            None => bloom_filter.insert_repulsive_pattern(&src_trail.accumulated_phase_delta),
        }
        src_trail.stamp_phase_conjugate_repulsion();
        st.active_branch_mode = EDBC_MODE_PHASE_CONJUGATE_REPULSION;
        st.repulsion_gain = LUT_REPULSION_GAIN[2];

        if let Some(scores) = attention_scores {
            let (_mean_coh, best_id, _coh_cnt) = src_trail
                .modulate_candidate_trail_scores_in_place(
                    scores.as_mut_slice(),
                    candidate_trails,
                    head_dim,
                    phase_interference_gain,
                    stigmergic_prior_gain,
                    pointer_link_bonus,
                    contradiction_threshold,
                    repulsion_penalty,
                );
            selected_id = best_id;
        }
    }

    let repulsion_active = !found_coherent;
    let dom_lane = branchless_select_u8(
        src_trail.target_anchor_id != 0,
        (src_trail.target_anchor_id as u8) & ATTRACTOR_SLOT_ID_MASK,
        src_trail.target_attractor_slot(),
    );
    let pre_evolve_cluster_id = src_trail.cluster_id;
    let evolved_ptr = src_trail.evolve_trail_mode_stigmergically(
        st.moving_z_entropy,
        repulsion_active,
        dom_lane,
        branchless_relu_f32(bif_thresh * 0.5),
        high_density_srl_thresh,
    );
    if (macro_anchor_id != 0) & (!repulsion_active) & (pre_evolve_cluster_id != 0) {
        src_trail.cluster_id = pre_evolve_cluster_id;
    }
    st.sample_temperature = if st.active_branch_mode == EDBC_MODE_NORMAL_ENTROPY {
        DEFAULT_SAMPLE_TEMPERATURE
    } else {
        eval_temperature_for_mode(
            st.active_branch_mode,
            DEFAULT_SAMPLE_TEMPERATURE,
            st.moving_z_entropy,
            cvoid_score,
            bif_thresh,
            alpha,
            beta,
            d0,
        )
    };

    Ok((
        st,
        evolved_ptr,
        selected_id,
        bayes_score,
        transition_log_prob,
        tunneling_x_star,
        found_coherent,
    ))
}

/// Register-resident core helper for [`EdbcClosedLoopCoordinator::step_via_gateway_frame`],
/// [`EdbcClosedLoopCoordinator::step_via_gateway_signal`], and
/// [`EdbcClosedLoopCoordinator::step_via_gateway_kmeans_64`] that operates directly on
/// `gw_w: &[u64; 8]` without reloading `frame.phase_deltas` from the stack.
#[inline(always)]
fn step_via_gateway_frame_with_words_core<
    T: EdbcBranchController + SheafCohomologyEvaluator + HierarchicalTreeTable + ?Sized,
>(
    engine: &mut T,
    token_z_score: f32,
    cvoid_score: f32,
    alpha: f32,
    beta: f32,
    d0: f32,
    high_density_srl_thresh: f32,
    frame: &GatewayFrame,
    gw_w: &[u64; NUM_SWARM_MICRO_CELLS],
    trail: &mut gcso_pointer_trail_t,
    logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    crystal_record_opt: Option<&mut CrystallizedAttractorRecord>,
) -> GcsoResult<(
    gcso_edbc_state_t,
    gcso_tagged_ptr_t,
    CausalProvenanceReport,
    f32,
    f32,
)> {
    if !branchless_is_finite_f32(high_density_srl_thresh) {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    if !branchless_is_finite_f32(cvoid_score)
        || !branchless_is_finite_f32(alpha)
        || !branchless_is_finite_f32(beta)
        || !branchless_is_finite_f32(d0)
        || beta <= 0.0
        || d0 <= 0.0
    {
        return Err(GCSO_ERROR_EDBC_SINGULARITY);
    }

    let mut raw_st = gcso_edbc_state_t::default();
    engine.eval_stateful(token_z_score, &mut raw_st)?;
    let moving_z_entropy = raw_st.moving_z_entropy;
    let sliding_entropy_rate = raw_st.sliding_entropy_rate;
    let bif_thresh = branchless_select_f32(
        (raw_st.bifurcation_threshold > 0.0)
            & branchless_is_finite_f32(raw_st.bifurcation_threshold),
        raw_st.bifurcation_threshold,
        DEFAULT_BIFURCATION_THRESHOLD,
    );

    let resolved_mode = engine.evaluate_branch_mode_transition(
        raw_st.active_branch_mode,
        moving_z_entropy,
        sliding_entropy_rate,
        cvoid_score,
        bif_thresh,
        DEFAULT_TAU_FLUX,
        DEFAULT_HYSTERESIS_BAND,
    );

    let frame_repulsion = (frame.dominant_cause_code == ROOT_CAUSE_COHOMOLOGY_REPULSION)
        | ((frame.provenance_bits & CAUSAL_CAUSE_COHOMOLOGY_REPULSION) != 0);
    let repulsion_triggered =
        (resolved_mode == EDBC_MODE_PHASE_CONJUGATE_REPULSION) | frame_repulsion;
    let final_mode = branchless_select_u32(
        repulsion_triggered,
        EDBC_MODE_PHASE_CONJUGATE_REPULSION,
        resolved_mode,
    );
    let final_rep_gain = LUT_REPULSION_GAIN[(final_mode as usize).min(2)];

    let cur_pull = branchless_flush_subnormal_f32(branchless_relu_f32(trail.attractor_pull_force));
    let gw_pull = branchless_flush_subnormal_f32(branchless_relu_f32(frame.effective_energy));
    let total_pull = cur_pull + gw_pull;
    let has_cur_pull = cur_pull > 1.0e-12;
    let qdps_step = frame.qdps_min_step_q7;
    #[allow(unused_mut)]
    let mut qdps_filtered_in_vec = false;
    let (mut next_w, next_eq_gw, mut next_eq_cur) = if gw_pull <= 1.0e-12 {
        (*trail_phase_words_ref(trail), false, true)
    } else if !has_cur_pull {
        (*gw_w, true, false)
    } else {
        let cur_w_ref = trail_phase_words_ref(trail);
        if (cur_w_ref[0] == gw_w[0]) && swar_words_equal_64(cur_w_ref, gw_w) {
            (*gw_w, true, true)
        } else {
            let ratio_f32 = branchless_clamp_f32(
                (gw_pull * fast_inv_pos_normal_f32(total_pull)) * 128.0 + 0.5,
                0.0,
                128.0,
            );
            let s = (ratio_f32 as u8) as u64;
            if s == 0 {
                (*cur_w_ref, false, true)
            } else if s >= 128 {
                (*gw_w, true, false)
            } else {
                let cur_uni = is_word_array_uniform_64(cur_w_ref);
                let gw_uni = is_word_array_uniform_64(gw_w);
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                {
                    if cur_uni && gw_uni {
                        let bias_restore =
                            (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                        let w0 = crate::traits::branchless::swar_slerp_word_unchecked_x8(
                            cur_w_ref[0],
                            gw_w[0],
                            s,
                            bias_restore,
                        );
                        ([w0; NUM_SWARM_MICRO_CELLS], false, false)
                    } else {
                        // SAFETY: `target_feature = "avx2"` is enabled; loads `trail.accumulated_phase_delta`
                        // directly from L1D and converts `gw_w` via register moves (`avx2_words64_to_vec64_with_uniform`)
                        // to eliminate 64B stack copy and 64-bit-to-256-bit Store-to-Load Forwarding (STLF) stalls.
                        let nw = unsafe {
                            let (vc0, vc1) =
                                crate::traits::branchless::avx2_words64_to_vec64_with_uniform(
                                    cur_w_ref, cur_uni,
                                );
                            let (vg0, vg1) =
                                crate::traits::branchless::avx2_words64_to_vec64_with_uniform(
                                    gw_w, gw_uni,
                                );
                            let (vr0, vr1) =
                                crate::traits::branchless::avx2_slerp_vec64(vc0, vc1, vg0, vg1, s);
                            let (vf0, vf1) = if qdps_step > 1 {
                                qdps_filtered_in_vec = true;
                                crate::traits::branchless::avx2_qdps_filter_vec64(
                                    vr0, vr1, qdps_step,
                                )
                            } else {
                                (vr0, vr1)
                            };
                            crate::traits::branchless::avx2_vec64_to_words64(vf0, vf1)
                        };
                        (nw, false, false)
                    }
                }
                #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                {
                    let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                    let (nw, _) = crate::traits::swar_slerp_words_with_uniform_flag_64(
                        cur_w_ref,
                        cur_uni,
                        gw_w,
                        gw_uni,
                        s,
                        bias_restore,
                    );
                    (nw, false, false)
                }
            }
        }
    };

    if (qdps_step > 1) & (!qdps_filtered_in_vec) & has_cur_pull & (!next_eq_gw)
        && (!swar_words_equal_64(&next_w, gw_w))
    {
        next_w = swar_q7_qdps_filter_words_64(&next_w, qdps_step);
        next_eq_cur = false;
    }
    if repulsion_triggered & (!frame_repulsion) {
        next_w = [
            swar_q7_neg_sat_x8(next_w[0]),
            swar_q7_neg_sat_x8(next_w[1]),
            swar_q7_neg_sat_x8(next_w[2]),
            swar_q7_neg_sat_x8(next_w[3]),
            swar_q7_neg_sat_x8(next_w[4]),
            swar_q7_neg_sat_x8(next_w[5]),
            swar_q7_neg_sat_x8(next_w[6]),
            swar_q7_neg_sat_x8(next_w[7]),
        ];
        next_eq_cur = false;
    }
    if (!has_cur_pull) | (!next_eq_cur) {
        *trail_phase_words_mut(trail) = next_w;
    }

    trail.prev_ptr = trail.current_ptr;
    trail.step_count = trail.step_count.wrapping_add(1);
    trail.attractor_pull_force = branchless_min_f32(total_pull, MAX_ATTRACTOR_PULL_FORCE);
    let cur_rho = branchless_flush_subnormal_f32(branchless_relu_f32(trail.stigmergic_density));
    let next_rho = branchless_select_f32(
        repulsion_triggered,
        cur_rho * 0.5,
        branchless_min_f32(
            cur_rho + 0.20 * branchless_clamp_f32(1.0 + frame.fact_persona_coherence, 0.25, 2.0),
            MAX_STIGMERGIC_DENSITY,
        ),
    );
    trail.stigmergic_density = branchless_flush_subnormal_f32(next_rho);

    let has_frame_anchor = frame.target_anchor_id != 0;
    let dom_anchor = branchless_select_u32(
        has_frame_anchor,
        frame.target_anchor_id,
        trail.target_anchor_id,
    );
    trail.target_anchor_id = dom_anchor;
    let dom_lane = branchless_select_u8(
        dom_anchor != 0,
        (dom_anchor as u8) & ATTRACTOR_SLOT_ID_MASK,
        frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK,
    );

    let base_ptr = gcso_tagged_ptr_t(frame.tagged_ptr);
    let low_entropy_skip_thresh = branchless_relu_f32(bif_thresh * 0.5);
    let mut evolved_ptr = base_ptr.evolve_mode_stigmergically(
        trail.stigmergic_density,
        moving_z_entropy,
        repulsion_triggered,
        dom_lane,
        low_entropy_skip_thresh,
        high_density_srl_thresh,
    );
    if ((frame.action_mode_flags & ACTION_MODE_LAYER_SKIP) != 0)
        & (final_mode == EDBC_MODE_NORMAL_ENTROPY)
        & (!repulsion_triggered)
    {
        evolved_ptr = evolved_ptr.with_action_and_slot(dom_lane, ACTION_PRESET_INERTIAL_SKIP);
    }
    trail.current_ptr = evolved_ptr.raw();
    trail.cluster_id = branchless_select_u32(
        (dom_anchor != 0) & (!repulsion_triggered),
        dom_anchor,
        evolved_ptr.target_attractor_slot() as u32,
    );
    trail.sync_flags_with_current_ptr();
    if repulsion_triggered {
        trail.flags |= TRAIL_FLAG_REPULSION_ACTIVE;
    }

    let gw_inv_temp = branchless_select_f32(
        branchless_is_finite_f32(frame.effective_inv_temp) & (frame.effective_inv_temp > 0.0),
        frame.effective_inv_temp,
        1.0,
    );
    let (edbc_temp, inv_edbc_temp) = if final_mode == EDBC_MODE_NORMAL_ENTROPY {
        (DEFAULT_SAMPLE_TEMPERATURE, 1.0_f32)
    } else {
        let t = eval_temperature_for_mode(
            final_mode,
            DEFAULT_SAMPLE_TEMPERATURE,
            moving_z_entropy,
            cvoid_score,
            bif_thresh,
            alpha,
            beta,
            d0,
        );
        (t, fast_inv_pos_normal_f32(branchless_max_f32(t, 0.05)))
    };
    let eff_inv_temp = branchless_clamp_f32(gw_inv_temp * inv_edbc_temp, 0.05, 8.0);
    if let Some(logits) = logits_opt {
        scale_f32_slice_unrolled8(logits.as_mut_slice(), eff_inv_temp);
    }

    let mut report = frame.to_causal_report();
    report.dominant_slot = evolved_ptr.target_attractor_slot();
    report.action_mode_flags = evolved_ptr.action_mode_flags();
    if repulsion_triggered {
        report.provenance_bits |= CAUSAL_CAUSE_COHOMOLOGY_REPULSION;
        report.dominant_cause_code = ROOT_CAUSE_COHOMOLOGY_REPULSION;
    } else if (evolved_ptr.action_mode_flags() & ACTION_MODE_LAYER_SKIP) != 0
        && (evolved_ptr.action_mode_flags() & ACTION_MODE_DPSR_STEER) == 0
    {
        report.provenance_bits |= CAUSAL_AXIS_SYSTEM_CONSTRAINT | CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE;
        if report.dominant_cause_code == ROOT_CAUSE_NOMINAL {
            report.dominant_cause_code = ROOT_CAUSE_SYSTEM_QOS_THROTTLE;
        }
    }

    let mut crystal_mass = 0.0_f32;
    if let Some(crystal) = crystal_record_opt {
        if (!repulsion_triggered) & (gw_pull > 1.0e-12) {
            let new_dom = evolved_ptr.target_attractor_slot();
            let new_act = evolved_ptr.action_mode_flags();
            crystal_mass = crystal.absorb_gateway_frame_with_words_and_overrides(
                frame,
                &next_w,
                new_dom,
                new_act,
                report.provenance_bits,
            );
            if (crystal.provenance_bits & CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED) != 0 {
                trail.flags |= TRAIL_FLAG_CRYSTALLIZED_CANDIDATE;
                report.provenance_bits |= CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED;
            }
        } else {
            crystal_mass = crystal.phase_gravity_mass;
        }
    }

    raw_st.bifurcation_threshold = bif_thresh;
    raw_st.repulsion_gain = final_rep_gain;
    raw_st.sample_temperature = edbc_temp;
    raw_st.active_branch_mode = final_mode;
    Ok((raw_st, evolved_ptr, report, eff_inv_temp, crystal_mass))
}

// ===================================================================
// 3. Blanket-Implemented Closed-Loop EDBC + Sheaf + Limina + Gateway Coordinator
// ===================================================================

/// Blanket-implemented closed-loop coordinator uniting [`EdbcBranchController`],
/// [`SheafCohomologyEvaluator`], [`HierarchicalTreeTable`], [`ManifoldGatewaySignal`],
/// Online Toroidal $k$-Means / DP-Means Coarse-to-Fine routing, and `.gcso`
/// [`CrystallizedAttractorRecord`] persistence.
///
/// Automatically implemented for all `T: EdbcBranchController + SheafCohomologyEvaluator + HierarchicalTreeTable + ?Sized`
/// (including [`EdbcEngine`], [`crate::traits::ColdPathObserver`], and [`crate::traits::FullStackGcsoEngine`]).
pub trait EdbcClosedLoopCoordinator:
    EdbcBranchController + SheafCohomologyEvaluator + HierarchicalTreeTable
{
    /// Computes the mean-square $\mathcal{C}_{\mathrm{void}}$ latent energy alongside the
    /// normalized $\ell_1 / \ell_2$ coherence ratio in a single AVX2 pass (`sum_abs_and_sq_f32_simd8`)
    /// with zero hardware `fdiv` instructions via `fast_inv_len_f32`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `key_vector` is empty.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if any element or accumulated norm is non-finite.
    #[inline(always)]
    fn eval_cvoid_with_coherence_ratio(
        &self,
        key_vector: &AlignedSlice32<'_, f32>,
    ) -> GcsoResult<(f32, f32)> {
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
        let mean_abs = sum_abs * inv_n;
        let valid_rms = mean_sq > 1.0e-24;
        let inv_rms = branchless_select_f32(
            valid_rms,
            branchless_rsqrt_f32(branchless_select_f32(valid_rms, mean_sq, 1.0)),
            0.0,
        );
        let coh_ratio =
            branchless_flush_subnormal_f32(branchless_clamp_f32(mean_abs * inv_rms, 0.0, 1.0));
        Ok((mean_sq, coh_ratio))
    }

    /// Evaluates Top-2 sparse surrogate entropy via the strictly branchless `0-ln` Taylor-Horner
    /// polynomial, combines it with inter-layer cosine distance into Moving Z-Score Normalized
    /// Attention Entropy $\tilde{H}$, and steps the EDBC controller via
    /// [`EntropyEvaluator::eval_stateful`].
    ///
    /// # Errors
    /// Propagates [`GCSO_ERROR_EDBC_SINGULARITY`] if computed entropy is non-finite.
    #[inline(always)]
    fn eval_surrogate_and_step(
        &mut self,
        p_max: f32,
        p_2nd: f32,
        cosine_dist: f32,
        mu_dist: f32,
        sigma_dist: f32,
        alpha_z: f32,
        gamma_mod: f32,
        state_out: &mut gcso_edbc_state_t,
    ) -> GcsoResult<f32> {
        let eps = branchless_select_f32(
            (state_out.singularity_eps > 0.0) & branchless_is_finite_f32(state_out.singularity_eps),
            state_out.singularity_eps,
            DEFAULT_SINGULARITY_EPS,
        );
        let h_surr = compute_sparse_surrogate_entropy(p_max, p_2nd, eps);
        let h_tilde = compute_moving_z_score_entropy(
            h_surr,
            cosine_dist,
            mu_dist,
            sigma_dist,
            eps,
            alpha_z,
            gamma_mod,
        );
        self.eval_stateful(h_tilde, state_out)?;
        Ok(state_out.moving_z_entropy)
    }

    /// Bridges all inter-memory **Limina** transitions across `trail_chain` ($M_1 \to \dots \to M_T$)
    /// and evaluates the **Joint Probability Chain Rule Log-Likelihood**
    /// $\ln P(M_1, \dots, M_T) = \sum_{t=1}^{T-1} \ln P(M_t \mid M_{t-1})$ and **$S^1$ Sheaf
    /// Holonomy Obstruction Energy** ($H^1(K; \mathcal{F})$) in a **single register-chained pass**,
    /// steps the stateful EDBC controller, and updates the terminal trail's pointer mode and
    /// Bloom repulsion memory.
    ///
    /// Returns `(edbc_state, evolved_ptr, joint_log_prob, holonomy_obstruction, repulsion_triggered)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `trail_chain` is empty or thresholds are non-finite.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if `token_z_score` or `cvoid_score` is non-finite.
    #[inline(always)]
    fn step_with_liminal_trail_chain_and_bloom_guard(
        &mut self,
        trail_chain: &mut [gcso_pointer_trail_t],
        interpolation_u7: u8,
        contradiction_threshold: f32,
        obstruction_threshold: f32,
        token_z_score: f32,
        cvoid_score: f32,
        high_density_srl_thresh: f32,
        dominant_attractor_slot: u8,
        bloom_filter: &mut ContrastivePhaseBloomFilter,
    ) -> GcsoResult<(gcso_edbc_state_t, gcso_tagged_ptr_t, f32, f32, bool)> {
        if trail_chain.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !branchless_is_finite_f32(cvoid_score) {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        let (joint_log_prob, holonomy_obs, is_obstructed, repulsed_transitions, last_w) =
            bridge_and_eval_liminal_trail_chain_single_pass(
                trail_chain,
                interpolation_u7,
                contradiction_threshold,
                obstruction_threshold,
                bloom_filter,
            )?;

        let mut st = gcso_edbc_state_t::default();
        self.eval_stateful(token_z_score, &mut st)?;
        let bif_thresh = branchless_select_f32(
            (st.bifurcation_threshold > 0.0) & branchless_is_finite_f32(st.bifurcation_threshold),
            st.bifurcation_threshold,
            DEFAULT_BIFURCATION_THRESHOLD,
        );
        st.bifurcation_threshold = bif_thresh;

        let last_idx = trail_chain.len() - 1;
        let repulsion_triggered = is_obstructed
            | (repulsed_transitions > 0)
            | (st.active_branch_mode == EDBC_MODE_PHASE_CONJUGATE_REPULSION);

        let low_entropy_skip_thresh = branchless_relu_f32(bif_thresh * 0.5);
        let mut evolved_ptr = gcso_tagged_ptr_t(trail_chain[last_idx].current_ptr)
            .evolve_mode_stigmergically(
                trail_chain[last_idx].stigmergic_density,
                st.moving_z_entropy,
                repulsion_triggered,
                dominant_attractor_slot & ATTRACTOR_SLOT_ID_MASK,
                low_entropy_skip_thresh,
                high_density_srl_thresh,
            );

        if repulsion_triggered {
            if (trail_chain[last_idx].flags & TRAIL_FLAG_REPULSION_ACTIVE) == 0 {
                bloom_filter.insert_repulsive_pattern_words_weighted(&last_w, 1);
                let w_dst = trail_phase_words_mut(&mut trail_chain[last_idx]);
                w_dst[0] = swar_q7_neg_sat_x8(last_w[0]);
                w_dst[1] = swar_q7_neg_sat_x8(last_w[1]);
                w_dst[2] = swar_q7_neg_sat_x8(last_w[2]);
                w_dst[3] = swar_q7_neg_sat_x8(last_w[3]);
                w_dst[4] = swar_q7_neg_sat_x8(last_w[4]);
                w_dst[5] = swar_q7_neg_sat_x8(last_w[5]);
                w_dst[6] = swar_q7_neg_sat_x8(last_w[6]);
                w_dst[7] = swar_q7_neg_sat_x8(last_w[7]);
                trail_chain[last_idx].flags |= TRAIL_FLAG_REPULSION_ACTIVE;
            }
            st.active_branch_mode = EDBC_MODE_PHASE_CONJUGATE_REPULSION;
            st.repulsion_gain = LUT_REPULSION_GAIN[2];
            st.sample_temperature =
                branchless_max_f32(st.sample_temperature, LUT_SAMPLE_TEMPERATURE[2]);
            let rep_mode = evolved_ptr.action_mode_flags() | ACTION_PRESET_REPULSION_ESCAPE;
            evolved_ptr = evolved_ptr
                .with_action_and_slot(ATTRACTOR_SLOT_REPULSION, rep_mode)
                .conjugate_phase();
            trail_chain[last_idx].cluster_id = ATTRACTOR_SLOT_REPULSION as u32;
        } else {
            let preserve_anchor = trail_chain[last_idx].target_anchor_id != 0;
            trail_chain[last_idx].cluster_id = branchless_select_u32(
                preserve_anchor,
                trail_chain[last_idx].target_anchor_id,
                evolved_ptr.target_attractor_slot() as u32,
            );
        }
        trail_chain[last_idx].current_ptr = evolved_ptr.raw();

        Ok((
            st,
            evolved_ptr,
            joint_log_prob,
            holonomy_obs,
            repulsion_triggered,
        ))
    }

    /// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) on the
    /// **Single Unified Attractor Manifold** into `trail` in register-resident order-invariant
    /// constant time via [`branchless_q7_unified_field_fold_words_with_init_weight_64`], binds the
    /// 32-bit dominant anchor index into both `trail.target_anchor_id` and `trail.cluster_id`,
    /// steps the EDBC controller, and evolves `trail.current_ptr` using
    /// `(dominant_anchor_id as u8) & 0x07` as the working lane.
    ///
    /// Returns `(edbc_state, evolved_ptr, total_resonant_pull, mean_coherence, dominant_anchor_id)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `anchor_phases.len() != anchor_base_energies.len()`.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if `token_z_score` or `cvoid_score` is non-finite.
    #[inline(always)]
    fn step_with_unified_field_and_pointer(
        &mut self,
        token_z_score: f32,
        cvoid_score: f32,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        high_density_srl_thresh: f32,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(gcso_edbc_state_t, gcso_tagged_ptr_t, f32, f32, u32)> {
        if anchor_phases.len() != anchor_base_energies.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !branchless_is_finite_f32(cvoid_score) {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        let init_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(trail.attractor_pull_force));
        let cur_w = *trail_phase_words_ref(trail);
        let (folded_w, total_weight, mean_coh, dom_idx, act_cnt) =
            branchless_q7_unified_field_fold_words_with_init_weight_64(
                &cur_w,
                init_pull,
                anchor_phases,
                anchor_base_energies,
            );
        if act_cnt > 0 {
            *trail_phase_words_mut(trail) = folded_w;
            trail.target_anchor_id = dom_idx;
            trail.cluster_id = dom_idx;
            trail.attractor_pull_force = branchless_min_f32(total_weight, MAX_ATTRACTOR_PULL_FORCE);
        }

        let mut st = gcso_edbc_state_t::default();
        self.eval_stateful(token_z_score, &mut st)?;
        let bif_thresh = branchless_select_f32(
            (st.bifurcation_threshold > 0.0) & branchless_is_finite_f32(st.bifurcation_threshold),
            st.bifurcation_threshold,
            DEFAULT_BIFURCATION_THRESHOLD,
        );
        st.bifurcation_threshold = bif_thresh;

        let dom_lane = (trail.target_anchor_id as u8) & ATTRACTOR_SLOT_ID_MASK;
        let is_repulsion = st.active_branch_mode == EDBC_MODE_PHASE_CONJUGATE_REPULSION;
        let low_entropy_skip_thresh = branchless_relu_f32(bif_thresh * 0.5);
        let evolved_ptr = gcso_tagged_ptr_t(trail.current_ptr).evolve_mode_stigmergically(
            trail.stigmergic_density,
            st.moving_z_entropy,
            is_repulsion,
            dom_lane,
            low_entropy_skip_thresh,
            high_density_srl_thresh,
        );
        trail.current_ptr = evolved_ptr.raw();
        let preserve_anchor = (trail.target_anchor_id != 0) & (!is_repulsion);
        trail.cluster_id = branchless_select_u32(
            preserve_anchor,
            trail.target_anchor_id,
            evolved_ptr.target_attractor_slot() as u32,
        );

        Ok((
            st,
            evolved_ptr,
            total_weight,
            mean_coh,
            trail.target_anchor_id,
        ))
    }

    /// Executes the single-pass closed-loop synthesis of **EDBC Langevin Pitchfork Bifurcation / Tunneling**,
    /// **Sequential Conditional Transition Probability** ($\sum_t \ln P(M_t \mid M_{t-1})$ $\times$ Attention phase interference),
    /// and **Indirect Pointer-Chain Selection & Rewiring** in `ActionHub`.
    ///
    /// Returns `(edbc_state, evolved_ptr, selected_trail_id, bayesian_score, transition_log_prob, tunneling_x_star, found_coherent)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `candidate_trails` is empty, `explore_head_start > 64`,
    /// or gains/thresholds are non-finite.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if `token_z_score`, `cvoid_score`, `alpha`, `beta <= 0.0`,
    /// `d0 <= 0.0`, or `head_dim` is invalid.
    #[inline(always)]
    fn step_with_bayesian_pointer_selection_and_tunneling(
        &mut self,
        src_trail: &mut gcso_pointer_trail_t,
        candidate_trails: &mut [gcso_pointer_trail_t],
        token_z_score: f32,
        cvoid_score: f32,
        alpha: f32,
        beta: f32,
        d0: f32,
        head_dim: u32,
        explore_head_start: usize,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        repulsion_penalty: f32,
        interpolation_u7: u8,
        contradiction_threshold: f32,
        high_density_srl_thresh: f32,
        bloom_filter: &mut ContrastivePhaseBloomFilter,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(
        gcso_edbc_state_t,
        gcso_tagged_ptr_t,
        u32,
        f32,
        f32,
        f32,
        bool,
    )> {
        if candidate_trails.is_empty()
            || explore_head_start > MAX_ATTENTION_HEADS
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(repulsion_penalty)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !branchless_is_finite_f32(cvoid_score)
            || !branchless_is_finite_f32(alpha)
            || !branchless_is_finite_f32(beta)
            || !branchless_is_finite_f32(d0)
            || beta <= 0.0
            || d0 <= 0.0
            || head_dim == 0
            || (head_dim & 1) != 0
        {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        step_edbc_tunneling_and_select_bayesian_core(
            self,
            src_trail,
            None,
            candidate_trails,
            token_z_score,
            cvoid_score,
            alpha,
            beta,
            d0,
            head_dim,
            explore_head_start,
            phase_interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
            repulsion_penalty,
            interpolation_u7,
            contradiction_threshold,
            high_density_srl_thresh,
            0,
            0,
            None,
            bloom_filter,
            attention_scores,
        )
    }

    /// Executes **Coarse-to-Fine Macro-to-Micro Routing ($\mathcal{O}(K) \to \mathcal{O}(M)$)**
    /// combined with **EDBC Pitchfork Metastable Tunneling & Bayesian Pointer-Trail Rewiring**:
    /// 1. **Coarse Step ($\mathcal{O}(K)$ Macro Centroid Routing)**: Routes `src_trail` to the
    ///    highest-gravity coherent macro centroid $k^*$ in `macro_centroids` via
    ///    `CrystallizedAttractorRecord::route_coarse_to_fine_trail_with_words_64`, Slerp-pulling
    ///    `src_trail` by `coarse_pull_u7`.
    /// 2. **Fine Step ($\mathcal{O}(M)$ + Pitchfork Tunneling)**: Passes `macro_anchor_id` and
    ///    register-resident `bridged_w` directly into `step_edbc_tunneling_and_select_bayesian_core`
    ///    so candidates stamped with `macro_anchor_id` receive the macro-cluster link alignment bonus
    ///    in a single pass without any separate pre-alignment scan over `candidate_trails`.
    ///
    /// Returns `(edbc_state, evolved_ptr, macro_cluster_idx, macro_anchor_id, selected_trail_id, bayesian_score, transition_log_prob, tunneling_x_star, found_coherent)`.
    ///
    /// # Errors
    /// Propagates any validation or singularity error from [`Self::step_with_bayesian_pointer_selection_and_tunneling`].
    #[inline(always)]
    fn route_coarse_to_fine_and_step_bayesian_tunneling(
        &mut self,
        macro_centroids: &[CrystallizedAttractorRecord],
        coarse_pull_u7: u8,
        src_trail: &mut gcso_pointer_trail_t,
        candidate_trails: &mut [gcso_pointer_trail_t],
        token_z_score: f32,
        cvoid_score: f32,
        alpha: f32,
        beta: f32,
        d0: f32,
        head_dim: u32,
        explore_head_start: usize,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        repulsion_penalty: f32,
        fine_interpolation_u7: u8,
        contradiction_threshold: f32,
        high_density_srl_thresh: f32,
        bloom_filter: &mut ContrastivePhaseBloomFilter,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(
        gcso_edbc_state_t,
        gcso_tagged_ptr_t,
        usize,
        u32,
        u32,
        f32,
        f32,
        f32,
        bool,
    )> {
        if candidate_trails.is_empty()
            || explore_head_start > MAX_ATTENTION_HEADS
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(repulsion_penalty)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !branchless_is_finite_f32(cvoid_score)
            || !branchless_is_finite_f32(alpha)
            || !branchless_is_finite_f32(beta)
            || !branchless_is_finite_f32(d0)
            || beta <= 0.0
            || d0 <= 0.0
            || head_dim == 0
            || (head_dim & 1) != 0
        {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        let (macro_idx, macro_anchor_id, _macro_coh, _macro_grav, bridged_w, win_w) =
            CrystallizedAttractorRecord::route_coarse_to_fine_trail_with_words_64(
                macro_centroids,
                src_trail,
                coarse_pull_u7,
            );
        let has_win_centroid =
            macro_idx < macro_centroids.len() && macro_centroids[macro_idx].sample_count > 0;

        let (
            st,
            evolved_ptr,
            selected_id,
            bayes_score,
            transition_log_prob,
            tunneling_x_star,
            found_coherent,
        ) = step_edbc_tunneling_and_select_bayesian_core(
            self,
            src_trail,
            Some(bridged_w),
            candidate_trails,
            token_z_score,
            cvoid_score,
            alpha,
            beta,
            d0,
            head_dim,
            explore_head_start,
            phase_interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
            repulsion_penalty,
            fine_interpolation_u7,
            contradiction_threshold,
            high_density_srl_thresh,
            macro_anchor_id,
            0,
            has_win_centroid.then_some(&win_w),
            bloom_filter,
            attention_scores,
        )?;

        Ok((
            st,
            evolved_ptr,
            macro_idx,
            macro_anchor_id,
            selected_id,
            bayes_score,
            transition_log_prob,
            tunneling_x_star,
            found_coherent,
        ))
    }

    /// Executes **Top-2 Liminal Macro-to-Micro Bifurcation Routing ($\mathcal{O}(K) \to \mathcal{O}(M)$)**
    /// united with **EDBC Pitchfork Metastable Tunneling & Bayesian Pointer-Trail Rewiring**:
    /// 1. **Coarse Step ($\mathcal{O}(K)$ Top-2 Liminal Centroid Routing)**: Routes `src_trail` to
    ///    the top-2 macro centroids `(idx1, idx2, aid1, aid2)` in `macro_centroids` via
    ///    `CrystallizedAttractorRecord::route_top2_liminal_centroids_with_words_64`, interpolating
    ///    onto the Limina geodesic midpoint when `(grav1 - grav2) <= bifurcation_margin * grav1`.
    /// 2. **Fine Step ($\mathcal{O}(M)$ + Pitchfork Tunneling)**: Evaluates stateful EDBC entropy,
    ///    injects Pitchfork metastable tunneling perturbation when in Bifurcation mode, and selects/rewires
    ///    the optimal candidate trail via `step_edbc_tunneling_and_select_bayesian_core`.
    ///
    /// Returns `(edbc_state, evolved_ptr, idx1, idx2, aid1, aid2, is_liminal, selected_trail_id, bayesian_score, transition_log_prob, tunneling_x_star, found_coherent)`.
    ///
    /// # Errors
    /// Propagates any validation or singularity error from [`Self::route_coarse_to_fine_and_step_bayesian_tunneling`].
    #[inline(always)]
    fn route_top2_liminal_and_step_bayesian_tunneling(
        &mut self,
        macro_centroids: &[CrystallizedAttractorRecord],
        coarse_pull_u7: u8,
        bifurcation_margin: f32,
        src_trail: &mut gcso_pointer_trail_t,
        candidate_trails: &mut [gcso_pointer_trail_t],
        token_z_score: f32,
        cvoid_score: f32,
        alpha: f32,
        beta: f32,
        d0: f32,
        head_dim: u32,
        explore_head_start: usize,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        repulsion_penalty: f32,
        fine_interpolation_u7: u8,
        contradiction_threshold: f32,
        high_density_srl_thresh: f32,
        bloom_filter: &mut ContrastivePhaseBloomFilter,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(
        gcso_edbc_state_t,
        gcso_tagged_ptr_t,
        usize,
        usize,
        u32,
        u32,
        bool,
        u32,
        f32,
        f32,
        f32,
        bool,
    )> {
        if candidate_trails.is_empty()
            || explore_head_start > MAX_ATTENTION_HEADS
            || !branchless_is_finite_f32(bifurcation_margin)
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(repulsion_penalty)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !branchless_is_finite_f32(cvoid_score)
            || !branchless_is_finite_f32(alpha)
            || !branchless_is_finite_f32(beta)
            || !branchless_is_finite_f32(d0)
            || beta <= 0.0
            || d0 <= 0.0
            || head_dim == 0
            || (head_dim & 1) != 0
        {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        let (idx1, idx2, aid1, aid2, _eff_coh, _tot_grav, is_liminal, bridged_w, w1, _w2) =
            CrystallizedAttractorRecord::route_top2_liminal_centroids_with_words_64(
                macro_centroids,
                src_trail,
                coarse_pull_u7,
                bifurcation_margin,
            );
        let has_win_centroid =
            idx1 < macro_centroids.len() && macro_centroids[idx1].sample_count > 0;

        let (
            st,
            evolved_ptr,
            selected_id,
            bayes_score,
            transition_log_prob,
            tunneling_x_star,
            found_coherent,
        ) = step_edbc_tunneling_and_select_bayesian_core(
            self,
            src_trail,
            Some(bridged_w),
            candidate_trails,
            token_z_score,
            cvoid_score,
            alpha,
            beta,
            d0,
            head_dim,
            explore_head_start,
            phase_interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
            repulsion_penalty,
            fine_interpolation_u7,
            contradiction_threshold,
            high_density_srl_thresh,
            aid1,
            branchless_select_u32(is_liminal, aid2, 0),
            has_win_centroid.then_some(&w1),
            bloom_filter,
            attention_scores,
        )?;

        Ok((
            st,
            evolved_ptr,
            idx1,
            idx2,
            aid1,
            aid2,
            is_liminal,
            selected_id,
            bayes_score,
            transition_log_prob,
            tunneling_x_star,
            found_coherent,
        ))
    }

    /// Modulates pre-Softmax candidate memory Attention scores (`attention_scores[..M]`) in-place
    /// in zero-copy, zero-allocation ($C_{\mathrm{alloc}} = 0$) constant time per candidate by
    /// combining:
    /// 1. Loop-hoisted candidate trail **Bayes-Attention Stigmergic Transition Score**
    ///    $\Delta z_{t \to j}$ and $H^1 \neq 0$ repulsion penalty via
    ///    [`StigmergicTrailOps::modulate_candidate_trail_scores_in_place`], and
    /// 2. EDBC's energy-guided dynamic sampling temperature $T_{\mathrm{sample}}$
    ///    fused with multiplicative **Phase-Interference Inverse-Temperature Scaling**
    ///    $\beta_{\mathrm{eff}} = \mathrm{clamp}(1.0 + \gamma \bar{\mathcal{C}}_{\mathrm{phase}}, 0.25, 4.0) / T_{\mathrm{sample}}$.
    ///
    /// Returns `(mean_coherence, effective_inv_temp, best_candidate_id, coherent_candidate_count)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `attention_scores` or `candidate_trails` is
    /// empty, `head_dim == 0`, `(head_dim & 1) != 0`, or any parameter is non-finite.
    #[inline(always)]
    fn modulate_candidate_trails_with_edbc_temperature(
        &self,
        attention_scores: &mut AlignedSliceMut32<'_, f32>,
        src_trail: &gcso_pointer_trail_t,
        candidate_trails: &[gcso_pointer_trail_t],
        head_dim: u32,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        contradiction_threshold: f32,
        repulsion_penalty: f32,
        base_temperature: f32,
        moving_z_entropy: f32,
        cvoid_score: f32,
        tau_eff: f32,
        alpha: f32,
        beta: f32,
        d0: f32,
    ) -> GcsoResult<(f32, f32, u32, u32)> {
        let count = attention_scores.len().min(candidate_trails.len());
        if count == 0
            || head_dim == 0
            || (head_dim & 1) != 0
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(contradiction_threshold)
            || !branchless_is_finite_f32(repulsion_penalty)
            || !branchless_is_finite_f32(base_temperature)
            || !branchless_is_finite_f32(moving_z_entropy)
            || !branchless_is_finite_f32(cvoid_score)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let edbc_temp = eval_temperature_for_mode(
            self.active_branch_mode(),
            base_temperature,
            moving_z_entropy,
            cvoid_score,
            tau_eff,
            alpha,
            beta,
            d0,
        );
        let inv_edbc_temp = fast_inv_pos_normal_f32(branchless_max_f32(edbc_temp, 0.05));
        let scores = &mut attention_scores.as_mut_slice()[..count];
        let (mean_coh, best_id, coherent_count) = src_trail
            .modulate_candidate_trail_scores_in_place(
                scores,
                &candidate_trails[..count],
                head_dim,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                contradiction_threshold,
                repulsion_penalty,
            );
        let phase_inv_temp =
            branchless_phase_coherence_inv_temp_scale(mean_coh, phase_interference_gain);
        let eff_inv_temp = branchless_clamp_f32(phase_inv_temp * inv_edbc_temp, 0.05, 8.0);
        scale_f32_slice_unrolled8(scores, eff_inv_temp);

        Ok((mean_coh, eff_inv_temp, best_id, coherent_count))
    }

    /// Evaluates 256-bit dyadic tree distance $d_{\mathcal{V}}(x, y) \in [0, 1)$, quantizes it
    /// with hysteresis into a discrete QDPT depth level, cascades `trail.current_ptr` via
    /// [`HierarchicalTreeTable::resolve_dyadic_qdpt_and_cascade_ptr`], steps the stateful EDBC
    /// controller, and evolves the cascaded pointer mode while preserving 32-bit `trail.target_anchor_id`.
    ///
    /// Returns `(edbc_state, qdpt_level, dyadic_distance, evolved_ptr)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `qdpt_addr_offsets` is empty or mismatched with
    /// `qdpt_phase_deltas`. Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if `token_z_score` or
    /// `cvoid_score` is non-finite.
    #[inline(always)]
    fn resolve_dyadic_tree_and_step_edbc(
        &mut self,
        current_mask: &gcso_paged_bitmask_t,
        anchor_mask: &gcso_paged_bitmask_t,
        w_key: f32,
        current_level: u32,
        qdpt_hysteresis_band: f32,
        qdpt_addr_offsets: &[u64],
        qdpt_phase_deltas: &[gcso_q7_t],
        token_z_score: f32,
        cvoid_score: f32,
        high_density_srl_thresh: f32,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(gcso_edbc_state_t, u32, f32, gcso_tagged_ptr_t)> {
        if !branchless_is_finite_f32(cvoid_score) {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }
        let (qdpt_lvl, d_v, cascaded_ptr) = self.resolve_dyadic_qdpt_and_cascade_ptr(
            gcso_tagged_ptr_t(trail.current_ptr),
            current_mask,
            anchor_mask,
            w_key,
            current_level,
            qdpt_hysteresis_band,
            qdpt_addr_offsets,
            qdpt_phase_deltas,
        )?;

        let mut st = gcso_edbc_state_t::default();
        self.eval_stateful(token_z_score, &mut st)?;
        let bif_thresh = branchless_select_f32(
            (st.bifurcation_threshold > 0.0) & branchless_is_finite_f32(st.bifurcation_threshold),
            st.bifurcation_threshold,
            DEFAULT_BIFURCATION_THRESHOLD,
        );
        st.bifurcation_threshold = bif_thresh;

        let dom_lane = branchless_select_u8(
            trail.target_anchor_id != 0,
            (trail.target_anchor_id as u8) & ATTRACTOR_SLOT_ID_MASK,
            cascaded_ptr.target_attractor_slot(),
        );
        let is_repulsion = st.active_branch_mode == EDBC_MODE_PHASE_CONJUGATE_REPULSION;
        let low_entropy_skip_thresh = branchless_relu_f32(bif_thresh * 0.5);
        let evolved_ptr = cascaded_ptr.evolve_mode_stigmergically(
            trail.stigmergic_density,
            st.moving_z_entropy,
            is_repulsion,
            dom_lane,
            low_entropy_skip_thresh,
            high_density_srl_thresh,
        );
        trail.prev_ptr = trail.current_ptr;
        trail.current_ptr = evolved_ptr.raw();
        let preserve_anchor = (trail.target_anchor_id != 0) & (!is_repulsion);
        trail.cluster_id = branchless_select_u32(
            preserve_anchor,
            trail.target_anchor_id,
            evolved_ptr.target_attractor_slot() as u32,
        );

        Ok((st, qdpt_lvl, d_v, evolved_ptr))
    }

    /// Executes a closed-loop EDBC step driven by a synthesized 128-byte [`GatewayFrame`]
    /// (**2-Origin Composite Manifold Gateway**):
    /// 1. Steps [`EntropyEvaluator::eval_stateful`] (`token_z_score`) and evaluates EDBC's
    ///    Pitchfork/Repulsion branch mode against `cvoid_score` and `frame.dominant_cause_code`.
    /// 2. Ingests `frame.phase_deltas` into `trail` in 8 register-resident `u64` SWAR words
    ///    (`next_w`), applying `frame.qdps_min_step_q7` step filtering and binding 32-bit
    ///    `frame.target_anchor_id` into `trail.target_anchor_id` and `trail.cluster_id`.
    /// 3. Evolves `trail.current_ptr` preserving `frame.zimms_section_offset48` (`48-bit` `.gcso`
    ///    direct RAG/Fact address) and reconciling `ACTION_MODE_LAYER_SKIP` with EDBC's
    ///    entropy state.
    /// 4. Combines `frame.effective_inv_temp` with EDBC's mode-aware temperature $T_{\mathrm{sample}}$
    ///    into $\beta_{\mathrm{eff}} = \mathrm{clamp}(\beta_{\mathrm{gateway}} / T_{\mathrm{sample}}, 0.05, 8.0)$
    ///    and optionally scales `logits_opt` in-place via AVX2 (`scale_f32_slice_unrolled8`).
    /// 5. Produces the unified $\mathcal{O}(1)$ [`CausalProvenanceReport`] (16B `Pod`) and, when
    ///    `crystal_record_opt` is supplied and the trajectory is coherent (`!repulsion_triggered`),
    ///    absorbs the register-resident `next_w` directly via
    ///    [`CrystallizedAttractorRecord::absorb_gateway_frame_with_words`] without reloading
    ///    `trail.accumulated_phase_delta` from memory!
    ///
    /// Returns `(edbc_state, evolved_ptr, causal_report, effective_inv_temp, crystallized_gravity_mass)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `high_density_srl_thresh` is non-finite.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if `token_z_score`, `cvoid_score`, `alpha`,
    /// `beta <= 0.0`, or `d0 <= 0.0` is invalid.
    #[inline(always)]
    fn step_via_gateway_frame(
        &mut self,
        token_z_score: f32,
        cvoid_score: f32,
        alpha: f32,
        beta: f32,
        d0: f32,
        high_density_srl_thresh: f32,
        frame: &GatewayFrame,
        trail: &mut gcso_pointer_trail_t,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
        crystal_record_opt: Option<&mut CrystallizedAttractorRecord>,
    ) -> GcsoResult<(
        gcso_edbc_state_t,
        gcso_tagged_ptr_t,
        CausalProvenanceReport,
        f32,
        f32,
    )> {
        step_via_gateway_frame_with_words_core(
            self,
            token_z_score,
            cvoid_score,
            alpha,
            beta,
            d0,
            high_density_srl_thresh,
            frame,
            frame.phase_words_ref(),
            trail,
            logits_opt,
            crystal_record_opt,
        )
    }

    /// Synthesizes a [`GatewayFrame`] and its register-resident 8-word phase profile (`gw_w`)
    /// from any composite [`ManifoldGatewaySignal`] (2/3/4-tuples, slice ensembles, `Option<T>`,
    /// or [`CrystallizedAttractorRecord`]) via [`ManifoldGatewaySignal::synthesize_gateway_frame_with_words`]
    /// and executes [`Self::step_via_gateway_frame`] without reloading `frame.phase_deltas` from memory.
    ///
    /// Returns `(edbc_state, evolved_ptr, causal_report, effective_inv_temp, crystallized_gravity_mass)`.
    ///
    /// # Errors
    /// Propagates any validation or singularity error from [`Self::step_via_gateway_frame`].
    #[inline(always)]
    fn step_via_gateway_signal<G: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        token_z_score: f32,
        cvoid_score: f32,
        alpha: f32,
        beta: f32,
        d0: f32,
        high_density_srl_thresh: f32,
        gateway_signal: &G,
        trail: &mut gcso_pointer_trail_t,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
        crystal_record_opt: Option<&mut CrystallizedAttractorRecord>,
    ) -> GcsoResult<(
        gcso_edbc_state_t,
        gcso_tagged_ptr_t,
        CausalProvenanceReport,
        f32,
        f32,
    )> {
        let (frame, gw_w) = gateway_signal.synthesize_gateway_frame_with_words();
        step_via_gateway_frame_with_words_core(
            self,
            token_z_score,
            cvoid_score,
            alpha,
            beta,
            d0,
            high_density_srl_thresh,
            &frame,
            &gw_w,
            trail,
            logits_opt,
            crystal_record_opt,
        )
    }

    /// Executes the closed-loop **2-Origin Composite Manifold Gateway $\to$ EDBC Step $\to$
    /// Classical Online Toroidal $k$-Means / DP-Means Macro Clustering** pipeline on
    /// `centroids: &mut [CrystallizedAttractorRecord]` in a single zero-allocation call,
    /// keeping `gw_w` in registers across both EDBC and Toroidal $k$-Means.
    ///
    /// Returns `(edbc_state, evolved_ptr, causal_report, effective_inv_temp, assigned_cluster_idx, assigned_anchor_id, updated_gravity_mass, spawned_new_cluster)`.
    ///
    /// # Errors
    /// Propagates any validation or singularity error from [`Self::step_via_gateway_frame`].
    #[inline(always)]
    fn step_via_gateway_kmeans_64<G: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        token_z_score: f32,
        cvoid_score: f32,
        alpha: f32,
        beta: f32,
        d0: f32,
        high_density_srl_thresh: f32,
        gateway_signal: &G,
        centroids: &mut [CrystallizedAttractorRecord],
        split_coherence_thresh: f32,
        trail: &mut gcso_pointer_trail_t,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(
        gcso_edbc_state_t,
        gcso_tagged_ptr_t,
        CausalProvenanceReport,
        f32,
        usize,
        u32,
        f32,
        bool,
    )> {
        let (frame, gw_w) = gateway_signal.synthesize_gateway_frame_with_words();
        let (st, evolved_ptr, mut report, eff_inv_temp, _single_mass) =
            step_via_gateway_frame_with_words_core(
                self,
                token_z_score,
                cvoid_score,
                alpha,
                beta,
                d0,
                high_density_srl_thresh,
                &frame,
                &gw_w,
                trail,
                logits_opt,
                None,
            )?;

        let (cluster_idx, anchor_id, _coh, mass, spawned) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_frame_with_words_64(
                centroids,
                &frame,
                &gw_w,
                split_coherence_thresh,
                Some(trail),
            );
        if !centroids.is_empty()
            && (centroids[cluster_idx].provenance_bits & CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED) != 0
        {
            report.provenance_bits |= CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED;
        }

        Ok((
            st,
            evolved_ptr,
            report,
            eff_inv_temp,
            cluster_idx,
            anchor_id,
            mass,
            spawned,
        ))
    }

    /// Folds a slice of `.gcso`-persisted [`CrystallizedAttractorRecord`]s (`records`) onto `trail`
    /// via **Phase-Proximity Gravity** ([`CrystallizedAttractorRecord::fold_crystallized_ensemble_words_with_init_pull_64`]),
    /// blending directly in 8 register-resident SWAR words when prior pull inertia is present,
    /// binds the dominant 32-bit `anchor_id` into `trail.target_anchor_id` and `trail.cluster_id`,
    /// steps the stateful EDBC controller, and evolves `trail.current_ptr`.
    ///
    /// Returns `(edbc_state, evolved_ptr, total_gravity_pull, mean_coherence, dominant_anchor_id)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `high_density_srl_thresh` is non-finite.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if `token_z_score` or `cvoid_score` is non-finite.
    #[inline(always)]
    fn step_with_crystallized_ensemble_and_pointer(
        &mut self,
        token_z_score: f32,
        cvoid_score: f32,
        records: &[CrystallizedAttractorRecord],
        high_density_srl_thresh: f32,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(gcso_edbc_state_t, gcso_tagged_ptr_t, f32, f32, u32)> {
        if !branchless_is_finite_f32(high_density_srl_thresh) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !branchless_is_finite_f32(cvoid_score) {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        let init_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(trail.attractor_pull_force));
        let (_after_w, raw_total_pull, dom_id, _dom_ripa_rad, mean_coh) =
            CrystallizedAttractorRecord::fold_crystallized_ensemble_words_and_coherence_64(
                records,
                &mut trail.accumulated_phase_delta,
                init_pull,
            );
        let total_pull = branchless_min_f32(raw_total_pull, MAX_ATTRACTOR_PULL_FORCE);
        if raw_total_pull > init_pull + 1.0e-6 {
            if dom_id != 0 {
                trail.target_anchor_id = dom_id;
                trail.cluster_id = dom_id;
            }
            trail.attractor_pull_force = total_pull;
            trail.flags |= TRAIL_FLAG_CRYSTALLIZED_CANDIDATE;
        }

        let mut st = gcso_edbc_state_t::default();
        self.eval_stateful(token_z_score, &mut st)?;
        let bif_thresh = branchless_select_f32(
            (st.bifurcation_threshold > 0.0) & branchless_is_finite_f32(st.bifurcation_threshold),
            st.bifurcation_threshold,
            DEFAULT_BIFURCATION_THRESHOLD,
        );
        st.bifurcation_threshold = bif_thresh;

        let dom_lane = (trail.target_anchor_id as u8) & ATTRACTOR_SLOT_ID_MASK;
        let is_repulsion = st.active_branch_mode == EDBC_MODE_PHASE_CONJUGATE_REPULSION;
        let low_entropy_skip_thresh = branchless_relu_f32(bif_thresh * 0.5);
        let evolved_ptr = gcso_tagged_ptr_t(trail.current_ptr).evolve_mode_stigmergically(
            trail.stigmergic_density,
            st.moving_z_entropy,
            is_repulsion,
            dom_lane,
            low_entropy_skip_thresh,
            high_density_srl_thresh,
        );
        trail.current_ptr = evolved_ptr.raw();
        let preserve_anchor = (trail.target_anchor_id != 0) & (!is_repulsion);
        trail.cluster_id = branchless_select_u32(
            preserve_anchor,
            trail.target_anchor_id,
            evolved_ptr.target_attractor_slot() as u32,
        );

        Ok((
            st,
            evolved_ptr,
            total_pull,
            mean_coh,
            trail.target_anchor_id,
        ))
    }
}

impl<T> EdbcClosedLoopCoordinator for T where
    T: EdbcBranchController + SheafCohomologyEvaluator + HierarchicalTreeTable + ?Sized
{
}

// ===================================================================
// 4. Stateful Entropy-Driven Decoding Branch Controller (`128B` Pod)
// ===================================================================

/// Entropy-Driven Decoding Branch Controller (EDBC) stateful engine (128 bytes = 2 cachelines, 32B aligned).
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdbcEngine {
    /// Active C-ABI EDBC state descriptor (64 bytes, 32-byte aligned).
    pub state: gcso_edbc_state_t,
    /// Hysteresis deadband width ($\delta_{\mathrm{hyst}}$) preventing mode chattering near thresholds.
    pub hysteresis_band: f32,
    /// Threshold for sliding-window entropy rate flux ($\tau_{\mathrm{flux}}$).
    pub tau_flux: f32,
    /// Fixed-size 8-slot ring buffer for Sliding-Window Entropy Rate Integrator ($\Phi_M(t)$, $M=5$).
    window_buf: [f32; EDBC_WINDOW_RING_CAPACITY],
    /// Total stateful evaluation steps processed.
    pub step_count: u64,
    /// Consecutive token steps spent in the currently active branch mode.
    pub mode_dwell_steps: u32,
    /// Minimum token dwell steps required before relaxing to a lower branch mode (`0` = immediate).
    pub min_mode_dwell_steps: u32,
    /// Ring buffer write index (`0..=255`, masked with `& 7`).
    window_idx: u8,
    /// Number of valid elements currently populated in `window_buf` (`0..=8`).
    window_len: u8,
    /// Controller active lifecycle flag (`1` = active, `0` = passive).
    pub is_active: bool,
    /// Explicit alignment padding to guarantee exact 128-byte (2-cacheline) POD layout.
    reserved_pad: [u8; 5],
}

// SAFETY: `EdbcEngine` is `#[repr(C, align(32))]` of size 128 bytes where every field is `Pod`.
unsafe impl Pod for EdbcEngine {}

const _: () = {
    assert!(size_of::<EdbcEngine>() == 128);
    assert!(align_of::<EdbcEngine>() == 32);
    assert!(offset_of!(EdbcEngine, state) == 0);
    assert!(offset_of!(EdbcEngine, hysteresis_band) == 64);
    assert!(offset_of!(EdbcEngine, tau_flux) == 68);
    assert!(offset_of!(EdbcEngine, window_buf) == 72);
    assert!(offset_of!(EdbcEngine, step_count) == 104);
    assert!(offset_of!(EdbcEngine, mode_dwell_steps) == 112);
    assert!(offset_of!(EdbcEngine, min_mode_dwell_steps) == 116);
    assert!(offset_of!(EdbcEngine, window_idx) == 120);
    assert!(offset_of!(EdbcEngine, window_len) == 121);
    assert!(offset_of!(EdbcEngine, is_active) == 122);
};

impl EdbcEngine {
    /// Constructs a new [`EdbcEngine`] with sanitized initial state configuration in branchless mode.
    #[inline(always)]
    pub const fn new(mut initial_state: gcso_edbc_state_t) -> Self {
        let valid_bif = branchless_is_finite_f32(initial_state.bifurcation_threshold)
            & (initial_state.bifurcation_threshold > 0.0);
        initial_state.bifurcation_threshold = branchless_select_f32(
            valid_bif,
            initial_state.bifurcation_threshold,
            DEFAULT_BIFURCATION_THRESHOLD,
        );

        let valid_eps = branchless_is_finite_f32(initial_state.singularity_eps)
            & (initial_state.singularity_eps > 0.0);
        initial_state.singularity_eps = branchless_select_f32(
            valid_eps,
            initial_state.singularity_eps,
            DEFAULT_SINGULARITY_EPS,
        );

        let valid_temp = branchless_is_finite_f32(initial_state.sample_temperature)
            & (initial_state.sample_temperature > 0.0);
        initial_state.sample_temperature = branchless_select_f32(
            valid_temp,
            initial_state.sample_temperature,
            DEFAULT_SAMPLE_TEMPERATURE,
        );

        Self {
            state: initial_state,
            hysteresis_band: DEFAULT_HYSTERESIS_BAND,
            tau_flux: DEFAULT_TAU_FLUX,
            window_buf: [0.0; EDBC_WINDOW_RING_CAPACITY],
            step_count: 0,
            mode_dwell_steps: 0,
            min_mode_dwell_steps: 0,
            window_idx: 0,
            window_len: 0,
            is_active: true,
            reserved_pad: [0; 5],
        }
    }

    /// Builder method setting the minimum token dwell steps (`min_mode_dwell_steps`) required
    /// before relaxing from Bifurcation/Repulsion modes back to lower modes.
    #[inline(always)]
    pub const fn with_min_mode_dwell_steps(mut self, min_steps: u32) -> Self {
        self.min_mode_dwell_steps = min_steps;
        self
    }

    /// Returns the current active C-ABI state descriptor snapshot.
    #[inline(always)]
    pub const fn state(&self) -> gcso_edbc_state_t {
        self.state
    }

    /// Returns `true` if the controller instance is currently active.
    #[inline(always)]
    pub const fn is_active(&self) -> bool {
        self.is_active
    }

    /// Sets the active lifecycle status flag.
    #[inline(always)]
    pub fn set_active(&mut self, active: bool) {
        self.is_active = active;
    }

    /// Returns the active decoding branch mode (`0` = Normal, `1` = Bifurcation, `2` = Repulsion).
    #[inline(always)]
    pub const fn active_branch_mode(&self) -> u32 {
        self.state.active_branch_mode
    }

    /// Returns `true` if operating in Normal Entropy Convergence Mode (`Mode 0`).
    #[inline(always)]
    pub const fn is_normal_mode(&self) -> bool {
        self.state.active_branch_mode == EDBC_MODE_NORMAL_ENTROPY
    }

    /// Returns `true` if operating in Pitchfork Bifurcation Exploration Mode (`Mode 1`).
    #[inline(always)]
    pub const fn is_bifurcation_mode(&self) -> bool {
        self.state.active_branch_mode == EDBC_MODE_PITCHFORK_BIFURCATION
    }

    /// Returns `true` if operating in Phase-Conjugate Repulsion Mode (`Mode 2`).
    #[inline(always)]
    pub const fn is_repulsion_mode(&self) -> bool {
        self.state.active_branch_mode == EDBC_MODE_PHASE_CONJUGATE_REPULSION
    }

    /// Returns the current Moving Z-Score Normalized Attention Entropy ($\tilde{H}$).
    #[inline(always)]
    pub const fn moving_z_entropy(&self) -> f32 {
        self.state.moving_z_entropy
    }

    /// Returns the current Sliding-Window Entropy Rate ($\Phi_M(t)$).
    #[inline(always)]
    pub const fn sliding_entropy_rate(&self) -> f32 {
        self.state.sliding_entropy_rate
    }

    /// Returns the active sampling temperature modifier (`sample_temperature`).
    #[inline(always)]
    pub const fn sample_temperature(&self) -> f32 {
        self.state.sample_temperature
    }

    /// Returns the active phase repulsion gain factor (`repulsion_gain`).
    #[inline(always)]
    pub const fn repulsion_gain(&self) -> f32 {
        self.state.repulsion_gain
    }

    /// Evaluates both the 1st-order sliding-window entropy flux $\Phi_M(t)$ and 2nd-order
    /// discrete entropy acceleration $\Delta^2 \tilde{H}(t)$ from the internal ring buffer in $\mathcal{O}(1)$.
    #[inline(always)]
    pub const fn eval_entropy_flux_acceleration(&self) -> (f32, f32) {
        let len = self.window_len;
        let has_2 = len >= 2;
        let has_3 = len >= 3;
        let mask = (EDBC_WINDOW_RING_CAPACITY - 1) as u8;
        let i0 = (self.window_idx.wrapping_sub(1) & mask) as usize;
        let i1 = (self.window_idx.wrapping_sub(2) & mask) as usize;
        let i2 = (self.window_idx.wrapping_sub(3) & mask) as usize;

        let h0 = branchless_select_f32(len >= 1, self.window_buf[i0], self.state.moving_z_entropy);
        let h1 = branchless_select_f32(has_2, self.window_buf[i1], h0);
        let h2 = branchless_select_f32(has_3, self.window_buf[i2], h1);

        let accel = branchless_select_f32(has_3, (h0 - h1) - (h1 - h2), 0.0);
        (
            self.state.sliding_entropy_rate,
            branchless_flush_subnormal_f32(accel),
        )
    }

    /// Evaluates PPRC Variable GOP keyframe scheduling and Game-Coexistence QoS compute budgeting
    /// in $\mathcal{O}(1)$ division-free (`0 fdiv`) branchless arithmetic from the current EDBC state.
    #[inline(always)]
    pub const fn evaluate_adaptive_compute_and_gop_budget(
        &self,
        steps_since_keyframe: u32,
        min_gop: u32,
        max_gop: u32,
        low_entropy_skip_thresh: f32,
    ) -> (bool, u32, bool, f32) {
        let safe_min = branchless_select_u32(min_gop > 0, min_gop, 1);
        let safe_max = branchless_select_u32(max_gop >= safe_min, max_gop, safe_min);
        let h_cur = branchless_relu_f32(self.state.moving_z_entropy);
        let flux_cur = self.state.sliding_entropy_rate;
        let tau = self.state.bifurcation_threshold;

        let surge = (self.state.active_branch_mode != EDBC_MODE_NORMAL_ENTROPY)
            | (h_cur > tau)
            | (flux_cur > self.tau_flux);
        let emit_iframe =
            (steps_since_keyframe >= safe_max) | ((steps_since_keyframe >= safe_min) & surge);
        let gop_len = branchless_select_u32(
            emit_iframe,
            branchless_select_u32(steps_since_keyframe > 0, steps_since_keyframe, 1),
            safe_max,
        );

        let can_skip = (self.state.active_branch_mode == EDBC_MODE_NORMAL_ENTROPY)
            & (h_cur <= low_entropy_skip_thresh)
            & (flux_cur <= self.tau_flux);
        let gate_scale = branchless_clamp_f32(h_cur * INV_LN_2, 0.0, 1.0);
        (emit_iframe, gop_len, can_skip, gate_scale)
    }

    /// Sets the pitchfork bifurcation threshold ($\tau_{\mathrm{bifurcation}} > 0$) branchlessly.
    #[inline(always)]
    pub fn set_bifurcation_threshold(&mut self, threshold: f32) {
        let valid = branchless_is_finite_f32(threshold) & (threshold > 0.0);
        self.state.bifurcation_threshold =
            branchless_select_f32(valid, threshold, self.state.bifurcation_threshold);
    }

    /// Sets the anti-chattering hysteresis deadband width ($\delta_{\mathrm{hyst}} \ge 0$) branchlessly.
    #[inline(always)]
    pub fn set_hysteresis_band(&mut self, band: f32) {
        let valid = branchless_is_finite_f32(band) & (band >= 0.0);
        self.hysteresis_band = branchless_select_f32(valid, band, self.hysteresis_band);
    }

    /// Sets the sliding-window entropy rate flux threshold ($\tau_{\mathrm{flux}} \ge 0$) branchlessly.
    #[inline(always)]
    pub fn set_tau_flux(&mut self, tau_flux: f32) {
        let valid = branchless_is_finite_f32(tau_flux) & (tau_flux >= 0.0);
        self.tau_flux = branchless_select_f32(valid, tau_flux, self.tau_flux);
    }

    /// Sets the minimum token dwell steps (`min_mode_dwell_steps`) before downward mode relaxation.
    #[inline(always)]
    pub fn set_min_mode_dwell_steps(&mut self, min_steps: u32) {
        self.min_mode_dwell_steps = min_steps;
    }

    /// Sets the logarithmic singularity epsilon guard ($\epsilon_{\mathrm{log}} > 0$) branchlessly.
    #[inline(always)]
    pub fn set_singularity_eps(&mut self, eps: f32) {
        let valid = branchless_is_finite_f32(eps) & (eps > 0.0);
        self.state.singularity_eps = branchless_select_f32(valid, eps, self.state.singularity_eps);
    }

    /// Sets the sampling temperature modifier (`sample_temperature > 0`) branchlessly.
    #[inline(always)]
    pub fn set_sample_temperature(&mut self, temp: f32) {
        let valid = branchless_is_finite_f32(temp) & (temp > 0.0);
        self.state.sample_temperature =
            branchless_select_f32(valid, temp, self.state.sample_temperature);
    }

    /// Resets transient entropy tracking accumulators, ring buffer, and branch modes in-place
    /// using width-matched scalar stores (`f32`/`u64`/`u32`/`u8`) to prevent unaligned 256-bit
    /// `vmovups` Store-to-Load Forwarding (STLF) pipeline stalls on the first post-reset step.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.state.moving_z_entropy = 0.0;
        self.state.sliding_entropy_rate = 0.0;
        self.state.repulsion_gain = 0.0;
        self.state.sample_temperature = DEFAULT_SAMPLE_TEMPERATURE;
        self.state.active_branch_mode = EDBC_MODE_NORMAL_ENTROPY;
        let mut i = 0usize;
        while i < EDBC_WINDOW_RING_CAPACITY {
            if self.window_buf[i].to_bits() != 0 {
                self.window_buf[i] = 0.0;
            }
            i += 1;
        }
        self.step_count = 0;
        self.mode_dwell_steps = 0;
        self.window_idx = 0;
        self.window_len = 0;
    }

    /// Restores the mutable runtime accumulators of `self` from `src` using width-matched scalar
    /// stores (`f32`, `u32`, `u64`, `u8`), avoiding the 256-bit `vmovaps` YMM store-to-scalar-load
    /// Store-to-Load Forwarding (STLF) pipeline stall caused by 128-byte `*self = *src` assignment
    /// while guaranteeing full 8-slot ring restoration across multi-step rollbacks.
    #[inline(always)]
    pub fn restore_from(&mut self, src: &Self) {
        self.state.moving_z_entropy = src.state.moving_z_entropy;
        self.state.bifurcation_threshold = src.state.bifurcation_threshold;
        self.state.singularity_eps = src.state.singularity_eps;
        self.state.sliding_entropy_rate = src.state.sliding_entropy_rate;
        self.state.repulsion_gain = src.state.repulsion_gain;
        self.state.sample_temperature = src.state.sample_temperature;
        self.state.active_branch_mode = src.state.active_branch_mode;
        self.hysteresis_band = src.hysteresis_band;
        self.tau_flux = src.tau_flux;
        let mut i = 0usize;
        while i < EDBC_WINDOW_RING_CAPACITY {
            let src_val = src.window_buf[i];
            if self.window_buf[i].to_bits() != src_val.to_bits() {
                self.window_buf[i] = src_val;
            }
            i += 1;
        }
        self.step_count = src.step_count;
        self.mode_dwell_steps = src.mode_dwell_steps;
        self.min_mode_dwell_steps = src.min_mode_dwell_steps;
        self.window_idx = src.window_idx;
        self.window_len = src.window_len;
        self.is_active = src.is_active;
    }

    /// Internal register-pipelined helper updating the EMA entropy and 8-slot sliding-window
    /// ring buffer from an already bounded/sanitized `token_z_safe` in `[0.0, MAX_VALID_TOKEN_Z_SCORE]`.
    #[inline(always)]
    fn advance_entropy_ring_presanitized(&mut self, token_z_safe: f32) -> (f32, f32, bool) {
        let prev_entropy = self.state.moving_z_entropy;
        let new_entropy = branchless_flush_subnormal_f32(
            (1.0 - EMA_ALPHA_ENTROPY) * prev_entropy + EMA_ALPHA_ENTROPY * token_z_safe,
        );

        let w_len = self.window_len;
        let w_idx = self.window_idx;
        let horizon_u8 = EDBC_SLIDING_WINDOW_HORIZON as u8;
        let cap_u8 = EDBC_WINDOW_RING_CAPACITY as u8;
        let mask_u8 = cap_u8 - 1;

        // Fast-path steady-state full ring buffer (`w_len == 8`, true for 999,992 of 1,000,000 tokens):
        // skips `was_empty` / `window_m` cmovs and avoids writing `self.window_len` on every token.
        if w_len == cap_u8 {
            let old_idx = (w_idx.wrapping_sub(horizon_u8) & mask_u8) as usize;
            let old_entropy = self.window_buf[old_idx];
            self.window_buf[(w_idx & mask_u8) as usize] = new_entropy;
            self.window_idx = w_idx.wrapping_add(1);
            let sliding_rate = branchless_flush_subnormal_f32(new_entropy - old_entropy);
            return (new_entropy, sliding_rate, false);
        }

        let was_empty = w_len == 0;
        let window_m = branchless_select_u8(w_len < horizon_u8, w_len, horizon_u8);
        let old_idx = (w_idx.wrapping_sub(window_m) & mask_u8) as usize;
        let old_entropy = branchless_select_f32(was_empty, 0.0_f32, self.window_buf[old_idx]);

        let write_idx = (w_idx & mask_u8) as usize;
        self.window_buf[write_idx] = new_entropy;
        self.window_idx = w_idx.wrapping_add(1);
        self.window_len = w_len.wrapping_add(1);

        let sliding_rate = branchless_flush_subnormal_f32(new_entropy - old_entropy);
        (new_entropy, sliding_rate, was_empty)
    }

    /// Internal register-pipelined helper updating the EMA entropy and 8-slot sliding-window
    /// ring buffer in a single pass using pure `u8` register arithmetic.
    #[inline(always)]
    fn advance_entropy_ring_in_registers(&mut self, token_z_score: f32) -> (f32, f32, bool) {
        let token_z_safe = branchless_sanitize_z_score_f32(token_z_score, MAX_VALID_TOKEN_Z_SCORE);
        self.advance_entropy_ring_presanitized(token_z_safe)
    }

    /// Internal register-pipelined implementation of `eval_stateful` assuming `token_z_score`
    /// has already been verified finite, writing only the 7 active scalar fields (28 bytes) into
    /// `state_out` without copying the 36-byte `reserved` padding array.
    #[inline(always)]
    fn eval_stateful_finite_into(
        &mut self,
        token_z_score: f32,
        state_out: &mut gcso_edbc_state_t,
    ) -> f32 {
        let current_mode = self.state.active_branch_mode;
        let tau = self.state.bifurcation_threshold;
        let eps = self.state.singularity_eps;
        let delta = self.hysteresis_band;
        let next_dwell = self.mode_dwell_steps.saturating_add(1);

        let token_z_safe = branchless_clamp_f32(token_z_score, 0.0, MAX_VALID_TOKEN_Z_SCORE);
        let (new_entropy, sliding_rate, is_first_step) =
            self.advance_entropy_ring_presanitized(token_z_safe);

        let (next_mode, final_dwell, sample_temp, rep_gain) =
            if (current_mode == EDBC_MODE_NORMAL_ENTROPY) & (new_entropy <= (tau + delta)) {
                (
                    EDBC_MODE_NORMAL_ENTROPY,
                    next_dwell,
                    DEFAULT_SAMPLE_TEMPERATURE,
                    0.0_f32,
                )
            } else {
                let tau_flux = self.tau_flux;
                let min_dwell = self.min_mode_dwell_steps;
                let can_relax = next_dwell >= min_dwell;
                let flux_surge = sliding_rate > (tau_flux + delta);
                let can_escalate = is_first_step | can_relax | flux_surge;
                let tau_15 = tau * 1.5;

                let cond_repulse_up = (new_entropy > (tau_15 + delta)) & flux_surge;
                let cond_bif_up =
                    (new_entropy > (tau + delta)) & (sliding_rate >= -delta) & can_escalate;
                let cond_norm_down = (new_entropy < (tau - delta)) & can_relax;
                let cond_repulse_down = ((new_entropy < (tau_15 - delta))
                    | (sliding_rate < (tau_flux - delta)))
                    & can_relax;

                let to_bif = ((current_mode == EDBC_MODE_NORMAL_ENTROPY) & cond_bif_up)
                    | ((current_mode == EDBC_MODE_PHASE_CONJUGATE_REPULSION) & cond_repulse_down);
                let mid_mode =
                    branchless_select_u32(to_bif, EDBC_MODE_PITCHFORK_BIFURCATION, current_mode);
                let nm = branchless_select_u32(
                    cond_repulse_up,
                    EDBC_MODE_PHASE_CONJUGATE_REPULSION,
                    branchless_select_u32(cond_norm_down, EDBC_MODE_NORMAL_ENTROPY, mid_mode),
                );
                let fd = branchless_select_u32(nm == current_mode, next_dwell, 1);
                let mode_idx = (nm as usize).min(2);
                (
                    nm,
                    fd,
                    LUT_SAMPLE_TEMPERATURE[mode_idx],
                    LUT_REPULSION_GAIN[mode_idx],
                )
            };

        self.state.moving_z_entropy = new_entropy;
        self.state.sliding_entropy_rate = sliding_rate;
        self.state.active_branch_mode = next_mode;
        self.state.sample_temperature = sample_temp;
        self.state.repulsion_gain = rep_gain;
        self.mode_dwell_steps = final_dwell;
        self.step_count = self.step_count.wrapping_add(1);

        state_out.moving_z_entropy = new_entropy;
        state_out.bifurcation_threshold = tau;
        state_out.singularity_eps = eps;
        state_out.sliding_entropy_rate = sliding_rate;
        state_out.repulsion_gain = rep_gain;
        state_out.sample_temperature = sample_temp;
        state_out.active_branch_mode = next_mode;
        new_entropy
    }

    /// Internal register-pipelined implementation of `eval_stateful_with_cvoid_and_pointer`
    /// assuming `token_z_score` and `cvoid_score` have already been verified finite.
    #[inline(always)]
    fn eval_stateful_with_cvoid_and_pointer_finite(
        &mut self,
        token_z_score: f32,
        cvoid_score: f32,
        alpha: f32,
        beta: f32,
        d0: f32,
        input_ptr: gcso_tagged_ptr_t,
        trail_density: f32,
        dominant_attractor_slot: u8,
        high_density_srl_thresh: f32,
    ) -> (gcso_edbc_state_t, gcso_tagged_ptr_t) {
        let prev_mode = self.state.active_branch_mode;
        let prev_dwell = self.mode_dwell_steps;
        let min_dwell = self.min_mode_dwell_steps;
        let tau = self.state.bifurcation_threshold;
        let tau_flux = self.tau_flux;
        let delta = self.hysteresis_band;

        let (new_entropy, sliding_rate, _was_empty) =
            self.advance_entropy_ring_in_registers(token_z_score);

        let candidate_mode = self.evaluate_branch_mode_transition(
            prev_mode,
            new_entropy,
            sliding_rate,
            cvoid_score,
            tau,
            tau_flux,
            delta,
        );

        let next_dwell = prev_dwell.saturating_add(1);
        let can_relax = next_dwell >= min_dwell;
        let is_downward_relax = candidate_mode < prev_mode;
        let resolved_mode =
            branchless_select_u32(is_downward_relax & !can_relax, prev_mode, candidate_mode);

        let final_dwell = branchless_select_u32(resolved_mode == prev_mode, next_dwell, 1);
        let mode_idx = (resolved_mode as usize).min(2);
        let rep_gain = LUT_REPULSION_GAIN[mode_idx];

        let is_repulsion = resolved_mode == EDBC_MODE_PHASE_CONJUGATE_REPULSION;
        let low_entropy_skip_thresh = branchless_relu_f32(tau * 0.5);
        let evolved_ptr = input_ptr.evolve_mode_stigmergically(
            trail_density,
            new_entropy,
            is_repulsion,
            dominant_attractor_slot & ATTRACTOR_SLOT_ID_MASK,
            low_entropy_skip_thresh,
            high_density_srl_thresh,
        );

        let sample_temp = eval_temperature_for_mode(
            resolved_mode,
            DEFAULT_SAMPLE_TEMPERATURE,
            new_entropy,
            cvoid_score,
            tau,
            alpha,
            beta,
            d0,
        );

        let eps = self.state.singularity_eps;
        self.state.moving_z_entropy = new_entropy;
        self.state.sliding_entropy_rate = sliding_rate;
        self.state.active_branch_mode = resolved_mode;
        self.state.repulsion_gain = rep_gain;
        self.state.sample_temperature = sample_temp;
        self.mode_dwell_steps = final_dwell;
        self.step_count = self.step_count.wrapping_add(1);

        let out_state = gcso_edbc_state_t {
            moving_z_entropy: new_entropy,
            bifurcation_threshold: tau,
            singularity_eps: eps,
            sliding_entropy_rate: sliding_rate,
            repulsion_gain: rep_gain,
            sample_temperature: sample_temp,
            active_branch_mode: resolved_mode,
            reserved: [0u8; 36],
        };
        (out_state, evolved_ptr)
    }

    /// Executes a single-writeback stateful EDBC step coupled with $\mathcal{C}_{\mathrm{void}}$
    /// dual-branch routing (Section 3.3), dwell-step anti-chattering inertia across all downward
    /// transitions (`candidate_mode < prev_mode`), Eyring-Kramers temperature modulation, and
    /// $\mathcal{O}(1)$ [`gcso_tagged_ptr_t`] mode evolution.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if `token_z_score` or `cvoid_score` is non-finite.
    #[inline(always)]
    pub fn eval_stateful_with_cvoid_and_pointer(
        &mut self,
        token_z_score: f32,
        cvoid_score: f32,
        alpha: f32,
        beta: f32,
        d0: f32,
        input_ptr: gcso_tagged_ptr_t,
        trail_density: f32,
        dominant_attractor_slot: u8,
        high_density_srl_thresh: f32,
    ) -> GcsoResult<(gcso_edbc_state_t, gcso_tagged_ptr_t)> {
        if !branchless_is_finite_f32(token_z_score) || !branchless_is_finite_f32(cvoid_score) {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }
        Ok(self.eval_stateful_with_cvoid_and_pointer_finite(
            token_z_score,
            cvoid_score,
            alpha,
            beta,
            d0,
            input_ptr,
            trail_density,
            dominant_attractor_slot,
            high_density_srl_thresh,
        ))
    }

    /// Fast register-pipelined Top-2 surrogate + inter-layer cosine distance Moving Z-Score step
    /// on [`EdbcEngine`] using `self.state.singularity_eps` and bypassing redundant `NaN`/`Inf`
    /// checks on the already-sanitized `h_tilde` (`0` branches, `0 fdiv`, `0 ln`).
    ///
    /// # Errors
    /// Never fails on valid finite/non-finite inputs because [`compute_sparse_surrogate_entropy`]
    /// and [`compute_moving_z_score_entropy`] sanitize non-finite inputs to `0.0`.
    #[inline(always)]
    pub fn eval_surrogate_and_step(
        &mut self,
        p_max: f32,
        p_2nd: f32,
        cosine_dist: f32,
        mu_dist: f32,
        sigma_dist: f32,
        alpha_z: f32,
        gamma_mod: f32,
        state_out: &mut gcso_edbc_state_t,
    ) -> GcsoResult<f32> {
        let eps = self.state.singularity_eps;
        let h_surr = compute_sparse_surrogate_entropy(p_max, p_2nd, eps);
        let h_tilde = compute_moving_z_score_entropy(
            h_surr,
            cosine_dist,
            mu_dist,
            sigma_dist,
            eps,
            alpha_z,
            gamma_mod,
        );
        Ok(self.eval_stateful_finite_into(h_tilde, state_out))
    }

    /// Fuses Top-2 Sparse Surrogate Attention Entropy, inter-layer Moving Z-Score normalization,
    /// SIMD $\mathcal{C}_{\mathrm{void}}$ + $\ell_1/\ell_2$ coherence evaluation on `key_vector`,
    /// and pointer-coupled EDBC mode evolution in a single zero-allocation call.
    ///
    /// Returns `(edbc_state, evolved_ptr, cvoid_mean_sq, key_coherence_ratio)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `key_vector` is empty.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if key norms or entropy parameters are non-finite.
    #[inline(always)]
    pub fn eval_surrogate_with_cvoid_and_pointer(
        &mut self,
        p_max: f32,
        p_2nd: f32,
        cosine_dist: f32,
        mu_dist: f32,
        sigma_dist: f32,
        alpha_z: f32,
        gamma_mod: f32,
        key_vector: &AlignedSlice32<'_, f32>,
        alpha: f32,
        beta: f32,
        d0: f32,
        input_ptr: gcso_tagged_ptr_t,
        trail_density: f32,
        dominant_attractor_slot: u8,
        high_density_srl_thresh: f32,
    ) -> GcsoResult<(gcso_edbc_state_t, gcso_tagged_ptr_t, f32, f32)> {
        let (cvoid_score, coh_ratio) = self.eval_cvoid_with_coherence_ratio(key_vector)?;
        let eps = self.state.singularity_eps;
        let h_surr = compute_sparse_surrogate_entropy(p_max, p_2nd, eps);
        let h_tilde = compute_moving_z_score_entropy(
            h_surr,
            cosine_dist,
            mu_dist,
            sigma_dist,
            eps,
            alpha_z,
            gamma_mod,
        );
        let (st, evolved_ptr) = self.eval_stateful_with_cvoid_and_pointer_finite(
            h_tilde,
            cvoid_score,
            alpha,
            beta,
            d0,
            input_ptr,
            trail_density,
            dominant_attractor_slot,
            high_density_srl_thresh,
        );
        Ok((st, evolved_ptr, cvoid_score, coh_ratio))
    }

    /// Executes [`Self::eval_stateful_with_cvoid_and_pointer`] and, when the resulting branch mode
    /// is non-normal (`Mode 1` or `Mode 2`), injects a deterministic SplitMix64-signed
    /// **Metastable Tunneling Phase Perturbation** ($\pm x^*_\pm$, Lemma 3.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `explore_head_start > MAX_ATTENTION_HEADS`.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if parameters are non-finite, `beta <= 0.0`,
    /// or `head_dim` is zero or odd.
    #[inline(always)]
    pub fn step_and_inject_tunneling_perturbation(
        &mut self,
        token_z_score: f32,
        cvoid_score: f32,
        alpha: f32,
        beta: f32,
        d0: f32,
        head_dim: u32,
        explore_head_start: usize,
        input_ptr: gcso_tagged_ptr_t,
        trail_density: f32,
        dominant_attractor_slot: u8,
        high_density_srl_thresh: f32,
        phases_inout: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<(gcso_edbc_state_t, gcso_tagged_ptr_t, f32)> {
        let (st, evolved_ptr) = self.eval_stateful_with_cvoid_and_pointer(
            token_z_score,
            cvoid_score,
            alpha,
            beta,
            d0,
            input_ptr,
            trail_density,
            dominant_attractor_slot,
            high_density_srl_thresh,
        )?;

        if st.active_branch_mode != EDBC_MODE_NORMAL_ENTROPY {
            let x_star = self.inject_metastable_tunneling_perturbation(
                phases_inout,
                evolved_ptr,
                cvoid_score,
                st.bifurcation_threshold,
                alpha,
                beta,
                head_dim,
                explore_head_start,
            )?;
            Ok((st, evolved_ptr, x_star))
        } else {
            Ok((st, evolved_ptr, 0.0))
        }
    }

    /// Evaluates Hodge cohomological obstruction $\mathbf{r}_{\mathrm{obs}} \in H^1(K; \mathcal{F})$
    /// and dual homology pairing $\langle \mathbf{r}_{\mathrm{obs}}, \boldsymbol{\gamma} \rangle$
    /// via [`SheafCohomologyEvaluator::evaluate_hodge_obstruction_and_dual_pairing`], steps the
    /// EDBC controller, overrides into Mode 2 (`EDBC_MODE_PHASE_CONJUGATE_REPULSION`) when a
    /// cohomological obstruction is detected, records the offending 64-head phase trajectory into
    /// `bloom_filter`, and conjugates `active_phases_inout` in-place ($-\boldsymbol{\Delta\theta}$).
    ///
    /// Returns `(edbc_state, evolved_ptr, mean_obstruction, dual_pairing, repulsion_triggered)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if cochain slices are empty or mismatched in length.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if norms or Z-score inputs are non-finite.
    #[inline(always)]
    pub fn step_with_cohomology_and_bloom_guard(
        &mut self,
        token_z_score: f32,
        cvoid_score: f32,
        alpha: f32,
        beta: f32,
        d0: f32,
        cochain_residual_r: &AlignedSlice32<'_, f32>,
        exact_projection_delta0_phi: &AlignedSlice32<'_, f32>,
        dual_homology_cycle_gamma: &AlignedSlice32<'_, f32>,
        obstruction_threshold: f32,
        input_ptr: gcso_tagged_ptr_t,
        trail_density: f32,
        dominant_attractor_slot: u8,
        high_density_srl_thresh: f32,
        active_phases_inout: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
        bloom_filter: &mut ContrastivePhaseBloomFilter,
    ) -> GcsoResult<(gcso_edbc_state_t, gcso_tagged_ptr_t, f32, f32, bool)> {
        let (mean_obs, dual_pairing, is_obstructed) = self
            .evaluate_hodge_obstruction_and_dual_pairing(
                cochain_residual_r,
                exact_projection_delta0_phi,
                dual_homology_cycle_gamma,
                obstruction_threshold,
            )?;

        let (mut st, mut evolved_ptr) = self.eval_stateful_with_cvoid_and_pointer(
            token_z_score,
            cvoid_score,
            alpha,
            beta,
            d0,
            input_ptr,
            trail_density,
            dominant_attractor_slot,
            high_density_srl_thresh,
        )?;

        let repulsion_triggered =
            is_obstructed | (st.active_branch_mode == EDBC_MODE_PHASE_CONJUGATE_REPULSION);

        if repulsion_triggered {
            let phase_w = load_q7_words_64(active_phases_inout);
            bloom_filter.insert_repulsive_pattern_words_weighted(&phase_w, 1);
            let conj_w = [
                swar_q7_neg_sat_x8(phase_w[0]),
                swar_q7_neg_sat_x8(phase_w[1]),
                swar_q7_neg_sat_x8(phase_w[2]),
                swar_q7_neg_sat_x8(phase_w[3]),
                swar_q7_neg_sat_x8(phase_w[4]),
                swar_q7_neg_sat_x8(phase_w[5]),
                swar_q7_neg_sat_x8(phase_w[6]),
                swar_q7_neg_sat_x8(phase_w[7]),
            ];
            store_q7_words_64(active_phases_inout, &conj_w);

            if st.active_branch_mode != EDBC_MODE_PHASE_CONJUGATE_REPULSION {
                let norm_h = branchless_clamp_f32(
                    branchless_relu_f32(st.moving_z_entropy) * INV_LN_2,
                    0.0,
                    1.0,
                );
                let obs_boost = branchless_clamp_f32(
                    branchless_relu_f32(mean_obs - obstruction_threshold) * 0.1,
                    0.0,
                    0.5,
                );
                let rep_temp = branchless_clamp_f32(
                    DEFAULT_SAMPLE_TEMPERATURE * (1.0 + 0.25 * norm_h + obs_boost),
                    st.sample_temperature,
                    5.0,
                );
                self.state.active_branch_mode = EDBC_MODE_PHASE_CONJUGATE_REPULSION;
                self.state.repulsion_gain = LUT_REPULSION_GAIN[2];
                self.state.sample_temperature = rep_temp;
                self.mode_dwell_steps = 1;
                st.active_branch_mode = EDBC_MODE_PHASE_CONJUGATE_REPULSION;
                st.repulsion_gain = LUT_REPULSION_GAIN[2];
                st.sample_temperature = rep_temp;
                let rep_mode = evolved_ptr.action_mode_flags() | ACTION_PRESET_REPULSION_ESCAPE;
                evolved_ptr = evolved_ptr
                    .with_action_and_slot(ATTRACTOR_SLOT_REPULSION, rep_mode)
                    .conjugate_phase();
            }
        }

        Ok((st, evolved_ptr, mean_obs, dual_pairing, repulsion_triggered))
    }

    /// Convenience entrypoint on [`EdbcEngine`] that synthesizes a [`GatewayFrame`] from any
    /// 2-Origin [`ManifoldGatewaySignal`] (`(F, P)`, `(S, F, P)`, `(S, F, P, G)`, slice ensembles,
    /// or [`CrystallizedAttractorRecord`]), executes [`EdbcClosedLoopCoordinator::step_via_gateway_frame`],
    /// and optionally updates `crystal_record_opt` for `.gcso` persistence.
    ///
    /// Returns `(edbc_state, evolved_ptr, causal_report, effective_inv_temp, crystallized_gravity_mass)`.
    ///
    /// # Errors
    /// Propagates any validation or singularity error from [`EdbcClosedLoopCoordinator::step_via_gateway_frame`].
    #[inline(always)]
    pub fn step_gateway_signal_with_defaults<G: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        token_z_score: f32,
        cvoid_score: f32,
        gateway_signal: &G,
        trail: &mut gcso_pointer_trail_t,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
        crystal_record_opt: Option<&mut CrystallizedAttractorRecord>,
    ) -> GcsoResult<(
        gcso_edbc_state_t,
        gcso_tagged_ptr_t,
        CausalProvenanceReport,
        f32,
        f32,
    )> {
        self.step_via_gateway_signal(
            token_z_score,
            cvoid_score,
            1.2,
            0.8,
            0.5,
            1.5,
            gateway_signal,
            trail,
            logits_opt,
            crystal_record_opt,
        )
    }

    /// Convenience entrypoint on [`EdbcEngine`] executing
    /// [`EdbcClosedLoopCoordinator::step_via_gateway_kmeans_64`] with default Langevin pitchfork
    /// parameters and [`DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE`] (`0.35`).
    ///
    /// Returns `(edbc_state, evolved_ptr, causal_report, effective_inv_temp, assigned_cluster_idx, assigned_anchor_id, updated_gravity_mass, spawned_new_cluster)`.
    ///
    /// # Errors
    /// Propagates any error from [`EdbcClosedLoopCoordinator::step_via_gateway_kmeans_64`].
    #[inline(always)]
    pub fn step_gateway_kmeans_with_defaults<G: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        token_z_score: f32,
        cvoid_score: f32,
        gateway_signal: &G,
        centroids: &mut [CrystallizedAttractorRecord],
        trail: &mut gcso_pointer_trail_t,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(
        gcso_edbc_state_t,
        gcso_tagged_ptr_t,
        CausalProvenanceReport,
        f32,
        usize,
        u32,
        f32,
        bool,
    )> {
        self.step_via_gateway_kmeans_64(
            token_z_score,
            cvoid_score,
            1.2,
            0.8,
            0.5,
            1.5,
            gateway_signal,
            centroids,
            DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
            trail,
            logits_opt,
        )
    }

    /// Convenience entrypoint on [`EdbcEngine`] that folds a slice of `.gcso`
    /// [`CrystallizedAttractorRecord`]s on $SO(2)^{64}$ weighted by Phase-Proximity Gravity and
    /// steps EDBC + pointer mode evolution in a single call.
    ///
    /// Returns `(edbc_state, evolved_ptr, total_gravity_pull, mean_coherence, dominant_anchor_id)`.
    ///
    /// # Errors
    /// Propagates any error from [`EdbcClosedLoopCoordinator::step_with_crystallized_ensemble_and_pointer`].
    #[inline(always)]
    pub fn step_crystallized_ensemble_with_defaults(
        &mut self,
        token_z_score: f32,
        cvoid_score: f32,
        records: &[CrystallizedAttractorRecord],
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(gcso_edbc_state_t, gcso_tagged_ptr_t, f32, f32, u32)> {
        self.step_with_crystallized_ensemble_and_pointer(
            token_z_score,
            cvoid_score,
            records,
            1.5,
            trail,
        )
    }
}

impl Default for EdbcEngine {
    #[inline(always)]
    fn default() -> Self {
        Self::new(gcso_edbc_state_t::default())
    }
}

// ===================================================================
// 5. Minimal Trait Delegations & Origin-1 `SystemConstraintSignal` Implementation
// ===================================================================

impl EdbcEngine {
    /// Returns the 8-bit active SWAR Micro-Swarm cell mask (`0x0F` in low-entropy inertial mode, `0xFF` otherwise).
    #[inline(always)]
    pub const fn active_swarm_cell_mask(&self) -> u8 {
        let low_e = (self.state.active_branch_mode == EDBC_MODE_NORMAL_ENTROPY)
            & (self.state.moving_z_entropy < 0.10)
            & (self.state.sliding_entropy_rate <= self.tau_flux);
        branchless_select_u8(low_e, 0x0F, 0xFF)
    }

    /// Returns `true` when the EDBC controller recommends inertial layer skipping (`ACTION_MODE_LAYER_SKIP`).
    #[inline(always)]
    pub const fn requests_inertial_layer_skip(&self) -> bool {
        let h_cur = branchless_relu_f32(self.state.moving_z_entropy);
        (self.state.active_branch_mode == EDBC_MODE_NORMAL_ENTROPY)
            & (h_cur <= 0.12)
            & (self.state.sliding_entropy_rate <= self.tau_flux)
    }
}

impl CausalProvenanceSignal for EdbcEngine {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let is_rep = self.state.active_branch_mode == EDBC_MODE_PHASE_CONJUGATE_REPULSION;
        let h_cur = branchless_relu_f32(self.state.moving_z_entropy);
        let can_skip = (self.state.active_branch_mode == EDBC_MODE_NORMAL_ENTROPY)
            & (h_cur <= 0.15)
            & (self.state.sliding_entropy_rate <= self.tau_flux);
        CAUSAL_AXIS_SYSTEM_CONSTRAINT
            | branchless_select_u32(is_rep, CAUSAL_CAUSE_COHOMOLOGY_REPULSION, 0)
            | branchless_select_u32(can_skip, CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE, 0)
    }

    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        let h_sev = branchless_clamp_f32(
            branchless_relu_f32(self.state.moving_z_entropy) * 0.25,
            0.0,
            1.0,
        );
        let rep_sev = branchless_select_f32(
            self.state.active_branch_mode == EDBC_MODE_PHASE_CONJUGATE_REPULSION,
            0.85,
            0.0,
        );
        branchless_max_f32(h_sev, rep_sev)
    }
}

impl SystemConstraintSignal for EdbcEngine {
    #[inline(always)]
    fn system_vram_pressure(&self) -> f32 {
        branchless_clamp_f32(
            branchless_relu_f32(self.state.moving_z_entropy) * 0.25,
            0.0,
            1.0,
        )
    }

    #[inline(always)]
    fn system_gpu_frame_overrun(&self) -> f32 {
        branchless_clamp_f32(
            branchless_relu_f32(self.state.sliding_entropy_rate),
            0.0,
            1.0,
        )
    }

    #[inline(always)]
    fn active_swarm_cells_mask(&self) -> u8 {
        self.active_swarm_cell_mask()
    }

    #[inline(always)]
    fn recommend_layer_skip(&self) -> bool {
        self.requests_inertial_layer_skip()
    }
}

impl ManifoldGatewaySignal for EdbcEngine {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        synthesize_multi_origin_gateway_frame_with_words::<
            Self,
            ObjectiveFact<[u8; 0]>,
            SubjectivePersona<[u8; 0]>,
            ContradictionGuard<()>,
        >(Some(self), None, None, None)
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        synthesize_multi_origin_gateway_frame::<
            Self,
            ObjectiveFact<[u8; 0]>,
            SubjectivePersona<[u8; 0]>,
            ContradictionGuard<()>,
        >(Some(self), None, None, None)
    }
}

impl EntropyEvaluator for EdbcEngine {
    /// Evaluates token Z-score activation entropy and updates dynamic pitchfork branch mode
    /// in a single register-pipelined pass with a 3-`cmov` state machine and zero Store-to-Load stalls.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if `token_z_score` is `NaN` or `Inf`.
    #[inline(always)]
    fn eval_stateful(
        &mut self,
        token_z_score: f32,
        state_out: &mut gcso_edbc_state_t,
    ) -> GcsoResult<()> {
        if !branchless_is_finite_f32(token_z_score) {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }
        let _ = self.eval_stateful_finite_into(token_z_score, state_out);
        Ok(())
    }

    /// Computes $\mathcal{C}_{\mathrm{void}}$ mean-square latent energy over a 32-byte aligned Key vector
    /// using the AVX2 tree-reduced accumulator (`sum_sq_f32_simd8`) and division-free reciprocal scaling.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `key_vector` is empty.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if the accumulated sum-of-squares is `NaN` or `Inf`.
    #[inline(always)]
    fn eval_cvoid_dyadic128(
        &self,
        key_vector: &AlignedSlice32<'_, f32>,
        void_score_out: &mut f32,
    ) -> GcsoResult<()> {
        let slice = key_vector.as_slice();
        let len = slice.len();
        if len == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let sum_sq = sum_sq_f32_simd8(slice);
        if !branchless_is_finite_f32(sum_sq) {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }
        *void_score_out = branchless_flush_subnormal_f32(sum_sq * fast_inv_len_f32(len));
        Ok(())
    }

    /// Calculates Eyring-Kramers potential barrier height:
    /// $\Delta V = \min\left(\frac{\tau_{\mathrm{eff}}}{\max(0, \mathcal{C}_{\mathrm{void}}) + \epsilon_{\mathrm{log}}}, \Delta V_{\max}\right)$
    /// in division-free (`0 fdiv`) mode, clamping `denom` to `F32_MIN_POSITIVE_NORMAL` and
    /// evaluating `fast_inv_pos_normal_f32` to eliminate subnormal-branch jitter near epsilon singularities.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if inputs are non-finite or `tau_eff <= 0.0`.
    #[inline(always)]
    fn eval_barrier(&self, void_score: f32, tau_eff: f32) -> GcsoResult<f32> {
        let valid = branchless_is_finite_f32(void_score)
            & branchless_is_finite_f32(tau_eff)
            & (tau_eff > 0.0);
        if !valid {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }
        let eps_raw = self.state.singularity_eps;
        let eps = branchless_select_f32(
            branchless_is_finite_f32(eps_raw) & (eps_raw > 0.0),
            eps_raw,
            DEFAULT_SINGULARITY_EPS,
        );
        let denom = branchless_max_f32(
            branchless_relu_f32(void_score) + eps,
            F32_MIN_POSITIVE_NORMAL,
        );
        let inv_denom = fast_inv_pos_normal_f32(denom);
        let barrier = tau_eff * inv_denom;
        if !branchless_is_finite_f32(barrier) {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }
        Ok(branchless_flush_subnormal_f32(branchless_min_f32(
            barrier,
            MAX_BARRIER_CEILING,
        )))
    }
}

impl EdbcBranchController for EdbcEngine {
    #[inline(always)]
    fn active_branch_mode(&self) -> u32 {
        self.state.active_branch_mode
    }

    #[inline(always)]
    fn set_bifurcation_threshold(&mut self, threshold: f32) {
        Self::set_bifurcation_threshold(self, threshold);
    }

    #[inline(always)]
    fn set_hysteresis_band(&mut self, band: f32) {
        Self::set_hysteresis_band(self, band);
    }

    #[inline(always)]
    fn set_tau_flux(&mut self, tau_flux: f32) {
        Self::set_tau_flux(self, tau_flux);
    }

    #[inline(always)]
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
        eval_temperature_for_mode(
            self.state.active_branch_mode,
            base_temperature,
            moving_z_entropy,
            cvoid_score,
            tau_eff,
            alpha,
            beta,
            d0,
        )
    }
}

impl SheafCohomologyEvaluator for EdbcEngine {}

impl HierarchicalTreeTable for EdbcEngine {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::aligned::AlignedBuffer32;
    use crate::traits::primitives::{
        ACTION_PRESET_EXPRESS_PHASE, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA,
    };
    use crate::traits::telemetry::PagedBitmaskOps;

    #[test]
    fn test_surrogate_and_moving_z_entropy() {
        let entropy = compute_sparse_surrogate_entropy(0.5, 0.5, 1e-12);
        assert!((entropy - LN_2).abs() < 1e-6);
        assert_eq!(compute_sparse_surrogate_entropy(1.0, 0.0, 1e-12), 0.0);
        assert_eq!(compute_sparse_surrogate_entropy(-0.5, 0.5, 1e-12), 0.0);

        let asym_h = compute_sparse_surrogate_entropy(0.8, 0.2, 1e-12);
        assert!(asym_h > 0.45 && asym_h < LN_2);

        let unnorm_h = compute_sparse_surrogate_entropy(1.6, 0.4, 1e-12);
        assert!((asym_h - unnorm_h).abs() < 1e-6);

        let tail_h = compute_sparse_surrogate_entropy(0.99, 0.01, 1e-12);
        assert!(tail_h > 0.03 && tail_h < 0.07);

        let h_tilde = compute_moving_z_score_entropy(0.5, 0.8, 0.2, 0.1, 1e-6, 1.0, 0.5);
        assert!(h_tilde > 0.5);
        assert_eq!(
            compute_moving_z_score_entropy(0.5, 0.1, 0.2, 0.1, 1e-6, 1.0, 0.5),
            0.5
        );
        assert_eq!(
            compute_moving_z_score_entropy(0.0, 0.8, 0.2, 0.1, 1e-6, 1.0, 0.5),
            0.0
        );
    }

    #[test]
    fn test_edbc_engine_init_setters_and_reset() {
        let mut engine = EdbcEngine::default();
        assert_eq!(engine.state().bifurcation_threshold, 1.5);
        assert_eq!(engine.state().singularity_eps, 1e-12);
        assert_eq!(engine.state().sample_temperature, 1.0);
        assert_eq!(engine.hysteresis_band, DEFAULT_HYSTERESIS_BAND);
        assert_eq!(engine.tau_flux, DEFAULT_TAU_FLUX);
        assert!(engine.is_active());
        assert!(engine.is_normal_mode());
        assert_eq!(engine.as_bytes().len(), 128);

        engine.set_bifurcation_threshold(2.0);
        engine.set_bifurcation_threshold(-1.0);
        assert_eq!(engine.state().bifurcation_threshold, 2.0);

        engine.state.moving_z_entropy = 2.5;
        engine.state.active_branch_mode = EDBC_MODE_PHASE_CONJUGATE_REPULSION;
        engine.reset();
        assert_eq!(engine.moving_z_entropy(), 0.0);
        assert!(engine.is_normal_mode());
    }

    #[test]
    fn test_eval_stateful_modes_spike_ceiling_and_dwell_anti_chattering() {
        let mut engine = EdbcEngine::default().with_min_mode_dwell_steps(3);
        engine.set_bifurcation_threshold(1.0);
        engine.set_hysteresis_band(0.05);
        engine.set_tau_flux(0.5);
        let mut out_state = gcso_edbc_state_t::default();

        engine.state.moving_z_entropy = 2.0;
        engine.eval_stateful(10.0, &mut out_state).unwrap();
        assert_eq!(
            out_state.active_branch_mode,
            EDBC_MODE_PHASE_CONJUGATE_REPULSION
        );
        assert_eq!(engine.mode_dwell_steps, 1);

        let (emit_iframe, gop_len, can_skip, gate_scale) =
            engine.evaluate_adaptive_compute_and_gop_budget(6, 4, 32, 0.20);
        assert!(emit_iframe);
        assert_eq!(gop_len, 6);
        assert!(!can_skip);
        assert!((gate_scale - 1.0).abs() < 1e-5);

        engine.state.moving_z_entropy = 0.2;
        engine.eval_stateful(0.0, &mut out_state).unwrap();
        assert_eq!(
            out_state.active_branch_mode,
            EDBC_MODE_PHASE_CONJUGATE_REPULSION
        );
        assert_eq!(engine.mode_dwell_steps, 2);

        engine.eval_stateful(0.0, &mut out_state).unwrap();
        assert_eq!(out_state.active_branch_mode, EDBC_MODE_NORMAL_ENTROPY);

        let (emit_after, _gop_after, can_skip_after, _gate_after) =
            engine.evaluate_adaptive_compute_and_gop_budget(2, 4, 32, 0.25);
        assert!(!emit_after);
        assert!(can_skip_after);

        engine.reset();
        engine.eval_stateful(1.0e5, &mut out_state).unwrap();
        assert!(out_state.moving_z_entropy <= 1.0 + 1e-5);
        for _ in 0..15 {
            engine.eval_stateful(0.0, &mut out_state).unwrap();
        }
        assert!(engine.is_normal_mode());

        let (_flux, accel) = engine.eval_entropy_flux_acceleration();
        assert!(accel.is_finite());

        assert_eq!(
            engine.eval_stateful(f32::NAN, &mut out_state),
            Err(GCSO_ERROR_EDBC_SINGULARITY)
        );
    }

    #[test]
    fn test_cvoid_barriers_pointer_coupled_cohomology_limina_and_unified_field() {
        let mut engine = EdbcEngine::default();
        let mut arr = [2.0f32; 35];
        arr[32..].copy_from_slice(&[3.0f32; 3]);
        let buf = AlignedBuffer32::new(arr);
        let slice = buf.as_aligned_slice();

        let mut score = 0.0f32;
        engine.eval_cvoid_dyadic128(&slice, &mut score).unwrap();
        let expected = (32.0 * 4.0 + 3.0 * 9.0) / 35.0;
        assert!((score - expected).abs() < 1e-4);

        let (mean_sq, coh) = engine.eval_cvoid_with_coherence_ratio(&slice).unwrap();
        assert!((mean_sq - expected).abs() < 1e-4);
        assert!(coh > 0.95 && coh <= 1.0);

        let buf128 = AlignedBuffer32::new([2.0f32; 128]);
        let mut score128 = 0.0f32;
        engine
            .eval_cvoid_dyadic128(&buf128.as_aligned_slice(), &mut score128)
            .unwrap();
        assert!((score128 - 4.0).abs() < 1e-6);

        assert!((engine.eval_barrier(2.0, 1.0).unwrap() - 0.5).abs() < 1e-5);
        assert!(engine.eval_barrier(0.0, 1.0).unwrap() <= MAX_BARRIER_CEILING);
        assert_eq!(
            engine.eval_barrier(2.0, -1.0),
            Err(GCSO_ERROR_EDBC_SINGULARITY)
        );
        assert!(
            (eval_barrier_pitchfork(2.0, 1.0, 1.0, 1.0, 128).unwrap() - (1.0 / 512.0)).abs() < 1e-6
        );
        assert_eq!(
            eval_barrier_pitchfork(0.5, 1.0, 1.0, 1.0, 128).unwrap(),
            0.0
        );

        let ptr = gcso_tagged_ptr_t::pack_action(
            0x4000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            16,
        );
        engine.state.moving_z_entropy = 2.0;
        let (st, evolved_ptr) = engine
            .eval_stateful_with_cvoid_and_pointer(
                10.0,
                0.2,
                1.0,
                1.0,
                0.5,
                ptr,
                1.0,
                ATTRACTOR_SLOT_KNOWLEDGE,
                1.5,
            )
            .unwrap();
        assert_eq!(st.active_branch_mode, EDBC_MODE_PHASE_CONJUGATE_REPULSION);
        assert!(st.sample_temperature > 1.0);
        assert_eq!(
            evolved_ptr.target_attractor_slot(),
            ATTRACTOR_SLOT_REPULSION
        );
        assert_eq!(
            evolved_ptr.action_mode_flags() & ACTION_PRESET_REPULSION_ESCAPE,
            ACTION_PRESET_REPULSION_ESCAPE
        );

        let mut phases = [0i8; MAX_ATTENTION_HEADS];
        let (st_bif, _ptr_bif, x_star) = engine
            .step_and_inject_tunneling_perturbation(
                10.0,
                3.0,
                4.0,
                1.0,
                0.5,
                128,
                48,
                ptr,
                1.0,
                ATTRACTOR_SLOT_KNOWLEDGE,
                1.5,
                &mut phases,
            )
            .unwrap();
        assert_eq!(st_bif.active_branch_mode, EDBC_MODE_PITCHFORK_BIFURCATION);
        assert!(x_star > 0.0);
        assert_eq!(phases[0], 0);
        assert_ne!(phases[48], 0);

        engine.reset();
        let r_cochain = AlignedBuffer32::new([2.0f32; 8]);
        let p_exact = AlignedBuffer32::new([0.5f32; 8]);
        let gamma_cycle = AlignedBuffer32::new([1.0f32; 8]);
        let mut active_p = [24i8; MAX_ATTENTION_HEADS];
        let mut bloom = ContrastivePhaseBloomFilter::new();

        let (st_coh, ptr_coh, mean_obs, dual_pair, rep_trig) = engine
            .step_with_cohomology_and_bloom_guard(
                0.1,
                0.9,
                1.0,
                1.0,
                0.5,
                &r_cochain.as_aligned_slice(),
                &p_exact.as_aligned_slice(),
                &gamma_cycle.as_aligned_slice(),
                1.0,
                ptr,
                1.0,
                ATTRACTOR_SLOT_PERSONA,
                1.5,
                &mut active_p,
                &mut bloom,
            )
            .unwrap();
        assert!(rep_trig);
        assert!((mean_obs - 2.25).abs() < 1e-5);
        assert!((dual_pair - 1.0).abs() < 1e-5);
        assert_eq!(
            st_coh.active_branch_mode,
            EDBC_MODE_PHASE_CONJUGATE_REPULSION
        );
        assert!(st_coh.sample_temperature > 1.0);
        assert_eq!(ptr_coh.target_attractor_slot(), ATTRACTOR_SLOT_REPULSION);
        assert_eq!(active_p[0], -24);
        assert_eq!(
            bloom.query_repulsive_density(&[24i8; MAX_ATTENTION_HEADS]),
            1
        );

        engine.reset();
        let unified_anchors = [[-96i8; MAX_ATTENTION_HEADS], [32i8; MAX_ATTENTION_HEADS]];
        let unified_energies = [0.2f32, 3.0f32];
        let mut uf_trail = gcso_pointer_trail_t {
            current_ptr: ptr.raw(),
            stigmergic_density: 1.0,
            accumulated_phase_delta: [28i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let (_st_uf, _ptr_uf, tot_pull, mean_c, dom_id) = engine
            .step_with_unified_field_and_pointer(
                0.1,
                0.8,
                &unified_anchors,
                &unified_energies,
                1.5,
                &mut uf_trail,
            )
            .unwrap();
        assert_eq!(dom_id, 1);
        assert_eq!(uf_trail.cluster_id, 1);
        assert!(tot_pull > 5.0 && mean_c > 0.85);

        let t_next = gcso_pointer_trail_t {
            current_ptr: ptr.raw(),
            stigmergic_density: 1.2,
            accumulated_phase_delta: [32i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let mut chain = [uf_trail, t_next];
        let (_st_lim, _ptr_lim, joint_lp, hol_obs, lim_rep) = engine
            .step_with_liminal_trail_chain_and_bloom_guard(
                &mut chain,
                64,
                0.25,
                0.20,
                0.1,
                0.8,
                1.5,
                ATTRACTOR_SLOT_PERSONA,
                &mut bloom,
            )
            .unwrap();
        assert!(!lim_rep && hol_obs < 0.05 && joint_lp > -0.05);
        assert_eq!(chain[1].cluster_id, 1);

        let t_contradict = gcso_pointer_trail_t {
            current_ptr: ptr.raw(),
            stigmergic_density: 1.2,
            accumulated_phase_delta: [-96i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let mut bad_chain = [uf_trail, t_contradict];
        let (st_bad, ptr_bad, _bad_lp, _bad_obs, bad_rep) = engine
            .step_with_liminal_trail_chain_and_bloom_guard(
                &mut bad_chain,
                64,
                0.25,
                0.20,
                0.1,
                0.8,
                1.5,
                ATTRACTOR_SLOT_PERSONA,
                &mut bloom,
            )
            .unwrap();
        assert!(bad_rep);
        assert_eq!(
            st_bad.active_branch_mode,
            EDBC_MODE_PHASE_CONJUGATE_REPULSION
        );
        assert_eq!(ptr_bad.target_attractor_slot(), ATTRACTOR_SLOT_REPULSION);

        let (_st_surr, _ptr_surr, cvoid_s, coh_s) = engine
            .eval_surrogate_with_cvoid_and_pointer(
                0.5,
                0.5,
                0.8,
                0.2,
                0.1,
                1.0,
                0.5,
                &slice,
                1.0,
                1.0,
                0.5,
                ptr,
                1.0,
                ATTRACTOR_SLOT_PERSONA,
                1.5,
            )
            .unwrap();
        assert!((cvoid_s - expected).abs() < 1e-4);
        assert!(coh_s > 0.95);
    }

    #[test]
    fn test_blanket_bayesian_pointer_selection_temperature_and_dyadic_tree() {
        let mut engine = EdbcEngine::default();
        engine.state.moving_z_entropy = 1.6;
        let ptr = gcso_tagged_ptr_t::pack_action(
            0x5000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            16,
        );

        let mut src_trail = gcso_pointer_trail_t {
            current_ptr: ptr.raw(),
            stigmergic_density: 1.8,
            attractor_pull_force: 2.0,
            target_anchor_id: 19,
            cluster_id: 19,
            accumulated_phase_delta: [28i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };

        let cand_bad = gcso_pointer_trail_t {
            current_ptr: 0x5100,
            stigmergic_density: 4.5,
            accumulated_phase_delta: [-96i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let cand_good = gcso_pointer_trail_t {
            current_ptr: 0x5200,
            stigmergic_density: 3.0,
            target_anchor_id: 27,
            accumulated_phase_delta: [32i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let mut candidates = [cand_bad, cand_good];
        let mut bloom = ContrastivePhaseBloomFilter::new();
        let mut bayes_scores = AlignedBuffer32::new([0.0f32; 8]);

        let (st_bayes, _ptr_bayes, sel_id, bayes_score, trans_lp, tun_x, found) = engine
            .step_with_bayesian_pointer_selection_and_tunneling(
                &mut src_trail,
                &mut candidates,
                2.0,
                2.5,
                2.0,
                1.0,
                0.5,
                128,
                48,
                2.0,
                0.5,
                0.25,
                1.0,
                64,
                0.25,
                1.5,
                &mut bloom,
                Some(&mut bayes_scores.as_aligned_slice_mut()),
            )
            .unwrap();
        assert!(found);
        assert_eq!(st_bayes.active_branch_mode, EDBC_MODE_PITCHFORK_BIFURCATION);
        assert!(tun_x > 0.0);
        assert_eq!(sel_id, 1);
        assert_eq!(src_trail.linked_trail_id, 1);
        assert_eq!(src_trail.target_anchor_id, 27);
        assert_eq!(src_trail.cluster_id, 27);
        assert!(bayes_score > 0.5 && trans_lp > -0.25);
        assert!(bayes_scores.data[1] > 0.8 && bayes_scores.data[0] < -0.5);

        let mut attn_scores = AlignedBuffer32::new([0.0f32; 8]);
        let (mean_coh, eff_inv_temp, best_id, coh_cnt) = engine
            .modulate_candidate_trails_with_edbc_temperature(
                &mut attn_scores.as_aligned_slice_mut(),
                &src_trail,
                &candidates,
                128,
                2.0,
                0.5,
                0.25,
                0.25,
                1.0,
                1.0,
                st_bayes.moving_z_entropy,
                0.9,
                1.5,
                1.5,
                0.5,
                0.5,
            )
            .unwrap();
        assert!(mean_coh.is_finite());
        assert!(eff_inv_temp > 0.5);
        assert_eq!(best_id, 1);
        assert_eq!(coh_cnt, 1);
        assert!(attn_scores.data[1] > attn_scores.data[0]);

        let mut m_cur = gcso_paged_bitmask_t::default();
        let mut m_anc = gcso_paged_bitmask_t::default();
        m_cur.set_bit_branchless(3);
        m_anc.set_bit_branchless(3);
        let (_st_tree, qdpt_lvl, d_v, tree_ptr) = engine
            .resolve_dyadic_tree_and_step_edbc(
                &m_cur,
                &m_anc,
                0.05,
                0,
                0.10,
                &[0x100, 0x200, 0x300, 0x400],
                &[4, 8, 12, 16],
                0.1,
                0.8,
                1.5,
                &mut src_trail,
            )
            .unwrap();
        assert_eq!(qdpt_lvl, 0);
        assert_eq!(d_v, 0.0);
        assert_eq!(tree_ptr.unpack_addr(), 0x5100);
        assert_eq!(src_trail.cluster_id, 27);
    }

    #[test]
    fn test_edbc_origin1_system_constraint_gateway_kmeans_and_gcso_crystallization() {
        let mut engine = EdbcEngine::default();
        assert_eq!(engine.active_swarm_cell_mask(), 0x0F);
        assert!(engine.requests_inertial_layer_skip());
        let sys_only_frame = engine.synthesize_gateway_frame();
        assert_eq!(sys_only_frame.active_swarm_cell_mask, 0x0F);
        assert_ne!(sys_only_frame.action_mode_flags & ACTION_MODE_LAYER_SKIP, 0);

        let mut st_init = gcso_edbc_state_t::default();
        engine.eval_stateful(2.5, &mut st_init).unwrap();
        assert_eq!(engine.active_swarm_cell_mask(), 0xFF);

        let fact = ObjectiveFact::new([28i8; MAX_ATTENTION_HEADS], 2.5, 0x0012_3400, 33);
        let persona = SubjectivePersona::new([30i8; MAX_ATTENTION_HEADS], 2.0, 0x0020_0000, 19);
        let composite_gateway = (engine, fact, persona);

        let mut trail = gcso_pointer_trail_t::default();
        let mut logits = AlignedBuffer32::new([1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25]);
        let mut crystal =
            CrystallizedAttractorRecord::new_empty(33, ATTRACTOR_SLOT_KNOWLEDGE, 0x0012_3400);

        let mut prev_mass = 0.0_f32;
        for _ in 0..4 {
            let (_st, evolved_ptr, report, eff_inv_temp, mass) = engine
                .step_gateway_signal_with_defaults(
                    0.35,
                    0.85,
                    &composite_gateway,
                    &mut trail,
                    Some(&mut logits.as_aligned_slice_mut()),
                    Some(&mut crystal),
                )
                .unwrap();
            assert_eq!(evolved_ptr.unpack_addr(), 0x0012_3400);
            assert_eq!(trail.target_anchor_id, 33);
            assert_eq!(trail.cluster_id, 33);
            assert!(eff_inv_temp > 1.0);
            assert!(mass > prev_mass);
            assert!(report.fact_persona_coherence > 0.90);
            prev_mass = mass;
        }
        assert_eq!(crystal.sample_count, 4);
        assert!(crystal.verify_integrity());
        assert_ne!(
            crystal.provenance_bits & CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
            0
        );

        let records = [crystal];
        let mut eval_trail = gcso_pointer_trail_t {
            attractor_pull_force: 1.5,
            accumulated_phase_delta: [26i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let (_st_ens, _ptr_ens, tot_pull, mean_coh, dom_id) = engine
            .step_crystallized_ensemble_with_defaults(0.25, 0.75, &records, &mut eval_trail)
            .unwrap();
        assert_eq!(dom_id, 33);
        assert_eq!(eval_trail.target_anchor_id, 33);
        assert_eq!(eval_trail.cluster_id, 33);
        assert!(tot_pull > 4.0 && mean_coh > 0.90);

        let mut macro_centroids = [
            CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_KNOWLEDGE, 0),
            CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_PERSONA, 0),
        ];
        let (_st_k1, _ptr_k1, _rep_k1, _inv_k1, c_idx1, c_id1, c_mass1, spawned1) = engine
            .step_gateway_kmeans_with_defaults(
                0.35,
                0.85,
                &composite_gateway,
                &mut macro_centroids,
                &mut eval_trail,
                None,
            )
            .unwrap();
        assert!(spawned1);
        assert_eq!(c_idx1, 0);
        assert_eq!(c_id1, 33);
        assert!(c_mass1 > 0.0);

        let opposed_persona =
            SubjectivePersona::new([-84i8; MAX_ATTENTION_HEADS], 2.8, 0x0028_0000, 99);
        let (_st_k2, _ptr_k2, _rep_k2, _inv_k2, c_idx2, c_id2, _c_mass2, spawned2) = engine
            .step_gateway_kmeans_with_defaults(
                0.40,
                0.85,
                &opposed_persona,
                &mut macro_centroids,
                &mut eval_trail,
                None,
            )
            .unwrap();
        assert!(spawned2);
        assert_eq!(c_idx2, 1);
        assert_eq!(c_id2, 99);

        let mut c2f_trail = gcso_pointer_trail_t {
            accumulated_phase_delta: [-80i8; MAX_ATTENTION_HEADS],
            stigmergic_density: 1.5,
            attractor_pull_force: 1.5,
            ..Default::default()
        };
        let mut c2f_candidates = [
            gcso_pointer_trail_t {
                cluster_id: 33,
                target_anchor_id: 33,
                stigmergic_density: 2.0,
                accumulated_phase_delta: [28i8; MAX_ATTENTION_HEADS],
                ..Default::default()
            },
            gcso_pointer_trail_t {
                cluster_id: 99,
                target_anchor_id: 990,
                stigmergic_density: 2.5,
                accumulated_phase_delta: [-82i8; MAX_ATTENTION_HEADS],
                ..Default::default()
            },
        ];
        let mut bloom = ContrastivePhaseBloomFilter::new();
        let (_st_c2f, _ptr_c2f, m_idx, m_id, sel_id, _bs, _lp, _tx, found_c2f) = engine
            .route_coarse_to_fine_and_step_bayesian_tunneling(
                &macro_centroids,
                64,
                &mut c2f_trail,
                &mut c2f_candidates,
                0.45,
                0.90,
                1.2,
                0.8,
                0.5,
                128,
                48,
                2.0,
                0.5,
                0.35,
                1.0,
                64,
                0.25,
                1.5,
                &mut bloom,
                None,
            )
            .unwrap();
        assert_eq!(m_idx, 1);
        assert_eq!(m_id, 99);
        assert!(found_c2f);
        assert_eq!(sel_id, 1);
        assert_eq!(c2f_trail.cluster_id, 99);
        assert_eq!(c2f_trail.target_anchor_id, 990);
        assert_eq!(c2f_candidates[1].cluster_id, 99);
    }
}
