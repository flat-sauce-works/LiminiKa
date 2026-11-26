// name: src/core/src/foundation/cohomology.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pure matrix-free sheaf cohomology, Chebyshev spectral filtering, and pseudo-ultrametric depth
//! quantization primitives.

use crate::foundation::aligned::{AlignedSlice32, AlignedSliceMut32};
use crate::foundation::scalar_math::{
    branchless_abs_f32, branchless_clamp_f32, branchless_flush_subnormal_f32,
    branchless_inv_pos_f32, branchless_is_finite_f32, branchless_rsqrt_f32, branchless_tanh_f32,
    fast_inv_len_f32,
};
use crate::foundation::simd_stats::{
    chebyshev_deg3_filter_slice_simd8, diff_norm_sq_f32_unrolled4,
    hodge_obstruction_and_dual_pairing_f32, max_pos_f32_unrolled4,
};
use crate::foundation::swar_simd::branchless_hysteresis_quantize_depth;
use crate::liminal::{
    LMNK_ERR_ENTROPY_SINGULARITY, LMNK_ERR_INVALID_ARGUMENT, LmnkResult, branchless_select_f32,
};

/// Polynomial degree of the Cold-Path Chebyshev spectral filter (`3`).
pub const CHEBYSHEV_FILTER_DEGREE: usize = 3;

/// Number of expansion coefficients $[c_0, c_1, c_2, c_3]$ in a 3rd-order Chebyshev filter (`4`).
pub const CHEBYSHEV_COEFF_LEN: usize = 4;

/// Evaluates the Gershgorin upper bound $\hat{\lambda}_{\max} = 2 \cdot \max_i D_{ii}$ in $\mathcal{O}(\vert V \vert)$ time (Section 1.2).
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `node_degrees` is empty.
#[inline]
pub fn eval_gershgorin_lambda_max(node_degrees: &AlignedSlice32<'_, f32>) -> LmnkResult<f32> {
    if node_degrees.is_empty() {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    Ok(branchless_flush_subnormal_f32(
        2.0 * max_pos_f32_unrolled4(node_degrees.as_slice()),
    ))
}

/// Computes 3rd-order Chebyshev expansion coefficients $[c_0, c_1, c_2, c_3]$ in $\mathcal{O}(1)$
/// using Gauss-Chebyshev nodes and division-free (`0 fdiv`) 4-way Montgomery inversion (Section 1.2).
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `lambda_max <= 0.0`, `epsilon <= 0.0`, or either is non-finite.
#[inline]
pub fn compute_tikhonov_chebyshev_coeffs_deg3(
    lambda_max: f32,
    epsilon: f32,
) -> LmnkResult<[f32; CHEBYSHEV_COEFF_LEN]> {
    if !branchless_is_finite_f32(lambda_max)
        || !branchless_is_finite_f32(epsilon)
        || lambda_max <= 0.0
        || epsilon <= 0.0
    {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
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
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `lambda_max <= 0.0`, `lambda_max` is non-finite,
/// `eigenvalues_in` is empty, or `eigenvalues_in.len() != filtered_out.len()`.
#[inline]
pub fn eval_chebyshev_spectral_filter_deg3(
    eigenvalues_in: &AlignedSlice32<'_, f32>,
    filtered_out: &mut AlignedSliceMut32<'_, f32>,
    lambda_max: f32,
    cheb_coeffs: &[f32; CHEBYSHEV_COEFF_LEN],
) -> LmnkResult<()> {
    if !branchless_is_finite_f32(lambda_max)
        || lambda_max <= 0.0
        || eigenvalues_in.is_empty()
        || eigenvalues_in.len() != filtered_out.len()
    {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
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
pub fn eval_pseudo_ultrametric_distance(l_lca: u32, l_max: u32, w_key: f32) -> f32 {
    let diff = l_max.saturating_sub(l_lca) as f32;
    let safe_w = branchless_select_f32(branchless_is_finite_f32(w_key) & (w_key > 0.0), w_key, 0.0);
    branchless_tanh_f32(safe_w * diff)
}

/// Evaluates the cohomological obstruction residual $\mathbf{r}_{\mathrm{obs}} = \mathbf{r} - \delta_0 \boldsymbol{\phi} \in \mathrm{im}(\delta_0)^\perp$ (Section 1.1).
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `cochain_residual_r` is empty or slice lengths differ.
/// Returns [`LMNK_ERR_ENTROPY_SINGULARITY`] if the squared residual norm overflows or is non-finite.
#[inline]
pub fn evaluate_cohomological_obstruction(
    cochain_residual_r: &AlignedSlice32<'_, f32>,
    exact_projection_delta0_phi: &AlignedSlice32<'_, f32>,
    obstruction_threshold: f32,
) -> LmnkResult<(f32, bool)> {
    let n = cochain_residual_r.len();
    if n == 0 || n != exact_projection_delta0_phi.len() {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    let norm_sq = diff_norm_sq_f32_unrolled4(
        cochain_residual_r.as_slice(),
        exact_projection_delta0_phi.as_slice(),
    );
    if !branchless_is_finite_f32(norm_sq) {
        return Err(LMNK_ERR_ENTROPY_SINGULARITY);
    }
    let mean_obs = branchless_flush_subnormal_f32(norm_sq * fast_inv_len_f32(n));
    Ok((mean_obs, mean_obs > obstruction_threshold))
}

/// Evaluates the Hodge obstruction residual $\mathbf{r}_{\mathrm{obs}} \in H^1(K; \mathcal{F})$
/// and normalized dual Cellular Sheaf Homology pairing $\frac{\vert\langle \mathbf{r}_{\mathrm{obs}}, \boldsymbol{\gamma} \rangle\vert}{\|\mathbf{r}_{\mathrm{obs}}\|_2 \|\boldsymbol{\gamma}\|_2} \in [0, 1]$ (Section 1.1).
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if any slice is empty or lengths differ.
/// Returns [`LMNK_ERR_ENTROPY_SINGULARITY`] if any accumulated norm or pairing inner product is non-finite.
#[inline]
pub fn evaluate_hodge_obstruction_and_dual_pairing(
    cochain_residual_r: &AlignedSlice32<'_, f32>,
    exact_projection_delta0_phi: &AlignedSlice32<'_, f32>,
    dual_homology_cycle_gamma: &AlignedSlice32<'_, f32>,
    obstruction_threshold: f32,
) -> LmnkResult<(f32, f32, bool)> {
    let n = cochain_residual_r.len();
    if n == 0 || exact_projection_delta0_phi.len() != n || dual_homology_cycle_gamma.len() != n {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
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
        return Err(LMNK_ERR_ENTROPY_SINGULARITY);
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

/// Quantizes continuous pseudo-ultrametric depth $d_{\mathcal{V}} \in [0, 1)$ into discrete QDPT level
/// $\bar{D}_{\mathcal{V},\mathrm{quantized}}$ with hysteresis band $\delta_{\mathrm{hyst}}$ (Section 1.3).
#[inline(always)]
#[must_use]
pub fn quantize_depth_hysteresis(
    d_v_distance: f32,
    current_level: u32,
    max_level: u32,
    hysteresis_band: f32,
) -> u32 {
    let scaled = branchless_clamp_f32(d_v_distance, 0.0, 1.0) * (max_level as f32);
    branchless_hysteresis_quantize_depth(scaled, current_level, hysteresis_band, max_level)
}
