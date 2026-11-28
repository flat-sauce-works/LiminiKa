// name: src/core/src/drivers/spine/entropy/kernels.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Constant-time branchless entropy, Pitchfork bifurcation, and register-pipelined stepping helpers.

use crate::drivers::spine::entropy::{
    DEFAULT_SAMPLE_TEMPERATURE, ENTROPY_GATE_MODE_NORMAL_ENTROPY,
    ENTROPY_GATE_MODE_PITCHFORK_BIFURCATION, INV_LN_2, MAX_VALID_TOKEN_Z_SCORE,
};
use crate::foundation::{
    F32_MIN_POSITIVE_NORMAL, branchless_clamp_f32, branchless_eyring_kramers_escape,
    branchless_flush_subnormal_f32, branchless_is_finite_f32, branchless_max_f32,
    branchless_min_f32, branchless_relu_f32, fast_inv_pos_normal_f32, tanh_lut_eval_nonneg,
};
use crate::liminal::{
    LMNK_ERR_ENTROPY_SINGULARITY, LmnkResult, MAX_SURROGATE_ENTROPY_LN2, branchless_select_f32,
};

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
/// The inter-layer cosine modulation factor $1 + \gamma_{\mathrm{mod}} \tanh(\alpha_z
/// \operatorname{ReLU}(z_{\mathrm{dist}}))$
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
/// $\Delta V = \frac{\alpha^2 (\max(0, \mathcal{E}_{\mathrm{lat}} - \tau_{\mathrm{eff}}))^2}{4 \beta
/// d_{\mathrm{head}}}$
/// in division-free (`0 fdiv`) mode (Lemma 3.1).
///
/// # Errors
/// Returns [`LMNK_ERR_ENTROPY_SINGULARITY`] if any parameter is `NaN`/`Inf`, `beta <= 0.0`,
/// or `head_dim` is zero or odd.
#[inline(always)]
pub const fn eval_barrier_pitchfork(
    liminal_lacuna: f32,
    tau_eff: f32,
    alpha: f32,
    beta: f32,
    head_dim: u32,
) -> LmnkResult<f32> {
    let valid_params = branchless_is_finite_f32(liminal_lacuna)
        & branchless_is_finite_f32(tau_eff)
        & branchless_is_finite_f32(alpha)
        & branchless_is_finite_f32(beta)
        & (beta > 0.0)
        & (head_dim != 0)
        & ((head_dim & 1) == 0);
    if !valid_params {
        return Err(LMNK_ERR_ENTROPY_SINGULARITY);
    }

    let diff = branchless_relu_f32(liminal_lacuna - tau_eff);
    let prod = alpha * diff;
    let num = prod * prod;
    let den = 4.0 * beta * (head_dim as f32);
    let safe_den = branchless_max_f32(den, F32_MIN_POSITIVE_NORMAL);
    let inv_den = fast_inv_pos_normal_f32(safe_den);
    let barrier = branchless_flush_subnormal_f32(num * inv_den);
    if !branchless_is_finite_f32(barrier) {
        return Err(LMNK_ERR_ENTROPY_SINGULARITY);
    }
    Ok(barrier)
}

/// Evaluates the energy-guided dynamic sampling temperature for an explicitly resolved `mode`
/// (`0` = Normal, `1` = Pitchfork Bifurcation, `2` = Repulsion) in **strictly branchless constant
/// time (`0` branches)** across all 3 modes.
#[inline(always)]
#[must_use]
// Hot-path register-passed SIMD/SWAR kernel leaf kept flat to avoid stack-frame spills.
#[allow(clippy::too_many_arguments)]
pub(super) const fn eval_temperature_for_mode(
    mode: u32,
    base_temperature: f32,
    moving_z_entropy: f32,
    lacuna_score: f32,
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
    if mode == ENTROPY_GATE_MODE_NORMAL_ENTROPY {
        return safe_base;
    }
    let norm_h = branchless_clamp_f32(branchless_relu_f32(moving_z_entropy) * INV_LN_2, 0.0, 1.0);
    let candidate = if mode == ENTROPY_GATE_MODE_PITCHFORK_BIFURCATION {
        let escape_p = branchless_eyring_kramers_escape(lacuna_score, tau_eff, alpha, beta, d0);
        safe_base * (1.0 + 0.35 * escape_p * norm_h)
    } else {
        safe_base * (1.0 + 0.25 * norm_h)
    };
    branchless_clamp_f32(candidate, 0.05, 5.0)
}
