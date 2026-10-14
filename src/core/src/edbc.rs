// name: src/core/src/edbc.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Entropy-Driven Decoding Branch Controller (EDBC) core engine implementation.
//!
//! Evaluates Moving Z-Score Normalized Attention Entropy ($\tilde{H}$) and controls
//! Pitchfork Bifurcation mode transitions, CVoid alignment scores, and Eyring-Kramers
//! potential barriers without dynamic heap allocations.

#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::inline_always)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::manual_clamp)]

#[cfg(not(feature = "std"))]
use core::f32::consts::LN_2;
#[cfg(feature = "std")]
use std::f32::consts::LN_2;

use crate::abi::{gcso_edbc_state_t, GCSO_ERROR_EDBC_SINGULARITY, GCSO_ERROR_INVALID_ARGUMENT};
use crate::traits::{AlignedSlice32, EntropyEvaluator, GcsoResult};

/// Normal entropy convergence mode (Mode 0).
pub const EDBC_MODE_NORMAL_ENTROPY: u32 = 0;
/// Pitchfork bifurcation exploration mode (Mode 1).
pub const EDBC_MODE_PITCHFORK_BIFURCATION: u32 = 1;
/// Phase-conjugate repulsion mode (Mode 2).
pub const EDBC_MODE_PHASE_CONJUGATE_REPULSION: u32 = 2;

/// Default singularity epsilon fallback constant (1e-12, aligned strictly with c_abi_spec.md).
pub const DEFAULT_SINGULARITY_EPS: f32 = 1e-12;
/// Default bifurcation threshold constant (1.5).
pub const DEFAULT_BIFURCATION_THRESHOLD: f32 = 1.5;
/// Default entropy rate flux threshold constant (0.5, aligned with GCSO Whitepaper Section 3.2).
pub const DEFAULT_TAU_FLUX: f32 = 0.5;
/// Default initial sample temperature (1.0).
pub const DEFAULT_SAMPLE_TEMPERATURE: f32 = 1.0;
/// Default hysteresis band width for anti-chattering mode stabilization (0.15).
pub const DEFAULT_HYSTERESIS_BAND: f32 = 0.15;

/// Exponential moving average smoothing factor for Z-score entropy tracking (0.1).
const EMA_ALPHA_ENTROPY: f32 = 0.1;

/// Cross-environment single-precision floating point natural logarithm wrapper.
///
/// Uses power-of-two exponent extraction and Remez polynomial series for high accuracy in `no_std`.
#[inline(always)]
fn ln_f32(val: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        val.ln()
    }
    #[cfg(not(feature = "std"))]
    {
        if val <= 0.0 || val.is_nan() {
            return 0.0;
        }
        let bits = val.to_bits();
        let exponent = ((bits >> 23) & 0xFF) as i32 - 127;
        let mantissa = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000);

        let u = (mantissa - 1.0) / (mantissa + 1.0);
        let u2 = u * u;
        let poly =
            u * (2.0 + u2 * (0.666_666_7 + u2 * (0.4 + u2 * (0.285_714_3 + u2 * 0.222_222_2))));

        (exponent as f32) * LN_2 + poly
    }
}

/// Cross-environment single-precision floating point hyperbolic tangent wrapper using Padé (5,4) approximation.
#[inline(always)]
fn tanh_f32(val: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        val.tanh()
    }
    #[cfg(not(feature = "std"))]
    {
        if val.is_nan() {
            return 0.0;
        }
        if val > 4.97 {
            1.0
        } else if val < -4.97 {
            -1.0
        } else {
            let x2 = val * val;
            let x4 = x2 * x2;
            let num = val * (945.0 + 105.0 * x2 + x4);
            let den = 945.0 + 420.0 * x2 + 15.0 * x4;
            num / den
        }
    }
}

/// Computes normalized Sparse Surrogate Attention Entropy from Top-2 attention probabilities.
///
/// Bounded strictly to the value range `[0, ln 2]` (approx `0.6931472`) for numerical stability.
#[inline]
pub fn compute_sparse_surrogate_entropy(p_max: f32, p_2nd: f32, eps: f32) -> f32 {
    if p_max <= 0.0 || p_max.is_nan() || p_2nd.is_nan() {
        return 0.0;
    }
    let p_2nd_clamped = if p_2nd < 0.0 { 0.0 } else { p_2nd };
    let sum = p_max + p_2nd_clamped;
    if sum <= 0.0 {
        return 0.0;
    }

    let p_max_norm = (p_max / sum).clamp(0.0, 1.0);
    let p_2nd_norm = (p_2nd_clamped / sum).clamp(0.0, 1.0);
    let eps_log = if eps > 0.0 && !eps.is_nan() && !eps.is_infinite() {
        eps
    } else {
        DEFAULT_SINGULARITY_EPS
    };

    let term1 = -p_max_norm * ln_f32(p_max_norm + eps_log);
    let term2 = -p_2nd_norm * ln_f32(p_2nd_norm + eps_log);

    (term1 + term2).clamp(0.0, LN_2)
}

/// Computes Moving Z-Score Normalized Attention Entropy ($\tilde{H}$) incorporating
/// surrogate entropy and inter-layer activation cosine distance variance according to GCSO Whitepaper Eq (3.1).
#[inline]
pub fn compute_moving_z_score_entropy(
    h_surrogate: f32,
    cosine_dist: f32,
    mu_dist: f32,
    sigma_dist: f32,
    sigma_min: f32,
    alpha_z: f32,
    gamma_mod: f32,
) -> f32 {
    if h_surrogate <= 0.0 || h_surrogate.is_nan() || cosine_dist.is_nan() || mu_dist.is_nan() {
        return 0.0;
    }
    let effective_sigma_min = if sigma_min > 0.0 && !sigma_min.is_nan() {
        sigma_min
    } else {
        1e-6
    };

    let effective_sigma = if sigma_dist > effective_sigma_min {
        sigma_dist
    } else {
        effective_sigma_min
    };

    let z_dist = (cosine_dist - mu_dist) / effective_sigma;
    let relu_z = if z_dist > 0.0 { z_dist } else { 0.0 };
    let alpha_z_safe = if alpha_z.is_nan() || alpha_z < 0.0 {
        1.0
    } else {
        alpha_z
    };
    let d_cosine_hat = tanh_f32(alpha_z_safe * relu_z);

    let gamma_mod_safe = if gamma_mod.is_nan() || gamma_mod < 0.0 {
        0.5
    } else {
        gamma_mod
    };
    let h_tilde = h_surrogate * (1.0 + gamma_mod_safe * d_cosine_hat);

    if h_tilde.is_nan() || h_tilde.is_infinite() {
        0.0
    } else {
        h_tilde
    }
}

/// Evaluates scale-invariant Pitchfork Bifurcation potential barrier height ($\Delta V$) according to
/// GCSO Whitepaper Lemma 3.1: $\Delta V = \frac{\alpha^2 (\mathcal{C}_{\mathrm{void}} - \tau_{\mathrm{eff}})^2}{4 \beta d_{\mathrm{head}}}$.
///
/// Under Convergence Mode ($\mathcal{C}_{\mathrm{void}} \le \tau_{\mathrm{eff}}$), origin $x^* = 0$ constitutes
/// the sole stable fixed point, yielding no potential escape barrier ($\Delta V = 0.0$).
///
/// # Errors
/// Returns `GCSO_ERROR_EDBC_SINGULARITY` if parameters evaluate to `NaN` or `Inf`, or if `head_dim` is zero.
#[inline]
pub fn eval_barrier_pitchfork(
    cvoid: f32,
    tau_eff: f32,
    alpha: f32,
    beta: f32,
    head_dim: u32,
) -> GcsoResult<f32> {
    if cvoid.is_nan()
        || cvoid.is_infinite()
        || tau_eff.is_nan()
        || tau_eff.is_infinite()
        || alpha.is_nan()
        || alpha.is_infinite()
        || beta <= 0.0
        || beta.is_nan()
        || beta.is_infinite()
        || head_dim == 0
    {
        return Err(GCSO_ERROR_EDBC_SINGULARITY);
    }

    // Convergence mode: cvoid <= tau_eff -> Origin is sole stable attractor, barrier is 0
    if cvoid <= tau_eff {
        return Ok(0.0);
    }

    let diff = cvoid - tau_eff;
    let num = (alpha * alpha) * (diff * diff);
    let den = 4.0 * beta * (head_dim as f32);

    let barrier = num / den;
    if barrier.is_nan() || barrier.is_infinite() {
        return Err(GCSO_ERROR_EDBC_SINGULARITY);
    }

    Ok(barrier)
}

/// Entropy-Driven Decoding Branch Controller (EDBC) stateful engine.
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdbcEngine {
    /// Active EDBC state descriptor.
    pub state: gcso_edbc_state_t,
    /// Controller active lifecycle flag.
    pub is_active: bool,
    /// Hysteresis band width preventing mode oscillation near threshold boundaries.
    pub hysteresis_band: f32,
    /// Threshold for sliding window entropy rate flux (tau_flux).
    pub tau_flux: f32,
    /// Fixed-size ring buffer for Sliding-Window Entropy Rate Integrator ($\Phi_M(t)$, M=5).
    window_buf: [f32; 8],
    /// Ring buffer write index.
    window_idx: u8,
    /// Number of valid elements currently in window.
    window_len: u8,
}

impl EdbcEngine {
    /// Constructs a new `EdbcEngine` with specified initial state configuration.
    #[inline]
    pub fn new(mut initial_state: gcso_edbc_state_t) -> Self {
        if initial_state.bifurcation_threshold <= 0.0
            || initial_state.bifurcation_threshold.is_nan()
            || initial_state.bifurcation_threshold.is_infinite()
        {
            initial_state.bifurcation_threshold = DEFAULT_BIFURCATION_THRESHOLD;
        }
        if initial_state.singularity_eps <= 0.0
            || initial_state.singularity_eps.is_nan()
            || initial_state.singularity_eps.is_infinite()
        {
            initial_state.singularity_eps = DEFAULT_SINGULARITY_EPS;
        }
        if initial_state.sample_temperature <= 0.0
            || initial_state.sample_temperature.is_nan()
            || initial_state.sample_temperature.is_infinite()
        {
            initial_state.sample_temperature = DEFAULT_SAMPLE_TEMPERATURE;
        }
        Self {
            state: initial_state,
            is_active: true,
            hysteresis_band: DEFAULT_HYSTERESIS_BAND,
            tau_flux: DEFAULT_TAU_FLUX,
            window_buf: [0.0; 8],
            window_idx: 0,
            window_len: 0,
        }
    }

    /// Returns the current active state descriptor snapshot.
    #[inline(always)]
    pub fn state(&self) -> gcso_edbc_state_t {
        self.state
    }

    /// Returns `true` if controller instance is currently active.
    #[inline(always)]
    pub fn is_active(&self) -> bool {
        self.is_active
    }

    /// Sets the active lifecycle status flag.
    #[inline(always)]
    pub fn set_active(&mut self, active: bool) {
        self.is_active = active;
    }

    /// Returns the active decoding branch mode (0 = Normal, 1 = Bifurcation, 2 = Repulsion).
    #[inline(always)]
    pub fn active_branch_mode(&self) -> u32 {
        self.state.active_branch_mode
    }

    /// Returns `true` if operating in Normal Entropy Convergence Mode (Mode 0).
    #[inline(always)]
    pub fn is_normal_mode(&self) -> bool {
        self.state.active_branch_mode == EDBC_MODE_NORMAL_ENTROPY
    }

    /// Returns `true` if operating in Pitchfork Bifurcation Exploration Mode (Mode 1).
    #[inline(always)]
    pub fn is_bifurcation_mode(&self) -> bool {
        self.state.active_branch_mode == EDBC_MODE_PITCHFORK_BIFURCATION
    }

    /// Returns `true` if operating in Phase-Conjugate Repulsion Mode (Mode 2).
    #[inline(always)]
    pub fn is_repulsion_mode(&self) -> bool {
        self.state.active_branch_mode == EDBC_MODE_PHASE_CONJUGATE_REPULSION
    }

    /// Returns the current moving Z-score entropy value ($\tilde{H}$).
    #[inline(always)]
    pub fn moving_z_entropy(&self) -> f32 {
        self.state.moving_z_entropy
    }

    /// Returns the current sliding entropy rate ($\Phi_M(t)$).
    #[inline(always)]
    pub fn sliding_entropy_rate(&self) -> f32 {
        self.state.sliding_entropy_rate
    }

    /// Returns the active sample temperature modifier.
    #[inline(always)]
    pub fn sample_temperature(&self) -> f32 {
        self.state.sample_temperature
    }

    /// Returns the active phase repulsion gain factor.
    #[inline(always)]
    pub fn repulsion_gain(&self) -> f32 {
        self.state.repulsion_gain
    }

    /// Sets the bifurcation threshold parameter dynamically.
    #[inline]
    pub fn set_bifurcation_threshold(&mut self, threshold: f32) {
        if threshold > 0.0 && !threshold.is_nan() && !threshold.is_infinite() {
            self.state.bifurcation_threshold = threshold;
        }
    }

    /// Sets the hysteresis band width parameter dynamically.
    #[inline]
    pub fn set_hysteresis_band(&mut self, band: f32) {
        if band >= 0.0 && !band.is_nan() && !band.is_infinite() {
            self.hysteresis_band = band;
        }
    }

    /// Sets the tau_flux threshold parameter dynamically.
    #[inline]
    pub fn set_tau_flux(&mut self, tau_flux: f32) {
        if tau_flux >= 0.0 && !tau_flux.is_nan() && !tau_flux.is_infinite() {
            self.tau_flux = tau_flux;
        }
    }

    /// Sets the singularity epsilon guard parameter dynamically.
    #[inline]
    pub fn set_singularity_eps(&mut self, eps: f32) {
        if eps > 0.0 && !eps.is_nan() && !eps.is_infinite() {
            self.state.singularity_eps = eps;
        }
    }

    /// Sets the sampling temperature modifier dynamically.
    #[inline]
    pub fn set_sample_temperature(&mut self, temp: f32) {
        if temp > 0.0 && !temp.is_nan() && !temp.is_infinite() {
            self.state.sample_temperature = temp;
        }
    }

    /// Resets transient entropy tracking accumulators and branch modes in-place.
    #[inline]
    pub fn reset(&mut self) {
        self.state.moving_z_entropy = 0.0;
        self.state.sliding_entropy_rate = 0.0;
        self.state.repulsion_gain = 0.0;
        self.state.sample_temperature = DEFAULT_SAMPLE_TEMPERATURE;
        self.state.active_branch_mode = EDBC_MODE_NORMAL_ENTROPY;
        self.window_buf = [0.0; 8];
        self.window_idx = 0;
        self.window_len = 0;
    }
}

impl Default for EdbcEngine {
    #[inline]
    fn default() -> Self {
        Self::new(gcso_edbc_state_t::default())
    }
}

impl EntropyEvaluator for EdbcEngine {
    /// Evaluates token Z-score activation entropy and updates dynamic pitchfork branch mode.
    ///
    /// Mode Mapping conforming strictly to c_abi_spec.md Section 3.1 & 5.4.G:
    /// - Mode 0 (`EDBC_MODE_NORMAL_ENTROPY`): Standard Convergence Mode
    /// - Mode 1 (`EDBC_MODE_PITCHFORK_BIFURCATION`): Pitchfork Exploration Mode
    /// - Mode 2 (`EDBC_MODE_PHASE_CONJUGATE_REPULSION`): Phase-Conjugate Repulsion Mode
    ///
    /// Incorporates Sliding-Window Entropy Rate Integrator ($\Phi_M(t) = \tilde{H}(t) - \tilde{H}(t-M)$, $M=5$)
    /// and Hysteresis Band ($\delta_{\mathrm{hyst}}$) anti-chattering stabilization per GCSO Whitepaper Section 3.2.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_EDBC_SINGULARITY` if `token_z_score` contains `NaN` or `Inf`.
    #[inline]
    fn eval_stateful(
        &mut self,
        token_z_score: f32,
        state_out: &mut gcso_edbc_state_t,
    ) -> GcsoResult<()> {
        if token_z_score.is_nan() || token_z_score.is_infinite() {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        let token_z_safe = if token_z_score < 0.0 {
            0.0
        } else {
            token_z_score
        };
        let prev_entropy = self.state.moving_z_entropy;
        let new_entropy =
            (1.0 - EMA_ALPHA_ENTROPY) * prev_entropy + EMA_ALPHA_ENTROPY * token_z_safe;

        // Push new entropy to sliding window buffer
        let idx = (self.window_idx as usize) & 7;
        self.window_buf[idx] = new_entropy;
        self.window_idx = self.window_idx.wrapping_add(1);
        if self.window_len < 8 {
            self.window_len += 1;
        }

        // Calculate Sliding-Window Entropy Rate Phi_M(t) = H~(t) - H~(t-M) with M=5
        let window_m = 5usize.min(self.window_len as usize);
        let old_idx = (self.window_idx.wrapping_sub(window_m as u8) as usize) & 7;
        let old_entropy = self.window_buf[old_idx];
        let sliding_rate = new_entropy - old_entropy;

        self.state.sliding_entropy_rate = sliding_rate;
        self.state.moving_z_entropy = new_entropy;

        let tau = self.state.bifurcation_threshold;
        let tau_flux = self.tau_flux;
        let delta = self.hysteresis_band;
        let current_mode = self.state.active_branch_mode;

        // Apply hysteresis state transitions with asymmetric anti-chattering thresholds
        let next_mode = match current_mode {
            EDBC_MODE_NORMAL_ENTROPY => {
                if new_entropy > tau * 1.5 + delta && sliding_rate > tau_flux + delta {
                    EDBC_MODE_PHASE_CONJUGATE_REPULSION
                } else if new_entropy > tau + delta {
                    EDBC_MODE_PITCHFORK_BIFURCATION
                } else {
                    EDBC_MODE_NORMAL_ENTROPY
                }
            }
            EDBC_MODE_PITCHFORK_BIFURCATION => {
                if new_entropy > tau * 1.5 + delta && sliding_rate > tau_flux + delta {
                    EDBC_MODE_PHASE_CONJUGATE_REPULSION
                } else if new_entropy < tau - delta {
                    EDBC_MODE_NORMAL_ENTROPY
                } else {
                    EDBC_MODE_PITCHFORK_BIFURCATION
                }
            }
            EDBC_MODE_PHASE_CONJUGATE_REPULSION => {
                if new_entropy < tau - delta {
                    EDBC_MODE_NORMAL_ENTROPY
                } else if new_entropy < tau * 1.5 - delta || sliding_rate < tau_flux - delta {
                    EDBC_MODE_PITCHFORK_BIFURCATION
                } else {
                    EDBC_MODE_PHASE_CONJUGATE_REPULSION
                }
            }
            _ => EDBC_MODE_NORMAL_ENTROPY,
        };

        self.state.active_branch_mode = next_mode;

        match next_mode {
            EDBC_MODE_PHASE_CONJUGATE_REPULSION => {
                self.state.sample_temperature = 1.5;
                self.state.repulsion_gain = 1.0;
            }
            EDBC_MODE_PITCHFORK_BIFURCATION => {
                self.state.sample_temperature = 1.2;
                self.state.repulsion_gain = 0.5;
            }
            _ => {
                self.state.sample_temperature = 1.0;
                self.state.repulsion_gain = 0.0;
            }
        }

        *state_out = self.state;
        Ok(())
    }

    /// Computes CVoid Coherent Vector Alignment Metric for Out-of-Distribution Latent Space.
    ///
    /// Evaluates $L_2$ norm squared over vector length ($d$) to score manifold alignment.
    /// Utilizes 4 independent accumulators over 8-element SIMD chunks to maximize ILP
    /// (Instruction-Level Parallelism) and FMA pipeline utilization without dynamic allocation.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `key_vector` is empty.
    /// Returns `GCSO_ERROR_EDBC_SINGULARITY` if numerical sum-of-squares is NaN/Inf.
    #[inline]
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

        let mut sum_sq0 = 0.0f32;
        let mut sum_sq1 = 0.0f32;
        let mut sum_sq2 = 0.0f32;
        let mut sum_sq3 = 0.0f32;

        // Process 8 floats per SIMD vector lane (32 bytes) with 4 independent accumulators for ILP
        let chunks = slice.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            sum_sq0 += chunk[0] * chunk[0];
            sum_sq1 += chunk[1] * chunk[1];
            sum_sq2 += chunk[2] * chunk[2];
            sum_sq3 += chunk[3] * chunk[3];
            sum_sq0 += chunk[4] * chunk[4];
            sum_sq1 += chunk[5] * chunk[5];
            sum_sq2 += chunk[6] * chunk[6];
            sum_sq3 += chunk[7] * chunk[7];
        }

        let mut sum_sq = (sum_sq0 + sum_sq1) + (sum_sq2 + sum_sq3);

        for &v in remainder {
            sum_sq += v * v;
        }

        if sum_sq.is_nan() || sum_sq.is_infinite() {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        *void_score_out = sum_sq / (len as f32);
        Ok(())
    }

    /// Calculates Eyring-Kramers potential barrier height: $\Delta V = \tau_{\mathrm{eff}} / (\mathcal{C}_{\mathrm{void}} + \epsilon)$.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_EDBC_SINGULARITY` if inputs contain `NaN`/`Inf`, $\tau_{\mathrm{eff}} \le 0.0$,
    /// or if the resulting barrier height is invalid.
    #[inline]
    fn eval_barrier(&self, void_score: f32, tau_eff: f32) -> GcsoResult<f32> {
        if void_score.is_nan()
            || void_score.is_infinite()
            || tau_eff.is_nan()
            || tau_eff.is_infinite()
            || tau_eff <= 0.0
        {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        let eps = if self.state.singularity_eps > 0.0
            && !self.state.singularity_eps.is_nan()
            && !self.state.singularity_eps.is_infinite()
        {
            self.state.singularity_eps
        } else {
            DEFAULT_SINGULARITY_EPS
        };

        let safe_void = if void_score < 0.0 { 0.0 } else { void_score };
        let barrier = tau_eff / (safe_void + eps);

        if barrier.is_nan() || barrier.is_infinite() {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        Ok(barrier)
    }

    /// Calculates scale-invariant Pitchfork Bifurcation potential barrier height ($\Delta V$)
    /// according to GCSO Whitepaper Lemma 3.1.
    #[inline]
    fn eval_barrier_pitchfork(
        &self,
        cvoid: f32,
        tau_eff: f32,
        alpha: f32,
        beta: f32,
        head_dim: u32,
    ) -> GcsoResult<f32> {
        eval_barrier_pitchfork(cvoid, tau_eff, alpha, beta, head_dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::gcso_edbc_state_t;

    #[test]
    fn test_compute_sparse_surrogate_entropy() {
        let entropy = compute_sparse_surrogate_entropy(0.5, 0.5, 1e-12);
        assert!((entropy - LN_2).abs() < 1e-5);

        let zero_entropy = compute_sparse_surrogate_entropy(1.0, 0.0, 1e-12);
        assert!(zero_entropy < 1e-4);

        let invalid_entropy = compute_sparse_surrogate_entropy(-0.5, 0.5, 1e-12);
        assert_eq!(invalid_entropy, 0.0);
    }

    #[test]
    fn test_compute_moving_z_score_entropy() {
        let h_tilde = compute_moving_z_score_entropy(0.5, 0.8, 0.2, 0.1, 1e-6, 1.0, 0.5);
        assert!(h_tilde > 0.5);

        let h_zero = compute_moving_z_score_entropy(0.0, 0.8, 0.2, 0.1, 1e-6, 1.0, 0.5);
        assert_eq!(h_zero, 0.0);
    }

    #[test]
    fn test_edbc_engine_init_defaults() {
        let state = gcso_edbc_state_t::default();
        let engine = EdbcEngine::new(state);

        assert_eq!(engine.state().bifurcation_threshold, 1.5);
        assert_eq!(engine.state().singularity_eps, 1e-12);
        assert_eq!(engine.state().sample_temperature, 1.0);
        assert_eq!(engine.hysteresis_band, DEFAULT_HYSTERESIS_BAND);
        assert_eq!(engine.tau_flux, DEFAULT_TAU_FLUX);
        assert!(engine.is_active());
        assert_eq!(engine.active_branch_mode(), EDBC_MODE_NORMAL_ENTROPY);
        assert!(engine.is_normal_mode());
        assert!(!engine.is_bifurcation_mode());
        assert!(!engine.is_repulsion_mode());
    }

    #[test]
    fn test_edbc_engine_setters() {
        let mut engine = EdbcEngine::default();
        engine.set_active(false);
        assert!(!engine.is_active());

        engine.set_bifurcation_threshold(2.0);
        assert_eq!(engine.state().bifurcation_threshold, 2.0);

        engine.set_hysteresis_band(0.1);
        assert_eq!(engine.hysteresis_band, 0.1);

        engine.set_tau_flux(0.6);
        assert_eq!(engine.tau_flux, 0.6);

        engine.set_singularity_eps(1e-8);
        assert_eq!(engine.state().singularity_eps, 1e-8);

        engine.set_sample_temperature(1.3);
        assert_eq!(engine.state().sample_temperature, 1.3);

        // Verify bounds checking on invalid inputs
        engine.set_bifurcation_threshold(-1.0);
        assert_eq!(engine.state().bifurcation_threshold, 2.0);
    }

    #[test]
    fn test_edbc_engine_reset() {
        let mut engine = EdbcEngine::default();
        engine.state.moving_z_entropy = 2.5;
        engine.state.sliding_entropy_rate = 0.8;
        engine.state.repulsion_gain = 1.0;
        engine.state.sample_temperature = 1.5;
        engine.state.active_branch_mode = EDBC_MODE_PHASE_CONJUGATE_REPULSION;

        engine.reset();

        assert_eq!(engine.moving_z_entropy(), 0.0);
        assert_eq!(engine.sliding_entropy_rate(), 0.0);
        assert_eq!(engine.active_branch_mode(), EDBC_MODE_NORMAL_ENTROPY);
        assert!(engine.is_normal_mode());
        assert_eq!(engine.sample_temperature(), 1.0);
        assert_eq!(engine.repulsion_gain(), 0.0);
    }

    #[test]
    fn test_eval_stateful_branch_modes_and_parameters() {
        let mut engine = EdbcEngine::default();
        engine.set_bifurcation_threshold(1.0);
        engine.set_hysteresis_band(0.05);
        let mut out_state = gcso_edbc_state_t::default();

        // Step 1: Low entropy -> Normal Mode
        engine.eval_stateful(0.5, &mut out_state).unwrap();
        assert_eq!(out_state.active_branch_mode, EDBC_MODE_NORMAL_ENTROPY);
        assert!(engine.is_normal_mode());
        assert_eq!(out_state.sample_temperature, 1.0);
        assert_eq!(out_state.repulsion_gain, 0.0);

        // Step 2: High entropy spike -> Pitchfork Bifurcation Mode
        for _ in 0..15 {
            engine.eval_stateful(12.0, &mut out_state).unwrap();
        }
        assert!(out_state.moving_z_entropy > 1.0);
        assert!(out_state.active_branch_mode >= EDBC_MODE_PITCHFORK_BIFURCATION);
        assert!(out_state.sample_temperature > 1.0);
    }

    #[test]
    fn test_eval_stateful_repulsion_mode() {
        let mut engine = EdbcEngine::default();
        engine.set_bifurcation_threshold(1.0);
        engine.set_hysteresis_band(0.05);
        engine.set_tau_flux(0.5);
        let mut out_state = gcso_edbc_state_t::default();

        // Induce high entropy and sudden rate spike
        engine.state.moving_z_entropy = 2.0;
        engine.eval_stateful(10.0, &mut out_state).unwrap();

        assert_eq!(
            out_state.active_branch_mode,
            EDBC_MODE_PHASE_CONJUGATE_REPULSION
        );
        assert!(engine.is_repulsion_mode());
        assert_eq!(out_state.sample_temperature, 1.5);
        assert_eq!(out_state.repulsion_gain, 1.0);
    }

    #[test]
    fn test_eval_stateful_invalid_input() {
        let mut engine = EdbcEngine::default();
        let mut out_state = gcso_edbc_state_t::default();

        assert_eq!(
            engine.eval_stateful(f32::NAN, &mut out_state),
            Err(GCSO_ERROR_EDBC_SINGULARITY)
        );
        assert_eq!(
            engine.eval_stateful(f32::INFINITY, &mut out_state),
            Err(GCSO_ERROR_EDBC_SINGULARITY)
        );
    }

    #[test]
    fn test_eval_cvoid_dyadic128_valid() {
        let engine = EdbcEngine::default();
        #[repr(align(32))]
        struct AlignedBuffer([f32; 32]);
        let buf = AlignedBuffer([2.0; 32]);
        let slice = AlignedSlice32::new(&buf.0).unwrap();

        let mut score = 0.0f32;
        engine.eval_cvoid_dyadic128(&slice, &mut score).unwrap();
        assert_eq!(score, 4.0);
    }

    #[test]
    fn test_eval_cvoid_dyadic128_remainder_handling() {
        let engine = EdbcEngine::default();
        #[repr(align(32))]
        struct AlignedBuffer([f32; 35]);
        let mut array = [0.0f32; 35];
        array[..32].copy_from_slice(&[2.0f32; 32]);
        array[32..].copy_from_slice(&[3.0f32; 3]);
        let buf = AlignedBuffer(array);
        let slice = AlignedSlice32::new(&buf.0).unwrap();

        let mut score = 0.0f32;
        engine.eval_cvoid_dyadic128(&slice, &mut score).unwrap();
        let expected = (32.0 * 4.0 + 3.0 * 9.0) / 35.0;
        assert!((score - expected).abs() < 1e-5);
    }

    #[test]
    fn test_eval_barrier_and_singularity() {
        let engine = EdbcEngine::default();

        let barrier = engine.eval_barrier(2.0, 1.0).unwrap();
        assert!((barrier - (1.0 / (2.0 + 1e-12))).abs() < 1e-5);

        // Barrier with zero void score (clamped to eps)
        let barrier_zero = engine.eval_barrier(0.0, 1.0).unwrap();
        assert!((barrier_zero - (1.0 / 1e-12)).abs() < 1e-3);

        // Singularity check with negative tau_eff
        let err = engine.eval_barrier(2.0, -1.0);
        assert_eq!(err, Err(GCSO_ERROR_EDBC_SINGULARITY));

        // Singularity check with NaN
        let err_nan = engine.eval_barrier(f32::NAN, 1.0);
        assert_eq!(err_nan, Err(GCSO_ERROR_EDBC_SINGULARITY));
    }

    #[test]
    fn test_eval_barrier_pitchfork() {
        // Exploration mode (cvoid > tau_eff) -> Valid barrier calculation
        let barrier = eval_barrier_pitchfork(2.0, 1.0, 1.0, 1.0, 128).unwrap();
        assert!((barrier - (1.0 / 512.0)).abs() < 1e-6);

        // Convergence mode (cvoid <= tau_eff) -> Potential barrier is 0.0
        let barrier_conv = eval_barrier_pitchfork(0.5, 1.0, 1.0, 1.0, 128).unwrap();
        assert_eq!(barrier_conv, 0.0);

        // Invalid beta parameter -> Err(GCSO_ERROR_EDBC_SINGULARITY)
        assert_eq!(
            eval_barrier_pitchfork(2.0, 1.0, 1.0, -1.0, 128),
            Err(GCSO_ERROR_EDBC_SINGULARITY)
        );
    }

    #[test]
    fn test_ln_and_tanh_f32_no_std_accuracy() {
        let ln_e = ln_f32(core::f32::consts::E);
        assert!((ln_e - 1.0).abs() < 1e-5);

        let tanh_zero = tanh_f32(0.0);
        assert_eq!(tanh_zero, 0.0);

        let tanh_large = tanh_f32(5.0);
        assert_eq!(tanh_large, 1.0);
    }
}
