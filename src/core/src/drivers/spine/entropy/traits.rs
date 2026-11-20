// name: src/core/src/drivers/spine/entropy/traits.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Entropy evaluation and pitchfork branch control traits (`EntropyEvaluator`, `DecodingGate`).

use core::f32::consts::{FRAC_PI_2, LN_2};

use crate::drivers::spine::entropy::gcso_edbc_state_t;
use crate::foundation::{
    branchless_clamp_f32, branchless_eyring_kramers_escape, branchless_flush_subnormal_f32,
    branchless_inv_pos_f32, branchless_is_finite_f32, branchless_is_nan_f32, branchless_max_f32,
    branchless_min_f32, branchless_popcount_rsqrt_u32, branchless_q7_add_sat, branchless_rad_to_q7,
    branchless_relu_f32, branchless_rsqrt_f32, branchless_sanitize_z_score_f32,
    branchless_splitmix64, branchless_sqrt_f32, branchless_tanh_f32, dot_and_norms_f32_unrolled4,
    fast_inv_len_f32, fast_inv_pos_normal_f32, gcso_tagged_ptr_t, load_q7_word_at,
    store_q7_word_at, sum_abs_and_sq_f32_simd8, swar_q7_add_sat_x8, AlignedSlice32,
    DEFAULT_MAX_VALID_Z_SCORE,
};
use crate::liminal::{
    branchless_select_f32, branchless_select_q7, branchless_select_u32, gcso_q7_t, GcsoResult,
    Liminal, DEFAULT_ENTROPY_EPS, GCSO_ERROR_EDBC_SINGULARITY, GCSO_ERROR_INVALID_ARGUMENT,
    MAX_ATTENTION_HEADS,
};
use crate::stigmergy::trail::StigmergicActionPtrOps;

/// Cold-path trait for entropy-driven branch evaluation and bifurcation tracking (**Macro Level**).
pub trait EntropyEvaluator: Liminal + Send + Sync {
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
    fn eval_sparse_surrogate_entropy(&self, p_max: f32, p_2nd: f32, _eps: f32) -> GcsoResult<f32> {
        if p_max <= 0.0 || branchless_is_nan_f32(p_max) || branchless_is_nan_f32(p_2nd) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        const C1: f32 = 0.5;
        const C2: f32 = 1.0 / 12.0;
        const C3: f32 = 1.0 / 30.0;
        const C4: f32 = 1.0 / 56.0;
        const C5: f32 = 1.0 / 90.0;
        const C6: f32 = 1.0 / 132.0;
        const C7: f32 = 1.0 / 182.0;
        const C8: f32 = 1.0 / 240.0;
        const C9_TAIL: f32 = LN_2 - (C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8);

        let p_2nd_clean = branchless_relu_f32(p_2nd);
        let raw_sum = p_max + p_2nd_clean;
        let raw_diff = p_max - p_2nd_clean;
        let valid_sum = (p_2nd_clean > 1.0e-30) & (raw_sum < f32::INFINITY);
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
        let raw_h = LN_2 - t * poly;
        Ok(branchless_select_f32(
            valid_sum & (t < 0.999_999_9),
            raw_h,
            0.0,
        ))
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

    /// Computes the Liminal Vector Alignment Metric ($\mathcal{L}_{\mathrm{void}} \in [0, 1]$)
    /// in division-free (`0 fdiv`) SIMD mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `key_vector` is empty.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if any element or accumulated norm of `key_vector` is non-finite.
    #[inline]
    fn eval_liminal_void_dyadic128(
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

    /// Calculates the Eyring-Kramers potential barrier height ($\Delta V = \tau_{\mathrm{eff}} / \mathcal{L}_{\mathrm{void}}$)
    /// in division-free (`0 fdiv`) mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if `void_score <= 1.0e-12`, `tau_eff < 0.0`, or either is non-finite.
    #[inline]
    fn eval_barrier(&self, void_score: f32, tau_eff: f32) -> GcsoResult<f32> {
        if !branchless_is_finite_f32(void_score)
            || !branchless_is_finite_f32(tau_eff)
            || void_score <= DEFAULT_ENTROPY_EPS
            || tau_eff < 0.0
        {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }
        Ok(branchless_flush_subnormal_f32(
            tau_eff * branchless_inv_pos_f32(void_score),
        ))
    }

    /// Evaluates the scale-invariant Pitchfork Bifurcation potential barrier height
    /// $\Delta V = \frac{\alpha^2 (\mathcal{L}_{\mathrm{void}} - \tau_{\mathrm{eff}})^2}{4 \beta d_{\mathrm{head}}}$
    /// in division-free (`0 fdiv`) mode (Lemma 3.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if any parameter is non-finite, `beta <= 0.0`,
    /// or `head_dim` is zero or odd.
    #[inline]
    fn eval_barrier_pitchfork(
        &self,
        liminal_void: f32,
        tau_eff: f32,
        alpha: f32,
        beta: f32,
        head_dim: u32,
    ) -> GcsoResult<f32> {
        if !branchless_is_finite_f32(liminal_void)
            || !branchless_is_finite_f32(tau_eff)
            || !branchless_is_finite_f32(alpha)
            || !branchless_is_finite_f32(beta)
            || beta <= 0.0
            || head_dim == 0
            || (head_dim & 1) != 0
        {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }
        let excess = branchless_relu_f32(liminal_void - tau_eff);
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
        liminal_void: f32,
        tau_eff: f32,
        alpha: f32,
        beta: f32,
        d0: f32,
    ) -> GcsoResult<f32> {
        if !branchless_is_finite_f32(liminal_void)
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
            liminal_void,
            tau_eff,
            alpha,
            beta,
            d0,
        ))
    }
}

/// Cold-path trait for stateful `EntropyController` parameters and hysteresis management (**Macro Level**).
pub trait DecodingGate: EntropyEvaluator {
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

    /// Branchlessly evaluates the next `EntropyController` branch mode (`0`, `1`, `2`).
    #[inline(always)]
    #[must_use]
    fn evaluate_branch_mode_transition(
        &self,
        current_mode: u32,
        moving_z_entropy: f32,
        sliding_flux: f32,
        void_score: f32,
        bifurcation_threshold: f32,
        tau_flux: f32,
        hysteresis_band: f32,
    ) -> u32 {
        let hyst = branchless_relu_f32(hysteresis_band);
        let upper_thresh = bifurcation_threshold + hyst;
        let lower_thresh = branchless_relu_f32(bifurcation_threshold - hyst);

        let surge_triggered = (moving_z_entropy > upper_thresh)
            | (sliding_flux > (tau_flux + hyst))
            | ((moving_z_entropy > bifurcation_threshold) & (void_score >= upper_thresh));
        let stay_active = (current_mode != 0) & (moving_z_entropy > lower_thresh);
        let active_non_normal = surge_triggered | stay_active;

        let active_target = branchless_select_u32(void_score >= bifurcation_threshold, 1, 2);
        branchless_select_u32(active_non_normal, active_target, 0)
    }

    /// Evaluates the next `EntropyController` branch mode (`0`, `1`, `2`) and evolves `tagged_ptr` in $\mathcal{O}(1)$.
    #[inline(always)]
    #[must_use]
    fn evaluate_branch_and_evolve_pointer(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        trail_density: f32,
        dominant_attractor_slot: u8,
        moving_z_entropy: f32,
        sliding_flux: f32,
        void_score: f32,
        bifurcation_threshold: f32,
        tau_flux: f32,
        hysteresis_band: f32,
        high_density_rank1_thresh: f32,
    ) -> (u32, gcso_tagged_ptr_t) {
        let next_mode = self.evaluate_branch_mode_transition(
            self.active_branch_mode(),
            moving_z_entropy,
            sliding_flux,
            void_score,
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
            high_density_rank1_thresh,
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
        void_score: f32,
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
        let norm_h = branchless_clamp_f32(
            branchless_relu_f32(moving_z_entropy) * (1.0 / LN_2),
            0.0,
            1.0,
        );
        if mode == 2 {
            return branchless_clamp_f32(safe_base * (1.0 + 0.25 * norm_h), 0.05, 5.0);
        }
        let escape_p = branchless_eyring_kramers_escape(void_score, tau_eff, alpha, beta, d0);
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
        void_score: f32,
        tau_eff: f32,
        alpha: f32,
        beta: f32,
        head_dim: u32,
        explore_head_start: usize,
    ) -> GcsoResult<f32> {
        if explore_head_start > MAX_ATTENTION_HEADS {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !branchless_is_finite_f32(void_score)
            || !branchless_is_finite_f32(tau_eff)
            || !branchless_is_finite_f32(alpha)
            || !branchless_is_finite_f32(beta)
            || beta <= 0.0
            || head_dim == 0
            || (head_dim & 1) != 0
        {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        let excess = branchless_relu_f32(void_score - tau_eff);
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
