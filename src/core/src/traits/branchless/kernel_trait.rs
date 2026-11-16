// name: src/core/src/traits/branchless/kernel_trait.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `BranchlessKernel` blanket trait exposing branchless arithmetic and phase primitives.

use super::phase_field::*;
use super::*;

/// Blanket trait providing branchless bitwise multiplexers, `#![no_std]` transcendentals,
/// division-free (`0 fdiv`) reciprocals, and 4/8-lane SWAR Q7 & `u8` arithmetic to all engine types.
pub trait BranchlessKernel {
    /// Branchless bitwise selection for `u8`.
    #[inline(always)]
    fn select_u8(&self, cond: bool, if_true: u8, if_false: u8) -> u8 {
        branchless_select_u8(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `i8`.
    #[inline(always)]
    fn select_i8(&self, cond: bool, if_true: i8, if_false: i8) -> i8 {
        branchless_select_i8(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `u16`.
    #[inline(always)]
    fn select_u16(&self, cond: bool, if_true: u16, if_false: u16) -> u16 {
        branchless_select_u16(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `u32`.
    #[inline(always)]
    fn select_u32(&self, cond: bool, if_true: u32, if_false: u32) -> u32 {
        branchless_select_u32(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `u64`.
    #[inline(always)]
    fn select_u64(&self, cond: bool, if_true: u64, if_false: u64) -> u64 {
        branchless_select_u64(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `usize`.
    #[inline(always)]
    fn select_usize(&self, cond: bool, if_true: usize, if_false: usize) -> usize {
        branchless_select_usize(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `i32`.
    #[inline(always)]
    fn select_i32(&self, cond: bool, if_true: i32, if_false: i32) -> i32 {
        branchless_select_i32(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `i64`.
    #[inline(always)]
    fn select_i64(&self, cond: bool, if_true: i64, if_false: i64) -> i64 {
        branchless_select_i64(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `gcso_q7_t`.
    #[inline(always)]
    fn select_q7(&self, cond: bool, if_true: gcso_q7_t, if_false: gcso_q7_t) -> gcso_q7_t {
        branchless_select_q7(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `f32`.
    #[inline(always)]
    fn select_f32(&self, cond: bool, if_true: f32, if_false: f32) -> f32 {
        branchless_select_f32(cond, if_true, if_false)
    }
    /// Branchless finiteness check for `f32`.
    #[inline(always)]
    fn is_finite_f32(&self, x: f32) -> bool {
        branchless_is_finite_f32(x)
    }
    /// Branchlessly flushes IEEE-754 subnormals and `-0.0` to `+0.0`.
    #[inline(always)]
    fn flush_subnormal_f32(&self, x: f32) -> f32 {
        branchless_flush_subnormal_f32(x)
    }
    /// Branchless IEEE-754 absolute value (`|x|`).
    #[inline(always)]
    fn abs_f32(&self, x: f32) -> f32 {
        branchless_abs_f32(x)
    }
    /// Branchless ReLU ($\max(0.0, x)$).
    #[inline(always)]
    fn relu_f32(&self, x: f32) -> f32 {
        branchless_relu_f32(x)
    }
    /// Branchless interval clamp.
    #[inline(always)]
    fn clamp_f32(&self, x: f32, lo: f32, hi: f32) -> f32 {
        branchless_clamp_f32(x, lo, hi)
    }
    /// Branchless Z-score sanitizer.
    #[inline(always)]
    fn sanitize_z_score_f32(&self, z_score: f32, max_ceiling: f32) -> f32 {
        branchless_sanitize_z_score_f32(z_score, max_ceiling)
    }
    /// Division-free (`0 fdiv`) branchless positive reciprocal ($1 / x$).
    #[inline(always)]
    fn inv_pos_f32(&self, val: f32) -> f32 {
        branchless_inv_pos_f32(val)
    }
    /// Branchless hyperbolic tangent ($\tanh(x)$).
    #[inline(always)]
    fn tanh_f32(&self, x: f32) -> f32 {
        branchless_tanh_f32(x)
    }
    /// Branchless logistic sigmoid ($\sigma(x)$).
    #[inline(always)]
    fn sigmoid_f32(&self, x: f32) -> f32 {
        branchless_sigmoid_f32(x)
    }
    /// Branchless RIPA soft-bounded phase clamp.
    #[inline(always)]
    fn ripa_clamp_rad(&self, delta_rad: f32, max_rad: f32) -> f32 {
        branchless_ripa_clamp_rad(delta_rad, max_rad)
    }
    /// Branchless natural logarithm ($\ln(x)$).
    #[inline(always)]
    fn ln_f32(&self, val: f32) -> f32 {
        branchless_ln_f32(val)
    }
    /// Branchless fast exponential ($e^x$).
    #[inline(always)]
    fn exp_f32(&self, x: f32) -> f32 {
        branchless_exp_f32(x)
    }
    /// Division-free (`0 fdiv`) branchless square root ($\sqrt{x}$).
    #[inline(always)]
    fn sqrt_f32(&self, val: f32) -> f32 {
        branchless_sqrt_f32(val)
    }
    /// Division-free (`0 fdiv`) branchless reciprocal square root ($1 / \sqrt{x}$).
    #[inline(always)]
    fn rsqrt_f32(&self, val: f32) -> f32 {
        branchless_rsqrt_f32(val)
    }
    /// Branchless fused $(\sin\theta, \cos\theta)$ for arbitrary finite angles.
    #[inline(always)]
    fn sincos_f32(&self, theta_rad: f32) -> (f32, f32) {
        branchless_sincos_f32(theta_rad)
    }
    /// Branchless direct $(\sin\theta, \cos\theta)$ for angles bounded in $[-\pi/2, \pi/2]$.
    #[inline(always)]
    fn sincos_half_pi_f32(&self, theta_rad: f32) -> (f32, f32) {
        branchless_sincos_half_pi_f32(theta_rad)
    }
    /// Division-free RIPA soft-clamped $(\sin\theta_{\mathrm{safe}}, \cos\theta_{\mathrm{safe}})$ from Q7.
    #[inline(always)]
    fn sincos_q7_ripa(&self, phase_q7: gcso_q7_t, max_rad: f32) -> (f32, f32) {
        branchless_sincos_q7_ripa(phase_q7, max_rad)
    }
    /// $\mathcal{O}(1)$ 256-entry LUT RIPA $(\sin\theta_{\mathrm{safe}}, \cos\theta_{\mathrm{safe}})$ under `DEFAULT_RIPA_CLAMP_MAX_RAD`.
    #[inline(always)]
    fn sincos_q7_ripa_default_lut(&self, phase_q7: gcso_q7_t) -> (f32, f32) {
        branchless_sincos_q7_ripa_default_lut(phase_q7)
    }
    /// $\mathcal{O}(1)$ 256-entry LUT combining $S^1$ geodesic folding and RIPA clamping.
    #[inline(always)]
    fn sincos_q7_geodesic_ripa_default_lut(&self, phase_q7: gcso_q7_t) -> (f32, f32) {
        branchless_sincos_q7_geodesic_ripa_default_lut(phase_q7)
    }
    /// Pre-inverted RIPA $(\sin\theta_{\mathrm{safe}}, \cos\theta_{\mathrm{safe}})$ from Q7.
    #[inline(always)]
    fn sincos_q7_ripa_preinv(
        &self,
        phase_q7: gcso_q7_t,
        max_rad: f32,
        q7_to_norm_scale: f32,
    ) -> (f32, f32) {
        branchless_sincos_q7_ripa_preinv(phase_q7, max_rad, q7_to_norm_scale)
    }
    /// Direct 256-entry LUT $(\sin\theta, \cos\theta)$ from Q7.
    #[inline(always)]
    fn sincos_q7(&self, phase_q7: gcso_q7_t) -> (f32, f32) {
        branchless_sincos_q7(phase_q7)
    }
    /// Branchless four-quadrant $\mathrm{atan2}(y, x) \in [-\pi, \pi]$.
    #[inline(always)]
    fn atan2_f32(&self, y: f32, x: f32) -> f32 {
        branchless_atan2_f32(y, x)
    }
    /// Branchless four-quadrant $\mathrm{atan2}(y, x)$ directly quantized into `gcso_q7_t`.
    #[inline(always)]
    fn atan2_to_q7_fast(&self, y: f32, x: f32) -> gcso_q7_t {
        branchless_atan2_to_q7_fast(y, x)
    }
    /// Branchless von Mises concentration parameter $\kappa_c$ from $\bar{R}_c \in [0, 1)$.
    #[inline(always)]
    fn von_mises_kappa(&self, r_bar: f32) -> f32 {
        branchless_von_mises_kappa(r_bar)
    }
    /// Evaluates circular resultant length $\bar{R}_c \in [0, 1]$ from 8 `u64` SWAR words.
    #[inline(always)]
    fn circular_r_bar_words_64(&self, words: &[u64; 8]) -> f32 {
        circular_r_bar_from_words_64(words)
    }
    /// Evaluates both 64-head phase coherence and circular resultant statistics in a single register pass.
    #[inline(always)]
    fn coherence_and_resultant_64(
        &self,
        phases_a: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        phases_b: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> (f32, f32, gcso_q7_t, f32) {
        branchless_q7_coherence_and_resultant_64(phases_a, phases_b)
    }
    /// Evaluates rounded arithmetic mean Q7 phase across 8 `u64` SWAR words.
    #[inline(always)]
    fn q7_mean_i8_words_64(&self, words: &[u64; 8]) -> gcso_q7_t {
        swar_q7_mean_i8_from_words_64(words)
    }
    /// Executes an Online Toroidal $k$-Means M-step centroid update in 8 `u64` SWAR words.
    #[inline(always)]
    fn toroidal_centroid_step_words_64(
        &self,
        centroid_words: &[u64; 8],
        prev_sample_count: u32,
        prev_base_energy: f32,
        prev_r_bar: f32,
        prev_kappa: f32,
        sample_words: &[u64; 8],
        sample_energy: f32,
    ) -> ([u64; 8], f32, f32, f32) {
        branchless_toroidal_centroid_step_words_64(
            centroid_words,
            prev_sample_count,
            prev_base_energy,
            prev_r_bar,
            prev_kappa,
            sample_words,
            sample_energy,
        )
    }
    /// Executes an Online Toroidal $k$-Means M-step centroid update in 8 `u64` SWAR words with an optional precomputed E-step coherence.
    #[inline(always)]
    fn toroidal_centroid_step_words_with_coh_64(
        &self,
        centroid_words: &[u64; 8],
        prev_sample_count: u32,
        prev_base_energy: f32,
        prev_r_bar: f32,
        prev_kappa: f32,
        sample_words: &[u64; 8],
        precomputed_coh_opt: Option<f32>,
        sample_energy: f32,
    ) -> ([u64; 8], f32, f32, f32) {
        branchless_toroidal_centroid_step_words_with_coh_64(
            centroid_words,
            prev_sample_count,
            prev_base_energy,
            prev_r_bar,
            prev_kappa,
            sample_words,
            precomputed_coh_opt,
            sample_energy,
        )
    }
    /// Evaluates Phase-Proximity Gravity $F_{\mathrm{grav}}$ on $SO(2)^{64}$.
    #[inline(always)]
    fn phase_proximity_gravity_f32(
        &self,
        gravity_mass: f32,
        phase_coherence: f32,
        softening_eps: f32,
    ) -> f32 {
        branchless_phase_proximity_gravity_f32(gravity_mass, phase_coherence, softening_eps)
    }
    /// Projects an `&[f32]` feature/embedding slice onto $SO(2)^{64}$ (`[gcso_q7_t; 64]`).
    #[inline(always)]
    fn project_f32_slice_to_q7_64(
        &self,
        values: &[f32],
        gain: f32,
    ) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        branchless_project_f32_slice_to_q7_64(values, gain)
    }
    /// Branchless scale-invariant Eyring-Kramers tunneling escape probability.
    #[inline(always)]
    fn eyring_kramers_escape(&self, cvoid: f32, tau_eff: f32, a: f32, b: f32, d0: f32) -> f32 {
        branchless_eyring_kramers_escape(cvoid, tau_eff, a, b, d0)
    }
    /// Branchless $SO(2)$ 2D rotation operator.
    #[inline(always)]
    fn rotate_so2_pair(&self, q0: f32, q1: f32, sin_t: f32, cos_t: f32) -> (f32, f32) {
        branchless_rotate_so2_pair(q0, q1, sin_t, cos_t)
    }
    /// Branchless composite $SO(2)$ rotation + isotropic scaling operator.
    #[inline(always)]
    fn rotate_so2_pair_scaled(
        &self,
        q0: f32,
        q1: f32,
        sin_t: f32,
        cos_t: f32,
        scale: f32,
    ) -> (f32, f32) {
        branchless_rotate_so2_pair_scaled(q0, q1, sin_t, cos_t, scale)
    }
    /// Converts Q7 phase to radians.
    #[inline(always)]
    fn q7_to_rad(&self, phase_q7: gcso_q7_t) -> f32 {
        branchless_q7_to_rad(phase_q7)
    }
    /// Quantizes radian angle to Q7 format.
    #[inline(always)]
    fn rad_to_q7(&self, rad: f32) -> gcso_q7_t {
        branchless_rad_to_q7(rad)
    }
    /// Quantizes radian QDPS threshold to ceiling `u8` step.
    #[inline(always)]
    fn rad_to_qdps_step_u8(&self, min_step_rad: f32) -> u8 {
        branchless_rad_to_qdps_step_u8(min_step_rad)
    }
    /// Branchless Q7 saturating addition.
    #[inline(always)]
    fn q7_add_sat(&self, a: gcso_q7_t, b: gcso_q7_t) -> gcso_q7_t {
        branchless_q7_add_sat(a, b)
    }
    /// Branchless Q7 saturating subtraction.
    #[inline(always)]
    fn q7_sub_sat(&self, a: gcso_q7_t, b: gcso_q7_t) -> gcso_q7_t {
        branchless_q7_sub_sat(a, b)
    }
    /// Branchless Q7 circular difference on $S^1$.
    #[inline(always)]
    fn q7_circular_diff(&self, a: gcso_q7_t, b: gcso_q7_t) -> gcso_q7_t {
        branchless_q7_circular_diff(a, b)
    }
    /// Branchless Q7 $S^1$ geodesic folding (`[-128, 127] -> [-64, 64]`).
    #[inline(always)]
    fn q7_fold_s1_geodesic(&self, phase_q7: gcso_q7_t) -> gcso_q7_t {
        branchless_q7_fold_s1_geodesic(phase_q7)
    }
    /// Branchless Q7 shortest-arc circular interpolation on $S^1$.
    #[inline(always)]
    fn q7_circular_lerp(&self, a: gcso_q7_t, b: gcso_q7_t, t_u7: u8) -> gcso_q7_t {
        branchless_q7_circular_lerp(a, b, t_u7)
    }
    /// Branchless Q7 saturating negation.
    #[inline(always)]
    fn q7_neg_sat(&self, x: gcso_q7_t) -> gcso_q7_t {
        branchless_q7_neg_sat(x)
    }
    /// Branchless Q7 fixed-point scaling.
    #[inline(always)]
    fn q7_scale_u7(&self, val: gcso_q7_t, scale_u7: u8) -> gcso_q7_t {
        branchless_q7_scale_u7(val, scale_u7)
    }
    /// SWAR 4-lane Q7 saturating addition.
    #[inline(always)]
    fn q7_add_sat_x4(&self, a_packed: u32, b_packed: u32) -> u32 {
        swar_q7_add_sat_x4(a_packed, b_packed)
    }
    /// SWAR 4-lane Q7 saturating subtraction.
    #[inline(always)]
    fn q7_sub_sat_x4(&self, a_packed: u32, b_packed: u32) -> u32 {
        swar_q7_sub_sat_x4(a_packed, b_packed)
    }
    /// SWAR 8-lane Q7 saturating addition.
    #[inline(always)]
    fn q7_add_sat_x8(&self, a_packed: u64, b_packed: u64) -> u64 {
        swar_q7_add_sat_x8(a_packed, b_packed)
    }
    /// SWAR 8-lane Q7 saturating subtraction.
    #[inline(always)]
    fn q7_sub_sat_x8(&self, a_packed: u64, b_packed: u64) -> u64 {
        swar_q7_sub_sat_x8(a_packed, b_packed)
    }
    /// SWAR 8-lane Q7 circular phase difference.
    #[inline(always)]
    fn q7_circular_diff_x8(&self, a_packed: u64, b_packed: u64) -> u64 {
        swar_q7_circular_diff_x8(a_packed, b_packed)
    }
    /// SWAR 8-lane Q7 circular phase addition.
    #[inline(always)]
    fn q7_circular_add_x8(&self, a_packed: u64, b_packed: u64) -> u64 {
        swar_q7_circular_add_x8(a_packed, b_packed)
    }
    /// SWAR 8-lane Q7 $S^1$ geodesic folding.
    #[inline(always)]
    fn q7_fold_s1_geodesic_x8(&self, packed: u64) -> u64 {
        swar_q7_fold_s1_geodesic_x8(packed)
    }
    /// SWAR 8-lane Q7 saturating negation.
    #[inline(always)]
    fn q7_neg_sat_x8(&self, packed: u64) -> u64 {
        swar_q7_neg_sat_x8(packed)
    }
    /// SWAR 8-lane Q7 shortest-arc geodesic Slerp.
    #[inline(always)]
    fn q7_geodesic_slerp_x8(&self, a_packed: u64, b_packed: u64, t_u7: u8) -> u64 {
        swar_q7_geodesic_slerp_x8(a_packed, b_packed, t_u7)
    }
    /// SWAR 8-lane Q7 fixed-point gain scaling.
    #[inline(always)]
    fn q7_scale_u7_x8(&self, packed: u64, scale_u7: u8) -> u64 {
        swar_q7_scale_u7_x8(packed, scale_u7)
    }
    /// SWAR 8-lane QDPS step filter.
    #[inline(always)]
    fn q7_qdps_filter_x8(&self, packed: u64, min_step_q7: u8) -> u64 {
        swar_q7_qdps_filter_x8(packed, min_step_q7)
    }
    /// SWAR 8-lane `u8` halving decay.
    #[inline(always)]
    fn u8_halve_decay_x8(&self, packed_u8: u64) -> u64 {
        swar_u8_halve_decay_x8(packed_u8)
    }
    /// SWAR 8-lane `u8` saturating addition.
    #[inline(always)]
    fn u8_add_sat_x8(&self, a: u64, b: u64) -> u64 {
        swar_u8_add_sat_x8(a, b)
    }
    /// SWAR 8-lane `u8` saturating subtraction.
    #[inline(always)]
    fn u8_sub_sat_x8(&self, packed_u8: u64, sub_step: u8) -> u64 {
        swar_u8_sub_sat_x8(packed_u8, sub_step)
    }
    /// SWAR 8-lane `u8` elementwise maximum.
    #[inline(always)]
    fn u8_max_x8(&self, a: u64, b: u64) -> u64 {
        swar_u8_max_x8(a, b)
    }
    /// Scalar Q7 QDPS step filter.
    #[inline(always)]
    fn qdps_filter_q7(&self, delta: gcso_q7_t, min_step_q7: i32) -> gcso_q7_t {
        branchless_qdps_filter_q7(delta, min_step_q7)
    }
    /// SplitMix64 bijective hash mixer.
    #[inline(always)]
    fn splitmix64(&self, x: u64) -> u64 {
        branchless_splitmix64(x)
    }
    /// Computes IEEE 802.3 CRC32 checksum via the 64B-aligned 4KB Slice-by-4 lookup table.
    #[inline(always)]
    fn crc32_ieee(&self, bytes: &[u8]) -> u32 {
        branchless_crc32_ieee(bytes)
    }
}

impl<T: ?Sized> BranchlessKernel for T {}

