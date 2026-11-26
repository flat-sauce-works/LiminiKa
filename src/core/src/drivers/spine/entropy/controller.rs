// name: src/core/src/drivers/spine/entropy/controller.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stateful 128B [`Pod`] [`EntropyController`] with inherent entropy evaluation and
//! decoding-branch gating methods.

use core::mem::{align_of, offset_of, size_of};

use crate::drivers::spine::entropy::kernels::{
    compute_moving_z_score_entropy, compute_sparse_surrogate_entropy, eval_temperature_for_mode,
};
use crate::drivers::spine::entropy::{
    DEFAULT_BIFURCATION_THRESHOLD, DEFAULT_HYSTERESIS_BAND, DEFAULT_SAMPLE_TEMPERATURE,
    DEFAULT_SINGULARITY_EPS, DEFAULT_TAU_FLUX, EMA_ALPHA_ENTROPY, ENTROPY_GATE_MODE_NORMAL_ENTROPY,
    ENTROPY_GATE_MODE_PHASE_CONJUGATE_REPULSION, ENTROPY_GATE_MODE_PITCHFORK_BIFURCATION,
    ENTROPY_GATE_SLIDING_WINDOW_HORIZON, ENTROPY_GATE_WINDOW_RING_CAPACITY, EntropyState, INV_LN_2,
    LUT_REPULSION_GAIN, LUT_SAMPLE_TEMPERATURE, MAX_BARRIER_CEILING, MAX_VALID_TOKEN_Z_SCORE,
};
use crate::foundation::{
    ATTRACTOR_SLOT_ID_MASK, AlignedSlice32, F32_MIN_POSITIVE_NORMAL, Pod, TaggedPtr,
    branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_is_finite_f32,
    branchless_max_f32, branchless_min_f32, branchless_relu_f32, branchless_sanitize_z_score_f32,
    fast_inv_len_f32, fast_inv_pos_normal_f32, sum_sq_f32_simd8,
};
use crate::liminal::{
    LMNK_ERR_ENTROPY_SINGULARITY, LMNK_ERR_INVALID_ARGUMENT, LmnkResult, branchless_select_f32,
    branchless_select_u8, branchless_select_u32,
};

/// `EntropyController` stateful engine (128 bytes = 2 cache lines, 64B aligned).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntropyController {
    /// Active C-ABI `EntropyController` state descriptor (32 bytes, 32-byte aligned, `offset 0..32`).
    pub state: EntropyState,
    /// Total stateful evaluation steps processed (`offset 32..40`).
    pub step_count: u64,
    /// Hysteresis deadband width ($\delta_{\mathrm{hyst}}$) preventing mode chattering near thresholds (`offset 40..44`).
    pub hysteresis_band: f32,
    /// Threshold for sliding-window entropy rate flux ($\tau_{\mathrm{flux}}$, `offset 44..48`).
    pub tau_flux: f32,
    /// Consecutive token steps spent in the currently active branch mode (`offset 48..52`).
    pub mode_dwell_steps: u32,
    /// Minimum token dwell steps required before relaxing to a lower branch mode (`0` = immediate, `offset 52..56`).
    pub min_mode_dwell_steps: u32,
    /// Ring buffer write index (`0..=255`, masked with `& 7`, `offset 56`).
    pub(crate) window_idx: u8,
    /// Number of valid elements currently populated in `window_buf` (`0..=8`, `offset 57`).
    pub(crate) window_len: u8,
    /// Controller active lifecycle flag (`1` = active, `0` = passive, `offset 58`).
    pub is_active: bool,
    /// Explicit cache-line 0 padding (`offset 59..64`).
    pub(crate) reserved_cl0: [u8; 5],
    /// Fixed-size 8-slot ring buffer for Sliding-Window Entropy Rate Integrator ($\Phi_M(t)$, $M=5$, `offset 64..96`).
    pub(crate) window_buf: [f32; ENTROPY_GATE_WINDOW_RING_CAPACITY],
    /// Explicit cache-line 1 reserved expansion space (`offset 96..128`).
    pub(crate) reserved_cl1: [u8; 32],
}

// SAFETY: `EntropyController` is `#[repr(C, align(64))]` of size 128 bytes where every field is `Pod`.
unsafe impl Pod for EntropyController {}

const _: () = {
    assert!(size_of::<EntropyController>() == 128);
    assert!(align_of::<EntropyController>() == 64);
    assert!(offset_of!(EntropyController, state) == 0);
    assert!(offset_of!(EntropyController, step_count) == 32);
    assert!(offset_of!(EntropyController, hysteresis_band) == 40);
    assert!(offset_of!(EntropyController, tau_flux) == 44);
    assert!(offset_of!(EntropyController, mode_dwell_steps) == 48);
    assert!(offset_of!(EntropyController, min_mode_dwell_steps) == 52);
    assert!(offset_of!(EntropyController, window_idx) == 56);
    assert!(offset_of!(EntropyController, window_len) == 57);
    assert!(offset_of!(EntropyController, is_active) == 58);
    assert!(offset_of!(EntropyController, reserved_cl0) == 59);
    assert!(offset_of!(EntropyController, window_buf) == 64);
    assert!(offset_of!(EntropyController, reserved_cl1) == 96);
};

impl EntropyController {
    /// Constructs a new [`EntropyController`] with sanitized initial state configuration in branchless mode.
    #[inline(always)]
    pub const fn new(mut initial_state: EntropyState) -> Self {
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
            step_count: 0,
            hysteresis_band: DEFAULT_HYSTERESIS_BAND,
            tau_flux: DEFAULT_TAU_FLUX,
            mode_dwell_steps: 0,
            min_mode_dwell_steps: 0,
            window_idx: 0,
            window_len: 0,
            is_active: true,
            reserved_cl0: [0; 5],
            window_buf: [0.0; ENTROPY_GATE_WINDOW_RING_CAPACITY],
            reserved_cl1: [0; 32],
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
    pub const fn state(&self) -> EntropyState {
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
        self.state.active_branch_mode == ENTROPY_GATE_MODE_NORMAL_ENTROPY
    }

    /// Returns `true` if operating in Pitchfork Bifurcation Exploration Mode (`Mode 1`).
    #[inline(always)]
    pub const fn is_bifurcation_mode(&self) -> bool {
        self.state.active_branch_mode == ENTROPY_GATE_MODE_PITCHFORK_BIFURCATION
    }

    /// Returns `true` if operating in Phase-Conjugate Repulsion Mode (`Mode 2`).
    #[inline(always)]
    pub const fn is_repulsion_mode(&self) -> bool {
        self.state.active_branch_mode == ENTROPY_GATE_MODE_PHASE_CONJUGATE_REPULSION
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
        let mask = (ENTROPY_GATE_WINDOW_RING_CAPACITY - 1) as u8;
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

    /// Evaluates KV-keyframe Variable GOP keyframe scheduling and Game-Coexistence QoS compute budgeting
    /// in $\mathcal{O}(1)$ division-free (`0 fdiv`) branchless arithmetic from the current `EntropyController` state.
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

        let surge = (self.state.active_branch_mode != ENTROPY_GATE_MODE_NORMAL_ENTROPY)
            | (h_cur > tau)
            | (flux_cur > self.tau_flux);
        let emit_iframe =
            (steps_since_keyframe >= safe_max) | ((steps_since_keyframe >= safe_min) & surge);
        let gop_len = branchless_select_u32(
            emit_iframe,
            branchless_select_u32(steps_since_keyframe > 0, steps_since_keyframe, 1),
            safe_max,
        );

        let can_skip = (self.state.active_branch_mode == ENTROPY_GATE_MODE_NORMAL_ENTROPY)
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
        self.state.active_branch_mode = ENTROPY_GATE_MODE_NORMAL_ENTROPY;
        let mut i = 0usize;
        while i < ENTROPY_GATE_WINDOW_RING_CAPACITY {
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
        self.state.flags = src.state.flags;
        self.hysteresis_band = src.hysteresis_band;
        self.tau_flux = src.tau_flux;
        let mut i = 0usize;
        while i < ENTROPY_GATE_WINDOW_RING_CAPACITY {
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
        let horizon_u8 = ENTROPY_GATE_SLIDING_WINDOW_HORIZON as u8;
        let cap_u8 = ENTROPY_GATE_WINDOW_RING_CAPACITY as u8;
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
    /// has already been verified finite, writing only the active scalar fields into
    /// `state_out` with width-matched stores (`flags` preserved).
    #[inline(always)]
    fn eval_stateful_finite_into(
        &mut self,
        token_z_score: f32,
        state_out: &mut EntropyState,
    ) -> f32 {
        let current_mode = self.state.active_branch_mode;
        let tau = self.state.bifurcation_threshold;
        let eps = self.state.singularity_eps;
        let delta = self.hysteresis_band;
        let next_dwell = self.mode_dwell_steps.saturating_add(1);

        let token_z_safe = branchless_clamp_f32(token_z_score, 0.0, MAX_VALID_TOKEN_Z_SCORE);
        let (new_entropy, sliding_rate, is_first_step) =
            self.advance_entropy_ring_presanitized(token_z_safe);

        let (next_mode, final_dwell, sample_temp, rep_gain) = if (current_mode
            == ENTROPY_GATE_MODE_NORMAL_ENTROPY)
            & (new_entropy <= (tau + delta))
        {
            (
                ENTROPY_GATE_MODE_NORMAL_ENTROPY,
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

            let to_bif = ((current_mode == ENTROPY_GATE_MODE_NORMAL_ENTROPY) & cond_bif_up)
                | ((current_mode == ENTROPY_GATE_MODE_PHASE_CONJUGATE_REPULSION)
                    & cond_repulse_down);
            let mid_mode = branchless_select_u32(
                to_bif,
                ENTROPY_GATE_MODE_PITCHFORK_BIFURCATION,
                current_mode,
            );
            let nm = branchless_select_u32(
                cond_repulse_up,
                ENTROPY_GATE_MODE_PHASE_CONJUGATE_REPULSION,
                branchless_select_u32(cond_norm_down, ENTROPY_GATE_MODE_NORMAL_ENTROPY, mid_mode),
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

    /// Internal register-pipelined implementation of `eval_stateful_with_lacuna_and_pointer`
    /// assuming `token_z_score` and `lacuna_score` have already been verified finite.
    #[inline]
    // Hot-path register-passed SIMD/SWAR kernel leaf kept flat to avoid stack-frame spills.
    #[allow(clippy::too_many_arguments)]
    fn eval_stateful_with_lacuna_and_pointer_finite(
        &mut self,
        token_z_score: f32,
        lacuna_score: f32,
        alpha: f32,
        beta: f32,
        d0: f32,
        input_ptr: TaggedPtr,
        trail_density: f32,
        dominant_attractor_slot: u8,
        high_density_rank1_thresh: f32,
    ) -> (EntropyState, TaggedPtr) {
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
            lacuna_score,
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

        let is_repulsion = resolved_mode == ENTROPY_GATE_MODE_PHASE_CONJUGATE_REPULSION;
        let low_entropy_skip_thresh = branchless_relu_f32(tau * 0.5);
        let evolved_ptr = input_ptr.evolve_mode_stigmergically(
            trail_density,
            new_entropy,
            is_repulsion,
            dominant_attractor_slot & ATTRACTOR_SLOT_ID_MASK,
            low_entropy_skip_thresh,
            high_density_rank1_thresh,
        );

        let sample_temp = eval_temperature_for_mode(
            resolved_mode,
            DEFAULT_SAMPLE_TEMPERATURE,
            new_entropy,
            lacuna_score,
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

        let out_state = EntropyState {
            moving_z_entropy: new_entropy,
            bifurcation_threshold: tau,
            singularity_eps: eps,
            sliding_entropy_rate: sliding_rate,
            repulsion_gain: rep_gain,
            sample_temperature: sample_temp,
            active_branch_mode: resolved_mode,
            flags: self.state.flags,
        };
        (out_state, evolved_ptr)
    }

    /// Executes a single-writeback stateful `EntropyController` step coupled with $\mathcal{C}_{\mathrm{lacuna}}$
    /// dual-branch routing, dwell-step anti-chattering inertia across all downward
    /// transitions (`candidate_mode < prev_mode`), Eyring-Kramers temperature modulation, and
    /// $\mathcal{O}(1)$ [`TaggedPtr`] mode evolution.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_ENTROPY_SINGULARITY`] if `token_z_score` or `lacuna_score` is non-finite.
    #[inline(always)]
    // v0.3.0 frozen public Rust API; parameter-struct consolidation deferred to v0.4.0.
    #[allow(clippy::too_many_arguments)]
    pub fn eval_stateful_with_lacuna_and_pointer(
        &mut self,
        token_z_score: f32,
        lacuna_score: f32,
        alpha: f32,
        beta: f32,
        d0: f32,
        input_ptr: TaggedPtr,
        trail_density: f32,
        dominant_attractor_slot: u8,
        high_density_rank1_thresh: f32,
    ) -> LmnkResult<(EntropyState, TaggedPtr)> {
        if !branchless_is_finite_f32(token_z_score) || !branchless_is_finite_f32(lacuna_score) {
            return Err(LMNK_ERR_ENTROPY_SINGULARITY);
        }
        Ok(self.eval_stateful_with_lacuna_and_pointer_finite(
            token_z_score,
            lacuna_score,
            alpha,
            beta,
            d0,
            input_ptr,
            trail_density,
            dominant_attractor_slot,
            high_density_rank1_thresh,
        ))
    }

    /// Fast register-pipelined Top-2 surrogate + inter-layer cosine distance Moving Z-Score step
    /// on [`EntropyController`] using `self.state.singularity_eps` and bypassing redundant `NaN`/`Inf`
    /// checks on the already-sanitized `h_tilde` (`0` branches, `0 fdiv`, `0 ln`).
    ///
    /// # Errors
    /// Never fails on valid finite/non-finite inputs because [`compute_sparse_surrogate_entropy`]
    /// and [`compute_moving_z_score_entropy`] sanitize non-finite inputs to `0.0`.
    #[inline(always)]
    // v0.3.0 frozen public Rust API; parameter-struct consolidation deferred to v0.4.0.
    #[allow(clippy::too_many_arguments)]
    pub fn eval_surrogate_and_step(
        &mut self,
        p_max: f32,
        p_2nd: f32,
        cosine_dist: f32,
        mu_dist: f32,
        sigma_dist: f32,
        alpha_z: f32,
        gamma_mod: f32,
        state_out: &mut EntropyState,
    ) -> LmnkResult<f32> {
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
}

impl Default for EntropyController {
    #[inline(always)]
    fn default() -> Self {
        Self::new(EntropyState::default())
    }
}

impl EntropyController {
    /// Returns the 8-bit active SWAR Micro-Swarm cell mask (`0x0F` in low-entropy inertial mode, `0xFF` otherwise).
    #[inline(always)]
    pub const fn active_swarm_cell_mask(&self) -> u8 {
        let low_e = (self.state.active_branch_mode == ENTROPY_GATE_MODE_NORMAL_ENTROPY)
            & (self.state.moving_z_entropy < 0.10)
            & (self.state.sliding_entropy_rate <= self.tau_flux);
        branchless_select_u8(low_e, 0x0F, 0xFF)
    }

    /// Returns `true` when the `EntropyController` recommends inertial layer skipping (`ACTION_MODE_LAYER_SKIP`).
    #[inline(always)]
    pub const fn requests_inertial_layer_skip(&self) -> bool {
        let h_cur = branchless_relu_f32(self.state.moving_z_entropy);
        (self.state.active_branch_mode == ENTROPY_GATE_MODE_NORMAL_ENTROPY)
            & (h_cur <= 0.12)
            & (self.state.sliding_entropy_rate <= self.tau_flux)
    }
}

// ===================================================================

impl EntropyController {
    /// Evaluates token Z-score activation entropy and updates dynamic pitchfork branch mode
    /// in a single register-pipelined pass with a 3-`cmov` state machine and zero Store-to-Load stalls.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_ENTROPY_SINGULARITY`] if `token_z_score` is `NaN` or `Inf`.
    #[inline(always)]
    pub fn eval_stateful(
        &mut self,
        token_z_score: f32,
        state_out: &mut EntropyState,
    ) -> LmnkResult<()> {
        if !branchless_is_finite_f32(token_z_score) {
            return Err(LMNK_ERR_ENTROPY_SINGULARITY);
        }
        let _ = self.eval_stateful_finite_into(token_z_score, state_out);
        Ok(())
    }

    /// Computes $\mathcal{E}_{\mathrm{lat}}$ mean-square latent energy over a 32-byte aligned Key vector
    /// using the AVX2 tree-reduced accumulator (`sum_sq_f32_simd8`) and division-free reciprocal scaling.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `key_vector` is empty.
    /// Returns [`LMNK_ERR_ENTROPY_SINGULARITY`] if the accumulated sum-of-squares is `NaN` or `Inf`.
    #[inline(always)]
    pub fn eval_lacuna_energy_dyadic128(
        &self,
        key_vector: &AlignedSlice32<'_, f32>,
        lacuna_score_out: &mut f32,
    ) -> LmnkResult<()> {
        let slice = key_vector.as_slice();
        let len = slice.len();
        if len == 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let sum_sq = sum_sq_f32_simd8(slice);
        if !branchless_is_finite_f32(sum_sq) {
            return Err(LMNK_ERR_ENTROPY_SINGULARITY);
        }
        *lacuna_score_out = branchless_flush_subnormal_f32(sum_sq * fast_inv_len_f32(len));
        Ok(())
    }

    /// Calculates Eyring-Kramers potential barrier height:
    /// $\Delta V = \min\left(\frac{\tau_{\mathrm{eff}}}{\max(0, \mathcal{E}_{\mathrm{lat}}) + \epsilon_{\mathrm{log}}}, \Delta V_{\max}\right)$
    /// in division-free (`0 fdiv`) mode, clamping `denom` to `F32_MIN_POSITIVE_NORMAL` and
    /// evaluating `fast_inv_pos_normal_f32` to eliminate subnormal-branch jitter near epsilon singularities.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_ENTROPY_SINGULARITY`] if inputs are non-finite or `tau_eff <= 0.0`.
    #[inline(always)]
    pub fn eval_barrier(&self, lacuna_score: f32, tau_eff: f32) -> LmnkResult<f32> {
        let valid = branchless_is_finite_f32(lacuna_score)
            & branchless_is_finite_f32(tau_eff)
            & (tau_eff > 0.0);
        if !valid {
            return Err(LMNK_ERR_ENTROPY_SINGULARITY);
        }
        let eps_raw = self.state.singularity_eps;
        let eps = branchless_select_f32(
            branchless_is_finite_f32(eps_raw) & (eps_raw > 0.0),
            eps_raw,
            DEFAULT_SINGULARITY_EPS,
        );
        let denom = branchless_max_f32(
            branchless_relu_f32(lacuna_score) + eps,
            F32_MIN_POSITIVE_NORMAL,
        );
        let inv_denom = fast_inv_pos_normal_f32(denom);
        let barrier = tau_eff * inv_denom;
        if !branchless_is_finite_f32(barrier) {
            return Err(LMNK_ERR_ENTROPY_SINGULARITY);
        }
        Ok(branchless_flush_subnormal_f32(branchless_min_f32(
            barrier,
            MAX_BARRIER_CEILING,
        )))
    }

    /// Branchlessly evaluates the next `EntropyController` branch mode (`0`, `1`, `2`).
    #[inline(always)]
    #[must_use]
    // v0.3.0 frozen public Rust API; parameter-struct consolidation deferred to v0.4.0.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_branch_mode_transition(
        &self,
        current_mode: u32,
        moving_z_entropy: f32,
        sliding_flux: f32,
        lacuna_score: f32,
        bifurcation_threshold: f32,
        tau_flux: f32,
        hysteresis_band: f32,
    ) -> u32 {
        let hyst = branchless_relu_f32(hysteresis_band);
        let upper_thresh = bifurcation_threshold + hyst;
        let lower_thresh = branchless_relu_f32(bifurcation_threshold - hyst);

        let surge_triggered = (moving_z_entropy > upper_thresh)
            | (sliding_flux > (tau_flux + hyst))
            | ((moving_z_entropy > bifurcation_threshold) & (lacuna_score >= upper_thresh));
        let stay_active = (current_mode != 0) & (moving_z_entropy > lower_thresh);
        let active_non_normal = surge_triggered | stay_active;

        let active_target = branchless_select_u32(lacuna_score >= bifurcation_threshold, 1, 2);
        branchless_select_u32(active_non_normal, active_target, 0)
    }

    /// Evaluates the energy-guided dynamic sampling temperature modifier (`sample_temperature`).
    #[inline(always)]
    #[must_use]
    // v0.3.0 frozen public Rust API; parameter-struct consolidation deferred to v0.4.0.
    #[allow(clippy::too_many_arguments)]
    pub fn eval_energy_guided_temperature(
        &self,
        base_temperature: f32,
        moving_z_entropy: f32,
        lacuna_score: f32,
        tau_eff: f32,
        alpha: f32,
        beta: f32,
        d0: f32,
    ) -> f32 {
        eval_temperature_for_mode(
            self.state.active_branch_mode,
            base_temperature,
            moving_z_entropy,
            lacuna_score,
            tau_eff,
            alpha,
            beta,
            d0,
        )
    }
}
