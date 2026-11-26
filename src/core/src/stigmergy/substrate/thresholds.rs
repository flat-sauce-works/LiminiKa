// name: src/core/src/stigmergy/substrate/thresholds.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Autonomy thresholds and 7-regime next-action selection for [`super::StigmergicSubstrate`].

use super::{
    DEFAULT_STIGMERGIC_CONTRADICTION_THRESH, DEFAULT_STIGMERGIC_CRYSTALLIZE_THRESH,
    DEFAULT_STIGMERGIC_EVAPORATION_RETAIN, DEFAULT_STIGMERGIC_EXPRESS_STEER_THRESH,
    DEFAULT_STIGMERGIC_INERTIAL_SKIP_ENTROPY, DEFAULT_STIGMERGIC_RANK1_INJECT_THRESH,
    DEFAULT_STIGMERGIC_REINFORCEMENT_GAIN, DEFAULT_STIGMERGIC_SUBHEAD_ROUTE_THRESH,
    StigmergicActionDecision,
};
use crate::foundation::{
    ACTION_MODE_PHASE_STEER, ATTRACTOR_SLOT_ID_MASK, AlignedSliceMut32,
    DEFAULT_PHASE_CLAMP_MAX_RAD, Pod, SubheadConfig, branchless_abs_f32, branchless_clamp_f32,
    branchless_flush_subnormal_f32, branchless_is_finite_f32, branchless_min_f32,
    branchless_rad_to_phase_step_u8, branchless_relu_f32,
    dispatch_fused_phase_clamp_and_subhead_words_core, phase_clamp_upper_rot_dim,
    scale_f32_slice_unrolled8, steer_from_swar_words_core_safe, swar_q7_phase_step_filter_words_64,
    swar_words_nonzero_64,
};
use crate::liminal::{
    LMNK_ERR_INVALID_ARGUMENT, LmnkResult, MAX_ATTENTION_HEADS, NUM_STIGMERGIC_GRANULES,
    branchless_select_f32, branchless_select_u8, branchless_select_u32,
};
use crate::stigmergy::anchor::DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE;
use crate::stigmergy::granule::StigmergicGranuleState;
use crate::stigmergy::{
    ATTRACTOR_SLOT_REPULSION, MAX_STIGMERGIC_DENSITY, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE,
    TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_PHASE_STEER_ACTIVE,
    TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_FLAG_SYNERGY_RESONANCE,
};

/// 32-byte (`#[repr(C, align(32))]`) [`Pod`] threshold configuration governing autonomous
/// stigmergic pheromone evaporation, reinforcement, and 7-regime next-action selection.
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StigmergicThresholds {
    /// Per-step pheromone retention multiplier $\lambda_{\mathrm{retain}} \in [0.0, 1.0]$ (default `0.95`).
    pub evaporation_retain: f32,
    /// Base pheromone reinforcement deposit $g_{\mathrm{reinforce}} \ge 0.0$ on coherent hops (default `0.35`).
    pub reinforcement_gain: f32,
    /// Low-entropy threshold $\tau_{\mathrm{skip}}$ in nats triggering [`StigmergicActionDecision::InertialLayerSkip`] (default `0.15`).
    pub inertial_skip_entropy_thresh: f32,
    /// Minimum stigmergic density $\tau_{\mathrm{steer}}$ triggering [`StigmergicActionDecision::ExpressPhaseSteer`] (default `0.35`).
    pub express_steer_density_thresh: f32,
    /// Minimum stigmergic density $\tau_{\mathrm{subhead}}$ triggering [`StigmergicActionDecision::SubHeadRoute`] (default `0.85`).
    pub subhead_route_density_thresh: f32,
    /// Minimum stigmergic density $\tau_{\mathrm{rank1}}$ triggering [`StigmergicActionDecision::FullOrchestration`] (default `1.50`).
    pub rank1_inject_density_thresh: f32,
    /// Minimum stigmergic density $\tau_{\mathrm{anchor}}$ triggering [`StigmergicActionDecision::CrystallizeMacroBasin`] (default `2.20`).
    pub crystallize_density_thresh: f32,
    /// Contradiction coherence threshold $\tau_{\mathrm{contradict}} \in [0.0, 1.0]$ where $\mathcal{C}_{\mathrm{phase}} < -\tau_{\mathrm{contradict}}$ triggers [`StigmergicActionDecision::RepulsionEscape`] (default `0.25`).
    pub contradiction_coherence_thresh: f32,
}

// SAFETY: `StigmergicThresholds` is a 32-byte `#[repr(C, align(32))]` struct of 8 `f32` fields with zero padding.
unsafe impl Pod for StigmergicThresholds {}

impl Default for StigmergicThresholds {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// Bundled parameters for [`StigmergicThresholds::execute_decision_steering_and_logit_scaling`].
#[derive(Clone, Copy)]
pub(super) struct DecisionSteerParams<'a> {
    pub decision: StigmergicActionDecision,
    pub steer_words: &'a [u64; NUM_STIGMERGIC_GRANULES],
    pub phase_step_min_rad: f32,
    pub phase_clamp_max: f32,
    pub head_dim: usize,
    pub num_heads: usize,
    pub best_coh: f32,
}

impl StigmergicThresholds {
    /// Creates the canonical LiminiKa stigmergic threshold ladder.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            evaporation_retain: DEFAULT_STIGMERGIC_EVAPORATION_RETAIN,
            reinforcement_gain: DEFAULT_STIGMERGIC_REINFORCEMENT_GAIN,
            inertial_skip_entropy_thresh: DEFAULT_STIGMERGIC_INERTIAL_SKIP_ENTROPY,
            express_steer_density_thresh: DEFAULT_STIGMERGIC_EXPRESS_STEER_THRESH,
            subhead_route_density_thresh: DEFAULT_STIGMERGIC_SUBHEAD_ROUTE_THRESH,
            rank1_inject_density_thresh: DEFAULT_STIGMERGIC_RANK1_INJECT_THRESH,
            crystallize_density_thresh: DEFAULT_STIGMERGIC_CRYSTALLIZE_THRESH,
            contradiction_coherence_thresh: DEFAULT_STIGMERGIC_CONTRADICTION_THRESH,
        }
    }

    /// Validates that all threshold parameters are finite and ordered cleanly.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if any threshold is non-finite or negative.
    #[inline(always)]
    pub const fn validate(&self) -> LmnkResult<()> {
        let valid = branchless_is_finite_f32(self.evaporation_retain)
            & (self.evaporation_retain >= 0.0)
            & (self.evaporation_retain <= 1.0)
            & branchless_is_finite_f32(self.reinforcement_gain)
            & (self.reinforcement_gain >= 0.0)
            & branchless_is_finite_f32(self.inertial_skip_entropy_thresh)
            & (self.inertial_skip_entropy_thresh >= 0.0)
            & branchless_is_finite_f32(self.express_steer_density_thresh)
            & (self.express_steer_density_thresh >= 0.0)
            & branchless_is_finite_f32(self.subhead_route_density_thresh)
            & (self.subhead_route_density_thresh >= self.express_steer_density_thresh)
            & branchless_is_finite_f32(self.rank1_inject_density_thresh)
            & (self.rank1_inject_density_thresh >= self.subhead_route_density_thresh)
            & branchless_is_finite_f32(self.crystallize_density_thresh)
            & (self.crystallize_density_thresh >= self.rank1_inject_density_thresh)
            & branchless_is_finite_f32(self.contradiction_coherence_thresh)
            & (self.contradiction_coherence_thresh >= 0.0)
            & (self.contradiction_coherence_thresh <= 1.0);
        if valid {
            Ok(())
        } else {
            Err(LMNK_ERR_INVALID_ARGUMENT)
        }
    }

    /// Combines integrated pheromone density $\rho_{\mathrm{int}}$ with a deadband-guarded positive
    /// rising derivative $\Delta\rho_+$ via a Q0.7 convex combination weighted by the branchless
    /// $4p(1-p)$ boundary sensitivity gate (`0 fdiv`, `0 alloc`).
    ///
    /// Returns `(effective_rho, exceeds_deadband)` where `effective_rho` stays strictly within
    /// `[0.0, MAX_STIGMERGIC_DENSITY]`, preserving the monotone threshold ladder.
    #[inline]
    #[must_use]
    pub const fn compute_effective_density_calculus(
        prev_rho: f32,
        integral_rho: f32,
        energy_gate: f32,
        alpha_q7: u8,
        noise_tolerance_q7: u8,
        prev_rising: bool,
    ) -> (f32, bool) {
        let p_rho = branchless_flush_subnormal_f32(branchless_relu_f32(prev_rho));
        let i_rho = branchless_min_f32(
            branchless_flush_subnormal_f32(branchless_relu_f32(integral_rho)),
            MAX_STIGMERGIC_DENSITY,
        );
        let e_gate = branchless_clamp_f32(energy_gate, 0.0, 1.0);

        let raw_delta = branchless_relu_f32(i_rho - p_rho);
        let deadband = (noise_tolerance_q7 as f32) * (1.0_f32 / 256.0_f32);
        let exceeds_deadband = raw_delta > deadband;
        let strong_rise = raw_delta > (deadband * 2.0_f32);
        // Guard against single-step quantized noise spikes: require either 2 consecutive rising
        // steps above deadband or a strong rise exceeding 2x deadband.
        let confirmed_rise = exceeds_deadband & (prev_rising | strong_rise);

        let p_norm = branchless_clamp_f32(i_rho * 0.25_f32 + e_gate * 0.5_f32, 0.0, 1.0);
        let p_q7 = (p_norm * 127.0_f32) as u8;
        let gate_q7 = StigmergicGranuleState::boundary_sensitivity_gate_q7(p_q7);
        let gate_f = (gate_q7 as f32) * (1.0_f32 / 128.0_f32);

        let net_delta = branchless_relu_f32(raw_delta - deadband);
        let deriv_scaled = branchless_min_f32(net_delta * 8.0_f32 * gate_f, MAX_STIGMERGIC_DENSITY);
        let deriv_term = branchless_select_f32(confirmed_rise, deriv_scaled, 0.0_f32);

        let clamped_alpha = if alpha_q7 > 127 { 127 } else { alpha_q7 };
        let alpha_f = (clamped_alpha as f32) * (1.0_f32 / 128.0_f32);
        let one_minus_alpha_f = ((128 - clamped_alpha) as f32) * (1.0_f32 / 128.0_f32);
        let eff = branchless_min_f32(
            branchless_flush_subnormal_f32(one_minus_alpha_f * i_rho + alpha_f * deriv_term),
            MAX_STIGMERGIC_DENSITY,
        );
        (eff, exceeds_deadband)
    }

    /// Evaluates the 7-regime stigmergic threshold ladder in **strictly branchless constant time
    /// (`0` data-dependent branches)**, returning
    /// `(decision, action_mode_flags, target_slot, extra_trail_flags)`.
    #[inline]
    #[must_use]
    pub const fn select_action_decision(
        &self,
        stigmergic_density: f32,
        entropy_h_tilde: f32,
        phase_coherence: f32,
        repulsion_triggered: bool,
        preferred_slot: u8,
    ) -> (StigmergicActionDecision, u8, u8, u32) {
        let rho = branchless_flush_subnormal_f32(branchless_relu_f32(stigmergic_density));
        let h_tilde = branchless_select_f32(
            branchless_is_finite_f32(entropy_h_tilde),
            branchless_relu_f32(entropy_h_tilde),
            0.0,
        );
        let coh = branchless_select_f32(
            branchless_is_finite_f32(phase_coherence),
            branchless_clamp_f32(phase_coherence, -1.0, 1.0),
            0.0,
        );
        let contradict_thresh = branchless_clamp_f32(
            branchless_abs_f32(self.contradiction_coherence_thresh),
            0.0,
            1.0,
        );
        let is_repulsed = repulsion_triggered | (coh < -contradict_thresh);

        let is_low_entropy_skip = (!is_repulsed) & (h_tilde <= self.inertial_skip_entropy_thresh);
        let rank1_entropy_min = self.inertial_skip_entropy_thresh * 1.6_f32;
        let subhead_entropy_fallback = self.inertial_skip_entropy_thresh * 3.6_f32;
        let ge_steer = rho >= self.express_steer_density_thresh;
        let ge_subhead = (rho >= self.subhead_route_density_thresh)
            | (ge_steer & (h_tilde >= subhead_entropy_fallback));
        let ge_rank1 = (rho >= self.rank1_inject_density_thresh) & (h_tilde > rank1_entropy_min);
        let ge_anchor = (rho >= self.crystallize_density_thresh)
            & (coh >= DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE);

        // Build decision discriminant bottom-up along the density ladder (`0..=6`):
        let mut disc = 0u8;
        disc = branchless_select_u8(
            ge_steer,
            StigmergicActionDecision::ExpressPhaseSteer as u8,
            disc,
        );
        disc = branchless_select_u8(
            ge_subhead,
            StigmergicActionDecision::SubHeadRoute as u8,
            disc,
        );
        disc = branchless_select_u8(
            ge_rank1,
            StigmergicActionDecision::FullOrchestration as u8,
            disc,
        );
        disc = branchless_select_u8(
            ge_anchor,
            StigmergicActionDecision::CrystallizeMacroBasin as u8,
            disc,
        );
        // Low-entropy laminar regime overrides active steering unless super-saturated crystallization is ready:
        disc = branchless_select_u8(
            is_low_entropy_skip & (!ge_anchor),
            StigmergicActionDecision::InertialLayerSkip as u8,
            disc,
        );
        // Repulsion escape unconditionally overrides all other regimes:
        disc = branchless_select_u8(
            is_repulsed,
            StigmergicActionDecision::RepulsionEscape as u8,
            disc,
        );

        let decision = StigmergicActionDecision::from_u8(disc);
        let mode_flags = decision.action_mode_flags();
        let clean_pref_slot = preferred_slot & ATTRACTOR_SLOT_ID_MASK;
        let target_slot =
            branchless_select_u8(is_repulsed, ATTRACTOR_SLOT_REPULSION, clean_pref_slot);

        let extra_flags = branchless_select_u32(
            (mode_flags & ACTION_MODE_PHASE_STEER) != 0,
            TRAIL_FLAG_PHASE_STEER_ACTIVE,
            0,
        ) | branchless_select_u32(
            is_low_entropy_skip & (!ge_anchor),
            TRAIL_FLAG_INERTIAL_SKIP_ACTIVE,
            0,
        ) | branchless_select_u32(
            ge_anchor & (!is_repulsed),
            TRAIL_FLAG_CRYSTALLIZED_CANDIDATE | TRAIL_FLAG_SYNERGY_RESONANCE,
            0,
        ) | branchless_select_u32(
            is_repulsed,
            TRAIL_FLAG_REPULSION_ACTIVE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE,
            0,
        );

        (decision, mode_flags, target_slot, extra_flags)
    }

    #[inline]
    pub(super) fn execute_decision_steering_and_logit_scaling(
        params: DecisionSteerParams<'_>,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> LmnkResult<(bool, bool, bool)> {
        let DecisionSteerParams {
            decision,
            steer_words,
            phase_step_min_rad,
            phase_clamp_max,
            head_dim,
            num_heads,
            best_coh,
        } = params;
        let mut steered = false;
        let mut subhead_routed = false;
        let mut skip_compensated = false;
        let min_step_q7 = branchless_rad_to_phase_step_u8(phase_step_min_rad);

        match decision {
            StigmergicActionDecision::PassiveSidecar => {}
            StigmergicActionDecision::InertialLayerSkip => {
                skip_compensated = true;
            }
            StigmergicActionDecision::ExpressPhaseSteer
            | StigmergicActionDecision::RepulsionEscape => {
                let swar_w = if min_step_q7 <= 1 {
                    *steer_words
                } else {
                    swar_q7_phase_step_filter_words_64(steer_words, min_step_q7)
                };
                if swar_words_nonzero_64(&swar_w) {
                    if let Some(q) = query_tensor {
                        steer_from_swar_words_core_safe(
                            &swar_w,
                            0xFF,
                            128,
                            head_dim,
                            num_heads,
                            false,
                            phase_clamp_max,
                            q,
                            key_tensor,
                        );
                        steered = true;
                    } else if let Some(k) = key_tensor {
                        steer_from_swar_words_core_safe(
                            &swar_w,
                            0xFF,
                            128,
                            head_dim,
                            num_heads,
                            false,
                            phase_clamp_max,
                            k,
                            None,
                        );
                        steered = true;
                    }
                }
            }
            StigmergicActionDecision::SubHeadRoute
            | StigmergicActionDecision::FullOrchestration
            | StigmergicActionDecision::CrystallizeMacroBasin => {
                let swar_w = if min_step_q7 <= 1 {
                    *steer_words
                } else {
                    swar_q7_phase_step_filter_words_64(steer_words, min_step_q7)
                };
                if swar_words_nonzero_64(&swar_w)
                    && let Some(q) = query_tensor
                {
                    let is_default_clamp = (phase_clamp_max.to_bits()
                        == DEFAULT_PHASE_CLAMP_MAX_RAD.to_bits())
                        & (num_heads > 0)
                        & (num_heads <= MAX_ATTENTION_HEADS);
                    if is_default_clamp {
                        let rot_dim = phase_clamp_upper_rot_dim(head_dim);
                        if let Some(k) = key_tensor {
                            crate::foundation::rope::rotate_fused_self_phase_clamp_subhead_q_and_clamp_k_default_words(
                                    q.as_mut_slice(),
                                    k.as_mut_slice(),
                                    &swar_w,
                                    num_heads,
                                    head_dim,
                                    rot_dim,
                                );
                        } else {
                            crate::foundation::rope::rotate_fused_self_phase_clamp_and_subhead_default_words(
                                    q.as_mut_slice(),
                                    &swar_w,
                                    num_heads,
                                    head_dim,
                                    rot_dim,
                                    true,
                                );
                        }
                    } else {
                        let f_heads = (num_heads >> 1) as u16;
                        let l_heads = (num_heads >> 2) as u16;
                        let e_heads = (num_heads as u16).saturating_sub(f_heads + l_heads);
                        let subhead_cfg = SubheadConfig {
                            num_fact_heads: f_heads,
                            num_logic_heads: l_heads,
                            num_explore_heads: e_heads,
                            flags: 0,
                            fact_phase_gain: 1.0,
                            logic_phase_gain: 1.0,
                            explore_phase_gain: 1.15,
                            reserved: [0; 12],
                        };
                        dispatch_fused_phase_clamp_and_subhead_words_core(
                            q,
                            &swar_w,
                            phase_clamp_max,
                            &subhead_cfg,
                            &swar_w,
                            &swar_w,
                            &swar_w,
                            phase_clamp_max,
                            0,
                            head_dim,
                            true,
                        )?;
                        if let Some(k) = key_tensor {
                            steer_from_swar_words_core_safe(
                                &swar_w,
                                0xFF,
                                128,
                                head_dim,
                                num_heads,
                                false,
                                phase_clamp_max,
                                k,
                                None,
                            );
                        }
                    }
                    steered = true;
                    subhead_routed = true;
                }
            }
        }

        if let Some(logits) = logits_opt
            && !logits.is_empty()
        {
            let inv_temp = branchless_select_f32(
                decision as u8 == StigmergicActionDecision::RepulsionEscape as u8,
                0.80,
                branchless_clamp_f32(1.0 + 0.15 * branchless_relu_f32(best_coh), 0.75, 1.35),
            );
            scale_f32_slice_unrolled8(logits.as_mut_slice(), inv_temp);
        }
        Ok((steered, subhead_routed, skip_compensated))
    }
}
