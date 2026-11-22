// name: src/core/src/runtime/gateway/bridges/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Distributed `impl` blocks integrating Lv.1-Lv.3 types (`ResidualAdapterEngine`,
//! `SystemSpineState`, `EgoDifferentialNet`, `LiminalEgoState`, `StigmergicSubstrate`,
//! `PhaseAnchorRecord`, `CognitiveTrajectoryLedger`, `CognitiveCircuitBank`,
//! `PointerTrail`, and `EntropyController`) with Lv.4 Gateway/Signal contracts.

use super::*;
use crate::drivers::{
    DistortionTension, EgoDifferentialNet, FrameDirective, LiminalEgoState,
    SpineFrameTransitionInput, SystemSpineState, EGO_TRIGGER_BASIN_CAPTURE,
    EGO_TRIGGER_BOUNDARY_CHATTER, EGO_TRIGGER_BOUNDARY_STAGNATION, EGO_TRIGGER_INTERNAL_CONFLICT,
    EGO_TRIGGER_LAYER_DISTORTION, EGO_TRIGGER_LIMINAL_EXTERIOR,
};
use crate::foundation::{
    branchless_clamp_f32, branchless_max_f32, branchless_relu_f32, pack_action_slot_tag,
    swar_q7_avg_round_x8, swar_q7_mean_i8_from_words_64, AlignedSlice32, AlignedSliceMut32,
    TaggedPtr, ATTRACTOR_SLOT_ID_MASK, SWAR_SIGN_MASK_64, SWAR_SPLAT_BYTE_64, TAG_ACTION_MODE_MASK,
};
use crate::liminal::{
    branchless_select_f32, branchless_select_i32, branchless_select_i8, branchless_select_u64,
    branchless_select_u8, BitCut8, Lacuna, LacunaPartition8, LiminalThresholdState, LmnkResult,
    PhaseQ7, MAX_ATTRACTOR_DOMAINS, NUM_LIMINAL_THRESHOLD_CHANNELS, NUM_STIGMERGIC_GRANULES,
};
use crate::runtime::boundary_delta::LiminalBoundaryDelta;
use crate::runtime::context::CoreStateStack;
use crate::runtime::horizon::EpistemicHorizonReading;
use crate::stigmergy::{
    OverlapDiscoveryReport, PointerTrail, ResidualAdapterEngine, StigmergicGranuleState,
    SubstrateStepReport,
};

mod anchor_bridge;
mod ledger_bridge;
mod substrate_bridge;

impl ResidualAdapterEngine {
    /// Applies norm-adaptive fused Rank-1 residual injection driven by `frame`.
    ///
    /// # Errors
    /// Propagates any validation or numerical singularity error from [`apply_gateway_frame_rank1_hidden_fused`].
    #[inline]
    pub fn apply_gateway_frame_rank1_hidden_fused(
        &self,
        frame: &GatewayFrame,
        y_out: &mut AlignedSliceMut32<'_, f32>,
        x_in: &AlignedSlice32<'_, f32>,
        u_vec: &AlignedSlice32<'_, f32>,
        v_vec: &AlignedSlice32<'_, f32>,
    ) -> LmnkResult<f32> {
        apply_gateway_frame_rank1_hidden_fused(frame, y_out, x_in, u_vec, v_vec)
    }
}

impl SystemSpineState {
    /// Physical `pinned` isolation barrier executed without branching after Ego modulates `ego_frame`.
    ///
    /// - Under `unobserved_locked != 0` or `irreversible_step_lock != 0`, returns `*baseline_frame`.
    /// - Always preserves every `pinned_action_mode_mask` bit from `baseline_frame` and blocks
    ///   slot overrides away from any `pinned_slot_mask` slot.
    #[inline]
    pub fn enforce_pinned_isolation_branchless(
        &self,
        baseline_frame: &GatewayFrame,
        ego_frame: &mut GatewayFrame,
    ) {
        let (full_lock, eff_slot, eff_modes) = self.isolate_slot_and_modes_branchless(
            baseline_frame.dominant_slot,
            baseline_frame.action_mode_flags,
            ego_frame.dominant_slot,
            ego_frame.action_mode_flags,
        );
        if full_lock {
            *ego_frame = *baseline_frame;
            return;
        }

        ego_frame.dominant_slot = eff_slot;
        ego_frame.action_mode_flags = eff_modes;

        let ptr_tag = pack_action_slot_tag(
            ego_frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK,
            ego_frame.action_mode_flags,
        );
        ego_frame.frame_flags = u16::from(ptr_tag);
        ego_frame.tagged_ptr = TaggedPtr::pack(
            ego_frame.mapped_section_offset48,
            ptr_tag,
            ego_frame.mean_phase_tag_q7,
        )
        .raw();
    }
}

#[inline(always)]
fn pframe_transition_allowed(spine: &SystemSpineState, partition: &LacunaPartition8) -> bool {
    let all_zero = (partition.self_ref_mask
        | partition.interior_mask
        | partition.exterior_mask
        | partition.unobserved_mask
        | partition.undefined_mask)
        == 0;
    let mut input = SpineFrameTransitionInput::from_lacuna_partition(partition, 0, 0, 0, 0, 0);
    if all_zero
        || (partition.interior_mask != 0
            && partition.exterior_mask == 0
            && partition.undefined_mask == 0)
    {
        input.lacuna_code = Lacuna::SelfRef.code();
        input.candidate_granule_mask = 0xFF;
    } else if input.candidate_granule_mask == 0 {
        input.candidate_granule_mask = 0xFF;
    }
    let (directive, _) = spine.evaluate_frame_transition_branchless(input);
    matches!(
        directive,
        FrameDirective::EmitDelta | FrameDirective::ReAnchorKeyframe
    )
}

impl EgoDifferentialNet {
    /// Evaluates the fixed-length [`crate::drivers::DistortionDelta`] network in-place on `frame`
    /// (1-hop parallel perspective ensemble when `max_hops == 1`, or sequential chain up to 4 hops
    /// when `max_hops >= 2`; `0` heap alloc, `0 fdiv`) and returns `(delta_norm_q7, hops_traversed)`.
    ///
    /// When Ego is inactive, tension scale is 0, or `spine` enforces `unobserved_locked` /
    /// `irreversible_step_lock`, leaves `frame` untouched and returns `(0, 0)` (zero-copy bypass).
    #[inline]
    pub fn evaluate_pframe_chain_branchless(
        &self,
        ego: &LiminalEgoState,
        spine: &SystemSpineState,
        partition: &LacunaPartition8,
        stigmergy: &StigmergicGranuleState,
        tension: &DistortionTension,
        frame: &mut GatewayFrame,
    ) -> (u8, u8) {
        let (norm, hops, _mask) = self.evaluate_pframe_chain_with_mask_branchless(
            ego, spine, partition, stigmergy, tension, frame,
        );
        (norm, hops)
    }

    /// Evaluates the fixed-length [`crate::drivers::DistortionDelta`] network in-place on `frame`
    /// and returns `(delta_norm_q7, hops_traversed, entry_idx)` (`entry_idx` in `0..8`) for
    /// reinforcement telemetry tracking.
    #[inline]
    pub fn evaluate_pframe_chain_with_entry_branchless(
        &self,
        ego: &LiminalEgoState,
        spine: &SystemSpineState,
        partition: &LacunaPartition8,
        stigmergy: &StigmergicGranuleState,
        tension: &DistortionTension,
        frame: &mut GatewayFrame,
    ) -> (u8, u8, u8) {
        if !pframe_transition_allowed(spine, partition) {
            return (0, 0, 0);
        }
        let (acc_words, avg_delta, hops_traversed, entry_idx, matched_mask) = self
            .evaluate_pframe_words_branchless(
                ego,
                partition,
                stigmergy,
                tension.effective_ego_scale_q7,
                frame.phase_words_ref(),
            );
        if matched_mask == 0 {
            return (0, 0, 0);
        }
        let (full_lock, eff_slot, eff_modes) = spine.isolate_slot_and_modes_branchless(
            frame.dominant_slot,
            frame.action_mode_flags,
            frame.dominant_slot,
            frame.action_mode_flags,
        );
        if full_lock {
            return (avg_delta, hops_traversed, entry_idx);
        }
        frame.dominant_slot = eff_slot;
        frame.action_mode_flags = eff_modes;
        *frame.phase_words_mut() = acc_words;
        frame.mean_phase_tag_q7 = swar_q7_mean_i8_from_words_64(&acc_words);
        let ptr_tag = pack_action_slot_tag(
            frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK,
            frame.action_mode_flags,
        );
        frame.frame_flags = u16::from(ptr_tag);
        frame.tagged_ptr = TaggedPtr::pack(
            frame.mapped_section_offset48,
            ptr_tag,
            frame.mean_phase_tag_q7,
        )
        .raw();
        (avg_delta, hops_traversed, entry_idx)
    }

    /// Evaluates the fixed-length [`crate::drivers::DistortionDelta`] network in-place on `frame`
    /// and returns `(delta_norm_q7, hops_traversed, matched_mask)` of every matched/traversed node
    /// for parallel perspective reinforcement via [`Self::record_matched_activations`].
    #[inline]
    pub fn evaluate_pframe_chain_with_mask_branchless(
        &self,
        ego: &LiminalEgoState,
        spine: &SystemSpineState,
        partition: &LacunaPartition8,
        stigmergy: &StigmergicGranuleState,
        tension: &DistortionTension,
        frame: &mut GatewayFrame,
    ) -> (u8, u8, u8) {
        if !pframe_transition_allowed(spine, partition) {
            return (0, 0, 0);
        }
        let (acc_words, avg_delta, hops_traversed, _entry_idx, matched_mask) = self
            .evaluate_pframe_words_branchless(
                ego,
                partition,
                stigmergy,
                tension.effective_ego_scale_q7,
                frame.phase_words_ref(),
            );
        if matched_mask == 0 {
            return (0, 0, 0);
        }
        let (full_lock, eff_slot, eff_modes) = spine.isolate_slot_and_modes_branchless(
            frame.dominant_slot,
            frame.action_mode_flags,
            frame.dominant_slot,
            frame.action_mode_flags,
        );
        if full_lock {
            return (avg_delta, hops_traversed, matched_mask);
        }
        frame.dominant_slot = eff_slot;
        frame.action_mode_flags = eff_modes;
        *frame.phase_words_mut() = acc_words;
        frame.mean_phase_tag_q7 = swar_q7_mean_i8_from_words_64(&acc_words);
        let ptr_tag = pack_action_slot_tag(
            frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK,
            frame.action_mode_flags,
        );
        frame.frame_flags = u16::from(ptr_tag);
        frame.tagged_ptr = TaggedPtr::pack(
            frame.mapped_section_offset48,
            ptr_tag,
            frame.mean_phase_tag_q7,
        )
        .raw();
        (avg_delta, hops_traversed, matched_mask)
    }
}

impl LiminalEgoState {
    /// Purely computes the Ego trigger mask (`fired_mask`) and modulates `frame` in-place without
    /// mutating `self`, `liminal`, or `stigmergy`. Used directly by read-only per-layer
    /// Hot-Path RoPE modulation so multi-layer passes never accumulate repeated sub-layer shifts.
    #[inline]
    pub fn modulate_frame_pure(
        &self,
        liminal: &LiminalThresholdState,
        stigmergy: &StigmergicGranuleState,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        overlap: &OverlapDiscoveryReport,
        frame: &mut GatewayFrame,
    ) -> u8 {
        let _ = (liminal, stigmergy);
        let pull_q7 = branchless_clamp_f32(
            branchless_relu_f32(frame.effective_energy) * 16.0,
            0.0,
            127.0,
        ) as u8;
        let t_basin = ((self.trigger_mask & EGO_TRIGGER_BASIN_CAPTURE) != 0)
            & (pull_q7 >= self.basin_escape_floor_q7)
            & (self.counter_basin_thrust_q7 > 0);
        let t_conflict = ((self.trigger_mask & EGO_TRIGGER_INTERNAL_CONFLICT) != 0)
            & horizon.is_internal_conflict()
            & (horizon.conflict_severity > 0.35);
        let t_liminal = ((self.trigger_mask & EGO_TRIGGER_LIMINAL_EXTERIOR) != 0)
            & (horizon.is_liminal() | (overlap.lacuna_exterior_mask != 0));
        let t_chatter = ((self.trigger_mask & EGO_TRIGGER_BOUNDARY_CHATTER) != 0)
            & (delta.chattering_channel_mask != 0);
        let t_stagnate = ((self.trigger_mask & EGO_TRIGGER_BOUNDARY_STAGNATION) != 0)
            & (delta.stationary_shortcut_ready != 0);
        let t_distortion = (self.trigger_mask & EGO_TRIGGER_LAYER_DISTORTION) != 0;

        let fired_mask = branchless_select_u8(t_basin, EGO_TRIGGER_BASIN_CAPTURE, 0)
            | branchless_select_u8(t_conflict, EGO_TRIGGER_INTERNAL_CONFLICT, 0)
            | branchless_select_u8(t_liminal, EGO_TRIGGER_LIMINAL_EXTERIOR, 0)
            | branchless_select_u8(t_chatter, EGO_TRIGGER_BOUNDARY_CHATTER, 0)
            | branchless_select_u8(t_stagnate, EGO_TRIGGER_BOUNDARY_STAGNATION, 0)
            | branchless_select_u8(t_distortion, EGO_TRIGGER_LAYER_DISTORTION, 0);
        let any_fired = fired_mask != 0;

        // Modulate GatewayFrame phases with counter-basin orbital thrust / transgressive leap
        let scale_to_word = |q7: PhaseQ7, w_u8: u8| -> u64 {
            let scaled_i32 = (i32::from(q7) * i32::from(w_u8.min(128))) >> 7;
            let clamped_i8 = scaled_i32.clamp(-128, 127) as i8;
            (clamped_i8 as u8 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64)
        };

        let trans_pos = self.transgression_bias_q7 > 0;
        let trans_abs = self.transgression_bias_q7.unsigned_abs();
        let strong_trans =
            (t_liminal | t_distortion | (t_chatter & (delta.phase_transition_pulse != 0)))
                & trans_pos
                & (trans_abs >= 20);
        let apply_phase_override = t_basin | t_stagnate | strong_trans | t_distortion;
        let thrust_q7 = self.counter_basin_thrust_q7.min(127);
        {
            let pw = frame.phase_words_mut();
            let mut cell = 0usize;
            while cell < NUM_STIGMERGIC_GRANULES {
                let w = pw[cell];
                let sov_word = scale_to_word(
                    self.sovereign_granule_phase_q7[cell],
                    self.granule_autonomy_q7[cell],
                );
                // Antipodal pi-radian SO(2) flip (`^ SWAR_SIGN_MASK_64`) blended with sovereign heading.
                let flipped_w = w ^ SWAR_SIGN_MASK_64;
                let counter_dir = swar_q7_avg_round_x8(flipped_w, sov_word);
                let basin_orbited = branchless_select_u64(
                    thrust_q7 >= 64,
                    counter_dir,
                    swar_q7_avg_round_x8(w, sov_word),
                );
                let trans_dir = swar_q7_avg_round_x8(w, sov_word);
                let next_w = branchless_select_u64(
                    t_basin,
                    basin_orbited,
                    branchless_select_u64(strong_trans | t_stagnate, trans_dir, w),
                );
                pw[cell] = branchless_select_u64(apply_phase_override, next_w, w);
                cell += 1;
            }
        }

        // Re-energize zero-gated frame when strong transgressive Ego intentionally steps past Liminal
        let trans_energy = f32::from(trans_abs) * (1.0 / 64.0);
        frame.effective_energy = branchless_select_f32(
            strong_trans,
            branchless_max_f32(frame.effective_energy, trans_energy),
            branchless_select_f32(
                t_stagnate,
                branchless_max_f32(frame.effective_energy, 0.35),
                frame.effective_energy,
            ),
        );

        // Induce sovereign slot preference / irrational defiance under conflict or capture threat
        let mut best_slot = frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK;
        let mut best_aff = 0i8;
        let mut s_idx = 0usize;
        while s_idx < MAX_ATTRACTOR_DOMAINS {
            let aff = self.slot_affinity_bias_q7[s_idx];
            let take_s = aff > best_aff;
            best_aff = branchless_select_i8(take_s, aff, best_aff);
            best_slot = branchless_select_u8(take_s, s_idx as u8, best_slot);
            s_idx += 1;
        }
        let defy_slot = (t_conflict | t_basin | strong_trans)
            & (self.irrational_defiance_q7 >= 32)
            & (best_aff > 0);
        frame.dominant_slot = branchless_select_u8(defy_slot, best_slot, frame.dominant_slot);

        let ego_modes = BitCut8::split(
            frame.action_mode_flags | self.forced_action_mode_or_mask,
            self.suppressed_action_mode_mask,
        )
        .outside;
        frame.action_mode_flags = branchless_select_u8(
            any_fired,
            ego_modes & TAG_ACTION_MODE_MASK,
            frame.action_mode_flags,
        );
        frame.mean_phase_tag_q7 = swar_q7_mean_i8_from_words_64(frame.phase_words_ref());
        let ptr_tag = pack_action_slot_tag(
            frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK,
            frame.action_mode_flags,
        );
        frame.frame_flags = ptr_tag as u16;
        frame.tagged_ptr = TaggedPtr::pack(
            frame.mapped_section_offset48,
            ptr_tag,
            frame.mean_phase_tag_q7,
        )
        .raw();

        fired_mask
    }

    /// Branchlessly evaluates active Ego triggers directly across Liminal (`liminal`), Stigmergic (`stigmergy`), temporal velocity (`delta`), and `overlap`, updating both
    /// sub-layers and injecting counter-basin orbital thrust or transgressive phase steering into `frame`
    /// in-place.
    #[inline]
    pub fn intervene_on_sublayers_and_frame_branchless(
        &mut self,
        liminal: &mut LiminalThresholdState,
        stigmergy: &mut StigmergicGranuleState,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        overlap: &OverlapDiscoveryReport,
        frame: &mut GatewayFrame,
    ) {
        let fired_mask =
            self.modulate_frame_pure(liminal, stigmergy, delta, horizon, overlap, frame);
        let any_fired = fired_mask != 0;
        let t_liminal = (fired_mask & EGO_TRIGGER_LIMINAL_EXTERIOR) != 0;
        let t_chatter = (fired_mask & EGO_TRIGGER_BOUNDARY_CHATTER) != 0;

        // 1. Shift Layer 0 (LiminalThresholdState) channel horizons & gains branchlessly
        let mut ch = 0usize;
        while ch < NUM_LIMINAL_THRESHOLD_CHANNELS {
            let nudge_f = i32::from(self.trait_floor_bias_q7[ch]) / 2;
            let nudge_g = i32::from(self.trait_gain_bias_q7[ch]) / 2;
            let ch_chatter = t_chatter & (((delta.chattering_channel_mask >> ch) & 1) != 0);
            let chatter_stabilize = branchless_select_i32(ch_chatter, 4, 0);
            let new_f = (i32::from(liminal.threshold_floor_q7[ch]) + nudge_f + chatter_stabilize)
                .clamp(8, 120) as u8;
            let new_g = (i32::from(liminal.aperture_gain_q7[ch]) + nudge_g).clamp(16, 127) as u8;
            liminal.threshold_floor_q7[ch] =
                branchless_select_u8(any_fired, new_f, liminal.threshold_floor_q7[ch]);
            liminal.aperture_gain_q7[ch] =
                branchless_select_u8(any_fired, new_g, liminal.aperture_gain_q7[ch]);
            ch += 1;
        }

        // 2. Shift StigmergicGranuleState autonomy & discovery threshold
        let mut g = 0usize;
        while g < NUM_STIGMERGIC_GRANULES {
            let boost = self.granule_autonomy_q7[g] >> 4;
            let next_sg = stigmergy.granule_step_gain_q7[g]
                .saturating_add(boost)
                .min(127);
            stigmergy.granule_step_gain_q7[g] =
                branchless_select_u8(any_fired, next_sg, stigmergy.granule_step_gain_q7[g]);
            g += 1;
        }
        let trans_pos = self.transgression_bias_q7 > 0;
        let trans_abs = self.transgression_bias_q7.unsigned_abs();
        let trans_step = trans_abs >> 3;
        stigmergy.discovery_floor_q7 = branchless_select_u8(
            t_liminal & trans_pos,
            stigmergy
                .discovery_floor_q7
                .saturating_sub(trans_step)
                .max(8),
            stigmergy.discovery_floor_q7,
        );

        self.last_intervened_mask = fired_mask;
        self.intervention_epochs = self.intervention_epochs.wrapping_add(u32::from(any_fired));
    }

    /// Branchlessly evaluates active Ego triggers across `stack`'s Layer 0 (`liminal`),
    /// Stigmergic (`stigmergy`), temporal velocity (`delta`), and `overlap`, modulating `frame`
    /// in-place and synchronizing `self` onto `stack.ego` after intervention.
    #[inline]
    pub fn intervene_on_layers_and_frame_branchless(
        &mut self,
        stack: &mut CoreStateStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        overlap: &OverlapDiscoveryReport,
        frame: &mut GatewayFrame,
    ) {
        self.intervene_on_sublayers_and_frame_branchless(
            &mut stack.liminal,
            &mut stack.stigmergy,
            delta,
            horizon,
            overlap,
            frame,
        );
        stack.ego = *self;
    }
}

/// Composable receptor trait invoked by [`crate::stigmergy::StigmergicSubstrate`] when Swarm consensus and mapped-storage
/// attractor pull elevate a pointer trail's stigmergic pheromone density across an action regime.
///
/// Instead of writing ad-hoc `if/else` semantic branches (e.g., `if needs_web_search { ... }`),
/// external or Cold-Path capabilities (such as asynchronous retrieval, mapped-storage RAG prefetch, or
/// macro-basin consolidation) implement [`StigmergicReceptor`] and are composed into tuples
/// `(ReceptorA, ReceptorB)` that react when `report.target_slot`, `report.dominant_anchor_id`, and
/// `report.action_decision()` cross the physical stigmergic threshold.
pub trait StigmergicReceptor {
    /// Invoked when a [`SubstrateStepReport`] emerges from [`crate::stigmergy::StigmergicSubstrate`] with a
    /// non-passive [`crate::stigmergy::StigmergicActionDecision`].
    ///
    /// # Errors
    /// Returns a [`LmnkResult`] if the downstream receptor encounters an execution or capacity error.
    fn on_stigmergic_emergence(
        &mut self,
        frame: &GatewayFrame,
        report: &SubstrateStepReport,
        trail: &PointerTrail,
    ) -> LmnkResult<()>;
}

impl StigmergicReceptor for () {
    #[inline(always)]
    fn on_stigmergic_emergence(
        &mut self,
        _frame: &GatewayFrame,
        _report: &SubstrateStepReport,
        _trail: &PointerTrail,
    ) -> LmnkResult<()> {
        Ok(())
    }
}

impl<A: StigmergicReceptor, B: StigmergicReceptor> StigmergicReceptor for (A, B) {
    #[inline(always)]
    fn on_stigmergic_emergence(
        &mut self,
        frame: &GatewayFrame,
        report: &SubstrateStepReport,
        trail: &PointerTrail,
    ) -> LmnkResult<()> {
        self.0.on_stigmergic_emergence(frame, report, trail)?;
        self.1.on_stigmergic_emergence(frame, report, trail)
    }
}
