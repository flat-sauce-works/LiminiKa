// name: src/core/src/runtime/gateway/bridges.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Distributed `impl` blocks integrating Lv.1-Lv.3 types (`ResidualAdapterEngine`,
//! `SystemSpineState`, `EgoDifferentialNet`, `LiminalEgoState`, `StigmergicSubstrate`,
//! `CrystallizedAttractorRecord`, `CognitiveTrajectoryLedger`, `CognitiveCircuitBank`,
//! `gcso_pointer_trail_t`, and `EntropyController`) with Lv.4 Gateway/Signal contracts.

use super::*;
use crate::drivers::{
    DistortionTension, EgoDifferentialNet, EntropyController, FrameDirective, LiminalEgoState,
    SpineFrameTransitionInput, SystemSpineState, EGO_TRIGGER_BASIN_CAPTURE,
    EGO_TRIGGER_BOUNDARY_CHATTER, EGO_TRIGGER_BOUNDARY_STAGNATION, EGO_TRIGGER_INTERNAL_CONFLICT,
    EGO_TRIGGER_LAYER_DISTORTION, EGO_TRIGGER_LIMINAL_EXTERIOR, ENTROPY_GATE_MODE_NORMAL_ENTROPY,
    ENTROPY_GATE_MODE_PHASE_CONJUGATE_REPULSION,
};
use crate::foundation::{
    branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_is_finite_f32,
    branchless_lerp_f32, branchless_max_f32, branchless_min_f32, branchless_rad_to_phase_step_u8,
    branchless_relu_f32, coherence_from_word_arrays_with_both_uniform_flags_64,
    fast_inv_pos_normal_f32, fast_ln_1p_bounded_f32, fast_mean_q7_tag_from_words_64,
    gcso_tagged_ptr_t, is_word_array_uniform_64, load_q7_words_64, pack_action_slot_tag,
    store_q7_words_64, swar_q7_avg_round_x8, swar_q7_circular_diff_x8,
    swar_q7_mean_i8_from_words_64, swar_q7_mean_i8_from_words_with_uniform_flag_64,
    swar_slerp_words_with_uniform_flag_64, swar_words_equal_64, swar_words_nonzero_64,
    words_to_q7_array_64, AlignedSlice32, AlignedSliceMut32, PhaseSteering, Pod,
    RawPhaseProjectable, ACTION_MODE_PHASE_REPULSE, ACTION_MODE_PHASE_STEER,
    ACTION_MODE_SUBHEAD_ROUTE, ACTION_PRESET_EXPRESS_PHASE, ACTION_PRESET_PASSIVE_SIDECAR,
    ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_ID_MASK, DEFAULT_PHASE_CLAMP_MAX_RAD,
    DEFAULT_PHASE_STEP_MIN_RAD, SWAR_SIGN_MASK_64, SWAR_SPLAT_BYTE_64, TAG_ACTION_MODE_MASK,
};
use crate::liminal::{
    branchless_select_f32, branchless_select_i32, branchless_select_i8, branchless_select_u32,
    branchless_select_u64, branchless_select_u8, branchless_select_usize, gcso_q7_t, BitCut8,
    EpistemicPartition, GcsoResult, Grounded, Indivisible, Lacuna, LiminalThresholdState, Scoped,
    ScopedCut, DEFAULT_ENTROPY_EPS, GCSO_ERROR_INVALID_ARGUMENT, MAX_ATTENTION_HEADS,
    MAX_ATTRACTOR_DOMAINS, NUM_LIMINAL_THRESHOLD_CHANNELS, NUM_STIGMERGIC_GRANULES,
};
use crate::runtime::boundary_delta::LiminalBoundaryDelta;
use crate::runtime::context::AutonomousCoreLayerStack;
use crate::runtime::horizon::EpistemicHorizonReading;
use crate::runtime::telemetry::{CoreTelemetryBank, LayerDistortionKind};
use crate::stigmergy::{
    gcso_anchor_type_t, gcso_pointer_trail_t, trail_phase_words_mut, trail_phase_words_ref,
    CognitiveCircuitActivation, CognitiveCircuitBank, CognitiveCircuitBankOps, CognitiveOpKind,
    CognitiveTrajectoryLedger, CrystallizedAttractorRecord, DissipativeOverflowReport,
    EpisodicPhaseSample, PhaseWordsRef64, RelativeAttractorNode, ResidualAdapterEngine,
    SandboxLifecycleEvent, StigmergicActionDecision, StigmergicGranuleState,
    StigmergicOverlapSerendipityReport, StigmergicScaleGranularity, StigmergicSubstrate,
    StigmergicSubstrateOps, SubstrateStepReport, WorkbenchSlotSpec, ATTRACTOR_SLOT_CRYSTALLIZED,
    ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA, ATTRACTOR_SLOT_REPULSION,
    DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE, GCSO_ANCHOR_TYPE_CRYSTALLIZED,
    MAX_ATTRACTOR_PULL_FORCE, SUBSTRATE_INLINE_TRAILS, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE,
    TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_SWARM_QUORUM_MASK, TRAIL_SWARM_QUORUM_SHIFT,
};

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
    ) -> GcsoResult<f32> {
        apply_gateway_frame_rank1_hidden_fused(frame, y_out, x_in, u_vec, v_vec)
    }
}

impl SystemSpineState {
    /// Physical `pinned` isolation barrier executed without branching after Ego modulates `ego_frame`.
    ///
    /// - Under `ood_void_locked != 0` or `irreversible_step_lock != 0`, returns `*baseline_frame`.
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
        ego_frame.tagged_ptr = gcso_tagged_ptr_t::pack(
            ego_frame.mapped_section_offset48,
            ptr_tag,
            ego_frame.mean_phase_tag_q7,
        )
        .raw();
    }
}

#[inline(always)]
fn pframe_transition_allowed(spine: &SystemSpineState, venn: &EpistemicPartition) -> bool {
    let lacuna_code = if (venn.shared_core_mask | venn.self_exclusive_mask) != 0 {
        Lacuna::SelfRef.code()
    } else if venn.known_unknown_mask != 0 {
        Lacuna::Exterior.code()
    } else if venn.true_void_mask != 0 {
        Lacuna::Unobserved.code()
    } else {
        Lacuna::SelfRef.code()
    };
    let (directive, _) = spine.evaluate_frame_transition_branchless(SpineFrameTransitionInput {
        lacuna_code,
        phase_delta_q7: 0,
        reanchor_delta_thresh_q7: 0,
        steps_since_keyframe: 0,
        max_gop_steps: 0,
        candidate_granule_mask: 0xFF,
        flags: 0,
        reserved: 0,
    });
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
    /// When Ego is inactive, tension scale is 0, or `spine` enforces `ood_void_locked` /
    /// `irreversible_step_lock`, leaves `frame` untouched and returns `(0, 0)` (zero-copy bypass).
    #[inline]
    pub fn evaluate_pframe_chain_branchless(
        &self,
        ego: &LiminalEgoState,
        spine: &SystemSpineState,
        venn: &EpistemicPartition,
        stigmergy: &StigmergicGranuleState,
        tension: &DistortionTension,
        frame: &mut GatewayFrame,
    ) -> (u8, u8) {
        let (norm, hops, _mask) = self.evaluate_pframe_chain_with_mask_branchless(
            ego, spine, venn, stigmergy, tension, frame,
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
        venn: &EpistemicPartition,
        stigmergy: &StigmergicGranuleState,
        tension: &DistortionTension,
        frame: &mut GatewayFrame,
    ) -> (u8, u8, u8) {
        if !pframe_transition_allowed(spine, venn) {
            return (0, 0, 0);
        }
        let (acc_words, avg_delta, hops_traversed, entry_idx, matched_mask) = self
            .evaluate_pframe_words_branchless(
                ego,
                venn,
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
        frame.tagged_ptr = gcso_tagged_ptr_t::pack(
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
        venn: &EpistemicPartition,
        stigmergy: &StigmergicGranuleState,
        tension: &DistortionTension,
        frame: &mut GatewayFrame,
    ) -> (u8, u8, u8) {
        if !pframe_transition_allowed(spine, venn) {
            return (0, 0, 0);
        }
        let (acc_words, avg_delta, hops_traversed, _entry_idx, matched_mask) = self
            .evaluate_pframe_words_branchless(
                ego,
                venn,
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
        frame.tagged_ptr = gcso_tagged_ptr_t::pack(
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
    /// mutating `self`, `liminal_layer`, or `stigmergic_layer`. Used directly by read-only per-layer
    /// Hot-Path RoPE modulation so multi-layer passes never accumulate repeated sub-layer shifts.
    #[inline]
    pub fn modulate_frame_pure(
        &self,
        liminal_layer: &LiminalThresholdState,
        stigmergic_layer: &StigmergicGranuleState,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        venn: &StigmergicOverlapSerendipityReport,
        frame: &mut GatewayFrame,
    ) -> u8 {
        let _ = (liminal_layer, stigmergic_layer);
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
            & (horizon.is_liminal() | (venn.void_exterior_mask != 0));
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
        let scale_to_word = |q7: gcso_q7_t, w_u8: u8| -> u64 {
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
        frame.tagged_ptr = gcso_tagged_ptr_t::pack(
            frame.mapped_section_offset48,
            ptr_tag,
            frame.mean_phase_tag_q7,
        )
        .raw();

        fired_mask
    }

    /// Branchlessly evaluates active Ego triggers directly across Core Layer 0 (`liminal_layer`),
    /// Core Layer 1 (`stigmergic_layer`), temporal velocity (`delta`), and `venn`, updating both
    /// sub-layers and injecting counter-basin orbital thrust or transgressive phase steering into `frame`
    /// in-place.
    #[inline]
    pub fn intervene_on_sublayers_and_frame_branchless(
        &mut self,
        liminal_layer: &mut LiminalThresholdState,
        stigmergic_layer: &mut StigmergicGranuleState,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        venn: &StigmergicOverlapSerendipityReport,
        frame: &mut GatewayFrame,
    ) {
        let fired_mask =
            self.modulate_frame_pure(liminal_layer, stigmergic_layer, delta, horizon, venn, frame);
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
            let new_f =
                (i32::from(liminal_layer.threshold_floor_q7[ch]) + nudge_f + chatter_stabilize)
                    .clamp(8, 120) as u8;
            let new_g =
                (i32::from(liminal_layer.aperture_gain_q7[ch]) + nudge_g).clamp(16, 127) as u8;
            liminal_layer.threshold_floor_q7[ch] =
                branchless_select_u8(any_fired, new_f, liminal_layer.threshold_floor_q7[ch]);
            liminal_layer.aperture_gain_q7[ch] =
                branchless_select_u8(any_fired, new_g, liminal_layer.aperture_gain_q7[ch]);
            ch += 1;
        }

        // 2. Shift Layer 1 (StigmergicGranuleState) autonomy & serendipity threshold
        let mut g = 0usize;
        while g < NUM_STIGMERGIC_GRANULES {
            let boost = self.granule_autonomy_q7[g] >> 4;
            let next_sg = stigmergic_layer.granule_step_gain_q7[g]
                .saturating_add(boost)
                .min(127);
            stigmergic_layer.granule_step_gain_q7[g] =
                branchless_select_u8(any_fired, next_sg, stigmergic_layer.granule_step_gain_q7[g]);
            g += 1;
        }
        let trans_pos = self.transgression_bias_q7 > 0;
        let trans_abs = self.transgression_bias_q7.unsigned_abs();
        let trans_step = trans_abs >> 3;
        stigmergic_layer.serendipity_floor_q7 = branchless_select_u8(
            t_liminal & trans_pos,
            stigmergic_layer
                .serendipity_floor_q7
                .saturating_sub(trans_step)
                .max(8),
            stigmergic_layer.serendipity_floor_q7,
        );

        self.last_intervened_mask = fired_mask;
        self.intervention_epochs = self.intervention_epochs.wrapping_add(u32::from(any_fired));
    }

    /// Branchlessly evaluates active Ego triggers across `stack`'s Layer 0 (`liminal_layer`),
    /// Layer 1 (`stigmergic_layer`), temporal velocity (`delta`), and `venn`, modulating `frame`
    /// in-place and synchronizing `self` onto `stack.ego_layer` after intervention.
    #[inline]
    pub fn intervene_on_layers_and_frame_branchless(
        &mut self,
        stack: &mut AutonomousCoreLayerStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        venn: &StigmergicOverlapSerendipityReport,
        frame: &mut GatewayFrame,
    ) {
        self.intervene_on_sublayers_and_frame_branchless(
            &mut stack.liminal_layer,
            &mut stack.stigmergic_layer,
            delta,
            horizon,
            venn,
            frame,
        );
        stack.ego_layer = *self;
    }
}

/// Composable receptor trait invoked by [`StigmergicSubstrate`] when Swarm consensus and mapped-storage
/// attractor pull elevate a pointer trail's stigmergic pheromone density across an action regime.
///
/// Instead of writing ad-hoc `if/else` semantic branches (e.g., `if needs_web_search { ... }`),
/// external or Cold-Path capabilities (such as asynchronous retrieval, mapped-storage RAG prefetch, or
/// macro-basin consolidation) implement [`StigmergicReceptor`] and are composed into tuples
/// `(ReceptorA, ReceptorB)` that react when `report.target_slot`, `report.dominant_anchor_id`, and
/// `report.action_decision()` cross the physical stigmergic threshold.
pub trait StigmergicReceptor {
    /// Invoked when a [`SubstrateStepReport`] emerges from [`StigmergicSubstrate`] with a
    /// non-passive [`StigmergicActionDecision`].
    ///
    /// # Errors
    /// Returns a [`GcsoResult`] if the downstream receptor encounters an execution or capacity error.
    fn on_stigmergic_emergence(
        &mut self,
        frame: &GatewayFrame,
        report: &SubstrateStepReport,
        trail: &gcso_pointer_trail_t,
    ) -> GcsoResult<()>;
}

impl StigmergicReceptor for () {
    #[inline(always)]
    fn on_stigmergic_emergence(
        &mut self,
        _frame: &GatewayFrame,
        _report: &SubstrateStepReport,
        _trail: &gcso_pointer_trail_t,
    ) -> GcsoResult<()> {
        Ok(())
    }
}

impl<A: StigmergicReceptor, B: StigmergicReceptor> StigmergicReceptor for (A, B) {
    #[inline(always)]
    fn on_stigmergic_emergence(
        &mut self,
        frame: &GatewayFrame,
        report: &SubstrateStepReport,
        trail: &gcso_pointer_trail_t,
    ) -> GcsoResult<()> {
        self.0.on_stigmergic_emergence(frame, report, trail)?;
        self.1.on_stigmergic_emergence(frame, report, trail)
    }
}

// ===================================================================
// 2. `StigmergicSubstrate` -- 2,816-Byte L1D-Resident Micro Passive Carrier
// ===================================================================

impl StigmergicSubstrate {
    /// Drives the isolated sandbox hypothesis lifecycle in zero-alloc mode:
    /// 1. Immediately releases unpinned sandbox lanes on [`StigmergicActionDecision::RepulsionEscape`]
    ///    or [`LayerDistortionKind::ActiveOverdrive`].
    /// 2. Settles an active sandbox lane via [`Self::settle_sandbox_lane`] once consecutive
    ///    coherence improvement reaches `settle_dwell_threshold`.
    /// 3. On sustained assimilation drift (`drift_dwell >= drift_dwell_threshold`), when `ego_active`
    ///    and `exterior_granule_mask != 0` (`(void_exterior_mask | serendipity_excursion_mask) & !venn_core_mask`),
    ///    leases a free lane via [`WorkbenchSlotSpec::auto_sandbox`] masked strictly to `exterior_granule_mask`.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn step_sandbox_hypothesis_lifecycle(
        &mut self,
        telemetry: &mut CoreTelemetryBank,
        active_sandbox_lane: &mut Option<u8>,
        decision: StigmergicActionDecision,
        ego_active: bool,
        exterior_granule_mask: u8,
        candidate_words: &[u64; NUM_STIGMERGIC_GRANULES],
        candidate_weight: f32,
        drift_dwell_threshold: u16,
        settle_dwell_threshold: u16,
    ) -> SandboxLifecycleEvent {
        let distortion = telemetry.compute_distortion_report();
        if matches!(decision, StigmergicActionDecision::RepulsionEscape)
            || matches!(distortion.kind, LayerDistortionKind::ActiveOverdrive)
        {
            let released = self.release_sandbox_lanes();
            if let Some(lane) = *active_sandbox_lane {
                let lane_bit = if usize::from(lane) < MAX_ATTRACTOR_DOMAINS {
                    1u8 << lane
                } else {
                    0
                };
                let still_sandbox = lane_bit != 0
                    && ((self.slot_enabled_mask & self.slot_sandbox_mask) & lane_bit) != 0;
                if !still_sandbox {
                    *active_sandbox_lane = None;
                }
            }
            telemetry.reset_assimilation_dwell();
            telemetry.reset_sandbox_improve_dwell();
            return if released > 0 {
                SandboxLifecycleEvent::Released(released)
            } else {
                SandboxLifecycleEvent::Idle
            };
        }

        if let Some(lane) = *active_sandbox_lane {
            let lane_bit = if usize::from(lane) < MAX_ATTRACTOR_DOMAINS {
                1u8 << lane
            } else {
                0
            };
            let still_sandbox = lane_bit != 0
                && ((self.slot_enabled_mask & self.slot_sandbox_mask) & lane_bit) != 0;
            if !still_sandbox {
                *active_sandbox_lane = None;
            } else if telemetry.sandbox_coherence_improve_dwell() >= settle_dwell_threshold.max(1) {
                if self.settle_sandbox_lane(lane).unwrap_or(false) {
                    *active_sandbox_lane = None;
                    telemetry.reset_assimilation_dwell();
                    telemetry.reset_sandbox_improve_dwell();
                    return SandboxLifecycleEvent::Settled(lane);
                }
            } else {
                return SandboxLifecycleEvent::Idle;
            }
        }

        if active_sandbox_lane.is_none()
            && ego_active
            && exterior_granule_mask != 0
            && telemetry.assimilation_drift_dwell() >= drift_dwell_threshold.max(1)
        {
            let mut masked_words = [0u64; NUM_STIGMERGIC_GRANULES];
            for (g, (dst, &src)) in masked_words
                .iter_mut()
                .zip(candidate_words.iter())
                .enumerate()
            {
                let active_mask = (((exterior_granule_mask >> g) & 1) as u64).wrapping_neg();
                *dst = src & active_mask;
            }
            if swar_words_nonzero_64(&masked_words) {
                let mut phases = [0 as gcso_q7_t; MAX_ATTENTION_HEADS];
                store_q7_words_64(&mut phases, &masked_words);
                let weight =
                    branchless_clamp_f32(branchless_relu_f32(candidate_weight), 0.125, 1.0);
                if let Ok(lane) =
                    self.allocate_lane(&WorkbenchSlotSpec::auto_sandbox(weight, phases))
                {
                    *active_sandbox_lane = Some(lane);
                    telemetry.reset_sandbox_improve_dwell();
                    return SandboxLifecycleEvent::Nucleated(lane);
                }
            }
        }

        SandboxLifecycleEvent::Idle
    }

    /// Executes a complete closed-loop **2-Origin Composite Manifold Gateway $\to$ Stigmergic
    /// Threshold Action Selection $\to$ Online Toroidal $k$-Means / DP-Means Macro Clustering**
    /// step in zero-allocation mode ($C_{\mathrm{alloc}} = 0$).
    ///
    /// Returns `(GatewayFrame, SubstrateStepReport, assigned_centroid_idx, assigned_anchor_id, assigned_coherence, spawned_new_cluster)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `centroids` is empty, `split_coherence_thresh`
    /// is non-finite, or tensor slices are invalid.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn commit_gateway_kmeans_and_select_action(
        &mut self,
        frame: &GatewayFrame,
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        entropy_h_tilde: f32,
        centroids: &mut [CrystallizedAttractorRecord],
        split_coherence_thresh: f32,
        trail_inout: &mut gcso_pointer_trail_t,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(SubstrateStepReport, usize, u32, f32, bool)> {
        if frame.target_anchor_id != 0 {
            trail_inout.target_anchor_id = frame.target_anchor_id;
        }
        trail_inout.attractor_pull_force = branchless_min_f32(
            branchless_max_f32(trail_inout.attractor_pull_force, frame.effective_energy),
            MAX_ATTRACTOR_PULL_FORCE,
        );

        let (assigned_idx, assigned_anchor_id, assigned_coh, _updated_mass, spawned_new) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_frame_with_words_64(
                centroids,
                frame,
                input_words,
                split_coherence_thresh,
                Some(trail_inout),
            );

        let win_w = *centroids[assigned_idx].phase_words_ref();
        let report = self.orchestrate_next_action_step_core(
            gcso_tagged_ptr_t(frame.tagged_ptr),
            entropy_h_tilde,
            frame.dominant_slot,
            trail_inout,
            attention_scores,
            query_tensor,
            key_tensor,
            logits_opt,
            false,
            Some(&win_w),
        )?;

        Ok((
            report,
            assigned_idx,
            assigned_anchor_id,
            assigned_coh,
            spawned_new,
        ))
    }

    /// Executes Gateway synthesis, Online Toroidal $k$-Means / DP-Means clustering, and
    /// Stigmergic action selection in a single `#![no_std]` zero-allocation pass.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `centroids` is empty, `split_coherence_thresh`
    /// is non-finite, or tensor slices are invalid.
    #[inline]
    pub fn orchestrate_gateway_kmeans_and_select_action<S: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        signal: &S,
        entropy_h_tilde: f32,
        centroids: &mut [CrystallizedAttractorRecord],
        split_coherence_thresh: f32,
        trail_inout: &mut gcso_pointer_trail_t,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(GatewayFrame, SubstrateStepReport, usize, u32, f32, bool)> {
        if centroids.is_empty() || !branchless_is_finite_f32(split_coherence_thresh) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (frame, frm_w) = signal.synthesize_gateway_frame_with_words();
        if swar_words_nonzero_64(&frm_w) {
            *trail_phase_words_mut(trail_inout) = frm_w;
        }
        let (report, assigned_idx, assigned_anchor_id, assigned_coh, spawned_new) = self
            .commit_gateway_kmeans_and_select_action(
                &frame,
                &frm_w,
                entropy_h_tilde,
                centroids,
                split_coherence_thresh,
                trail_inout,
                attention_scores,
                query_tensor,
                key_tensor,
                logits_opt,
            )?;

        Ok((
            frame,
            report,
            assigned_idx,
            assigned_anchor_id,
            assigned_coh,
            spawned_new,
        ))
    }

    /// Executes the full **Gateway-Origin Trait Composition $\to$ 8-Cell Micro-Swarm Consensus $\to$
    /// mapped-storage Attractor Basin Pull $\to$ Stigmergic Pheromone Action Selection & Receptor Dispatch**
    /// closed loop without manual hyperparameter arguments or semantic `if/else` task branching.
    ///
    /// 1. Synthesizes `(GatewayFrame, [u64; 8])` from `signal: &S` ([`ManifoldGatewaySignal`]).
    /// 2. Resolves the 8-cell SWAR Micro-Swarm (`try_resolve_action_in_swarm_from_words`) using
    ///    autonomous physical parameters derived directly from the [`GatewayFrame`].
    /// 3. Steps the mapped-storage Online Toroidal $k$-Means / DP-Means attractor basins (`centroids`) and
    ///    evolves `trail_inout`'s stigmergic pheromone density (`orchestrate_next_action_step_core`).
    /// 4. Dispatches `receptor.on_stigmergic_emergence(&frame, &report, trail_inout)` whenever
    ///    a non-passive [`StigmergicActionDecision`] emerges from the trail's accumulated density.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `centroids` is empty or tensor slices are invalid.
    #[inline]
    pub fn orchestrate_gateway_attractor_swarm_step<S, R>(
        &mut self,
        signal: &S,
        entropy_h_tilde: f32,
        centroids: &mut [CrystallizedAttractorRecord],
        trail_inout: &mut gcso_pointer_trail_t,
        receptor: &mut R,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(GatewayFrame, SubstrateStepReport, usize, u32, f32, bool)>
    where
        S: ManifoldGatewaySignal + ?Sized,
        R: StigmergicReceptor + ?Sized,
    {
        if centroids.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (frame, init_w) = signal.synthesize_gateway_frame_with_words();
        let (_swarm_rep, swarm_w) = trail_inout.try_resolve_action_in_swarm_from_words(
            gcso_tagged_ptr_t(frame.tagged_ptr),
            &init_w,
            frame.active_swarm_cell_mask,
            frame.autonomous_swarm_step_gain_u7(),
            frame.qdps_min_step_q7,
            frame.autonomous_swarm_tolerance_q7(),
            frame.autonomous_swarm_quorum_threshold(),
            entropy_h_tilde,
            self.thresholds.inertial_skip_entropy_thresh,
            Some(&self.bloom_filter),
            3,
            96,
        );
        let (report, assigned_idx, assigned_anchor_id, assigned_coh, spawned_new) = self
            .commit_gateway_kmeans_and_select_action(
                &frame,
                &swarm_w,
                entropy_h_tilde,
                centroids,
                DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE,
                trail_inout,
                attention_scores,
                query_tensor,
                key_tensor,
                logits_opt,
            )?;

        if report.action_decision() != StigmergicActionDecision::PassiveSidecar {
            receptor.on_stigmergic_emergence(&frame, &report, trail_inout)?;
        }

        Ok((
            frame,
            report,
            assigned_idx,
            assigned_anchor_id,
            assigned_coh,
            spawned_new,
        ))
    }
}

impl CausalProvenanceSignal for StigmergicSubstrate {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let dec = StigmergicActionDecision::from_u8(self.last_decision);
        CAUSAL_AXIS_SYSTEM_CONSTRAINT
            | branchless_select_u32(
                dec == StigmergicActionDecision::InertialLayerSkip,
                CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE,
                0,
            )
            | branchless_select_u32(
                dec == StigmergicActionDecision::CrystallizeMacroBasin,
                CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED | CAUSAL_AXIS_SUBJECTIVE_PERSONA,
                0,
            )
            | branchless_select_u32(
                dec == StigmergicActionDecision::RepulsionEscape,
                CAUSAL_AXIS_CONTRADICTION_GUARD | CAUSAL_CAUSE_COHOMOLOGY_REPULSION,
                CAUSAL_AXIS_OBJECTIVE_FACT | CAUSAL_AXIS_SUBJECTIVE_PERSONA,
            )
    }

    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        self.last_anomaly_severity
    }
}

impl SystemConstraintSignal for StigmergicSubstrate {
    #[inline(always)]
    fn system_vram_pressure(&self) -> f32 {
        0.0
    }

    #[inline(always)]
    fn system_gpu_frame_overrun(&self) -> f32 {
        0.0
    }

    #[inline(always)]
    fn active_swarm_cells_mask(&self) -> u8 {
        self.slot_enabled_mask
    }

    #[inline(always)]
    fn recommend_layer_skip(&self) -> bool {
        self.last_decision == (StigmergicActionDecision::InertialLayerSkip as u8)
    }

    #[inline(always)]
    fn qdps_min_step_rad_override(&self) -> f32 {
        self.phase_steering.phase_step_min_rad()
    }
}

impl ManifoldGatewaySignal for StigmergicSubstrate {
    #[inline]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let sel_idx = (self.last_selected_trail_id as usize) & (SUBSTRATE_INLINE_TRAILS - 1);
        let latest_idx =
            (self.write_cursor.wrapping_sub(1) as usize) & (SUBSTRATE_INLINE_TRAILS - 1);
        let idx = if (self.active_count > 0) & (self.trails[sel_idx].current_ptr == 0) {
            latest_idx
        } else {
            sel_idx
        };
        let active_trail = &self.trails[idx];
        let words = *trail_phase_words_ref(active_trail);
        let dec = StigmergicActionDecision::from_u8(self.last_decision);
        let mode_flags = dec.action_mode_flags();
        let cur_tp = gcso_tagged_ptr_t(active_trail.current_ptr);
        let dom_slot = branchless_select_u8(
            dec == StigmergicActionDecision::RepulsionEscape,
            ATTRACTOR_SLOT_REPULSION,
            cur_tp.target_attractor_slot(),
        );
        let cause_code = branchless_select_u8(
            dec == StigmergicActionDecision::RepulsionEscape,
            ROOT_CAUSE_COHOMOLOGY_REPULSION,
            branchless_select_u8(
                dec == StigmergicActionDecision::InertialLayerSkip,
                ROOT_CAUSE_SYSTEM_QOS_THROTTLE,
                ROOT_CAUSE_NOMINAL,
            ),
        );
        let qdps_min_rad = self.phase_steering.phase_step_min_rad();
        let num_heads_u8 = (self
            .phase_steering
            .num_heads()
            .min(MAX_ATTENTION_HEADS as u32)) as u8;
        let fact_h = num_heads_u8 >> 1;
        let logic_h = num_heads_u8 >> 2;
        let explore_h = num_heads_u8.saturating_sub(fact_h + logic_h);
        let mut frame = GatewayFrame {
            tagged_ptr: active_trail.current_ptr,
            mapped_section_offset48: cur_tp.unpack_addr(),
            target_anchor_id: active_trail.target_anchor_id,
            effective_energy: branchless_max_f32(
                active_trail.stigmergic_density,
                active_trail.attractor_pull_force,
            ),
            ripa_clamp_rad: self.phase_steering.phase_clamp_max_rad(),
            qdps_min_step_rad: qdps_min_rad,
            effective_inv_temp: 1.0,
            fact_persona_coherence: 1.0 - 2.0 * self.last_anomaly_severity,
            repulsion_severity: self.last_anomaly_severity,
            provenance_bits: self.causal_provenance_bits(),
            dominant_cause_code: cause_code,
            dominant_slot: dom_slot,
            action_mode_flags: mode_flags,
            active_swarm_cell_mask: self.slot_enabled_mask,
            qdps_min_step_q7: branchless_rad_to_phase_step_u8(qdps_min_rad),
            subhead_fact_heads: fact_h,
            subhead_logic_heads: logic_h,
            subhead_explore_heads: explore_h,
            subhead_fact_gain_q7: 128,
            subhead_logic_gain_q7: 128,
            subhead_explore_gain_q7: 128,
            mean_phase_tag_q7: cur_tp.unpack_phase(),
            active_head_count: u16::from(num_heads_u8),
            frame_flags: u16::from(dec == StigmergicActionDecision::RepulsionEscape),
            phase_deltas: [0; MAX_ATTENTION_HEADS],
        };
        *frame.phase_words_mut() = words;
        (frame, words)
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

const _: () = {
    const fn assert_substrate_bounds<
        T: StigmergicSubstrateOps
            + SubstrateGatewayExt
            + GatewayPhaseSteeringExt
            + CausalProvenanceSignal
            + SystemConstraintSignal
            + ManifoldGatewaySignal
            + Pod,
    >() {
    }
    assert_substrate_bounds::<StigmergicSubstrate>();
};

impl CrystallizedAttractorRecord {
    #[inline(always)]
    fn absorb_gateway_frame_with_cur_coh_and_uni_unstamped(
        &mut self,
        cur_w: &[u64; NUM_STIGMERGIC_GRANULES],
        frame: &GatewayFrame,
        frame_words: Indivisible<Grounded, PhaseWordsRef64<'_>>,
        precomputed_coh_opt: Option<f32>,
        uni_flags_opt: Option<(bool, bool)>,
    ) -> ([u64; NUM_STIGMERGIC_GRANULES], f32, bool, bool) {
        if frame.is_repulsed() {
            return (*cur_w, self.phase_gravity_mass, false, false);
        }

        if self.sample_count == 0 {
            if frame.target_anchor_id != 0 {
                self.anchor_id = frame.target_anchor_id;
            }
            if frame.mapped_section_offset48 != 0 {
                self.mapped_section_offset48 = frame.mapped_section_offset48;
            }
            self.dominant_slot = frame.dominant_slot;
            self.ripa_clamp_rad = frame.ripa_clamp_rad;
            self.effective_inv_temp = frame.effective_inv_temp;
        } else {
            if self.ripa_clamp_rad.to_bits() != frame.ripa_clamp_rad.to_bits() {
                self.ripa_clamp_rad =
                    branchless_lerp_f32(self.ripa_clamp_rad, frame.ripa_clamp_rad, 0.25);
            }
            if self.effective_inv_temp.to_bits() != frame.effective_inv_temp.to_bits() {
                self.effective_inv_temp =
                    branchless_lerp_f32(self.effective_inv_temp, frame.effective_inv_temp, 0.25);
            }
        }
        self.action_mode_flags = frame.action_mode_flags;
        self.active_swarm_mask = frame.active_swarm_cell_mask;
        self.provenance_bits |= frame.provenance_bits;

        let (next_w, mass, words_changed) = self
            .absorb_sample_words_with_cur_coh_and_uni_unstamped(
                cur_w,
                frame_words,
                precomputed_coh_opt,
                uni_flags_opt,
                frame.effective_energy,
                1,
            );
        (next_w, mass, true, words_changed)
    }

    /// Absorbs a [`GatewayFrame`] into this [`CrystallizedAttractorRecord`] and returns `phase_gravity_mass`.
    #[inline]
    pub fn absorb_gateway_frame(&mut self, frame: &GatewayFrame) -> f32 {
        let grounded_w = match Scoped::from_boundary(PhaseWordsRef64(frame.phase_words_ref()))
            .cut_by_lacuna(!frame.is_repulsed(), frame.repulsion_severity, 0.50)
        {
            ScopedCut::SelfRef(aligned) => aligned.into_grounded(),
            _ => return self.phase_gravity_mass,
        };
        let prev_crc = self.checksum_crc32;
        let prev_aid = self.anchor_id;
        let prev_count = self.sample_count;
        let prev_zoff = self.mapped_section_offset48;
        let prev_be_bits = self.base_energy.to_bits();
        let prev_rb_bits = self.resultant_r_bar.to_bits();
        let prev_kp_bits = self.von_mises_kappa.to_bits();
        let prev_gm_bits = self.phase_gravity_mass.to_bits();
        let prev_clamp_bits = self.ripa_clamp_rad.to_bits();
        let prev_temp_bits = self.effective_inv_temp.to_bits();
        let prev_prov = self.provenance_bits;
        let prev_w7_lo = self.pack_w7_lo();
        let cur_w = *self.phase_words_ref();
        let (next_w, mass, touched, words_changed) = self
            .absorb_gateway_frame_with_cur_coh_and_uni_unstamped(
                &cur_w, frame, grounded_w, None, None,
            );
        if touched {
            self.stamp_or_update_absorption_crc32(
                prev_crc,
                prev_aid,
                prev_count,
                prev_zoff,
                prev_be_bits,
                prev_rb_bits,
                prev_kp_bits,
                prev_gm_bits,
                prev_clamp_bits,
                prev_temp_bits,
                prev_prov,
                prev_w7_lo,
                &cur_w,
                &next_w,
                words_changed,
            );
        }
        mass
    }

    #[inline(always)]
    fn select_or_spawn_centroid_slot_64(
        centroids: &mut [Self],
        frame: &GatewayFrame,
        nearest_idx: usize,
        best_coh: f32,
        empty_opt: Option<usize>,
        weakest_idx: usize,
        best_w: [u64; NUM_STIGMERGIC_GRANULES],
        safe_split: f32,
    ) -> (usize, bool, [u64; NUM_STIGMERGIC_GRANULES]) {
        let needs_split = (centroids[nearest_idx].sample_count == 0) | (best_coh < safe_split);
        if needs_split {
            if let Some(empty_i) = empty_opt {
                let assigned_id = if frame.target_anchor_id != 0 {
                    frame.target_anchor_id
                } else {
                    (empty_i as u32) + 1
                };
                centroids[empty_i].init_spawned_slot_header_in_place(
                    assigned_id,
                    frame.dominant_slot,
                    frame.mapped_section_offset48,
                );
                (empty_i, true, [0u64; NUM_STIGMERGIC_GRANULES])
            } else if frame.effective_energy > centroids[weakest_idx].phase_gravity_mass {
                let assigned_id = if frame.target_anchor_id != 0 {
                    frame.target_anchor_id
                } else {
                    (weakest_idx as u32) + 1
                };
                centroids[weakest_idx].init_spawned_slot_header_in_place(
                    assigned_id,
                    frame.dominant_slot,
                    frame.mapped_section_offset48,
                );
                (weakest_idx, true, [0u64; NUM_STIGMERGIC_GRANULES])
            } else {
                (nearest_idx, false, best_w)
            }
        } else {
            (nearest_idx, false, best_w)
        }
    }

    #[inline(always)]
    fn commit_frame_absorption_at_index(
        centroids: &mut [Self],
        target_idx: usize,
        spawned_new: bool,
        best_coh: f32,
        best_is_uniform: bool,
        sample_is_uniform: bool,
        cur_target_w: &[u64; NUM_STIGMERGIC_GRANULES],
        frame: &GatewayFrame,
        sample_w: Indivisible<Grounded, PhaseWordsRef64<'_>>,
        trail_opt: Option<&mut gcso_pointer_trail_t>,
    ) -> (u32, f32, f32) {
        let pre_coh = Some(branchless_select_f32(spawned_new, 1.0, best_coh));
        let eff_best_uni = spawned_new | best_is_uniform;
        let uni_flags = Some((eff_best_uni, sample_is_uniform));
        let fallback_aid = (target_idx as u32) + 1;
        let target = &mut centroids[target_idx];
        let prev_crc = target.checksum_crc32;
        let prev_count = target.sample_count;
        let prev_aid = branchless_select_u32(spawned_new, 0, target.anchor_id);
        let prev_zoff = branchless_select_u64(spawned_new, 0, target.mapped_section_offset48);
        let prev_be_bits = target.base_energy.to_bits();
        let prev_gm_bits = target.phase_gravity_mass.to_bits();
        let prev_prov = target.provenance_bits;
        let prev_rb_bits = target.resultant_r_bar.to_bits();
        let prev_kp_bits = target.von_mises_kappa.to_bits();
        let prev_clamp_bits = target.ripa_clamp_rad.to_bits();
        let prev_temp_bits = target.effective_inv_temp.to_bits();
        let prev_w7_lo = target.pack_w7_lo();
        let (next_w, updated_mass, _modified, words_changed) = target
            .absorb_gateway_frame_with_cur_coh_and_uni_unstamped(
                cur_target_w,
                frame,
                sample_w,
                pre_coh,
                uni_flags,
            );
        let winning_anchor_id =
            branchless_select_u32(target.anchor_id != 0, target.anchor_id, fallback_aid);
        target.anchor_id = winning_anchor_id;
        let eff_prev_count = branchless_select_u32(spawned_new, 0, prev_count);
        target.stamp_or_update_absorption_crc32(
            prev_crc,
            prev_aid,
            eff_prev_count,
            prev_zoff,
            prev_be_bits,
            prev_rb_bits,
            prev_kp_bits,
            prev_gm_bits,
            prev_clamp_bits,
            prev_temp_bits,
            prev_prov,
            prev_w7_lo,
            cur_target_w,
            &next_w,
            words_changed,
        );
        let final_coh = branchless_select_f32(spawned_new, 1.0, best_coh);

        if let Some(trail) = trail_opt {
            trail.target_anchor_id = winning_anchor_id;
            trail.cluster_id = winning_anchor_id;
            trail.flags |= TRAIL_FLAG_CRYSTALLIZED_CANDIDATE;
        }
        (winning_anchor_id, final_coh, updated_mass)
    }

    #[inline]
    pub(crate) fn step_online_toroidal_kmeans_frame_with_words_64(
        centroids: &mut [Self],
        frame: &GatewayFrame,
        sample_w: &[u64; NUM_STIGMERGIC_GRANULES],
        split_coherence_thresh: f32,
        trail_opt: Option<&mut gcso_pointer_trail_t>,
    ) -> (usize, u32, f32, f32, bool) {
        let k_len = centroids.len();
        if k_len == 0 {
            return (0, frame.target_anchor_id, 0.0, 0.0, false);
        }

        let (
            nearest_idx,
            best_coh,
            _best_grav,
            empty_opt,
            weakest_idx,
            best_w,
            sample_is_uniform,
            best_is_uniform,
            _,
        ) = Self::find_nearest_toroidal_centroid_with_cache_64(centroids, sample_w, None);

        let grounded_w = match Scoped::from_boundary(PhaseWordsRef64(sample_w)).cut_by_lacuna(
            !frame.is_repulsed(),
            frame.repulsion_severity,
            0.50,
        ) {
            ScopedCut::SelfRef(aligned) => aligned.into_grounded(),
            _ => {
                let dom_id = centroids[nearest_idx].anchor_id;
                return (
                    nearest_idx,
                    dom_id,
                    best_coh,
                    centroids[nearest_idx].phase_gravity_mass,
                    false,
                );
            }
        };

        let safe_split = branchless_clamp_f32(split_coherence_thresh, -0.95, 0.95);
        let (target_idx, spawned_new, cur_target_w) = Self::select_or_spawn_centroid_slot_64(
            centroids,
            frame,
            nearest_idx,
            best_coh,
            empty_opt,
            weakest_idx,
            best_w,
            safe_split,
        );
        let (winning_anchor_id, final_coh, updated_mass) = Self::commit_frame_absorption_at_index(
            centroids,
            target_idx,
            spawned_new,
            best_coh,
            best_is_uniform,
            sample_is_uniform,
            &cur_target_w,
            frame,
            grounded_w,
            trail_opt,
        );

        (
            target_idx,
            winning_anchor_id,
            final_coh,
            updated_mass,
            spawned_new,
        )
    }
}

impl CrystallizedAttractorRecord {
    /// Branchless 3-stage **Dissipative Overflow -> Residual Scatter -> Autonomous Daughter Birth** step:
    /// 1. Evaluates saturation & dispersion pressure (`1 - resultant_r_bar`, `sample_count`) on the nearest attractor $A_1$.
    /// 2. If $A_1$ overflows (`effective_coh < split_coherence_thresh`), peels $A_1$'s phase from `sample_w` via SWAR circular subtraction and scans remaining active slots for a secondary basin $A_2$ without branches.
    /// 3. If the overflow residual settles on $A_2$, absorbs into $A_2$ (`scattered = true`); otherwise autonomously spawns a daughter attractor into an empty or weaker slot (`spawned_new = true`).
    #[inline]
    pub fn step_dissipative_overflow_scatter_or_spawn_64(
        centroids: &mut [Self],
        frame: &GatewayFrame,
        sample_w: &[u64; NUM_STIGMERGIC_GRANULES],
        split_coherence_thresh: f32,
        trail_opt: Option<&mut gcso_pointer_trail_t>,
    ) -> DissipativeOverflowReport {
        let k_len = centroids.len();
        if k_len == 0 {
            return DissipativeOverflowReport::default();
        }

        let is_repulsed = (frame.repulsion_severity > 0.50)
            | ((frame.action_mode_flags & ACTION_MODE_PHASE_REPULSE) != 0)
            | (frame.dominant_cause_code == ROOT_CAUSE_COHOMOLOGY_REPULSION)
            | ((frame.provenance_bits & CAUSAL_CAUSE_COHOMOLOGY_REPULSION) != 0);
        let (idx_1, coh_1, _grav_1, empty_opt, weakest_idx, w_1, sample_uni, best_1_uni, _) =
            Self::find_nearest_toroidal_centroid_with_cache_64(centroids, sample_w, None);
        let primary_aid = centroids[idx_1].anchor_id;
        let primary_count = centroids[idx_1].sample_count;
        let primary_mass = centroids[idx_1].phase_gravity_mass;
        let r_bar_1 = branchless_clamp_f32(centroids[idx_1].resultant_r_bar, 0.0, 1.0);
        let overflow_pressure = branchless_clamp_f32(
            (1.0 - r_bar_1) * 0.35 + fast_ln_1p_bounded_f32(primary_count as f32) * 0.06,
            0.0,
            0.85,
        );
        let res_w = [
            swar_q7_circular_diff_x8(sample_w[0], w_1[0]),
            swar_q7_circular_diff_x8(sample_w[1], w_1[1]),
            swar_q7_circular_diff_x8(sample_w[2], w_1[2]),
            swar_q7_circular_diff_x8(sample_w[3], w_1[3]),
            swar_q7_circular_diff_x8(sample_w[4], w_1[4]),
            swar_q7_circular_diff_x8(sample_w[5], w_1[5]),
            swar_q7_circular_diff_x8(sample_w[6], w_1[6]),
            swar_q7_circular_diff_x8(sample_w[7], w_1[7]),
        ];

        if is_repulsed {
            return DissipativeOverflowReport {
                residual_words: res_w,
                primary_anchor_id: primary_aid,
                landed_anchor_id: primary_aid,
                primary_coherence: coh_1,
                scatter_coherence: 0.0,
                overflow_pressure,
                updated_gravity_mass: primary_mass,
                primary_idx: idx_1 as u16,
                landed_idx: idx_1 as u16,
                overflowed: false,
                scattered: false,
                spawned_new: false,
                _reserved: 0,
            };
        }

        let safe_split = branchless_clamp_f32(split_coherence_thresh, -0.95, 0.95);
        let has_primary = primary_count > 0;
        let overflowed = has_primary & ((coh_1 - overflow_pressure) < safe_split);

        // Branchless secondary scatter scan over `res_w` across all slots `i != idx_1`.
        let res_uni = is_word_array_uniform_64(&res_w);
        let mut best_2_idx = idx_1;
        let mut best_2_coh = -2.0_f32;
        let mut best_2_uni = false;
        let mut any_2 = false;
        let mut best_2_w = [0u64; NUM_STIGMERGIC_GRANULES];
        let mut i = 0usize;
        while i < k_len {
            let rec = &centroids[i];
            let eligible = (i != idx_1)
                & (rec.sample_count > 0)
                & (rec.phase_gravity_mass > DEFAULT_ENTROPY_EPS);
            let cw = rec.phase_words_ref();
            let c_uni = is_word_array_uniform_64(cw);
            let raw_coh =
                coherence_from_word_arrays_with_both_uniform_flags_64(&res_w, res_uni, cw, c_uni);
            let eff_coh = branchless_select_f32(eligible, raw_coh, -2.0);
            let take = eligible & ((!any_2) | (eff_coh > best_2_coh));
            best_2_idx = branchless_select_usize(take, i, best_2_idx);
            best_2_coh = branchless_select_f32(take, eff_coh, best_2_coh);
            best_2_uni = (take & c_uni) | ((!take) & best_2_uni);
            best_2_w = [
                branchless_select_u64(take, cw[0], best_2_w[0]),
                branchless_select_u64(take, cw[1], best_2_w[1]),
                branchless_select_u64(take, cw[2], best_2_w[2]),
                branchless_select_u64(take, cw[3], best_2_w[3]),
                branchless_select_u64(take, cw[4], best_2_w[4]),
                branchless_select_u64(take, cw[5], best_2_w[5]),
                branchless_select_u64(take, cw[6], best_2_w[6]),
                branchless_select_u64(take, cw[7], best_2_w[7]),
            ];
            any_2 |= eligible;
            i += 1;
        }
        let clean_scatter_coh = branchless_select_f32(any_2, best_2_coh, 0.0);

        let settled_on_scatter = overflowed & any_2 & (best_2_coh >= safe_split);
        let needs_spawn = (!has_primary) | (overflowed & !settled_on_scatter);
        let (has_empty, empty_i) = match empty_opt {
            Some(idx) => (true, idx),
            None => (false, 0),
        };
        let can_evict =
            (!has_empty) & (frame.effective_energy > centroids[weakest_idx].phase_gravity_mass);
        let spawn_slot = branchless_select_usize(has_empty, empty_i, weakest_idx);
        let effective_spawn = needs_spawn & (has_empty | can_evict);

        let target_idx = branchless_select_usize(
            settled_on_scatter,
            best_2_idx,
            branchless_select_usize(effective_spawn, spawn_slot, idx_1),
        );
        let assigned_spawn_id = branchless_select_u32(
            frame.target_anchor_id != 0,
            frame.target_anchor_id,
            (target_idx as u32) + 1,
        );

        let peel_for_daughter = effective_spawn & overflowed & (coh_1 >= safe_split);
        let use_res_words = settled_on_scatter | peel_for_daughter;
        let absorbed_w = [
            branchless_select_u64(use_res_words, res_w[0], sample_w[0]),
            branchless_select_u64(use_res_words, res_w[1], sample_w[1]),
            branchless_select_u64(use_res_words, res_w[2], sample_w[2]),
            branchless_select_u64(use_res_words, res_w[3], sample_w[3]),
            branchless_select_u64(use_res_words, res_w[4], sample_w[4]),
            branchless_select_u64(use_res_words, res_w[5], sample_w[5]),
            branchless_select_u64(use_res_words, res_w[6], sample_w[6]),
            branchless_select_u64(use_res_words, res_w[7], sample_w[7]),
        ];
        let grounded_absorbed_w = match Scoped::from_boundary(PhaseWordsRef64(&absorbed_w))
            .cut_by_lacuna(!frame.is_repulsed(), frame.repulsion_severity, 0.50)
        {
            ScopedCut::SelfRef(aligned) => aligned.into_grounded(),
            _ => {
                return DissipativeOverflowReport {
                    residual_words: res_w,
                    primary_anchor_id: primary_aid,
                    landed_anchor_id: primary_aid,
                    primary_coherence: coh_1,
                    scatter_coherence: clean_scatter_coh,
                    overflow_pressure,
                    updated_gravity_mass: primary_mass,
                    primary_idx: idx_1 as u16,
                    landed_idx: idx_1 as u16,
                    overflowed: false,
                    scattered: false,
                    spawned_new: false,
                    _reserved: 0,
                };
            }
        };
        if effective_spawn {
            centroids[target_idx].init_spawned_slot_header_in_place(
                assigned_spawn_id,
                frame.dominant_slot,
                frame.mapped_section_offset48,
            );
        }

        let absorbed_uni = (use_res_words & res_uni) | ((!use_res_words) & sample_uni);
        let cur_target_w = [
            branchless_select_u64(
                effective_spawn,
                0,
                branchless_select_u64(settled_on_scatter, best_2_w[0], w_1[0]),
            ),
            branchless_select_u64(
                effective_spawn,
                0,
                branchless_select_u64(settled_on_scatter, best_2_w[1], w_1[1]),
            ),
            branchless_select_u64(
                effective_spawn,
                0,
                branchless_select_u64(settled_on_scatter, best_2_w[2], w_1[2]),
            ),
            branchless_select_u64(
                effective_spawn,
                0,
                branchless_select_u64(settled_on_scatter, best_2_w[3], w_1[3]),
            ),
            branchless_select_u64(
                effective_spawn,
                0,
                branchless_select_u64(settled_on_scatter, best_2_w[4], w_1[4]),
            ),
            branchless_select_u64(
                effective_spawn,
                0,
                branchless_select_u64(settled_on_scatter, best_2_w[5], w_1[5]),
            ),
            branchless_select_u64(
                effective_spawn,
                0,
                branchless_select_u64(settled_on_scatter, best_2_w[6], w_1[6]),
            ),
            branchless_select_u64(
                effective_spawn,
                0,
                branchless_select_u64(settled_on_scatter, best_2_w[7], w_1[7]),
            ),
        ];
        let target_coh = branchless_select_f32(settled_on_scatter, best_2_coh, coh_1);
        let target_uni = (settled_on_scatter & best_2_uni) | ((!settled_on_scatter) & best_1_uni);
        let (landed_aid, _final_coh, updated_mass) = Self::commit_frame_absorption_at_index(
            centroids,
            target_idx,
            effective_spawn,
            target_coh,
            target_uni,
            absorbed_uni,
            &cur_target_w,
            frame,
            grounded_absorbed_w,
            trail_opt,
        );

        DissipativeOverflowReport {
            residual_words: res_w,
            primary_anchor_id: primary_aid,
            landed_anchor_id: landed_aid,
            primary_coherence: coh_1,
            scatter_coherence: clean_scatter_coh,
            overflow_pressure,
            updated_gravity_mass: updated_mass,
            primary_idx: idx_1 as u16,
            landed_idx: target_idx as u16,
            overflowed,
            scattered: settled_on_scatter,
            spawned_new: effective_spawn,
            _reserved: 0,
        }
    }

    /// Evaluates [`step_dissipative_overflow_scatter_or_spawn_64`](Self::step_dissipative_overflow_scatter_or_spawn_64)
    /// directly from any [`ManifoldGatewaySignal`] implementor.
    #[inline]
    pub fn step_dissipative_overflow_signal_64<S: ManifoldGatewaySignal + ?Sized>(
        centroids: &mut [Self],
        signal: &S,
        split_coherence_thresh: f32,
        trail_opt: Option<&mut gcso_pointer_trail_t>,
    ) -> (GatewayFrame, DissipativeOverflowReport) {
        let (frame, frm_w) = signal.synthesize_gateway_frame_with_words();
        let report = Self::step_dissipative_overflow_scatter_or_spawn_64(
            centroids,
            &frame,
            &frm_w,
            split_coherence_thresh,
            trail_opt,
        );
        (frame, report)
    }

    /// Autonomously nucleates a new attractor inside `centroids` when approximation noise in
    /// [`StigmergicOverlapSerendipityReport`] pushes a fragmented granule outside all existing
    /// attractors' Venn apertures into `Liminal` (`serendipity_triggered != 0`).
    ///
    /// Returns `(spawned_slot_idx, spawned_anchor_id, did_nucleate)` in branchless constant time
    /// relative to centroid slot contents.
    #[inline]
    pub fn nucleate_serendipitous_attractor_from_void_64(
        centroids: &mut [Self],
        venn_report: &StigmergicOverlapSerendipityReport,
        drifted_words: &[u64; NUM_STIGMERGIC_GRANULES],
        new_anchor_id: u32,
        trail_opt: Option<&mut gcso_pointer_trail_t>,
    ) -> (usize, u32, bool) {
        if (venn_report.serendipity_triggered == 0) || centroids.is_empty() {
            return (0, 0, false);
        }
        let mut empty_idx = 0usize;
        let mut has_empty = false;
        let mut weakest_idx = 0usize;
        let mut weakest_mass = f32::MAX;
        let mut i = 0usize;
        while i < centroids.len() {
            let rec = &centroids[i];
            let is_empty = rec.sample_count == 0;
            let take_empty = (!has_empty) & is_empty;
            empty_idx = branchless_select_usize(take_empty, i, empty_idx);
            has_empty |= is_empty;

            let take_weak = (!is_empty) & (rec.phase_gravity_mass < weakest_mass);
            weakest_idx = branchless_select_usize(take_weak, i, weakest_idx);
            weakest_mass = branchless_select_f32(take_weak, rec.phase_gravity_mass, weakest_mass);
            i += 1;
        }

        let seed_energy = branchless_max_f32(venn_report.void_excursion_energy, 0.25);
        let can_evict = (!has_empty) & (seed_energy > weakest_mass);
        if !(has_empty | can_evict) {
            return (0, 0, false);
        }
        let grounded_w = match Scoped::from_delta(PhaseWordsRef64(drifted_words)).cut_by_lacuna_q7(
            venn_report.serendipity_triggered != 0,
            127,
            0,
        ) {
            ScopedCut::Exterior(spec) => {
                match spec.bridge_to_shared(venn_report.serendipity_excursion_mask | 1) {
                    Ok(bridged) => bridged.into_grounded(),
                    Err(_) => return (0, 0, false),
                }
            }
            _ => return (0, 0, false),
        };
        let target_idx = branchless_select_usize(has_empty, empty_idx, weakest_idx);
        let eff_anchor_id =
            branchless_select_u32(new_anchor_id != 0, new_anchor_id, (target_idx as u32) + 1);

        let mut seed_frame = GatewayFrame {
            target_anchor_id: eff_anchor_id,
            effective_energy: seed_energy,
            ripa_clamp_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
            qdps_min_step_rad: DEFAULT_PHASE_STEP_MIN_RAD,
            effective_inv_temp: 1.0,
            provenance_bits: CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
            dominant_slot: ATTRACTOR_SLOT_CRYSTALLIZED,
            action_mode_flags: ACTION_PRESET_EXPRESS_PHASE,
            active_swarm_cell_mask: venn_report.serendipity_excursion_mask | 0x01,
            ..GatewayFrame::default()
        };
        *seed_frame.phase_words_mut() = *drifted_words;
        seed_frame.mean_phase_tag_q7 =
            swar_q7_mean_i8_from_words_with_uniform_flag_64(drifted_words, false);

        centroids[target_idx].init_spawned_slot_header_in_place(
            eff_anchor_id,
            ATTRACTOR_SLOT_CRYSTALLIZED,
            0,
        );
        let empty_cur = [0u64; NUM_STIGMERGIC_GRANULES];
        let (final_aid, _coh, _mass) = Self::commit_frame_absorption_at_index(
            centroids,
            target_idx,
            true,
            1.0,
            true,
            is_word_array_uniform_64(drifted_words),
            &empty_cur,
            &seed_frame,
            grounded_w,
            trail_opt,
        );
        (target_idx, final_aid, true)
    }

    /// **Classical Online Toroidal $k$-Means / DP-Means Closed-Loop E+M Step ($\mathcal{O}(K)$)**:
    /// Returns `(assigned_cluster_idx, assigned_anchor_id, centroid_coherence, updated_gravity_mass, spawned_new_cluster)`.
    #[inline(always)]
    pub fn step_online_toroidal_kmeans_frame_64(
        centroids: &mut [Self],
        frame: &GatewayFrame,
        split_coherence_thresh: f32,
        trail_opt: Option<&mut gcso_pointer_trail_t>,
    ) -> (usize, u32, f32, f32, bool) {
        Self::step_online_toroidal_kmeans_frame_with_words_64(
            centroids,
            frame,
            frame.phase_words_ref(),
            split_coherence_thresh,
            trail_opt,
        )
    }

    /// Executes Online Toroidal $k$-Means / DP-Means across a batch of [`GatewayFrame`]s,
    /// returning `(last_assigned_idx, last_assigned_anchor_id, mean_coherence, spawned_clusters_count)`.
    #[inline]
    pub fn step_online_toroidal_kmeans_batch_frames_64(
        centroids: &mut [Self],
        frames: &[GatewayFrame],
        split_coherence_thresh: f32,
        mut trail_opt: Option<&mut gcso_pointer_trail_t>,
    ) -> (usize, u32, f32, u32) {
        if centroids.is_empty() || frames.is_empty() {
            return (0, 0, 0.0, 0);
        }
        let mut valid_count = 0u32;
        let mut spawned_count = 0u32;
        let mut coh_sum = 0.0_f32;
        let mut last_idx = 0usize;
        let mut last_aid = 0u32;
        for frame in frames {
            if frame.is_repulsed() {
                continue;
            }
            let (idx, aid, coh, _mass, spawned) =
                Self::step_online_toroidal_kmeans_frame_with_words_64(
                    centroids,
                    frame,
                    frame.phase_words_ref(),
                    split_coherence_thresh,
                    trail_opt.as_deref_mut(),
                );
            valid_count += 1;
            if spawned {
                spawned_count += 1;
            }
            coh_sum += coh;
            last_idx = idx;
            last_aid = aid;
        }
        let mean_coh = if valid_count > 0 {
            branchless_clamp_f32(
                coh_sum * fast_inv_pos_normal_f32(valid_count as f32),
                -1.0,
                1.0,
            )
        } else {
            0.0
        };
        (last_idx, last_aid, mean_coh, spawned_count)
    }

    /// **Online Toroidal $k$-Means / DP-Means on Any [`ManifoldGatewaySignal`]**:
    /// Synthesizes `signal` into a [`GatewayFrame`] and executes E-step + M-step over `centroids`.
    #[inline]
    pub fn step_online_toroidal_kmeans_signal_64<S: ManifoldGatewaySignal + ?Sized>(
        centroids: &mut [Self],
        signal: &S,
        split_coherence_thresh: f32,
        trail_opt: Option<&mut gcso_pointer_trail_t>,
    ) -> (GatewayFrame, usize, u32, f32, f32, bool) {
        let (frame, frm_w) = signal.synthesize_gateway_frame_with_words();
        let (cluster_idx, anchor_id, coh, mass, spawned) =
            Self::step_online_toroidal_kmeans_frame_with_words_64(
                centroids,
                &frame,
                &frm_w,
                split_coherence_thresh,
                trail_opt,
            );
        (frame, cluster_idx, anchor_id, coh, mass, spawned)
    }
}

impl CausalProvenanceSignal for CrystallizedAttractorRecord {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let is_mature = (self.sample_count >= 2) & (self.resultant_r_bar >= 0.45);
        self.provenance_bits
            | branchless_select_u32(is_mature, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, 0)
    }
}

impl CognitiveGroundingSignal for CrystallizedAttractorRecord {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.phase_profile
    }
    #[inline(always)]
    fn grounding_phases_ref_64(&self) -> Option<&[gcso_q7_t; MAX_ATTENTION_HEADS]> {
        Some(&self.phase_profile)
    }
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        *self.phase_words_ref()
    }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        self.phase_gravity_mass
    }
    #[inline(always)]
    fn mapped_section_offset48(&self) -> u64 {
        self.mapped_section_offset48
    }
    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 {
        self.anchor_id
    }
}

impl ObjectiveFactSignal for CrystallizedAttractorRecord {
    #[inline(always)]
    fn fact_ripa_clamp_rad(&self) -> f32 {
        self.ripa_clamp_rad
    }
}

impl SubjectivePersonaSignal for CrystallizedAttractorRecord {
    #[inline(always)]
    fn persona_ripa_clamp_rad(&self) -> f32 {
        self.ripa_clamp_rad
    }
}

impl ManifoldGatewaySignal for CrystallizedAttractorRecord {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let is_active = (self.sample_count > 0) & (self.phase_gravity_mass > DEFAULT_ENTROPY_EPS);
        if !is_active {
            let packed_tag =
                pack_action_slot_tag(self.dominant_slot, ACTION_PRESET_PASSIVE_SIDECAR);
            return (
                GatewayFrame {
                    tagged_ptr: gcso_tagged_ptr_t::pack(
                        self.mapped_section_offset48,
                        packed_tag,
                        0,
                    )
                    .raw(),
                    mapped_section_offset48: self.mapped_section_offset48,
                    target_anchor_id: self.anchor_id,
                    dominant_slot: self.dominant_slot,
                    action_mode_flags: ACTION_PRESET_PASSIVE_SIDECAR,
                    active_swarm_cell_mask: self.active_swarm_mask,
                    dominant_cause_code: self.dominant_cause_code,
                    frame_flags: packed_tag as u16,
                    ..GatewayFrame::default()
                },
                [0u64; NUM_STIGMERGIC_GRANULES],
            );
        }
        let is_mature = (self.sample_count >= 2) & (self.resultant_r_bar >= 0.45);
        let w = *self.phase_words_ref();
        let tag_q7 = fast_mean_q7_tag_from_words_64(&w);
        let eff_action = self.action_mode_flags;
        let packed_tag = pack_action_slot_tag(self.dominant_slot, eff_action);
        let tagged_ptr =
            gcso_tagged_ptr_t::pack(self.mapped_section_offset48, packed_tag, tag_q7).raw();
        let prov_bits = self.provenance_bits
            | branchless_select_u32(is_mature, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, 0);

        let is_fact_slot = self.dominant_slot == ATTRACTOR_SLOT_KNOWLEDGE;
        let is_persona_slot = self.dominant_slot == ATTRACTOR_SLOT_PERSONA;
        let subhead_fact = branchless_select_u8(
            is_fact_slot,
            36,
            branchless_select_u8(is_persona_slot, 16, 24),
        );
        let subhead_logic = branchless_select_u8(is_fact_slot, 20, 24);
        let subhead_explore = branchless_select_u8(
            is_fact_slot,
            8,
            branchless_select_u8(is_persona_slot, 24, 16),
        );
        let fact_gain_q7 = branchless_select_u8(is_fact_slot, 147, 128);
        let explore_gain_q7 = branchless_select_u8(is_persona_slot, 154, 128);
        let logic_gain_q7 = ((fact_gain_q7 as u16 + explore_gain_q7 as u16) >> 1) as u8;

        (
            GatewayFrame {
                tagged_ptr,
                mapped_section_offset48: self.mapped_section_offset48,
                target_anchor_id: self.anchor_id,
                effective_energy: branchless_min_f32(
                    self.phase_gravity_mass,
                    MAX_ATTRACTOR_PULL_FORCE,
                ),
                ripa_clamp_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
                qdps_min_step_rad: DEFAULT_PHASE_STEP_MIN_RAD,
                effective_inv_temp: self.effective_inv_temp,
                fact_persona_coherence: self.resultant_r_bar,
                repulsion_severity: 0.0,
                provenance_bits: prov_bits,
                dominant_cause_code: self.dominant_cause_code,
                dominant_slot: self.dominant_slot,
                action_mode_flags: eff_action,
                active_swarm_cell_mask: self.active_swarm_mask,
                qdps_min_step_q7: branchless_rad_to_phase_step_u8(DEFAULT_PHASE_STEP_MIN_RAD),
                subhead_fact_heads: subhead_fact,
                subhead_logic_heads: subhead_logic,
                subhead_explore_heads: subhead_explore,
                subhead_fact_gain_q7: fact_gain_q7,
                subhead_logic_gain_q7: logic_gain_q7,
                subhead_explore_gain_q7: explore_gain_q7,
                mean_phase_tag_q7: tag_q7,
                active_head_count: MAX_ATTENTION_HEADS as u16,
                frame_flags: packed_tag as u16,
                phase_deltas: self.phase_profile,
            },
            w,
        )
    }

    #[inline(always)]
    fn synthesize_gateway_ensemble_item(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let is_active = (self.sample_count > 0) & (self.phase_gravity_mass > DEFAULT_ENTROPY_EPS);
        if !is_active {
            return (
                GatewayFrame {
                    mapped_section_offset48: self.mapped_section_offset48,
                    target_anchor_id: self.anchor_id,
                    dominant_slot: self.dominant_slot,
                    action_mode_flags: ACTION_PRESET_PASSIVE_SIDECAR,
                    active_swarm_cell_mask: self.active_swarm_mask,
                    dominant_cause_code: self.dominant_cause_code,
                    ..GatewayFrame::default()
                },
                [0u64; NUM_STIGMERGIC_GRANULES],
            );
        }
        let is_mature = (self.sample_count >= 2) & (self.resultant_r_bar >= 0.45);
        let w = *self.phase_words_ref();
        let prov_bits = self.provenance_bits
            | branchless_select_u32(is_mature, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, 0);
        let is_fact_slot = self.dominant_slot == ATTRACTOR_SLOT_KNOWLEDGE;
        let is_persona_slot = self.dominant_slot == ATTRACTOR_SLOT_PERSONA;
        let subhead_fact = branchless_select_u8(
            is_fact_slot,
            36,
            branchless_select_u8(is_persona_slot, 16, 24),
        );
        let subhead_logic = branchless_select_u8(is_fact_slot, 20, 24);
        let subhead_explore = branchless_select_u8(
            is_fact_slot,
            8,
            branchless_select_u8(is_persona_slot, 24, 16),
        );
        let fact_gain_q7 = branchless_select_u8(is_fact_slot, 147, 128);
        let explore_gain_q7 = branchless_select_u8(is_persona_slot, 154, 128);
        let logic_gain_q7 = ((fact_gain_q7 as u16 + explore_gain_q7 as u16) >> 1) as u8;

        (
            GatewayFrame {
                tagged_ptr: 0,
                mapped_section_offset48: self.mapped_section_offset48,
                target_anchor_id: self.anchor_id,
                effective_energy: branchless_min_f32(
                    self.phase_gravity_mass,
                    MAX_ATTRACTOR_PULL_FORCE,
                ),
                ripa_clamp_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
                qdps_min_step_rad: DEFAULT_PHASE_STEP_MIN_RAD,
                effective_inv_temp: self.effective_inv_temp,
                fact_persona_coherence: self.resultant_r_bar,
                repulsion_severity: 0.0,
                provenance_bits: prov_bits,
                dominant_cause_code: self.dominant_cause_code,
                dominant_slot: self.dominant_slot,
                action_mode_flags: self.action_mode_flags,
                active_swarm_cell_mask: self.active_swarm_mask,
                qdps_min_step_q7: branchless_rad_to_phase_step_u8(DEFAULT_PHASE_STEP_MIN_RAD),
                subhead_fact_heads: subhead_fact,
                subhead_logic_heads: subhead_logic,
                subhead_explore_heads: subhead_explore,
                subhead_fact_gain_q7: fact_gain_q7,
                subhead_logic_gain_q7: logic_gain_q7,
                subhead_explore_gain_q7: explore_gain_q7,
                mean_phase_tag_q7: 0,
                active_head_count: MAX_ATTENTION_HEADS as u16,
                frame_flags: 0,
                phase_deltas: [0i8; MAX_ATTENTION_HEADS],
            },
            w,
        )
    }

    #[inline(always)]
    fn has_active_gateway_ensemble_signal(&self) -> bool {
        (self.sample_count > 0) & (self.phase_gravity_mass > DEFAULT_ENTROPY_EPS)
    }

    #[inline]
    fn synthesize_gateway_slice_with_words(
        records: &[Self],
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let len = records.len();
        if len == 0 {
            return (GatewayFrame::default(), [0u64; NUM_STIGMERGIC_GRANULES]);
        }
        if len == 1 {
            return records[0].synthesize_gateway_frame_with_words();
        }

        let mut acc_w = [0u64; NUM_STIGMERGIC_GRANULES];
        let mut acc_is_uniform = true;
        let mut tot_e = 0.0_f32;
        let mut best_e = 0.0_f32;
        let mut weighted_inv_temp = 0.0_f32;
        let mut weighted_coh = 0.0_f32;
        let mut dom_idx = 0usize;
        let mut dom_storage_addr = 0u64;
        let mut action_flags = ACTION_PRESET_PASSIVE_SIDECAR;
        let mut swarm_mask = 0xFFu8;
        let mut prov_bits = 0u32;
        let mut dom_cause = ROOT_CAUSE_NOMINAL;
        let mut has_active = false;

        let mut i = 0usize;
        while i < len {
            let rec = &records[i];
            if (rec.sample_count > 0) & (rec.phase_gravity_mass > DEFAULT_ENTROPY_EPS) {
                let rec_w = rec.phase_words_ref();
                let rec_is_uniform = if !has_active || acc_is_uniform {
                    is_word_array_uniform_64(rec_w)
                } else {
                    false
                };
                let f_e = branchless_min_f32(
                    branchless_flush_subnormal_f32(branchless_relu_f32(rec.phase_gravity_mass)),
                    MAX_ATTRACTOR_PULL_FORCE,
                );
                let is_mature = (rec.sample_count >= 2) & (rec.resultant_r_bar >= 0.45);
                let rec_prov = rec.provenance_bits
                    | branchless_select_u32(is_mature, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, 0);

                if !has_active {
                    acc_w = *rec_w;
                    acc_is_uniform = rec_is_uniform;
                    tot_e = f_e;
                    best_e = f_e;
                    weighted_inv_temp = f_e * rec.effective_inv_temp;
                    weighted_coh = f_e * rec.resultant_r_bar;
                    dom_idx = i;
                    dom_storage_addr = rec.mapped_section_offset48;
                    action_flags = rec.action_mode_flags;
                    swarm_mask = rec.active_swarm_mask;
                    prov_bits = rec_prov;
                    dom_cause = rec.dominant_cause_code;
                    has_active = true;
                } else if f_e > DEFAULT_ENTROPY_EPS {
                    let new_tot = tot_e + f_e;
                    if !swar_words_equal_64(&acc_w, rec_w) {
                        let safe_tot =
                            branchless_select_f32(new_tot > DEFAULT_ENTROPY_EPS, new_tot, 1.0);
                        let w_ratio =
                            branchless_clamp_f32(f_e * fast_inv_pos_normal_f32(safe_tot), 0.0, 1.0);
                        let s_u8 = ((w_ratio * 128.0 + 0.5) as u8).min(128);
                        let s_u64 = s_u8 as u64;
                        let bias_restore =
                            (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                        let (next_acc, next_uni) = swar_slerp_words_with_uniform_flag_64(
                            &acc_w,
                            acc_is_uniform,
                            rec_w,
                            rec_is_uniform,
                            s_u64,
                            bias_restore,
                        );
                        acc_w = next_acc;
                        acc_is_uniform = next_uni;
                    }
                    weighted_inv_temp += f_e * rec.effective_inv_temp;
                    weighted_coh += f_e * rec.resultant_r_bar;
                    let take_dom = f_e > best_e;
                    best_e = branchless_select_f32(take_dom, f_e, best_e);
                    dom_idx = branchless_select_usize(take_dom, i, dom_idx);
                    dom_storage_addr = branchless_select_u64(
                        take_dom & (rec.mapped_section_offset48 != 0),
                        rec.mapped_section_offset48,
                        branchless_select_u64(
                            dom_storage_addr != 0,
                            dom_storage_addr,
                            rec.mapped_section_offset48,
                        ),
                    );
                    action_flags |= rec.action_mode_flags;
                    swarm_mask &= rec.active_swarm_mask;
                    prov_bits |= rec_prov;
                    dom_cause = branchless_select_u8(
                        rec.dominant_cause_code > dom_cause,
                        rec.dominant_cause_code,
                        dom_cause,
                    );
                    tot_e = new_tot;
                }
            }
            i += 1;
        }

        let safe_tot_e = branchless_select_f32(tot_e > DEFAULT_ENTROPY_EPS, tot_e, 1.0);
        let inv_tot_e = fast_inv_pos_normal_f32(safe_tot_e);
        let eff_inv_temp = branchless_select_f32(
            tot_e > DEFAULT_ENTROPY_EPS,
            branchless_clamp_f32(weighted_inv_temp * inv_tot_e, 0.25, 4.0),
            records[dom_idx].effective_inv_temp,
        );
        let fp_coh = branchless_select_f32(
            tot_e > DEFAULT_ENTROPY_EPS,
            branchless_clamp_f32(weighted_coh * inv_tot_e, 0.0, 1.0),
            records[dom_idx].resultant_r_bar,
        );
        let dom = &records[dom_idx];
        let dom_slot = dom.dominant_slot;
        let is_fact_slot = dom_slot == ATTRACTOR_SLOT_KNOWLEDGE;
        let is_persona_slot = dom_slot == ATTRACTOR_SLOT_PERSONA;
        let subhead_fact = branchless_select_u8(
            is_fact_slot,
            36,
            branchless_select_u8(is_persona_slot, 16, 24),
        );
        let subhead_logic = branchless_select_u8(is_fact_slot, 20, 24);
        let subhead_explore = branchless_select_u8(
            is_fact_slot,
            8,
            branchless_select_u8(is_persona_slot, 24, 16),
        );
        let fact_gain_q7 = branchless_select_u8(is_fact_slot, 147, 128);
        let explore_gain_q7 = branchless_select_u8(is_persona_slot, 154, 128);
        let logic_gain_q7 = ((fact_gain_q7 as u16 + explore_gain_q7 as u16) >> 1) as u8;

        let phase_deltas = words_to_q7_array_64(&acc_w);
        let tag_q7 = swar_q7_mean_i8_from_words_with_uniform_flag_64(&acc_w, acc_is_uniform);
        let packed_tag = pack_action_slot_tag(dom_slot, action_flags);
        let tagged_ptr = gcso_tagged_ptr_t::pack(dom_storage_addr, packed_tag, tag_q7).raw();

        (
            GatewayFrame {
                tagged_ptr,
                mapped_section_offset48: dom_storage_addr,
                target_anchor_id: dom.anchor_id,
                effective_energy: branchless_min_f32(tot_e, MAX_ATTRACTOR_PULL_FORCE),
                ripa_clamp_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
                qdps_min_step_rad: DEFAULT_PHASE_STEP_MIN_RAD,
                effective_inv_temp: eff_inv_temp,
                fact_persona_coherence: fp_coh,
                repulsion_severity: 0.0,
                provenance_bits: prov_bits,
                dominant_cause_code: dom_cause,
                dominant_slot: dom_slot,
                action_mode_flags: action_flags,
                active_swarm_cell_mask: swarm_mask,
                qdps_min_step_q7: branchless_rad_to_phase_step_u8(DEFAULT_PHASE_STEP_MIN_RAD),
                subhead_fact_heads: subhead_fact,
                subhead_logic_heads: subhead_logic,
                subhead_explore_heads: subhead_explore,
                subhead_fact_gain_q7: fact_gain_q7,
                subhead_logic_gain_q7: logic_gain_q7,
                subhead_explore_gain_q7: explore_gain_q7,
                mean_phase_tag_q7: tag_q7,
                active_head_count: MAX_ATTENTION_HEADS as u16,
                frame_flags: packed_tag as u16,
                phase_deltas,
            },
            acc_w,
        )
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

impl CognitiveTrajectoryLedger {
    /// Evaluates this trajectory circuit against `(frm_w, trail_w)` at `scale` and modulates
    /// `frame`'s `SubheadRouter` head allocation, sub-head gains, `EntropyController` inverse-temperature scale, and
    /// anti-resonance guard severity in place without dataset-specific constants.
    ///
    /// When the trajectory ledger is empty (`trajectory_depth == 0`), `frame` is returned unmodified.
    #[inline]
    pub fn modulate_gateway_frame_at_scale(
        &self,
        frame: &mut GatewayFrame,
        frm_w: &[u64; NUM_STIGMERGIC_GRANULES],
        trail_w: &[u64; NUM_STIGMERGIC_GRANULES],
        scale: StigmergicScaleGranularity,
    ) -> CognitiveCircuitActivation {
        let act = self.evaluate_circuit_activation(frm_w, trail_w, scale);
        if act.trajectory_depth == 0 {
            return act;
        }
        const MIN_BUCKET_HEADS: u8 = 8;
        match act.op_kind() {
            CognitiveOpKind::DecomposePeel | CognitiveOpKind::RouteSubHead => {
                let shift = ((act.peel_gate_q7.saturating_sub(56)) >> 3).min(8);
                let take_logic =
                    (shift >> 1).min(frame.subhead_logic_heads.saturating_sub(MIN_BUCKET_HEADS));
                let rem = shift.saturating_sub(take_logic);
                let take_explore =
                    rem.min(frame.subhead_explore_heads.saturating_sub(MIN_BUCKET_HEADS));
                let gained = take_logic.saturating_add(take_explore);
                frame.subhead_logic_heads = frame.subhead_logic_heads.saturating_sub(take_logic);
                frame.subhead_explore_heads =
                    frame.subhead_explore_heads.saturating_sub(take_explore);
                frame.subhead_fact_heads = frame.subhead_fact_heads.saturating_add(gained);
                frame.subhead_fact_gain_q7 = frame
                    .subhead_fact_gain_q7
                    .saturating_add(act.peel_gate_q7 >> 3);
            }
            CognitiveOpKind::SynthesizeFuse | CognitiveOpKind::CrystallizeCommit => {
                let shift = ((act.fuse_gate_q7.saturating_sub(56)) >> 3).min(8);
                let take_fact =
                    (shift >> 1).min(frame.subhead_fact_heads.saturating_sub(MIN_BUCKET_HEADS));
                let rem = shift.saturating_sub(take_fact);
                let take_explore =
                    rem.min(frame.subhead_explore_heads.saturating_sub(MIN_BUCKET_HEADS));
                let gained = take_fact.saturating_add(take_explore);
                frame.subhead_fact_heads = frame.subhead_fact_heads.saturating_sub(take_fact);
                frame.subhead_explore_heads =
                    frame.subhead_explore_heads.saturating_sub(take_explore);
                frame.subhead_logic_heads = frame.subhead_logic_heads.saturating_add(gained);
                frame.subhead_logic_gain_q7 = frame
                    .subhead_logic_gain_q7
                    .saturating_add(act.fuse_gate_q7 >> 3);
                let boost = 1.0_f32 + (f32::from(act.fuse_gate_q7) * (0.25_f32 / 128.0_f32));
                frame.effective_inv_temp =
                    branchless_clamp_f32(frame.effective_inv_temp * boost, 0.25, 4.0);
            }
            CognitiveOpKind::RepulseAudit => {
                let corroborated = act.circuit_resonance_q7 < 0
                    || frame.fact_persona_coherence < 0.0
                    || frame.repulsion_severity > 0.25;
                if corroborated {
                    let sev_floor = branchless_clamp_f32(
                        f32::from(act.repulse_sensitivity_q7) / 128.0_f32,
                        0.55,
                        1.0,
                    );
                    frame.repulsion_severity =
                        branchless_max_f32(frame.repulsion_severity, sev_floor);
                    frame.action_mode_flags |= ACTION_MODE_PHASE_REPULSE;
                    frame.provenance_bits |= CAUSAL_CAUSE_COHOMOLOGY_REPULSION;
                    frame.subhead_explore_gain_q7 = frame
                        .subhead_explore_gain_q7
                        .saturating_add(act.repulse_sensitivity_q7 >> 3);
                }
            }
            CognitiveOpKind::BridgeShift
            | CognitiveOpKind::SteerAlign
            | CognitiveOpKind::Observe => {
                let inc = act.intensity_q7 >> 4;
                frame.subhead_fact_gain_q7 = frame.subhead_fact_gain_q7.saturating_add(inc);
                frame.subhead_logic_gain_q7 = frame.subhead_logic_gain_q7.saturating_add(inc);
            }
        }
        act
    }
}

impl CausalProvenanceSignal for CognitiveTrajectoryLedger {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let (hist, depth) = self.op_counts();
        if depth == 0 {
            return 0;
        }
        let repulse_active = hist[CognitiveOpKind::RepulseAudit as usize]
            > hist[CognitiveOpKind::SynthesizeFuse as usize];
        CAUSAL_AXIS_SUBJECTIVE_PERSONA
            | branchless_select_u32(repulse_active, CAUSAL_CAUSE_COHOMOLOGY_REPULSION, 0)
    }
}

impl CognitiveGroundingSignal for CognitiveTrajectoryLedger {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.project_to_q7_64()
    }

    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        self.to_phase_words_64()
    }

    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        let (_hist, depth) = self.op_counts();
        (depth as f32) * 0.25
    }
}

impl SubjectivePersonaSignal for CognitiveTrajectoryLedger {}

impl ManifoldGatewaySignal for CognitiveTrajectoryLedger {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let words = self.to_phase_words_64();
        let (hist, depth) = self.op_counts();
        if depth == 0 {
            return (GatewayFrame::default(), [0u64; NUM_STIGMERGIC_GRANULES]);
        }
        let tag_q7 = fast_mean_q7_tag_from_words_64(&words);
        let peel_dominates = hist[CognitiveOpKind::DecomposePeel as usize]
            > hist[CognitiveOpKind::SynthesizeFuse as usize];
        let subhead_fact = branchless_select_u8(peel_dominates, 32, 20);
        let subhead_logic = branchless_select_u8(peel_dominates, 20, 24);
        let subhead_explore = MAX_ATTENTION_HEADS as u8 - subhead_fact - subhead_logic;
        let action_mode = if peel_dominates {
            ACTION_MODE_PHASE_STEER | ACTION_MODE_SUBHEAD_ROUTE
        } else {
            ACTION_PRESET_EXPRESS_PHASE
        };
        let packed_tag = pack_action_slot_tag(ATTRACTOR_SLOT_PERSONA, action_mode);
        let prov = self.causal_provenance_bits();
        (
            GatewayFrame {
                tagged_ptr: gcso_tagged_ptr_t::pack(0, packed_tag, tag_q7).raw(),
                effective_energy: (depth as f32) * 0.25,
                provenance_bits: prov,
                dominant_slot: ATTRACTOR_SLOT_PERSONA,
                action_mode_flags: action_mode,
                active_swarm_cell_mask: 0xFF,
                subhead_fact_heads: subhead_fact,
                subhead_logic_heads: subhead_logic,
                subhead_explore_heads: subhead_explore,
                mean_phase_tag_q7: tag_q7,
                active_head_count: MAX_ATTENTION_HEADS as u16,
                frame_flags: packed_tag as u16,
                phase_deltas: words_to_q7_array_64(&words),
                ..GatewayFrame::default()
            },
            words,
        )
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

impl<const N: usize> CausalProvenanceSignal for CognitiveCircuitBank<N> {
    #[inline]
    fn causal_provenance_bits(&self) -> u32 {
        self.to_consolidated_ledger().causal_provenance_bits()
    }
}

impl<const N: usize> CognitiveGroundingSignal for CognitiveCircuitBank<N> {
    #[inline]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.project_to_q7_64()
    }

    #[inline]
    fn grounding_base_energy(&self) -> f32 {
        self.to_consolidated_ledger().grounding_base_energy()
    }
}

impl<const N: usize> SubjectivePersonaSignal for CognitiveCircuitBank<N> {}

impl<const N: usize> ManifoldGatewaySignal for CognitiveCircuitBank<N> {
    #[inline]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.to_consolidated_ledger().synthesize_gateway_frame()
    }
}

impl CausalProvenanceSignal for gcso_pointer_trail_t {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let rep = (self.flags & TRAIL_FLAG_REPULSION_ACTIVE) != 0;
        let has_anchor = self.target_anchor_id != 0;
        branchless_select_u32(has_anchor, CAUSAL_AXIS_OBJECTIVE_FACT, 0)
            | branchless_select_u32(
                rep,
                CAUSAL_AXIS_CONTRADICTION_GUARD | CAUSAL_CAUSE_COHOMOLOGY_REPULSION,
                0,
            )
    }
}

impl ManifoldGatewaySignal for gcso_pointer_trail_t {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let w = *trail_phase_words_ref(self);
        let tag_q7 = fast_mean_q7_tag_from_words_64(&w);
        let ptr = gcso_tagged_ptr_t(self.current_ptr);
        let rep = (self.flags & TRAIL_FLAG_REPULSION_ACTIVE) != 0;
        let dom_slot = ptr.unpack_tag() & ATTRACTOR_SLOT_ID_MASK;
        let action_flags = branchless_select_u8(
            rep,
            ACTION_PRESET_REPULSION_ESCAPE,
            ACTION_PRESET_EXPRESS_PHASE,
        );
        let packed_tag = pack_action_slot_tag(dom_slot, action_flags);
        (
            GatewayFrame {
                tagged_ptr: self.current_ptr,
                mapped_section_offset48: ptr.unpack_addr(),
                target_anchor_id: self.target_anchor_id,
                effective_energy: branchless_min_f32(
                    branchless_max_f32(self.attractor_pull_force, self.stigmergic_density),
                    MAX_ATTRACTOR_PULL_FORCE,
                ),
                repulsion_severity: branchless_select_f32(rep, 0.85, 0.0),
                provenance_bits: self.causal_provenance_bits(),
                dominant_cause_code: branchless_select_u8(
                    rep,
                    ROOT_CAUSE_COHOMOLOGY_REPULSION,
                    ROOT_CAUSE_NOMINAL,
                ),
                dominant_slot: dom_slot,
                action_mode_flags: action_flags,
                active_swarm_cell_mask: ((self.flags & TRAIL_SWARM_QUORUM_MASK)
                    >> TRAIL_SWARM_QUORUM_SHIFT) as u8,
                mean_phase_tag_q7: tag_q7,
                active_head_count: MAX_ATTENTION_HEADS as u16,
                frame_flags: packed_tag as u16,
                phase_deltas: self.accumulated_phase_delta,
                ..GatewayFrame::default()
            },
            w,
        )
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

impl StigmergicScaleLayer for gcso_pointer_trail_t {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        if self.linked_trail_id != 0 {
            StigmergicScaleGranularity::MacroCoalition
        } else {
            StigmergicScaleGranularity::CellTrail
        }
    }
}

impl EpisodicPhaseSample for GatewayFrame {
    #[inline(always)]
    fn sample_phase_coords(&self, _domain_salt: u32) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.phase_deltas
    }

    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        branchless_clamp_f32(
            branchless_flush_subnormal_f32(branchless_relu_f32(self.effective_energy)),
            0.0,
            MAX_ATTRACTOR_PULL_FORCE,
        )
    }

    #[inline(always)]
    fn sample_coherence_prior(&self) -> f32 {
        branchless_clamp_f32(0.5 * (1.0 + self.fact_persona_coherence), 0.25, 1.0)
    }

    #[inline(always)]
    fn preferred_working_lane(&self) -> u32 {
        u32::from(self.dominant_slot) & ((MAX_ATTRACTOR_DOMAINS as u32) - 1)
    }

    #[inline(always)]
    fn sample_anchor_type(&self) -> gcso_anchor_type_t {
        GCSO_ANCHOR_TYPE_CRYSTALLIZED
    }
}

impl CausalProvenanceSignal for EntropyController {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let is_rep = self.state.active_branch_mode == ENTROPY_GATE_MODE_PHASE_CONJUGATE_REPULSION;
        let h_cur = branchless_relu_f32(self.state.moving_z_entropy);
        let can_skip = (self.state.active_branch_mode == ENTROPY_GATE_MODE_NORMAL_ENTROPY)
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
            self.state.active_branch_mode == ENTROPY_GATE_MODE_PHASE_CONJUGATE_REPULSION,
            0.85,
            0.0,
        );
        branchless_max_f32(h_sev, rep_sev)
    }
}

impl SystemConstraintSignal for EntropyController {
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

impl ManifoldGatewaySignal for EntropyController {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
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

impl RelativeAttractorNode for GatewayFrame {
    #[inline(always)]
    fn relative_phase_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        *self.phase_words_ref()
    }
    #[inline(always)]
    fn relative_energy(&self) -> f32 {
        self.effective_energy
    }
    #[inline(always)]
    fn relative_anchor_id(&self) -> u32 {
        self.target_anchor_id
    }
}

impl<T: RawPhaseProjectable> RelativeAttractorNode for ObjectiveFact<T> {
    #[inline(always)]
    fn relative_phase_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        self.payload.project_to_words_64()
    }
    #[inline(always)]
    fn relative_energy(&self) -> f32 {
        self.base_energy
    }
    #[inline(always)]
    fn relative_anchor_id(&self) -> u32 {
        self.anchor_id
    }
}

impl<T: RawPhaseProjectable> RelativeAttractorNode for SubjectivePersona<T> {
    #[inline(always)]
    fn relative_phase_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        self.payload.project_to_words_64()
    }
    #[inline(always)]
    fn relative_energy(&self) -> f32 {
        self.base_energy
    }
    #[inline(always)]
    fn relative_anchor_id(&self) -> u32 {
        self.anchor_id
    }
}

impl RelativeAttractorNode for GuardPhaseAnchor {
    #[inline(always)]
    fn relative_phase_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        load_q7_words_64(&self.phase_deltas)
    }
    #[inline(always)]
    fn relative_energy(&self) -> f32 {
        self.cohomology_obstruction
    }
    #[inline(always)]
    fn relative_anchor_id(&self) -> u32 {
        0
    }
}

impl RelativeAttractorNode for ScaleLayerProjection {
    #[inline(always)]
    fn relative_phase_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        *self.frame.phase_words_ref()
    }
    #[inline(always)]
    fn relative_energy(&self) -> f32 {
        branchless_relu_f32(self.frame.effective_energy)
    }
    #[inline(always)]
    fn relative_anchor_id(&self) -> u32 {
        self.frame.target_anchor_id
    }
}
