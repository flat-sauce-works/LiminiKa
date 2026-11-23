// name: src/core/src/runtime/gateway/bridges/ledger_bridge.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`CognitiveTrajectoryLedger`], [`CognitiveCircuitBank`], [`PointerTrail`], and [`EntropyController`] Gateway bridges.

use super::super::*;
use crate::drivers::{
    EntropyController, ENTROPY_GATE_MODE_NORMAL_ENTROPY,
    ENTROPY_GATE_MODE_PHASE_CONJUGATE_REPULSION,
};
use crate::foundation::{
    branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_max_f32, branchless_min_f32,
    branchless_relu_f32, fast_mean_q7_tag_from_words_64, load_q7_words_64, pack_action_slot_tag,
    words_to_q7_array_64, RawPhaseProjectable, TaggedPtr, ACTION_MODE_PHASE_REPULSE,
    ACTION_MODE_PHASE_STEER, ACTION_MODE_SUBHEAD_ROUTE, ACTION_PRESET_EXPRESS_PHASE,
    ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_ID_MASK,
};
use crate::liminal::{
    branchless_select_f32, branchless_select_u32, branchless_select_u8, PhaseQ7,
    MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS, NUM_STIGMERGIC_GRANULES,
};
use crate::stigmergy::{
    trail_phase_words_ref, CognitiveCircuitActivation, CognitiveCircuitBank,
    CognitiveCircuitBankOps, CognitiveOpKind, CognitiveTrajectoryLedger, EpisodicPhaseSample,
    PointerTrail, RelativeAttractorNode, StigmergicScaleGranularity, ATTRACTOR_SLOT_PERSONA,
    MAX_ATTRACTOR_PULL_FORCE, TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_SWARM_QUORUM_MASK,
    TRAIL_SWARM_QUORUM_SHIFT,
};

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
                    frame.lineage_flags |= CAUSAL_CAUSE_COHOMOLOGY_REPULSION;
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
    fn project_grounding_phases_64(&self) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
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
                tagged_ptr: TaggedPtr::pack(0, packed_tag, tag_q7).raw(),
                effective_energy: (depth as f32) * 0.25,
                lineage_flags: prov,
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
    fn project_grounding_phases_64(&self) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
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

impl CausalProvenanceSignal for PointerTrail {
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

impl ManifoldGatewaySignal for PointerTrail {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let w = *trail_phase_words_ref(self);
        let tag_q7 = fast_mean_q7_tag_from_words_64(&w);
        let ptr = TaggedPtr(self.current_ptr);
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
                lineage_flags: self.causal_provenance_bits(),
                attribution_tag: branchless_select_u8(
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

impl StigmergicScaleLayer for PointerTrail {
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
    fn sample_phase_coords(&self, _domain_salt: u32) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
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
