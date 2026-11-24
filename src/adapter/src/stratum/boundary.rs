// name: src/adapter/src/stratum/boundary.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Upper boundary arbitration stratum (`SubstrateBoundaryStratum`) and core
//! [`StratumGatewayMapping`] projection (`StratumSignal::map_to_gateway`).

use super::{RecallStratum, StratumContext, StratumDirective, StratumSignal};
use crate::{
    AttractorRole, LacunaBoundaryState, RecallBoundaryGate, RecallBoundaryMetrics,
    TextAttractorEntry,
};
use liminika_core::abi::lmnk_pointer_trail_t;
use liminika_core::liminal::MAX_ATTENTION_HEADS;
use liminika_core::runtime::{
    AdaptiveThresholdPolicy, CognitiveCircuitThresholdPolicy, GatewayFrame, GuardPhaseAnchor,
    ManifoldGatewaySignal, ObjectiveFact, ScaleLayerNode, ScaleLayerOrchestrator,
    SubjectivePersona,
};
use liminika_core::stigmergy::{
    CognitiveOpKind, CognitiveStepOp, StigmergicActionDecision, StigmergicScaleGranularity,
    StigmergicSubstrate, StigmergicThresholds,
};

/// Stratum 2 (`n = 2`): upper boundary arbiter combining [`RecallBoundaryGate`] with
/// substrate [`StigmergicActionDecision`] regime selection and top-down directives.
pub struct SubstrateBoundaryStratum<'a> {
    /// Boundary gate classifying `lacuna_state` and suppression.
    pub gate: &'a dyn RecallBoundaryGate,
}

impl RecallStratum for SubstrateBoundaryStratum<'_> {
    fn name(&self) -> &'static str {
        "substrate_boundary"
    }

    fn infer_up(&self, ctx: &StratumContext<'_>, signal: &mut StratumSignal) -> Result<(), String> {
        let mut fact_scores: Vec<f32> = Vec::new();
        let mut persona_scores: Vec<f32> = Vec::new();
        let mut max_guard_score = -1.0_f32;
        for (entry, &score) in ctx.entries.iter().zip(signal.candidate_scores.iter()) {
            match entry.role {
                AttractorRole::Fact => fact_scores.push(score),
                AttractorRole::Persona => persona_scores.push(score),
                AttractorRole::Guard => {
                    if score > max_guard_score {
                        max_guard_score = score;
                    }
                }
            }
        }
        fact_scores.sort_by(|a, b| b.total_cmp(a));
        persona_scores.sort_by(|a, b| b.total_cmp(a));

        let top1_fact_score = fact_scores.first().copied().unwrap_or(-1.0);
        let top2_fact_score = fact_scores.get(1).copied().unwrap_or(-1.0);
        let top1_persona_score = persona_scores.first().copied().unwrap_or(-1.0);
        let guard_conflict_margin = max_guard_score - top1_fact_score.max(top1_persona_score);

        let metrics = RecallBoundaryMetrics {
            top1_fact_score,
            top2_fact_score,
            top1_persona_score,
            max_guard_score,
            guard_conflict_margin,
        };
        let decision = self.gate.evaluate(&metrics, ctx.policy);
        signal.boundary_metrics = metrics;
        signal.gate_decision = decision;

        let repulsed_by_guard = guard_conflict_margin > ctx.policy.guard_suppress_margin;
        if repulsed_by_guard {
            signal.action_decision = StigmergicActionDecision::RepulsionEscape;
        }

        let last_layer = ctx
            .entries
            .first()
            .map(|e| e.per_layer_q7.len().saturating_sub(1))
            .unwrap_or(0);
        let persona_floor = ctx.policy.min_coherence * 0.75;
        let mut frame = signal
            .map_to_gateway_with_persona_floor(ctx.entries, last_layer, persona_floor)
            .frame;
        frame.active_head_count =
            (ctx.num_q_heads as u16).clamp(1, liminika_core::liminal::MAX_ATTENTION_HEADS as u16);

        let repulsed = repulsed_by_guard || frame.is_repulsed();
        let has_active_persona = top1_persona_score >= persona_floor;
        let is_unobserved_exterior =
            matches!(decision.lacuna_state, LacunaBoundaryState::LiminalExterior)
                && !has_active_persona
                && !repulsed;

        let fact_margin = (top1_fact_score - top2_fact_score).max(0.0);
        let margin_ratio = (fact_margin / ctx.policy.boundary_margin.max(0.01)).clamp(0.0, 3.0);
        let channel_disagreement = signal.telemetry.channel_disagreement;
        let partial_swarm_consensus = signal.swarm_report.resolved_cell_count > 0
            && signal.swarm_report.resolved_cell_count < 6;

        let thresholds = StigmergicThresholds::default();
        let circuit_policy = AdaptiveThresholdPolicy::default();
        let cell_act = signal.telemetry.layer_activation;
        let circuit_peel_active = matches!(
            cell_act.op_kind(),
            CognitiveOpKind::DecomposePeel | CognitiveOpKind::RouteSubHead
        ) && cell_act.peel_gate_q7
            >= circuit_policy.peel_escalation_floor_q7();
        let circuit_fuse_active = matches!(
            cell_act.op_kind(),
            CognitiveOpKind::SynthesizeFuse | CognitiveOpKind::CrystallizeCommit
        ) && cell_act.fuse_gate_q7
            >= circuit_policy.early_exit_fuse_floor_q7();

        let (rho, h_tilde) = if repulsed {
            (frame.effective_energy, 0.75_f32)
        } else if is_unobserved_exterior {
            (0.0_f32, 0.0_f32)
        } else if matches!(decision.lacuna_state, LacunaBoundaryState::LiminalExterior)
            && has_active_persona
        {
            // Persona-only active grounding outside factual basins uses lightweight ExpressPhaseSteer.
            (thresholds.express_steer_density_thresh.max(0.45), 0.35_f32)
        } else if signal.local_channel_preferred
            || signal.decomposition_triggered
            || (channel_disagreement && partial_swarm_consensus)
            || (circuit_peel_active
                && !matches!(decision.lacuna_state, LacunaBoundaryState::LiminalExterior))
        {
            // Bifurcated, local-sharpened, or circuit-peel-biased channel triggers sub-head routing.
            (
                thresholds
                    .subhead_route_density_thresh
                    .max(frame.effective_energy.min(1.35)),
                0.65_f32,
            )
        } else if circuit_fuse_active && margin_ratio >= 1.2 {
            let boost = thresholds
                .crystallize_density_thresh
                .max(frame.effective_energy * 1.5);
            (boost, 0.40_f32)
        } else if matches!(decision.lacuna_state, LacunaBoundaryState::LiminalBoundary) {
            (thresholds.express_steer_density_thresh.max(0.45), 0.35_f32)
        } else if margin_ratio >= 2.0 {
            let boost = thresholds
                .crystallize_density_thresh
                .max(frame.effective_energy * 1.5);
            (boost, 0.40_f32)
        } else {
            let rank1_rho = thresholds
                .rank1_inject_density_thresh
                .max(frame.effective_energy.min(2.0));
            (rank1_rho, 0.40_f32)
        };

        let (action_decision, mode_flags, target_slot, _) = thresholds.select_action_decision(
            rho,
            h_tilde,
            frame.fact_persona_coherence,
            repulsed,
            frame.dominant_slot,
        );
        frame.action_mode_flags = mode_flags;
        frame.dominant_slot = target_slot;

        let final_top1_anchor = ctx
            .entries
            .iter()
            .zip(signal.candidate_scores.iter())
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(e, _)| e.anchor_id);
        signal.action_decision = action_decision;
        signal.gateway_frame = frame;
        signal.telemetry.final_top1_anchor = final_top1_anchor;
        signal.telemetry.provenance = frame.to_causal_report();

        // Execute cross-scale substrate orchestration so Gateway synthesis, Stigmergy, and
        // recorded CognitiveTrajectoryLedger close into one unified FastPathLedger ledger.
        let mut boundary_substrate = StigmergicSubstrate::new();
        let mut ledger_slot = Default::default();
        signal
            .telemetry
            .cognitive_trajectory
            .write_to_ledger_slot(&mut ledger_slot);
        let mut boundary_trail = lmnk_pointer_trail_t::default();
        boundary_trail
            .copy_from_with_words_fast(&lmnk_pointer_trail_t::default(), frame.phase_words_ref());
        let scale_stack = [
            ScaleLayerNode::new(frame, StigmergicScaleGranularity::TypeGateway),
            ScaleLayerNode::new(
                signal
                    .telemetry
                    .cognitive_trajectory
                    .synthesize_gateway_frame(),
                StigmergicScaleGranularity::MacroCoalition,
            ),
        ];
        if let Ok((proj, mut ledger, _rep)) = boundary_substrate.orchestrate_adaptive_depth_n(
            &scale_stack,
            scale_stack.len(),
            liminika_core::foundation::TaggedPtr(frame.tagged_ptr),
            frame.phase_words_ref(),
            &mut ledger_slot,
            &mut boundary_trail,
            None,
        ) {
            let final_op = match action_decision {
                StigmergicActionDecision::RepulsionEscape => CognitiveOpKind::RepulseAudit,
                StigmergicActionDecision::CrystallizeMacroBasin => {
                    CognitiveOpKind::CrystallizeCommit
                }
                StigmergicActionDecision::SubHeadRoute => {
                    if signal.decomposition_triggered {
                        CognitiveOpKind::DecomposePeel
                    } else {
                        CognitiveOpKind::RouteSubHead
                    }
                }
                StigmergicActionDecision::FullOrchestration
                | StigmergicActionDecision::ExpressPhaseSteer => CognitiveOpKind::SynthesizeFuse,
                StigmergicActionDecision::PassiveSidecar
                | StigmergicActionDecision::InertialLayerSkip => CognitiveOpKind::Observe,
            };
            let intensity = (proj.activation.intensity_q7 >> 4).clamp(1, 7);
            ledger.push_step(CognitiveStepOp::pack(
                StigmergicScaleGranularity::MacroCoalition,
                final_op,
                intensity,
            ));
            ledger.write_to_ledger_slot(&mut boundary_substrate.ledger_slot);
            signal.telemetry.cognitive_trajectory = ledger;
            signal.telemetry.layer_activation = proj.activation;
        }
        Ok(())
    }

    fn adjust_down(
        &self,
        _ctx: &StratumContext<'_>,
        directive: &mut StratumDirective,
        signal: &mut StratumSignal,
    ) {
        directive.suppress_injection = signal.gate_decision.suppress_fact_injection;
        directive.repulse_guard = matches!(
            signal.action_decision,
            StigmergicActionDecision::RepulsionEscape
        );
        directive.top_down_gain = match signal.action_decision {
            StigmergicActionDecision::RepulsionEscape
            | StigmergicActionDecision::InertialLayerSkip
            | StigmergicActionDecision::PassiveSidecar => 0.0,
            StigmergicActionDecision::ExpressPhaseSteer => 0.65,
            StigmergicActionDecision::SubHeadRoute => 0.85,
            StigmergicActionDecision::FullOrchestration => 1.0,
            StigmergicActionDecision::CrystallizeMacroBasin => 1.15,
        };
        let min_lacuna = signal
            .local_lacuna_depths
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        directive.min_local_lacuna_depth = if min_lacuna.is_finite() {
            min_lacuna
        } else {
            1.0
        };
    }
}

/// Core `ManifoldGatewaySignal` projection synthesized from adapter-layer [`StratumSignal`] channels.
#[derive(Debug, Clone, Copy)]
pub struct StratumGatewayMapping {
    /// Winning `ObjectiveFact` axis when a non-suppressed Fact basin is active.
    pub fact: Option<ObjectiveFact<[i8; MAX_ATTENTION_HEADS]>>,
    /// Winning `SubjectivePersona` axis when a Persona attractor is active.
    pub persona: Option<SubjectivePersona<[i8; MAX_ATTENTION_HEADS]>>,
    /// Guard repulsion axis when a Guard attractor is present.
    pub guard: Option<GuardPhaseAnchor>,
    /// Synthesized 128-byte [`GatewayFrame`] produced via `liminika_core` blanket traits.
    pub frame: GatewayFrame,
}

impl StratumSignal {
    /// Maps this stratum signal's winning role axes into `liminika_core` [`ManifoldGatewaySignal`]
    /// primitives and synthesizes a unified 128-byte [`GatewayFrame`] via blanket composition.
    #[must_use]
    pub fn map_to_gateway(
        &self,
        entries: &[TextAttractorEntry],
        layer_idx: usize,
    ) -> StratumGatewayMapping {
        self.map_to_gateway_with_persona_floor(entries, layer_idx, f32::NEG_INFINITY)
    }

    pub(crate) fn map_to_gateway_with_persona_floor(
        &self,
        entries: &[TextAttractorEntry],
        layer_idx: usize,
        persona_floor: f32,
    ) -> StratumGatewayMapping {
        let mut best_fact: Option<(usize, f32)> = None;
        let mut best_persona: Option<(usize, f32)> = None;
        let mut best_guard: Option<(usize, f32)> = None;

        for (i, (entry, &score)) in entries.iter().zip(self.candidate_scores.iter()).enumerate() {
            let slot = match entry.role {
                AttractorRole::Fact => &mut best_fact,
                AttractorRole::Persona => &mut best_persona,
                AttractorRole::Guard => &mut best_guard,
            };
            if slot.is_none_or(|(_, best_s)| score > best_s) {
                *slot = Some((i, score));
            }
        }

        let is_repulsion = matches!(
            self.action_decision,
            StigmergicActionDecision::RepulsionEscape
        );

        let fact = if !is_repulsion && !self.gate_decision.suppress_fact_injection {
            best_fact.and_then(|(idx, score)| {
                let entry = &entries[idx];
                let q7 = entry
                    .per_layer_q7
                    .get(layer_idx)
                    .copied()
                    .or_else(|| entry.per_layer_q7.last().copied())?;
                let w = self.stigmergic_weights.get(idx).copied().unwrap_or(1.0);
                Some(ObjectiveFact::new(
                    q7,
                    (entry.energy * score.max(0.1) * w).max(0.1),
                    layer_idx as u64,
                    entry.anchor_id,
                ))
            })
        } else {
            None
        };

        let persona = if !is_repulsion {
            best_persona
                .filter(|(_, score)| *score >= persona_floor)
                .and_then(|(idx, score)| {
                    let entry = &entries[idx];
                    let q7 = entry
                        .per_layer_q7
                        .get(layer_idx)
                        .copied()
                        .or_else(|| entry.per_layer_q7.last().copied())?;
                    Some(SubjectivePersona::new(
                        q7,
                        (entry.energy * score.max(0.05)).max(0.05),
                        layer_idx as u64,
                        entry.anchor_id,
                    ))
                })
        } else {
            None
        };

        let guard = best_guard.and_then(|(idx, _)| {
            let entry = &entries[idx];
            let q7 = entry
                .per_layer_q7
                .get(layer_idx)
                .copied()
                .or_else(|| entry.per_layer_q7.last().copied())?;
            let raw_obs = (self.boundary_metrics.guard_conflict_margin + 0.5).max(0.0);
            let obs = if is_repulsion {
                raw_obs.max(0.55)
            } else {
                raw_obs
            };
            Some(GuardPhaseAnchor::new(q7, obs, 0.5))
        });

        let frame = if is_repulsion {
            guard
                .map(|g| g.synthesize_gateway_frame())
                .unwrap_or_default()
        } else {
            match (fact, persona) {
                (Some(f), Some(p)) => (f, p).synthesize_gateway_frame(),
                (Some(f), None) => f.synthesize_gateway_frame(),
                (None, Some(p)) => p.synthesize_gateway_frame(),
                (None, None) => GatewayFrame::default(),
            }
        };

        StratumGatewayMapping {
            fact,
            persona,
            guard,
            frame,
        }
    }
}
