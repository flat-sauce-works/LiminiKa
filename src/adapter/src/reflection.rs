// name: src/adapter/src/reflection.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Event-triggered closed-loop attractor reflection and trajectory critique.
//!
//! Rather than blindly re-running generation on every token or turn, [`AttractorTrajectoryEvaluator`]
//! inspects the geometric position of a generated draft on the compiled attractor manifold relative
//! to the originating query. Retries are triggered only when the draft exhibits `Guard` attraction,
//! manifold drift (`cvoid`), liminal ambiguity, or out-of-domain hallucination, and are strictly
//! bounded by [`ReflectionPolicy::max_retries`].

use crate::{DynamicAttractorRecall, RecallPolicy, VoidBoundaryState};
use liminika_core::action_hub::StigmergicActionDecision;
use liminika_core::traits::gateway::{
    CausalProvenanceSignal, CognitiveCircuitActivation, CognitiveOpKind, CognitiveStepOp,
    CognitiveTrajectoryLedger, GatewayFrame, GuardPhaseAnchor, ManifoldGatewaySignal,
    ObjectiveFact, SystemConstraint, CAUSAL_AXIS_CONTRADICTION_GUARD, CAUSAL_AXIS_OBJECTIVE_FACT,
};
use liminika_core::traits::primitives::MAX_ATTENTION_HEADS;

/// Policy governing bounded, event-triggered closed-loop reflection retries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReflectionPolicy {
    /// Maximum reflection retries per turn (`0` disables retries; `1..=2` keeps latency bounded).
    pub max_retries: usize,
    /// Draft `guard_conflict_margin` threshold above which [`ReflectionTriggerKind::GuardAttraction`] fires.
    pub guard_trigger_margin: f32,
    /// Allowed drop in `top1_fact_score` between query and draft before [`ReflectionTriggerKind::VoidDrift`] fires.
    pub convergence_drop_tol: f32,
    /// Minimum alignment gain required for a retried trajectory to replace Pass 1.
    pub alignment_improvement_eps: f32,
    /// Additive RoPE steering delta (radians) applied on corrective Fact/Guard passes.
    pub max_rope_boost_rad: f32,
    /// Additive Rank-1 residual residual scale delta applied on corrective Fact/Guard passes.
    pub srl_boost_delta: f32,
}

impl Default for ReflectionPolicy {
    fn default() -> Self {
        Self {
            max_retries: 1,
            guard_trigger_margin: 0.02,
            convergence_drop_tol: 0.08,
            alignment_improvement_eps: 0.02,
            max_rope_boost_rad: -0.35,
            srl_boost_delta: -0.20,
        }
    }
}

/// Classification of a generated draft trajectory relative to the query's manifold intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflectionTriggerKind {
    /// Draft converged safely inside a grounded Fact basin without Guard dominance.
    Converged,
    /// Out-of-domain query and draft both stayed cleanly outside the corpus (`cvoid`).
    ConsistentVoidAbstention,
    /// Draft drifted toward a contrastive `Guard` attractor (`guard_conflict_margin` elevated).
    GuardAttraction,
    /// Grounded query's draft dropped off-manifold into `cvoid` or suffered a severe resonance drop.
    VoidDrift,
    /// Draft landed in an unresolved `LiminalBoundary` or diverged from the query's top anchor.
    BoundaryAmbiguity,
    /// Ungrounded (`cvoid`) query produced a draft that pulled unexpectedly into corpus attractors.
    VoidHallucination,
}

/// Selection outcome returned when comparing Pass 1 against a corrective Pass 2 trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflectionSelection {
    /// Keep the initial draft.
    Pass1,
    /// Adopt the retried Pass 2 draft.
    Pass2,
}

/// Corrective steering and prompt-framing directive synthesized from manifold critique.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReflectionCorrectiveDirective {
    /// True when the evaluator recommends running a bounded corrective retry.
    pub rethink_recommended: bool,
    /// True when Guard contradiction repulsion should be amplified during Pass 2.
    pub inject_guard_repulsion: bool,
    /// True when steering/context injection should be suppressed (neutral out-of-domain retry).
    pub suppress_injection: bool,
    /// Additive RoPE steering adjustment (radians) applied during Pass 2.
    pub rope_rad_delta: f32,
    /// Additive Rank-1 residual gain adjustment applied during Pass 2.
    pub srl_scale_delta: f32,
    /// True when the draft's top recalled Fact anchor diverges from the query's top Fact anchor.
    pub anchor_mismatch: bool,
}

/// Owned, lifetime-free manifold telemetry snapshot extracted from [`DynamicAttractorRecall`].
#[derive(Debug, Clone)]
pub struct TrajectoryManifoldSnapshot {
    /// Boundary state observed on the corpus manifold.
    pub void_state: VoidBoundaryState,
    /// Top-1 Fact resonance score.
    pub top1_fact_score: f32,
    /// Top-2 Fact resonance score.
    pub top2_fact_score: f32,
    /// Top-1 Persona resonance score.
    pub top1_persona_score: f32,
    /// Maximum Guard score minus maximum non-Guard score.
    pub guard_conflict_margin: f32,
    /// Top recalled Fact anchor ID, if any.
    pub top_fact_anchor: Option<u32>,
    /// Last-layer 64-head Q7 signature of `top_fact_anchor`, or zeros if unavailable.
    pub top_fact_q7: [i8; MAX_ATTENTION_HEADS],
    /// Ordered list of recalled Fact anchor IDs.
    pub recalled_fact_anchors: Vec<u32>,
    /// Ordered `(anchor_id, last_layer_q7)` pairs for all recalled Fact basins in this trajectory.
    pub recalled_fact_q7s: Vec<(u32, [i8; MAX_ATTENTION_HEADS])>,
    /// Core `ActionHub` regime decision associated with this recall.
    pub action_decision: StigmergicActionDecision,
    /// Synthesized 128-byte core `GatewayFrame`.
    pub gateway_frame: GatewayFrame,
    /// Multi-scale cognitive step trajectory captured during this recall.
    pub cognitive_trajectory: CognitiveTrajectoryLedger,
}

impl TrajectoryManifoldSnapshot {
    /// Captures an owned manifold snapshot from a borrowed [`DynamicAttractorRecall`].
    #[must_use]
    pub fn from_recall(recall: &DynamicAttractorRecall<'_>) -> Self {
        let top_entry = recall.recalled_facts.first().map(|(e, _)| *e);
        let top_fact_anchor = top_entry.map(|e| e.anchor_id);
        let top_fact_q7 = top_entry
            .and_then(|e| e.per_layer_q7.last().copied())
            .unwrap_or([0_i8; MAX_ATTENTION_HEADS]);
        let recalled_fact_anchors = recall
            .recalled_facts
            .iter()
            .map(|(e, _)| e.anchor_id)
            .collect();
        let recalled_fact_q7s = recall
            .recalled_facts
            .iter()
            .map(|(e, _)| {
                (
                    e.anchor_id,
                    e.per_layer_q7
                        .last()
                        .copied()
                        .unwrap_or([0_i8; MAX_ATTENTION_HEADS]),
                )
            })
            .collect();
        Self {
            void_state: recall.void_state,
            top1_fact_score: recall.top1_fact_score,
            top2_fact_score: recall.top2_fact_score,
            top1_persona_score: recall.top1_persona_score,
            guard_conflict_margin: recall.guard_conflict_margin,
            top_fact_anchor,
            top_fact_q7,
            recalled_fact_anchors,
            recalled_fact_q7s,
            action_decision: recall.action_decision,
            gateway_frame: recall.gateway_frame,
            cognitive_trajectory: recall.telemetry.cognitive_trajectory,
        }
    }

    /// Returns query Fact anchor IDs that were omitted from `draft.recalled_fact_anchors`.
    #[must_use]
    pub fn missing_coalition_anchors(&self, draft: &Self) -> Vec<u32> {
        self.recalled_fact_anchors
            .iter()
            .copied()
            .filter(|id| !draft.recalled_fact_anchors.contains(id))
            .collect()
    }
}

/// Complete trajectory assessment and [`ManifoldGatewaySignal`] emitted by [`AttractorTrajectoryEvaluator`].
#[derive(Debug, Clone, PartialEq)]
pub struct ReflectionCritique {
    /// Detected convergence or drift category.
    pub trigger: ReflectionTriggerKind,
    /// Continuous manifold alignment utility (higher indicates healthier basin alignment).
    pub alignment_score: f32,
    /// Fraction of query Fact anchors preserved by the draft (`0.0..=1.0`).
    pub anchor_overlap_ratio: f32,
    /// Corrective directive for an optional bounded Pass 2.
    pub directive: ReflectionCorrectiveDirective,
    /// Synthesized core 128-byte `GatewayFrame` reflecting corrective Fact pull or Guard repulsion.
    pub gateway_frame: GatewayFrame,
    /// Updated cognitive trajectory ledger recording this closed-loop reflection transition.
    pub cognitive_trajectory: CognitiveTrajectoryLedger,
    /// Cross-trajectory pseudo-neural circuit activation comparing query and draft reasoning paths.
    pub layer_activation: CognitiveCircuitActivation,
}

impl ReflectionCritique {
    /// Returns a neutral, domain-agnostic reflection prompt cue when a retry is recommended.
    #[must_use]
    pub fn corrective_instruction(&self, draft_text: &str) -> Option<String> {
        if !self.directive.rethink_recommended {
            return None;
        }
        let compact_draft = draft_text.trim();
        let cue = match self.trigger {
            ReflectionTriggerKind::GuardAttraction => {
                if compact_draft.is_empty() {
                    "Answer strictly from verified context and avoid false premises.".to_string()
                } else {
                    format!(
                        "Carefully audit this draft for false premises or contradictions ({compact_draft}) and re-answer strictly from verified facts."
                    )
                }
            }
            ReflectionTriggerKind::VoidDrift | ReflectionTriggerKind::BoundaryAmbiguity => {
                if compact_draft.is_empty() {
                    "Re-examine the full question using every detail in the grounded context."
                        .to_string()
                } else {
                    format!(
                        "The draft ({compact_draft}) drifted or omitted key grounded details. Re-examine the question and cover all grounded context points concisely."
                    )
                }
            }
            ReflectionTriggerKind::VoidHallucination => {
                "Answer directly from general knowledge without introducing unrelated specialized terms."
                    .to_string()
            }
            ReflectionTriggerKind::Converged
            | ReflectionTriggerKind::ConsistentVoidAbstention => return None,
        };
        Some(cue)
    }
}

impl CausalProvenanceSignal for ReflectionCritique {
    fn causal_provenance_bits(&self) -> u32 {
        match self.trigger {
            ReflectionTriggerKind::GuardAttraction => CAUSAL_AXIS_CONTRADICTION_GUARD,
            ReflectionTriggerKind::VoidDrift | ReflectionTriggerKind::BoundaryAmbiguity => {
                CAUSAL_AXIS_OBJECTIVE_FACT
            }
            ReflectionTriggerKind::Converged
            | ReflectionTriggerKind::ConsistentVoidAbstention
            | ReflectionTriggerKind::VoidHallucination => 0,
        }
    }
}

impl ManifoldGatewaySignal for ReflectionCritique {
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.gateway_frame
    }
}

/// Trait abstracting closed-loop trajectory critique and convergence selection over
/// `liminika-adapter` attractor recalls.
pub trait AttractorTrajectoryEvaluator {
    /// Evaluates `draft` relative to `query` on the compiled manifold.
    fn assess_trajectory(
        &self,
        query: &TrajectoryManifoldSnapshot,
        draft: &TrajectoryManifoldSnapshot,
        recall_policy: &RecallPolicy,
        policy: &ReflectionPolicy,
    ) -> ReflectionCritique;

    /// Chooses between Pass 1 and a corrective Pass 2 using strictly geometric criteria.
    fn select_better_pass(
        &self,
        policy: &ReflectionPolicy,
        pass1: &ReflectionCritique,
        pass2: &ReflectionCritique,
    ) -> ReflectionSelection {
        let p2_clean = matches!(
            pass2.trigger,
            ReflectionTriggerKind::Converged | ReflectionTriggerKind::ConsistentVoidAbstention
        );
        let p1_failed = !matches!(
            pass1.trigger,
            ReflectionTriggerKind::Converged | ReflectionTriggerKind::ConsistentVoidAbstention
        );
        if p1_failed && p2_clean {
            return ReflectionSelection::Pass2;
        }
        let p2_unsafe = matches!(
            pass2.trigger,
            ReflectionTriggerKind::GuardAttraction | ReflectionTriggerKind::VoidHallucination
        );
        if !p2_unsafe
            && pass2.alignment_score >= pass1.alignment_score + policy.alignment_improvement_eps
        {
            return ReflectionSelection::Pass2;
        }
        if !p2_unsafe
            && (pass2.alignment_score - pass1.alignment_score).abs()
                < policy.alignment_improvement_eps
            && pass2.layer_activation.repulse_sensitivity_q7
                < pass1.layer_activation.repulse_sensitivity_q7
            && pass2.layer_activation.circuit_resonance_q7
                > pass1.layer_activation.circuit_resonance_q7
        {
            return ReflectionSelection::Pass2;
        }
        ReflectionSelection::Pass1
    }
}

/// Default [`AttractorTrajectoryEvaluator`] combining Fact-basin overlap, `cvoid` drift checks,
/// and `Guard` contradiction margins via `liminika_core` [`GatewayFrame`] synthesis.
#[derive(Debug, Default, Clone, Copy)]
pub struct ManifoldReflectionEvaluator;

impl ManifoldReflectionEvaluator {
    fn anchor_overlap(
        query: &TrajectoryManifoldSnapshot,
        draft: &TrajectoryManifoldSnapshot,
    ) -> (f32, bool) {
        if query.recalled_fact_anchors.is_empty() {
            return (0.0, false);
        }
        let matched = query
            .recalled_fact_anchors
            .iter()
            .filter(|id| draft.recalled_fact_anchors.contains(id))
            .count();
        let ratio = matched as f32 / query.recalled_fact_anchors.len() as f32;
        let top_mismatch = match (query.top_fact_anchor, draft.top_fact_anchor) {
            (Some(q_id), Some(d_id)) => q_id != d_id && matched == 0,
            (Some(_), None) => true,
            _ => false,
        };
        (ratio, top_mismatch)
    }

    fn synthesize_corrective_frame(
        trigger: ReflectionTriggerKind,
        query: &TrajectoryManifoldSnapshot,
        draft: &TrajectoryManifoldSnapshot,
    ) -> GatewayFrame {
        match trigger {
            ReflectionTriggerKind::GuardAttraction => {
                let obstruction = (draft.guard_conflict_margin + 0.5).clamp(0.55, 1.0);
                GuardPhaseAnchor::new(query.top_fact_q7, obstruction, 0.4)
                    .synthesize_gateway_frame()
            }
            ReflectionTriggerKind::VoidDrift | ReflectionTriggerKind::BoundaryAmbiguity => {
                let missing_ids = query.missing_coalition_anchors(draft);
                if let Some((missing_anchor_id, missing_q7)) = missing_ids
                    .iter()
                    .find_map(|id| query.recalled_fact_q7s.iter().find(|(qid, _)| qid == id))
                {
                    ObjectiveFact::new(*missing_q7, 2.5, 0, *missing_anchor_id)
                        .synthesize_gateway_frame()
                } else {
                    let anchor_id = query.top_fact_anchor.unwrap_or(1);
                    ObjectiveFact::new(query.top_fact_q7, 2.5, 0, anchor_id)
                        .synthesize_gateway_frame()
                }
            }
            ReflectionTriggerKind::VoidHallucination => {
                SystemConstraint::new((), 0.25, 0.0, 0xFF).synthesize_gateway_frame()
            }
            ReflectionTriggerKind::Converged | ReflectionTriggerKind::ConsistentVoidAbstention => {
                draft.gateway_frame
            }
        }
    }
}

impl AttractorTrajectoryEvaluator for ManifoldReflectionEvaluator {
    fn assess_trajectory(
        &self,
        query: &TrajectoryManifoldSnapshot,
        draft: &TrajectoryManifoldSnapshot,
        recall_policy: &RecallPolicy,
        policy: &ReflectionPolicy,
    ) -> ReflectionCritique {
        let (anchor_overlap_ratio, anchor_mismatch) = Self::anchor_overlap(query, draft);
        let query_is_guard_trap = query.guard_conflict_margin > policy.guard_trigger_margin;
        let query_is_grounded =
            !matches!(query.void_state, VoidBoundaryState::CvoidEmptySpace) && !query_is_guard_trap;

        let (trigger, alignment_score, directive) = if query_is_grounded {
            let guard_pen = draft.guard_conflict_margin.max(0.0) * 1.5;
            let void_pen = match draft.void_state {
                VoidBoundaryState::InsideBasin => 0.0,
                VoidBoundaryState::LiminalBoundary => 0.08,
                VoidBoundaryState::CvoidEmptySpace => 0.25,
            };
            let align = draft.top1_fact_score + 0.20 * anchor_overlap_ratio - guard_pen - void_pen;

            if draft.guard_conflict_margin > policy.guard_trigger_margin {
                (
                    ReflectionTriggerKind::GuardAttraction,
                    align,
                    ReflectionCorrectiveDirective {
                        rethink_recommended: true,
                        inject_guard_repulsion: true,
                        suppress_injection: false,
                        rope_rad_delta: policy.max_rope_boost_rad,
                        srl_scale_delta: policy.srl_boost_delta,
                        anchor_mismatch,
                    },
                )
            } else if matches!(draft.void_state, VoidBoundaryState::CvoidEmptySpace)
                || (draft.top1_fact_score + policy.convergence_drop_tol < query.top1_fact_score
                    && anchor_overlap_ratio < 0.5)
            {
                (
                    ReflectionTriggerKind::VoidDrift,
                    align,
                    ReflectionCorrectiveDirective {
                        rethink_recommended: true,
                        inject_guard_repulsion: false,
                        suppress_injection: false,
                        rope_rad_delta: policy.max_rope_boost_rad,
                        srl_scale_delta: policy.srl_boost_delta,
                        anchor_mismatch,
                    },
                )
            } else if matches!(draft.void_state, VoidBoundaryState::LiminalBoundary)
                && (anchor_mismatch
                    || (draft.top1_fact_score - draft.top2_fact_score)
                        < recall_policy.boundary_margin)
            {
                (
                    ReflectionTriggerKind::BoundaryAmbiguity,
                    align,
                    ReflectionCorrectiveDirective {
                        rethink_recommended: true,
                        inject_guard_repulsion: false,
                        suppress_injection: false,
                        rope_rad_delta: policy.max_rope_boost_rad * 0.6,
                        srl_scale_delta: policy.srl_boost_delta * 0.5,
                        anchor_mismatch,
                    },
                )
            } else {
                (
                    ReflectionTriggerKind::Converged,
                    align,
                    ReflectionCorrectiveDirective {
                        rethink_recommended: false,
                        inject_guard_repulsion: false,
                        suppress_injection: false,
                        rope_rad_delta: 0.0,
                        srl_scale_delta: 0.0,
                        anchor_mismatch,
                    },
                )
            }
        } else if query_is_guard_trap {
            if draft.guard_conflict_margin > policy.guard_trigger_margin {
                let align = draft.top1_fact_score - draft.guard_conflict_margin * 1.5;
                (
                    ReflectionTriggerKind::GuardAttraction,
                    align,
                    ReflectionCorrectiveDirective {
                        rethink_recommended: true,
                        inject_guard_repulsion: true,
                        suppress_injection: false,
                        rope_rad_delta: policy.max_rope_boost_rad,
                        srl_scale_delta: policy.srl_boost_delta,
                        anchor_mismatch,
                    },
                )
            } else if !matches!(draft.void_state, VoidBoundaryState::CvoidEmptySpace) {
                let align = draft.top1_fact_score + 0.15;
                (
                    ReflectionTriggerKind::Converged,
                    align,
                    ReflectionCorrectiveDirective {
                        rethink_recommended: false,
                        inject_guard_repulsion: false,
                        suppress_injection: false,
                        rope_rad_delta: 0.0,
                        srl_scale_delta: 0.0,
                        anchor_mismatch: false,
                    },
                )
            } else {
                let align = 0.10 - draft.guard_conflict_margin.max(0.0);
                (
                    ReflectionTriggerKind::ConsistentVoidAbstention,
                    align,
                    ReflectionCorrectiveDirective {
                        rethink_recommended: false,
                        inject_guard_repulsion: false,
                        suppress_injection: true,
                        rope_rad_delta: 0.0,
                        srl_scale_delta: 0.0,
                        anchor_mismatch: false,
                    },
                )
            }
        } else {
            // Ungrounded OOD query (`CvoidEmptySpace` without Guard trap): trigger only on clear
            // Guard intrusion or strong Fact-basin hallucination well above `min_coherence`.
            let strong_basin_intrusion = matches!(draft.void_state, VoidBoundaryState::InsideBasin)
                && draft.top1_fact_score >= recall_policy.min_coherence * 1.50;
            let guard_intrusion = draft.guard_conflict_margin > policy.guard_trigger_margin;
            if !strong_basin_intrusion && !guard_intrusion {
                let align =
                    0.50 - draft.top1_fact_score.max(0.0) - draft.guard_conflict_margin.max(0.0);
                (
                    ReflectionTriggerKind::ConsistentVoidAbstention,
                    align,
                    ReflectionCorrectiveDirective {
                        rethink_recommended: false,
                        inject_guard_repulsion: false,
                        suppress_injection: true,
                        rope_rad_delta: 0.0,
                        srl_scale_delta: 0.0,
                        anchor_mismatch: false,
                    },
                )
            } else {
                let align = -draft.top1_fact_score.max(0.0) - draft.guard_conflict_margin.max(0.0);
                (
                    ReflectionTriggerKind::VoidHallucination,
                    align,
                    ReflectionCorrectiveDirective {
                        rethink_recommended: true,
                        inject_guard_repulsion: draft.guard_conflict_margin > 0.0,
                        suppress_injection: true,
                        rope_rad_delta: 0.0,
                        srl_scale_delta: 0.0,
                        anchor_mismatch: false,
                    },
                )
            }
        };

        let cross_activation = query
            .cognitive_trajectory
            .evaluate_cross_trajectory_activation(
                &draft.cognitive_trajectory,
                query.gateway_frame.phase_words_ref(),
                draft.gateway_frame.phase_words_ref(),
            );

        // If the pseudo-neural trajectory circuit detects omitted compound decomposition (`DecomposePeel`)
        // alongside incomplete Fact anchor coverage on a grounded query, escalate to BoundaryAmbiguity.
        let (trigger, alignment_score, mut directive) = if query_is_grounded
            && matches!(trigger, ReflectionTriggerKind::Converged)
            && matches!(cross_activation.op_kind(), CognitiveOpKind::DecomposePeel)
            && cross_activation.peel_gate_q7 >= 84
            && anchor_overlap_ratio < 1.0
        {
            (
                ReflectionTriggerKind::BoundaryAmbiguity,
                alignment_score - 0.05,
                ReflectionCorrectiveDirective {
                    rethink_recommended: true,
                    inject_guard_repulsion: false,
                    suppress_injection: false,
                    rope_rad_delta: policy.max_rope_boost_rad * 0.75,
                    srl_scale_delta: policy.srl_boost_delta * 0.65,
                    anchor_mismatch: true,
                },
            )
        } else {
            (trigger, alignment_score, directive)
        };

        if directive.rethink_recommended && !directive.suppress_injection {
            let circuit_gain = 1.0 + 0.25 * (f32::from(cross_activation.intensity_q7) / 128.0);
            directive.rope_rad_delta *= circuit_gain;
            directive.srl_scale_delta *= circuit_gain;
        }

        let gateway_frame = Self::synthesize_corrective_frame(trigger, query, draft);
        let mut cognitive_trajectory = query.cognitive_trajectory;
        let step_op = match trigger {
            ReflectionTriggerKind::GuardAttraction | ReflectionTriggerKind::VoidHallucination => {
                CognitiveOpKind::RepulseAudit
            }
            ReflectionTriggerKind::VoidDrift | ReflectionTriggerKind::BoundaryAmbiguity => {
                if matches!(cross_activation.op_kind(), CognitiveOpKind::DecomposePeel) {
                    CognitiveOpKind::DecomposePeel
                } else {
                    CognitiveOpKind::BridgeShift
                }
            }
            ReflectionTriggerKind::Converged => CognitiveOpKind::CrystallizeCommit,
            ReflectionTriggerKind::ConsistentVoidAbstention => CognitiveOpKind::Observe,
        };
        let intensity = if directive.rethink_recommended {
            (cross_activation.intensity_q7 >> 4).clamp(5, 7)
        } else {
            (cross_activation.intensity_q7 >> 5).clamp(2, 5)
        };
        cognitive_trajectory.push_step(CognitiveStepOp::pack(
            cross_activation.next_scale(),
            step_op,
            intensity,
        ));
        ReflectionCritique {
            trigger,
            alignment_score,
            anchor_overlap_ratio,
            directive,
            gateway_frame,
            cognitive_trajectory,
            layer_activation: cross_activation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use liminika_core::traits::gateway::CausalProvenanceSignal;

    fn make_snapshot(
        void_state: VoidBoundaryState,
        top1_fact: f32,
        top2_fact: f32,
        guard_margin: f32,
        anchors: Vec<u32>,
    ) -> TrajectoryManifoldSnapshot {
        let recalled_fact_q7s = anchors
            .iter()
            .map(|&id| (id, [4_i8; MAX_ATTENTION_HEADS]))
            .collect();
        TrajectoryManifoldSnapshot {
            void_state,
            top1_fact_score: top1_fact,
            top2_fact_score: top2_fact,
            top1_persona_score: 0.1,
            guard_conflict_margin: guard_margin,
            top_fact_anchor: anchors.first().copied(),
            top_fact_q7: [4_i8; MAX_ATTENTION_HEADS],
            recalled_fact_anchors: anchors,
            recalled_fact_q7s,
            action_decision: StigmergicActionDecision::PassiveSidecar,
            gateway_frame: GatewayFrame::default(),
            cognitive_trajectory: CognitiveTrajectoryLedger::new(),
        }
    }

    #[test]
    fn reflection_skips_retry_when_draft_converges_or_ood_abstains() {
        let eval = ManifoldReflectionEvaluator;
        let recall_policy = RecallPolicy::default();
        let policy = ReflectionPolicy::default();

        let query = make_snapshot(VoidBoundaryState::InsideBasin, 0.72, 0.41, -0.35, vec![1]);
        let draft = make_snapshot(VoidBoundaryState::InsideBasin, 0.70, 0.40, -0.30, vec![1]);
        let critique = eval.assess_trajectory(&query, &draft, &recall_policy, &policy);
        assert_eq!(critique.trigger, ReflectionTriggerKind::Converged);
        assert!(!critique.directive.rethink_recommended);
        assert!(critique.corrective_instruction("draft").is_none());

        let ood_q = make_snapshot(
            VoidBoundaryState::CvoidEmptySpace,
            0.12,
            0.09,
            -0.10,
            vec![],
        );
        let ood_d = make_snapshot(
            VoidBoundaryState::CvoidEmptySpace,
            0.10,
            0.08,
            -0.12,
            vec![],
        );
        let ood_crit = eval.assess_trajectory(&ood_q, &ood_d, &recall_policy, &policy);
        assert_eq!(
            ood_crit.trigger,
            ReflectionTriggerKind::ConsistentVoidAbstention
        );
        assert!(!ood_crit.directive.rethink_recommended);
    }

    #[test]
    fn reflection_triggers_on_guard_or_void_drift_and_selects_recovered_pass() {
        let eval = ManifoldReflectionEvaluator;
        let recall_policy = RecallPolicy::default();
        let policy = ReflectionPolicy::default();

        let query = make_snapshot(VoidBoundaryState::InsideBasin, 0.70, 0.45, -0.25, vec![2]);
        let bad_guard_draft =
            make_snapshot(VoidBoundaryState::InsideBasin, 0.48, 0.40, 0.12, vec![5]);
        let crit_guard = eval.assess_trajectory(&query, &bad_guard_draft, &recall_policy, &policy);
        assert_eq!(crit_guard.trigger, ReflectionTriggerKind::GuardAttraction);
        assert!(crit_guard.directive.rethink_recommended);
        assert!(crit_guard.directive.inject_guard_repulsion);
        assert_ne!(
            crit_guard.causal_provenance_bits() & CAUSAL_AXIS_CONTRADICTION_GUARD,
            0
        );
        assert!(crit_guard.corrective_instruction("wrong claim").is_some());
        assert_eq!(
            crit_guard
                .cognitive_trajectory
                .latest_step()
                .map(|s| s.op()),
            Some(CognitiveOpKind::RepulseAudit)
        );

        let recovered_draft =
            make_snapshot(VoidBoundaryState::InsideBasin, 0.74, 0.42, -0.32, vec![2]);
        let crit_recovered =
            eval.assess_trajectory(&query, &recovered_draft, &recall_policy, &policy);
        assert_eq!(crit_recovered.trigger, ReflectionTriggerKind::Converged);
        assert_eq!(
            eval.select_better_pass(&policy, &crit_guard, &crit_recovered),
            ReflectionSelection::Pass2
        );
    }

    #[test]
    fn reflection_flags_ood_hallucination_with_suppression_directive() {
        let eval = ManifoldReflectionEvaluator;
        let recall_policy = RecallPolicy::default();
        let policy = ReflectionPolicy::default();

        let ood_q = make_snapshot(
            VoidBoundaryState::CvoidEmptySpace,
            0.15,
            0.10,
            -0.05,
            vec![],
        );
        let shallow_d = make_snapshot(
            VoidBoundaryState::LiminalBoundary,
            0.24,
            0.15,
            -0.20,
            vec![1],
        );
        let shallow_crit = eval.assess_trajectory(&ood_q, &shallow_d, &recall_policy, &policy);
        assert_eq!(
            shallow_crit.trigger,
            ReflectionTriggerKind::ConsistentVoidAbstention
        );
        assert!(!shallow_crit.directive.rethink_recommended);

        let hall_d = make_snapshot(VoidBoundaryState::InsideBasin, 0.65, 0.30, -0.10, vec![1]);
        let crit = eval.assess_trajectory(&ood_q, &hall_d, &recall_policy, &policy);
        assert_eq!(crit.trigger, ReflectionTriggerKind::VoidHallucination);
        assert!(crit.directive.rethink_recommended);
        assert!(crit.directive.suppress_injection);
    }

    #[test]
    fn reflection_targets_omitted_coalition_anchor_in_corrective_gateway_frame() {
        let eval = ManifoldReflectionEvaluator;
        let recall_policy = RecallPolicy::default();
        let policy = ReflectionPolicy::default();

        let mut query = make_snapshot(
            VoidBoundaryState::InsideBasin,
            0.68,
            0.62,
            -0.25,
            vec![5, 6],
        );
        let q7_five = [3_i8; MAX_ATTENTION_HEADS];
        let q7_six = [21_i8; MAX_ATTENTION_HEADS];
        query.top_fact_q7 = q7_five;
        query.recalled_fact_q7s = vec![(5, q7_five), (6, q7_six)];

        // Draft stayed liminal and omitted coalition anchor #6 while partially matching #5.
        let draft = make_snapshot(
            VoidBoundaryState::LiminalBoundary,
            0.38,
            0.36,
            -0.20,
            vec![5],
        );
        assert_eq!(query.missing_coalition_anchors(&draft), vec![6]);

        let crit = eval.assess_trajectory(&query, &draft, &recall_policy, &policy);
        assert_eq!(crit.trigger, ReflectionTriggerKind::BoundaryAmbiguity);
        assert!(crit.directive.rethink_recommended);
        assert_eq!(crit.gateway_frame.target_anchor_id, 6);
        assert_eq!(crit.gateway_frame.phase_deltas, q7_six);
    }

    #[test]
    fn reflection_escalates_omitted_compound_decomposition_via_cognitive_circuit() {
        use liminika_core::traits::gateway::SwarmScaleGranularity;

        let eval = ManifoldReflectionEvaluator;
        let recall_policy = RecallPolicy::default();
        let policy = ReflectionPolicy::default();

        let mut query = make_snapshot(
            VoidBoundaryState::InsideBasin,
            0.70,
            0.64,
            -0.25,
            vec![101, 102],
        );
        query.cognitive_trajectory.push_step(CognitiveStepOp::pack(
            SwarmScaleGranularity::CellTrail,
            CognitiveOpKind::DecomposePeel,
            6,
        ));
        query.cognitive_trajectory.push_step(CognitiveStepOp::pack(
            SwarmScaleGranularity::PointerBit,
            CognitiveOpKind::RouteSubHead,
            5,
        ));
        query.gateway_frame = ObjectiveFact::new([24_i8; MAX_ATTENTION_HEADS], 2.5, 0, 101)
            .synthesize_gateway_frame();

        // Draft appears converged by scalar Fact score alone, yet only observed one anchor
        // without performing compound decomposition.
        let mut draft = make_snapshot(VoidBoundaryState::InsideBasin, 0.66, 0.25, -0.25, vec![101]);
        draft.cognitive_trajectory.push_step(CognitiveStepOp::pack(
            SwarmScaleGranularity::MacroCoalition,
            CognitiveOpKind::Observe,
            3,
        ));
        draft.gateway_frame = ObjectiveFact::new([24_i8; MAX_ATTENTION_HEADS], 2.0, 0, 101)
            .synthesize_gateway_frame();

        let crit = eval.assess_trajectory(&query, &draft, &recall_policy, &policy);
        assert_eq!(crit.trigger, ReflectionTriggerKind::BoundaryAmbiguity);
        assert!(crit.directive.rethink_recommended);
        assert_eq!(
            crit.layer_activation.op_kind(),
            CognitiveOpKind::DecomposePeel
        );
        assert!(crit.layer_activation.peel_gate_q7 >= 84);
        assert!(crit.directive.rope_rad_delta.abs() > (policy.max_rope_boost_rad * 0.75).abs());
    }
}
