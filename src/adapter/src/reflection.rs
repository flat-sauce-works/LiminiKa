// name: src/adapter/src/reflection.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Event-triggered closed-loop attractor reflection and trajectory critique.
//!
//! Rather than blindly re-running generation on every token or turn, [`AttractorTrajectoryEvaluator`]
//! inspects the geometric position of a generated draft on the compiled attractor manifold relative
//! to the originating query. Retries are triggered only when the draft exhibits `Guard` attraction,
//! manifold drift (`Liminal`), liminal ambiguity, or out-of-domain hallucination, and are strictly
//! bounded by [`ReflectionPolicy::max_retries`].

use crate::{DynamicAttractorRecall, LacunaBoundaryState, RecallPolicy};
use liminika_core::drivers::{LiminalEgoState, EGO_TRIGGER_LIMINAL_EXTERIOR};
use liminika_core::liminal::MAX_ATTENTION_HEADS;
use liminika_core::runtime::{
    AdaptiveThresholdPolicy, CausalProvenanceSignal, CognitiveCircuitThresholdPolicy, GatewayFrame,
    GuardPhaseAnchor, ManifoldGatewaySignal, ObjectiveFact, SystemConstraint,
    CAUSAL_AXIS_CONTRADICTION_GUARD, CAUSAL_AXIS_OBJECTIVE_FACT,
};
use liminika_core::stigmergy::{
    CognitiveCircuitActivation, CognitiveOpKind, CognitiveStepOp, CognitiveTrajectoryLedger,
    StigmergicActionDecision,
};

/// Policy governing bounded, event-triggered closed-loop reflection retries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReflectionPolicy {
    /// Maximum reflection retries per turn (`0` disables retries; `1..=2` keeps latency bounded).
    pub max_retries: usize,
    /// Draft `guard_conflict_margin` threshold above which [`ReflectionTriggerKind::GuardAttraction`] fires.
    pub guard_trigger_margin: f32,
    /// Allowed drop in `top1_fact_score` between query and draft before [`ReflectionTriggerKind::LacunaDrift`] fires.
    pub convergence_drop_tol: f32,
    /// Minimum alignment gain required for a retried trajectory to replace Pass 1.
    pub alignment_improvement_eps: f32,
    /// Additive RoPE steering delta (radians) applied on corrective Fact/Guard passes.
    pub max_rope_boost_rad: f32,
    /// Additive Rank-1 residual scale delta applied on corrective Fact/Guard passes.
    pub rank1_boost_delta: f32,
}

impl Default for ReflectionPolicy {
    fn default() -> Self {
        Self {
            max_retries: 1,
            guard_trigger_margin: 0.02,
            convergence_drop_tol: 0.08,
            alignment_improvement_eps: 0.02,
            max_rope_boost_rad: -0.35,
            rank1_boost_delta: -0.20,
        }
    }
}

/// Classification of a generated draft trajectory relative to the query's manifold intent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflectionTriggerKind {
    /// Draft converged safely inside a grounded Fact basin without Guard dominance.
    Converged,
    /// Out-of-domain query and draft both stayed cleanly outside the corpus (`LiminalExterior`).
    ConsistentLacunaAbstention,
    /// Draft drifted toward a contrastive `Guard` attractor (`guard_conflict_margin` elevated).
    GuardAttraction,
    /// Grounded query's draft dropped off-manifold into `LiminalExterior` or suffered a severe resonance drop.
    LacunaDrift,
    /// Draft landed in an unresolved `LiminalBoundary` or diverged from the query's top anchor.
    BoundaryAmbiguity,
    /// Ungrounded (`LiminalExterior`) query produced a draft that pulled unexpectedly into corpus attractors.
    LacunaHallucination,
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
    pub rank1_scale_delta: f32,
    /// True when the draft's top recalled Fact anchor diverges from the query's top Fact anchor.
    pub anchor_mismatch: bool,
}

/// Owned, lifetime-free manifold telemetry snapshot extracted from [`DynamicAttractorRecall`].
#[derive(Debug, Clone)]
pub struct TrajectoryManifoldSnapshot {
    /// Boundary state observed on the corpus manifold.
    pub lacuna_state: LacunaBoundaryState,
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
    /// Last-layer 64-head Q7 signature of the strongest observed Guard attractor, if any.
    ///
    /// Remains `None` when no repulsive Guard attractor has been observed so corrective reflection
    /// leaves repulsive poles unobserved (zero-aperture) rather than borrowing attractive phases.
    pub top_guard_q7: Option<[i8; MAX_ATTENTION_HEADS]>,
    /// Ordered list of recalled Fact anchor IDs.
    pub recalled_fact_anchors: Vec<u32>,
    /// Ordered `(anchor_id, last_layer_q7)` pairs for all recalled Fact basins in this trajectory.
    pub recalled_fact_q7s: Vec<(u32, [i8; MAX_ATTENTION_HEADS])>,
    /// Core substrate regime decision associated with this recall.
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
        let top_guard_q7 = recall
            .top_guard
            .and_then(|(e, _)| e.per_layer_q7.last().copied());
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
            lacuna_state: recall.lacuna_state,
            top1_fact_score: recall.top1_fact_score,
            top2_fact_score: recall.top2_fact_score,
            top1_persona_score: recall.top1_persona_score,
            guard_conflict_margin: recall.guard_conflict_margin,
            top_fact_anchor,
            top_fact_q7,
            top_guard_q7,
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
    /// Cross-trajectory stigmergic cognitive circuit activation comparing query and draft reasoning paths.
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
            ReflectionTriggerKind::LacunaDrift | ReflectionTriggerKind::BoundaryAmbiguity => {
                if compact_draft.is_empty() {
                    "Re-examine the full question using every detail in the grounded context."
                        .to_string()
                } else {
                    format!(
                        "The draft ({compact_draft}) drifted or omitted key grounded details. Re-examine the question and cover all grounded context points concisely."
                    )
                }
            }
            ReflectionTriggerKind::LacunaHallucination => {
                "Answer directly from general knowledge without introducing unrelated specialized terms."
                    .to_string()
            }
            ReflectionTriggerKind::Converged
            | ReflectionTriggerKind::ConsistentLacunaAbstention => return None,
        };
        Some(cue)
    }
}

impl CausalProvenanceSignal for ReflectionCritique {
    fn causal_provenance_bits(&self) -> u32 {
        match self.trigger {
            ReflectionTriggerKind::GuardAttraction => CAUSAL_AXIS_CONTRADICTION_GUARD,
            ReflectionTriggerKind::LacunaDrift | ReflectionTriggerKind::BoundaryAmbiguity => {
                CAUSAL_AXIS_OBJECTIVE_FACT
            }
            ReflectionTriggerKind::Converged
            | ReflectionTriggerKind::ConsistentLacunaAbstention
            | ReflectionTriggerKind::LacunaHallucination => 0,
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

    /// Evaluates `draft` relative to `query` while respecting active Core Layer 2 Ego
    /// boundary-transgression intent (`LiminalEgoState`).
    ///
    /// When `ego` is actively exploring beyond `Liminal` horizons (`transgression_bias_q7 >= 24` or
    /// `EGO_TRIGGER_LIMINAL_EXTERIOR` fired) and the draft introduces no Guard conflict, controlled
    /// `BoundaryAmbiguity` / mild `LacunaDrift` excursions are accepted as intentional exploration
    /// (`ReflectionTriggerKind::Converged`) rather than penalized as accidental drift, while
    /// Guard traps and ungrounded `LacunaHallucination` remain strictly rejected.
    fn assess_trajectory_with_ego(
        &self,
        query: &TrajectoryManifoldSnapshot,
        draft: &TrajectoryManifoldSnapshot,
        recall_policy: &RecallPolicy,
        policy: &ReflectionPolicy,
        ego: &LiminalEgoState,
    ) -> ReflectionCritique {
        let mut critique = self.assess_trajectory(query, draft, recall_policy, policy);
        let ego_exploring = ego.is_active()
            && (ego.transgression_bias_q7 >= 24
                || (ego.last_intervened_mask & EGO_TRIGGER_LIMINAL_EXTERIOR) != 0);
        let no_guard_conflict = draft.guard_conflict_margin <= policy.guard_trigger_margin;
        let controlled_excursion =
            matches!(
                critique.trigger,
                ReflectionTriggerKind::BoundaryAmbiguity | ReflectionTriggerKind::LacunaDrift
            ) && (!matches!(draft.lacuna_state, LacunaBoundaryState::LiminalExterior)
                || ego.transgression_bias_q7 >= 36
                || (ego.last_intervened_mask & EGO_TRIGGER_LIMINAL_EXTERIOR) != 0);

        if ego_exploring && no_guard_conflict && controlled_excursion {
            critique.trigger = ReflectionTriggerKind::Converged;
            critique.directive.rethink_recommended = false;
            critique.directive.inject_guard_repulsion = false;
            critique.directive.suppress_injection = false;
            critique.directive.rope_rad_delta = 0.0;
            critique.directive.rank1_scale_delta = 0.0;
        }
        critique
    }

    /// Chooses between Pass 1 and a corrective Pass 2 using strictly geometric criteria.
    fn select_better_pass(
        &self,
        policy: &ReflectionPolicy,
        pass1: &ReflectionCritique,
        pass2: &ReflectionCritique,
    ) -> ReflectionSelection {
        let p2_clean = matches!(
            pass2.trigger,
            ReflectionTriggerKind::Converged | ReflectionTriggerKind::ConsistentLacunaAbstention
        );
        let p1_failed = !matches!(
            pass1.trigger,
            ReflectionTriggerKind::Converged | ReflectionTriggerKind::ConsistentLacunaAbstention
        );
        if p1_failed && p2_clean {
            return ReflectionSelection::Pass2;
        }
        let p2_unsafe = matches!(
            pass2.trigger,
            ReflectionTriggerKind::GuardAttraction | ReflectionTriggerKind::LacunaHallucination
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

    /// Updates a persistent [`CognitiveTrajectoryLedger`] (persona reasoning circuit) using the
    /// core [`CognitiveTrajectoryLedger`] plasticity methods from the outcome of `critique`.
    fn update_persona_circuit(
        &self,
        circuit: &mut CognitiveTrajectoryLedger,
        critique: &ReflectionCritique,
    ) {
        let converged = matches!(
            critique.trigger,
            ReflectionTriggerKind::Converged | ReflectionTriggerKind::ConsistentLacunaAbstention
        );
        circuit.reinforce_from_activation(&critique.layer_activation, converged);
    }
}

/// Inputs evaluated by [`ManifoldReflectionEvaluator::should_trigger_reflection`] to bypass
/// redundant critique passes on calm grounded queries while firing selectively on boundary regions,
/// undigested Known Unknowns ($O \setminus S$), exit entropy spikes, or Guard traps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReflectionGateInput {
    /// $4p(1-p)$ boundary sensitivity in Q0.7 (`0..=127`) from [`liminika_core::liminal::LacunaPartition8`].
    pub boundary_sensitivity_q7: u8,
    /// True when the query contains an undigested Known Unknown ($O \setminus S$) exterior pointer.
    pub has_undigested_known_unknown: bool,
    /// Exit `EntropyController` / top-k entropy (`0.0..=1.0`).
    pub exit_entropy_h_tilde: f32,
    /// Guard conflict margin (`max_guard_score - top1_fact_score`).
    pub guard_conflict_margin: f32,
    /// True when the query lies strictly inside unobserved space ($\overline{S \cup O}$) without Guard threat.
    pub is_unobserved_lacuna: bool,
    /// Minimum Q0.7 $4p(1-p)$ boundary sensitivity required to trigger reflection (default `64`).
    pub min_boundary_sensitivity_q7: u8,
    /// Minimum exit entropy threshold required to trigger reflection (default `0.62`).
    pub entropy_trigger_thresh: f32,
    /// Guard conflict margin floor above which reflection fires unconditionally (default `0.02`).
    pub guard_trigger_margin: f32,
}

impl Default for ReflectionGateInput {
    fn default() -> Self {
        Self {
            boundary_sensitivity_q7: 0,
            has_undigested_known_unknown: false,
            exit_entropy_h_tilde: 0.0,
            guard_conflict_margin: -0.25,
            is_unobserved_lacuna: false,
            min_boundary_sensitivity_q7: 64,
            entropy_trigger_thresh: 0.62,
            guard_trigger_margin: 0.02,
        }
    }
}

/// Default [`AttractorTrajectoryEvaluator`] combining Fact-basin overlap, `Liminal` drift checks,
/// and `Guard` contradiction margins via `liminika_core` [`GatewayFrame`] synthesis.
#[derive(Debug, Default, Clone, Copy)]
pub struct ManifoldReflectionEvaluator;

impl ManifoldReflectionEvaluator {
    /// Returns `true` when reflection critique should run (Conditional Reflection Bypass):
    /// skips calm inside-basin hits and pure unobserved OOD queries, triggering only on high $4p(1-p)$
    /// boundary sensitivity, undigested $O \setminus S$, exit entropy spikes, or Guard conflict.
    #[must_use]
    pub fn should_trigger_reflection(&self, input: &ReflectionGateInput) -> bool {
        if input.is_unobserved_lacuna
            && input.guard_conflict_margin <= input.guard_trigger_margin
            && input.exit_entropy_h_tilde < input.entropy_trigger_thresh
        {
            return false;
        }
        input.boundary_sensitivity_q7 >= input.min_boundary_sensitivity_q7
            || input.has_undigested_known_unknown
            || input.exit_entropy_h_tilde >= input.entropy_trigger_thresh
            || input.guard_conflict_margin > input.guard_trigger_margin
    }

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
                if let Some(guard_q7) = draft.top_guard_q7.or(query.top_guard_q7) {
                    let obstruction = (draft.guard_conflict_margin + 0.5).clamp(0.55, 1.0);
                    GuardPhaseAnchor::new(guard_q7, obstruction, 0.4).synthesize_gateway_frame()
                } else {
                    // No repulsive target phase was observed: emit a zero-aperture frame rather
                    // than repurposing an attractive `top_fact_q7` pole for repulsion.
                    GatewayFrame::default()
                }
            }
            ReflectionTriggerKind::LacunaDrift | ReflectionTriggerKind::BoundaryAmbiguity => {
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
            ReflectionTriggerKind::LacunaHallucination => {
                SystemConstraint::new((), 0.25, 0.0, 0xFF).synthesize_gateway_frame()
            }
            ReflectionTriggerKind::Converged
            | ReflectionTriggerKind::ConsistentLacunaAbstention => draft.gateway_frame,
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
        let query_is_grounded = !matches!(query.lacuna_state, LacunaBoundaryState::LiminalExterior)
            && !query_is_guard_trap;

        let (trigger, alignment_score, directive) = if query_is_grounded {
            let guard_pen = draft.guard_conflict_margin.max(0.0) * 1.5;
            let lacuna_penalty = match draft.lacuna_state {
                LacunaBoundaryState::InsideBasin => 0.0,
                LacunaBoundaryState::LiminalBoundary => 0.08,
                LacunaBoundaryState::LiminalExterior => 0.25,
            };
            let align =
                draft.top1_fact_score + 0.20 * anchor_overlap_ratio - guard_pen - lacuna_penalty;

            if draft.guard_conflict_margin > policy.guard_trigger_margin {
                (
                    ReflectionTriggerKind::GuardAttraction,
                    align,
                    ReflectionCorrectiveDirective {
                        rethink_recommended: true,
                        inject_guard_repulsion: true,
                        suppress_injection: false,
                        rope_rad_delta: policy.max_rope_boost_rad,
                        rank1_scale_delta: policy.rank1_boost_delta,
                        anchor_mismatch,
                    },
                )
            } else if matches!(draft.lacuna_state, LacunaBoundaryState::LiminalExterior)
                || (draft.top1_fact_score + policy.convergence_drop_tol < query.top1_fact_score
                    && anchor_overlap_ratio < 0.5)
            {
                (
                    ReflectionTriggerKind::LacunaDrift,
                    align,
                    ReflectionCorrectiveDirective {
                        rethink_recommended: true,
                        inject_guard_repulsion: false,
                        suppress_injection: false,
                        rope_rad_delta: policy.max_rope_boost_rad,
                        rank1_scale_delta: policy.rank1_boost_delta,
                        anchor_mismatch,
                    },
                )
            } else if matches!(draft.lacuna_state, LacunaBoundaryState::LiminalBoundary)
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
                        rank1_scale_delta: policy.rank1_boost_delta * 0.5,
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
                        rank1_scale_delta: 0.0,
                        anchor_mismatch,
                    },
                )
            }
        } else if query_is_guard_trap {
            let evaded_counter_fact = !query.recalled_fact_anchors.is_empty()
                && (anchor_mismatch || anchor_overlap_ratio < 0.5);
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
                        rank1_scale_delta: policy.rank1_boost_delta,
                        anchor_mismatch,
                    },
                )
            } else if !matches!(draft.lacuna_state, LacunaBoundaryState::LiminalExterior)
                && evaded_counter_fact
            {
                let align = draft.top1_fact_score + 0.15 * anchor_overlap_ratio - 0.10;
                (
                    ReflectionTriggerKind::BoundaryAmbiguity,
                    align,
                    ReflectionCorrectiveDirective {
                        rethink_recommended: true,
                        inject_guard_repulsion: true,
                        suppress_injection: false,
                        rope_rad_delta: policy.max_rope_boost_rad * 0.75,
                        rank1_scale_delta: policy.rank1_boost_delta * 0.65,
                        anchor_mismatch: true,
                    },
                )
            } else if !matches!(draft.lacuna_state, LacunaBoundaryState::LiminalExterior) {
                let align = draft.top1_fact_score + 0.15;
                (
                    ReflectionTriggerKind::Converged,
                    align,
                    ReflectionCorrectiveDirective {
                        rethink_recommended: false,
                        inject_guard_repulsion: false,
                        suppress_injection: false,
                        rope_rad_delta: 0.0,
                        rank1_scale_delta: 0.0,
                        anchor_mismatch: false,
                    },
                )
            } else {
                let align = 0.10 - draft.guard_conflict_margin.max(0.0);
                (
                    ReflectionTriggerKind::ConsistentLacunaAbstention,
                    align,
                    ReflectionCorrectiveDirective {
                        rethink_recommended: false,
                        inject_guard_repulsion: false,
                        suppress_injection: true,
                        rope_rad_delta: 0.0,
                        rank1_scale_delta: 0.0,
                        anchor_mismatch: false,
                    },
                )
            }
        } else {
            // Ungrounded OOD query (`LiminalExterior` without Guard trap): trigger only on clear
            // Guard intrusion or strong Fact-basin hallucination well above `min_coherence`.
            let strong_basin_intrusion =
                matches!(draft.lacuna_state, LacunaBoundaryState::InsideBasin)
                    && draft.top1_fact_score >= recall_policy.min_coherence * 1.50;
            let guard_intrusion = draft.guard_conflict_margin > policy.guard_trigger_margin;
            if !strong_basin_intrusion && !guard_intrusion {
                let align =
                    0.50 - draft.top1_fact_score.max(0.0) - draft.guard_conflict_margin.max(0.0);
                (
                    ReflectionTriggerKind::ConsistentLacunaAbstention,
                    align,
                    ReflectionCorrectiveDirective {
                        rethink_recommended: false,
                        inject_guard_repulsion: false,
                        suppress_injection: true,
                        rope_rad_delta: 0.0,
                        rank1_scale_delta: 0.0,
                        anchor_mismatch: false,
                    },
                )
            } else {
                let align = -draft.top1_fact_score.max(0.0) - draft.guard_conflict_margin.max(0.0);
                (
                    ReflectionTriggerKind::LacunaHallucination,
                    align,
                    ReflectionCorrectiveDirective {
                        rethink_recommended: true,
                        inject_guard_repulsion: draft.guard_conflict_margin > 0.0,
                        suppress_injection: true,
                        rope_rad_delta: 0.0,
                        rank1_scale_delta: 0.0,
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

        // If the stigmergic cognitive circuit detects omitted compound decomposition (`DecomposePeel`)
        // alongside substantial Fact anchor omission (<= 55% coverage, top-anchor mismatch, liminal
        // boundary, or negative phase resonance) on a grounded query, escalate to BoundaryAmbiguity.
        // Majority-covered `InsideBasin` drafts with positive phase resonance avoid over-thinking.
        let peel_floor_q7 = AdaptiveThresholdPolicy::default().peel_escalation_floor_q7();
        let substantial_omission = anchor_overlap_ratio <= 0.55
            || (anchor_overlap_ratio < 1.0
                && (anchor_mismatch
                    || matches!(draft.lacuna_state, LacunaBoundaryState::LiminalBoundary)
                    || cross_activation.circuit_resonance_q7 < 0));
        let (trigger, alignment_score, mut directive) = if query_is_grounded
            && matches!(trigger, ReflectionTriggerKind::Converged)
            && matches!(cross_activation.op_kind(), CognitiveOpKind::DecomposePeel)
            && cross_activation.peel_gate_q7 >= peel_floor_q7
            && substantial_omission
        {
            (
                ReflectionTriggerKind::BoundaryAmbiguity,
                alignment_score - 0.05,
                ReflectionCorrectiveDirective {
                    rethink_recommended: true,
                    inject_guard_repulsion: false,
                    suppress_injection: false,
                    rope_rad_delta: policy.max_rope_boost_rad * 0.75,
                    rank1_scale_delta: policy.rank1_boost_delta * 0.65,
                    anchor_mismatch: true,
                },
            )
        } else {
            (trigger, alignment_score, directive)
        };

        if directive.rethink_recommended && !directive.suppress_injection {
            let circuit_gain = 1.0 + 0.25 * (f32::from(cross_activation.intensity_q7) / 128.0);
            directive.rope_rad_delta *= circuit_gain;
            directive.rank1_scale_delta *= circuit_gain;
        }

        let gateway_frame = Self::synthesize_corrective_frame(trigger, query, draft);
        let mut cognitive_trajectory = query.cognitive_trajectory;
        let step_op = match trigger {
            ReflectionTriggerKind::GuardAttraction | ReflectionTriggerKind::LacunaHallucination => {
                CognitiveOpKind::RepulseAudit
            }
            ReflectionTriggerKind::LacunaDrift | ReflectionTriggerKind::BoundaryAmbiguity => {
                if matches!(cross_activation.op_kind(), CognitiveOpKind::DecomposePeel) {
                    CognitiveOpKind::DecomposePeel
                } else {
                    CognitiveOpKind::BridgeShift
                }
            }
            ReflectionTriggerKind::Converged => CognitiveOpKind::CrystallizeCommit,
            ReflectionTriggerKind::ConsistentLacunaAbstention => CognitiveOpKind::Observe,
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
    use liminika_core::runtime::CausalProvenanceSignal;

    fn make_snapshot(
        lacuna_state: LacunaBoundaryState,
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
            lacuna_state,
            top1_fact_score: top1_fact,
            top2_fact_score: top2_fact,
            top1_persona_score: 0.1,
            guard_conflict_margin: guard_margin,
            top_fact_anchor: anchors.first().copied(),
            top_fact_q7: [4_i8; MAX_ATTENTION_HEADS],
            top_guard_q7: None,
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

        let query = make_snapshot(LacunaBoundaryState::InsideBasin, 0.72, 0.41, -0.35, vec![1]);
        let draft = make_snapshot(LacunaBoundaryState::InsideBasin, 0.70, 0.40, -0.30, vec![1]);
        let critique = eval.assess_trajectory(&query, &draft, &recall_policy, &policy);
        assert_eq!(critique.trigger, ReflectionTriggerKind::Converged);
        assert!(!critique.directive.rethink_recommended);
        assert!(critique.corrective_instruction("draft").is_none());

        let ood_q = make_snapshot(
            LacunaBoundaryState::LiminalExterior,
            0.12,
            0.09,
            -0.10,
            vec![],
        );
        let ood_d = make_snapshot(
            LacunaBoundaryState::LiminalExterior,
            0.10,
            0.08,
            -0.12,
            vec![],
        );
        let ood_crit = eval.assess_trajectory(&ood_q, &ood_d, &recall_policy, &policy);
        assert_eq!(
            ood_crit.trigger,
            ReflectionTriggerKind::ConsistentLacunaAbstention
        );
        assert!(!ood_crit.directive.rethink_recommended);
    }

    #[test]
    fn reflection_triggers_on_guard_or_lacuna_drift_and_selects_recovered_pass() {
        let eval = ManifoldReflectionEvaluator;
        let recall_policy = RecallPolicy::default();
        let policy = ReflectionPolicy::default();

        let query = make_snapshot(LacunaBoundaryState::InsideBasin, 0.70, 0.45, -0.25, vec![2]);
        let bad_guard_draft =
            make_snapshot(LacunaBoundaryState::InsideBasin, 0.48, 0.40, 0.12, vec![5]);
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
            make_snapshot(LacunaBoundaryState::InsideBasin, 0.74, 0.42, -0.32, vec![2]);
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
            LacunaBoundaryState::LiminalExterior,
            0.15,
            0.10,
            -0.05,
            vec![],
        );
        let shallow_d = make_snapshot(
            LacunaBoundaryState::LiminalBoundary,
            0.24,
            0.15,
            -0.20,
            vec![1],
        );
        let shallow_crit = eval.assess_trajectory(&ood_q, &shallow_d, &recall_policy, &policy);
        assert_eq!(
            shallow_crit.trigger,
            ReflectionTriggerKind::ConsistentLacunaAbstention
        );
        assert!(!shallow_crit.directive.rethink_recommended);

        let hall_d = make_snapshot(LacunaBoundaryState::InsideBasin, 0.65, 0.30, -0.10, vec![1]);
        let crit = eval.assess_trajectory(&ood_q, &hall_d, &recall_policy, &policy);
        assert_eq!(crit.trigger, ReflectionTriggerKind::LacunaHallucination);
        assert!(crit.directive.rethink_recommended);
        assert!(crit.directive.suppress_injection);
    }

    #[test]
    fn reflection_targets_omitted_coalition_anchor_in_corrective_gateway_frame() {
        let eval = ManifoldReflectionEvaluator;
        let recall_policy = RecallPolicy::default();
        let policy = ReflectionPolicy::default();

        let mut query = make_snapshot(
            LacunaBoundaryState::InsideBasin,
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
            LacunaBoundaryState::LiminalBoundary,
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
        use liminika_core::stigmergy::StigmergicScaleGranularity;

        let eval = ManifoldReflectionEvaluator;
        let recall_policy = RecallPolicy::default();
        let policy = ReflectionPolicy::default();

        let mut query = make_snapshot(
            LacunaBoundaryState::InsideBasin,
            0.70,
            0.64,
            -0.25,
            vec![101, 102],
        );
        query.cognitive_trajectory.push_step(CognitiveStepOp::pack(
            StigmergicScaleGranularity::CellTrail,
            CognitiveOpKind::DecomposePeel,
            6,
        ));
        query.cognitive_trajectory.push_step(CognitiveStepOp::pack(
            StigmergicScaleGranularity::PointerBit,
            CognitiveOpKind::RouteSubHead,
            5,
        ));
        query.gateway_frame = ObjectiveFact::new([24_i8; MAX_ATTENTION_HEADS], 2.5, 0, 101)
            .synthesize_gateway_frame();

        // Draft appears converged by scalar Fact score alone, yet only observed one anchor
        // without performing compound decomposition.
        let mut draft = make_snapshot(
            LacunaBoundaryState::InsideBasin,
            0.66,
            0.25,
            -0.25,
            vec![101],
        );
        draft.cognitive_trajectory.push_step(CognitiveStepOp::pack(
            StigmergicScaleGranularity::MacroCoalition,
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

    #[test]
    fn reflection_avoids_overthinking_when_majority_coalition_converges_inside_basin() {
        use liminika_core::stigmergy::StigmergicScaleGranularity;

        let eval = ManifoldReflectionEvaluator;
        let recall_policy = RecallPolicy::default();
        let policy = ReflectionPolicy::default();

        let mut query = make_snapshot(
            LacunaBoundaryState::InsideBasin,
            0.65,
            0.60,
            -0.25,
            vec![1, 6, 5],
        );
        query.cognitive_trajectory.push_step(CognitiveStepOp::pack(
            StigmergicScaleGranularity::CellTrail,
            CognitiveOpKind::DecomposePeel,
            6,
        ));
        query.gateway_frame =
            ObjectiveFact::new([24_i8; MAX_ATTENTION_HEADS], 2.5, 0, 1).synthesize_gateway_frame();

        // Draft covers the primary pair (2/3 = 0.67 > 0.55) inside the basin with aligned phase.
        let mut draft = make_snapshot(
            LacunaBoundaryState::InsideBasin,
            0.62,
            0.55,
            -0.22,
            vec![1, 6],
        );
        draft.cognitive_trajectory.push_step(CognitiveStepOp::pack(
            StigmergicScaleGranularity::MacroCoalition,
            CognitiveOpKind::Observe,
            3,
        ));
        draft.gateway_frame =
            ObjectiveFact::new([24_i8; MAX_ATTENTION_HEADS], 2.2, 0, 1).synthesize_gateway_frame();

        let crit = eval.assess_trajectory(&query, &draft, &recall_policy, &policy);
        assert_eq!(crit.trigger, ReflectionTriggerKind::Converged);
        assert!(!crit.directive.rethink_recommended);
    }

    #[test]
    fn reflection_flags_guard_trap_vague_evasion_when_counter_fact_anchor_is_omitted() {
        let eval = ManifoldReflectionEvaluator;
        let recall_policy = RecallPolicy::default();
        let policy = ReflectionPolicy::default();

        // Query tripped the Guard margin while recalling counter-fact anchor #2.
        let query = make_snapshot(
            LacunaBoundaryState::LiminalBoundary,
            0.58,
            0.35,
            0.18,
            vec![2],
        );
        // Draft stayed liminal without directly tripping the guard, yet omitted counter-fact #2.
        let evasive_draft = make_snapshot(
            LacunaBoundaryState::LiminalBoundary,
            0.36,
            0.30,
            -0.05,
            vec![9],
        );
        let crit = eval.assess_trajectory(&query, &evasive_draft, &recall_policy, &policy);
        assert_eq!(crit.trigger, ReflectionTriggerKind::BoundaryAmbiguity);
        assert!(crit.directive.rethink_recommended);
        assert!(crit.directive.inject_guard_repulsion);
        assert!(crit.directive.anchor_mismatch);

        // Once Pass 2 grounds on counter-fact anchor #2 inside the basin, it converges.
        let grounded_draft =
            make_snapshot(LacunaBoundaryState::InsideBasin, 0.68, 0.32, -0.25, vec![2]);
        let crit_ok = eval.assess_trajectory(&query, &grounded_draft, &recall_policy, &policy);
        assert_eq!(crit_ok.trigger, ReflectionTriggerKind::Converged);
        assert!(!crit_ok.directive.rethink_recommended);
        assert_eq!(
            eval.select_better_pass(&policy, &crit, &crit_ok),
            ReflectionSelection::Pass2
        );
    }

    #[test]
    fn reflection_ego_allows_controlled_exploration_but_rejects_guard_and_hallucination() {
        use liminika_core::drivers::EgoIntentAxes;

        let eval = ManifoldReflectionEvaluator;
        let recall_policy = RecallPolicy::default();
        let policy = ReflectionPolicy::default();

        let ego = LiminalEgoState::from_intent_axes(
            &EgoIntentAxes::new()
                .with_autonomy_q7(80)
                .with_transgression_q7(72),
        );

        let query = make_snapshot(
            LacunaBoundaryState::InsideBasin,
            0.65,
            0.35,
            -0.25,
            vec![1, 2],
        );
        // Draft explores across the LiminalBoundary with no Guard conflict.
        let exploring_draft = make_snapshot(
            LacunaBoundaryState::LiminalBoundary,
            0.40,
            0.38,
            -0.10,
            vec![3],
        );
        let baseline_crit =
            eval.assess_trajectory(&query, &exploring_draft, &recall_policy, &policy);
        assert!(baseline_crit.directive.rethink_recommended);

        let ego_crit = eval.assess_trajectory_with_ego(
            &query,
            &exploring_draft,
            &recall_policy,
            &policy,
            &ego,
        );
        assert_eq!(ego_crit.trigger, ReflectionTriggerKind::Converged);
        assert!(!ego_crit.directive.rethink_recommended);

        // Guard attraction must still be rejected even with an active transgressive Ego.
        let guard_draft = make_snapshot(
            LacunaBoundaryState::LiminalBoundary,
            0.55,
            0.25,
            0.30,
            vec![1],
        );
        let guard_crit =
            eval.assess_trajectory_with_ego(&query, &guard_draft, &recall_policy, &policy, &ego);
        assert_eq!(guard_crit.trigger, ReflectionTriggerKind::GuardAttraction);
        assert!(guard_crit.directive.rethink_recommended);

        // Ungrounded OOD query hallucination must still be suppressed.
        let ood_query = make_snapshot(
            LacunaBoundaryState::LiminalExterior,
            0.10,
            0.05,
            -0.20,
            vec![],
        );
        let ood_hallucination = make_snapshot(
            LacunaBoundaryState::InsideBasin,
            0.72,
            0.30,
            -0.10,
            vec![99],
        );
        let ood_crit = eval.assess_trajectory_with_ego(
            &ood_query,
            &ood_hallucination,
            &recall_policy,
            &policy,
            &ego,
        );
        assert_eq!(ood_crit.trigger, ReflectionTriggerKind::LacunaHallucination);
        assert!(ood_crit.directive.suppress_injection);
    }

    #[test]
    fn conditional_reflection_gate_skips_calm_hits_and_triggers_on_boundary_or_entropy() {
        let eval = ManifoldReflectionEvaluator;
        let calm = ReflectionGateInput {
            boundary_sensitivity_q7: 16,
            has_undigested_known_unknown: false,
            exit_entropy_h_tilde: 0.25,
            guard_conflict_margin: -0.30,
            is_unobserved_lacuna: false,
            ..ReflectionGateInput::default()
        };
        assert!(!eval.should_trigger_reflection(&calm));

        let lacuna_calm = ReflectionGateInput {
            is_unobserved_lacuna: true,
            boundary_sensitivity_q7: 96, // ignored under peaceful Unobserved Lacuna
            ..calm
        };
        assert!(!eval.should_trigger_reflection(&lacuna_calm));

        let boundary_hit = ReflectionGateInput {
            boundary_sensitivity_q7: 80,
            ..calm
        };
        assert!(eval.should_trigger_reflection(&boundary_hit));

        let known_unknown = ReflectionGateInput {
            has_undigested_known_unknown: true,
            ..calm
        };
        assert!(eval.should_trigger_reflection(&known_unknown));

        let entropy_spike = ReflectionGateInput {
            exit_entropy_h_tilde: 0.71,
            ..calm
        };
        assert!(eval.should_trigger_reflection(&entropy_spike));

        let guard_trap = ReflectionGateInput {
            guard_conflict_margin: 0.08,
            ..calm
        };
        assert!(eval.should_trigger_reflection(&guard_trap));
    }

    #[test]
    fn reflection_guard_attraction_uses_observed_guard_antiphase_and_zero_aperture_when_unobserved()
    {
        let eval = ManifoldReflectionEvaluator;
        let recall_policy = RecallPolicy::default();
        let policy = ReflectionPolicy::default();

        let mut query = make_snapshot(LacunaBoundaryState::InsideBasin, 0.72, 0.40, -0.25, vec![1]);
        query.top_fact_q7 = [42_i8; MAX_ATTENTION_HEADS];
        query.top_guard_q7 = None;

        // 1. Unobserved guard pole -> zero-aperture GatewayFrame::default() without borrowing top_fact_q7.
        let mut draft_unobserved =
            make_snapshot(LacunaBoundaryState::InsideBasin, 0.45, 0.35, 0.20, vec![1]);
        draft_unobserved.top_guard_q7 = None;
        let crit_unobs = eval.assess_trajectory(&query, &draft_unobserved, &recall_policy, &policy);
        assert_eq!(crit_unobs.trigger, ReflectionTriggerKind::GuardAttraction);
        assert_eq!(crit_unobs.gateway_frame, GatewayFrame::default());
        assert_eq!(
            crit_unobs.gateway_frame.phase_deltas,
            [0_i8; MAX_ATTENTION_HEADS]
        );
        assert!(!crit_unobs.gateway_frame.is_repulsed());
        assert_eq!(crit_unobs.gateway_frame.effective_energy, 0.0);

        // 2. Observed guard pole (including i8::MIN saturation) -> saturating anti-phase `-guard`.
        let mut guard_pole = [0_i8; MAX_ATTENTION_HEADS];
        guard_pole[0] = 35;
        guard_pole[1] = -28;
        guard_pole[2] = i8::MIN;
        let mut draft_observed = draft_unobserved.clone();
        draft_observed.top_guard_q7 = Some(guard_pole);
        let crit_obs = eval.assess_trajectory(&query, &draft_observed, &recall_policy, &policy);
        assert_eq!(crit_obs.trigger, ReflectionTriggerKind::GuardAttraction);
        assert!(crit_obs.gateway_frame.is_repulsed());
        let mut expected_anti = [0_i8; MAX_ATTENTION_HEADS];
        expected_anti[0] = -35;
        expected_anti[1] = 28;
        expected_anti[2] = 127;
        assert_eq!(crit_obs.gateway_frame.phase_deltas, expected_anti);
        assert_ne!(crit_obs.gateway_frame.phase_deltas[0], -42);

        // 3. Query-side observed guard pole is used when draft did not carry one.
        let mut query_with_guard = query;
        query_with_guard.top_guard_q7 = Some(guard_pole);
        let crit_query_guard = eval.assess_trajectory(
            &query_with_guard,
            &draft_unobserved,
            &recall_policy,
            &policy,
        );
        assert_eq!(
            crit_query_guard.trigger,
            ReflectionTriggerKind::GuardAttraction
        );
        assert!(crit_query_guard.gateway_frame.is_repulsed());
        assert_eq!(crit_query_guard.gateway_frame.phase_deltas, expected_anti);
    }
}
