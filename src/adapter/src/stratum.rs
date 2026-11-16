// name: src/adapter/src/stratum.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hierarchical Society-of-Mind strata (`n - 1 <-> n <-> n + 1`) built over [`liminika_core`]
//! `ActionHub` and role-scoped `cvoid` boundary geometry.
//!
//! Each [`SocietyStratum`] operates only on signals exposed by adjacent layers, allowing dynamic
//! insertion, removal, or replacement via [`StratumPipeline`] without coupling lower feature
//! workers directly to global dialogue policy.

use crate::{
    AsymmetricDominoVoidPerspective, AttractorPerspectiveContext, AttractorRole,
    AttractorVoidPerspective, CoherenceGuardGate, RecallBoundaryGate, RecallBoundaryMetrics,
    RecallGateDecision, RecallPolicy, TextAttractorCorpus, TextAttractorEntry, VoidBoundaryState,
};
use liminika_core::abi::{gcso_pointer_trail_t, gcso_tagged_ptr_t};
use liminika_core::action_hub::{
    ActionHubEngine, ActionOrchestrationReport, StigmergicActionDecision, StigmergicActionReceptor,
    StigmergicThresholds, ACTION_HUB_INLINE_TRAILS,
};
use liminika_core::traits::gateway::{
    ActionHubScaleOrchestrator, CausalProvenanceReport, CognitiveCircuitActivation,
    CognitiveOpKind, CognitiveStepOp, CognitiveTrajectoryLedger, GatewayFrame, GuardPhaseAnchor,
    ManifoldGatewaySignal, ObjectiveFact, ScaleLayerNode, SubjectivePersona, SwarmScaleGranularity,
};
use liminika_core::traits::primitives::{load_q7_words_64, GcsoResult, MAX_ATTENTION_HEADS};
use liminika_core::traits::telemetry::{
    StigmergicSwarmOps, SwarmResolutionReport, NUM_SWARM_MICRO_CELLS,
};

/// Uncollapsed per-query-token resonance observation deposited into stigmergic swarm trails.
#[derive(Debug, Clone)]
pub struct TokenResonanceDeposit {
    /// Unsupervised token discriminability weight (`w_t * disc^2`, or uniform fallback weight).
    pub disc_weight: f32,
    /// Per-corpus-entry centered MaxSim excess (`row_sim[i] - mean_row`) for this token.
    pub entry_excess: Vec<f32>,
    /// Pre-baked 64-head Q7 phase words (`[u64; 8]`) for 8-cell Micro-Swarm evaluation at the
    /// final active RoPE layer.
    pub token_phase_words: [u64; NUM_SWARM_MICRO_CELLS],
}

/// Uncollapsed local multi-token channel observation passed into [`TextAttractorCorpus`].
#[derive(Debug, Clone, Default)]
pub struct LocalChannelObservation {
    /// Combined symmetric Chamfer score (`0.65 * q_to_d + 0.35 * d_to_q`) per corpus entry.
    pub combined_scores: Vec<f32>,
    /// Independent query-to-doc discriminative centered MaxSim scores per corpus entry.
    pub q_to_d_scores: Vec<f32>,
    /// Independent doc-to-query Chamfer specificity coverage scores per corpus entry.
    pub d_to_q_scores: Vec<f32>,
    /// Uncollapsed per-query-token resonance deposits evaluated by the stigmergic swarm.
    pub token_deposits: Vec<TokenResonanceDeposit>,
}

/// [`StigmergicActionReceptor`] tracking multi-trail pointer emergence when [`ActionHubEngine`]
/// links disjoint factual sub-basins across residual unexplained query tokens.
#[derive(Debug, Clone, Default)]
pub struct ActionHubInterpretiveReceptor {
    /// Ordered sequence of dominant anchor IDs observed during trail emergence.
    pub emerged_anchor_chain: Vec<u32>,
    /// Last core [`ActionOrchestrationReport`] emitted upon stigmergic emergence.
    pub last_report: Option<ActionOrchestrationReport>,
}

impl StigmergicActionReceptor for ActionHubInterpretiveReceptor {
    fn on_stigmergic_action_emerged(
        &mut self,
        _frame: &GatewayFrame,
        report: &ActionOrchestrationReport,
        trail: &gcso_pointer_trail_t,
    ) -> GcsoResult<()> {
        self.last_report = Some(*report);
        if report.dominant_anchor_id != 0
            && !self
                .emerged_anchor_chain
                .contains(&report.dominant_anchor_id)
        {
            self.emerged_anchor_chain.push(report.dominant_anchor_id);
        }
        if trail.target_anchor_id != 0
            && !self.emerged_anchor_chain.contains(&trail.target_anchor_id)
        {
            self.emerged_anchor_chain.push(trail.target_anchor_id);
        }
        Ok(())
    }
}

/// Per-query diagnostic telemetry bridging adapter stratum channels with [`liminika_core`]
/// [`CausalProvenanceReport`] so `ActionHub` and evaluation harnesses observe channel agreement.
#[derive(Debug, Clone, Copy, Default)]
pub struct StratumProbeTelemetry {
    /// Winning anchor ID from the global pooled cosine channel before stratum modulation.
    pub global_top1_anchor: Option<u32>,
    /// Winning score from the global pooled cosine channel.
    pub global_top1_score: f32,
    /// Winning anchor ID from the multi-token MaxSim channel (`local_token_scores`), if active.
    pub local_top1_anchor: Option<u32>,
    /// Winning score from the multi-token MaxSim channel (`local_token_scores`).
    pub local_top1_score: f32,
    /// Winning anchor ID after full `StratumPipeline` arbitration.
    pub final_top1_anchor: Option<u32>,
    /// True when both channels are active and disagree on the top-1 anchor ID.
    pub channel_disagreement: bool,
    /// Core 16-byte causal provenance report synthesized via [`GatewayFrame::to_causal_report`].
    pub provenance: CausalProvenanceReport,
    /// Recorded 8-step multi-scale cognitive trajectory from `ActionHub` FastPathLedger mini ledger.
    pub cognitive_trajectory: CognitiveTrajectoryLedger,
    /// Latest pseudo-neural cognitive circuit gating activation computed from trajectory history.
    pub layer_activation: CognitiveCircuitActivation,
}

/// Bottom-up and top-down signal flowing across a Society-of-Mind [`StratumPipeline`].
///
/// Lower strata (`n - 1`) populate local features and stigmergic pointer weights without needing
/// global semantic awareness, while upper strata (`n + 1`) arbitrate `cvoid` boundaries, Guard
/// conflicts, and `ActionHub` regimes.
#[derive(Debug, Clone)]
pub struct StratumSignal {
    /// Per-attractor score (`-1.0..=1.0+`) updated by each stratum in the stack.
    pub candidate_scores: Vec<f32>,
    /// Independent per-attractor multi-token MaxSim score (`-1.0..=1.0`) supplied by a
    /// representation sensor, or empty when operating in single-vector mode.
    pub local_token_scores: Vec<f32>,
    /// Independent query-to-doc discriminative centered MaxSim scores per corpus entry.
    pub q_to_d_scores: Vec<f32>,
    /// Independent doc-to-query Chamfer specificity coverage scores per corpus entry.
    pub d_to_q_scores: Vec<f32>,
    /// Uncollapsed per-query-token resonance deposits evaluated by the stigmergic swarm.
    pub token_deposits: Vec<TokenResonanceDeposit>,
    /// Ordered multi-trail Fact decomposition `(entry_idx, score)` selected by [`ActionHubEngine`].
    pub linked_fact_indices: Vec<(usize, f32)>,
    /// True when [`ActionHubEngine`] interpretive decomposition linked disjoint Fact sub-basins.
    pub decomposition_triggered: bool,
    /// Per-attractor local `cvoid` gap depth (`0.0` = deep inside basin).
    pub local_void_depths: Vec<f32>,
    /// Per-attractor stigmergic pointer trace / pheromone strength (`>= 0.0`).
    pub stigmergic_weights: Vec<f32>,
    /// Aggregated boundary metrics observed at the upper boundary stratum.
    pub boundary_metrics: RecallBoundaryMetrics,
    /// Boundary gate decision (`InsideBasin`, `LiminalBoundary`, or `CvoidEmptySpace`).
    pub gate_decision: RecallGateDecision,
    /// `ActionHub` regime chosen for top-down orchestration.
    pub action_decision: StigmergicActionDecision,
    /// Synthesized 128-byte core [`GatewayFrame`] serving as the universal arbitration gate.
    pub gateway_frame: GatewayFrame,
    /// 8-cell Micro-Swarm consensus report between global and local phase channels.
    pub swarm_report: SwarmResolutionReport,
    /// True when the local multi-token channel exhibits sharper unimodal support than global pooling.
    pub local_channel_preferred: bool,
    /// Cross-channel probe telemetry and core [`CausalProvenanceReport`].
    pub telemetry: StratumProbeTelemetry,
}

/// Top-down directive sent from stratum `n + 1` down to `n` and `n - 1` after upward inference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StratumDirective {
    /// Whether injection should be suppressed (`CvoidEmptySpace` or Guard repulsion).
    pub suppress_injection: bool,
    /// Whether active Guard repulsion (`RepulsionEscape`) was triggered.
    pub repulse_guard: bool,
    /// Top-down gain multiplier (`0.0` when skipped/repulsed, `< 1.0` on liminal boundary, `>= 1.0` inside basin).
    pub top_down_gain: f32,
    /// Minimum local `cvoid` depth observed across the stack.
    pub min_local_void_depth: f32,
}

impl Default for StratumDirective {
    fn default() -> Self {
        Self {
            suppress_injection: false,
            repulse_guard: false,
            top_down_gain: 1.0,
            min_local_void_depth: 1.0,
        }
    }
}

/// Read-only shared context passed to each [`SocietyStratum`] during upward and downward passes.
pub struct StratumContext<'a> {
    /// Corpus entries aligned with `signal.candidate_scores`.
    pub entries: &'a [TextAttractorEntry],
    /// Unprojected steered-layer residual signature for the query.
    pub raw_sig: &'a [f32],
    /// Common-mode-projected normalized query signature.
    pub query_sig: &'a [f32],
    /// Common-mode-projected normalized corpus signatures.
    pub entry_sigs: &'a [Vec<f32>],
    /// Rotary dimension per steered layer chunk (`q_rot_len`).
    pub layer_chunk_dim: usize,
    /// Number of query attention heads (`H_Q`) inside each steered layer chunk.
    pub num_q_heads: usize,
    /// Active rotated dimensions per attention head (`rotary_dim`).
    pub rotary_dim: usize,
    /// Active recall policy thresholds.
    pub policy: &'a RecallPolicy,
}

/// Pluggable hierarchical layer in a Society-of-Mind recall/steering stack (`n - 1 <-> n <-> n + 1`).
///
/// Strata can be dynamically added, inserted, or removed via [`StratumPipeline`] without altering
/// neighboring layers.
pub trait SocietyStratum {
    /// Stable identifier for inspection and dynamic removal/insertion.
    fn name(&self) -> &'static str;

    /// Bottom-up inference step (`n - 1 -> n -> n + 1`): transforms `signal` using local rules.
    ///
    /// # Errors
    /// Returns an error if input dimensions or invariants required by this stratum are invalid.
    fn infer_up(&self, ctx: &StratumContext<'_>, signal: &mut StratumSignal) -> Result<(), String>;

    /// Top-down self-adjustment step (`n + 1 -> n -> n - 1`): updates `directive` or modulates
    /// lower-stratum `stigmergic_weights` / scores. Default implementation is a no-op.
    fn adjust_down(
        &self,
        _ctx: &StratumContext<'_>,
        _directive: &mut StratumDirective,
        _signal: &mut StratumSignal,
    ) {
    }
}

/// Stratum 0 (`n = 0`): local feature worker extracting per-layer viewing-angle contrast relative
/// to same-role peers on common-mode-projected signatures and recording stigmergic weights.
#[derive(Debug, Clone, Copy)]
pub struct LocalFeatureWorkerStratum {
    /// Weight applied to top-layer viewing-angle contrast modulated by positive global support.
    pub layer_contrast_gain: f32,
}

impl Default for LocalFeatureWorkerStratum {
    fn default() -> Self {
        Self {
            layer_contrast_gain: 0.35,
        }
    }
}

impl SocietyStratum for LocalFeatureWorkerStratum {
    fn name(&self) -> &'static str {
        "local_feature_worker"
    }

    fn infer_up(&self, ctx: &StratumContext<'_>, signal: &mut StratumSignal) -> Result<(), String> {
        let n_entries = ctx.entries.len();
        let expected = ctx.layer_chunk_dim;
        let use_projected = !ctx.query_sig.is_empty()
            && ctx.query_sig.len() == ctx.raw_sig.len()
            && ctx.entry_sigs.len() == n_entries;
        let query_vec = if use_projected {
            ctx.query_sig
        } else {
            ctx.raw_sig
        };

        if expected > 0 && query_vec.len() >= expected && query_vec.len().is_multiple_of(expected) {
            let num_steered = query_vec.len() / expected;
            if num_steered >= 2 {
                let mut layer_cos: Vec<Vec<f32>> = vec![vec![0.0_f32; num_steered]; n_entries];
                for (i, entry) in ctx.entries.iter().enumerate() {
                    let entry_vec: &[f32] = if use_projected {
                        &ctx.entry_sigs[i]
                    } else {
                        &entry.residual_sig
                    };
                    if entry_vec.len() != query_vec.len() {
                        continue;
                    }
                    for (l, slot) in layer_cos[i].iter_mut().enumerate().take(num_steered) {
                        let start = l * expected;
                        let end = start + expected;
                        *slot = TextAttractorCorpus::cosine_f32(
                            &query_vec[start..end],
                            &entry_vec[start..end],
                        );
                    }
                }
                for role in [
                    AttractorRole::Fact,
                    AttractorRole::Persona,
                    AttractorRole::Guard,
                ] {
                    let idxs: Vec<usize> = ctx
                        .entries
                        .iter()
                        .enumerate()
                        .filter(|(_, e)| e.role == role)
                        .map(|(i, _)| i)
                        .collect();
                    if idxs.len() <= 1 {
                        continue;
                    }
                    let mut role_layer_mean = vec![0.0_f32; num_steered];
                    for &i in &idxs {
                        for (l, mean_slot) in role_layer_mean.iter_mut().enumerate() {
                            *mean_slot += layer_cos[i][l] / (idxs.len() as f32);
                        }
                    }
                    let top_l = (num_steered / 4).max(1);
                    for &i in &idxs {
                        let mut diffs: Vec<f32> = (0..num_steered)
                            .map(|l| layer_cos[i][l] - role_layer_mean[l])
                            .collect();
                        diffs.sort_by(|a, b| b.total_cmp(a));
                        let top_contrast = diffs[..top_l].iter().sum::<f32>() / (top_l as f32);
                        let pos = signal.candidate_scores[i].max(0.0);
                        signal.candidate_scores[i] += self.layer_contrast_gain * pos * top_contrast;
                        signal.stigmergic_weights[i] = (1.0 + top_contrast).clamp(0.5, 1.5);
                    }
                }
            }
        }
        Ok(())
    }

    fn adjust_down(
        &self,
        _ctx: &StratumContext<'_>,
        directive: &mut StratumDirective,
        signal: &mut StratumSignal,
    ) {
        if directive.repulse_guard || directive.suppress_injection {
            for w in &mut signal.stigmergic_weights {
                *w *= 0.25;
            }
        } else {
            for w in &mut signal.stigmergic_weights {
                *w = (*w * directive.top_down_gain).clamp(0.25, 2.0);
            }
        }
    }
}

/// Stratum 0.5: multi-head viewing-angle worker treating each attention head (`0..num_q_heads`)
/// across steered layers as an independent perspective on common-mode-projected signatures so
/// specialized entity heads are not diluted by head-wide averaging or shared template framing.
#[derive(Debug, Clone, Copy)]
pub struct HeadAngleWorkerStratum {
    /// Gain applied to top-head viewing-angle contrast modulated by positive global support.
    pub head_contrast_gain: f32,
}

impl Default for HeadAngleWorkerStratum {
    fn default() -> Self {
        Self {
            head_contrast_gain: 0.40,
        }
    }
}

impl SocietyStratum for HeadAngleWorkerStratum {
    fn name(&self) -> &'static str {
        "head_angle_worker"
    }

    fn infer_up(&self, ctx: &StratumContext<'_>, signal: &mut StratumSignal) -> Result<(), String> {
        let n_entries = ctx.entries.len();
        let expected = ctx.layer_chunk_dim;
        let n_heads = ctx.num_q_heads;
        let r_dim = ctx.rotary_dim;
        let use_projected = !ctx.query_sig.is_empty()
            && ctx.query_sig.len() == ctx.raw_sig.len()
            && ctx.entry_sigs.len() == n_entries;
        let query_vec = if use_projected {
            ctx.query_sig
        } else {
            ctx.raw_sig
        };

        if n_entries == 0
            || n_heads < 2
            || r_dim == 0
            || expected != n_heads * r_dim
            || query_vec.len() < expected
            || !query_vec.len().is_multiple_of(expected)
        {
            return Ok(());
        }
        let num_steered = query_vec.len() / expected;
        let mut head_cos: Vec<Vec<f32>> = vec![vec![0.0_f32; n_heads]; n_entries];
        for (i, entry) in ctx.entries.iter().enumerate() {
            let entry_vec: &[f32] = if use_projected {
                &ctx.entry_sigs[i]
            } else {
                &entry.residual_sig
            };
            if entry_vec.len() != query_vec.len() {
                continue;
            }
            for (h, slot) in head_cos[i].iter_mut().enumerate() {
                let mut dot = 0.0_f64;
                let mut nq = 0.0_f64;
                let mut ne = 0.0_f64;
                for l in 0..num_steered {
                    let base = l * expected + h * r_dim;
                    for d in 0..r_dim {
                        let qv = f64::from(query_vec[base + d]);
                        let ev = f64::from(entry_vec[base + d]);
                        dot += qv * ev;
                        nq += qv * qv;
                        ne += ev * ev;
                    }
                }
                let denom = (nq * ne).sqrt();
                *slot = if denom > 1e-12 {
                    (dot / denom).clamp(-1.0, 1.0) as f32
                } else {
                    0.0
                };
            }
        }

        let top_k_heads = (n_heads / 6).max(2).min(n_heads);
        for role in [
            AttractorRole::Fact,
            AttractorRole::Persona,
            AttractorRole::Guard,
        ] {
            let idxs: Vec<usize> = ctx
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.role == role)
                .map(|(i, _)| i)
                .collect();
            if idxs.len() <= 1 {
                continue;
            }
            let mut role_head_mean = vec![0.0_f32; n_heads];
            for &i in &idxs {
                for (h, mean_slot) in role_head_mean.iter_mut().enumerate() {
                    *mean_slot += head_cos[i][h] / (idxs.len() as f32);
                }
            }
            for &i in &idxs {
                let mut diffs: Vec<f32> = (0..n_heads)
                    .map(|h| head_cos[i][h] - role_head_mean[h])
                    .collect();
                diffs.sort_by(|a, b| b.total_cmp(a));
                let top_contrast = diffs[..top_k_heads].iter().sum::<f32>() / (top_k_heads as f32);
                let pos = signal.candidate_scores[i].max(0.0);
                signal.candidate_scores[i] += self.head_contrast_gain * pos * top_contrast;
                signal.stigmergic_weights[i] =
                    (signal.stigmergic_weights[i] * (1.0 + top_contrast)).clamp(0.4, 2.0);
            }
        }
        Ok(())
    }

    fn adjust_down(
        &self,
        _ctx: &StratumContext<'_>,
        directive: &mut StratumDirective,
        signal: &mut StratumSignal,
    ) {
        if directive.repulse_guard || directive.suppress_injection {
            for w in &mut signal.stigmergic_weights {
                *w *= 0.5;
            }
        }
    }
}

/// Stratum 1 (`n = 1`): evaluates role-scoped local neighborhood baselines and perspective-dependent
/// `cvoid` depths via an [`AttractorVoidPerspective`] trait object.
pub struct PerspectiveDominoStratum<'a> {
    /// Perspective evaluator measuring `perspective_affinity` and `local_void_depth`.
    pub perspective: &'a dyn AttractorVoidPerspective,
}

impl SocietyStratum for PerspectiveDominoStratum<'_> {
    fn name(&self) -> &'static str {
        "perspective_domino_cvoid"
    }

    fn infer_up(&self, ctx: &StratumContext<'_>, signal: &mut StratumSignal) -> Result<(), String> {
        let n_entries = ctx.entries.len();
        let (query_mean_cosine, query_std_cosine) = if signal.candidate_scores.is_empty() {
            (0.0_f32, 0.05_f32)
        } else {
            let n = signal.candidate_scores.len() as f32;
            let mean = signal.candidate_scores.iter().sum::<f32>() / n;
            let var = signal
                .candidate_scores
                .iter()
                .map(|&s| (s - mean) * (s - mean))
                .sum::<f32>()
                / n;
            (mean, var.sqrt())
        };

        let mut neighbor_stats: Vec<(f32, f32)> = vec![(0.0, 0.05); n_entries];
        let mut role_global_means = [0.0_f32; 3];
        for role in [
            AttractorRole::Fact,
            AttractorRole::Persona,
            AttractorRole::Guard,
        ] {
            let role_indices: Vec<usize> = ctx
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.role == role)
                .map(|(i, _)| i)
                .collect();
            if role_indices.len() <= 1 {
                continue;
            }
            let neighbor_k = (role_indices.len() - 1).clamp(1, 4);
            let mut sum_k_mean = 0.0_f32;
            for &i in &role_indices {
                let mut sims: Vec<f32> = role_indices
                    .iter()
                    .copied()
                    .filter(|&j| j != i)
                    .map(|j| {
                        TextAttractorCorpus::cosine_f32(&ctx.entry_sigs[i], &ctx.entry_sigs[j])
                    })
                    .collect();
                sims.sort_by(|a, b| b.total_cmp(a));
                let top_slice = &sims[..neighbor_k.min(sims.len())];
                let k_len = top_slice.len().max(1) as f32;
                let k_mean = top_slice.iter().sum::<f32>() / k_len;
                let all_mean = sims.iter().sum::<f32>() / (sims.len().max(1) as f32);
                let all_var = sims
                    .iter()
                    .map(|&s| (s - all_mean) * (s - all_mean))
                    .sum::<f32>()
                    / (sims.len().max(1) as f32);
                neighbor_stats[i] = (k_mean, all_var.sqrt());
                sum_k_mean += k_mean;
            }
            let role_slot = match role {
                AttractorRole::Fact => 0,
                AttractorRole::Persona => 1,
                AttractorRole::Guard => 2,
            };
            role_global_means[role_slot] = sum_k_mean / (role_indices.len() as f32);
        }

        for (idx, entry) in ctx.entries.iter().enumerate() {
            let (neighbor_mean_cosine, neighbor_std_cosine) = neighbor_stats[idx];
            let role_slot = match entry.role {
                AttractorRole::Fact => 0,
                AttractorRole::Persona => 1,
                AttractorRole::Guard => 2,
            };
            let reading = self.perspective.measure_perspective(
                &AttractorPerspectiveContext {
                    entry,
                    raw_cosine: signal.candidate_scores[idx],
                    neighbor_mean_cosine,
                    neighbor_std_cosine,
                    global_neighbor_mean: role_global_means[role_slot],
                    query_mean_cosine,
                    query_std_cosine,
                },
                ctx.policy.min_coherence,
            );
            signal.candidate_scores[idx] = reading.perspective_affinity;
            signal.local_void_depths[idx] = reading.local_void_depth;
        }
        Ok(())
    }
}

/// Stratum 2 (`n = 2`): upper Society-of-Mind arbiter combining [`RecallBoundaryGate`] with
/// `ActionHub` [`StigmergicActionDecision`] regime selection and top-down directives.
pub struct ActionHubBoundaryStratum<'a> {
    /// Boundary gate classifying `void_state` and suppression.
    pub gate: &'a dyn RecallBoundaryGate,
}

impl SocietyStratum for ActionHubBoundaryStratum<'_> {
    fn name(&self) -> &'static str {
        "action_hub_boundary"
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

        let repulsed = repulsed_by_guard || frame.is_repulsed();
        let has_active_persona = top1_persona_score >= persona_floor;
        let is_total_cvoid = matches!(decision.void_state, VoidBoundaryState::CvoidEmptySpace)
            && !has_active_persona
            && !repulsed;

        let fact_margin = (top1_fact_score - top2_fact_score).max(0.0);
        let margin_ratio = (fact_margin / ctx.policy.boundary_margin.max(0.01)).clamp(0.0, 3.0);
        let channel_disagreement = signal.telemetry.channel_disagreement;
        let partial_swarm_consensus = signal.swarm_report.resolved_cell_count > 0
            && signal.swarm_report.resolved_cell_count < 6;

        let thresholds = StigmergicThresholds::default();
        let (rho, h_tilde) = if repulsed {
            (frame.effective_energy, 0.75_f32)
        } else if is_total_cvoid {
            (0.0_f32, 0.0_f32)
        } else if signal.local_channel_preferred
            || signal.decomposition_triggered
            || (channel_disagreement && partial_swarm_consensus)
        {
            // Bifurcated or local-sharpened channel triggers PSPM sub-head routing via core ladder.
            (
                thresholds
                    .pspm_route_density_thresh
                    .max(frame.effective_energy.min(1.35)),
                0.65_f32,
            )
        } else if matches!(decision.void_state, VoidBoundaryState::CvoidEmptySpace)
            && has_active_persona
        {
            // Persona-only active grounding outside factual basins uses lightweight ExpressPhaseSteer.
            (thresholds.express_steer_density_thresh.max(0.45), 0.35_f32)
        } else if matches!(decision.void_state, VoidBoundaryState::LiminalBoundary) {
            (thresholds.express_steer_density_thresh.max(0.45), 0.35_f32)
        } else if margin_ratio >= 2.0 {
            let boost = thresholds
                .crystallize_density_thresh
                .max(frame.effective_energy * 1.5);
            (boost, 0.40_f32)
        } else {
            let srl_rho = thresholds
                .srl_inject_density_thresh
                .max(frame.effective_energy.min(2.0));
            (srl_rho, 0.40_f32)
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

        // Execute cross-scale ActionHub orchestration so Gateway synthesis, Stigmergy, and
        // recorded CognitiveTrajectoryLedger close into one unified FastPathLedger ledger.
        let mut boundary_hub = ActionHubEngine::new();
        let mut ledger_slot = Default::default();
        signal
            .telemetry
            .cognitive_trajectory
            .write_to_ledger_slot(&mut ledger_slot);
        let mut boundary_trail = gcso_pointer_trail_t::default();
        boundary_trail
            .copy_from_with_words_fast(&gcso_pointer_trail_t::default(), frame.phase_words_ref());
        let scale_stack = [
            ScaleLayerNode::new(frame, SwarmScaleGranularity::TypeGateway),
            ScaleLayerNode::new(
                signal
                    .telemetry
                    .cognitive_trajectory
                    .synthesize_gateway_frame(),
                SwarmScaleGranularity::MacroCoalition,
            ),
        ];
        if let Ok((proj, mut ledger, _rep)) = boundary_hub.orchestrate_adaptive_depth_n(
            &scale_stack,
            scale_stack.len(),
            gcso_tagged_ptr_t(frame.tagged_ptr),
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
                StigmergicActionDecision::PspmSubHeadRoute => {
                    if signal.decomposition_triggered {
                        CognitiveOpKind::DecomposePeel
                    } else {
                        CognitiveOpKind::RouteSubHead
                    }
                }
                StigmergicActionDecision::FullOrchestrationSrl
                | StigmergicActionDecision::ExpressPhaseSteer => CognitiveOpKind::SynthesizeFuse,
                StigmergicActionDecision::PassiveSidecar
                | StigmergicActionDecision::InertialLayerSkip => CognitiveOpKind::Observe,
            };
            let intensity = (proj.activation.intensity_q7 >> 4).clamp(1, 7);
            ledger.push_step(CognitiveStepOp::pack(
                SwarmScaleGranularity::MacroCoalition,
                final_op,
                intensity,
            ));
            ledger.write_to_ledger_slot(&mut boundary_hub.ledger_slot);
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
            StigmergicActionDecision::PspmSubHeadRoute => 0.85,
            StigmergicActionDecision::FullOrchestrationSrl => 1.0,
            StigmergicActionDecision::CrystallizeMacroBasin => 1.15,
        };
        let min_void = signal
            .local_void_depths
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        directive.min_local_void_depth = if min_void.is_finite() { min_void } else { 1.0 };
    }
}

/// Composable Society-of-Mind pipeline supporting dynamic insertion, removal, and finite
/// bidirectional (`infer_up` then `adjust_down`) arbitration across [`SocietyStratum`] layers.
pub struct StratumPipeline<'a> {
    strata: Vec<Box<dyn SocietyStratum + 'a>>,
}

impl<'a> StratumPipeline<'a> {
    /// Creates an empty [`StratumPipeline`].
    #[must_use]
    pub fn new() -> Self {
        Self { strata: Vec::new() }
    }

    /// Appends a stratum at the top (`n + 1`) of the pipeline.
    pub fn push_stratum(&mut self, stratum: Box<dyn SocietyStratum + 'a>) {
        self.strata.push(stratum);
    }

    /// Inserts a stratum at `index` (clamped to `self.strata.len()`).
    pub fn insert_stratum(&mut self, index: usize, stratum: Box<dyn SocietyStratum + 'a>) {
        let pos = index.min(self.strata.len());
        self.strata.insert(pos, stratum);
    }

    /// Removes and returns the first stratum matching `name`, if present.
    pub fn remove_stratum(&mut self, name: &str) -> Option<Box<dyn SocietyStratum + 'a>> {
        self.strata
            .iter()
            .position(|s| s.name() == name)
            .map(|idx| self.strata.remove(idx))
    }

    /// Returns the ordered list of active stratum names from bottom (`0`) to top.
    #[must_use]
    pub fn stratum_names(&self) -> Vec<&'static str> {
        self.strata.iter().map(|s| s.name()).collect()
    }

    /// Executes one bottom-up pass (`0..len`) followed by one top-down pass (`len..0`).
    ///
    /// # Errors
    /// Returns an error if any stratum's `infer_up` fails.
    pub fn evaluate(
        &self,
        ctx: &StratumContext<'_>,
        mut signal: StratumSignal,
    ) -> Result<(StratumSignal, StratumDirective), String> {
        let best_anchor = |scores: &[f32]| -> (Option<u32>, f32) {
            ctx.entries
                .iter()
                .zip(scores.iter())
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(e, &s)| (Some(e.anchor_id), s))
                .unwrap_or((None, -1.0))
        };
        let (global_top1_anchor, global_top1_score) = best_anchor(&signal.candidate_scores);
        let (local_top1_anchor, local_top1_score) = best_anchor(&signal.local_token_scores);
        signal.telemetry.global_top1_anchor = global_top1_anchor;
        signal.telemetry.global_top1_score = global_top1_score;
        signal.telemetry.local_top1_anchor = local_top1_anchor;
        signal.telemetry.local_top1_score = local_top1_score;
        signal.telemetry.channel_disagreement = global_top1_anchor.is_some()
            && local_top1_anchor.is_some()
            && global_top1_anchor != local_top1_anchor;

        for stratum in &self.strata {
            stratum.infer_up(ctx, &mut signal)?;
        }
        let mut directive = StratumDirective::default();
        for stratum in self.strata.iter().rev() {
            stratum.adjust_down(ctx, &mut directive, &mut signal);
        }

        if signal.telemetry.final_top1_anchor.is_none() {
            let (final_top1_anchor, _) = best_anchor(&signal.candidate_scores);
            let last_layer = ctx
                .entries
                .first()
                .map(|e| e.per_layer_q7.len().saturating_sub(1))
                .unwrap_or(0);
            let frame = signal
                .map_to_gateway_with_persona_floor(
                    ctx.entries,
                    last_layer,
                    ctx.policy.min_coherence * 0.75,
                )
                .frame;
            signal.gateway_frame = frame;
            signal.telemetry.final_top1_anchor = final_top1_anchor;
            signal.telemetry.provenance = frame.to_causal_report();
        }
        Ok((signal, directive))
    }
}

/// Trait abstracting multi-point stigmergic coalition decomposition over token resonance
/// deposits and 8-cell Micro-Swarm phase consensus (`ActionHubEngine`).
pub trait StigmergicCoalitionDecomposer: Send + Sync {
    /// Decomposes a multi-token query's residual resonance field into an ordered chain of
    /// complementary Fact attractors (`t0 -> t1 -> ...`) inside `hub`.
    fn decompose_fact_coalition(
        &self,
        ctx: &StratumContext<'_>,
        signal: &mut StratumSignal,
        hub: &mut ActionHubEngine,
        selected_indices: &[usize],
        trail_states: &[gcso_pointer_trail_t],
        last_layer: usize,
    );
}

/// Default [`StigmergicCoalitionDecomposer`] applying symmetric Partition-of-Unity lateral
/// inhibition across token profiles and iterative residual peeling up to `max_coalition_hops`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualPeelingSwarmDecomposer {
    /// Maximum number of Fact trails linked into a single decomposed pointer chain (`>= 2`).
    pub max_coalition_hops: usize,
}

impl Default for ResidualPeelingSwarmDecomposer {
    fn default() -> Self {
        Self {
            max_coalition_hops: 3,
        }
    }
}

/// Stratum 0.75: multi-token late-interaction resonance stratum arbitrating global pooled
/// resonance against `local_token_scores` via statistical unimodality, 8-cell Micro-Swarm
/// phase consensus (`StigmergicSwarmOps`), and [`StigmergicCoalitionDecomposer`].
#[derive(Debug, Clone, Copy)]
pub struct LocalTokenResonanceStratum {
    /// Gain applied when promoting non-negative local contrast under sharper local unimodality.
    pub token_contrast_gain: f32,
    /// Stigmergic decomposer building multi-hop Fact coalitions on residual token fields.
    pub decomposer: ResidualPeelingSwarmDecomposer,
}

impl Default for LocalTokenResonanceStratum {
    fn default() -> Self {
        Self {
            token_contrast_gain: 0.45,
            decomposer: ResidualPeelingSwarmDecomposer::default(),
        }
    }
}

impl LocalTokenResonanceStratum {
    fn channel_sharpness_on_indices(
        scores: &[f32],
        idxs: &[usize],
        min_std: f32,
    ) -> Option<(usize, f32, f32, f32, f32, f32)> {
        if idxs.is_empty() {
            return None;
        }
        let mut indexed: Vec<(usize, f32)> = idxs.iter().map(|&i| (i, scores[i])).collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        let (top1_idx, top1) = indexed[0];
        let top2 = indexed.get(1).map(|x| x.1).unwrap_or(top1);
        let n = indexed.len() as f32;
        let mean = indexed.iter().map(|&(_, s)| s).sum::<f32>() / n;
        let var = indexed
            .iter()
            .map(|&(_, s)| (s - mean) * (s - mean))
            .sum::<f32>()
            / n;
        let std = var.sqrt().max(min_std);
        let z_peak = (top1 - mean) / std;
        let z_margin = (top1 - top2) / std;
        Some((top1_idx, top1, top2, mean, std, 0.5 * (z_peak + z_margin)))
    }

    fn channel_sharpness(scores: &[f32]) -> Option<(usize, f32, f32)> {
        let all: Vec<usize> = (0..scores.len()).collect();
        Self::channel_sharpness_on_indices(scores, &all, 0.08)
            .map(|(idx, top1, _, _, _, sharp)| (idx, top1, sharp))
    }

    fn resolve_pair_swarm(
        entries: &[TextAttractorEntry],
        g_idx: usize,
        l_idx: usize,
        last_layer: usize,
    ) -> SwarmResolutionReport {
        let g_q7 = entries[g_idx]
            .per_layer_q7
            .get(last_layer)
            .copied()
            .unwrap_or([0_i8; MAX_ATTENTION_HEADS]);
        let l_q7 = entries[l_idx]
            .per_layer_q7
            .get(last_layer)
            .copied()
            .unwrap_or([0_i8; MAX_ATTENTION_HEADS]);
        let g_words = load_q7_words_64(&g_q7);
        let l_words = load_q7_words_64(&l_q7);
        let mut swarm_trail = gcso_pointer_trail_t::default();
        swarm_trail.copy_from_with_words_fast(&gcso_pointer_trail_t::default(), &g_words);
        let (swarm_report, _) = swarm_trail.try_resolve_action_in_swarm_from_words(
            Default::default(),
            &l_words,
            0xFF,
            64,
            4,
            24,
            4,
            0.40,
            0.15,
            None,
            3,
            96,
        );
        swarm_report
    }

    /// Selects up to `ACTION_HUB_INLINE_TRAILS` (16) active attractor candidates covering global,
    /// combined local, and single-token specialist peaks, then deposits uncollapsed token
    /// observations through the 8-cell Micro-Swarm into an [`ActionHubEngine`].
    fn deposit_token_swarm_trails(
        entries: &[TextAttractorEntry],
        signal: &mut StratumSignal,
        last_layer: usize,
    ) -> (ActionHubEngine, Vec<usize>, Vec<gcso_pointer_trail_t>) {
        let n_entries = entries.len();
        let mut hub = ActionHubEngine::new();
        for entry in entries {
            if entry.role == AttractorRole::Guard {
                let g_q7 = entry
                    .per_layer_q7
                    .get(last_layer)
                    .copied()
                    .unwrap_or([0_i8; MAX_ATTENTION_HEADS]);
                let g_words = load_q7_words_64(&g_q7);
                hub.bloom_filter
                    .insert_repulsive_pattern_words_weighted(&g_words, 2);
            }
        }

        let mut ranked: Vec<(usize, f32)> = (0..n_entries)
            .map(|i| {
                let g = signal.candidate_scores.get(i).copied().unwrap_or(0.0);
                let l = signal.local_token_scores.get(i).copied().unwrap_or(0.0);
                let tok_peak = signal
                    .token_deposits
                    .iter()
                    .filter_map(|d| d.entry_excess.get(i).copied())
                    .fold(0.0_f32, f32::max);
                (i, g.max(l).max(tok_peak * 0.8))
            })
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked.truncate(ACTION_HUB_INLINE_TRAILS);

        let selected_indices: Vec<usize> = ranked.iter().map(|&(i, _)| i).collect();
        let mut trail_states: Vec<gcso_pointer_trail_t> = selected_indices
            .iter()
            .map(|&entry_idx| {
                let q7 = entries[entry_idx]
                    .per_layer_q7
                    .get(last_layer)
                    .copied()
                    .unwrap_or([0_i8; MAX_ATTENTION_HEADS]);
                let words = load_q7_words_64(&q7);
                let mut trail = gcso_pointer_trail_t::default();
                trail.copy_from_with_words_fast(&gcso_pointer_trail_t::default(), &words);
                trail.target_anchor_id = entries[entry_idx].anchor_id;
                trail.user_data = entry_idx as u64;
                trail.stigmergic_density = signal
                    .candidate_scores
                    .get(entry_idx)
                    .copied()
                    .unwrap_or(0.0)
                    .max(0.0)
                    * 0.5;
                trail
            })
            .collect();

        if signal.token_deposits.is_empty() {
            return (hub, selected_indices, trail_states);
        }

        let mut bloom_repulsions = vec![0_u32; selected_indices.len()];
        for dep in &signal.token_deposits {
            if dep.disc_weight <= 0.0 || dep.entry_excess.len() != n_entries {
                continue;
            }
            let max_ex = dep.entry_excess.iter().copied().fold(0.0_f32, f32::max);
            if max_ex <= 1e-6 {
                continue;
            }
            for (slot, &entry_idx) in selected_indices.iter().enumerate() {
                let ex = dep.entry_excess[entry_idx];
                if ex <= 0.0 {
                    continue;
                }
                let tok_coh = (ex / max_ex).clamp(0.0, 1.0);
                if tok_coh < 0.45 {
                    continue;
                }
                let (rep, _swarm_w) = trail_states[slot].try_resolve_action_in_swarm_from_words(
                    gcso_tagged_ptr_t::default(),
                    &dep.token_phase_words,
                    0xFF,
                    64,
                    4,
                    24,
                    4,
                    0.45,
                    0.15,
                    Some(&hub.bloom_filter),
                    3,
                    96,
                );
                let consensus = (rep.resolved_cell_count as f32) / (NUM_SWARM_MICRO_CELLS as f32);
                let bloom_repulsed = rep.stigmergic_delta_density < 0.0;
                if bloom_repulsed {
                    bloom_repulsions[slot] += 1;
                }
                let weighted_coh = (tok_coh * (0.5 + 0.5 * consensus)).clamp(0.0, 1.0);
                hub.evaporate_and_deposit_pheromone(
                    &mut trail_states[slot],
                    weighted_coh,
                    bloom_repulsed,
                );
            }
        }

        for (slot, &entry_idx) in selected_indices.iter().enumerate() {
            let density = trail_states[slot].stigmergic_density;
            if density > 0.0 {
                let factor = if bloom_repulsions[slot] > 0 {
                    0.75
                } else {
                    1.0 + 0.25 * density.min(1.5)
                };
                signal.stigmergic_weights[entry_idx] =
                    (signal.stigmergic_weights[entry_idx] * factor).clamp(0.30, 2.5);
            }
        }

        (hub, selected_indices, trail_states)
    }
}

impl StigmergicCoalitionDecomposer for ResidualPeelingSwarmDecomposer {
    /// Evaluates multi-hop residual-peeling set cover over Fact trails when multiple specialized
    /// attractors jointly explain distinct query tokens that single global pooling collapses.
    fn decompose_fact_coalition(
        &self,
        ctx: &StratumContext<'_>,
        signal: &mut StratumSignal,
        hub: &mut ActionHubEngine,
        selected_indices: &[usize],
        trail_states: &[gcso_pointer_trail_t],
        last_layer: usize,
    ) {
        let n_entries = ctx.entries.len();
        if ctx.policy.top_k_facts < 2
            || self.max_coalition_hops < 2
            || signal.token_deposits.is_empty()
            || signal.q_to_d_scores.len() != n_entries
            || signal.d_to_q_scores.len() != n_entries
        {
            return;
        }

        let max_guard_raw = ctx
            .entries
            .iter()
            .zip(signal.candidate_scores.iter())
            .filter(|(e, _)| e.role == AttractorRole::Guard)
            .map(|(_, &s)| s)
            .fold(f32::NEG_INFINITY, f32::max);
        let max_non_guard_raw = ctx
            .entries
            .iter()
            .zip(signal.candidate_scores.iter())
            .filter(|(e, _)| e.role != AttractorRole::Guard)
            .map(|(_, &s)| s)
            .fold(f32::NEG_INFINITY, f32::max);
        if max_guard_raw.is_finite()
            && max_non_guard_raw.is_finite()
            && (max_guard_raw - max_non_guard_raw) > ctx.policy.guard_suppress_margin
        {
            return;
        }

        let fact_idxs: Vec<usize> = ctx
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.role == AttractorRole::Fact)
            .map(|(i, _)| i)
            .collect();
        if fact_idxs.len() < 2 {
            return;
        }

        let total_disc_w: f32 = signal
            .token_deposits
            .iter()
            .map(|d| d.disc_weight.max(0.0))
            .sum();
        if total_disc_w <= 1e-8 {
            return;
        }

        // Compute Fact-exclusivity and cross-token redundancy weights per token deposit so
        // repeated subwords of a shared corpus qualifier cannot drown out distinct single-Fact
        // specialist concepts during compound decomposition.
        let n_dep = signal.token_deposits.len();
        let mut decomp_weights = vec![0.0_f32; n_dep];
        let mut fact_profiles: Vec<Vec<f32>> = Vec::with_capacity(n_dep);
        for (idx, d) in signal.token_deposits.iter().enumerate() {
            let mut pos_vals: Vec<f32> = fact_idxs
                .iter()
                .map(|&fi| d.entry_excess[fi].max(0.0))
                .collect();
            let norm: f32 = pos_vals.iter().map(|&v| v * v).sum::<f32>().sqrt();
            let mut sorted_vals = pos_vals.clone();
            sorted_vals.sort_by(|a, b| b.total_cmp(a));
            let f_top1 = sorted_vals.first().copied().unwrap_or(0.0);
            let f_top2 = sorted_vals.get(1).copied().unwrap_or(0.0);
            let exclusivity = if f_top1 > 1e-6 {
                ((f_top1 - f_top2) / f_top1).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut redundancy_damp = 1.0_f32;
            if norm > 1e-6 {
                for v in &mut pos_vals {
                    *v /= norm;
                }
                let mut max_prev_cos = 0.0_f32;
                for prev in &fact_profiles {
                    let c: f32 = pos_vals.iter().zip(prev.iter()).map(|(&a, &b)| a * b).sum();
                    if c > max_prev_cos {
                        max_prev_cos = c;
                    }
                }
                if max_prev_cos > 0.75 && exclusivity < 0.50 {
                    redundancy_damp = (1.0 - max_prev_cos * max_prev_cos).max(0.50);
                }
            }
            fact_profiles.push(pos_vals);
            decomp_weights[idx] =
                d.disc_weight.max(0.0) * (0.40 + 0.60 * exclusivity) * redundancy_damp;
        }
        let total_decomp_w: f32 = decomp_weights.iter().sum();
        if total_decomp_w <= 1e-8 {
            return;
        }

        let (g_fact_top1, g_fact_top1_score) = fact_idxs
            .iter()
            .map(|&i| (i, signal.candidate_scores[i]))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((fact_idxs[0], -1.0));
        let excl_q_to_d_for = |fact_idx: usize| -> f32 {
            signal
                .token_deposits
                .iter()
                .zip(decomp_weights.iter())
                .map(|(d, &w)| w * d.entry_excess[fact_idx])
                .sum::<f32>()
                / total_decomp_w
        };
        let excl_single_scores: Vec<(usize, f32)> =
            fact_idxs.iter().map(|&i| (i, excl_q_to_d_for(i))).collect();
        let g_top1_excl = excl_q_to_d_for(g_fact_top1);

        let mut best_pair: Option<(usize, usize, f32, f32, f32)> = None;
        for a in 0..fact_idxs.len() {
            let (i, gain_i) = excl_single_scores[a];
            let gain_i_pos = gain_i.max(0.0);
            if gain_i_pos <= 0.004 {
                continue;
            }
            for &(j, gain_j) in &excl_single_scores[(a + 1)..] {
                let gain_j_pos = gain_j.max(0.0);
                if gain_j_pos <= 0.004 {
                    continue;
                }
                let joint_excl = signal
                    .token_deposits
                    .iter()
                    .zip(decomp_weights.iter())
                    .map(|(d, &w)| w * d.entry_excess[i].max(d.entry_excess[j]))
                    .sum::<f32>()
                    / total_decomp_w;
                let second_contrib = (joint_excl - gain_i_pos)
                    .min(joint_excl - gain_j_pos)
                    .max(0.0);
                if second_contrib <= 0.012 {
                    continue;
                }
                let pair_d_to_q = 0.5 * (signal.d_to_q_scores[i] + signal.d_to_q_scores[j]);
                let utility = 0.70 * joint_excl + 0.25 * second_contrib + 0.20 * pair_d_to_q;
                if best_pair.is_none_or(|(_, _, _, _, best_u)| utility > best_u) {
                    best_pair = Some((i, j, joint_excl, second_contrib, utility));
                }
            }
        }

        let Some((cand_a, cand_b, joint_excl, second_contrib, _utility)) = best_pair else {
            return;
        };

        // Ensure no single Fact `k` subsumes `>= 78%` of `{cand_a, cand_b}`'s covered token mass.
        let max_subsumption = fact_idxs
            .iter()
            .map(|&k| {
                signal
                    .token_deposits
                    .iter()
                    .zip(decomp_weights.iter())
                    .map(|(d, &w)| {
                        let ex_k = d.entry_excess[k];
                        let ex_pair = d.entry_excess[cand_a].max(d.entry_excess[cand_b]);
                        w * ex_k.min(ex_pair)
                    })
                    .sum::<f32>()
                    / total_decomp_w
            })
            .fold(0.0_f32, f32::max);

        let raw_joint_q_to_d = signal
            .token_deposits
            .iter()
            .map(|d| d.disc_weight.max(0.0) * d.entry_excess[cand_a].max(d.entry_excess[cand_b]))
            .sum::<f32>()
            / total_disc_w;
        let basin_supported =
            g_fact_top1_score >= ctx.policy.min_coherence * 0.55 || raw_joint_q_to_d >= 0.065;
        let strictly_bimodal = joint_excl >= 0.040
            && second_contrib >= 0.015
            && second_contrib >= 0.25 * joint_excl
            && (joint_excl - max_subsumption) >= 0.22 * joint_excl;
        let global_chamfer_top1 =
            0.70 * g_top1_excl.max(0.0) + 0.20 * signal.d_to_q_scores[g_fact_top1];
        let pair_chamfer = 0.70 * joint_excl
            + 0.20 * 0.5 * (signal.d_to_q_scores[cand_a] + signal.d_to_q_scores[cand_b]);
        let outperforms_global_hub = (cand_a != g_fact_top1
            && cand_b != g_fact_top1
            && pair_chamfer > global_chamfer_top1 + 0.020)
            || signal.candidate_scores[cand_a] < ctx.policy.min_coherence * 0.75
            || signal.candidate_scores[cand_b] < ctx.policy.min_coherence * 0.75;

        if !(basin_supported && strictly_bimodal && outperforms_global_hub) {
            return;
        }

        let solo_rank_score =
            |idx: usize| -> f32 { excl_q_to_d_for(idx) + 0.35 * signal.d_to_q_scores[idx] };
        let (first_idx, second_idx) = if solo_rank_score(cand_a) >= solo_rank_score(cand_b) {
            (cand_a, cand_b)
        } else {
            (cand_b, cand_a)
        };

        let mut coalition = vec![first_idx, second_idx];
        let hop_cap = self
            .max_coalition_hops
            .min(ACTION_HUB_INLINE_TRAILS)
            .min(fact_idxs.len());
        if hop_cap >= 3 {
            let mut best_third: Option<(usize, f32)> = None;
            for &k in &fact_idxs {
                if k == first_idx || k == second_idx {
                    continue;
                }
                if signal.q_to_d_scores[k] <= 0.0 && signal.local_token_scores[k] <= 0.0 {
                    continue;
                }
                let mut res_gain = 0.0_f32;
                let mut has_specialist_peak = false;
                for (d, &w) in signal.token_deposits.iter().zip(decomp_weights.iter()) {
                    let pair_ex = d.entry_excess[first_idx]
                        .max(d.entry_excess[second_idx])
                        .max(0.0);
                    let ex_k = d.entry_excess[k];
                    if ex_k <= pair_ex {
                        continue;
                    }
                    let other_rem_max = fact_idxs
                        .iter()
                        .copied()
                        .filter(|&m| m != first_idx && m != second_idx && m != k)
                        .map(|m| d.entry_excess[m].max(0.0))
                        .fold(0.0_f32, f32::max);
                    let rem_excl = if ex_k > 1e-6 {
                        ((ex_k - other_rem_max) / ex_k).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let delta = ex_k - pair_ex;
                    let eff_w = w + 0.0015 * rem_excl;
                    res_gain += eff_w * delta;
                    if ex_k >= 0.08 && delta >= 0.025 && rem_excl >= 0.40 {
                        has_specialist_peak = true;
                    }
                }
                res_gain /= total_decomp_w;
                if res_gain >= 0.004 && has_specialist_peak {
                    let score = res_gain
                        + 0.20 * signal.q_to_d_scores[k].max(0.0)
                        + 0.15 * signal.d_to_q_scores[k].max(0.0);
                    if best_third.is_none_or(|(_, best_s)| score > best_s) {
                        best_third = Some((k, score));
                    }
                }
            }
            if let Some((third_idx, _)) = best_third {
                coalition.push(third_idx);
            }
        }

        // Wire the decomposed coalition onto `ActionHubEngine` inline trails as a closed pointer
        // ring and notify the `StigmergicActionReceptor`.
        let make_trail_for = |entry_idx: usize| -> gcso_pointer_trail_t {
            if let Some(pos) = selected_indices.iter().position(|&idx| idx == entry_idx) {
                trail_states[pos]
            } else {
                let q7 = ctx.entries[entry_idx]
                    .per_layer_q7
                    .get(last_layer)
                    .copied()
                    .unwrap_or([0_i8; MAX_ATTENTION_HEADS]);
                let words = load_q7_words_64(&q7);
                let mut t = gcso_pointer_trail_t::default();
                t.copy_from_with_words_fast(&gcso_pointer_trail_t::default(), &words);
                t.target_anchor_id = ctx.entries[entry_idx].anchor_id;
                t.user_data = entry_idx as u64;
                t
            }
        };
        let chain_len = coalition.len();
        for (m, &entry_idx) in coalition.iter().enumerate() {
            let mut t = make_trail_for(entry_idx);
            t.linked_trail_id = ((m + 1) % chain_len) as u32;
            let min_rho = (0.65 - 0.05 * (m as f32)).max(0.50);
            t.stigmergic_density = t.stigmergic_density.max(min_rho);
            hub.trails[m] = t;
        }
        hub.active_count = hub.active_count.max(chain_len as u32);

        let mut chain_trail = gcso_pointer_trail_t::default();
        chain_trail.copy_from_fast(&hub.trails[0]);
        let mut receptor = ActionHubInterpretiveReceptor::default();
        if let Ok((chain_report, _hops, _term_idx, _term_anchor, _mean_coh)) = hub
            .traverse_pointer_chain_and_orchestrate(
                0,
                chain_len,
                0.58,
                &mut chain_trail,
                None,
                None,
                None,
                None,
            )
        {
            let _ = receptor.on_stigmergic_action_emerged(
                &signal.gateway_frame,
                &chain_report,
                &chain_trail,
            );
        }

        let first_score = (g_fact_top1_score + 0.035).max(ctx.policy.min_coherence * 1.15);
        signal.linked_fact_indices.clear();
        for (m, &entry_idx) in coalition.iter().enumerate() {
            let floor_score = (first_score - 0.015 * (m as f32))
                .max(ctx.policy.min_coherence * if m == 0 { 1.15 } else { 0.95 });
            signal.candidate_scores[entry_idx] =
                signal.candidate_scores[entry_idx].max(floor_score);
            let boost = (1.35 - 0.05 * (m as f32)).max(1.15);
            signal.stigmergic_weights[entry_idx] =
                (signal.stigmergic_weights[entry_idx] * boost).clamp(0.4, 2.5);
            signal
                .linked_fact_indices
                .push((entry_idx, signal.candidate_scores[entry_idx]));
        }
        signal.decomposition_triggered = true;
        signal.local_channel_preferred = true;
        signal.swarm_report = LocalTokenResonanceStratum::resolve_pair_swarm(
            ctx.entries,
            coalition[0],
            coalition[1],
            last_layer,
        );
    }
}

impl SocietyStratum for LocalTokenResonanceStratum {
    fn name(&self) -> &'static str {
        "local_token_resonance"
    }

    fn infer_up(&self, ctx: &StratumContext<'_>, signal: &mut StratumSignal) -> Result<(), String> {
        let n_entries = ctx.entries.len();
        if n_entries == 0 || signal.local_token_scores.len() != n_entries {
            return Ok(());
        }

        let Some((g_idx, _g_score, g_sharp)) = Self::channel_sharpness(&signal.candidate_scores)
        else {
            return Ok(());
        };
        let Some((l_idx, l_score, l_sharp)) = Self::channel_sharpness(&signal.local_token_scores)
        else {
            return Ok(());
        };

        let last_layer = ctx
            .entries
            .first()
            .map(|e| e.per_layer_q7.len().saturating_sub(1))
            .unwrap_or(0);
        // Deposit uncollapsed per-token observations onto 8-cell Micro-Swarm trails in ActionHub.
        let (mut hub, selected_indices, trail_states) =
            Self::deposit_token_swarm_trails(ctx.entries, signal, last_layer);

        let mut ledger = CognitiveTrajectoryLedger::from_ledger_slot(&hub.ledger_slot);
        if !signal.token_deposits.is_empty() {
            ledger.push_step(CognitiveStepOp::pack(
                SwarmScaleGranularity::PointerBit,
                CognitiveOpKind::Observe,
                (signal.token_deposits.len().min(7)) as u8,
            ));
        }

        // Run core 8-cell Micro-Swarm consensus between the global and local winners' 64-head Q7 profiles.
        signal.swarm_report = Self::resolve_pair_swarm(ctx.entries, g_idx, l_idx, last_layer);
        let g_words = load_q7_words_64(
            &ctx.entries[g_idx]
                .per_layer_q7
                .get(last_layer)
                .copied()
                .unwrap_or([0_i8; MAX_ATTENTION_HEADS]),
        );
        let l_words = load_q7_words_64(
            &ctx.entries[l_idx]
                .per_layer_q7
                .get(last_layer)
                .copied()
                .unwrap_or([0_i8; MAX_ATTENTION_HEADS]),
        );
        let cell_activation = ledger.evaluate_circuit_activation(
            &g_words,
            &l_words,
            SwarmScaleGranularity::CellTrail,
        );
        ledger.push_step(CognitiveStepOp::pack(
            SwarmScaleGranularity::CellTrail,
            if g_idx != l_idx {
                CognitiveOpKind::BridgeShift
            } else {
                CognitiveOpKind::RouteSubHead
            },
            signal.swarm_report.resolved_cell_count.min(7),
        ));
        ledger.write_to_ledger_slot(&mut hub.ledger_slot);
        signal.telemetry.layer_activation = cell_activation;

        let global_local_sharper = g_idx != l_idx && l_score > 0.0 && l_sharp > g_sharp * 1.15;
        signal.local_channel_preferred = global_local_sharper;

        if g_idx == l_idx && l_score > 0.0 {
            signal.stigmergic_weights[g_idx] = (signal.stigmergic_weights[g_idx]
                * (1.0 + 0.35 * l_score.min(1.0)))
            .clamp(0.35, 2.5);
        }

        // First test whether Fact tokens form a multi-basin compound coalition better solved via
        // ActionHub linked trails than unimodal winner-take-all sharpening.
        self.decomposer.decompose_fact_coalition(
            ctx,
            signal,
            &mut hub,
            &selected_indices,
            &trail_states,
            last_layer,
        );
        if signal.decomposition_triggered {
            ledger.push_step(CognitiveStepOp::pack(
                SwarmScaleGranularity::MacroCoalition,
                CognitiveOpKind::DecomposePeel,
                (signal.linked_fact_indices.len().min(7)) as u8,
            ));
            ledger.write_to_ledger_slot(&mut hub.ledger_slot);
        }
        signal.telemetry.cognitive_trajectory = ledger;

        // Arbitrate per causal role subspace (Fact / Persona / Guard) so a strong Persona style
        // hub cannot mask Fact-level disagreement between global pooling and token-level MaxSim.
        for role in [
            AttractorRole::Fact,
            AttractorRole::Persona,
            AttractorRole::Guard,
        ] {
            if role == AttractorRole::Fact && signal.decomposition_triggered {
                continue;
            }
            let idxs: Vec<usize> = ctx
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.role == role)
                .map(|(i, _)| i)
                .collect();
            if idxs.len() <= 1 {
                continue;
            }
            let Some((g_r_idx, _g_r_top1, _g_r_top2, g_r_mean, g_r_std, g_r_sharp)) =
                Self::channel_sharpness_on_indices(&signal.candidate_scores, &idxs, 0.05)
            else {
                continue;
            };
            let Some((l_r_idx, l_r_top1, l_r_top2, l_r_mean, l_r_std, l_r_sharp)) =
                Self::channel_sharpness_on_indices(&signal.local_token_scores, &idxs, 0.035)
            else {
                continue;
            };

            if g_r_idx == l_r_idx {
                if l_r_top1 > 0.0 {
                    signal.stigmergic_weights[g_r_idx] = (signal.stigmergic_weights[g_r_idx]
                        * (1.0 + 0.25 * l_r_top1.min(1.0)))
                    .clamp(0.35, 2.5);
                }
                continue;
            }

            let l_r_margin = (l_r_top1 - l_r_top2).max(0.0);
            let role_local_sharper = (global_local_sharper && l_r_top1 > 0.04)
                || (l_r_top1 > 0.04 && l_r_margin >= 0.025 && l_r_sharp > g_r_sharp * 1.15);
            if !role_local_sharper {
                continue;
            }

            if role == AttractorRole::Fact {
                signal.local_channel_preferred = true;
                signal.swarm_report =
                    Self::resolve_pair_swarm(ctx.entries, g_r_idx, l_r_idx, last_layer);
            }

            let alpha = (self.token_contrast_gain * 1.6).clamp(0.0, 0.80);
            let scale = g_r_std.max(0.10);
            for &i in &idxs {
                let z_loc = ((signal.local_token_scores[i] - l_r_mean) / l_r_std).clamp(-2.0, 3.0);
                let matched_loc = g_r_mean + z_loc * scale;
                signal.candidate_scores[i] =
                    (1.0 - alpha) * signal.candidate_scores[i] + alpha * matched_loc;
                let pos_contrast = (signal.local_token_scores[i] - l_r_mean).max(0.0);
                if pos_contrast > 0.0 {
                    signal.stigmergic_weights[i] =
                        (signal.stigmergic_weights[i] * (1.0 + pos_contrast)).clamp(0.35, 2.5);
                }
            }
        }
        Ok(())
    }

    fn adjust_down(
        &self,
        _ctx: &StratumContext<'_>,
        directive: &mut StratumDirective,
        signal: &mut StratumSignal,
    ) {
        if directive.repulse_guard || directive.suppress_injection {
            for w in &mut signal.stigmergic_weights {
                *w *= 0.5;
            }
        }
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
            let obs = (self.boundary_metrics.guard_conflict_margin + 0.5).max(0.0);
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

impl Default for StratumPipeline<'static> {
    fn default() -> Self {
        let mut p = Self::new();
        p.push_stratum(Box::new(LocalFeatureWorkerStratum::default()));
        p.push_stratum(Box::new(HeadAngleWorkerStratum::default()));
        p.push_stratum(Box::new(LocalTokenResonanceStratum::default()));
        p.push_stratum(Box::new(PerspectiveDominoStratum {
            perspective: &DEFAULT_PERSPECTIVE,
        }));
        p.push_stratum(Box::new(ActionHubBoundaryStratum {
            gate: &DEFAULT_GATE,
        }));
        p
    }
}

static DEFAULT_PERSPECTIVE: AsymmetricDominoVoidPerspective = AsymmetricDominoVoidPerspective {
    neighbor_k: 4,
    hubness_damping: 0.50,
    perspective_z_gain: 0.025,
};
static DEFAULT_GATE: CoherenceGuardGate = CoherenceGuardGate;
