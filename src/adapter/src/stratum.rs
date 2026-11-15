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
use liminika_core::action_hub::StigmergicActionDecision;
use liminika_core::traits::gateway::{
    CausalProvenanceReport, GatewayFrame, GuardPhaseAnchor, ManifoldGatewaySignal, ObjectiveFact,
    SubjectivePersona,
};
use liminika_core::traits::primitives::MAX_ATTENTION_HEADS;

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
        let action_decision = if guard_conflict_margin > ctx.policy.guard_suppress_margin {
            StigmergicActionDecision::RepulsionEscape
        } else {
            match decision.void_state {
                VoidBoundaryState::CvoidEmptySpace => StigmergicActionDecision::InertialLayerSkip,
                VoidBoundaryState::LiminalBoundary => StigmergicActionDecision::ExpressPhaseSteer,
                VoidBoundaryState::InsideBasin => {
                    if (top1_fact_score - top2_fact_score) >= 2.0 * ctx.policy.boundary_margin {
                        StigmergicActionDecision::CrystallizeMacroBasin
                    } else {
                        StigmergicActionDecision::FullOrchestrationSrl
                    }
                }
            }
        };

        signal.boundary_metrics = metrics;
        signal.gate_decision = decision;
        signal.action_decision = action_decision;
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

        for stratum in &self.strata {
            stratum.infer_up(ctx, &mut signal)?;
        }
        let mut directive = StratumDirective::default();
        for stratum in self.strata.iter().rev() {
            stratum.adjust_down(ctx, &mut directive, &mut signal);
        }

        let (final_top1_anchor, _) = best_anchor(&signal.candidate_scores);
        let last_layer = ctx
            .entries
            .first()
            .map(|e| e.per_layer_q7.len().saturating_sub(1))
            .unwrap_or(0);
        let provenance = signal
            .map_to_gateway(ctx.entries, last_layer)
            .frame
            .to_causal_report();
        signal.telemetry = StratumProbeTelemetry {
            global_top1_anchor,
            global_top1_score,
            local_top1_anchor,
            local_top1_score,
            final_top1_anchor,
            channel_disagreement: global_top1_anchor.is_some()
                && local_top1_anchor.is_some()
                && global_top1_anchor != local_top1_anchor,
            provenance,
        };
        Ok((signal, directive))
    }
}

/// Stratum 0.75: multi-token late-interaction resonance stratum interpreting the independent
/// `local_token_scores` channel (without coupling the `cvoid` layers to tokenizers or model pooling).
#[derive(Debug, Clone, Copy)]
pub struct LocalTokenResonanceStratum {
    /// Gain applied to role-centered MaxSim contrast (`local_score - role_mean`).
    pub token_contrast_gain: f32,
}

impl Default for LocalTokenResonanceStratum {
    fn default() -> Self {
        Self {
            token_contrast_gain: 0.45,
        }
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
            let role_mean = idxs
                .iter()
                .map(|&i| signal.local_token_scores[i])
                .sum::<f32>()
                / (idxs.len() as f32);
            for &i in &idxs {
                let local_score = signal.local_token_scores[i];
                let contrast = local_score - role_mean;
                let support = local_score.max(signal.candidate_scores[i]).max(0.0);
                if support > 0.0 {
                    signal.candidate_scores[i] += self.token_contrast_gain * contrast;
                    signal.stigmergic_weights[i] =
                        (signal.stigmergic_weights[i] * (1.0 + contrast)).clamp(0.35, 2.0);
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

        let fact = if !self.gate_decision.suppress_fact_injection {
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

        let persona = best_persona.and_then(|(idx, score)| {
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
        });

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

        let frame = if matches!(
            self.action_decision,
            StigmergicActionDecision::RepulsionEscape
        ) {
            guard
                .map(|g| g.synthesize_gateway_frame())
                .unwrap_or_default()
        } else {
            match (fact, persona, guard) {
                (Some(f), Some(p), _) => (f, p).synthesize_gateway_frame(),
                (Some(f), None, _) => f.synthesize_gateway_frame(),
                (None, Some(p), _) => p.synthesize_gateway_frame(),
                (None, None, Some(g)) => g.synthesize_gateway_frame(),
                (None, None, None) => GatewayFrame::default(),
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
