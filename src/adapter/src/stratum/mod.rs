// name: src/adapter/src/stratum/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hierarchical recall strata (`n - 1 <-> n <-> n + 1`) built over [`liminika_core`]
//! `StigmergicSubstrate` and role-scoped `Liminal` boundary geometry.
//!
//! Each [`RecallStratum`] operates only on signals exposed by adjacent layers, allowing dynamic
//! insertion, removal, or replacement via [`StratumPipeline`] without coupling lower feature
//! workers directly to global dialogue policy.

mod boundary;
mod coalition;
mod workers;

pub use boundary::{StratumGatewayMapping, SubstrateBoundaryStratum};
pub use coalition::{
    CoalitionThresholds, LocalTokenResonanceStratum, ResidualPeelingStigmergicDecomposer,
    StigmergicCoalitionDecomposer,
};
pub use workers::{HeadAngleWorkerStratum, LocalFeatureWorkerStratum, PerspectiveHubnessStratum};

use crate::{
    AsymmetricHubnessPerspective, CoherenceGuardGate, NeighborhoodStatsProvider,
    RecallBoundaryMetrics, RecallGateDecision, RecallPolicy, TextAttractorEntry,
};
use liminika_core::abi::lmnk_pointer_trail_t;
use liminika_core::liminal::{LmnkResult, NUM_STIGMERGIC_GRANULES};
use liminika_core::runtime::{CausalProvenanceReport, GatewayFrame, StigmergicReceptor};
use liminika_core::stigmergy::{
    CognitiveCircuitActivation, CognitiveTrajectoryLedger, StigmergicActionDecision,
    StigmergicReconstitutionReport, SubstrateStepReport,
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
    pub token_phase_words: [u64; NUM_STIGMERGIC_GRANULES],
}

/// Uncollapsed local multi-token channel observation passed into [`crate::TextAttractorCorpus`].
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
    /// SIF-weighted fraction (`0.0..=1.0`) of salient query tokens not explained by corpus Facts.
    pub unexplained_token_ratio: f32,
    /// Original query token sequence length (`S`, `0` when single-vector or unspecified).
    pub query_seq_len: usize,
}

/// [`StigmergicReceptor`] tracking multi-trail pointer emergence when
/// [`liminika_core::stigmergy::StigmergicSubstrate`] links disjoint factual sub-basins across
/// residual unexplained query tokens.
#[derive(Debug, Clone, Default)]
pub struct TrailEmergenceReceptor {
    /// Ordered sequence of dominant anchor IDs observed during trail emergence.
    pub emerged_anchor_chain: Vec<u32>,
    /// Last core [`SubstrateStepReport`] emitted upon stigmergic emergence.
    pub last_report: Option<SubstrateStepReport>,
}

impl StigmergicReceptor for TrailEmergenceReceptor {
    fn on_stigmergic_emergence(
        &mut self,
        _frame: &GatewayFrame,
        report: &SubstrateStepReport,
        trail: &lmnk_pointer_trail_t,
    ) -> LmnkResult<()> {
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
/// [`CausalProvenanceReport`] so `StigmergicSubstrate` and evaluation harnesses observe channel agreement.
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
    /// Recorded 8-step multi-scale cognitive trajectory from substrate FastPathLedger mini ledger.
    pub cognitive_trajectory: CognitiveTrajectoryLedger,
    /// Latest stigmergic cognitive circuit gating activation computed from trajectory history.
    pub layer_activation: CognitiveCircuitActivation,
    /// Orthogonal residual norm ratio `||P_perp(q - mu)|| / ||q - mu||` (`0.0..=1.0`) after
    /// common-mode projection.
    pub query_ortho_ratio: f32,
    /// SIF-weighted fraction (`0.0..=1.0`) of salient query tokens unexplained by corpus Facts.
    pub unexplained_token_ratio: f32,
    /// Effective Layer 1 hard-block Guard conflict margin resolved for this turn (`0.0` = default `0.05`).
    pub guard_hard_margin: f32,
    /// True when Layer 2 soft-caution Guard mode (`0 < margin <= hard_margin`) activated on this turn.
    pub is_soft_guard_caution: bool,
}

/// Bottom-up and top-down signal flowing across a hierarchical [`StratumPipeline`].
///
/// Lower strata (`n - 1`) populate local features and stigmergic pointer weights without needing
/// global semantic awareness, while upper strata (`n + 1`) arbitrate `Liminal` boundaries, Guard
/// conflicts, and substrate regimes.
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
    /// Ordered multi-trail Fact decomposition `(entry_idx, score)` selected by
    /// [`liminika_core::stigmergy::StigmergicSubstrate`].
    pub linked_fact_indices: Vec<(usize, f32)>,
    /// True when [`liminika_core::stigmergy::StigmergicSubstrate`] interpretive decomposition
    /// linked disjoint Fact sub-basins.
    pub decomposition_triggered: bool,
    /// Per-attractor local `Liminal` gap depth (`0.0` = deep inside basin).
    pub local_lacuna_depths: Vec<f32>,
    /// Per-attractor stigmergic pointer trace / pheromone strength (`>= 0.0`).
    pub stigmergic_weights: Vec<f32>,
    /// Aggregated boundary metrics observed at the upper boundary stratum.
    pub boundary_metrics: RecallBoundaryMetrics,
    /// Boundary gate decision (`InsideBasin`, `LiminalBoundary`, or `LiminalExterior`).
    pub gate_decision: RecallGateDecision,
    /// Substrate regime chosen for top-down orchestration.
    pub action_decision: StigmergicActionDecision,
    /// Synthesized 128-byte core [`GatewayFrame`] serving as the universal arbitration gate.
    pub gateway_frame: GatewayFrame,
    /// 8-cell Micro-Swarm consensus report between global and local phase channels.
    pub swarm_report: StigmergicReconstitutionReport,
    /// True when the local multi-token channel exhibits sharper unimodal support than global pooling.
    pub local_channel_preferred: bool,
    /// Cross-channel probe telemetry and core [`CausalProvenanceReport`].
    pub telemetry: StratumProbeTelemetry,
}

/// Top-down directive sent from stratum `n + 1` down to `n` and `n - 1` after upward inference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StratumDirective {
    /// Whether injection should be suppressed (`LiminalExterior` or Guard repulsion).
    pub suppress_injection: bool,
    /// Whether active Guard repulsion (`RepulsionEscape`) was triggered.
    pub repulse_guard: bool,
    /// Top-down gain multiplier (`0.0` when skipped/repulsed, `< 1.0` on liminal boundary, `>= 1.0` inside basin).
    pub top_down_gain: f32,
    /// Minimum local `Liminal` depth observed across the stack.
    pub min_local_lacuna_depth: f32,
}

impl Default for StratumDirective {
    fn default() -> Self {
        Self {
            suppress_injection: false,
            repulse_guard: false,
            top_down_gain: 1.0,
            min_local_lacuna_depth: 1.0,
        }
    }
}

/// Read-only shared context passed to each [`RecallStratum`] during upward and downward passes.
pub struct StratumContext<'a> {
    /// Corpus entries aligned with `signal.candidate_scores`.
    pub entries: &'a [TextAttractorEntry],
    /// Unprojected steered-layer residual signature for the query.
    pub raw_sig: &'a [f32],
    /// Common-mode-projected normalized query signature.
    pub query_sig: &'a [f32],
    /// Flat row-major common-mode-projected normalized corpus signatures (`entries.len() * projected_sig_dim`).
    pub projected_sigs_flat: &'a [f32],
    /// Fixed dimension of each projected corpus signature row (`0` when unavailable).
    pub projected_sig_dim: usize,
    /// Precomputed role-scoped neighborhood statistics and role index partitions.
    pub neighborhood_stats: &'a dyn NeighborhoodStatsProvider,
    /// Rotary dimension per steered layer chunk (`q_rot_len`).
    pub layer_chunk_dim: usize,
    /// Number of query attention heads (`H_Q`) inside each steered layer chunk.
    pub num_q_heads: usize,
    /// Active rotated dimensions per attention head (`rotary_dim`).
    pub rotary_dim: usize,
    /// Query token sequence length (`S`, `0` when unspecified/single-vector) for finite-sample
    /// shrinkage of short-query viewing-angle contrast boosts.
    pub query_seq_len: usize,
    /// Orthogonal residual norm ratio `||P_perp(q - mu)|| / ||q - mu||` (`0.0..=1.0`).
    pub query_ortho_ratio: f32,
    /// Active recall policy thresholds.
    pub policy: &'a RecallPolicy,
    /// Calibrated coalition-peeling and relative-scale thresholds (defaults to 1D fallback when uncalibrated).
    pub coalition_thresholds: CoalitionThresholds,
}

impl<'a> StratumContext<'a> {
    /// Returns the common-mode-projected signature slice for `entry_idx` from the flat buffer.
    #[inline]
    #[must_use]
    pub fn projected_entry_sig(&self, entry_idx: usize) -> Option<&'a [f32]> {
        if self.projected_sig_dim == 0 {
            return None;
        }
        let start = entry_idx.checked_mul(self.projected_sig_dim)?;
        let end = start.checked_add(self.projected_sig_dim)?;
        self.projected_sigs_flat.get(start..end)
    }
}

/// Pluggable hierarchical layer in a multi-stratum recall/steering stack (`n - 1 <-> n <-> n + 1`).
///
/// Strata can be dynamically added, inserted, or removed via [`StratumPipeline`] without altering
/// neighboring layers.
pub trait RecallStratum {
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

/// Composable hierarchical recall pipeline supporting dynamic insertion, removal, and finite
/// bidirectional (`infer_up` then `adjust_down`) arbitration across [`RecallStratum`] layers.
pub struct StratumPipeline<'a> {
    strata: Vec<Box<dyn RecallStratum + 'a>>,
}

impl<'a> StratumPipeline<'a> {
    /// Creates an empty [`StratumPipeline`].
    #[must_use]
    pub fn new() -> Self {
        Self { strata: Vec::new() }
    }

    /// Appends a stratum at the top (`n + 1`) of the pipeline.
    pub fn push_stratum(&mut self, stratum: Box<dyn RecallStratum + 'a>) {
        self.strata.push(stratum);
    }

    /// Inserts a stratum at `index` (clamped to `self.strata.len()`).
    pub fn insert_stratum(&mut self, index: usize, stratum: Box<dyn RecallStratum + 'a>) {
        let pos = index.min(self.strata.len());
        self.strata.insert(pos, stratum);
    }

    /// Removes and returns the first stratum matching `name`, if present.
    pub fn remove_stratum(&mut self, name: &str) -> Option<Box<dyn RecallStratum + 'a>> {
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
            let mut frame = signal
                .map_to_gateway_with_persona_floor(
                    ctx.entries,
                    last_layer,
                    ctx.policy.min_coherence * 0.75,
                )
                .frame;
            frame.active_head_count = (ctx.num_q_heads as u16)
                .clamp(1, liminika_core::liminal::MAX_ATTENTION_HEADS as u16);
            signal.gateway_frame = frame;
            signal.telemetry.final_top1_anchor = final_top1_anchor;
            signal.telemetry.provenance = frame.to_causal_report();
        }
        Ok((signal, directive))
    }
}

impl Default for StratumPipeline<'static> {
    fn default() -> Self {
        let mut p = Self::new();
        p.push_stratum(Box::new(LocalFeatureWorkerStratum::default()));
        p.push_stratum(Box::new(HeadAngleWorkerStratum::default()));
        p.push_stratum(Box::new(LocalTokenResonanceStratum::default()));
        p.push_stratum(Box::new(PerspectiveHubnessStratum {
            perspective: &DEFAULT_PERSPECTIVE,
        }));
        p.push_stratum(Box::new(SubstrateBoundaryStratum {
            gate: &DEFAULT_GATE,
        }));
        p
    }
}

static DEFAULT_PERSPECTIVE: AsymmetricHubnessPerspective = AsymmetricHubnessPerspective {
    neighbor_k: 4,
    hubness_damping: 0.50,
    perspective_z_gain: 0.025,
};
static DEFAULT_GATE: CoherenceGuardGate = CoherenceGuardGate;
