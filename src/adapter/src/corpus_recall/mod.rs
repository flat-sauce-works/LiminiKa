// name: src/adapter/src/corpus_recall/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dynamic natural-language attractor corpus indexing, Liminal 2D boundary gates,
//! and persistent token-sequence recall ([`TextAttractorCorpus`]).

mod boundary_2d;
mod projection;

pub use boundary_2d::{
    Boundary2DDecision, Boundary2DInput, Boundary2DRegion, Boundary2DThresholds,
    CoherenceGuardGate, DEFAULT_GUARD_HARD_MARGIN, DEFAULT_GUARD_OBSTRUCTION_THRESHOLD,
    DEFAULT_SOFT_GUARD_Q7_SCALE, GuardLayerClassification, GuardLayerMode, RecallBoundaryGate,
    RecallBoundaryMetrics, RecallGateDecision, classify_guard_tier, effective_guard_hard_margin,
};
pub use projection::attenuate_guard_q7_soft;
pub(crate) use projection::{
    AttractorRecallProjection, RecalledAnchorRef,
    build_exterior_sandbox_anchors_per_layer_from_projection,
    build_fact_anchors_per_layer_from_projection, build_guard_anchors_per_layer_from_projection,
    build_persona_anchors_per_layer_from_projection, classify_guard_tier_from_projection,
    compute_lacuna_partition_8, context_fingerprint_from_projection,
    join_exterior_context_from_projection, join_fact_context_from_projection,
    join_persona_context_from_projection,
};

use super::*;
use crate::corpus_recall_cache::{CorpusRecallCache, LazyRecallCache};
use crate::pooling::NullCalibrationEnvelope;
use crate::stratum::CoalitionThresholds;

/// Semantic origin role of a natural-language text attractor inside [`TextAttractorCorpus`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttractorRole {
    /// Factual / declarative knowledge chunk (`ObjectiveFact` origin).
    Fact,
    /// Stylistic / behavioral / personality log (`SubjectivePersona` origin).
    Persona,
    /// Contrastive / negative guard boundary (`ContradictionGuard` origin).
    Guard,
}

impl AttractorRole {
    /// Returns a dense slot index (`0..3`) for fixed-size per-role arrays.
    #[inline]
    #[must_use]
    pub const fn slot_index(self) -> usize {
        match self {
            Self::Fact => 0,
            Self::Persona => 1,
            Self::Guard => 2,
        }
    }
}

/// Geometric classification of a query's position relative to the compiled attractor manifold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LacunaBoundaryState {
    /// Query clearly resides inside a resonant attractor basin.
    InsideBasin,
    /// Query sits on the `Liminal` boundary between two competing attractors.
    LiminalBoundary,
    /// Query falls in ungrounded exterior space outside all basins; avoid forced steering.
    LiminalExterior,
}

/// Single natural-language-backed attractor entry compiled for the active model's RoPE manifold.
#[derive(Debug, Clone)]
pub struct TextAttractorEntry {
    /// Unique stable anchor identifier (`1..`).
    pub anchor_id: u32,
    /// Origin role (`Fact`, `Persona`, or `Guard`).
    pub role: AttractorRole,
    /// Lightweight natural-language source text (reproducible source of truth).
    pub text: String,
    /// Per-layer 64-head Q7 phase signatures (`num_layers` entries).
    pub per_layer_q7: Vec<[lmnk_q7_t; MAX_ATTENTION_HEADS]>,
    /// Base gravitational pull energy for gateway composition.
    pub energy: f32,
    /// L2-normalized centered residual Q signature across steered layers for high-resolution semantic recall.
    pub(crate) residual_sig: Vec<f32>,
}

impl RelativeAttractorNode for TextAttractorEntry {
    fn relative_phase_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        let q7 = self
            .per_layer_q7
            .last()
            .copied()
            .unwrap_or([0_i8; MAX_ATTENTION_HEADS]);
        load_q7_words_64(&q7)
    }

    fn relative_energy(&self) -> f32 {
        self.energy
    }

    fn relative_anchor_id(&self) -> u32 {
        self.anchor_id
    }
}

impl CausalProvenanceSignal for TextAttractorEntry {
    fn causal_provenance_bits(&self) -> u32 {
        match self.role {
            AttractorRole::Fact => CAUSAL_AXIS_OBJECTIVE_FACT,
            AttractorRole::Persona => CAUSAL_AXIS_SUBJECTIVE_PERSONA,
            AttractorRole::Guard => CAUSAL_AXIS_CONTRADICTION_GUARD,
        }
    }
}

impl ManifoldGatewaySignal for TextAttractorEntry {
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        let q7 = self
            .per_layer_q7
            .last()
            .copied()
            .unwrap_or([0_i8; MAX_ATTENTION_HEADS]);
        let layer_idx = self.per_layer_q7.len().saturating_sub(1) as u64;
        match self.role {
            AttractorRole::Fact => ObjectiveFact::new(q7, self.energy, layer_idx, self.anchor_id)
                .synthesize_gateway_frame(),
            AttractorRole::Persona => {
                SubjectivePersona::new(q7, self.energy, layer_idx, self.anchor_id)
                    .synthesize_gateway_frame()
            }
            AttractorRole::Guard => GuardPhaseAnchor::new(q7, self.energy.clamp(0.0, 1.0), 0.5)
                .synthesize_gateway_frame(),
        }
    }
}

/// Dynamic top-K recall result from [`TextAttractorCorpus::recall_for_query`].
#[derive(Debug, Clone)]
pub struct DynamicAttractorRecall<'a> {
    /// Whether the query landed inside a basin, on a liminal boundary, or in `Liminal` empty space.
    pub lacuna_state: LacunaBoundaryState,
    /// Highest geometric resonance score among Fact attractors.
    pub top1_fact_score: f32,
    /// Second-highest geometric resonance score among Fact attractors.
    pub top2_fact_score: f32,
    /// Highest geometric resonance score among Persona attractors.
    pub top1_persona_score: f32,
    /// Smallest per-attractor local `Liminal` depth observed across all attractors (`0.0` = deep in a basin).
    pub min_local_lacuna_depth: f32,
    /// Highest Guard resonance minus highest Fact/Persona resonance (`> 0` indicates guard-dominant conflict).
    pub guard_conflict_margin: f32,
    /// Substrate regime selected by the upper boundary stratum (`PassiveSidecar`, `InertialLayerSkip`,
    /// `ExpressPhaseSteer`, `SubHeadRoute`, `FullOrchestration`, `CrystallizeMacroBasin`, or `RepulsionEscape`).
    pub action_decision: StigmergicActionDecision,
    /// Authoritative 128-byte core [`GatewayFrame`] synthesized and arbitrated by the pipeline.
    pub gateway_frame: GatewayFrame,
    /// Cross-channel probe telemetry and core [`CausalProvenanceReport`].
    pub telemetry: StratumProbeTelemetry,
    /// Top-K recalled Fact entries (empty when `lacuna_state == LiminalExterior`).
    pub recalled_facts: Vec<(&'a TextAttractorEntry, f32)>,
    /// Top-K recalled Persona entries (suppressed only on Guard repulsion or foreign OOD tokens).
    pub recalled_personas: Vec<(&'a TextAttractorEntry, f32)>,
    /// Strongest observed Guard attractor in the corpus for this query, if any Guard exists.
    pub top_guard: Option<(&'a TextAttractorEntry, f32)>,
    /// Branchless 4-region lacuna partition (`S & O`, `S \ O`, `O \ S`, Unobserved) for this query.
    pub lacuna_partition: LacunaPartition8,
    /// Non-copy exterior Known Unknown ($O \setminus S$) pointers routed strictly through sandbox staging.
    pub exterior_sandbox_facts: Vec<(&'a TextAttractorEntry, f32)>,
    /// True when a validated Known Unknown ($O \setminus S$) exterior pointer was staged via $S \cap O$ bridge.
    pub is_known_unknown_exterior: bool,
}

impl DynamicAttractorRecall<'_> {
    /// Converts this borrowed recall outcome into a lifetime-free [`OwnedAttractorRecall`]
    /// without copying un-recalled corpus entries or full-resolution residual signatures.
    #[must_use]
    pub fn to_owned_recall(&self) -> OwnedAttractorRecall {
        OwnedAttractorRecall {
            lacuna_state: self.lacuna_state,
            top1_fact_score: self.top1_fact_score,
            top2_fact_score: self.top2_fact_score,
            top1_persona_score: self.top1_persona_score,
            min_local_lacuna_depth: self.min_local_lacuna_depth,
            guard_conflict_margin: self.guard_conflict_margin,
            action_decision: self.action_decision,
            gateway_frame: self.gateway_frame,
            telemetry: self.telemetry,
            recalled_facts: self
                .recalled_facts
                .iter()
                .map(|&(e, s)| RecalledAnchorRecord::from_entry(e, s))
                .collect(),
            recalled_personas: self
                .recalled_personas
                .iter()
                .map(|&(e, s)| RecalledAnchorRecord::from_entry(e, s))
                .collect(),
            top_guard: self
                .top_guard
                .map(|(e, s)| RecalledAnchorRecord::from_entry(e, s)),
            lacuna_partition: self.lacuna_partition,
            exterior_sandbox_facts: self
                .exterior_sandbox_facts
                .iter()
                .map(|&(e, s)| RecalledAnchorRecord::from_entry(e, s))
                .collect(),
            is_known_unknown_exterior: self.is_known_unknown_exterior,
        }
    }

    /// Materializes per-layer `ObjectiveFact<[i8; 64]>` anchors for
    /// [`ResonantSignalSlice`](liminika_core::runtime::ResonantSignalSlice).
    #[must_use]
    pub fn build_fact_anchors_per_layer(
        &self,
        num_layers: usize,
    ) -> Vec<Vec<ObjectiveFact<[lmnk_q7_t; MAX_ATTENTION_HEADS]>>> {
        build_fact_anchors_per_layer_from_projection(self, num_layers)
    }

    /// Materializes per-layer `SubjectivePersona<[i8; 64]>` anchors for
    /// [`ResonantSignalSlice`](liminika_core::runtime::ResonantSignalSlice).
    #[must_use]
    pub fn build_persona_anchors_per_layer(
        &self,
        num_layers: usize,
    ) -> Vec<Vec<SubjectivePersona<[lmnk_q7_t; MAX_ATTENTION_HEADS]>>> {
        build_persona_anchors_per_layer_from_projection(self, num_layers)
    }

    /// Concatenates all recalled Fact and Persona attractor texts in recall order.
    #[must_use]
    pub fn recalled_context_text(&self) -> String {
        self.recalled_facts
            .iter()
            .map(|(e, _)| e.text.as_str())
            .chain(self.recalled_personas.iter().map(|(e, _)| e.text.as_str()))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Concatenates only recalled Fact attractor texts (delegating $S \cap O$ persona rules to steering).
    #[must_use]
    pub fn recalled_facts_only_context_text(&self) -> String {
        join_fact_context_from_projection(self)
    }

    /// Concatenates non-copy Known Unknown ($O \setminus S$) exterior pointer texts staged via sandbox.
    #[must_use]
    pub fn exterior_pointer_context_text(&self) -> String {
        join_exterior_context_from_projection(self)
    }

    /// Computes a deterministic 64-bit FNV-1a context fingerprint across recalled attractors,
    /// returning `0` when the query lies in ungrounded `Liminal` or recalled text is empty.
    #[must_use]
    pub fn context_fingerprint_u64(&self) -> u64 {
        context_fingerprint_from_projection(self)
    }
}

/// Policy parameters controlling top-K attractor recall and `Liminal` boundary classification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecallPolicy {
    /// Maximum number of fact attractors to return when inside a basin or liminal boundary.
    pub top_k_facts: usize,
    /// Maximum number of persona attractors to return.
    pub top_k_personas: usize,
    /// Minimum top-1 geometric coherence required to stay inside an attractor basin (below this -> `Liminal`).
    pub min_coherence: f32,
    /// Minimum top-1 minus top-2 score gap to avoid `LiminalBoundary`.
    pub boundary_margin: f32,
    /// Guard conflict margin (`max_guard_score - top1_fact_score`) above which Layer 1 suppression applies.
    pub guard_suppress_margin: f32,
}

impl Default for RecallPolicy {
    fn default() -> Self {
        Self {
            top_k_facts: 2,
            top_k_personas: 1,
            min_coherence: 0.35,
            boundary_margin: 0.03,
            guard_suppress_margin: 0.0,
        }
    }
}

/// Relational neighborhood context observed from a single attractor's vantage point, enabling
/// asymmetric, perspective-dependent `Liminal` measurements (analogous to viewing a chain from the
/// left edge vs. the right edge or per-head Attention affinity).
#[derive(Debug, Clone, Copy)]
pub struct AttractorPerspectiveContext<'a> {
    /// Target attractor whose local frame is being evaluated.
    pub entry: &'a TextAttractorEntry,
    /// Raw cosine similarity between the query and this attractor.
    pub raw_cosine: f32,
    /// Mean cosine from this attractor to its local nearest neighbors inside the corpus.
    pub neighbor_mean_cosine: f32,
    /// Standard deviation of cosine similarity around this attractor's local neighborhood.
    pub neighbor_std_cosine: f32,
    /// Corpus-wide average of `neighbor_mean_cosine`.
    pub global_neighbor_mean: f32,
    /// Query's mean cosine similarity across the corpus (query-side baseline).
    pub query_mean_cosine: f32,
    /// Standard deviation of the query's cosine similarity across the corpus.
    pub query_std_cosine: f32,
}

/// Per-attractor measurement returned by [`AttractorBoundaryPerspective::measure_perspective`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerspectiveBoundaryReading {
    /// Perspective-calibrated affinity score as viewed from this attractor's angle.
    pub perspective_affinity: f32,
    /// Non-negative local `Liminal` gap seen from this attractor (`0.0` = well inside basin).
    pub local_lacuna_depth: f32,
}

/// Trait computing perspective-dependent affinity and local `Liminal` boundary depth per attractor.
///
/// Rather than judging every attractor from a single isotropic origin, each attractor can have a
/// distinct local density and boundary geometry so central hubs do not eclipse peripheral basins.
pub trait AttractorBoundaryPerspective {
    /// Measures affinity and local `Liminal` depth from `ctx.entry`'s vantage point.
    fn measure_perspective(
        &self,
        ctx: &AttractorPerspectiveContext<'_>,
        min_coherence: f32,
    ) -> PerspectiveBoundaryReading;
}

/// Default number of same-role nearest neighbors used for local hubness baseline statistics.
pub const DEFAULT_NEIGHBOR_K: usize = 4;

/// Maximum same-role or global entry count using exact $O(N^2)$ pairwise cosine scans before
/// switching to deterministic SWAR `BitCut64` + stride-sampled quasi-linear calibration.
pub const EXACT_CORPUS_STATS_LIMIT: usize = 64;

/// Precomputed role-scoped local neighborhood statistics for a single corpus attractor entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntryNeighborhoodStats {
    /// Mean cosine similarity from this attractor to its top-`k` same-role neighbors.
    pub neighbor_mean_cosine: f32,
    /// Standard deviation of cosine similarity across all other same-role entries.
    pub neighbor_std_cosine: f32,
}

impl Default for EntryNeighborhoodStats {
    #[inline]
    fn default() -> Self {
        Self {
            neighbor_mean_cosine: 0.0,
            neighbor_std_cosine: 0.05,
        }
    }
}

/// Read-only provider supplying role-scoped local neighborhood statistics and role index partitions.
///
/// Decouples [`PerspectiveHubnessStratum`] and lower feature workers from corpus internals so
/// per-query recall reuses precomputed $O(N^2)$ neighborhood baselines in $O(1)$ or accepts
/// alternative/mock providers in tests.
pub trait NeighborhoodStatsProvider: Send + Sync {
    /// Returns the local neighborhood mean and standard deviation for corpus entry `entry_idx`.
    fn entry_stats(&self, entry_idx: usize) -> EntryNeighborhoodStats;

    /// Returns the corpus-wide mean of `neighbor_mean_cosine` for `role`.
    fn role_global_mean(&self, role: AttractorRole) -> f32;

    /// Returns the ordered corpus entry indices belonging to `role`.
    fn role_indices(&self, role: AttractorRole) -> &[usize];
}

/// Default [`AttractorBoundaryPerspective`] implementing bidirectional CSLS hubness normalization and
/// asymmetric Z-score scaling across each attractor's local neighborhood.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AsymmetricHubnessPerspective {
    /// Number of nearest corpus neighbors defining each attractor's local baseline
    /// (matches [`DEFAULT_NEIGHBOR_K`] in cached corpus statistics).
    pub neighbor_k: usize,
    /// Damping weight applied to centered attractor hubness (`neighbor_mean - global_mean`).
    pub hubness_damping: f32,
    /// Weight applied to the query/attractor bidirectional Z-score modulated by positive raw cosine.
    pub perspective_z_gain: f32,
}

impl Default for AsymmetricHubnessPerspective {
    fn default() -> Self {
        Self {
            neighbor_k: DEFAULT_NEIGHBOR_K,
            hubness_damping: 0.50,
            perspective_z_gain: 0.025,
        }
    }
}

impl AttractorBoundaryPerspective for AsymmetricHubnessPerspective {
    fn measure_perspective(
        &self,
        ctx: &AttractorPerspectiveContext<'_>,
        min_coherence: f32,
    ) -> PerspectiveBoundaryReading {
        let hub_shift = ctx.neighbor_mean_cosine - ctx.global_neighbor_mean;
        let z_from_attractor =
            (ctx.raw_cosine - ctx.neighbor_mean_cosine) / ctx.neighbor_std_cosine.max(0.05);
        let z_from_query =
            (ctx.raw_cosine - ctx.query_mean_cosine) / ctx.query_std_cosine.max(0.05);
        let rel_z = (0.5 * (z_from_attractor + z_from_query)).clamp(-3.0, 3.0);
        let positive_support = ctx.raw_cosine.max(0.0);
        // Gate hubness compensation by positive raw support so unrelated singleton entries never
        // gain phantom positive affinity when `raw_cosine <= 0`, while true dense hubs still damp.
        let support_gate = (positive_support / min_coherence.max(0.05)).clamp(0.0, 1.0);
        let damped_hub = if hub_shift < 0.0 {
            hub_shift * support_gate * support_gate
        } else {
            hub_shift * support_gate
        };
        let perspective_affinity = ctx.raw_cosine - self.hubness_damping * damped_hub
            + self.perspective_z_gain * positive_support * rel_z;
        let local_lacuna_depth = (min_coherence - perspective_affinity).max(0.0);
        PerspectiveBoundaryReading {
            perspective_affinity,
            local_lacuna_depth,
        }
    }
}

/// Model-agnostic Cold-Path corpus that compiles natural-language logs, facts, personas, and
/// contrastive guards into a dynamic attractor manifold with explicit `Liminal` 2D boundary detection.
#[derive(Debug, Default, Clone)]
pub struct TextAttractorCorpus {
    entries: Vec<TextAttractorEntry>,
    next_anchor_id: u32,
    common_mode_basis: Vec<Vec<f32>>,
    null_calibration: Option<Box<NullCalibrationEnvelope>>,
    recall_cache: LazyRecallCache,
}

impl TextAttractorCorpus {
    /// Creates an empty [`TextAttractorCorpus`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_anchor_id: 1,
            common_mode_basis: Vec::new(),
            null_calibration: None,
            recall_cache: LazyRecallCache::new(),
        }
    }

    /// Returns all stored attractor entries.
    #[must_use]
    pub fn entries(&self) -> &[TextAttractorEntry] {
        &self.entries
    }

    /// Attaches or clears an in-memory 2-fold OOD [`NullCalibrationEnvelope`] powering calibrated
    /// 2D `(top1_coh, unexp)` decision boundaries and relative-scale coalition thresholds.
    pub fn set_null_calibration(&mut self, envelope: Option<NullCalibrationEnvelope>) {
        self.null_calibration = envelope.map(Box::new);
    }

    /// Returns the currently attached [`NullCalibrationEnvelope`], or `None` under `--no-base-prior`.
    #[must_use]
    pub fn null_calibration(&self) -> Option<NullCalibrationEnvelope> {
        self.null_calibration.as_deref().copied()
    }

    /// Derives active [`Boundary2DThresholds`] for `min_coherence` (calibrated when
    /// [`Self::null_calibration`] is present, or exact 1D fallback under `--no-base-prior`).
    #[must_use]
    pub fn boundary_2d_thresholds(&self, min_coherence: f32) -> Boundary2DThresholds {
        Boundary2DThresholds::from_optional_envelope(
            self.null_calibration.as_deref(),
            min_coherence,
        )
    }

    #[inline]
    pub(crate) fn recall_cache(&self) -> &CorpusRecallCache {
        self.recall_cache
            .get_or_init(|| CorpusRecallCache::build(&self.entries, &self.common_mode_basis))
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn is_recall_cache_populated(&self) -> bool {
        self.recall_cache.is_populated()
    }

    pub(crate) fn build_steered_residual_sig(
        geom: &RopeGeometry,
        q_layers: &[&[f32]],
        ref_layers: &[&[f32]],
    ) -> Vec<f32> {
        let mut sig = Vec::new();
        for (l, (q, r)) in q_layers.iter().zip(ref_layers.iter()).enumerate() {
            if !geom.is_rope_layer(l) {
                continue;
            }
            let start = sig.len();
            for (&qv, &rv) in q.iter().zip(r.iter()) {
                sig.push(qv - rv);
            }
            // Equalize per-layer contribution so late-layer norm spikes cannot drown mid-layer semantics.
            Self::l2_normalize_in_place(&mut sig[start..]);
        }
        Self::l2_normalize_in_place(&mut sig);
        sig
    }

    fn l2_normalize_in_place(v: &mut [f32]) {
        let norm_sq: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        let norm = norm_sq.sqrt();
        if norm > 1e-12 {
            let inv = (1.0 / norm) as f32;
            for x in v.iter_mut() {
                *x *= inv;
            }
        }
    }

    pub(crate) fn project_and_normalize_in_place_with_ratio(
        out: &mut [f32],
        basis: &[Vec<f32>],
    ) -> f32 {
        if basis.is_empty() || out.is_empty() {
            return 1.0;
        }
        let raw_norm_sq: f64 = out.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        let raw_norm = raw_norm_sq.sqrt();
        for dir in basis {
            if dir.len() != out.len() {
                continue;
            }
            let dot: f64 = out
                .iter()
                .zip(dir.iter())
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let d = dot as f32;
            for (x, &c) in out.iter_mut().zip(dir.iter()) {
                *x -= d * c;
            }
        }
        let perp_norm_sq: f64 = out.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        let perp_norm = perp_norm_sq.sqrt();
        let ratio = if raw_norm > 1e-12 {
            (perp_norm / raw_norm).clamp(0.0, 1.0) as f32
        } else {
            0.0
        };
        if perp_norm > 1e-12 {
            let inv = (1.0 / perp_norm) as f32;
            for x in out.iter_mut() {
                *x *= inv;
            }
        }
        ratio
    }

    pub(crate) fn project_and_normalize_with_ratio(
        v: &[f32],
        basis: &[Vec<f32>],
    ) -> (Vec<f32>, f32) {
        let mut out = v.to_vec();
        let ratio = Self::project_and_normalize_in_place_with_ratio(&mut out, basis);
        (out, ratio)
    }

    pub(crate) fn project_and_normalize(v: &[f32], basis: &[Vec<f32>]) -> Vec<f32> {
        Self::project_and_normalize_with_ratio(v, basis).0
    }

    fn orthogonalize_against_basis(v: &mut [f32], basis: &[Vec<f32>]) {
        for dir in basis {
            let dot: f64 = v
                .iter()
                .zip(dir.iter())
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let d = dot as f32;
            for (dst, &c) in v.iter_mut().zip(dir.iter()) {
                *dst -= d * c;
            }
        }
    }

    pub(crate) fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut dot = 0.0_f64;
        let mut na = 0.0_f64;
        let mut nb = 0.0_f64;
        for i in 0..len {
            let x = f64::from(a[i]);
            let y = f64::from(b[i]);
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        let denom = (na * nb).sqrt();
        if denom > 1e-12 {
            (dot / denom).clamp(-1.0, 1.0) as f32
        } else {
            0.0
        }
    }

    /// Estimates up to `max_components` orthonormal common-mode directions from borrowed `sigs`
    /// via deflated power iteration using $O(D)$ scratch memory (zero corpus cloning).
    ///
    /// Because $P_\perp = I - Q Q^T$ is an orthogonal projector, for any $v \in Q^\perp$ we have
    /// $(X P_\perp)^T (X P_\perp) v = P_\perp (X^T X v)$, allowing exact subspace deflation without
    /// duplicating or mutating the input signature matrix.
    pub(crate) fn fit_common_mode_subspace_from_sigs(
        sigs: &[&[f32]],
        max_components: usize,
    ) -> Vec<Vec<f32>> {
        let mut basis: Vec<Vec<f32>> = Vec::new();
        if sigs.len() < 4 || max_components == 0 {
            return basis;
        }
        let dim = sigs[0].len();
        if dim == 0 || sigs.iter().any(|s| s.len() != dim) {
            return basis;
        }
        let rank_cap = max_components.min(sigs.len() / 3);
        let mut sum_x = vec![0.0_f32; dim];
        for s in sigs {
            for (dst, &src) in sum_x.iter_mut().zip(s.iter()) {
                *dst += src;
            }
        }
        let mut next = vec![0.0_f32; dim];

        for _ in 0..rank_cap {
            let mut v = sum_x.clone();
            Self::orthogonalize_against_basis(&mut v, &basis);
            Self::l2_normalize_in_place(&mut v);
            if v.iter().all(|&x| x == 0.0) {
                break;
            }
            for _ in 0..8 {
                Self::orthogonalize_against_basis(&mut v, &basis);
                next.fill(0.0);
                for s in sigs {
                    let dot: f64 = s
                        .iter()
                        .zip(v.iter())
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum();
                    let d = dot as f32;
                    for (dst, &src) in next.iter_mut().zip(s.iter()) {
                        *dst += d * src;
                    }
                }
                Self::orthogonalize_against_basis(&mut next, &basis);
                let norm_sq: f64 = next.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
                if norm_sq < 1e-12 {
                    break;
                }
                Self::l2_normalize_in_place(&mut next);
                std::mem::swap(&mut v, &mut next);
            }
            let norm_sq: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            if norm_sq < 0.5 {
                break;
            }
            basis.push(v);
        }
        basis
    }

    /// Projects every stored entry into a CRC32-stamped [`PhaseAnchorRecord`] for `.lmnkc`
    /// container persistence or domain-agnostic [`AnchorWorkingSetPager`] paging.
    #[must_use]
    pub fn to_phase_anchor_records(&self) -> Vec<PhaseAnchorRecord> {
        self.entries
            .iter()
            .map(|entry| {
                let slot = match entry.role {
                    AttractorRole::Fact => ATTRACTOR_SLOT_KNOWLEDGE,
                    AttractorRole::Persona => ATTRACTOR_SLOT_PERSONA,
                    AttractorRole::Guard => ATTRACTOR_SLOT_REPULSION,
                };
                let mut rec = PhaseAnchorRecord::new_empty(entry.anchor_id, slot, 0);
                rec.base_energy = entry.energy;
                rec.phase_gravity_mass = entry.energy;
                rec.sample_count = 1;
                if let Some(last_layer) = entry.per_layer_q7.last() {
                    rec.phase_profile = *last_layer;
                }
                rec.stamp_crc32();
                rec
            })
            .collect()
    }

    /// Constructs a domain-agnostic [`AnchorWorkingSetPager`] backed by all entries in this corpus.
    #[must_use]
    pub fn build_working_set_pager(&self, config: AnchorPagerConfig) -> AnchorWorkingSetPager {
        let records = self.to_phase_anchor_records();
        AnchorWorkingSetPager::from_records(&records, config)
    }

    /// Estimates up to `max_components` orthonormal common-mode directions from corpus signatures
    /// via deflated power iteration (`3` components by default removes shared prompt bias).
    pub fn fit_common_mode_subspace(&mut self, max_components: usize) {
        let sig_refs: Vec<&[f32]> = if self.entries.len() <= EXACT_CORPUS_STATS_LIMIT {
            self.entries
                .iter()
                .map(|e| e.residual_sig.as_slice())
                .collect()
        } else {
            let n = self.entries.len();
            (0..EXACT_CORPUS_STATS_LIMIT)
                .map(|s| {
                    self.entries[(s * n) / EXACT_CORPUS_STATS_LIMIT]
                        .residual_sig
                        .as_slice()
                })
                .collect()
        };
        self.common_mode_basis =
            Self::fit_common_mode_subspace_from_sigs(&sig_refs, max_components);
        self.recall_cache.invalidate();
    }

    /// Appends external orthonormal prior directions (e.g., startup base-model framing modes)
    /// into the common-mode projection basis via modified Gram-Schmidt orthogonalization.
    pub fn extend_common_mode_basis(&mut self, extra_dirs: &[Vec<f32>], max_total: usize) {
        let Some(expected_dim) = self
            .entries
            .first()
            .map(|e| e.residual_sig.len())
            .filter(|&d| d > 0)
        else {
            return;
        };
        let mut changed = false;
        for dir in extra_dirs {
            if self.common_mode_basis.len() >= max_total {
                break;
            }
            if dir.len() != expected_dim {
                continue;
            }
            let mut orth = dir.clone();
            for existing in &self.common_mode_basis {
                let dot: f64 = orth
                    .iter()
                    .zip(existing.iter())
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                let d = dot as f32;
                for (dst, &c) in orth.iter_mut().zip(existing.iter()) {
                    *dst -= d * c;
                }
            }
            let norm_sq: f64 = orth.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            if norm_sq >= 0.25 {
                Self::l2_normalize_in_place(&mut orth);
                self.common_mode_basis.push(orth);
                changed = true;
            }
        }
        if changed {
            self.recall_cache.invalidate();
        }
    }

    /// Calibrates [`RecallPolicy`] from the empirical off-diagonal similarity distribution of the
    /// corpus (after common-mode projection), avoiding hardcoded model-specific thresholds.
    #[must_use]
    pub fn calibrate_recall_policy(
        &self,
        top_k_facts: usize,
        top_k_personas: usize,
    ) -> RecallPolicy {
        if self.entries.len() < 2 {
            return RecallPolicy {
                top_k_facts,
                top_k_personas,
                ..RecallPolicy::default()
            };
        }
        let cache = self.recall_cache();
        let n_entries = self.entries.len();
        let sample_indices: Vec<usize> = if n_entries <= EXACT_CORPUS_STATS_LIMIT {
            (0..n_entries).collect()
        } else {
            (0..EXACT_CORPUS_STATS_LIMIT)
                .map(|s| (s * n_entries) / EXACT_CORPUS_STATS_LIMIT)
                .collect()
        };
        let m = sample_indices.len();
        let mut sims = Vec::with_capacity((m * m.saturating_sub(1)) / 2);
        for a in 0..m {
            let sig_i = cache.projected_sig(sample_indices[a]);
            for &j in &sample_indices[(a + 1)..] {
                sims.push(f64::from(Self::cosine_f32(sig_i, cache.projected_sig(j))));
            }
        }
        let n = sims.len().max(1) as f64;
        let mean = sims.iter().sum::<f64>() / n;
        let var = sims.iter().map(|&s| (s - mean) * (s - mean)).sum::<f64>() / n;
        let std = var.sqrt();
        let min_coherence = (mean + 1.25 * std).clamp(0.15, 0.75) as f32;
        let boundary_margin = (0.30 * std).clamp(0.01, 0.12) as f32;
        RecallPolicy {
            top_k_facts,
            top_k_personas,
            min_coherence,
            boundary_margin,
            guard_suppress_margin: 0.0,
        }
    }

    /// Compiles and appends a single natural-language text chunk into the attractor corpus using
    /// the active model's `adapter` and per-layer baseline reference activations.
    ///
    /// # Errors
    ///
    /// Returns an error if `per_layer_q.len() != baseline_ref_per_layer.len()` or any layer slice
    /// mismatches `adapter.geometry().q_rot_len()`.
    pub fn ingest_text_activations(
        &mut self,
        adapter: &ModelSteeringAdapter,
        role: AttractorRole,
        text: impl Into<String>,
        per_layer_q: &[&[f32]],
        baseline_ref_per_layer: &[&[f32]],
        energy: f32,
    ) -> Result<u32, String> {
        self.ingest_text_activations_with_phase_ref(
            adapter,
            role,
            text,
            per_layer_q,
            baseline_ref_per_layer,
            baseline_ref_per_layer,
            energy,
        )
    }

    /// Compiles and appends a text chunk while allowing a role-specific Procrustes phase baseline
    /// (`phase_ref_per_layer`, e.g. non-Guard centroid for `Guard` poles) separate from the unified
    /// recall residual centering reference (`residual_ref_per_layer`).
    ///
    /// # Errors
    ///
    /// Returns an error if layer counts or per-layer `q_rot` lengths mismatch `adapter.geometry()`.
    // v0.3.0 frozen public Rust API; parameter-struct consolidation deferred to v0.4.0.
    #[allow(clippy::too_many_arguments)]
    pub fn ingest_text_activations_with_phase_ref(
        &mut self,
        adapter: &ModelSteeringAdapter,
        role: AttractorRole,
        text: impl Into<String>,
        per_layer_q: &[&[f32]],
        residual_ref_per_layer: &[&[f32]],
        phase_ref_per_layer: &[&[f32]],
        energy: f32,
    ) -> Result<u32, String> {
        if per_layer_q.len() != residual_ref_per_layer.len()
            || per_layer_q.len() != phase_ref_per_layer.len()
        {
            return Err(format!(
                "layer count mismatch: per_layer_q={}, residual_ref={}, phase_ref={}",
                per_layer_q.len(),
                residual_ref_per_layer.len(),
                phase_ref_per_layer.len()
            ));
        }
        let mut per_layer_q7 = Vec::with_capacity(per_layer_q.len());
        for (q, r) in per_layer_q.iter().zip(phase_ref_per_layer.iter()) {
            let q7 = adapter.bake_layer_procrustes_q7(r, q)?;
            let mut arr64 = [0i8; MAX_ATTENTION_HEADS];
            let n = q7.len().min(MAX_ATTENTION_HEADS);
            arr64[..n].copy_from_slice(&q7[..n]);
            per_layer_q7.push(arr64);
        }
        let residual_sig = Self::build_steered_residual_sig(
            adapter.geometry(),
            per_layer_q,
            residual_ref_per_layer,
        );
        let id = self.next_anchor_id;
        self.next_anchor_id = self.next_anchor_id.saturating_add(1);
        self.entries.push(TextAttractorEntry {
            anchor_id: id,
            role,
            text: text.into(),
            per_layer_q7,
            energy: energy.max(0.1),
            residual_sig,
        });
        self.recall_cache.invalidate();
        Ok(id)
    }

    /// Scores a query's per-layer Q activations against the corpus using the default
    /// multi-stratum pipeline, [`AsymmetricHubnessPerspective`], and
    /// [`CoherenceGuardGate`] to classify [`LacunaBoundaryState`].
    ///
    /// # Errors
    ///
    /// Returns an error if query/baseline layer slices mismatch `adapter.geometry()`.
    pub fn recall_for_query(
        &self,
        adapter: &ModelSteeringAdapter,
        query_per_layer_q: &[&[f32]],
        baseline_ref_per_layer: &[&[f32]],
        policy: &RecallPolicy,
    ) -> Result<DynamicAttractorRecall<'_>, String> {
        let perspective = AsymmetricHubnessPerspective::default();
        let mut pipeline = StratumPipeline::new();
        pipeline.push_stratum(Box::new(LocalFeatureWorkerStratum::default()));
        pipeline.push_stratum(Box::new(HeadAngleWorkerStratum::default()));
        pipeline.push_stratum(Box::new(LocalTokenResonanceStratum::default()));
        pipeline.push_stratum(Box::new(PerspectiveHubnessStratum {
            perspective: &perspective,
        }));
        pipeline.push_stratum(Box::new(SubstrateBoundaryStratum {
            gate: &CoherenceGuardGate,
        }));
        self.recall_for_query_with_strata(
            adapter,
            query_per_layer_q,
            baseline_ref_per_layer,
            policy,
            &pipeline,
        )
    }

    /// Scores a query's per-layer Q activations through a dynamically composable [`StratumPipeline`]
    /// (`n - 1 <-> n <-> n + 1`), enabling runtime addition or removal of strata.
    ///
    /// # Errors
    ///
    /// Returns an error if query/baseline layer slices mismatch `adapter.geometry()` or a stratum fails.
    pub fn recall_for_query_with_strata(
        &self,
        adapter: &ModelSteeringAdapter,
        query_per_layer_q: &[&[f32]],
        baseline_ref_per_layer: &[&[f32]],
        policy: &RecallPolicy,
        pipeline: &StratumPipeline<'_>,
    ) -> Result<DynamicAttractorRecall<'_>, String> {
        self.recall_for_query_with_strata_and_local_scores(
            adapter,
            query_per_layer_q,
            baseline_ref_per_layer,
            policy,
            pipeline,
            None,
        )
    }

    /// Scores a query's per-layer Q activations through a [`StratumPipeline`] while supplying an
    /// independent `local_token_scores` channel (e.g., ColBERT-style MaxSim point-cloud scores).
    ///
    /// # Errors
    ///
    /// Returns an error if query/baseline layer slices mismatch `adapter.geometry()`,
    /// `local_token_scores` length mismatches `self.entries().len()`, or a stratum fails.
    pub fn recall_for_query_with_strata_and_local_scores(
        &self,
        adapter: &ModelSteeringAdapter,
        query_per_layer_q: &[&[f32]],
        baseline_ref_per_layer: &[&[f32]],
        policy: &RecallPolicy,
        pipeline: &StratumPipeline<'_>,
        local_token_scores: Option<&[f32]>,
    ) -> Result<DynamicAttractorRecall<'_>, String> {
        let obs = local_token_scores.map(|scores| LocalChannelObservation {
            combined_scores: scores.to_vec(),
            ..Default::default()
        });
        self.recall_for_query_with_strata_observation(
            adapter,
            query_per_layer_q,
            baseline_ref_per_layer,
            policy,
            pipeline,
            obs,
        )
    }

    /// Scores a query's per-layer Q activations through a [`StratumPipeline`] while preserving an
    /// uncollapsed [`LocalChannelObservation`] (bidirectional Chamfer scores + per-token swarm
    /// deposits) for [`liminika_core::stigmergy::StigmergicSubstrate`] trail decomposition.
    ///
    /// # Errors
    ///
    /// Returns an error if query/baseline layer slices mismatch `adapter.geometry()`,
    /// observation vectors mismatch `self.entries().len()`, or a stratum fails.
    pub fn recall_for_query_with_strata_observation(
        &self,
        adapter: &ModelSteeringAdapter,
        query_per_layer_q: &[&[f32]],
        baseline_ref_per_layer: &[&[f32]],
        policy: &RecallPolicy,
        pipeline: &StratumPipeline<'_>,
        local_observation: Option<LocalChannelObservation>,
    ) -> Result<DynamicAttractorRecall<'_>, String> {
        self.recall_for_query_with_strata_observation_and_circuit(
            adapter,
            query_per_layer_q,
            baseline_ref_per_layer,
            policy,
            pipeline,
            local_observation,
            None,
        )
    }

    /// Scores a query's per-layer Q activations against the corpus using a custom [`StratumPipeline`],
    /// an optional [`LocalChannelObservation`], and an optional prior [`CognitiveTrajectoryLedger`]
    /// injected at the start of Pass 1 stratum inference.
    ///
    /// # Errors
    /// Returns an error if query/baseline layer slices mismatch `adapter.geometry()`,
    /// observation vectors mismatch `self.entries().len()`, or a stratum fails.
    // v0.3.0 frozen public Rust API; parameter-struct consolidation deferred to v0.4.0.
    #[allow(clippy::too_many_arguments)]
    pub fn recall_for_query_with_strata_observation_and_circuit(
        &self,
        adapter: &ModelSteeringAdapter,
        query_per_layer_q: &[&[f32]],
        baseline_ref_per_layer: &[&[f32]],
        policy: &RecallPolicy,
        pipeline: &StratumPipeline<'_>,
        local_observation: Option<LocalChannelObservation>,
        prior_circuit: Option<CognitiveTrajectoryLedger>,
    ) -> Result<DynamicAttractorRecall<'_>, String> {
        if query_per_layer_q.len() != baseline_ref_per_layer.len() {
            return Err(format!(
                "query/baseline layer mismatch: {} vs {}",
                query_per_layer_q.len(),
                baseline_ref_per_layer.len()
            ));
        }
        let geom = adapter.geometry();
        let expected = geom.q_rot_len();
        for (l, (q, r)) in query_per_layer_q
            .iter()
            .zip(baseline_ref_per_layer.iter())
            .enumerate()
        {
            if !geom.is_rope_layer(l) {
                continue;
            }
            if q.len() != expected || r.len() != expected {
                return Err(format!(
                    "layer {l}: expected q_rot length {expected}, got query={} baseline={}",
                    q.len(),
                    r.len()
                ));
            }
        }
        let raw_sig =
            Self::build_steered_residual_sig(geom, query_per_layer_q, baseline_ref_per_layer);
        let (query_sig, query_ortho_ratio) =
            Self::project_and_normalize_with_ratio(&raw_sig, &self.common_mode_basis);
        let cache = self.recall_cache();
        let boundary_2d = self.boundary_2d_thresholds(policy.min_coherence);
        let coalition_thresholds = CoalitionThresholds::from_boundary_2d(&boundary_2d);
        let ortho_gate = if self.common_mode_basis.is_empty() {
            1.0_f32
        } else {
            // Only attenuate degenerate near-pure-common-mode queries (`query_ortho_ratio` below
            // `corpus_ortho_ref * 0.45`, clamped to `[0.15, 0.25]`) so legitimate queries keep
            // full sensitivity while common-mode-heavy OOD probes never weaken the knee.
            let ortho_knee = (cache.corpus_ortho_ref * 0.45).clamp(0.15, 0.25);
            (query_ortho_ratio / ortho_knee).clamp(0.0, 1.0)
        };
        let n_entries = self.entries.len();
        let raw_cosines: Vec<f32> = (0..n_entries)
            .map(|i| {
                let c = Self::cosine_f32(&query_sig, cache.projected_sig(i));
                if c > 0.0 { c * ortho_gate } else { c }
            })
            .collect();
        let (
            token_channel,
            q_to_d_scores,
            d_to_q_scores,
            token_deposits,
            unexplained_token_ratio,
            query_seq_len,
        ) = match local_observation {
            Some(obs) => {
                if obs.combined_scores.len() != n_entries {
                    return Err(format!(
                        "local_token_scores length mismatch: expected {n_entries}, got {}",
                        obs.combined_scores.len()
                    ));
                }
                (
                    obs.combined_scores,
                    obs.q_to_d_scores,
                    obs.d_to_q_scores,
                    obs.token_deposits,
                    obs.unexplained_token_ratio.clamp(0.0, 1.0),
                    obs.query_seq_len,
                )
            }
            None => (Vec::new(), Vec::new(), Vec::new(), Vec::new(), 0.0_f32, 0),
        };
        let initial_signal = StratumSignal {
            candidate_scores: raw_cosines.clone(),
            local_token_scores: token_channel,
            q_to_d_scores,
            d_to_q_scores,
            token_deposits,
            linked_fact_indices: Vec::new(),
            decomposition_triggered: false,
            local_lacuna_depths: raw_cosines
                .iter()
                .map(|&s| (policy.min_coherence - s).max(0.0))
                .collect(),
            stigmergic_weights: vec![1.0_f32; n_entries],
            boundary_metrics: RecallBoundaryMetrics {
                top1_fact_score: -1.0,
                top2_fact_score: -1.0,
                top1_persona_score: -1.0,
                max_guard_score: -1.0,
                guard_conflict_margin: 0.0,
            },
            gate_decision: RecallGateDecision {
                lacuna_state: LacunaBoundaryState::LiminalExterior,
                suppress_fact_injection: true,
            },
            action_decision: StigmergicActionDecision::PassiveSidecar,
            gateway_frame: GatewayFrame::default(),
            swarm_report: StigmergicReconstitutionReport::default(),
            local_channel_preferred: false,
            telemetry: StratumProbeTelemetry {
                cognitive_trajectory: prior_circuit.unwrap_or_default(),
                query_ortho_ratio,
                unexplained_token_ratio,
                ..StratumProbeTelemetry::default()
            },
        };
        let stratum_ctx = StratumContext {
            entries: &self.entries,
            raw_sig: &raw_sig,
            query_sig: &query_sig,
            projected_sigs_flat: &cache.projected_sigs_flat,
            projected_sig_dim: cache.sig_dim,
            neighborhood_stats: cache,
            layer_chunk_dim: expected,
            num_q_heads: geom.num_q_heads,
            rotary_dim: geom.rotary_dim,
            query_seq_len,
            query_ortho_ratio,
            policy,
            coalition_thresholds,
        };
        let (mut signal, directive) = pipeline.evaluate(&stratum_ctx, initial_signal)?;

        let mut fact_scores: Vec<(&TextAttractorEntry, f32)> = Vec::new();
        let mut persona_scores: Vec<(&TextAttractorEntry, f32)> = Vec::new();
        let mut top_guard: Option<(&TextAttractorEntry, f32)> = None;
        for (entry, &score) in self.entries.iter().zip(signal.candidate_scores.iter()) {
            match entry.role {
                AttractorRole::Fact => fact_scores.push((entry, score)),
                AttractorRole::Persona => persona_scores.push((entry, score)),
                AttractorRole::Guard => {
                    if top_guard.is_none_or(|(_, best_s)| score > best_s) {
                        top_guard = Some((entry, score));
                    }
                }
            }
        }
        fact_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        persona_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top1_fact_candidate = fact_scores.first().copied();
        let top_any_entry = top1_fact_candidate.or_else(|| persona_scores.first().copied());

        let align_num = signal
            .boundary_metrics
            .top1_fact_score
            .max(signal.boundary_metrics.top1_persona_score)
            .max(signal.telemetry.local_top1_score * 0.90)
            .max(0.0);
        let lacuna_partition = compute_lacuna_partition_8(
            adapter,
            top_any_entry,
            align_num,
            boundary_2d.calibrated_min_coherence,
            query_per_layer_q,
            baseline_ref_per_layer,
        );

        let hard_margin =
            effective_guard_hard_margin(policy.guard_suppress_margin, boundary_2d.coherence_scale);
        let guard_floor = policy.min_coherence * 0.70;
        let guard_tier = classify_guard_tier(
            signal.boundary_metrics.max_guard_score,
            guard_floor,
            signal.boundary_metrics.guard_conflict_margin,
            hard_margin,
            directive.repulse_guard,
        );
        signal.telemetry.guard_hard_margin = guard_tier.hard_margin;
        signal.telemetry.is_soft_guard_caution = guard_tier.is_soft_guard_caution;

        let hard_guard_block = guard_tier.mode == GuardLayerMode::HardBlock;
        let suppress_facts = hard_guard_block
            || directive.suppress_injection
            || signal.gate_decision.suppress_fact_injection;
        let has_grounded_counter_fact = hard_guard_block
            && signal.boundary_metrics.top1_fact_score >= policy.min_coherence * 0.65;
        let mut recalled_facts = if has_grounded_counter_fact {
            // Surface the strongest grounded counter-fact so downstream generation and reflection
            // can actively refute the false premise while preserving `RepulsionEscape` gating.
            fact_scores.into_iter().take(1).collect()
        } else if suppress_facts {
            Vec::new()
        } else if signal.decomposition_triggered && !signal.linked_fact_indices.is_empty() {
            let max_coalition = policy.top_k_facts.max(signal.linked_fact_indices.len());
            signal
                .linked_fact_indices
                .iter()
                .filter_map(|&(idx, score)| {
                    self.entries
                        .get(idx)
                        .filter(|e| e.role == AttractorRole::Fact)
                        .map(|e| (e, signal.candidate_scores[idx].max(score)))
                })
                .take(max_coalition)
                .collect()
        } else {
            let top1_s = fact_scores.first().map_or(0.0_f32, |(_, s)| *s);
            let min_keep = (policy.min_coherence * 0.75).min(top1_s * 0.95);
            fact_scores
                .into_iter()
                .filter(|(_, s)| *s >= min_keep)
                .take(policy.top_k_facts)
                .collect()
        };

        let mut final_lacuna_state = signal.gate_decision.lacuna_state;
        let mut final_action_decision = signal.action_decision;
        let mut exterior_sandbox_facts = Vec::new();
        let decision_2d = boundary_2d.classify(&Boundary2DInput {
            top1_coh: signal.boundary_metrics.top1_fact_score,
            top2_coh: signal.boundary_metrics.top2_fact_score,
            boundary_margin: policy.boundary_margin,
            unexplained_token_ratio,
            query_seq_len,
            has_recalled_facts: !recalled_facts.is_empty(),
            has_top1_fact_candidate: top1_fact_candidate.is_some(),
            repulse_guard: hard_guard_block,
            guard_conflict_margin: signal.boundary_metrics.guard_conflict_margin,
            is_sandbox_eligible: lacuna_partition.is_sandbox_eligible(),
            bridge_count: lacuna_partition.bridge_count,
        });

        if !hard_guard_block {
            match decision_2d.region {
                Boundary2DRegion::Unobserved => {
                    recalled_facts.clear();
                    final_lacuna_state = LacunaBoundaryState::LiminalExterior;
                    if matches!(
                        final_action_decision,
                        StigmergicActionDecision::SubHeadRoute
                            | StigmergicActionDecision::CrystallizeMacroBasin
                            | StigmergicActionDecision::FullOrchestration
                    ) {
                        final_action_decision = StigmergicActionDecision::PassiveSidecar;
                    }
                }
                Boundary2DRegion::Exterior => {
                    final_lacuna_state = LacunaBoundaryState::LiminalExterior;
                    if decision_2d.stage_top1_candidate_to_exterior {
                        if let Some(cand) = top1_fact_candidate {
                            exterior_sandbox_facts.push(cand);
                        }
                    } else {
                        exterior_sandbox_facts.append(&mut recalled_facts);
                    }
                }
                Boundary2DRegion::SelfRef => {}
            }
        }
        let is_known_unknown_exterior = !exterior_sandbox_facts.is_empty();

        // Persona grounding operates orthogonally to factual `Liminal` gating and is suppressed
        // only on Layer 1 hard-block Guard repulsion or foreign-OOD token gating.
        let recalled_personas = if hard_guard_block || decision_2d.foreign_ood_tokens {
            Vec::new()
        } else {
            persona_scores
                .into_iter()
                .filter(|(_, s)| *s >= policy.min_coherence * 0.75)
                .take(policy.top_k_personas)
                .collect()
        };

        Ok(DynamicAttractorRecall {
            lacuna_state: final_lacuna_state,
            top1_fact_score: signal.boundary_metrics.top1_fact_score,
            top2_fact_score: signal.boundary_metrics.top2_fact_score,
            top1_persona_score: signal.boundary_metrics.top1_persona_score,
            min_local_lacuna_depth: directive.min_local_lacuna_depth,
            guard_conflict_margin: signal.boundary_metrics.guard_conflict_margin,
            action_decision: final_action_decision,
            gateway_frame: signal.gateway_frame,
            telemetry: signal.telemetry,
            recalled_facts,
            recalled_personas,
            top_guard,
            lacuna_partition,
            exterior_sandbox_facts,
            is_known_unknown_exterior,
        })
    }
}
