// name: src/adapter/src/corpus_recall.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dynamic natural-language attractor corpus indexing, Liminal boundary gates,
//! and persistent token-sequence recall ([`TextAttractorCorpus`]).

use super::*;

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

/// Geometric classification of a query's position relative to the compiled attractor manifold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoidBoundaryState {
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
    pub per_layer_q7: Vec<[gcso_q7_t; MAX_ATTENTION_HEADS]>,
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
    pub void_state: VoidBoundaryState,
    /// Highest geometric resonance score among Fact attractors.
    pub top1_fact_score: f32,
    /// Second-highest geometric resonance score among Fact attractors.
    pub top2_fact_score: f32,
    /// Highest geometric resonance score among Persona attractors.
    pub top1_persona_score: f32,
    /// Smallest per-attractor local `Liminal` depth observed across all attractors (`0.0` = deep in a basin).
    pub min_local_void_depth: f32,
    /// Highest Guard resonance minus highest Fact/Persona resonance (`> 0` indicates guard-dominant conflict).
    pub guard_conflict_margin: f32,
    /// `ActionHub` regime selected by the upper boundary stratum (`PassiveSidecar`, `InertialLayerSkip`,
    /// `ExpressPhaseSteer`, `SubHeadRoute`, `FullOrchestration`, `CrystallizeMacroBasin`, or `RepulsionEscape`).
    pub action_decision: StigmergicActionDecision,
    /// Authoritative 128-byte core [`GatewayFrame`] synthesized and arbitrated by the pipeline.
    pub gateway_frame: GatewayFrame,
    /// Cross-channel probe telemetry and core [`CausalProvenanceReport`].
    pub telemetry: StratumProbeTelemetry,
    /// Top-K recalled Fact entries (empty when `void_state == Liminal`).
    pub recalled_facts: Vec<(&'a TextAttractorEntry, f32)>,
    /// Top-K recalled Persona entries (suppressed only on Guard repulsion).
    pub recalled_personas: Vec<(&'a TextAttractorEntry, f32)>,
}

impl<'a> DynamicAttractorRecall<'a> {
    /// Materializes per-layer `ObjectiveFact<[i8; 64]>` anchors for
    /// [`ResonantSignalSlice`](liminika_core::traits::gateway::ResonantSignalSlice).
    #[must_use]
    pub fn build_fact_anchors_per_layer(
        &self,
        num_layers: usize,
    ) -> Vec<Vec<ObjectiveFact<[gcso_q7_t; MAX_ATTENTION_HEADS]>>> {
        (0..num_layers)
            .map(|l| {
                self.recalled_facts
                    .iter()
                    .filter_map(|(entry, score)| {
                        entry.per_layer_q7.get(l).map(|&arr| {
                            let scaled_e = (entry.energy * score.clamp(0.25, 1.25)).max(0.25);
                            ObjectiveFact::new(arr, scaled_e, 0, entry.anchor_id)
                        })
                    })
                    .collect()
            })
            .collect()
    }

    /// Materializes per-layer `SubjectivePersona<[i8; 64]>` anchors for
    /// [`ResonantSignalSlice`](liminika_core::traits::gateway::ResonantSignalSlice).
    #[must_use]
    pub fn build_persona_anchors_per_layer(
        &self,
        num_layers: usize,
    ) -> Vec<Vec<SubjectivePersona<[gcso_q7_t; MAX_ATTENTION_HEADS]>>> {
        (0..num_layers)
            .map(|l| {
                self.recalled_personas
                    .iter()
                    .filter_map(|(entry, score)| {
                        entry.per_layer_q7.get(l).map(|&arr| {
                            let scaled_e = (entry.energy * score.clamp(0.25, 1.25)).max(0.25);
                            SubjectivePersona::new(arr, scaled_e, 0, entry.anchor_id)
                        })
                    })
                    .collect()
            })
            .collect()
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

    /// Computes a deterministic 64-bit FNV-1a context fingerprint across recalled attractors,
    /// returning `0` when the query lies in ungrounded `Liminal` or recalled text is empty.
    #[must_use]
    pub fn context_fingerprint_u64(&self) -> u64 {
        if matches!(self.void_state, VoidBoundaryState::LiminalExterior) {
            return 0;
        }
        let ctx = self.recalled_context_text();
        if ctx.is_empty() {
            return 0;
        }
        ctx.as_bytes()
            .iter()
            .fold(0xCBF2_9CE4_8422_2325_u64, |h, &b| {
                (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3)
            })
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
    /// Guard conflict margin (`max_guard_score - top1_fact_score`) above which fact injection is suppressed.
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

/// Geometric boundary measurements evaluated by a [`RecallBoundaryGate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecallBoundaryMetrics {
    /// Highest geometric resonance score among Fact attractors.
    pub top1_fact_score: f32,
    /// Second-highest geometric resonance score among Fact attractors.
    pub top2_fact_score: f32,
    /// Highest geometric resonance score among Persona attractors.
    pub top1_persona_score: f32,
    /// Highest geometric resonance score among Guard attractors.
    pub max_guard_score: f32,
    /// `max_guard_score - max(top1_fact_score, top1_persona_score)`.
    pub guard_conflict_margin: f32,
}

/// Decision produced by a [`RecallBoundaryGate`] for a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecallGateDecision {
    /// Classified boundary state (`InsideBasin`, `LiminalBoundary`, or `Liminal`).
    pub void_state: VoidBoundaryState,
    /// Whether forced Fact/Persona attractor injection should be suppressed (e.g., `Liminal` or Guard dominance).
    pub suppress_fact_injection: bool,
}

/// Trait abstracting `Liminal` boundary classification and Guard-conflict suppression policies.
pub trait RecallBoundaryGate {
    /// Evaluates boundary metrics under the active [`RecallPolicy`].
    fn evaluate(
        &self,
        metrics: &RecallBoundaryMetrics,
        policy: &RecallPolicy,
    ) -> RecallGateDecision;
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

/// Per-attractor measurement returned by [`AttractorVoidPerspective::measure_perspective`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerspectiveVoidReading {
    /// Perspective-calibrated affinity score as viewed from this attractor's angle.
    pub perspective_affinity: f32,
    /// Non-negative local `Liminal` gap seen from this attractor (`0.0` = well inside basin).
    pub local_void_depth: f32,
}

/// Trait computing perspective-dependent affinity and local `Liminal` void depth per attractor.
///
/// Rather than judging every attractor from a single isotropic origin, each attractor can have a
/// distinct local density and boundary geometry so central hubs do not eclipse peripheral basins.
pub trait AttractorVoidPerspective {
    /// Measures affinity and local `Liminal` depth from `ctx.entry`'s vantage point.
    fn measure_perspective(
        &self,
        ctx: &AttractorPerspectiveContext<'_>,
        min_coherence: f32,
    ) -> PerspectiveVoidReading;
}

/// Default [`AttractorVoidPerspective`] implementing bidirectional CSLS hubness normalization and
/// asymmetric Z-score scaling across each attractor's local neighborhood.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AsymmetricDominoVoidPerspective {
    /// Number of nearest corpus neighbors defining each attractor's local baseline.
    pub neighbor_k: usize,
    /// Damping weight applied to centered attractor hubness (`neighbor_mean - global_mean`).
    pub hubness_damping: f32,
    /// Weight applied to the query/attractor bidirectional Z-score modulated by positive raw cosine.
    pub perspective_z_gain: f32,
}

impl Default for AsymmetricDominoVoidPerspective {
    fn default() -> Self {
        Self {
            neighbor_k: 4,
            hubness_damping: 0.50,
            perspective_z_gain: 0.025,
        }
    }
}

impl AttractorVoidPerspective for AsymmetricDominoVoidPerspective {
    fn measure_perspective(
        &self,
        ctx: &AttractorPerspectiveContext<'_>,
        min_coherence: f32,
    ) -> PerspectiveVoidReading {
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
        let local_void_depth = (min_coherence - perspective_affinity).max(0.0);
        PerspectiveVoidReading {
            perspective_affinity,
            local_void_depth,
        }
    }
}

/// Default [`RecallBoundaryGate`] enforcing both `Liminal` coherence floor and Guard dominance suppression.
#[derive(Debug, Default, Clone, Copy)]
pub struct CoherenceGuardGate;

impl RecallBoundaryGate for CoherenceGuardGate {
    fn evaluate(
        &self,
        metrics: &RecallBoundaryMetrics,
        policy: &RecallPolicy,
    ) -> RecallGateDecision {
        if metrics.guard_conflict_margin > policy.guard_suppress_margin {
            return RecallGateDecision {
                void_state: VoidBoundaryState::LiminalExterior,
                suppress_fact_injection: true,
            };
        }
        let margin = metrics.top1_fact_score - metrics.top2_fact_score;
        let sharp_liminal_match = metrics.top1_fact_score >= policy.min_coherence * 0.65
            && margin >= policy.boundary_margin
            && metrics.top1_fact_score >= metrics.top2_fact_score.max(0.0) * 1.5;

        if metrics.top1_fact_score < policy.min_coherence && !sharp_liminal_match {
            RecallGateDecision {
                void_state: VoidBoundaryState::LiminalExterior,
                suppress_fact_injection: true,
            }
        } else if sharp_liminal_match && metrics.top1_fact_score < policy.min_coherence
            || margin < policy.boundary_margin
        {
            RecallGateDecision {
                void_state: VoidBoundaryState::LiminalBoundary,
                suppress_fact_injection: false,
            }
        } else {
            RecallGateDecision {
                void_state: VoidBoundaryState::InsideBasin,
                suppress_fact_injection: false,
            }
        }
    }
}

/// Model-agnostic Cold-Path corpus that compiles natural-language logs, facts, personas, and
/// contrastive guards into a dynamic attractor manifold with explicit `Liminal` boundary detection.
#[derive(Debug, Default, Clone)]
pub struct TextAttractorCorpus {
    entries: Vec<TextAttractorEntry>,
    next_anchor_id: u32,
    common_mode_basis: Vec<Vec<f32>>,
}

impl TextAttractorCorpus {
    /// Creates an empty [`TextAttractorCorpus`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_anchor_id: 1,
            common_mode_basis: Vec::new(),
        }
    }

    /// Returns all stored attractor entries.
    #[must_use]
    pub fn entries(&self) -> &[TextAttractorEntry] {
        &self.entries
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

    pub(crate) fn project_and_normalize(v: &[f32], basis: &[Vec<f32>]) -> Vec<f32> {
        let mut out = v.to_vec();
        if !basis.is_empty() {
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
            Self::l2_normalize_in_place(&mut out);
        }
        out
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

    /// Estimates up to `max_components` orthonormal common-mode directions from `sigs` via
    /// deflated power iteration so shared template/attention-sink directions can be projected out.
    pub(crate) fn fit_common_mode_subspace_from_sigs(
        sigs: &[Vec<f32>],
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
        let mut work: Vec<Vec<f32>> = sigs.to_vec();

        for _ in 0..rank_cap {
            let mut v = vec![0.0_f32; dim];
            for w in &work {
                for (dst, &src) in v.iter_mut().zip(w.iter()) {
                    *dst += src;
                }
            }
            for dir in &basis {
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
            Self::l2_normalize_in_place(&mut v);
            if v.iter().all(|&x| x == 0.0) {
                break;
            }
            for _ in 0..8 {
                let mut next = vec![0.0_f32; dim];
                for w in &work {
                    let dot: f64 = w
                        .iter()
                        .zip(v.iter())
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum();
                    let d = dot as f32;
                    for (dst, &src) in next.iter_mut().zip(w.iter()) {
                        *dst += d * src;
                    }
                }
                for dir in &basis {
                    let dot: f64 = next
                        .iter()
                        .zip(dir.iter())
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum();
                    let d = dot as f32;
                    for (dst, &c) in next.iter_mut().zip(dir.iter()) {
                        *dst -= d * c;
                    }
                }
                let norm_sq: f64 = next.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
                if norm_sq < 1e-12 {
                    break;
                }
                Self::l2_normalize_in_place(&mut next);
                v = next;
            }
            let norm_sq: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            if norm_sq < 0.5 {
                break;
            }
            for w in &mut work {
                let dot: f64 = w
                    .iter()
                    .zip(v.iter())
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                let d = dot as f32;
                for (dst, &c) in w.iter_mut().zip(v.iter()) {
                    *dst -= d * c;
                }
            }
            basis.push(v);
        }
        basis
    }

    /// Estimates up to `max_components` orthonormal common-mode directions from corpus signatures
    /// via deflated power iteration.
    pub fn fit_common_mode_subspace(&mut self, max_components: usize) {
        let sigs: Vec<Vec<f32>> = self
            .entries
            .iter()
            .map(|e| e.residual_sig.clone())
            .collect();
        self.common_mode_basis = Self::fit_common_mode_subspace_from_sigs(&sigs, max_components);
    }

    /// Estimates and stores the dominant unsupervised common-mode subspace (up to 3 components)
    /// so recall projects out shared prompt/template bias.
    pub fn fit_common_mode_projection(&mut self) {
        self.fit_common_mode_subspace(3);
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
            }
        }
    }

    /// Returns common-mode-projected normalized signatures across the corpus.
    fn projected_corpus_sigs(&self) -> Vec<Vec<f32>> {
        self.entries
            .iter()
            .map(|e| Self::project_and_normalize(&e.residual_sig, &self.common_mode_basis))
            .collect()
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
        let projected = self.projected_corpus_sigs();
        let mut sims = Vec::new();
        for i in 0..projected.len() {
            for j in (i + 1)..projected.len() {
                sims.push(f64::from(Self::cosine_f32(&projected[i], &projected[j])));
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
        if per_layer_q.len() != baseline_ref_per_layer.len() {
            return Err(format!(
                "layer count mismatch: per_layer_q={}, baseline={}",
                per_layer_q.len(),
                baseline_ref_per_layer.len()
            ));
        }
        let mut per_layer_q7 = Vec::with_capacity(per_layer_q.len());
        for (q, r) in per_layer_q.iter().zip(baseline_ref_per_layer.iter()) {
            let q7 = adapter.bake_layer_procrustes_q7(r, q)?;
            let mut arr64 = [0i8; MAX_ATTENTION_HEADS];
            let n = q7.len().min(MAX_ATTENTION_HEADS);
            arr64[..n].copy_from_slice(&q7[..n]);
            per_layer_q7.push(arr64);
        }
        let residual_sig = Self::build_steered_residual_sig(
            adapter.geometry(),
            per_layer_q,
            baseline_ref_per_layer,
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
        Ok(id)
    }

    /// Re-bakes or scores a query's per-layer Q activations against the corpus purely via
    /// steered-layer geometric cosine similarity (zero keyword heuristics), using the default
    /// [`CoherenceGuardGate`] to classify [`VoidBoundaryState`].
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
        self.recall_for_query_with_gate(
            adapter,
            query_per_layer_q,
            baseline_ref_per_layer,
            policy,
            &CoherenceGuardGate,
        )
    }

    /// Scores a query's per-layer Q activations against the corpus using the default
    /// [`AsymmetricDominoVoidPerspective`] and delegates boundary/guard gating to a
    /// caller-supplied [`RecallBoundaryGate`] trait object.
    ///
    /// # Errors
    ///
    /// Returns an error if query/baseline layer slices mismatch `adapter.geometry()`.
    pub fn recall_for_query_with_gate(
        &self,
        adapter: &ModelSteeringAdapter,
        query_per_layer_q: &[&[f32]],
        baseline_ref_per_layer: &[&[f32]],
        policy: &RecallPolicy,
        gate: &dyn RecallBoundaryGate,
    ) -> Result<DynamicAttractorRecall<'_>, String> {
        self.recall_for_query_with_perspective(
            adapter,
            query_per_layer_q,
            baseline_ref_per_layer,
            policy,
            gate,
            &AsymmetricDominoVoidPerspective::default(),
        )
    }

    /// Scores a query's per-layer Q activations against the corpus using caller-supplied
    /// [`AttractorVoidPerspective`] and [`RecallBoundaryGate`] trait objects inside the default
    /// 3-stratum Society-of-Mind pipeline.
    ///
    /// # Errors
    ///
    /// Returns an error if query/baseline layer slices mismatch `adapter.geometry()`.
    pub fn recall_for_query_with_perspective(
        &self,
        adapter: &ModelSteeringAdapter,
        query_per_layer_q: &[&[f32]],
        baseline_ref_per_layer: &[&[f32]],
        policy: &RecallPolicy,
        gate: &dyn RecallBoundaryGate,
        perspective: &dyn AttractorVoidPerspective,
    ) -> Result<DynamicAttractorRecall<'_>, String> {
        let mut pipeline = StratumPipeline::new();
        pipeline.push_stratum(Box::new(LocalFeatureWorkerStratum::default()));
        pipeline.push_stratum(Box::new(HeadAngleWorkerStratum::default()));
        pipeline.push_stratum(Box::new(LocalTokenResonanceStratum::default()));
        pipeline.push_stratum(Box::new(PerspectiveDominoStratum { perspective }));
        pipeline.push_stratum(Box::new(ActionHubBoundaryStratum { gate }));
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
    /// deposits) for [`liminika_core::action_hub::ActionHubEngine`] trail decomposition.
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
        let query_sig = Self::project_and_normalize(&raw_sig, &self.common_mode_basis);
        let entry_sigs = self.projected_corpus_sigs();
        let raw_cosines: Vec<f32> = entry_sigs
            .iter()
            .map(|sig| Self::cosine_f32(&query_sig, sig))
            .collect();
        let n_entries = entry_sigs.len();
        let (token_channel, q_to_d_scores, d_to_q_scores, token_deposits) = match local_observation
        {
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
                )
            }
            None => (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
        };
        let initial_signal = StratumSignal {
            candidate_scores: raw_cosines.clone(),
            local_token_scores: token_channel,
            q_to_d_scores,
            d_to_q_scores,
            token_deposits,
            linked_fact_indices: Vec::new(),
            decomposition_triggered: false,
            local_void_depths: raw_cosines
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
                void_state: VoidBoundaryState::LiminalExterior,
                suppress_fact_injection: true,
            },
            action_decision: StigmergicActionDecision::PassiveSidecar,
            gateway_frame: GatewayFrame::default(),
            swarm_report: StigmergicReconstitutionReport::default(),
            local_channel_preferred: false,
            telemetry: StratumProbeTelemetry {
                cognitive_trajectory: prior_circuit.unwrap_or_default(),
                ..StratumProbeTelemetry::default()
            },
        };
        let stratum_ctx = StratumContext {
            entries: &self.entries,
            raw_sig: &raw_sig,
            query_sig: &query_sig,
            entry_sigs: &entry_sigs,
            layer_chunk_dim: expected,
            num_q_heads: geom.num_q_heads,
            rotary_dim: geom.rotary_dim,
            policy,
        };
        let (signal, directive) = pipeline.evaluate(&stratum_ctx, initial_signal)?;

        let mut fact_scores: Vec<(&TextAttractorEntry, f32)> = Vec::new();
        let mut persona_scores: Vec<(&TextAttractorEntry, f32)> = Vec::new();
        for (entry, &score) in self.entries.iter().zip(signal.candidate_scores.iter()) {
            match entry.role {
                AttractorRole::Fact => fact_scores.push((entry, score)),
                AttractorRole::Persona => persona_scores.push((entry, score)),
                AttractorRole::Guard => {}
            }
        }
        fact_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        persona_scores.sort_by(|a, b| b.1.total_cmp(&a.1));

        let suppress_facts = directive.repulse_guard
            || directive.suppress_injection
            || signal.gate_decision.suppress_fact_injection;
        let has_grounded_counter_fact = directive.repulse_guard
            && signal.boundary_metrics.top1_fact_score >= policy.min_coherence * 0.65;
        let recalled_facts = if has_grounded_counter_fact {
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
            fact_scores
                .into_iter()
                .enumerate()
                .filter(|(rank, (_, s))| *rank == 0 || *s >= policy.min_coherence * 0.75)
                .map(|(_, pair)| pair)
                .take(policy.top_k_facts)
                .collect()
        };
        // Persona grounding operates orthogonally to factual `Liminal` gating and is suppressed only on Guard repulsion.
        let recalled_personas = if directive.repulse_guard {
            Vec::new()
        } else {
            persona_scores
                .into_iter()
                .filter(|(_, s)| *s >= policy.min_coherence * 0.75)
                .take(policy.top_k_personas)
                .collect()
        };

        Ok(DynamicAttractorRecall {
            void_state: signal.gate_decision.void_state,
            top1_fact_score: signal.boundary_metrics.top1_fact_score,
            top2_fact_score: signal.boundary_metrics.top2_fact_score,
            top1_persona_score: signal.boundary_metrics.top1_persona_score,
            min_local_void_depth: directive.min_local_void_depth,
            guard_conflict_margin: signal.boundary_metrics.guard_conflict_margin,
            action_decision: signal.action_decision,
            gateway_frame: signal.gateway_frame,
            telemetry: signal.telemetry,
            recalled_facts,
            recalled_personas,
        })
    }
}
