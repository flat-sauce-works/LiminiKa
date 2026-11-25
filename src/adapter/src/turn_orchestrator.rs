// name: src/adapter/src/turn_orchestrator.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Central `liminal` turn-boundary gate ([`TurnOrchestrator`]), abstract recall source
//! ([`AttractorRecallSource`]), and sealed [`TurnSteeringPlan`].
//!
//! All per-turn recall, snapshot composition, KV-cache stepping, cognitive circuit binding,
//! `LacunaPartition8` separation ($S \cap O$ vs. $O \setminus S$ vs. $\bot$), and core-layer
//! synchronization pass through [`TurnOrchestrator::prepare_turn`] as a single atomic boundary.

use crate::corpus_recall::{
    build_exterior_sandbox_anchors_per_layer_from_projection,
    build_fact_anchors_per_layer_from_projection, build_guard_anchors_per_layer_from_projection,
    build_persona_anchors_per_layer_from_projection, context_fingerprint_from_projection,
    join_exterior_context_from_projection, join_fact_context_from_projection,
    join_persona_context_from_projection, AttractorRecallProjection, RecalledAnchorRef,
};
use crate::{
    AttractorRole, CompiledTokenCorpus, DynamicAttractorRecall, LacunaBoundaryState,
    ModelSteeringAdapter, ReflectionCritique, ReflectionGateInput, ReflectionTriggerKind,
    StratumProbeTelemetry, TextAttractorEntry, TrajectoryManifoldSnapshot,
};
use liminika_core::abi::lmnk_q7_t;
use liminika_core::liminal::{LacunaPartition8, LmnkResult, MAX_ATTENTION_HEADS};
use liminika_core::runtime::{
    CognitiveCircuitThresholdPolicy, EpistemicHorizonReading, GatewayFrame, GuardPhaseAnchor,
    ObjectiveFact, RamPhaseKvGopCache, RamPhaseKvStepOutcome, SubjectivePersona,
};
use liminika_core::stigmergy::{
    CognitiveCircuitActivation, CognitiveCircuitBankOps, CognitiveTrajectoryLedger,
    OverlapDiscoveryReport, StigmergicActionDecision, StigmergicFragmentEnsembleReport,
};

/// Owned, lifetime-free attractor anchor record returned by an [`AttractorRecallSource`].
///
/// Drops full-resolution retrieval signatures (`residual_sig`) and retains only the compact
/// per-layer Q7 phase geometry, text, and score required by [`TurnOrchestrator`].
#[derive(Debug, Clone)]
pub struct RecalledAnchorRecord {
    /// Stable anchor identifier (`1..`).
    pub anchor_id: u32,
    /// Origin role (`Fact`, `Persona`, or `Guard`).
    pub role: AttractorRole,
    /// Natural-language source text corresponding to this anchor.
    pub text: String,
    /// Per-layer 64-head Q7 phase signatures (`num_layers` entries).
    pub per_layer_q7: Vec<[lmnk_q7_t; MAX_ATTENTION_HEADS]>,
    /// Base gravitational pull energy for gateway composition.
    pub energy: f32,
    /// Query-relative resonance score observed during recall.
    pub score: f32,
}

impl RecalledAnchorRecord {
    /// Constructs an owned [`RecalledAnchorRecord`] from a borrowed [`TextAttractorEntry`].
    #[must_use]
    pub fn from_entry(entry: &TextAttractorEntry, score: f32) -> Self {
        Self {
            anchor_id: entry.anchor_id,
            role: entry.role,
            text: entry.text.clone(),
            per_layer_q7: entry.per_layer_q7.clone(),
            energy: entry.energy,
            score,
        }
    }

    #[inline]
    fn as_ref_view(&self) -> RecalledAnchorRef<'_> {
        RecalledAnchorRef {
            anchor_id: self.anchor_id,
            text: self.text.as_str(),
            per_layer_q7: &self.per_layer_q7,
            energy: self.energy,
            score: self.score,
        }
    }
}

/// Owned, lifetime-free recall outcome returned by an abstract [`AttractorRecallSource`].
#[derive(Debug, Clone)]
pub struct OwnedAttractorRecall {
    /// Classified boundary state (`InsideBasin`, `LiminalBoundary`, or `LiminalExterior`).
    pub lacuna_state: LacunaBoundaryState,
    /// Highest geometric resonance score among Fact attractors.
    pub top1_fact_score: f32,
    /// Second-highest geometric resonance score among Fact attractors.
    pub top2_fact_score: f32,
    /// Highest geometric resonance score among Persona attractors.
    pub top1_persona_score: f32,
    /// Smallest per-attractor local `Liminal` depth observed across all attractors.
    pub min_local_lacuna_depth: f32,
    /// Highest Guard resonance minus highest Fact/Persona resonance (`> 0` = guard-dominant).
    pub guard_conflict_margin: f32,
    /// Substrate regime selected by the upper boundary stratum.
    pub action_decision: StigmergicActionDecision,
    /// Authoritative 128-byte core [`GatewayFrame`] synthesized for this query.
    pub gateway_frame: GatewayFrame,
    /// Cross-channel probe telemetry and core causal provenance report.
    pub telemetry: StratumProbeTelemetry,
    /// Top-K recalled $S \cap O$ (`SelfRef`) Fact anchors.
    pub recalled_facts: Vec<RecalledAnchorRecord>,
    /// Top-K recalled Persona anchors (suppressed only on Guard repulsion).
    pub recalled_personas: Vec<RecalledAnchorRecord>,
    /// Strongest observed Guard attractor, if any (`None` keeps Guard repulsion zero-aperture).
    pub top_guard: Option<RecalledAnchorRecord>,
    /// Branchless 4-region lacuna partition (`S & O`, `S \ O`, `O \ S`, Unobserved) for this query.
    pub lacuna_partition: LacunaPartition8,
    /// Isolated $O \setminus S$ (`Exterior`) known-unknown sandbox pointers.
    pub exterior_sandbox_facts: Vec<RecalledAnchorRecord>,
    /// True when a validated $O \setminus S$ exterior pointer was staged via the $S \cap O$ bridge.
    pub is_known_unknown_exterior: bool,
}

impl From<&DynamicAttractorRecall<'_>> for OwnedAttractorRecall {
    fn from(recall: &DynamicAttractorRecall<'_>) -> Self {
        recall.to_owned_recall()
    }
}

impl AttractorRecallProjection for OwnedAttractorRecall {
    #[inline]
    fn lacuna_state(&self) -> LacunaBoundaryState {
        self.lacuna_state
    }
    #[inline]
    fn top1_fact_score(&self) -> f32 {
        self.top1_fact_score
    }
    #[inline]
    fn top2_fact_score(&self) -> f32 {
        self.top2_fact_score
    }
    #[inline]
    fn top1_persona_score(&self) -> f32 {
        self.top1_persona_score
    }
    #[inline]
    fn guard_conflict_margin(&self) -> f32 {
        self.guard_conflict_margin
    }
    #[inline]
    fn action_decision(&self) -> StigmergicActionDecision {
        self.action_decision
    }
    #[inline]
    fn gateway_frame(&self) -> &GatewayFrame {
        &self.gateway_frame
    }
    #[inline]
    fn telemetry(&self) -> &StratumProbeTelemetry {
        &self.telemetry
    }
    #[inline]
    fn lacuna_partition(&self) -> LacunaPartition8 {
        self.lacuna_partition
    }
    #[inline]
    fn is_known_unknown_exterior(&self) -> bool {
        self.is_known_unknown_exterior
    }
    #[inline]
    fn recalled_fact_len(&self) -> usize {
        self.recalled_facts.len()
    }
    #[inline]
    fn recalled_fact_at(&self, idx: usize) -> Option<RecalledAnchorRef<'_>> {
        self.recalled_facts
            .get(idx)
            .map(RecalledAnchorRecord::as_ref_view)
    }
    #[inline]
    fn recalled_persona_len(&self) -> usize {
        self.recalled_personas.len()
    }
    #[inline]
    fn recalled_persona_at(&self, idx: usize) -> Option<RecalledAnchorRef<'_>> {
        self.recalled_personas
            .get(idx)
            .map(RecalledAnchorRecord::as_ref_view)
    }
    #[inline]
    fn exterior_sandbox_len(&self) -> usize {
        self.exterior_sandbox_facts.len()
    }
    #[inline]
    fn exterior_sandbox_at(&self, idx: usize) -> Option<RecalledAnchorRef<'_>> {
        self.exterior_sandbox_facts
            .get(idx)
            .map(RecalledAnchorRecord::as_ref_view)
    }
    #[inline]
    fn top_guard_ref(&self) -> Option<RecalledAnchorRef<'_>> {
        self.top_guard
            .as_ref()
            .map(RecalledAnchorRecord::as_ref_view)
    }
}

/// Abstract recall observation source supplying an [`OwnedAttractorRecall`] for a query token
/// sequence capture.
///
/// Decouples [`TurnOrchestrator`] from concrete corpus storage (in-memory [`CompiledTokenCorpus`],
/// paged `.lmnkc` containers, or mock unit-test sources).
pub trait AttractorRecallSource {
    /// Evaluates `query_token_qs` (`query_seq_len` tokens) against this source under `adapter`'s
    /// active geometry and optional `prior_circuit`.
    ///
    /// # Errors
    /// Returns an error if query shapes disagree with `adapter` or recall fails.
    fn recall_query(
        &self,
        adapter: &ModelSteeringAdapter,
        query_token_qs: &[Vec<f32>],
        query_seq_len: usize,
        prior_circuit: Option<CognitiveTrajectoryLedger>,
    ) -> Result<OwnedAttractorRecall, String>;
}

impl AttractorRecallSource for CompiledTokenCorpus {
    fn recall_query(
        &self,
        adapter: &ModelSteeringAdapter,
        query_token_qs: &[Vec<f32>],
        query_seq_len: usize,
        prior_circuit: Option<CognitiveTrajectoryLedger>,
    ) -> Result<OwnedAttractorRecall, String> {
        self.recall_token_sequence_with_circuit(
            adapter,
            query_token_qs,
            query_seq_len,
            prior_circuit,
        )
        .map(|recall| recall.to_owned_recall())
    }
}

/// Per-turn inputs passed into [`TurnOrchestrator::prepare_turn`].
pub struct TurnPrepareInput<
    'a,
    P: CognitiveCircuitThresholdPolicy + ?Sized,
    B: CognitiveCircuitBankOps,
> {
    /// Per-layer flattened query token activations `[seq_len * q_rot_len]`.
    pub query_token_qs: &'a [Vec<f32>],
    /// Number of tokens in the raw query sequence (`seq_len`).
    pub query_seq_len: usize,
    /// Monotonic 1-based turn index for KV-cache recency tracking.
    pub turn_index: u32,
    /// Threshold policy governing KV-cache circuit blending and resonance gating.
    pub kv_policy: &'a P,
    /// Optional override for minimum KV-cache resonance threshold (`Q0.7`).
    pub kv_min_resonance_override: Option<lmnk_q7_t>,
    /// Optional persistent persona cognitive circuit bank seeded into recall and KV stepping.
    pub persona_bank: Option<&'a B>,
    /// Optional per-turn RoPE layer steering mask override (`None` inherits `adapter.geometry()`).
    pub layer_steer_mask_override: Option<u64>,
}

/// Sealed per-turn steering plan issued exclusively by [`TurnOrchestrator::prepare_turn`].
///
/// Enforces strict `liminal` topological separation between `SelfRef` ($S \cap O$),
/// `Exterior` ($O \setminus S$), and `Persona` channels via private fields and read-only
/// accessors so downstream callers cannot accidentally conflate external sandbox pointers into
/// self-referential steering slices.
#[derive(Debug, Clone)]
pub struct TurnSteeringPlan {
    query_snapshot: TrajectoryManifoldSnapshot,
    min_local_lacuna_depth: f32,
    lacuna_partition: LacunaPartition8,
    telemetry: StratumProbeTelemetry,
    horizon: EpistemicHorizonReading,
    overlap_report: OverlapDiscoveryReport,
    ensemble_report: Option<StigmergicFragmentEnsembleReport>,
    kv_outcome: RamPhaseKvStepOutcome,
    is_known_unknown_exterior: bool,
    is_guard_trap: bool,
    context_fingerprint: u64,
    layer_steer_mask_override: Option<u64>,
    self_ref_context: String,
    exterior_context: String,
    persona_context: String,
    self_ref_anchors_per_layer: Vec<Vec<ObjectiveFact<[lmnk_q7_t; MAX_ATTENTION_HEADS]>>>,
    exterior_sandbox_anchors_per_layer: Vec<Vec<ObjectiveFact<[lmnk_q7_t; MAX_ATTENTION_HEADS]>>>,
    persona_anchors_per_layer: Vec<Vec<SubjectivePersona<[lmnk_q7_t; MAX_ATTENTION_HEADS]>>>,
    guard_anchors_per_layer: Vec<Vec<GuardPhaseAnchor>>,
}

impl TurnSteeringPlan {
    /// Returns the owned [`TrajectoryManifoldSnapshot`] for downstream reflection critique.
    #[inline]
    #[must_use]
    pub const fn query_snapshot(&self) -> &TrajectoryManifoldSnapshot {
        &self.query_snapshot
    }

    /// Returns the classified [`LacunaBoundaryState`] for this turn.
    #[inline]
    #[must_use]
    pub const fn lacuna_state(&self) -> LacunaBoundaryState {
        self.query_snapshot.lacuna_state
    }

    /// Returns the branchless 4-region [`LacunaPartition8`] computed for this turn.
    #[inline]
    #[must_use]
    pub const fn lacuna_partition(&self) -> LacunaPartition8 {
        self.lacuna_partition
    }

    /// Returns the selected substrate [`StigmergicActionDecision`].
    #[inline]
    #[must_use]
    pub const fn action_decision(&self) -> StigmergicActionDecision {
        self.query_snapshot.action_decision
    }

    /// Returns the authoritative 128-byte core [`GatewayFrame`].
    #[inline]
    #[must_use]
    pub const fn gateway_frame(&self) -> &GatewayFrame {
        &self.query_snapshot.gateway_frame
    }

    /// Returns the cross-channel [`StratumProbeTelemetry`] captured during recall.
    #[inline]
    #[must_use]
    pub const fn telemetry(&self) -> &StratumProbeTelemetry {
        &self.telemetry
    }

    /// Returns the core [`EpistemicHorizonReading`] synchronized onto the adapter.
    #[inline]
    #[must_use]
    pub const fn horizon(&self) -> &EpistemicHorizonReading {
        &self.horizon
    }

    /// Returns the core [`OverlapDiscoveryReport`] synchronized onto the adapter.
    #[inline]
    #[must_use]
    pub const fn overlap_report(&self) -> &OverlapDiscoveryReport {
        &self.overlap_report
    }

    /// Returns the optional micro-ensemble report when `>= 2` self-ref anchors were synthesized.
    #[inline]
    #[must_use]
    pub const fn ensemble_report(&self) -> Option<&StigmergicFragmentEnsembleReport> {
        self.ensemble_report.as_ref()
    }

    /// Returns the RAM phase KV-cache step outcome recorded for this turn.
    #[inline]
    #[must_use]
    pub const fn kv_outcome(&self) -> RamPhaseKvStepOutcome {
        self.kv_outcome
    }

    /// Returns the top-1 Fact resonance score.
    #[inline]
    #[must_use]
    pub const fn top1_fact_score(&self) -> f32 {
        self.query_snapshot.top1_fact_score
    }

    /// Returns the top-2 Fact resonance score.
    #[inline]
    #[must_use]
    pub const fn top2_fact_score(&self) -> f32 {
        self.query_snapshot.top2_fact_score
    }

    /// Returns the top-1 Persona resonance score.
    #[inline]
    #[must_use]
    pub const fn top1_persona_score(&self) -> f32 {
        self.query_snapshot.top1_persona_score
    }

    /// Returns the smallest local `Liminal` depth observed across corpus attractors.
    #[inline]
    #[must_use]
    pub const fn min_local_lacuna_depth(&self) -> f32 {
        self.min_local_lacuna_depth
    }

    /// Returns the Guard conflict margin (`max_guard_score - top1_fact_or_persona`).
    #[inline]
    #[must_use]
    pub const fn guard_conflict_margin(&self) -> f32 {
        self.query_snapshot.guard_conflict_margin
    }

    /// Returns `true` when an isolated $O \setminus S$ (`Exterior`) known-unknown pointer is staged.
    #[inline]
    #[must_use]
    pub const fn is_known_unknown_exterior(&self) -> bool {
        self.is_known_unknown_exterior
    }

    /// Returns `true` when Guard dominance or `RepulsionEscape` requires defensive steering.
    #[inline]
    #[must_use]
    pub const fn is_guard_trap(&self) -> bool {
        self.is_guard_trap
    }

    /// Returns `true` when the query lies strictly inside unobserved space ($\overline{S \cup O}$)
    /// without a staged $O \setminus S$ known-unknown pointer.
    #[inline]
    #[must_use]
    pub const fn is_unobserved_lacuna(&self) -> bool {
        matches!(
            self.query_snapshot.lacuna_state,
            LacunaBoundaryState::LiminalExterior
        ) && !self.is_known_unknown_exterior
    }

    /// Returns `true` when RoPE/Rank-1 steering is authorized for this turn (any Guard trap,
    /// `SelfRef` basin/boundary match, or staged $O \setminus S$ exterior pointer).
    #[inline]
    #[must_use]
    pub const fn steering_eligible(&self) -> bool {
        self.is_guard_trap || !self.is_unobserved_lacuna()
    }

    /// Returns the deterministic 64-bit FNV-1a context fingerprint used for KV-cache lookup.
    #[inline]
    #[must_use]
    pub const fn context_fingerprint(&self) -> u64 {
        self.context_fingerprint
    }

    /// Returns the ordered anchor IDs in the $S \cap O$ (`SelfRef`) channel.
    #[inline]
    #[must_use]
    pub fn self_ref_anchor_ids(&self) -> &[u32] {
        &self.query_snapshot.recalled_fact_anchors
    }

    /// Returns the concatenated text of recalled $S \cap O$ (`SelfRef`) anchors only.
    #[inline]
    #[must_use]
    pub fn self_ref_context(&self) -> &str {
        &self.self_ref_context
    }

    /// Returns the concatenated text of isolated $O \setminus S$ (`Exterior`) sandbox anchors only.
    #[inline]
    #[must_use]
    pub fn exterior_context(&self) -> &str {
        &self.exterior_context
    }

    /// Returns the concatenated text of recalled `Persona` anchors only.
    #[inline]
    #[must_use]
    pub fn persona_context(&self) -> &str {
        &self.persona_context
    }

    /// Returns per-layer `ObjectiveFact` anchors strictly from the $S \cap O$ (`SelfRef`) channel.
    #[inline]
    #[must_use]
    pub fn self_ref_anchors_per_layer(
        &self,
    ) -> &[Vec<ObjectiveFact<[lmnk_q7_t; MAX_ATTENTION_HEADS]>>] {
        &self.self_ref_anchors_per_layer
    }

    /// Returns per-layer `ObjectiveFact` anchors strictly from the isolated $O \setminus S$
    /// (`Exterior`) sandbox channel (damped energy, empty unless `is_known_unknown_exterior`).
    #[inline]
    #[must_use]
    pub fn exterior_sandbox_anchors_per_layer(
        &self,
    ) -> &[Vec<ObjectiveFact<[lmnk_q7_t; MAX_ATTENTION_HEADS]>>] {
        &self.exterior_sandbox_anchors_per_layer
    }

    /// Returns per-layer `SubjectivePersona` anchors from the orthogonal persona channel.
    #[inline]
    #[must_use]
    pub fn persona_anchors_per_layer(
        &self,
    ) -> &[Vec<SubjectivePersona<[lmnk_q7_t; MAX_ATTENTION_HEADS]>>] {
        &self.persona_anchors_per_layer
    }

    /// Returns per-layer [`GuardPhaseAnchor`] poles carrying dynamic turn obstruction on Guard
    /// traps or positive `guard_conflict_margin` turns (empty slices on non-Guard turns).
    #[inline]
    #[must_use]
    pub fn guard_anchors_per_layer(&self) -> &[Vec<GuardPhaseAnchor>] {
        &self.guard_anchors_per_layer
    }

    /// Returns `true` if at least one $S \cap O$ (`SelfRef`) anchor is present.
    #[inline]
    #[must_use]
    pub fn has_self_ref_anchors(&self) -> bool {
        !self.query_snapshot.recalled_fact_anchors.is_empty()
    }

    /// Returns `true` if at least one $O \setminus S$ (`Exterior`) sandbox anchor is present.
    #[inline]
    #[must_use]
    pub fn has_exterior_sandbox_anchors(&self) -> bool {
        self.exterior_sandbox_anchors_per_layer
            .iter()
            .any(|layer| !layer.is_empty())
    }

    /// Returns the optional per-turn RoPE layer steering mask override (default `None`).
    #[inline]
    #[must_use]
    pub const fn layer_steer_mask_override(&self) -> Option<u64> {
        self.layer_steer_mask_override
    }

    /// Resolves the effective RoPE steering layer bitmask for this turn (`override` or adapter mask).
    #[inline]
    #[must_use]
    pub const fn effective_layer_steer_mask(&self, adapter: &ModelSteeringAdapter) -> u64 {
        match self.layer_steer_mask_override {
            Some(mask) => mask,
            None => adapter.geometry().layer_rope_mask,
        }
    }

    /// Synthesizes a [`ReflectionGateInput`] combining this turn's sealed `liminal` boundary
    /// partition with the live Pass-1 exit entropy (`exit_entropy_h_tilde`) observed after decode.
    #[must_use]
    pub fn reflection_gate_input(&self, exit_entropy_h_tilde: f32) -> ReflectionGateInput {
        ReflectionGateInput {
            boundary_sensitivity_q7: self.lacuna_partition.boundary_sensitivity_q7,
            has_undigested_known_unknown: self.is_known_unknown_exterior,
            exit_entropy_h_tilde,
            guard_conflict_margin: self.guard_conflict_margin(),
            is_unobserved_lacuna: self.is_unobserved_lacuna(),
            ..ReflectionGateInput::default()
        }
    }
}

/// Adapter-level trait abstracting per-turn GOP KV-cache stepping over any
/// [`RamPhaseKvGopCache`] (or custom mock cache) without modifying `liminika-core`.
pub trait TurnKvCacheStepping {
    /// Executes one GOP KV-cache step for the turn's `frame` and `circuit`.
    ///
    /// # Errors
    /// Propagates invalid-argument or keyframe integrity errors from the backing KV cache.
    #[allow(clippy::too_many_arguments)]
    fn step_turn_kv<Pol: CognitiveCircuitThresholdPolicy + ?Sized>(
        &mut self,
        context_hash: u64,
        token_index: u32,
        frame: &GatewayFrame,
        circuit: &mut CognitiveTrajectoryLedger,
        activation: CognitiveCircuitActivation,
        policy: &Pol,
        override_min_resonance_q7: Option<lmnk_q7_t>,
        epoch: u32,
    ) -> LmnkResult<RamPhaseKvStepOutcome>;
}

impl<const N: usize, const P: usize> TurnKvCacheStepping for RamPhaseKvGopCache<N, P> {
    #[inline]
    fn step_turn_kv<Pol: CognitiveCircuitThresholdPolicy + ?Sized>(
        &mut self,
        context_hash: u64,
        token_index: u32,
        frame: &GatewayFrame,
        circuit: &mut CognitiveTrajectoryLedger,
        activation: CognitiveCircuitActivation,
        policy: &Pol,
        override_min_resonance_q7: Option<lmnk_q7_t>,
        epoch: u32,
    ) -> LmnkResult<RamPhaseKvStepOutcome> {
        self.step_with_gateway_and_override(
            context_hash,
            token_index,
            frame,
            circuit,
            activation,
            policy,
            override_min_resonance_q7,
            epoch,
        )
    }
}

/// Stateless central `liminal` turn-boundary gate coordinating recall, KV stepping, cognitive
/// circuit synchronization, core-layer adaptation, and channel separation.
#[derive(Debug, Default, Clone, Copy)]
pub struct TurnOrchestrator;

impl TurnOrchestrator {
    /// Executes the complete pre-generation turn-boundary pipeline atomically and emits a sealed
    /// [`TurnSteeringPlan`] with strictly separated $S \cap O$, $O \setminus S$, and `Persona`
    /// channels.
    ///
    /// # Errors
    /// Returns an error if `source.recall_query` or `kv_cache.step_turn_kv` fails.
    #[must_use = "turn steering plan carries recalled anchors, guard repulsion poles, and telemetry"]
    pub fn prepare_turn<S, K, P, B>(
        source: &S,
        adapter: &mut ModelSteeringAdapter,
        kv_cache: &mut K,
        input: &TurnPrepareInput<'_, P, B>,
    ) -> Result<TurnSteeringPlan, String>
    where
        S: AttractorRecallSource + ?Sized,
        K: TurnKvCacheStepping + ?Sized,
        P: CognitiveCircuitThresholdPolicy + ?Sized,
        B: CognitiveCircuitBankOps,
    {
        let prior_circuit = input
            .persona_bank
            .map(CognitiveCircuitBankOps::to_consolidated_ledger);
        let recall = source.recall_query(
            adapter,
            input.query_token_qs,
            input.query_seq_len,
            prior_circuit,
        )?;

        let num_layers = adapter.geometry().num_layers;
        let self_ref_context = join_fact_context_from_projection(&recall);
        let persona_context = join_persona_context_from_projection(&recall);
        let exterior_context = join_exterior_context_from_projection(&recall);
        let self_ref_anchors_per_layer =
            build_fact_anchors_per_layer_from_projection(&recall, num_layers);
        let exterior_sandbox_anchors_per_layer =
            build_exterior_sandbox_anchors_per_layer_from_projection(&recall, num_layers);
        let persona_anchors_per_layer =
            build_persona_anchors_per_layer_from_projection(&recall, num_layers);

        let mut query_snapshot = TrajectoryManifoldSnapshot::from_owned_recall(&recall);
        if let Some(bank) = input.persona_bank {
            let cons = bank.to_consolidated_ledger();
            query_snapshot.cognitive_trajectory = cons.compose_persona_circuit(
                &query_snapshot.cognitive_trajectory,
                input.kv_policy.kv_circuit_blend_q7(&recall.gateway_frame),
            );
        }

        let context_fingerprint = context_fingerprint_from_projection(&recall);
        let kv_outcome = kv_cache
            .step_turn_kv(
                context_fingerprint,
                input.query_seq_len as u32,
                &recall.gateway_frame,
                &mut query_snapshot.cognitive_trajectory,
                recall.telemetry.layer_activation,
                input.kv_policy,
                input.kv_min_resonance_override,
                input.turn_index,
            )
            .map_err(|e| format!("ram_kv_cache error: {e}"))?;

        if input.persona_bank.is_some() {
            adapter.set_active_circuit(query_snapshot.cognitive_trajectory);
        } else {
            adapter.clear_active_circuit();
        }

        let (horizon, overlap_report, ensemble_report) =
            adapter.sync_core_layers_from_recall(&recall, None);

        let is_guard_trap = recall.guard_conflict_margin > 0.05
            || matches!(
                recall.action_decision,
                StigmergicActionDecision::RepulsionEscape
            );
        let guard_anchors_per_layer =
            build_guard_anchors_per_layer_from_projection(&recall, num_layers, is_guard_trap);

        Ok(TurnSteeringPlan {
            query_snapshot,
            min_local_lacuna_depth: recall.min_local_lacuna_depth,
            lacuna_partition: recall.lacuna_partition,
            telemetry: recall.telemetry,
            horizon,
            overlap_report,
            ensemble_report,
            kv_outcome,
            is_known_unknown_exterior: recall.is_known_unknown_exterior,
            is_guard_trap,
            context_fingerprint,
            layer_steer_mask_override: input.layer_steer_mask_override,
            self_ref_context,
            exterior_context,
            persona_context,
            self_ref_anchors_per_layer,
            exterior_sandbox_anchors_per_layer,
            persona_anchors_per_layer,
            guard_anchors_per_layer,
        })
    }

    /// Commits post-turn cognitive circuit plasticity from `plan` (and optional reflection
    /// `critique`) into `bank` and synchronizes the consolidated ledger onto `adapter`.
    pub fn commit_turn_circuit<B: CognitiveCircuitBankOps>(
        adapter: &mut ModelSteeringAdapter,
        plan: &TurnSteeringPlan,
        bank: &mut B,
        critique: Option<&ReflectionCritique>,
    ) {
        if let Some(step) = plan.telemetry().cognitive_trajectory.latest_step() {
            bank.push_step_hierarchical(step);
        }
        if let Some(crit) = critique {
            let converged = matches!(
                crit.trigger,
                ReflectionTriggerKind::Converged
                    | ReflectionTriggerKind::ConsistentLacunaAbstention
            );
            bank.reinforce_bank(&crit.layer_activation, converged);
        }
        adapter.set_active_circuit(bank.to_consolidated_ledger());
    }
}
