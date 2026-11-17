// name: src/core/src/traits/gateway/epistemic.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Branchless epistemic horizon separation (`KnownCoherent` vs. `InternalConflict` vs. `LiminalVoid`),
//! signed relative attractor coupling ($W_{ij} \in [-127, +127]$), autonomous dynamic
//! [`LiminalApertureField`] horizon shifting, and `#![no_std]` self-introspecting Mermaid topology
//! telemetry over [`ManifoldGatewaySignal`].
//!
//! Rather than hard-coding static `Fact`/`Guard` roles or branch-heavy conditionals, every
//! attractor participates in a signed relational graph from the active perspective $i$:
//! - $W_{ij} > 0$ projects node $j$ onto the attractive basin axis;
//! - $W_{ij} < 0$ projects node $j$ onto the opposing/repulsive axis;
//! - When total projection energy onto the local memory cut falls below `horizon_floor`, the
//!   manifold enters [`EpistemicRegime::LiminalVoid`] (*Limina* to the unknowable) and scales
//!   output steering energy cleanly to `0.0` without branching.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use super::*;
use crate::crystal::DissipativeOverflowReport;
use crate::traits::branchless::{LiminalVoid, branchless_q7_inhibit_words_64};
use crate::traits::primitives::{branchless_sincos_q7, swar_u8_mean_round_x8};
use crate::traits::telemetry::{DEFAULT_STIGMERGIC_QUORUM_THRESHOLD, NUM_STIGMERGIC_GRANULES};

/// Epistemic regime classification separating internal memory conflict from out-of-memory unknowability.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicRegime {
    /// Query falls inside the carved memory horizon with coherent dominant alignment ($H^0$).
    KnownCoherent = 0,
    /// Query falls inside the carved memory horizon with competing / opposing attractors ($H^1 \neq 0$).
    InternalConflict = 1,
    /// Query lies outside the carved memory horizon (*Limina* between cognition and the unknowable).
    LiminalVoid = 2,
}

impl EpistemicRegime {
    /// Decodes a compact `u8` regime code (`0`, `1`, `2`) into [`EpistemicRegime`].
    #[inline(always)]
    #[must_use]
    pub const fn from_code(code: u8) -> Self {
        match code {
            1 => Self::InternalConflict,
            2 => Self::LiminalVoid,
            _ => Self::KnownCoherent,
        }
    }
}

/// Branchless epistemic horizon measurement over a finite local memory manifold (`16`-byte aligned).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EpistemicHorizonReading {
    /// Summed positive projection energy onto the local memory slice.
    pub total_memory_support: f32,
    /// Peak positive coherence observed across attractive / uncoupled memory nodes.
    pub peak_positive_affinity: f32,
    /// Accumulated competing / opposing projection energy inside the memory cut.
    pub internal_conflict_energy: f32,
    /// Branchless amplitude gate (`1.0` inside cognitive horizon, `0.0` in [`EpistemicRegime::LiminalVoid`]).
    pub horizon_gate: f32,
    /// Normalized internal contradiction severity in $[0.0, 1.0]$.
    pub conflict_severity: f32,
    /// Anchor ID of the strongest attractive/aligned node inside the cut (`0` if none).
    pub dominant_anchor_id: u32,
    /// Anchor ID of the strongest opposing/conflicting node inside the cut (`0` if none).
    pub opposing_anchor_id: u32,
    /// Encoded [`EpistemicRegime`] byte (`0` = Coherent, `1` = InternalConflict, `2` = LiminalVoid).
    pub regime_code: u8,
    /// Alignment padding bytes.
    pub reserved_align: [u8; 3],
}

impl EpistemicHorizonReading {
    /// Returns the decoded [`EpistemicRegime`].
    #[inline(always)]
    #[must_use]
    pub const fn regime(&self) -> EpistemicRegime {
        EpistemicRegime::from_code(self.regime_code)
    }

    /// Returns `true` when the query lies outside the carved memory cut ([`EpistemicRegime::LiminalVoid`]).
    #[inline(always)]
    #[must_use]
    pub const fn is_liminal_void(&self) -> bool {
        self.regime_code == EpistemicRegime::LiminalVoid as u8
    }

    /// Returns `true` when the query lies inside the memory cut yet triggers internal opposition ($H^1 \neq 0$).
    #[inline(always)]
    #[must_use]
    pub const fn is_internal_conflict(&self) -> bool {
        self.regime_code == EpistemicRegime::InternalConflict as u8
    }

    /// Modulates any synthesized [`GatewayFrame`] strictly branchlessly according to this epistemic
    /// horizon: zeroes amplitude in [`EpistemicRegime::LiminalVoid`] and elevates repulsive flags in
    /// [`EpistemicRegime::InternalConflict`].
    #[inline(always)]
    #[must_use]
    pub fn apply_to_frame_branchless(&self, mut frame: GatewayFrame) -> GatewayFrame {
        let is_void = self.is_liminal_void();
        let is_conflict = self.is_internal_conflict();
        let gate = branchless_clamp_f32(branchless_relu_f32(self.horizon_gate), 0.0, 1.0);

        frame.effective_energy =
            branchless_select_f32(is_void, 0.0, frame.effective_energy * gate);
        frame.repulsion_severity = branchless_select_f32(
            is_void,
            0.0,
            branchless_select_f32(
                is_conflict,
                branchless_max_f32(frame.repulsion_severity, self.conflict_severity),
                frame.repulsion_severity,
            ),
        );

        let high_repulse = (!is_void) & is_conflict & (self.conflict_severity > 0.50);
        frame.action_mode_flags = branchless_select_u8(
            is_void,
            ACTION_PRESET_PASSIVE_SIDECAR,
            branchless_select_u8(
                high_repulse,
                frame.action_mode_flags | ACTION_MODE_PHASE_REPULSE,
                frame.action_mode_flags,
            ),
        );
        frame.dominant_cause_code = branchless_select_u8(
            is_void,
            ROOT_CAUSE_FACT_RETRIEVAL_MISS,
            branchless_select_u8(
                high_repulse,
                ROOT_CAUSE_COHOMOLOGY_REPULSION,
                frame.dominant_cause_code,
            ),
        );
        frame.provenance_bits = branchless_select_u32(
            is_void,
            (frame.provenance_bits & !CAUSAL_CAUSE_COHOMOLOGY_REPULSION)
                | CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK,
            branchless_select_u32(
                is_conflict,
                frame.provenance_bits | CAUSAL_CAUSE_COHOMOLOGY_REPULSION,
                frame.provenance_bits,
            ),
        );

        // Zero phase words branchlessly when outside the carved memory horizon.
        let keep_mask = branchless_select_u64(is_void, 0, u64::MAX);
        {
            let pw = frame.phase_words_mut();
            let mut w = 0usize;
            while w < NUM_SWARM_MICRO_CELLS {
                pw[w] &= keep_mask;
                w += 1;
            }
        }
        frame.mean_phase_tag_q7 = branchless_select_i8(is_void, 0, frame.mean_phase_tag_q7);
        let eff_slot = frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK;
        let ptr_tag = pack_action_slot_tag(eff_slot, frame.action_mode_flags);
        frame.frame_flags = ptr_tag as u16;
        frame.tagged_ptr =
            gcso_tagged_ptr_t::pack(frame.mapped_section_offset48, ptr_tag, frame.mean_phase_tag_q7)
                .raw();
        frame
    }
}

impl CausalProvenanceSignal for EpistemicHorizonReading {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let is_void = self.is_liminal_void();
        let is_conflict = self.is_internal_conflict();
        branchless_select_u32(
            is_void,
            CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK,
            CAUSAL_AXIS_OBJECTIVE_FACT
                | branchless_select_u32(
                    is_conflict,
                    CAUSAL_AXIS_CONTRADICTION_GUARD | CAUSAL_CAUSE_COHOMOLOGY_REPULSION,
                    0,
                ),
        )
    }

    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        branchless_select_f32(
            self.is_liminal_void(),
            0.0,
            branchless_clamp_f32(self.conflict_severity, 0.0, 1.0),
        )
    }
}

impl ManifoldGatewaySignal for EpistemicHorizonReading {
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        let base = GatewayFrame {
            target_anchor_id: branchless_select_u32(
                self.is_liminal_void(),
                0,
                branchless_select_u32(
                    self.is_internal_conflict() && self.opposing_anchor_id != 0,
                    self.opposing_anchor_id,
                    self.dominant_anchor_id,
                ),
            ),
            effective_energy: branchless_max_f32(
                self.peak_positive_affinity,
                self.total_memory_support * 0.25,
            ),
            dominant_slot: branchless_select_u8(
                self.is_internal_conflict(),
                ATTRACTOR_SLOT_REPULSION,
                ATTRACTOR_SLOT_KNOWLEDGE,
            ),
            ..GatewayFrame::default()
        };
        self.apply_to_frame_branchless(base)
    }
}

/// Uniform node abstraction enabling relative signed coupling and epistemic horizon evaluation
/// across any attractor representation (`CrystallizedAttractorRecord`, `GatewayFrame`, `ObjectiveFact`, etc.)
/// atop [`LiminalVoid`].
pub trait RelativeAttractorNode: LiminalVoid {
    /// Static cognitive aperture mask for relative attractor participation.
    const RELATIVE_APERTURE_MASK: u32 = APERTURE_RELATIVE_COUPLING;
    /// Returns the node's 64-head Q7 phase profile packed into 8 register-resident `u64` words.
    fn relative_phase_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS];

    /// Returns the node's positive gravitational / activation energy ($\ge 0$).
    fn relative_energy(&self) -> f32;

    /// Returns the node's 32-bit manifold anchor ID.
    fn relative_anchor_id(&self) -> u32;

    /// Returns `true` when all 64 phase lanes share an identical Q7 value (enabling $\mathcal{O}(1)$ coherence).
    #[inline(always)]
    fn relative_phase_uniform(&self) -> bool {
        is_word_array_uniform_64(&self.relative_phase_words_64())
    }
}

impl<T: RelativeAttractorNode + ?Sized> RelativeAttractorNode for &T {
    #[inline(always)]
    fn relative_phase_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        (**self).relative_phase_words_64()
    }
    #[inline(always)]
    fn relative_energy(&self) -> f32 {
        (**self).relative_energy()
    }
    #[inline(always)]
    fn relative_anchor_id(&self) -> u32 {
        (**self).relative_anchor_id()
    }
    #[inline(always)]
    fn relative_phase_uniform(&self) -> bool {
        (**self).relative_phase_uniform()
    }
}

impl RelativeAttractorNode for CrystallizedAttractorRecord {
    #[inline(always)]
    fn relative_phase_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        *self.phase_words_ref()
    }
    #[inline(always)]
    fn relative_energy(&self) -> f32 {
        branchless_select_f32(self.sample_count > 0, self.phase_gravity_mass, 0.0)
    }
    #[inline(always)]
    fn relative_anchor_id(&self) -> u32 {
        self.anchor_id
    }
}

impl RelativeAttractorNode for GatewayFrame {
    #[inline(always)]
    fn relative_phase_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
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
    fn relative_phase_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
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
    fn relative_phase_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
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
    fn relative_phase_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
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

impl RelativeAttractorNode for gcso_pointer_trail_t {
    #[inline(always)]
    fn relative_phase_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        *trail_phase_words_ref(self)
    }
    #[inline(always)]
    fn relative_energy(&self) -> f32 {
        branchless_relu_f32(self.stigmergic_density)
    }
    #[inline(always)]
    fn relative_anchor_id(&self) -> u32 {
        self.target_anchor_id
    }
}

impl RelativeAttractorNode for ScaleLayerProjection {
    #[inline(always)]
    fn relative_phase_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
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

impl RelativeAttractorNode for crate::attractor::AttractorAnchor {
    #[inline(always)]
    fn relative_phase_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        *self.phase_words_ref()
    }
    #[inline(always)]
    fn relative_energy(&self) -> f32 {
        branchless_select_f32(
            (self.is_active != 0) & (self.id != 0),
            branchless_relu_f32(self.weight),
            0.0,
        )
    }
    #[inline(always)]
    fn relative_anchor_id(&self) -> u32 {
        self.id
    }
}

impl RelativeAttractorNode for crate::attractor::CrystallizedAnchorRecord {
    #[inline(always)]
    fn relative_phase_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        load_q7_words_64(&self.phase_deltas)
    }
    #[inline(always)]
    fn relative_energy(&self) -> f32 {
        branchless_select_f32(
            (self.is_active != 0) & (self.id != 0),
            branchless_relu_f32(self.weight),
            0.0,
        )
    }
    #[inline(always)]
    fn relative_anchor_id(&self) -> u32 {
        self.id
    }
}

#[inline]
fn accumulate_words_sincos_weighted(
    nw: &[u64; NUM_SWARM_MICRO_CELLS],
    weight: f32,
    sin_acc: &mut [f32; MAX_ATTENTION_HEADS],
    cos_acc: &mut [f32; MAX_ATTENTION_HEADS],
) {
    let mut cell = 0usize;
    while cell < NUM_SWARM_MICRO_CELLS {
        let word = nw[cell];
        let base = cell << 3;
        let mut lane = 0usize;
        while lane < 8 {
            let q7 = ((word >> (lane << 3)) & 0xFF) as u8 as i8;
            let (s_val, c_val) = branchless_sincos_q7(q7);
            let h = base + lane;
            sin_acc[h] += weight * s_val;
            cos_acc[h] += weight * c_val;
            lane += 1;
        }
        cell += 1;
    }
}

/// Self-similarly folds an arbitrary slice of [`RelativeAttractorNode`]s on $SO(2)^{64}$ into
/// `(folded_phase_words, total_energy, dominant_anchor_id)` strictly branchlessly (`0` allocations).
#[inline]
#[must_use]
pub fn fold_fractal_nodes_words_64<Node: RelativeAttractorNode>(
    nodes: &[Node],
) -> ([u64; NUM_SWARM_MICRO_CELLS], f32, u32) {
    let mut sin_acc = [0.0_f32; MAX_ATTENTION_HEADS];
    let mut cos_acc = [0.0_f32; MAX_ATTENTION_HEADS];
    let mut total_energy = 0.0_f32;
    let mut best_energy = 0.0_f32;
    let mut best_anchor_id = 0u32;

    let mut idx = 0usize;
    while idx < nodes.len() {
        let node = &nodes[idx];
        let e = branchless_relu_f32(node.relative_energy());
        let nw = node.relative_phase_words_64();
        accumulate_words_sincos_weighted(&nw, e, &mut sin_acc, &mut cos_acc);
        total_energy += e;
        let take_best = e > best_energy;
        best_energy = branchless_select_f32(take_best, e, best_energy);
        best_anchor_id =
            branchless_select_u32(take_best, node.relative_anchor_id(), best_anchor_id);
        idx += 1;
    }

    let words = fold_sincos64_to_words(&sin_acc, &cos_acc);
    (words, total_energy, best_anchor_id)
}

impl<A: RelativeAttractorNode, B: RelativeAttractorNode> RelativeAttractorNode for (A, B) {
    #[inline]
    fn relative_phase_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        let mut sin_acc = [0.0_f32; MAX_ATTENTION_HEADS];
        let mut cos_acc = [0.0_f32; MAX_ATTENTION_HEADS];
        let ea = branchless_relu_f32(self.0.relative_energy());
        let eb = branchless_relu_f32(self.1.relative_energy());
        accumulate_words_sincos_weighted(&self.0.relative_phase_words_64(), ea, &mut sin_acc, &mut cos_acc);
        accumulate_words_sincos_weighted(&self.1.relative_phase_words_64(), eb, &mut sin_acc, &mut cos_acc);
        fold_sincos64_to_words(&sin_acc, &cos_acc)
    }

    #[inline(always)]
    fn relative_energy(&self) -> f32 {
        branchless_relu_f32(self.0.relative_energy()) + branchless_relu_f32(self.1.relative_energy())
    }

    #[inline(always)]
    fn relative_anchor_id(&self) -> u32 {
        let ea = branchless_relu_f32(self.0.relative_energy());
        let eb = branchless_relu_f32(self.1.relative_energy());
        branchless_select_u32(
            eb > ea,
            self.1.relative_anchor_id(),
            self.0.relative_anchor_id(),
        )
    }
}

impl<Node: RelativeAttractorNode, const K: usize> RelativeAttractorNode for [Node; K] {
    #[inline]
    fn relative_phase_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        fold_fractal_nodes_words_64(self.as_slice()).0
    }

    #[inline]
    fn relative_energy(&self) -> f32 {
        fold_fractal_nodes_words_64(self.as_slice()).1
    }

    #[inline]
    fn relative_anchor_id(&self) -> u32 {
        fold_fractal_nodes_words_64(self.as_slice()).2
    }
}

/// Fixed-capacity stack-resident signed relational coupling matrix (`[[i8; N]; N]`) over $N$
/// attractors (`#![no_std]`, $C_{\mathrm{alloc}} = 0$).
///
/// Entry `weights_q7[i][j]` encodes the perspective-relative relation when node `i` is active:
/// - `> 0`: node `j` is synergistic/attractive relative to `i`;
/// - `< 0`: node `j` is mutually opposing/repulsive relative to `i`;
/// - `== 0`: node `j` is uncoupled from `i`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelativeCouplingMatrix<const N: usize> {
    /// Signed Q7 coupling weights (`-127..=127`) indexed `[perspective_idx][target_idx]`.
    pub weights_q7: [[i8; N]; N],
}

impl<const N: usize> Default for RelativeCouplingMatrix<N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> RelativeCouplingMatrix<N> {
    /// Creates an uncoupled zero-initialized matrix with `+127` self-coupling on the diagonal.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        let mut weights_q7 = [[0i8; N]; N];
        let mut i = 0usize;
        while i < N {
            weights_q7[i][i] = 127;
            i += 1;
        }
        Self { weights_q7 }
    }

    /// Returns the signed Q7 coupling perspective row for active attractor `perspective_idx`.
    #[inline(always)]
    #[must_use]
    pub const fn row(&self, perspective_idx: usize) -> &[i8; N] {
        let idx = if perspective_idx < N { perspective_idx } else { 0 };
        &self.weights_q7[idx]
    }

    /// Sets symmetric signed Q7 coupling (`(i, j)` and `(j, i)`) clamped to `[-127, 127]`.
    #[inline]
    pub fn set_symmetric(&mut self, i: usize, j: usize, weight_q7: i8) {
        let valid = (i < N) & (j < N) & (i != j);
        let clamped = branchless_select_i8(weight_q7 == -128, -127, weight_q7);
        let safe_i = branchless_select_usize(valid, i, 0);
        let safe_j = branchless_select_usize(valid, j, 0);
        let next = branchless_select_i8(valid, clamped, self.weights_q7[safe_i][safe_j]);
        self.weights_q7[safe_i][safe_j] = next;
        self.weights_q7[safe_j][safe_i] = next;
    }

    /// Branchlessly reinforces either positive synergy (`!mutually_exclusive`) or negative
    /// lateral opposition (`mutually_exclusive`) between `i` and `j`.
    #[inline]
    pub fn reinforce_relation_branchless(
        &mut self,
        i: usize,
        j: usize,
        mutually_exclusive: bool,
        step_q7: u8,
    ) {
        let valid = (i < N) & (j < N) & (i != j);
        let safe_i = branchless_select_usize(valid, i, 0);
        let safe_j = branchless_select_usize(valid, j, 0);
        let cur = self.weights_q7[safe_i][safe_j] as i32;
        let step = (step_q7.min(127)) as i32;
        let delta = branchless_select_i32(mutually_exclusive, -step, step);
        let updated_i32 = (cur + delta).clamp(-127, 127);
        let updated = updated_i32 as i8;
        let final_w = branchless_select_i8(valid, updated, self.weights_q7[safe_i][safe_j]);
        self.weights_q7[safe_i][safe_j] = final_w;
        self.weights_q7[safe_j][safe_i] = final_w;
    }

    /// Self-organizes relative couplings directly from a [`DissipativeOverflowReport`] without
    /// branches:
    /// - Overflow scatter (`report.scattered`) strengthens positive bridge synergy between
    ///   `primary_idx` and `landed_idx`.
    /// - Autonomous daughter birth (`report.spawned_new && report.overflowed`) establishes
    ///   negative lateral differentiation between parent `primary_idx` and daughter `landed_idx`.
    #[inline]
    pub fn record_dissipative_transition_branchless(
        &mut self,
        report: &DissipativeOverflowReport,
        step_q7: u8,
    ) {
        let edge_active = (report.scattered | (report.spawned_new & report.overflowed))
            & (report.primary_idx != report.landed_idx);
        let eff_step = branchless_select_u8(edge_active, step_q7, 0);
        let is_opposed = report.spawned_new & report.overflowed;
        self.reinforce_relation_branchless(
            report.primary_idx,
            report.landed_idx,
            is_opposed,
            eff_step,
        );
    }
}

/// Evaluates [`EpistemicHorizonReading`] over any slice of [`RelativeAttractorNode`]s relative
/// to `query_words` and an optional signed perspective row `coupling_row_q7` (`0` data branches).
#[inline]
#[must_use]
pub fn evaluate_epistemic_horizon_words_64<Node: RelativeAttractorNode>(
    nodes: &[Node],
    query_words: &[u64; NUM_SWARM_MICRO_CELLS],
    coupling_row_q7: Option<&[i8]>,
    horizon_floor: f32,
) -> EpistemicHorizonReading {
    let safe_floor = branchless_max_f32(branchless_relu_f32(horizon_floor), DEFAULT_ENTROPY_EPS);
    let q_uni = is_word_array_uniform_64(query_words);

    let mut total_memory_support = 0.0_f32;
    let mut internal_support_energy = 0.0_f32;
    let mut peak_positive_affinity = 0.0_f32;
    let mut peak_opposing_affinity = 0.0_f32;
    let mut internal_conflict_energy = 0.0_f32;
    let mut dominant_anchor_id = 0u32;
    let mut opposing_anchor_id = 0u32;

    let mut idx = 0usize;
    while idx < nodes.len() {
        let node = &nodes[idx];
        let e = branchless_relu_f32(node.relative_energy());
        let nw = node.relative_phase_words_64();
        let n_uni = is_word_array_uniform_64(&nw);
        let coh =
            coherence_from_word_arrays_with_both_uniform_flags_64(query_words, q_uni, &nw, n_uni);
        let pos_coh = branchless_relu_f32(coh);

        let w_q7 = coupling_row_q7
            .and_then(|row| row.get(idx).copied())
            .unwrap_or(127);
        let w_i32 = w_q7 as i32;
        let is_rep = w_i32 < 0;
        let is_att = (w_i32 > 0) | coupling_row_q7.is_none();
        let rel_mag = branchless_select_f32(
            coupling_row_q7.is_some(),
            (w_i32.unsigned_abs() as f32) * (1.0 / 127.0),
            1.0,
        );

        let raw_support = pos_coh * e;
        let weighted_support = raw_support * (pos_coh * rel_mag);
        total_memory_support += raw_support;

        let take_dom = is_att & (pos_coh > peak_positive_affinity) & (e > DEFAULT_ENTROPY_EPS);
        peak_positive_affinity = branchless_select_f32(take_dom, pos_coh, peak_positive_affinity);
        dominant_anchor_id =
            branchless_select_u32(take_dom, node.relative_anchor_id(), dominant_anchor_id);

        let take_opp = is_rep & (pos_coh > peak_opposing_affinity) & (e > DEFAULT_ENTROPY_EPS);
        peak_opposing_affinity = branchless_select_f32(take_opp, pos_coh, peak_opposing_affinity);
        opposing_anchor_id =
            branchless_select_u32(take_opp, node.relative_anchor_id(), opposing_anchor_id);

        internal_support_energy += branchless_select_f32(is_att, weighted_support, 0.0);
        internal_conflict_energy += branchless_select_f32(is_rep, weighted_support, 0.0);
        idx += 1;
    }

    let max_coh = branchless_max_f32(peak_positive_affinity, peak_opposing_affinity);
    let inside_horizon = (total_memory_support >= safe_floor) | (max_coh >= safe_floor);
    let horizon_gate = branchless_select_f32(inside_horizon, 1.0, 0.0);

    let conflict_ratio = branchless_clamp_f32(
        internal_conflict_energy * branchless_inv_pos_f32(internal_support_energy + DEFAULT_ENTROPY_EPS),
        0.0,
        2.0,
    );
    let conflict_active = inside_horizon
        & ((internal_conflict_energy > 0.0)
            & ((peak_opposing_affinity > peak_positive_affinity) | (conflict_ratio > 0.85)));
    let conflict_severity = branchless_select_f32(
        inside_horizon,
        branchless_clamp_f32(
            0.5 * conflict_ratio
                + 0.5 * branchless_relu_f32(peak_opposing_affinity - peak_positive_affinity + 0.5),
            0.0,
            1.0,
        ),
        0.0,
    );
    let regime_code = branchless_select_u8(
        !inside_horizon,
        EpistemicRegime::LiminalVoid as u8,
        branchless_select_u8(
            conflict_active,
            EpistemicRegime::InternalConflict as u8,
            EpistemicRegime::KnownCoherent as u8,
        ),
    );

    EpistemicHorizonReading {
        total_memory_support,
        peak_positive_affinity,
        internal_conflict_energy,
        horizon_gate,
        conflict_severity,
        dominant_anchor_id,
        opposing_anchor_id,
        regime_code,
        reserved_align: [0; 3],
    }
}

/// Folds a 64-head `[f32; 64]` `(sin, cos)` vector accumulator into 8 packed SWAR `u64` words.
#[inline]
fn fold_sincos64_to_words(sin_acc: &[f32; MAX_ATTENTION_HEADS], cos_acc: &[f32; MAX_ATTENTION_HEADS]) -> [u64; NUM_SWARM_MICRO_CELLS] {
    let mut words = [0u64; NUM_SWARM_MICRO_CELLS];
    let mut cell = 0usize;
    while cell < NUM_SWARM_MICRO_CELLS {
        let base = cell << 3;
        let sin_arr = [
            sin_acc[base],
            sin_acc[base + 1],
            sin_acc[base + 2],
            sin_acc[base + 3],
            sin_acc[base + 4],
            sin_acc[base + 5],
            sin_acc[base + 6],
            sin_acc[base + 7],
        ];
        let cos_arr = [
            cos_acc[base],
            cos_acc[base + 1],
            cos_acc[base + 2],
            cos_acc[base + 3],
            cos_acc[base + 4],
            cos_acc[base + 5],
            cos_acc[base + 6],
            cos_acc[base + 7],
        ];
        words[cell] = atan2_to_q7_word_x8_simd(&sin_arr, &cos_arr);
        cell += 1;
    }
    words
}

/// Branchlessly synthesizes a 128-byte [`GatewayFrame`] and [`EpistemicHorizonReading`] over
/// `nodes` strictly from the active perspective defined by `coupling_row_q7` (`0` static role branches).
///
/// Positive entries in `coupling_row_q7` pull toward synergistic nodes; negative entries inhibit
/// and project opposing nodes onto the repulsive axis (`ACTION_MODE_PHASE_REPULSE`). When
/// `query_words` falls into [`EpistemicRegime::LiminalVoid`] (`< horizon_floor`), steering amplitude
/// is scaled cleanly to `0.0`.
#[inline]
#[must_use]
pub fn synthesize_relative_gateway_from_perspective_words_64<Node: RelativeAttractorNode>(
    nodes: &[Node],
    coupling_row_q7: &[i8],
    query_words: &[u64; NUM_SWARM_MICRO_CELLS],
    horizon_floor: f32,
) -> (GatewayFrame, EpistemicHorizonReading) {
    let horizon = evaluate_epistemic_horizon_words_64(
        nodes,
        query_words,
        Some(coupling_row_q7),
        horizon_floor,
    );
    let q_uni = is_word_array_uniform_64(query_words);

    let mut att_sin = [0.0_f32; MAX_ATTENTION_HEADS];
    let mut att_cos = [0.0_f32; MAX_ATTENTION_HEADS];
    let mut rep_sin = [0.0_f32; MAX_ATTENTION_HEADS];
    let mut rep_cos = [0.0_f32; MAX_ATTENTION_HEADS];
    let mut sum_att = 0.0_f32;
    let mut sum_rep = 0.0_f32;
    let mut best_att_pull = 0.0_f32;
    let mut best_rep_pull = 0.0_f32;
    let mut best_att_id = 0u32;
    let mut best_rep_id = 0u32;

    let count = nodes.len().min(coupling_row_q7.len());
    let mut idx = 0usize;
    while idx < count {
        let w_i32 = coupling_row_q7[idx] as i32;
        let is_rep = w_i32 < 0;
        let is_att = w_i32 > 0;
        let mag = (w_i32.unsigned_abs() as f32) * (1.0 / 127.0);

        let node = &nodes[idx];
        let e = branchless_relu_f32(node.relative_energy());
        let nw = node.relative_phase_words_64();
        let n_uni = is_word_array_uniform_64(&nw);
        let coh =
            coherence_from_word_arrays_with_both_uniform_flags_64(query_words, q_uni, &nw, n_uni);
        let pos_coh = branchless_relu_f32(coh);
        let pull = mag * e * (pos_coh * pos_coh);

        let att_w = branchless_select_f32(is_att, pull, 0.0);
        let rep_w = branchless_select_f32(is_rep, pull, 0.0);
        sum_att += att_w;
        sum_rep += rep_w;

        let take_att = is_att & (att_w > best_att_pull);
        best_att_pull = branchless_select_f32(take_att, att_w, best_att_pull);
        best_att_id = branchless_select_u32(take_att, node.relative_anchor_id(), best_att_id);

        let take_rep = is_rep & (rep_w > best_rep_pull);
        best_rep_pull = branchless_select_f32(take_rep, rep_w, best_rep_pull);
        best_rep_id = branchless_select_u32(take_rep, node.relative_anchor_id(), best_rep_id);

        let mut cell = 0usize;
        while cell < NUM_SWARM_MICRO_CELLS {
            let word = nw[cell];
            let base = cell << 3;
            let mut lane = 0usize;
            while lane < 8 {
                let q7 = ((word >> (lane << 3)) & 0xFF) as u8 as i8;
                let (s_val, c_val) = branchless_sincos_q7(q7);
                let h = base + lane;
                att_sin[h] += att_w * s_val;
                att_cos[h] += att_w * c_val;
                rep_sin[h] += rep_w * s_val;
                rep_cos[h] += rep_w * c_val;
                lane += 1;
            }
            cell += 1;
        }
        idx += 1;
    }

    let mut att_w = fold_sincos64_to_words(&att_sin, &att_cos);
    let rep_w = fold_sincos64_to_words(&rep_sin, &rep_cos);
    let rep_strength = branchless_clamp_f32(
        sum_rep * branchless_inv_pos_f32(sum_att + sum_rep + DEFAULT_ENTROPY_EPS),
        0.0,
        1.0,
    );
    let att_post = branchless_q7_inhibit_words_64(&mut att_w, sum_att, &rep_w, rep_strength);

    let rep_dominates = (sum_rep > sum_att) & (sum_rep > DEFAULT_ENTROPY_EPS);
    let mut final_w = [0u64; NUM_SWARM_MICRO_CELLS];
    let mut c = 0usize;
    while c < NUM_SWARM_MICRO_CELLS {
        // Flip opposing vector by pi (XOR 0x80 per Q7 lane) when repulsion dominates.
        let rep_flipped = rep_w[c] ^ SWAR_SIGN_MASK_64;
        final_w[c] = branchless_select_u64(rep_dominates, rep_flipped, att_w[c]);
        c += 1;
    }

    let eff_energy = branchless_min_f32(
        branchless_select_f32(rep_dominates, sum_rep, att_post),
        MAX_ATTRACTOR_PULL_FORCE,
    );
    let mean_tag = fast_mean_q7_tag_from_words_64(&final_w);
    let dom_slot = branchless_select_u8(
        rep_dominates,
        ATTRACTOR_SLOT_REPULSION,
        ATTRACTOR_SLOT_KNOWLEDGE,
    );
    let mode_flags = branchless_select_u8(
        rep_dominates,
        ACTION_PRESET_REPULSION_ESCAPE,
        ACTION_PRESET_EXPRESS_PHASE | ACTION_MODE_SUBHEAD_ROUTE,
    );
    let target_id = branchless_select_u32(rep_dominates, best_rep_id, best_att_id);
    let base_coh = branchless_clamp_f32(
        horizon.peak_positive_affinity - horizon.conflict_severity * 0.5,
        -1.0,
        1.0,
    );
    let norm_rep_sev = branchless_select_f32(
        rep_dominates,
        branchless_clamp_f32(0.55 + 0.45 * rep_strength, 0.55, 1.0),
        rep_strength * 0.50,
    );

    let packed_tag = pack_action_slot_tag(dom_slot, mode_flags);
    let qdps_step_u8 = branchless_rad_to_qdps_step_u8(DEFAULT_QDPS_MIN_STEP_RAD);

    let mut frame = GatewayFrame {
        tagged_ptr: gcso_tagged_ptr_t::pack(0, packed_tag, mean_tag).raw(),
        target_anchor_id: target_id,
        effective_energy: eff_energy,
        ripa_clamp_rad: DEFAULT_FACT_RIPA_CLAMP_RAD,
        qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD,
        effective_inv_temp: 1.0 + branchless_clamp_f32(eff_energy * 0.25, 0.0, 1.0),
        fact_persona_coherence: base_coh,
        repulsion_severity: norm_rep_sev,
        provenance_bits: horizon.causal_provenance_bits(),
        dominant_cause_code: branchless_select_u8(
            rep_dominates,
            ROOT_CAUSE_COHOMOLOGY_REPULSION,
            ROOT_CAUSE_NOMINAL,
        ),
        dominant_slot: dom_slot,
        action_mode_flags: mode_flags,
        active_swarm_cell_mask: 0xFF,
        qdps_min_step_q7: qdps_step_u8,
        mean_phase_tag_q7: mean_tag,
        frame_flags: packed_tag as u16,
        ..GatewayFrame::default()
    };
    *frame.phase_words_mut() = final_w;
    let gated_frame = horizon.apply_to_frame_branchless(frame);
    (gated_frame, horizon)
}

/// Blanket-implemented epistemic & relative-coupling trait over any [`RelativeAttractorNode`].
pub trait EpistemicManifoldOps: RelativeAttractorNode {
    /// Evaluates [`EpistemicHorizonReading`] for a slice of `Self` relative to `query_words`.
    #[inline(always)]
    fn evaluate_slice_epistemic_horizon(
        slice: &[Self],
        query_words: &[u64; NUM_SWARM_MICRO_CELLS],
        coupling_row_q7: Option<&[i8]>,
        horizon_floor: f32,
    ) -> EpistemicHorizonReading
    where
        Self: Sized,
    {
        evaluate_epistemic_horizon_words_64(slice, query_words, coupling_row_q7, horizon_floor)
    }

    /// Self-similarly folds a slice of `Self` on $SO(2)^{64}$ into `(phase_words, total_energy, dominant_anchor_id)`.
    #[inline(always)]
    fn fold_slice_as_fractal_node(slice: &[Self]) -> ([u64; NUM_SWARM_MICRO_CELLS], f32, u32)
    where
        Self: Sized,
    {
        fold_fractal_nodes_words_64(slice)
    }

    /// Synthesizes a perspective-relative [`GatewayFrame`] and [`EpistemicHorizonReading`] over
    /// a slice of `Self` using signed relative weights `coupling_row_q7` (`0` static role branches).
    #[inline(always)]
    fn synthesize_slice_relative_gateway(
        slice: &[Self],
        coupling_row_q7: &[i8],
        query_words: &[u64; NUM_SWARM_MICRO_CELLS],
        horizon_floor: f32,
    ) -> (GatewayFrame, EpistemicHorizonReading)
    where
        Self: Sized,
    {
        synthesize_relative_gateway_from_perspective_words_64(
            slice,
            coupling_row_q7,
            query_words,
            horizon_floor,
        )
    }

    /// Computes normalized 2D phase affinity `cos(query, self) * self.relative_energy()` branchlessly
    /// without checking static role labels.
    #[inline(always)]
    fn relative_projection_energy(&self, query_words: &[u64; NUM_SWARM_MICRO_CELLS]) -> f32 {
        let nw = self.relative_phase_words_64();
        let coh = coherence_from_word_arrays_with_both_uniform_flags_64(
            query_words,
            is_word_array_uniform_64(query_words),
            &nw,
            is_word_array_uniform_64(&nw),
        );
        let pos = branchless_relu_f32(coh);
        let e = branchless_relu_f32(self.relative_energy());
        pos * e
    }

    /// Evaluates Venn-diagram multi-attractor aperture overlap and noise-driven serendipity in
    /// [`LiminalVoid`] across a slice of `Self` (`0` allocations, `0` branches).
    #[inline(always)]
    fn evaluate_slice_venn_and_serendipity(
        slice: &[Self],
        query_words: &[u64; NUM_STIGMERGIC_GRANULES],
        noisy_drifted_words: &[u64; NUM_STIGMERGIC_GRANULES],
        aperture_floor: f32,
        serendipity_floor_q7: u8,
    ) -> StigmergicVennSerendipityReport
    where
        Self: Sized,
    {
        evaluate_stigmergic_venn_and_serendipity_words_64(
            slice,
            query_words,
            noisy_drifted_words,
            aperture_floor,
            serendipity_floor_q7,
        )
    }
}

impl<T: RelativeAttractorNode + ?Sized> EpistemicManifoldOps for T {}

// ===================================================================
// Dynamic LiminalVoid Aperture Shifting & Self-Introspecting Telemetry
// ===================================================================

/// Dynamic, branchlessly adaptable cognitive aperture controller (`#[repr(C, align(16))]`, 32 B)
/// shifting the [`LiminalVoid`] boundary (`void_floor_q7`, `aperture_gain_q7`, `phase_shift_q7`)
/// across the 8 trait horizon channels in response to [`CognitiveTrajectoryLedger`] habits,
/// [`EpistemicHorizonReading`] state, and [`DissipativeOverflowReport`] crystallization events.
///
/// Channel indices:
/// - `0`: `SystemConstraintSignal`
/// - `1`: `ObjectiveFactSignal`
/// - `2`: `SubjectivePersonaSignal`
/// - `3`: `ContradictionGuardSignal`
/// - `4`: `CognitiveTrajectoryLedger` (`Circuit`)
/// - `5`: `RamPhaseKvCacheOps` (`KvCache`)
/// - `6`: `RelativeCouplingMatrix` (`RelativeCoupling`)
/// - `7`: `ActionHub` (Wide Execution Aperture)
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiminalApertureField {
    /// Per-channel `LiminalVoid` threshold floor in Q0.7 (`[8..=120]`): higher values expand the
    /// unknowable exterior for that channel, filtering low-support signals cleanly to `0`.
    pub void_floor_q7: [u8; NUM_LIMINAL_APERTURE_CHANNELS],
    /// Per-channel cognitive window opening gain in Q0.7 (`[16..=127]`).
    pub aperture_gain_q7: [u8; NUM_LIMINAL_APERTURE_CHANNELS],
    /// Per-channel signed Q7 focal phase offset (`[-127..=127]`).
    pub phase_shift_q7: [i8; NUM_LIMINAL_APERTURE_CHANNELS],
    /// Active cognitive aperture bitmask reflecting dynamically open vs. collapsed channels.
    pub active_aperture_mask: u32,
    /// Monotonic count of autonomous aperture adaptation cycles applied.
    pub adaptation_epochs: u32,
}

impl Default for LiminalApertureField {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl LiminalApertureField {
    /// Neutral baseline cognitive aperture configuration (`32 B`, `#[repr(C, align(16))]`).
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            void_floor_q7: [20, 28, 32, 24, 24, 24, 24, 16],
            aperture_gain_q7: [96; NUM_LIMINAL_APERTURE_CHANNELS],
            phase_shift_q7: [0; NUM_LIMINAL_APERTURE_CHANNELS],
            active_aperture_mask: APERTURE_ACTION_HUB_WIDE,
            adaptation_epochs: 0,
        }
    }

    /// Returns channel `ch`'s normalized `LiminalVoid` threshold floor in `[0.0, 1.0]`.
    #[inline(always)]
    #[must_use]
    pub const fn channel_void_floor_f32(&self, ch: usize) -> f32 {
        (self.void_floor_q7[ch & (NUM_LIMINAL_APERTURE_CHANNELS - 1)] as f32) * (1.0 / 127.0)
    }

    /// Returns channel `ch`'s normalized cognitive aperture gain in `[0.0, 1.0]`.
    #[inline(always)]
    #[must_use]
    pub const fn channel_gain_f32(&self, ch: usize) -> f32 {
        (self.aperture_gain_q7[ch & (NUM_LIMINAL_APERTURE_CHANNELS - 1)] as f32) * (1.0 / 127.0)
    }

    /// Branchlessly adapts each channel's dynamic `LiminalVoid` boundary from `ledger`, `horizon`,
    /// and optional `overflow_opt` without heap allocation or data-dependent branches.
    #[inline]
    pub fn adapt_from_circuit_and_overflow_branchless(
        &mut self,
        ledger: &CognitiveTrajectoryLedger,
        horizon: &EpistemicHorizonReading,
        overflow_opt: Option<&DissipativeOverflowReport>,
    ) {
        let (op_hist, _depth) = ledger.op_counts();
        let peel_n = op_hist[CognitiveOpKind::DecomposePeel as usize]
            .saturating_add(op_hist[CognitiveOpKind::RouteSubHead as usize]);
        let fuse_n = op_hist[CognitiveOpKind::SynthesizeFuse as usize]
            .saturating_add(op_hist[CognitiveOpKind::CrystallizeCommit as usize]);
        let rep_n = op_hist[CognitiveOpKind::RepulseAudit as usize];

        let peel_dom = (peel_n > fuse_n) | horizon.is_internal_conflict() | (rep_n > 0);
        let scattered = match overflow_opt {
            Some(r) => r.scattered,
            None => false,
        };
        let spawned = match overflow_opt {
            Some(r) => r.spawned_new & r.overflowed,
            None => false,
        };
        let res_mean_q7 = match overflow_opt {
            Some(r) => swar_q7_mean_i8_from_words_64(&r.residual_words),
            None => 0,
        };
        let fuse_dom = (fuse_n > peel_n) | scattered;
        let only_fuse = fuse_dom & !peel_dom;
        let void_active = horizon.is_liminal_void();

        // Channel 1 (ObjectiveFact) & Channel 3 (ContradictionGuard): narrow void under peel/conflict.
        let next_f1 = branchless_select_u8(
            peel_dom,
            self.void_floor_q7[1].saturating_sub(4).max(8),
            self.void_floor_q7[1],
        );
        self.void_floor_q7[1] = branchless_select_u8(
            void_active,
            next_f1.saturating_add(4).min(120),
            next_f1,
        );
        self.aperture_gain_q7[1] = branchless_select_u8(
            peel_dom,
            self.aperture_gain_q7[1].saturating_add(8).min(127),
            branchless_select_u8(
                void_active,
                self.aperture_gain_q7[1].saturating_sub(6).max(16),
                self.aperture_gain_q7[1],
            ),
        );

        let next_f3 = branchless_select_u8(
            peel_dom,
            self.void_floor_q7[3].saturating_sub(4).max(8),
            self.void_floor_q7[3],
        );
        self.void_floor_q7[3] = branchless_select_u8(
            void_active,
            next_f3.saturating_add(4).min(120),
            next_f3,
        );
        self.aperture_gain_q7[3] = branchless_select_u8(
            peel_dom,
            self.aperture_gain_q7[3].saturating_add(8).min(127),
            branchless_select_u8(
                void_active,
                self.aperture_gain_q7[3].saturating_sub(6).max(16),
                self.aperture_gain_q7[3],
            ),
        );

        // Channel 2 (SubjectivePersona): push into LiminalVoid during peel/conflict; open under fuse.
        let p2_floor_peel = branchless_select_u8(
            peel_dom,
            self.void_floor_q7[2].saturating_add(8).min(120),
            self.void_floor_q7[2],
        );
        let p2_floor_fuse = branchless_select_u8(
            only_fuse,
            p2_floor_peel.saturating_sub(6).max(8),
            p2_floor_peel,
        );
        self.void_floor_q7[2] = branchless_select_u8(
            void_active,
            p2_floor_fuse.saturating_add(4).min(120),
            p2_floor_fuse,
        );
        let p2_gain_peel = branchless_select_u8(
            peel_dom,
            self.aperture_gain_q7[2].saturating_sub(8).max(24),
            self.aperture_gain_q7[2],
        );
        let p2_gain_fuse = branchless_select_u8(
            only_fuse,
            p2_gain_peel.saturating_add(6).min(127),
            p2_gain_peel,
        );
        self.aperture_gain_q7[2] = branchless_select_u8(
            void_active,
            p2_gain_fuse.saturating_sub(6).max(16),
            p2_gain_fuse,
        );

        // Channel 5 (KvCache): open under integrative fuse or scatter transitions.
        self.void_floor_q7[5] = branchless_select_u8(
            fuse_dom,
            self.void_floor_q7[5].saturating_sub(6).max(8),
            self.void_floor_q7[5],
        );
        self.aperture_gain_q7[5] = branchless_select_u8(
            fuse_dom,
            self.aperture_gain_q7[5].saturating_add(6).min(127),
            self.aperture_gain_q7[5],
        );

        // Channel 6 (RelativeCoupling): open under fuse/scatter/birth and track scattered phase shift.
        let widen_rel = only_fuse | scattered | spawned;
        self.void_floor_q7[6] = branchless_select_u8(
            widen_rel,
            self.void_floor_q7[6].saturating_sub(6).max(8),
            self.void_floor_q7[6],
        );
        self.aperture_gain_q7[6] = branchless_select_u8(
            widen_rel,
            self.aperture_gain_q7[6].saturating_add(6).min(127),
            self.aperture_gain_q7[6],
        );
        let blended_shift =
            ((i32::from(self.phase_shift_q7[6]) + i32::from(res_mean_q7)) >> 1).clamp(-127, 127)
                as i8;
        self.phase_shift_q7[6] = branchless_select_i8(
            scattered | spawned,
            blended_shift,
            self.phase_shift_q7[6],
        );

        // Recompute active_aperture_mask branchlessly from channel floor saturation (< 96).
        let mut mask = APERTURE_ACTION_HUB_WIDE & !0x0000_00FF;
        let mut ch = 0usize;
        while ch < NUM_LIMINAL_APERTURE_CHANNELS {
            let is_open = (self.void_floor_q7[ch] < 96) & (!void_active | (ch == 0) | (ch == 7));
            mask |= branchless_select_u32(is_open, 1u32 << ch, 0);
            ch += 1;
        }
        self.active_aperture_mask = mask;
        self.adaptation_epochs = self.adaptation_epochs.wrapping_add(1);
    }

    /// Branchlessly modulates `frame` with this [`LiminalApertureField`] and `horizon`:
    /// enforces the dynamic Fact `LiminalVoid` floor, scales Fact/Logic/Explore sub-head gains,
    /// and applies channel `7`'s focal phase shift when inside the horizon.
    #[inline(always)]
    #[must_use]
    pub fn modulate_gateway_frame_branchless(
        &self,
        frame: GatewayFrame,
        horizon: &EpistemicHorizonReading,
    ) -> GatewayFrame {
        let mut out = horizon.apply_to_frame_branchless(frame);
        let below_fact_floor = horizon.total_memory_support < self.channel_void_floor_f32(1);
        let is_void = horizon.is_liminal_void() | below_fact_floor;

        out.effective_energy = branchless_select_f32(is_void, 0.0, out.effective_energy);

        let scale_gain = |base_q7: u8, ap_q7: u8| -> u8 {
            let scaled = (u16::from(base_q7) * (u16::from(ap_q7) + 32)) >> 7;
            scaled.clamp(16, 255) as u8
        };
        out.subhead_fact_gain_q7 =
            scale_gain(out.subhead_fact_gain_q7, self.aperture_gain_q7[1]);
        out.subhead_logic_gain_q7 =
            scale_gain(out.subhead_logic_gain_q7, self.aperture_gain_q7[3]);
        out.subhead_explore_gain_q7 =
            scale_gain(out.subhead_explore_gain_q7, self.aperture_gain_q7[2]);

        let shift_byte = self.phase_shift_q7[7] as u8 as u64;
        let shift_word = shift_byte.wrapping_mul(SWAR_SPLAT_BYTE_64);
        let keep_mask = branchless_select_u64(is_void, 0, u64::MAX);
        {
            let pw = out.phase_words_mut();
            let mut c = 0usize;
            while c < NUM_SWARM_MICRO_CELLS {
                let shifted = swar_q7_circular_add_x8(pw[c], shift_word);
                pw[c] = shifted & keep_mask;
                c += 1;
            }
        }
        out.mean_phase_tag_q7 = swar_q7_mean_i8_from_words_64(out.phase_words_ref());
        let eff_slot = out.dominant_slot & ATTRACTOR_SLOT_ID_MASK;
        let ptr_tag = pack_action_slot_tag(eff_slot, out.action_mode_flags);
        out.frame_flags = ptr_tag as u16;
        out.tagged_ptr =
            gcso_tagged_ptr_t::pack(out.mapped_section_offset48, ptr_tag, out.mean_phase_tag_q7)
                .raw();
        out
    }
}

/// Static compile-time descriptor binding a trait horizon node to its static cognitive aperture
/// and unknowable exterior complement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiminalHorizonNodeDescriptor {
    /// Mermaid graph node identifier (`"N0"`..`"N7"`).
    pub node_id: &'static str,
    /// Human-readable Rust trait name refining [`LiminalVoid`].
    pub trait_name: &'static str,
    /// Channel index (`0..8`) inside [`LiminalApertureField`].
    pub channel_idx: usize,
    /// Static cognitive aperture mask owned by this trait node.
    pub static_aperture_mask: u32,
    /// Type-level unknowable exterior complement (`!static_aperture_mask`).
    pub unknowable_void_mask: u32,
}

/// Canonical 8-channel trait topology table derived from type-level [`LiminalVoid`] refinements.
pub const GCSO_LIMINAL_HORIZON_TOPOLOGY: [LiminalHorizonNodeDescriptor;
    NUM_LIMINAL_APERTURE_CHANNELS] = [
    LiminalHorizonNodeDescriptor {
        node_id: "N0",
        trait_name: "SystemConstraintSignal",
        channel_idx: 0,
        static_aperture_mask: APERTURE_SYSTEM_CONSTRAINT,
        unknowable_void_mask: !APERTURE_SYSTEM_CONSTRAINT,
    },
    LiminalHorizonNodeDescriptor {
        node_id: "N1",
        trait_name: "ObjectiveFactSignal",
        channel_idx: 1,
        static_aperture_mask: APERTURE_OBJECTIVE_FACT,
        unknowable_void_mask: !APERTURE_OBJECTIVE_FACT,
    },
    LiminalHorizonNodeDescriptor {
        node_id: "N2",
        trait_name: "SubjectivePersonaSignal",
        channel_idx: 2,
        static_aperture_mask: APERTURE_SUBJECTIVE_PERSONA,
        unknowable_void_mask: !APERTURE_SUBJECTIVE_PERSONA,
    },
    LiminalHorizonNodeDescriptor {
        node_id: "N3",
        trait_name: "ContradictionGuardSignal",
        channel_idx: 3,
        static_aperture_mask: APERTURE_CONTRADICTION_GUARD,
        unknowable_void_mask: !APERTURE_CONTRADICTION_GUARD,
    },
    LiminalHorizonNodeDescriptor {
        node_id: "N4",
        trait_name: "CognitiveTrajectoryLedger",
        channel_idx: 4,
        static_aperture_mask: APERTURE_COGNITIVE_CIRCUIT,
        unknowable_void_mask: !APERTURE_COGNITIVE_CIRCUIT,
    },
    LiminalHorizonNodeDescriptor {
        node_id: "N5",
        trait_name: "RamPhaseKvCacheOps",
        channel_idx: 5,
        static_aperture_mask: APERTURE_KV_PHASE_CACHE,
        unknowable_void_mask: !APERTURE_KV_PHASE_CACHE,
    },
    LiminalHorizonNodeDescriptor {
        node_id: "N6",
        trait_name: "RelativeCouplingMatrix",
        channel_idx: 6,
        static_aperture_mask: APERTURE_RELATIVE_COUPLING,
        unknowable_void_mask: !APERTURE_RELATIVE_COUPLING,
    },
    LiminalHorizonNodeDescriptor {
        node_id: "N7",
        trait_name: "ActionHub",
        channel_idx: 7,
        static_aperture_mask: APERTURE_ACTION_HUB_WIDE,
        unknowable_void_mask: !APERTURE_ACTION_HUB_WIDE,
    },
];

/// Compact `32`-byte (`#[repr(C, align(16))]`) self-introspecting telemetry snapshot capturing
/// live [`LiminalApertureField`] boundaries, active epistemic regime, and crystal population counts.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LiminalTopologyTelemetrySnapshot {
    /// Per-channel `LiminalVoid` floor values in Q0.7 (`0..=127`).
    pub void_floor_q7: [u8; NUM_LIMINAL_APERTURE_CHANNELS],
    /// Per-channel cognitive aperture gain values in Q0.7 (`0..=127`).
    pub aperture_gain_q7: [u8; NUM_LIMINAL_APERTURE_CHANNELS],
    /// Per-channel signed focal phase shifts in Q7 (`-128..=127`).
    pub phase_shift_q7: [i8; NUM_LIMINAL_APERTURE_CHANNELS],
    /// Live active cognitive aperture bitmask.
    pub active_aperture_mask: u32,
    /// Encoded [`EpistemicRegime`] byte (`0`, `1`, `2`).
    pub regime_code: u8,
    /// Quantized conflict severity in Q0.7 (`0..=127`).
    pub conflict_severity_q7: u8,
    /// Number of active macro attractors currently populated.
    pub active_crystals: u8,
    /// Number of autonomously spawned daughter attractors recorded.
    pub spawned_daughters: u8,
}

struct BufCursor<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl<'a> BufCursor<'a> {
    #[inline]
    fn write_bytes(&mut self, bytes: &[u8]) {
        let rem = self.buf.len().saturating_sub(self.pos);
        let n = bytes.len().min(rem);
        self.buf[self.pos..self.pos + n].copy_from_slice(&bytes[..n]);
        self.pos += n;
    }

    #[inline]
    fn write_str(&mut self, s: &str) {
        self.write_bytes(s.as_bytes());
    }

    #[inline]
    fn write_u8_dec(&mut self, val: u8) {
        let mut tmp = [0u8; 3];
        let mut v = val;
        let mut len = 0usize;
        if v >= 100 {
            tmp[len] = b'0' + (v / 100);
            len += 1;
            v %= 100;
            tmp[len] = b'0' + (v / 10);
            len += 1;
            tmp[len] = b'0' + (v % 10);
            len += 1;
        } else if v >= 10 {
            tmp[len] = b'0' + (v / 10);
            len += 1;
            tmp[len] = b'0' + (v % 10);
            len += 1;
        } else {
            tmp[len] = b'0' + v;
            len += 1;
        }
        self.write_bytes(&tmp[..len]);
    }

    #[inline]
    fn write_i8_dec(&mut self, val: i8) {
        if val < 0 {
            self.write_bytes(b"-");
            self.write_u8_dec(val.unsigned_abs());
        } else {
            self.write_bytes(b"+");
            self.write_u8_dec(val as u8);
        }
    }

    #[inline]
    fn write_hex_u16(&mut self, val: u32) {
        const HEX: &[u8; 16] = b"0123456789ABCDEF";
        let out = [
            b'0',
            b'x',
            HEX[((val >> 12) & 0xF) as usize],
            HEX[((val >> 8) & 0xF) as usize],
            HEX[((val >> 4) & 0xF) as usize],
            HEX[(val & 0xF) as usize],
        ];
        self.write_bytes(&out);
    }

    #[inline]
    fn write_hex_u8(&mut self, val: u8) {
        const HEX: &[u8; 16] = b"0123456789ABCDEF";
        let out = [
            b'0',
            b'x',
            HEX[((val >> 4) & 0xF) as usize],
            HEX[(val & 0xF) as usize],
        ];
        self.write_bytes(&out);
    }
}

impl LiminalTopologyTelemetrySnapshot {
    /// Captures a compact self-introspecting telemetry snapshot from `field` and `horizon`.
    #[inline]
    #[must_use]
    pub fn capture(
        field: &LiminalApertureField,
        horizon: &EpistemicHorizonReading,
        active_crystals: u8,
        spawned_daughters: u8,
    ) -> Self {
        let sev = (branchless_clamp_f32(horizon.conflict_severity, 0.0, 1.0) * 127.0 + 0.5) as u8;
        Self {
            void_floor_q7: field.void_floor_q7,
            aperture_gain_q7: field.aperture_gain_q7,
            phase_shift_q7: field.phase_shift_q7,
            active_aperture_mask: field.active_aperture_mask,
            regime_code: horizon.regime_code,
            conflict_severity_q7: sev,
            active_crystals,
            spawned_daughters,
        }
    }

    #[inline]
    fn format_mermaid_flowchart_core(
        &self,
        layer1_opt: Option<(&StigmergicGranuleField, &StigmergicVennSerendipityReport)>,
        buf: &mut [u8],
    ) -> usize {
        let mut cur = BufCursor { buf, pos: 0 };
        let regime_str = match EpistemicRegime::from_code(self.regime_code) {
            EpistemicRegime::KnownCoherent => "KnownCoherent",
            EpistemicRegime::InternalConflict => "InternalConflict",
            EpistemicRegime::LiminalVoid => "LiminalVoid",
        };
        cur.write_str("flowchart TD\n");
        cur.write_str("    LVOID[\"Core Layer 0: LiminalVoid (Epistemic Horizon & Exterior)\"]\n");

        match layer1_opt {
            Some((layer1, venn)) => {
                cur.write_str("    STIG[\"Core Layer 1: StigmergicGranuleField | VennCore:");
                cur.write_hex_u8(venn.venn_core_mask);
                cur.write_str(" | Serendipity:");
                cur.write_hex_u8(venn.serendipity_excursion_mask);
                cur.write_str(" | Quorum:");
                cur.write_u8_dec(layer1.quorum_threshold);
                cur.write_str("/8 | Epochs:");
                cur.write_u8_dec((layer1.adaptation_epochs & 0xFF) as u8);
                cur.write_str("\"]\n");
            }
            None => {
                cur.write_str("    STIG[\"Core Layer 1: Stigmergic Approximation | Apt:");
                cur.write_hex_u16(APERTURE_SWARM_CONSENSUS);
                cur.write_str("\"]\n");
            }
        }

        let mut i = 0usize;
        while i < 7 {
            let desc = &GCSO_LIMINAL_HORIZON_TOPOLOGY[i];
            let ch = desc.channel_idx;
            let active = (self.active_aperture_mask & (1u32 << ch)) != 0;
            cur.write_str("    ");
            cur.write_str(desc.node_id);
            cur.write_str("[\"");
            cur.write_str(desc.trait_name);
            cur.write_str(" | Apt:");
            cur.write_hex_u16(desc.static_aperture_mask);
            cur.write_str(" | VoidFloor:");
            cur.write_u8_dec(self.void_floor_q7[ch]);
            cur.write_str("/127 | Gain:");
            cur.write_u8_dec(self.aperture_gain_q7[ch]);
            cur.write_str("/127 | Shift:");
            cur.write_i8_dec(self.phase_shift_q7[ch]);
            if active {
                cur.write_str(" | ACTIVE\"]\n");
            } else {
                cur.write_str(" | COLLAPSED\"]\n");
            }
            i += 1;
        }

        cur.write_str("    GW[\"ManifoldGatewaySignal | Apt:");
        cur.write_hex_u16(APERTURE_MANIFOLD_GATEWAY);
        cur.write_str(" | Regime:");
        cur.write_str(regime_str);
        cur.write_str(" | Conflict:");
        cur.write_u8_dec(self.conflict_severity_q7);
        cur.write_str("/127\"]\n");

        cur.write_str("    HUB[\"ActionHub | Apt:");
        cur.write_hex_u16(APERTURE_ACTION_HUB_WIDE);
        cur.write_str(" | VoidFloor:");
        cur.write_u8_dec(self.void_floor_q7[7]);
        cur.write_str("/127 | Crystals:");
        cur.write_u8_dec(self.active_crystals);
        cur.write_str(" | Daughters:");
        cur.write_u8_dec(self.spawned_daughters);
        cur.write_str("\"]\n");

        cur.write_str("    LVOID -. \"Cognitive Cut\" .-> N0 & N1 & N2 & N3 & N4 & N5 & N6\n");
        cur.write_str("    LVOID <-->|\"Cross-Layer Feedback\"| STIG\n");
        cur.write_str("    N0 & N1 & N2 & N3 & N4 & N5 & N6 --> GW\n");
        cur.write_str("    GW --> HUB\n");
        cur.write_str("    STIG -->|\"Frag / Drift / Reconstitute\"| HUB\n");
        cur.pos
    }

    /// Renders this system snapshot as a `#![no_std]` zero-allocation Mermaid `flowchart TD` diagram
    /// directly into `buf`, returning the exact byte count written.
    #[inline]
    pub fn format_mermaid_flowchart(&self, buf: &mut [u8]) -> usize {
        self.format_mermaid_flowchart_core(None, buf)
    }

    /// Renders both Core Layer 0 (`LiminalVoid`) and Core Layer 1 (`StigmergicGranuleField` + Venn
    /// overlap & serendipity telemetry) as a `#![no_std]` zero-allocation Mermaid `flowchart TD` diagram.
    #[inline]
    pub fn format_core_layers_mermaid(
        &self,
        layer1: &StigmergicGranuleField,
        venn: &StigmergicVennSerendipityReport,
        buf: &mut [u8],
    ) -> usize {
        self.format_mermaid_flowchart_core(Some((layer1, venn)), buf)
    }
}

// ===================================================================
// Venn-Diagram Overlap, Noise-Driven Serendipity & 2-Layer Autonomous Core Stack
// ===================================================================

/// Branchless 32-byte (`#[repr(C, align(16))]`) report capturing Venn-diagram overlap across the
/// 8 SWAR granules (`venn_core_mask`, `venn_penumbra_mask`) and serendipitous approximation-noise
/// excursions into [`LiminalVoid`] (`serendipity_excursion_mask`, `serendipity_triggered`).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct StigmergicVennSerendipityReport {
    /// Cooperative resonance energy accumulated across granules covered by `>= 2` attractors (`offset 0..4`).
    pub venn_intersection_energy: f32,
    /// Attenuated fringe energy across granules covered by exactly `1` attractor (`offset 4..8`).
    pub penumbra_attenuation_energy: f32,
    /// Residual noise excursion energy outside all attractors' apertures (`offset 8..12`).
    pub void_excursion_energy: f32,
    /// Dominant attractor anchor ID participating in the highest-overlap Venn region (`offset 12..16`).
    pub dominant_overlap_anchor_id: u32,
    /// Packed 8-head `u64` SWAR granule word of the strongest noisy excursion in `LiminalVoid` (`offset 16..24`).
    pub serendipitous_seed_word: u64,
    /// 8-bit mask where bit `g == 1` iff granule `g` is covered by `>= 2` attractors (`offset 24`).
    pub venn_core_mask: u8,
    /// 8-bit mask where bit `g == 1` iff granule `g` is covered by exactly `1` attractor (`offset 25`).
    pub venn_penumbra_mask: u8,
    /// 8-bit mask where bit `g == 1` iff granule `g` lies outside all attractors (`overlap == 0`, `offset 26`).
    pub void_exterior_mask: u8,
    /// 8-bit mask where bit `g == 1` iff granule `g` lies in `void_exterior_mask` with drift `>= floor` (`offset 27`).
    pub serendipity_excursion_mask: u8,
    /// Maximum number of attractors overlapping on any single granule (`offset 28`).
    pub max_overlap_depth: u8,
    /// Granule index (`0..=7`) hosting `serendipitous_seed_word` (`offset 29`).
    pub serendipitous_granule_idx: u8,
    /// `1` if approximation noise in `LiminalVoid` triggered unknown attractor discovery, `0` otherwise (`offset 30`).
    pub serendipity_triggered: u8,
    /// Explicit 1-byte alignment padding (`offset 31`).
    pub reserved_align: u8,
}

/// Evaluates Venn-diagram multi-attractor cognitive aperture overlap and noise-driven serendipity
/// in [`LiminalVoid`] across the 8 SWAR granules (`0` allocations, `0` data-dependent branches).
#[inline]
#[must_use]
pub fn evaluate_stigmergic_venn_and_serendipity_words_64<Node: RelativeAttractorNode>(
    nodes: &[Node],
    query_words: &[u64; NUM_STIGMERGIC_GRANULES],
    noisy_drifted_words: &[u64; NUM_STIGMERGIC_GRANULES],
    aperture_floor: f32,
    serendipity_floor_q7: u8,
) -> StigmergicVennSerendipityReport {
    let safe_floor = branchless_max_f32(branchless_relu_f32(aperture_floor), DEFAULT_ENTROPY_EPS);
    let safe_serendipity_floor = branchless_select_u8(
        serendipity_floor_q7 < 4,
        4,
        branchless_select_u8(serendipity_floor_q7 > 120, 120, serendipity_floor_q7),
    );

    let mut overlap_counts = [0u8; NUM_STIGMERGIC_GRANULES];
    let mut granule_energy = [0.0_f32; NUM_STIGMERGIC_GRANULES];
    let mut granule_best_pull = [0.0_f32; NUM_STIGMERGIC_GRANULES];
    let mut granule_best_anchor = [0u32; NUM_STIGMERGIC_GRANULES];

    let mut idx = 0usize;
    while idx < nodes.len() {
        let node = &nodes[idx];
        let e = branchless_relu_f32(node.relative_energy());
        let nw = node.relative_phase_words_64();
        let aid = node.relative_anchor_id();
        let active_e = e > DEFAULT_ENTROPY_EPS;

        let mut g = 0usize;
        while g < NUM_STIGMERGIC_GRANULES {
            let coh = coherence_from_uniform_words_x8(noisy_drifted_words[g], nw[g]);
            let covered = active_e & (coh >= safe_floor);
            overlap_counts[g] = overlap_counts[g].saturating_add(u8::from(covered));
            let pull_g = branchless_select_f32(covered, branchless_relu_f32(coh) * e, 0.0);
            granule_energy[g] += pull_g;
            let better = covered & (pull_g > granule_best_pull[g]);
            granule_best_pull[g] = branchless_select_f32(better, pull_g, granule_best_pull[g]);
            granule_best_anchor[g] = branchless_select_u32(better, aid, granule_best_anchor[g]);
            g += 1;
        }
        idx += 1;
    }

    let mut venn_intersection_energy = 0.0_f32;
    let mut penumbra_attenuation_energy = 0.0_f32;
    let mut void_excursion_energy = 0.0_f32;
    let mut dominant_overlap_anchor_id = 0u32;
    let mut best_core_score = 0.0_f32;
    let mut best_any_anchor_id = 0u32;
    let mut best_any_pull = 0.0_f32;
    let mut serendipitous_seed_word = 0u64;
    let mut venn_core_mask = 0u8;
    let mut venn_penumbra_mask = 0u8;
    let mut void_exterior_mask = 0u8;
    let mut serendipity_excursion_mask = 0u8;
    let mut max_overlap_depth = 0u8;
    let mut best_serendip_q7 = 0u8;
    let mut serendipitous_granule_idx = 0u8;
    let mut serendipity_triggered = 0u8;

    let mut g = 0usize;
    while g < NUM_STIGMERGIC_GRANULES {
        let ov = overlap_counts[g];
        let bit = 1u8 << g;
        let is_core = ov >= 2;
        let is_penumbra = ov == 1;
        let is_void = ov == 0;

        venn_core_mask |= branchless_select_u8(is_core, bit, 0);
        venn_penumbra_mask |= branchless_select_u8(is_penumbra, bit, 0);
        void_exterior_mask |= branchless_select_u8(is_void, bit, 0);

        let depth_gt = ov > max_overlap_depth;
        max_overlap_depth = branchless_select_u8(depth_gt, ov, max_overlap_depth);

        let coop_factor = 1.0 + 0.35 * (f32::from(ov.saturating_sub(1)));
        let core_e = branchless_select_f32(is_core, granule_energy[g] * coop_factor, 0.0);
        let pen_e = branchless_select_f32(is_penumbra, granule_energy[g] * 0.5, 0.0);
        venn_intersection_energy += core_e;
        penumbra_attenuation_energy += pen_e;

        let take_core_dom = is_core & (core_e > best_core_score);
        best_core_score = branchless_select_f32(take_core_dom, core_e, best_core_score);
        dominant_overlap_anchor_id = branchless_select_u32(
            take_core_dom,
            granule_best_anchor[g],
            dominant_overlap_anchor_id,
        );
        let take_any_dom = (!is_void) & (granule_best_pull[g] > best_any_pull);
        best_any_pull = branchless_select_f32(take_any_dom, granule_best_pull[g], best_any_pull);
        best_any_anchor_id =
            branchless_select_u32(take_any_dom, granule_best_anchor[g], best_any_anchor_id);

        let drift_word = swar_q7_circular_diff_x8(noisy_drifted_words[g], query_words[g]);
        let drift_q7 = swar_u8_mean_round_x8(swar_q7_abs_u8_x8(drift_word));
        let serendip = is_void & (drift_q7 >= safe_serendipity_floor);
        serendipity_excursion_mask |= branchless_select_u8(serendip, bit, 0);
        void_excursion_energy +=
            branchless_select_f32(is_void, f32::from(drift_q7) * (1.0 / 127.0), 0.0);

        let take_serendip = serendip & (drift_q7 > best_serendip_q7);
        best_serendip_q7 = branchless_select_u8(take_serendip, drift_q7, best_serendip_q7);
        serendipitous_seed_word = branchless_select_u64(
            take_serendip,
            noisy_drifted_words[g],
            serendipitous_seed_word,
        );
        serendipitous_granule_idx =
            branchless_select_u8(take_serendip, g as u8, serendipitous_granule_idx);
        serendipity_triggered |= u8::from(serendip);
        g += 1;
    }

    let final_dom_id = branchless_select_u32(
        venn_core_mask != 0,
        dominant_overlap_anchor_id,
        best_any_anchor_id,
    );

    StigmergicVennSerendipityReport {
        venn_intersection_energy,
        penumbra_attenuation_energy,
        void_excursion_energy,
        dominant_overlap_anchor_id: final_dom_id,
        serendipitous_seed_word,
        venn_core_mask,
        venn_penumbra_mask,
        void_exterior_mask,
        serendipity_excursion_mask,
        max_overlap_depth,
        serendipitous_granule_idx,
        serendipity_triggered,
        reserved_align: 0,
    }
}

/// Autonomous Core Layer 1 controller (`#[repr(C, align(16))]`, 32 B) governing per-granule
/// stigmergic approximation parameters, Venn-overlap reinforcement, and serendipitous noise discovery.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StigmergicGranuleField {
    /// Per-granule pheromone step gain in Q0.7 (`[32..=127]`, `offset 0..8`).
    pub granule_step_gain_q7: [u8; NUM_STIGMERGIC_GRANULES],
    /// Per-granule tolerated phase approximation residual in Q7 (`[2..=32]`, `offset 8..16`).
    pub granule_noise_tolerance_q7: [u8; NUM_STIGMERGIC_GRANULES],
    /// Per-granule local terrain pheromone density in Q0.7 (`[8..=127]`, `offset 16..24`).
    pub granule_pheromone_q7: [u8; NUM_STIGMERGIC_GRANULES],
    /// Active Venn overlap core mask across the 8 SWAR granules (`offset 24`).
    pub venn_overlap_mask: u8,
    /// Mask of granules whose approximation noise reached `LiminalVoid` discovery (`offset 25`).
    pub serendipity_excursion_mask: u8,
    /// Adaptive granule consensus quorum requirement (`4..=7` out of 8, `offset 26`).
    pub quorum_threshold: u8,
    /// Q7 threshold above which noisy excursions in `LiminalVoid` nucleate a new attractor (`offset 27`).
    pub serendipity_floor_q7: u8,
    /// Monotonic count of Core Layer 1 self-adaptation epochs (`offset 28..32`).
    pub adaptation_epochs: u32,
}

impl Default for StigmergicGranuleField {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl StigmergicGranuleField {
    /// Constructs a neutral 32-byte [`StigmergicGranuleField`] baseline.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            granule_step_gain_q7: [96; NUM_STIGMERGIC_GRANULES],
            granule_noise_tolerance_q7: [4; NUM_STIGMERGIC_GRANULES],
            granule_pheromone_q7: [32; NUM_STIGMERGIC_GRANULES],
            venn_overlap_mask: 0,
            serendipity_excursion_mask: 0,
            quorum_threshold: DEFAULT_STIGMERGIC_QUORUM_THRESHOLD,
            serendipity_floor_q7: 24,
            adaptation_epochs: 0,
        }
    }

    /// Branchlessly adapts per-granule pheromone gains, tolerated approximation noise, and
    /// serendipity discovery floors from `horizon`, `venn_report`, and `ledger`.
    #[inline]
    pub fn adapt_from_horizon_and_venn_branchless(
        &mut self,
        horizon: &EpistemicHorizonReading,
        venn_report: &StigmergicVennSerendipityReport,
        ledger: &CognitiveTrajectoryLedger,
    ) {
        let (op_hist, _depth) = ledger.op_counts();
        let peel_dom = op_hist[CognitiveOpKind::DecomposePeel as usize]
            > op_hist[CognitiveOpKind::SynthesizeFuse as usize];
        let mut g = 0usize;
        while g < NUM_STIGMERGIC_GRANULES {
            let bit = 1u8 << g;
            let in_core = (venn_report.venn_core_mask & bit) != 0;
            let in_penumbra = (venn_report.venn_penumbra_mask & bit) != 0;
            let in_serendip = (venn_report.serendipity_excursion_mask & bit) != 0;

            let sg0 = self.granule_step_gain_q7[g];
            let sg_core = branchless_select_u8(in_core, sg0.saturating_add(6).min(127), sg0);
            let sg_ser =
                branchless_select_u8(in_serendip, sg_core.saturating_add(8).min(127), sg_core);
            self.granule_step_gain_q7[g] = branchless_select_u8(
                !in_core & !in_penumbra & !in_serendip,
                sg_ser.saturating_sub(2).max(32),
                sg_ser,
            );

            // High Venn overlap tolerates coarser low-spec noise; edge penumbra tightens precision.
            let nt0 = self.granule_noise_tolerance_q7[g];
            let nt_core = branchless_select_u8(in_core, nt0.saturating_add(2).min(28), nt0);
            let nt_pen =
                branchless_select_u8(in_penumbra, nt_core.saturating_sub(1).max(3), nt_core);
            self.granule_noise_tolerance_q7[g] =
                branchless_select_u8(in_serendip, nt_pen.saturating_add(2).min(24), nt_pen);

            let ph0 = self.granule_pheromone_q7[g];
            let ph_core = branchless_select_u8(in_core, ph0.saturating_add(8).min(127), ph0);
            let ph_pen =
                branchless_select_u8(in_penumbra, ph_core.saturating_add(2).min(127), ph_core);
            let ph_ser = branchless_select_u8(in_serendip, ph_pen.max(48), ph_pen);
            self.granule_pheromone_q7[g] = branchless_select_u8(
                !in_core & !in_penumbra & !in_serendip,
                ph_ser.saturating_sub(4).max(8),
                ph_ser,
            );
            g += 1;
        }

        let core_cnt = venn_report.venn_core_mask.count_ones() as u8;
        let in_conflict = horizon.is_internal_conflict();
        self.quorum_threshold = branchless_select_u8(
            in_conflict,
            self.quorum_threshold.saturating_add(1).min(7),
            branchless_select_u8(
                core_cnt >= 4,
                self.quorum_threshold.saturating_sub(1).max(4),
                self.quorum_threshold,
            ),
        );
        self.serendipity_floor_q7 = branchless_select_u8(
            in_conflict,
            self.serendipity_floor_q7.saturating_add(4).min(64),
            branchless_select_u8(
                peel_dom & (core_cnt < 3),
                self.serendipity_floor_q7.saturating_sub(2).max(12),
                self.serendipity_floor_q7,
            ),
        );
        self.venn_overlap_mask = venn_report.venn_core_mask;
        self.serendipity_excursion_mask = venn_report.serendipity_excursion_mask;
        self.adaptation_epochs = self.adaptation_epochs.wrapping_add(1);
    }
}

/// Unified 64-byte (1 cache line, `#[repr(C, align(64))]`) 2-layer autonomous core stack pairing:
/// - **Layer 0 ([`LiminalApertureField`])**: Epistemic Horizon & Trait Apertures (`Void` boundary)
/// - **Layer 1 ([`StigmergicGranuleField`])**: Stigmergic Fragmentation, Venn Overlap & Serendipity
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AutonomousCoreLayerStack {
    /// Core Layer 0: Epistemic Horizon & Trait Cognitive Aperture Controller (`offset 0..32`).
    pub liminal_void_layer: LiminalApertureField,
    /// Core Layer 1: Stigmergic Granule Approximation & Serendipity Controller (`offset 32..64`).
    pub stigmergic_layer: StigmergicGranuleField,
}

impl AutonomousCoreLayerStack {
    /// Constructs a neutral 64-byte [`AutonomousCoreLayerStack`].
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            liminal_void_layer: LiminalApertureField::new(),
            stigmergic_layer: StigmergicGranuleField::new(),
        }
    }

    /// Branchlessly co-adapts Layer 0 (`LiminalApertureField`) and Layer 1 (`StigmergicGranuleField`)
    /// with cross-layer feedback (serendipitous discovery in Layer 1 widens Layer 0's horizon).
    #[inline]
    pub fn adapt_layers_branchless(
        &mut self,
        ledger: &CognitiveTrajectoryLedger,
        horizon: &EpistemicHorizonReading,
        venn_report: &StigmergicVennSerendipityReport,
        overflow_opt: Option<&DissipativeOverflowReport>,
    ) {
        self.liminal_void_layer
            .adapt_from_circuit_and_overflow_branchless(ledger, horizon, overflow_opt);
        let discovered = venn_report.serendipity_triggered != 0;
        self.liminal_void_layer.void_floor_q7[1] = branchless_select_u8(
            discovered,
            self.liminal_void_layer.void_floor_q7[1]
                .saturating_sub(4)
                .max(8),
            self.liminal_void_layer.void_floor_q7[1],
        );
        self.liminal_void_layer.void_floor_q7[6] = branchless_select_u8(
            discovered,
            self.liminal_void_layer.void_floor_q7[6]
                .saturating_sub(4)
                .max(8),
            self.liminal_void_layer.void_floor_q7[6],
        );
        self.liminal_void_layer.aperture_gain_q7[6] = branchless_select_u8(
            discovered,
            self.liminal_void_layer.aperture_gain_q7[6]
                .saturating_add(6)
                .min(127),
            self.liminal_void_layer.aperture_gain_q7[6],
        );
        self.liminal_void_layer.active_aperture_mask |=
            branchless_select_u32(discovered, (1 << 1) | (1 << 6), 0);

        self.stigmergic_layer
            .adapt_from_horizon_and_venn_branchless(horizon, venn_report, ledger);
    }

    /// Branchlessly modulates `frame` across both autonomous core layers (`LiminalVoid` gating +
    /// Venn-overlap cooperative amplification + active granule mask).
    #[inline(always)]
    #[must_use]
    pub fn modulate_gateway_frame_branchless(
        &self,
        frame: GatewayFrame,
        horizon: &EpistemicHorizonReading,
        venn_report: &StigmergicVennSerendipityReport,
    ) -> GatewayFrame {
        let mut out = self
            .liminal_void_layer
            .modulate_gateway_frame_branchless(frame, horizon);
        let core_cnt = venn_report.venn_core_mask.count_ones() as f32;
        let has_venn = (venn_report.venn_core_mask | venn_report.venn_penumbra_mask) != 0;
        let coop_boost = branchless_clamp_f32(1.0 + 0.05 * core_cnt, 1.0, 1.40);
        let void_gated = horizon.is_liminal_void()
            | (horizon.total_memory_support < self.liminal_void_layer.channel_void_floor_f32(1));
        out.effective_energy = branchless_select_f32(
            !void_gated & (venn_report.venn_core_mask != 0),
            out.effective_energy * coop_boost,
            out.effective_energy,
        );
        let active_granules =
            venn_report.venn_core_mask | venn_report.venn_penumbra_mask | venn_report.serendipity_excursion_mask;
        out.active_swarm_cell_mask =
            branchless_select_u8(has_venn, active_granules, out.active_swarm_cell_mask);
        out
    }
}
