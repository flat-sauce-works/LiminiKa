// name: src/core/src/traits/gateway/epistemic.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Branchless epistemic horizon separation (`KnownCoherent` vs. `InternalConflict` vs. `Liminal`),
//! signed relative attractor coupling ($W_{ij} \in [-127, +127]$), autonomous dynamic
//! [`LiminalApertureField`] horizon shifting, and `#![no_std]` self-introspecting Mermaid topology
//! telemetry over [`ManifoldGatewaySignal`].
//!
//! Rather than hard-coding static `Fact`/`Guard` roles or branch-heavy conditionals, every
//! attractor participates in a signed relational graph from the active perspective $i$:
//! - $W_{ij} > 0$ projects node $j$ onto the attractive basin axis;
//! - $W_{ij} < 0$ projects node $j$ onto the opposing/repulsive axis;
//! - When total projection energy onto the local memory cut falls below `horizon_floor`, the
//!   manifold enters [`EpistemicRegime::Liminal`] (*Limina* to the unknowable) and scales
//!   output steering energy cleanly to `0.0` without branching.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use super::*;
use super::core_layers::*;
use crate::crystal::DissipativeOverflowReport;
use crate::traits::branchless::{Liminal, branchless_q7_inhibit_words_64};
use crate::traits::primitives::branchless_sincos_q7;
use crate::traits::telemetry::NUM_STIGMERGIC_GRANULES;

/// Epistemic regime classification separating internal memory conflict from out-of-memory unknowability.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicRegime {
    /// Query falls inside the carved memory horizon with coherent dominant alignment ($H^0$).
    KnownCoherent = 0,
    /// Query falls inside the carved memory horizon with competing / opposing attractors ($H^1 \neq 0$).
    InternalConflict = 1,
    /// Query lies outside the carved memory horizon (*Limina* between cognition and the unknowable).
    Liminal = 2,
}

impl EpistemicRegime {
    /// Decodes a compact `u8` regime code (`0`, `1`, `2`) into [`EpistemicRegime`].
    #[inline(always)]
    #[must_use]
    pub const fn from_code(code: u8) -> Self {
        match code {
            1 => Self::InternalConflict,
            2 => Self::Liminal,
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
    /// Branchless amplitude gate (`1.0` inside cognitive horizon, `0.0` in [`EpistemicRegime::Liminal`]).
    pub horizon_gate: f32,
    /// Normalized internal contradiction severity in $[0.0, 1.0]$.
    pub conflict_severity: f32,
    /// Anchor ID of the strongest attractive/aligned node inside the cut (`0` if none).
    pub dominant_anchor_id: u32,
    /// Anchor ID of the strongest opposing/conflicting node inside the cut (`0` if none).
    pub opposing_anchor_id: u32,
    /// Encoded [`EpistemicRegime`] byte (`0` = Coherent, `1` = InternalConflict, `2` = Liminal).
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

    /// Returns `true` when the query lies outside the carved memory cut ([`EpistemicRegime::Liminal`]).
    #[inline(always)]
    #[must_use]
    pub const fn is_liminal(&self) -> bool {
        self.regime_code == EpistemicRegime::Liminal as u8
    }

    /// Returns `true` when the query lies inside the memory cut yet triggers internal opposition ($H^1 \neq 0$).
    #[inline(always)]
    #[must_use]
    pub const fn is_internal_conflict(&self) -> bool {
        self.regime_code == EpistemicRegime::InternalConflict as u8
    }

    /// Modulates any synthesized [`GatewayFrame`] strictly branchlessly according to this epistemic
    /// horizon: zeroes amplitude in [`EpistemicRegime::Liminal`] and elevates repulsive flags in
    /// [`EpistemicRegime::InternalConflict`].
    #[inline(always)]
    #[must_use]
    pub fn apply_to_frame_branchless(&self, mut frame: GatewayFrame) -> GatewayFrame {
        let is_void = self.is_liminal();
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
            while w < NUM_STIGMERGIC_GRANULES {
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
        let is_void = self.is_liminal();
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
            self.is_liminal(),
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
                self.is_liminal(),
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
/// atop [`Liminal`].
pub trait RelativeAttractorNode: Liminal {
    /// Static cognitive aperture mask for relative attractor participation.
    const RELATIVE_APERTURE_MASK: u32 = APERTURE_RELATIVE_COUPLING;
    /// Returns the node's 64-head Q7 phase profile packed into 8 register-resident `u64` words.
    fn relative_phase_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES];

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
    fn relative_phase_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
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
    fn relative_phase_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
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
    fn relative_phase_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
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
    fn relative_phase_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
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
    fn relative_phase_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
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
    fn relative_phase_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
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
    fn relative_phase_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
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
    fn relative_phase_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
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
    fn relative_phase_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
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
    fn relative_phase_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
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
    nw: &[u64; NUM_STIGMERGIC_GRANULES],
    weight: f32,
    sin_acc: &mut [f32; MAX_ATTENTION_HEADS],
    cos_acc: &mut [f32; MAX_ATTENTION_HEADS],
) {
    let mut cell = 0usize;
    while cell < NUM_STIGMERGIC_GRANULES {
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
) -> ([u64; NUM_STIGMERGIC_GRANULES], f32, u32) {
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
    fn relative_phase_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
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
    fn relative_phase_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
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
    query_words: &[u64; NUM_STIGMERGIC_GRANULES],
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
        EpistemicRegime::Liminal as u8,
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
fn fold_sincos64_to_words(sin_acc: &[f32; MAX_ATTENTION_HEADS], cos_acc: &[f32; MAX_ATTENTION_HEADS]) -> [u64; NUM_STIGMERGIC_GRANULES] {
    let mut words = [0u64; NUM_STIGMERGIC_GRANULES];
    let mut cell = 0usize;
    while cell < NUM_STIGMERGIC_GRANULES {
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
/// `query_words` falls into [`EpistemicRegime::Liminal`] (`< horizon_floor`), steering amplitude
/// is scaled cleanly to `0.0`.
#[inline]
#[must_use]
pub fn synthesize_relative_gateway_from_perspective_words_64<Node: RelativeAttractorNode>(
    nodes: &[Node],
    coupling_row_q7: &[i8],
    query_words: &[u64; NUM_STIGMERGIC_GRANULES],
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
        while cell < NUM_STIGMERGIC_GRANULES {
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
    let mut final_w = [0u64; NUM_STIGMERGIC_GRANULES];
    let mut c = 0usize;
    while c < NUM_STIGMERGIC_GRANULES {
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
        query_words: &[u64; NUM_STIGMERGIC_GRANULES],
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
    fn fold_slice_as_fractal_node(slice: &[Self]) -> ([u64; NUM_STIGMERGIC_GRANULES], f32, u32)
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
        query_words: &[u64; NUM_STIGMERGIC_GRANULES],
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
    fn relative_projection_energy(&self, query_words: &[u64; NUM_STIGMERGIC_GRANULES]) -> f32 {
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
    /// [`Liminal`] across a slice of `Self` (`0` allocations, `0` branches).
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
// Dynamic Liminal Aperture Shifting & Self-Introspecting Telemetry
// ===================================================================

/// Dynamic, branchlessly adaptable cognitive aperture controller (`#[repr(C, align(16))]`, 32 B)
/// shifting the [`Liminal`] boundary (`void_floor_q7`, `aperture_gain_q7`, `phase_shift_q7`)
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
    /// Per-channel `Liminal` threshold floor in Q0.7 (`[8..=120]`): higher values expand the
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

    /// Returns channel `ch`'s normalized `Liminal` threshold floor in `[0.0, 1.0]`.
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

    /// Branchlessly adapts each channel's dynamic `Liminal` boundary from `ledger`, `horizon`,
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
        let void_active = horizon.is_liminal();

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

        // Channel 2 (SubjectivePersona): push into Liminal during peel/conflict; open under fuse.
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
    /// enforces the dynamic Fact `Liminal` floor, scales Fact/Logic/Explore sub-head gains,
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
        let is_void = horizon.is_liminal() | below_fact_floor;

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
            while c < NUM_STIGMERGIC_GRANULES {
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
    /// Human-readable Rust trait name refining [`Liminal`].
    pub trait_name: &'static str,
    /// Channel index (`0..8`) inside [`LiminalApertureField`].
    pub channel_idx: usize,
    /// Static cognitive aperture mask owned by this trait node.
    pub static_aperture_mask: u32,
    /// Type-level unknowable exterior complement (`!static_aperture_mask`).
    pub unknowable_void_mask: u32,
}

/// Canonical 8-channel trait topology table derived from type-level [`Liminal`] refinements.
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
    /// Per-channel `Liminal` floor values in Q0.7 (`0..=127`).
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
        layer2_opt: Option<&LiminalEgoField>,
        buf: &mut [u8],
    ) -> usize {
        let mut cur = BufCursor { buf, pos: 0 };
        let regime_str = match EpistemicRegime::from_code(self.regime_code) {
            EpistemicRegime::KnownCoherent => "KnownCoherent",
            EpistemicRegime::InternalConflict => "InternalConflict",
            EpistemicRegime::Liminal => "Liminal",
        };
        cur.write_str("flowchart TD\n");
        cur.write_str("    LVOID[\"Core Layer 0: Liminal (Epistemic Horizon & Exterior)\"]\n");

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
                cur.write_hex_u16(APERTURE_STIGMERGIC_CONSENSUS);
                cur.write_str("\"]\n");
            }
        }

        if let Some(ego) = layer2_opt {
            cur.write_str("    EGO[\"Core Layer 2: LiminalEgoField | Thrust:");
            cur.write_u8_dec(ego.counter_vortex_thrust_q7);
            cur.write_str("/127 | Transgress:");
            cur.write_i8_dec(ego.transgression_bias_q7);
            cur.write_str(" | Fired:");
            cur.write_hex_u8(ego.last_intervened_mask);
            cur.write_str(" | Epochs:");
            cur.write_u8_dec((ego.intervention_epochs & 0xFF) as u8);
            cur.write_str("\"]\n");
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
        if layer2_opt.is_some() {
            cur.write_str("    EGO -. \"Top-Down Intervention\" .-> LVOID & STIG\n");
            cur.write_str("    EGO -->|\"Counter-Vortex / Defiance\"| GW\n");
        }
        cur.write_str("    N0 & N1 & N2 & N3 & N4 & N5 & N6 --> GW\n");
        cur.write_str("    GW --> HUB\n");
        cur.write_str("    STIG -->|\"Frag / Drift / Reconstitute\"| HUB\n");
        cur.pos
    }

    /// Renders this system snapshot as a `#![no_std]` zero-allocation Mermaid `flowchart TD` diagram
    /// directly into `buf`, returning the exact byte count written.
    #[inline]
    pub fn format_mermaid_flowchart(&self, buf: &mut [u8]) -> usize {
        self.format_mermaid_flowchart_core(None, None, buf)
    }

    /// Renders Core Layer 0 (`Liminal`) and Core Layer 1 (`StigmergicGranuleField` + Venn
    /// overlap & serendipity telemetry) as a `#![no_std]` zero-allocation Mermaid `flowchart TD` diagram.
    #[inline]
    pub fn format_core_layers_mermaid(
        &self,
        layer1: &StigmergicGranuleField,
        venn: &StigmergicVennSerendipityReport,
        buf: &mut [u8],
    ) -> usize {
        self.format_mermaid_flowchart_core(Some((layer1, venn)), None, buf)
    }

    /// Renders all three autonomous core layers (`Liminal` Layer 0, `StigmergicGranuleField`
    /// Layer 1, and `LiminalEgoField` Layer 2) alongside Venn overlap & serendipity telemetry as a
    /// `#![no_std]` zero-allocation Mermaid `flowchart TD` diagram.
    #[inline]
    pub fn format_three_core_layers_mermaid(
        &self,
        stack: &AutonomousCoreLayerStack,
        venn: &StigmergicVennSerendipityReport,
        buf: &mut [u8],
    ) -> usize {
        self.format_mermaid_flowchart_core(
            Some((&stack.stigmergic_layer, venn)),
            Some(&stack.ego_layer),
            buf,
        )
    }
}
