// name: src/core/src/traits/gateway/epistemic.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Branchless epistemic horizon separation (`KnownCoherent` vs. `InternalConflict` vs. `TrueVoid`)
//! and signed relative attractor coupling ($W_{ij} \in [-127, +127]$) over [`ManifoldGatewaySignal`].
//!
//! Rather than hard-coding static `Fact`/`Guard` roles or branch-heavy conditionals, every
//! attractor participates in a signed relational graph from the active perspective $i$:
//! - $W_{ij} > 0$ projects node $j$ onto the attractive basin axis;
//! - $W_{ij} < 0$ projects node $j$ onto the opposing/repulsive axis;
//! - When total projection energy onto the local memory cut falls below `horizon_floor`, the
//!   manifold enters [`EpistemicRegime::TrueVoid`] (*Limina* to the unknowable) and scales
//!   output steering energy cleanly to `0.0` without branching.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use super::*;
use crate::crystal::DissipativeOverflowReport;
use crate::traits::branchless::branchless_q7_inhibit_words_64;
use crate::traits::primitives::branchless_sincos_q7;

/// Epistemic regime classification separating internal memory conflict from out-of-memory unknowability.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicRegime {
    /// Query falls inside the carved memory horizon with coherent dominant alignment ($H^0$).
    KnownCoherent = 0,
    /// Query falls inside the carved memory horizon with competing / opposing attractors ($H^1 \neq 0$).
    InternalConflict = 1,
    /// Query lies outside the carved memory horizon (*Limina* between cognition and the unknowable).
    TrueVoid = 2,
}

impl EpistemicRegime {
    /// Decodes a compact `u8` regime code (`0`, `1`, `2`) into [`EpistemicRegime`].
    #[inline(always)]
    #[must_use]
    pub const fn from_code(code: u8) -> Self {
        match code {
            1 => Self::InternalConflict,
            2 => Self::TrueVoid,
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
    /// Branchless amplitude gate (`1.0` inside cognitive horizon, `0.0` in [`EpistemicRegime::TrueVoid`]).
    pub horizon_gate: f32,
    /// Normalized internal contradiction severity in $[0.0, 1.0]$.
    pub conflict_severity: f32,
    /// Anchor ID of the strongest attractive/aligned node inside the cut (`0` if none).
    pub dominant_anchor_id: u32,
    /// Anchor ID of the strongest opposing/conflicting node inside the cut (`0` if none).
    pub opposing_anchor_id: u32,
    /// Encoded [`EpistemicRegime`] byte (`0` = Coherent, `1` = InternalConflict, `2` = TrueVoid).
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

    /// Returns `true` when the query lies outside the carved memory cut ([`EpistemicRegime::TrueVoid`]).
    #[inline(always)]
    #[must_use]
    pub const fn is_true_void(&self) -> bool {
        self.regime_code == EpistemicRegime::TrueVoid as u8
    }

    /// Returns `true` when the query lies inside the memory cut yet triggers internal opposition ($H^1 \neq 0$).
    #[inline(always)]
    #[must_use]
    pub const fn is_internal_conflict(&self) -> bool {
        self.regime_code == EpistemicRegime::InternalConflict as u8
    }

    /// Modulates any synthesized [`GatewayFrame`] strictly branchlessly according to this epistemic
    /// horizon: zeroes amplitude in [`EpistemicRegime::TrueVoid`] and elevates repulsive flags in
    /// [`EpistemicRegime::InternalConflict`].
    #[inline(always)]
    #[must_use]
    pub fn apply_to_frame_branchless(&self, mut frame: GatewayFrame) -> GatewayFrame {
        let is_void = self.is_true_void();
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
        let is_void = self.is_true_void();
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
            self.is_true_void(),
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
                self.is_true_void(),
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
/// across any attractor representation (`CrystallizedAttractorRecord`, `GatewayFrame`, `ObjectiveFact`, etc.).
pub trait RelativeAttractorNode {
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
        EpistemicRegime::TrueVoid as u8,
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
/// `query_words` falls into [`EpistemicRegime::TrueVoid`] (`< horizon_floor`), steering amplitude
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
}

impl<T: RelativeAttractorNode + ?Sized> EpistemicManifoldOps for T {}
