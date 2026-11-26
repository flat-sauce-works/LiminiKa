// name: src/core/src/runtime/horizon/epistemic.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Epistemic Horizon separation (`KnownCoherent` / `InternalConflict` / `Liminal`),
//! relative perspective Gateway synthesis, and dynamic threshold/granule horizon bridges.

use crate::foundation::{
    ACTION_MODE_PHASE_REPULSE, ACTION_MODE_SUBHEAD_ROUTE, ACTION_PRESET_EXPRESS_PHASE,
    ACTION_PRESET_PASSIVE_SIDECAR, ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_ID_MASK,
    DEFAULT_PHASE_STEP_MIN_RAD, SWAR_SIGN_MASK_64, SWAR_SPLAT_BYTE_64, TaggedPtr,
    branchless_clamp_f32, branchless_inv_pos_f32, branchless_max_f32, branchless_min_f32,
    branchless_q7_inhibit_words_64, branchless_rad_to_phase_step_u8, branchless_relu_f32,
    branchless_sincos_q7, coherence_from_word_arrays_with_both_uniform_flags_64,
    fast_mean_q7_tag_from_words_64, is_word_array_uniform_64, pack_action_slot_tag,
    swar_q7_circular_add_x8, swar_q7_mean_i8_from_words_64,
};
use crate::liminal::{
    BitCut32, DEFAULT_ENTROPY_EPS, EpistemicRegime, LiminalThresholdState, MAX_ATTENTION_HEADS,
    NUM_LIMINAL_THRESHOLD_CHANNELS, NUM_STIGMERGIC_GRANULES, THRESHOLD_STIGMERGY_WIDE,
    branchless_select_f32, branchless_select_i8, branchless_select_u8, branchless_select_u32,
    branchless_select_u64,
};
use crate::runtime::gateway::{
    CAUSAL_AXIS_CONTRADICTION_GUARD, CAUSAL_AXIS_OBJECTIVE_FACT, CAUSAL_CAUSE_COHOMOLOGY_REPULSION,
    CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK, CausalProvenanceSignal, GatewayFrame, ManifoldGatewaySignal,
    ROOT_CAUSE_COHOMOLOGY_REPULSION, ROOT_CAUSE_FACT_RETRIEVAL_MISS, ROOT_CAUSE_NOMINAL,
};
use crate::stigmergy::{
    ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_REPULSION, CognitiveOpKind, CognitiveTrajectoryLedger,
    DEFAULT_FACT_PHASE_CLAMP_RAD, DissipativeOverflowReport, MAX_ATTRACTOR_PULL_FORCE,
    OverlapDiscoveryReport, RelativeAttractorNode, StigmergicGranuleState,
    evaluate_stigmergic_overlap_and_discovery_words_64, fold_fractal_nodes_words_64,
    fold_sincos64_to_words,
};

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

    /// Modulates any synthesized [`GatewayFrame`] strictly branchlessly in-place according to this
    /// epistemic horizon: zeroes amplitude in [`EpistemicRegime::Liminal`] and elevates repulsive
    /// flags in [`EpistemicRegime::InternalConflict`].
    #[inline(always)]
    pub fn apply_to_frame_branchless(&self, frame: &mut GatewayFrame) {
        let is_unobserved = self.is_liminal();
        let is_conflict = self.is_internal_conflict();
        let gate = branchless_clamp_f32(branchless_relu_f32(self.horizon_gate), 0.0, 1.0);

        frame.effective_energy =
            branchless_select_f32(is_unobserved, 0.0, frame.effective_energy * gate);
        frame.repulsion_severity = branchless_select_f32(
            is_unobserved,
            0.0,
            branchless_select_f32(
                is_conflict,
                branchless_max_f32(frame.repulsion_severity, self.conflict_severity),
                frame.repulsion_severity,
            ),
        );

        let high_repulse = (!is_unobserved) & is_conflict & (self.conflict_severity > 0.50);
        frame.action_mode_flags = branchless_select_u8(
            is_unobserved,
            ACTION_PRESET_PASSIVE_SIDECAR,
            branchless_select_u8(
                high_repulse,
                frame.action_mode_flags | ACTION_MODE_PHASE_REPULSE,
                frame.action_mode_flags,
            ),
        );
        frame.root_cause_code = branchless_select_u8(
            is_unobserved,
            ROOT_CAUSE_FACT_RETRIEVAL_MISS,
            branchless_select_u8(
                high_repulse,
                ROOT_CAUSE_COHOMOLOGY_REPULSION,
                frame.root_cause_code,
            ),
        );
        frame.lineage_flags = branchless_select_u32(
            is_unobserved,
            BitCut32::split(frame.lineage_flags, CAUSAL_CAUSE_COHOMOLOGY_REPULSION).outside
                | CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK,
            branchless_select_u32(
                high_repulse,
                frame.lineage_flags | CAUSAL_CAUSE_COHOMOLOGY_REPULSION,
                frame.lineage_flags,
            ),
        );

        // Zero phase words branchlessly when outside the carved memory horizon.
        let keep_mask = branchless_select_u64(is_unobserved, 0, u64::MAX);
        {
            let pw = frame.phase_words_mut();
            let mut w = 0usize;
            while w < NUM_STIGMERGIC_GRANULES {
                pw[w] &= keep_mask;
                w += 1;
            }
        }
        frame.mean_phase_tag_q7 = branchless_select_i8(is_unobserved, 0, frame.mean_phase_tag_q7);
        let eff_slot = frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK;
        let ptr_tag = pack_action_slot_tag(eff_slot, frame.action_mode_flags);
        frame.frame_flags = ptr_tag as u16;
        frame.tagged_ptr = TaggedPtr::pack(
            frame.mapped_section_offset48,
            ptr_tag,
            frame.mean_phase_tag_q7,
        )
        .raw();
    }
}

impl CausalProvenanceSignal for EpistemicHorizonReading {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let is_unobserved = self.is_liminal();
        let is_conflict = self.is_internal_conflict();
        let high_repulse = is_conflict & (self.conflict_severity > 0.50);
        branchless_select_u32(
            is_unobserved,
            CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK,
            CAUSAL_AXIS_OBJECTIVE_FACT
                | branchless_select_u32(
                    high_repulse,
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
        let mut frame = base;
        self.apply_to_frame_branchless(&mut frame);
        frame
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
            (w_i32.unsigned_abs() as f32) * (1.0 / 128.0),
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
        internal_conflict_energy
            * branchless_inv_pos_f32(internal_support_energy + DEFAULT_ENTROPY_EPS),
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
        let mag = (w_i32.unsigned_abs() as f32) * (1.0 / 128.0);

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
    let min_step_q7 = branchless_rad_to_phase_step_u8(DEFAULT_PHASE_STEP_MIN_RAD);

    let mut frame = GatewayFrame {
        tagged_ptr: TaggedPtr::pack(0, packed_tag, mean_tag).raw(),
        target_anchor_id: target_id,
        effective_energy: eff_energy,
        phase_clamp_rad: DEFAULT_FACT_PHASE_CLAMP_RAD,
        phase_step_min_rad: DEFAULT_PHASE_STEP_MIN_RAD,
        effective_inv_temp: 1.0 + branchless_clamp_f32(eff_energy * 0.25, 0.0, 1.0),
        fact_persona_coherence: base_coh,
        repulsion_severity: norm_rep_sev,
        lineage_flags: horizon.causal_provenance_bits(),
        root_cause_code: branchless_select_u8(
            rep_dominates,
            ROOT_CAUSE_COHOMOLOGY_REPULSION,
            ROOT_CAUSE_NOMINAL,
        ),
        dominant_slot: dom_slot,
        action_mode_flags: mode_flags,
        active_swarm_cell_mask: 0xFF,
        phase_step_min_q7: min_step_q7,
        mean_phase_tag_q7: mean_tag,
        frame_flags: packed_tag as u16,
        ..GatewayFrame::default()
    };
    *frame.phase_words_mut() = final_w;
    horizon.apply_to_frame_branchless(&mut frame);
    (frame, horizon)
}

/// Blanket-implemented epistemic & relative-coupling trait over any [`RelativeAttractorNode`].
pub trait EpistemicManifoldOps: RelativeAttractorNode {
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

    /// Evaluates stigmergic multi-attractor overlap and excursion discovery in
    /// `Liminal` across a slice of `Self` (`0` allocations, `0` branches).
    #[inline(always)]
    fn evaluate_slice_overlap_and_discovery(
        slice: &[Self],
        query_words: &[u64; NUM_STIGMERGIC_GRANULES],
        noisy_drifted_words: &[u64; NUM_STIGMERGIC_GRANULES],
        aperture_floor: f32,
        discovery_floor_q7: u8,
    ) -> OverlapDiscoveryReport
    where
        Self: Sized,
    {
        evaluate_stigmergic_overlap_and_discovery_words_64(
            slice,
            query_words,
            noisy_drifted_words,
            aperture_floor,
            discovery_floor_q7,
        )
    }
}

impl<T: RelativeAttractorNode + ?Sized> EpistemicManifoldOps for T {}

impl LiminalThresholdState {
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
        let liminal_active = horizon.is_liminal();

        // Channel 1 (ObjectiveFact) & Channel 3 (ContradictionGuard): narrow aperture under peel/conflict.
        let next_f1 = branchless_select_u8(
            peel_dom,
            self.threshold_floor_q7[1].saturating_sub(4).max(8),
            self.threshold_floor_q7[1],
        );
        self.threshold_floor_q7[1] =
            branchless_select_u8(liminal_active, next_f1.saturating_add(4).min(120), next_f1);
        self.aperture_gain_q7[1] = branchless_select_u8(
            peel_dom,
            self.aperture_gain_q7[1].saturating_add(8).min(127),
            branchless_select_u8(
                liminal_active,
                self.aperture_gain_q7[1].saturating_sub(6).max(16),
                self.aperture_gain_q7[1],
            ),
        );

        let next_f3 = branchless_select_u8(
            peel_dom,
            self.threshold_floor_q7[3].saturating_sub(4).max(8),
            self.threshold_floor_q7[3],
        );
        self.threshold_floor_q7[3] =
            branchless_select_u8(liminal_active, next_f3.saturating_add(4).min(120), next_f3);
        self.aperture_gain_q7[3] = branchless_select_u8(
            peel_dom,
            self.aperture_gain_q7[3].saturating_add(8).min(127),
            branchless_select_u8(
                liminal_active,
                self.aperture_gain_q7[3].saturating_sub(6).max(16),
                self.aperture_gain_q7[3],
            ),
        );

        // Channel 2 (SubjectivePersona): push into Liminal during peel/conflict; open under fuse.
        let p2_floor_peel = branchless_select_u8(
            peel_dom,
            self.threshold_floor_q7[2].saturating_add(8).min(120),
            self.threshold_floor_q7[2],
        );
        let p2_floor_fuse = branchless_select_u8(
            only_fuse,
            p2_floor_peel.saturating_sub(6).max(8),
            p2_floor_peel,
        );
        self.threshold_floor_q7[2] = branchless_select_u8(
            liminal_active,
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
            liminal_active,
            p2_gain_fuse.saturating_sub(6).max(16),
            p2_gain_fuse,
        );

        // Channel 5 (KvCache): open under integrative fuse or scatter transitions.
        self.threshold_floor_q7[5] = branchless_select_u8(
            fuse_dom,
            self.threshold_floor_q7[5].saturating_sub(6).max(8),
            self.threshold_floor_q7[5],
        );
        self.aperture_gain_q7[5] = branchless_select_u8(
            fuse_dom,
            self.aperture_gain_q7[5].saturating_add(6).min(127),
            self.aperture_gain_q7[5],
        );

        // Channel 6 (RelativeCoupling): open under fuse/scatter/birth and track scattered phase shift.
        let widen_rel = only_fuse | scattered | spawned;
        self.threshold_floor_q7[6] = branchless_select_u8(
            widen_rel,
            self.threshold_floor_q7[6].saturating_sub(6).max(8),
            self.threshold_floor_q7[6],
        );
        self.aperture_gain_q7[6] = branchless_select_u8(
            widen_rel,
            self.aperture_gain_q7[6].saturating_add(6).min(127),
            self.aperture_gain_q7[6],
        );
        let blended_shift = ((i32::from(self.phase_shift_q7[6]) + i32::from(res_mean_q7)) >> 1)
            .clamp(-127, 127) as i8;
        self.phase_shift_q7[6] =
            branchless_select_i8(scattered | spawned, blended_shift, self.phase_shift_q7[6]);

        // Recompute active_aperture_mask branchlessly from channel floor saturation (< 96).
        let mut mask = BitCut32::split(THRESHOLD_STIGMERGY_WIDE, 0x0000_00FF).outside;
        let mut ch = 0usize;
        while ch < NUM_LIMINAL_THRESHOLD_CHANNELS {
            let is_open =
                (self.threshold_floor_q7[ch] < 96) & (!liminal_active | (ch == 0) | (ch == 7));
            mask |= branchless_select_u32(is_open, 1u32 << ch, 0);
            ch += 1;
        }
        self.active_aperture_mask = mask;
        self.adaptation_epochs = self.adaptation_epochs.wrapping_add(1);
    }

    /// Branchlessly modulates `frame` in-place with this [`LiminalThresholdState`] and `horizon`:
    /// enforces the dynamic Fact `Liminal` floor, scales Fact/Logic/Explore sub-head gains,
    /// and applies channel `7`'s focal phase shift when inside the horizon.
    #[inline(always)]
    pub fn modulate_gateway_frame_branchless(
        &self,
        frame: &mut GatewayFrame,
        horizon: &EpistemicHorizonReading,
    ) {
        horizon.apply_to_frame_branchless(frame);
        let below_fact_floor = horizon.total_memory_support < self.channel_threshold_floor_f32(1);
        let is_unobserved = horizon.is_liminal() | below_fact_floor;

        frame.effective_energy = branchless_select_f32(is_unobserved, 0.0, frame.effective_energy);

        let scale_gain = |base_q7: u8, ap_q7: u8| -> u8 {
            let scaled = (u16::from(base_q7) * (u16::from(ap_q7) + 32)) >> 7;
            scaled.clamp(16, 255) as u8
        };
        frame.subhead_fact_gain_q7 =
            scale_gain(frame.subhead_fact_gain_q7, self.aperture_gain_q7[1]);
        frame.subhead_logic_gain_q7 =
            scale_gain(frame.subhead_logic_gain_q7, self.aperture_gain_q7[3]);
        frame.subhead_explore_gain_q7 =
            scale_gain(frame.subhead_explore_gain_q7, self.aperture_gain_q7[2]);

        let shift_byte = self.phase_shift_q7[7] as u8 as u64;
        let shift_word = shift_byte.wrapping_mul(SWAR_SPLAT_BYTE_64);
        let keep_mask = branchless_select_u64(is_unobserved, 0, u64::MAX);
        {
            let pw = frame.phase_words_mut();
            let mut c = 0usize;
            while c < NUM_STIGMERGIC_GRANULES {
                let shifted = swar_q7_circular_add_x8(pw[c], shift_word);
                pw[c] = shifted & keep_mask;
                c += 1;
            }
        }
        frame.mean_phase_tag_q7 = swar_q7_mean_i8_from_words_64(frame.phase_words_ref());
        let eff_slot = frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK;
        let ptr_tag = pack_action_slot_tag(eff_slot, frame.action_mode_flags);
        frame.frame_flags = ptr_tag as u16;
        frame.tagged_ptr = TaggedPtr::pack(
            frame.mapped_section_offset48,
            ptr_tag,
            frame.mean_phase_tag_q7,
        )
        .raw();
    }
}

impl StigmergicGranuleState {
    /// Branchlessly adapts per-granule pheromone gains, tolerated approximation noise, and
    /// excursion discovery floors from `horizon`, `overlap_report`, and `ledger`.
    #[inline]
    pub fn adapt_from_horizon_and_overlap_branchless(
        &mut self,
        horizon: &EpistemicHorizonReading,
        overlap_report: &OverlapDiscoveryReport,
        ledger: &CognitiveTrajectoryLedger,
    ) {
        self.adapt_from_conflict_and_overlap_branchless(
            horizon.is_internal_conflict(),
            overlap_report,
            ledger,
        );
    }
}
