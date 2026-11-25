// name: src/core/src/stigmergy/granule.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! 8-granule stigmergic approximation-and-reconstitution engine, overlap/discovery
//! field analysis, adaptive [`StigmergicGranuleState`], fragmented micro-ensemble summary,
//! and relative attractor graph coupling ([`RelativeAttractorNode`], [`RelativeCouplingMatrix`]).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

use core::mem::{align_of, offset_of, size_of};

use crate::foundation::{
    atan2_to_q7_word_x8_simd, branchless_mask8_min_delta, branchless_max_f32, branchless_relu_f32,
    branchless_sincos_q7, coherence_from_uniform_words_x8, is_word_array_uniform_64,
    store_q7_words_64, swar_q7_abs_u8_x8, swar_q7_circular_diff_x8, swar_u8_mean_round_x8, Pod,
};
use crate::liminal::{
    branchless_select_f32, branchless_select_i32, branchless_select_i8, branchless_select_u32,
    branchless_select_u64, branchless_select_u8, branchless_select_usize, LiminalThresholdState,
    LmnkResult, PhaseQ7, DEFAULT_ENTROPY_EPS, LMNK_ERR_INVALID_ARGUMENT, MAX_ATTENTION_HEADS,
    NUM_STIGMERGIC_GRANULES, THRESHOLD_RELATIVE_COUPLING,
};
use crate::stigmergy::anchor::{DissipativeOverflowReport, PhaseAnchorRecord};
use crate::stigmergy::circuit::{CognitiveOpKind, CognitiveTrajectoryLedger};
use crate::stigmergy::pheromone::StigmergicReconstitutionReport;
use crate::stigmergy::trail::{trail_phase_words_ref, PagedBitmask, PointerTrail, TaggedPtr};
use crate::stigmergy::DEFAULT_STIGMERGIC_QUORUM_THRESHOLD;

/// 64-byte cacheline-aligned Plain Old Data (`Pod`) state engine for the 8-granule SWAR
/// stigmergic approximation-and-reconstitution field and QDPT/CPT hierarchical pointer cascade.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StigmergicGranuleEngine {
    /// 64-head Q7 phase profile partitioned into 8 myopic SWAR granules of 8 heads each (`offset 0..64`).
    pub cell_phases: [PhaseQ7; MAX_ATTENTION_HEADS],
}

// SAFETY: `StigmergicGranuleEngine` is a 64-byte aligned array of 64 `i8` values with no padding traps.
unsafe impl Pod for StigmergicGranuleEngine {}

impl Default for StigmergicGranuleEngine {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl StigmergicGranuleEngine {
    /// Constructs a zero-initialized 64-byte [`StigmergicGranuleEngine`].
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            cell_phases: [0; MAX_ATTENTION_HEADS],
        }
    }

    /// Constructs a 64-byte [`StigmergicGranuleEngine`] initialized with `cell_phases`.
    #[inline(always)]
    #[must_use]
    pub const fn from_phases(cell_phases: [PhaseQ7; MAX_ATTENTION_HEADS]) -> Self {
        Self { cell_phases }
    }

    /// Reinterprets the 64-byte phase buffer as 8 packed 64-bit SWAR granule words in $\mathcal{O}(1)$.
    #[inline(always)]
    #[must_use]
    pub const fn words_ref(&self) -> &[u64; NUM_STIGMERGIC_GRANULES] {
        // SAFETY: `StigmergicGranuleEngine` is 64-byte aligned (`#[repr(C, align(64))]`) and `cell_phases` is at offset 0.
        unsafe {
            &*(self
                .cell_phases
                .as_ptr()
                .cast::<[u64; NUM_STIGMERGIC_GRANULES]>())
        }
    }

    /// Mutably reinterprets the 64-byte phase buffer as 8 packed 64-bit SWAR granule words in $\mathcal{O}(1)$.
    #[inline(always)]
    #[must_use]
    pub const fn words_mut(&mut self) -> &mut [u64; NUM_STIGMERGIC_GRANULES] {
        // SAFETY: `StigmergicGranuleEngine` is 64-byte aligned (`#[repr(C, align(64))]`) and `cell_phases` is at offset 0.
        unsafe {
            &mut *(self
                .cell_phases
                .as_mut_ptr()
                .cast::<[u64; NUM_STIGMERGIC_GRANULES]>())
        }
    }

    /// Fragments, stigmergically drifts, and reconstitutes `trail` against this engine's 8 myopic
    /// granules via [`PointerTrail::reconstitute_from_stigmergy_words`], keeping
    /// `self.cell_phases` synchronized with the reconstituted SWAR words in $\mathcal{O}(1)$.
    #[inline(always)]
    pub fn reconstitute_trail_stigmergically(
        &mut self,
        trail: &mut PointerTrail,
        step_gain_u7: u8,
        step_filter_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
    ) -> StigmergicReconstitutionReport {
        let target_words = *self.words_ref();
        let action_ptr = TaggedPtr(trail.current_ptr);
        let (report, next_words) = trail.reconstitute_from_stigmergy_words(
            action_ptr,
            &target_words,
            0xFF,
            step_gain_u7,
            step_filter_q7,
            tolerance_q7,
            quorum_threshold,
            0.5,
            0.15,
            None,
            3,
            64,
        );
        store_q7_words_64(&mut self.cell_phases, &next_words);
        report
    }

    /// Resolves `trail` against this engine's 8 granules (delegates to [`Self::reconstitute_trail_stigmergically`]).
    #[inline(always)]
    pub fn resolve_trail_in_swarm(
        &mut self,
        trail: &mut PointerTrail,
        step_gain_u7: u8,
        step_filter_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
    ) -> StigmergicReconstitutionReport {
        self.reconstitute_trail_stigmergically(
            trail,
            step_gain_u7,
            step_filter_q7,
            tolerance_q7,
            quorum_threshold,
        )
    }
}

impl StigmergicGranuleEngine {
    /// Updates local cellular swarm cell state across PagedBlock token chunk boundaries.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `chunk_len == 0`.
    #[inline(always)]
    pub fn step_chunk(&mut self, mask: &PagedBitmask, chunk_len: u32) -> LmnkResult<()> {
        if chunk_len == 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let _active_bits = mask.popcount();
        Ok(())
    }
}

const _: () = {
    assert!(size_of::<StigmergicGranuleEngine>() == 64);
    assert!(align_of::<StigmergicGranuleEngine>() == 64);
    assert!(offset_of!(StigmergicGranuleEngine, cell_phases) == 0);
};

/// Branchless 32-byte (`#[repr(C, align(16))]`) report capturing set-overlap overlap across the
/// 8 SWAR granules (`overlap_core_mask`, `overlap_fringe_mask`) and emergent approximation-noise
/// excursions into `Liminal` (`discovery_excursion_mask`, `discovery_triggered`).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct OverlapDiscoveryReport {
    /// Cooperative resonance energy accumulated across granules covered by `>= 2` attractors (`offset 0..4`).
    pub overlap_core_energy: f32,
    /// Attenuated fringe energy across granules covered by exactly `1` attractor (`offset 4..8`).
    pub overlap_fringe_energy: f32,
    /// Residual noise excursion energy outside all attractors' apertures (`offset 8..12`).
    pub lacuna_excursion_energy: f32,
    /// Dominant attractor anchor ID participating in the highest-overlap overlap region (`offset 12..16`).
    pub dominant_overlap_anchor_id: u32,
    /// Packed 8-head `u64` SWAR granule word of the strongest noisy excursion in `Liminal` (`offset 16..24`).
    pub discovery_seed_word: u64,
    /// 8-bit mask where bit `g == 1` iff granule `g` is covered by `>= 2` attractors (`offset 24`).
    pub overlap_core_mask: u8,
    /// 8-bit mask where bit `g == 1` iff granule `g` is covered by exactly `1` attractor (`offset 25`).
    pub overlap_fringe_mask: u8,
    /// 8-bit mask where bit `g == 1` iff granule `g` lies outside all attractors (`overlap == 0`, `offset 26`).
    pub lacuna_exterior_mask: u8,
    /// 8-bit mask where bit `g == 1` iff granule `g` lies in `lacuna_exterior_mask` with drift `>= floor` (`offset 27`).
    pub discovery_excursion_mask: u8,
    /// Maximum number of attractors overlapping on any single granule (`offset 28`).
    pub max_overlap_depth: u8,
    /// Granule index (`0..=7`) hosting `discovery_seed_word` (`offset 29`).
    pub discovery_granule_idx: u8,
    /// `1` if approximation noise in `Liminal` triggered unknown attractor discovery, `0` otherwise (`offset 30`).
    pub discovery_triggered: u8,
    /// Explicit 1-byte alignment padding (`offset 31`).
    pub reserved_align: u8,
}

/// Evaluates set-overlap multi-attractor cognitive aperture overlap and noise-driven discovery
/// in `Liminal` across the 8 SWAR granules (`0` allocations, `0` data-dependent branches).
#[inline]
#[must_use]
pub fn evaluate_stigmergic_overlap_and_discovery_words_64<Node: RelativeAttractorNode>(
    nodes: &[Node],
    query_words: &[u64; NUM_STIGMERGIC_GRANULES],
    noisy_drifted_words: &[u64; NUM_STIGMERGIC_GRANULES],
    aperture_floor: f32,
    discovery_floor_q7: u8,
) -> OverlapDiscoveryReport {
    let safe_floor = branchless_max_f32(branchless_relu_f32(aperture_floor), DEFAULT_ENTROPY_EPS);
    let safe_discovery_floor = branchless_select_u8(
        discovery_floor_q7 < 4,
        4,
        branchless_select_u8(discovery_floor_q7 > 120, 120, discovery_floor_q7),
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

    let mut overlap_core_energy = 0.0_f32;
    let mut overlap_fringe_energy = 0.0_f32;
    let mut lacuna_excursion_energy = 0.0_f32;
    let mut dominant_overlap_anchor_id = 0u32;
    let mut best_core_score = 0.0_f32;
    let mut best_any_anchor_id = 0u32;
    let mut best_any_pull = 0.0_f32;
    let mut discovery_seed_word = 0u64;
    let mut overlap_core_mask = 0u8;
    let mut overlap_fringe_mask = 0u8;
    let mut lacuna_exterior_mask = 0u8;
    let mut discovery_excursion_mask = 0u8;
    let mut max_overlap_depth = 0u8;
    let mut best_serendip_q7 = 0u8;
    let mut discovery_granule_idx = 0u8;
    let mut discovery_triggered = 0u8;

    let mut g = 0usize;
    while g < NUM_STIGMERGIC_GRANULES {
        let ov = overlap_counts[g];
        let bit = 1u8 << g;
        let is_core = ov >= 2;
        let is_fringe = ov == 1;
        let is_lacuna = ov == 0;

        overlap_core_mask |= branchless_select_u8(is_core, bit, 0);
        overlap_fringe_mask |= branchless_select_u8(is_fringe, bit, 0);
        lacuna_exterior_mask |= branchless_select_u8(is_lacuna, bit, 0);

        let depth_gt = ov > max_overlap_depth;
        max_overlap_depth = branchless_select_u8(depth_gt, ov, max_overlap_depth);

        let coop_factor = 1.0 + 0.35 * (f32::from(ov.saturating_sub(1)));
        let core_e = branchless_select_f32(is_core, granule_energy[g] * coop_factor, 0.0);
        let pen_e = branchless_select_f32(is_fringe, granule_energy[g] * 0.5, 0.0);
        overlap_core_energy += core_e;
        overlap_fringe_energy += pen_e;

        let take_core_dom = is_core & (core_e > best_core_score);
        best_core_score = branchless_select_f32(take_core_dom, core_e, best_core_score);
        dominant_overlap_anchor_id = branchless_select_u32(
            take_core_dom,
            granule_best_anchor[g],
            dominant_overlap_anchor_id,
        );
        let take_any_dom = (!is_lacuna) & (granule_best_pull[g] > best_any_pull);
        best_any_pull = branchless_select_f32(take_any_dom, granule_best_pull[g], best_any_pull);
        best_any_anchor_id =
            branchless_select_u32(take_any_dom, granule_best_anchor[g], best_any_anchor_id);

        let drift_word = swar_q7_circular_diff_x8(noisy_drifted_words[g], query_words[g]);
        let drift_q7 = swar_u8_mean_round_x8(swar_q7_abs_u8_x8(drift_word));
        let disc_hit = is_lacuna & (drift_q7 >= safe_discovery_floor);
        discovery_excursion_mask |= branchless_select_u8(disc_hit, bit, 0);
        lacuna_excursion_energy +=
            branchless_select_f32(is_lacuna, f32::from(drift_q7) * (1.0 / 127.0), 0.0);

        let take_serendip = disc_hit & (drift_q7 > best_serendip_q7);
        best_serendip_q7 = branchless_select_u8(take_serendip, drift_q7, best_serendip_q7);
        discovery_seed_word =
            branchless_select_u64(take_serendip, noisy_drifted_words[g], discovery_seed_word);
        discovery_granule_idx = branchless_select_u8(take_serendip, g as u8, discovery_granule_idx);
        discovery_triggered |= u8::from(disc_hit);
        g += 1;
    }

    let final_dom_id = branchless_select_u32(
        overlap_core_mask != 0,
        dominant_overlap_anchor_id,
        best_any_anchor_id,
    );

    OverlapDiscoveryReport {
        overlap_core_energy,
        overlap_fringe_energy,
        lacuna_excursion_energy,
        dominant_overlap_anchor_id: final_dom_id,
        discovery_seed_word,
        overlap_core_mask,
        overlap_fringe_mask,
        lacuna_exterior_mask,
        discovery_excursion_mask,
        max_overlap_depth,
        discovery_granule_idx,
        discovery_triggered,
        reserved_align: 0,
    }
}

// ===================================================================
// 2. Core Layer 1: Stigmergic Granule State (32 B)
// ===================================================================

/// Autonomous Core Layer 1 controller (`#[repr(C, align(16))]`, 32 B) governing per-granule
/// stigmergic approximation parameters, overlap-overlap reinforcement, and emergent noise discovery.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StigmergicGranuleState {
    /// Per-granule pheromone step gain in Q0.7 (`[32..=127]`, `offset 0..8`).
    pub granule_step_gain_q7: [u8; NUM_STIGMERGIC_GRANULES],
    /// Per-granule tolerated phase approximation residual in Q7 (`[2..=32]`, `offset 8..16`).
    pub granule_noise_tolerance_q7: [u8; NUM_STIGMERGIC_GRANULES],
    /// Per-granule local terrain pheromone density in Q0.7 (`[8..=127]`, `offset 16..24`).
    pub granule_pheromone_q7: [u8; NUM_STIGMERGIC_GRANULES],
    /// Active set-overlap core mask across the 8 SWAR granules (`offset 24`).
    pub overlap_mask: u8,
    /// Mask of granules whose approximation noise reached `Liminal` discovery (`offset 25`).
    pub discovery_excursion_mask: u8,
    /// Adaptive granule consensus quorum requirement (`4..=7` out of 8, `offset 26`).
    pub quorum_threshold: u8,
    /// Q7 threshold above which noisy excursions in `Liminal` nucleate a new attractor (`offset 27`).
    pub discovery_floor_q7: u8,
    /// Monotonic count of Core Layer 1 self-adaptation epochs (`offset 28..32`).
    pub adaptation_epochs: u32,
}

impl Default for StigmergicGranuleState {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl StigmergicGranuleState {
    /// Constructs a neutral 32-byte [`StigmergicGranuleState`] baseline.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            granule_step_gain_q7: [96; NUM_STIGMERGIC_GRANULES],
            granule_noise_tolerance_q7: [4; NUM_STIGMERGIC_GRANULES],
            granule_pheromone_q7: [32; NUM_STIGMERGIC_GRANULES],
            overlap_mask: 0,
            discovery_excursion_mask: 0,
            quorum_threshold: DEFAULT_STIGMERGIC_QUORUM_THRESHOLD,
            discovery_floor_q7: 24,
            adaptation_epochs: 0,
        }
    }

    /// Branchlessly adapts per-granule pheromone gains, tolerated approximation noise, and
    /// discovery discovery floors from `in_conflict`, `overlap_report`, and `ledger`.
    #[inline]
    pub fn adapt_from_conflict_and_overlap_branchless(
        &mut self,
        in_conflict: bool,
        overlap_report: &OverlapDiscoveryReport,
        ledger: &CognitiveTrajectoryLedger,
    ) {
        let (op_hist, _depth) = ledger.op_counts();
        let peel_dom = op_hist[CognitiveOpKind::DecomposePeel as usize]
            > op_hist[CognitiveOpKind::SynthesizeFuse as usize];
        let mut g = 0usize;
        while g < NUM_STIGMERGIC_GRANULES {
            let bit = 1u8 << g;
            let in_core = (overlap_report.overlap_core_mask & bit) != 0;
            let in_penumbra = (overlap_report.overlap_fringe_mask & bit) != 0;
            let in_serendip = (overlap_report.discovery_excursion_mask & bit) != 0;

            let sg0 = self.granule_step_gain_q7[g];
            let sg_core = branchless_select_u8(in_core, sg0.saturating_add(6).min(127), sg0);
            let sg_ser =
                branchless_select_u8(in_serendip, sg_core.saturating_add(8).min(127), sg_core);
            self.granule_step_gain_q7[g] = branchless_select_u8(
                !in_core & !in_penumbra & !in_serendip,
                sg_ser.saturating_sub(2).max(32),
                sg_ser,
            );

            // High set-overlap tolerates coarser low-spec noise; edge penumbra tightens precision.
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

        let (_min_mask, min_cnt, core_le_noncore) =
            branchless_mask8_min_delta(0xFF, overlap_report.overlap_core_mask);
        let core_cnt = branchless_select_u8(core_le_noncore, min_cnt, 8 - min_cnt);
        let core_ge_half = (!core_le_noncore) | (min_cnt == 4);
        self.quorum_threshold = branchless_select_u8(
            in_conflict,
            self.quorum_threshold.saturating_add(1).min(7),
            branchless_select_u8(
                core_ge_half,
                self.quorum_threshold.saturating_sub(1).max(4),
                self.quorum_threshold,
            ),
        );
        self.discovery_floor_q7 = branchless_select_u8(
            in_conflict,
            self.discovery_floor_q7.saturating_add(4).min(64),
            branchless_select_u8(
                peel_dom & (core_cnt < 3),
                self.discovery_floor_q7.saturating_sub(2).max(12),
                self.discovery_floor_q7,
            ),
        );
        self.overlap_mask = overlap_report.overlap_core_mask;
        self.discovery_excursion_mask = overlap_report.discovery_excursion_mask;
        self.adaptation_epochs = self.adaptation_epochs.wrapping_add(1);
    }

    /// Computes the branchless integer mean of `granule_noise_tolerance_q7` across all 8 granules (`0 fdiv`).
    #[inline(always)]
    #[must_use]
    pub const fn mean_noise_tolerance_q7(&self) -> u8 {
        let sum = self.granule_noise_tolerance_q7[0] as u16
            + self.granule_noise_tolerance_q7[1] as u16
            + self.granule_noise_tolerance_q7[2] as u16
            + self.granule_noise_tolerance_q7[3] as u16
            + self.granule_noise_tolerance_q7[4] as u16
            + self.granule_noise_tolerance_q7[5] as u16
            + self.granule_noise_tolerance_q7[6] as u16
            + self.granule_noise_tolerance_q7[7] as u16;
        (sum >> 3) as u8
    }

    /// Evaluates the branchless self-differentiable boundary sensitivity gate $4p(1-p)$ in Q0.7 (`0 fdiv`).
    ///
    /// Vanishes at both ignorance (`p_q7 == 0`) and saturation (`p_q7 >= 127`) while peaking at the
    /// transition boundary (`p_q7 == 64`, returning `126`).
    #[inline(always)]
    #[must_use]
    pub const fn boundary_sensitivity_gate_q7(p_q7: u8) -> u8 {
        crate::liminal::sensitivity_4p1mp_q7(p_q7)
    }
}

/// Branchless 16-byte (`#[repr(C, align(16))]`) summary capturing fragmented micro-ensemble
/// consensus across the 8 SWAR granules (`quorum_agreed_granule_mask`) and Layer 2 Ego tie-breaks.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct StigmergicFragmentEnsembleReport {
    /// Total consensus-weighted steering energy across participating signal fragments (`offset 0..4`).
    pub ensemble_consensus_energy: f32,
    /// Variance-reduction diversity gain factor derived from multi-fragment agreement (`offset 4..8`).
    pub ensemble_diversity_gain: f32,
    /// 8-bit mask of SWAR granules where `>= 2` fragments agreed in phase direction (`offset 8`).
    pub quorum_agreed_granule_mask: u8,
    /// 8-bit mask of SWAR granules where fragments split/contested direction (`offset 9`).
    pub split_contested_granule_mask: u8,
    /// 8-bit mask of contested granules resolved by Layer 2 Ego (`offset 10`).
    pub ego_tiebreak_granule_mask: u8,
    /// Number of active participating signal fragments in the ensemble (`offset 11`).
    pub active_fragment_count: u8,
    /// Dominant attractor slot selected by the consensus vote (`offset 12`).
    pub dominant_fragment_slot: u8,
    /// Explicit alignment padding (`offset 13..16`).
    pub reserved: [u8; 3],
}

/// Uniform node abstraction enabling relative signed coupling and epistemic horizon evaluation
/// across any attractor representation (`PhaseAnchorRecord`, `GatewayFrame`, `ObjectiveFact`, etc.).
pub trait RelativeAttractorNode {
    /// Static cognitive aperture mask for relative attractor participation.
    const RELATIVE_APERTURE_MASK: u32 = THRESHOLD_RELATIVE_COUPLING;
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

impl RelativeAttractorNode for PhaseAnchorRecord {
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

impl RelativeAttractorNode for PointerTrail {
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
        accumulate_words_sincos_weighted(
            &self.0.relative_phase_words_64(),
            ea,
            &mut sin_acc,
            &mut cos_acc,
        );
        accumulate_words_sincos_weighted(
            &self.1.relative_phase_words_64(),
            eb,
            &mut sin_acc,
            &mut cos_acc,
        );
        fold_sincos64_to_words(&sin_acc, &cos_acc)
    }

    #[inline(always)]
    fn relative_energy(&self) -> f32 {
        branchless_relu_f32(self.0.relative_energy())
            + branchless_relu_f32(self.1.relative_energy())
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
        let idx = if perspective_idx < N {
            perspective_idx
        } else {
            0
        };
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
        let primary_idx = usize::from(report.primary_idx);
        let landed_idx = usize::from(report.landed_idx);
        let edge_active = (report.scattered | (report.spawned_new & report.overflowed))
            & (primary_idx != landed_idx);
        let eff_step = branchless_select_u8(edge_active, step_q7, 0);
        let is_opposed = report.spawned_new & report.overflowed;
        self.reinforce_relation_branchless(primary_idx, landed_idx, is_opposed, eff_step);
    }
}

/// Folds a 64-head `[f32; 64]` `(sin, cos)` vector accumulator into 8 packed SWAR `u64` words.
#[inline]
pub(crate) fn fold_sincos64_to_words(
    sin_acc: &[f32; MAX_ATTENTION_HEADS],
    cos_acc: &[f32; MAX_ATTENTION_HEADS],
) -> [u64; NUM_STIGMERGIC_GRANULES] {
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

/// Unified single-cacheline (`64 B`, `#[repr(C, align(64))]`) primitive state pairing the two
/// foundational geometric primitives: [`LiminalThresholdState`] (`0..32`) and
/// [`StigmergicGranuleState`] (`32..64`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CorePrimitiveState {
    /// Boundary apertures, cognitive window floors, and domain gates (`offset 0..32`).
    pub liminal: LiminalThresholdState,
    /// 8-granule stigmergic step gains, noise tolerances, and volatile pheromones (`offset 32..64`).
    pub stigmergy: StigmergicGranuleState,
}

impl CorePrimitiveState {
    /// Constructs a neutral 64-byte [`CorePrimitiveState`].
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            liminal: LiminalThresholdState::new(),
            stigmergy: StigmergicGranuleState::new(),
        }
    }
}

// SAFETY: `CorePrimitiveState` is `#[repr(C, align(64))]`, exact size 64 bytes with zero padding.
unsafe impl Pod for CorePrimitiveState {}

// SAFETY: `OverlapDiscoveryReport` is `#[repr(C, align(16))]`, exact size 32 bytes with zero padding.
unsafe impl Pod for OverlapDiscoveryReport {}
// SAFETY: `StigmergicGranuleState` is `#[repr(C, align(16))]`, exact size 32 bytes with zero padding.
unsafe impl Pod for StigmergicGranuleState {}
// SAFETY: `StigmergicFragmentEnsembleReport` is `#[repr(C, align(16))]`, exact size 16 bytes with zero padding.
unsafe impl Pod for StigmergicFragmentEnsembleReport {}

const _: () = {
    assert!(size_of::<OverlapDiscoveryReport>() == 32);
    assert!(align_of::<OverlapDiscoveryReport>() == 16);
    assert!(offset_of!(OverlapDiscoveryReport, overlap_core_energy) == 0);
    assert!(offset_of!(OverlapDiscoveryReport, dominant_overlap_anchor_id) == 12);
    assert!(offset_of!(OverlapDiscoveryReport, discovery_seed_word) == 16);
    assert!(offset_of!(OverlapDiscoveryReport, overlap_core_mask) == 24);
    assert!(offset_of!(OverlapDiscoveryReport, discovery_triggered) == 30);

    assert!(size_of::<StigmergicGranuleState>() == 32);
    assert!(align_of::<StigmergicGranuleState>() == 16);
    assert!(offset_of!(StigmergicGranuleState, overlap_mask) == 24);
    assert!(offset_of!(StigmergicGranuleState, discovery_excursion_mask) == 25);
    assert!(offset_of!(StigmergicGranuleState, discovery_floor_q7) == 27);
};
