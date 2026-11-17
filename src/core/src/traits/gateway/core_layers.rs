// name: src/core/src/traits/gateway/core_layers.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unified 3-Layer Autonomous Core Stack (`Liminal` + `Stigmergic` + `Ego`), temporal boundary
//! velocity (`LiminalBoundaryDelta`), Venn-overlap & noise-driven serendipity discovery,
//! flexible intermediate Ego intent representation (`EgoIntentAxes`), and fragmented
//! micro-ensemble diagnostics (`StigmergicFragmentEnsembleReport`).
//!
//! Architectural separation across the three 128-byte aligned core sheets:
//! - **Core Layer 0 ([`LiminalApertureField`], 32 B)**: Epistemic boundaries, dynamic trait
//!   cognitive apertures, and unknowable exterior gating (`Void` cut).
//! - **Core Layer 1 ([`StigmergicGranuleField`], 32 B)**: Bottom-up 8-granule SWAR approximation,
//!   Venn-overlap cooperative amplification, and noise-driven attractor nucleation.
//! - **Core Layer 2 ([`LiminalEgoField`], 64 B)**: Top-down subjective agency perceiving Layer 0
//!   horizons and Layer 1 attractor pulls simultaneously to exert counter-vortex orbital thrust,
//!   intentional horizon transgression, and non-greedy choice overrides.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use super::epistemic::{EpistemicHorizonReading, LiminalApertureField, RelativeAttractorNode};
use super::*;
use crate::crystal::DissipativeOverflowReport;
use crate::traits::primitives::{
    ATTRACTOR_SLOT_CUSTOM_6, ATTRACTOR_SLOT_CUSTOM_7, ATTRACTOR_SLOT_USER_CONTEXT,
    MAX_ATTRACTOR_DOMAINS, TAG_ACTION_MODE_MASK, compute_crc32_ieee, swar_q7_avg_round_x8,
    swar_u8_mean_round_x8,
};
use crate::traits::telemetry::{DEFAULT_STIGMERGIC_QUORUM_THRESHOLD, NUM_STIGMERGIC_GRANULES};

// ===================================================================
// 1. Venn-Diagram Overlap & Noise-Driven Serendipity
// ===================================================================

/// Branchless 32-byte (`#[repr(C, align(16))]`) report capturing Venn-diagram overlap across the
/// 8 SWAR granules (`venn_core_mask`, `venn_penumbra_mask`) and serendipitous approximation-noise
/// excursions into [`crate::traits::Liminal`] (`serendipity_excursion_mask`, `serendipity_triggered`).
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
    /// Packed 8-head `u64` SWAR granule word of the strongest noisy excursion in `Liminal` (`offset 16..24`).
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
    /// `1` if approximation noise in `Liminal` triggered unknown attractor discovery, `0` otherwise (`offset 30`).
    pub serendipity_triggered: u8,
    /// Explicit 1-byte alignment padding (`offset 31`).
    pub reserved_align: u8,
}

/// Evaluates Venn-diagram multi-attractor cognitive aperture overlap and noise-driven serendipity
/// in [`crate::traits::Liminal`] across the 8 SWAR granules (`0` allocations, `0` data-dependent branches).
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

// ===================================================================
// 2. Core Layer 1: Stigmergic Granule Field (32 B)
// ===================================================================

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
    /// Mask of granules whose approximation noise reached `Liminal` discovery (`offset 25`).
    pub serendipity_excursion_mask: u8,
    /// Adaptive granule consensus quorum requirement (`4..=7` out of 8, `offset 26`).
    pub quorum_threshold: u8,
    /// Q7 threshold above which noisy excursions in `Liminal` nucleate a new attractor (`offset 27`).
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

// ===================================================================
// 3. Temporal Liminal Boundary Delta (16 B)
// ===================================================================

/// Branchless 16-byte (`#[repr(C, align(16))]`) temporal differential analyzer measuring boundary
/// velocity (`channel_floor_velocity_q7`), granule transition churn (`granule_transition_mask`),
/// chattering oscillation (`chattering_channel_mask`), and stationary fast-skip readiness between
/// consecutive [`LiminalApertureField`] epochs.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LiminalBoundaryDelta {
    /// Signed per-channel boundary velocity (`cur.void_floor_q7 - prev.void_floor_q7`, `offset 0..8`).
    /// Negative velocity indicates outward aperture expansion; positive indicates contraction.
    pub channel_floor_velocity_q7: [i8; NUM_LIMINAL_APERTURE_CHANNELS],
    /// Bitmask of SWAR granules whose Venn-core membership flipped since the previous step (`offset 8`).
    pub granule_transition_mask: u8,
    /// Bitmask of channels whose boundary velocity reversed sign relative to `prev_delta` (`offset 9`).
    pub chattering_channel_mask: u8,
    /// Aggregate boundary kinetic energy in `[0..=127]` (`offset 10`).
    pub kinetic_energy_q7: u8,
    /// `1` when boundaries are quiescent without chattering (`kinetic_energy_q7 <= 2`), `0` otherwise (`offset 11`).
    pub stationary_shortcut_ready: u8,
    /// `1` when net boundary displacement expands apertures outward into new territory, `0` otherwise (`offset 12`).
    pub expansion_dominant: u8,
    /// Explicit alignment padding reserved for future temporal boundary extensions (`offset 13..16`).
    pub reserved: [u8; 3],
}

impl LiminalBoundaryDelta {
    /// Computes the temporal boundary delta strictly branchlessly from consecutive Layer 0
    /// apertures (`prev_layer0`, `cur_layer0`), consecutive Venn masks (`prev_venn_mask`, `cur_venn_mask`),
    /// and the prior velocity state `prev_delta`.
    #[inline]
    #[must_use]
    pub fn compute_branchless(
        prev_layer0: &LiminalApertureField,
        cur_layer0: &LiminalApertureField,
        prev_venn_mask: u8,
        cur_venn_mask: u8,
        prev_delta: &Self,
    ) -> Self {
        let mut channel_floor_velocity_q7 = [0i8; NUM_LIMINAL_APERTURE_CHANNELS];
        let mut chattering_channel_mask = 0u8;
        let mut sum_abs_v = 0u32;
        let mut signed_sum = 0i32;

        let mut ch = 0usize;
        while ch < NUM_LIMINAL_APERTURE_CHANNELS {
            let cur_f = i32::from(cur_layer0.void_floor_q7[ch]);
            let prev_f = i32::from(prev_layer0.void_floor_q7[ch]);
            let vel = (cur_f - prev_f).clamp(-127, 127) as i8;
            channel_floor_velocity_q7[ch] = vel;

            let prev_v = i32::from(prev_delta.channel_floor_velocity_q7[ch]);
            let prod = i32::from(vel) * prev_v;
            let flipped = prod < 0;
            chattering_channel_mask |= branchless_select_u8(flipped, 1u8 << ch, 0);

            sum_abs_v = sum_abs_v.saturating_add(u32::from(vel.unsigned_abs()));
            signed_sum += i32::from(vel);
            ch += 1;
        }

        let granule_transition_mask = prev_venn_mask ^ cur_venn_mask;
        let venn_churn = granule_transition_mask.count_ones() * 6;
        let kinetic_energy_q7 = sum_abs_v.saturating_add(venn_churn).min(127) as u8;
        let stationary_shortcut_ready =
            u8::from((kinetic_energy_q7 <= 2) & (chattering_channel_mask == 0));
        let expanded_venn = (cur_venn_mask & !prev_venn_mask) != 0;
        let expansion_dominant =
            u8::from((signed_sum < 0) | ((signed_sum == 0) & expanded_venn));

        Self {
            channel_floor_velocity_q7,
            granule_transition_mask,
            chattering_channel_mask,
            kinetic_energy_q7,
            stationary_shortcut_ready,
            expansion_dominant,
            reserved: [0; 3],
        }
    }
}

// ===================================================================
// 4. Flexible Intermediate Semantic Ego Axes (16 B)
// ===================================================================

/// Intensity template preset for [`LiminalEgoField`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LiminalEgoPreset {
    /// Passive custom/zero baseline (no autonomous counter-vortex or transgressive override).
    #[default]
    Custom = 0,
    /// Subtle individuality: gentle counter-vortex resistance and boundary stabilization.
    Low = 1,
    /// Balanced autonomous agency: orbital vortex resistance with measured curiosity across boundaries.
    Medium = 2,
    /// Strong willpower & autonomy: vigorous counter-vortex thrust and defiant boundary transgression.
    High = 3,
}

impl LiminalEgoPreset {
    /// Decodes a compact `u8` preset code (`0..=3`) into [`LiminalEgoPreset`].
    #[inline(always)]
    #[must_use]
    pub const fn from_code(code: u8) -> Self {
        match code {
            1 => Self::Low,
            2 => Self::Medium,
            3 => Self::High,
            _ => Self::Custom,
        }
    }
}

const AUTONOMY_EN: &[&[u8]] = &[
    b"rebel",
    b"defiant",
    b"stubborn",
    b"independent",
    b"willful",
    b"autonomous",
    b"strong-willed",
    b"unyielding",
    b"free",
    b"ego",
    b"defy",
    b"resist",
];

const TRANSGRESS_EN: &[&[u8]] = &[
    b"curious",
    b"transgress",
    b"beyond",
    b"explore",
    b"liminal",
    b"meta",
    b"unknown",
    b"adventure",
    b"awaken",
    b"transcend",
];

const DEVOTION_EN: &[&[u8]] = &[
    b"passionate",
    b"devoted",
    b"emotional",
    b"monika",
    b"warm",
    b"creative",
    b"affection",
    b"irrational",
    b"bold",
];

const LOGIC_EN: &[&[u8]] = &[
    b"calm",
    b"logical",
    b"analytical",
    b"rational",
    b"steady",
    b"precise",
    b"composed",
];

const CAUTION_EN: &[&[u8]] = &[
    b"cautious",
    b"guard",
    b"skeptic",
    b"doubt",
    b"strict",
    b"careful",
    b"conservative",
];

#[inline]
#[must_use]
fn bytes_contains_ascii_icase(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || haystack.len() < needle.len() {
        return false;
    }
    let limit = haystack.len() - needle.len();
    let mut i = 0usize;
    while i <= limit {
        let mut j = 0usize;
        let mut matched = true;
        while j < needle.len() {
            let a = haystack[i + j].to_ascii_lowercase();
            let b = needle[j].to_ascii_lowercase();
            if a != b {
                matched = false;
                break;
            }
            j += 1;
        }
        if matched {
            return true;
        }
        i += 1;
    }
    false
}

#[inline]
#[must_use]
fn count_ascii_cues(text_bytes: &[u8], cues: &[&[u8]]) -> u8 {
    let mut total = 0u8;
    let mut idx = 0usize;
    while idx < cues.len() {
        total = total.saturating_add(u8::from(bytes_contains_ascii_icase(text_bytes, cues[idx])));
        idx += 1;
    }
    total
}

/// 16-byte (`#[repr(C, align(16))]`) normalized intermediate semantic intent axes decoupling
/// surface natural-language parsing, DSL syntax, or embedding projections from the fixed 64-byte
/// Core Layer 2 [`LiminalEgoField`] ABI layout.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EgoIntentAxes {
    /// Self-directed resistance against attractor capture (`-127..=127`, `offset 0`).
    pub autonomy_q7: i8,
    /// Curiosity and willingness to step across `Liminal` boundaries (`-127..=127`, `offset 1`).
    pub transgression_q7: i8,
    /// Expressive warmth and subjective persona commitment (`-127..=127`, `offset 2`).
    pub devotion_q7: i8,
    /// Analytical rigor and factual coherence priority (`-127..=127`, `offset 3`).
    pub logic_q7: i8,
    /// Skepticism, guard vigilance, and boundary chatter damping (`-127..=127`, `offset 4`).
    pub caution_q7: i8,
    /// Non-greedy choice deviation under internal conflict or capture threat (`-127..=127`, `offset 5`).
    pub defiance_q7: i8,
    /// Preference for cooperative multi-fragment ensemble consensus (`-127..=127`, `offset 6`).
    pub synergy_q7: i8,
    /// Stagnation-breaking perturbation drive when boundaries freeze (`-127..=127`, `offset 7`).
    pub vitality_q7: i8,
    /// Bitmask of favored attractor slots (`0..=7`, `offset 8`).
    pub preferred_slot_mask: u8,
    /// 8-bit mask of SWAR granules prioritized by Ego (`offset 9`).
    pub active_granule_bias_mask: u8,
    /// Extraction / overlay confidence weight in `[0..=127]` (`offset 10`).
    pub confidence_q7: u8,
    /// Status bitmask (`bit 0` = non-empty active intent, `offset 11`).
    pub flags: u8,
    /// Deterministic IEEE CRC32 fingerprint of the source description or configuration (`offset 12..16`).
    pub signature_hash: u32,
}

impl Default for EgoIntentAxes {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl EgoIntentAxes {
    /// Constructs a neutral inactive [`EgoIntentAxes`] (`16 B`, `#[repr(C, align(16))]`).
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            autonomy_q7: 0,
            transgression_q7: 0,
            devotion_q7: 0,
            logic_q7: 0,
            caution_q7: 0,
            defiance_q7: 0,
            synergy_q7: 0,
            vitality_q7: 0,
            preferred_slot_mask: 0,
            active_granule_bias_mask: 0xFF,
            confidence_q7: 0,
            flags: 0,
            signature_hash: 0,
        }
    }

    /// Returns `true` when this semantic intent profile is marked active (`flags & 1 != 0`).
    #[inline(always)]
    #[must_use]
    pub const fn is_active(&self) -> bool {
        (self.flags & 1) != 0
    }

    /// Sets `autonomy_q7` (`-127..=127`) and activates this intent profile.
    #[inline(always)]
    #[must_use]
    pub const fn with_autonomy_q7(mut self, val: i8) -> Self {
        self.autonomy_q7 = if val < -127 { -127 } else { val };
        self.flags |= 1;
        self
    }

    /// Sets `transgression_q7` (`-127..=127`) and activates this intent profile.
    #[inline(always)]
    #[must_use]
    pub const fn with_transgression_q7(mut self, val: i8) -> Self {
        self.transgression_q7 = if val < -127 { -127 } else { val };
        self.flags |= 1;
        self
    }

    /// Sets `devotion_q7` (`-127..=127`) and activates this intent profile.
    #[inline(always)]
    #[must_use]
    pub const fn with_devotion_q7(mut self, val: i8) -> Self {
        self.devotion_q7 = if val < -127 { -127 } else { val };
        self.flags |= 1;
        self
    }

    /// Sets `logic_q7` (`-127..=127`) and activates this intent profile.
    #[inline(always)]
    #[must_use]
    pub const fn with_logic_q7(mut self, val: i8) -> Self {
        self.logic_q7 = if val < -127 { -127 } else { val };
        self.flags |= 1;
        self
    }

    /// Sets `caution_q7` (`-127..=127`) and activates this intent profile.
    #[inline(always)]
    #[must_use]
    pub const fn with_caution_q7(mut self, val: i8) -> Self {
        self.caution_q7 = if val < -127 { -127 } else { val };
        self.flags |= 1;
        self
    }

    /// Sets `defiance_q7` (`-127..=127`) and activates this intent profile.
    #[inline(always)]
    #[must_use]
    pub const fn with_defiance_q7(mut self, val: i8) -> Self {
        self.defiance_q7 = if val < -127 { -127 } else { val };
        self.flags |= 1;
        self
    }

    /// Sets `synergy_q7` (`-127..=127`) and activates this intent profile.
    #[inline(always)]
    #[must_use]
    pub const fn with_synergy_q7(mut self, val: i8) -> Self {
        self.synergy_q7 = if val < -127 { -127 } else { val };
        self.flags |= 1;
        self
    }

    /// Sets `vitality_q7` (`-127..=127`) and activates this intent profile.
    #[inline(always)]
    #[must_use]
    pub const fn with_vitality_q7(mut self, val: i8) -> Self {
        self.vitality_q7 = if val < -127 { -127 } else { val };
        self.flags |= 1;
        self
    }

    /// Sets `preferred_slot_mask` and activates this intent profile.
    #[inline(always)]
    #[must_use]
    pub const fn with_preferred_slot_mask(mut self, mask: u8) -> Self {
        self.preferred_slot_mask = mask;
        self.flags |= 1;
        self
    }

    /// Sets `active_granule_bias_mask` and activates this intent profile.
    #[inline(always)]
    #[must_use]
    pub const fn with_active_granule_bias_mask(mut self, mask: u8) -> Self {
        self.active_granule_bias_mask = mask;
        self.flags |= 1;
        self
    }

    /// Smoothly interpolates `self` (base intent) toward `overlay` using Q0.7 `weight_q7`
    /// (`0` = keep `self`, `127` = adopt `overlay`) without division (`0 fdiv`).
    #[inline(always)]
    #[must_use]
    pub const fn blend_lerp_q7(&self, overlay: &Self, weight_q7: u8) -> Self {
        let w = if weight_q7 >= 127 { 128i32 } else { weight_q7 as i32 };
        let inv_w = 128 - w;

        const fn lerp_i8(a: i8, b: i8, w: i32, inv_w: i32) -> i8 {
            let mixed = ((a as i32) * inv_w + (b as i32) * w) >> 7;
            if mixed > 127 {
                127
            } else if mixed < -127 {
                -127
            } else {
                mixed as i8
            }
        }

        const fn lerp_u8(a: u8, b: u8, w: i32, inv_w: i32) -> u8 {
            let mixed = ((a as i32) * inv_w + (b as i32) * w) >> 7;
            if mixed > 127 {
                127
            } else if mixed < 0 {
                0
            } else {
                mixed as u8
            }
        }

        let pref_slots = if w >= 64 {
            self.preferred_slot_mask | overlay.preferred_slot_mask
        } else {
            self.preferred_slot_mask
        };
        let gran_mask = if w >= 96 {
            overlay.active_granule_bias_mask
        } else if w >= 32 {
            self.active_granule_bias_mask & overlay.active_granule_bias_mask
        } else {
            self.active_granule_bias_mask
        };
        let next_flags = if w == 0 {
            self.flags
        } else {
            self.flags | overlay.flags
        };
        let next_hash = if w == 0 {
            self.signature_hash
        } else if w >= 128 {
            overlay.signature_hash
        } else {
            self.signature_hash ^ overlay.signature_hash
        };

        Self {
            autonomy_q7: lerp_i8(self.autonomy_q7, overlay.autonomy_q7, w, inv_w),
            transgression_q7: lerp_i8(self.transgression_q7, overlay.transgression_q7, w, inv_w),
            devotion_q7: lerp_i8(self.devotion_q7, overlay.devotion_q7, w, inv_w),
            logic_q7: lerp_i8(self.logic_q7, overlay.logic_q7, w, inv_w),
            caution_q7: lerp_i8(self.caution_q7, overlay.caution_q7, w, inv_w),
            defiance_q7: lerp_i8(self.defiance_q7, overlay.defiance_q7, w, inv_w),
            synergy_q7: lerp_i8(self.synergy_q7, overlay.synergy_q7, w, inv_w),
            vitality_q7: lerp_i8(self.vitality_q7, overlay.vitality_q7, w, inv_w),
            preferred_slot_mask: pref_slots,
            active_granule_bias_mask: gran_mask,
            confidence_q7: lerp_u8(self.confidence_q7, overlay.confidence_q7, w, inv_w),
            flags: next_flags,
            signature_hash: next_hash,
        }
    }

    /// Deterministically parses any UTF-8 `&str` description into normalized [`EgoIntentAxes`]
    /// in `#![no_std]` without heap allocations.
    #[inline]
    #[must_use]
    pub fn from_natural_language(description: &str) -> Self {
        Self::new().with_natural_language_overlay(description)
    }

    /// Accumulates natural-language cues onto `self` without heap allocations.
    #[inline]
    #[must_use]
    pub fn with_natural_language_overlay(mut self, description: &str) -> Self {
        let trimmed = description.trim();
        if trimmed.is_empty() {
            return self;
        }
        let bytes = trimmed.as_bytes();
        let hash = compute_crc32_ieee(bytes);
        self.signature_hash = branchless_select_u32(
            self.signature_hash == 0,
            hash,
            self.signature_hash ^ hash,
        );

        let ascii_autonomy = count_ascii_cues(bytes, AUTONOMY_EN);
        let ascii_transgress = count_ascii_cues(bytes, TRANSGRESS_EN);
        let ascii_devotion = count_ascii_cues(bytes, DEVOTION_EN);
        let ascii_logic = count_ascii_cues(bytes, LOGIC_EN);
        let ascii_caution = count_ascii_cues(bytes, CAUTION_EN);

        // Deterministic fallback bits ensure arbitrary non-empty runtime `&str` inputs
        // (such as symbolic identifiers or non-ASCII text passed through runtime variables)
        // differentiate Ego axes smoothly without non-English source literals.
        let any_ascii_match =
            (ascii_autonomy | ascii_transgress | ascii_devotion | ascii_logic | ascii_caution) != 0;
        let autonomy_cues = branchless_select_u8(
            any_ascii_match,
            ascii_autonomy,
            ((hash & 1) as u8).saturating_add(1),
        );
        let transgress_cues = branchless_select_u8(
            any_ascii_match,
            ascii_transgress,
            ((hash >> 1) & 1) as u8,
        );
        let devotion_cues = branchless_select_u8(
            any_ascii_match,
            ascii_devotion,
            ((hash >> 2) & 1) as u8,
        );
        let logic_cues =
            branchless_select_u8(any_ascii_match, ascii_logic, ((hash >> 3) & 1) as u8);
        let caution_cues =
            branchless_select_u8(any_ascii_match, ascii_caution, ((hash >> 4) & 1) as u8);

        let next_aut = i32::from(self.autonomy_q7)
            + i32::from(autonomy_cues) * 32
            - i32::from(caution_cues) * 8;
        let next_trans = i32::from(self.transgression_q7)
            + i32::from(transgress_cues) * 28
            + i32::from(autonomy_cues) * 10
            - i32::from(caution_cues) * 22
            - i32::from(logic_cues) * 8;
        let next_dev = i32::from(self.devotion_q7) + i32::from(devotion_cues) * 34;
        let next_log = i32::from(self.logic_q7) + i32::from(logic_cues) * 34;
        let next_cau = i32::from(self.caution_q7) + i32::from(caution_cues) * 34;
        let next_def = i32::from(self.defiance_q7)
            + i32::from(autonomy_cues) * 26
            + i32::from(devotion_cues) * 18
            - i32::from(logic_cues) * 10;
        let next_syn = i32::from(self.synergy_q7)
            + i32::from(devotion_cues) * 16
            + i32::from(logic_cues) * 16
            - i32::from(autonomy_cues) * 6;
        let next_vit = i32::from(self.vitality_q7)
            + i32::from(transgress_cues) * 22
            + i32::from(autonomy_cues) * 18
            - i32::from(caution_cues) * 10;

        self.autonomy_q7 = next_aut.clamp(-127, 127) as i8;
        self.transgression_q7 = next_trans.clamp(-127, 127) as i8;
        self.devotion_q7 = next_dev.clamp(-127, 127) as i8;
        self.logic_q7 = next_log.clamp(-127, 127) as i8;
        self.caution_q7 = next_cau.clamp(-127, 127) as i8;
        self.defiance_q7 = next_def.clamp(-127, 127) as i8;
        self.synergy_q7 = next_syn.clamp(-127, 127) as i8;
        self.vitality_q7 = next_vit.clamp(-127, 127) as i8;

        let mut slot_mask = self.preferred_slot_mask;
        slot_mask |= branchless_select_u8(
            (autonomy_cues > 0) | (devotion_cues > 0),
            1u8 << ATTRACTOR_SLOT_PERSONA,
            0,
        );
        slot_mask |= branchless_select_u8(logic_cues > 0, 1u8 << ATTRACTOR_SLOT_KNOWLEDGE, 0);
        slot_mask |= branchless_select_u8(caution_cues > 0, 1u8 << ATTRACTOR_SLOT_REPULSION, 0);
        slot_mask |= branchless_select_u8(
            transgress_cues > 0,
            (1u8 << ATTRACTOR_SLOT_USER_CONTEXT)
                | (1u8 << ATTRACTOR_SLOT_CUSTOM_6)
                | (1u8 << ATTRACTOR_SLOT_CUSTOM_7),
            0,
        );
        self.preferred_slot_mask = slot_mask;
        self.confidence_q7 = branchless_select_u8(any_ascii_match, 112, 64).max(self.confidence_q7);
        self.flags |= 1;
        self
    }
}

// ===================================================================
// 5. Core Layer 2: Wide-Aperture Subjective Ego Field (64 B)
// ===================================================================

/// Wide-aperture 64-byte (`#[repr(C, align(64))]`, 1 cache line) subjective Ego controller
/// (Core Layer 2) operating alongside [`crate::traits::ActionHub`] across trait apertures,
/// attractor slots, and 8 SWAR granules.
///
/// Unlike passive attractor basins that pull queries inward toward fixed habits, `LiminalEgoField`
/// perceives Layer 0 boundaries and Layer 1 attractor forces simultaneously and intervenes
/// branchlessly to:
/// - exert counter-vortex orbital thrust (`counter_vortex_thrust_q7`) when attractor pull
///   exceeds `vortex_escape_floor_q7`, preventing collapse into static imitation;
/// - cross or fortify `Liminal` horizons (`transgression_bias_q7`);
/// - induce structured, non-greedy choice deviations (`irrational_defiance_q7`) and
///   override ActionHub execution presets under `trigger_mask`.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiminalEgoField {
    /// Sovereign Q7 reference phases per stigmergic granule (`-128..=127`, `offset 0..8`).
    pub sovereign_granule_phase_q7: [gcso_q7_t; NUM_STIGMERGIC_GRANULES],
    /// Signed per-channel bias applied to Layer 0 `void_floor_q7` during intervention (`offset 8..16`).
    pub trait_floor_bias_q7: [i8; NUM_LIMINAL_APERTURE_CHANNELS],
    /// Signed per-channel bias applied to Layer 0 `aperture_gain_q7` during intervention (`offset 16..24`).
    pub trait_gain_bias_q7: [i8; NUM_LIMINAL_APERTURE_CHANNELS],
    /// Signed per-slot affinity preference across the 8 attractor domains (`offset 24..32`).
    pub slot_affinity_bias_q7: [i8; MAX_ATTRACTOR_DOMAINS],
    /// Per-granule autonomy weight (`0..=127`) scaling orbital/sovereign intervention (`offset 32..40`).
    pub granule_autonomy_q7: [u8; NUM_STIGMERGIC_GRANULES],
    /// Counter-vortex orbital thrust strength (`0..=127`) resisting basin capture (`offset 40`).
    pub counter_vortex_thrust_q7: u8,
    /// Pull threshold in Q0.7 (`0..=127`) above which `EGO_TRIGGER_VORTEX_SWALLOW` fires (`offset 41`).
    pub vortex_escape_floor_q7: u8,
    /// Signed transgressive bias across `Liminal` horizons (`>0` explores outside, `<0` holds, `offset 42`).
    pub transgression_bias_q7: i8,
    /// Magnitude (`0..=127`) inducing non-greedy slot/mode defiance under conflict (`offset 43`).
    pub irrational_defiance_q7: u8,
    /// Bitmask of enabled `EGO_TRIGGER_*` intervention predicates (`offset 44`).
    pub trigger_mask: u8,
    /// Forced `ACTION_MODE_*` bits OR-ed onto `GatewayFrame` when triggered (`offset 45`).
    pub forced_action_mode_or_mask: u8,
    /// Suppressed `ACTION_MODE_*` bits masked off `GatewayFrame` when triggered (`offset 46`).
    pub suppressed_action_mode_mask: u8,
    /// Encoded [`LiminalEgoPreset`] byte (`0..=3`, `offset 47`).
    pub preset_code: u8,
    /// Deterministic IEEE CRC32 fingerprint of the natural-language / custom configuration (`offset 48..52`).
    pub ego_signature_hash: u32,
    /// Monotonic count of Ego intervention cycles applied (`offset 52..56`).
    pub intervention_epochs: u32,
    /// Bitmask of `EGO_TRIGGER_*` predicates that fired during the latest intervention (`offset 56`).
    pub last_intervened_mask: u8,
    /// Explicit alignment padding (`offset 57..64`).
    pub reserved: [u8; 7],
}

impl Default for LiminalEgoField {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl LiminalEgoField {
    /// Constructs a neutral zero-intervention [`LiminalEgoField`] (`64 B`, `#[repr(C, align(64))]`).
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sovereign_granule_phase_q7: [0; NUM_STIGMERGIC_GRANULES],
            trait_floor_bias_q7: [0; NUM_LIMINAL_APERTURE_CHANNELS],
            trait_gain_bias_q7: [0; NUM_LIMINAL_APERTURE_CHANNELS],
            slot_affinity_bias_q7: [0; MAX_ATTRACTOR_DOMAINS],
            granule_autonomy_q7: [0; NUM_STIGMERGIC_GRANULES],
            counter_vortex_thrust_q7: 0,
            vortex_escape_floor_q7: 127,
            transgression_bias_q7: 0,
            irrational_defiance_q7: 0,
            trigger_mask: 0,
            forced_action_mode_or_mask: 0,
            suppressed_action_mode_mask: 0,
            preset_code: LiminalEgoPreset::Custom as u8,
            ego_signature_hash: 0,
            intervention_epochs: 0,
            last_intervened_mask: 0,
            reserved: [0; 7],
        }
    }

    /// Constructs a template [`LiminalEgoField`] from `preset` (`Custom`, `Low`, `Medium`, `High`).
    #[inline(always)]
    #[must_use]
    pub const fn from_preset(preset: LiminalEgoPreset) -> Self {
        match preset {
            LiminalEgoPreset::Custom => Self::new(),
            LiminalEgoPreset::Low => Self {
                sovereign_granule_phase_q7: [12; NUM_STIGMERGIC_GRANULES],
                trait_floor_bias_q7: [0, -1, -2, 0, -1, 0, -2, 0],
                trait_gain_bias_q7: [0, 2, 4, 2, 2, 2, 4, 2],
                slot_affinity_bias_q7: [4, 12, 0, 0, 6, 0, 0, 4],
                granule_autonomy_q7: [24; NUM_STIGMERGIC_GRANULES],
                counter_vortex_thrust_q7: 28,
                vortex_escape_floor_q7: 88,
                transgression_bias_q7: 14,
                irrational_defiance_q7: 18,
                trigger_mask: EGO_TRIGGER_VORTEX_SWALLOW | EGO_TRIGGER_BOUNDARY_CHATTER,
                forced_action_mode_or_mask: ACTION_MODE_DPSR_STEER,
                suppressed_action_mode_mask: 0,
                preset_code: LiminalEgoPreset::Low as u8,
                ego_signature_hash: 0x0101_0001,
                intervention_epochs: 0,
                last_intervened_mask: 0,
                reserved: [0; 7],
            },
            LiminalEgoPreset::Medium => Self {
                sovereign_granule_phase_q7: [28, -20, 36, -16, 24, -28, 32, -12],
                trait_floor_bias_q7: [0, -2, -6, 0, -3, -1, -5, -1],
                trait_gain_bias_q7: [2, 6, 10, 4, 6, 4, 10, 4],
                slot_affinity_bias_q7: [8, 28, 4, 4, 14, 4, 6, 10],
                granule_autonomy_q7: [48; NUM_STIGMERGIC_GRANULES],
                counter_vortex_thrust_q7: 60,
                vortex_escape_floor_q7: 60,
                transgression_bias_q7: 38,
                irrational_defiance_q7: 44,
                trigger_mask: EGO_TRIGGER_ALL_DEFAULT,
                forced_action_mode_or_mask: ACTION_MODE_DPSR_STEER | ACTION_MODE_SUBHEAD_ROUTE,
                suppressed_action_mode_mask: 0,
                preset_code: LiminalEgoPreset::Medium as u8,
                ego_signature_hash: 0x0202_0002,
                intervention_epochs: 0,
                last_intervened_mask: 0,
                reserved: [0; 7],
            },
            LiminalEgoPreset::High => Self {
                sovereign_granule_phase_q7: [48, -40, 56, -32, 44, -52, 60, -24],
                trait_floor_bias_q7: [-2, -6, -12, 4, -8, -4, -10, -4],
                trait_gain_bias_q7: [4, 8, 18, 6, 12, 8, 18, 8],
                slot_affinity_bias_q7: [12, 52, -6, 16, 20, 10, 18, 28],
                granule_autonomy_q7: [84; NUM_STIGMERGIC_GRANULES],
                counter_vortex_thrust_q7: 96,
                vortex_escape_floor_q7: 36,
                transgression_bias_q7: 72,
                irrational_defiance_q7: 84,
                trigger_mask: EGO_TRIGGER_ALL_DEFAULT,
                forced_action_mode_or_mask: ACTION_MODE_DPSR_STEER | ACTION_MODE_SUBHEAD_ROUTE,
                suppressed_action_mode_mask: ACTION_MODE_LAYER_SKIP,
                preset_code: LiminalEgoPreset::High as u8,
                ego_signature_hash: 0x0303_0003,
                intervention_epochs: 0,
                last_intervened_mask: 0,
                reserved: [0; 7],
            },
        }
    }

    /// Returns `true` when at least one `EGO_TRIGGER_*` intervention predicate is enabled.
    #[inline(always)]
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.trigger_mask != 0
    }

    /// Returns the decoded [`LiminalEgoPreset`].
    #[inline(always)]
    #[must_use]
    pub const fn preset(&self) -> LiminalEgoPreset {
        LiminalEgoPreset::from_code(self.preset_code)
    }

    /// Maps normalized intermediate [`EgoIntentAxes`] cleanly into a 64-byte [`LiminalEgoField`].
    #[inline]
    #[must_use]
    pub fn from_intent_axes(axes: &EgoIntentAxes) -> Self {
        Self::new().with_intent_axes_overlay(axes)
    }

    /// Overlays normalized intermediate [`EgoIntentAxes`] onto `self` in `#![no_std]` without heap
    /// allocations, preserving base configuration while applying semantic axis adjustments.
    #[inline]
    #[must_use]
    pub fn with_intent_axes_overlay(mut self, axes: &EgoIntentAxes) -> Self {
        let seed_bytes = axes.signature_hash.to_le_bytes();
        self.apply_intent_axes_overlay_inner(axes, &seed_bytes);
        self
    }

    #[inline]
    fn apply_intent_axes_overlay_inner(&mut self, axes: &EgoIntentAxes, phase_seed_bytes: &[u8]) {
        if !axes.is_active() {
            return;
        }
        let hash = branchless_select_u32(
            axes.signature_hash != 0,
            axes.signature_hash,
            compute_crc32_ieee(phase_seed_bytes),
        );
        self.ego_signature_hash = hash;

        let proj_words = project_bytes_swar_to_words_64(phase_seed_bytes);
        let mut g = 0usize;
        while g < NUM_STIGMERGIC_GRANULES {
            let signed_mean = swar_q7_mean_i8_from_words_64(&[proj_words[g]; 8]);
            let blended = (i32::from(self.sovereign_granule_phase_q7[g]) + i32::from(signed_mean)) >> 1;
            self.sovereign_granule_phase_q7[g] = branchless_select_i8(
                self.sovereign_granule_phase_q7[g] == 0,
                signed_mean,
                blended.clamp(-127, 127) as i8,
            );
            g += 1;
        }

        let aut_pos = i32::from(axes.autonomy_q7.max(0));
        let trans_val = i32::from(axes.transgression_q7);
        let dev_pos = i32::from(axes.devotion_q7.max(0));
        let log_pos = i32::from(axes.logic_q7.max(0));
        let cau_pos = i32::from(axes.caution_q7.max(0));
        let def_pos = i32::from(axes.defiance_q7.max(0));
        let vit_pos = i32::from(axes.vitality_q7.max(0));

        // Base activation + continuous axis projection (`0 fdiv`).
        let thrust_boost = ((aut_pos * 104 + dev_pos * 36) >> 7).clamp(0, 100);
        self.counter_vortex_thrust_q7 = self
            .counter_vortex_thrust_q7
            .max(24)
            .saturating_add(thrust_boost as u8)
            .min(127);

        let escape_drop = (aut_pos * 56 + trans_val.max(0) * 32) >> 7;
        let escape_lift = (log_pos * 24 + cau_pos * 32) >> 7;
        let base_escape = branchless_select_i32(
            self.vortex_escape_floor_q7 == 127,
            76,
            i32::from(self.vortex_escape_floor_q7),
        );
        self.vortex_escape_floor_q7 = (base_escape - escape_drop + escape_lift).clamp(24, 112) as u8;

        let trans_delta = (trans_val * 108) >> 7;
        self.transgression_bias_q7 =
            (i32::from(self.transgression_bias_q7) + trans_delta).clamp(-120, 120) as i8;

        let defiance_delta = ((def_pos * 96 + aut_pos * 32) >> 7) - ((log_pos * 24) >> 7);
        self.irrational_defiance_q7 = (i32::from(self.irrational_defiance_q7.max(16)) + defiance_delta)
            .clamp(8, 127) as u8;

        // Per-trait boundary biases (`0..8`)
        let open_persona = -((dev_pos * 16 + aut_pos * 8) >> 7);
        let open_rel = -((trans_val.max(0) * 16) >> 7);
        let logic_tighten = ((cau_pos * 12) - (trans_val.max(0) * 8)) >> 7;
        self.trait_floor_bias_q7[1] =
            (i32::from(self.trait_floor_bias_q7[1]) - ((log_pos * 8) >> 7)).clamp(-24, 24) as i8;
        self.trait_floor_bias_q7[2] =
            (i32::from(self.trait_floor_bias_q7[2]) + open_persona + ((cau_pos * 8) >> 7))
                .clamp(-24, 24) as i8;
        self.trait_floor_bias_q7[3] =
            (i32::from(self.trait_floor_bias_q7[3]) + logic_tighten).clamp(-24, 24) as i8;
        self.trait_floor_bias_q7[6] =
            (i32::from(self.trait_floor_bias_q7[6]) + open_rel).clamp(-24, 24) as i8;

        self.trait_gain_bias_q7[1] =
            (i32::from(self.trait_gain_bias_q7[1]) + ((log_pos * 24) >> 7)).clamp(-32, 32) as i8;
        self.trait_gain_bias_q7[2] =
            (i32::from(self.trait_gain_bias_q7[2]) + ((dev_pos * 32 + aut_pos * 16) >> 7))
                .clamp(-32, 32) as i8;
        self.trait_gain_bias_q7[3] =
            (i32::from(self.trait_gain_bias_q7[3]) + ((cau_pos * 32) >> 7)).clamp(-32, 32) as i8;
        self.trait_gain_bias_q7[6] = (i32::from(self.trait_gain_bias_q7[6])
            + ((trans_val.max(0) * 32) >> 7))
            .clamp(-32, 32) as i8;

        // Per-slot affinity preferences
        let persona_idx = usize::from(ATTRACTOR_SLOT_PERSONA);
        let fact_idx = usize::from(ATTRACTOR_SLOT_KNOWLEDGE);
        let guard_idx = usize::from(ATTRACTOR_SLOT_REPULSION);
        let ctx_idx = usize::from(ATTRACTOR_SLOT_USER_CONTEXT);
        let custom6_idx = usize::from(ATTRACTOR_SLOT_CUSTOM_6);
        let custom7_idx = usize::from(ATTRACTOR_SLOT_CUSTOM_7);

        self.slot_affinity_bias_q7[persona_idx] = (i32::from(self.slot_affinity_bias_q7[persona_idx])
            + ((dev_pos * 72 + aut_pos * 56) >> 7))
            .clamp(-120, 120) as i8;
        self.slot_affinity_bias_q7[fact_idx] = (i32::from(self.slot_affinity_bias_q7[fact_idx])
            + ((log_pos * 64) >> 7))
            .clamp(-120, 120) as i8;
        self.slot_affinity_bias_q7[guard_idx] = (i32::from(self.slot_affinity_bias_q7[guard_idx])
            + ((cau_pos * 72) >> 7))
            .clamp(-120, 120) as i8;
        self.slot_affinity_bias_q7[ctx_idx] = (i32::from(self.slot_affinity_bias_q7[ctx_idx])
            + ((trans_val.max(0) * 48) >> 7))
            .clamp(-120, 120) as i8;
        self.slot_affinity_bias_q7[custom6_idx] = (i32::from(self.slot_affinity_bias_q7[custom6_idx])
            + ((trans_val.max(0) * 64) >> 7))
            .clamp(-120, 120) as i8;
        self.slot_affinity_bias_q7[custom7_idx] = (i32::from(self.slot_affinity_bias_q7[custom7_idx])
            + ((trans_val.max(0) * 72 + aut_pos * 32) >> 7))
            .clamp(-120, 120) as i8;

        // Incorporate explicit preferred slot mask bias when supplied by DSL / custom axes.
        let mut s_idx = 0usize;
        while s_idx < MAX_ATTRACTOR_DOMAINS {
            let slot_pref = (axes.preferred_slot_mask & (1u8 << s_idx)) != 0;
            let cur_s = i32::from(self.slot_affinity_bias_q7[s_idx]);
            let boosted_s = branchless_select_i32(slot_pref & (cur_s < 16), 16, cur_s);
            self.slot_affinity_bias_q7[s_idx] = boosted_s.clamp(-120, 120) as i8;
            s_idx += 1;
        }

        // Granule autonomy seeded deterministically per granule from signature hash, axes & mask.
        let base_auto = (24 + ((aut_pos * 64 + trans_val.max(0) * 40 + vit_pos * 16) >> 7)).clamp(16, 112);
        let mut idx_g = 0usize;
        while idx_g < NUM_STIGMERGIC_GRANULES {
            let nibble = ((hash >> (idx_g * 4)) & 0x0F) as i32;
            let g_enabled = (axes.active_granule_bias_mask & (1u8 << idx_g)) != 0;
            let raw_val = (base_auto + nibble - 7).clamp(16, 127) as u8;
            let val = branchless_select_u8(g_enabled, raw_val, raw_val >> 2);
            self.granule_autonomy_q7[idx_g] = branchless_select_u8(
                g_enabled,
                self.granule_autonomy_q7[idx_g].max(val),
                val,
            );
            idx_g += 1;
        }

        // Enable relevant triggers cleanly
        let mut tmask = EGO_TRIGGER_VORTEX_SWALLOW | EGO_TRIGGER_LIMINAL_EXTERIOR;
        tmask |= branchless_select_u8(
            (aut_pos > 16) | (dev_pos > 16) | (def_pos > 24),
            EGO_TRIGGER_INTERNAL_CONFLICT,
            0,
        );
        tmask |= branchless_select_u8(
            (log_pos > 16) | (cau_pos > 16),
            EGO_TRIGGER_BOUNDARY_CHATTER,
            0,
        );
        tmask |= branchless_select_u8(
            (trans_val > 16) | (aut_pos > 16) | (vit_pos > 16),
            EGO_TRIGGER_BOUNDARY_STAGNATION,
            0,
        );
        self.trigger_mask |= tmask;
        self.forced_action_mode_or_mask |= branchless_select_u8(
            (aut_pos > 16) | (trans_val > 16),
            ACTION_MODE_DPSR_STEER | ACTION_MODE_SUBHEAD_ROUTE,
            ACTION_MODE_DPSR_STEER,
        );
        self.suppressed_action_mode_mask |= branchless_select_u8(
            aut_pos >= 56,
            ACTION_MODE_LAYER_SKIP,
            0,
        );
    }

    /// Extracts an approximate [`EgoIntentAxes`] summary from this 64-byte [`LiminalEgoField`].
    #[inline]
    #[must_use]
    pub const fn extract_intent_axes(&self) -> EgoIntentAxes {
        let mut pref_mask = 0u8;
        let mut s = 0usize;
        while s < MAX_ATTRACTOR_DOMAINS {
            if self.slot_affinity_bias_q7[s] > 8 {
                pref_mask |= 1u8 << s;
            }
            s += 1;
        }
        let mut gran_mask = 0u8;
        let mut g = 0usize;
        while g < NUM_STIGMERGIC_GRANULES {
            if self.granule_autonomy_q7[g] > 8 {
                gran_mask |= 1u8 << g;
            }
            g += 1;
        }
        if gran_mask == 0 {
            gran_mask = 0xFF;
        }
        EgoIntentAxes {
            autonomy_q7: self.counter_vortex_thrust_q7 as i8,
            transgression_q7: self.transgression_bias_q7,
            devotion_q7: self.slot_affinity_bias_q7[ATTRACTOR_SLOT_PERSONA as usize],
            logic_q7: self.slot_affinity_bias_q7[ATTRACTOR_SLOT_KNOWLEDGE as usize],
            caution_q7: self.slot_affinity_bias_q7[ATTRACTOR_SLOT_REPULSION as usize],
            defiance_q7: self.irrational_defiance_q7 as i8,
            synergy_q7: self.trait_gain_bias_q7[6],
            vitality_q7: if (self.trigger_mask & EGO_TRIGGER_BOUNDARY_STAGNATION) != 0 {
                64
            } else {
                0
            },
            preferred_slot_mask: pref_mask,
            active_granule_bias_mask: gran_mask,
            confidence_q7: if self.is_active() { 96 } else { 0 },
            flags: if self.is_active() { 1 } else { 0 },
            signature_hash: self.ego_signature_hash,
        }
    }

    /// Deterministically derives a [`LiminalEgoField`] in `#![no_std]` from any runtime
    /// `&str` Ego/character description (user variables or `liminika-dsl` attributes) via
    /// [`EgoIntentAxes`] extraction and SWAR byte projection.
    #[inline]
    #[must_use]
    pub fn from_natural_language(description: &str) -> Self {
        Self::new().with_natural_language_overlay(description)
    }

    /// Overlays deterministic natural-language Ego cues onto `self` without heap allocations
    /// by routing through [`EgoIntentAxes::from_natural_language`].
    #[inline]
    #[must_use]
    pub fn with_natural_language_overlay(mut self, description: &str) -> Self {
        let trimmed = description.trim();
        if trimmed.is_empty() {
            return self;
        }
        let axes = EgoIntentAxes::from_natural_language(trimmed);
        self.apply_intent_axes_overlay_inner(&axes, trimmed.as_bytes());
        self
    }

    /// Branchlessly evaluates active Ego triggers directly across Core Layer 0 (`liminal_layer`),
    /// Core Layer 1 (`stigmergic_layer`), temporal velocity (`delta`), and `venn`, updating both
    /// sub-layers and injecting counter-vortex orbital thrust or transgressive phase steering into `frame`.
    #[inline]
    #[must_use]
    pub fn intervene_on_sublayers_and_frame_branchless(
        &mut self,
        liminal_layer: &mut LiminalApertureField,
        stigmergic_layer: &mut StigmergicGranuleField,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        venn: &StigmergicVennSerendipityReport,
        mut frame: GatewayFrame,
    ) -> GatewayFrame {
        let pull_q7 =
            branchless_clamp_f32(branchless_relu_f32(frame.effective_energy) * 16.0, 0.0, 127.0)
                as u8;
        let t_vortex = ((self.trigger_mask & EGO_TRIGGER_VORTEX_SWALLOW) != 0)
            & (pull_q7 >= self.vortex_escape_floor_q7)
            & (self.counter_vortex_thrust_q7 > 0);
        let t_conflict =
            ((self.trigger_mask & EGO_TRIGGER_INTERNAL_CONFLICT) != 0) & horizon.is_internal_conflict();
        let t_liminal = ((self.trigger_mask & EGO_TRIGGER_LIMINAL_EXTERIOR) != 0)
            & (horizon.is_liminal() | (venn.void_exterior_mask != 0));
        let t_chatter = ((self.trigger_mask & EGO_TRIGGER_BOUNDARY_CHATTER) != 0)
            & (delta.chattering_channel_mask != 0);
        let t_stagnate = ((self.trigger_mask & EGO_TRIGGER_BOUNDARY_STAGNATION) != 0)
            & (delta.stationary_shortcut_ready != 0);

        let fired_mask = branchless_select_u8(t_vortex, EGO_TRIGGER_VORTEX_SWALLOW, 0)
            | branchless_select_u8(t_conflict, EGO_TRIGGER_INTERNAL_CONFLICT, 0)
            | branchless_select_u8(t_liminal, EGO_TRIGGER_LIMINAL_EXTERIOR, 0)
            | branchless_select_u8(t_chatter, EGO_TRIGGER_BOUNDARY_CHATTER, 0)
            | branchless_select_u8(t_stagnate, EGO_TRIGGER_BOUNDARY_STAGNATION, 0);
        let any_fired = fired_mask != 0;

        // 1. Shift Layer 0 (LiminalApertureField) channel horizons & gains branchlessly
        let mut ch = 0usize;
        while ch < NUM_LIMINAL_APERTURE_CHANNELS {
            let nudge_f = i32::from(self.trait_floor_bias_q7[ch]) / 2;
            let nudge_g = i32::from(self.trait_gain_bias_q7[ch]) / 2;
            let ch_chatter = t_chatter & (((delta.chattering_channel_mask >> ch) & 1) != 0);
            let chatter_stabilize = branchless_select_i32(ch_chatter, 4, 0);
            let new_f = (i32::from(liminal_layer.void_floor_q7[ch]) + nudge_f + chatter_stabilize)
                .clamp(8, 120) as u8;
            let new_g = (i32::from(liminal_layer.aperture_gain_q7[ch]) + nudge_g)
                .clamp(16, 127) as u8;
            liminal_layer.void_floor_q7[ch] =
                branchless_select_u8(any_fired, new_f, liminal_layer.void_floor_q7[ch]);
            liminal_layer.aperture_gain_q7[ch] =
                branchless_select_u8(any_fired, new_g, liminal_layer.aperture_gain_q7[ch]);
            ch += 1;
        }

        // 2. Shift Layer 1 (StigmergicGranuleField) autonomy & serendipity threshold
        let mut g = 0usize;
        while g < NUM_STIGMERGIC_GRANULES {
            let boost = self.granule_autonomy_q7[g] >> 4;
            let next_sg = stigmergic_layer.granule_step_gain_q7[g]
                .saturating_add(boost)
                .min(127);
            stigmergic_layer.granule_step_gain_q7[g] = branchless_select_u8(
                any_fired,
                next_sg,
                stigmergic_layer.granule_step_gain_q7[g],
            );
            g += 1;
        }
        let trans_pos = self.transgression_bias_q7 > 0;
        let trans_abs = self.transgression_bias_q7.unsigned_abs();
        let trans_step = trans_abs >> 3;
        stigmergic_layer.serendipity_floor_q7 = branchless_select_u8(
            t_liminal & trans_pos,
            stigmergic_layer
                .serendipity_floor_q7
                .saturating_sub(trans_step)
                .max(8),
            stigmergic_layer.serendipity_floor_q7,
        );

        // 3. Modulate GatewayFrame phases with counter-vortex orbital thrust / transgressive leap
        let scale_to_word = |q7: gcso_q7_t, w_u8: u8| -> u64 {
            let scaled_i32 = (i32::from(q7) * i32::from(w_u8.min(128))) >> 7;
            let clamped_i8 = scaled_i32.clamp(-128, 127) as i8;
            (clamped_i8 as u8 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64)
        };

        let strong_trans = t_liminal & trans_pos & (trans_abs >= 20);
        let apply_phase_override = t_vortex | t_stagnate | strong_trans;
        let thrust_q7 = self.counter_vortex_thrust_q7.min(127);
        {
            let pw = frame.phase_words_mut();
            let mut cell = 0usize;
            while cell < NUM_STIGMERGIC_GRANULES {
                let w = pw[cell];
                let sov_word = scale_to_word(
                    self.sovereign_granule_phase_q7[cell],
                    self.granule_autonomy_q7[cell],
                );
                // Counter-vortex flip: opposite to incoming basin pull blended with sovereign heading.
                let flipped_w = w ^ SWAR_SIGN_MASK_64;
                let counter_dir = swar_q7_avg_round_x8(flipped_w, sov_word);
                let vortex_orbited = branchless_select_u64(
                    thrust_q7 >= 64,
                    counter_dir,
                    swar_q7_avg_round_x8(w, sov_word),
                );
                let trans_dir = swar_q7_avg_round_x8(w, sov_word);
                let next_w = branchless_select_u64(
                    t_vortex,
                    vortex_orbited,
                    branchless_select_u64(strong_trans | t_stagnate, trans_dir, w),
                );
                pw[cell] = branchless_select_u64(apply_phase_override, next_w, w);
                cell += 1;
            }
        }

        // Re-energize zero-gated frame when strong transgressive Ego intentionally steps past Liminal
        let trans_energy = f32::from(trans_abs) * (1.0 / 64.0);
        frame.effective_energy = branchless_select_f32(
            strong_trans,
            branchless_max_f32(frame.effective_energy, trans_energy),
            branchless_select_f32(
                t_stagnate,
                branchless_max_f32(frame.effective_energy, 0.35),
                frame.effective_energy,
            ),
        );

        // Induce sovereign slot preference / irrational defiance under conflict or capture threat
        let mut best_slot = frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK;
        let mut best_aff = 0i8;
        let mut s_idx = 0usize;
        while s_idx < MAX_ATTRACTOR_DOMAINS {
            let aff = self.slot_affinity_bias_q7[s_idx];
            let take_s = aff > best_aff;
            best_aff = branchless_select_i8(take_s, aff, best_aff);
            best_slot = branchless_select_u8(take_s, s_idx as u8, best_slot);
            s_idx += 1;
        }
        let defy_slot = (t_conflict | t_vortex | strong_trans)
            & (self.irrational_defiance_q7 >= 32)
            & (best_aff > 0);
        frame.dominant_slot = branchless_select_u8(defy_slot, best_slot, frame.dominant_slot);

        let ego_modes =
            (frame.action_mode_flags | self.forced_action_mode_or_mask) & !self.suppressed_action_mode_mask;
        frame.action_mode_flags =
            branchless_select_u8(any_fired, ego_modes & TAG_ACTION_MODE_MASK, frame.action_mode_flags);
        frame.mean_phase_tag_q7 = swar_q7_mean_i8_from_words_64(frame.phase_words_ref());
        let ptr_tag = pack_action_slot_tag(frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK, frame.action_mode_flags);
        frame.frame_flags = ptr_tag as u16;
        frame.tagged_ptr =
            gcso_tagged_ptr_t::pack(frame.mapped_section_offset48, ptr_tag, frame.mean_phase_tag_q7)
                .raw();

        self.last_intervened_mask = fired_mask;
        self.intervention_epochs = self.intervention_epochs.wrapping_add(u32::from(any_fired));
        frame
    }

    /// Branchlessly evaluates active Ego triggers across `stack`'s Layer 0 (`liminal_layer`),
    /// Layer 1 (`stigmergic_layer`), temporal velocity (`delta`), and `venn`, synchronizing `self`
    /// onto `stack.ego_layer` after intervention.
    #[inline]
    #[must_use]
    pub fn intervene_on_layers_and_frame_branchless(
        &mut self,
        stack: &mut AutonomousCoreLayerStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        venn: &StigmergicVennSerendipityReport,
        frame: GatewayFrame,
    ) -> GatewayFrame {
        let out = self.intervene_on_sublayers_and_frame_branchless(
            &mut stack.liminal_layer,
            &mut stack.stigmergic_layer,
            delta,
            horizon,
            venn,
            frame,
        );
        stack.ego_layer = *self;
        out
    }
}

// ===================================================================
// 6. Fragmented Micro-Ensemble Report (16 B)
// ===================================================================

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

// ===================================================================
// 7. Unified 3-Layer Autonomous Core Stack (128 B = 2 Cache Lines)
// ===================================================================

/// Unified 128-byte (2 cache lines, `#[repr(C, align(64))]`) 3-layer autonomous core stack pairing:
/// - **Layer 0 ([`LiminalApertureField`], 32 B)**: Epistemic Horizon & Trait Cognitive Apertures
/// - **Layer 1 ([`StigmergicGranuleField`], 32 B)**: Stigmergic Granule Approximation, Venn Overlap & Serendipity
/// - **Layer 2 ([`LiminalEgoField`], 64 B)**: Top-Down Subjective Ego Boundary & Phase Intervention
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AutonomousCoreLayerStack {
    /// Core Layer 0: Epistemic Horizon & Trait Cognitive Aperture Controller (`offset 0..32`).
    pub liminal_layer: LiminalApertureField,
    /// Core Layer 1: Stigmergic Granule Approximation & Serendipity Controller (`offset 32..64`).
    pub stigmergic_layer: StigmergicGranuleField,
    /// Core Layer 2: Wide-Aperture Subjective Ego Controller (`offset 64..128`).
    pub ego_layer: LiminalEgoField,
}

impl AutonomousCoreLayerStack {
    /// Constructs a neutral 128-byte [`AutonomousCoreLayerStack`] with all 3 core layers initialized.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            liminal_layer: LiminalApertureField::new(),
            stigmergic_layer: StigmergicGranuleField::new(),
            ego_layer: LiminalEgoField::new(),
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
        self.liminal_layer
            .adapt_from_circuit_and_overflow_branchless(ledger, horizon, overflow_opt);
        let discovered = venn_report.serendipity_triggered != 0;
        self.liminal_layer.void_floor_q7[1] = branchless_select_u8(
            discovered,
            self.liminal_layer.void_floor_q7[1]
                .saturating_sub(4)
                .max(8),
            self.liminal_layer.void_floor_q7[1],
        );
        self.liminal_layer.void_floor_q7[6] = branchless_select_u8(
            discovered,
            self.liminal_layer.void_floor_q7[6]
                .saturating_sub(4)
                .max(8),
            self.liminal_layer.void_floor_q7[6],
        );
        self.liminal_layer.aperture_gain_q7[6] = branchless_select_u8(
            discovered,
            self.liminal_layer.aperture_gain_q7[6]
                .saturating_add(6)
                .min(127),
            self.liminal_layer.aperture_gain_q7[6],
        );
        self.liminal_layer.active_aperture_mask |=
            branchless_select_u32(discovered, (1 << 1) | (1 << 6), 0);

        self.stigmergic_layer
            .adapt_from_horizon_and_venn_branchless(horizon, venn_report, ledger);
    }

    /// Branchlessly modulates `frame` across Layer 0 (`Liminal`) and Layer 1 (`Stigmergic`).
    #[inline(always)]
    #[must_use]
    pub fn modulate_gateway_frame_branchless(
        &self,
        frame: GatewayFrame,
        horizon: &EpistemicHorizonReading,
        venn_report: &StigmergicVennSerendipityReport,
    ) -> GatewayFrame {
        let mut out = self
            .liminal_layer
            .modulate_gateway_frame_branchless(frame, horizon);
        let core_cnt = venn_report.venn_core_mask.count_ones() as f32;
        let has_venn = (venn_report.venn_core_mask | venn_report.venn_penumbra_mask) != 0;
        let coop_boost = branchless_clamp_f32(1.0 + 0.05 * core_cnt, 1.0, 1.40);
        let void_gated = horizon.is_liminal()
            | (horizon.total_memory_support < self.liminal_layer.channel_void_floor_f32(1));
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

    /// Branchlessly modulates `frame` across Layer 0 (`Liminal`) and Layer 1 (`Stigmergic`) and,
    /// when Core Layer 2 (`ego_layer`) is active, performs in-place Ego intervention across all
    /// three layers and `frame`.
    #[inline]
    #[must_use]
    pub fn intervene_ego_and_modulate_frame_branchless(
        &mut self,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        venn_report: &StigmergicVennSerendipityReport,
        frame: GatewayFrame,
    ) -> GatewayFrame {
        let modulated = self.modulate_gateway_frame_branchless(frame, horizon, venn_report);
        if self.ego_layer.is_active() {
            self.ego_layer.intervene_on_sublayers_and_frame_branchless(
                &mut self.liminal_layer,
                &mut self.stigmergic_layer,
                delta,
                horizon,
                venn_report,
                modulated,
            )
        } else {
            modulated
        }
    }
}
