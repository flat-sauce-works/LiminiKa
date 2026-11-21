// name: src/core/src/drivers/ego/differential.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Non-parametric directed P-Frame residual network ([`EgoDifferentialNet`]) and
//! directed P-Frame [`DistortionDelta`] packets (128 B) evaluating word-level P-Frame
//! ensembles and chains without importing `SystemSpineState` or `GatewayFrame`.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

use crate::drivers::ego::intent::{EgoIntentAxes, LiminalEgoState};
use crate::foundation::{
    branchless_q7_weighted_ensemble_words_64, load_q7_words_64, swar_q7_circular_add_x8,
    swar_q7_scale_le128_with_bias_x8, Pod, SWAR_SPLAT_BYTE_64,
};
use crate::liminal::{
    branchless_select_u32, branchless_select_u8, gcso_q7_t, AttributionTag, BitCut8,
    EpistemicPartition, Scoped, ScopedCut, MAX_ATTENTION_HEADS, NUM_STIGMERGIC_GRANULES,
    SCOPE_REGION_KNOWN_UNKNOWN, SCOPE_REGION_SELF_EXCLUSIVE, SCOPE_REGION_SHARED_CORE,
};
use crate::stigmergy::crystal::{CrystallizedAttractorRecord, PhaseProfile64};
use crate::stigmergy::granule::StigmergicGranuleState;
use crate::stigmergy::ATTRACTOR_SLOT_SYSTEM_BASELINE;

/// Maximum number of inline P-Frame [`DistortionDelta`] nodes hosted by [`EgoDifferentialNet`].
pub const MAX_EGO_PFRAME_NODES: usize = 8;

/// Sentinel next-hop index marking termination of a [`DistortionDelta`] pointer chain.
pub const DISTORTION_DELTA_NO_NEXT_HOP: u16 = 0xFFFF;

/// Default maximum hop depth allowed when evaluating branched [`DistortionDelta`] chains (`4`).
pub const MAX_DISTORTION_CHAIN_HOPS: u8 = 4;

/// Flag bit `0`: [`DistortionDelta`] node is active and eligible for Venn trigger matching.
pub const DISTORTION_DELTA_FLAG_ACTIVE: u32 = 1 << 0;

/// Flag bit `1`: [`DistortionDelta`] node was consolidated into an I-Frame crystal anchor.
pub const DISTORTION_DELTA_FLAG_CONSOLIDATED_IFRAME: u32 = 1 << 1;

/// Flag bit `2`: [`DistortionDelta`] node carries valid model-baked 64-head Q7 phase deltas.
pub const DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID: u32 = 1 << 2;

/// First-class 128-byte (`#[repr(C, align(64))]`) directed P-Frame differential node representing
/// a gauge-relative deviation $\vec{\Delta}_t$ from `SystemSpineState` trajectory $f_{\mathrm{sys}}$.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistortionDelta {
    /// Logical P-Frame node identifier (`offset 0..4`).
    pub node_id: u32,
    /// Reference logical I-Frame anchor identifier (`offset 4..8`).
    pub anchor_ref_id: u32,
    /// Next chained P-Frame node index (`0..8`, or `0xFFFF` if terminal, `offset 8..10`).
    pub next_hop_idx: u16,
    /// Hop depth along the chain (`1..=4`, `offset 10`).
    pub hop_depth: u8,
    /// 8-granule Venn support bitmask required to trigger this delta (`offset 11`).
    pub venn_trigger_mask: u8,
    /// Bitmask of allowed Venn regions (`bit 0=S&O`, `bit 1=S\O`, `bit 2=O\S`, `offset 12`).
    pub venn_region_selector: u8,
    /// Stigmergic trace density in Q0.7 (`0..=127`, `offset 13`).
    pub pheromone_q7: u8,
    /// Directed delta scaling gain in Q0.7 (`0..=127`, `offset 14`).
    pub gain_q7: u8,
    /// 2-bit [`AttributionTag`] (`Stigmergy`/`Ego`, `offset 15`).
    pub attribution_tag: u8,
    /// Model-agnostic logical `EgoIntentAxes` transition (`16 B`, `offset 16..32`).
    pub intent_delta: EgoIntentAxes,
    /// Accumulated positive reinforcement counter for Cold-Path I-Frame consolidation (`offset 32..36`).
    pub reinforce_count: u32,
    /// Last active step epoch for stigmergic evaporation pruning (`offset 36..40`).
    pub last_active_epoch: u32,
    /// Optional integrity checksum over logical delta payload (`offset 40..44`).
    pub checksum_crc32: u32,
    /// Lifecycle and validity bitflags (`DISTORTION_DELTA_FLAG_*`, `offset 44..48`).
    pub flags: u32,
    /// Reserved aligned header bytes (`offset 48..64`).
    pub reserved_header: [u8; 16],
    /// Model-baked 64-head gauge-relative phase delta $\vec{\Delta}_t$ (`offset 64..128`).
    pub delta_phase_q7: [gcso_q7_t; MAX_ATTENTION_HEADS],
}

impl Default for DistortionDelta {
    #[inline(always)]
    fn default() -> Self {
        Self::new_empty(0, 0)
    }
}

impl DistortionDelta {
    /// Constructs a zeroed inactive 128-byte [`DistortionDelta`] packet.
    #[inline(always)]
    #[must_use]
    pub const fn new_empty(node_id: u32, anchor_ref_id: u32) -> Self {
        Self {
            node_id,
            anchor_ref_id,
            next_hop_idx: DISTORTION_DELTA_NO_NEXT_HOP,
            hop_depth: 1,
            venn_trigger_mask: 0xFF,
            venn_region_selector: SCOPE_REGION_SHARED_CORE | SCOPE_REGION_SELF_EXCLUSIVE,
            pheromone_q7: 64,
            gain_q7: 64,
            attribution_tag: AttributionTag::STIGMERGY_EGO.raw(),
            intent_delta: EgoIntentAxes::new(),
            reinforce_count: 0,
            last_active_epoch: 0,
            checksum_crc32: 0,
            flags: 0,
            reserved_header: [0; 16],
            delta_phase_q7: [0; MAX_ATTENTION_HEADS],
        }
    }

    /// Returns `true` when this node is marked active (`DISTORTION_DELTA_FLAG_ACTIVE`).
    #[inline(always)]
    #[must_use]
    pub const fn is_active(&self) -> bool {
        (self.flags & DISTORTION_DELTA_FLAG_ACTIVE) != 0
    }

    /// Returns `true` when this node is active (`DISTORTION_DELTA_FLAG_ACTIVE`) and carries
    /// valid model-baked 64-head deltas (`DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID`).
    #[inline(always)]
    #[must_use]
    pub const fn is_baked_active(&self) -> bool {
        const MASK: u32 = DISTORTION_DELTA_FLAG_ACTIVE | DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID;
        (self.flags & MASK) == MASK
    }
}

// SAFETY: `DistortionDelta` is `#[repr(C, align(64))]`, exact size 128 bytes with zero padding.
unsafe impl Pod for DistortionDelta {}

/// Inline fixed-capacity (`8` nodes, `0` heap allocations) non-parametric P-Frame residual
/// neural network evaluating directed [`DistortionDelta`] pointer chains (up to `max_hops`
/// in `1..=4`) triggered by Venn region support and stigmergic density.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EgoDifferentialNet {
    /// Fixed array of up to 8 directed P-Frame [`DistortionDelta`] nodes.
    pub nodes: [DistortionDelta; MAX_EGO_PFRAME_NODES],
    /// 8-bit bitmask marking populated nodes (`0..=7`).
    pub active_node_mask: u8,
    /// Maximum hop traversal bound (`1..=4`, allows clean 1-hop vs multi-hop comparison).
    pub max_hops: u8,
    /// Minimum hop depth before Cold-Path GOP consolidation crystallizes an I-Frame anchor.
    pub consolidation_hop_limit: u8,
    /// Stigmergic pheromone floor (`0..=127`) below which unused P-Frame nodes evaporate.
    pub evaporation_floor_q7: u8,
    /// Hop count traversed on the most recent evaluation.
    pub last_hops_traversed: u8,
    /// Mean differential magnitude (`0..=127`) observed on the most recent evaluation.
    pub last_delta_norm_q7: u8,
    /// Monotonic Cold-Path maintenance epoch counter.
    pub epoch: u32,
}

impl Default for EgoDifferentialNet {
    #[inline(always)]
    fn default() -> Self {
        Self::new_empty(1)
    }
}

impl EgoDifferentialNet {
    /// Creates an empty inert [`EgoDifferentialNet`] with bounded `max_hops` (`1..=4`).
    #[inline(always)]
    #[must_use]
    pub const fn new_empty(max_hops: u8) -> Self {
        let clamped_hops = if max_hops == 0 {
            1
        } else if max_hops > MAX_DISTORTION_CHAIN_HOPS {
            MAX_DISTORTION_CHAIN_HOPS
        } else {
            max_hops
        };
        Self {
            nodes: [DistortionDelta::new_empty(0, 0); MAX_EGO_PFRAME_NODES],
            active_node_mask: 0,
            max_hops: clamped_hops,
            consolidation_hop_limit: 3,
            evaporation_floor_q7: 16,
            last_hops_traversed: 0,
            last_delta_norm_q7: 0,
            epoch: 0,
        }
    }

    /// Deterministically initializes the 3-node P-Frame net from [`LiminalEgoState`] and
    /// [`EgoIntentAxes`], evaluated as a 1-hop parallel perspective ensemble when `max_hops == 1`
    /// or as a sequential multi-hop chain when `max_hops >= 2`:
    /// - Node 0 (`S & O`): persona heading alignment delta
    /// - Node 1 (`S \ O`): sovereign counter-basin stylistic delta
    /// - Node 2 (`O \ S`): transgressive exterior bridge delta
    #[inline]
    #[must_use]
    pub fn from_ego_state_and_axes(
        ego: &LiminalEgoState,
        axes: &EgoIntentAxes,
        max_hops: u8,
    ) -> Self {
        let mut net = Self::new_empty(max_hops);
        if !ego.is_active() {
            return net;
        }

        let make_phase = |base: &[gcso_q7_t; NUM_STIGMERGIC_GRANULES],
                          shift: i8,
                          flip_odd: bool|
         -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
            let mut out = [0 as gcso_q7_t; MAX_ATTENTION_HEADS];
            let mut g = 0usize;
            while g < NUM_STIGMERGIC_GRANULES {
                let gval = i32::from(base[g]);
                let dir = if flip_odd && ((g & 1) == 1) {
                    -gval
                } else {
                    gval
                };
                let s = (dir + i32::from(shift)).clamp(-127, 127) as i8;
                let base_h = g << 3;
                let mut k = 0usize;
                while k < 8 {
                    let mod_off = ((k as i32) - 3) * (i32::from(shift) >> 4);
                    out[base_h + k] = (i32::from(s) + mod_off).clamp(-127, 127) as gcso_q7_t;
                    k += 1;
                }
                g += 1;
            }
            out
        };

        let mut n0 = DistortionDelta::new_empty(1, u32::from(ATTRACTOR_SLOT_SYSTEM_BASELINE) + 1);
        n0.next_hop_idx = 1;
        n0.hop_depth = 1;
        n0.venn_trigger_mask = if axes.active_granule_bias_mask == 0 {
            0xFF
        } else {
            axes.active_granule_bias_mask
        };
        n0.venn_region_selector = SCOPE_REGION_SHARED_CORE;
        n0.pheromone_q7 = 88;
        n0.gain_q7 = ego.counter_basin_thrust_q7.clamp(32, 127);
        n0.intent_delta = *axes;
        n0.flags = DISTORTION_DELTA_FLAG_ACTIVE | DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID;
        n0.delta_phase_q7 = make_phase(&ego.sovereign_granule_phase_q7, 4, false);

        let mut n1 = DistortionDelta::new_empty(2, u32::from(ATTRACTOR_SLOT_SYSTEM_BASELINE) + 2);
        n1.next_hop_idx = 2;
        n1.hop_depth = 2;
        n1.venn_trigger_mask = 0xAA;
        n1.venn_region_selector = SCOPE_REGION_SHARED_CORE | SCOPE_REGION_SELF_EXCLUSIVE;
        n1.pheromone_q7 = 76;
        n1.gain_q7 = ego.irrational_defiance_q7.clamp(24, 127);
        n1.intent_delta = *axes;
        n1.flags = DISTORTION_DELTA_FLAG_ACTIVE | DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID;
        n1.delta_phase_q7 = make_phase(
            &ego.sovereign_granule_phase_q7,
            ego.transgression_bias_q7 >> 2,
            true,
        );

        let mut n2 = DistortionDelta::new_empty(3, u32::from(ATTRACTOR_SLOT_SYSTEM_BASELINE) + 3);
        n2.next_hop_idx = DISTORTION_DELTA_NO_NEXT_HOP;
        n2.hop_depth = 3;
        n2.venn_trigger_mask = 0x55;
        n2.venn_region_selector =
            SCOPE_REGION_KNOWN_UNKNOWN | SCOPE_REGION_SELF_EXCLUSIVE | SCOPE_REGION_SHARED_CORE;
        n2.pheromone_q7 = 64;
        n2.gain_q7 = ego.transgression_bias_q7.unsigned_abs().clamp(24, 127);
        n2.intent_delta = *axes;
        n2.flags = DISTORTION_DELTA_FLAG_ACTIVE | DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID;
        n2.delta_phase_q7 = make_phase(
            &ego.sovereign_granule_phase_q7,
            -(ego.transgression_bias_q7 >> 2),
            false,
        );

        net.nodes[0] = n0;
        net.nodes[1] = n1;
        net.nodes[2] = n2;
        net.active_node_mask = 0b0000_0111;
        net
    }

    /// Records positive traversal along up to `hops_traversed` chain hops starting at `entry_idx`,
    /// advancing `reinforce_count` and stamping `last_active_epoch = self.epoch` for Cold-Path GOP
    /// consolidation and evaporation pruning.
    #[inline]
    pub fn record_chain_activation(&mut self, entry_idx: usize, hops_traversed: u8) {
        self.last_hops_traversed = hops_traversed;
        let mut cur = entry_idx;
        let mut h = 0u8;
        let cap = hops_traversed.min(MAX_DISTORTION_CHAIN_HOPS);
        while h < cap && cur < MAX_EGO_PFRAME_NODES {
            if (self.active_node_mask & (1u8 << cur)) == 0 {
                break;
            }
            let node = &mut self.nodes[cur];
            node.reinforce_count = node.reinforce_count.saturating_add(1);
            node.last_active_epoch = self.epoch;
            cur = usize::from(node.next_hop_idx);
            h = h.saturating_add(1);
        }
    }

    /// Records positive activation for every node bit set in `matched_mask`, advancing
    /// `reinforce_count`, nudging `pheromone_q7` (capped at `127`), and stamping
    /// `last_active_epoch = self.epoch` for Cold-Path GOP consolidation and evaporation pruning.
    #[inline]
    pub fn record_matched_activations(&mut self, matched_mask: u8) {
        let active_matched = matched_mask & self.active_node_mask;
        self.last_hops_traversed = if self.max_hops <= 1 {
            u8::from(active_matched != 0)
        } else {
            (active_matched.count_ones() as u8).min(self.max_hops)
        };
        for (idx, node) in self.nodes.iter_mut().enumerate().take(MAX_EGO_PFRAME_NODES) {
            if ((active_matched >> idx) & 1) != 0 {
                node.reinforce_count = node.reinforce_count.saturating_add(1);
                node.pheromone_q7 = node.pheromone_q7.saturating_add(1).min(127);
                node.last_active_epoch = self.epoch;
            }
        }
    }

    /// Cold-Path GOP maintenance: consolidates reinforced multi-hop P-Frame chains into an I-Frame
    /// [`CrystallizedAttractorRecord`] and evaporates stale unreinforced P-Frame nodes below `evaporation_floor_q7`.
    #[inline]
    pub fn consolidate_and_evaporate_cold_path(
        &mut self,
        current_epoch: u32,
    ) -> Option<CrystallizedAttractorRecord> {
        self.epoch = current_epoch;
        let mut crystallized: Option<CrystallizedAttractorRecord> = None;

        for (idx, node) in self.nodes.iter_mut().enumerate().take(MAX_EGO_PFRAME_NODES) {
            let bit = 1u8 << idx;
            if (self.active_node_mask & bit) == 0 {
                continue;
            }
            if !node.is_active() {
                self.active_node_mask &= !bit;
                continue;
            }
            if node.hop_depth >= self.consolidation_hop_limit
                && node.reinforce_count >= 4
                && (node.flags & DISTORTION_DELTA_FLAG_CONSOLIDATED_IFRAME) == 0
            {
                let bridge_count = node.reinforce_count.min(u32::from(u8::MAX)) as u8;
                let grounded_profile = match Scoped::from_delta(PhaseProfile64(node.delta_phase_q7))
                    .cut_by_lacuna_q7(true, node.gain_q7.max(1), 0)
                {
                    ScopedCut::SelfRef(aligned) => Some(aligned.into_grounded()),
                    ScopedCut::Exterior(spec) => spec
                        .bridge_to_shared(bridge_count)
                        .ok()
                        .map(Scoped::into_grounded),
                    ScopedCut::Unmarked(_) | ScopedCut::Bottom(_) => None,
                };
                if let Some(grounded) = grounded_profile {
                    let rec = CrystallizedAttractorRecord::from_grounded_delta_64(
                        node.anchor_ref_id,
                        ATTRACTOR_SLOT_SYSTEM_BASELINE,
                        node.reinforce_count,
                        f32::from(node.gain_q7) * (1.0 / 127.0),
                        grounded,
                    );
                    node.flags |= DISTORTION_DELTA_FLAG_CONSOLIDATED_IFRAME;
                    crystallized = Some(rec);
                }
            }

            if current_epoch.saturating_sub(node.last_active_epoch) >= 4 {
                let decay = (node.pheromone_q7 >> 3).max(1);
                node.pheromone_q7 = node.pheromone_q7.saturating_sub(decay);
                if node.pheromone_q7 < self.evaporation_floor_q7 {
                    node.flags &= !DISTORTION_DELTA_FLAG_ACTIVE;
                    self.active_node_mask &= !bit;
                }
            }
        }
        crystallized
    }

    /// Single-source-of-truth word-level P-Frame evaluator independent of `SystemSpineState`,
    /// `DistortionTension`, and `GatewayFrame`.
    ///
    /// - When `self.max_hops <= 1`, executes the **1-Hop Parallel Perspective Ensemble**: evaluates
    ///   all active baked nodes against `epistemic` in parallel, accumulates weighted per-head
    ///   Q7 deltas in `i32`, and normalizes by an integer reciprocal LUT (`0 fdiv`, `0` heap alloc)
    ///   before circular Q7 addition onto `base_words`.
    /// - When `self.max_hops >= 2`, executes the **Multi-Hop Sequential Chain**: selects the
    ///   strongest matching entry node and follows `next_hop_idx` up to
    ///   `self.max_hops.min(MAX_DISTORTION_CHAIN_HOPS)`.
    ///
    /// Returns `(out_words, delta_norm_q7, hops_traversed, entry_idx, matched_mask)`.
    #[inline]
    #[must_use]
    pub fn evaluate_pframe_words_branchless(
        &self,
        ego: &LiminalEgoState,
        epistemic: &EpistemicPartition,
        stigmergy: &StigmergicGranuleState,
        effective_ego_scale_q7: u8,
        base_words: &[u64; NUM_STIGMERGIC_GRANULES],
    ) -> ([u64; NUM_STIGMERGIC_GRANULES], u8, u8, u8, u8) {
        if !ego.is_active() || effective_ego_scale_q7 == 0 || self.active_node_mask == 0 {
            return (*base_words, 0, 0, 0, 0);
        }

        let region_bits =
            branchless_select_u8(epistemic.shared_core_mask != 0, SCOPE_REGION_SHARED_CORE, 0)
                | branchless_select_u8(
                    epistemic.self_exclusive_mask != 0,
                    SCOPE_REGION_SELF_EXCLUSIVE,
                    0,
                )
                | branchless_select_u8(
                    epistemic.known_unknown_mask != 0,
                    SCOPE_REGION_KNOWN_UNKNOWN,
                    0,
                );
        let region_fallback = branchless_select_u8(
            region_bits != 0,
            region_bits,
            SCOPE_REGION_SHARED_CORE | SCOPE_REGION_SELF_EXCLUSIVE,
        );
        let combined_support = epistemic.shared_core_mask
            | epistemic.self_exclusive_mask
            | epistemic.known_unknown_mask;
        let support_mask = branchless_select_u8(combined_support != 0, combined_support, 0xFF);

        let sens_boost_q7 = u32::from(epistemic.boundary_sensitivity_q7).max(48);
        let eff_scale = u32::from(effective_ego_scale_q7);
        let mut ph_sum = 0u32;
        for &ph in stigmergy
            .granule_pheromone_q7
            .iter()
            .take(NUM_STIGMERGIC_GRANULES)
        {
            ph_sum += u32::from(ph);
        }
        let mean_stig_ph = (ph_sum >> 3).clamp(24, 127);

        if self.max_hops <= 1 {
            let mut matched_mask = 0u8;
            let mut weights_q7 = [0u8; MAX_EGO_PFRAME_NODES];

            for (i, node) in self.nodes.iter().enumerate().take(MAX_EGO_PFRAME_NODES) {
                let active = ((self.active_node_mask >> i) & 1) != 0 && node.is_baked_active();
                let reg_ok = BitCut8::split(node.venn_region_selector, region_fallback).inside != 0;
                let trig_ok = BitCut8::split(node.venn_trigger_mask, support_mask).inside != 0;
                if active && reg_ok && trig_ok {
                    matched_mask |= 1u8 << i;
                    let hop_weight_q7 =
                        (((u32::from(node.gain_q7) * eff_scale) >> 7) * sens_boost_q7) >> 7;
                    let ph_blend_q7 = (u32::from(node.pheromone_q7) + mean_stig_ph) >> 1;
                    let dens_weight_q7 = ((hop_weight_q7 * ph_blend_q7) >> 7).clamp(8, 127) as u8;
                    weights_q7[i] = dens_weight_q7;
                }
            }

            if matched_mask == 0 {
                return (*base_words, 0, 0, 0, 0);
            }

            let profiles: [&[gcso_q7_t; MAX_ATTENTION_HEADS]; MAX_EGO_PFRAME_NODES] =
                core::array::from_fn(|i| &self.nodes[i].delta_phase_q7);
            let (out_words, avg_delta, entry_idx) = branchless_q7_weighted_ensemble_words_64(
                profiles,
                &weights_q7,
                matched_mask,
                base_words,
            );
            return (out_words, avg_delta, 1, entry_idx, matched_mask);
        }

        let mut best_entry_idx = 0usize;
        let mut best_entry_score = 0u32;
        for (i, n) in self.nodes.iter().enumerate().take(MAX_EGO_PFRAME_NODES) {
            let active = ((self.active_node_mask >> i) & 1) != 0 && n.is_baked_active();
            let reg_ok = BitCut8::split(n.venn_region_selector, region_fallback).inside != 0;
            let trig_overlap = BitCut8::split(n.venn_trigger_mask, support_mask)
                .inside
                .count_ones();
            let score = if active && reg_ok && trig_overlap > 0 {
                u32::from(n.pheromone_q7).saturating_mul(trig_overlap)
            } else {
                0
            };
            let take = score > best_entry_score;
            best_entry_score = branchless_select_u32(take, score, best_entry_score);
            best_entry_idx = if take { i } else { best_entry_idx };
        }
        if best_entry_score == 0 {
            return (*base_words, 0, 0, 0, 0);
        }

        let hop_cap = self.max_hops.min(MAX_DISTORTION_CHAIN_HOPS);
        let mut acc_words = *base_words;
        let mut accum_norm_sum = 0u32;
        let mut hops_traversed = 0u8;
        let mut matched_mask = 0u8;
        let mut cur_idx = best_entry_idx as u16;
        let mut hop = 0u8;

        while hop < MAX_DISTORTION_CHAIN_HOPS {
            let in_bounds = (hop < hop_cap)
                && (usize::from(cur_idx) < MAX_EGO_PFRAME_NODES)
                && (((self.active_node_mask >> (cur_idx & 7)) & 1) != 0);
            if !in_bounds {
                break;
            }
            let node = &self.nodes[usize::from(cur_idx)];
            if !node.is_baked_active() {
                break;
            }
            matched_mask |= 1u8 << (cur_idx & 7);
            let hop_weight_q7 = (((u32::from(node.gain_q7) * eff_scale) >> 7) * sens_boost_q7) >> 7;
            let dens_weight_q7 = ((hop_weight_q7 * mean_stig_ph) >> 7).clamp(8, 127) as u64;
            let bias_restore =
                (128u64.wrapping_sub(dens_weight_q7)).wrapping_mul(SWAR_SPLAT_BYTE_64);
            let node_words = load_q7_words_64(&node.delta_phase_q7);
            for (acc_w, &nw) in acc_words
                .iter_mut()
                .zip(node_words.iter())
                .take(NUM_STIGMERGIC_GRANULES)
            {
                let d_w = swar_q7_scale_le128_with_bias_x8(nw, dens_weight_q7, bias_restore);
                *acc_w = swar_q7_circular_add_x8(*acc_w, d_w);
            }
            accum_norm_sum = accum_norm_sum.saturating_add(dens_weight_q7 as u32);
            hops_traversed = hops_traversed.saturating_add(1);
            cur_idx = node.next_hop_idx;
            hop = hop.saturating_add(1);
        }

        const INV_HOPS_Q7: [u32; 5] = [0, 128, 64, 43, 32];
        let avg_delta = ((accum_norm_sum * INV_HOPS_Q7[usize::from(hops_traversed.min(4))]) >> 7)
            .min(127) as u8;
        (
            acc_words,
            avg_delta,
            hops_traversed,
            best_entry_idx as u8,
            matched_mask,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drivers::ego::intent::LiminalEgoPreset;

    #[test]
    fn test_one_hop_parallel_perspective_ensemble_and_reinforcement() {
        let ego = LiminalEgoState::from_preset(LiminalEgoPreset::High);
        let axes = EgoIntentAxes::new()
            .with_autonomy_q7(72)
            .with_transgression_q7(48);
        let mut net_1 = EgoDifferentialNet::from_ego_state_and_axes(&ego, &axes, 1);
        let net_3 = EgoDifferentialNet::from_ego_state_and_axes(&ego, &axes, 3);
        let epistemic = EpistemicPartition {
            shared_core_mask: 0x0F,
            self_exclusive_mask: 0x30,
            known_unknown_mask: 0x40,
            true_void_mask: 0x80,
            boundary_sensitivity_q7: 96,
            ..EpistemicPartition::default()
        };
        let stig = StigmergicGranuleState::new();
        let base_words = [0u64; NUM_STIGMERGIC_GRANULES];

        let (w1, norm1, hops1, entry1, mask1) =
            net_1.evaluate_pframe_words_branchless(&ego, &epistemic, &stig, 96, &base_words);
        let (w3, _norm3, hops3, _entry3, mask3) =
            net_3.evaluate_pframe_words_branchless(&ego, &epistemic, &stig, 96, &base_words);

        assert_eq!(hops1, 1);
        assert!(norm1 > 0);
        assert!(
            mask1.count_ones() >= 2,
            "1-hop ensemble should fire multiple perspectives, got mask={mask1:#010b}"
        );
        assert_eq!(entry1, mask1.trailing_zeros() as u8);
        assert_eq!(hops3, 3);
        assert_eq!(mask3, 0b0000_0111);
        assert_ne!(
            w1, w3,
            "1-hop normalized ensemble should differ from 3-hop chain"
        );

        let ph_before: [u8; 3] = [
            net_1.nodes[0].pheromone_q7,
            net_1.nodes[1].pheromone_q7,
            net_1.nodes[2].pheromone_q7,
        ];
        net_1.epoch = 5;
        net_1.record_matched_activations(mask1);
        for (idx, prev_ph) in ph_before.iter().enumerate() {
            if ((mask1 >> idx) & 1) != 0 {
                assert_eq!(net_1.nodes[idx].reinforce_count, 1);
                assert_eq!(net_1.nodes[idx].last_active_epoch, 5);
                assert_eq!(
                    net_1.nodes[idx].pheromone_q7,
                    prev_ph.saturating_add(1).min(127)
                );
            }
        }
    }
}
