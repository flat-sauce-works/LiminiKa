// name: src/core/src/drivers/ego/differential.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Non-parametric directed P-Frame residual network ([`EgoDifferentialNet`]) evaluating
//! word-level P-Frame chains without importing `SystemSpineState` or `GatewayFrame`.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

use crate::drivers::distortion::{
    DistortionDelta, DistortionTension, DISTORTION_DELTA_FLAG_ACTIVE,
    DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID, DISTORTION_DELTA_FLAG_CONSOLIDATED_IFRAME,
    DISTORTION_DELTA_NO_NEXT_HOP, MAX_DISTORTION_CHAIN_HOPS, MAX_EGO_PFRAME_NODES,
};
use crate::drivers::ego::intent::{EgoIntentAxes, LiminalEgoField};
use crate::foundation::{
    load_q7_words_64, swar_q7_circular_add_x8, swar_q7_mean_i8_from_words_64,
    swar_q7_scale_le128_with_bias_x8, SWAR_SPLAT_BYTE_64,
};
use crate::liminal::{
    branchless_select_u32, branchless_select_u8, gcso_q7_t, EpistemicPartition, Scoped, ScopedCut,
    ATTRACTOR_SLOT_SYSTEM_BASELINE, MAX_ATTENTION_HEADS, NUM_STIGMERGIC_GRANULES,
    SCOPE_REGION_KNOWN_UNKNOWN, SCOPE_REGION_SELF_EXCLUSIVE, SCOPE_REGION_SHARED_CORE,
};
use crate::stigmergy::crystal::CrystallizedAttractorRecord;
use crate::stigmergy::granule::StigmergicGranuleField;

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

    /// Deterministically initializes a 3-hop P-Frame chain from [`LiminalEgoField`] and [`EgoIntentAxes`]:
    /// - Node 0 (`S & O`): persona heading alignment delta
    /// - Node 1 (`S \ O`): sovereign counter-basin stylistic delta
    /// - Node 2 (`O \ S`): transgressive exterior bridge delta
    #[inline]
    #[must_use]
    pub fn from_ego_field_and_axes(
        ego: &LiminalEgoField,
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
                let grounded_profile = match Scoped::from_delta(node.delta_phase_q7)
                    .cut_by_lacuna_q7(true, node.gain_q7.max(1), 0)
                {
                    ScopedCut::SelfRef(aligned) => Some(aligned.seal_grounded()),
                    ScopedCut::Exterior(spec) => spec
                        .bridge_to_shared(bridge_count)
                        .ok()
                        .map(Scoped::seal_grounded),
                    ScopedCut::Void(_) => None,
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

    /// Word-level P-Frame chain core independent of `SystemSpineState` and `GatewayFrame`.
    /// Returns `(modulated_words, mean_phase_tag_q7, avg_delta_q7, hops_traversed, entry_idx, applied)`.
    #[inline]
    #[must_use]
    pub fn evaluate_pframe_words_branchless(
        &self,
        ego: &LiminalEgoField,
        spine_locked: bool,
        part: &EpistemicPartition,
        stigmergy: &StigmergicGranuleField,
        tension: &DistortionTension,
        base_words: &[u64; NUM_STIGMERGIC_GRANULES],
    ) -> ([u64; NUM_STIGMERGIC_GRANULES], gcso_q7_t, u8, u8, u8, bool) {
        if !ego.is_active()
            || tension.effective_ego_scale_q7 == 0
            || spine_locked
            || self.active_node_mask == 0
        {
            return (*base_words, 0, 0, 0, 0, false);
        }

        let region_bits =
            branchless_select_u8(part.shared_core_mask != 0, SCOPE_REGION_SHARED_CORE, 0)
                | branchless_select_u8(
                    part.self_exclusive_mask != 0,
                    SCOPE_REGION_SELF_EXCLUSIVE,
                    0,
                )
                | branchless_select_u8(part.known_unknown_mask != 0, SCOPE_REGION_KNOWN_UNKNOWN, 0);
        let region_fallback = branchless_select_u8(
            region_bits != 0,
            region_bits,
            SCOPE_REGION_SHARED_CORE | SCOPE_REGION_SELF_EXCLUSIVE,
        );
        let combined_support =
            part.shared_core_mask | part.self_exclusive_mask | part.known_unknown_mask;
        let support_mask = branchless_select_u8(combined_support != 0, combined_support, 0xFF);

        let mut best_entry_idx = 0usize;
        let mut best_entry_score = 0u32;
        for (i, n) in self.nodes.iter().enumerate().take(MAX_EGO_PFRAME_NODES) {
            let active = ((self.active_node_mask >> i) & 1) != 0 && n.is_baked_active();
            let reg_ok = (n.venn_region_selector & region_fallback) != 0;
            let trig_overlap = (n.venn_trigger_mask & support_mask).count_ones();
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
            return (*base_words, 0, 0, 0, 0, false);
        }

        let hop_cap = self.max_hops.min(MAX_DISTORTION_CHAIN_HOPS);
        let sens_boost_q7 = u32::from(part.boundary_sensitivity_q7).max(48);
        let eff_scale = u32::from(tension.effective_ego_scale_q7);
        let mut ph_sum = 0u32;
        for &ph in stigmergy
            .granule_pheromone_q7
            .iter()
            .take(NUM_STIGMERGIC_GRANULES)
        {
            ph_sum += u32::from(ph);
        }
        let mean_stig_ph = (ph_sum >> 3).clamp(24, 127);

        let mut acc_words = *base_words;
        let mut accum_norm_sum = 0u32;
        let mut hops_traversed = 0u8;
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

        let mean_phase = swar_q7_mean_i8_from_words_64(&acc_words);
        const INV_HOPS_Q7: [u32; 5] = [0, 128, 64, 43, 32];
        let avg_delta = ((accum_norm_sum * INV_HOPS_Q7[usize::from(hops_traversed.min(4))]) >> 7)
            .min(127) as u8;
        (
            acc_words,
            mean_phase,
            avg_delta,
            hops_traversed,
            best_entry_idx as u8,
            true,
        )
    }
}
