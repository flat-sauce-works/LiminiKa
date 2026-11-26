// name: src/core/src/runtime/gateway/frame/signal.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`ManifoldGatewaySignal`] trait, granule voting, aperture shaping, and SWAR active-head folding.

use super::super::*;
use super::GatewayFrame;
use super::synthesis::combine_pair_aperture_frame_pure;
use crate::foundation::{
    ACTION_MODE_LAYER_SKIP, ACTION_PRESET_EXPRESS_PHASE, ATTRACTOR_SLOT_ID_MASK,
    DEFAULT_PHASE_CLAMP_MAX_RAD, DEFAULT_PHASE_STEP_MIN_RAD, SWAR_SPLAT_BYTE_64,
    TAG_ACTION_MODE_MASK, TaggedPtr, branchless_clamp_f32, branchless_flush_subnormal_f32,
    branchless_lerp_f32, branchless_max_f32, branchless_min_f32, branchless_rad_to_phase_step_u8,
    branchless_relu_f32, coherence_from_word_arrays_with_both_uniform_flags_64,
    fast_inv_pos_normal_f32, is_word_array_uniform_64, pack_action_slot_tag, swar_q7_avg_round_x8,
    swar_q7_circular_add_x8, swar_q7_mean_i8_from_words_64,
    swar_q7_mean_i8_from_words_with_uniform_flag_64, swar_q7_scale_le128_with_bias_x8,
    swar_slerp_words_with_uniform_flag_64, swar_words_equal_64, swar_words_nonzero_64,
    words_to_q7_array_64,
};
use crate::liminal::{
    BitCut32, DEFAULT_ENTROPY_EPS, MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS,
    NUM_STIGMERGIC_GRANULES, THRESHOLD_MANIFOLD_GATEWAY, branchless_select_f32,
    branchless_select_u8, branchless_select_u32, branchless_select_u64,
};
use crate::runtime::boundary_delta::LiminalBoundaryDelta;
use crate::runtime::context::CoreStateStack;
use crate::runtime::horizon::EpistemicHorizonReading;
use crate::stigmergy::{
    ATTRACTOR_SLOT_SYSTEM_BASELINE, MAX_ATTRACTOR_PULL_FORCE, OverlapDiscoveryReport,
    StigmergicFragmentEnsembleReport,
};

/// **The Single Composite Manifold Gateway Trait (`ManifoldGatewaySignal: CausalProvenanceSignal`)**.
pub trait ManifoldGatewaySignal: CausalProvenanceSignal {
    /// Wide composite cognitive aperture combining causal origins, circuit plasticity, KV phase
    /// cache, relative coupling, and dissipative crystallization (`0x0000_00FF`).
    const GATEWAY_APERTURE_MASK: u32 = THRESHOLD_MANIFOLD_GATEWAY;

    /// Complement bitmask outside the Gateway horizon (`!THRESHOLD_MANIFOLD_GATEWAY`).
    const GATEWAY_UNOBSERVED_LACUNA_MASK: u32 = !Self::GATEWAY_APERTURE_MASK;

    /// Branchlessly partitions `self.causal_provenance_bits()` into
    /// `(cognized_bits, unobserved_lacuna_bits)` for [`Self::GATEWAY_APERTURE_MASK`].
    #[inline(always)]
    fn partition_gateway_horizon(&self) -> (u32, u32) {
        let cut = BitCut32::split(self.causal_provenance_bits(), Self::GATEWAY_APERTURE_MASK);
        (cut.inside, cut.outside)
    }

    /// Synthesizes all constituent blanket signals into a 128-byte [`GatewayFrame`] in
    /// constant-time branchless arithmetic ($C_{\mathrm{alloc}} = 0$).
    fn synthesize_gateway_frame(&self) -> GatewayFrame;

    /// Synthesizes `self` into both a 128-byte [`GatewayFrame`] and its register-resident 8 `u64`
    /// SWAR phase words (`[u64; 8]`), eliminating stack store-to-load forwarding stalls across
    /// downstream ensemble folding, [`crate::stigmergy::StigmergicSubstrate`] steering, and Online Toroidal $k$-Means steps.
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let frame = self.synthesize_gateway_frame();
        let words = *frame.phase_words_ref();
        (frame, words)
    }

    /// Lightweight internal/ensemble fold hook returning `(GatewayFrame, [u64; 8])` where
    /// `GatewayFrame.phase_deltas`, `mean_phase_tag_q7`, and `tagged_ptr` may be deferred until
    /// the outer `[T]` ensemble accumulator completes its final reduction.
    #[inline(always)]
    fn synthesize_gateway_ensemble_item(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        self.synthesize_gateway_frame_with_words()
    }

    /// Fast predicate returning `true` if `self` carries an active non-passive signal during `[T]` ensemble folding.
    #[inline(always)]
    fn has_active_gateway_ensemble_signal(&self) -> bool {
        true
    }

    /// Synthesizes a contiguous slice of `Self` into a 128-byte [`GatewayFrame`] and its 8
    /// register-resident `u64` SWAR phase words (`[u64; 8]`), accumulating scalar metadata in
    /// registers and materializing `GatewayFrame` only once upon loop completion.
    #[inline(always)]
    fn synthesize_gateway_slice_with_words(
        slice: &[Self],
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES])
    where
        Self: Sized,
    {
        fold_gateway_slice_with_query_words_core(slice, None)
    }

    /// Evaluates this signal through its live dynamic cognitive aperture against all three core
    /// layers (`Liminal` + `Stigmergic` + `Ego`) prior to final frame synthesis, gating out-of-window
    /// SWAR granules and applying in-place three-layer Ego modulation.
    #[inline]
    fn synthesize_gateway_with_core_layers(
        &self,
        stack: &mut CoreStateStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        overlap_report: &OverlapDiscoveryReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let mut frame = self.synthesize_gateway_with_aperture_from_stack(stack);
        stack.intervene_ego_and_modulate_frame_branchless(
            delta,
            horizon,
            overlap_report,
            &mut frame,
        );
        let out_words = *frame.phase_words_ref();
        (frame, out_words)
    }

    /// Read-only counterpart to [`Self::synthesize_gateway_with_core_layers`] for per-layer
    /// Hot-Path RoPE steering: resolves the dynamic trait cognitive aperture and modulates `frame`
    /// across all three core layers (`Liminal` + `Stigmergic` + `Ego`) without mutating `stack`.
    #[inline]
    fn synthesize_gateway_with_core_layers_readonly(
        &self,
        stack: &CoreStateStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        overlap_report: &OverlapDiscoveryReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let mut frame = self.synthesize_gateway_with_aperture_from_stack(stack);
        stack.modulate_gateway_frame_with_ego_readonly(delta, horizon, overlap_report, &mut frame);
        let out_words = *frame.phase_words_ref();
        (frame, out_words)
    }

    /// Shared helper resolving `self`'s dynamic cognitive aperture against `stack` and shaping
    /// the base synthesized [`GatewayFrame`].
    #[inline(always)]
    fn synthesize_gateway_with_aperture_from_stack(&self, stack: &CoreStateStack) -> GatewayFrame {
        let (dyn_mask, dyn_gain_q7, dyn_shift_q7) = self.resolve_dynamic_core_aperture(stack);
        let (mut frame, mut words) = self.synthesize_gateway_frame_with_words();
        apply_dynamic_aperture_to_frame_and_words(
            &mut frame,
            &mut words,
            dyn_mask,
            dyn_gain_q7,
            dyn_shift_q7,
        );
        frame
    }

    /// Dynamically routes and combines `self` (`A`) and `other` (`B`) across the 8 SWAR granules
    /// according to their live core-layer cognitive apertures:
    /// - `mask_a & !mask_b`: exclusive routing to `A`
    /// - `!mask_a & mask_b`: exclusive routing to `B`
    /// - `mask_a & mask_b`: aperture overlap cooperative fusion or Layer 2 Ego conflict tie-breaking
    /// - `!mask_a & !mask_b`: gated to `Liminal` exterior (`0`)
    #[inline]
    fn combine_pair_with_core_layers<B: ManifoldGatewaySignal + ?Sized>(
        &self,
        other: &B,
        stack: &mut CoreStateStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        overlap_report: &OverlapDiscoveryReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let mut combined = combine_pair_aperture_frame_pure(self, other, stack);
        stack.intervene_ego_and_modulate_frame_branchless(
            delta,
            horizon,
            overlap_report,
            &mut combined,
        );
        let final_words = *combined.phase_words_ref();
        (combined, final_words)
    }

    /// Read-only counterpart to [`Self::combine_pair_with_core_layers`] for per-layer
    /// Hot-Path steering without mutating `stack`.
    #[inline]
    fn combine_pair_with_core_layers_readonly<B: ManifoldGatewaySignal + ?Sized>(
        &self,
        other: &B,
        stack: &CoreStateStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        overlap_report: &OverlapDiscoveryReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let mut combined = combine_pair_aperture_frame_pure(self, other, stack);
        stack.modulate_gateway_frame_with_ego_readonly(
            delta,
            horizon,
            overlap_report,
            &mut combined,
        );
        let final_words = *combined.phase_words_ref();
        (combined, final_words)
    }

    /// Aggregates a slice of fragmented signal experts (`fragments`) across the 8 SWAR granules
    /// using per-fragment dynamic core apertures, quorum agreement amplification, Layer 0
    /// uncertainty damping, and Layer 2 Ego tie-breaking (`0` heap allocations).
    #[inline]
    fn synthesize_fragment_ensemble_with_core_layers(
        fragments: &[Self],
        stack: &mut CoreStateStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        overlap_report: &OverlapDiscoveryReport,
    ) -> (
        GatewayFrame,
        [u64; NUM_STIGMERGIC_GRANULES],
        StigmergicFragmentEnsembleReport,
    )
    where
        Self: Sized,
    {
        if fragments.is_empty() {
            return (
                GatewayFrame::default(),
                [0u64; NUM_STIGMERGIC_GRANULES],
                StigmergicFragmentEnsembleReport::default(),
            );
        }

        let mut pos_votes = [0u8; NUM_STIGMERGIC_GRANULES];
        let mut neg_votes = [0u8; NUM_STIGMERGIC_GRANULES];
        let mut pos_energy = [0.0_f32; NUM_STIGMERGIC_GRANULES];
        let mut neg_energy = [0.0_f32; NUM_STIGMERGIC_GRANULES];
        let mut pos_words = [0u64; NUM_STIGMERGIC_GRANULES];
        let mut neg_words = [0u64; NUM_STIGMERGIC_GRANULES];
        let mut slot_votes = [0.0_f32; MAX_ATTRACTOR_DOMAINS];

        let mut active_frag_count = 0u8;
        let mut best_e = 0.0_f32;
        let mut seed_frame = fragments[0].synthesize_gateway_frame();
        let mut prov_bits = 0u32;
        let mut rep_sev = 0.0_f32;
        let mut action_flags = 0u8;

        let overlap_has_info = (overlap_report.overlap_core_mask
            | overlap_report.overlap_fringe_mask
            | overlap_report.lacuna_exterior_mask)
            != 0;
        let overlap_core_boost = 1.25_f32
            + 0.25_f32 * branchless_clamp_f32(overlap_report.overlap_core_energy * 0.25, 0.0, 1.0);

        let mut idx = 0usize;
        while idx < fragments.len() {
            let frag = &fragments[idx];
            if !frag.has_active_gateway_ensemble_signal() {
                idx += 1;
                continue;
            }
            let (frag_mask, frag_gain_q7, frag_shift_q7) =
                frag.resolve_dynamic_core_aperture(stack);
            let (f, fw) = frag.synthesize_gateway_ensemble_item();
            let raw_e = branchless_flush_subnormal_f32(branchless_relu_f32(f.effective_energy));
            let f_e = raw_e * ((f32::from(frag_gain_q7) + 32.0) * (1.0 / 128.0));
            let core_overlap_cnt =
                (frag_mask & overlap_report.overlap_core_mask).count_ones() as f32;
            let pen_overlap_cnt =
                (frag_mask & overlap_report.overlap_fringe_mask).count_ones() as f32;
            let overlap_frag_mult = if overlap_has_info {
                0.50_f32 + 0.20_f32 * core_overlap_cnt + 0.05_f32 * pen_overlap_cnt
            } else {
                1.0_f32
            };
            let overlap_weighted_e = f_e * overlap_frag_mult;

            if f_e > DEFAULT_ENTROPY_EPS || f.lineage_flags != 0 {
                active_frag_count = active_frag_count.saturating_add(1);
                prov_bits |= f.lineage_flags;
                rep_sev = branchless_max_f32(rep_sev, f.repulsion_severity);
                action_flags |= f.action_mode_flags;
                let s_idx = usize::from(f.dominant_slot & ATTRACTOR_SLOT_ID_MASK);
                slot_votes[s_idx] += overlap_weighted_e;
                if overlap_weighted_e > best_e {
                    best_e = overlap_weighted_e;
                    seed_frame = f;
                }

                let mut g = 0usize;
                while g < NUM_STIGMERGIC_GRANULES {
                    let bit = 1u8 << g;
                    let shaped = shape_granule_word_by_aperture(fw[g], frag_gain_q7, frag_shift_q7);
                    let in_window =
                        ((frag_mask & bit) != 0) & (f_e > DEFAULT_ENTROPY_EPS) & (shaped != 0);
                    if in_window {
                        let overlap_g_weight = if !overlap_has_info {
                            1.0_f32
                        } else if (overlap_report.overlap_core_mask & bit) != 0 {
                            overlap_core_boost
                        } else if (overlap_report.overlap_fringe_mask & bit) != 0 {
                            0.75_f32
                        } else if (overlap_report.discovery_excursion_mask & bit) != 0 {
                            0.90_f32
                        } else {
                            0.35_f32
                        };
                        let g_e = f_e * overlap_g_weight;
                        let mean_g = swar_q7_mean_i8_from_words_64(&[shaped; 8]);
                        if mean_g >= 0 {
                            accumulate_fragment_granule_vote(
                                &mut pos_words[g],
                                &mut pos_energy[g],
                                &mut pos_votes[g],
                                shaped,
                                g_e,
                            );
                        } else {
                            accumulate_fragment_granule_vote(
                                &mut neg_words[g],
                                &mut neg_energy[g],
                                &mut neg_votes[g],
                                shaped,
                                g_e,
                            );
                        }
                    }
                    g += 1;
                }
            }
            idx += 1;
        }

        let ego_active = stack.ego.is_active();
        let mut out_words = [0u64; NUM_STIGMERGIC_GRANULES];
        let mut quorum_agreed_mask = 0u8;
        let mut split_contested_mask = 0u8;
        let mut ego_tiebreak_mask = 0u8;
        let mut consensus_energy = 0.0_f32;
        let mut active_granule_mask = 0u8;

        let mut g = 0usize;
        while g < NUM_STIGMERGIC_GRANULES {
            let bit = 1u8 << g;
            let pv = pos_votes[g];
            let nv = neg_votes[g];
            let has_any = (pv | nv) != 0;
            active_granule_mask |= branchless_select_u8(has_any, bit, 0);

            let maj_votes = pv.max(nv);
            let min_votes = pv.min(nv);
            let agreed = (maj_votes >= 2) & (maj_votes > min_votes);
            let contested = (pv > 0) & (nv > 0) & (pv == nv);
            quorum_agreed_mask |= branchless_select_u8(agreed, bit, 0);
            split_contested_mask |= branchless_select_u8(contested, bit, 0);

            let ego_can_break = ego_active & (stack.ego.granule_autonomy_q7[g] >= 16);
            let ego_tie = contested & ego_can_break;
            ego_tiebreak_mask |= branchless_select_u8(ego_tie, bit, 0);

            let ego_pref_pos = stack.ego.sovereign_granule_phase_q7[g] >= 0;
            let e_pref_pos = pos_energy[g] >= neg_energy[g];
            let choose_pos = branchless_select_u8(
                contested,
                branchless_select_u8(ego_can_break, u8::from(ego_pref_pos), u8::from(e_pref_pos)),
                u8::from((pv > nv) | ((pv == nv) & e_pref_pos)),
            ) != 0;

            let chosen_w = branchless_select_u64(choose_pos, pos_words[g], neg_words[g]);
            let ego_word = (stack.ego.sovereign_granule_phase_q7[g] as u8 as u64)
                .wrapping_mul(SWAR_SPLAT_BYTE_64);
            let damped_contested = swar_q7_scale_le128_with_bias_x8(
                chosen_w,
                64,
                64u64.wrapping_mul(SWAR_SPLAT_BYTE_64),
            );
            let ego_resolved = swar_q7_avg_round_x8(chosen_w, ego_word);

            let gw = branchless_select_u64(
                !has_any,
                0,
                branchless_select_u64(
                    ego_tie,
                    ego_resolved,
                    branchless_select_u64(contested, damped_contested, chosen_w),
                ),
            );
            out_words[g] = gw;

            let win_e = branchless_select_f32(choose_pos, pos_energy[g], neg_energy[g]);
            let ag_mult = 1.0 + 0.12 * f32::from(maj_votes.saturating_sub(1).min(4));
            let g_contrib = branchless_select_f32(
                agreed,
                win_e * ag_mult,
                branchless_select_f32(contested & !ego_can_break, win_e * 0.5, win_e),
            );
            consensus_energy += g_contrib * (1.0 / (NUM_STIGMERGIC_GRANULES as f32));
            g += 1;
        }

        let mut dom_slot = seed_frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK;
        let mut best_slot_score = -1.0_f32;
        let mut s = 0usize;
        while s < MAX_ATTRACTOR_DOMAINS {
            let mut sc = slot_votes[s];
            if ego_active {
                sc += f32::from(stack.ego.slot_affinity_bias_q7[s].max(0)) * (1.0 / 64.0);
            }
            if sc > best_slot_score {
                best_slot_score = sc;
                dom_slot = s as u8;
            }
            s += 1;
        }

        let agreed_cnt = quorum_agreed_mask.count_ones() as f32;
        let diversity_gain = if active_frag_count > 1 {
            1.0 + 0.08 * agreed_cnt
        } else {
            1.0
        };

        let mut out_frame = seed_frame;
        out_frame.effective_energy =
            branchless_min_f32(consensus_energy * diversity_gain, MAX_ATTRACTOR_PULL_FORCE);
        out_frame.lineage_flags = prov_bits;
        out_frame.repulsion_severity = rep_sev;
        out_frame.dominant_slot = dom_slot;
        out_frame.action_mode_flags = branchless_select_u8(
            action_flags != 0,
            action_flags & TAG_ACTION_MODE_MASK,
            out_frame.action_mode_flags,
        );
        out_frame.active_swarm_cell_mask = active_granule_mask;
        *out_frame.phase_words_mut() = out_words;
        let tag_q7 = swar_q7_mean_i8_from_words_64(out_frame.phase_words_ref());
        let ptr_tag = pack_action_slot_tag(dom_slot, out_frame.action_mode_flags);
        out_frame.mean_phase_tag_q7 = tag_q7;
        out_frame.frame_flags = ptr_tag as u16;
        out_frame.tagged_ptr =
            TaggedPtr::pack(out_frame.mapped_section_offset48, ptr_tag, tag_q7).raw();

        stack.intervene_ego_and_modulate_frame_branchless(
            delta,
            horizon,
            overlap_report,
            &mut out_frame,
        );
        let final_words = *out_frame.phase_words_ref();
        let report = StigmergicFragmentEnsembleReport {
            ensemble_consensus_energy: out_frame.effective_energy,
            ensemble_diversity_gain: diversity_gain,
            quorum_agreed_granule_mask: quorum_agreed_mask,
            split_contested_granule_mask: split_contested_mask,
            ego_tiebreak_granule_mask: ego_tiebreak_mask,
            active_fragment_count: active_frag_count,
            dominant_fragment_slot: out_frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK,
            reserved: [0; 3],
        };
        (out_frame, final_words, report)
    }
}

#[inline(always)]
fn accumulate_fragment_granule_vote(
    acc_word: &mut u64,
    acc_energy: &mut f32,
    acc_votes: &mut u8,
    shaped: u64,
    f_e: f32,
) {
    let new_e = *acc_energy + f_e;
    if *acc_votes == 0 {
        *acc_word = shaped;
    } else {
        let w_ratio = branchless_clamp_f32(
            f_e * fast_inv_pos_normal_f32(branchless_max_f32(new_e, DEFAULT_ENTROPY_EPS)),
            0.0,
            1.0,
        );
        let s_u64 = ((w_ratio * 128.0 + 0.5) as u8).min(128) as u64;
        let inv_s = 128u64.wrapping_sub(s_u64);
        *acc_word = swar_q7_circular_add_x8(
            swar_q7_scale_le128_with_bias_x8(
                *acc_word,
                inv_s,
                s_u64.wrapping_mul(SWAR_SPLAT_BYTE_64),
            ),
            swar_q7_scale_le128_with_bias_x8(shaped, s_u64, inv_s.wrapping_mul(SWAR_SPLAT_BYTE_64)),
        );
    }
    *acc_energy = new_e;
    *acc_votes = acc_votes.saturating_add(1);
}

#[inline(always)]
#[must_use]
pub(super) fn shape_granule_word_by_aperture(word: u64, gain_q7: u8, shift_q7: i8) -> u64 {
    let shift_word = (shift_q7 as u8 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let shifted = swar_q7_circular_add_x8(word, shift_word);
    let scale128 = (u64::from(gain_q7) + 32).min(128);
    let bias = (128u64.wrapping_sub(scale128)).wrapping_mul(SWAR_SPLAT_BYTE_64);
    swar_q7_scale_le128_with_bias_x8(shifted, scale128, bias)
}

#[inline(always)]
fn apply_dynamic_aperture_to_frame_and_words(
    frame: &mut GatewayFrame,
    words: &mut [u64; NUM_STIGMERGIC_GRANULES],
    dyn_mask: u8,
    dyn_gain_q7: u8,
    dyn_shift_q7: i8,
) {
    let mut g = 0usize;
    while g < NUM_STIGMERGIC_GRANULES {
        let open = ((dyn_mask >> g) & 1) != 0;
        let shaped = shape_granule_word_by_aperture(words[g], dyn_gain_q7, dyn_shift_q7);
        words[g] = branchless_select_u64(open, shaped, 0);
        g += 1;
    }
    let open_frac =
        f32::from(dyn_mask.count_ones() as u8) * (1.0 / (NUM_STIGMERGIC_GRANULES as f32));
    let gain_scale = (f32::from(dyn_gain_q7) + 32.0) * (1.0 / 128.0);
    frame.effective_energy = branchless_min_f32(
        branchless_relu_f32(frame.effective_energy) * open_frac * gain_scale,
        MAX_ATTRACTOR_PULL_FORCE,
    );
    frame.active_swarm_cell_mask &= dyn_mask;
    *frame.phase_words_mut() = *words;
    let tag_q7 = swar_q7_mean_i8_from_words_64(frame.phase_words_ref());
    let ptr_tag = pack_action_slot_tag(
        frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK,
        frame.action_mode_flags,
    );
    frame.mean_phase_tag_q7 = tag_q7;
    frame.frame_flags = ptr_tag as u16;
    frame.tagged_ptr = TaggedPtr::pack(frame.mapped_section_offset48, ptr_tag, tag_q7).raw();
}

/// Folds an arbitrary-length slice of [`ManifoldGatewaySignal`] items into a single
/// [`GatewayFrame`] and register-resident SWAR phase words, optionally weighting each item
/// by its $SO(2)^{64}$ phase coherence against `query_words_opt`.
#[inline(always)]
pub(super) fn mask_words_for_active_heads_64(
    words: &[u64; NUM_STIGMERGIC_GRANULES],
    active_heads: u16,
) -> [u64; NUM_STIGMERGIC_GRANULES] {
    if active_heads >= MAX_ATTENTION_HEADS as u16 {
        return *words;
    }
    let h = active_heads as usize;
    let mut out = [0u64; NUM_STIGMERGIC_GRANULES];
    let mut w_idx = 0usize;
    while w_idx < NUM_STIGMERGIC_GRANULES {
        let base = w_idx * 8;
        if base + 8 <= h {
            out[w_idx] = words[w_idx];
        } else if base < h {
            let rem = (h - base) as u32;
            let mask = (1u64 << (rem * 8)) - 1;
            out[w_idx] = words[w_idx] & mask;
        }
        w_idx += 1;
    }
    out
}

/// Evaluates $SO(2)^H$ circular phase coherence restricted to the `active_heads` ($H \in [1, 64]$)
/// active attention heads, algebraically subtracting the $+1.0$ zero-padding bias contributed by
/// the $(64 - H)$ inactive SIMD/SWAR lanes.
#[inline(always)]
pub(super) fn coherence_for_active_heads_64(
    a: &[u64; NUM_STIGMERGIC_GRANULES],
    a_uni: bool,
    b: &[u64; NUM_STIGMERGIC_GRANULES],
    b_uni: bool,
    active_heads: u16,
) -> f32 {
    let h = active_heads.clamp(1, MAX_ATTENTION_HEADS as u16);
    if h >= MAX_ATTENTION_HEADS as u16 || (a_uni && b_uni) {
        return coherence_from_word_arrays_with_both_uniform_flags_64(a, a_uni, b, b_uni);
    }
    let a_masked = mask_words_for_active_heads_64(a, h);
    let b_masked = mask_words_for_active_heads_64(b, h);
    let c64 =
        coherence_from_word_arrays_with_both_uniform_flags_64(&a_masked, false, &b_masked, false);
    let h_f32 = f32::from(h);
    let pad_f32 = (MAX_ATTENTION_HEADS as f32) - h_f32;
    branchless_clamp_f32(
        ((MAX_ATTENTION_HEADS as f32) * c64 - pad_f32) / h_f32,
        -1.0,
        1.0,
    )
}

/// Evaluates scale-invariant multi-head tangent cosine similarity in $[-1.0, 1.0]$ over the first
/// `active_heads` ($H \in [1, 64]$) lanes, eliminating both zero-padding bias and small-amplitude
/// circular cosine bias when comparing heterogeneous tangent steering vectors.
#[inline(always)]
pub(crate) fn tangent_cosine_active_heads_64(
    a: &[u64; NUM_STIGMERGIC_GRANULES],
    b: &[u64; NUM_STIGMERGIC_GRANULES],
    active_heads: u16,
) -> f32 {
    let h_usize = (active_heads.clamp(1, MAX_ATTENTION_HEADS as u16)) as usize;
    let a_bytes = words_to_q7_array_64(a);
    let b_bytes = words_to_q7_array_64(b);
    let mut dot = 0i32;
    let mut norm_a = 0i32;
    let mut norm_b = 0i32;
    let mut idx = 0usize;
    while idx < h_usize {
        let av = i32::from(a_bytes[idx]);
        let bv = i32::from(b_bytes[idx]);
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
        idx += 1;
    }
    if norm_a <= 0 || norm_b <= 0 {
        return 0.0;
    }
    let denom_sq = (norm_a as f32) * (norm_b as f32);
    let half = 0.5_f32 * denom_sq;
    let y0 = f32::from_bits(0x5f37_5a86_u32.wrapping_sub(denom_sq.to_bits() >> 1));
    let y1 = y0 * (1.5_f32 - half * y0 * y0);
    let inv_denom = y1 * (1.5_f32 - half * y1 * y1);
    branchless_clamp_f32((dot as f32) * inv_denom, -1.0, 1.0)
}

/// Linearly interpolates two 64-lane signed Q7 tangent arrays in $[-\theta_{\mathrm{clamp}}, +\theta_{\mathrm{clamp}}]^{64}$
/// (`[-127, +127]`) without modular $\pm 128$ wrapping overflow when $|b_h - a_h| > 127$.
#[inline(always)]
pub(super) fn swar_tangent_lerp_words_with_uniform_flag_64(
    a: &[u64; NUM_STIGMERGIC_GRANULES],
    a_uni: bool,
    b: &[u64; NUM_STIGMERGIC_GRANULES],
    b_uni: bool,
    s_u64: u64,
) -> ([u64; NUM_STIGMERGIC_GRANULES], bool) {
    let s = if s_u64 > 128 { 128 } else { s_u64 };
    if s == 0 {
        return (*a, a_uni);
    }
    if s >= 128 {
        return (*b, b_uni);
    }
    let inv_s = 128u64.wrapping_sub(s);
    let bias_a = s.wrapping_mul(SWAR_SPLAT_BYTE_64);
    let bias_b = inv_s.wrapping_mul(SWAR_SPLAT_BYTE_64);
    if a_uni && b_uni {
        let a0 = swar_q7_scale_le128_with_bias_x8(a[0], inv_s, bias_a);
        let b0 = swar_q7_scale_le128_with_bias_x8(b[0], s, bias_b);
        let w0 = swar_q7_circular_add_x8(a0, b0);
        return ([w0; NUM_STIGMERGIC_GRANULES], true);
    }
    let mut out = [0u64; NUM_STIGMERGIC_GRANULES];
    let mut k = 0usize;
    while k < NUM_STIGMERGIC_GRANULES {
        let ak = swar_q7_scale_le128_with_bias_x8(a[k], inv_s, bias_a);
        let bk = swar_q7_scale_le128_with_bias_x8(b[k], s, bias_b);
        out[k] = swar_q7_circular_add_x8(ak, bk);
        k += 1;
    }
    let is_uni = is_word_array_uniform_64(&out);
    (out, is_uni)
}

#[inline(always)]
pub(super) fn fold_gateway_slice_with_query_words_core<S: ManifoldGatewaySignal>(
    slice: &[S],
    query_words_opt: Option<&[u64; NUM_STIGMERGIC_GRANULES]>,
) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
    fold_gateway_slice_with_query_words_and_heads_core(
        slice,
        query_words_opt,
        MAX_ATTENTION_HEADS as u16,
    )
}

#[inline]
pub(super) fn fold_gateway_slice_with_query_words_and_heads_core<S: ManifoldGatewaySignal>(
    slice: &[S],
    query_words_opt: Option<&[u64; NUM_STIGMERGIC_GRANULES]>,
    query_active_heads: u16,
) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
    let len = slice.len();
    if len == 0 {
        return (GatewayFrame::default(), [0u64; NUM_STIGMERGIC_GRANULES]);
    }
    let use_query_resonance = query_words_opt.is_some_and(swar_words_nonzero_64);
    if len == 1 && !use_query_resonance {
        return slice[0].synthesize_gateway_frame_with_words();
    }
    let q_is_uniform = query_words_opt.is_some_and(is_word_array_uniform_64);

    let mut acc_w = [0u64; NUM_STIGMERGIC_GRANULES];
    let mut acc_is_uniform = true;
    let mut tot_e = 0.0_f32;
    let mut best_e = 0.0_f32;
    let mut has_active = false;

    let mut storage_addr = 0u64;
    let mut target_anchor_id = 0u32;
    let mut phase_clamp_rad = DEFAULT_PHASE_CLAMP_MAX_RAD;
    let mut phase_step_min_rad = DEFAULT_PHASE_STEP_MIN_RAD;
    let mut eff_inv_temp = 1.0_f32;
    let mut fp_coh = 1.0_f32;
    let mut rep_sev = 0.0_f32;
    let mut prov_bits = 0u32;
    let mut dom_cause = ROOT_CAUSE_NOMINAL;
    let mut dom_slot = ATTRACTOR_SLOT_SYSTEM_BASELINE;
    let mut action_flags = ACTION_PRESET_EXPRESS_PHASE;
    let mut swarm_mask = 0xFFu8;
    let mut phase_step_q7 = branchless_rad_to_phase_step_u8(DEFAULT_PHASE_STEP_MIN_RAD);
    let mut subhead_fact = 24u8;
    let mut subhead_logic = 24u8;
    let mut subhead_explore = 16u8;
    let mut fact_gain_q7 = 128u8;
    let mut logic_gain_q7 = 128u8;
    let mut explore_gain_q7 = 128u8;
    let mut active_heads = query_active_heads.clamp(1, MAX_ATTENTION_HEADS as u16);

    let mut i = 0usize;
    while i < len {
        if !slice[i].has_active_gateway_ensemble_signal() {
            i += 1;
            continue;
        }
        let (f, f_w) = slice[i].synthesize_gateway_ensemble_item();
        let raw_e = branchless_flush_subnormal_f32(branchless_relu_f32(f.effective_energy));
        let f_w_uniform = is_word_array_uniform_64(&f_w);
        let item_heads = f
            .active_head_count
            .min(query_active_heads)
            .clamp(1, MAX_ATTENTION_HEADS as u16);
        let f_e = if use_query_resonance {
            let q_w = query_words_opt.unwrap_or(&f_w);
            let coh =
                coherence_for_active_heads_64(q_w, q_is_uniform, &f_w, f_w_uniform, item_heads);
            let res = branchless_max_f32(coh, 0.0);
            branchless_flush_subnormal_f32(raw_e * (res * res + 0.02))
        } else {
            raw_e
        };
        let f_has_signal = (f_e > DEFAULT_ENTROPY_EPS)
            | (f.repulsion_severity > DEFAULT_ENTROPY_EPS)
            | (f.lineage_flags != 0)
            | ((f.action_mode_flags & ACTION_MODE_LAYER_SKIP) != 0);

        if f_has_signal {
            if !has_active {
                acc_w = f_w;
                acc_is_uniform = f_w_uniform;
                tot_e = f_e;
                best_e = f_e;
                storage_addr = f.mapped_section_offset48;
                target_anchor_id = f.target_anchor_id;
                phase_clamp_rad = f.phase_clamp_rad;
                phase_step_min_rad = f.phase_step_min_rad;
                eff_inv_temp = f.effective_inv_temp;
                fp_coh = f.fact_persona_coherence;
                rep_sev = f.repulsion_severity;
                prov_bits = f.lineage_flags;
                dom_cause = f.attribution_tag;
                dom_slot = f.dominant_slot;
                action_flags = f.action_mode_flags;
                swarm_mask = f.active_swarm_cell_mask;
                phase_step_q7 = f.phase_step_min_q7;
                subhead_fact = f.subhead_fact_heads;
                subhead_logic = f.subhead_logic_heads;
                subhead_explore = f.subhead_explore_heads;
                fact_gain_q7 = f.subhead_fact_gain_q7;
                logic_gain_q7 = f.subhead_logic_gain_q7;
                explore_gain_q7 = f.subhead_explore_gain_q7;
                active_heads = item_heads;
                has_active = true;
            } else {
                if f_e > DEFAULT_ENTROPY_EPS {
                    let new_tot = tot_e + f_e;
                    let safe_tot =
                        branchless_select_f32(new_tot > DEFAULT_ENTROPY_EPS, new_tot, 1.0);
                    let w_ratio =
                        branchless_clamp_f32(f_e * fast_inv_pos_normal_f32(safe_tot), 0.0, 1.0);
                    let s_u64 = ((w_ratio * 128.0 + 0.5) as u8).min(128) as u64;
                    if s_u64 >= 128 {
                        acc_w = f_w;
                        acc_is_uniform = f_w_uniform;
                    } else if s_u64 > 0 && !swar_words_equal_64(&acc_w, &f_w) {
                        let f_is_uniform = acc_is_uniform && f_w_uniform;
                        let bias_restore =
                            (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                        let (next_acc, next_uni) = swar_slerp_words_with_uniform_flag_64(
                            &acc_w,
                            acc_is_uniform,
                            &f_w,
                            f_is_uniform,
                            s_u64,
                            bias_restore,
                        );
                        acc_w = next_acc;
                        acc_is_uniform = next_uni;
                    }
                    eff_inv_temp = branchless_lerp_f32(eff_inv_temp, f.effective_inv_temp, w_ratio);
                    fp_coh = branchless_lerp_f32(fp_coh, f.fact_persona_coherence, w_ratio);

                    let take_dom = f_e > best_e;
                    best_e = branchless_select_f32(take_dom, f_e, best_e);
                    target_anchor_id =
                        branchless_select_u32(take_dom, f.target_anchor_id, target_anchor_id);
                    storage_addr = branchless_select_u64(
                        take_dom & (f.mapped_section_offset48 != 0),
                        f.mapped_section_offset48,
                        storage_addr,
                    );
                    dom_slot = branchless_select_u8(take_dom, f.dominant_slot, dom_slot);
                    subhead_fact =
                        branchless_select_u8(take_dom, f.subhead_fact_heads, subhead_fact);
                    subhead_logic =
                        branchless_select_u8(take_dom, f.subhead_logic_heads, subhead_logic);
                    subhead_explore =
                        branchless_select_u8(take_dom, f.subhead_explore_heads, subhead_explore);
                    fact_gain_q7 =
                        branchless_select_u8(take_dom, f.subhead_fact_gain_q7, fact_gain_q7);
                    logic_gain_q7 =
                        branchless_select_u8(take_dom, f.subhead_logic_gain_q7, logic_gain_q7);
                    explore_gain_q7 =
                        branchless_select_u8(take_dom, f.subhead_explore_gain_q7, explore_gain_q7);
                    tot_e = new_tot;
                }
                action_flags |= f.action_mode_flags;
                swarm_mask &= f.active_swarm_cell_mask;
                prov_bits |= f.lineage_flags;
                rep_sev = branchless_max_f32(rep_sev, f.repulsion_severity);
                if f.attribution_tag > dom_cause {
                    dom_cause = f.attribution_tag;
                }
            }
        }
        i += 1;
    }

    if !has_active {
        return slice[0].synthesize_gateway_frame_with_words();
    }

    let phase_deltas = words_to_q7_array_64(&acc_w);
    let tag_q7 = swar_q7_mean_i8_from_words_with_uniform_flag_64(&acc_w, acc_is_uniform);
    let packed_tag = pack_action_slot_tag(dom_slot, action_flags);
    let tagged_ptr = TaggedPtr::pack(storage_addr, packed_tag, tag_q7).raw();
    let out_energy = if use_query_resonance {
        branchless_min_f32(best_e, MAX_ATTRACTOR_PULL_FORCE)
    } else {
        branchless_min_f32(tot_e, MAX_ATTRACTOR_PULL_FORCE)
    };
    (
        GatewayFrame {
            tagged_ptr,
            mapped_section_offset48: storage_addr,
            target_anchor_id,
            effective_energy: out_energy,
            phase_clamp_rad,
            phase_step_min_rad,
            effective_inv_temp: eff_inv_temp,
            fact_persona_coherence: fp_coh,
            repulsion_severity: rep_sev,
            lineage_flags: prov_bits,
            attribution_tag: dom_cause,
            dominant_slot: dom_slot,
            action_mode_flags: action_flags,
            active_swarm_cell_mask: swarm_mask,
            phase_step_min_q7: phase_step_q7,
            subhead_fact_heads: subhead_fact,
            subhead_logic_heads: subhead_logic,
            subhead_explore_heads: subhead_explore,
            subhead_fact_gain_q7: fact_gain_q7,
            subhead_logic_gain_q7: logic_gain_q7,
            subhead_explore_gain_q7: explore_gain_q7,
            mean_phase_tag_q7: tag_q7,
            active_head_count: active_heads,
            frame_flags: packed_tag as u16,
            phase_deltas,
        },
        acc_w,
    )
}

impl<T: ManifoldGatewaySignal + ?Sized> ManifoldGatewaySignal for &T {
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        (**self).synthesize_gateway_frame()
    }
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        (**self).synthesize_gateway_frame_with_words()
    }
    #[inline(always)]
    fn synthesize_gateway_ensemble_item(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        (**self).synthesize_gateway_ensemble_item()
    }
    #[inline(always)]
    fn has_active_gateway_ensemble_signal(&self) -> bool {
        (**self).has_active_gateway_ensemble_signal()
    }
    #[inline]
    fn synthesize_gateway_with_core_layers(
        &self,
        stack: &mut CoreStateStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        overlap_report: &OverlapDiscoveryReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        (**self).synthesize_gateway_with_core_layers(stack, delta, horizon, overlap_report)
    }
    #[inline]
    fn synthesize_gateway_with_core_layers_readonly(
        &self,
        stack: &CoreStateStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        overlap_report: &OverlapDiscoveryReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        (**self).synthesize_gateway_with_core_layers_readonly(stack, delta, horizon, overlap_report)
    }
    #[inline]
    fn combine_pair_with_core_layers<B: ManifoldGatewaySignal + ?Sized>(
        &self,
        other: &B,
        stack: &mut CoreStateStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        overlap_report: &OverlapDiscoveryReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        (**self).combine_pair_with_core_layers(other, stack, delta, horizon, overlap_report)
    }
    #[inline]
    fn combine_pair_with_core_layers_readonly<B: ManifoldGatewaySignal + ?Sized>(
        &self,
        other: &B,
        stack: &CoreStateStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        overlap_report: &OverlapDiscoveryReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        (**self).combine_pair_with_core_layers_readonly(
            other,
            stack,
            delta,
            horizon,
            overlap_report,
        )
    }
}
