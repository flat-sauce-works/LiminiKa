// name: src/adapter/src/stratum/coalition.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-token late-interaction resonance arbitration and stigmergic coalition peeling strata
//! (`StigmergicCoalitionDecomposer`, `ResidualPeelingStigmergicDecomposer`, and
//! `LocalTokenResonanceStratum`).

use super::{
    RecallStratum, StratumContext, StratumDirective, StratumSignal, TokenResonanceDeposit,
    TrailEmergenceReceptor,
};
use crate::AttractorRole;
use liminika_core::abi::{lmnk_pointer_trail_t, lmnk_tagged_ptr_t};
use liminika_core::foundation::load_q7_words_64;
use liminika_core::liminal::{MAX_ATTENTION_HEADS, NUM_STIGMERGIC_GRANULES};
use liminika_core::runtime::StigmergicReceptor;
use liminika_core::stigmergy::{
    CognitiveOpKind, CognitiveStepOp, CognitiveTrajectoryLedger, SUBSTRATE_INLINE_TRAILS,
    StigmergicReconstitutionReport, StigmergicScaleGranularity, StigmergicSubstrate,
};

/// Trait abstracting multi-point stigmergic coalition decomposition over token resonance
/// deposits and 8-cell Micro-Swarm phase consensus (`StigmergicSubstrate`).
pub trait StigmergicCoalitionDecomposer: Send + Sync {
    /// Decomposes a multi-token query's residual resonance field into an ordered chain of
    /// complementary Fact attractors (`t0 -> t1 -> ...`) inside `substrate`.
    fn decompose_fact_coalition(
        &self,
        ctx: &StratumContext<'_>,
        signal: &mut StratumSignal,
        substrate: &mut StigmergicSubstrate,
        selected_indices: &[usize],
        trail_states: &[lmnk_pointer_trail_t],
        last_layer: usize,
    );
}

/// Default [`StigmergicCoalitionDecomposer`] applying symmetric Partition-of-Unity lateral
/// inhibition across token profiles and iterative residual peeling up to `max_coalition_hops`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualPeelingStigmergicDecomposer {
    /// Maximum number of Fact trails linked into a single decomposed pointer chain (`>= 2`).
    pub max_coalition_hops: usize,
}

impl Default for ResidualPeelingStigmergicDecomposer {
    fn default() -> Self {
        Self {
            max_coalition_hops: 3,
        }
    }
}

/// Stratum 0.75: multi-token late-interaction resonance stratum arbitrating global pooled
/// resonance against `local_token_scores` via statistical unimodality, 8-cell Micro-Swarm
/// phase consensus (`StigmergicReconstitutionOps`), and [`StigmergicCoalitionDecomposer`].
#[derive(Debug, Clone, Copy)]
pub struct LocalTokenResonanceStratum {
    /// Gain applied when promoting non-negative local contrast under sharper local unimodality.
    pub token_contrast_gain: f32,
    /// Stigmergic decomposer building multi-hop Fact coalitions on residual token fields.
    pub decomposer: ResidualPeelingStigmergicDecomposer,
}

impl Default for LocalTokenResonanceStratum {
    fn default() -> Self {
        Self {
            token_contrast_gain: 0.45,
            decomposer: ResidualPeelingStigmergicDecomposer::default(),
        }
    }
}

impl LocalTokenResonanceStratum {
    fn channel_sharpness_on_indices(
        scores: &[f32],
        idxs: &[usize],
        min_std: f32,
    ) -> Option<(usize, f32, f32, f32, f32, f32)> {
        if idxs.is_empty() {
            return None;
        }
        let mut indexed: Vec<(usize, f32)> = idxs.iter().map(|&i| (i, scores[i])).collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        let (top1_idx, top1) = indexed[0];
        let top2 = indexed.get(1).map(|x| x.1).unwrap_or(top1);
        let n = indexed.len() as f32;
        let mean = indexed.iter().map(|&(_, s)| s).sum::<f32>() / n;
        let var = indexed
            .iter()
            .map(|&(_, s)| (s - mean) * (s - mean))
            .sum::<f32>()
            / n;
        let std = var.sqrt().max(min_std);
        let z_peak = (top1 - mean) / std;
        let z_margin = (top1 - top2) / std;
        Some((top1_idx, top1, top2, mean, std, 0.5 * (z_peak + z_margin)))
    }

    fn channel_sharpness(scores: &[f32]) -> Option<(usize, f32, f32)> {
        let all: Vec<usize> = (0..scores.len()).collect();
        Self::channel_sharpness_on_indices(scores, &all, 0.08)
            .map(|(idx, top1, _, _, _, sharp)| (idx, top1, sharp))
    }

    fn resolve_pair_swarm(
        entries: &[crate::TextAttractorEntry],
        g_idx: usize,
        l_idx: usize,
        last_layer: usize,
    ) -> StigmergicReconstitutionReport {
        let g_q7 = entries[g_idx]
            .per_layer_q7
            .get(last_layer)
            .copied()
            .unwrap_or([0_i8; MAX_ATTENTION_HEADS]);
        let l_q7 = entries[l_idx]
            .per_layer_q7
            .get(last_layer)
            .copied()
            .unwrap_or([0_i8; MAX_ATTENTION_HEADS]);
        let g_words = load_q7_words_64(&g_q7);
        let l_words = load_q7_words_64(&l_q7);
        let mut swarm_trail = lmnk_pointer_trail_t::default();
        swarm_trail.copy_from_with_words_fast(&lmnk_pointer_trail_t::default(), &g_words);
        let (swarm_report, _) = swarm_trail.try_resolve_action_in_swarm_from_words(
            &l_words,
            liminika_core::stigmergy::SwarmResolveParams {
                action_ptr: Default::default(),
                active_cell_mask: 0xFF,
                step_gain_u7: 64,
                min_step_q7: 4,
                tolerance_q7: 24,
                quorum_threshold: 4,
                entropy_h_tilde: 0.40,
                low_entropy_skip_thresh: 0.15,
                bloom_filter: None,
                repulsion_threshold: 3,
                repulsion_scale_u7: 96,
            },
        );
        swarm_report
    }

    /// Selects up to `SUBSTRATE_INLINE_TRAILS` (16) active attractor candidates covering global,
    /// combined local, and single-token specialist peaks, then deposits uncollapsed token
    /// observations through the 8-cell Micro-Swarm into a [`StigmergicSubstrate`].
    fn deposit_token_swarm_trails(
        entries: &[crate::TextAttractorEntry],
        signal: &mut StratumSignal,
        last_layer: usize,
    ) -> (StigmergicSubstrate, Vec<usize>, Vec<lmnk_pointer_trail_t>) {
        let n_entries = entries.len();
        let mut substrate = StigmergicSubstrate::new();
        for entry in entries {
            if entry.role == AttractorRole::Guard {
                let g_q7 = entry
                    .per_layer_q7
                    .get(last_layer)
                    .copied()
                    .unwrap_or([0_i8; MAX_ATTENTION_HEADS]);
                let g_words = load_q7_words_64(&g_q7);
                substrate
                    .bloom_filter
                    .insert_repulsive_pattern_words_weighted(&g_words, 2);
            }
        }

        let mut ranked: Vec<(usize, f32)> = (0..n_entries)
            .map(|i| {
                let g = signal.candidate_scores.get(i).copied().unwrap_or(0.0);
                let l = signal.local_token_scores.get(i).copied().unwrap_or(0.0);
                let tok_peak = signal
                    .token_deposits
                    .iter()
                    .filter_map(|d| d.entry_excess.get(i).copied())
                    .fold(0.0_f32, f32::max);
                (i, g.max(l).max(tok_peak * 0.8))
            })
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked.truncate(SUBSTRATE_INLINE_TRAILS);

        let selected_indices: Vec<usize> = ranked.iter().map(|&(i, _)| i).collect();
        let mut trail_states: Vec<lmnk_pointer_trail_t> = selected_indices
            .iter()
            .map(|&entry_idx| {
                let q7 = entries[entry_idx]
                    .per_layer_q7
                    .get(last_layer)
                    .copied()
                    .unwrap_or([0_i8; MAX_ATTENTION_HEADS]);
                let words = load_q7_words_64(&q7);
                let mut trail = lmnk_pointer_trail_t::default();
                trail.copy_from_with_words_fast(&lmnk_pointer_trail_t::default(), &words);
                trail.target_anchor_id = entries[entry_idx].anchor_id;
                trail.user_data = entry_idx as u64;
                trail.stigmergic_density = signal
                    .candidate_scores
                    .get(entry_idx)
                    .copied()
                    .unwrap_or(0.0)
                    .max(0.0)
                    * 0.5;
                trail
            })
            .collect();

        if signal.token_deposits.is_empty() {
            return (substrate, selected_indices, trail_states);
        }

        let mut bloom_repulsions = vec![0_u32; selected_indices.len()];
        for dep in &signal.token_deposits {
            if dep.disc_weight <= 0.0 || dep.entry_excess.len() != n_entries {
                continue;
            }
            let max_ex = dep.entry_excess.iter().copied().fold(0.0_f32, f32::max);
            if max_ex <= 1e-6 {
                continue;
            }
            for (slot, &entry_idx) in selected_indices.iter().enumerate() {
                let ex = dep.entry_excess[entry_idx];
                if ex <= 0.0 {
                    continue;
                }
                let tok_coh = (ex / max_ex).clamp(0.0, 1.0);
                if tok_coh < 0.45 {
                    continue;
                }
                let (rep, _swarm_w) = trail_states[slot].try_resolve_action_in_swarm_from_words(
                    &dep.token_phase_words,
                    liminika_core::stigmergy::SwarmResolveParams {
                        action_ptr: lmnk_tagged_ptr_t::default(),
                        active_cell_mask: 0xFF,
                        step_gain_u7: 64,
                        min_step_q7: 4,
                        tolerance_q7: 24,
                        quorum_threshold: 4,
                        entropy_h_tilde: 0.45,
                        low_entropy_skip_thresh: 0.15,
                        bloom_filter: Some(&substrate.bloom_filter),
                        repulsion_threshold: 3,
                        repulsion_scale_u7: 96,
                    },
                );
                let consensus = (rep.resolved_cell_count as f32) / (NUM_STIGMERGIC_GRANULES as f32);
                let bloom_repulsed = rep.stigmergic_delta_density < 0.0;
                if bloom_repulsed {
                    bloom_repulsions[slot] += 1;
                }
                let weighted_coh = (tok_coh * (0.5 + 0.5 * consensus)).clamp(0.0, 1.0);
                substrate.evaporate_and_deposit_pheromone(
                    &mut trail_states[slot],
                    weighted_coh,
                    bloom_repulsed,
                );
            }
        }

        for (slot, &entry_idx) in selected_indices.iter().enumerate() {
            let density = trail_states[slot].stigmergic_density;
            if density > 0.0 {
                let factor = if bloom_repulsions[slot] > 0 {
                    0.75
                } else {
                    1.0 + 0.25 * density.min(1.5)
                };
                signal.stigmergic_weights[entry_idx] =
                    (signal.stigmergic_weights[entry_idx] * factor).clamp(0.30, 2.5);
            }
        }

        (substrate, selected_indices, trail_states)
    }
}

struct RoleCoalitionPair {
    cand_a: usize,
    cand_b: usize,
    joint_excl: f32,
    second_contrib: f32,
    pair_chamfer: f32,
    max_subsumption: f32,
    decomp_weights: Vec<f32>,
    total_decomp_w: f32,
}

/// Evaluates role-scoped exclusivity weights and finds the highest-utility 2-attractor
/// coalition over `token_deposits` for any `AttractorRole` (`Fact` or `Guard`).
fn evaluate_role_coalition_pair(
    token_deposits: &[TokenResonanceDeposit],
    d_to_q_scores: &[f32],
    role_idxs: &[usize],
) -> Option<(RoleCoalitionPair, Vec<(usize, f32)>)> {
    if role_idxs.len() < 2 || token_deposits.is_empty() {
        return None;
    }
    let n_dep = token_deposits.len();
    let mut decomp_weights = vec![0.0_f32; n_dep];
    let mut role_profiles: Vec<Vec<f32>> = Vec::with_capacity(n_dep);
    for (idx, d) in token_deposits.iter().enumerate() {
        let mut pos_vals: Vec<f32> = role_idxs
            .iter()
            .map(|&ri| d.entry_excess[ri].max(0.0))
            .collect();
        let norm: f32 = pos_vals.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let mut sorted_vals = pos_vals.clone();
        sorted_vals.sort_by(|a, b| b.total_cmp(a));
        let r_top1 = sorted_vals.first().copied().unwrap_or(0.0);
        let r_top2 = sorted_vals.get(1).copied().unwrap_or(0.0);
        let exclusivity = if r_top1 > 1e-6 {
            ((r_top1 - r_top2) / r_top1).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut redundancy_damp = 1.0_f32;
        if norm > 1e-6 {
            for v in &mut pos_vals {
                *v /= norm;
            }
            let mut max_prev_cos = 0.0_f32;
            for prev in &role_profiles {
                let c: f32 = pos_vals.iter().zip(prev.iter()).map(|(&a, &b)| a * b).sum();
                if c > max_prev_cos {
                    max_prev_cos = c;
                }
            }
            if max_prev_cos > 0.75 && exclusivity < 0.50 {
                redundancy_damp = (1.0 - max_prev_cos * max_prev_cos).max(0.50);
            }
        }
        role_profiles.push(pos_vals);
        decomp_weights[idx] =
            d.disc_weight.max(0.0) * (0.40 + 0.60 * exclusivity) * redundancy_damp;
    }
    let total_decomp_w: f32 = decomp_weights.iter().sum();
    if total_decomp_w <= 1e-8 {
        return None;
    }

    let excl_single_scores: Vec<(usize, f32)> = role_idxs
        .iter()
        .map(|&ri| {
            let s = token_deposits
                .iter()
                .zip(decomp_weights.iter())
                .map(|(d, &w)| w * d.entry_excess[ri])
                .sum::<f32>()
                / total_decomp_w;
            (ri, s)
        })
        .collect();

    // For large corpora (> 32 entries of the same role), bound pairwise coalition search to the
    // top-32 single-entry candidates (`<= 496` pairs, O(1)) while keeping `<= 32` bit-identical.
    let pair_candidates: Vec<(usize, f32)> = if excl_single_scores.len() <= 32 {
        excl_single_scores.clone()
    } else {
        let mut ranked = excl_single_scores.clone();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(32);
        ranked
    };

    let mut best_pair: Option<(usize, usize, f32, f32, f32)> = None;
    for a in 0..pair_candidates.len() {
        let (i, gain_i) = pair_candidates[a];
        let gain_i_pos = gain_i.max(0.0);
        if gain_i_pos <= 0.004 {
            continue;
        }
        for &(j, gain_j) in &pair_candidates[(a + 1)..] {
            let gain_j_pos = gain_j.max(0.0);
            if gain_j_pos <= 0.004 {
                continue;
            }
            let joint_excl = token_deposits
                .iter()
                .zip(decomp_weights.iter())
                .map(|(d, &w)| w * d.entry_excess[i].max(d.entry_excess[j]))
                .sum::<f32>()
                / total_decomp_w;
            let second_contrib = (joint_excl - gain_i_pos)
                .min(joint_excl - gain_j_pos)
                .max(0.0);
            if second_contrib <= 0.012 {
                continue;
            }
            let pair_d_to_q = 0.5 * (d_to_q_scores[i] + d_to_q_scores[j]);
            let utility = 0.70 * joint_excl + 0.25 * second_contrib + 0.20 * pair_d_to_q;
            if best_pair.is_none_or(|(_, _, _, _, best_u)| utility > best_u) {
                best_pair = Some((i, j, joint_excl, second_contrib, utility));
            }
        }
    }

    let (cand_a, cand_b, joint_excl, second_contrib, _utility) = best_pair?;
    let max_subsumption = role_idxs
        .iter()
        .map(|&k| {
            token_deposits
                .iter()
                .zip(decomp_weights.iter())
                .map(|(d, &w)| {
                    let ex_k = d.entry_excess[k];
                    let ex_pair = d.entry_excess[cand_a].max(d.entry_excess[cand_b]);
                    w * ex_k.min(ex_pair)
                })
                .sum::<f32>()
                / total_decomp_w
        })
        .fold(0.0_f32, f32::max);
    let pair_chamfer =
        0.70 * joint_excl + 0.20 * 0.5 * (d_to_q_scores[cand_a] + d_to_q_scores[cand_b]);

    Some((
        RoleCoalitionPair {
            cand_a,
            cand_b,
            joint_excl,
            second_contrib,
            pair_chamfer,
            max_subsumption,
            decomp_weights,
            total_decomp_w,
        },
        excl_single_scores,
    ))
}

impl StigmergicCoalitionDecomposer for ResidualPeelingStigmergicDecomposer {
    /// Evaluates multi-hop residual-peeling set cover over Fact trails when multiple specialized
    /// attractors jointly explain distinct query tokens that single global pooling collapses.
    fn decompose_fact_coalition(
        &self,
        ctx: &StratumContext<'_>,
        signal: &mut StratumSignal,
        substrate: &mut StigmergicSubstrate,
        selected_indices: &[usize],
        trail_states: &[lmnk_pointer_trail_t],
        last_layer: usize,
    ) {
        let n_entries = ctx.entries.len();
        if ctx.policy.top_k_facts < 2
            || self.max_coalition_hops < 2
            || signal.token_deposits.is_empty()
            || signal.q_to_d_scores.len() != n_entries
            || signal.d_to_q_scores.len() != n_entries
        {
            return;
        }
        // Foreign OOD queries whose salient tokens are overwhelmingly unexplained by corpus Facts
        // must never fabricate multi-hop Fact coalitions (`floor_score >= min_coherence * 1.15`).
        if ctx.query_seq_len > 0 && signal.telemetry.unexplained_token_ratio >= 0.72 {
            return;
        }

        let max_guard_raw = ctx
            .entries
            .iter()
            .zip(signal.candidate_scores.iter())
            .filter(|(e, _)| e.role == AttractorRole::Guard)
            .map(|(_, &s)| s)
            .fold(f32::NEG_INFINITY, f32::max);
        let max_non_guard_raw = ctx
            .entries
            .iter()
            .zip(signal.candidate_scores.iter())
            .filter(|(e, _)| e.role != AttractorRole::Guard)
            .map(|(_, &s)| s)
            .fold(f32::NEG_INFINITY, f32::max);
        if max_guard_raw.is_finite()
            && max_non_guard_raw.is_finite()
            && (max_guard_raw - max_non_guard_raw) > ctx.policy.guard_suppress_margin
        {
            return;
        }

        let fact_idxs = ctx.neighborhood_stats.role_indices(AttractorRole::Fact);
        if fact_idxs.len() < 2 {
            return;
        }

        let total_disc_w: f32 = signal
            .token_deposits
            .iter()
            .map(|d| d.disc_weight.max(0.0))
            .sum();
        if total_disc_w <= 1e-8 {
            return;
        }

        let Some((pair, excl_single_scores)) =
            evaluate_role_coalition_pair(&signal.token_deposits, &signal.d_to_q_scores, fact_idxs)
        else {
            return;
        };
        let RoleCoalitionPair {
            cand_a,
            cand_b,
            joint_excl,
            second_contrib,
            pair_chamfer,
            max_subsumption,
            decomp_weights,
            total_decomp_w,
        } = pair;

        let (g_fact_top1, g_fact_top1_score) = fact_idxs
            .iter()
            .map(|&i| (i, signal.candidate_scores[i]))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((fact_idxs[0], -1.0));
        let excl_q_to_d_for = |fact_idx: usize| -> f32 {
            excl_single_scores
                .iter()
                .find(|&&(idx, _)| idx == fact_idx)
                .map(|&(_, s)| s)
                .unwrap_or(0.0)
        };
        let g_top1_excl = excl_q_to_d_for(g_fact_top1);

        let raw_joint_q_to_d = signal
            .token_deposits
            .iter()
            .map(|d| d.disc_weight.max(0.0) * d.entry_excess[cand_a].max(d.entry_excess[cand_b]))
            .sum::<f32>()
            / total_disc_w;
        let basin_supported = if ctx.query_seq_len > 0 {
            g_fact_top1_score >= ctx.policy.min_coherence * 0.50
                || (raw_joint_q_to_d >= 0.080 && signal.telemetry.unexplained_token_ratio < 0.68)
        } else {
            g_fact_top1_score >= ctx.policy.min_coherence * 0.55 || raw_joint_q_to_d >= 0.065
        };
        let strictly_bimodal = joint_excl >= 0.040
            && second_contrib >= 0.015
            && second_contrib >= 0.25 * joint_excl
            && (joint_excl - max_subsumption) >= 0.22 * joint_excl;
        let global_chamfer_top1 =
            0.70 * g_top1_excl.max(0.0) + 0.20 * signal.d_to_q_scores[g_fact_top1];
        let outperforms_global_hub = pair_chamfer > global_chamfer_top1 + 0.020
            || signal.candidate_scores[cand_a] < ctx.policy.min_coherence * 0.75
            || signal.candidate_scores[cand_b] < ctx.policy.min_coherence * 0.75;

        if !(basin_supported && strictly_bimodal && outperforms_global_hub) {
            return;
        }

        let solo_rank_score =
            |idx: usize| -> f32 { excl_q_to_d_for(idx) + 0.35 * signal.d_to_q_scores[idx] };
        let (first_idx, second_idx) = if solo_rank_score(cand_a) >= solo_rank_score(cand_b) {
            (cand_a, cand_b)
        } else {
            (cand_b, cand_a)
        };

        let mut coalition = vec![first_idx, second_idx];
        let hop_cap = self
            .max_coalition_hops
            .min(SUBSTRATE_INLINE_TRAILS)
            .min(fact_idxs.len());
        if hop_cap >= 3 {
            // Precompute per-deposit top-2 remaining Fact excess values in O(T_q * N) so the
            // 3rd-hop specialist search stays linear in `fact_idxs.len()` instead of O(N^2 * T_q).
            let rem_top2: Vec<(usize, f32, f32)> = signal
                .token_deposits
                .iter()
                .map(|d| {
                    let mut best_idx = usize::MAX;
                    let mut best_v = 0.0_f32;
                    let mut second_v = 0.0_f32;
                    for &m in fact_idxs {
                        if m == first_idx || m == second_idx {
                            continue;
                        }
                        let v = d.entry_excess[m].max(0.0);
                        if v > best_v {
                            second_v = best_v;
                            best_v = v;
                            best_idx = m;
                        } else if v > second_v {
                            second_v = v;
                        }
                    }
                    (best_idx, best_v, second_v)
                })
                .collect();

            let mut best_third: Option<(usize, f32)> = None;
            for &k in fact_idxs {
                if k == first_idx || k == second_idx {
                    continue;
                }
                if signal.q_to_d_scores[k] <= 0.0 && signal.local_token_scores[k] <= 0.0 {
                    continue;
                }
                let mut res_gain = 0.0_f32;
                let mut has_specialist_peak = false;
                for ((d, &w), &(top_m, top_v, second_v)) in signal
                    .token_deposits
                    .iter()
                    .zip(decomp_weights.iter())
                    .zip(rem_top2.iter())
                {
                    let pair_ex = d.entry_excess[first_idx]
                        .max(d.entry_excess[second_idx])
                        .max(0.0);
                    let ex_k = d.entry_excess[k];
                    if ex_k <= pair_ex {
                        continue;
                    }
                    let other_rem_max = if top_m == k { second_v } else { top_v };
                    let rem_excl = if ex_k > 1e-6 {
                        ((ex_k - other_rem_max) / ex_k).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let delta = ex_k - pair_ex;
                    let eff_w = w + 0.0015 * rem_excl;
                    res_gain += eff_w * delta;
                    if ex_k >= 0.08 && delta >= 0.025 && rem_excl >= 0.40 {
                        has_specialist_peak = true;
                    }
                }
                res_gain /= total_decomp_w;
                if res_gain >= 0.004 && has_specialist_peak {
                    let score = res_gain
                        + 0.20 * signal.q_to_d_scores[k].max(0.0)
                        + 0.15 * signal.d_to_q_scores[k].max(0.0);
                    if best_third.is_none_or(|(_, best_s)| score > best_s) {
                        best_third = Some((k, score));
                    }
                }
            }
            if let Some((third_idx, _)) = best_third {
                coalition.push(third_idx);
            }
        }

        // Wire the decomposed coalition onto `StigmergicSubstrate` inline trails as a closed pointer
        // ring and notify the `StigmergicReceptor`.
        let make_trail_for = |entry_idx: usize| -> lmnk_pointer_trail_t {
            if let Some(pos) = selected_indices.iter().position(|&idx| idx == entry_idx) {
                trail_states[pos]
            } else {
                let q7 = ctx.entries[entry_idx]
                    .per_layer_q7
                    .get(last_layer)
                    .copied()
                    .unwrap_or([0_i8; MAX_ATTENTION_HEADS]);
                let words = load_q7_words_64(&q7);
                let mut t = lmnk_pointer_trail_t::default();
                t.copy_from_with_words_fast(&lmnk_pointer_trail_t::default(), &words);
                t.target_anchor_id = ctx.entries[entry_idx].anchor_id;
                t.user_data = entry_idx as u64;
                t
            }
        };
        let chain_len = coalition.len();
        for (m, &entry_idx) in coalition.iter().enumerate() {
            let mut t = make_trail_for(entry_idx);
            t.linked_trail_id = ((m + 1) % chain_len) as u32;
            let min_rho = (0.65 - 0.05 * (m as f32)).max(0.50);
            t.stigmergic_density = t.stigmergic_density.max(min_rho);
            substrate.trails[m] = t;
        }
        substrate.active_count = substrate.active_count.max(chain_len as u32);

        let mut chain_trail = lmnk_pointer_trail_t::default();
        chain_trail.copy_from_fast(&substrate.trails[0]);
        let mut receptor = TrailEmergenceReceptor::default();
        if let Ok((chain_report, _hops, _term_idx, _term_anchor, _mean_coh)) = substrate
            .traverse_pointer_chain_and_orchestrate(
                liminika_core::stigmergy::ChainTraverseParams {
                    start_trail_idx: 0,
                    max_hops: chain_len,
                    entropy_h_tilde: 0.58,
                },
                &mut chain_trail,
                None,
                None,
                None,
                None,
            )
        {
            let _ = receptor.on_stigmergic_emergence(
                &signal.gateway_frame,
                &chain_report,
                &chain_trail,
            );
        }

        let first_score = (g_fact_top1_score + 0.035).max(ctx.policy.min_coherence * 1.15);
        signal.linked_fact_indices.clear();
        for (m, &entry_idx) in coalition.iter().enumerate() {
            let floor_score = (first_score - 0.015 * (m as f32))
                .max(ctx.policy.min_coherence * if m == 0 { 1.15 } else { 0.95 });
            signal.candidate_scores[entry_idx] =
                signal.candidate_scores[entry_idx].max(floor_score);
            let boost = (1.35 - 0.05 * (m as f32)).max(1.15);
            signal.stigmergic_weights[entry_idx] =
                (signal.stigmergic_weights[entry_idx] * boost).clamp(0.4, 2.5);
            signal
                .linked_fact_indices
                .push((entry_idx, signal.candidate_scores[entry_idx]));
        }
        signal.decomposition_triggered = true;
        signal.local_channel_preferred = true;
        signal.swarm_report = LocalTokenResonanceStratum::resolve_pair_swarm(
            ctx.entries,
            coalition[0],
            coalition[1],
            last_layer,
        );
    }
}

impl RecallStratum for LocalTokenResonanceStratum {
    fn name(&self) -> &'static str {
        "local_token_resonance"
    }

    fn infer_up(&self, ctx: &StratumContext<'_>, signal: &mut StratumSignal) -> Result<(), String> {
        let n_entries = ctx.entries.len();
        if n_entries == 0 || signal.local_token_scores.len() != n_entries {
            return Ok(());
        }

        let Some((g_idx, _g_score, g_sharp)) = Self::channel_sharpness(&signal.candidate_scores)
        else {
            return Ok(());
        };
        let Some((l_idx, l_score, l_sharp)) = Self::channel_sharpness(&signal.local_token_scores)
        else {
            return Ok(());
        };

        let last_layer = ctx
            .entries
            .first()
            .map(|e| e.per_layer_q7.len().saturating_sub(1))
            .unwrap_or(0);
        // Deposit uncollapsed per-token observations onto 8-cell Micro-Swarm trails in StigmergicSubstrate.
        let (mut substrate, selected_indices, trail_states) =
            Self::deposit_token_swarm_trails(ctx.entries, signal, last_layer);

        let mut ledger = if signal
            .telemetry
            .cognitive_trajectory
            .latest_step()
            .is_some()
        {
            signal.telemetry.cognitive_trajectory
        } else {
            CognitiveTrajectoryLedger::from_ledger_slot(&substrate.ledger_slot)
        };
        if !signal.token_deposits.is_empty() {
            ledger.push_step(CognitiveStepOp::pack(
                StigmergicScaleGranularity::PointerBit,
                CognitiveOpKind::Observe,
                (signal.token_deposits.len().min(7)) as u8,
            ));
        }

        // Run core 8-cell Micro-Swarm consensus between the global and local winners' 64-head Q7 profiles.
        signal.swarm_report = Self::resolve_pair_swarm(ctx.entries, g_idx, l_idx, last_layer);
        let g_words = load_q7_words_64(
            &ctx.entries[g_idx]
                .per_layer_q7
                .get(last_layer)
                .copied()
                .unwrap_or([0_i8; MAX_ATTENTION_HEADS]),
        );
        let l_words = load_q7_words_64(
            &ctx.entries[l_idx]
                .per_layer_q7
                .get(last_layer)
                .copied()
                .unwrap_or([0_i8; MAX_ATTENTION_HEADS]),
        );
        let cell_activation = ledger.evaluate_circuit_activation(
            &g_words,
            &l_words,
            StigmergicScaleGranularity::CellTrail,
        );
        ledger.push_step(CognitiveStepOp::pack(
            StigmergicScaleGranularity::CellTrail,
            if g_idx != l_idx {
                CognitiveOpKind::BridgeShift
            } else {
                CognitiveOpKind::RouteSubHead
            },
            signal.swarm_report.resolved_cell_count.min(7),
        ));
        ledger.write_to_ledger_slot(&mut substrate.ledger_slot);
        signal.telemetry.layer_activation = cell_activation;

        let global_local_sharper = g_idx != l_idx && l_score > 0.0 && l_sharp > g_sharp * 1.15;
        signal.local_channel_preferred = global_local_sharper;

        if g_idx == l_idx && l_score > 0.0 {
            signal.stigmergic_weights[g_idx] = (signal.stigmergic_weights[g_idx]
                * (1.0 + 0.35 * l_score.min(1.0)))
            .clamp(0.35, 2.5);
        }

        // First test whether Fact tokens form a multi-basin compound coalition better solved via
        // StigmergicSubstrate linked trails than unimodal winner-take-all sharpening.
        self.decomposer.decompose_fact_coalition(
            ctx,
            signal,
            &mut substrate,
            &selected_indices,
            &trail_states,
            last_layer,
        );
        if signal.decomposition_triggered {
            ledger.push_step(CognitiveStepOp::pack(
                StigmergicScaleGranularity::MacroCoalition,
                CognitiveOpKind::DecomposePeel,
                (signal.linked_fact_indices.len().min(7)) as u8,
            ));
            ledger.write_to_ledger_slot(&mut substrate.ledger_slot);
        }
        signal.telemetry.cognitive_trajectory = ledger;

        // Arbitrate per causal role subspace (Fact / Persona / Guard) so a strong Persona style
        // hub cannot mask Fact-level disagreement between global pooling and token-level MaxSim.
        for role in [
            AttractorRole::Fact,
            AttractorRole::Persona,
            AttractorRole::Guard,
        ] {
            if role == AttractorRole::Fact && signal.decomposition_triggered {
                continue;
            }
            let idxs = ctx.neighborhood_stats.role_indices(role);
            if idxs.len() <= 1 {
                continue;
            }
            let Some((g_r_idx, g_r_top1, _g_r_top2, g_r_mean, g_r_std, g_r_sharp)) =
                Self::channel_sharpness_on_indices(&signal.candidate_scores, idxs, 0.05)
            else {
                continue;
            };
            let Some((l_r_idx, l_r_top1, l_r_top2, l_r_mean, l_r_std, l_r_sharp)) =
                Self::channel_sharpness_on_indices(&signal.local_token_scores, idxs, 0.035)
            else {
                continue;
            };

            let energy_floor = ctx.policy.min_coherence * 0.50;
            if g_r_top1 < energy_floor && l_r_top1 < energy_floor {
                continue;
            }

            let l_r_margin = (l_r_top1 - l_r_top2).max(0.0);
            let agree_floor = ctx.policy.min_coherence * 0.65;
            let role_local_sharper =
                (global_local_sharper && g_r_idx != l_r_idx && l_r_top1 > 0.04)
                    || (l_r_margin >= 0.025
                        && l_r_sharp > g_r_sharp * 1.15
                        && ((g_r_idx != l_r_idx && l_r_top1 >= energy_floor)
                            || (g_r_idx == l_r_idx
                                && g_r_top1 >= agree_floor
                                && l_r_top1 >= agree_floor)));

            if g_r_idx == l_r_idx && l_r_top1 > 0.0 {
                signal.stigmergic_weights[g_r_idx] = (signal.stigmergic_weights[g_r_idx]
                    * (1.0 + 0.25 * l_r_top1.min(1.0)))
                .clamp(0.35, 2.5);
            }
            if !role_local_sharper {
                continue;
            }

            if role == AttractorRole::Fact && g_r_idx != l_r_idx {
                signal.local_channel_preferred = true;
                signal.swarm_report =
                    Self::resolve_pair_swarm(ctx.entries, g_r_idx, l_r_idx, last_layer);
            }

            let alpha = (self.token_contrast_gain * 1.6).clamp(0.0, 0.80);
            let basin_scale_floor = ctx.policy.min_coherence * 0.75;
            let scale = if g_r_top1 >= basin_scale_floor || l_r_top1 >= basin_scale_floor {
                g_r_std.max(0.10)
            } else {
                g_r_std.max(0.03)
            };
            for &i in idxs {
                let z_loc = ((signal.local_token_scores[i] - l_r_mean) / l_r_std).clamp(-2.0, 3.0);
                let matched_loc = g_r_mean + z_loc * scale;
                signal.candidate_scores[i] =
                    (1.0 - alpha) * signal.candidate_scores[i] + alpha * matched_loc;
                let pos_contrast = (signal.local_token_scores[i] - l_r_mean).max(0.0);
                if pos_contrast > 0.0 {
                    signal.stigmergic_weights[i] =
                        (signal.stigmergic_weights[i] * (1.0 + pos_contrast)).clamp(0.35, 2.5);
                }
            }
        }

        // Symmetric fractal Guard coalition check: when false premises split across multiple
        // ContrastiveGuard entries (`PyTorch` + `24 GB VRAM`) that jointly cover discriminative
        // token deposits better than any single Fact, elevate the Guard coalition so the upper
        // boundary stratum detects the multi-premise trap.
        if !signal.decomposition_triggered
            && !signal.token_deposits.is_empty()
            && signal.d_to_q_scores.len() == n_entries
            && signal.q_to_d_scores.len() == n_entries
        {
            let guard_idxs = ctx.neighborhood_stats.role_indices(AttractorRole::Guard);
            if let Some((g_pair, _)) = evaluate_role_coalition_pair(
                &signal.token_deposits,
                &signal.d_to_q_scores,
                guard_idxs,
            ) {
                let g_bimodal = g_pair.joint_excl >= 0.040
                    && g_pair.second_contrib >= 0.015
                    && g_pair.second_contrib >= 0.25 * g_pair.joint_excl
                    && (g_pair.joint_excl - g_pair.max_subsumption) >= 0.22 * g_pair.joint_excl;
                let best_fact_chamfer = ctx
                    .entries
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| e.role == AttractorRole::Fact)
                    .map(|(i, _)| {
                        0.70 * signal.q_to_d_scores[i].max(0.0) + 0.20 * signal.d_to_q_scores[i]
                    })
                    .fold(0.0_f32, f32::max);
                let guard_chamfer_floor = if ctx.query_seq_len > 0 { 0.045 } else { 0.0 };
                let guard_tok_support = signal.q_to_d_scores[g_pair.cand_a] > 0.0
                    && signal.q_to_d_scores[g_pair.cand_b] > 0.0
                    && (ctx.query_seq_len == 0 || signal.telemetry.unexplained_token_ratio < 0.72);
                if g_bimodal
                    && guard_tok_support
                    && g_pair.pair_chamfer >= guard_chamfer_floor
                    && g_pair.pair_chamfer > best_fact_chamfer + 0.010
                {
                    let max_fact_score = ctx
                        .entries
                        .iter()
                        .enumerate()
                        .filter(|(_, e)| e.role == AttractorRole::Fact)
                        .map(|(i, _)| signal.candidate_scores[i])
                        .fold(0.0_f32, f32::max);
                    let guard_floor = (max_fact_score + (g_pair.pair_chamfer - best_fact_chamfer))
                        .max(ctx.policy.min_coherence);
                    signal.candidate_scores[g_pair.cand_a] =
                        signal.candidate_scores[g_pair.cand_a].max(guard_floor);
                    signal.candidate_scores[g_pair.cand_b] =
                        signal.candidate_scores[g_pair.cand_b].max(guard_floor);
                }
            }
        }
        Ok(())
    }

    fn adjust_down(
        &self,
        _ctx: &StratumContext<'_>,
        directive: &mut StratumDirective,
        signal: &mut StratumSignal,
    ) {
        if directive.repulse_guard || directive.suppress_injection {
            for w in &mut signal.stigmergic_weights {
                *w *= 0.5;
            }
        }
    }
}
