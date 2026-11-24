// name: src/adapter/src/pooling/calibrator.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-entry token specificity and empirical MaxSim hubness calibration strategies.

use crate::{AttractorRole, TextAttractorCorpus, TextAttractorEntry};

/// Default corpus entry count threshold up to which [`AdaptiveTokenSpecificityCalibrator`]
/// evaluates exact fused all-pairs token specificity and hubness statistics.
pub const EXACT_TOKEN_STATS_LIMIT: usize = 64;

/// Calibrated per-token specificity weights and per-entry leave-one-out MaxSim hubness baselines.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenCalibrationStats {
    /// Unsupervised cross-entry token specificity weights (`[entry][token]`) in `[0.10, 1.0]`.
    pub entry_token_weights: Vec<Vec<f32>>,
    /// Leave-one-out empirical MaxSim hubness baseline per corpus entry.
    pub entry_token_hub_bias: Vec<f32>,
    /// Mean leave-one-out MaxSim hubness across all corpus entries.
    pub global_token_hub_mean: f32,
}

/// Strategy trait calibrating per-token specificity weights and empirical MaxSim hubness baselines.
pub trait TokenSpecificityCalibrator: Send + Sync {
    /// Computes calibrated token specificity weights and hubness baselines for `entry_token_sigs`
    /// aligned with `entries`.
    fn calibrate_token_stats(
        &self,
        entry_token_sigs: &[Vec<Vec<f32>>],
        entries: &[TextAttractorEntry],
    ) -> TokenCalibrationStats;
}

/// Default [`TokenSpecificityCalibrator`] combining a single-pass upper-triangle exact solver
/// (bit-identical to two-pass all-pairs up to `exact_limit`, default [`EXACT_TOKEN_STATS_LIMIT`])
/// with deterministic stride-sampled representative calibration ($O(N \cdot M \cdot T^2)$) above
/// that threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveTokenSpecificityCalibrator {
    /// Maximum entry count calibrated with exact all-pairs statistics before switching to
    /// deterministic stride-sampled representatives of size `exact_limit`.
    pub exact_limit: usize,
}

impl Default for AdaptiveTokenSpecificityCalibrator {
    #[inline]
    fn default() -> Self {
        Self {
            exact_limit: EXACT_TOKEN_STATS_LIMIT,
        }
    }
}

impl AdaptiveTokenSpecificityCalibrator {
    /// Selects up to `budget` deterministic, strictly increasing representatives from `pool`
    /// using integer stride indexing (zero randomness).
    fn deterministic_stride_sample(pool: &[usize], budget: usize) -> Vec<usize> {
        let len = pool.len();
        if budget == 0 || len == 0 {
            return Vec::new();
        }
        if len <= budget {
            return pool.to_vec();
        }
        let mut reps = Vec::with_capacity(budget);
        for s in 0..budget {
            reps.push(pool[(s * len) / budget]);
        }
        reps
    }

    /// Computes unweighted directed MaxSim from `query_tokens` to `doc_tokens`.
    pub(crate) fn compute_unweighted_maxsim(
        query_tokens: &[Vec<f32>],
        doc_tokens: &[Vec<f32>],
    ) -> f32 {
        if query_tokens.is_empty() || doc_tokens.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0_f32;
        for q_tok in query_tokens {
            let max_sim = doc_tokens
                .iter()
                .map(|d_tok| TextAttractorCorpus::cosine_f32(q_tok, d_tok))
                .fold(f32::NEG_INFINITY, f32::max);
            if max_sim.is_finite() {
                sum += max_sim;
            }
        }
        sum / (query_tokens.len() as f32)
    }

    /// Fused single-pass exact calibration over upper-triangle pairs `(i, j)` with `i < j`.
    ///
    /// Every entry `k` visits partners `0..k` (as `j`) followed by `(k + 1)..N` (as `i`) in exact
    /// ascending order, producing bit-identical sums to separate specificity and hubness loops
    /// while evaluating each `(token_a, token_b)` cosine only once.
    fn calibrate_exact_single_pass(
        entry_token_sigs: &[Vec<Vec<f32>>],
        entries: &[TextAttractorEntry],
    ) -> TokenCalibrationStats {
        let n_entries = entry_token_sigs.len();
        let mut other_sim_sq: Vec<Vec<f32>> = entry_token_sigs
            .iter()
            .map(|toks| vec![0.0_f32; toks.len()])
            .collect();
        let mut hub_sums = vec![0.0_f32; n_entries];
        let mut col_max: Vec<f32> = Vec::new();

        for i in 0..n_entries {
            let toks_i = &entry_token_sigs[i];
            if toks_i.is_empty() {
                continue;
            }
            let role_i = entries.get(i).map(|e| e.role);
            for j in (i + 1)..n_entries {
                let toks_j = &entry_token_sigs[j];
                if toks_j.is_empty() {
                    continue;
                }
                let role_j = entries.get(j).map(|e| e.role);
                let same_role = role_i.zip(role_j).is_none_or(|(ri, rj)| ri == rj);

                col_max.clear();
                col_max.resize(toks_j.len(), f32::NEG_INFINITY);

                let mut sum_i_to_j = 0.0_f32;
                for (a, tok_a) in toks_i.iter().enumerate() {
                    let mut row_max = f32::NEG_INFINITY;
                    for (b, tok_b) in toks_j.iter().enumerate() {
                        let c = TextAttractorCorpus::cosine_f32(tok_a, tok_b);
                        if c > row_max {
                            row_max = c;
                        }
                        if c > col_max[b] {
                            col_max[b] = c;
                        }
                    }
                    if row_max.is_finite() {
                        sum_i_to_j += row_max;
                        if same_role {
                            let excess = (row_max.max(0.0) - 0.30).max(0.0);
                            other_sim_sq[i][a] += excess * excess;
                        }
                    }
                }
                hub_sums[j] += sum_i_to_j / (toks_i.len() as f32);

                let mut sum_j_to_i = 0.0_f32;
                for (b, &best_i) in col_max.iter().enumerate() {
                    if best_i.is_finite() {
                        sum_j_to_i += best_i;
                        if same_role {
                            let excess = (best_i.max(0.0) - 0.30).max(0.0);
                            other_sim_sq[j][b] += excess * excess;
                        }
                    }
                }
                hub_sums[i] += sum_j_to_i / (toks_j.len() as f32);
            }
        }

        let entry_token_weights: Vec<Vec<f32>> = other_sim_sq
            .into_iter()
            .map(|sq_row| {
                sq_row
                    .into_iter()
                    .map(|sq| (1.0 / (1.0 + 3.0 * sq)).clamp(0.10, 1.0))
                    .collect()
            })
            .collect();

        let mut entry_token_hub_bias = vec![0.0_f32; n_entries];
        let mut global_token_hub_mean = 0.0_f32;
        if n_entries >= 2 {
            let denom = (n_entries - 1) as f32;
            for (dst, &sum) in entry_token_hub_bias.iter_mut().zip(hub_sums.iter()) {
                *dst = sum / denom;
            }
            global_token_hub_mean = entry_token_hub_bias.iter().sum::<f32>() / (n_entries as f32);
        }

        TokenCalibrationStats {
            entry_token_weights,
            entry_token_hub_bias,
            global_token_hub_mean,
        }
    }

    /// Deterministic stride-sampled calibration for corpora exceeding `exact_limit`, bounding
    /// runtime to $O(N \cdot M \cdot T^2)$ where $M = \max(\text{exact\_limit}, 1)$.
    fn calibrate_representative_stride(
        entry_token_sigs: &[Vec<Vec<f32>>],
        entries: &[TextAttractorEntry],
        rep_budget: usize,
    ) -> TokenCalibrationStats {
        let n_entries = entry_token_sigs.len();
        let budget = rep_budget.max(1);

        // Partition entries by role (plus a 4th slot if `entries` is shorter than `entry_token_sigs`).
        let mut role_pools: [Vec<usize>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for i in 0..n_entries {
            let bucket = entries.get(i).map_or(3, |e| e.role.slot_index());
            role_pools[bucket].push(i);
        }
        let role_reps: [Vec<usize>; 4] = std::array::from_fn(|slot| {
            Self::deterministic_stride_sample(&role_pools[slot], budget)
        });

        let mut entry_token_weights: Vec<Vec<f32>> = vec![Vec::new(); n_entries];
        for (bucket, pool) in role_pools.iter().enumerate() {
            if pool.is_empty() {
                continue;
            }
            let reps = &role_reps[bucket];
            let total_peers = pool.len().saturating_sub(1);
            for &i in pool {
                let toks_i = &entry_token_sigs[i];
                let mut weights_i = Vec::with_capacity(toks_i.len());
                if toks_i.is_empty() {
                    entry_token_weights[i] = weights_i;
                    continue;
                }
                let compared_reps = reps.iter().filter(|&&j| j != i).count();
                let peer_scale = if compared_reps > 0 {
                    (total_peers as f32) / (compared_reps as f32)
                } else {
                    0.0_f32
                };
                for d_tok in toks_i {
                    let mut other_sim_sq = 0.0_f32;
                    for &j in reps {
                        if i == j {
                            continue;
                        }
                        let other_toks = &entry_token_sigs[j];
                        if other_toks.is_empty() {
                            continue;
                        }
                        let best_j = other_toks
                            .iter()
                            .map(|od| TextAttractorCorpus::cosine_f32(d_tok, od))
                            .fold(0.0_f32, f32::max);
                        let excess = (best_j - 0.30).max(0.0);
                        other_sim_sq += excess * excess;
                    }
                    let scaled_sq = other_sim_sq * peer_scale;
                    let spec = (1.0 / (1.0 + 3.0 * scaled_sq)).clamp(0.10, 1.0);
                    weights_i.push(spec);
                }
                entry_token_weights[i] = weights_i;
            }
        }

        let all_indices: Vec<usize> = (0..n_entries).collect();
        let global_reps = Self::deterministic_stride_sample(&all_indices, budget);
        let mut entry_token_hub_bias = vec![0.0_f32; n_entries];
        let mut global_token_hub_mean = 0.0_f32;
        if n_entries >= 2 && !global_reps.is_empty() {
            for j in 0..n_entries {
                let mut sum = 0.0_f32;
                let mut count = 0_usize;
                for &i in &global_reps {
                    if i == j {
                        continue;
                    }
                    sum +=
                        Self::compute_unweighted_maxsim(&entry_token_sigs[i], &entry_token_sigs[j]);
                    count += 1;
                }
                if count > 0 {
                    entry_token_hub_bias[j] = sum / (count as f32);
                }
            }
            global_token_hub_mean = entry_token_hub_bias.iter().sum::<f32>() / (n_entries as f32);
        }

        // Ensure role slot mapping stays aligned with `AttractorRole` variants.
        let _ = AttractorRole::Fact.slot_index();

        TokenCalibrationStats {
            entry_token_weights,
            entry_token_hub_bias,
            global_token_hub_mean,
        }
    }
}

impl TokenSpecificityCalibrator for AdaptiveTokenSpecificityCalibrator {
    fn calibrate_token_stats(
        &self,
        entry_token_sigs: &[Vec<Vec<f32>>],
        entries: &[TextAttractorEntry],
    ) -> TokenCalibrationStats {
        if entry_token_sigs.len() <= self.exact_limit {
            Self::calibrate_exact_single_pass(entry_token_sigs, entries)
        } else {
            Self::calibrate_representative_stride(entry_token_sigs, entries, self.exact_limit)
        }
    }
}
