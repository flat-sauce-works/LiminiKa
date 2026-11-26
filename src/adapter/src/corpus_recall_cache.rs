// name: src/adapter/src/corpus_recall_cache.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lazily populated contiguous recall cache (`CorpusRecallCache`) with deterministic
//! SWAR [`BitCut64`](liminika_core::liminal::BitCut64) + stride-sampled quasi-linear
//! neighborhood calibration above [`EXACT_CORPUS_STATS_LIMIT`].

use std::sync::OnceLock;

use crate::{
    AttractorRole, DEFAULT_NEIGHBOR_K, EXACT_CORPUS_STATS_LIMIT, EntryNeighborhoodStats,
    NeighborhoodStatsProvider, TextAttractorCorpus, TextAttractorEntry,
};
use liminika_core::liminal::BitCut64;

#[inline]
pub(crate) fn residual_sig_bitcut64(sig: &[f32]) -> BitCut64 {
    if sig.is_empty() {
        return BitCut64::default();
    }
    let mut nonzero = 0_u64;
    let mut pos = 0_u64;
    let len = sig.len();
    for b in 0..64_usize {
        let start = (b * len) / 64;
        let end = (((b + 1) * len) / 64).max(start + 1).min(len);
        let sum: f32 = sig[start..end].iter().sum();
        if sum.abs() > 1e-9 {
            let bit = 1_u64 << b;
            nonzero |= bit;
            if sum > 0.0 {
                pos |= bit;
            }
        }
    }
    BitCut64::split(nonzero, pos)
}

#[inline]
fn bitcut64_coarse_score(a: BitCut64, b: BitCut64) -> i32 {
    let agreed = (a.intersect(b).count_ones() + (a.outside & b.outside).count_ones()) as i32;
    let opposed = (a.excess_over(b).count_ones() + b.excess_over(a).count_ones()) as i32;
    agreed - opposed
}

/// Lazily populated contiguous recall cache stored inside [`TextAttractorCorpus`].
#[derive(Debug)]
pub(crate) struct CorpusRecallCache {
    /// Flat row-major common-mode-projected normalized corpus signatures (`entries.len() * sig_dim`).
    pub(crate) projected_sigs_flat: Vec<f32>,
    /// Fixed dimension of each projected signature row (`0` when empty or non-uniform).
    pub(crate) sig_dim: usize,
    /// Lower-quartile orthogonal residual ratio (`||P_perp(e)|| / ||e||`) across corpus entries,
    /// anchoring query orthogonal-residual attenuation without hardcoded magic thresholds.
    pub(crate) corpus_ortho_ref: f32,
    /// Ordered corpus entry indices partitioned by [`AttractorRole::slot_index`].
    role_indices: [Vec<usize>; 3],
    /// Precomputed role-scoped neighborhood mean/std per corpus entry.
    entry_stats: Vec<EntryNeighborhoodStats>,
    /// Corpus-wide mean of `neighbor_mean_cosine` per [`AttractorRole::slot_index`].
    role_global_means: [f32; 3],
}

impl CorpusRecallCache {
    pub(crate) fn build(entries: &[TextAttractorEntry], common_mode_basis: &[Vec<f32>]) -> Self {
        let n_entries = entries.len();
        let sig_dim = entries
            .first()
            .map_or(0, |e| e.residual_sig.len())
            .filter_nonzero_if_uniform(entries);

        let mut projected_sigs_flat = Vec::with_capacity(n_entries.saturating_mul(sig_dim));
        let mut entry_ortho_ratios: Vec<f32> = Vec::with_capacity(n_entries);
        if sig_dim > 0 {
            for entry in entries {
                let start = projected_sigs_flat.len();
                projected_sigs_flat.extend_from_slice(&entry.residual_sig);
                let ratio = TextAttractorCorpus::project_and_normalize_in_place_with_ratio(
                    &mut projected_sigs_flat[start..start + sig_dim],
                    common_mode_basis,
                );
                entry_ortho_ratios.push(ratio);
            }
        }
        let corpus_ortho_ref = if common_mode_basis.is_empty() || entry_ortho_ratios.is_empty() {
            1.0
        } else {
            entry_ortho_ratios.sort_by(f32::total_cmp);
            let q25_idx = entry_ortho_ratios.len() / 4;
            entry_ortho_ratios[q25_idx].clamp(0.15, 1.0)
        };

        let mut role_indices: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (i, entry) in entries.iter().enumerate() {
            role_indices[entry.role.slot_index()].push(i);
        }

        let mut entry_stats = vec![EntryNeighborhoodStats::default(); n_entries];
        let mut role_global_means = [0.0_f32; 3];

        for role_slot in 0..3 {
            let idxs = &role_indices[role_slot];
            let m = idxs.len();
            if m <= 1 {
                continue;
            }
            let neighbor_k = (m - 1).clamp(1, DEFAULT_NEIGHBOR_K);
            let mut sum_k_mean = 0.0_f32;
            if m <= EXACT_CORPUS_STATS_LIMIT {
                let mut pair_cos = vec![0.0_f32; m * m];
                for a in 0..m {
                    let sig_a = Self::slice_from_flat(&projected_sigs_flat, sig_dim, idxs[a]);
                    for b in (a + 1)..m {
                        let sig_b = Self::slice_from_flat(&projected_sigs_flat, sig_dim, idxs[b]);
                        let c = TextAttractorCorpus::cosine_f32(sig_a, sig_b);
                        pair_cos[a * m + b] = c;
                        pair_cos[b * m + a] = c;
                    }
                }
                let mut sims = Vec::with_capacity(m - 1);
                for a in 0..m {
                    sims.clear();
                    let row = &pair_cos[a * m..(a + 1) * m];
                    for (b, &val) in row.iter().enumerate() {
                        if b != a {
                            sims.push(val);
                        }
                    }
                    sims.sort_by(|x, y| y.total_cmp(x));
                    let k_len = neighbor_k.min(sims.len()).max(1);
                    let k_mean = sims[..k_len].iter().sum::<f32>() / (k_len as f32);
                    let all_len = sims.len().max(1) as f32;
                    let all_mean = sims.iter().sum::<f32>() / all_len;
                    let all_var = sims
                        .iter()
                        .map(|&s| (s - all_mean) * (s - all_mean))
                        .sum::<f32>()
                        / all_len;
                    entry_stats[idxs[a]] = EntryNeighborhoodStats {
                        neighbor_mean_cosine: k_mean,
                        neighbor_std_cosine: all_var.sqrt(),
                    };
                    sum_k_mean += k_mean;
                }
            } else {
                let budget = EXACT_CORPUS_STATS_LIMIT;
                let sig_cuts: Vec<BitCut64> = idxs
                    .iter()
                    .map(|&idx| {
                        residual_sig_bitcut64(Self::slice_from_flat(
                            &projected_sigs_flat,
                            sig_dim,
                            idx,
                        ))
                    })
                    .collect();
                let stride_peers: Vec<usize> = (0..budget).map(|s| (s * m) / budget).collect();
                let mut coarse: Vec<(usize, i32)> = Vec::with_capacity(m);
                let mut top_cos = Vec::with_capacity(budget);
                let mut bg_cos = Vec::with_capacity(budget);
                for a in 0..m {
                    let sig_a = Self::slice_from_flat(&projected_sigs_flat, sig_dim, idxs[a]);
                    let cut_a = sig_cuts[a];
                    coarse.clear();
                    for (b, &cut_b) in sig_cuts.iter().enumerate() {
                        if b != a {
                            coarse.push((b, bitcut64_coarse_score(cut_a, cut_b)));
                        }
                    }
                    coarse.sort_by(|x, y| y.1.cmp(&x.1).then_with(|| x.0.cmp(&y.0)));
                    top_cos.clear();
                    for &(b, _) in coarse.iter().take(budget) {
                        let sig_b = Self::slice_from_flat(&projected_sigs_flat, sig_dim, idxs[b]);
                        top_cos.push(TextAttractorCorpus::cosine_f32(sig_a, sig_b));
                    }
                    top_cos.sort_by(|x, y| y.total_cmp(x));
                    let k_len = neighbor_k.min(top_cos.len()).max(1);
                    let k_mean = top_cos[..k_len].iter().sum::<f32>() / (k_len as f32);
                    bg_cos.clear();
                    for &b in &stride_peers {
                        if b != a {
                            let sig_b =
                                Self::slice_from_flat(&projected_sigs_flat, sig_dim, idxs[b]);
                            bg_cos.push(TextAttractorCorpus::cosine_f32(sig_a, sig_b));
                        }
                    }
                    let bg_len = bg_cos.len().max(1) as f32;
                    let bg_mean = bg_cos.iter().sum::<f32>() / bg_len;
                    let bg_var = bg_cos
                        .iter()
                        .map(|&s| (s - bg_mean) * (s - bg_mean))
                        .sum::<f32>()
                        / bg_len;
                    entry_stats[idxs[a]] = EntryNeighborhoodStats {
                        neighbor_mean_cosine: k_mean,
                        neighbor_std_cosine: bg_var.sqrt(),
                    };
                    sum_k_mean += k_mean;
                }
            }
            role_global_means[role_slot] = sum_k_mean / (m as f32);
        }

        Self {
            projected_sigs_flat,
            sig_dim,
            corpus_ortho_ref,
            role_indices,
            entry_stats,
            role_global_means,
        }
    }

    #[inline]
    fn slice_from_flat(flat: &[f32], sig_dim: usize, idx: usize) -> &[f32] {
        if sig_dim == 0 {
            return &[];
        }
        let Some(start) = idx.checked_mul(sig_dim) else {
            return &[];
        };
        let Some(end) = start.checked_add(sig_dim) else {
            return &[];
        };
        flat.get(start..end).unwrap_or(&[])
    }

    #[inline]
    pub(crate) fn projected_sig(&self, idx: usize) -> &[f32] {
        Self::slice_from_flat(&self.projected_sigs_flat, self.sig_dim, idx)
    }
}

trait UniformSigDimExt {
    fn filter_nonzero_if_uniform(self, entries: &[TextAttractorEntry]) -> usize;
}

impl UniformSigDimExt for usize {
    fn filter_nonzero_if_uniform(self, entries: &[TextAttractorEntry]) -> usize {
        if self > 0 && entries.iter().all(|e| e.residual_sig.len() == self) {
            self
        } else {
            0
        }
    }
}

impl NeighborhoodStatsProvider for CorpusRecallCache {
    #[inline]
    fn entry_stats(&self, entry_idx: usize) -> EntryNeighborhoodStats {
        self.entry_stats.get(entry_idx).copied().unwrap_or_default()
    }

    #[inline]
    fn role_global_mean(&self, role: AttractorRole) -> f32 {
        self.role_global_means[role.slot_index()]
    }

    #[inline]
    fn role_indices(&self, role: AttractorRole) -> &[usize] {
        &self.role_indices[role.slot_index()]
    }
}

/// Private `OnceLock<Box<_>>` wrapper providing `Clone` (empty reset), `Default`, `Debug`, and
/// `Send + Sync` while keeping [`TextAttractorCorpus`] compact when the lazy cache is unpopulated.
pub(crate) struct LazyRecallCache(OnceLock<Box<CorpusRecallCache>>);

impl LazyRecallCache {
    pub(crate) const fn new() -> Self {
        Self(OnceLock::new())
    }

    #[inline]
    pub(crate) fn invalidate(&mut self) {
        let _ = self.0.take();
    }

    #[inline]
    pub(crate) fn get_or_init(
        &self,
        init: impl FnOnce() -> CorpusRecallCache,
    ) -> &CorpusRecallCache {
        self.0.get_or_init(|| Box::new(init())).as_ref()
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn is_populated(&self) -> bool {
        self.0.get().is_some()
    }
}

impl Default for LazyRecallCache {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for LazyRecallCache {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for LazyRecallCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyRecallCache")
            .field("populated", &self.0.get().is_some())
            .finish()
    }
}
