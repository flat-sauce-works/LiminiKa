// name: src/adapter/src/stratum/workers.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Feature-level and perspective-hubness recall strata (`LocalFeatureWorkerStratum`,
//! `HeadAngleWorkerStratum`, and `PerspectiveHubnessStratum`).

use super::{RecallStratum, StratumContext, StratumDirective, StratumSignal};
use crate::{
    AttractorBoundaryPerspective, AttractorPerspectiveContext, AttractorRole, TextAttractorCorpus,
};

/// Applies leave-one-out top-k viewing-angle contrast across a subset of corpus `indices`
/// to eliminate `1/N` self-dilution bias across differently sized role populations.
#[allow(clippy::too_many_arguments)]
fn apply_loo_top_k_contrast_on_indices(
    cos_matrix: &[Vec<f32>],
    indices: &[usize],
    num_axes: usize,
    top_k: usize,
    gain: f32,
    multiply_weight: bool,
    weight_min: f32,
    weight_max: f32,
    signal: &mut StratumSignal,
) {
    let n_sub = indices.len();
    if n_sub <= 1 || num_axes == 0 || top_k == 0 {
        return;
    }
    let mut axis_sum = vec![0.0_f32; num_axes];
    for &i in indices {
        for (a, &val) in cos_matrix[i].iter().enumerate().take(num_axes) {
            axis_sum[a] += val;
        }
    }
    let denom = (n_sub - 1) as f32;
    let k = top_k.min(num_axes);
    for &i in indices {
        let row = &cos_matrix[i];
        let mut diffs: Vec<f32> = (0..num_axes)
            .map(|a| {
                let loo_mean = (axis_sum[a] - row[a]) / denom;
                row[a] - loo_mean
            })
            .collect();
        diffs.sort_by(|a, b| b.total_cmp(a));
        let top_contrast = diffs[..k].iter().sum::<f32>() / (k as f32);
        let pos = signal.candidate_scores[i].max(0.0);
        signal.candidate_scores[i] += gain * pos * top_contrast;
        let base_w = if multiply_weight {
            signal.stigmergic_weights[i]
        } else {
            1.0
        };
        signal.stigmergic_weights[i] =
            (base_w * (1.0 + top_contrast)).clamp(weight_min, weight_max);
    }
}

/// Stratum 0 (`n = 0`): local feature worker extracting per-layer viewing-angle contrast relative
/// to the unified corpus manifold on common-mode-projected signatures and recording stigmergic weights.
#[derive(Debug, Clone, Copy)]
pub struct LocalFeatureWorkerStratum {
    /// Weight applied to top-layer viewing-angle contrast modulated by positive global support.
    pub layer_contrast_gain: f32,
}

impl Default for LocalFeatureWorkerStratum {
    fn default() -> Self {
        Self {
            layer_contrast_gain: 0.35,
        }
    }
}

impl RecallStratum for LocalFeatureWorkerStratum {
    fn name(&self) -> &'static str {
        "local_feature_worker"
    }

    fn infer_up(&self, ctx: &StratumContext<'_>, signal: &mut StratumSignal) -> Result<(), String> {
        let n_entries = ctx.entries.len();
        let expected = ctx.layer_chunk_dim;
        let use_projected = !ctx.query_sig.is_empty()
            && ctx.query_sig.len() == ctx.raw_sig.len()
            && ctx.entry_sigs.len() == n_entries;
        let query_vec = if use_projected {
            ctx.query_sig
        } else {
            ctx.raw_sig
        };

        if expected > 0 && query_vec.len() >= expected && query_vec.len().is_multiple_of(expected) {
            let num_steered = query_vec.len() / expected;
            if num_steered >= 2 {
                let mut layer_cos: Vec<Vec<f32>> = vec![vec![0.0_f32; num_steered]; n_entries];
                for (i, entry) in ctx.entries.iter().enumerate() {
                    let entry_vec: &[f32] = if use_projected {
                        &ctx.entry_sigs[i]
                    } else {
                        &entry.residual_sig
                    };
                    if entry_vec.len() != query_vec.len() {
                        continue;
                    }
                    for (l, slot) in layer_cos[i].iter_mut().enumerate().take(num_steered) {
                        let start = l * expected;
                        let end = start + expected;
                        *slot = TextAttractorCorpus::cosine_f32(
                            &query_vec[start..end],
                            &entry_vec[start..end],
                        );
                    }
                }
                let top_l = (num_steered / 4).max(1);
                for role in [
                    AttractorRole::Fact,
                    AttractorRole::Persona,
                    AttractorRole::Guard,
                ] {
                    let idxs: Vec<usize> = ctx
                        .entries
                        .iter()
                        .enumerate()
                        .filter(|(_, e)| e.role == role)
                        .map(|(i, _)| i)
                        .collect();
                    apply_loo_top_k_contrast_on_indices(
                        &layer_cos,
                        &idxs,
                        num_steered,
                        top_l,
                        self.layer_contrast_gain,
                        false,
                        0.5,
                        1.5,
                        signal,
                    );
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
                *w *= 0.25;
            }
        } else {
            for w in &mut signal.stigmergic_weights {
                *w = (*w * directive.top_down_gain).clamp(0.25, 2.0);
            }
        }
    }
}

/// Stratum 0.5: multi-head viewing-angle worker treating each attention head (`0..num_q_heads`)
/// across steered layers as an independent perspective on common-mode-projected signatures so
/// specialized entity heads are not diluted by head-wide averaging or shared template framing.
#[derive(Debug, Clone, Copy)]
pub struct HeadAngleWorkerStratum {
    /// Gain applied to top-head viewing-angle contrast modulated by positive global support.
    pub head_contrast_gain: f32,
}

impl Default for HeadAngleWorkerStratum {
    fn default() -> Self {
        Self {
            head_contrast_gain: 0.40,
        }
    }
}

impl RecallStratum for HeadAngleWorkerStratum {
    fn name(&self) -> &'static str {
        "head_angle_worker"
    }

    fn infer_up(&self, ctx: &StratumContext<'_>, signal: &mut StratumSignal) -> Result<(), String> {
        let n_entries = ctx.entries.len();
        let expected = ctx.layer_chunk_dim;
        let n_heads = ctx.num_q_heads;
        let r_dim = ctx.rotary_dim;
        let use_projected = !ctx.query_sig.is_empty()
            && ctx.query_sig.len() == ctx.raw_sig.len()
            && ctx.entry_sigs.len() == n_entries;
        let query_vec = if use_projected {
            ctx.query_sig
        } else {
            ctx.raw_sig
        };

        if n_entries == 0
            || n_heads < 2
            || r_dim == 0
            || expected != n_heads * r_dim
            || query_vec.len() < expected
            || !query_vec.len().is_multiple_of(expected)
        {
            return Ok(());
        }
        let num_steered = query_vec.len() / expected;
        let mut head_cos: Vec<Vec<f32>> = vec![vec![0.0_f32; n_heads]; n_entries];
        for (i, entry) in ctx.entries.iter().enumerate() {
            let entry_vec: &[f32] = if use_projected {
                &ctx.entry_sigs[i]
            } else {
                &entry.residual_sig
            };
            if entry_vec.len() != query_vec.len() {
                continue;
            }
            for (h, slot) in head_cos[i].iter_mut().enumerate() {
                let mut dot = 0.0_f64;
                let mut nq = 0.0_f64;
                let mut ne = 0.0_f64;
                for l in 0..num_steered {
                    let base = l * expected + h * r_dim;
                    for d in 0..r_dim {
                        let qv = f64::from(query_vec[base + d]);
                        let ev = f64::from(entry_vec[base + d]);
                        dot += qv * ev;
                        nq += qv * qv;
                        ne += ev * ev;
                    }
                }
                let denom = (nq * ne).sqrt();
                *slot = if denom > 1e-12 {
                    (dot / denom).clamp(-1.0, 1.0) as f32
                } else {
                    0.0
                };
            }
        }

        let top_k_heads = (n_heads / 6).max(2).min(n_heads);
        for role in [
            AttractorRole::Fact,
            AttractorRole::Persona,
            AttractorRole::Guard,
        ] {
            let idxs: Vec<usize> = ctx
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.role == role)
                .map(|(i, _)| i)
                .collect();
            apply_loo_top_k_contrast_on_indices(
                &head_cos,
                &idxs,
                n_heads,
                top_k_heads,
                self.head_contrast_gain,
                true,
                0.4,
                2.0,
                signal,
            );
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

/// Stratum 1 (`n = 1`): evaluates role-scoped local neighborhood baselines and perspective-dependent
/// `Liminal` depths via an [`AttractorBoundaryPerspective`] trait object.
pub struct PerspectiveHubnessStratum<'a> {
    /// Perspective evaluator measuring `perspective_affinity` and `local_lacuna_depth`.
    pub perspective: &'a dyn AttractorBoundaryPerspective,
}

impl RecallStratum for PerspectiveHubnessStratum<'_> {
    fn name(&self) -> &'static str {
        "perspective_hubness_liminal_lacuna"
    }

    fn infer_up(&self, ctx: &StratumContext<'_>, signal: &mut StratumSignal) -> Result<(), String> {
        let n_entries = ctx.entries.len();
        let (query_mean_cosine, query_std_cosine) = if signal.candidate_scores.is_empty() {
            (0.0_f32, 0.05_f32)
        } else {
            let n = signal.candidate_scores.len() as f32;
            let mean = signal.candidate_scores.iter().sum::<f32>() / n;
            let var = signal
                .candidate_scores
                .iter()
                .map(|&s| (s - mean) * (s - mean))
                .sum::<f32>()
                / n;
            (mean, var.sqrt())
        };

        let mut neighbor_stats: Vec<(f32, f32)> = vec![(0.0, 0.05); n_entries];
        let mut role_global_means = [0.0_f32; 3];
        for role in [
            AttractorRole::Fact,
            AttractorRole::Persona,
            AttractorRole::Guard,
        ] {
            let role_indices: Vec<usize> = ctx
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.role == role)
                .map(|(i, _)| i)
                .collect();
            if role_indices.len() <= 1 {
                continue;
            }
            let neighbor_k = (role_indices.len() - 1).clamp(1, 4);
            let mut sum_k_mean = 0.0_f32;
            for &i in &role_indices {
                let mut sims: Vec<f32> = role_indices
                    .iter()
                    .copied()
                    .filter(|&j| j != i)
                    .map(|j| {
                        TextAttractorCorpus::cosine_f32(&ctx.entry_sigs[i], &ctx.entry_sigs[j])
                    })
                    .collect();
                sims.sort_by(|a, b| b.total_cmp(a));
                let top_slice = &sims[..neighbor_k.min(sims.len())];
                let k_len = top_slice.len().max(1) as f32;
                let k_mean = top_slice.iter().sum::<f32>() / k_len;
                let all_mean = sims.iter().sum::<f32>() / (sims.len().max(1) as f32);
                let all_var = sims
                    .iter()
                    .map(|&s| (s - all_mean) * (s - all_mean))
                    .sum::<f32>()
                    / (sims.len().max(1) as f32);
                neighbor_stats[i] = (k_mean, all_var.sqrt());
                sum_k_mean += k_mean;
            }
            let role_slot = match role {
                AttractorRole::Fact => 0,
                AttractorRole::Persona => 1,
                AttractorRole::Guard => 2,
            };
            role_global_means[role_slot] = sum_k_mean / (role_indices.len() as f32);
        }

        for (idx, entry) in ctx.entries.iter().enumerate() {
            let (neighbor_mean_cosine, neighbor_std_cosine) = neighbor_stats[idx];
            let role_slot = match entry.role {
                AttractorRole::Fact => 0,
                AttractorRole::Persona => 1,
                AttractorRole::Guard => 2,
            };
            let reading = self.perspective.measure_perspective(
                &AttractorPerspectiveContext {
                    entry,
                    raw_cosine: signal.candidate_scores[idx],
                    neighbor_mean_cosine,
                    neighbor_std_cosine,
                    global_neighbor_mean: role_global_means[role_slot],
                    query_mean_cosine,
                    query_std_cosine,
                },
                ctx.policy.min_coherence,
            );
            signal.candidate_scores[idx] = reading.perspective_affinity;
            signal.local_lacuna_depths[idx] = reading.local_lacuna_depth;
        }
        Ok(())
    }
}
