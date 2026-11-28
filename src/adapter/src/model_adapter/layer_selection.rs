// name: src/adapter/src/model_adapter/layer_selection.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model-agnostic RoPE steering layer bitmask helpers and deterministic relative-`EnvProbe`
//! adaptive layer selection generalized across arbitrary depths `L in 1..=64`.

use crate::pooling::{ScriptGroup, classify_script_group};
use crate::{SequenceActivationPooler, TokenSequenceEntry};

/// Default lower fractional depth of the candidate steering band (`floor(0.35 * L)`).
pub const DEFAULT_CANDIDATE_BAND_START_RATIO: f32 = 0.35;

/// Default upper fractional depth of the candidate steering band (`L`).
pub const DEFAULT_CANDIDATE_BAND_END_RATIO: f32 = 1.0;

/// Default target fraction of model layers selected (`floor(0.50 * L)`).
pub const DEFAULT_TARGET_LAYER_RATIO: f32 = 0.50;

/// Bitmask covering valid transformer layer indices `0..num_layers.min(64)`.
#[inline]
#[must_use]
pub const fn active_layers_mask(num_layers: usize) -> u64 {
    if num_layers == 0 {
        0
    } else if num_layers >= 64 {
        u64::MAX
    } else {
        (1_u64 << num_layers) - 1
    }
}

/// Clips `mask` to valid layer indices `0..num_layers.min(64)`.
#[inline]
#[must_use]
pub const fn clip_layer_mask(mask: u64, num_layers: usize) -> u64 {
    mask & active_layers_mask(num_layers)
}

/// Proportional upper-half layer mask covering layers `L/2..L` (`0xFFFF_0000` for 32 layers,
/// `0x03FF_E000` for 26 layers).
#[inline]
#[must_use]
pub const fn top_half_layer_mask(num_layers: usize) -> u64 {
    let n = if num_layers > 64 { 64 } else { num_layers };
    let start = n / 2;
    let count = n - start;
    if count == 0 {
        0
    } else if count >= 64 {
        u64::MAX
    } else {
        ((1_u64 << count) - 1) << start
    }
}

/// Computes a layer bitmask spanning the fractional depth band
/// `[floor(start_ratio * L), ceil(end_ratio * L))` clipped to `0..L.min(64)`.
///
/// Non-finite ratios fall back safely to `[DEFAULT_CANDIDATE_BAND_START_RATIO,
/// DEFAULT_CANDIDATE_BAND_END_RATIO]`. Returns `0` when `num_layers == 0` or `start >= end`.
#[must_use]
pub fn ratio_band_layer_mask(num_layers: usize, start_ratio: f32, end_ratio: f32) -> u64 {
    let n = num_layers.min(64);
    if n == 0 {
        return 0;
    }
    let s_ratio = if start_ratio.is_finite() {
        start_ratio.clamp(0.0, 1.0)
    } else {
        DEFAULT_CANDIDATE_BAND_START_RATIO
    };
    let e_ratio = if end_ratio.is_finite() {
        end_ratio.clamp(0.0, 1.0)
    } else {
        DEFAULT_CANDIDATE_BAND_END_RATIO
    };
    if s_ratio >= e_ratio {
        return 0;
    }
    let start = ((s_ratio * (n as f32)).floor() as usize).min(n);
    let mut end = ((e_ratio * (n as f32)).ceil() as usize).clamp(start, n);
    if end == start && start < n {
        end = start + 1;
    }
    let count = end.saturating_sub(start);
    if count == 0 {
        0
    } else if count >= 64 {
        u64::MAX
    } else {
        ((1_u64 << count) - 1) << start
    }
}

/// Configuration for deterministic relative-`EnvProbe` adaptive layer selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveLayerSelectConfig {
    /// Lower fractional depth bound of candidate layers (default `0.35`).
    pub band_start_ratio: f32,
    /// Upper fractional depth bound of candidate layers (default `1.0`).
    pub band_end_ratio: f32,
    /// Target fraction of model layers selected (default `0.50`, clamped to `[min(4, L), L]`).
    pub target_layer_ratio: f32,
    /// Numerical stability floor added to background variance (default `1e-6`).
    pub variance_eps: f32,
}

impl Default for AdaptiveLayerSelectConfig {
    #[inline]
    fn default() -> Self {
        Self {
            band_start_ratio: DEFAULT_CANDIDATE_BAND_START_RATIO,
            band_end_ratio: DEFAULT_CANDIDATE_BAND_END_RATIO,
            target_layer_ratio: DEFAULT_TARGET_LAYER_RATIO,
            variance_eps: 1e-6,
        }
    }
}

fn infer_valid_rotary_dim(entries: &[TokenSequenceEntry], n_layers: usize) -> Option<usize> {
    let first = entries.first()?;
    if first.seq_len == 0 || first.per_layer_token_qs.len() != n_layers {
        return None;
    }
    let first_len = first.per_layer_token_qs.first()?.len();
    if first_len == 0 || !first_len.is_multiple_of(first.seq_len) {
        return None;
    }
    let dim = first_len / first.seq_len;
    if dim == 0 {
        return None;
    }
    Some(dim)
}

fn pool_layer_unit_vectors(
    entries: &[TokenSequenceEntry],
    layer_idx: usize,
    n_layers: usize,
    dim: usize,
) -> Vec<(ScriptGroup, Vec<f32>)> {
    let mut vecs = Vec::with_capacity(entries.len());
    for entry in entries {
        if entry.per_layer_token_qs.len() != n_layers || entry.seq_len == 0 {
            continue;
        }
        let mask = entry.token_include_mask.as_deref();
        if let Ok(mut pooled) = SequenceActivationPooler::uniform_unit_mean_masked(
            &entry.per_layer_token_qs[layer_idx],
            entry.seq_len,
            dim,
            mask,
        ) {
            let norm_sq: f64 = pooled.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            if norm_sq > 1e-12 {
                // Project onto the unit sphere S^{d-1} so spherical dispersion matches
                // `build_steered_residual_sig`'s per-layer normalization without sequence-length
                // or late-layer token-isotropy norm shrinkage bias.
                let inv = (1.0 / norm_sq.sqrt()) as f32;
                for v in &mut pooled {
                    *v *= inv;
                }
                vecs.push((classify_script_group(&entry.text), pooled));
            }
        }
    }
    vecs
}

fn compute_centroid_and_variance(vecs: &[&[f32]], dim: usize) -> (Vec<f32>, f64) {
    if vecs.is_empty() || dim == 0 {
        return (vec![0.0_f32; dim], 0.0);
    }
    let n = vecs.len() as f64;
    let mut centroid = vec![0.0_f64; dim];
    for v in vecs {
        for (dst, &src) in centroid.iter_mut().zip(v.iter()) {
            *dst += f64::from(src);
        }
    }
    for c in &mut centroid {
        *c /= n;
    }
    let mut var_sum = 0.0_f64;
    for v in vecs {
        for (&src, &mu) in v.iter().zip(centroid.iter()) {
            let d = f64::from(src) - mu;
            var_sum += d * d;
        }
    }
    let mu_f32 = centroid.into_iter().map(|c| c as f32).collect();
    (mu_f32, var_sum / n)
}

/// Computes pooled within-script variance (Law of Total Variance decomposition) whenever both
/// `AsciiDominant` and `NonAsciiDominant` groups have at least 2 vectors, preventing cross-script
/// mean separation in mixed-language prior probes from inflating `bg_var(l)` on upper layers.
fn compute_within_script_variance(vecs: &[(ScriptGroup, Vec<f32>)], dim: usize) -> f64 {
    if vecs.len() < 2 || dim == 0 {
        return 0.0;
    }
    let mut ascii_refs: Vec<&[f32]> = Vec::new();
    let mut non_ascii_refs: Vec<&[f32]> = Vec::new();
    for (group, v) in vecs {
        match group {
            ScriptGroup::AsciiDominant => ascii_refs.push(v.as_slice()),
            ScriptGroup::NonAsciiDominant => non_ascii_refs.push(v.as_slice()),
        }
    }
    if ascii_refs.len() >= 2 && non_ascii_refs.len() >= 2 {
        let (_, var_a) = compute_centroid_and_variance(&ascii_refs, dim);
        let (_, var_b) = compute_centroid_and_variance(&non_ascii_refs, dim);
        let na = ascii_refs.len() as f64;
        let nb = non_ascii_refs.len() as f64;
        (na * var_a + nb * var_b) / (na + nb)
    } else {
        let all_refs: Vec<&[f32]> = vecs.iter().map(|(_, v)| v.as_slice()).collect();
        compute_centroid_and_variance(&all_refs, dim).1
    }
}

/// Computes the flattened symmetric centered Gram matrix `K = R * R^T` (`N * N`) for one layer's
/// corpus vectors, providing a representation invariant to orthogonal rotations and head permutations
/// across independent `W_Q^{(l)}` projection matrices.
fn build_centered_gram(vecs: &[(ScriptGroup, Vec<f32>)], dim: usize) -> Vec<f64> {
    let n = vecs.len();
    if n < 2 || dim == 0 {
        return Vec::new();
    }
    let refs: Vec<&[f32]> = vecs.iter().map(|(_, v)| v.as_slice()).collect();
    let (mu, _) = compute_centroid_and_variance(&refs, dim);
    let mut gram = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in i..n {
            let mut dot = 0.0_f64;
            for ((&a, &b), &m) in refs[i].iter().zip(refs[j].iter()).zip(mu.iter()) {
                dot += f64::from(a - m) * f64::from(b - m);
            }
            gram[i * n + j] = dot;
            gram[j * n + i] = dot;
        }
    }
    gram
}

/// Linear Centered Kernel Alignment (CKA) between two adjacent layers' centered Gram matrices:
/// `<K_a, K_b>_F / (||K_a||_F * ||K_b||_F) in [0, 1]`, invariant to `W_Q` rotations/permutations.
fn centered_gram_cka(gram_a: &[f64], gram_b: &[f64]) -> f64 {
    if gram_a.is_empty() || gram_a.len() != gram_b.len() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for (&a, &b) in gram_a.iter().zip(gram_b.iter()) {
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    let denom = (na * nb).sqrt();
    if denom > 1e-12 {
        (dot / denom).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Deterministically selects the top `K = clamp(floor(target_layer_ratio * L), min(4, L), L)`
/// transformer layers inside the candidate ratio band `[floor(band_start_ratio * L),
/// ceil(band_end_ratio * L))` using relative discriminative response gain weighted by
/// rotation- and head-permutation-invariant adjacent-layer Linear CKA coherence.
///
/// For each candidate layer `l`, computes:
/// - `corpus_var(l)`: pooled within-script spherical variance of unit-normalized pooled corpus
///   pre-RoPE Q directions;
/// - `bg_var(l)`: pooled within-script spherical variance of neutral `prior_samples` (defaults to
///   `1.0` when fewer than 2 prior samples are available, e.g. `--no-base-prior`), eliminating
///   between-script mean-offset inflation on mixed-language priors;
/// - `adj_coh(l)`: mean Linear CKA alignment with immediately adjacent candidate layers
///   (`0.5 + 0.5 * clamp(mean_cka, 0.0, 1.0)`).
///
/// Scores `gain(l) = (corpus_var(l) / (bg_var(l) + eps)) * adj_coh(l)` and selects the top `K`
/// layers using `f64::total_cmp` with an upper-layer tie-breaker (`l` descending) so uniform
/// scores naturally reproduce [`top_half_layer_mask`].
///
/// Falls back safely to [`top_half_layer_mask`] whenever `num_layers == 0`, fewer
/// than 2 corpus entries are supplied, tensor shapes are invalid, or all candidate layers have
/// zero discriminative variance.
#[must_use]
pub fn select_adaptive_layer_mask(
    num_layers: usize,
    corpus_entries: &[TokenSequenceEntry],
    prior_samples: &[TokenSequenceEntry],
    config: AdaptiveLayerSelectConfig,
) -> u64 {
    let n_layers = num_layers.min(64);
    if n_layers == 0 {
        return 0;
    }
    let fallback = top_half_layer_mask(n_layers);
    if corpus_entries.len() < 2 {
        return fallback;
    }
    let Some(dim) = infer_valid_rotary_dim(corpus_entries, n_layers) else {
        return fallback;
    };

    let target_ratio = if config.target_layer_ratio.is_finite() {
        config.target_layer_ratio.clamp(0.0, 1.0)
    } else {
        DEFAULT_TARGET_LAYER_RATIO
    };
    let k = ((target_ratio * (n_layers as f32)).floor() as usize).clamp(n_layers.min(4), n_layers);

    let band_mask = ratio_band_layer_mask(n_layers, config.band_start_ratio, config.band_end_ratio);
    let candidates: Vec<usize> = (0..n_layers)
        .filter(|&l| ((band_mask >> l) & 1) != 0)
        .collect();
    if candidates.is_empty() {
        return fallback;
    }
    if candidates.len() <= k {
        return band_mask;
    }

    let eps = if config.variance_eps.is_finite() {
        f64::from(config.variance_eps.max(1e-9))
    } else {
        1e-6_f64
    };

    let mut layer_corpus_grams: Vec<Vec<f64>> = Vec::with_capacity(candidates.len());
    let mut layer_corpus_var: Vec<f64> = Vec::with_capacity(candidates.len());
    let mut layer_bg_var: Vec<f64> = Vec::with_capacity(candidates.len());
    let mut max_corpus_var = 0.0_f64;

    for &l in &candidates {
        let c_vecs = pool_layer_unit_vectors(corpus_entries, l, n_layers, dim);
        if c_vecs.len() < 2 {
            return fallback;
        }
        let c_var = compute_within_script_variance(&c_vecs, dim);
        if c_var > max_corpus_var {
            max_corpus_var = c_var;
        }
        let c_gram = build_centered_gram(&c_vecs, dim);

        let p_vecs = if prior_samples.len() >= 2 {
            pool_layer_unit_vectors(prior_samples, l, n_layers, dim)
        } else {
            Vec::new()
        };
        let bg_var = if p_vecs.len() >= 2 {
            compute_within_script_variance(&p_vecs, dim)
        } else {
            1.0_f64
        };

        layer_corpus_grams.push(c_gram);
        layer_corpus_var.push(c_var);
        layer_bg_var.push(bg_var);
    }

    if max_corpus_var <= eps {
        return fallback;
    }

    let mut scored: Vec<(usize, f64)> = Vec::with_capacity(candidates.len());
    for (pos, &layer_idx) in candidates.iter().enumerate() {
        let mut adj_sum = 0.0_f64;
        let mut adj_cnt = 0_usize;
        if pos > 0 && candidates[pos - 1] + 1 == layer_idx {
            adj_sum += centered_gram_cka(&layer_corpus_grams[pos], &layer_corpus_grams[pos - 1]);
            adj_cnt += 1;
        }
        if pos + 1 < candidates.len() && candidates[pos + 1] == layer_idx + 1 {
            adj_sum += centered_gram_cka(&layer_corpus_grams[pos], &layer_corpus_grams[pos + 1]);
            adj_cnt += 1;
        }
        let adj_coh = if adj_cnt > 0 {
            0.5 + 0.5 * (adj_sum / (adj_cnt as f64)).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let rel_gain = layer_corpus_var[pos] / (layer_bg_var[pos] + eps);
        let score = if rel_gain.is_finite() {
            rel_gain * adj_coh
        } else {
            0.0
        };
        scored.push((layer_idx, score));
    }

    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| b.0.cmp(&a.0)));
    let mut selected_mask = 0_u64;
    for &(layer_idx, _) in scored.iter().take(k) {
        selected_mask |= 1_u64 << layer_idx;
    }
    if selected_mask == 0 {
        fallback
    } else {
        selected_mask
    }
}
