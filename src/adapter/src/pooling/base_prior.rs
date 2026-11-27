// name: src/adapter/src/pooling/base_prior.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Startup base-model attractor profile extraction ([`ModelBaseAttractorProfile`]) and
//! cross-entry token common-mode subspace helpers.

use super::null_calibration::{ScriptGroup, classify_script_group};
use super::{
    CompiledTokenCorpus, STREAMING_PCA_SAMPLE_BUDGET, SequenceActivationPooler, TokenSequenceEntry,
    validate_token_include_mask,
};
use crate::{ModelSteeringAdapter, TextAttractorCorpus};

/// Startup base-model attractor profile distilled from unsteered baseline generations/prompts.
///
/// Captures the host model's own background linguistic gravity (layer-wise mean direction, global
/// and token-level common-mode framing subspaces, and online cosine-clustered sub-basins) so
/// external knowledge compilation can orthogonalize against the model's default priors.
#[derive(Debug, Clone)]
pub struct ModelBaseAttractorProfile {
    /// Mean unsteered Q direction per transformer layer.
    pub novelty_reference_per_layer: Vec<Vec<f32>>,
    /// Orthonormal global framing/prior directions extracted from unsteered model sequences.
    pub pooled_common_mode_basis: Vec<Vec<f32>>,
    /// Orthonormal token-level framing/sink directions extracted from unsteered model tokens.
    pub token_common_mode_basis: Vec<Vec<f32>>,
    /// Distinct regime sub-attractors (`[cluster_k][steered_dim]`) partitioned from the base model.
    pub sub_attractors: Vec<Vec<f32>>,
    /// Number of unsteered startup sequences processed to construct this profile.
    pub sample_count: usize,
    /// Raw startup probe sequences retained transiently for non-leaking 2-fold OOD null calibration.
    pub(crate) calibration_samples: Vec<TokenSequenceEntry>,
}

impl ModelBaseAttractorProfile {
    /// Extracts a [`ModelBaseAttractorProfile`] from unsteered startup sequences via two-pass
    /// pooling, PCA common-mode fitting, and cosine-threshold online sub-basin partitioning.
    ///
    /// Tokens with `token_include_mask[t] == false` are strictly excluded from centroids,
    /// common-mode subspaces, and sub-attractor clustering.
    ///
    /// # Errors
    /// Returns an error if `samples` is empty, has zero valid unmasked sequences, or any sample's
    /// tensor shape or mask length disagrees with `adapter`.
    pub fn extract(
        adapter: &ModelSteeringAdapter,
        samples: &[TokenSequenceEntry],
        pooler: SequenceActivationPooler,
        merge_cosine_thresh: f32,
    ) -> Result<Self, String> {
        if samples.is_empty() {
            return Err("cannot extract ModelBaseAttractorProfile from empty samples".to_string());
        }
        let geom = adapter.geometry();
        let num_layers = geom.num_layers;
        let dim = geom.q_rot_len();

        let mut active_samples: Vec<&TokenSequenceEntry> = Vec::with_capacity(samples.len());
        for s in samples {
            if s.per_layer_token_qs.len() != num_layers {
                return Err(format!(
                    "sample layer count mismatch: expected {num_layers}, got {}",
                    s.per_layer_token_qs.len()
                ));
            }
            let included = validate_token_include_mask(s.seq_len, s.token_include_mask.as_deref())?;
            if included > 0 {
                active_samples.push(s);
            }
        }
        if active_samples.is_empty() {
            return Err(
                "cannot extract ModelBaseAttractorProfile: all sample tokens excluded by mask"
                    .to_string(),
            );
        }
        let n = active_samples.len() as f32;

        let sample_scripts: Vec<ScriptGroup> = active_samples
            .iter()
            .map(|s| classify_script_group(&s.text))
            .collect();
        let mut ascii_count = 0_usize;
        let mut non_ascii_count = 0_usize;
        for &g in &sample_scripts {
            match g {
                ScriptGroup::AsciiDominant => ascii_count += 1,
                ScriptGroup::NonAsciiDominant => non_ascii_count += 1,
            }
        }
        let split_script_refs = ascii_count >= 2 && non_ascii_count >= 2;

        let mut novelty_reference_per_layer = vec![vec![0.0_f32; dim]; num_layers];
        let mut ascii_ref_per_layer = if split_script_refs {
            vec![vec![0.0_f32; dim]; num_layers]
        } else {
            Vec::new()
        };
        let mut non_ascii_ref_per_layer = if split_script_refs {
            vec![vec![0.0_f32; dim]; num_layers]
        } else {
            Vec::new()
        };
        let inv_ascii = if ascii_count > 0 {
            1.0_f32 / (ascii_count as f32)
        } else {
            0.0
        };
        let inv_non_ascii = if non_ascii_count > 0 {
            1.0_f32 / (non_ascii_count as f32)
        } else {
            0.0
        };

        for (s, &group) in active_samples.iter().zip(sample_scripts.iter()) {
            let mask = s.token_include_mask.as_deref();
            for (l, flat) in s.per_layer_token_qs.iter().enumerate() {
                let u =
                    SequenceActivationPooler::uniform_unit_mean_masked(flat, s.seq_len, dim, mask)?;
                for (dst, &src) in novelty_reference_per_layer[l].iter_mut().zip(u.iter()) {
                    *dst += src / n;
                }
                if split_script_refs {
                    match group {
                        ScriptGroup::AsciiDominant => {
                            for (dst, &src) in ascii_ref_per_layer[l].iter_mut().zip(u.iter()) {
                                *dst += src * inv_ascii;
                            }
                        }
                        ScriptGroup::NonAsciiDominant => {
                            for (dst, &src) in non_ascii_ref_per_layer[l].iter_mut().zip(u.iter()) {
                                *dst += src * inv_non_ascii;
                            }
                        }
                    }
                }
            }
        }

        let grand_ref_slices: Vec<&[f32]> = novelty_reference_per_layer
            .iter()
            .map(Vec::as_slice)
            .collect();
        let ascii_ref_slices: Vec<&[f32]> = ascii_ref_per_layer.iter().map(Vec::as_slice).collect();
        let non_ascii_ref_slices: Vec<&[f32]> =
            non_ascii_ref_per_layer.iter().map(Vec::as_slice).collect();

        let mut pooled_sigs: Vec<Vec<f32>> = Vec::with_capacity(active_samples.len());
        let mut raw_tok_sigs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(active_samples.len());
        for (s, &group) in active_samples.iter().zip(sample_scripts.iter()) {
            let mask = s.token_include_mask.as_deref();
            let (layer_ref, ref_slices) = if split_script_refs {
                match group {
                    ScriptGroup::AsciiDominant => (&ascii_ref_per_layer, &ascii_ref_slices),
                    ScriptGroup::NonAsciiDominant => {
                        (&non_ascii_ref_per_layer, &non_ascii_ref_slices)
                    }
                }
            } else {
                (&novelty_reference_per_layer, &grand_ref_slices)
            };
            let pooled = pooler.pool_sequence_masked(
                &s.per_layer_token_qs,
                s.seq_len,
                dim,
                Some(layer_ref),
                mask,
            )?;
            let p_refs: Vec<&[f32]> = pooled.iter().map(Vec::as_slice).collect();
            pooled_sigs.push(TextAttractorCorpus::build_steered_residual_sig(
                geom, &p_refs, ref_slices,
            ));
            let indexed_toks = CompiledTokenCorpus::build_raw_token_sigs_for_entry(
                adapter,
                &s.per_layer_token_qs,
                s.seq_len,
                layer_ref,
                mask,
            )?;
            raw_tok_sigs.push(indexed_toks.into_iter().map(|(_, sig)| sig).collect());
        }

        // Remove residual within-script group means before pooled PCA whenever multiple scripts are
        // present so cross-script mean offsets never masquerade as global common-mode directions.
        let pca_pooled_sigs: Vec<Vec<f32>> = if split_script_refs {
            center_pooled_sigs_within_script(&pooled_sigs, &sample_scripts)
        } else {
            pooled_sigs.clone()
        };
        let pooled_sig_refs: Vec<&[f32]> = pca_pooled_sigs.iter().map(Vec::as_slice).collect();
        let pooled_common_mode_basis =
            TextAttractorCorpus::fit_common_mode_subspace_from_sigs(&pooled_sig_refs, 3);
        let token_common_mode_basis =
            fit_cross_entry_token_common_mode_with_scripts(&raw_tok_sigs, Some(&sample_scripts), 6);

        // Partition the base model's unsteered responses into coherent regime sub-attractors.
        let thresh = merge_cosine_thresh.clamp(0.10, 0.99);
        let mut sub_attractors: Vec<Vec<f32>> = Vec::new();
        let mut cluster_counts: Vec<usize> = Vec::new();
        for sig in &pooled_sigs {
            let norm: f32 = sig.iter().map(|&v| v * v).sum::<f32>().sqrt();
            if norm <= 1e-6 {
                continue;
            }
            let best = sub_attractors
                .iter()
                .enumerate()
                .map(|(idx, c)| (idx, TextAttractorCorpus::cosine_f32(sig, c)))
                .max_by(|a, b| a.1.total_cmp(&b.1));
            match best {
                Some((best_idx, sim)) if sim >= thresh => {
                    let count = cluster_counts[best_idx] as f32;
                    let lr = 1.0 / (count + 1.0);
                    let mut sum_sq = 0.0_f64;
                    for (dst, &src) in sub_attractors[best_idx].iter_mut().zip(sig.iter()) {
                        *dst = (1.0 - lr) * *dst + lr * src;
                        sum_sq += f64::from(*dst) * f64::from(*dst);
                    }
                    let inv = (1.0 / sum_sq.sqrt().max(1e-12)) as f32;
                    for v in &mut sub_attractors[best_idx] {
                        *v *= inv;
                    }
                    cluster_counts[best_idx] += 1;
                }
                _ => {
                    sub_attractors.push(sig.clone());
                    cluster_counts.push(1);
                }
            }
        }

        Ok(Self {
            novelty_reference_per_layer,
            pooled_common_mode_basis,
            token_common_mode_basis,
            sub_attractors,
            sample_count: active_samples.len(),
            calibration_samples: active_samples.into_iter().cloned().collect(),
        })
    }
}

/// Subtracts each [`ScriptGroup`]'s own mean signature from its members so cross-script mean
/// offsets never inject a rank-1 between-language spike into pooled common-mode PCA.
fn center_pooled_sigs_within_script(
    pooled_sigs: &[Vec<f32>],
    scripts: &[ScriptGroup],
) -> Vec<Vec<f32>> {
    let Some(dim) = pooled_sigs.first().map(Vec::len) else {
        return Vec::new();
    };
    if dim == 0 || pooled_sigs.len() != scripts.len() {
        return pooled_sigs.to_vec();
    }
    let mut ascii_sum = vec![0.0_f64; dim];
    let mut non_ascii_sum = vec![0.0_f64; dim];
    let mut ascii_n = 0_usize;
    let mut non_ascii_n = 0_usize;
    for (sig, &group) in pooled_sigs.iter().zip(scripts.iter()) {
        if sig.len() != dim {
            continue;
        }
        match group {
            ScriptGroup::AsciiDominant => {
                ascii_n += 1;
                for (dst, &src) in ascii_sum.iter_mut().zip(sig.iter()) {
                    *dst += f64::from(src);
                }
            }
            ScriptGroup::NonAsciiDominant => {
                non_ascii_n += 1;
                for (dst, &src) in non_ascii_sum.iter_mut().zip(sig.iter()) {
                    *dst += f64::from(src);
                }
            }
        }
    }
    let ascii_mean: Option<Vec<f32>> = (ascii_n >= 2).then(|| {
        let inv = 1.0 / (ascii_n as f64);
        ascii_sum.iter().map(|&v| (v * inv) as f32).collect()
    });
    let non_ascii_mean: Option<Vec<f32>> = (non_ascii_n >= 2).then(|| {
        let inv = 1.0 / (non_ascii_n as f64);
        non_ascii_sum.iter().map(|&v| (v * inv) as f32).collect()
    });

    pooled_sigs
        .iter()
        .zip(scripts.iter())
        .map(|(sig, &group)| {
            let mean = match group {
                ScriptGroup::AsciiDominant => ascii_mean.as_deref(),
                ScriptGroup::NonAsciiDominant => non_ascii_mean.as_deref(),
            };
            if let Some(mu) = mean
                && sig.len() == dim
            {
                sig.iter().zip(mu.iter()).map(|(&x, &m)| x - m).collect()
            } else {
                sig.clone()
            }
        })
        .collect()
}

/// Fits cross-entry token-space common-mode directions using strict-majority entry support
/// (`> 50%`, bounded by entry count). When `script_groups` contains both ASCII-dominant and
/// non-ASCII-dominant groups with at least 4 samples each, candidate directions must also carry
/// support in both groups so single-language directions are not over-extracted as global priors.
pub(super) fn fit_cross_entry_token_common_mode_with_scripts(
    per_entry_tok_sigs: &[Vec<Vec<f32>>],
    script_groups: Option<&[ScriptGroup]>,
    max_components: usize,
) -> Vec<Vec<f32>> {
    if per_entry_tok_sigs.len() < 3 || max_components == 0 {
        return Vec::new();
    }
    let sampled_indices: Vec<usize> = if per_entry_tok_sigs.len() <= STREAMING_PCA_SAMPLE_BUDGET {
        (0..per_entry_tok_sigs.len()).collect()
    } else {
        (0..STREAMING_PCA_SAMPLE_BUDGET)
            .map(|s| (s * per_entry_tok_sigs.len()) / STREAMING_PCA_SAMPLE_BUDGET)
            .collect()
    };
    let total_toks: usize = sampled_indices
        .iter()
        .map(|&i| per_entry_tok_sigs[i].len())
        .sum();
    if total_toks == 0 {
        return Vec::new();
    }
    let mut flat_refs: Vec<&[f32]> = Vec::with_capacity(total_toks);
    for &i in &sampled_indices {
        for tok in &per_entry_tok_sigs[i] {
            flat_refs.push(tok.as_slice());
        }
    }
    let candidates =
        TextAttractorCorpus::fit_common_mode_subspace_from_sigs(&flat_refs, max_components);

    // Strict majority (`n / 2 + 1`, clamped to `2..=n`) prevents an evenly split single-language
    // half (e.g. 12/24) from dominating the global token common-mode subspace.
    let n_sampled = sampled_indices.len();
    let min_entry_support = (n_sampled / 2 + 1).min(n_sampled).max(2);

    let cross_script_min = script_groups.and_then(|groups| {
        if groups.len() != per_entry_tok_sigs.len() {
            return None;
        }
        let mut ascii_n = 0_usize;
        let mut non_ascii_n = 0_usize;
        for &i in &sampled_indices {
            match groups[i] {
                ScriptGroup::AsciiDominant => ascii_n += 1,
                ScriptGroup::NonAsciiDominant => non_ascii_n += 1,
            }
        }
        if ascii_n >= 4 && non_ascii_n >= 4 {
            Some(((ascii_n / 4).max(2), (non_ascii_n / 4).max(2)))
        } else {
            None
        }
    });

    candidates
        .into_iter()
        .filter(|dir| {
            let mut total_support = 0_usize;
            let mut ascii_support = 0_usize;
            let mut non_ascii_support = 0_usize;
            let mut ascii_sum = vec![0.0_f32; dir.len()];
            let mut non_ascii_sum = vec![0.0_f32; dir.len()];
            for &i in &sampled_indices {
                let mut best_c = 0.0_f32;
                let mut best_tok: Option<&[f32]> = None;
                for tok in &per_entry_tok_sigs[i] {
                    let c = TextAttractorCorpus::cosine_f32(tok, dir);
                    if c * c > 0.08 && c.abs() > best_c.abs() {
                        best_c = c;
                        best_tok = Some(tok.as_slice());
                    }
                }
                if let Some(tok) = best_tok {
                    total_support += 1;
                    if let Some(groups) = script_groups {
                        // Accumulate unflipped token vectors so antipodal cross-script offsets
                        // (`+Delta` vs `-Delta`) yield negative centroid cosine and are rejected,
                        // whereas genuinely shared framing directions yield positive cosine.
                        match groups[i] {
                            ScriptGroup::AsciiDominant => {
                                ascii_support += 1;
                                for (dst, &src) in ascii_sum.iter_mut().zip(tok.iter()) {
                                    *dst += src;
                                }
                            }
                            ScriptGroup::NonAsciiDominant => {
                                non_ascii_support += 1;
                                for (dst, &src) in non_ascii_sum.iter_mut().zip(tok.iter()) {
                                    *dst += src;
                                }
                            }
                        }
                    }
                }
            }
            if total_support < min_entry_support {
                return false;
            }
            if let Some((min_ascii, min_non_ascii)) = cross_script_min {
                if ascii_support < min_ascii || non_ascii_support < min_non_ascii {
                    return false;
                }
                // Reject degenerate diagonal mixes of two orthogonal single-language directions
                // by requiring sign-aligned script centroids to agree along the shared axis.
                TextAttractorCorpus::cosine_f32(&ascii_sum, &non_ascii_sum) > 0.25
            } else {
                true
            }
        })
        .collect()
}

/// Computes the orthogonal residual norm (`[0, 1]`) of a unit token signature after projecting out
/// orthonormal common-mode directions in `basis`.
pub(super) fn token_orthogonal_residual_norm(raw_tok_sig: &[f32], basis: &[Vec<f32>]) -> f32 {
    if basis.is_empty() || raw_tok_sig.is_empty() {
        return 1.0;
    }
    let mut work = raw_tok_sig.to_vec();
    for dir in basis {
        if dir.len() != work.len() {
            continue;
        }
        let dot: f64 = work
            .iter()
            .zip(dir.iter())
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        let d = dot as f32;
        for (x, &c) in work.iter_mut().zip(dir.iter()) {
            *x -= d * c;
        }
    }
    let norm_sq: f64 = work.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    norm_sq.sqrt().clamp(0.0, 1.0) as f32
}

/// Appends orthogonalized unit directions from `extra` onto `base` via modified Gram-Schmidt up to
/// `max_total` basis vectors.
pub(super) fn merge_orthonormal_basis(
    base: &[Vec<f32>],
    extra: &[Vec<f32>],
    max_total: usize,
) -> Vec<Vec<f32>> {
    let mut out = base.to_vec();
    let Some(expected_dim) = out.first().or_else(|| extra.first()).map(Vec::len) else {
        return out;
    };
    for dir in extra {
        if out.len() >= max_total {
            break;
        }
        if dir.len() != expected_dim {
            continue;
        }
        let mut orth = dir.clone();
        for existing in &out {
            let dot: f64 = orth
                .iter()
                .zip(existing.iter())
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let d = dot as f32;
            for (dst, &c) in orth.iter_mut().zip(existing.iter()) {
                *dst -= d * c;
            }
        }
        let norm_sq: f64 = orth.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        if norm_sq >= 0.25 {
            let inv = (1.0 / norm_sq.sqrt()) as f32;
            for v in &mut orth {
                *v *= inv;
            }
            out.push(orth);
        }
    }
    out
}
