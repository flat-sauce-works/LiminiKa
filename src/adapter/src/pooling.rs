// name: src/adapter/src/pooling.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model-agnostic unsupervised sequence-token pooling and two-pass corpus compilation.
//!
//! External LLM runtimes (Candle, `llama.cpp`, ONNX) supply raw per-token, per-layer pre-RoPE Q
//! activations (`[seq_len, q_rot_len]`) to [`CompiledTokenCorpus`]. Keeping sequence aggregation
//! inside `liminika-adapter` rather than in example binaries ensures identical geometric pooling,
//! BOS/attention-sink normalization, and zero-overfitting unsupervised novelty weighting across all
//! backends and CLI entrypoints.

use crate::{
    AttractorRole, DynamicAttractorRecall, LocalChannelObservation, ModelSteeringAdapter,
    RecallPolicy, StratumPipeline, TextAttractorCorpus, TextAttractorEntry, TokenResonanceDeposit,
};
use liminika_core::traits::gateway::ManifoldProbeBaseline;
use liminika_core::traits::primitives::{load_q7_words_64, MAX_ATTENTION_HEADS};
use liminika_core::traits::telemetry::NUM_STIGMERGIC_GRANULES;

/// Unsupervised strategy collapsing per-token Q activations (`[seq_len, dim]`) into `[dim]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SequencePoolingMode {
    /// Uniform average across unit-normalized token directions (`1 / seq_len`).
    UniformMean,
    /// Unsupervised Smooth-Inverse-Frequency (SIF) novelty weighting relative to an empirical
    /// corpus reference direction.
    ///
    /// Prevents BOS/attention-sink magnitude spikes via per-token L2 normalization and attenuates
    /// common syntactic framing while smoothly falling back to uniform averaging when all tokens
    /// share similar alignment to the reference.
    CorpusNoveltySif {
        /// Exponent `alpha >= 0.0` shaping `(1.0 - max(0, cos(token, ref))).powf(alpha)`.
        novelty_power: f32,
        /// Minimum relative weight in `(0.0..=1.0]` preventing brittle single-token collapse.
        weight_floor: f32,
    },
}

impl Default for SequencePoolingMode {
    fn default() -> Self {
        Self::CorpusNoveltySif {
            novelty_power: 1.5,
            weight_floor: 0.08,
        }
    }
}

/// Stateless geometric sequence pooler transforming `[seq_len * dim]` buffers into `[dim]`.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct SequenceActivationPooler {
    /// Active pooling rule applied consistently to corpus entries and queries.
    pub mode: SequencePoolingMode,
}

impl SequenceActivationPooler {
    /// Creates a [`SequenceActivationPooler`] with the supplied [`SequencePoolingMode`].
    #[must_use]
    pub const fn new(mode: SequencePoolingMode) -> Self {
        Self { mode }
    }

    /// Computes uniform average over raw (unnormalized) token vectors for one layer.
    ///
    /// # Errors
    /// Returns an error if `seq_len == 0`, `dim == 0`, or `token_flat.len() != seq_len * dim`.
    pub fn uniform_raw_mean(
        token_flat: &[f32],
        seq_len: usize,
        dim: usize,
    ) -> Result<Vec<f32>, String> {
        if seq_len == 0 || dim == 0 {
            return Err("seq_len and dim must be > 0 for sequence pooling".to_string());
        }
        let expected = seq_len
            .checked_mul(dim)
            .ok_or_else(|| "seq_len * dim overflow".to_string())?;
        if token_flat.len() != expected {
            return Err(format!(
                "token buffer length mismatch: expected {} ({seq_len}x{dim}), got {}",
                expected,
                token_flat.len()
            ));
        }
        let inv = 1.0 / (seq_len as f32);
        let mut out = vec![0.0_f32; dim];
        for t in 0..seq_len {
            let slice = &token_flat[t * dim..(t + 1) * dim];
            for (dst, &src) in out.iter_mut().zip(slice.iter()) {
                *dst += src * inv;
            }
        }
        Ok(out)
    }

    /// Computes uniform average over unit-normalized token vectors for one layer.
    ///
    /// # Errors
    /// Returns an error if `seq_len == 0`, `dim == 0`, or `token_flat.len() != seq_len * dim`.
    pub fn uniform_unit_mean(
        token_flat: &[f32],
        seq_len: usize,
        dim: usize,
    ) -> Result<Vec<f32>, String> {
        Self::pool_layer_with_mode(
            SequencePoolingMode::UniformMean,
            token_flat,
            seq_len,
            dim,
            None,
        )
    }

    /// Pools one layer's flattened `[seq_len, dim]` activations into `[dim]` using `self.mode`.
    ///
    /// # Errors
    /// Returns an error on empty shapes, mismatched slice length, or invalid reference length.
    pub fn pool_layer(
        &self,
        token_flat: &[f32],
        seq_len: usize,
        dim: usize,
        reference: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        Self::pool_layer_with_mode(self.mode, token_flat, seq_len, dim, reference)
    }

    fn pool_layer_with_mode(
        mode: SequencePoolingMode,
        token_flat: &[f32],
        seq_len: usize,
        dim: usize,
        reference: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        if seq_len == 0 || dim == 0 {
            return Err("seq_len and dim must be > 0 for sequence pooling".to_string());
        }
        let expected = seq_len
            .checked_mul(dim)
            .ok_or_else(|| "seq_len * dim overflow".to_string())?;
        if token_flat.len() != expected {
            return Err(format!(
                "token buffer length mismatch: expected {} ({seq_len}x{dim}), got {}",
                expected,
                token_flat.len()
            ));
        }
        if let Some(r) = reference {
            if r.len() != dim {
                return Err(format!(
                    "reference dimension mismatch: expected {dim}, got {}",
                    r.len()
                ));
            }
        }

        let mut weights = Vec::with_capacity(seq_len);
        let mut unit_tokens: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let slice = &token_flat[t * dim..(t + 1) * dim];
            let norm_sq: f64 = slice.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            let norm = norm_sq.sqrt();
            let mut unit = vec![0.0_f32; dim];
            if norm > 1e-12 {
                let inv = (1.0 / norm) as f32;
                for (dst, &src) in unit.iter_mut().zip(slice.iter()) {
                    *dst = src * inv;
                }
            }
            let w = match (mode, reference) {
                (SequencePoolingMode::UniformMean, _) | (_, None) => 1.0_f32,
                (
                    SequencePoolingMode::CorpusNoveltySif {
                        novelty_power,
                        weight_floor,
                    },
                    Some(ref_vec),
                ) => {
                    let cos_pos = TextAttractorCorpus::cosine_f32(&unit, ref_vec).max(0.0);
                    let floor = weight_floor.clamp(1e-3, 1.0);
                    let novelty = (1.0 - cos_pos).clamp(floor, 1.0);
                    novelty.powf(novelty_power.max(0.0))
                }
            };
            weights.push(w);
            unit_tokens.push(unit);
        }

        let weight_sum: f64 = weights.iter().map(|&w| f64::from(w)).sum();
        let inv_sum = if weight_sum > 1e-12 {
            (1.0 / weight_sum) as f32
        } else {
            1.0 / (seq_len as f32)
        };

        let mut out = vec![0.0_f32; dim];
        for (w, unit) in weights.iter().zip(unit_tokens.iter()) {
            let scale = *w * inv_sum;
            for (dst, &src) in out.iter_mut().zip(unit.iter()) {
                *dst += scale * src;
            }
        }
        Ok(out)
    }

    /// Pools every transformer layer's `[seq_len, dim]` activations into `[num_layers][dim]`.
    ///
    /// # Errors
    /// Returns an error if any layer buffer or optional reference slice mismatches dimensions.
    pub fn pool_sequence(
        &self,
        per_layer_token_qs: &[Vec<f32>],
        seq_len: usize,
        dim: usize,
        ref_per_layer: Option<&[Vec<f32>]>,
    ) -> Result<Vec<Vec<f32>>, String> {
        if let Some(refs) = ref_per_layer {
            if refs.len() != per_layer_token_qs.len() {
                return Err(format!(
                    "reference layer count mismatch: {} vs {}",
                    refs.len(),
                    per_layer_token_qs.len()
                ));
            }
        }
        let mut out = Vec::with_capacity(per_layer_token_qs.len());
        for (l, token_flat) in per_layer_token_qs.iter().enumerate() {
            let ref_slice = ref_per_layer
                .and_then(|refs| refs.get(l))
                .map(Vec::as_slice);
            out.push(self.pool_layer(token_flat, seq_len, dim, ref_slice)?);
        }
        Ok(out)
    }
}

/// Raw unpooled per-token sequence capture for one natural-language attractor entry.
#[derive(Debug, Clone)]
pub struct TokenSequenceEntry {
    /// Target semantic role (`Fact`, `Persona`, or `Guard`).
    pub role: AttractorRole,
    /// Original natural-language text corresponding to this sequence.
    pub text: String,
    /// Initial attractor basin energy.
    pub energy: f32,
    /// Number of tokens in this captured sequence (`S`).
    pub seq_len: usize,
    /// Per-layer flattened row-major token activations `[seq_len * q_rot_len]`.
    pub per_layer_token_qs: Vec<Vec<f32>>,
}

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
}

impl ModelBaseAttractorProfile {
    /// Extracts a [`ModelBaseAttractorProfile`] from unsteered startup sequences via two-pass
    /// pooling, PCA common-mode fitting, and cosine-threshold online sub-basin partitioning.
    ///
    /// # Errors
    /// Returns an error if `samples` is empty or any sample's tensor shape disagrees with `adapter`.
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
        let n = samples.len() as f32;

        let mut novelty_reference_per_layer = vec![vec![0.0_f32; dim]; num_layers];
        for s in samples {
            if s.per_layer_token_qs.len() != num_layers {
                return Err(format!(
                    "sample layer count mismatch: expected {num_layers}, got {}",
                    s.per_layer_token_qs.len()
                ));
            }
            for (l, flat) in s.per_layer_token_qs.iter().enumerate() {
                let u = SequenceActivationPooler::uniform_unit_mean(flat, s.seq_len, dim)?;
                for (dst, &src) in novelty_reference_per_layer[l].iter_mut().zip(u.iter()) {
                    *dst += src / n;
                }
            }
        }

        let ref_slices: Vec<&[f32]> = novelty_reference_per_layer
            .iter()
            .map(Vec::as_slice)
            .collect();
        let mut pooled_sigs: Vec<Vec<f32>> = Vec::with_capacity(samples.len());
        let mut raw_tok_sigs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(samples.len());
        for s in samples {
            let pooled = pooler.pool_sequence(
                &s.per_layer_token_qs,
                s.seq_len,
                dim,
                Some(&novelty_reference_per_layer),
            )?;
            let p_refs: Vec<&[f32]> = pooled.iter().map(Vec::as_slice).collect();
            pooled_sigs.push(TextAttractorCorpus::build_steered_residual_sig(
                geom,
                &p_refs,
                &ref_slices,
            ));
            raw_tok_sigs.push(CompiledTokenCorpus::build_raw_token_sigs_for_entry(
                adapter,
                &s.per_layer_token_qs,
                s.seq_len,
                &novelty_reference_per_layer,
            )?);
        }

        let pooled_common_mode_basis =
            TextAttractorCorpus::fit_common_mode_subspace_from_sigs(&pooled_sigs, 3);
        let token_common_mode_basis =
            CompiledTokenCorpus::fit_cross_entry_token_common_mode(&raw_tok_sigs, 6);

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
            sample_count: samples.len(),
        })
    }
}

/// Self-contained Cold-Path corpus compiled from raw multi-token Q activations via two-pass
/// unsupervised centroid estimation + SIF novelty pooling + salient token point clouds.
#[derive(Debug, Clone)]
pub struct CompiledTokenCorpus {
    /// Compiled attractor manifold corpus.
    pub corpus: TextAttractorCorpus,
    /// Pass-1 uniform centroid per layer used strictly as the unsupervised token-novelty baseline.
    pub novelty_reference_per_layer: Vec<Vec<f32>>,
    /// Pass-2 centroid matching the pooled distribution of `corpus` entries.
    pub corpus_centroid_per_layer: Vec<Vec<f32>>,
    /// Unsupervised cross-entry common-mode subspace extracted strictly in token space.
    pub token_common_mode_basis: Vec<Vec<f32>>,
    /// Top salient token signatures per corpus entry (`[entry][token_k][steered_dim]`) for
    /// ColBERT-style late interaction (`local_token_scores`).
    pub entry_token_sigs: Vec<Vec<Vec<f32>>>,
    /// Unsupervised cross-entry token specificity weights (`1 / (1 + sum_{j!=i} max_sim_+^2)`)
    /// down-weighting corpus-frequent tokens without keyword heuristics.
    pub entry_token_weights: Vec<Vec<f32>>,
    /// Leave-one-out empirical MaxSim hubness baseline per corpus entry.
    pub entry_token_hub_bias: Vec<f32>,
    /// Mean leave-one-out MaxSim hubness across the corpus.
    pub global_token_hub_mean: f32,
    /// Core Cold-Path manifold baseline synthesized from empirical corpus statistics.
    pub probe_baseline: ManifoldProbeBaseline,
    /// Empirically calibrated [`RecallPolicy`] derived from projected corpus similarities.
    pub recall_policy: RecallPolicy,
    /// Pooler configuration applied identically during compilation and query recall.
    pub pooler: SequenceActivationPooler,
}

impl CompiledTokenCorpus {
    fn fit_cross_entry_token_common_mode(
        per_entry_tok_sigs: &[Vec<Vec<f32>>],
        max_components: usize,
    ) -> Vec<Vec<f32>> {
        if per_entry_tok_sigs.len() < 3 || max_components == 0 {
            return Vec::new();
        }
        let flat_sigs: Vec<Vec<f32>> = per_entry_tok_sigs
            .iter()
            .flat_map(|toks| toks.iter().cloned())
            .collect();
        let candidates =
            TextAttractorCorpus::fit_common_mode_subspace_from_sigs(&flat_sigs, max_components);
        let min_entry_support = (per_entry_tok_sigs.len() / 2).max(2);
        candidates
            .into_iter()
            .filter(|dir| {
                let supporting_entries = per_entry_tok_sigs
                    .iter()
                    .filter(|toks| {
                        toks.iter().any(|tok| {
                            let c = TextAttractorCorpus::cosine_f32(tok, dir);
                            c * c > 0.08
                        })
                    })
                    .count();
                supporting_entries >= min_entry_support
            })
            .collect()
    }

    fn token_orthogonal_residual_norm(raw_tok_sig: &[f32], basis: &[Vec<f32>]) -> f32 {
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

    fn compute_unweighted_maxsim(query_tokens: &[Vec<f32>], doc_tokens: &[Vec<f32>]) -> f32 {
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

    fn build_raw_token_sigs_for_entry(
        adapter: &ModelSteeringAdapter,
        per_layer_token_qs: &[Vec<f32>],
        seq_len: usize,
        novelty_reference_per_layer: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, String> {
        let geom = adapter.geometry();
        let dim = geom.q_rot_len();
        let ref_slices: Vec<&[f32]> = novelty_reference_per_layer
            .iter()
            .map(Vec::as_slice)
            .collect();
        let mut out = Vec::with_capacity(seq_len);
        let mut unit_tok_layers: Vec<Vec<f32>> = vec![vec![0.0_f32; dim]; per_layer_token_qs.len()];
        for t in 0..seq_len {
            for (l, layer_flat) in per_layer_token_qs.iter().enumerate() {
                let start = t * dim;
                let end = start + dim;
                if end > layer_flat.len() {
                    return Err(format!(
                        "token slice {start}..{end} exceeds {}",
                        layer_flat.len()
                    ));
                }
                let src = &layer_flat[start..end];
                let norm_sq: f64 = src.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
                let norm = norm_sq.sqrt();
                let inv = if norm > 1e-12 {
                    (1.0 / norm) as f32
                } else {
                    0.0
                };
                for (dst, &v) in unit_tok_layers[l].iter_mut().zip(src.iter()) {
                    *dst = v * inv;
                }
            }
            let tok_refs: Vec<&[f32]> = unit_tok_layers.iter().map(Vec::as_slice).collect();
            out.push(TextAttractorCorpus::build_steered_residual_sig(
                geom,
                &tok_refs,
                &ref_slices,
            ));
        }
        Ok(out)
    }

    fn extract_salient_token_sigs_indexed(
        adapter: &ModelSteeringAdapter,
        per_layer_token_qs: &[Vec<f32>],
        seq_len: usize,
        novelty_reference_per_layer: &[Vec<f32>],
        token_common_mode_basis: &[Vec<f32>],
        max_tokens: usize,
        pooler: SequenceActivationPooler,
    ) -> Result<Vec<(usize, Vec<f32>, f32)>, String> {
        let geom = adapter.geometry();
        let dim = geom.q_rot_len();
        if seq_len == 0 || max_tokens == 0 {
            return Ok(Vec::new());
        }
        let raw_tok_sigs = Self::build_raw_token_sigs_for_entry(
            adapter,
            per_layer_token_qs,
            seq_len,
            novelty_reference_per_layer,
        )?;
        let mut token_scores: Vec<(usize, f32)> = Vec::with_capacity(seq_len);
        let (power, floor) = match pooler.mode {
            SequencePoolingMode::UniformMean => (0.0_f32, 1.0_f32),
            SequencePoolingMode::CorpusNoveltySif {
                novelty_power,
                weight_floor,
            } => (novelty_power.max(0.0), weight_floor.clamp(1e-3, 1.0)),
        };

        for (t, raw_sig) in raw_tok_sigs.iter().enumerate() {
            let mut sum_w = 0.0_f32;
            let mut count = 0_usize;
            for (l, layer_flat) in per_layer_token_qs.iter().enumerate() {
                if !geom.is_rope_layer(l) {
                    continue;
                }
                let start = t * dim;
                let end = start + dim;
                let slice = &layer_flat[start..end];
                let cos_pos =
                    TextAttractorCorpus::cosine_f32(slice, &novelty_reference_per_layer[l])
                        .max(0.0);
                let w = (1.0 - cos_pos).clamp(floor, 1.0).powf(power);
                sum_w += w;
                count += 1;
            }
            let mean_w = if count > 0 {
                sum_w / (count as f32)
            } else {
                1.0
            };
            let ortho_ratio =
                Self::token_orthogonal_residual_norm(raw_sig, token_common_mode_basis);
            // Suppress tokens whose energy lies mostly inside the cross-entry common-mode subspace
            // before re-normalization amplifies small residual noise.
            let salience = mean_w * ortho_ratio * ortho_ratio;
            token_scores.push((t, salience));
        }

        token_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        token_scores.truncate(max_tokens.min(seq_len));

        let mut out = Vec::with_capacity(token_scores.len());
        for (t, w) in token_scores {
            let proj_tok_sig = TextAttractorCorpus::project_and_normalize(
                &raw_tok_sigs[t],
                token_common_mode_basis,
            );
            out.push((t, proj_tok_sig, w));
        }
        Ok(out)
    }
    fn merge_orthonormal_basis(
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

    /// Compiles a list of raw token-sequence captures into a calibrated [`CompiledTokenCorpus`].
    ///
    /// Pass 1 computes an unsupervised uniform-mean centroid across all entries to serve as a
    /// neutral boilerplate reference. Pass 2 pools each entry with `pooler` against that neutral
    /// reference, builds the matched Pass-2 centroid, fits the common-mode projection subspace, and
    /// calibrates [`RecallPolicy`] automatically from empirical off-diagonal corpus geometry.
    ///
    /// # Errors
    /// Returns an error if `entries` is empty or any layer tensor shape disagrees with `adapter`.
    pub fn compile(
        adapter: &ModelSteeringAdapter,
        entries: &[TokenSequenceEntry],
        pooler: SequenceActivationPooler,
        top_k_facts: usize,
        top_k_personas: usize,
    ) -> Result<Self, String> {
        Self::compile_with_base_prior(adapter, entries, pooler, top_k_facts, top_k_personas, None)
    }

    /// Compiles raw token-sequence captures while optionally incorporating a pre-rendered
    /// [`ModelBaseAttractorProfile`] so the host LLM's background framing and default gravity
    /// directions are orthogonalized before calibrating recall boundaries.
    ///
    /// # Errors
    /// Returns an error if `entries` is empty or any layer tensor shape disagrees with `adapter`.
    pub fn compile_with_base_prior(
        adapter: &ModelSteeringAdapter,
        entries: &[TokenSequenceEntry],
        pooler: SequenceActivationPooler,
        top_k_facts: usize,
        top_k_personas: usize,
        base_prior: Option<&ModelBaseAttractorProfile>,
    ) -> Result<Self, String> {
        if entries.is_empty() {
            return Err("cannot compile an empty TokenSequenceEntry slice".to_string());
        }
        let num_layers = adapter.geometry().num_layers;
        let dim = adapter.geometry().q_rot_len();
        let n = entries.len() as f32;

        // Pass 1: unsupervised uniform centroid over unit-normalized token vectors, optionally
        // blended with the startup base-model reference when available.
        let mut novelty_reference_per_layer = vec![vec![0.0_f32; dim]; num_layers];
        for entry in entries {
            if entry.per_layer_token_qs.len() != num_layers {
                return Err(format!(
                    "entry layer count mismatch: expected {num_layers}, got {}",
                    entry.per_layer_token_qs.len()
                ));
            }
            for (l, flat) in entry.per_layer_token_qs.iter().enumerate() {
                let u = SequenceActivationPooler::uniform_unit_mean(flat, entry.seq_len, dim)?;
                for (dst, &src) in novelty_reference_per_layer[l].iter_mut().zip(u.iter()) {
                    *dst += src / n;
                }
            }
        }
        if let Some(prior) = base_prior {
            if prior.novelty_reference_per_layer.len() == num_layers {
                for (dst_layer, prior_layer) in novelty_reference_per_layer
                    .iter_mut()
                    .zip(prior.novelty_reference_per_layer.iter())
                {
                    if dst_layer.len() == prior_layer.len() {
                        for (dst, &src) in dst_layer.iter_mut().zip(prior_layer.iter()) {
                            *dst = 0.70 * *dst + 0.30 * src;
                        }
                    }
                }
            }
        }

        // Pass 2: pool every entry against Pass-1 reference and compute matched centroid.
        let mut pooled_entries: Vec<Vec<Vec<f32>>> = Vec::with_capacity(entries.len());
        let mut corpus_centroid_per_layer = vec![vec![0.0_f32; dim]; num_layers];
        for entry in entries {
            let pooled = pooler.pool_sequence(
                &entry.per_layer_token_qs,
                entry.seq_len,
                dim,
                Some(&novelty_reference_per_layer),
            )?;
            for (l, vec_l) in pooled.iter().enumerate() {
                for (dst, &src) in corpus_centroid_per_layer[l].iter_mut().zip(vec_l.iter()) {
                    *dst += src / n;
                }
            }
            pooled_entries.push(pooled);
        }

        let centroid_refs: Vec<&[f32]> = corpus_centroid_per_layer
            .iter()
            .map(Vec::as_slice)
            .collect();
        let mut corpus = TextAttractorCorpus::new();
        for (entry, pooled) in entries.iter().zip(pooled_entries.iter()) {
            let q_refs: Vec<&[f32]> = pooled.iter().map(Vec::as_slice).collect();
            corpus.ingest_text_activations(
                adapter,
                entry.role,
                entry.text.as_str(),
                &q_refs,
                &centroid_refs,
                entry.energy,
            )?;
        }
        corpus.fit_common_mode_projection();
        if let Some(prior) = base_prior {
            corpus.extend_common_mode_basis(&prior.pooled_common_mode_basis, 5);
        }
        let recall_policy = corpus.calibrate_recall_policy(top_k_facts, top_k_personas);

        // Estimate unsupervised token-space common mode across entries (distinct from pooled
        // document PCA) so BOS/attention-sink/template token directions are removed before MaxSim.
        let mut per_entry_raw_toks: Vec<Vec<Vec<f32>>> = Vec::with_capacity(entries.len());
        for entry in entries {
            per_entry_raw_toks.push(Self::build_raw_token_sigs_for_entry(
                adapter,
                &entry.per_layer_token_qs,
                entry.seq_len,
                &novelty_reference_per_layer,
            )?);
        }
        let corpus_tok_basis = Self::fit_cross_entry_token_common_mode(&per_entry_raw_toks, 10);
        let token_common_mode_basis = if let Some(prior) = base_prior {
            Self::merge_orthonormal_basis(&corpus_tok_basis, &prior.token_common_mode_basis, 12)
        } else {
            corpus_tok_basis
        };

        let mut entry_token_sigs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(entries.len());
        for entry in entries {
            let salient = Self::extract_salient_token_sigs_indexed(
                adapter,
                &entry.per_layer_token_qs,
                entry.seq_len,
                &novelty_reference_per_layer,
                &token_common_mode_basis,
                24,
                pooler,
            )?;
            entry_token_sigs.push(salient.into_iter().map(|(_, sig, _)| sig).collect());
        }

        let (entry_token_weights, entry_token_hub_bias, global_token_hub_mean) =
            Self::recompute_token_specificity_and_hubness(&entry_token_sigs, corpus.entries());

        let geom = adapter.geometry();
        let steered_layers = (0..geom.num_layers)
            .filter(|&l| geom.is_rope_layer(l))
            .count()
            .max(1) as u16;
        let probe_baseline = ManifoldProbeBaseline {
            active_heads: geom.num_q_heads.min(MAX_ATTENTION_HEADS) as u16,
            steered_layers,
            baseline_entropy: 0.35,
            baseline_inter_layer_coherence: recall_policy.min_coherence,
            baseline_guard_noise_floor: 0.45,
            probe_epoch: 1,
        };

        Ok(Self {
            corpus,
            novelty_reference_per_layer,
            corpus_centroid_per_layer,
            token_common_mode_basis,
            entry_token_sigs,
            entry_token_weights,
            entry_token_hub_bias,
            global_token_hub_mean,
            probe_baseline,
            recall_policy,
            pooler,
        })
    }

    fn recompute_token_specificity_and_hubness(
        entry_token_sigs: &[Vec<Vec<f32>>],
        entries: &[TextAttractorEntry],
    ) -> (Vec<Vec<f32>>, Vec<f32>, f32) {
        // Estimate same-role cross-entry doc-token specificity weights
        // (`1 / (1 + sum_{j!=i, role_j == role_i} max_sim_+^2)`) so contrastive Guards echoing
        // factual entity names do not dilute factual token specificity, alongside leave-one-out
        // empirical MaxSim hubness baselines.
        let n_entries = entry_token_sigs.len();
        let mut entry_token_weights: Vec<Vec<f32>> = Vec::with_capacity(n_entries);
        for i in 0..n_entries {
            let role_i = entries.get(i).map(|e| e.role);
            let mut weights_i = Vec::with_capacity(entry_token_sigs[i].len());
            for d_tok in &entry_token_sigs[i] {
                let mut other_sim_sq = 0.0_f32;
                for (j, other_toks) in entry_token_sigs.iter().enumerate() {
                    let same_role = role_i
                        .zip(entries.get(j).map(|e| e.role))
                        .is_none_or(|(ri, rj)| ri == rj);
                    if i == j || other_toks.is_empty() || !same_role {
                        continue;
                    }
                    let best_j = other_toks
                        .iter()
                        .map(|od| TextAttractorCorpus::cosine_f32(d_tok, od))
                        .fold(0.0_f32, f32::max);
                    // Only accumulate genuine cross-entry overlap above random cosine noise floor.
                    let excess = (best_j - 0.30).max(0.0);
                    other_sim_sq += excess * excess;
                }
                let spec = (1.0 / (1.0 + 3.0 * other_sim_sq)).clamp(0.10, 1.0);
                weights_i.push(spec);
            }
            entry_token_weights.push(weights_i);
        }

        let mut entry_token_hub_bias = vec![0.0_f32; n_entries];
        let mut global_token_hub_mean = 0.0_f32;
        if n_entries >= 2 {
            let denom = (n_entries - 1) as f32;
            for j in 0..n_entries {
                let mut sum = 0.0_f32;
                for i in 0..n_entries {
                    if i == j {
                        continue;
                    }
                    sum +=
                        Self::compute_unweighted_maxsim(&entry_token_sigs[i], &entry_token_sigs[j]);
                }
                entry_token_hub_bias[j] = sum / denom;
            }
            global_token_hub_mean = entry_token_hub_bias.iter().sum::<f32>() / (n_entries as f32);
        }

        (
            entry_token_weights,
            entry_token_hub_bias,
            global_token_hub_mean,
        )
    }

    /// Appends auxiliary per-layer token Q activations (such as a space-prefixed BPE alias of a
    /// sentence-initial token) into `entry_idx`'s salient token cloud without altering `pooled_q`
    /// or `corpus_centroid_per_layer`, then recalibrates cross-entry token specificity weights.
    ///
    /// # Errors
    /// Returns an error if `entry_idx` is out of range or `alias_token_qs` mismatches `adapter`.
    pub fn register_bpe_prefix_alias_tokens(
        &mut self,
        adapter: &ModelSteeringAdapter,
        entry_idx: usize,
        alias_token_qs: &[Vec<f32>],
        alias_seq_len: usize,
    ) -> Result<(), String> {
        if entry_idx >= self.entry_token_sigs.len() {
            return Err(format!(
                "entry_idx {entry_idx} out of range (len={})",
                self.entry_token_sigs.len()
            ));
        }
        let num_layers = adapter.geometry().num_layers;
        if alias_token_qs.len() != num_layers {
            return Err(format!(
                "alias layer count mismatch: expected {num_layers}, got {}",
                alias_token_qs.len()
            ));
        }
        if alias_seq_len == 0 {
            return Ok(());
        }
        let salient = Self::extract_salient_token_sigs_indexed(
            adapter,
            alias_token_qs,
            alias_seq_len,
            &self.novelty_reference_per_layer,
            &self.token_common_mode_basis,
            alias_seq_len,
            self.pooler,
        )?;
        for (_, sig, _) in salient {
            self.entry_token_sigs[entry_idx].push(sig);
        }
        let (weights, hub_bias, global_hub_mean) = Self::recompute_token_specificity_and_hubness(
            &self.entry_token_sigs,
            self.corpus.entries(),
        );
        self.entry_token_weights = weights;
        self.entry_token_hub_bias = hub_bias;
        self.global_token_hub_mean = global_hub_mean;
        Ok(())
    }

    /// Pools a raw query token sequence using the frozen pooler and recalls via the default
    /// [`StratumPipeline`].
    ///
    /// # Errors
    /// Returns an error if layer count, token length, or stratum inference fails.
    pub fn recall_token_sequence<'a>(
        &'a self,
        adapter: &ModelSteeringAdapter,
        query_token_qs: &[Vec<f32>],
        query_seq_len: usize,
    ) -> Result<DynamicAttractorRecall<'a>, String> {
        self.recall_token_sequence_with_strata_and_circuit(
            adapter,
            query_token_qs,
            query_seq_len,
            &StratumPipeline::default(),
            None,
        )
    }

    /// Scores a raw query token sequence while seeding Pass 1 stratum inference with `prior_circuit`.
    ///
    /// # Errors
    /// Returns an error if layer count, token length, or stratum inference fails.
    pub fn recall_token_sequence_with_circuit<'a>(
        &'a self,
        adapter: &ModelSteeringAdapter,
        query_token_qs: &[Vec<f32>],
        query_seq_len: usize,
        prior_circuit: Option<liminika_core::traits::gateway::CognitiveTrajectoryLedger>,
    ) -> Result<DynamicAttractorRecall<'a>, String> {
        self.recall_token_sequence_with_strata_and_circuit(
            adapter,
            query_token_qs,
            query_seq_len,
            &StratumPipeline::default(),
            prior_circuit,
        )
    }

    /// Builds an uncollapsed [`LocalChannelObservation`] (bidirectional Chamfer scores and
    /// per-token resonance deposits) for a query token sequence.
    ///
    /// # Errors
    /// Returns an error if layer count or token slice dimensions mismatch `adapter`.
    pub fn build_local_observation(
        &self,
        adapter: &ModelSteeringAdapter,
        query_token_qs: &[Vec<f32>],
        query_seq_len: usize,
    ) -> Result<LocalChannelObservation, String> {
        let num_layers = adapter.geometry().num_layers;
        let dim = adapter.geometry().q_rot_len();
        if query_token_qs.len() != num_layers {
            return Err(format!(
                "query layer count mismatch: expected {num_layers}, got {}",
                query_token_qs.len()
            ));
        }
        let query_salient = Self::extract_salient_token_sigs_indexed(
            adapter,
            query_token_qs,
            query_seq_len,
            &self.novelty_reference_per_layer,
            &self.token_common_mode_basis,
            24,
            self.pooler,
        )?;
        let n_docs = self.entry_token_sigs.len();
        let mut combined_scores = vec![0.0_f32; n_docs];
        let mut q_to_d_scores = vec![0.0_f32; n_docs];
        let mut d_to_q_scores = vec![0.0_f32; n_docs];
        let mut token_deposits: Vec<TokenResonanceDeposit> = Vec::new();
        let last_rope_layer = (0..num_layers)
            .rfind(|&l| adapter.geometry().is_rope_layer(l))
            .unwrap_or(0);

        if n_docs > 0 && !query_salient.is_empty() {
            // Unsupervised token-level discriminative centering (ColBERT-IDF equivalent):
            // down-weight query tokens that respond uniformly across all corpus entries and
            // center each token's MaxSim row around its corpus-wide mean before aggregation.
            let mut weighted_sums = vec![0.0_f32; n_docs];
            let mut total_disc_weight = 0.0_f32;
            let mut raw_weighted_sums = vec![0.0_f32; n_docs];
            let mut total_raw_weight = 0.0_f32;

            for (tok_idx, q_tok, w_t) in &query_salient {
                let mut row_sims = vec![0.0_f32; n_docs];
                let mut sum_row = 0.0_f32;
                let mut max_row = f32::NEG_INFINITY;
                let mut valid_docs = 0_usize;
                for (i, doc_tokens) in self.entry_token_sigs.iter().enumerate() {
                    if doc_tokens.is_empty() {
                        continue;
                    }
                    let doc_weights = self.entry_token_weights.get(i);
                    let max_sim = doc_tokens
                        .iter()
                        .enumerate()
                        .map(|(k, d_tok)| {
                            let c = TextAttractorCorpus::cosine_f32(q_tok, d_tok);
                            if c > 0.0 {
                                let spec =
                                    doc_weights.and_then(|ws| ws.get(k).copied()).unwrap_or(1.0);
                                c * spec
                            } else {
                                c
                            }
                        })
                        .fold(f32::NEG_INFINITY, f32::max);
                    if max_sim.is_finite() {
                        row_sims[i] = max_sim;
                        sum_row += max_sim;
                        if max_sim > max_row {
                            max_row = max_sim;
                        }
                        valid_docs += 1;
                    }
                }
                if valid_docs == 0 {
                    continue;
                }
                let mean_row = sum_row / (valid_docs as f32);
                let disc = (max_row - mean_row).max(0.0);
                let eff_w = *w_t * disc * disc;
                if eff_w > 1e-8 {
                    let mut entry_excess = Vec::with_capacity(n_docs);
                    for (dst, &s) in weighted_sums.iter_mut().zip(row_sims.iter()) {
                        let ex = s - mean_row;
                        *dst += eff_w * ex;
                        entry_excess.push(ex);
                    }
                    total_disc_weight += eff_w;

                    let mut token_phase_words = [0_u64; NUM_STIGMERGIC_GRANULES];
                    let start = tok_idx.saturating_mul(dim);
                    let end = start.saturating_add(dim);
                    if let (Some(layer_flat), Some(ref_slice)) = (
                        query_token_qs.get(last_rope_layer),
                        self.novelty_reference_per_layer.get(last_rope_layer),
                    ) {
                        if end <= layer_flat.len() {
                            if let Ok(q7) =
                                adapter.bake_layer_procrustes_q7(ref_slice, &layer_flat[start..end])
                            {
                                let mut arr64 = [0_i8; MAX_ATTENTION_HEADS];
                                let n = q7.len().min(MAX_ATTENTION_HEADS);
                                arr64[..n].copy_from_slice(&q7[..n]);
                                token_phase_words = load_q7_words_64(&arr64);
                            }
                        }
                    }
                    token_deposits.push(TokenResonanceDeposit {
                        disc_weight: eff_w,
                        entry_excess,
                        token_phase_words,
                    });
                }
                for (dst, &s) in raw_weighted_sums.iter_mut().zip(row_sims.iter()) {
                    *dst += *w_t * s;
                }
                total_raw_weight += *w_t;
            }

            // Symmetric Chamfer doc-to-query specificity coverage: down-weights broad overview docs
            // whose extra tokens do not appear in the query, centered across valid entries.
            let mut d_to_q = vec![0.0_f32; n_docs];
            let mut sum_d_to_q = 0.0_f32;
            let mut valid_d_to_q = 0_usize;
            for (i, doc_tokens) in self.entry_token_sigs.iter().enumerate() {
                if doc_tokens.is_empty() {
                    continue;
                }
                let doc_weights = self.entry_token_weights.get(i);
                let mut sum_cov = 0.0_f32;
                let mut sum_spec = 0.0_f32;
                for (k, d_tok) in doc_tokens.iter().enumerate() {
                    let spec = doc_weights.and_then(|ws| ws.get(k).copied()).unwrap_or(1.0);
                    let max_q = query_salient
                        .iter()
                        .map(|(_, q_tok, _)| TextAttractorCorpus::cosine_f32(d_tok, q_tok))
                        .fold(f32::NEG_INFINITY, f32::max);
                    if max_q.is_finite() {
                        sum_cov += spec * max_q;
                        sum_spec += spec;
                    }
                }
                if sum_spec > 1e-8 {
                    let mean_cov = sum_cov / sum_spec;
                    d_to_q[i] = mean_cov;
                    sum_d_to_q += mean_cov;
                    valid_d_to_q += 1;
                }
            }
            let mean_d_to_q = if valid_d_to_q > 0 {
                sum_d_to_q / (valid_d_to_q as f32)
            } else {
                0.0
            };

            if total_disc_weight > 1e-8 {
                for (i, &ws) in weighted_sums.iter().enumerate() {
                    let q_to_d_score = ws / total_disc_weight;
                    let d_to_q_centered = if self.entry_token_sigs[i].is_empty() {
                        0.0
                    } else {
                        d_to_q[i] - mean_d_to_q
                    };
                    q_to_d_scores[i] = q_to_d_score;
                    d_to_q_scores[i] = d_to_q_centered;
                    combined_scores[i] = 0.65 * q_to_d_score + 0.35 * d_to_q_centered;
                }
            } else if total_raw_weight > 1e-12 {
                for (i, &ws) in raw_weighted_sums.iter().enumerate() {
                    let raw_maxsim = ws / total_raw_weight;
                    let hub_shift = self.entry_token_hub_bias.get(i).copied().unwrap_or(0.0)
                        - self.global_token_hub_mean;
                    q_to_d_scores[i] = raw_maxsim;
                    combined_scores[i] = raw_maxsim - 0.65 * hub_shift;
                }
            }
        }

        Ok(LocalChannelObservation {
            combined_scores,
            q_to_d_scores,
            d_to_q_scores,
            token_deposits,
        })
    }

    /// Pools a raw query token sequence and recalls through a caller-supplied [`StratumPipeline`].
    ///
    /// # Errors
    /// Returns an error if layer count, token length, or stratum inference fails.
    pub fn recall_token_sequence_with_strata<'a>(
        &'a self,
        adapter: &ModelSteeringAdapter,
        query_token_qs: &[Vec<f32>],
        query_seq_len: usize,
        pipeline: &StratumPipeline<'_>,
    ) -> Result<DynamicAttractorRecall<'a>, String> {
        self.recall_token_sequence_with_strata_and_circuit(
            adapter,
            query_token_qs,
            query_seq_len,
            pipeline,
            None,
        )
    }

    /// Scores a raw query token sequence using a custom [`StratumPipeline`] and optional prior
    /// [`CognitiveTrajectoryLedger`](liminika_core::traits::gateway::CognitiveTrajectoryLedger).
    ///
    /// # Errors
    /// Returns an error if layer count, token length, or stratum inference fails.
    pub fn recall_token_sequence_with_strata_and_circuit<'a>(
        &'a self,
        adapter: &ModelSteeringAdapter,
        query_token_qs: &[Vec<f32>],
        query_seq_len: usize,
        pipeline: &StratumPipeline<'_>,
        prior_circuit: Option<liminika_core::traits::gateway::CognitiveTrajectoryLedger>,
    ) -> Result<DynamicAttractorRecall<'a>, String> {
        let num_layers = adapter.geometry().num_layers;
        let dim = adapter.geometry().q_rot_len();
        if query_token_qs.len() != num_layers {
            return Err(format!(
                "query layer count mismatch: expected {num_layers}, got {}",
                query_token_qs.len()
            ));
        }
        let pooled_q = self.pooler.pool_sequence(
            query_token_qs,
            query_seq_len,
            dim,
            Some(&self.novelty_reference_per_layer),
        )?;
        let obs = self.build_local_observation(adapter, query_token_qs, query_seq_len)?;
        let q_refs: Vec<&[f32]> = pooled_q.iter().map(Vec::as_slice).collect();
        let centroid_refs: Vec<&[f32]> = self
            .corpus_centroid_per_layer
            .iter()
            .map(Vec::as_slice)
            .collect();
        self.corpus
            .recall_for_query_with_strata_observation_and_circuit(
                adapter,
                &q_refs,
                &centroid_refs,
                &self.recall_policy,
                pipeline,
                Some(obs),
                prior_circuit,
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RopeGeometry, VoidBoundaryState};

    #[test]
    fn novelty_sif_pooling_resists_attention_sink_norm_spike_and_boilerplate() {
        let dim = 8;
        // Reference represents boilerplate direction [1, 0, 0, ...].
        let ref_dir = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // Sequence:
        //   token 0: huge 25x BOS/sink spike along boilerplate axis [25, 0, ...]
        //   token 1..3: normal boilerplate tokens [1, 0, ...]
        //   token 4: rare entity signal along orthogonal axis [0, 0, 1, 0, ...]
        let mut seq = Vec::new();
        seq.extend_from_slice(&[25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        for _ in 0..3 {
            seq.extend_from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }
        seq.extend_from_slice(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let pooler = SequenceActivationPooler::default();
        let pooled_sif = pooler
            .pool_layer(&seq, 5, dim, Some(&ref_dir))
            .expect("sif pool");
        let pooled_mean =
            SequenceActivationPooler::uniform_unit_mean(&seq, 5, dim).expect("mean pool");

        // Novelty weighting must amplify the distinctive entity coordinate (`idx = 2`) well above
        // uniform mean, while the 25x magnitude sink at token 0 cannot blow up `idx = 0`.
        assert!(
            pooled_sif[2] > pooled_mean[2] * 2.5,
            "expected SIF novelty boost on idx=2: sif={} mean={}",
            pooled_sif[2],
            pooled_mean[2]
        );
        assert!(pooled_sif[2] > pooled_sif[0]);
    }

    #[test]
    fn novelty_sif_pooling_is_token_order_invariant() {
        let dim = 4;
        let ref_dir = vec![1.0_f32, 0.0, 0.0, 0.0];
        let a = [1.0_f32, 0.1, 0.0, 0.0];
        let b = [0.2_f32, 0.9, 0.3, 0.0];
        let c = [0.1_f32, 0.0, 0.0, 1.0];

        let mut fwd = Vec::new();
        fwd.extend_from_slice(&a);
        fwd.extend_from_slice(&b);
        fwd.extend_from_slice(&c);

        let mut rev = Vec::new();
        rev.extend_from_slice(&c);
        rev.extend_from_slice(&a);
        rev.extend_from_slice(&b);

        let pooler = SequenceActivationPooler::default();
        let p1 = pooler.pool_layer(&fwd, 3, dim, Some(&ref_dir)).expect("p1");
        let p2 = pooler.pool_layer(&rev, 3, dim, Some(&ref_dir)).expect("p2");
        for (x, y) in p1.iter().zip(p2.iter()) {
            assert!((x - y).abs() < 1e-6, "order variance detected: {x} vs {y}");
        }
    }

    #[test]
    fn compiled_token_corpus_recovers_diluted_entity_while_rejecting_ood() {
        let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
        let num_layers = adapter.geometry().num_layers;
        let dim = adapter.geometry().q_rot_len();

        let make_seq = |boilerplate_tokens: usize, entity_dim: usize| -> Vec<Vec<f32>> {
            let mut layer_buf = vec![0.0_f32; (boilerplate_tokens + 1) * dim];
            for t in 0..boilerplate_tokens {
                layer_buf[t * dim] = 1.0;
            }
            let e_off = boilerplate_tokens * dim;
            layer_buf[e_off + entity_dim] = 1.0;
            vec![layer_buf; num_layers]
        };

        let entries = vec![
            TokenSequenceEntry {
                role: AttractorRole::Fact,
                text: "Entity A".to_string(),
                energy: 2.0,
                seq_len: 7,
                per_layer_token_qs: make_seq(6, 1),
            },
            TokenSequenceEntry {
                role: AttractorRole::Fact,
                text: "Entity B".to_string(),
                energy: 2.0,
                seq_len: 7,
                per_layer_token_qs: make_seq(6, 2),
            },
            TokenSequenceEntry {
                role: AttractorRole::Fact,
                text: "Entity C".to_string(),
                energy: 2.0,
                seq_len: 7,
                per_layer_token_qs: make_seq(6, 3),
            },
            TokenSequenceEntry {
                role: AttractorRole::Fact,
                text: "Entity D".to_string(),
                energy: 2.0,
                seq_len: 7,
                per_layer_token_qs: make_seq(6, 4),
            },
        ];

        let compiled = CompiledTokenCorpus::compile(
            &adapter,
            &entries,
            SequenceActivationPooler::default(),
            1,
            1,
        )
        .expect("compile");

        // In-domain diluted query targeting Entity C (entity_dim = 3) recovers #3.
        let query_c = make_seq(6, 3);
        let recall_c = compiled
            .recall_token_sequence(&adapter, &query_c, 7)
            .expect("recall C");
        assert_eq!(
            recall_c.recalled_facts.first().map(|(e, _)| e.anchor_id),
            Some(3)
        );

        // Completely out-of-domain entity (entity_dim = 500) must fall in Liminal.
        let query_ood = make_seq(6, 500);
        let recall_ood = compiled
            .recall_token_sequence(&adapter, &query_ood, 7)
            .expect("recall OOD");
        assert_eq!(recall_ood.void_state, VoidBoundaryState::LiminalExterior);
        assert!(recall_ood.recalled_facts.is_empty());
    }

    #[test]
    fn stratum_signal_maps_to_core_gateway_blanket_frame() {
        use crate::{
            RecallBoundaryMetrics, RecallGateDecision, StratumProbeTelemetry, StratumSignal,
            TextAttractorEntry,
        };
        use liminika_core::action_hub::StigmergicActionDecision;

        let entries = vec![
            TextAttractorEntry {
                anchor_id: 7,
                role: AttractorRole::Fact,
                text: "Fact 7".to_string(),
                per_layer_q7: vec![[12_i8; 64]],
                energy: 2.0,
                residual_sig: vec![],
            },
            TextAttractorEntry {
                anchor_id: 101,
                role: AttractorRole::Persona,
                text: "Persona 101".to_string(),
                per_layer_q7: vec![[5_i8; 64]],
                energy: 1.5,
                residual_sig: vec![],
            },
        ];
        let signal = StratumSignal {
            candidate_scores: vec![0.45, 0.30],
            local_token_scores: vec![0.60, 0.25],
            q_to_d_scores: vec![],
            d_to_q_scores: vec![],
            token_deposits: vec![],
            linked_fact_indices: vec![],
            decomposition_triggered: false,
            local_void_depths: vec![0.0, 0.0],
            stigmergic_weights: vec![1.2, 1.0],
            boundary_metrics: RecallBoundaryMetrics {
                top1_fact_score: 0.45,
                top2_fact_score: 0.10,
                top1_persona_score: 0.30,
                max_guard_score: -0.2,
                guard_conflict_margin: -0.65,
            },
            gate_decision: RecallGateDecision {
                void_state: VoidBoundaryState::InsideBasin,
                suppress_fact_injection: false,
            },
            action_decision: StigmergicActionDecision::CrystallizeMacroBasin,
            gateway_frame: Default::default(),
            swarm_report: Default::default(),
            local_channel_preferred: false,
            telemetry: StratumProbeTelemetry::default(),
        };

        let mapped = signal.map_to_gateway(&entries, 0);
        assert_eq!(mapped.fact.map(|f| f.anchor_id), Some(7));
        assert_eq!(mapped.persona.map(|p| p.anchor_id), Some(101));
        assert_eq!(mapped.frame.target_anchor_id, 7);
    }

    #[test]
    fn compiled_token_corpus_assigns_lower_specificity_to_shared_tokens() {
        let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
        let geom = adapter.geometry();
        let dim = geom.q_rot_len();
        let num_layers = geom.num_layers;

        let make_seq_with_tokens = |token_dims: &[usize]| -> Vec<Vec<f32>> {
            let s = token_dims.len();
            let mut per_layer = vec![vec![0.0_f32; s * dim]; num_layers];
            for layer_buf in &mut per_layer {
                for (t, &d_idx) in token_dims.iter().enumerate() {
                    let pos = d_idx % dim;
                    layer_buf[t * dim + pos] = 1.0;
                    layer_buf[t * dim + ((pos + 1) % dim)] = 0.25;
                }
            }
            per_layer
        };

        // Dimension `10` is shared across 3 entries; dimension `25` appears only in entry 0.
        let e0 = TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Shared 10 and unique 25".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq_with_tokens(&[10, 25]),
        };
        let e1 = TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Shared 10 and unique 40".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq_with_tokens(&[10, 40]),
        };
        let e2 = TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Shared 10 and unique 55".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq_with_tokens(&[10, 55]),
        };
        let compiled = CompiledTokenCorpus::compile(
            &adapter,
            &[e0, e1, e2],
            SequenceActivationPooler::default(),
            2,
            1,
        )
        .expect("compile");
        assert_eq!(compiled.entry_token_weights.len(), 3);
        let min_w0 = compiled.entry_token_weights[0]
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let max_w0 = compiled.entry_token_weights[0]
            .iter()
            .copied()
            .fold(0.0_f32, f32::max);
        assert!(
            max_w0 > min_w0 * 1.25,
            "unique token must have strictly higher specificity weight than cross-entry shared token: min={min_w0} max={max_w0}"
        );
    }

    #[test]
    fn compiled_token_corpus_decomposes_bimodal_compound_query_end_to_end() {
        let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
        let geom = adapter.geometry();
        let dim = geom.q_rot_len();
        let num_layers = geom.num_layers;

        let make_seq_with_tokens = |token_dims: &[usize]| -> Vec<Vec<f32>> {
            let s = token_dims.len();
            let mut per_layer = vec![vec![0.0_f32; s * dim]; num_layers];
            for layer_buf in &mut per_layer {
                for (t, &d_idx) in token_dims.iter().enumerate() {
                    let pos = d_idx % dim;
                    layer_buf[t * dim + pos] = 1.0;
                    layer_buf[t * dim + ((pos + 2) % dim)] = 0.25;
                }
            }
            per_layer
        };

        let entries = vec![
            TokenSequenceEntry {
                role: AttractorRole::Fact,
                text: "Overview Hub (10, 12, 14)".to_string(),
                energy: 2.0,
                seq_len: 3,
                per_layer_token_qs: make_seq_with_tokens(&[10, 12, 14]),
            },
            TokenSequenceEntry {
                role: AttractorRole::Fact,
                text: "Shared Sibling (10, 20)".to_string(),
                energy: 2.0,
                seq_len: 2,
                per_layer_token_qs: make_seq_with_tokens(&[10, 20]),
            },
            TokenSequenceEntry {
                role: AttractorRole::Fact,
                text: "Specialist Alpha (35, 39)".to_string(),
                energy: 2.0,
                seq_len: 2,
                per_layer_token_qs: make_seq_with_tokens(&[35, 39]),
            },
            TokenSequenceEntry {
                role: AttractorRole::Fact,
                text: "Specialist Beta (55, 59)".to_string(),
                energy: 2.0,
                seq_len: 2,
                per_layer_token_qs: make_seq_with_tokens(&[55, 59]),
            },
            TokenSequenceEntry {
                role: AttractorRole::Guard,
                text: "Forbidden Guard (75, 79)".to_string(),
                energy: 2.0,
                seq_len: 2,
                per_layer_token_qs: make_seq_with_tokens(&[75, 79]),
            },
        ];

        let compiled = CompiledTokenCorpus::compile(
            &adapter,
            &entries,
            SequenceActivationPooler::default(),
            2,
            1,
        )
        .expect("compile");

        // Query asks about shared domain framing (10, 10) plus both Alpha (35) and Beta (55).
        let q_seq = make_seq_with_tokens(&[10, 10, 35, 55]);
        let recall = compiled
            .recall_token_sequence(&adapter, &q_seq, 4)
            .expect("bimodal recall");
        let recalled_ids: Vec<u32> = recall
            .recalled_facts
            .iter()
            .map(|(e, _)| e.anchor_id)
            .collect();
        assert_eq!(recalled_ids.len(), 2, "got {recalled_ids:?}");
        assert!(
            recalled_ids.contains(&3) && recalled_ids.contains(&4),
            "expected Alpha(#3) and Beta(#4), got {recalled_ids:?}"
        );
    }

    #[test]
    fn model_base_attractor_profile_partitions_sub_basins_and_orthogonalizes_corpus_prior() {
        let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
        let geom = adapter.geometry();
        let dim = geom.q_rot_len();
        let num_layers = geom.num_layers;

        let make_seq = |tok_specs: &[(usize, f32)]| -> Vec<Vec<f32>> {
            let s = tok_specs.len();
            let mut per_layer = vec![vec![0.0_f32; s * dim]; num_layers];
            for layer_buf in &mut per_layer {
                for (t, &(d_idx, framing_strength)) in tok_specs.iter().enumerate() {
                    layer_buf[t * dim] = framing_strength;
                    let pos = d_idx % dim;
                    layer_buf[t * dim + pos] = 1.0;
                }
            }
            per_layer
        };

        // Six baseline unsteered samples spanning two distinct model regimes (`20` and `60`)
        // plus shared model framing along dimension `0`.
        let base_samples = vec![
            TokenSequenceEntry {
                role: AttractorRole::Persona,
                text: "Base regime A sample 1".to_string(),
                energy: 1.0,
                seq_len: 2,
                per_layer_token_qs: make_seq(&[(20, 0.85), (21, 0.70)]),
            },
            TokenSequenceEntry {
                role: AttractorRole::Persona,
                text: "Base regime A sample 2".to_string(),
                energy: 1.0,
                seq_len: 2,
                per_layer_token_qs: make_seq(&[(20, 0.80), (21, 0.65)]),
            },
            TokenSequenceEntry {
                role: AttractorRole::Persona,
                text: "Base regime A sample 3".to_string(),
                energy: 1.0,
                seq_len: 2,
                per_layer_token_qs: make_seq(&[(20, 0.90), (21, 0.75)]),
            },
            TokenSequenceEntry {
                role: AttractorRole::Persona,
                text: "Base regime B sample 1".to_string(),
                energy: 1.0,
                seq_len: 2,
                per_layer_token_qs: make_seq(&[(60, 0.85), (61, 0.70)]),
            },
            TokenSequenceEntry {
                role: AttractorRole::Persona,
                text: "Base regime B sample 2".to_string(),
                energy: 1.0,
                seq_len: 2,
                per_layer_token_qs: make_seq(&[(60, 0.80), (61, 0.65)]),
            },
            TokenSequenceEntry {
                role: AttractorRole::Persona,
                text: "Base regime B sample 3".to_string(),
                energy: 1.0,
                seq_len: 2,
                per_layer_token_qs: make_seq(&[(60, 0.90), (61, 0.75)]),
            },
        ];

        let prior = ModelBaseAttractorProfile::extract(
            &adapter,
            &base_samples,
            SequenceActivationPooler::default(),
            0.50,
        )
        .expect("extract prior");
        assert_eq!(prior.sample_count, 6);
        assert_eq!(
            prior.sub_attractors.len(),
            2,
            "expected two distinct base sub-attractors, got {}",
            prior.sub_attractors.len()
        );
        assert!(
            !prior.pooled_common_mode_basis.is_empty(),
            "expected at least one global base-model framing component"
        );

        // Compile external facts with the base-model prior and verify recall selects target #2.
        let facts = vec![
            TokenSequenceEntry {
                role: AttractorRole::Fact,
                text: "Fact Alpha".to_string(),
                energy: 2.0,
                seq_len: 2,
                per_layer_token_qs: make_seq(&[(30, 0.85), (31, 0.65)]),
            },
            TokenSequenceEntry {
                role: AttractorRole::Fact,
                text: "Fact Beta".to_string(),
                energy: 2.0,
                seq_len: 2,
                per_layer_token_qs: make_seq(&[(80, 0.85), (81, 0.65)]),
            },
        ];
        let compiled = CompiledTokenCorpus::compile_with_base_prior(
            &adapter,
            &facts,
            SequenceActivationPooler::default(),
            2,
            1,
            Some(&prior),
        )
        .expect("compile with base prior");
        let query = make_seq(&[(80, 0.90), (81, 0.75)]);
        let recall = compiled
            .recall_token_sequence(&adapter, &query, 2)
            .expect("recall with base prior");
        assert_eq!(
            recall.recalled_facts.first().map(|(e, _)| e.anchor_id),
            Some(2)
        );
    }

    #[test]
    fn register_bpe_prefix_alias_tokens_preserves_pooled_sigs_and_enables_split_token_recall() {
        let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
        let num_layers = adapter.geometry().num_layers;
        let dim = adapter.geometry().q_rot_len();
        let make_seq = |toks: &[(usize, f32)]| -> Vec<Vec<f32>> {
            let mut per_layer = Vec::with_capacity(num_layers);
            for _ in 0..num_layers {
                let mut flat = vec![0.0_f32; toks.len() * dim];
                for (t, &(slot, amp)) in toks.iter().enumerate() {
                    flat[t * dim + slot] = amp;
                    flat[t * dim + (slot + 1) % dim] = 0.20;
                }
                per_layer.push(flat);
            }
            per_layer
        };

        let entries = vec![
            TokenSequenceEntry {
                role: AttractorRole::Fact,
                text: "Overview Fact".to_string(),
                energy: 2.0,
                seq_len: 2,
                per_layer_token_qs: make_seq(&[(10, 0.9), (12, 0.7)]),
            },
            TokenSequenceEntry {
                role: AttractorRole::Fact,
                text: "Initial-split Fact".to_string(),
                energy: 2.0,
                seq_len: 2,
                per_layer_token_qs: make_seq(&[(40, 0.9), (12, 0.7)]),
            },
        ];

        let mut compiled = CompiledTokenCorpus::compile(
            &adapter,
            &entries,
            SequenceActivationPooler::default(),
            2,
            1,
        )
        .expect("compile");
        let before_sig = compiled.corpus.entries()[1].residual_sig.clone();

        // Register space-prefixed alias tokens at orthogonal dimensions (70, 72) for entry 1.
        let alias_qs = make_seq(&[(70, 0.95), (72, 0.90)]);
        compiled
            .register_bpe_prefix_alias_tokens(&adapter, 1, &alias_qs, 2)
            .expect("register alias");
        assert_eq!(compiled.corpus.entries()[1].residual_sig, before_sig);
        assert_eq!(compiled.entry_token_sigs[1].len(), 4);

        let err = compiled.register_bpe_prefix_alias_tokens(&adapter, 9, &alias_qs, 2);
        assert!(err.is_err());

        let obs = compiled
            .build_local_observation(&adapter, &alias_qs, 2)
            .expect("obs");
        assert!(
            obs.q_to_d_scores[1] > obs.q_to_d_scores[0],
            "expected entry 1 alias match ({}) > entry 0 ({})",
            obs.q_to_d_scores[1],
            obs.q_to_d_scores[0]
        );
    }

    #[test]
    fn same_role_token_specificity_ignores_contrastive_guard_echoes() {
        let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
        let num_layers = adapter.geometry().num_layers;
        let dim = adapter.geometry().q_rot_len();
        let make_seq = |toks: &[(usize, f32)]| -> Vec<Vec<f32>> {
            let mut per_layer = Vec::with_capacity(num_layers);
            for _ in 0..num_layers {
                let mut flat = vec![0.0_f32; toks.len() * dim];
                for (t, &(slot, amp)) in toks.iter().enumerate() {
                    flat[t * dim + slot] = amp;
                    flat[t * dim + (slot + 1) % dim] = 0.15;
                }
                per_layer.push(flat);
            }
            per_layer
        };

        // Entry 0 (Fact) and Entry 2 (Guard) share entity token slot 30, while Entry 1 (Fact)
        // shares token slot 50 with nothing else; Guard echo must not dilute Entry 0's specificity.
        let entries = vec![
            TokenSequenceEntry {
                role: AttractorRole::Fact,
                text: "Entity Fact".to_string(),
                energy: 2.0,
                seq_len: 2,
                per_layer_token_qs: make_seq(&[(30, 0.95), (70, 0.85)]),
            },
            TokenSequenceEntry {
                role: AttractorRole::Fact,
                text: "Other Fact".to_string(),
                energy: 2.0,
                seq_len: 2,
                per_layer_token_qs: make_seq(&[(50, 0.95), (90, 0.85)]),
            },
            TokenSequenceEntry {
                role: AttractorRole::Guard,
                text: "Contrastive Guard echoing entity".to_string(),
                energy: 2.0,
                seq_len: 2,
                per_layer_token_qs: make_seq(&[(30, 0.95), (110, 0.85)]),
            },
        ];

        let compiled = CompiledTokenCorpus::compile(
            &adapter,
            &entries,
            SequenceActivationPooler::default(),
            2,
            1,
        )
        .expect("compile");
        assert!(
            compiled.entry_token_weights[0].iter().all(|&w| w > 0.85),
            "Guard echo should not penalize Fact token specificity: {:?}",
            compiled.entry_token_weights[0]
        );
    }
}
