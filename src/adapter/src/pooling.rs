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
    AttractorRole, DynamicAttractorRecall, ModelSteeringAdapter, RecallPolicy, StratumPipeline,
    TextAttractorCorpus,
};
use liminika_core::traits::gateway::ManifoldProbeBaseline;
use liminika_core::traits::primitives::MAX_ATTENTION_HEADS;

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

    fn extract_salient_token_sigs(
        adapter: &ModelSteeringAdapter,
        per_layer_token_qs: &[Vec<f32>],
        seq_len: usize,
        novelty_reference_per_layer: &[Vec<f32>],
        token_common_mode_basis: &[Vec<f32>],
        max_tokens: usize,
        pooler: SequenceActivationPooler,
    ) -> Result<Vec<(Vec<f32>, f32)>, String> {
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
            out.push((proj_tok_sig, w));
        }
        Ok(out)
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
        if entries.is_empty() {
            return Err("cannot compile an empty TokenSequenceEntry slice".to_string());
        }
        let num_layers = adapter.geometry().num_layers;
        let dim = adapter.geometry().q_rot_len();
        let n = entries.len() as f32;

        // Pass 1: unsupervised uniform centroid over unit-normalized token vectors.
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
        let token_common_mode_basis =
            Self::fit_cross_entry_token_common_mode(&per_entry_raw_toks, 10);

        let mut entry_token_sigs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(entries.len());
        for entry in entries {
            let salient = Self::extract_salient_token_sigs(
                adapter,
                &entry.per_layer_token_qs,
                entry.seq_len,
                &novelty_reference_per_layer,
                &token_common_mode_basis,
                24,
                pooler,
            )?;
            entry_token_sigs.push(salient.into_iter().map(|(sig, _)| sig).collect());
        }

        // Estimate cross-entry doc-token specificity weights (`1 / (1 + sum_{j!=i} max_sim_+^2)`)
        // and leave-one-out empirical MaxSim hubness baseline per entry.
        let n_entries = entry_token_sigs.len();
        let mut entry_token_weights: Vec<Vec<f32>> = Vec::with_capacity(n_entries);
        for i in 0..n_entries {
            let mut weights_i = Vec::with_capacity(entry_token_sigs[i].len());
            for d_tok in &entry_token_sigs[i] {
                let mut other_sim_sq = 0.0_f32;
                for (j, other_toks) in entry_token_sigs.iter().enumerate() {
                    if i == j || other_toks.is_empty() {
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
        self.recall_token_sequence_with_strata(
            adapter,
            query_token_qs,
            query_seq_len,
            &StratumPipeline::default(),
        )
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
        let query_salient = Self::extract_salient_token_sigs(
            adapter,
            query_token_qs,
            query_seq_len,
            &self.novelty_reference_per_layer,
            &self.token_common_mode_basis,
            24,
            self.pooler,
        )?;
        let n_docs = self.entry_token_sigs.len();
        let mut local_token_scores = vec![0.0_f32; n_docs];
        if n_docs > 0 && !query_salient.is_empty() {
            // Unsupervised token-level discriminative centering (ColBERT-IDF equivalent):
            // down-weight query tokens that respond uniformly across all corpus entries and
            // center each token's MaxSim row around its corpus-wide mean before aggregation.
            let mut weighted_sums = vec![0.0_f32; n_docs];
            let mut total_disc_weight = 0.0_f32;
            let mut raw_weighted_sums = vec![0.0_f32; n_docs];
            let mut total_raw_weight = 0.0_f32;

            for (q_tok, w_t) in &query_salient {
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
                    for (dst, &s) in weighted_sums.iter_mut().zip(row_sims.iter()) {
                        *dst += eff_w * (s - mean_row);
                    }
                    total_disc_weight += eff_w;
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
                        .map(|(q_tok, _)| TextAttractorCorpus::cosine_f32(d_tok, q_tok))
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
                    local_token_scores[i] = 0.65 * q_to_d_score + 0.35 * d_to_q_centered;
                }
            } else if total_raw_weight > 1e-12 {
                for (i, &ws) in raw_weighted_sums.iter().enumerate() {
                    let raw_maxsim = ws / total_raw_weight;
                    let hub_shift = self.entry_token_hub_bias.get(i).copied().unwrap_or(0.0)
                        - self.global_token_hub_mean;
                    local_token_scores[i] = raw_maxsim - 0.65 * hub_shift;
                }
            }
        }

        let q_refs: Vec<&[f32]> = pooled_q.iter().map(Vec::as_slice).collect();
        let centroid_refs: Vec<&[f32]> = self
            .corpus_centroid_per_layer
            .iter()
            .map(Vec::as_slice)
            .collect();
        self.corpus.recall_for_query_with_strata_and_local_scores(
            adapter,
            &q_refs,
            &centroid_refs,
            &self.recall_policy,
            pipeline,
            Some(&local_token_scores),
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

        // Completely out-of-domain entity (entity_dim = 500) must fall in CvoidEmptySpace.
        let query_ood = make_seq(6, 500);
        let recall_ood = compiled
            .recall_token_sequence(&adapter, &query_ood, 7)
            .expect("recall OOD");
        assert_eq!(recall_ood.void_state, VoidBoundaryState::CvoidEmptySpace);
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
}
