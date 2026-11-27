// name: src/adapter/src/pooling/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model-agnostic unsupervised sequence-token pooling and two-pass corpus compilation.
//!
//! External LLM runtimes (Candle, `llama.cpp`, ONNX) supply raw per-token, per-layer pre-RoPE Q
//! activations (`[seq_len, q_rot_len]`) to [`CompiledTokenCorpus`]. Keeping sequence aggregation
//! inside `liminika-adapter` rather than in example binaries ensures identical geometric pooling,
//! BOS/attention-sink normalization, and zero-overfitting unsupervised novelty weighting across all
//! backends and CLI entrypoints.

mod base_prior;
mod calibrator;
mod encoder;
mod null_calibration;
mod streaming;
#[cfg(test)]
mod tests;

pub use base_prior::ModelBaseAttractorProfile;
pub use calibrator::{
    AdaptiveTokenSpecificityCalibrator, EXACT_TOKEN_STATS_LIMIT, TokenCalibrationStats,
    TokenSpecificityCalibrator,
};
pub use encoder::{
    CapturedTokenSequence, ContrastiveAnchorBank, ContrastiveAnchorEntry,
    DEFAULT_STARTUP_PRIOR_PROBES, PooledSubtitleEncoder, PrefixAliasProbe, TokenSequenceEncoder,
    extract_leading_alias_word,
};
pub use null_calibration::{
    DEFAULT_TOKEN_EXPLAIN_THRESHOLD, NullCalibrationEnvelope, OOD_COH_MAD_FLOOR,
    OOD_UNEXP_MAD_FLOOR,
};
pub use streaming::{
    DEFAULT_STREAMING_SALIENT_TOKENS, STREAMING_PCA_SAMPLE_BUDGET, StreamingCorpusDistiller,
    StreamingDistillConfig, StreamingDistillReport, split_raw_document_chunks,
};

use crate::{
    AttractorRole, DynamicAttractorRecall, LacunaBoundaryState, LocalChannelObservation,
    ModelSteeringAdapter, RecallPolicy, StratumPipeline, StratumProbeTelemetry,
    TextAttractorCorpus, TokenResonanceDeposit,
};
use base_prior::{
    fit_cross_entry_token_common_mode_with_scripts, merge_orthonormal_basis,
    token_orthogonal_residual_norm,
};
use liminika_core::foundation::load_q7_words_64;
use liminika_core::liminal::{LacunaPartition8, MAX_ATTENTION_HEADS, NUM_STIGMERGIC_GRANULES};
use liminika_core::runtime::{GatewayFrame, ManifoldProbeBaseline};
use liminika_core::stigmergy::StigmergicActionDecision;
use null_calibration::{NullCalibrationContext, calibrate_null_envelope_2fold};

/// Validates an optional per-token inclusion mask against `seq_len` (fail-closed on mismatch) and
/// returns the count of included (`true`) tokens.
pub(crate) fn validate_token_include_mask(
    seq_len: usize,
    token_include_mask: Option<&[bool]>,
) -> Result<usize, String> {
    if let Some(mask) = token_include_mask {
        if mask.len() != seq_len {
            return Err(format!(
                "token_include_mask length mismatch: expected {seq_len}, got {}",
                mask.len()
            ));
        }
        Ok(mask.iter().filter(|&&keep| keep).count())
    } else {
        Ok(seq_len)
    }
}

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
        Self::uniform_raw_mean_masked(token_flat, seq_len, dim, None)
    }

    /// Computes uniform average over raw (unnormalized) token vectors for one layer, completely
    /// excluding tokens where `token_include_mask[t] == false`.
    ///
    /// When `token_include_mask` is `Some` and masks out every token, returns a zero vector
    /// `[0.0; dim]` without division by zero or NaN.
    ///
    /// # Errors
    /// Returns an error if `seq_len == 0`, `dim == 0`, `token_flat.len() != seq_len * dim`, or
    /// `token_include_mask` length disagrees with `seq_len`.
    pub fn uniform_raw_mean_masked(
        token_flat: &[f32],
        seq_len: usize,
        dim: usize,
        token_include_mask: Option<&[bool]>,
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
        let included = validate_token_include_mask(seq_len, token_include_mask)?;
        if included == 0 {
            return Ok(vec![0.0_f32; dim]);
        }
        let inv = 1.0 / (included as f32);
        let mut out = vec![0.0_f32; dim];
        for t in 0..seq_len {
            if token_include_mask.is_some_and(|m| !m[t]) {
                continue;
            }
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
        Self::uniform_unit_mean_masked(token_flat, seq_len, dim, None)
    }

    /// Computes uniform average over unit-normalized token vectors for one layer, completely
    /// excluding tokens where `token_include_mask[t] == false`.
    ///
    /// # Errors
    /// Returns an error if `seq_len == 0`, `dim == 0`, `token_flat.len() != seq_len * dim`, or
    /// `token_include_mask` length disagrees with `seq_len`.
    pub fn uniform_unit_mean_masked(
        token_flat: &[f32],
        seq_len: usize,
        dim: usize,
        token_include_mask: Option<&[bool]>,
    ) -> Result<Vec<f32>, String> {
        Self::pool_layer_with_mode(
            SequencePoolingMode::UniformMean,
            token_flat,
            seq_len,
            dim,
            None,
            token_include_mask,
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
        Self::pool_layer_with_mode(self.mode, token_flat, seq_len, dim, reference, None)
    }

    /// Pools one layer's flattened `[seq_len, dim]` activations into `[dim]` using `self.mode`
    /// while excluding any token with `token_include_mask[t] == false`.
    ///
    /// When `token_include_mask` excludes all tokens, returns a zero vector `[0.0; dim]` without
    /// NaN or division by zero.
    ///
    /// # Errors
    /// Returns an error on empty shapes, mismatched slice/mask length, or invalid reference length.
    pub fn pool_layer_masked(
        &self,
        token_flat: &[f32],
        seq_len: usize,
        dim: usize,
        reference: Option<&[f32]>,
        token_include_mask: Option<&[bool]>,
    ) -> Result<Vec<f32>, String> {
        Self::pool_layer_with_mode(
            self.mode,
            token_flat,
            seq_len,
            dim,
            reference,
            token_include_mask,
        )
    }

    fn pool_layer_with_mode(
        mode: SequencePoolingMode,
        token_flat: &[f32],
        seq_len: usize,
        dim: usize,
        reference: Option<&[f32]>,
        token_include_mask: Option<&[bool]>,
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
        if let Some(r) = reference
            && r.len() != dim
        {
            return Err(format!(
                "reference dimension mismatch: expected {dim}, got {}",
                r.len()
            ));
        }
        let included = validate_token_include_mask(seq_len, token_include_mask)?;
        if included == 0 {
            return Ok(vec![0.0_f32; dim]);
        }

        // Two-pass streaming over included tokens avoids materializing `[seq_len][dim]` unit copies
        // while preserving exact per-token normalization rounding order (`(w * inv_sum) * (src * inv)`).
        let mut active: Vec<(usize, f32, f32)> = Vec::with_capacity(included);
        let mut weight_sum = 0.0_f64;
        for t in 0..seq_len {
            if token_include_mask.is_some_and(|m| !m[t]) {
                continue;
            }
            let slice = &token_flat[t * dim..(t + 1) * dim];
            let norm_sq: f64 = slice.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            let norm = norm_sq.sqrt();
            let inv = if norm > 1e-12 {
                (1.0 / norm) as f32
            } else {
                0.0_f32
            };
            let w = match (mode, reference) {
                (SequencePoolingMode::UniformMean, _) | (_, None) => 1.0_f32,
                (
                    SequencePoolingMode::CorpusNoveltySif {
                        novelty_power,
                        weight_floor,
                    },
                    Some(ref_vec),
                ) => {
                    let cos_pos = if inv > 0.0 {
                        TextAttractorCorpus::cosine_f32(slice, ref_vec).max(0.0)
                    } else {
                        0.0
                    };
                    let floor = weight_floor.clamp(1e-3, 1.0);
                    let novelty = (1.0 - cos_pos).clamp(floor, 1.0);
                    novelty.powf(novelty_power.max(0.0))
                }
            };
            weight_sum += f64::from(w);
            active.push((t, w, inv));
        }

        let inv_sum = if weight_sum > 1e-12 {
            (1.0 / weight_sum) as f32
        } else {
            1.0 / (included as f32)
        };
        let mut out = vec![0.0_f32; dim];
        for (t, w, inv) in active {
            if inv == 0.0 {
                continue;
            }
            let scale = w * inv_sum;
            let slice = &token_flat[t * dim..(t + 1) * dim];
            for (dst, &src) in out.iter_mut().zip(slice.iter()) {
                *dst += scale * (src * inv);
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
        self.pool_sequence_masked(per_layer_token_qs, seq_len, dim, ref_per_layer, None)
    }

    /// Pools every transformer layer's `[seq_len, dim]` activations into `[num_layers][dim]`
    /// while excluding tokens where `token_include_mask[t] == false`.
    ///
    /// # Errors
    /// Returns an error if any layer buffer, optional reference slice, or mask length mismatches.
    pub fn pool_sequence_masked(
        &self,
        per_layer_token_qs: &[Vec<f32>],
        seq_len: usize,
        dim: usize,
        ref_per_layer: Option<&[Vec<f32>]>,
        token_include_mask: Option<&[bool]>,
    ) -> Result<Vec<Vec<f32>>, String> {
        let _ = validate_token_include_mask(seq_len, token_include_mask)?;
        if let Some(refs) = ref_per_layer
            && refs.len() != per_layer_token_qs.len()
        {
            return Err(format!(
                "reference layer count mismatch: {} vs {}",
                refs.len(),
                per_layer_token_qs.len()
            ));
        }
        let mut out = Vec::with_capacity(per_layer_token_qs.len());
        for (l, token_flat) in per_layer_token_qs.iter().enumerate() {
            let ref_slice = ref_per_layer
                .and_then(|refs| refs.get(l))
                .map(Vec::as_slice);
            out.push(self.pool_layer_masked(
                token_flat,
                seq_len,
                dim,
                ref_slice,
                token_include_mask,
            )?);
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
    /// Optional per-token inclusion mask (`len == seq_len`, `true` = include, `false` = exclude).
    pub token_include_mask: Option<Vec<bool>>,
}

/// Auxiliary per-token Q activation observation to append into an existing compiled corpus entry.
#[derive(Debug, Clone)]
pub struct EntryTokenObservation {
    /// Target corpus entry index inside [`CompiledTokenCorpus::entry_token_sigs`].
    pub entry_idx: usize,
    /// Number of observed tokens (`S`).
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
    /// Empirically calibrated [`RecallPolicy`] derived from projected corpus similarities and
    /// optional out-of-sample base-prior null calibration.
    pub recall_policy: RecallPolicy,
    /// Optional non-leaking 2-fold OOD null calibration envelope derived from `base_prior`.
    pub null_calibration: Option<NullCalibrationEnvelope>,
    /// Token cosine threshold below which a salient query token counts as unexplained by corpus Facts.
    pub(crate) token_explain_threshold: f32,
    /// Pooler configuration applied identically during compilation and query recall.
    pub pooler: SequenceActivationPooler,
}

impl CompiledTokenCorpus {
    pub(super) fn fit_cross_entry_token_common_mode(
        per_entry_tok_sigs: &[Vec<Vec<f32>>],
        max_components: usize,
    ) -> Vec<Vec<f32>> {
        fit_cross_entry_token_common_mode_with_scripts(per_entry_tok_sigs, None, max_components)
    }

    pub(super) fn token_orthogonal_residual_norm(raw_tok_sig: &[f32], basis: &[Vec<f32>]) -> f32 {
        token_orthogonal_residual_norm(raw_tok_sig, basis)
    }

    pub(super) fn merge_orthonormal_basis(
        base: &[Vec<f32>],
        extra: &[Vec<f32>],
        max_total: usize,
    ) -> Vec<Vec<f32>> {
        merge_orthonormal_basis(base, extra, max_total)
    }

    /// Builds raw steered residual signatures `(orig_token_idx, sig)` strictly for included tokens
    /// (`token_include_mask[t] == true`, or all tokens when `None`).
    pub(super) fn build_raw_token_sigs_for_entry(
        adapter: &ModelSteeringAdapter,
        per_layer_token_qs: &[Vec<f32>],
        seq_len: usize,
        novelty_reference_per_layer: &[Vec<f32>],
        token_include_mask: Option<&[bool]>,
    ) -> Result<Vec<(usize, Vec<f32>)>, String> {
        let included = validate_token_include_mask(seq_len, token_include_mask)?;
        let geom = adapter.geometry();
        let dim = geom.q_rot_len();
        let ref_slices: Vec<&[f32]> = novelty_reference_per_layer
            .iter()
            .map(Vec::as_slice)
            .collect();
        let mut out = Vec::with_capacity(included);
        let mut unit_tok_layers: Vec<Vec<f32>> = vec![vec![0.0_f32; dim]; per_layer_token_qs.len()];
        for t in 0..seq_len {
            if token_include_mask.is_some_and(|m| !m[t]) {
                continue;
            }
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
            out.push((
                t,
                TextAttractorCorpus::build_steered_residual_sig(geom, &tok_refs, &ref_slices),
            ));
        }
        Ok(out)
    }

    fn extract_salient_token_sigs_from_raw(
        adapter: &ModelSteeringAdapter,
        per_layer_token_qs: &[Vec<f32>],
        raw_tok_sigs: &[(usize, Vec<f32>)],
        novelty_reference_per_layer: &[Vec<f32>],
        token_common_mode_basis: &[Vec<f32>],
        max_tokens: usize,
        pooler: SequenceActivationPooler,
    ) -> Vec<(usize, Vec<f32>, f32)> {
        let geom = adapter.geometry();
        let dim = geom.q_rot_len();
        let n_active = raw_tok_sigs.len();
        if n_active == 0 || max_tokens == 0 {
            return Vec::new();
        }
        let mut token_scores: Vec<(usize, usize, f32)> = Vec::with_capacity(n_active);
        let (power, floor) = match pooler.mode {
            SequencePoolingMode::UniformMean => (0.0_f32, 1.0_f32),
            SequencePoolingMode::CorpusNoveltySif {
                novelty_power,
                weight_floor,
            } => (novelty_power.max(0.0), weight_floor.clamp(1e-3, 1.0)),
        };

        for (raw_pos, (orig_t, raw_sig)) in raw_tok_sigs.iter().enumerate() {
            let mut sum_w = 0.0_f32;
            let mut count = 0_usize;
            for (l, layer_flat) in per_layer_token_qs.iter().enumerate() {
                if !geom.is_rope_layer(l) {
                    continue;
                }
                let start = orig_t * dim;
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
            token_scores.push((raw_pos, *orig_t, salience));
        }

        token_scores.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.1.cmp(&b.1)));
        token_scores.truncate(max_tokens.min(n_active));

        let mut out = Vec::with_capacity(token_scores.len());
        for (raw_pos, orig_t, w) in token_scores {
            let proj_tok_sig = TextAttractorCorpus::project_and_normalize(
                &raw_tok_sigs[raw_pos].1,
                token_common_mode_basis,
            );
            out.push((orig_t, proj_tok_sig, w));
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn extract_salient_token_sigs_indexed_masked(
        adapter: &ModelSteeringAdapter,
        per_layer_token_qs: &[Vec<f32>],
        seq_len: usize,
        novelty_reference_per_layer: &[Vec<f32>],
        token_common_mode_basis: &[Vec<f32>],
        max_tokens: usize,
        pooler: SequenceActivationPooler,
        token_include_mask: Option<&[bool]>,
    ) -> Result<Vec<(usize, Vec<f32>, f32)>, String> {
        let included = validate_token_include_mask(seq_len, token_include_mask)?;
        if included == 0 || max_tokens == 0 {
            return Ok(Vec::new());
        }
        let raw_tok_sigs = Self::build_raw_token_sigs_for_entry(
            adapter,
            per_layer_token_qs,
            seq_len,
            novelty_reference_per_layer,
            token_include_mask,
        )?;
        Ok(Self::extract_salient_token_sigs_from_raw(
            adapter,
            per_layer_token_qs,
            &raw_tok_sigs,
            novelty_reference_per_layer,
            token_common_mode_basis,
            max_tokens,
            pooler,
        ))
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
    /// Returns an error if `entries` is empty or any layer tensor shape or mask length disagrees
    /// with `adapter`.
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

        // Pass 1: unsupervised uniform centroid over unit-normalized included token vectors.
        // Keeping the SIF novelty reference pure to the corpus preserves sharp in-domain Fact and
        // Guard separation; `base_prior` is applied strictly to out-of-sample null calibration and
        // common-mode subspace projection without shifting the corpus centroid.
        let mut novelty_reference_per_layer = vec![vec![0.0_f32; dim]; num_layers];
        for entry in entries {
            if entry.per_layer_token_qs.len() != num_layers {
                return Err(format!(
                    "entry layer count mismatch: expected {num_layers}, got {}",
                    entry.per_layer_token_qs.len()
                ));
            }
            let mask = entry.token_include_mask.as_deref();
            for (l, flat) in entry.per_layer_token_qs.iter().enumerate() {
                let u = SequenceActivationPooler::uniform_unit_mean_masked(
                    flat,
                    entry.seq_len,
                    dim,
                    mask,
                )?;
                for (dst, &src) in novelty_reference_per_layer[l].iter_mut().zip(u.iter()) {
                    *dst += src / n;
                }
            }
        }

        // Pass 2: pool every entry against Pass-1 reference and compute matched centroid, plus a
        // non-Guard (`Fact` + `Persona`) centroid for Guard Procrustes baking so Guard ratio
        // growth never shrinks the anti-phase repulsion angle.
        let mut pooled_entries: Vec<Vec<Vec<f32>>> = Vec::with_capacity(entries.len());
        let mut corpus_centroid_per_layer = vec![vec![0.0_f32; dim]; num_layers];
        let mut non_guard_sum_per_layer = vec![vec![0.0_f32; dim]; num_layers];
        let mut non_guard_count = 0_usize;
        for entry in entries {
            let pooled = pooler.pool_sequence_masked(
                &entry.per_layer_token_qs,
                entry.seq_len,
                dim,
                Some(&novelty_reference_per_layer),
                entry.token_include_mask.as_deref(),
            )?;
            let is_non_guard = entry.role != AttractorRole::Guard;
            if is_non_guard {
                non_guard_count += 1;
            }
            for (l, vec_l) in pooled.iter().enumerate() {
                for (dst, &src) in corpus_centroid_per_layer[l].iter_mut().zip(vec_l.iter()) {
                    *dst += src / n;
                }
                if is_non_guard {
                    for (dst, &src) in non_guard_sum_per_layer[l].iter_mut().zip(vec_l.iter()) {
                        *dst += src;
                    }
                }
            }
            pooled_entries.push(pooled);
        }
        let guard_phase_centroid_per_layer = if non_guard_count > 0 {
            let inv_ng = 1.0_f32 / (non_guard_count as f32);
            for layer in &mut non_guard_sum_per_layer {
                for v in layer.iter_mut() {
                    *v *= inv_ng;
                }
            }
            non_guard_sum_per_layer
        } else {
            corpus_centroid_per_layer.clone()
        };

        let centroid_refs: Vec<&[f32]> = corpus_centroid_per_layer
            .iter()
            .map(Vec::as_slice)
            .collect();
        let guard_phase_refs: Vec<&[f32]> = guard_phase_centroid_per_layer
            .iter()
            .map(Vec::as_slice)
            .collect();
        let mut corpus = TextAttractorCorpus::new();
        for (entry, pooled) in entries.iter().zip(pooled_entries.iter()) {
            let q_refs: Vec<&[f32]> = pooled.iter().map(Vec::as_slice).collect();
            let phase_refs = if entry.role == AttractorRole::Guard {
                &guard_phase_refs
            } else {
                &centroid_refs
            };
            corpus.ingest_text_activations_with_phase_ref(
                adapter,
                entry.role,
                entry.text.as_str(),
                &q_refs,
                &centroid_refs,
                phase_refs,
                entry.energy,
            )?;
        }
        corpus.fit_common_mode_subspace(3);
        let corpus_pre_prior = corpus.clone();
        if let Some(prior) = base_prior {
            corpus.extend_common_mode_basis(&prior.pooled_common_mode_basis, 3);
        }
        let mut recall_policy = corpus.calibrate_recall_policy(top_k_facts, top_k_personas);
        let corpus_offdiag_floor = recall_policy.min_coherence;

        // Estimate unsupervised token-space common mode across included entry tokens only.
        let mut per_entry_raw_toks: Vec<Vec<(usize, Vec<f32>)>> = Vec::with_capacity(entries.len());
        let mut per_entry_tok_vecs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(entries.len());
        for entry in entries {
            let raw_indexed = Self::build_raw_token_sigs_for_entry(
                adapter,
                &entry.per_layer_token_qs,
                entry.seq_len,
                &novelty_reference_per_layer,
                entry.token_include_mask.as_deref(),
            )?;
            per_entry_tok_vecs.push(raw_indexed.iter().map(|(_, s)| s.clone()).collect());
            per_entry_raw_toks.push(raw_indexed);
        }
        let corpus_tok_basis = Self::fit_cross_entry_token_common_mode(&per_entry_tok_vecs, 10);
        let token_common_mode_basis = if let Some(prior) = base_prior {
            Self::merge_orthonormal_basis(&corpus_tok_basis, &prior.token_common_mode_basis, 12)
        } else {
            corpus_tok_basis.clone()
        };

        let extract_all_entries = |basis: &[Vec<f32>]| -> Vec<Vec<Vec<f32>>> {
            entries
                .iter()
                .zip(per_entry_raw_toks.iter())
                .map(|(entry, raw_toks)| {
                    Self::extract_salient_token_sigs_from_raw(
                        adapter,
                        &entry.per_layer_token_qs,
                        raw_toks,
                        &novelty_reference_per_layer,
                        basis,
                        24,
                        pooler,
                    )
                    .into_iter()
                    .map(|(_, sig, _)| sig)
                    .collect()
                })
                .collect()
        };

        let entry_token_sigs = extract_all_entries(&token_common_mode_basis);

        let TokenCalibrationStats {
            entry_token_weights,
            entry_token_hub_bias,
            global_token_hub_mean,
        } = AdaptiveTokenSpecificityCalibrator::default()
            .calibrate_token_stats(&entry_token_sigs, corpus.entries());

        // Non-leaking 2-fold cross-fitted OOD null calibration with script-group bimodal defense.
        let mut null_calibration: Option<NullCalibrationEnvelope> = None;
        let mut token_explain_threshold = 0.40_f32;
        if let Some(prior) = base_prior
            && let Some((env, tok_thresh)) = calibrate_null_envelope_2fold(
                &NullCalibrationContext {
                    adapter,
                    prior,
                    corpus_pre_prior: &corpus_pre_prior,
                    corpus_tok_basis: &corpus_tok_basis,
                    novelty_reference_per_layer: &novelty_reference_per_layer,
                    corpus_centroid_per_layer: &corpus_centroid_per_layer,
                    recall_policy,
                    corpus_offdiag_floor,
                    pooler,
                },
                extract_all_entries,
            )
        {
            recall_policy.min_coherence = env.calibrated_min_coherence;
            token_explain_threshold = tok_thresh;
            corpus.set_null_calibration(Some(env));
            null_calibration = Some(env);
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
            null_calibration,
            token_explain_threshold,
            pooler,
        })
    }

    /// Recomputes token specificity weights and empirical hubness baselines using a caller-supplied
    /// [`TokenSpecificityCalibrator`].
    pub fn recalibrate_token_stats(&mut self, calibrator: &dyn TokenSpecificityCalibrator) {
        let stats = calibrator.calibrate_token_stats(&self.entry_token_sigs, self.corpus.entries());
        self.entry_token_weights = stats.entry_token_weights;
        self.entry_token_hub_bias = stats.entry_token_hub_bias;
        self.global_token_hub_mean = stats.global_token_hub_mean;
    }

    /// Appends a batch of auxiliary per-token Q observations into their target corpus entries
    /// without altering pooled entry signatures or centroids, then recalibrates token specificity
    /// and hubness statistics once at the end of the batch.
    ///
    /// Validation and feature extraction run before mutating `self`, so any error leaves the corpus
    /// unchanged.
    ///
    /// # Errors
    /// Returns an error if any observation's `entry_idx` is out of range or its activation tensor
    /// shape disagrees with `adapter`.
    pub fn append_entry_token_observations(
        &mut self,
        adapter: &ModelSteeringAdapter,
        observations: &[EntryTokenObservation],
    ) -> Result<(), String> {
        if observations.is_empty() {
            return Ok(());
        }
        let num_layers = adapter.geometry().num_layers;
        let mut staged: Vec<(usize, Vec<Vec<f32>>)> = Vec::with_capacity(observations.len());
        let mut appended_any = false;

        for obs in observations {
            if obs.entry_idx >= self.entry_token_sigs.len() {
                return Err(format!(
                    "entry_idx {} out of range (len={})",
                    obs.entry_idx,
                    self.entry_token_sigs.len()
                ));
            }
            if obs.per_layer_token_qs.len() != num_layers {
                return Err(format!(
                    "observation layer count mismatch: expected {num_layers}, got {}",
                    obs.per_layer_token_qs.len()
                ));
            }
            if obs.seq_len == 0 {
                continue;
            }
            let salient = Self::extract_salient_token_sigs_indexed_masked(
                adapter,
                &obs.per_layer_token_qs,
                obs.seq_len,
                &self.novelty_reference_per_layer,
                &self.token_common_mode_basis,
                obs.seq_len,
                self.pooler,
                None,
            )?;
            let sigs: Vec<Vec<f32>> = salient.into_iter().map(|(_, sig, _)| sig).collect();
            if !sigs.is_empty() {
                appended_any = true;
                staged.push((obs.entry_idx, sigs));
            }
        }

        if !appended_any {
            return Ok(());
        }

        for (entry_idx, sigs) in staged {
            self.entry_token_sigs[entry_idx].extend(sigs);
        }
        self.recalibrate_token_stats(&AdaptiveTokenSpecificityCalibrator::default());
        Ok(())
    }

    /// Captures a slice of `(role, text, energy)` source records through `encoder` into raw
    /// [`TokenSequenceEntry`] buffers ready for [`Self::compile`] or [`Self::compile_with_base_prior`].
    ///
    /// # Errors
    /// Returns an error if `encoder` fails on any source text.
    pub fn capture_entries<E: TokenSequenceEncoder + ?Sized>(
        encoder: &mut E,
        sources: &[(AttractorRole, &str, f32)],
    ) -> Result<Vec<TokenSequenceEntry>, String> {
        let mut staged = Vec::with_capacity(sources.len());
        for &(role, text, energy) in sources {
            let captured = encoder.encode_token_sequence(text)?;
            staged.push(captured.into_token_sequence_entry(role, text, energy));
        }
        Ok(staged)
    }

    /// Probes every [`AttractorRole::Fact`] entry's leading span via `probe` for continuation-context
    /// subword aliases and atomically appends all discovered observations into this compiled corpus,
    /// returning the number of alias observations appended.
    ///
    /// # Errors
    /// Returns an error if `probe` fails or any captured observation disagrees with `adapter`.
    pub fn append_leading_alias_observations<P: PrefixAliasProbe + ?Sized>(
        &mut self,
        adapter: &ModelSteeringAdapter,
        entries: &[TokenSequenceEntry],
        probe: &mut P,
    ) -> Result<usize, String> {
        let dim = adapter.geometry().q_rot_len();
        let mut observations: Vec<EntryTokenObservation> = Vec::new();
        for (entry_idx, entry) in entries.iter().enumerate() {
            if entry.role != AttractorRole::Fact {
                continue;
            }
            let Some(word) = extract_leading_alias_word(&entry.text) else {
                continue;
            };
            if let Some(captured) = probe.probe_leading_alias(word, dim)? {
                observations.push(EntryTokenObservation {
                    entry_idx,
                    seq_len: captured.seq_len,
                    per_layer_token_qs: captured.per_layer_token_qs,
                });
            }
        }
        let count = observations.len();
        self.append_entry_token_observations(adapter, &observations)?;
        Ok(count)
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
        self.recall_token_sequence_with_mask_strata_and_circuit(
            adapter,
            query_token_qs,
            query_seq_len,
            None,
            &StratumPipeline::default(),
            None,
        )
    }

    /// Pools a raw query token sequence while excluding tokens where
    /// `token_include_mask[t] == false` and recalls via the default [`StratumPipeline`].
    ///
    /// # Errors
    /// Returns an error if layer count, token length, mask length, or stratum inference fails.
    pub fn recall_token_sequence_with_mask<'a>(
        &'a self,
        adapter: &ModelSteeringAdapter,
        query_token_qs: &[Vec<f32>],
        query_seq_len: usize,
        token_include_mask: Option<&[bool]>,
    ) -> Result<DynamicAttractorRecall<'a>, String> {
        self.recall_token_sequence_with_mask_strata_and_circuit(
            adapter,
            query_token_qs,
            query_seq_len,
            token_include_mask,
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
        prior_circuit: Option<liminika_core::stigmergy::CognitiveTrajectoryLedger>,
    ) -> Result<DynamicAttractorRecall<'a>, String> {
        self.recall_token_sequence_with_mask_strata_and_circuit(
            adapter,
            query_token_qs,
            query_seq_len,
            None,
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
        self.build_local_observation_masked(adapter, query_token_qs, query_seq_len, None)
    }

    /// Builds an uncollapsed [`LocalChannelObservation`] strictly over tokens where
    /// `token_include_mask[t] == true` (excluding masked tokens from MaxSim, deposits, and the
    /// `unexplained_token_ratio` denominator).
    ///
    /// When `token_include_mask` excludes all tokens, returns a fail-closed empty observation with
    /// `unexplained_token_ratio = 1.0` and `query_seq_len = 0` without NaN or division by zero.
    ///
    /// # Errors
    /// Returns an error if layer count, token slice dimensions, or mask length mismatch `adapter`.
    pub fn build_local_observation_masked(
        &self,
        adapter: &ModelSteeringAdapter,
        query_token_qs: &[Vec<f32>],
        query_seq_len: usize,
        token_include_mask: Option<&[bool]>,
    ) -> Result<LocalChannelObservation, String> {
        let num_layers = adapter.geometry().num_layers;
        let dim = adapter.geometry().q_rot_len();
        if query_token_qs.len() != num_layers {
            return Err(format!(
                "query layer count mismatch: expected {num_layers}, got {}",
                query_token_qs.len()
            ));
        }
        let effective_seq_len = validate_token_include_mask(query_seq_len, token_include_mask)?;
        let n_docs = self.entry_token_sigs.len();
        if effective_seq_len == 0 {
            return Ok(LocalChannelObservation {
                combined_scores: vec![0.0_f32; n_docs],
                q_to_d_scores: vec![0.0_f32; n_docs],
                d_to_q_scores: vec![0.0_f32; n_docs],
                token_deposits: Vec::new(),
                unexplained_token_ratio: 1.0_f32,
                query_seq_len: 0,
            });
        }
        let query_salient = Self::extract_salient_token_sigs_indexed_masked(
            adapter,
            query_token_qs,
            query_seq_len,
            &self.novelty_reference_per_layer,
            &self.token_common_mode_basis,
            24,
            self.pooler,
            token_include_mask,
        )?;
        let mut combined_scores = vec![0.0_f32; n_docs];
        let mut q_to_d_scores = vec![0.0_f32; n_docs];
        let mut d_to_q_scores = vec![0.0_f32; n_docs];
        let mut token_deposits: Vec<TokenResonanceDeposit> = Vec::new();
        let last_rope_layer = (0..num_layers)
            .rfind(|&l| adapter.geometry().is_rope_layer(l))
            .unwrap_or(0);

        let mut unexplained_token_ratio = 0.0_f32;
        if n_docs > 0 && !query_salient.is_empty() {
            // Unsupervised token-level discriminative centering (ColBERT-IDF equivalent):
            // down-weight query tokens that respond uniformly across all corpus entries and
            // center each token's MaxSim row around its corpus-wide mean before aggregation.
            let mut weighted_sums = vec![0.0_f32; n_docs];
            let mut total_disc_weight = 0.0_f32;
            let mut raw_weighted_sums = vec![0.0_f32; n_docs];
            let mut total_raw_weight = 0.0_f32;
            let mut unexplained_w_sum = 0.0_f32;
            let mut explain_w_total = 0.0_f32;
            let has_fact_docs = self.corpus.entries().iter().enumerate().any(|(i, e)| {
                e.role == AttractorRole::Fact && !self.entry_token_sigs[i].is_empty()
            });

            for (tok_idx, q_tok, w_t) in &query_salient {
                let mut row_sims = vec![0.0_f32; n_docs];
                let mut sum_row = 0.0_f32;
                let mut max_row = f32::NEG_INFINITY;
                let mut best_fact_raw_cos = 0.0_f32;
                let mut valid_docs = 0_usize;
                for (i, doc_tokens) in self.entry_token_sigs.iter().enumerate() {
                    if doc_tokens.is_empty() {
                        continue;
                    }
                    let is_fact_doc = self
                        .corpus
                        .entries()
                        .get(i)
                        .is_some_and(|e| e.role == AttractorRole::Fact);
                    let doc_weights = self.entry_token_weights.get(i);
                    let mut max_sim = f32::NEG_INFINITY;
                    for (k, d_tok) in doc_tokens.iter().enumerate() {
                        let c = TextAttractorCorpus::cosine_f32(q_tok, d_tok);
                        if is_fact_doc && c > best_fact_raw_cos {
                            best_fact_raw_cos = c;
                        }
                        let weighted_c = if c > 0.0 {
                            let spec = doc_weights.and_then(|ws| ws.get(k).copied()).unwrap_or(1.0);
                            c * spec
                        } else {
                            c
                        };
                        if weighted_c > max_sim {
                            max_sim = weighted_c;
                        }
                    }
                    if max_sim.is_finite() {
                        row_sims[i] = max_sim;
                        sum_row += max_sim;
                        if max_sim > max_row {
                            max_row = max_sim;
                        }
                        valid_docs += 1;
                    }
                }
                if has_fact_docs && *w_t > 0.0 {
                    explain_w_total += *w_t;
                    if best_fact_raw_cos < self.token_explain_threshold {
                        unexplained_w_sum += *w_t;
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
                    ) && end <= layer_flat.len()
                        && let Ok(q7) =
                            adapter.bake_layer_procrustes_q7(ref_slice, &layer_flat[start..end])
                    {
                        let mut arr64 = [0_i8; MAX_ATTENTION_HEADS];
                        let n = q7.len().min(MAX_ATTENTION_HEADS);
                        arr64[..n].copy_from_slice(&q7[..n]);
                        token_phase_words = load_q7_words_64(&arr64);
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
            if explain_w_total > 1e-8 {
                unexplained_token_ratio = (unexplained_w_sum / explain_w_total).clamp(0.0, 1.0);
            }
        }

        Ok(LocalChannelObservation {
            combined_scores,
            q_to_d_scores,
            d_to_q_scores,
            token_deposits,
            unexplained_token_ratio,
            query_seq_len: effective_seq_len,
        })
    }

    /// Scores a raw query token sequence using a custom [`StratumPipeline`] and optional prior
    /// [`CognitiveTrajectoryLedger`](liminika_core::stigmergy::CognitiveTrajectoryLedger).
    ///
    /// # Errors
    /// Returns an error if layer count, token length, or stratum inference fails.
    pub fn recall_token_sequence_with_strata_and_circuit<'a>(
        &'a self,
        adapter: &ModelSteeringAdapter,
        query_token_qs: &[Vec<f32>],
        query_seq_len: usize,
        pipeline: &StratumPipeline<'_>,
        prior_circuit: Option<liminika_core::stigmergy::CognitiveTrajectoryLedger>,
    ) -> Result<DynamicAttractorRecall<'a>, String> {
        self.recall_token_sequence_with_mask_strata_and_circuit(
            adapter,
            query_token_qs,
            query_seq_len,
            None,
            pipeline,
            prior_circuit,
        )
    }

    /// Scores a raw query token sequence while excluding masked tokens (`mask[t] == false`) using
    /// a custom [`StratumPipeline`] and optional prior
    /// [`CognitiveTrajectoryLedger`](liminika_core::stigmergy::CognitiveTrajectoryLedger).
    ///
    /// When `token_include_mask` excludes every token in the sequence, fails closed directly into
    /// an unobserved `LiminalExterior` recall outcome with zero NaN/Inf values.
    ///
    /// # Errors
    /// Returns an error if layer count, token length, mask length, or stratum inference fails.
    pub fn recall_token_sequence_with_mask_strata_and_circuit<'a>(
        &'a self,
        adapter: &ModelSteeringAdapter,
        query_token_qs: &[Vec<f32>],
        query_seq_len: usize,
        token_include_mask: Option<&[bool]>,
        pipeline: &StratumPipeline<'_>,
        prior_circuit: Option<liminika_core::stigmergy::CognitiveTrajectoryLedger>,
    ) -> Result<DynamicAttractorRecall<'a>, String> {
        let num_layers = adapter.geometry().num_layers;
        let dim = adapter.geometry().q_rot_len();
        if query_token_qs.len() != num_layers {
            return Err(format!(
                "query layer count mismatch: expected {num_layers}, got {}",
                query_token_qs.len()
            ));
        }
        let effective_seq_len = validate_token_include_mask(query_seq_len, token_include_mask)?;
        if effective_seq_len == 0 {
            // Fail-closed unobserved degradation when every token was masked out (zero NaN/Inf).
            return Ok(DynamicAttractorRecall {
                lacuna_state: LacunaBoundaryState::LiminalExterior,
                top1_fact_score: 0.0,
                top2_fact_score: 0.0,
                top1_persona_score: 0.0,
                min_local_lacuna_depth: self.recall_policy.min_coherence,
                guard_conflict_margin: 0.0,
                action_decision: StigmergicActionDecision::PassiveSidecar,
                gateway_frame: GatewayFrame::default(),
                telemetry: StratumProbeTelemetry {
                    cognitive_trajectory: prior_circuit.unwrap_or_default(),
                    query_ortho_ratio: 0.0,
                    unexplained_token_ratio: 1.0,
                    ..StratumProbeTelemetry::default()
                },
                recalled_facts: Vec::new(),
                recalled_personas: Vec::new(),
                top_guard: None,
                lacuna_partition: LacunaPartition8::default(),
                exterior_sandbox_facts: Vec::new(),
                is_known_unknown_exterior: false,
            });
        }

        let pooled_q = self.pooler.pool_sequence_masked(
            query_token_qs,
            query_seq_len,
            dim,
            Some(&self.novelty_reference_per_layer),
            token_include_mask,
        )?;
        let obs = self.build_local_observation_masked(
            adapter,
            query_token_qs,
            query_seq_len,
            token_include_mask,
        )?;
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
