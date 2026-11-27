// name: src/adapter/src/pooling/streaming.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Single-pass streaming raw-document distillation ([`StreamingCorpusDistiller`]) eliminating
//! full-corpus raw activation retention and $O(N^2)$ pairwise compilation bottlenecks.

use super::base_prior::{
    build_corpus_discriminative_subspace, collect_corpus_pooled_sigs,
    compute_entry_token_centroids, orthogonalize_prior_basis_against_discriminative,
};
use super::null_calibration::{
    DEFAULT_TOKEN_EXPLAIN_THRESHOLD, NullCalibrationContext, calibrate_null_envelope_2fold,
};
use super::{
    AdaptiveTokenSpecificityCalibrator, CapturedTokenSequence, CompiledTokenCorpus,
    ModelBaseAttractorProfile, NullCalibrationEnvelope, SequenceActivationPooler,
    SequencePoolingMode, TokenCalibrationStats, TokenSequenceEncoder, TokenSpecificityCalibrator,
    validate_token_include_mask,
};
use crate::{AttractorRole, ModelSteeringAdapter, TextAttractorCorpus};
use liminika_core::liminal::MAX_ATTENTION_HEADS;
use liminika_core::runtime::ManifoldProbeBaseline;

/// Default maximum salient token signatures retained per distilled chunk in streaming mode (`10`).
pub const DEFAULT_STREAMING_SALIENT_TOKENS: usize = 10;

/// Maximum entry count sampled when fitting pooled or token common-mode PCA subspaces (`64`),
/// bounding subspace fitting to $O(1)$ entries for arbitrarily long raw documents.
pub const STREAMING_PCA_SAMPLE_BUDGET: usize = 64;

/// Configuration for [`StreamingCorpusDistiller`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamingDistillConfig {
    /// Sequence pooler configuration applied identically to corpus chunks and queries.
    pub pooler: SequenceActivationPooler,
    /// Number of top Fact attractors returned per recall query.
    pub top_k_facts: usize,
    /// Number of top Persona attractors returned per recall query.
    pub top_k_personas: usize,
    /// Maximum salient token signatures retained per chunk (`1..=24`, default `10`).
    pub max_salient_tokens_per_entry: usize,
}

impl Default for StreamingDistillConfig {
    #[inline]
    fn default() -> Self {
        Self {
            pooler: SequenceActivationPooler::default(),
            top_k_facts: 2,
            top_k_personas: 1,
            max_salient_tokens_per_entry: DEFAULT_STREAMING_SALIENT_TOKENS,
        }
    }
}

/// Summary telemetry emitted by [`StreamingCorpusDistiller::finalize`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamingDistillReport {
    /// Total raw document chunks ingested in a single forward pass.
    pub chunks_ingested: usize,
    /// Total source tokens processed across all ingested chunks.
    pub total_tokens_processed: usize,
    /// Total compact salient token signatures retained across all compiled entries.
    pub retained_salient_tokens: usize,
    /// Cumulative raw per-token activation bytes immediately discarded after each chunk.
    pub raw_activation_bytes_discarded: u64,
    /// Number of compiled attractor anchors available for paging/persistence.
    pub anchors_generated: usize,
}

/// Deterministically splits a raw long document (Markdown, prose, or logs) into UTF-8 safe,
/// paragraph/heading-aware chunks bounded by `max_chars_per_chunk` with optional overlap.
#[must_use]
pub fn split_raw_document_chunks(
    raw_text: &str,
    max_chars_per_chunk: usize,
    overlap_chars: usize,
) -> Vec<String> {
    let cap = max_chars_per_chunk.max(64);
    let overlap = overlap_chars.min(cap / 2);
    let mut chunks = Vec::new();
    let mut current = String::new();

    for raw_para in raw_text.split("\n\n") {
        let para = raw_para.trim();
        if para.is_empty() {
            continue;
        }
        if !current.is_empty() && current.chars().count() + 2 + para.chars().count() > cap {
            chunks.push(std::mem::take(&mut current));
            if overlap > 0 {
                let tail: String = chunks
                    .last()
                    .map(|prev| {
                        let chars: Vec<char> = prev.chars().collect();
                        let start = chars.len().saturating_sub(overlap);
                        chars[start..].iter().collect()
                    })
                    .unwrap_or_default();
                let trimmed_tail = tail.trim();
                if !trimmed_tail.is_empty() {
                    current.push_str(trimmed_tail);
                    current.push('\n');
                }
            }
        }
        if para.chars().count() <= cap {
            if !current.is_empty() {
                current.push_str("\n\n");
            }
            current.push_str(para);
        } else {
            if !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            let chars: Vec<char> = para.chars().collect();
            let step = (cap - overlap).max(1);
            let mut pos = 0_usize;
            while pos < chars.len() {
                let end = (pos + cap).min(chars.len());
                let piece: String = chars[pos..end].iter().collect();
                let trimmed = piece.trim();
                if !trimmed.is_empty() {
                    chunks.push(trimmed.to_string());
                }
                if end == chars.len() {
                    break;
                }
                pos = pos.saturating_add(step);
            }
        }
    }
    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }
    chunks
}

struct CompactStreamedEntry {
    role: AttractorRole,
    text: String,
    energy: f32,
    pooled_per_layer: Vec<Vec<f32>>,
    salient_raw_tok_sigs: Vec<Vec<f32>>,
    salient_token_weights: Vec<f32>,
}

fn deterministic_stride_indices(len: usize, budget: usize) -> Vec<usize> {
    if len == 0 || budget == 0 {
        return Vec::new();
    }
    if len <= budget {
        return (0..len).collect();
    }
    (0..budget).map(|s| (s * len) / budget).collect()
}

/// Single-pass streaming distiller that consumes one chunk's [`CapturedTokenSequence`] at a time,
/// extracts compact per-layer pooled summaries and bounded salient token clouds, and immediately
/// frees the raw `[num_layers][seq_len * q_rot_len]` activation matrices.
pub struct StreamingCorpusDistiller {
    config: StreamingDistillConfig,
    num_layers: usize,
    q_rot_len: usize,
    novelty_sum_per_layer: Vec<Vec<f32>>,
    corpus_sum_per_layer: Vec<Vec<f32>>,
    non_guard_sum_per_layer: Vec<Vec<f32>>,
    non_guard_count: usize,
    entries: Vec<CompactStreamedEntry>,
    total_tokens_processed: usize,
    raw_activation_bytes_discarded: u64,
}

impl StreamingCorpusDistiller {
    /// Creates a fresh [`StreamingCorpusDistiller`] sized to `adapter.geometry()`.
    #[must_use]
    pub fn new(adapter: &ModelSteeringAdapter, config: StreamingDistillConfig) -> Self {
        let geom = adapter.geometry();
        let num_layers = geom.num_layers;
        let q_rot_len = geom.q_rot_len();
        Self {
            config: StreamingDistillConfig {
                max_salient_tokens_per_entry: config.max_salient_tokens_per_entry.clamp(1, 24),
                ..config
            },
            num_layers,
            q_rot_len,
            novelty_sum_per_layer: vec![vec![0.0_f32; q_rot_len]; num_layers],
            corpus_sum_per_layer: vec![vec![0.0_f32; q_rot_len]; num_layers],
            non_guard_sum_per_layer: vec![vec![0.0_f32; q_rot_len]; num_layers],
            non_guard_count: 0,
            entries: Vec::new(),
            total_tokens_processed: 0,
            raw_activation_bytes_discarded: 0,
        }
    }

    /// Returns the number of chunks distilled so far.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no chunks have been distilled yet.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Encodes one text chunk via `encoder` (strictly 1 forward pass), distills compact summaries,
    /// and immediately drops the raw token activation matrix.
    ///
    /// # Errors
    /// Returns an error if encoding fails or captured tensor shapes disagree with `adapter`.
    pub fn push_text<E: TokenSequenceEncoder + ?Sized>(
        &mut self,
        encoder: &mut E,
        adapter: &ModelSteeringAdapter,
        role: AttractorRole,
        text: impl Into<String>,
        energy: f32,
    ) -> Result<usize, String> {
        let text = text.into();
        let captured = encoder.encode_token_sequence(&text)?;
        self.push_captured(adapter, role, text, energy, captured)
    }

    /// Consumes one chunk's [`CapturedTokenSequence`] by value, updates online centroids, extracts
    /// compact pooled vectors + bounded salient token signatures, and drops the raw activation
    /// buffers before returning.
    ///
    /// # Errors
    /// Returns an error if layer count, `seq_len`, or flattened activation dimensions are invalid.
    pub fn push_captured(
        &mut self,
        adapter: &ModelSteeringAdapter,
        role: AttractorRole,
        text: impl Into<String>,
        energy: f32,
        captured: CapturedTokenSequence,
    ) -> Result<usize, String> {
        let geom = adapter.geometry();
        if captured.per_layer_token_qs.len() != self.num_layers {
            return Err(format!(
                "streaming entry layer count mismatch: expected {}, got {}",
                self.num_layers,
                captured.per_layer_token_qs.len()
            ));
        }
        if captured.seq_len == 0 {
            return Err("streaming entry seq_len must be > 0".to_string());
        }
        let dim = self.q_rot_len;
        let seq_len = captured.seq_len;
        let mask = captured.token_include_mask.as_deref();
        let included = validate_token_include_mask(seq_len, mask)?;
        if included == 0 {
            return Err("streaming entry has zero included tokens after token_include_mask".into());
        }
        let raw_bytes: u64 = captured
            .per_layer_token_qs
            .iter()
            .map(|v| (v.len() * std::mem::size_of::<f32>()) as u64)
            .sum();

        // 1. Update running pure-corpus uniform reference (online Pass-1 equivalent).
        let mut uniform_per_layer: Vec<Vec<f32>> = Vec::with_capacity(self.num_layers);
        for (l, flat) in captured.per_layer_token_qs.iter().enumerate() {
            let u = SequenceActivationPooler::uniform_unit_mean_masked(flat, seq_len, dim, mask)?;
            for (dst, &src) in self.novelty_sum_per_layer[l].iter_mut().zip(u.iter()) {
                *dst += src;
            }
            uniform_per_layer.push(u);
        }

        let count_now = (self.entries.len() + 1) as f32;
        let inv_count = 1.0_f32 / count_now;
        let running_ref_per_layer: Vec<Vec<f32>> = self
            .novelty_sum_per_layer
            .iter()
            .map(|sum_l| sum_l.iter().map(|&v| v * inv_count).collect())
            .collect();

        // 2. Pool this chunk on the fly against the running pure-corpus reference.
        let pooled_per_layer = if self.entries.is_empty() {
            uniform_per_layer
        } else {
            self.config.pooler.pool_sequence_masked(
                &captured.per_layer_token_qs,
                seq_len,
                dim,
                Some(&running_ref_per_layer),
                mask,
            )?
        };

        let is_non_guard = role != AttractorRole::Guard;
        if is_non_guard {
            self.non_guard_count += 1;
        }
        for (l, vec_l) in pooled_per_layer.iter().enumerate() {
            for (dst, &src) in self.corpus_sum_per_layer[l].iter_mut().zip(vec_l.iter()) {
                *dst += src;
            }
            if is_non_guard {
                for (dst, &src) in self.non_guard_sum_per_layer[l].iter_mut().zip(vec_l.iter()) {
                    *dst += src;
                }
            }
        }

        // 3. Score included tokens on the fly and materialize steered signatures ONLY for Top-K.
        let (power, floor) = match self.config.pooler.mode {
            SequencePoolingMode::UniformMean => (0.0_f32, 1.0_f32),
            SequencePoolingMode::CorpusNoveltySif {
                novelty_power,
                weight_floor,
            } => (novelty_power.max(0.0), weight_floor.clamp(1e-3, 1.0)),
        };
        let mut token_scores: Vec<(usize, f32)> = Vec::with_capacity(included);
        for t in 0..seq_len {
            if mask.is_some_and(|m| !m[t]) {
                continue;
            }
            let mut sum_w = 0.0_f32;
            let mut cnt = 0_usize;
            for (l, flat) in captured.per_layer_token_qs.iter().enumerate() {
                if !geom.is_rope_layer(l) {
                    continue;
                }
                let slice = &flat[t * dim..(t + 1) * dim];
                let cos_pos =
                    TextAttractorCorpus::cosine_f32(slice, &running_ref_per_layer[l]).max(0.0);
                sum_w += (1.0 - cos_pos).clamp(floor, 1.0).powf(power);
                cnt += 1;
            }
            let mean_w = if cnt > 0 { sum_w / (cnt as f32) } else { 1.0 };
            token_scores.push((t, mean_w));
        }
        token_scores.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let keep = self.config.max_salient_tokens_per_entry.min(included);
        token_scores.truncate(keep);

        let ref_slices: Vec<&[f32]> = running_ref_per_layer.iter().map(Vec::as_slice).collect();
        let mut unit_tok_layers: Vec<Vec<f32>> = vec![vec![0.0_f32; dim]; self.num_layers];
        let mut salient_raw_tok_sigs = Vec::with_capacity(token_scores.len());
        let mut salient_token_weights = Vec::with_capacity(token_scores.len());
        for (t, w) in token_scores {
            for (l, flat) in captured.per_layer_token_qs.iter().enumerate() {
                if !geom.is_rope_layer(l) {
                    continue;
                }
                let src = &flat[t * dim..(t + 1) * dim];
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
            salient_raw_tok_sigs.push(TextAttractorCorpus::build_steered_residual_sig(
                geom,
                &tok_refs,
                &ref_slices,
            ));
            salient_token_weights.push(w);
        }

        // Explicitly drop raw `[num_layers][seq_len * dim]` buffers right here.
        drop(captured);
        self.total_tokens_processed = self.total_tokens_processed.saturating_add(seq_len);
        self.raw_activation_bytes_discarded = self
            .raw_activation_bytes_discarded
            .saturating_add(raw_bytes);

        let entry_idx = self.entries.len();
        self.entries.push(CompactStreamedEntry {
            role,
            text: text.into(),
            energy,
            pooled_per_layer,
            salient_raw_tok_sigs,
            salient_token_weights,
        });
        Ok(entry_idx)
    }

    /// Finalizes the distilled corpus from compact streamed statistics without re-running the
    /// model forward pass, preserving pure-corpus centroids, non-Guard Guard baking, quasi-linear
    /// calibration, and non-leaking 2-fold `base_prior` null calibration.
    ///
    /// # Errors
    /// Returns an error if no chunks were ingested or corpus ingestion fails.
    pub fn finalize(
        self,
        adapter: &ModelSteeringAdapter,
        base_prior: Option<&ModelBaseAttractorProfile>,
    ) -> Result<(CompiledTokenCorpus, StreamingDistillReport), String> {
        if self.entries.is_empty() {
            return Err("cannot finalize an empty StreamingCorpusDistiller".to_string());
        }
        let n_entries = self.entries.len();
        let inv_n = 1.0_f32 / (n_entries as f32);

        let novelty_reference_per_layer: Vec<Vec<f32>> = self
            .novelty_sum_per_layer
            .into_iter()
            .map(|mut sum_l| {
                for v in &mut sum_l {
                    *v *= inv_n;
                }
                sum_l
            })
            .collect();
        let corpus_centroid_per_layer: Vec<Vec<f32>> = self
            .corpus_sum_per_layer
            .into_iter()
            .map(|mut sum_l| {
                for v in &mut sum_l {
                    *v *= inv_n;
                }
                sum_l
            })
            .collect();
        let guard_phase_centroid_per_layer = if self.non_guard_count > 0 {
            let inv_ng = 1.0_f32 / (self.non_guard_count as f32);
            self.non_guard_sum_per_layer
                .into_iter()
                .map(|mut sum_l| {
                    for v in &mut sum_l {
                        *v *= inv_ng;
                    }
                    sum_l
                })
                .collect()
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
        for entry in &self.entries {
            let q_refs: Vec<&[f32]> = entry.pooled_per_layer.iter().map(Vec::as_slice).collect();
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

        // Fit pooled common-mode subspace in O(min(N, 64) * D) using deterministic stride sampling.
        corpus.fit_common_mode_subspace(3);
        let pooled_disc_basis = build_corpus_discriminative_subspace(
            &collect_corpus_pooled_sigs(&corpus),
            corpus.common_mode_basis(),
        );
        let corpus_pre_prior = if base_prior.is_some_and(|p| p.calibration_samples.len() >= 4) {
            Some(corpus.clone())
        } else {
            None
        };
        if let Some(prior) = base_prior {
            let safe_prior_pooled = orthogonalize_prior_basis_against_discriminative(
                &prior.pooled_common_mode_basis,
                &pooled_disc_basis,
            );
            corpus.extend_common_mode_basis(&safe_prior_pooled, 3);
        }
        let mut recall_policy =
            corpus.calibrate_recall_policy(self.config.top_k_facts, self.config.top_k_personas);
        let corpus_offdiag_floor = recall_policy.min_coherence;

        // Fit token-space common mode over stride-sampled entries (`<= 64` entries).
        let sample_idxs = deterministic_stride_indices(n_entries, STREAMING_PCA_SAMPLE_BUDGET);
        let sampled_toks: Vec<Vec<Vec<f32>>> = sample_idxs
            .iter()
            .map(|&i| self.entries[i].salient_raw_tok_sigs.clone())
            .collect();
        let corpus_tok_basis =
            CompiledTokenCorpus::fit_cross_entry_token_common_mode(&sampled_toks, 10);
        let tok_disc_basis = build_corpus_discriminative_subspace(
            &compute_entry_token_centroids(&sampled_toks),
            &corpus_tok_basis,
        );
        let token_common_mode_basis = if let Some(prior) = base_prior {
            let safe_prior_tok = orthogonalize_prior_basis_against_discriminative(
                &prior.token_common_mode_basis,
                &tok_disc_basis,
            );
            CompiledTokenCorpus::merge_orthonormal_basis(&corpus_tok_basis, &safe_prior_tok, 12)
        } else {
            corpus_tok_basis.clone()
        };

        let project_compact_entries =
            |entries: &[CompactStreamedEntry], basis: &[Vec<f32>]| -> Vec<Vec<Vec<f32>>> {
                entries
                    .iter()
                    .map(|e| {
                        let mut scored: Vec<(usize, f32)> = e
                            .salient_raw_tok_sigs
                            .iter()
                            .zip(e.salient_token_weights.iter())
                            .enumerate()
                            .map(|(idx, (sig, &w))| {
                                let ortho =
                                    CompiledTokenCorpus::token_orthogonal_residual_norm(sig, basis);
                                (idx, w * ortho * ortho)
                            })
                            .collect();
                        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                        scored
                            .into_iter()
                            .map(|(idx, _)| {
                                TextAttractorCorpus::project_and_normalize(
                                    &e.salient_raw_tok_sigs[idx],
                                    basis,
                                )
                            })
                            .collect()
                    })
                    .collect()
            };

        let entry_token_sigs = project_compact_entries(&self.entries, &token_common_mode_basis);
        let retained_salient_tokens: usize = entry_token_sigs.iter().map(Vec::len).sum();

        let TokenCalibrationStats {
            entry_token_weights,
            entry_token_hub_bias,
            global_token_hub_mean,
        } = AdaptiveTokenSpecificityCalibrator::default()
            .calibrate_token_stats(&entry_token_sigs, corpus.entries());

        // Non-leaking 2-fold cross-fitted OOD null calibration over compact streamed entries.
        let mut null_calibration: Option<NullCalibrationEnvelope> = None;
        let mut token_explain_threshold = DEFAULT_TOKEN_EXPLAIN_THRESHOLD;
        if let (Some(prior), Some(pre_prior_corpus)) = (base_prior, corpus_pre_prior)
            && let Some((env, tok_thresh)) = calibrate_null_envelope_2fold(
                &NullCalibrationContext {
                    adapter,
                    prior,
                    corpus_pre_prior: &pre_prior_corpus,
                    corpus_tok_basis: &corpus_tok_basis,
                    pooled_disc_basis: &pooled_disc_basis,
                    tok_disc_basis: &tok_disc_basis,
                    novelty_reference_per_layer: &novelty_reference_per_layer,
                    corpus_centroid_per_layer: &corpus_centroid_per_layer,
                    recall_policy,
                    corpus_offdiag_floor,
                    pooler: self.config.pooler,
                },
                |fold_tok_basis| project_compact_entries(&self.entries, fold_tok_basis),
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

        let anchors_generated = corpus.entries().len();
        let compiled = CompiledTokenCorpus {
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
            pooler: self.config.pooler,
        };
        let report = StreamingDistillReport {
            chunks_ingested: n_entries,
            total_tokens_processed: self.total_tokens_processed,
            retained_salient_tokens,
            raw_activation_bytes_discarded: self.raw_activation_bytes_discarded,
            anchors_generated,
        };
        Ok((compiled, report))
    }
}

impl CompiledTokenCorpus {
    /// Encodes and distills `(role, text, energy)` slices in a strict single-pass streaming
    /// pipeline via [`StreamingCorpusDistiller`], dropping each chunk's raw token activations
    /// immediately after distillation.
    ///
    /// # Errors
    /// Returns an error if `sources` is empty, encoding fails, or tensor shapes disagree.
    pub fn compile_streaming_with_base_prior<E: TokenSequenceEncoder + ?Sized>(
        adapter: &ModelSteeringAdapter,
        encoder: &mut E,
        sources: &[(AttractorRole, &str, f32)],
        config: StreamingDistillConfig,
        base_prior: Option<&ModelBaseAttractorProfile>,
    ) -> Result<(Self, StreamingDistillReport), String> {
        let mut distiller = StreamingCorpusDistiller::new(adapter, config);
        for &(role, text, energy) in sources {
            let _ = distiller.push_text(encoder, adapter, role, text, energy)?;
        }
        distiller.finalize(adapter, base_prior)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::{AnchorPagerConfig, PagerStepQuery};
    use crate::{RopeGeometry, RopeLayout};
    use liminika_core::liminal::evaluate_lacuna_partition_branchless;

    struct SyntheticChunkEncoder {
        layers: usize,
        dim: usize,
        forward_calls: usize,
    }

    impl TokenSequenceEncoder for SyntheticChunkEncoder {
        fn encode_token_sequence(&mut self, text: &str) -> Result<CapturedTokenSequence, String> {
            self.forward_calls += 1;
            let seq_len = text.split_whitespace().count().clamp(6, 32);
            let mut seed = 0xCBF2_9CE4_8422_2325_u64;
            for b in text.as_bytes() {
                seed = (seed ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01B3);
            }
            let mut layers = Vec::with_capacity(self.layers);
            for l in 0..self.layers {
                let mut flat = Vec::with_capacity(seq_len * self.dim);
                for t in 0..seq_len {
                    for d in 0..self.dim {
                        let mix = seed
                            .wrapping_add((l as u64).wrapping_mul(31))
                            .wrapping_add((t as u64).wrapping_mul(131))
                            .wrapping_add((d as u64).wrapping_mul(17));
                        let x = ((mix % 200) as f32) * 0.01 - 1.0;
                        flat.push(x);
                    }
                }
                layers.push(flat);
            }
            Ok(CapturedTokenSequence::new(seq_len, layers))
        }
    }

    #[test]
    fn split_raw_document_chunks_handles_utf8_and_bounds() {
        let mut doc = String::new();
        for i in 0..40 {
            doc.push_str(&format!(
                "## Section {i}\nParagraph {i} describes deterministic geometric phase \
                 manifolds and bounded paging.\n\n"
            ));
        }
        let chunks = split_raw_document_chunks(&doc, 180, 24);
        assert!(chunks.len() >= 15);
        for c in &chunks {
            assert!(!c.is_empty());
        }
    }

    #[test]
    fn streaming_distill_280_chunks_single_pass_and_pages_beyond_64() {
        let geom = RopeGeometry {
            layout: RopeLayout::HalfSplit,
            head_dim: 8,
            rotary_dim: 8,
            num_q_heads: 4,
            num_kv_heads: 2,
            num_layers: 4,
            layer_rope_mask: 0x0C,
        };
        let adapter = ModelSteeringAdapter::new(geom).expect("adapter");
        let mut encoder = SyntheticChunkEncoder {
            layers: 4,
            dim: adapter.geometry().q_rot_len(),
            forward_calls: 0,
        };

        let mut distiller =
            StreamingCorpusDistiller::new(&adapter, StreamingDistillConfig::default());
        for idx in 0..280_usize {
            let role = if idx == 279 {
                AttractorRole::Guard
            } else {
                AttractorRole::Fact
            };
            let text = format!(
                "Chunk {idx}: synthetic architectural specification section {} anchor topic {}",
                idx / 10,
                idx % 29
            );
            distiller
                .push_text(&mut encoder, &adapter, role, text, 1.5)
                .expect("push_text");
        }
        // Strictly 1 forward pass per chunk.
        assert_eq!(encoder.forward_calls, 280);

        let (compiled, report) = distiller.finalize(&adapter, None).expect("finalize");
        assert_eq!(report.chunks_ingested, 280);
        assert_eq!(report.anchors_generated, 280);
        assert!(report.raw_activation_bytes_discarded > 0);
        assert!(report.retained_salient_tokens <= 280 * DEFAULT_STREAMING_SALIENT_TOKENS);

        // Flow > 64 compiled anchors seamlessly into AnchorWorkingSetPager.
        let mut pager = compiled
            .corpus
            .build_working_set_pager(AnchorPagerConfig::default());
        assert_eq!(pager.backing_len(), 280);
        assert_eq!(pager.resident_len(), 64);

        let probe_anchor = pager.get_entry(250).expect("anchor 250");
        let query = PagerStepQuery {
            phase_profile: probe_anchor.record.phase_profile,
            lacuna_partition: evaluate_lacuna_partition_branchless(0xFF, &[96; 8], &[16; 8], 32),
            desired_count: 4,
            allow_exterior_staging: false,
        };
        let out = pager.step_working_set(&query);
        assert!(!out.skipped_unobserved);
        assert!(pager.resident_len() <= 64);
        assert!(pager.is_resident(250));
    }
}
