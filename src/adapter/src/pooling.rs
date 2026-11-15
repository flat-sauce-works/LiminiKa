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
/// unsupervised centroid estimation + SIF novelty pooling + Stratum calibration.
#[derive(Debug, Clone)]
pub struct CompiledTokenCorpus {
    /// Compiled attractor manifold corpus.
    pub corpus: TextAttractorCorpus,
    /// Pass-1 uniform centroid per layer used strictly as the unsupervised token-novelty baseline.
    pub novelty_reference_per_layer: Vec<Vec<f32>>,
    /// Pass-2 centroid matching the pooled distribution of `corpus` entries.
    pub corpus_centroid_per_layer: Vec<Vec<f32>>,
    /// Empirically calibrated [`RecallPolicy`] derived from projected corpus similarities.
    pub recall_policy: RecallPolicy,
    /// Pooler configuration applied identically during compilation and query recall.
    pub pooler: SequenceActivationPooler,
}

impl CompiledTokenCorpus {
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

        Ok(Self {
            corpus,
            novelty_reference_per_layer,
            corpus_centroid_per_layer,
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
        let q_refs: Vec<&[f32]> = pooled_q.iter().map(Vec::as_slice).collect();
        let centroid_refs: Vec<&[f32]> = self
            .corpus_centroid_per_layer
            .iter()
            .map(Vec::as_slice)
            .collect();
        self.corpus.recall_for_query_with_strata(
            adapter,
            &q_refs,
            &centroid_refs,
            &self.recall_policy,
            pipeline,
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
}
