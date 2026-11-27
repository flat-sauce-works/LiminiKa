// name: src/adapter/src/pooling/encoder.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model-agnostic input distillation boundary traits and contrastive anchor banking.
//!
//! Concrete LLM backends (Candle, `llama.cpp`, ONNX) implement [`TokenSequenceEncoder`] and
//! optionally [`PrefixAliasProbe`] to supply raw pre-RoPE query activations without leaking
//! model-specific tensor code into `liminika-adapter`.

use crate::{
    AttractorRole, ModelSteeringAdapter, SequenceActivationPooler, SourceTextRecord,
    SubtitleActivationEncoder, TokenSequenceEntry,
};
use liminika_core::abi::lmnk_q7_t;
use liminika_core::liminal::MAX_ATTENTION_HEADS;
use liminika_core::runtime::{GuardPhaseAnchor, ObjectiveFact, SubjectivePersona};

/// Maximum character count for a leading alias span extracted from ASCII-only text.
const MAX_ASCII_LEADING_ALIAS_CHARS: usize = 24;

/// Maximum character count for a leading alias span containing non-ASCII characters (preventing
/// unspaced CJK/Japanese sentences from collapsing into whole-line alias probes).
const MAX_NON_ASCII_LEADING_ALIAS_CHARS: usize = 8;

/// Default neutral multilingual prompts spanning six general domains (technical, arithmetic,
/// factual, code, daily conversation, and summarization) across English (12 original prompts
/// preserved verbatim) and Japanese/CJK/symbol-numeric prompts (12 added prompts), interleaved in
/// parity-balanced pairs so 2-fold cross-fitting receives 6 English + 6 non-ASCII prompts and all
/// 6 domains in each fold.
pub const DEFAULT_STARTUP_PRIOR_PROBES: [&str; 24] = [
    // Domain 0: Technical / Architecture (EN even/odd, JA/symbol even/odd)
    "Explain the basic design trade-offs of a modular systems architecture.",
    "Summarize how memory alignment affects hardware cache performance.",
    "分散システムでキャッシュ整合性を保つための基本的な設計手法を簡潔に説明してください。",
    "HTTP/2とTCP接続の多重化がレイテンシに与える影響を2文で要約してください。",
    // Domain 1: Arithmetic / Numeric + Symbols (EN even/odd, JA/numeric even/odd)
    "What is 23 multiplied by 4?",
    "Calculate the remainder when 50 is divided by 7.",
    "37 × 6 + 14 の計算結果と途中の手順を簡潔に示してください。",
    "時速45kmで2時間20分移動したときの合計距離（km）を求めてください。",
    // Domain 2: General Factual / Geography & Units (EN even/odd, JA/CJK even/odd)
    "Which ocean lies between Europe and North America?",
    "How many seconds are in a single hour?",
    "地球上で最も深い海溝の名称と、それが位置する海域を教えてください。",
    "標準大気圧のもとで純水が沸騰する温度は摂氏何度ですか。",
    // Domain 3: Code / Data Structures (EN even/odd, JA + ASCII symbols even/odd)
    "Write a short function signature in C that returns the square of an integer.",
    "What does a hash table use to map keys to array slots?",
    "整数配列から最大値を返す関数 `max_val(a[], n)` の時間計算量は何ですか。",
    "二分探索木で要素を探索するとき、平均計算量がO(log n)になる理由を述べてください。",
    // Domain 4: Daily Conversation / Practical (EN even/odd, JA/CJK even/odd)
    "Suggest two quick ways to keep fresh herbs crisp in the refrigerator.",
    "How can someone politely follow up on an unanswered email?",
    "湿気の多い季節に革靴のカビを防ぐための簡単な保管方法を2つ挙げてください。",
    "会議の日程変更を社外の関係者へ丁寧に伝える短い案内文の要点を示してください。",
    // Domain 5: Science & Definitions (EN even/odd, JA/CJK even/odd)
    "Briefly explain how evaporation cools a liquid surface.",
    "Define the term algorithm in one clear sentence.",
    "冬に窓ガラスの内側へ結露が生じる物理的な仕組みを一文で説明してください。",
    "データ圧縮における可逆圧縮と非可逆圧縮の決定的な違いを簡潔に述べてください。",
];

/// Raw per-token, per-layer pre-RoPE query activations captured from one text sequence,
/// optionally paired with a final-layer hidden state for contrastive Rank-1 residual vectors
/// and an abstract per-token inclusion mask (`None` includes every token).
#[derive(Debug, Clone)]
pub struct CapturedTokenSequence {
    /// Number of tokens in this captured sequence (`S`).
    pub seq_len: usize,
    /// Per-layer flattened row-major token activations `[seq_len * q_rot_len]`.
    pub per_layer_token_qs: Vec<Vec<f32>>,
    /// Optional final-layer hidden representation `[hidden_size]` for Rank-1 residual distillation.
    pub final_hidden: Option<Vec<f32>>,
    /// Optional per-token inclusion mask (`len == seq_len`, `true` = include, `false` = exclude).
    /// `None` preserves full-sequence pooling with zero overhead for plain text inputs.
    pub token_include_mask: Option<Vec<bool>>,
}

impl CapturedTokenSequence {
    /// Creates a [`CapturedTokenSequence`] without a final-layer hidden state or exclusion mask.
    #[must_use]
    pub const fn new(seq_len: usize, per_layer_token_qs: Vec<Vec<f32>>) -> Self {
        Self {
            seq_len,
            per_layer_token_qs,
            final_hidden: None,
            token_include_mask: None,
        }
    }

    /// Attaches a final-layer hidden vector used when building Rank-1 residual directions.
    #[must_use]
    pub fn with_final_hidden(mut self, hidden: Vec<f32>) -> Self {
        self.final_hidden = Some(hidden);
        self
    }

    /// Attaches an abstract per-token inclusion mask (`true` = keep, `false` = exclude).
    #[must_use]
    pub fn with_token_include_mask(mut self, mask: Vec<bool>) -> Self {
        self.token_include_mask = Some(mask);
        self
    }

    /// Converts this captured sequence into a [`TokenSequenceEntry`] for corpus compilation.
    #[must_use]
    pub fn into_token_sequence_entry(
        self,
        role: AttractorRole,
        text: impl Into<String>,
        energy: f32,
    ) -> TokenSequenceEntry {
        TokenSequenceEntry {
            role,
            text: text.into(),
            energy,
            seq_len: self.seq_len,
            per_layer_token_qs: self.per_layer_token_qs,
            token_include_mask: self.token_include_mask,
        }
    }
}

/// Abstract backend encoder capturing raw per-token, per-layer pre-RoPE query activations from
/// natural-language inputs, RAG passages, or dialogue logs.
pub trait TokenSequenceEncoder {
    /// Encodes `text` into per-layer flattened `[seq_len * q_rot_len]` pre-RoPE query activations.
    ///
    /// # Errors
    /// Returns an error if tokenization or model forward capture fails.
    fn encode_token_sequence(&mut self, text: &str) -> Result<CapturedTokenSequence, String>;
}

/// Adapter bridging any [`TokenSequenceEncoder`] to [`SubtitleActivationEncoder`] via uniform
/// unit-vector sequence pooling and deterministic target-dimension resizing for cross-model rebake.
#[derive(Debug)]
pub struct PooledSubtitleEncoder<E> {
    encoder: E,
    model_rotary_dim: usize,
    target_rotary_dim: usize,
}

impl<E> PooledSubtitleEncoder<E> {
    /// Wraps `encoder` to pool captured sequences of width `model_rotary_dim` and resize each
    /// layer vector to `target_rotary_dim` (zero-padding or truncating as needed).
    #[must_use]
    pub const fn new(encoder: E, model_rotary_dim: usize, target_rotary_dim: usize) -> Self {
        Self {
            encoder,
            model_rotary_dim,
            target_rotary_dim,
        }
    }

    /// Consumes this wrapper and returns the inner encoder.
    #[must_use]
    pub fn into_inner(self) -> E {
        self.encoder
    }
}

impl<E: TokenSequenceEncoder> SubtitleActivationEncoder for PooledSubtitleEncoder<E> {
    fn encode_subtitle_activations(
        &mut self,
        record: &SourceTextRecord,
    ) -> Result<Vec<Vec<f32>>, String> {
        let captured = self.encoder.encode_token_sequence(&record.text)?;
        let mask = captured.token_include_mask.as_deref();
        let mut out = Vec::with_capacity(captured.per_layer_token_qs.len());
        for tok_flat in captured.per_layer_token_qs {
            let mut layer_q = SequenceActivationPooler::uniform_unit_mean_masked(
                &tok_flat,
                captured.seq_len,
                self.model_rotary_dim,
                mask,
            )?;
            layer_q.resize(self.target_rotary_dim, 0.0);
            out.push(layer_q);
        }
        Ok(out)
    }
}

#[inline]
fn is_alias_span_boundary(ch: char) -> bool {
    ch.is_whitespace()
        || matches!(
            ch,
            ',' | '.'
                | ':'
                | ';'
                | '!'
                | '?'
                | '('
                | ')'
                | '['
                | ']'
                | '{'
                | '}'
                | '<'
                | '>'
                | '"'
                | '\''
                | '\u{3001}' // 、
                | '\u{3002}' // 。
                | '\u{300C}' // 「
                | '\u{300D}' // 」
                | '\u{300E}' // 『
                | '\u{300F}' // 』
                | '\u{FF08}' // （
                | '\u{FF09}' // ）
                | '\u{3010}' // 【
                | '\u{3011}' // 】
                | '\u{3008}' // 〈
                | '\u{3009}' // 〉
                | '\u{300A}' // 《
                | '\u{300B}' // 》
                | '\u{30FB}' // ・
                | '\u{FF1A}' // ：
                | '\u{FF1B}' // ；
                | '\u{FF01}' // ！
                | '\u{FF1F}' // ？
                | '\u{FF0C}' // ，
                | '\u{FF0E}' // ．
                | '\u{201C}' // “
                | '\u{201D}' // ”
                | '\u{2018}' // ‘
                | '\u{2019}' // ’
                | '\u{2014}' // —
                | '\u{2026}' // …
        )
}

#[inline]
fn is_leading_alias_skip(ch: char) -> bool {
    is_alias_span_boundary(ch) || matches!(ch, '-' | '/' | '\\' | '#' | '*')
}

/// Extracts a UTF-8-safe leading non-punctuation span (`>= 2` Unicode scalar values) from `text`
/// for subword-prefix alias probing without assuming ASCII whitespace word segmentation.
///
/// Leading whitespace, brackets, quotes, and punctuation are skipped so bracketed terms such as
/// `「東京」` still yield `"東京"`. Spans terminate at ASCII or Unicode clause/bracket boundaries
/// and are bounded at the character boundary (`<= 24` chars for ASCII-only spans, `<= 8` chars
/// once any non-ASCII character appears) so unspaced CJK/Japanese sentences never collapse into
/// whole-line probes while preserving exact first-word extraction on space-delimited English text.
#[must_use]
pub fn extract_leading_alias_word(text: &str) -> Option<&str> {
    let start_byte = text
        .char_indices()
        .find(|&(_, ch)| !is_leading_alias_skip(ch))
        .map(|(idx, _)| idx)?;
    let tail = &text[start_byte..];

    let mut end_byte = tail.len();
    let mut has_non_ascii = false;
    for (char_count, (rel_byte, ch)) in tail.char_indices().enumerate() {
        if is_alias_span_boundary(ch) {
            end_byte = rel_byte;
            break;
        }
        if !ch.is_ascii() {
            has_non_ascii = true;
        }
        let max_chars = if has_non_ascii {
            MAX_NON_ASCII_LEADING_ALIAS_CHARS
        } else {
            MAX_ASCII_LEADING_ALIAS_CHARS
        };
        if char_count >= max_chars {
            end_byte = rel_byte;
            break;
        }
    }

    let span = &tail[..end_byte];
    if span.chars().count() >= 2 {
        Some(span)
    } else {
        None
    }
}

/// Optional tokenizer-aware probe capturing continuation-conditioned token activations when a
/// leading span tokenizes differently in mid-sentence context than at sequence start (e.g., BPE
/// space-prefix splits or SentencePiece word-initial marker divergence across scripts).
pub trait PrefixAliasProbe {
    /// Probes whether `word` (a UTF-8-safe leading span extracted by
    /// [`extract_leading_alias_word`]) diverges in continuation tokenization context and, if so,
    /// returns the sliced per-layer activations for the continuation alias tokens (`Ok(None)`
    /// otherwise).
    ///
    /// # Errors
    /// Returns an error if tokenization or model forward capture fails.
    fn probe_leading_alias(
        &mut self,
        word: &str,
        q_rot_len: usize,
    ) -> Result<Option<CapturedTokenSequence>, String>;
}

/// Role-tagged captured sequence supplied to [`ContrastiveAnchorBank::from_entries`].
#[derive(Debug, Clone)]
pub struct ContrastiveAnchorEntry {
    /// Polarity/origin role separating attractive (`Fact`/`Persona`) from repulsive (`Guard`) poles.
    pub role: AttractorRole,
    /// Captured per-layer token activations and optional final-layer hidden state.
    pub captured: CapturedTokenSequence,
}

/// Polarity-separated contrastive anchor bank distilled from raw per-token activations.
///
/// Attractive entries ([`AttractorRole::Fact`] and [`AttractorRole::Persona`]) are baked relative
/// to the repulsive centroid (`repulsive_centroid -> target`), while repulsive entries
/// ([`AttractorRole::Guard`]) are baked explicitly against the attractive centroid
/// (`attractive_centroid -> guard`) into [`GuardPhaseAnchor`] so negative constraints never invert
/// polarity under shared-centroid subtraction.
#[derive(Debug, Clone)]
pub struct ContrastiveAnchorBank {
    /// Combined per-layer Q7 Procrustes deltas (`repulsive_centroid -> attractive_centroid`).
    pub per_layer_q7: Vec<Vec<lmnk_q7_t>>,
    /// Fact-specific per-layer Q7 Procrustes deltas (`repulsive_centroid -> fact_centroid`).
    pub fact_q7: Vec<Vec<lmnk_q7_t>>,
    /// Persona-specific per-layer Q7 Procrustes deltas (`repulsive_centroid -> persona_centroid`).
    pub persona_q7: Vec<Vec<lmnk_q7_t>>,
    /// Per-layer repulsive (`Guard`) raw mean centroid `[layer][q_rot_len]`.
    pub repulsive_centroid_per_layer: Vec<Vec<f32>>,
    /// Per-layer attractive (`Fact` + `Persona`) raw mean centroid `[layer][q_rot_len]`.
    pub attractive_centroid_per_layer: Vec<Vec<f32>>,
    /// Per-layer attractive `ObjectiveFact` anchors baked from `repulsive_centroid -> fact`.
    pub fact_anchors_per_layer: Vec<Vec<ObjectiveFact<[lmnk_q7_t; MAX_ATTENTION_HEADS]>>>,
    /// Per-layer attractive `SubjectivePersona` anchors baked from `repulsive_centroid -> persona`.
    pub persona_anchors_per_layer: Vec<Vec<SubjectivePersona<[lmnk_q7_t; MAX_ATTENTION_HEADS]>>>,
    /// Per-layer repulsive `GuardPhaseAnchor` poles baked from `attractive_centroid -> guard`.
    pub guard_anchors_per_layer: Vec<Vec<GuardPhaseAnchor>>,
    /// Unit-normalized contrastive Rank-1 hidden direction `u` (`attractive_h - repulsive_h`), or empty.
    pub rank1_u: Vec<f32>,
    /// Unit-normalized contrastive Rank-1 hidden direction `v` (`attractive_h - repulsive_h`), or empty.
    pub rank1_v: Vec<f32>,
}

fn normalize_unit_f32(v: &mut [f32]) {
    let norm = v.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-8);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

impl ContrastiveAnchorBank {
    /// Distills a polarity-separated [`ContrastiveAnchorBank`] from role-tagged captured sequences
    /// via [`ModelSteeringAdapter::bake_layer_procrustes_q7`].
    ///
    /// # Errors
    /// Returns an error if `entries` is empty, layer counts, `q_rot_len` slices, or optional
    /// `token_include_mask` lengths disagree with `adapter.geometry()`, or provided `final_hidden`
    /// vectors have inconsistent dimensions.
    pub fn from_entries(
        adapter: &ModelSteeringAdapter,
        entries: &[ContrastiveAnchorEntry],
    ) -> Result<Self, String> {
        if entries.is_empty() {
            return Err("cannot build ContrastiveAnchorBank from empty entries".to_string());
        }
        let geom = adapter.geometry();
        let num_layers = geom.num_layers;
        let q_len = geom.q_rot_len();

        let mut fact_n = 0_usize;
        let mut persona_n = 0_usize;
        let mut guard_n = 0_usize;
        let mut hidden_size: Option<usize> = None;

        for (idx, entry) in entries.iter().enumerate() {
            if entry.captured.per_layer_token_qs.len() != num_layers {
                return Err(format!(
                    "entry {idx} layer count mismatch: expected {num_layers}, got {}",
                    entry.captured.per_layer_token_qs.len()
                ));
            }
            if let Some(ref h) = entry.captured.final_hidden {
                match hidden_size {
                    None => hidden_size = Some(h.len()),
                    Some(expected_h) if h.len() != expected_h => {
                        return Err(format!(
                            "entry {idx} hidden_size mismatch: expected {expected_h}, got {}",
                            h.len()
                        ));
                    }
                    Some(_) => {}
                }
            }
            match entry.role {
                AttractorRole::Fact => fact_n += 1,
                AttractorRole::Persona => persona_n += 1,
                AttractorRole::Guard => guard_n += 1,
            }
        }

        let fact_count = fact_n.max(1) as f32;
        let persona_count = persona_n.max(1) as f32;
        let pos_count = (fact_n + persona_n).max(1) as f32;
        let neg_count = guard_n.max(1) as f32;

        let mut fact_sum = vec![vec![0.0_f32; q_len]; num_layers];
        let mut persona_sum = vec![vec![0.0_f32; q_len]; num_layers];
        let mut pos_sum = vec![vec![0.0_f32; q_len]; num_layers];
        let mut neg_sum = vec![vec![0.0_f32; q_len]; num_layers];
        let h_dim = hidden_size.unwrap_or(0);
        let mut pos_h = vec![0.0_f32; h_dim];
        let mut neg_h = vec![0.0_f32; h_dim];

        let mut per_layer_fact_qs: Vec<Vec<Vec<f32>>> =
            vec![Vec::with_capacity(fact_n); num_layers];
        let mut per_layer_persona_qs: Vec<Vec<Vec<f32>>> =
            vec![Vec::with_capacity(persona_n); num_layers];
        let mut per_layer_guard_qs: Vec<Vec<Vec<f32>>> =
            vec![Vec::with_capacity(guard_n); num_layers];

        for entry in entries {
            let mask = entry.captured.token_include_mask.as_deref();
            match entry.role {
                AttractorRole::Fact => {
                    for (l, tok_flat) in entry.captured.per_layer_token_qs.iter().enumerate() {
                        let q = SequenceActivationPooler::uniform_raw_mean_masked(
                            tok_flat,
                            entry.captured.seq_len,
                            q_len,
                            mask,
                        )?;
                        for ((f_dst, p_dst), &src) in fact_sum[l]
                            .iter_mut()
                            .zip(pos_sum[l].iter_mut())
                            .zip(q.iter())
                        {
                            *f_dst += src / fact_count;
                            *p_dst += src / pos_count;
                        }
                        per_layer_fact_qs[l].push(q);
                    }
                    if let Some(ref h_vec) = entry.captured.final_hidden {
                        for (dst, &src) in pos_h.iter_mut().zip(h_vec.iter()) {
                            *dst += src / pos_count;
                        }
                    }
                }
                AttractorRole::Persona => {
                    for (l, tok_flat) in entry.captured.per_layer_token_qs.iter().enumerate() {
                        let q = SequenceActivationPooler::uniform_raw_mean_masked(
                            tok_flat,
                            entry.captured.seq_len,
                            q_len,
                            mask,
                        )?;
                        for ((per_dst, p_dst), &src) in persona_sum[l]
                            .iter_mut()
                            .zip(pos_sum[l].iter_mut())
                            .zip(q.iter())
                        {
                            *per_dst += src / persona_count;
                            *p_dst += src / pos_count;
                        }
                        per_layer_persona_qs[l].push(q);
                    }
                    if let Some(ref h_vec) = entry.captured.final_hidden {
                        for (dst, &src) in pos_h.iter_mut().zip(h_vec.iter()) {
                            *dst += src / pos_count;
                        }
                    }
                }
                AttractorRole::Guard => {
                    for (l, tok_flat) in entry.captured.per_layer_token_qs.iter().enumerate() {
                        let q = SequenceActivationPooler::uniform_raw_mean_masked(
                            tok_flat,
                            entry.captured.seq_len,
                            q_len,
                            mask,
                        )?;
                        for (dst, &src) in neg_sum[l].iter_mut().zip(q.iter()) {
                            *dst += src / neg_count;
                        }
                        per_layer_guard_qs[l].push(q);
                    }
                    if let Some(ref h_vec) = entry.captured.final_hidden {
                        for (dst, &src) in neg_h.iter_mut().zip(h_vec.iter()) {
                            *dst += src / neg_count;
                        }
                    }
                }
            }
        }

        let mut per_layer_q7 = Vec::with_capacity(num_layers);
        let mut fact_q7 = Vec::with_capacity(num_layers);
        let mut persona_q7 = Vec::with_capacity(num_layers);
        let mut fact_anchors_per_layer = Vec::with_capacity(num_layers);
        let mut persona_anchors_per_layer = Vec::with_capacity(num_layers);
        let mut guard_anchors_per_layer = Vec::with_capacity(num_layers);

        for l in 0..num_layers {
            per_layer_q7.push(adapter.bake_layer_procrustes_q7(&neg_sum[l], &pos_sum[l])?);
            fact_q7.push(adapter.bake_layer_procrustes_q7(&neg_sum[l], &fact_sum[l])?);
            persona_q7.push(adapter.bake_layer_procrustes_q7(&neg_sum[l], &persona_sum[l])?);

            let mut layer_facts = Vec::with_capacity(per_layer_fact_qs[l].len());
            for (fact_idx, q_fact) in per_layer_fact_qs[l].iter().enumerate() {
                let q7 = adapter.bake_layer_procrustes_q7(&neg_sum[l], q_fact)?;
                let mut arr64 = [0_i8; MAX_ATTENTION_HEADS];
                let n = q7.len().min(MAX_ATTENTION_HEADS);
                arr64[..n].copy_from_slice(&q7[..n]);
                layer_facts.push(ObjectiveFact::new(arr64, 2.5, 0, (fact_idx + 1) as u32));
            }
            fact_anchors_per_layer.push(layer_facts);

            let mut layer_personas = Vec::with_capacity(per_layer_persona_qs[l].len());
            for (per_idx, q_per) in per_layer_persona_qs[l].iter().enumerate() {
                let q7 = adapter.bake_layer_procrustes_q7(&neg_sum[l], q_per)?;
                let mut arr64 = [0_i8; MAX_ATTENTION_HEADS];
                let n = q7.len().min(MAX_ATTENTION_HEADS);
                arr64[..n].copy_from_slice(&q7[..n]);
                layer_personas.push(SubjectivePersona::new(
                    arr64,
                    1.5,
                    0,
                    (per_idx + 101) as u32,
                ));
            }
            persona_anchors_per_layer.push(layer_personas);

            let mut layer_guards = Vec::with_capacity(per_layer_guard_qs[l].len());
            for q_guard in &per_layer_guard_qs[l] {
                let q7 = adapter.bake_layer_procrustes_q7(&pos_sum[l], q_guard)?;
                let mut arr64 = [0_i8; MAX_ATTENTION_HEADS];
                let n = q7.len().min(MAX_ATTENTION_HEADS);
                arr64[..n].copy_from_slice(&q7[..n]);
                layer_guards.push(GuardPhaseAnchor::new(arr64, 0.0, 0.72));
            }
            guard_anchors_per_layer.push(layer_guards);
        }

        let (rank1_u, rank1_v) = if h_dim > 0 {
            let mut u: Vec<f32> = pos_h
                .iter()
                .zip(neg_h.iter())
                .map(|(&p, &n)| p - n)
                .collect();
            let mut v = u.clone();
            normalize_unit_f32(&mut u);
            normalize_unit_f32(&mut v);
            (u, v)
        } else {
            (Vec::new(), Vec::new())
        };

        Ok(Self {
            per_layer_q7,
            fact_q7,
            persona_q7,
            repulsive_centroid_per_layer: neg_sum,
            attractive_centroid_per_layer: pos_sum,
            fact_anchors_per_layer,
            persona_anchors_per_layer,
            guard_anchors_per_layer,
            rank1_u,
            rank1_v,
        })
    }
}
