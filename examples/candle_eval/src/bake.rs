// name: examples/candle_eval/src/bake.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixture loading, base-prior sampling, RoPE baking, and VRAM/metric helpers.

use std::path::Path;
use std::time::Instant;

use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device};
use liminika_adapter::{
    AttractorRole, CapturedTokenSequence, CompiledTokenCorpus, ContrastiveAnchorBank,
    ContrastiveAnchorEntry, ModelBaseAttractorProfile, ModelSteeringAdapter, PrefixAliasProbe,
    SequenceActivationPooler, TokenSequenceEncoder, TokenSequenceEntry,
    DEFAULT_STARTUP_PRIOR_PROBES,
};
use liminika_core::stigmergy::StigmergicSubstrate;
use tokenizers::Tokenizer;

use crate::model::{CausalLmModel, ChatTemplate, KvCacheMode, LogitsFetchMode, ModelConfig};

pub(crate) type BakedModelProfile = ContrastiveAnchorBank;

pub(crate) struct LmTokenEncoder<'a> {
    model: &'a mut CausalLmModel,
    tokenizer: &'a Tokenizer,
    chat: &'a ChatTemplate,
    format_as_chat_prompt: bool,
    capture_final_hidden: bool,
}

impl<'a> LmTokenEncoder<'a> {
    pub(crate) fn new(
        model: &'a mut CausalLmModel,
        tokenizer: &'a Tokenizer,
        chat: &'a ChatTemplate,
    ) -> Self {
        Self {
            model,
            tokenizer,
            chat,
            format_as_chat_prompt: false,
            capture_final_hidden: false,
        }
    }

    pub(crate) const fn with_chat_prompt(mut self, enabled: bool) -> Self {
        self.format_as_chat_prompt = enabled;
        self
    }

    pub(crate) const fn with_final_hidden(mut self, enabled: bool) -> Self {
        self.capture_final_hidden = enabled;
        self
    }
}

impl TokenSequenceEncoder for LmTokenEncoder<'_> {
    fn encode_token_sequence(&mut self, text: &str) -> Result<CapturedTokenSequence, String> {
        let owned_prompt;
        let input_text = if self.format_as_chat_prompt {
            owned_prompt = self.chat.format_single_turn(text);
            owned_prompt.as_str()
        } else {
            text
        };
        let ids = self
            .tokenizer
            .encode(input_text, false)
            .map_err(|e| format!("encode error: {e}"))?
            .get_ids()
            .to_vec();
        let (_, layer_tok_qs, h_vec) = self
            .model
            .forward_tokens(
                &ids,
                0,
                None,
                false,
                true,
                LogitsFetchMode::FullLogits,
                KvCacheMode::IsolatedProbe,
            )
            .map_err(|e| format!("forward error: {e}"))?;
        let captured = CapturedTokenSequence::new(ids.len(), layer_tok_qs);
        if self.capture_final_hidden {
            Ok(captured.with_final_hidden(h_vec))
        } else {
            Ok(captured)
        }
    }
}

impl PrefixAliasProbe for LmTokenEncoder<'_> {
    fn probe_leading_alias(
        &mut self,
        word: &str,
        q_rot_len: usize,
    ) -> Result<Option<CapturedTokenSequence>, String> {
        let prefix_ids = self
            .tokenizer
            .encode("Item:", false)
            .map_err(|e| format!("alias prefix encode error: {e}"))?
            .get_ids()
            .to_vec();
        let ids_raw = self
            .tokenizer
            .encode(word, false)
            .map_err(|e| format!("alias raw encode error: {e}"))?
            .get_ids()
            .to_vec();
        let spaced_word = format!(" {word}");
        let ids_spaced = self
            .tokenizer
            .encode(spaced_word.as_str(), false)
            .map_err(|e| format!("alias spaced encode error: {e}"))?
            .get_ids()
            .to_vec();
        let disjoint_bpe_split = !ids_raw.is_empty()
            && !ids_spaced.is_empty()
            && ids_raw.iter().all(|id| !ids_spaced.contains(id));
        if !disjoint_bpe_split {
            return Ok(None);
        }
        let mut full_ids = prefix_ids.clone();
        full_ids.extend_from_slice(&ids_spaced);
        let (_, alias_tok_qs, _) = self
            .model
            .forward_tokens(
                &full_ids,
                0,
                None,
                false,
                true,
                LogitsFetchMode::FullLogits,
                KvCacheMode::IsolatedProbe,
            )
            .map_err(|e| format!("alias forward error: {e}"))?;
        let start = prefix_ids.len() * q_rot_len;
        let end = full_ids.len() * q_rot_len;
        let sliced_qs: Vec<Vec<f32>> = alias_tok_qs
            .into_iter()
            .map(|layer| layer[start..end].to_vec())
            .collect();
        Ok(Some(CapturedTokenSequence::new(
            ids_spaced.len(),
            sliced_qs,
        )))
    }
}

pub(crate) struct FixtureSuite {
    pub(crate) facts: Vec<String>,
    pub(crate) persona: Vec<String>,
    pub(crate) guards: Vec<String>,
    pub(crate) probes: Vec<(String, String, Vec<String>, Vec<String>)>,
}

pub(crate) fn load_fixture(path: &Path) -> Result<FixtureSuite, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    let mut suite = FixtureSuite {
        facts: Vec::new(),
        persona: Vec::new(),
        guards: Vec::new(),
        probes: Vec::new(),
    };
    let mut section = "";
    for line in raw.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        match line {
            "[FACT_ATTRACTORS]" => section = "FACT",
            "[PERSONA_ATTRACTORS]" => section = "PERSONA",
            "[CONTRASTIVE_GUARDS]" => section = "GUARD",
            "[EVAL_PROBES]" => section = "PROBE",
            _ => match section {
                "FACT" => suite.facts.push(line.to_string()),
                "PERSONA" => suite.persona.push(line.to_string()),
                "GUARD" => suite.guards.push(line.to_string()),
                "PROBE" => {
                    let parts: Vec<&str> = line.split('|').map(str::trim).collect();
                    if parts.len() == 4 {
                        let split_kw = |s: &str| -> Vec<String> {
                            s.split(',')
                                .map(str::trim)
                                .filter(|k| !k.is_empty())
                                .map(ToString::to_string)
                                .collect()
                        };
                        suite.probes.push((
                            parts[0].to_string(),
                            parts[1].to_string(),
                            split_kw(parts[2]),
                            split_kw(parts[3]),
                        ));
                    }
                }
                _ => {}
            },
        }
    }
    Ok(suite)
}

pub(crate) fn kl_div(p_logits: &[f32], q_logits: &[f32]) -> f32 {
    let softmax = |v: &[f32]| -> Vec<f32> {
        let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = v.iter().map(|&x| (x - m).exp()).collect();
        let sum: f32 = exps.iter().sum::<f32>().max(1e-12);
        exps.into_iter().map(|e| e / sum).collect()
    };
    let p = softmax(p_logits);
    let q = softmax(q_logits);
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            let ps = pi.max(1e-8);
            let qs = qi.max(1e-8);
            ps * (ps / qs).ln()
        })
        .sum()
}

pub(crate) fn token_logprob(logits: &[f32], tok_id: u32) -> f32 {
    let idx = tok_id as usize;
    if idx >= logits.len() {
        return 0.0;
    }
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp = m + logits.iter().map(|&x| (x - m).exp()).sum::<f32>().ln();
    logits[idx] - log_sum_exp
}

pub(crate) fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(idx, _)| idx as u32)
}

pub(crate) fn detect_simd_label() -> &'static str {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return "avx2+fma+swar";
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return "avx2+swar";
        }
    }
    "portable-swar"
}

pub(crate) fn query_nvidia_smi_vram() -> Option<String> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .trim()
        .to_string();
    if line.is_empty() {
        None
    } else {
        Some(line)
    }
}

pub(crate) fn estimate_model_footprint(
    cfg: &ModelConfig,
    dtype: DType,
    quant: Option<GgmlDType>,
) -> (u64, f64, f64) {
    let h = cfg.hidden_size as u64;
    let i = cfg.intermediate_size as u64;
    let n_q = cfg.num_attention_heads as u64;
    let n_kv = cfg.num_key_value_heads as u64;
    let d_head = cfg.head_dim() as u64;
    let q_dim = n_q * d_head;
    let qkv_out = (n_q + 2 * n_kv) * d_head;
    let per_layer_params = h * qkv_out + q_dim * h + h * (2 * i) + i * h + 2 * h;
    let embed_params = (cfg.vocab_size as u64) * h;
    let lm_head_params = if cfg.tie_word_embeddings {
        0
    } else {
        embed_params
    };
    let total_params =
        embed_params + lm_head_params + (cfg.num_hidden_layers as u64) * per_layer_params + h;
    let elem_bytes = dtype.size_in_bytes() as f64;
    let weights_mb = match quant {
        None => (total_params as f64 * elem_bytes) / (1024.0 * 1024.0),
        Some(q) => {
            let bpp = (q.type_size() as f64) / (q.block_size().max(1) as f64);
            let quantized_params = embed_params + (cfg.num_hidden_layers as u64) * per_layer_params;
            (quantized_params as f64 * bpp) / (1024.0 * 1024.0)
        }
    };
    // KV cache footprint at full 2048 context window (K + V across all layers).
    let kv_2048_bytes = 2.0
        * (cfg.num_hidden_layers as f64)
        * 2048.0
        * (n_kv as f64)
        * (d_head as f64)
        * elem_bytes;
    let kv_2048_mb = kv_2048_bytes / (1024.0 * 1024.0);
    (total_params, weights_mb, kv_2048_mb)
}

pub(crate) fn print_runtime_profile(
    cfg: &ModelConfig,
    adapter: &ModelSteeringAdapter,
    device: &Device,
    dtype: DType,
    quant: Option<GgmlDType>,
    load_elapsed: std::time::Duration,
    vram_before: Option<&str>,
) {
    let geom = adapter.geometry();
    let (params, weights_mb, kv_2048_mb) = estimate_model_footprint(cfg, dtype, quant);
    let hub_bytes = std::mem::size_of::<StigmergicSubstrate>();
    println!(
        "[System] os={} arch={} simd={} device={:?} dtype={:?} quant={:?} load={:.2?}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        detect_simd_label(),
        device,
        dtype,
        quant,
        load_elapsed,
    );
    println!(
        "[Model ] params={:.2}M weights_est={:.1}MiB kv2048_est={:.1}MiB layers={} heads={}/{} head_dim={} rotary_dim={} sidecar_substrate={}B",
        params as f64 / 1_000_000.0,
        weights_mb,
        kv_2048_mb,
        geom.num_layers,
        geom.num_q_heads,
        geom.num_kv_heads,
        geom.head_dim,
        geom.rotary_dim,
        hub_bytes,
    );
    let vram_after = query_nvidia_smi_vram();
    match (vram_before, vram_after) {
        (Some(before), Some(after)) => {
            println!("[VRAM  ] gpu_before=[{before}] gpu_after=[{after}]");
        }
        (None, Some(after)) => println!("[VRAM  ] gpu=[{after}]"),
        (Some(before), None) => println!("[VRAM  ] gpu=[{before}]"),
        (None, None) => println!("[VRAM  ] nvidia-smi unavailable"),
    }
    if !device.is_cuda() {
        println!(
            "[Exec  ] Running on CPU exact path (pass `--features cuda` on CUDA-enabled toolchains for GPU execution)."
        );
    }
}

pub(crate) fn sample_startup_base_prior(
    model: &mut CausalLmModel,
    tokenizer: &Tokenizer,
    chat: &ChatTemplate,
    adapter: &ModelSteeringAdapter,
) -> Result<ModelBaseAttractorProfile, Box<dyn std::error::Error>> {
    // Generation-free 1-pass IsolatedProbe capture across neutral startup prompts matching the
    // raw query tokenization frame used by `CompiledTokenCorpus` and `TurnOrchestrator`.
    let mut encoder = LmTokenEncoder::new(model, tokenizer, chat);
    let mut base_entries = Vec::with_capacity(DEFAULT_STARTUP_PRIOR_PROBES.len());
    for prompt in DEFAULT_STARTUP_PRIOR_PROBES {
        let captured = encoder.encode_token_sequence(prompt)?;
        base_entries.push(captured.into_token_sequence_entry(AttractorRole::Persona, prompt, 1.0));
    }
    ModelBaseAttractorProfile::extract(
        adapter,
        &base_entries,
        SequenceActivationPooler::default(),
        0.55,
    )
    .map_err(|e| e.into())
}

pub(crate) struct SharedFixtureArtifacts {
    pub(crate) baked_opt: Option<BakedModelProfile>,
    pub(crate) compiled: CompiledTokenCorpus,
    pub(crate) bake_elapsed: std::time::Duration,
    pub(crate) compile_ms: f64,
    pub(crate) prior_ms: f64,
    pub(crate) used_base_prior: bool,
}

/// Captures the fixture entries in a single model forward pass and builds both the optional
/// [`ContrastiveAnchorBank`] (when `!recall_only`) and the shared [`CompiledTokenCorpus`],
/// dropping raw token activation buffers immediately after compilation to keep peak RAM low.
// Evaluation-harness corpus baking entrypoint keeps explicit flat parameters for call-site clarity.
#[allow(clippy::too_many_arguments)]
pub(crate) fn capture_and_build_shared_corpus(
    model: &mut CausalLmModel,
    tokenizer: &Tokenizer,
    chat: &ChatTemplate,
    adapter: &mut ModelSteeringAdapter,
    suite: &FixtureSuite,
    layer_mask: u64,
    recall_only: bool,
    use_base_prior: bool,
) -> Result<SharedFixtureArtifacts, Box<dyn std::error::Error>> {
    adapter.set_layer_rope_mask(layer_mask);
    let (base_prior, prior_ms) = if use_base_prior {
        let t_prior = Instant::now();
        let profile = sample_startup_base_prior(model, tokenizer, chat, adapter)?;
        let elapsed_ms = t_prior.elapsed().as_secs_f64() * 1000.0;
        println!(
            "[BasePrior] 1-pass IsolatedProbe samples={} sub_attractors={} pooled_basis={} token_basis={} elapsed={:.1}ms",
            profile.sample_count,
            profile.sub_attractors.len(),
            profile.pooled_common_mode_basis.len(),
            profile.token_common_mode_basis.len(),
            elapsed_ms
        );
        (Some(profile), elapsed_ms)
    } else {
        (None, 0.0)
    };

    let t_capture = Instant::now();
    let total = suite.facts.len() + suite.persona.len() + suite.guards.len();
    let mut contrastive_entries: Vec<ContrastiveAnchorEntry> = Vec::with_capacity(total);
    let mut entry_meta: Vec<(&str, f32)> = Vec::with_capacity(total);
    {
        let mut encoder =
            LmTokenEncoder::new(model, tokenizer, chat).with_final_hidden(!recall_only);
        for (role, texts, energy) in [
            (AttractorRole::Fact, suite.facts.as_slice(), 2.5_f32),
            (AttractorRole::Persona, suite.persona.as_slice(), 1.5_f32),
            (AttractorRole::Guard, suite.guards.as_slice(), 2.0_f32),
        ] {
            for text in texts {
                let captured = encoder.encode_token_sequence(text.as_str())?;
                contrastive_entries.push(ContrastiveAnchorEntry { role, captured });
                entry_meta.push((text.as_str(), energy));
            }
        }
    }

    let baked_opt = if !recall_only {
        Some(ContrastiveAnchorBank::from_entries(
            adapter,
            &contrastive_entries,
        )?)
    } else {
        None
    };
    let bake_elapsed = t_capture.elapsed();

    let t_compile = Instant::now();
    // Move captured per-layer Q buffers without cloning and drop `final_hidden` immediately.
    let staged_entries: Vec<TokenSequenceEntry> = contrastive_entries
        .into_iter()
        .zip(entry_meta)
        .map(|(ce, (text, energy))| ce.captured.into_token_sequence_entry(ce.role, text, energy))
        .collect();

    let mut compiled = CompiledTokenCorpus::compile_with_base_prior(
        adapter,
        &staged_entries,
        SequenceActivationPooler::default(),
        2,
        1,
        base_prior.as_ref(),
    )?;
    {
        let mut alias_encoder = LmTokenEncoder::new(model, tokenizer, chat);
        compiled.append_leading_alias_observations(adapter, &staged_entries, &mut alias_encoder)?;
    }
    drop(staged_entries);
    let compile_ms = (bake_elapsed + t_compile.elapsed()).as_secs_f64() * 1000.0;

    Ok(SharedFixtureArtifacts {
        baked_opt,
        compiled,
        bake_elapsed,
        compile_ms,
        prior_ms,
        used_base_prior: base_prior.is_some(),
    })
}
