// name: examples/candle_eval/src/bake.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixture loading, base-prior sampling, RoPE baking, and VRAM/metric helpers.

use std::path::Path;

use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device};
use liminika_adapter::{
    AttractorRole, ModelBaseAttractorProfile, ModelSteeringAdapter, SequenceActivationPooler,
    TokenSequenceEntry,
};
use liminika_core::abi::lmnk_q7_t;
use liminika_core::runtime::{GuardPhaseAnchor, ObjectiveFact, SubjectivePersona};
use liminika_core::stigmergy::StigmergicSubstrate;
use tokenizers::Tokenizer;

use crate::model::{Phi3Config, Phi3Model};

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

pub(crate) fn format_phi4_prompt(user_prompt: &str) -> String {
    format!("<|user|>{user_prompt}<|end|><|assistant|>")
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

pub(crate) fn estimate_phi3_footprint(
    cfg: &Phi3Config,
    dtype: DType,
    quant: Option<GgmlDType>,
) -> (u64, f64, f64) {
    let h = cfg.hidden_size as u64;
    let i = cfg.intermediate_size as u64;
    let n_q = cfg.num_attention_heads as u64;
    let n_kv = cfg.num_key_value_heads as u64;
    let d_head = h / n_q.max(1);
    let qkv_out = (n_q + 2 * n_kv) * d_head;
    let per_layer_params = h * qkv_out + h * h + h * (2 * i) + i * h + 2 * h;
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
    cfg: &Phi3Config,
    adapter: &ModelSteeringAdapter,
    device: &Device,
    dtype: DType,
    quant: Option<GgmlDType>,
    load_elapsed: std::time::Duration,
    vram_before: Option<&str>,
) {
    let geom = adapter.geometry();
    let (params, weights_mb, kv_2048_mb) = estimate_phi3_footprint(cfg, dtype, quant);
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
    model: &mut Phi3Model,
    tokenizer: &Tokenizer,
    adapter: &ModelSteeringAdapter,
) -> Result<ModelBaseAttractorProfile, Box<dyn std::error::Error>> {
    let probe_prompts = [
        "Explain the basic design trade-offs of a modular systems architecture.",
        "Summarize how memory alignment affects hardware cache performance.",
        "Describe step by step how an algorithm filters noisy input signals.",
        "Contrast deterministic engineering constraints with approximate heuristics.",
        "What is the boiling point of water at standard atmospheric pressure?",
        "Write a concise technical note defining a protocol boundary version.",
    ];
    let mut base_entries = Vec::with_capacity(probe_prompts.len());
    for prompt in probe_prompts {
        model.clear_cache();
        let formatted = format_phi4_prompt(prompt);
        let mut seq_ids = tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| format!("base prior encode error: {e}"))?
            .get_ids()
            .to_vec();
        let (mut logits, _, _) = model.forward_tokens(&seq_ids, 0, None, false, false)?;
        let mut gen_ids = Vec::with_capacity(8);
        for offset in (seq_ids.len()..).take(8) {
            let next = argmax(&logits);
            if next == 199999 {
                break;
            }
            gen_ids.push(next);
            let (next_logits, _, _) = model.forward_tokens(&[next], offset, None, false, false)?;
            logits = next_logits;
        }
        seq_ids.extend_from_slice(&gen_ids);
        model.clear_cache();
        let (_, layer_tok_qs, _) = model.forward_tokens(&seq_ids, 0, None, false, true)?;
        base_entries.push(TokenSequenceEntry {
            role: AttractorRole::Persona,
            text: prompt.to_string(),
            energy: 1.0,
            seq_len: seq_ids.len(),
            per_layer_token_qs: layer_tok_qs,
        });
    }
    ModelBaseAttractorProfile::extract(
        adapter,
        &base_entries,
        SequenceActivationPooler::default(),
        0.55,
    )
    .map_err(|e| e.into())
}

pub(crate) struct BakedModelProfile {
    pub(crate) per_layer_q7: Vec<Vec<lmnk_q7_t>>,
    pub(crate) fact_q7: Vec<Vec<lmnk_q7_t>>,
    pub(crate) persona_q7: Vec<Vec<lmnk_q7_t>>,
    pub(crate) neg_sum: Vec<Vec<f32>>,
    pub(crate) pos_sum: Vec<Vec<f32>>,
    pub(crate) fact_anchors_per_layer: Vec<Vec<ObjectiveFact<[i8; 64]>>>,
    pub(crate) persona_anchors_per_layer: Vec<Vec<SubjectivePersona<[i8; 64]>>>,
    pub(crate) guard_anchors_per_layer: Vec<Vec<GuardPhaseAnchor>>,
    pub(crate) srl_u: Vec<f32>,
    pub(crate) srl_v: Vec<f32>,
}

pub(crate) fn normalize_unit(v: &mut [f32]) {
    let norm = v.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-8);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

pub(crate) fn bake_via_adapter(
    model: &mut Phi3Model,
    tokenizer: &Tokenizer,
    adapter: &ModelSteeringAdapter,
    hidden_size: usize,
    suite: &FixtureSuite,
) -> Result<BakedModelProfile, Box<dyn std::error::Error>> {
    let num_layers = adapter.geometry().num_layers;
    let q_len = adapter.geometry().q_rot_len();
    let mut fact_sum = vec![vec![0.0_f32; q_len]; num_layers];
    let mut persona_sum = vec![vec![0.0_f32; q_len]; num_layers];
    let mut pos_sum = vec![vec![0.0_f32; q_len]; num_layers];
    let mut neg_sum = vec![vec![0.0_f32; q_len]; num_layers];
    let mut pos_h = vec![0.0_f32; hidden_size];
    let mut neg_h = vec![0.0_f32; hidden_size];

    let fact_count = suite.facts.len().max(1) as f32;
    let persona_count = suite.persona.len().max(1) as f32;
    let pos_count = (suite.facts.len() + suite.persona.len()).max(1) as f32;
    let neg_count = suite.guards.len().max(1) as f32;

    let mut per_layer_fact_qs: Vec<Vec<Vec<f32>>> =
        vec![Vec::with_capacity(suite.facts.len()); num_layers];
    let mut per_layer_persona_qs: Vec<Vec<Vec<f32>>> =
        vec![Vec::with_capacity(suite.persona.len()); num_layers];
    let mut per_layer_guard_qs: Vec<Vec<Vec<f32>>> =
        vec![Vec::with_capacity(suite.guards.len()); num_layers];

    for text in &suite.facts {
        model.clear_cache();
        let ids = tokenizer
            .encode(text.as_str(), false)
            .map_err(|e| format!("encode error: {e}"))?
            .get_ids()
            .to_vec();
        let (_, layer_tok_qs, h_vec) = model.forward_tokens(&ids, 0, None, false, true)?;
        for (l, tok_flat) in layer_tok_qs.iter().enumerate() {
            let q = SequenceActivationPooler::uniform_raw_mean(tok_flat, ids.len(), q_len)?;
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
        for (dst, &src) in pos_h.iter_mut().zip(h_vec.iter()) {
            *dst += src / pos_count;
        }
    }

    for text in &suite.persona {
        model.clear_cache();
        let ids = tokenizer
            .encode(text.as_str(), false)
            .map_err(|e| format!("encode error: {e}"))?
            .get_ids()
            .to_vec();
        let (_, layer_tok_qs, h_vec) = model.forward_tokens(&ids, 0, None, false, true)?;
        for (l, tok_flat) in layer_tok_qs.iter().enumerate() {
            let q = SequenceActivationPooler::uniform_raw_mean(tok_flat, ids.len(), q_len)?;
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
        for (dst, &src) in pos_h.iter_mut().zip(h_vec.iter()) {
            *dst += src / pos_count;
        }
    }

    for guard in &suite.guards {
        model.clear_cache();
        let ids = tokenizer
            .encode(guard.as_str(), false)
            .map_err(|e| format!("encode error: {e}"))?
            .get_ids()
            .to_vec();
        let (_, layer_tok_qs, h_vec) = model.forward_tokens(&ids, 0, None, false, true)?;
        for (l, tok_flat) in layer_tok_qs.iter().enumerate() {
            let q = SequenceActivationPooler::uniform_raw_mean(tok_flat, ids.len(), q_len)?;
            for (dst, &src) in neg_sum[l].iter_mut().zip(q.iter()) {
                *dst += src / neg_count;
            }
            per_layer_guard_qs[l].push(q);
        }
        for (dst, &src) in neg_h.iter_mut().zip(h_vec.iter()) {
            *dst += src / neg_count;
        }
    }

    let mut per_layer_q7 = Vec::with_capacity(num_layers);
    let mut fact_q7 = Vec::with_capacity(num_layers);
    let mut persona_q7 = Vec::with_capacity(num_layers);
    let mut fact_anchors_per_layer = Vec::with_capacity(num_layers);
    let mut persona_anchors_per_layer = Vec::with_capacity(num_layers);
    let mut guard_anchors_per_layer = Vec::with_capacity(num_layers);
    for l in 0..num_layers {
        out_push_q7(&mut per_layer_q7, adapter, &neg_sum[l], &pos_sum[l])?;
        out_push_q7(&mut fact_q7, adapter, &neg_sum[l], &fact_sum[l])?;
        out_push_q7(&mut persona_q7, adapter, &neg_sum[l], &persona_sum[l])?;

        let mut layer_anchors = Vec::with_capacity(per_layer_fact_qs[l].len());
        for (fact_idx, q_fact) in per_layer_fact_qs[l].iter().enumerate() {
            let q7 = adapter.bake_layer_procrustes_q7(&neg_sum[l], q_fact)?;
            let mut arr64 = [0i8; 64];
            let n = q7.len().min(64);
            arr64[..n].copy_from_slice(&q7[..n]);
            layer_anchors.push(ObjectiveFact::new(arr64, 2.5, 0, (fact_idx + 1) as u32));
        }
        fact_anchors_per_layer.push(layer_anchors);

        let mut layer_personas = Vec::with_capacity(per_layer_persona_qs[l].len());
        for (per_idx, q_per) in per_layer_persona_qs[l].iter().enumerate() {
            let q7 = adapter.bake_layer_procrustes_q7(&neg_sum[l], q_per)?;
            let mut arr64 = [0i8; 64];
            let n = q7.len().min(64);
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
            let mut arr64 = [0i8; 64];
            let n = q7.len().min(64);
            arr64[..n].copy_from_slice(&q7[..n]);
            layer_guards.push(GuardPhaseAnchor::new(arr64, 0.0, 0.72));
        }
        guard_anchors_per_layer.push(layer_guards);
    }

    let mut srl_u: Vec<f32> = pos_h
        .iter()
        .zip(neg_h.iter())
        .map(|(&p, &n)| p - n)
        .collect();
    let mut srl_v = srl_u.clone();
    normalize_unit(&mut srl_u);
    normalize_unit(&mut srl_v);

    Ok(BakedModelProfile {
        per_layer_q7,
        fact_q7,
        persona_q7,
        neg_sum,
        pos_sum,
        fact_anchors_per_layer,
        persona_anchors_per_layer,
        guard_anchors_per_layer,
        srl_u,
        srl_v,
    })
}

pub(crate) fn out_push_q7(
    out: &mut Vec<Vec<lmnk_q7_t>>,
    adapter: &ModelSteeringAdapter,
    neg: &[f32],
    pos: &[f32],
) -> Result<(), String> {
    out.push(adapter.bake_layer_procrustes_q7(neg, pos)?);
    Ok(())
}
