// name: examples/candle_eval/src/main.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Standalone Candle + Safetensors evaluation harness for real-LLM RoPE steering.
//!
//! Loads `Phi-4-mini-instruct` (`Phi3ForCausalLM`) via mmaped safetensors, extracts
//! post-RoPE query representations on `tests/fixtures/gcso_eval_samples.txt` to bake
//! an in-memory phase profile, and runs an angle-sweep benchmark (`0.00..=0.25 rad`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Module, Result as CResult, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};
use serde::Deserialize;
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize)]
struct Phi3Config {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    #[serde(default = "default_partial_rotary")]
    partial_rotary_factor: f64,
    #[serde(default)]
    tie_word_embeddings: bool,
}

fn default_partial_rotary() -> f64 {
    1.0
}

impl Phi3Config {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn rotary_dim(&self) -> usize {
        let raw = (self.head_dim() as f64 * self.partial_rotary_factor).round() as usize;
        raw - (raw % 2)
    }
}

struct RotaryCache {
    cos: Tensor,
    sin: Tensor,
    rotary_dim: usize,
}

impl RotaryCache {
    fn new(cfg: &Phi3Config, max_seq: usize, dtype: DType, dev: &Device) -> CResult<Self> {
        let rot_dim = cfg.rotary_dim();
        let half = rot_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0_f32 / (cfg.rope_theta as f32).powf((2 * i) as f32 / rot_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), dev)?;
        let pos: Vec<f32> = (0..max_seq).map(|p| p as f32).collect();
        let pos = Tensor::from_vec(pos, (max_seq, 1), dev)?;
        let freqs = pos.matmul(&inv_freq)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        Ok(Self {
            cos,
            sin,
            rotary_dim: rot_dim,
        })
    }

    /// Applies HalfSplit RoPE on `0..rotary_dim` and optional LiminiKa phase steering on `q`.
    fn apply_with_steering(
        &self,
        x: &Tensor,
        offset: usize,
        steer_cos_sin: Option<(&Tensor, &Tensor)>,
    ) -> CResult<Tensor> {
        let (_b, _h, seq_len, head_dim) = x.dims4()?;
        let rot = self.rotary_dim;
        let half = rot / 2;

        let x_rot = x.narrow(D::Minus1, 0, rot)?;
        let x1 = x_rot.narrow(D::Minus1, 0, half)?;
        let x2 = x_rot.narrow(D::Minus1, half, half)?;

        let cos = self.cos.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = self.sin.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;

        let mut r1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let mut r2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

        if let Some((sc, ss)) = steer_cos_sin {
            let sr1 = (r1.broadcast_mul(sc)? - r2.broadcast_mul(ss)?)?;
            let sr2 = (r1.broadcast_mul(ss)? + r2.broadcast_mul(sc)?)?;
            r1 = sr1;
            r2 = sr2;
        }

        if rot < head_dim {
            let x_pass = x.narrow(D::Minus1, rot, head_dim - rot)?;
            Tensor::cat(&[&r1, &r2, &x_pass], D::Minus1)
        } else {
            Tensor::cat(&[&r1, &r2], D::Minus1)
        }
    }
}

struct Phi3Attention {
    qkv_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Phi3Attention {
    fn load(cfg: &Phi3Config, vb: VarBuilder) -> CResult<Self> {
        let head_dim = cfg.head_dim();
        let q_size = cfg.num_attention_heads * head_dim;
        let kv_size = cfg.num_key_value_heads * head_dim;
        let qkv_proj = linear_no_bias(cfg.hidden_size, q_size + 2 * kv_size, vb.pp("qkv_proj"))?;
        let o_proj = linear_no_bias(q_size, cfg.hidden_size, vb.pp("o_proj"))?;
        Ok(Self {
            qkv_proj,
            o_proj,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            kv_cache: None,
        })
    }

    fn clear_cache(&mut self) {
        self.kv_cache = None;
    }

    fn forward(
        &mut self,
        x: &Tensor,
        rope: &RotaryCache,
        offset: usize,
        steer_cos_sin: Option<(&Tensor, &Tensor)>,
        collect_q_summary: bool,
    ) -> CResult<(Tensor, Option<Vec<f32>>)> {
        let (b, seq_len, _) = x.dims3()?;
        let qkv = self.qkv_proj.forward(x)?;
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        let q = qkv
            .narrow(D::Minus1, 0, q_dim)?
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = qkv
            .narrow(D::Minus1, q_dim, kv_dim)?
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = qkv
            .narrow(D::Minus1, q_dim + kv_dim, kv_dim)?
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = rope.apply_with_steering(&q, offset, steer_cos_sin)?;
        let k = rope.apply_with_steering(&k, offset, None)?;

        let q_summary = if collect_q_summary {
            let last_q = q
                .i((0, .., seq_len - 1, 0..rope.rotary_dim))?
                .mean(0)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            Some(last_q)
        } else {
            None
        };

        let (k, v) = match &self.kv_cache {
            Some((prev_k, prev_v)) => (
                Tensor::cat(&[prev_k, &k], 2)?,
                Tensor::cat(&[prev_v, &v], 2)?,
            ),
            None => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut att = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        if seq_len > 1 {
            let mask = causal_mask(seq_len, offset, x.dtype(), x.device())?;
            att = att.broadcast_add(&mask)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&att)?;
        let ctx = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, seq_len, q_dim))?;
        Ok((self.o_proj.forward(&ctx)?, q_summary))
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> CResult<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, n_kv, s, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, n_kv, n_rep, s, d))?
        .reshape((b, n_kv * n_rep, s, d))
}

fn causal_mask(seq_len: usize, offset: usize, dtype: DType, dev: &Device) -> CResult<Tensor> {
    let total = offset + seq_len;
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            let pos = offset + i;
            (0..total).map(move |j| if j <= pos { 0.0 } else { f32::NEG_INFINITY })
        })
        .collect();
    Tensor::from_vec(mask, (1, 1, seq_len, total), dev)?.to_dtype(dtype)
}

struct Phi3Mlp {
    gate_up_proj: Linear,
    down_proj: Linear,
    intermediate_size: usize,
}

impl Phi3Mlp {
    fn load(cfg: &Phi3Config, vb: VarBuilder) -> CResult<Self> {
        let gate_up_proj =
            linear_no_bias(cfg.hidden_size, 2 * cfg.intermediate_size, vb.pp("gate_up_proj"))?;
        let down_proj = linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size: cfg.intermediate_size,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let up = self.gate_up_proj.forward(x)?;
        let gate = up.narrow(D::Minus1, 0, self.intermediate_size)?;
        let val = up.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        let act = (candle_nn::ops::silu(&gate)? * val)?;
        self.down_proj.forward(&act)
    }
}

struct Phi3Layer {
    input_layernorm: RmsNorm,
    self_attn: Phi3Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Phi3Mlp,
}

impl Phi3Layer {
    fn load(cfg: &Phi3Config, vb: VarBuilder) -> CResult<Self> {
        Ok(Self {
            input_layernorm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            self_attn: Phi3Attention::load(cfg, vb.pp("self_attn"))?,
            post_attention_layernorm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Phi3Mlp::load(cfg, vb.pp("mlp"))?,
        })
    }
}

struct Phi3Model {
    embed_tokens: Embedding,
    layers: Vec<Phi3Layer>,
    norm: RmsNorm,
    lm_head: Linear,
    rope: RotaryCache,
    device: Device,
    dtype: DType,
}

impl Phi3Model {
    fn load(model_dir: &Path, cfg: &Phi3Config, dev: &Device) -> Result<Self, Box<dyn std::error::Error>> {
        let dtype = if dev.is_cuda() { DType::BF16 } else { DType::F32 };
        let files = vec![
            model_dir.join("model-00001-of-00002.safetensors"),
            model_dir.join("model-00002-of-00002.safetensors"),
        ];
        // SAFETY: The safetensors files are read-only model weights on disk.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, dev)? };
        let mvb = vb.pp("model");

        let embed_weight = mvb.pp("embed_tokens").get((cfg.vocab_size, cfg.hidden_size), "weight")?;
        let embed_tokens = Embedding::new(embed_weight.clone(), cfg.hidden_size);
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_weight, None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Phi3Layer::load(cfg, mvb.pp(format!("layers.{i}")))?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, mvb.pp("norm"))?;
        let rope = RotaryCache::new(cfg, 2048, dtype, dev)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            device: dev.clone(),
            dtype,
        })
    }

    fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.self_attn.clear_cache();
        }
    }

    fn forward_tokens(
        &mut self,
        tokens: &[u32],
        offset: usize,
        steer_cos_sin: Option<(&Tensor, &Tensor)>,
        collect_q_summary: bool,
    ) -> CResult<(Vec<f32>, Option<Vec<f32>>)> {
        let input = Tensor::from_slice(tokens, (1, tokens.len()), &self.device)?;
        let mut h = self.embed_tokens.forward(&input)?;
        let rot_dim = self.rope.rotary_dim;
        let mut q_accum = if collect_q_summary {
            vec![0.0_f32; rot_dim]
        } else {
            Vec::new()
        };

        for layer in &mut self.layers {
            let normed = layer.input_layernorm.forward(&h)?;
            let (attn_out, q_sum) = layer.self_attn.forward(
                &normed,
                &self.rope,
                offset,
                steer_cos_sin,
                collect_q_summary,
            )?;
            if let Some(qs) = q_sum {
                for (d, s) in q_accum.iter_mut().zip(qs.iter()) {
                    *d += s;
                }
            }
            h = (h + attn_out)?;
            let post = layer.post_attention_layernorm.forward(&h)?;
            let mlp_out = layer.mlp.forward(&post)?;
            h = (h + mlp_out)?;
        }

        if collect_q_summary && !self.layers.is_empty() {
            let inv = 1.0 / self.layers.len() as f32;
            for v in &mut q_accum {
                *v *= inv;
            }
        }

        let h = self.norm.forward(&h)?;
        let last = h.i((0, tokens.len() - 1, ..))?.unsqueeze(0)?;
        let logits = self
            .lm_head
            .forward(&last)?
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        Ok((
            logits,
            if collect_q_summary {
                Some(q_accum)
            } else {
                None
            },
        ))
    }

    fn make_steer_tensors(
        &self,
        profile: &[f32],
        max_rad: f32,
    ) -> CResult<Option<(Tensor, Tensor)>> {
        if max_rad <= 0.0 {
            return Ok(None);
        }
        let half = self.rope.rotary_dim / 2;
        let cos_vals: Vec<f32> = (0..half)
            .map(|i| (profile[i] * max_rad).cos())
            .collect();
        let sin_vals: Vec<f32> = (0..half)
            .map(|i| (profile[i] * max_rad).sin())
            .collect();
        let c = Tensor::from_vec(cos_vals, (1, 1, 1, half), &self.device)?.to_dtype(self.dtype)?;
        let s = Tensor::from_vec(sin_vals, (1, 1, 1, half), &self.device)?.to_dtype(self.dtype)?;
        Ok(Some((c, s)))
    }
}

struct FixtureSuite {
    facts: Vec<String>,
    persona: Vec<String>,
    guards: Vec<String>,
    probes: Vec<(String, String, Vec<String>, Vec<String>)>,
}

fn load_fixture(path: &Path) -> Result<FixtureSuite, Box<dyn std::error::Error>> {
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

fn kl_div(p_logits: &[f32], q_logits: &[f32]) -> f32 {
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

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(idx, _)| idx as u32)
}

fn format_phi4_prompt(user_prompt: &str) -> String {
    format!("<|user|>{user_prompt}<|end|><|assistant|>")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let model_dir = repo_root.join("models/Phi-4-mini-instruct");
    let fixture_path = repo_root.join("tests/fixtures/gcso_eval_samples.txt");

    let cfg: Phi3Config = serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json"))?)?;
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| format!("tokenizer load error: {e}"))?;
    let suite = load_fixture(&fixture_path)?;

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!(
        "LiminiKa Candle Eval | device={:?} | layers={} heads={}/{} head_dim={} rotary_dim={}",
        device,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim(),
        cfg.rotary_dim()
    );

    let t0 = Instant::now();
    let mut model = Phi3Model::load(&model_dir, &cfg, &device)?;
    println!("Loaded Phi-4-mini-instruct in {:.2?}", t0.elapsed());

    // 1. In-memory bake of phase profile from fixture facts/persona vs guards.
    let rot_dim = cfg.rotary_dim();
    let mut accum = vec![0.0_f32; rot_dim];
    for text in suite.facts.iter().chain(suite.persona.iter()) {
        model.clear_cache();
        let ids = tokenizer
            .encode(text.as_str(), false)
            .map_err(|e| format!("encode error: {e}"))?
            .get_ids()
            .to_vec();
        let (_, q_opt) = model.forward_tokens(&ids, 0, None, true)?;
        if let Some(q) = q_opt {
            for (d, s) in accum.iter_mut().zip(q.iter()) {
                *d += s;
            }
        }
    }
    for guard in &suite.guards {
        model.clear_cache();
        let ids = tokenizer
            .encode(guard.as_str(), false)
            .map_err(|e| format!("encode error: {e}"))?
            .get_ids()
            .to_vec();
        let (_, q_opt) = model.forward_tokens(&ids, 0, None, true)?;
        if let Some(q) = q_opt {
            for (d, s) in accum.iter_mut().zip(q.iter()) {
                *d -= 0.5 * s;
            }
        }
    }
    let max_abs = accum.iter().fold(0.0_f32, |m, &v| m.max(v.abs())).max(1e-6);
    for v in &mut accum {
        *v /= max_abs;
    }
    println!("Baked in-memory phase profile ({rot_dim} dims). Running angle sweep...");

    // 2. Angle sweep across probes.
    let angles = [0.0_f32, 0.05, 0.10, 0.15, 0.25];
    let mut baselines: Vec<Vec<f32>> = Vec::with_capacity(suite.probes.len());
    for (_, prompt, _, _) in &suite.probes {
        model.clear_cache();
        let formatted = format_phi4_prompt(prompt);
        let ids = tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| format!("encode error: {e}"))?
            .get_ids()
            .to_vec();
        let (logits, _) = model.forward_tokens(&ids, 0, None, false)?;
        baselines.push(logits);
    }

    println!(
        "{:>8} | {:>10} | {:>11} | {:>11}",
        "max_rad", "mean_KL", "expect_hit", "forbid_hit"
    );
    for &angle in &angles {
        let steer = model.make_steer_tensors(&accum, angle)?;
        let steer_ref = steer.as_ref().map(|(c, s)| (c, s));
        let mut sum_kl = 0.0_f32;
        let mut exp_hits = 0_usize;
        let mut exp_total = 0_usize;
        let mut forb_hits = 0_usize;
        let mut forb_total = 0_usize;

        for ((_, prompt, expected, forbidden), base_logits) in
            suite.probes.iter().zip(baselines.iter())
        {
            model.clear_cache();
            let formatted = format_phi4_prompt(prompt);
            let ids = tokenizer
                .encode(formatted.as_str(), false)
                .map_err(|e| format!("encode error: {e}"))?
                .get_ids()
                .to_vec();
            let (mut logits, _) = model.forward_tokens(&ids, 0, steer_ref, false)?;
            sum_kl += kl_div(base_logits, &logits);

            let mut gen_ids = Vec::with_capacity(32);
            let mut offset = ids.len();
            for _ in 0..32 {
                let next = argmax(&logits);
                if next == 199999 {
                    break;
                }
                gen_ids.push(next);
                let (next_logits, _) = model.forward_tokens(&[next], offset, steer_ref, false)?;
                logits = next_logits;
                offset += 1;
            }
            let text = tokenizer
                .decode(&gen_ids, true)
                .map_err(|e| format!("decode error: {e}"))?
                .to_lowercase();
            exp_hits += expected
                .iter()
                .filter(|kw| text.contains(&kw.to_lowercase()))
                .count();
            exp_total += expected.len();
            forb_hits += forbidden
                .iter()
                .filter(|kw| text.contains(&kw.to_lowercase()))
                .count();
            forb_total += forbidden.len();
        }

        let n = suite.probes.len() as f32;
        let eh = if exp_total == 0 {
            0.0
        } else {
            exp_hits as f32 / exp_total as f32 * 100.0
        };
        let fh = if forb_total == 0 {
            0.0
        } else {
            forb_hits as f32 / forb_total as f32 * 100.0
        };
        println!("{angle:>8.2} | {:>10.4} | {eh:>10.1}% | {fh:>10.1}%", sum_kl / n);
    }

    Ok(())
}
