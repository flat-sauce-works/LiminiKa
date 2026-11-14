// name: examples/candle_eval/src/main.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Thin Candle + Safetensors evaluation harness delegating all RoPE geometry mapping,
//! Q7 phase baking, and steering kernels to `liminika-adapter` (and `liminika-core`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Module, Result as CResult, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};
use liminika_adapter::{ModelSteeringAdapter, RopeGeometry, RopeLayout};
use liminika_core::abi::{gcso_pointer_trail_t, gcso_q7_t};
use liminika_core::traits::gateway::{ObjectiveFact, ResonantSignalSlice, SubjectivePersona};
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

    fn to_rope_geometry(&self) -> RopeGeometry {
        let mask = if self.num_hidden_layers >= 64 {
            u64::MAX
        } else {
            (1_u64 << self.num_hidden_layers) - 1
        };
        RopeGeometry {
            layout: RopeLayout::HalfSplit,
            head_dim: self.head_dim(),
            rotary_dim: self.rotary_dim(),
            num_q_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            num_layers: self.num_hidden_layers,
            layer_rope_mask: mask,
        }
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

    /// Applies HalfSplit RoPE on `0..rotary_dim` and delegates steering to `ModelSteeringAdapter`.
    fn apply_with_adapter(
        &self,
        x: &Tensor,
        offset: usize,
        steer: Option<(
            &mut ModelSteeringAdapter,
            usize,
            &[gcso_q7_t],
            Option<&[gcso_q7_t]>,
            Option<(&[ObjectiveFact<[i8; 64]>], &[f32])>,
            f32,
            &mut gcso_pointer_trail_t,
        )>,
    ) -> CResult<Tensor> {
        let (b, num_heads, seq_len, head_dim) = x.dims4()?;
        let rot = self.rotary_dim;
        let half = rot / 2;

        let x_rot = x.narrow(D::Minus1, 0, rot)?;
        let x1 = x_rot.narrow(D::Minus1, 0, half)?;
        let x2 = x_rot.narrow(D::Minus1, half, half)?;

        let cos = self.cos.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = self.sin.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;

        let r1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let r2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        let mut q_rot = Tensor::cat(&[&r1, &r2], D::Minus1)?;

        if let Some((adapter, layer_idx, q7_deltas, persona_opt, resonant_opt, max_rad, trail)) =
            steer
        {
            if max_rad != 0.0 && b == 1 {
                let mut flat = q_rot
                    .transpose(1, 2)?
                    .contiguous()?
                    .flatten_all()?
                    .to_dtype(DType::F32)?
                    .to_vec1::<f32>()?;
                let tok_stride = num_heads * rot;
                if max_rad < 0.0 {
                    if let Some((anchors, neg_ref)) = resonant_opt {
                        for tok in 0..seq_len {
                            let slice = &mut flat[tok * tok_stride..(tok + 1) * tok_stride];
                            let probe = adapter
                                .bake_layer_procrustes_q7(neg_ref, slice)
                                .map_err(candle_core::Error::Msg)?;
                            let resonant = ResonantSignalSlice::from_q7_probe(anchors, &probe);
                            if max_rad <= -1.5 {
                                adapter
                                    .orchestrate_layer_rope_with_action_hub(
                                        layer_idx,
                                        &resonant,
                                        trail,
                                        None,
                                        128,
                                        slice,
                                        None,
                                    )
                                    .map_err(candle_core::Error::Msg)?;
                            } else {
                                adapter
                                    .steer_layer_with_gateway(
                                        layer_idx,
                                        &resonant,
                                        trail,
                                        128,
                                        slice,
                                        None,
                                    )
                                    .map_err(candle_core::Error::Msg)?;
                            }
                        }
                    } else {
                        let mut fact_arr = [0i8; 64];
                        let n = q7_deltas.len().min(64);
                        fact_arr[..n].copy_from_slice(&q7_deltas[..n]);
                        let fact = ObjectiveFact::new(fact_arr, 2.5, 0, layer_idx as u32 + 1);
                        if let Some(p_deltas) = persona_opt {
                            let mut persona_arr = [0i8; 64];
                            let pn = p_deltas.len().min(64);
                            persona_arr[..pn].copy_from_slice(&p_deltas[..pn]);
                            let persona =
                                SubjectivePersona::new(persona_arr, 1.5, 0, layer_idx as u32 + 101);
                            let composite = (fact, persona);
                            for tok in 0..seq_len {
                                let slice = &mut flat[tok * tok_stride..(tok + 1) * tok_stride];
                                adapter
                                    .steer_layer_with_gateway(
                                        layer_idx,
                                        &composite,
                                        trail,
                                        128,
                                        slice,
                                        None,
                                    )
                                    .map_err(candle_core::Error::Msg)?;
                            }
                        } else {
                            for tok in 0..seq_len {
                                let slice = &mut flat[tok * tok_stride..(tok + 1) * tok_stride];
                                adapter
                                    .steer_layer_with_gateway(
                                        layer_idx,
                                        &fact,
                                        trail,
                                        128,
                                        slice,
                                        None,
                                    )
                                    .map_err(candle_core::Error::Msg)?;
                            }
                        }
                    }
                } else {
                    for tok in 0..seq_len {
                        let slice = &mut flat[tok * tok_stride..(tok + 1) * tok_stride];
                        adapter
                            .steer_layer_q_rot(layer_idx, slice, q7_deltas, max_rad)
                            .map_err(candle_core::Error::Msg)?;
                    }
                }
                q_rot = Tensor::from_vec(flat, (1, seq_len, num_heads, rot), x.device())?
                    .transpose(1, 2)?
                    .to_dtype(x.dtype())?;
            }
        }

        if rot < head_dim {
            let x_pass = x.narrow(D::Minus1, rot, head_dim - rot)?;
            Tensor::cat(&[&q_rot, &x_pass], D::Minus1)
        } else {
            Ok(q_rot)
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
        steer: Option<(
            &mut ModelSteeringAdapter,
            usize,
            &[gcso_q7_t],
            Option<&[gcso_q7_t]>,
            Option<(&[ObjectiveFact<[i8; 64]>], &[f32])>,
            f32,
            &mut gcso_pointer_trail_t,
        )>,
        collect_q_heads: bool,
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

        let q = rope.apply_with_adapter(&q, offset, steer)?;
        let k = rope.apply_with_adapter(&k, offset, None)?;

        let q_heads = if collect_q_heads {
            let last_q = q
                .i((0, .., seq_len - 1, 0..rope.rotary_dim))?
                .flatten_all()?
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
        Ok((self.o_proj.forward(&ctx)?, q_heads))
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
        mut adapter: Option<&mut ModelSteeringAdapter>,
        q7s: Option<&[Vec<gcso_q7_t>]>,
        persona_q7s: Option<&[Vec<gcso_q7_t>]>,
        resonant_ctx: Option<(&[Vec<ObjectiveFact<[i8; 64]>>], &[Vec<f32>])>,
        rope_rad: f32,
        trail_opt: Option<&mut gcso_pointer_trail_t>,
        srl: Option<(&[f32], &[f32], f32)>,
        use_edbc: bool,
        collect_bake_data: bool,
    ) -> CResult<(Vec<f32>, Vec<Vec<f32>>, Vec<f32>)> {
        let input = Tensor::from_slice(tokens, (1, tokens.len()), &self.device)?;
        let mut h = self.embed_tokens.forward(&input)?;
        let mut per_layer_q = if collect_bake_data {
            Vec::with_capacity(self.layers.len())
        } else {
            Vec::new()
        };
        let mut fallback_trail = gcso_pointer_trail_t::default();
        let trail_ref = trail_opt.unwrap_or(&mut fallback_trail);

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let p_slice = persona_q7s.and_then(|ps| ps.get(layer_idx)).map(Vec::as_slice);
            let res_slice = resonant_ctx.and_then(|(anchors, negs)| {
                anchors
                    .get(layer_idx)
                    .zip(negs.get(layer_idx))
                    .map(|(a, n)| (a.as_slice(), n.as_slice()))
            });
            let steer_ref = match (adapter.as_deref_mut(), q7s) {
                (Some(ad), Some(qs)) if rope_rad != 0.0 => qs.get(layer_idx).map(|q7| {
                    (
                        ad,
                        layer_idx,
                        q7.as_slice(),
                        p_slice,
                        res_slice,
                        rope_rad,
                        &mut *trail_ref,
                    )
                }),
                _ => None,
            };
            let normed = layer.input_layernorm.forward(&h)?;
            let (attn_out, q_opt) = layer.self_attn.forward(
                &normed,
                &self.rope,
                offset,
                steer_ref,
                collect_bake_data,
            )?;
            if let Some(q) = q_opt {
                per_layer_q.push(q);
            }
            h = (h + attn_out)?;
            let post = layer.post_attention_layernorm.forward(&h)?;
            let mlp_out = layer.mlp.forward(&post)?;
            h = (h + mlp_out)?;
        }

        let h = self.norm.forward(&h)?;
        let mut last = h.i((0, tokens.len() - 1, ..))?.unsqueeze(0)?;
        let last_hidden = if collect_bake_data {
            last.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?
        } else {
            Vec::new()
        };

        if let (Some(ad), Some((u_vec, v_vec, srl_scale))) = (adapter.as_deref(), srl) {
            if srl_scale != 0.0 {
                let mut h_vec = last.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                if srl_scale < 0.0 {
                    let mut fact_arr = [0i8; 64];
                    if let Some(last_q7) = q7s.and_then(|qs| qs.last()) {
                        let n = last_q7.len().min(64);
                        fact_arr[..n].copy_from_slice(&last_q7[..n]);
                    }
                    let fact = ObjectiveFact::new(fact_arr, 2.5, 0, 1);
                    if let Some(last_p) = persona_q7s.and_then(|ps| ps.last()) {
                        let mut persona_arr = [0i8; 64];
                        let pn = last_p.len().min(64);
                        persona_arr[..pn].copy_from_slice(&last_p[..pn]);
                        let persona = SubjectivePersona::new(persona_arr, 1.5, 0, 101);
                        let composite = (fact, persona);
                        let _ = ad
                            .apply_srl_with_gateway(&composite, &mut h_vec, u_vec, v_vec)
                            .map_err(candle_core::Error::Msg)?;
                    } else if rope_rad <= -1.5 {
                        let _ = ad
                            .orchestrate_layer_srl_with_action_hub(&fact, &mut h_vec, u_vec, v_vec)
                            .map_err(candle_core::Error::Msg)?;
                    } else {
                        let _ = ad
                            .apply_srl_with_gateway(&fact, &mut h_vec, u_vec, v_vec)
                            .map_err(candle_core::Error::Msg)?;
                    }
                } else {
                    ad.apply_srl_hidden_fused(&mut h_vec, u_vec, v_vec, srl_scale)
                        .map_err(candle_core::Error::Msg)?;
                }
                let len = h_vec.len();
                last = Tensor::from_vec(h_vec, (1, len), &self.device)?.to_dtype(last.dtype())?;
            }
        }

        let mut logits = self
            .lm_head
            .forward(&last)?
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        if use_edbc {
            if let Some(ad) = adapter.as_deref_mut() {
                if rope_rad <= -1.5 {
                    let _ = ad
                        .measure_edbc_entropy(&logits)
                        .map_err(candle_core::Error::Msg)?;
                } else {
                    let _ = ad
                        .step_edbc_logits(&mut logits)
                        .map_err(candle_core::Error::Msg)?;
                }
            }
        }
        Ok((logits, per_layer_q, last_hidden))
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

fn token_logprob(logits: &[f32], tok_id: u32) -> f32 {
    let idx = tok_id as usize;
    if idx >= logits.len() {
        return 0.0;
    }
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp = m + logits.iter().map(|&x| (x - m).exp()).sum::<f32>().ln();
    logits[idx] - log_sum_exp
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

struct BakedModelProfile {
    per_layer_q7: Vec<Vec<gcso_q7_t>>,
    fact_q7: Vec<Vec<gcso_q7_t>>,
    persona_q7: Vec<Vec<gcso_q7_t>>,
    neg_sum: Vec<Vec<f32>>,
    fact_anchors_per_layer: Vec<Vec<ObjectiveFact<[i8; 64]>>>,
    srl_u: Vec<f32>,
    srl_v: Vec<f32>,
}

fn normalize_unit(v: &mut [f32]) {
    let norm = v.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-8);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

fn bake_via_adapter(
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

    let mut per_layer_fact_qs: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(suite.facts.len()); num_layers];

    for text in &suite.facts {
        model.clear_cache();
        let ids = tokenizer
            .encode(text.as_str(), false)
            .map_err(|e| format!("encode error: {e}"))?
            .get_ids()
            .to_vec();
        let (_, layer_qs, h_vec) =
            model.forward_tokens(&ids, 0, None, None, None, None, 0.0, None, None, false, true)?;
        for (l, q) in layer_qs.iter().enumerate() {
            per_layer_fact_qs[l].push(q.clone());
            for ((f_dst, p_dst), &src) in fact_sum[l].iter_mut().zip(pos_sum[l].iter_mut()).zip(q.iter()) {
                *f_dst += src / fact_count;
                *p_dst += src / pos_count;
            }
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
        let (_, layer_qs, h_vec) =
            model.forward_tokens(&ids, 0, None, None, None, None, 0.0, None, None, false, true)?;
        for (l, q) in layer_qs.iter().enumerate() {
            for ((per_dst, p_dst), &src) in persona_sum[l].iter_mut().zip(pos_sum[l].iter_mut()).zip(q.iter()) {
                *per_dst += src / persona_count;
                *p_dst += src / pos_count;
            }
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
        let (_, layer_qs, h_vec) =
            model.forward_tokens(&ids, 0, None, None, None, None, 0.0, None, None, false, true)?;
        for (l, q) in layer_qs.iter().enumerate() {
            for (dst, &src) in neg_sum[l].iter_mut().zip(q.iter()) {
                *dst += src / neg_count;
            }
        }
        for (dst, &src) in neg_h.iter_mut().zip(h_vec.iter()) {
            *dst += src / neg_count;
        }
    }

    let mut per_layer_q7 = Vec::with_capacity(num_layers);
    let mut fact_q7 = Vec::with_capacity(num_layers);
    let mut persona_q7 = Vec::with_capacity(num_layers);
    let mut fact_anchors_per_layer = Vec::with_capacity(num_layers);
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
    }

    let mut srl_u: Vec<f32> = pos_h.iter().zip(neg_h.iter()).map(|(&p, &n)| p - n).collect();
    let mut srl_v = srl_u.clone();
    normalize_unit(&mut srl_u);
    normalize_unit(&mut srl_v);

    Ok(BakedModelProfile {
        per_layer_q7,
        fact_q7,
        persona_q7,
        neg_sum,
        fact_anchors_per_layer,
        srl_u,
        srl_v,
    })
}

fn out_push_q7(
    out: &mut Vec<Vec<gcso_q7_t>>,
    adapter: &ModelSteeringAdapter,
    neg: &[f32],
    pos: &[f32],
) -> Result<(), String> {
    out.push(adapter.bake_layer_procrustes_q7(neg, pos)?);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let model_dir = repo_root.join("models/Phi-4-mini-instruct");
    let fixture_path = repo_root.join("tests/fixtures/gcso_eval_samples.txt");

    let cfg: Phi3Config =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json"))?)?;
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| format!("tokenizer load error: {e}"))?;
    let suite = load_fixture(&fixture_path)?;

    let mut adapter = ModelSteeringAdapter::new(cfg.to_rope_geometry())?;

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!(
        "LiminiKa Candle Eval (RoPE + EDBC + SRL via liminika-adapter) | device={:?} | layers={} heads={}/{} head_dim={} rotary_dim={}",
        device,
        adapter.geometry().num_layers,
        adapter.geometry().num_q_heads,
        adapter.geometry().num_kv_heads,
        adapter.geometry().head_dim,
        adapter.geometry().rotary_dim
    );

    let t0 = Instant::now();
    let mut model = Phi3Model::load(&model_dir, &cfg, &device)?;
    println!("Loaded Phi-4-mini-instruct in {:.2?}", t0.elapsed());

    let baked = bake_via_adapter(&mut model, &tokenizer, &adapter, cfg.hidden_size, &suite)?;
    println!(
        "Baked {} x {} Q7 phase deltas + {}-dim SRL vectors via ModelSteeringAdapter.",
        baked.per_layer_q7.len(),
        adapter.geometry().num_q_heads,
        cfg.hidden_size
    );

    let mut target_toks: Vec<Option<u32>> = Vec::with_capacity(suite.probes.len());
    for (_, _, expected, _) in &suite.probes {
        let tok = expected.first().and_then(|kw| {
            tokenizer
                .encode(kw.as_str(), false)
                .ok()
                .and_then(|enc| enc.get_ids().first().copied())
        });
        target_toks.push(tok);
    }

    let mut baselines: Vec<Vec<f32>> = Vec::with_capacity(suite.probes.len());
    for (_, prompt, _, _) in &suite.probes {
        model.clear_cache();
        let formatted = format_phi4_prompt(prompt);
        let ids = tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| format!("encode error: {e}"))?
            .get_ids()
            .to_vec();
        let (logits, _, _) =
            model.forward_tokens(&ids, 0, None, None, None, None, 0.0, None, None, false, false)?;
        baselines.push(logits);
    }

    // Compare manual max_rad vs Gateway structural hyperparameters vs Resonant slice vs ActionHub.
    // tuple: (label, layer_mask, rope_rad, srl_scale, use_two_origin, use_resonant_slice)
    let configs: [(&str, u64, f32, f32, bool, bool); 6] = [
        ("Baseline", 0xFFFF_FFFF, 0.00, 0.00, false, false),
        ("RoPE (Top16, 0.10)", 0xFFFF_0000, 0.10, 0.00, false, false),
        ("Top16 + GW(Fact)", 0xFFFF_0000, -1.00, 0.00, false, false),
        ("Top16 + GW(Fact+SRL)", 0xFFFF_0000, -1.00, -1.00, false, false),
        ("Top16 + GW(Resonant)", 0xFFFF_0000, -1.00, -1.00, false, true),
        ("Top16 + ActionHub", 0xFFFF_0000, -2.00, -1.00, false, true),
    ];

    println!(
        "\n{:>20} | {:>10} | {:>12} | {:>11} | {:>11}",
        "mode", "mean_KL", "d_logprob", "expect_hit", "forbid_hit"
    );

    for (label, layer_mask, rope_rad, srl_scale, use_two_origin, use_resonant) in configs {
        adapter.set_layer_rope_mask(layer_mask);
        adapter.reset_action_hub_stats();
        let use_edbc = rope_rad <= -1.5;
        let mut sum_kl = 0.0_f32;
        let mut sum_d_lp = 0.0_f32;
        let mut d_lp_count = 0_usize;
        let mut exp_hits = 0_usize;
        let mut exp_total = 0_usize;
        let mut forb_hits = 0_usize;
        let mut forb_total = 0_usize;
        let mut p1_sample = String::new();

        for (p_idx, ((_, prompt, expected, forbidden), base_logits)) in
            suite.probes.iter().zip(baselines.iter()).enumerate()
        {
            model.clear_cache();
            let formatted = format_phi4_prompt(prompt);
            let ids = tokenizer
                .encode(formatted.as_str(), false)
                .map_err(|e| format!("encode error: {e}"))?
                .get_ids()
                .to_vec();
            let mut trail = gcso_pointer_trail_t::default();
            let q7_primary = if use_two_origin {
                baked.fact_q7.as_slice()
            } else {
                baked.per_layer_q7.as_slice()
            };
            let q7_persona = if use_two_origin {
                Some(baked.persona_q7.as_slice())
            } else {
                None
            };
            let res_ctx = if use_resonant {
                Some((
                    baked.fact_anchors_per_layer.as_slice(),
                    baked.neg_sum.as_slice(),
                ))
            } else {
                None
            };
            let srl_arg = if srl_scale != 0.0 {
                Some((baked.srl_u.as_slice(), baked.srl_v.as_slice(), srl_scale))
            } else {
                None
            };
            let (mut logits, _, _) = model.forward_tokens(
                &ids,
                0,
                Some(&mut adapter),
                Some(q7_primary),
                q7_persona,
                res_ctx,
                rope_rad,
                Some(&mut trail),
                srl_arg,
                use_edbc,
                false,
            )?;
            sum_kl += kl_div(base_logits, &logits);

            if let Some(tok) = target_toks[p_idx] {
                sum_d_lp += token_logprob(&logits, tok) - token_logprob(base_logits, tok);
                d_lp_count += 1;
            }

            let mut gen_ids = Vec::with_capacity(32);
            let mut offset = ids.len();
            for _ in 0..32 {
                let next = argmax(&logits);
                if next == 199999 {
                    break;
                }
                gen_ids.push(next);
                let (next_logits, _, _) = model.forward_tokens(
                    &[next],
                    offset,
                    Some(&mut adapter),
                    Some(q7_primary),
                    q7_persona,
                    res_ctx,
                    rope_rad,
                    Some(&mut trail),
                    srl_arg,
                    use_edbc,
                    false,
                )?;
                logits = next_logits;
                offset += 1;
            }
            let text = tokenizer
                .decode(&gen_ids, true)
                .map_err(|e| format!("decode error: {e}"))?;
            if p_idx == 0 {
                p1_sample = text.replace('\n', " ");
            }
            let lower = text.to_lowercase();
            exp_hits += expected
                .iter()
                .filter(|kw| lower.contains(&kw.to_lowercase()))
                .count();
            exp_total += expected.len();
            forb_hits += forbidden
                .iter()
                .filter(|kw| lower.contains(&kw.to_lowercase()))
                .count();
            forb_total += forbidden.len();
        }

        let n = suite.probes.len() as f32;
        let mean_d_lp = if d_lp_count == 0 {
            0.0
        } else {
            sum_d_lp / d_lp_count as f32
        };
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
        println!(
            "{label:>20} | {:>10.4} | {mean_d_lp:>+12.4} | {eh:>10.1}% | {fh:>10.1}%",
            sum_kl / n
        );
        println!("   [P1] {p1_sample}");
        if rope_rad <= -1.5 {
            let counts = adapter.action_hub_decision_counts();
            let total: u64 = counts.iter().sum::<u64>().max(1);
            println!(
                "   [ActionHub] total={} Passive={:.1}% Skip={:.1}% Express={:.1}% PSPM={:.1}% SRL={:.1}% Crystal={:.1}% Repulse={:.1}%",
                total,
                counts[0] as f64 / total as f64 * 100.0,
                counts[1] as f64 / total as f64 * 100.0,
                counts[2] as f64 / total as f64 * 100.0,
                counts[3] as f64 / total as f64 * 100.0,
                counts[4] as f64 / total as f64 * 100.0,
                counts[5] as f64 / total as f64 * 100.0,
                counts[6] as f64 / total as f64 * 100.0,
            );
        }
    }

    Ok(())
}
