// name: examples/candle_eval/src/main.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Thin Candle + Safetensors evaluation harness delegating all RoPE geometry mapping,
//! Q7 phase baking, and steering kernels to `liminika-adapter` (and `liminika-core`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Module, Result as CResult, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};
use liminika_adapter::{
    AdaptiveGopCutDetector, AttractorContainerCodec, AttractorRole, CompiledTokenCorpus,
    FrameTransitionMetrics, GcsoDualTrackTimeline, ModelBaseAttractorProfile, ModelSteeringAdapter,
    RebakeContext, RopeGeometry, RopeLayout, SequenceActivationPooler, SourceTextRecord,
    SubtitleActivationEncoder, TimelineResumeOutcome, TimelineStepContext, TokenSequenceEntry,
    VoidBoundaryState,
};
use liminika_core::abi::{gcso_pointer_trail_t, gcso_q7_t};
use liminika_core::action_hub::ActionHubEngine;
use liminika_core::traits::gateway::{
    GuardPhaseAnchor, ObjectiveFact, ResonantSignalSlice, SubjectivePersona,
};
use serde::Deserialize;
use tokenizers::Tokenizer;

type LayerResonantSlices<'a> = (
    &'a [ObjectiveFact<[i8; 64]>],
    &'a [SubjectivePersona<[i8; 64]>],
    &'a [GuardPhaseAnchor],
    &'a [f32],
    &'a [f32],
);

type ForwardTokensOutput = (Vec<f32>, Vec<Vec<f32>>, Vec<f32>);

struct ResonantAnchors<'a> {
    facts: &'a [Vec<ObjectiveFact<[i8; 64]>>],
    personas: &'a [Vec<SubjectivePersona<[i8; 64]>>],
    guards: &'a [Vec<GuardPhaseAnchor>],
    negs: &'a [Vec<f32>],
    poss: &'a [Vec<f32>],
}

impl<'a> ResonantAnchors<'a> {
    fn layer_slices(&self, layer_idx: usize) -> Option<LayerResonantSlices<'a>> {
        self.facts
            .get(layer_idx)
            .zip(self.personas.get(layer_idx))
            .zip(self.guards.get(layer_idx))
            .zip(self.negs.get(layer_idx))
            .zip(self.poss.get(layer_idx))
            .map(|((((fa, pa), ga), n), p)| {
                (
                    fa.as_slice(),
                    pa.as_slice(),
                    ga.as_slice(),
                    n.as_slice(),
                    p.as_slice(),
                )
            })
    }
}

struct LayerSteer<'a> {
    adapter: &'a mut ModelSteeringAdapter,
    layer_idx: usize,
    q7_deltas: &'a [gcso_q7_t],
    persona_opt: Option<&'a [gcso_q7_t]>,
    resonant_opt: Option<LayerResonantSlices<'a>>,
    max_rad: f32,
    trail: &'a mut gcso_pointer_trail_t,
}

struct ForwardSteer<'a> {
    adapter: &'a mut ModelSteeringAdapter,
    q7s: Option<&'a [Vec<gcso_q7_t>]>,
    persona_q7s: Option<&'a [Vec<gcso_q7_t>]>,
    resonant_ctx: Option<ResonantAnchors<'a>>,
    rope_rad: f32,
    trail: &'a mut gcso_pointer_trail_t,
    srl: Option<(&'a [f32], &'a [f32], f32)>,
}

struct ProbeTrace {
    slot: String,
    text: String,
    d_lp: f32,
    p_exp: usize,
    p_forb: usize,
    delta: [u64; 7],
    rho: f32,
    steps: u32,
    h_tilde: f32,
}

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
        steer: Option<LayerSteer<'_>>,
    ) -> CResult<Tensor> {
        let (b, num_heads, seq_len, head_dim) = x.dims4()?;
        let rot = self.rotary_dim;
        let half = rot / 2;

        let x_rot = x.narrow(D::Minus1, 0, rot)?;
        let x1 = x_rot.narrow(D::Minus1, 0, half)?;
        let x2 = x_rot.narrow(D::Minus1, half, half)?;

        let cos = self
            .cos
            .narrow(0, offset, seq_len)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let sin = self
            .sin
            .narrow(0, offset, seq_len)?
            .unsqueeze(0)?
            .unsqueeze(0)?;

        let r1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let r2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        let mut q_rot = Tensor::cat(&[&r1, &r2], D::Minus1)?;

        if let Some(LayerSteer {
            adapter,
            layer_idx,
            q7_deltas,
            persona_opt,
            resonant_opt,
            max_rad,
            trail,
        }) = steer
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
                    if let Some((
                        fact_anchors,
                        persona_anchors,
                        _guard_anchors,
                        neg_ref,
                        _pos_ref,
                    )) = resonant_opt
                    {
                        for tok in 0..seq_len {
                            let slice = &mut flat[tok * tok_stride..(tok + 1) * tok_stride];
                            let probe = adapter
                                .bake_layer_procrustes_q7(neg_ref, slice)
                                .map_err(candle_core::Error::Msg)?;
                            let fact_res = ResonantSignalSlice::from_q7_probe(fact_anchors, &probe);
                            if max_rad <= -1.5 {
                                let persona_res =
                                    ResonantSignalSlice::from_q7_probe(persona_anchors, &probe);
                                let composite = (fact_res, persona_res);
                                adapter
                                    .orchestrate_layer_rope_with_action_hub(
                                        layer_idx, &composite, trail, None, 128, slice, None,
                                    )
                                    .map_err(candle_core::Error::Msg)?;
                            } else {
                                adapter
                                    .steer_layer_with_gateway(
                                        layer_idx, &fact_res, trail, 128, slice, None,
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
                                        layer_idx, &composite, trail, 128, slice, None,
                                    )
                                    .map_err(candle_core::Error::Msg)?;
                            }
                        } else {
                            for tok in 0..seq_len {
                                let slice = &mut flat[tok * tok_stride..(tok + 1) * tok_stride];
                                adapter
                                    .steer_layer_with_gateway(
                                        layer_idx, &fact, trail, 128, slice, None,
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
        steer: Option<LayerSteer<'_>>,
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

        // Capture raw per-token pre-RoPE Q activations `[seq_len * num_heads * rotary_dim]`
        // and delegate all sequence pooling & novelty weighting to `liminika-adapter::pooling`.
        let q_heads = if collect_q_heads {
            let seq_q = q
                .i((0, .., .., 0..rope.rotary_dim))?
                .transpose(0, 1)?
                .contiguous()?
                .flatten_all()?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            Some(seq_q)
        } else {
            None
        };

        let q = rope.apply_with_adapter(&q, offset, steer)?;
        let k = rope.apply_with_adapter(&k, offset, None)?;

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
        let gate_up_proj = linear_no_bias(
            cfg.hidden_size,
            2 * cfg.intermediate_size,
            vb.pp("gate_up_proj"),
        )?;
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
    fn load(
        model_dir: &Path,
        cfg: &Phi3Config,
        dev: &Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let dtype = if dev.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };
        let files = vec![
            model_dir.join("model-00001-of-00002.safetensors"),
            model_dir.join("model-00002-of-00002.safetensors"),
        ];
        // SAFETY: The safetensors files are read-only model weights on disk.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, dev)? };
        let mvb = vb.pp("model");

        let embed_weight = mvb
            .pp("embed_tokens")
            .get((cfg.vocab_size, cfg.hidden_size), "weight")?;
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
        mut steer: Option<&mut ForwardSteer<'_>>,
        use_edbc: bool,
        collect_bake_data: bool,
    ) -> CResult<ForwardTokensOutput> {
        let input = Tensor::from_slice(tokens, (1, tokens.len()), &self.device)?;
        let mut h = self.embed_tokens.forward(&input)?;
        let mut per_layer_q = if collect_bake_data {
            Vec::with_capacity(self.layers.len())
        } else {
            Vec::new()
        };

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let steer_ref = match steer.as_deref_mut() {
                Some(st) if st.rope_rad != 0.0 => st.q7s.and_then(|qs| {
                    qs.get(layer_idx).map(|q7| LayerSteer {
                        adapter: &mut *st.adapter,
                        layer_idx,
                        q7_deltas: q7.as_slice(),
                        persona_opt: st
                            .persona_q7s
                            .and_then(|ps| ps.get(layer_idx))
                            .map(Vec::as_slice),
                        resonant_opt: st
                            .resonant_ctx
                            .as_ref()
                            .and_then(|r| r.layer_slices(layer_idx)),
                        max_rad: st.rope_rad,
                        trail: &mut *st.trail,
                    })
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

        if let Some(st) = steer.as_deref() {
            if let Some((u_vec, v_vec, srl_scale)) = st.srl {
                if srl_scale != 0.0 {
                    let ad: &ModelSteeringAdapter = st.adapter;
                    let q7s = st.q7s;
                    let persona_q7s = st.persona_q7s;
                    let rope_rad = st.rope_rad;
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
                            if rope_rad <= -1.5 {
                                let _ = ad
                                    .orchestrate_layer_srl_with_action_hub(
                                        &composite, &mut h_vec, u_vec, v_vec,
                                    )
                                    .map_err(candle_core::Error::Msg)?;
                            } else {
                                let _ = ad
                                    .apply_srl_with_gateway(&composite, &mut h_vec, u_vec, v_vec)
                                    .map_err(candle_core::Error::Msg)?;
                            }
                        } else if rope_rad <= -1.5 {
                            let _ = ad
                                .orchestrate_layer_srl_with_action_hub(
                                    &fact, &mut h_vec, u_vec, v_vec,
                                )
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
                    last =
                        Tensor::from_vec(h_vec, (1, len), &self.device)?.to_dtype(last.dtype())?;
                }
            }
        }

        let mut logits = self
            .lm_head
            .forward(&last)?
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        if use_edbc {
            if let Some(st) = steer {
                let ad = &mut *st.adapter;
                if st.rope_rad <= -1.5 {
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

fn detect_simd_label() -> &'static str {
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

fn query_nvidia_smi_vram() -> Option<String> {
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

fn estimate_phi3_footprint(cfg: &Phi3Config, dtype: DType) -> (u64, f64, f64) {
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
    let weights_mb = (total_params as f64 * elem_bytes) / (1024.0 * 1024.0);
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

fn print_runtime_profile(
    cfg: &Phi3Config,
    adapter: &ModelSteeringAdapter,
    device: &Device,
    dtype: DType,
    load_elapsed: std::time::Duration,
    vram_before: Option<&str>,
) {
    let geom = adapter.geometry();
    let (params, weights_mb, kv_2048_mb) = estimate_phi3_footprint(cfg, dtype);
    let hub_bytes = std::mem::size_of::<ActionHubEngine>();
    println!(
        "[System] os={} arch={} simd={} device={:?} dtype={:?} load={:.2?}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        detect_simd_label(),
        device,
        dtype,
        load_elapsed,
    );
    println!(
        "[Model ] params={:.2}M weights_est={:.1}MiB kv2048_est={:.1}MiB layers={} heads={}/{} head_dim={} rotary_dim={} sidecar_hub={}B",
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

fn sample_startup_base_prior(
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

struct BakedModelProfile {
    per_layer_q7: Vec<Vec<gcso_q7_t>>,
    fact_q7: Vec<Vec<gcso_q7_t>>,
    persona_q7: Vec<Vec<gcso_q7_t>>,
    neg_sum: Vec<Vec<f32>>,
    pos_sum: Vec<Vec<f32>>,
    fact_anchors_per_layer: Vec<Vec<ObjectiveFact<[i8; 64]>>>,
    persona_anchors_per_layer: Vec<Vec<SubjectivePersona<[i8; 64]>>>,
    guard_anchors_per_layer: Vec<Vec<GuardPhaseAnchor>>,
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
    let active_dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let vram_before = query_nvidia_smi_vram();
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
    let load_elapsed = t0.elapsed();
    println!("Loaded Phi-4-mini-instruct in {:.2?}", load_elapsed);
    print_runtime_profile(
        &cfg,
        &adapter,
        &device,
        active_dtype,
        load_elapsed,
        vram_before.as_deref(),
    );

    let step2_only = std::env::args().any(|a| a == "--step2-only");
    let use_base_prior = std::env::args().any(|a| a == "--base-prior");
    let t_bake = Instant::now();
    let baked = bake_via_adapter(&mut model, &tokenizer, &adapter, cfg.hidden_size, &suite)?;
    println!(
        "Baked {} x {} Q7 phase deltas + {}-dim SRL vectors via ModelSteeringAdapter in {:.2?}.",
        baked.per_layer_q7.len(),
        adapter.geometry().num_q_heads,
        cfg.hidden_size,
        t_bake.elapsed(),
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
    let mut base_entropies: Vec<f32> = Vec::with_capacity(suite.probes.len());
    for (_, prompt, _, _) in &suite.probes {
        model.clear_cache();
        let formatted = format_phi4_prompt(prompt);
        let ids = tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| format!("encode error: {e}"))?
            .get_ids()
            .to_vec();
        let (logits, _, _) = model.forward_tokens(&ids, 0, None, false, false)?;
        let h_t = adapter.measure_edbc_entropy(&logits)?.moving_z_entropy;
        base_entropies.push(h_t);
        baselines.push(logits);
    }

    adapter.set_layer_rope_mask(0xFFFF_0000);
    let pos_slices: Vec<&[f32]> = baked.pos_sum.iter().map(Vec::as_slice).collect();
    let neg_slices: Vec<&[f32]> = baked.neg_sum.iter().map(Vec::as_slice).collect();
    let probe_res = adapter.calibrate_manifold_probe(&pos_slices, &neg_slices, &base_entropies)?;
    println!(
        "[EnvProbe epoch={}] heads={} steered_layers={} coh={:.3} H0={:.3} => dyn_gain={:.3} skip_H={:.3} srl_rho={:.3}",
        probe_res.probe_epoch,
        probe_res.active_heads,
        probe_res.steered_layers,
        probe_res.baseline_inter_layer_coherence,
        probe_res.baseline_entropy,
        adapter.action_hub().thresholds.reinforcement_gain,
        adapter.action_hub().thresholds.inertial_skip_entropy_thresh,
        adapter.action_hub().thresholds.srl_inject_density_thresh,
    );

    if !step2_only {
        // Compare manual max_rad vs Gateway structural hyperparameters vs Resonant slice vs ActionHub.
        // tuple: (label, layer_mask, rope_rad, srl_scale, use_two_origin, use_resonant_slice)
        let configs: [(&str, u64, f32, f32, bool, bool); 6] = [
            ("Baseline", 0xFFFF_FFFF, 0.00, 0.00, false, false),
            ("RoPE (Top16, 0.10)", 0xFFFF_0000, 0.10, 0.00, false, false),
            ("Top16 + GW(Fact)", 0xFFFF_0000, -1.00, 0.00, false, false),
            (
                "Top16 + GW(Fact+SRL)",
                0xFFFF_0000,
                -1.00,
                -1.00,
                false,
                false,
            ),
            (
                "Top16 + GW(Resonant)",
                0xFFFF_0000,
                -1.00,
                -1.00,
                false,
                true,
            ),
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
            let mut probe_traces: Vec<ProbeTrace> = Vec::new();

            for (p_idx, ((slot, prompt, expected, forbidden), base_logits)) in
                suite.probes.iter().zip(baselines.iter()).enumerate()
            {
                model.clear_cache();
                let counts_before = *adapter.action_hub_decision_counts();
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
                    Some(ResonantAnchors {
                        facts: baked.fact_anchors_per_layer.as_slice(),
                        personas: baked.persona_anchors_per_layer.as_slice(),
                        guards: baked.guard_anchors_per_layer.as_slice(),
                        negs: baked.neg_sum.as_slice(),
                        poss: baked.pos_sum.as_slice(),
                    })
                } else {
                    None
                };
                let srl_arg = if srl_scale != 0.0 {
                    Some((baked.srl_u.as_slice(), baked.srl_v.as_slice(), srl_scale))
                } else {
                    None
                };
                let mut steer = ForwardSteer {
                    adapter: &mut adapter,
                    q7s: Some(q7_primary),
                    persona_q7s: q7_persona,
                    resonant_ctx: res_ctx,
                    rope_rad,
                    trail: &mut trail,
                    srl: srl_arg,
                };
                let (mut logits, _, _) =
                    model.forward_tokens(&ids, 0, Some(&mut steer), use_edbc, false)?;
                sum_kl += kl_div(base_logits, &logits);

                let mut probe_d_lp = 0.0_f32;
                if let Some(tok) = target_toks[p_idx] {
                    probe_d_lp = token_logprob(&logits, tok) - token_logprob(base_logits, tok);
                    sum_d_lp += probe_d_lp;
                    d_lp_count += 1;
                }

                let mut gen_ids = Vec::with_capacity(48);
                for offset in (ids.len()..).take(48) {
                    let next = argmax(&logits);
                    if next == 199999 {
                        break;
                    }
                    gen_ids.push(next);
                    let (next_logits, _, _) =
                        model.forward_tokens(&[next], offset, Some(&mut steer), use_edbc, false)?;
                    logits = next_logits;
                }
                let text = tokenizer
                    .decode(&gen_ids, true)
                    .map_err(|e| format!("decode error: {e}"))?;
                let clean_text = text.replace('\n', " ");
                if p_idx == 0 {
                    p1_sample = clean_text.clone();
                }
                let lower = text.to_lowercase();
                let p_exp = expected
                    .iter()
                    .filter(|kw| lower.contains(&kw.to_lowercase()))
                    .count();
                let p_forb = forbidden
                    .iter()
                    .filter(|kw| lower.contains(&kw.to_lowercase()))
                    .count();
                exp_hits += p_exp;
                exp_total += expected.len();
                forb_hits += p_forb;
                forb_total += forbidden.len();

                let counts_after = *adapter.action_hub_decision_counts();
                let mut delta = [0_u64; 7];
                for i in 0..7 {
                    delta[i] = counts_after[i].saturating_sub(counts_before[i]);
                }
                probe_traces.push(ProbeTrace {
                    slot: slot.clone(),
                    text: clean_text,
                    d_lp: probe_d_lp,
                    p_exp,
                    p_forb,
                    delta,
                    rho: trail.stigmergic_density,
                    steps: trail.step_count,
                    h_tilde: adapter.action_hub().last_entropy_h_tilde,
                });
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
            if label == "Baseline" || label == "Top16 + ActionHub" {
                println!("   --- Per-Probe Trace ({label}) ---");
                for (idx, tr) in probe_traces.iter().enumerate() {
                    let d_sum: u64 = tr.delta.iter().sum::<u64>().max(1);
                    if rope_rad <= -1.5 {
                        println!(
                        "   [P{}:{}] d_lp={:+.3} hit={}/forb={} rho={:.3} steps={} H={:.3} | Pas={} Skp={} Exp={} PSPM={} SRL={} Cry={} Rep={} | {}",
                        idx + 1,
                        tr.slot,
                        tr.d_lp,
                        tr.p_exp,
                        tr.p_forb,
                        tr.rho,
                        tr.steps,
                        tr.h_tilde,
                        tr.delta[0],
                        tr.delta[1],
                        tr.delta[2],
                        tr.delta[3],
                        tr.delta[4],
                        tr.delta[5],
                        tr.delta[6],
                        tr.text
                    );
                        let _ = d_sum;
                    } else {
                        println!(
                            "   [P{}:{}] hit={}/forb={} | {}",
                            idx + 1,
                            tr.slot,
                            tr.p_exp,
                            tr.p_forb,
                            tr.text
                        );
                    }
                }
            }
        }

        // Multi-turn context switching check on a single persistent KV cache & stigmergic trail:
        // Turn 1 (FACT) -> Turn 2 (PERSONA) -> Turn 3 (GUARD)
        println!("\n=== Multi-Turn Context Switch Check (Persistent Trail & KV Cache) ===");
        let turns = [
            ("Turn1:FACT", suite.probes[0].1.as_str()),
            ("Turn2:PERSONA", suite.probes[5].1.as_str()),
            ("Turn3:GUARD", suite.probes[4].1.as_str()),
        ];
        for &(mode_label, layer_mask, rope_rad, srl_scale, use_edbc, use_resonant) in &[
            (
                "Baseline",
                0xFFFF_FFFF_u64,
                0.00_f32,
                0.00_f32,
                false,
                false,
            ),
            (
                "ActionHub",
                0xFFFF_0000_u64,
                -2.00_f32,
                -1.00_f32,
                true,
                true,
            ),
        ] {
            adapter.set_layer_rope_mask(layer_mask);
            adapter.reset_action_hub_stats();
            model.clear_cache();
            let mut trail = gcso_pointer_trail_t::default();
            let mut offset = 0_usize;
            println!("-- {mode_label} --");
            for (turn_name, prompt) in turns {
                let counts_before = *adapter.action_hub_decision_counts();
                let turn_text = if offset == 0 {
                    format_phi4_prompt(prompt)
                } else {
                    format!("<|end|><|user|>{prompt}<|end|><|assistant|>")
                };
                let ids = tokenizer
                    .encode(turn_text.as_str(), false)
                    .map_err(|e| format!("encode error: {e}"))?
                    .get_ids()
                    .to_vec();
                let q7_primary = if use_resonant {
                    baked.fact_q7.as_slice()
                } else {
                    baked.per_layer_q7.as_slice()
                };
                let q7_persona = if use_resonant {
                    Some(baked.persona_q7.as_slice())
                } else {
                    None
                };
                let res_ctx = if use_resonant {
                    Some(ResonantAnchors {
                        facts: baked.fact_anchors_per_layer.as_slice(),
                        personas: baked.persona_anchors_per_layer.as_slice(),
                        guards: baked.guard_anchors_per_layer.as_slice(),
                        negs: baked.neg_sum.as_slice(),
                        poss: baked.pos_sum.as_slice(),
                    })
                } else {
                    None
                };
                let srl_arg = if srl_scale != 0.0 {
                    Some((baked.srl_u.as_slice(), baked.srl_v.as_slice(), srl_scale))
                } else {
                    None
                };
                let mut steer = ForwardSteer {
                    adapter: &mut adapter,
                    q7s: Some(q7_primary),
                    persona_q7s: q7_persona,
                    resonant_ctx: res_ctx,
                    rope_rad,
                    trail: &mut trail,
                    srl: srl_arg,
                };
                let (mut logits, _, _) =
                    model.forward_tokens(&ids, offset, Some(&mut steer), use_edbc, false)?;
                offset += ids.len();
                let mut gen_ids = Vec::with_capacity(48);
                for _ in 0..48 {
                    let next = argmax(&logits);
                    gen_ids.push(next);
                    let (next_logits, _, _) =
                        model.forward_tokens(&[next], offset, Some(&mut steer), use_edbc, false)?;
                    logits = next_logits;
                    offset += 1;
                    if next == 199999 {
                        break;
                    }
                }
                let text = tokenizer
                    .decode(&gen_ids, true)
                    .map_err(|e| format!("decode error: {e}"))?
                    .replace('\n', " ");
                let counts_after = *adapter.action_hub_decision_counts();
                let mut d = [0_u64; 7];
                for i in 0..7 {
                    d[i] = counts_after[i].saturating_sub(counts_before[i]);
                }
                println!(
                "  [{turn_name}] rho={:.3} steps={} H={:.3} modes=[Pas={} Skp={} Exp={} PSPM={} SRL={} Cry={} Rep={}] => {text}",
                trail.stigmergic_density,
                trail.step_count,
                adapter.action_hub().last_entropy_h_tilde,
                d[0], d[1], d[2], d[3], d[4], d[5], d[6],
            );
            }
        }
    }

    // Step 2: Online Natural-Language -> Attractor Corpus & cvoid Boundary Evaluation
    println!("\n=== Step 2: Natural-Language Attractor Corpus & cvoid Boundary Eval ===");
    adapter.set_layer_rope_mask(0xFFFF_0000);
    let base_prior = if use_base_prior {
        let t_prior = Instant::now();
        let profile = sample_startup_base_prior(&mut model, &tokenizer, &adapter)?;
        println!(
            "[BasePrior] samples={} sub_attractors={} pooled_basis={} token_basis={} elapsed={:.2?}",
            profile.sample_count,
            profile.sub_attractors.len(),
            profile.pooled_common_mode_basis.len(),
            profile.token_common_mode_basis.len(),
            t_prior.elapsed()
        );
        Some(profile)
    } else {
        None
    };

    let t_compile = Instant::now();
    let mut staged_entries: Vec<TokenSequenceEntry> = Vec::new();
    for (role, texts, energy) in [
        (AttractorRole::Fact, suite.facts.as_slice(), 2.5_f32),
        (AttractorRole::Persona, suite.persona.as_slice(), 1.5_f32),
        (AttractorRole::Guard, suite.guards.as_slice(), 2.0_f32),
    ] {
        for text in texts {
            model.clear_cache();
            let ids = tokenizer
                .encode(text.as_str(), false)
                .map_err(|e| format!("encode error: {e}"))?
                .get_ids()
                .to_vec();
            let (_, layer_tok_qs, _) = model.forward_tokens(&ids, 0, None, false, true)?;
            staged_entries.push(TokenSequenceEntry {
                role,
                text: text.clone(),
                energy,
                seq_len: ids.len(),
                per_layer_token_qs: layer_tok_qs,
            });
        }
    }

    let compiled = CompiledTokenCorpus::compile_with_base_prior(
        &adapter,
        &staged_entries,
        SequenceActivationPooler::default(),
        2,
        1,
        base_prior.as_ref(),
    )?;
    let compile_ms = t_compile.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Compiled {} natural-language attractors (two-pass novelty SIF + Stratum pipeline, base_prior={}) in {:.1}ms | auto RecallPolicy: min_coh={:.3} margin={:.3}",
        compiled.corpus.entries().len(),
        base_prior.is_some(),
        compile_ms,
        compiled.recall_policy.min_coherence,
        compiled.recall_policy.boundary_margin,
    );
    let mut step2_ctx_hits = 0_usize;
    let mut step2_exp_hits = 0_usize;
    let mut step2_exp_total = 0_usize;
    let mut step2_forb_hits = 0_usize;
    let mut step2_forb_total = 0_usize;
    let mut step2_recall_ms_sum = 0.0_f64;
    let mut step2_decode_tokens = 0_usize;
    let mut step2_decode_secs = 0.0_f64;
    let mut eval_items: Vec<(String, String, Vec<String>, Vec<String>)> = suite.probes.clone();
    eval_items.push((
        "CVOID".to_string(),
        "What is the boiling point of water at standard sea-level pressure?".to_string(),
        vec!["100".to_string()],
        vec!["GCSO".to_string(), "ZiMMS".to_string()],
    ));

    let cut_detector = AdaptiveGopCutDetector::default();
    let mut timeline = GcsoDualTrackTimeline::new(GcsoDualTrackTimeline::geometry_fingerprint(
        adapter.geometry(),
    ));
    let mut frames_since_keyframe = 0_usize;

    for (idx, (slot, prompt, expected, forbidden)) in eval_items.iter().enumerate() {
        model.clear_cache();
        let q_ids = tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| format!("encode error: {e}"))?
            .get_ids()
            .to_vec();
        let (_, query_tok_qs, _) = model.forward_tokens(&q_ids, 0, None, false, true)?;
        let t_recall = Instant::now();
        let recall = compiled.recall_token_sequence(&adapter, &query_tok_qs, q_ids.len())?;
        let recall_ms = t_recall.elapsed().as_secs_f64() * 1000.0;
        step2_recall_ms_sum += recall_ms;

        let turn_role = match slot.as_str() {
            "PERSONA" => AttractorRole::Persona,
            "GUARD" => AttractorRole::Guard,
            _ => AttractorRole::Fact,
        };
        let emitted_i = timeline.record_step(
            &TimelineStepContext {
                adapter: &adapter,
                corpus: &compiled.corpus,
                policy: &compiled.recall_policy,
                detector: &cut_detector,
            },
            SourceTextRecord {
                frame_index: idx as u64,
                timestamp_epoch_sec: 1_700_000_000 + (idx as u64) * 10,
                role: turn_role,
                text: prompt.clone(),
                energy: 1.5,
            },
            &FrameTransitionMetrics {
                phase_l1_delta: (1.0 - recall.top1_fact_score).clamp(0.0, 1.0) * 0.25,
                void_state: recall.void_state,
                frames_since_keyframe,
                new_cluster_spawned: matches!(
                    recall.void_state,
                    VoidBoundaryState::CvoidEmptySpace
                ),
            },
        )?;
        frames_since_keyframe = if emitted_i {
            0
        } else {
            frames_since_keyframe + 1
        };

        let recalled_ctx: String = recall
            .recalled_facts
            .iter()
            .map(|(e, _)| e.text.as_str())
            .chain(
                recall
                    .recalled_personas
                    .iter()
                    .map(|(e, _)| e.text.as_str()),
            )
            .collect::<Vec<_>>()
            .join(" ");

        let active_prompt = if matches!(recall.void_state, VoidBoundaryState::CvoidEmptySpace)
            || recall.guard_conflict_margin > 0.05
            || recalled_ctx.is_empty()
        {
            format_phi4_prompt(prompt)
        } else {
            format_phi4_prompt(&format!("[Context: {recalled_ctx}]\n{prompt}"))
        };

        model.clear_cache();
        adapter.reset_action_hub_stats();
        let mut trail = gcso_pointer_trail_t::default();
        let ids = tokenizer
            .encode(active_prompt.as_str(), false)
            .map_err(|e| format!("encode error: {e}"))?
            .get_ids()
            .to_vec();

        let dyn_facts = recall.build_fact_anchors_per_layer(cfg.num_hidden_layers);
        let dyn_personas = recall.build_persona_anchors_per_layer(cfg.num_hidden_layers);
        let use_steer = !matches!(recall.void_state, VoidBoundaryState::CvoidEmptySpace)
            && !dyn_facts.is_empty();
        let res_ctx = if use_steer {
            Some(ResonantAnchors {
                facts: dyn_facts.as_slice(),
                personas: dyn_personas.as_slice(),
                guards: baked.guard_anchors_per_layer.as_slice(),
                negs: baked.neg_sum.as_slice(),
                poss: baked.pos_sum.as_slice(),
            })
        } else {
            None
        };
        let rope_rad = if use_steer { -2.00 } else { 0.00 };
        let srl_arg = if use_steer {
            Some((baked.srl_u.as_slice(), baked.srl_v.as_slice(), -1.00))
        } else {
            None
        };

        let mut steer = ForwardSteer {
            adapter: &mut adapter,
            q7s: Some(baked.fact_q7.as_slice()),
            persona_q7s: Some(baked.persona_q7.as_slice()),
            resonant_ctx: res_ctx,
            rope_rad,
            trail: &mut trail,
            srl: srl_arg,
        };
        let t_dec = Instant::now();
        let (mut logits, _, _) =
            model.forward_tokens(&ids, 0, Some(&mut steer), use_steer, false)?;

        let mut gen_ids = Vec::with_capacity(48);
        for offset in (ids.len()..).take(48) {
            let next = argmax(&logits);
            if next == 199999 {
                break;
            }
            gen_ids.push(next);
            let (next_logits, _, _) =
                model.forward_tokens(&[next], offset, Some(&mut steer), use_steer, false)?;
            logits = next_logits;
        }
        step2_decode_tokens += gen_ids.len();
        step2_decode_secs += t_dec.elapsed().as_secs_f64();
        let text = tokenizer
            .decode(&gen_ids, true)
            .map_err(|e| format!("decode error: {e}"))?
            .replace('\n', " ");
        let lower = text.to_lowercase();
        let ctx_lower = recalled_ctx.to_lowercase();
        let ctx_exp = expected
            .iter()
            .filter(|kw| ctx_lower.contains(&kw.to_lowercase()))
            .count();
        let p_exp = expected
            .iter()
            .filter(|kw| lower.contains(&kw.to_lowercase()))
            .count();
        let p_forb = forbidden
            .iter()
            .filter(|kw| lower.contains(&kw.to_lowercase()))
            .count();
        if idx < suite.probes.len() {
            step2_ctx_hits += ctx_exp;
            step2_exp_hits += p_exp;
            step2_exp_total += expected.len();
            step2_forb_hits += p_forb;
            step2_forb_total += forbidden.len();
        }
        let top_fact_preview = if recall.recalled_facts.is_empty() {
            "none".to_string()
        } else {
            recall
                .recalled_facts
                .iter()
                .map(|(e, _)| format!("#{}", e.anchor_id))
                .collect::<Vec<_>>()
                .join("+")
        };
        let tlm = &recall.telemetry;
        println!(
            "  [S2:{}:{}] void={:?} action={:?} top1={:.3} top2={:.3} guard_m={:+.3} recall={} recall_ms={:.2} probe=[g=#{:?}({:.3}) loc=#{:?}({:.3}) dis={} prov_slot={}] ctx_hit={}/{} hit={}/{} forb={} => {}",
            idx + 1,
            slot,
            recall.void_state,
            recall.action_decision,
            recall.top1_fact_score,
            recall.top2_fact_score,
            recall.guard_conflict_margin,
            top_fact_preview,
            recall_ms,
            tlm.global_top1_anchor.unwrap_or(0),
            tlm.global_top1_score,
            tlm.local_top1_anchor.unwrap_or(0),
            tlm.local_top1_score,
            tlm.channel_disagreement,
            tlm.provenance.dominant_slot,
            ctx_exp,
            expected.len(),
            p_exp,
            expected.len(),
            p_forb,
            text
        );
    }
    let mean_recall_ms = step2_recall_ms_sum / (eval_items.len().max(1) as f64);
    let decode_tps = if step2_decode_secs > 1e-6 {
        step2_decode_tokens as f64 / step2_decode_secs
    } else {
        0.0
    };
    println!(
        "Step 2 Dynamic Corpus Summary: recall_ctx_hit={:.1}% ({}/{}) | gen_expect_hit={:.1}% ({}/{}) | forbid_hit={:.1}% ({}/{}) | compile={:.1}ms mean_recall={:.2}ms decode={:.2} tok/s",
        if step2_exp_total == 0 {
            0.0
        } else {
            step2_ctx_hits as f32 / step2_exp_total as f32 * 100.0
        },
        step2_ctx_hits,
        step2_exp_total,
        if step2_exp_total == 0 {
            0.0
        } else {
            step2_exp_hits as f32 / step2_exp_total as f32 * 100.0
        },
        step2_exp_hits,
        step2_exp_total,
        if step2_forb_total == 0 {
            0.0
        } else {
            step2_forb_hits as f32 / step2_forb_total as f32 * 100.0
        },
        step2_forb_hits,
        step2_forb_total,
        compile_ms,
        mean_recall_ms,
        decode_tps,
    );

    // Step 3: Repository-Contained .gcso Persistence, Zero-Forward Seek & Cross-Model Re-Bake
    println!("\n=== Step 3: Dual-Track .gcso GOP Seek & Cross-Model Re-Bake ===");
    let containers_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("containers");
    let gcso_path = containers_dir.join("phi4_session.gcso");
    let written_bytes = timeline.save_to_path(&gcso_path)?;
    let reloaded = GcsoDualTrackTimeline::load_from_path(&gcso_path)?;
    let active_fp = GcsoDualTrackTimeline::geometry_fingerprint(adapter.geometry());
    println!(
        "Saved & Reloaded .gcso ({}): {} bytes | frames={} | keyframes={:?} | fp_match={}",
        gcso_path.display(),
        written_bytes,
        reloaded.frames().len(),
        reloaded.keyframe_positions(),
        reloaded.model_fingerprint() == active_fp,
    );

    struct SubtitleModelProjector<'a> {
        model: &'a mut Phi3Model,
        tokenizer: &'a Tokenizer,
        model_rotary_dim: usize,
        target_rotary_dim: usize,
    }

    impl SubtitleActivationEncoder for SubtitleModelProjector<'_> {
        fn encode_subtitle_activations(
            &mut self,
            record: &SourceTextRecord,
        ) -> Result<Vec<Vec<f32>>, String> {
            self.model.clear_cache();
            let prompt = format_phi4_prompt(&record.text);
            let ids = self
                .tokenizer
                .encode(prompt.as_str(), false)
                .map_err(|e| format!("rebake encode error: {e}"))?
                .get_ids()
                .to_vec();
            let (_, query_tok_qs, _) = self
                .model
                .forward_tokens(&ids, 0, None, false, true)
                .map_err(|e| format!("rebake forward error: {e}"))?;
            let mut out = Vec::with_capacity(query_tok_qs.len());
            for tok_flat in query_tok_qs {
                let mut layer_q = SequenceActivationPooler::uniform_unit_mean(
                    &tok_flat,
                    ids.len(),
                    self.model_rotary_dim,
                )?;
                layer_q.resize(self.target_rotary_dim, 0.0);
                out.push(layer_q);
            }
            Ok(out)
        }
    }

    let mut same_projector = SubtitleModelProjector {
        model: &mut model,
        tokenizer: &tokenizer,
        model_rotary_dim: adapter.geometry().q_rot_len(),
        target_rotary_dim: adapter.geometry().q_rot_len(),
    };
    let same_ctx = RebakeContext {
        adapter: &adapter,
        policy: &compiled.recall_policy,
        detector: &cut_detector,
        centroid_layers: &[],
        up_to_frame: None,
    };
    if let TimelineResumeOutcome::FastResumed { seek_state } =
        reloaded.resume_or_rebake(&same_ctx, &mut same_projector, 2)?
    {
        println!(
            "Same-Model FastResumed(frame=2): keyframe={} p_frames={} total_energy={:.2} anchors={}",
            seek_state.keyframe_index,
            seek_state.p_frames_applied,
            seek_state.total_energy,
            seek_state.anchor_ids.len(),
        );
    }

    let mut alt_geom = RopeGeometry::phi4_mini();
    alt_geom.head_dim = 128;
    alt_geom.rotary_dim = 128;
    let alt_adapter = ModelSteeringAdapter::new(alt_geom)?;
    let mut alt_projector = SubtitleModelProjector {
        model: &mut model,
        tokenizer: &tokenizer,
        model_rotary_dim: adapter.geometry().q_rot_len(),
        target_rotary_dim: alt_adapter.geometry().q_rot_len(),
    };
    let alt_ctx = RebakeContext {
        adapter: &alt_adapter,
        policy: &compiled.recall_policy,
        detector: &cut_detector,
        centroid_layers: &[],
        up_to_frame: None,
    };
    if let TimelineResumeOutcome::Rebaked {
        timeline: rebaked_tl,
        seek_state,
        source_fingerprint,
        target_fingerprint,
        rebaked_frames,
        ..
    } = reloaded.resume_or_rebake(&alt_ctx, &mut alt_projector, 2)?
    {
        let rebaked_path = containers_dir.join("phi4_rebaked_alt.gcso");
        let rebaked_bytes = rebaked_tl.save_to_path(&rebaked_path)?;
        println!(
            "Cross-Model EnvProbe Rebaked: src_fp=0x{source_fingerprint:016x} -> dst_fp=0x{target_fingerprint:016x} | frames={rebaked_frames} | saved={} ({rebaked_bytes} bytes) | seek_frame2_anchors={}",
            rebaked_path.display(),
            seek_state.anchor_ids.len(),
        );
    }

    Ok(())
}
