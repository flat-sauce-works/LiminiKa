// name: examples/candle_eval/src/model.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Minimal Phi-3 / Phi-4-mini Transformer wrapper with pre-RoPE and post-RoPE hooks.

use std::path::Path;

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Result as CResult, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};
use liminika_adapter::{
    CorpusRecallEscalationProvider, ModelSteeringAdapter, RopeGeometry, RopeLayout,
};
use liminika_core::abi::{gcso_pointer_trail_t, gcso_q7_t};
use liminika_core::traits::gateway::{
    GuardPhaseAnchor, ManifoldGatewaySignal, ObjectiveFact, ResonantSignalSlice, SubjectivePersona,
};
use serde::Deserialize;

use crate::eval::HorizonAblationMode;
use crate::LayerResonantSlices;

pub(crate) type ForwardTokensOutput = (Vec<f32>, Vec<Vec<f32>>, Vec<f32>);

#[derive(Clone, Copy)]
pub(crate) struct ResonantAnchors<'a> {
    pub(crate) facts: &'a [Vec<ObjectiveFact<[i8; 64]>>],
    pub(crate) personas: &'a [Vec<SubjectivePersona<[i8; 64]>>],
    pub(crate) guards: &'a [Vec<GuardPhaseAnchor>],
    pub(crate) negs: &'a [Vec<f32>],
    pub(crate) poss: &'a [Vec<f32>],
}

impl<'a> ResonantAnchors<'a> {
    pub(crate) fn layer_slices(&self, layer_idx: usize) -> Option<LayerResonantSlices<'a>> {
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

pub(crate) struct LayerSteer<'a, 'c, 'e> {
    pub(crate) adapter: &'a mut ModelSteeringAdapter,
    pub(crate) layer_idx: usize,
    pub(crate) q7_deltas: &'a [gcso_q7_t],
    pub(crate) persona_opt: Option<&'a [gcso_q7_t]>,
    pub(crate) resonant_opt: Option<LayerResonantSlices<'a>>,
    pub(crate) max_rad: f32,
    pub(crate) trail: &'a mut gcso_pointer_trail_t,
    pub(crate) horizon_mode: HorizonAblationMode,
    pub(crate) escalation_provider: Option<&'e mut CorpusRecallEscalationProvider<'c>>,
}

pub(crate) struct ForwardSteer<'a, 'c, 'e> {
    pub(crate) adapter: &'a mut ModelSteeringAdapter,
    pub(crate) q7s: Option<&'a [Vec<gcso_q7_t>]>,
    pub(crate) persona_q7s: Option<&'a [Vec<gcso_q7_t>]>,
    pub(crate) resonant_ctx: Option<ResonantAnchors<'a>>,
    pub(crate) rope_rad: f32,
    pub(crate) trail: &'a mut gcso_pointer_trail_t,
    pub(crate) srl: Option<(&'a [f32], &'a [f32], f32)>,
    pub(crate) horizon_mode: HorizonAblationMode,
    pub(crate) escalation_provider: Option<&'e mut CorpusRecallEscalationProvider<'c>>,
    pub(crate) compute_dispatch: Option<liminika_adapter::ComputeDispatchMask>,
    pub(crate) steer_bypass_hits: usize,
    pub(crate) steer_eligible_layers: usize,
}

fn dispatch_rope_orchestration<S: ManifoldGatewaySignal + ?Sized>(
    adapter: &mut ModelSteeringAdapter,
    horizon_mode: HorizonAblationMode,
    escalation_provider: Option<&mut CorpusRecallEscalationProvider<'_>>,
    layer_idx: usize,
    signal: &S,
    trail: &mut liminika_core::abi::gcso_pointer_trail_t,
    slice: &mut [f32],
) -> Result<(), String> {
    if matches!(
        horizon_mode,
        HorizonAblationMode::Spillover | HorizonAblationMode::SlowLoop
    ) {
        if let Some(esc) = escalation_provider {
            adapter.orchestrate_layer_rope_with_spillover(
                layer_idx, signal, esc, trail, None, 128, slice, None,
            )?;
        } else {
            let mut no_esc = ();
            adapter.orchestrate_layer_rope_with_spillover(
                layer_idx,
                signal,
                &mut no_esc,
                trail,
                None,
                128,
                slice,
                None,
            )?;
        }
    } else {
        adapter.orchestrate_layer_rope_with_action_hub(
            layer_idx, signal, trail, None, 128, slice, None,
        )?;
    }
    Ok(())
}

pub(crate) struct ProbeTrace {
    pub(crate) slot: String,
    pub(crate) text: String,
    pub(crate) d_lp: f32,
    pub(crate) p_exp: usize,
    pub(crate) p_forb: usize,
    pub(crate) delta: [u64; 7],
    pub(crate) rho: f32,
    pub(crate) steps: u32,
    pub(crate) h_tilde: f32,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Phi3Config {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) vocab_size: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f64,
    #[serde(default = "default_partial_rotary")]
    pub(crate) partial_rotary_factor: f64,
    #[serde(default)]
    pub(crate) tie_word_embeddings: bool,
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

    pub(crate) fn to_rope_geometry(&self) -> RopeGeometry {
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

pub(crate) struct RotaryCache {
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
        steer: Option<LayerSteer<'_, '_, '_>>,
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
            horizon_mode,
            mut escalation_provider,
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
                    if let Some((fact_anchors, persona_anchors, guard_anchors, neg_ref, _pos_ref)) =
                        resonant_opt
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
                                if guard_anchors.is_empty() {
                                    let composite = (fact_res, persona_res);
                                    dispatch_rope_orchestration(
                                        adapter,
                                        horizon_mode,
                                        escalation_provider.as_deref_mut(),
                                        layer_idx,
                                        &composite,
                                        trail,
                                        slice,
                                    )
                                    .map_err(candle_core::Error::Msg)?;
                                } else {
                                    let guard_res =
                                        ResonantSignalSlice::from_q7_probe(guard_anchors, &probe);
                                    let sys_res = *adapter.system_constraint();
                                    let composite = (sys_res, fact_res, persona_res, guard_res);
                                    dispatch_rope_orchestration(
                                        adapter,
                                        horizon_mode,
                                        escalation_provider.as_deref_mut(),
                                        layer_idx,
                                        &composite,
                                        trail,
                                        slice,
                                    )
                                    .map_err(candle_core::Error::Msg)?;
                                }
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

pub(crate) enum LinearLayer {
    Dense(Linear),
    Quantized(QMatMul),
}

impl LinearLayer {
    fn load_no_bias(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
        quant: Option<GgmlDType>,
        dev: &Device,
    ) -> CResult<Self> {
        match quant {
            None => Ok(Self::Dense(linear_no_bias(in_dim, out_dim, vb)?)),
            Some(q_dtype) => {
                let w = vb.get((out_dim, in_dim), "weight")?.to_device(dev)?;
                let qt = QTensor::quantize(&w, q_dtype)?;
                Ok(Self::Quantized(QMatMul::from_qtensor(qt)?))
            }
        }
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        match self {
            Self::Dense(l) => l.forward(x),
            Self::Quantized(q) => q.forward(x),
        }
    }
}

fn load_rms_norm_on_device(
    size: usize,
    eps: f64,
    vb: VarBuilder,
    dev: &Device,
) -> CResult<RmsNorm> {
    if vb.device().same_device(dev) {
        rms_norm(size, eps, vb)
    } else {
        let w = vb.get(size, "weight")?.to_device(dev)?;
        Ok(RmsNorm::new(w, eps))
    }
}

pub(crate) struct Phi3Attention {
    qkv_proj: LinearLayer,
    o_proj: LinearLayer,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Phi3Attention {
    pub(crate) fn load(
        cfg: &Phi3Config,
        vb: VarBuilder,
        quant: Option<GgmlDType>,
        dev: &Device,
    ) -> CResult<Self> {
        let head_dim = cfg.head_dim();
        let q_size = cfg.num_attention_heads * head_dim;
        let kv_size = cfg.num_key_value_heads * head_dim;
        let qkv_proj = LinearLayer::load_no_bias(
            cfg.hidden_size,
            q_size + 2 * kv_size,
            vb.pp("qkv_proj"),
            quant,
            dev,
        )?;
        let o_proj =
            LinearLayer::load_no_bias(q_size, cfg.hidden_size, vb.pp("o_proj"), quant, dev)?;
        Ok(Self {
            qkv_proj,
            o_proj,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            kv_cache: None,
        })
    }

    pub(crate) fn clear_cache(&mut self) {
        self.kv_cache = None;
    }

    fn forward(
        &mut self,
        x: &Tensor,
        rope: &RotaryCache,
        offset: usize,
        steer: Option<LayerSteer<'_, '_, '_>>,
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
        let q = q.contiguous()?;
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let mut att = (q.matmul(&k_t)? * scale)?;
        if seq_len > 1 {
            let mask = causal_mask(seq_len, offset, x.dtype(), x.device())?;
            att = att.broadcast_add(&mask)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&att)?;
        let ctx = probs
            .matmul(&v.contiguous()?)?
            .transpose(1, 2)?
            .contiguous()?
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

pub(crate) struct Phi3Mlp {
    gate_up_proj: LinearLayer,
    down_proj: LinearLayer,
    pub(crate) intermediate_size: usize,
}

impl Phi3Mlp {
    pub(crate) fn load(
        cfg: &Phi3Config,
        vb: VarBuilder,
        quant: Option<GgmlDType>,
        dev: &Device,
    ) -> CResult<Self> {
        let gate_up_proj = LinearLayer::load_no_bias(
            cfg.hidden_size,
            2 * cfg.intermediate_size,
            vb.pp("gate_up_proj"),
            quant,
            dev,
        )?;
        let down_proj = LinearLayer::load_no_bias(
            cfg.intermediate_size,
            cfg.hidden_size,
            vb.pp("down_proj"),
            quant,
            dev,
        )?;
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

pub(crate) struct Phi3Layer {
    input_layernorm: RmsNorm,
    self_attn: Phi3Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Phi3Mlp,
}

impl Phi3Layer {
    pub(crate) fn load(
        cfg: &Phi3Config,
        vb: VarBuilder,
        quant: Option<GgmlDType>,
        dev: &Device,
    ) -> CResult<Self> {
        Ok(Self {
            input_layernorm: load_rms_norm_on_device(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
                dev,
            )?,
            self_attn: Phi3Attention::load(cfg, vb.pp("self_attn"), quant, dev)?,
            post_attention_layernorm: load_rms_norm_on_device(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
                dev,
            )?,
            mlp: Phi3Mlp::load(cfg, vb.pp("mlp"), quant, dev)?,
        })
    }
}

pub(crate) struct Phi3Model {
    embed_tokens: Embedding,
    embed_on_cpu: bool,
    layers: Vec<Phi3Layer>,
    norm: RmsNorm,
    lm_head: LinearLayer,
    rope: RotaryCache,
    device: Device,
}

fn dispatch_srl_signal<S: ManifoldGatewaySignal + ?Sized>(
    ad: &mut ModelSteeringAdapter,
    rope_rad: f32,
    signal: &S,
    h_vec: &mut [f32],
    u_vec: &[f32],
    v_vec: &[f32],
) -> CResult<()> {
    if rope_rad <= -1.5 {
        let _ = ad
            .orchestrate_layer_srl_with_action_hub(signal, h_vec, u_vec, v_vec)
            .map_err(candle_core::Error::Msg)?;
    } else {
        let _ = ad
            .apply_srl_with_gateway_mut(signal, h_vec, u_vec, v_vec)
            .map_err(candle_core::Error::Msg)?;
    }
    Ok(())
}

impl Phi3Model {
    pub(crate) fn load(
        model_dir: &Path,
        cfg: &Phi3Config,
        dev: &Device,
        quant: Option<GgmlDType>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let dtype = if quant.is_some() || !dev.is_cuda() {
            DType::F32
        } else {
            DType::BF16
        };
        let files = vec![
            model_dir.join("model-00001-of-00002.safetensors"),
            model_dir.join("model-00002-of-00002.safetensors"),
        ];
        // When quantizing, load safetensors lazily on CPU so only compact quantized blocks enter VRAM.
        let vb_dev = if quant.is_some() { &Device::Cpu } else { dev };
        // SAFETY: The safetensors files are read-only model weights on disk.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, vb_dev)? };
        let mvb = vb.pp("model");

        let embed_weight = mvb
            .pp("embed_tokens")
            .get((cfg.vocab_size, cfg.hidden_size), "weight")?;
        let embed_on_cpu = quant.is_some() && dev.is_cuda();
        let embed_tokens = Embedding::new(embed_weight.clone(), cfg.hidden_size);
        let lm_head = match quant {
            None => {
                if cfg.tie_word_embeddings {
                    LinearLayer::Dense(Linear::new(embed_weight, None))
                } else {
                    LinearLayer::Dense(linear_no_bias(
                        cfg.hidden_size,
                        cfg.vocab_size,
                        vb.pp("lm_head"),
                    )?)
                }
            }
            Some(q_dtype) => {
                let w = if cfg.tie_word_embeddings {
                    embed_weight.to_device(dev)?
                } else {
                    vb.pp("lm_head")
                        .get((cfg.vocab_size, cfg.hidden_size), "weight")?
                        .to_device(dev)?
                };
                let qt = QTensor::quantize(&w, q_dtype)?;
                LinearLayer::Quantized(QMatMul::from_qtensor(qt)?)
            }
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Phi3Layer::load(
                cfg,
                mvb.pp(format!("layers.{i}")),
                quant,
                dev,
            )?);
        }
        let norm = load_rms_norm_on_device(cfg.hidden_size, cfg.rms_norm_eps, mvb.pp("norm"), dev)?;
        let rope = RotaryCache::new(cfg, 2048, dtype, dev)?;

        Ok(Self {
            embed_tokens,
            embed_on_cpu,
            layers,
            norm,
            lm_head,
            rope,
            device: dev.clone(),
        })
    }

    pub(crate) fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.self_attn.clear_cache();
        }
    }

    pub(crate) fn forward_tokens(
        &mut self,
        tokens: &[u32],
        offset: usize,
        mut steer: Option<&mut ForwardSteer<'_, '_, '_>>,
        use_edbc: bool,
        collect_bake_data: bool,
    ) -> CResult<ForwardTokensOutput> {
        let mut h = if self.embed_on_cpu {
            let input_cpu = Tensor::from_slice(tokens, (1, tokens.len()), &Device::Cpu)?;
            self.embed_tokens
                .forward(&input_cpu)?
                .to_device(&self.device)?
        } else {
            let input = Tensor::from_slice(tokens, (1, tokens.len()), &self.device)?;
            self.embed_tokens.forward(&input)?
        };
        let mut per_layer_q = if collect_bake_data {
            Vec::with_capacity(self.layers.len())
        } else {
            Vec::new()
        };

        let whole_bypass = steer
            .as_deref()
            .and_then(|st| st.compute_dispatch)
            .is_some_and(|d| d.whole_token_steering_bypass != 0);

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let steer_ref = match steer.as_deref_mut() {
                Some(st) if st.rope_rad != 0.0 => {
                    let bypassed = whole_bypass
                        || st.compute_dispatch.is_some_and(|d| {
                            layer_idx < 64 && (d.bypass_mask & (1_u64 << layer_idx)) != 0
                        });
                    if st.q7s.is_some() {
                        st.steer_eligible_layers += 1;
                        if bypassed {
                            st.steer_bypass_hits += 1;
                        }
                    }
                    st.q7s.filter(|_| !bypassed).and_then(|qs| {
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
                            horizon_mode: st.horizon_mode,
                            escalation_provider: st.escalation_provider.as_deref_mut(),
                        })
                    })
                }
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

        if !whole_bypass {
            if let Some(st) = steer.as_deref_mut() {
                if let Some((u_vec, v_vec, srl_scale)) = st.srl {
                    if srl_scale != 0.0 {
                        let ad: &mut ModelSteeringAdapter = &mut *st.adapter;
                        let q7s = st.q7s;
                        let persona_q7s = st.persona_q7s;
                        let rope_rad = st.rope_rad;
                        let mut h_vec = last.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                        if srl_scale < 0.0 {
                            let last_layer_idx = self.layers.len().saturating_sub(1);
                            let dyn_srl = st
                                .resonant_ctx
                                .and_then(|rc| rc.layer_slices(last_layer_idx))
                                .filter(|(fa, _, _, _, _)| !fa.is_empty());
                            if let Some((fact_anchors, persona_anchors, guard_anchors, _, _)) =
                                dyn_srl
                            {
                                let fact_res =
                                    ResonantSignalSlice::from_trail(fact_anchors, st.trail);
                                if !persona_anchors.is_empty() && !guard_anchors.is_empty() {
                                    let persona_res =
                                        ResonantSignalSlice::from_trail(persona_anchors, st.trail);
                                    let guard_res =
                                        ResonantSignalSlice::from_trail(guard_anchors, st.trail);
                                    let sys_res = *ad.system_constraint();
                                    let composite = (sys_res, fact_res, persona_res, guard_res);
                                    dispatch_srl_signal(
                                        ad, rope_rad, &composite, &mut h_vec, u_vec, v_vec,
                                    )?;
                                } else if !persona_anchors.is_empty() {
                                    let persona_res =
                                        ResonantSignalSlice::from_trail(persona_anchors, st.trail);
                                    let composite = (fact_res, persona_res);
                                    dispatch_srl_signal(
                                        ad, rope_rad, &composite, &mut h_vec, u_vec, v_vec,
                                    )?;
                                } else {
                                    dispatch_srl_signal(
                                        ad, rope_rad, &fact_res, &mut h_vec, u_vec, v_vec,
                                    )?;
                                }
                            } else {
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
                                    dispatch_srl_signal(
                                        ad, rope_rad, &composite, &mut h_vec, u_vec, v_vec,
                                    )?;
                                } else {
                                    dispatch_srl_signal(
                                        ad, rope_rad, &fact, &mut h_vec, u_vec, v_vec,
                                    )?;
                                }
                            }
                        } else {
                            ad.apply_srl_hidden_fused(&mut h_vec, u_vec, v_vec, srl_scale)
                                .map_err(candle_core::Error::Msg)?;
                        }
                        let len = h_vec.len();
                        last = Tensor::from_vec(h_vec, (1, len), &self.device)?
                            .to_dtype(last.dtype())?;
                    }
                }
            }
        }

        let mut logits = self
            .lm_head
            .forward(&last)?
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        if use_edbc && !whole_bypass {
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
