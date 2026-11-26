// name: examples/candle_eval/src/model/layers.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Transformer attention, SwiGLU MLP, and linear/quantized layer blocks supporting both fused
//! (`Phi-4-mini`) and separate (`Ministral 3 3B`) projection layouts.

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Result as CResult, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, Linear, RmsNorm, VarBuilder};

use crate::model::config::ModelConfig;
use crate::model::rope::{LayerSteer, RotaryCache};
use crate::model::KvCacheMode;

pub(crate) enum LinearLayer {
    Dense(Linear),
    Quantized(QMatMul),
}

impl LinearLayer {
    pub(crate) fn load_no_bias(
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

    pub(crate) fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        match self {
            Self::Dense(l) => l.forward(x),
            Self::Quantized(q) => q.forward(x),
        }
    }
}

pub(crate) fn load_rms_norm_on_device(
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

pub(crate) enum QkvProjection {
    Fused(LinearLayer),
    Separate {
        q_proj: LinearLayer,
        k_proj: LinearLayer,
        v_proj: LinearLayer,
    },
}

pub(crate) struct TransformerAttention {
    pub(crate) qkv: QkvProjection,
    pub(crate) o_proj: LinearLayer,
    pub(crate) num_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) kv_cache: Option<(Tensor, Tensor)>,
}

impl TransformerAttention {
    pub(crate) fn load(
        cfg: &ModelConfig,
        vb: VarBuilder,
        quant: Option<GgmlDType>,
        dev: &Device,
    ) -> CResult<Self> {
        let head_dim = cfg.head_dim();
        let q_size = cfg.num_attention_heads * head_dim;
        let kv_size = cfg.num_key_value_heads * head_dim;
        let qkv = if cfg.separate_projections {
            let q_proj =
                LinearLayer::load_no_bias(cfg.hidden_size, q_size, vb.pp("q_proj"), quant, dev)?;
            let k_proj =
                LinearLayer::load_no_bias(cfg.hidden_size, kv_size, vb.pp("k_proj"), quant, dev)?;
            let v_proj =
                LinearLayer::load_no_bias(cfg.hidden_size, kv_size, vb.pp("v_proj"), quant, dev)?;
            QkvProjection::Separate {
                q_proj,
                k_proj,
                v_proj,
            }
        } else {
            let qkv_proj = LinearLayer::load_no_bias(
                cfg.hidden_size,
                q_size + 2 * kv_size,
                vb.pp("qkv_proj"),
                quant,
                dev,
            )?;
            QkvProjection::Fused(qkv_proj)
        };
        let o_proj =
            LinearLayer::load_no_bias(q_size, cfg.hidden_size, vb.pp("o_proj"), quant, dev)?;
        Ok(Self {
            qkv,
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

    pub(crate) fn kv_len(&self) -> usize {
        self.kv_cache
            .as_ref()
            .and_then(|(k, _)| k.dim(2).ok())
            .unwrap_or(0)
    }

    pub(crate) fn truncate_cache(&mut self, target_len: usize) -> CResult<()> {
        if target_len == 0 {
            self.kv_cache = None;
            return Ok(());
        }
        let cur = self.kv_len();
        if target_len > cur {
            return Err(candle_core::Error::Msg(format!(
                "truncate_cache target_len {target_len} exceeds current kv_len {cur}"
            )));
        }
        if target_len < cur {
            if let Some((ref k, ref v)) = self.kv_cache {
                self.kv_cache = Some((k.narrow(2, 0, target_len)?, v.narrow(2, 0, target_len)?));
            }
        }
        Ok(())
    }

    pub(crate) fn forward(
        &mut self,
        x: &Tensor,
        rope: &RotaryCache,
        offset: usize,
        steer: Option<LayerSteer<'_, '_, '_>>,
        collect_q_heads: bool,
        kv_mode: KvCacheMode,
    ) -> CResult<(Tensor, Option<Vec<f32>>)> {
        if kv_mode == KvCacheMode::Append {
            debug_assert_eq!(self.kv_len(), offset);
        }
        let (b, seq_len, _) = x.dims3()?;
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        let (q, k, v) = match &self.qkv {
            QkvProjection::Fused(qkv_proj) => {
                let qkv = qkv_proj.forward(x)?;
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
                (q, k, v)
            }
            QkvProjection::Separate {
                q_proj,
                k_proj,
                v_proj,
            } => {
                let q = q_proj
                    .forward(x)?
                    .reshape((b, seq_len, self.num_heads, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?;
                let k = k_proj
                    .forward(x)?
                    .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?;
                let v = v_proj
                    .forward(x)?
                    .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?;
                (q, k, v)
            }
        };

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
        let q = rope.apply_query_temp_scale(q, offset, seq_len)?;
        let k = rope.apply_with_adapter(&k, offset, None)?;

        let (k, v, mask_offset) = match kv_mode {
            KvCacheMode::Append => {
                let (k, v) = match &self.kv_cache {
                    Some((prev_k, prev_v)) => (
                        Tensor::cat(&[prev_k, &k], 2)?,
                        Tensor::cat(&[prev_v, &v], 2)?,
                    ),
                    None => (k, v),
                };
                self.kv_cache = Some((k.clone(), v.clone()));
                (k, v, offset)
            }
            KvCacheMode::IsolatedProbe => (k, v, 0),
        };

        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let q = q.contiguous()?;
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let mut att = (q.matmul(&k_t)? * scale)?;
        if seq_len > 1 {
            let mask = causal_mask(seq_len, mask_offset, x.dtype(), x.device())?;
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

pub(crate) enum MlpProjection {
    FusedGateUp {
        gate_up_proj: LinearLayer,
        intermediate_size: usize,
    },
    SeparateGateUp {
        gate_proj: LinearLayer,
        up_proj: LinearLayer,
    },
}

pub(crate) struct TransformerMlp {
    pub(crate) proj: MlpProjection,
    pub(crate) down_proj: LinearLayer,
}

impl TransformerMlp {
    pub(crate) fn load(
        cfg: &ModelConfig,
        vb: VarBuilder,
        quant: Option<GgmlDType>,
        dev: &Device,
    ) -> CResult<Self> {
        let proj = if cfg.separate_projections {
            let gate_proj = LinearLayer::load_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("gate_proj"),
                quant,
                dev,
            )?;
            let up_proj = LinearLayer::load_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("up_proj"),
                quant,
                dev,
            )?;
            MlpProjection::SeparateGateUp { gate_proj, up_proj }
        } else {
            let gate_up_proj = LinearLayer::load_no_bias(
                cfg.hidden_size,
                2 * cfg.intermediate_size,
                vb.pp("gate_up_proj"),
                quant,
                dev,
            )?;
            MlpProjection::FusedGateUp {
                gate_up_proj,
                intermediate_size: cfg.intermediate_size,
            }
        };
        let down_proj = LinearLayer::load_no_bias(
            cfg.intermediate_size,
            cfg.hidden_size,
            vb.pp("down_proj"),
            quant,
            dev,
        )?;
        Ok(Self { proj, down_proj })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let act = match &self.proj {
            MlpProjection::FusedGateUp {
                gate_up_proj,
                intermediate_size,
            } => {
                let up = gate_up_proj.forward(x)?;
                let gate = up.narrow(D::Minus1, 0, *intermediate_size)?;
                let val = up.narrow(D::Minus1, *intermediate_size, *intermediate_size)?;
                (candle_nn::ops::silu(&gate)? * val)?
            }
            MlpProjection::SeparateGateUp { gate_proj, up_proj } => {
                let gate = gate_proj.forward(x)?;
                let val = up_proj.forward(x)?;
                (candle_nn::ops::silu(&gate)? * val)?
            }
        };
        self.down_proj.forward(&act)
    }
}

pub(crate) struct TransformerLayer {
    pub(crate) input_layernorm: RmsNorm,
    pub(crate) self_attn: TransformerAttention,
    pub(crate) post_attention_layernorm: RmsNorm,
    pub(crate) mlp: TransformerMlp,
}

impl TransformerLayer {
    pub(crate) fn load(
        cfg: &ModelConfig,
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
            self_attn: TransformerAttention::load(cfg, vb.pp("self_attn"), quant, dev)?,
            post_attention_layernorm: load_rms_norm_on_device(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
                dev,
            )?,
            mlp: TransformerMlp::load(cfg, vb.pp("mlp"), quant, dev)?,
        })
    }
}
