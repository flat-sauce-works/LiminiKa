// name: examples/candle_eval/src/model/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-architecture Transformer evaluation wrapper (`Phi-4-mini`, `Ministral 3 3B`) with
//! pre-RoPE and post-RoPE `ModelSteeringAdapter` hooks and chunked prefill.

use std::path::Path;

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Result as CResult, Tensor};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder, linear_no_bias};
use liminika_adapter::{
    AnchorWorkingSetPager, ModelSteeringAdapter, PagerStepQuery, SystemPressureReading,
};
use liminika_core::runtime::{
    ManifoldGatewaySignal, ObjectiveFact, ResonantSignalSlice, SubjectivePersona,
};

pub(crate) mod config;
pub(crate) mod layers;
pub(crate) mod rope;
#[cfg(test)]
mod tests;

pub(crate) use config::{
    ChatTemplate, ChatTemplateArg, ChatTemplateKind, ModelArchArg, ModelConfig, active_layers_mask,
    clip_layer_mask, resolve_safetensors_files, top_half_layer_mask,
};
pub(crate) use layers::{LinearLayer, TransformerLayer, load_rms_norm_on_device};
pub(crate) use rope::{
    EvalOscilloscopeSignal, ForwardSteer, LayerSteer, ProbeCoordinateFrame, ResonantAnchors,
    RotaryCache,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum KvCacheMode {
    #[default]
    Append,
    IsolatedProbe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LogitsFetchMode {
    FullLogits,
    DeviceArgmaxOnly,
    SkipLogits,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum LogitsOutcome {
    ObservedFull(Vec<f32>),
    UnobservedDeviceArgmax(u32),
    SkippedIntermediate,
}

/// Default prefill chunk width (`256` tokens) preventing $O(L^2)$ attention VRAM spikes while
/// keeping short single-turn prompts (`<= 256` tokens) on the exact single-chunk path.
pub(crate) const DEFAULT_PREFILL_CHUNK_SIZE: usize = 256;

/// Coarse-grained telemetry emitted by [`CausalLmModel::prefill_tokens_chunked`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ChunkedPrefillReport {
    pub(crate) total_tokens: usize,
    pub(crate) chunk_size: usize,
    pub(crate) chunks_executed: usize,
    pub(crate) liminal_steps: usize,
    pub(crate) pager_page_ins: u64,
    pub(crate) pager_page_outs: u64,
}

impl LogitsOutcome {
    pub(crate) fn into_full_logits(self) -> CResult<Vec<f32>> {
        match self {
            Self::ObservedFull(v) => Ok(v),
            Self::UnobservedDeviceArgmax(_) | Self::SkippedIntermediate => Err(
                candle_core::Error::Msg("expected FullLogits outcome".to_string()),
            ),
        }
    }
}

pub(crate) fn extract_device_argmax_token(logits_1d: &Tensor) -> CResult<u32> {
    if let Ok(tok_t) = logits_1d.argmax(0)
        && let Ok(tok_id) = tok_t.to_scalar::<u32>()
    {
        return Ok(tok_id);
    }
    logits_1d
        .to_dtype(DType::F32)?
        .argmax(0)?
        .to_scalar::<u32>()
}

pub(crate) type ForwardTokensOutput = (LogitsOutcome, Vec<Vec<f32>>, Vec<f32>);

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

fn dispatch_rank1_signal_inner<S: ManifoldGatewaySignal + ?Sized>(
    ad: &mut ModelSteeringAdapter,
    rope_rad: f32,
    signal: &S,
    h_vec: &mut [f32],
    u_vec: &[f32],
    v_vec: &[f32],
) -> CResult<()> {
    if rope_rad <= -1.5 {
        let _ = ad
            .orchestrate_layer_rank1_with_substrate(signal, h_vec, u_vec, v_vec)
            .map_err(candle_core::Error::Msg)?;
    } else {
        let _ = ad
            .apply_rank1_with_gateway_mut(signal, h_vec, u_vec, v_vec)
            .map_err(candle_core::Error::Msg)?;
    }
    Ok(())
}

// Internal Rank-1 dispatcher forwards per-layer evaluation slices without intermediate wrappers.
#[allow(clippy::too_many_arguments)]
fn dispatch_rank1_signal<S: ManifoldGatewaySignal + ?Sized>(
    ad: &mut ModelSteeringAdapter,
    rope_rad: f32,
    layer_idx: usize,
    signal: &S,
    h_vec: &mut [f32],
    u_vec: &[f32],
    v_vec: &[f32],
    phase_clamp_override: Option<f32>,
    random_push_seed: Option<u64>,
) -> CResult<()> {
    if phase_clamp_override.is_some() || random_push_seed.is_some() {
        let wrapped = EvalOscilloscopeSignal {
            inner: signal,
            layer_idx,
            phase_clamp_override,
            random_push_seed,
        };
        dispatch_rank1_signal_inner(ad, rope_rad, &wrapped, h_vec, u_vec, v_vec)
    } else {
        dispatch_rank1_signal_inner(ad, rope_rad, signal, h_vec, u_vec, v_vec)
    }
}

pub(crate) struct CausalLmModel {
    pub(crate) embed_tokens: Embedding,
    pub(crate) embed_on_cpu: bool,
    pub(crate) layers: Vec<TransformerLayer>,
    pub(crate) norm: RmsNorm,
    pub(crate) lm_head: LinearLayer,
    pub(crate) rope: RotaryCache,
    pub(crate) device: Device,
}

impl CausalLmModel {
    pub(crate) fn load(
        model_dir: &Path,
        cfg: &ModelConfig,
        dev: &Device,
        quant: Option<GgmlDType>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let dtype = if quant.is_some() || !dev.is_cuda() {
            DType::F32
        } else {
            DType::BF16
        };
        let files = resolve_safetensors_files(model_dir)?;
        // When quantizing, load safetensors lazily on CPU so only compact quantized blocks enter VRAM.
        let vb_dev = if quant.is_some() { &Device::Cpu } else { dev };
        // SAFETY: The safetensors files are read-only model weights on disk.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, vb_dev)? };
        let mvb = vb.pp(&cfg.tensor_prefix);
        let lm_head_vb = if let Some(parent) = cfg.tensor_prefix.strip_suffix(".model") {
            vb.pp(parent).pp("lm_head")
        } else {
            vb.pp("lm_head")
        };

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
                    LinearLayer::Dense(linear_no_bias(cfg.hidden_size, cfg.vocab_size, lm_head_vb)?)
                }
            }
            Some(q_dtype) => {
                let w = if cfg.tie_word_embeddings {
                    embed_weight.to_device(dev)?
                } else {
                    lm_head_vb
                        .get((cfg.vocab_size, cfg.hidden_size), "weight")?
                        .to_device(dev)?
                };
                let qt = QTensor::quantize(&w, q_dtype)?;
                LinearLayer::Quantized(QMatMul::from_qtensor(qt)?)
            }
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(TransformerLayer::load(
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

    pub(crate) fn rope_cached_len(&self) -> usize {
        self.rope.cached_len()
    }

    pub(crate) fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.self_attn.clear_cache();
        }
    }

    pub(crate) fn kv_len(&self) -> usize {
        self.layers
            .first()
            .map_or(0, |layer| layer.self_attn.kv_len())
    }

    pub(crate) fn truncate_cache(&mut self, target_len: usize) -> CResult<()> {
        if target_len == 0 {
            self.clear_cache();
            return Ok(());
        }
        for layer in &mut self.layers {
            layer.self_attn.truncate_cache(target_len)?;
        }
        Ok(())
    }

    pub(crate) fn embed_token_ids(&self, tokens: &[u32]) -> CResult<Tensor> {
        if tokens.is_empty() {
            return Err(candle_core::Error::Msg(
                "embed_token_ids requires at least one token".to_string(),
            ));
        }
        if self.embed_on_cpu {
            let input_cpu = Tensor::from_slice(tokens, (1, tokens.len()), &Device::Cpu)?;
            self.embed_tokens
                .forward(&input_cpu)?
                .to_device(&self.device)
        } else {
            let input = Tensor::from_slice(tokens, (1, tokens.len()), &self.device)?;
            self.embed_tokens.forward(&input)
        }
    }

    pub(crate) fn active_dtype(&self) -> DType {
        self.embed_tokens.embeddings().dtype()
    }

    // Forward entrypoint mirrors `forward_embeds` flags directly across evaluation callers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_tokens(
        &mut self,
        tokens: &[u32],
        offset: usize,
        steer: Option<&mut ForwardSteer<'_, '_, '_>>,
        use_entropy_gate: bool,
        collect_bake_data: bool,
        fetch_mode: LogitsFetchMode,
        kv_mode: KvCacheMode,
    ) -> CResult<ForwardTokensOutput> {
        if tokens.is_empty() {
            return Err(candle_core::Error::Msg(
                "forward_tokens requires at least one token".to_string(),
            ));
        }
        let h = self.embed_token_ids(tokens)?;
        self.forward_embeds(
            &h,
            offset,
            steer,
            use_entropy_gate,
            collect_bake_data,
            fetch_mode,
            kv_mode,
        )
    }

    /// Runs the transformer backbone directly from pre-built `[1, seq_len, hidden_size]`
    /// embeddings (supporting multimodal `[IMG]` splice inputs) across both `Append` and
    /// `IsolatedProbe` KV modes with dynamic RoPE expansion and Llama-4 scaling parity.
    // Multimodal forward entrypoint mirrors `forward_tokens` flags directly across evaluation callers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_embeds(
        &mut self,
        embeds: &Tensor,
        offset: usize,
        mut steer: Option<&mut ForwardSteer<'_, '_, '_>>,
        use_entropy_gate: bool,
        collect_bake_data: bool,
        fetch_mode: LogitsFetchMode,
        kv_mode: KvCacheMode,
    ) -> CResult<ForwardTokensOutput> {
        if kv_mode == KvCacheMode::Append {
            debug_assert_eq!(self.kv_len(), offset);
        }
        let (batch, seq_len, _) = embeds.dims3()?;
        if batch != 1 || seq_len == 0 {
            return Err(candle_core::Error::Msg(format!(
                "forward_embeds requires [1, seq_len >= 1, hidden], got [{batch}, {seq_len}, _]"
            )));
        }
        let required_len = offset.checked_add(seq_len).ok_or_else(|| {
            candle_core::Error::Msg("forward_embeds offset + len overflow".to_string())
        })?;
        self.rope.ensure_capacity(required_len)?;
        let mut h = if embeds.device().same_device(&self.device) {
            embeds.clone()
        } else {
            embeds.to_device(&self.device)?
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
                    let layer_enabled = st.layer_steer_mask.map_or_else(
                        || st.adapter.geometry().is_rope_layer(layer_idx),
                        |mask| layer_idx < 64 && ((mask >> layer_idx) & 1) != 0,
                    );
                    if !layer_enabled {
                        None
                    } else {
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
                                probe_frame: st.probe_frame,
                                phase_clamp_override: st.phase_clamp_override,
                                random_push_seed: st.random_push_seed,
                            })
                        })
                    }
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
                kv_mode,
            )?;
            if let Some(q) = q_opt {
                per_layer_q.push(q);
            }
            h = (h + attn_out)?;
            let post = layer.post_attention_layernorm.forward(&h)?;
            let mlp_out = layer.mlp.forward(&post)?;
            h = (h + mlp_out)?;
        }

        if matches!(fetch_mode, LogitsFetchMode::SkipLogits) && !collect_bake_data {
            return Ok((LogitsOutcome::SkippedIntermediate, per_layer_q, Vec::new()));
        }

        let h = self.norm.forward(&h)?;
        let mut last = h.i((0, seq_len - 1, ..))?.unsqueeze(0)?;
        let last_hidden = if collect_bake_data {
            last.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?
        } else {
            Vec::new()
        };

        if !whole_bypass
            && let Some(st) = steer.as_deref_mut()
            && let Some((u_vec, v_vec, rank1_scale)) = st.rank1
            && rank1_scale != 0.0
        {
            let ad: &mut ModelSteeringAdapter = &mut *st.adapter;
            let q7s = st.q7s;
            let persona_q7s = st.persona_q7s;
            let rope_rad = st.rope_rad;
            let phase_clamp_override = st.phase_clamp_override;
            let random_push_seed = st.random_push_seed;
            let last_layer_idx = self.layers.len().saturating_sub(1);
            let mut h_vec = last.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            if rank1_scale < 0.0 {
                let dyn_rank1 = st
                    .resonant_ctx
                    .and_then(|rc| rc.layer_slices(last_layer_idx))
                    .filter(|(fa, _, _, _, _)| !fa.is_empty());
                if let Some((fact_anchors, persona_anchors, guard_anchors, _, _)) = dyn_rank1 {
                    let fact_res = ResonantSignalSlice::from_trail(fact_anchors, st.trail);
                    if !persona_anchors.is_empty() && !guard_anchors.is_empty() {
                        let persona_res =
                            ResonantSignalSlice::from_trail(persona_anchors, st.trail);
                        let guard_res = ResonantSignalSlice::from_trail(guard_anchors, st.trail);
                        let sys_res = *ad.system_constraint();
                        let composite = (sys_res, fact_res, persona_res, guard_res);
                        dispatch_rank1_signal(
                            ad,
                            rope_rad,
                            last_layer_idx,
                            &composite,
                            &mut h_vec,
                            u_vec,
                            v_vec,
                            phase_clamp_override,
                            random_push_seed,
                        )?;
                    } else if !persona_anchors.is_empty() {
                        let persona_res =
                            ResonantSignalSlice::from_trail(persona_anchors, st.trail);
                        let composite = (fact_res, persona_res);
                        dispatch_rank1_signal(
                            ad,
                            rope_rad,
                            last_layer_idx,
                            &composite,
                            &mut h_vec,
                            u_vec,
                            v_vec,
                            phase_clamp_override,
                            random_push_seed,
                        )?;
                    } else {
                        dispatch_rank1_signal(
                            ad,
                            rope_rad,
                            last_layer_idx,
                            &fact_res,
                            &mut h_vec,
                            u_vec,
                            v_vec,
                            phase_clamp_override,
                            random_push_seed,
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
                        dispatch_rank1_signal(
                            ad,
                            rope_rad,
                            last_layer_idx,
                            &composite,
                            &mut h_vec,
                            u_vec,
                            v_vec,
                            phase_clamp_override,
                            random_push_seed,
                        )?;
                    } else {
                        dispatch_rank1_signal(
                            ad,
                            rope_rad,
                            last_layer_idx,
                            &fact,
                            &mut h_vec,
                            u_vec,
                            v_vec,
                            phase_clamp_override,
                            random_push_seed,
                        )?;
                    }
                }
            } else {
                ad.apply_rank1_hidden_fused(&mut h_vec, u_vec, v_vec, rank1_scale)
                    .map_err(candle_core::Error::Msg)?;
            }
            let len = h_vec.len();
            last = Tensor::from_vec(h_vec, (1, len), &self.device)?.to_dtype(last.dtype())?;
        }

        let logits_1d = self.lm_head.forward(&last)?.squeeze(0)?;
        let outcome = match fetch_mode {
            LogitsFetchMode::SkipLogits => LogitsOutcome::SkippedIntermediate,
            LogitsFetchMode::DeviceArgmaxOnly if !use_entropy_gate || whole_bypass => {
                let tok_id = extract_device_argmax_token(&logits_1d)?;
                LogitsOutcome::UnobservedDeviceArgmax(tok_id)
            }
            _ => {
                let mut logits = logits_1d.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                if use_entropy_gate
                    && !whole_bypass
                    && let Some(st) = steer
                {
                    let ad = &mut *st.adapter;
                    if st.rope_rad <= -1.5 {
                        let _ = ad
                            .measure_entropy_state(&logits)
                            .map_err(candle_core::Error::Msg)?;
                    } else {
                        let _ = ad
                            .step_entropy_logits(&mut logits)
                            .map_err(candle_core::Error::Msg)?;
                    }
                }
                LogitsOutcome::ObservedFull(logits)
            }
        };
        Ok((outcome, per_layer_q, last_hidden))
    }

    /// Prefills an arbitrarily long token sequence into the persistent KV cache in bounded chunks
    /// (`chunk_size`, default [`DEFAULT_PREFILL_CHUNK_SIZE`]), preventing $O(L^2)$ attention VRAM
    /// spikes while preserving mathematical equivalence to monolithic causal prefill.
    // Chunked prefill entrypoint keeps explicit steering/callback parameters for evaluation callers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_tokens_chunked(
        &mut self,
        tokens: &[u32],
        start_offset: usize,
        chunk_size: usize,
        mut steer: Option<&mut ForwardSteer<'_, '_, '_>>,
        mut pager: Option<&mut AnchorWorkingSetPager>,
        use_entropy_gate: bool,
        final_fetch_mode: LogitsFetchMode,
    ) -> CResult<(LogitsOutcome, ChunkedPrefillReport)> {
        if tokens.is_empty() {
            return Err(candle_core::Error::Msg(
                "prefill_tokens_chunked requires at least one token".to_string(),
            ));
        }
        let required_len = start_offset.checked_add(tokens.len()).ok_or_else(|| {
            candle_core::Error::Msg("prefill_tokens_chunked offset + len overflow".to_string())
        })?;
        self.rope.ensure_capacity(required_len)?;

        let eff_chunk = chunk_size.max(1);
        let mut pos = 0_usize;
        let mut chunks_executed = 0_usize;
        let mut liminal_steps = 0_usize;
        let pager_in_before = pager.as_ref().map_or(0, |p| p.stats().page_ins);
        let pager_out_before = pager.as_ref().map_or(0, |p| p.stats().page_outs);
        let mut last_outcome = LogitsOutcome::SkippedIntermediate;

        while pos < tokens.len() {
            let end = (pos + eff_chunk).min(tokens.len());
            let chunk = &tokens[pos..end];
            let is_last = end == tokens.len();
            let chunk_offset = start_offset + pos;

            let exec_mask_opt = if let Some(st) = steer.as_deref_mut() {
                if st.rope_rad != 0.0 {
                    let mut dispatch = st.adapter.step_compute_dispatch_with_spine(
                        SystemPressureReading {
                            frame_time_pressure_q15: 8192,
                            bandwidth_pressure_q15: 8192,
                            l3_load_pressure_q15: 8192,
                            irreversible_step: u8::from(is_last),
                            _pad: 0,
                        },
                        false,
                    );
                    // Never bypass prefill KV population; preserve Spine safety floor across chunks.
                    dispatch.whole_token_steering_bypass = 0;
                    st.compute_dispatch = Some(dispatch);
                    liminal_steps += 1;
                    if let Some(pg) = pager.as_deref_mut() {
                        let _ = pg.step_working_set(&PagerStepQuery {
                            phase_profile: st.trail.accumulated_phase_delta,
                            lacuna_partition: *st.adapter.last_lacuna_partition(),
                            desired_count: 4,
                            allow_exterior_staging: false,
                        });
                    }
                    Some(dispatch.execute_mask)
                } else {
                    None
                }
            } else {
                None
            };

            let fetch_mode = if is_last {
                final_fetch_mode
            } else {
                LogitsFetchMode::SkipLogits
            };
            let (outcome, _, _) = self.forward_tokens(
                chunk,
                chunk_offset,
                steer.as_deref_mut(),
                is_last && use_entropy_gate,
                false,
                fetch_mode,
                KvCacheMode::Append,
            )?;
            last_outcome = outcome;

            if let (Some(exec_mask), Some(st)) = (exec_mask_opt, steer.as_deref_mut()) {
                st.adapter.commit_compute_exit_reward(exec_mask, 64);
                if !is_last {
                    st.adapter.step_distortion_tension_hot_path(0, 0, false);
                }
            }

            chunks_executed += 1;
            pos = end;
        }

        let pager_page_ins = pager
            .as_ref()
            .map_or(0, |p| p.stats().page_ins.saturating_sub(pager_in_before));
        let pager_page_outs = pager
            .as_ref()
            .map_or(0, |p| p.stats().page_outs.saturating_sub(pager_out_before));
        Ok((
            last_outcome,
            ChunkedPrefillReport {
                total_tokens: tokens.len(),
                chunk_size: eff_chunk,
                chunks_executed,
                liminal_steps,
                pager_page_ins,
                pager_page_outs,
            },
        ))
    }
}
