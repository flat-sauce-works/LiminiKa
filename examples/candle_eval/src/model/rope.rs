// name: examples/candle_eval/src/model/rope.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dynamic RoPE cache (standard + YaRN + Llama-4 long-context query scaling) with SSOT
//! `HalfSplit`/`Interleaved` dispatch and pre-RoPE/post-RoPE `ModelSteeringAdapter` hooks.

use candle_core::{D, DType, Device, Result as CResult, Tensor};
use liminika_adapter::{CorpusRecallEscalationProvider, ModelSteeringAdapter, RopeLayout};
use liminika_core::abi::{lmnk_pointer_trail_t, lmnk_q7_t};
use liminika_core::foundation::TaggedPtr;
use liminika_core::liminal::MAX_ATTENTION_HEADS;
use liminika_core::runtime::{
    CausalProvenanceSignal, GatewayFrame, GuardPhaseAnchor, ManifoldGatewaySignal, ObjectiveFact,
    ResonantSignalSlice, SubjectivePersona,
};

use crate::LayerResonantSlices;
use crate::eval::HorizonAblationMode;
use crate::model::config::{ModelConfig, YarnRopeConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ProbeCoordinateFrame {
    #[default]
    PreRope,
    PostRope,
}

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
    pub(crate) q7_deltas: &'a [lmnk_q7_t],
    pub(crate) persona_opt: Option<&'a [lmnk_q7_t]>,
    pub(crate) resonant_opt: Option<LayerResonantSlices<'a>>,
    pub(crate) max_rad: f32,
    pub(crate) trail: &'a mut lmnk_pointer_trail_t,
    pub(crate) horizon_mode: HorizonAblationMode,
    pub(crate) escalation_provider: Option<&'e mut CorpusRecallEscalationProvider<'c>>,
    pub(crate) probe_frame: ProbeCoordinateFrame,
    pub(crate) phase_clamp_override: Option<f32>,
    pub(crate) random_push_seed: Option<u64>,
}

pub(crate) struct ForwardSteer<'a, 'c, 'e> {
    pub(crate) adapter: &'a mut ModelSteeringAdapter,
    pub(crate) q7s: Option<&'a [Vec<lmnk_q7_t>]>,
    pub(crate) persona_q7s: Option<&'a [Vec<lmnk_q7_t>]>,
    pub(crate) resonant_ctx: Option<ResonantAnchors<'a>>,
    pub(crate) rope_rad: f32,
    pub(crate) trail: &'a mut lmnk_pointer_trail_t,
    pub(crate) rank1: Option<(&'a [f32], &'a [f32], f32)>,
    pub(crate) horizon_mode: HorizonAblationMode,
    pub(crate) escalation_provider: Option<&'e mut CorpusRecallEscalationProvider<'c>>,
    pub(crate) compute_dispatch: Option<liminika_adapter::ComputeDispatchMask>,
    pub(crate) layer_steer_mask: Option<u64>,
    pub(crate) probe_frame: ProbeCoordinateFrame,
    pub(crate) phase_clamp_override: Option<f32>,
    pub(crate) random_push_seed: Option<u64>,
    pub(crate) steer_bypass_hits: usize,
    pub(crate) steer_eligible_layers: usize,
}

#[inline]
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Generates a deterministic norm-matched random Q7 phase perturbation over `0..active_heads`
/// using `splitmix64` (zero external RNG dependencies) while preserving zero-aperture frames.
pub(crate) fn generate_norm_matched_random_q7(
    reference: &[lmnk_q7_t; MAX_ATTENTION_HEADS],
    active_heads: usize,
    seed: u64,
    layer_idx: usize,
) -> [lmnk_q7_t; MAX_ATTENTION_HEADS] {
    let n = active_heads.clamp(1, MAX_ATTENTION_HEADS);
    let ref_sq: f64 = reference[..n]
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum();
    if ref_sq <= 0.0 {
        return [0_i8; MAX_ATTENTION_HEADS];
    }
    let ref_norm = ref_sq.sqrt();
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ ((layer_idx as u64)
            .wrapping_add(1)
            .wrapping_mul(0xBF58_476D_1CE4_E5B9));
    let mut raw = [0.0_f64; MAX_ATTENTION_HEADS];
    let mut raw_sq = 0.0_f64;
    for slot in raw.iter_mut().take(n) {
        let bits = splitmix64_next(&mut state);
        let unit = ((bits >> 11) as f64) / ((1_u64 << 53) as f64);
        let signed = unit * 2.0 - 1.0;
        *slot = signed;
        raw_sq += signed * signed;
    }
    if raw_sq <= 1e-18 {
        raw[0] = 1.0;
        raw_sq = 1.0;
    }
    let scale = ref_norm / raw_sq.sqrt();
    let mut out = [0_i8; MAX_ATTENTION_HEADS];
    for i in 0..n {
        out[i] = (raw[i] * scale).round().clamp(-127.0, 127.0) as lmnk_q7_t;
    }
    out
}

/// Evaluation-only [`ManifoldGatewaySignal`] wrapper that injects `PhaseClamp` overrides and
/// norm-matched random Q7 phase controls strictly inside `examples/candle_eval` without modifying
/// `liminika-core` or adding adapter backdoors.
pub(crate) struct EvalOscilloscopeSignal<'a, S: ManifoldGatewaySignal + ?Sized> {
    pub(crate) inner: &'a S,
    pub(crate) layer_idx: usize,
    pub(crate) phase_clamp_override: Option<f32>,
    pub(crate) random_push_seed: Option<u64>,
}

impl<S: ManifoldGatewaySignal + ?Sized> CausalProvenanceSignal for EvalOscilloscopeSignal<'_, S> {
    #[inline]
    fn causal_provenance_bits(&self) -> u32 {
        self.inner.causal_provenance_bits()
    }

    #[inline]
    fn causal_anomaly_severity(&self) -> f32 {
        self.inner.causal_anomaly_severity()
    }
}

impl<S: ManifoldGatewaySignal + ?Sized> ManifoldGatewaySignal for EvalOscilloscopeSignal<'_, S> {
    #[inline]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        let mut frame = self.inner.synthesize_gateway_frame();
        if let Some(clamp_rad) = self.phase_clamp_override {
            frame.phase_clamp_rad = clamp_rad;
        }
        if let Some(seed) = self.random_push_seed {
            let active_heads = (frame.active_head_count as usize).clamp(1, MAX_ATTENTION_HEADS);
            let rand_q7 = generate_norm_matched_random_q7(
                &frame.phase_deltas,
                active_heads,
                seed,
                self.layer_idx,
            );
            frame.phase_deltas = rand_q7;
            let sum: i32 = rand_q7[..active_heads].iter().map(|&v| i32::from(v)).sum();
            let mean = (sum / active_heads as i32).clamp(-127, 127) as lmnk_q7_t;
            frame.mean_phase_tag_q7 = mean;
            frame.tagged_ptr = TaggedPtr(frame.tagged_ptr).with_phase(mean).raw();
        }
        frame
    }
}

fn dispatch_rope_orchestration_inner<S: ManifoldGatewaySignal + ?Sized>(
    adapter: &mut ModelSteeringAdapter,
    horizon_mode: HorizonAblationMode,
    escalation_provider: Option<&mut CorpusRecallEscalationProvider<'_>>,
    layer_idx: usize,
    signal: &S,
    trail: &mut lmnk_pointer_trail_t,
    slice: &mut [f32],
) -> Result<(), String> {
    if matches!(
        horizon_mode,
        HorizonAblationMode::Spillover | HorizonAblationMode::SlowLoop
    ) {
        let step_input = liminika_adapter::LayerRopeStepInput {
            layer_idx,
            signal,
            entropy_h_tilde_opt: None,
            unresolved_damping_u7: 128,
            q_rot_inout: slice,
            logits_opt: None,
        };
        if let Some(esc) = escalation_provider {
            adapter.orchestrate_layer_rope_with_spillover(step_input, esc, trail)?;
        } else {
            let mut no_esc = ();
            adapter.orchestrate_layer_rope_with_spillover(step_input, &mut no_esc, trail)?;
        }
    } else {
        let step_input = liminika_adapter::LayerRopeStepInput {
            layer_idx,
            signal,
            entropy_h_tilde_opt: None,
            unresolved_damping_u7: 128,
            q_rot_inout: slice,
            logits_opt: None,
        };
        adapter.orchestrate_layer_rope_with_substrate(step_input, trail)?;
    }
    Ok(())
}

// Internal RoPE steering dispatcher forwards per-layer evaluation slices without intermediate structs.
#[allow(clippy::too_many_arguments)]
fn dispatch_rope_orchestration<S: ManifoldGatewaySignal + ?Sized>(
    adapter: &mut ModelSteeringAdapter,
    horizon_mode: HorizonAblationMode,
    escalation_provider: Option<&mut CorpusRecallEscalationProvider<'_>>,
    layer_idx: usize,
    signal: &S,
    trail: &mut lmnk_pointer_trail_t,
    slice: &mut [f32],
    phase_clamp_override: Option<f32>,
    random_push_seed: Option<u64>,
) -> Result<(), String> {
    if phase_clamp_override.is_some() || random_push_seed.is_some() {
        let wrapped = EvalOscilloscopeSignal {
            inner: signal,
            layer_idx,
            phase_clamp_override,
            random_push_seed,
        };
        dispatch_rope_orchestration_inner(
            adapter,
            horizon_mode,
            escalation_provider,
            layer_idx,
            &wrapped,
            trail,
            slice,
        )
    } else {
        dispatch_rope_orchestration_inner(
            adapter,
            horizon_mode,
            escalation_provider,
            layer_idx,
            signal,
            trail,
            slice,
        )
    }
}

/// Computes `(inv_freq, attention_factor)` for standard or YaRN RoPE matching Hugging Face
/// `transformers` `_compute_yarn_parameters`.
pub(crate) fn compute_rope_inv_freq(
    rotary_dim: usize,
    rope_theta: f64,
    yarn: Option<YarnRopeConfig>,
) -> (Vec<f32>, f32) {
    let half = rotary_dim / 2;
    let base_inv: Vec<f64> = (0..half)
        .map(|i| 1.0_f64 / rope_theta.powf((2 * i) as f64 / rotary_dim as f64))
        .collect();

    let Some(y) = yarn else {
        return (base_inv.into_iter().map(|v| v as f32).collect(), 1.0_f32);
    };
    if !y.factor.is_finite() || y.factor <= 1.0 {
        return (base_inv.into_iter().map(|v| v as f32).collect(), 1.0_f32);
    }

    let dim_f = rotary_dim as f64;
    let orig_max = y.original_max_position_embeddings.max(1) as f64;
    let find_correction_dim = |num_rotations: f64| -> f64 {
        (dim_f * (orig_max / (num_rotations * 2.0 * std::f64::consts::PI)).ln())
            / (2.0 * rope_theta.ln())
    };
    let low = find_correction_dim(y.beta_fast).floor().max(0.0);
    let high = find_correction_dim(y.beta_slow)
        .ceil()
        .min((rotary_dim.saturating_sub(1)) as f64);
    let span = (high - low).max(1e-3);

    let inv_freq: Vec<f32> = base_inv
        .into_iter()
        .enumerate()
        .map(|(i, pos_freq)| {
            let inv_extrap = pos_freq;
            let inv_interp = pos_freq / y.factor;
            let ramp = (((i as f64) - low) / span).clamp(0.0, 1.0);
            let extrap_factor = 1.0 - ramp;
            (inv_interp * (1.0 - extrap_factor) + inv_extrap * extrap_factor) as f32
        })
        .collect();
    let attn_factor = y.attention_factor() as f32;
    (inv_freq, attn_factor)
}

/// Computes the Llama-4 / Ministral-3 long-context attention query scale factor at `pos`:
/// `1.0 + beta * ln(1.0 + floor(pos / original_max_position_embeddings))`.
#[inline]
pub(crate) fn llama_4_attn_scale_scalar(pos: usize, beta: f64, orig_max_pos: usize) -> f32 {
    if orig_max_pos == 0 || pos < orig_max_pos {
        return 1.0;
    }
    let ratio = (pos / orig_max_pos) as f64;
    (1.0 + beta * (1.0 + ratio).ln()) as f32
}

pub(crate) struct RotaryCache {
    layout: RopeLayout,
    cos: Tensor,
    sin: Tensor,
    inv_freq: Tensor,
    attention_factor: f32,
    llama_4_scaling: Option<(f64, usize)>,
    pub(crate) rotary_dim: usize,
    cached_len: usize,
    max_position_embeddings: usize,
    dtype: DType,
    device: Device,
}

impl RotaryCache {
    pub(crate) fn new(
        cfg: &ModelConfig,
        initial_seq: usize,
        dtype: DType,
        dev: &Device,
    ) -> CResult<Self> {
        let rot_dim = cfg.rotary_dim();
        let half = rot_dim / 2;
        let max_pos = cfg.max_position_embeddings.max(1);
        let init_len = initial_seq.clamp(1, max_pos);
        let (inv_freq_vec, attention_factor) =
            compute_rope_inv_freq(rot_dim, cfg.rope_theta, cfg.yarn);
        let llama_4_scaling = cfg.yarn.and_then(|y| {
            y.llama_4_scaling_beta
                .filter(|&b| b.is_finite() && b != 0.0)
                .map(|b| (b, y.original_max_position_embeddings.max(1)))
        });

        let inv_freq = Tensor::from_vec(inv_freq_vec, (1, half), dev)?;
        let pos: Vec<f32> = (0..init_len).map(|p| p as f32).collect();
        let pos = Tensor::from_vec(pos, (init_len, 1), dev)?;
        let freqs = pos.matmul(&inv_freq)?;
        let mut cos = freqs.cos()?;
        let mut sin = freqs.sin()?;
        if (attention_factor - 1.0).abs() > 1e-6 {
            cos = (cos * f64::from(attention_factor))?;
            sin = (sin * f64::from(attention_factor))?;
        }
        let cos = cos.to_dtype(dtype)?;
        let sin = sin.to_dtype(dtype)?;
        Ok(Self {
            layout: cfg.rope_layout,
            cos,
            sin,
            inv_freq,
            attention_factor,
            llama_4_scaling,
            rotary_dim: rot_dim,
            cached_len: init_len,
            max_position_embeddings: max_pos,
            dtype,
            device: dev.clone(),
        })
    }

    pub(crate) fn cached_len(&self) -> usize {
        self.cached_len
    }

    /// Dynamically extends the precomputed RoPE `cos`/`sin` tables up to `required_len` (growing
    /// in power-of-two increments capped at `max_position_embeddings`), preserving bit-identical
    /// prefixes and returning an explicit error without panicking if `max_position_embeddings` is
    /// exceeded.
    pub(crate) fn ensure_capacity(&mut self, required_len: usize) -> CResult<()> {
        if required_len <= self.cached_len {
            return Ok(());
        }
        if required_len > self.max_position_embeddings {
            return Err(candle_core::Error::Msg(format!(
                "sequence length {required_len} exceeds model max_position_embeddings {}",
                self.max_position_embeddings
            )));
        }
        let target_len = required_len
            .checked_next_power_of_two()
            .unwrap_or(self.max_position_embeddings)
            .max(self.cached_len.saturating_mul(2))
            .min(self.max_position_embeddings);
        let add_len = target_len.saturating_sub(self.cached_len);
        if add_len == 0 {
            return Ok(());
        }
        let pos: Vec<f32> = (self.cached_len..target_len).map(|p| p as f32).collect();
        let pos = Tensor::from_vec(pos, (add_len, 1), &self.device)?;
        let freqs = pos.matmul(&self.inv_freq)?;
        let mut tail_cos = freqs.cos()?;
        let mut tail_sin = freqs.sin()?;
        if (self.attention_factor - 1.0).abs() > 1e-6 {
            tail_cos = (tail_cos * f64::from(self.attention_factor))?;
            tail_sin = (tail_sin * f64::from(self.attention_factor))?;
        }
        let tail_cos = tail_cos.to_dtype(self.dtype)?;
        let tail_sin = tail_sin.to_dtype(self.dtype)?;
        self.cos = Tensor::cat(&[&self.cos, &tail_cos], 0)?;
        self.sin = Tensor::cat(&[&self.sin, &tail_sin], 0)?;
        self.cached_len = target_len;
        Ok(())
    }

    /// Applies Llama-4 / Ministral-3 query temperature scaling when `offset + seq_len` crosses
    /// `original_max_position_embeddings` (zero-overhead no-op below the threshold).
    pub(crate) fn apply_query_temp_scale(
        &self,
        q: Tensor,
        offset: usize,
        seq_len: usize,
    ) -> CResult<Tensor> {
        let Some((beta, orig_max_pos)) = self.llama_4_scaling else {
            return Ok(q);
        };
        if offset.saturating_add(seq_len) <= orig_max_pos {
            return Ok(q);
        }
        let scales: Vec<f32> = (offset..offset + seq_len)
            .map(|p| llama_4_attn_scale_scalar(p, beta, orig_max_pos))
            .collect();
        let scale_t =
            Tensor::from_vec(scales, (1, 1, seq_len, 1), q.device())?.to_dtype(q.dtype())?;
        q.broadcast_mul(&scale_t)
    }

    /// Applies layout-matched RoPE (`HalfSplit` or `Interleaved`) to `[b, num_heads, seq_len, rot]`.
    pub(crate) fn apply_rope_slice(
        &self,
        x_rot: &Tensor,
        offset: usize,
        seq_len: usize,
    ) -> CResult<Tensor> {
        let end = offset.checked_add(seq_len).ok_or_else(|| {
            candle_core::Error::Msg("RoPE position offset + seq_len overflow".to_string())
        })?;
        if end > self.cached_len {
            return Err(candle_core::Error::Msg(format!(
                "RoPE span {offset}..{end} exceeds cached_len {}",
                self.cached_len
            )));
        }
        match self.layout {
            RopeLayout::HalfSplit => {
                let half = self.rotary_dim / 2;
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
                Tensor::cat(&[&r1, &r2], D::Minus1)
            }
            RopeLayout::Interleaved => {
                let cos = self.cos.narrow(0, offset, seq_len)?.contiguous()?;
                let sin = self.sin.narrow(0, offset, seq_len)?.contiguous()?;
                candle_nn::rotary_emb::rope_i(&x_rot.contiguous()?, &cos, &sin)
            }
        }
    }

    /// Steers a rotary query slice `[1, num_heads, seq_len, rotary_dim]` via `ModelSteeringAdapter`.
    fn steer_rotary_slice(
        &self,
        rot_tensor: Tensor,
        num_heads: usize,
        seq_len: usize,
        out_dtype: DType,
        steer: LayerSteer<'_, '_, '_>,
    ) -> CResult<Tensor> {
        let rot = self.rotary_dim;
        let LayerSteer {
            adapter,
            layer_idx,
            q7_deltas,
            persona_opt,
            resonant_opt,
            max_rad,
            trail,
            horizon_mode,
            mut escalation_provider,
            probe_frame: _,
            phase_clamp_override,
            random_push_seed,
        } = steer;

        let mut flat = rot_tensor
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
                                phase_clamp_override,
                                random_push_seed,
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
                                phase_clamp_override,
                                random_push_seed,
                            )
                            .map_err(candle_core::Error::Msg)?;
                        }
                    } else if phase_clamp_override.is_some() || random_push_seed.is_some() {
                        let wrapped = EvalOscilloscopeSignal {
                            inner: &fact_res,
                            layer_idx,
                            phase_clamp_override,
                            random_push_seed,
                        };
                        adapter
                            .steer_layer_with_gateway(layer_idx, &wrapped, trail, 128, slice, None)
                            .map_err(candle_core::Error::Msg)?;
                    } else {
                        adapter
                            .steer_layer_with_gateway(layer_idx, &fact_res, trail, 128, slice, None)
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
                        if phase_clamp_override.is_some() || random_push_seed.is_some() {
                            let wrapped = EvalOscilloscopeSignal {
                                inner: &composite,
                                layer_idx,
                                phase_clamp_override,
                                random_push_seed,
                            };
                            adapter
                                .steer_layer_with_gateway(
                                    layer_idx, &wrapped, trail, 128, slice, None,
                                )
                                .map_err(candle_core::Error::Msg)?;
                        } else {
                            adapter
                                .steer_layer_with_gateway(
                                    layer_idx, &composite, trail, 128, slice, None,
                                )
                                .map_err(candle_core::Error::Msg)?;
                        }
                    }
                } else {
                    for tok in 0..seq_len {
                        let slice = &mut flat[tok * tok_stride..(tok + 1) * tok_stride];
                        if phase_clamp_override.is_some() || random_push_seed.is_some() {
                            let wrapped = EvalOscilloscopeSignal {
                                inner: &fact,
                                layer_idx,
                                phase_clamp_override,
                                random_push_seed,
                            };
                            adapter
                                .steer_layer_with_gateway(
                                    layer_idx, &wrapped, trail, 128, slice, None,
                                )
                                .map_err(candle_core::Error::Msg)?;
                        } else {
                            adapter
                                .steer_layer_with_gateway(layer_idx, &fact, trail, 128, slice, None)
                                .map_err(candle_core::Error::Msg)?;
                        }
                    }
                }
            }
        } else {
            let eff_max_rad = phase_clamp_override.unwrap_or(max_rad);
            let rand_deltas = random_push_seed.map(|seed| {
                let mut ref_arr = [0_i8; MAX_ATTENTION_HEADS];
                let n = q7_deltas.len().min(MAX_ATTENTION_HEADS);
                ref_arr[..n].copy_from_slice(&q7_deltas[..n]);
                generate_norm_matched_random_q7(&ref_arr, n, seed, layer_idx)
            });
            let active_deltas = match rand_deltas.as_ref() {
                Some(r) => &r[..q7_deltas.len().min(MAX_ATTENTION_HEADS)],
                None => q7_deltas,
            };
            for tok in 0..seq_len {
                let slice = &mut flat[tok * tok_stride..(tok + 1) * tok_stride];
                adapter
                    .steer_layer_q_rot(layer_idx, slice, active_deltas, eff_max_rad)
                    .map_err(candle_core::Error::Msg)?;
            }
        }
        Tensor::from_vec(flat, (1, seq_len, num_heads, rot), rot_tensor.device())?
            .transpose(1, 2)?
            .to_dtype(out_dtype)
    }

    /// Applies layout-matched RoPE on `0..rotary_dim` and delegates steering to
    /// `ModelSteeringAdapter`.
    pub(crate) fn apply_with_adapter(
        &self,
        x: &Tensor,
        offset: usize,
        steer: Option<LayerSteer<'_, '_, '_>>,
    ) -> CResult<Tensor> {
        let (b, num_heads, seq_len, head_dim) = x.dims4()?;
        let rot = self.rotary_dim;
        let x_rot = x.narrow(D::Minus1, 0, rot)?;

        let q_rot = match steer {
            Some(st) if st.max_rad != 0.0 && b == 1 => match st.probe_frame {
                ProbeCoordinateFrame::PreRope => {
                    let steered_pre =
                        self.steer_rotary_slice(x_rot, num_heads, seq_len, x.dtype(), st)?;
                    self.apply_rope_slice(&steered_pre, offset, seq_len)?
                }
                ProbeCoordinateFrame::PostRope => {
                    let roped = self.apply_rope_slice(&x_rot, offset, seq_len)?;
                    self.steer_rotary_slice(roped, num_heads, seq_len, x.dtype(), st)?
                }
            },
            _ => self.apply_rope_slice(&x_rot, offset, seq_len)?,
        };

        if rot < head_dim {
            let x_pass = x.narrow(D::Minus1, rot, head_dim - rot)?;
            Tensor::cat(&[&q_rot, &x_pass], D::Minus1)
        } else {
            Ok(q_rot)
        }
    }
}
