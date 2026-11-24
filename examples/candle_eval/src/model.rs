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
use liminika_core::abi::{lmnk_pointer_trail_t, lmnk_q7_t};
use liminika_core::foundation::TaggedPtr;
use liminika_core::liminal::MAX_ATTENTION_HEADS;
use liminika_core::runtime::{
    CausalProvenanceSignal, GatewayFrame, GuardPhaseAnchor, ManifoldGatewaySignal, ObjectiveFact,
    ResonantSignalSlice, SubjectivePersona,
};
use serde::Deserialize;

use crate::eval::HorizonAblationMode;
use crate::LayerResonantSlices;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum KvCacheMode {
    #[default]
    Append,
    IsolatedProbe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ProbeCoordinateFrame {
    #[default]
    PreRope,
    PostRope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LogitsFetchMode {
    FullLogits,
    DeviceArgmaxOnly,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum LogitsOutcome {
    ObservedFull(Vec<f32>),
    UnobservedDeviceArgmax(u32),
}

impl LogitsOutcome {
    pub(crate) fn into_full_logits(self) -> CResult<Vec<f32>> {
        match self {
            Self::ObservedFull(v) => Ok(v),
            Self::UnobservedDeviceArgmax(_) => Err(candle_core::Error::Msg(
                "expected FullLogits outcome".to_string(),
            )),
        }
    }
}

pub(crate) fn extract_device_argmax_token(logits_1d: &Tensor) -> CResult<u32> {
    if let Ok(tok_t) = logits_1d.argmax(0) {
        if let Ok(tok_id) = tok_t.to_scalar::<u32>() {
            return Ok(tok_id);
        }
    }
    logits_1d
        .to_dtype(DType::F32)?
        .argmax(0)?
        .to_scalar::<u32>()
}

pub(crate) type ForwardTokensOutput = (LogitsOutcome, Vec<Vec<f32>>, Vec<f32>);

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
struct EvalOscilloscopeSignal<'a, S: ManifoldGatewaySignal + ?Sized> {
    inner: &'a S,
    layer_idx: usize,
    phase_clamp_override: Option<f32>,
    random_push_seed: Option<u64>,
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
    trail: &mut liminika_core::abi::lmnk_pointer_trail_t,
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
        adapter.orchestrate_layer_rope_with_substrate(
            layer_idx, signal, trail, None, 128, slice, None,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dispatch_rope_orchestration<S: ManifoldGatewaySignal + ?Sized>(
    adapter: &mut ModelSteeringAdapter,
    horizon_mode: HorizonAblationMode,
    escalation_provider: Option<&mut CorpusRecallEscalationProvider<'_>>,
    layer_idx: usize,
    signal: &S,
    trail: &mut liminika_core::abi::lmnk_pointer_trail_t,
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

    /// Applies HalfSplit RoPE (`cos`/`sin`) to the rotary slice `[b, num_heads, seq_len, rot]`.
    fn apply_half_split_rope(
        &self,
        x_rot: &Tensor,
        offset: usize,
        seq_len: usize,
    ) -> CResult<Tensor> {
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

    /// Applies HalfSplit RoPE on `0..rotary_dim` and delegates steering to `ModelSteeringAdapter`.
    ///
    /// In `ProbeCoordinateFrame::PreRope` (default), probing (`bake_layer_procrustes_q7`) and
    /// SO(2) phase steering run on `x_rot` before RoPE `cos/sin` so probes share the exact
    /// pre-RoPE coordinate frame as baked anchors with zero extra D2H/H2D transfers.
    fn apply_with_adapter(
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
                    self.apply_half_split_rope(&steered_pre, offset, seq_len)?
                }
                ProbeCoordinateFrame::PostRope => {
                    let roped = self.apply_half_split_rope(&x_rot, offset, seq_len)?;
                    self.steer_rotary_slice(roped, num_heads, seq_len, x.dtype(), st)?
                }
            },
            _ => self.apply_half_split_rope(&x_rot, offset, seq_len)?,
        };

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

    fn forward(
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_tokens(
        &mut self,
        tokens: &[u32],
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

        let h = self.norm.forward(&h)?;
        let mut last = h.i((0, tokens.len() - 1, ..))?.unsqueeze(0)?;
        let last_hidden = if collect_bake_data {
            last.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?
        } else {
            Vec::new()
        };

        if !whole_bypass {
            if let Some(st) = steer.as_deref_mut() {
                if let Some((u_vec, v_vec, rank1_scale)) = st.rank1 {
                    if rank1_scale != 0.0 {
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
                            if let Some((fact_anchors, persona_anchors, guard_anchors, _, _)) =
                                dyn_rank1
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
                        last = Tensor::from_vec(h_vec, (1, len), &self.device)?
                            .to_dtype(last.dtype())?;
                    }
                }
            }
        }

        let logits_1d = self.lm_head.forward(&last)?.squeeze(0)?;
        let outcome = match fetch_mode {
            LogitsFetchMode::DeviceArgmaxOnly if !use_entropy_gate || whole_bypass => {
                let tok_id = extract_device_argmax_token(&logits_1d)?;
                LogitsOutcome::UnobservedDeviceArgmax(tok_id)
            }
            _ => {
                let mut logits = logits_1d.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                if use_entropy_gate && !whole_bypass {
                    if let Some(st) = steer {
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
                }
                LogitsOutcome::ObservedFull(logits)
            }
        };
        Ok((outcome, per_layer_q, last_hidden))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> Phi3Config {
        Phi3Config {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            vocab_size: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            partial_rotary_factor: 1.0,
            tie_word_embeddings: true,
        }
    }

    fn deterministic_attention(cfg: &Phi3Config) -> Phi3Attention {
        let h = cfg.hidden_size;
        let qkv_out = 3 * h;
        let qkv_w: Vec<f32> = (0..qkv_out * h)
            .map(|i| (((i * 37 + 13) % 29) as f32 - 14.0) * 0.04)
            .collect();
        let o_w: Vec<f32> = (0..h * h)
            .map(|i| (((i * 23 + 7) % 31) as f32 - 15.0) * 0.04)
            .collect();
        let qkv_t = Tensor::from_vec(qkv_w, (qkv_out, h), &Device::Cpu).expect("qkv_w");
        let o_t = Tensor::from_vec(o_w, (h, h), &Device::Cpu).expect("o_w");
        Phi3Attention {
            qkv_proj: LinearLayer::Dense(Linear::new(qkv_t, None)),
            o_proj: LinearLayer::Dense(Linear::new(o_t, None)),
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim(),
            kv_cache: None,
        }
    }

    fn tensor_flat_f32(t: &Tensor) -> Vec<f32> {
        t.contiguous()
            .expect("contiguous")
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to_vec1")
    }

    #[test]
    fn isolated_probe_preserves_kv_cache_and_truncate_rewinds_exact_prefix() {
        let cfg = tiny_config();
        let rope = RotaryCache::new(&cfg, 64, DType::F32, &Device::Cpu).expect("rope");
        let mut attn = deterministic_attention(&cfg);
        let mut ref_attn = deterministic_attention(&cfg);

        let x_init: Vec<f32> = (0..3 * cfg.hidden_size)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
            .collect();
        let x_init_t = Tensor::from_vec(x_init, (1, 3, cfg.hidden_size), &Device::Cpu).expect("x0");

        let _ = attn
            .forward(&x_init_t, &rope, 0, None, false, KvCacheMode::Append)
            .expect("append init");
        let _ = ref_attn
            .forward(&x_init_t, &rope, 0, None, false, KvCacheMode::Append)
            .expect("ref append init");
        assert_eq!(attn.kv_len(), 3);

        let (k0, v0) = attn.kv_cache.as_ref().expect("kv populated");
        let k0_vec = tensor_flat_f32(k0);
        let v0_vec = tensor_flat_f32(v0);

        let x_probe: Vec<f32> = (0..2 * cfg.hidden_size)
            .map(|i| ((i * 5 + 3) % 19) as f32 * 0.08 - 0.6)
            .collect();
        let x_probe_t =
            Tensor::from_vec(x_probe, (1, 2, cfg.hidden_size), &Device::Cpu).expect("x_probe");
        let (_, q_probe) = attn
            .forward(&x_probe_t, &rope, 0, None, true, KvCacheMode::IsolatedProbe)
            .expect("isolated probe offset 0");
        assert_eq!(
            q_probe.expect("q_heads captured").len(),
            2 * cfg.num_attention_heads * cfg.rotary_dim()
        );
        // Non-zero offset probe must also respect local causal mask shape without touching KV.
        let _ = attn
            .forward(
                &x_probe_t,
                &rope,
                5,
                None,
                false,
                KvCacheMode::IsolatedProbe,
            )
            .expect("isolated probe offset 5");

        assert_eq!(attn.kv_len(), 3);
        let (k_after_probe, v_after_probe) = attn.kv_cache.as_ref().expect("kv intact");
        assert_eq!(tensor_flat_f32(k_after_probe), k0_vec);
        assert_eq!(tensor_flat_f32(v_after_probe), v0_vec);

        let x_step: Vec<f32> = (0..cfg.hidden_size)
            .map(|i| (i as f32 - 7.5) * 0.12)
            .collect();
        let x_step_t =
            Tensor::from_vec(x_step, (1, 1, cfg.hidden_size), &Device::Cpu).expect("x_step");
        let (out_with_probe, _) = attn
            .forward(&x_step_t, &rope, 3, None, false, KvCacheMode::Append)
            .expect("append step");
        let (out_ref, _) = ref_attn
            .forward(&x_step_t, &rope, 3, None, false, KvCacheMode::Append)
            .expect("ref append step");
        assert_eq!(attn.kv_len(), 4);
        assert_eq!(tensor_flat_f32(&out_with_probe), tensor_flat_f32(&out_ref));

        // Truncating back to 3 restores the exact initial KV prefix; over-truncation fails closed.
        attn.truncate_cache(3).expect("truncate to 3");
        assert_eq!(attn.kv_len(), 3);
        let (k_trunc, v_trunc) = attn.kv_cache.as_ref().expect("kv truncated");
        assert_eq!(tensor_flat_f32(k_trunc), k0_vec);
        assert_eq!(tensor_flat_f32(v_trunc), v0_vec);
        assert!(attn.truncate_cache(5).is_err());

        attn.truncate_cache(0).expect("truncate to 0");
        assert_eq!(attn.kv_len(), 0);
        assert!(attn.kv_cache.is_none());
    }

    #[test]
    fn pre_rope_commutes_with_fixed_angle_and_keeps_resonant_probe_position_invariant() {
        let cfg = tiny_config();
        let rope = RotaryCache::new(&cfg, 64, DType::F32, &Device::Cpu).expect("rope");
        let mut adapter = ModelSteeringAdapter::new(cfg.to_rope_geometry()).expect("adapter");

        let q_pre_vec: Vec<f32> = (0..cfg.num_attention_heads * cfg.head_dim())
            .map(|i| ((i * 11 + 5) % 13) as f32 * 0.15 - 0.8)
            .collect();
        let q_pre = Tensor::from_vec(
            q_pre_vec.clone(),
            (1, cfg.num_attention_heads, 1, cfg.head_dim()),
            &Device::Cpu,
        )
        .expect("q_pre");

        // 1. Fixed-angle SO(2) commutativity: steering before vs after HalfSplit RoPE agrees.
        let fixed_q7: [lmnk_q7_t; 2] = [64, -48];
        let mut trail_a = lmnk_pointer_trail_t::default();
        let mut trail_b = lmnk_pointer_trail_t::default();
        let out_pre = rope
            .apply_with_adapter(
                &q_pre,
                19,
                Some(LayerSteer {
                    adapter: &mut adapter,
                    layer_idx: 1,
                    q7_deltas: &fixed_q7,
                    persona_opt: None,
                    resonant_opt: None,
                    max_rad: 0.20,
                    trail: &mut trail_a,
                    horizon_mode: HorizonAblationMode::Core3,
                    escalation_provider: None,
                    probe_frame: ProbeCoordinateFrame::PreRope,
                    phase_clamp_override: None,
                    random_push_seed: None,
                }),
            )
            .expect("pre-rope fixed steer");
        let out_post = rope
            .apply_with_adapter(
                &q_pre,
                19,
                Some(LayerSteer {
                    adapter: &mut adapter,
                    layer_idx: 1,
                    q7_deltas: &fixed_q7,
                    persona_opt: None,
                    resonant_opt: None,
                    max_rad: 0.20,
                    trail: &mut trail_b,
                    horizon_mode: HorizonAblationMode::Core3,
                    escalation_provider: None,
                    probe_frame: ProbeCoordinateFrame::PostRope,
                    phase_clamp_override: None,
                    random_push_seed: None,
                }),
            )
            .expect("post-rope fixed steer");
        for (a, b) in tensor_flat_f32(&out_pre)
            .iter()
            .zip(tensor_flat_f32(&out_post).iter())
        {
            assert!(
                (a - b).abs() < 1e-5,
                "HalfSplit SO(2) commutativity mismatch: pre={a} vs post={b}"
            );
        }

        // 2. Resonant Procrustes probe invariance in `PreRope` vs position drift in `PostRope`.
        let neg_ref: Vec<f32> = q_pre_vec
            .iter()
            .enumerate()
            .map(|(i, &v)| if i % 2 == 0 { -v } else { v * 0.5 + 0.3 })
            .collect();
        let anchor_q7_vec = adapter
            .bake_layer_procrustes_q7(&neg_ref, &q_pre_vec)
            .expect("bake pre-rope anchor");
        let mut arr64 = [0_i8; 64];
        arr64[..anchor_q7_vec.len()].copy_from_slice(&anchor_q7_vec);
        let facts = [ObjectiveFact::new(arr64, 2.5, 0, 1)];
        let personas: [SubjectivePersona<[i8; 64]>; 0] = [];
        let guards: [GuardPhaseAnchor; 0] = [];
        let res_slice: LayerResonantSlices<'_> = (&facts, &personas, &guards, &neg_ref, &q_pre_vec);

        let mut trail_off0 = lmnk_pointer_trail_t::default();
        let steered_at_0 = rope
            .apply_with_adapter(
                &q_pre,
                0,
                Some(LayerSteer {
                    adapter: &mut adapter,
                    layer_idx: 1,
                    q7_deltas: &fixed_q7,
                    persona_opt: None,
                    resonant_opt: Some(res_slice),
                    max_rad: -1.0,
                    trail: &mut trail_off0,
                    horizon_mode: HorizonAblationMode::Core3,
                    escalation_provider: None,
                    probe_frame: ProbeCoordinateFrame::PreRope,
                    phase_clamp_override: None,
                    random_push_seed: None,
                }),
            )
            .expect("pre-rope resonant off=0");
        // Rotating the `offset=0` steered vector by RoPE(25) must match `PreRope` at `offset=25`.
        let expected_at_25 = rope
            .apply_half_split_rope(&steered_at_0, 25, 1)
            .expect("rope shift");

        let mut trail_off25 = lmnk_pointer_trail_t::default();
        let pre_at_25 = rope
            .apply_with_adapter(
                &q_pre,
                25,
                Some(LayerSteer {
                    adapter: &mut adapter,
                    layer_idx: 1,
                    q7_deltas: &fixed_q7,
                    persona_opt: None,
                    resonant_opt: Some(res_slice),
                    max_rad: -1.0,
                    trail: &mut trail_off25,
                    horizon_mode: HorizonAblationMode::Core3,
                    escalation_provider: None,
                    probe_frame: ProbeCoordinateFrame::PreRope,
                    phase_clamp_override: None,
                    random_push_seed: None,
                }),
            )
            .expect("pre-rope resonant off=25");
        for (a, b) in tensor_flat_f32(&expected_at_25)
            .iter()
            .zip(tensor_flat_f32(&pre_at_25).iter())
        {
            assert!(
                (a - b).abs() < 1e-5,
                "PreRope resonant probe must be position-invariant: expected={a} got={b}"
            );
        }

        // Verify `PostRope` mixes pre-RoPE `neg_ref` with RoPE-rotated `q` at `offset=25`.
        let post_roped_25 = tensor_flat_f32(
            &rope
                .apply_half_split_rope(&q_pre, 25, 1)
                .expect("post roped 25"),
        );
        let probe_pre = adapter
            .bake_layer_procrustes_q7(&neg_ref, &q_pre_vec)
            .expect("probe pre");
        let probe_post = adapter
            .bake_layer_procrustes_q7(&neg_ref, &post_roped_25)
            .expect("probe post");
        assert_ne!(
            probe_pre, probe_post,
            "PostRope probe at offset=25 should diverge from pre-RoPE anchor frame"
        );
    }

    #[test]
    fn random_push_control_is_deterministic_norm_matched_and_zero_preserving() {
        let zeros = [0_i8; MAX_ATTENTION_HEADS];
        assert_eq!(generate_norm_matched_random_q7(&zeros, 24, 42, 16), zeros);

        let mut ref_q7 = [0_i8; MAX_ATTENTION_HEADS];
        for (i, slot) in ref_q7.iter_mut().take(24).enumerate() {
            *slot = if i % 2 == 0 { 40 } else { -30 };
        }
        let r1 = generate_norm_matched_random_q7(&ref_q7, 24, 42, 16);
        let r2 = generate_norm_matched_random_q7(&ref_q7, 24, 42, 16);
        let r_other_layer = generate_norm_matched_random_q7(&ref_q7, 24, 42, 17);
        let r_other_seed = generate_norm_matched_random_q7(&ref_q7, 24, 99, 16);

        assert_eq!(r1, r2);
        assert_ne!(r1, r_other_layer);
        assert_ne!(r1, r_other_seed);
        assert!(r1[24..].iter().all(|&v| v == 0));

        let ref_norm: f64 = ref_q7[..24]
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            .sqrt();
        let rand_norm: f64 = r1[..24]
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            .sqrt();
        assert!(
            (ref_norm - rand_norm).abs() / ref_norm < 0.05,
            "norm mismatch: ref={ref_norm:.2} rand={rand_norm:.2}"
        );

        let fact = ObjectiveFact::new(ref_q7, 2.0, 0, 7);
        let wrapped = EvalOscilloscopeSignal {
            inner: &fact,
            layer_idx: 16,
            phase_clamp_override: Some(0.24),
            random_push_seed: Some(42),
        };
        let frame = wrapped.synthesize_gateway_frame();
        assert!((frame.phase_clamp_rad - 0.24).abs() < 1e-6);
        assert_ne!(frame.phase_deltas, ref_q7);
    }
}
