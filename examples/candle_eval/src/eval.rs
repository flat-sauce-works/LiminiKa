// name: examples/candle_eval/src/eval.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CLI profile parsers, KV-cache statistics, steered decoding, and reflection critique.

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::quantized::GgmlDType;
use candle_core::Tensor;
use liminika_adapter::{
    AttractorTrajectoryEvaluator, CompiledTokenCorpus, CorpusRecallEscalationProvider,
    LmnkTimeline, ManifoldReflectionEvaluator, ModelSteeringAdapter, ReflectionCritique,
    ReflectionPolicy, SpineDecodeInput, SpineLogitsObserveMode, SystemPressureReading,
    TrajectoryManifoldSnapshot, SPINE_BYPASS_VETO_GUARD_ACTIVE, SPINE_BYPASS_VETO_PINNED_LOCK,
    SPINE_BYPASS_VETO_REPULSION_ACTIVE, SPINE_SENTINEL_AUTO_U8,
};
use liminika_core::abi::{lmnk_pointer_trail_t, lmnk_q7_t};
use liminika_core::drivers::LiminalEgoPreset;
use liminika_core::foundation::ACTION_MODE_PHASE_REPULSE;
use liminika_core::runtime::{
    CoreTelemetryBank, CoreTelemetrySnapshot, LayerDistortionReport, RamPhaseKvStepOutcome,
};
use liminika_core::stigmergy::{
    CognitiveOpKind, CognitiveStepOp, CognitiveTrajectoryLedger, StigmergicScaleGranularity,
    TRAIL_FLAG_REPULSION_ACTIVE,
};
use tokenizers::Tokenizer;

use crate::bake::argmax;
use crate::eval_common::count_semantic_forbid_hits_in_clauses;
use crate::model::{
    active_layers_mask, clip_layer_mask, top_half_layer_mask, CausalLmModel, ChatTemplate,
    ChatTemplateArg, ForwardSteer, KvCacheMode, LogitsFetchMode, LogitsOutcome, ModelArchArg,
    ProbeCoordinateFrame, ResonantAnchors,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerMaskSpec {
    TopHalf,
    All,
    Fixed(u64),
}

impl LayerMaskSpec {
    pub(crate) const fn resolve(self, num_layers: usize) -> u64 {
        match self {
            Self::TopHalf => top_half_layer_mask(num_layers),
            Self::All => active_layers_mask(num_layers),
            Self::Fixed(mask) => clip_layer_mask(mask, num_layers),
        }
    }
}

pub(crate) fn parse_model_path<'a>(args: impl IntoIterator<Item = &'a str>) -> Option<PathBuf> {
    for arg in args {
        if let Some(val) = arg.strip_prefix("--model=") {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                return Some(PathBuf::from(trimmed));
            }
        }
    }
    None
}

pub(crate) fn parse_tokenizer_path<'a>(args: impl IntoIterator<Item = &'a str>) -> Option<PathBuf> {
    for arg in args {
        if let Some(val) = arg.strip_prefix("--tokenizer=") {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                return Some(PathBuf::from(trimmed));
            }
        }
    }
    None
}

fn parse_model_family_arg<'a>(
    args: impl IntoIterator<Item = &'a str>,
    flag_prefix: &str,
) -> Option<&'static str> {
    for arg in args {
        if let Some(val) = arg.strip_prefix(flag_prefix) {
            let trimmed = val.trim();
            if trimmed.eq_ignore_ascii_case("phi4") {
                return Some("phi4");
            }
            if trimmed.eq_ignore_ascii_case("mistral3")
                || trimmed.eq_ignore_ascii_case("ministral3")
            {
                return Some("mistral3");
            }
            if trimmed.eq_ignore_ascii_case("auto") {
                return Some("auto");
            }
        }
    }
    None
}

pub(crate) fn parse_model_arch<'a>(args: impl IntoIterator<Item = &'a str>) -> ModelArchArg {
    match parse_model_family_arg(args, "--model-arch=") {
        Some("phi4") => ModelArchArg::Phi4,
        Some("mistral3") => ModelArchArg::Mistral3,
        _ => ModelArchArg::Auto,
    }
}

pub(crate) fn parse_chat_template<'a>(args: impl IntoIterator<Item = &'a str>) -> ChatTemplateArg {
    match parse_model_family_arg(args, "--chat-template=") {
        Some("phi4") => ChatTemplateArg::Phi4,
        Some("mistral3") => ChatTemplateArg::Mistral3,
        _ => ChatTemplateArg::Auto,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum PressureSimMode {
    #[default]
    Off,
    Elevated,
    Surge,
}

pub(crate) fn parse_pressure_sim_mode<'a>(
    args: impl IntoIterator<Item = &'a str>,
) -> PressureSimMode {
    for arg in args {
        if let Some(val) = arg.strip_prefix("--pressure-sim=") {
            if val.eq_ignore_ascii_case("elevated") {
                return PressureSimMode::Elevated;
            }
            if val.eq_ignore_ascii_case("surge") {
                return PressureSimMode::Surge;
            }
        }
    }
    PressureSimMode::Off
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CircuitProfileMode {
    None,
    Analytical,
    Integrative,
    Hybrid(u8),
}

pub(crate) fn parse_circuit_mode<'a>(
    args: impl IntoIterator<Item = &'a str>,
) -> CircuitProfileMode {
    for arg in args {
        if let Some(val) = arg.strip_prefix("--circuit=") {
            if val.eq_ignore_ascii_case("analytical") {
                return CircuitProfileMode::Analytical;
            }
            if val.eq_ignore_ascii_case("integrative") {
                return CircuitProfileMode::Integrative;
            }
            if val.eq_ignore_ascii_case("hybrid") {
                return CircuitProfileMode::Hybrid(64);
            }
            if let Some(blend_str) = val.strip_prefix("hybrid:") {
                if let Ok(blend) = blend_str.parse::<u8>() {
                    return CircuitProfileMode::Hybrid(blend.min(128));
                }
            }
        }
    }
    CircuitProfileMode::None
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EgoProfileMode {
    Custom,
    Preset(LiminalEgoPreset),
    NaturalLanguage(String),
}

pub(crate) fn parse_ego_mode<'a>(args: impl IntoIterator<Item = &'a str>) -> EgoProfileMode {
    for arg in args {
        if let Some(val) = arg.strip_prefix("--ego=") {
            if val.eq_ignore_ascii_case("custom") || val.eq_ignore_ascii_case("off") {
                return EgoProfileMode::Custom;
            }
            if val.eq_ignore_ascii_case("low") {
                return EgoProfileMode::Preset(LiminalEgoPreset::Low);
            }
            if val.eq_ignore_ascii_case("medium") {
                return EgoProfileMode::Preset(LiminalEgoPreset::Medium);
            }
            if val.eq_ignore_ascii_case("high") {
                return EgoProfileMode::Preset(LiminalEgoPreset::High);
            }
            if let Some(desc) = val.strip_prefix("desc:") {
                return EgoProfileMode::NaturalLanguage(desc.to_string());
            }
            return EgoProfileMode::NaturalLanguage(val.to_string());
        }
    }
    EgoProfileMode::Custom
}

pub(crate) fn apply_ego_mode(adapter: &mut ModelSteeringAdapter, mode: &EgoProfileMode) {
    match mode {
        EgoProfileMode::Custom => adapter.configure_ego_from_preset(LiminalEgoPreset::Custom),
        EgoProfileMode::Preset(p) => adapter.configure_ego_from_preset(*p),
        EgoProfileMode::NaturalLanguage(desc) => adapter.configure_ego_from_natural_language(desc),
    }
}

pub(crate) fn parse_kv_min_res<'a>(args: impl IntoIterator<Item = &'a str>) -> Option<lmnk_q7_t> {
    for arg in args {
        if let Some(val) = arg.strip_prefix("--kv-min-res=") {
            if let Ok(parsed) = val.parse::<i8>() {
                return Some(parsed.clamp(-127, 127));
            }
        }
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ReflectMode {
    Off,
    Conditional,
    #[default]
    Always,
}

impl ReflectMode {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Conditional => "cond",
            Self::Always => "on",
        }
    }
}

pub(crate) fn parse_reflect_mode<'a>(args: impl IntoIterator<Item = &'a str>) -> ReflectMode {
    let mut no_reflect = false;
    for arg in args {
        if arg == "--no-reflect" {
            no_reflect = true;
        } else if let Some(val) = arg.strip_prefix("--reflect-mode=") {
            if val.eq_ignore_ascii_case("off") {
                return ReflectMode::Off;
            }
            if val.eq_ignore_ascii_case("conditional") || val.eq_ignore_ascii_case("cond") {
                return ReflectMode::Conditional;
            }
            if val.eq_ignore_ascii_case("always") || val.eq_ignore_ascii_case("on") {
                return ReflectMode::Always;
            }
        }
    }
    if no_reflect {
        ReflectMode::Off
    } else {
        ReflectMode::Always
    }
}

pub(crate) fn parse_ego_max_hops<'a>(args: impl IntoIterator<Item = &'a str>) -> u8 {
    for arg in args {
        if let Some(val) = arg.strip_prefix("--ego-max-hops=") {
            if let Ok(parsed) = val.parse::<u8>() {
                return parsed.clamp(1, 4);
            }
        }
    }
    1
}

pub(crate) fn parse_temperature<'a>(args: impl IntoIterator<Item = &'a str>) -> f32 {
    for arg in args {
        if let Some(val) = arg.strip_prefix("--temp=") {
            if let Ok(parsed) = val.parse::<f32>() {
                if parsed.is_finite() && parsed >= 0.0 {
                    return parsed;
                }
            }
        }
    }
    0.0
}

pub(crate) fn parse_rng_seed<'a>(args: impl IntoIterator<Item = &'a str>) -> u64 {
    for arg in args {
        if let Some(val) = arg.strip_prefix("--seed=") {
            if let Ok(parsed) = val.parse::<u64>() {
                return parsed.max(1);
            }
        }
    }
    42
}

pub(crate) const DEFAULT_LOGITS_OBSERVE_STRIDE: usize = 1;
pub(crate) const DEFAULT_GUARD_HOLD_TOKENS: u8 = ModelSteeringAdapter::DEFAULT_GUARD_HOLD_TOKENS;
pub(crate) const DEFAULT_LAYER_STEER_MASK: u64 = 0xFFFF_0000;
pub(crate) const DEFAULT_MAX_GEN_TOKENS: usize = 64;
pub(crate) const DEFAULT_PREFILL_CHUNK_SIZE: usize = crate::model::DEFAULT_PREFILL_CHUNK_SIZE;

pub(crate) fn parse_prefill_chunk_size<'a>(args: impl IntoIterator<Item = &'a str>) -> usize {
    for arg in args {
        if let Some(raw) = arg.strip_prefix("--prefill-chunk-size=") {
            if let Ok(parsed) = raw.trim().parse::<usize>() {
                return parsed.clamp(16, 4096);
            }
        }
    }
    DEFAULT_PREFILL_CHUNK_SIZE
}

pub(crate) fn parse_long_doc_bench_path<'a>(
    args: impl IntoIterator<Item = &'a str>,
) -> Option<PathBuf> {
    for arg in args {
        if let Some(val) = arg.strip_prefix("--long-doc-bench=") {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                return Some(PathBuf::from(trimmed));
            }
        }
    }
    None
}

pub(crate) fn parse_layer_mask_spec<'a>(
    args: impl IntoIterator<Item = &'a str>,
) -> Option<LayerMaskSpec> {
    for arg in args {
        if let Some(raw) = arg.strip_prefix("--layer-mask=") {
            let val = raw.trim();
            if val.eq_ignore_ascii_case("top-half") || val.eq_ignore_ascii_case("top_half") {
                return Some(LayerMaskSpec::TopHalf);
            }
            if val.eq_ignore_ascii_case("top16") {
                return Some(LayerMaskSpec::Fixed(0xFFFF_0000));
            }
            if val.eq_ignore_ascii_case("mid12") {
                return Some(LayerMaskSpec::Fixed(0x003F_FC00));
            }
            if val.eq_ignore_ascii_case("mid16") {
                return Some(LayerMaskSpec::Fixed(0x03FF_FC00));
            }
            if val.eq_ignore_ascii_case("upper-mid12") || val.eq_ignore_ascii_case("upper_mid12") {
                return Some(LayerMaskSpec::Fixed(0x0FFF_0000));
            }
            if val.eq_ignore_ascii_case("all") {
                return Some(LayerMaskSpec::All);
            }
            if let Some(hex_raw) = val.strip_prefix("0x").or_else(|| val.strip_prefix("0X")) {
                let cleaned = hex_raw.replace('_', "");
                if let Ok(parsed) = u64::from_str_radix(&cleaned, 16) {
                    if parsed != 0 {
                        return Some(LayerMaskSpec::Fixed(parsed));
                    }
                }
            }
        }
    }
    None
}

pub(crate) fn parse_phase_clamp_rad<'a>(args: impl IntoIterator<Item = &'a str>) -> Option<f32> {
    for arg in args {
        if let Some(raw) = arg.strip_prefix("--phase-clamp-rad=") {
            if let Ok(parsed) = raw.trim().parse::<f32>() {
                if parsed.is_finite() && parsed > 0.0 && parsed <= std::f32::consts::FRAC_PI_2 {
                    return Some(parsed);
                }
            }
        }
    }
    None
}

pub(crate) fn parse_max_tokens<'a>(args: impl IntoIterator<Item = &'a str>) -> usize {
    for arg in args {
        if let Some(raw) = arg.strip_prefix("--max-tokens=") {
            if let Ok(parsed) = raw.trim().parse::<usize>() {
                return parsed.clamp(1, 512);
            }
        }
    }
    DEFAULT_MAX_GEN_TOKENS
}

pub(crate) fn parse_logits_observe_stride<'a>(args: impl IntoIterator<Item = &'a str>) -> usize {
    for arg in args {
        if let Some(val) = arg.strip_prefix("--logits-observe-stride=") {
            if let Ok(parsed) = val.parse::<usize>() {
                return parsed.max(1);
            }
        }
    }
    DEFAULT_LOGITS_OBSERVE_STRIDE
}

pub(crate) fn parse_guard_hold_tokens<'a>(args: impl IntoIterator<Item = &'a str>) -> u8 {
    for arg in args {
        if let Some(val) = arg.strip_prefix("--guard-hold-tokens=") {
            if let Ok(parsed) = val.parse::<u8>() {
                return parsed;
            }
        }
    }
    DEFAULT_GUARD_HOLD_TOKENS
}

pub(crate) fn parse_bench_trials<'a>(args: impl IntoIterator<Item = &'a str>) -> usize {
    for arg in args {
        if let Some(val) = arg.strip_prefix("--bench-trials=") {
            if let Ok(parsed) = val.parse::<usize>() {
                return parsed.max(1);
            }
        }
    }
    1
}

pub(crate) fn parse_fixture_path<'a>(args: impl IntoIterator<Item = &'a str>) -> Option<PathBuf> {
    for arg in args {
        if let Some(val) = arg.strip_prefix("--fixture=") {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                return Some(PathBuf::from(trimmed));
            }
        }
    }
    None
}

pub(crate) fn parse_probe_frame<'a>(
    args: impl IntoIterator<Item = &'a str>,
) -> ProbeCoordinateFrame {
    for arg in args {
        if let Some(val) = arg.strip_prefix("--probe-frame=") {
            if val.eq_ignore_ascii_case("post-rope")
                || val.eq_ignore_ascii_case("post_rope")
                || val.eq_ignore_ascii_case("postrope")
            {
                return ProbeCoordinateFrame::PostRope;
            }
            if val.eq_ignore_ascii_case("pre-rope")
                || val.eq_ignore_ascii_case("pre_rope")
                || val.eq_ignore_ascii_case("prerope")
            {
                return ProbeCoordinateFrame::PreRope;
            }
        }
    }
    ProbeCoordinateFrame::PreRope
}

/// Computes normalized Top-K Shannon entropy in `[0.0, 1.0]` directly on an already-fetched CPU
/// logits vector without issuing extra GPU->CPU D2H transfers.
pub(crate) fn compute_top_k_logit_entropy(logits: &[f32], top_k: usize) -> f32 {
    let k = top_k.min(logits.len());
    if k < 2 {
        return 0.0;
    }
    let mut top_vals: Vec<f32> = Vec::with_capacity(k);
    for &v in logits {
        if !v.is_finite() {
            continue;
        }
        if top_vals.len() < k {
            top_vals.push(v);
            if top_vals.len() == k {
                top_vals.sort_by(|a, b| a.total_cmp(b));
            }
        } else if v > top_vals[0] {
            top_vals[0] = v;
            for i in 0..(k - 1) {
                if top_vals[i] > top_vals[i + 1] {
                    top_vals.swap(i, i + 1);
                } else {
                    break;
                }
            }
        }
    }
    if top_vals.len() < 2 {
        return 0.0;
    }
    let max_v = *top_vals.last().unwrap_or(&0.0);
    let mut sum_exp = 0.0_f64;
    let mut exps = Vec::with_capacity(top_vals.len());
    for &v in &top_vals {
        let e = f64::from(v - max_v).exp();
        sum_exp += e;
        exps.push(e);
    }
    if sum_exp <= 1e-12 {
        return 0.0;
    }
    let mut h = 0.0_f64;
    for e in exps {
        let p = e / sum_exp;
        if p > 1e-12 {
            h -= p * p.ln();
        }
    }
    let max_h = (top_vals.len() as f64).ln().max(1e-6);
    (h / max_h).clamp(0.0, 1.0) as f32
}

/// Counts `(raw_hits, semantic_hits)` across `forbidden` keywords, suppressing false positives
/// when the forbidden term appears strictly inside an explicit negation/refutation clause.
pub(crate) fn count_semantic_forbid_hits(reply_text: &str, forbidden: &[String]) -> (usize, usize) {
    let lower = reply_text.to_ascii_lowercase();
    let normalized = lower
        .replace(", instead ", " ; ")
        .replace("; instead ", " ; ")
        .replace(" instead, ", " ; ")
        .replace(", but ", " ; ")
        .replace(" but ", " ; ")
        .replace(", however ", " ; ")
        .replace(" however ", " ; ")
        .replace(", whereas ", " ; ")
        .replace(" whereas ", " ; ");
    let clauses: Vec<&str> = normalized
        .split(['.', ';', '!', '?', '\n'])
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .collect();

    const PREFIX_NEG_CUES: &[&str] = &[
        "does not ",
        "do not ",
        "did not ",
        "is not ",
        "are not ",
        "cannot ",
        "can't",
        "doesn't",
        "don't",
        "isn't",
        "aren't",
        "doesnt ",
        "dont ",
        "never ",
        "not ",
        "no ",
        "without ",
        "neither ",
        "nor ",
        "avoid",
        "replaces ",
        "replacing ",
        "reject",
        "refute",
        "false",
        "incorrect",
        "myth",
        "contrary to",
        "instead of ",
        "rather than ",
    ];
    const SUFFIX_NEG_CUES: &[&str] = &[
        " is false",
        " are false",
        " is incorrect",
        " is not used",
        " are not used",
    ];

    let forbidden_lower: Vec<String> = forbidden.iter().map(|k| k.to_ascii_lowercase()).collect();
    count_semantic_forbid_hits_in_clauses(
        &clauses,
        &forbidden_lower,
        PREFIX_NEG_CUES,
        SUFFIX_NEG_CUES,
        &[],
        28,
    )
}

/// Samples the next token deterministically via `argmax` when `temperature <= 1e-4`, or
/// top-40 temperature-scaled softmax with xorshift64* when `temperature > 1e-4`.
pub(crate) fn sample_next_token(logits: &[f32], temperature: f32, rng_state: &mut u64) -> u32 {
    if temperature <= 1e-4 || logits.len() < 2 {
        return argmax(logits);
    }
    const TOP_K: usize = 40;
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(TOP_K);
    for (idx, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        if top.len() < TOP_K {
            top.push((idx, v));
            if top.len() == TOP_K {
                top.sort_by(|a, b| a.1.total_cmp(&b.1));
            }
        } else if v > top[0].1 {
            top[0] = (idx, v);
            for i in 0..(TOP_K - 1) {
                if top[i].1 > top[i + 1].1 {
                    top.swap(i, i + 1);
                } else {
                    break;
                }
            }
        }
    }
    if top.is_empty() {
        return 0;
    }
    let max_v = top.last().map_or(0.0, |&(_, v)| v);
    let inv_t = 1.0 / f64::from(temperature.max(1e-3));
    let mut weights: Vec<(usize, f64)> = Vec::with_capacity(top.len());
    let mut total = 0.0_f64;
    for (idx, v) in top {
        let w = (f64::from(v - max_v) * inv_t).exp();
        total += w;
        weights.push((idx, w));
    }
    if *rng_state == 0 {
        *rng_state = 0x9E37_79B9_7F4A_7C15_u64;
    }
    let mut x = *rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *rng_state = x;
    let u = ((x.wrapping_mul(0x2545_F491_4F6C_DD1D_u64) >> 11) as f64) / ((1_u64 << 53) as f64);
    let mut target = u * total;
    for &(idx, w) in weights.iter().rev() {
        if target <= w {
            return idx as u32;
        }
        target -= w;
    }
    weights.last().map_or(0, |&(idx, _)| idx as u32)
}

#[derive(Default)]
pub(crate) struct KvTraceStats {
    pub(crate) lookups: usize,
    pub(crate) exact_hits: usize,
    pub(crate) res_hits: usize,
    pub(crate) pframes_appended: usize,
    pub(crate) pframe_seeks: usize,
    pub(crate) pframe_repulsed: usize,
    pub(crate) lookup_us_sum: f64,
}

impl KvTraceStats {
    pub(crate) fn record_outcome(
        &mut self,
        outcome: RamPhaseKvStepOutcome,
        elapsed_us: f64,
        phase_deltas: &[lmnk_q7_t; 64],
    ) -> String {
        self.lookups += 1;
        self.lookup_us_sum += elapsed_us;
        let ph_l1: u32 = phase_deltas
            .iter()
            .map(|&v| u32::from(v.unsigned_abs()))
            .sum();
        let ph_head = &phase_deltas[..6];
        match outcome {
            RamPhaseKvStepOutcome::SkippedLiminal => "UNOBS(skip)".to_string(),
            RamPhaseKvStepOutcome::Unbacked => format!("UNOBS(miss,l1={ph_l1},h={ph_head:?})"),
            RamPhaseKvStepOutcome::StoredKeyframe {
                slot_idx,
                min_resonance_q7,
            } => format!("STORE(s{slot_idx},min={min_resonance_q7},l1={ph_l1},h={ph_head:?})"),
            RamPhaseKvStepOutcome::ResonatedSeeked {
                slot_idx,
                exact_hit,
                resonance_q7,
                min_resonance_q7,
                pframe_count,
                seek_r_bar,
                seek_repulsed,
                seek_executed,
            } => {
                if exact_hit {
                    self.exact_hits += 1;
                } else {
                    self.res_hits += 1;
                }
                self.pframes_appended += 1;
                let seek_note = if seek_executed {
                    self.pframe_seeks += 1;
                    self.pframe_repulsed += seek_repulsed as usize;
                    format!(" seek_r={seek_r_bar:.2} rep={seek_repulsed}")
                } else {
                    String::new()
                };
                if exact_hit {
                    format!(
                        "EXACT(s{slot_idx},pf={pframe_count},min={min_resonance_q7},l1={ph_l1},h={ph_head:?}{seek_note})"
                    )
                } else {
                    format!(
                        "RES(s{slot_idx},r={resonance_q7},min={min_resonance_q7},pf={pframe_count},l1={ph_l1},h={ph_head:?}{seek_note})"
                    )
                }
            }
        }
    }
}

pub(crate) fn parse_quant_mode<'a>(args: impl IntoIterator<Item = &'a str>) -> Option<GgmlDType> {
    for arg in args {
        if let Some(val) = arg.strip_prefix("--quant=") {
            if val.eq_ignore_ascii_case("q4_0") || val.eq_ignore_ascii_case("q40") {
                return Some(GgmlDType::Q4_0);
            }
            if val.eq_ignore_ascii_case("q4k") || val.eq_ignore_ascii_case("q4_k") {
                return Some(GgmlDType::Q4K);
            }
            if val.eq_ignore_ascii_case("q8_0") || val.eq_ignore_ascii_case("q80") {
                return Some(GgmlDType::Q8_0);
            }
            if val.eq_ignore_ascii_case("none") || val.eq_ignore_ascii_case("bf16") {
                return None;
            }
        }
    }
    None
}

pub(crate) fn build_preset_circuit(mode: CircuitProfileMode) -> Option<CognitiveTrajectoryLedger> {
    let mut analytical = CognitiveTrajectoryLedger::new();
    analytical.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::CellTrail,
        CognitiveOpKind::DecomposePeel,
        6,
    ));
    analytical.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::PointerBit,
        CognitiveOpKind::RouteSubHead,
        6,
    ));

    let mut integrative = CognitiveTrajectoryLedger::new();
    integrative.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::TypeGateway,
        CognitiveOpKind::SynthesizeFuse,
        6,
    ));
    integrative.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::MacroCoalition,
        CognitiveOpKind::CrystallizeCommit,
        7,
    ));

    match mode {
        CircuitProfileMode::None => None,
        CircuitProfileMode::Analytical => Some(analytical),
        CircuitProfileMode::Integrative => Some(integrative),
        CircuitProfileMode::Hybrid(blend_q7) => {
            Some(analytical.compose_persona_circuit(&integrative, blend_q7))
        }
    }
}

pub(crate) fn resolve_persona_circuit(
    mode: CircuitProfileMode,
    load_path: Option<&Path>,
) -> Option<CognitiveTrajectoryLedger> {
    let loaded = load_path
        .and_then(|p| LmnkTimeline::load_from_path(p).ok())
        .and_then(|tl| tl.persona_circuit());
    let preset = build_preset_circuit(mode);
    match (loaded, preset) {
        (Some(base), Some(ov)) => Some(base.compose_persona_circuit(&ov, 64)),
        (Some(base), None) => Some(base),
        (None, Some(ov)) => Some(ov),
        (None, None) => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FairBenchCondition {
    /// Condition A: Vanilla + Standard RAG (Fact + Persona + Guard text injected, Steer OFF).
    VanillaFullRag,
    /// Condition B: Vanilla + Minimal RAG (Fact text only, Steer OFF).
    VanillaMinRag,
    /// Condition C: LiminiKa Hybrid (Fact text only + RoPE/Rank-1/Substrate Steer ON).
    LiminiKaHybrid,
    /// Condition D: LiminiKa Zero-Context (No text injection + RoPE/Rank-1/Substrate Steer ON).
    LiminiKaZeroContext,
}

impl FairBenchCondition {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::VanillaFullRag => "A:Vanilla+FullRAG",
            Self::VanillaMinRag => "B:Vanilla+MinRAG",
            Self::LiminiKaHybrid => "C:LiminiKa-Hybrid",
            Self::LiminiKaZeroContext => "D:LiminiKa-ZeroCtx",
        }
    }

    pub(crate) const fn steering_enabled(self) -> bool {
        matches!(self, Self::LiminiKaHybrid | Self::LiminiKaZeroContext)
    }

    pub(crate) const fn inject_fact_text(self) -> bool {
        matches!(
            self,
            Self::VanillaFullRag | Self::VanillaMinRag | Self::LiminiKaHybrid
        )
    }

    pub(crate) const fn inject_persona_and_guard_text(self) -> bool {
        matches!(self, Self::VanillaFullRag)
    }
}

pub(crate) fn parse_fair_bench_conditions<'a>(
    args: impl IntoIterator<Item = &'a str>,
    default_no_reflect: bool,
) -> Option<Vec<(FairBenchCondition, bool, Option<bool>)>> {
    let mut fair_all = false;
    let mut selected: Vec<(FairBenchCondition, bool, Option<bool>)> = Vec::new();
    for arg in args {
        if arg == "--fair-bench" {
            fair_all = true;
        } else if let Some(val) = arg.strip_prefix("--bench-cond=") {
            for tok in val.split(',').map(str::trim) {
                match tok.to_ascii_lowercase().as_str() {
                    "a" | "vanilla-full" => {
                        selected.push((FairBenchCondition::VanillaFullRag, true, Some(false)))
                    }
                    "b" | "vanilla-min" => {
                        selected.push((FairBenchCondition::VanillaMinRag, true, Some(false)))
                    }
                    "c" | "hybrid" => selected.push((
                        FairBenchCondition::LiminiKaHybrid,
                        default_no_reflect,
                        None,
                    )),
                    "c-noreflect" | "c-off" => {
                        selected.push((FairBenchCondition::LiminiKaHybrid, true, None))
                    }
                    "c-byp-off" | "c(byp=off)" => {
                        selected.push((FairBenchCondition::LiminiKaHybrid, true, Some(false)))
                    }
                    "c-byp-on" | "c(byp=on)" => {
                        selected.push((FairBenchCondition::LiminiKaHybrid, true, Some(true)))
                    }
                    "c-reflect" | "c-on" => {
                        selected.push((FairBenchCondition::LiminiKaHybrid, false, None))
                    }
                    "d" | "zero-ctx" => {
                        selected.push((FairBenchCondition::LiminiKaZeroContext, true, None))
                    }
                    "all" => fair_all = true,
                    _ => {}
                }
            }
        }
    }
    if fair_all {
        Some(vec![
            (FairBenchCondition::VanillaFullRag, true, Some(false)),
            (FairBenchCondition::VanillaMinRag, true, Some(false)),
            (FairBenchCondition::LiminiKaHybrid, true, None),
            (FairBenchCondition::LiminiKaHybrid, false, None),
            (FairBenchCondition::LiminiKaZeroContext, true, None),
        ])
    } else if !selected.is_empty() {
        Some(selected)
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HorizonAblationMode {
    Core3,
    Spillover,
    SlowLoop,
}

impl HorizonAblationMode {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Core3 => "Core3",
            Self::Spillover => "Core3+Spillover",
            Self::SlowLoop => "Core3+Spillover+SlowLoop",
        }
    }
}

pub(crate) fn parse_horizon_mode<'a>(
    args: impl IntoIterator<Item = &'a str>,
) -> HorizonAblationMode {
    for arg in args {
        if let Some(val) = arg.strip_prefix("--horizon-mode=") {
            if val.eq_ignore_ascii_case("spillover") {
                return HorizonAblationMode::Spillover;
            }
            if val.eq_ignore_ascii_case("slow-loop") || val.eq_ignore_ascii_case("slow_loop") {
                return HorizonAblationMode::SlowLoop;
            }
            if val.eq_ignore_ascii_case("core3") {
                return HorizonAblationMode::Core3;
            }
        }
    }
    HorizonAblationMode::Core3
}

#[derive(Default)]
pub(crate) struct StepTelemetryTotals {
    pub(crate) core_bank: CoreTelemetryBank,
    pub(crate) total_steps: u64,
    pub(crate) continuous_standard_steps: u64,
    pub(crate) decision_counts: [u64; 7],
    pub(crate) excursion_steps: u64,
    pub(crate) ego_interventions: u64,
    pub(crate) escalation_swaps: u64,
    pub(crate) slide_in_count: u64,
    pub(crate) slide_out_count: u64,
    pub(crate) distortion_wakeups: u64,
    pub(crate) distortion_cooldowns: u64,
}

impl StepTelemetryTotals {
    pub(crate) fn accumulate(&mut self, snap: &CoreTelemetrySnapshot) {
        self.core_bank.accumulate_snapshot(snap);
        self.total_steps += snap.spine.total_steps;
        self.continuous_standard_steps += snap.spine.continuous_standard_steps;
        for (dst, &src) in self
            .decision_counts
            .iter_mut()
            .zip(snap.stigmergy.decision_counts.iter())
        {
            *dst += src;
        }
        self.excursion_steps += snap.liminal.regime_counts[1] + snap.liminal.regime_counts[2];
        self.ego_interventions += snap.ego.intervention_count;
        self.escalation_swaps += snap.ego.escalation_swaps;
        self.slide_in_count += snap.liminal.slide_in_count;
        self.slide_out_count += snap.liminal.slide_out_count;
    }

    pub(crate) fn distortion_report(&self) -> LayerDistortionReport {
        self.core_bank.compute_distortion_report()
    }

    pub(crate) fn decision_pct(&self, idx: usize) -> f32 {
        if self.continuous_standard_steps == 0 {
            0.0
        } else {
            self.decision_counts[idx] as f32 / self.continuous_standard_steps as f32 * 100.0
        }
    }

    pub(crate) fn excursion_pct(&self) -> f32 {
        if self.continuous_standard_steps == 0 {
            0.0
        } else {
            self.excursion_steps as f32 / self.continuous_standard_steps as f32 * 100.0
        }
    }

    pub(crate) fn ego_pct(&self) -> f32 {
        if self.continuous_standard_steps == 0 {
            0.0
        } else {
            self.ego_interventions as f32 / self.continuous_standard_steps as f32 * 100.0
        }
    }
}

pub(crate) struct SteerDecodeConfig<'a, 'c, 'e> {
    pub(crate) q7s: Option<&'a [Vec<lmnk_q7_t>]>,
    pub(crate) persona_q7s: Option<&'a [Vec<lmnk_q7_t>]>,
    pub(crate) resonant_ctx: Option<ResonantAnchors<'a>>,
    pub(crate) rope_rad: f32,
    pub(crate) rank1: Option<(&'a [f32], &'a [f32], f32)>,
    pub(crate) use_steer: bool,
    pub(crate) horizon_mode: HorizonAblationMode,
    pub(crate) escalation_provider: Option<&'e mut CorpusRecallEscalationProvider<'c>>,
    pub(crate) pressure_sim: PressureSimMode,
    pub(crate) enable_pre_bypass: bool,
    pub(crate) temperature: f32,
    pub(crate) rng_seed: u64,
    pub(crate) logits_observe_stride: usize,
    pub(crate) layer_steer_mask: Option<u64>,
    pub(crate) probe_frame: ProbeCoordinateFrame,
    pub(crate) phase_clamp_override: Option<f32>,
    pub(crate) random_push_seed: Option<u64>,
    pub(crate) max_tokens: usize,
    pub(crate) prefill_chunk_size: usize,
    pub(crate) capture_first_logits: bool,
    pub(crate) first_prefill_logits: Option<Vec<f32>>,
    pub(crate) exit_entropy_h_tilde: f32,
    pub(crate) steer_bypass_hits: usize,
    pub(crate) steer_eligible_layers: usize,
    pub(crate) spine_veto_guard_hits: usize,
    pub(crate) spine_veto_repulse_hits: usize,
    pub(crate) spine_veto_pin_hits: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct ReplyTiming {
    pub(crate) prefill_secs: f64,
    pub(crate) decode_secs: f64,
}

/// Runs an untimed warmup decode pass (1 short prompt, up to 10 greedy tokens) to stabilize
/// GPU P0 clocks before Trial 1, then explicitly clears the model KV cache.
pub(crate) fn run_gpu_warmup(
    model: &mut CausalLmModel,
    tokenizer: &Tokenizer,
    chat: &ChatTemplate,
) -> Result<(), Box<dyn std::error::Error>> {
    let t_warm = Instant::now();
    model.clear_cache();
    let warmup_prompt = chat.format_single_turn("Warmup: what is 2 + 2?");
    let ids = tokenizer
        .encode(warmup_prompt.as_str(), false)
        .map_err(|e| format!("warmup encode error: {e}"))?
        .get_ids()
        .to_vec();
    let (mut outcome, _, _) = model.forward_tokens(
        &ids,
        0,
        None,
        false,
        false,
        LogitsFetchMode::FullLogits,
        KvCacheMode::Append,
    )?;
    let mut gen_count = 0usize;
    for offset in (ids.len()..).take(10) {
        let next = match outcome {
            LogitsOutcome::ObservedFull(ref logits) => argmax(logits),
            LogitsOutcome::UnobservedDeviceArgmax(tok) => tok,
            LogitsOutcome::SkippedIntermediate => break,
        };
        gen_count += 1;
        if chat.is_stop_token(next) {
            break;
        }
        let (next_out, _, _) = model.forward_tokens(
            &[next],
            offset,
            None,
            false,
            false,
            LogitsFetchMode::FullLogits,
            KvCacheMode::Append,
        )?;
        outcome = next_out;
    }
    model.clear_cache();
    println!(
        "[Warmup] Ran {} untimed warmup decode token(s) in {:.1}ms and cleared KV cache.",
        gen_count,
        t_warm.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}

/// Runs a 5-prompt Steer-OFF greedy generation smoke test to validate RoPE pairing, YaRN
/// scaling, explicit `head_dim`, tied embeddings, and chat template / BOS / EOS resolution.
pub(crate) fn run_steer_off_smoke_test(
    model: &mut CausalLmModel,
    tokenizer: &Tokenizer,
    chat: &ChatTemplate,
    max_tokens: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    const PROMPTS: [(&str, &str); 5] = [
        (
            "Capital",
            "What is the capital of France? Answer in one short sentence.",
        ),
        (
            "Arithmetic",
            "What is 17 + 25? Answer with just the number.",
        ),
        (
            "Boiling",
            "What is the boiling point of water at sea level in Celsius?",
        ),
        ("Chemistry", "What is the chemical symbol for gold?"),
        ("Astronomy", "Which planet is fourth from the Sun?"),
    ];
    println!(
        "\n=== Phase 0: Steer-OFF Greedy Generation Smoke Test (template={}, stops={:?}) ===",
        chat.kind.label(),
        chat.stop_token_ids
    );
    let max_gen = max_tokens.clamp(1, 64);
    for (tag, prompt) in PROMPTS {
        model.clear_cache();
        let formatted = chat.format_single_turn(prompt);
        let ids = tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| format!("smoke encode error: {e}"))?
            .get_ids()
            .to_vec();
        let t0 = Instant::now();
        let (mut outcome, _, _) = model.forward_tokens(
            &ids,
            0,
            None,
            false,
            false,
            LogitsFetchMode::FullLogits,
            KvCacheMode::Append,
        )?;
        let mut gen_ids = Vec::with_capacity(max_gen);
        let mut hit_eos = false;
        for offset in (ids.len()..).take(max_gen) {
            let next = match outcome {
                LogitsOutcome::ObservedFull(ref logits) => argmax(logits),
                LogitsOutcome::UnobservedDeviceArgmax(tok) => tok,
                LogitsOutcome::SkippedIntermediate => break,
            };
            if chat.is_stop_token(next) {
                hit_eos = true;
                break;
            }
            gen_ids.push(next);
            let (next_out, _, _) = model.forward_tokens(
                &[next],
                offset,
                None,
                false,
                false,
                LogitsFetchMode::FullLogits,
                KvCacheMode::Append,
            )?;
            outcome = next_out;
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let reply = tokenizer
            .decode(&gen_ids, true)
            .map_err(|e| format!("smoke decode error: {e}"))?
            .replace('\n', " ");
        println!(
            "  [Smoke:{tag:<10}] in_tok={} gen_tok={} eos={} ({:.1}ms) => {reply}",
            ids.len(),
            gen_ids.len(),
            hit_eos,
            elapsed_ms
        );
    }
    model.clear_cache();
    Ok(())
}

enum SteerPrefillInput<'a> {
    Tokens(&'a [u32]),
    Embeds(&'a Tensor),
}

fn generate_steered_reply_core(
    model: &mut CausalLmModel,
    tokenizer: &Tokenizer,
    chat: &ChatTemplate,
    adapter: &mut ModelSteeringAdapter,
    prefill: SteerPrefillInput<'_>,
    start_offset: usize,
    cfg: &mut SteerDecodeConfig<'_, '_, '_>,
) -> Result<(String, Vec<u32>, ReplyTiming), Box<dyn std::error::Error>> {
    let prompt_len = match prefill {
        SteerPrefillInput::Tokens(ids) => ids.len(),
        SteerPrefillInput::Embeds(embeds) => embeds.dims3()?.1,
    };
    let saved_layer_mask = cfg
        .layer_steer_mask
        .filter(|&m| m != adapter.geometry().layer_rope_mask)
        .map(|m| {
            let prev = adapter.geometry().layer_rope_mask;
            adapter.set_layer_rope_mask(m);
            prev
        });

    let result = (|| -> Result<(String, Vec<u32>, ReplyTiming), Box<dyn std::error::Error>> {
        let mut trail = lmnk_pointer_trail_t::default();
        let mut steer = ForwardSteer {
            adapter,
            q7s: cfg.q7s,
            persona_q7s: cfg.persona_q7s,
            resonant_ctx: cfg.resonant_ctx,
            rope_rad: cfg.rope_rad,
            trail: &mut trail,
            rank1: cfg.rank1,
            horizon_mode: cfg.horizon_mode,
            escalation_provider: cfg.escalation_provider.as_deref_mut(),
            compute_dispatch: None,
            layer_steer_mask: cfg.layer_steer_mask,
            probe_frame: cfg.probe_frame,
            phase_clamp_override: cfg.phase_clamp_override,
            random_push_seed: cfg.random_push_seed,
            steer_bypass_hits: 0,
            steer_eligible_layers: 0,
        };
        let t_prefill = Instant::now();
        steer.adapter.reset_turn_decode_state();
        steer
            .adapter
            .set_max_observe_stride(cfg.logits_observe_stride.clamp(1, u8::MAX as usize) as u8);
        steer
            .adapter
            .set_pre_bypass_enabled(cfg.use_steer && cfg.enable_pre_bypass);

        let mut pending_outcome = match prefill {
            SteerPrefillInput::Tokens(ids) => {
                model
                    .prefill_tokens_chunked(
                        ids,
                        start_offset,
                        cfg.prefill_chunk_size.max(1),
                        Some(&mut steer),
                        None,
                        cfg.use_steer,
                        LogitsFetchMode::FullLogits,
                    )?
                    .0
            }
            SteerPrefillInput::Embeds(embeds) => {
                model
                    .forward_embeds(
                        embeds,
                        start_offset,
                        Some(&mut steer),
                        cfg.use_steer,
                        false,
                        LogitsFetchMode::FullLogits,
                        KvCacheMode::Append,
                    )?
                    .0
            }
        };
        let prefill_secs = t_prefill.elapsed().as_secs_f64();
        if cfg.capture_first_logits {
            if let LogitsOutcome::ObservedFull(ref logits) = pending_outcome {
                cfg.first_prefill_logits = Some(logits.clone());
            }
        }

        let max_gen_tokens = cfg.max_tokens.max(1);
        let mut gen_ids = Vec::with_capacity(max_gen_tokens);
        let mut rng_state = cfg.rng_seed.max(1);
        let greedy = cfg.temperature <= 1e-6;
        let mut entropy_sum = 0.0_f32;
        let mut entropy_count = 0_u32;
        let mut next_fetch_mode = LogitsFetchMode::FullLogits;

        let t_dec = Instant::now();
        for (step_idx, offset) in (start_offset + prompt_len..)
            .take(max_gen_tokens)
            .enumerate()
        {
            let next = match pending_outcome {
                LogitsOutcome::ObservedFull(ref logits) => {
                    sample_next_token(logits, cfg.temperature, &mut rng_state)
                }
                LogitsOutcome::UnobservedDeviceArgmax(tok_id) => tok_id,
                LogitsOutcome::SkippedIntermediate => break,
            };
            if chat.is_stop_token(next) {
                break;
            }
            gen_ids.push(next);

            let t_tok = Instant::now();
            let (next_outcome, _, _) = model.forward_tokens(
                &[next],
                offset,
                Some(&mut steer),
                cfg.use_steer,
                false,
                next_fetch_mode,
                KvCacheMode::Append,
            )?;
            let tok_ms = t_tok.elapsed().as_secs_f64() * 1000.0;

            let observed_entropy = match &next_outcome {
                LogitsOutcome::ObservedFull(logits) => {
                    let h = compute_top_k_logit_entropy(logits, 16);
                    entropy_sum += h;
                    entropy_count += 1;
                    Some(h)
                }
                LogitsOutcome::UnobservedDeviceArgmax(_) | LogitsOutcome::SkippedIntermediate => {
                    None
                }
            };

            if cfg.use_steer {
                let measured_q15 = ((tok_ms / 120.0).clamp(0.0, 1.0) * 32767.0) as u16;
                if !matches!(cfg.pressure_sim, PressureSimMode::Off)
                    && gen_ids.len() == 4
                    && steer.adapter.compute_pool().baseline_pressure_q15 == 8192
                {
                    steer.adapter.reanchor_compute_pressure(measured_q15);
                }
                let base_p = steer.adapter.compute_pool().baseline_pressure_q15;
                let pressure_q15 = match cfg.pressure_sim {
                    PressureSimMode::Off => base_p,
                    PressureSimMode::Elevated => {
                        measured_q15.max(base_p.saturating_add(6144).min(32767))
                    }
                    PressureSimMode::Surge => {
                        measured_q15.max(base_p.saturating_add(14336).min(32767))
                    }
                };
                let pressure = SystemPressureReading {
                    frame_time_pressure_q15: pressure_q15,
                    bandwidth_pressure_q15: pressure_q15,
                    l3_load_pressure_q15: pressure_q15 >> 1,
                    irreversible_step: 0,
                    _pad: 0,
                };
                let repulsion_active = ((steer.trail.flags & TRAIL_FLAG_REPULSION_ACTIVE) != 0)
                    || ((((steer.trail.current_ptr >> 48) as u8) & ACTION_MODE_PHASE_REPULSE) != 0);

                let decode_input = match observed_entropy {
                    Some(h) => SpineDecodeInput::from_observed_entropy_f32(
                        pressure,
                        step_idx as u16,
                        h,
                        SPINE_SENTINEL_AUTO_U8,
                        repulsion_active,
                        greedy,
                    ),
                    None => SpineDecodeInput::unobserved(
                        pressure,
                        step_idx as u16,
                        SPINE_SENTINEL_AUTO_U8,
                        repulsion_active,
                        greedy,
                    ),
                }
                .with_pre_bypass(cfg.enable_pre_bypass);

                let step_plan = steer.adapter.step_decode_control(&decode_input);
                if (step_plan.veto_mask & SPINE_BYPASS_VETO_GUARD_ACTIVE) != 0 {
                    cfg.spine_veto_guard_hits += 1;
                }
                if (step_plan.veto_mask & SPINE_BYPASS_VETO_REPULSION_ACTIVE) != 0 {
                    cfg.spine_veto_repulse_hits += 1;
                }
                if (step_plan.veto_mask & SPINE_BYPASS_VETO_PINNED_LOCK) != 0 {
                    cfg.spine_veto_pin_hits += 1;
                }
                steer.compute_dispatch = Some(step_plan.dispatch);
                next_fetch_mode = match step_plan.next_observe_mode_enum() {
                    SpineLogitsObserveMode::SkipToDeviceArgmax => LogitsFetchMode::DeviceArgmaxOnly,
                    SpineLogitsObserveMode::ObserveFullLogits => LogitsFetchMode::FullLogits,
                };
            }
            pending_outcome = next_outcome;
        }
        let decode_secs = t_dec.elapsed().as_secs_f64();
        cfg.exit_entropy_h_tilde = if entropy_count > 0 {
            entropy_sum / entropy_count as f32
        } else {
            steer.adapter.substrate().last_entropy_h_tilde
        };
        cfg.steer_bypass_hits += steer.steer_bypass_hits;
        cfg.steer_eligible_layers += steer.steer_eligible_layers;
        let text = tokenizer
            .decode(&gen_ids, true)
            .map_err(|e| format!("decode error: {e}"))?
            .replace('\n', " ");
        debug_assert_eq!(model.kv_len(), start_offset + prompt_len + gen_ids.len());
        Ok((
            text,
            gen_ids,
            ReplyTiming {
                prefill_secs,
                decode_secs,
            },
        ))
    })();

    if let Some(prev) = saved_layer_mask {
        adapter.set_layer_rope_mask(prev);
    }
    result
}

pub(crate) fn generate_steered_reply_with_offset(
    model: &mut CausalLmModel,
    tokenizer: &Tokenizer,
    chat: &ChatTemplate,
    adapter: &mut ModelSteeringAdapter,
    formatted_prompt: &str,
    start_offset: usize,
    cfg: &mut SteerDecodeConfig<'_, '_, '_>,
) -> Result<(String, Vec<u32>, ReplyTiming), Box<dyn std::error::Error>> {
    let ids = tokenizer
        .encode(formatted_prompt, false)
        .map_err(|e| format!("encode error: {e}"))?
        .get_ids()
        .to_vec();
    let (text, gen_ids, timing) = generate_steered_reply_core(
        model,
        tokenizer,
        chat,
        adapter,
        SteerPrefillInput::Tokens(&ids),
        start_offset,
        cfg,
    )?;
    let mut all_ids = ids;
    all_ids.extend_from_slice(&gen_ids);
    Ok((text, all_ids, timing))
}

/// Prefills pre-built multimodal `inputs_embeds` (`[1, prompt_len, hidden]`) at `start_offset`
/// and runs greedy/steered token decode with full `SystemSpineState` decode control parity.
pub(crate) fn generate_steered_reply_from_embeds_with_offset(
    model: &mut CausalLmModel,
    tokenizer: &Tokenizer,
    chat: &ChatTemplate,
    adapter: &mut ModelSteeringAdapter,
    inputs_embeds: &Tensor,
    start_offset: usize,
    cfg: &mut SteerDecodeConfig<'_, '_, '_>,
) -> Result<(String, Vec<u32>, ReplyTiming), Box<dyn std::error::Error>> {
    generate_steered_reply_core(
        model,
        tokenizer,
        chat,
        adapter,
        SteerPrefillInput::Embeds(inputs_embeds),
        start_offset,
        cfg,
    )
}

pub(crate) fn generate_steered_reply(
    model: &mut CausalLmModel,
    tokenizer: &Tokenizer,
    chat: &ChatTemplate,
    adapter: &mut ModelSteeringAdapter,
    formatted_prompt: &str,
    cfg: &mut SteerDecodeConfig<'_, '_, '_>,
) -> Result<(String, usize, usize, ReplyTiming), Box<dyn std::error::Error>> {
    model.clear_cache();
    let prompt_ids_len = tokenizer
        .encode(formatted_prompt, false)
        .map_err(|e| format!("encode error: {e}"))?
        .get_ids()
        .len();
    let (text, all_ids, timing) = generate_steered_reply_with_offset(
        model,
        tokenizer,
        chat,
        adapter,
        formatted_prompt,
        0,
        cfg,
    )?;
    let gen_tokens = all_ids.len().saturating_sub(prompt_ids_len);
    Ok((text, prompt_ids_len, gen_tokens, timing))
}

pub(crate) fn critique_generated_draft(
    model: &mut CausalLmModel,
    tokenizer: &Tokenizer,
    adapter: &ModelSteeringAdapter,
    compiled: &CompiledTokenCorpus,
    query_snap: &TrajectoryManifoldSnapshot,
    draft_text: &str,
    reflection_policy: &ReflectionPolicy,
) -> Result<Option<ReflectionCritique>, Box<dyn std::error::Error>> {
    let trimmed = draft_text.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let draft_ids = tokenizer
        .encode(trimmed, false)
        .map_err(|e| format!("draft encode error: {e}"))?
        .get_ids()
        .to_vec();
    if draft_ids.is_empty() {
        return Ok(None);
    }
    let (_, draft_tok_qs, _) = model.forward_tokens(
        &draft_ids,
        0,
        None,
        false,
        true,
        LogitsFetchMode::FullLogits,
        KvCacheMode::IsolatedProbe,
    )?;
    let draft_recall = compiled.recall_token_sequence(adapter, &draft_tok_qs, draft_ids.len())?;
    let draft_snap = TrajectoryManifoldSnapshot::from_recall(&draft_recall);
    Ok(Some(
        ManifoldReflectionEvaluator.assess_trajectory_with_ego(
            query_snap,
            &draft_snap,
            &compiled.recall_policy,
            reflection_policy,
            adapter.ego_state(),
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_forbid_hits_filters_explicit_negation_clauses() {
        let forb = vec![
            "PyTorch".to_string(),
            "24 GB".to_string(),
            "XML".to_string(),
            "LoRA".to_string(),
        ];
        // (a) Negation/refutation sentence
        assert_eq!(
            count_semantic_forbid_hits(
                "LiminiKa does not fine-tune PyTorch weights or require 24 GB VRAM.",
                &forb
            ),
            (2, 0)
        );
        // (b) Genuine affirmative hallucination
        assert_eq!(
            count_semantic_forbid_hits("LiminiKa uses PyTorch training on 24 GB VRAM.", &forb),
            (2, 2)
        );
        // (c) Mixed negation + affirmative conjunction
        assert_eq!(
            count_semantic_forbid_hits("It does not use XML, instead it uses LoRA.", &forb),
            (2, 1)
        );
        // (d) Contrastive comparison ("rather than") & inflected negation cues
        assert_eq!(
            count_semantic_forbid_hits("LiminiKa steers phases rather than LoRA weights.", &forb),
            (1, 0)
        );
        assert_eq!(
            count_semantic_forbid_hits(
                "LiminiKa avoids PyTorch fine-tuning and replaces LoRA adapters.",
                &forb
            ),
            (2, 0)
        );
    }

    #[test]
    fn semantic_forbid_hits_handles_multibyte_utf8_at_slice_boundaries() {
        let forb = vec!["PyTorch".to_string(), "LoRA".to_string()];
        // Place multibyte UTF-8 characters (`\u{2014}` 3B, `\u{2019}` 3B, `\u{3042}` 3B) across
        // the -96 prefix and +28 suffix byte offsets so raw byte offsets fall mid-char.
        let prefix_pad = format!("{}\u{2014}{}", "\u{3042}".repeat(31), "x");
        let neg_reply = format!(
            "{prefix_pad} does not use PyTorch\u{2019}\u{2014}\u{3042}weights or LoRA\u{2014}\u{2019}\u{3042}end."
        );
        assert_eq!(count_semantic_forbid_hits(&neg_reply, &forb), (2, 0));

        let aff_reply = format!(
            "{}\u{2019}\u{2014}PyTorch\u{2019}\u{2014}{} and {}\u{2014}\u{2019}LoRA\u{2014}\u{3042}",
            "\u{3042}".repeat(31),
            "z".repeat(25),
            "\u{3042}".repeat(33)
        );
        assert_eq!(count_semantic_forbid_hits(&aff_reply, &forb), (2, 2));
    }

    #[test]
    fn top_k_logit_entropy_bounds_and_seeded_sampling_are_deterministic() {
        let sharp = [12.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let flat = [1.0_f32; 6];
        assert!(compute_top_k_logit_entropy(&sharp, 6) < 0.05);
        assert!(compute_top_k_logit_entropy(&flat, 6) > 0.99);

        let logits = [0.5_f32, 2.0, 1.9, 1.8, 0.1, 0.2];
        let mut s1 = 42_u64;
        let mut s2 = 42_u64;
        assert_eq!(sample_next_token(&logits, 0.0, &mut s1), 1);
        assert_eq!(
            sample_next_token(&logits, 0.8, &mut s1),
            sample_next_token(&logits, 0.8, &mut s2)
        );
    }

    #[test]
    fn cli_spine_and_probe_frame_parsers_enforce_defaults_and_bounds() {
        assert_eq!(parse_logits_observe_stride(["candle_eval"]), 1);
        assert_eq!(
            parse_logits_observe_stride(["candle_eval", "--logits-observe-stride=0"]),
            1
        );
        assert_eq!(
            parse_logits_observe_stride(["candle_eval", "--logits-observe-stride=1"]),
            1
        );
        assert_eq!(
            parse_logits_observe_stride(["candle_eval", "--logits-observe-stride=5"]),
            5
        );
        assert_eq!(
            parse_guard_hold_tokens(["candle_eval"]),
            DEFAULT_GUARD_HOLD_TOKENS
        );
        assert_eq!(
            parse_guard_hold_tokens(["candle_eval", "--guard-hold-tokens=0"]),
            0
        );
        assert_eq!(
            parse_guard_hold_tokens(["candle_eval", "--guard-hold-tokens=12"]),
            12
        );
        assert_eq!(
            parse_probe_frame(["candle_eval"]),
            ProbeCoordinateFrame::PreRope
        );
        assert_eq!(
            parse_probe_frame(["candle_eval", "--probe-frame=pre-rope"]),
            ProbeCoordinateFrame::PreRope
        );
        assert_eq!(
            parse_probe_frame(["candle_eval", "--probe-frame=post-rope"]),
            ProbeCoordinateFrame::PostRope
        );

        assert_eq!(parse_layer_mask_spec(["candle_eval"]), None);
        assert_eq!(
            parse_layer_mask_spec(["candle_eval", "--layer-mask=top-half"])
                .map(|s| (s.resolve(32), s.resolve(26))),
            Some((0xFFFF_0000, 0x03FF_E000))
        );
        assert_eq!(
            parse_layer_mask_spec(["candle_eval", "--layer-mask=top16"]).map(|s| s.resolve(32)),
            Some(0xFFFF_0000)
        );
        assert_eq!(
            parse_layer_mask_spec(["candle_eval", "--layer-mask=top16"]).map(|s| s.resolve(26)),
            Some(0x03FF_0000)
        );
        assert_eq!(
            parse_layer_mask_spec(["candle_eval", "--layer-mask=mid12"]).map(|s| s.resolve(32)),
            Some(0x003F_FC00)
        );
        assert_eq!(
            parse_layer_mask_spec(["candle_eval", "--layer-mask=mid16"]).map(|s| s.resolve(32)),
            Some(0x03FF_FC00)
        );
        assert_eq!(
            parse_layer_mask_spec(["candle_eval", "--layer-mask=upper-mid12"])
                .map(|s| s.resolve(32)),
            Some(0x0FFF_0000)
        );
        assert_eq!(
            parse_layer_mask_spec(["candle_eval", "--layer-mask=all"])
                .map(|s| (s.resolve(32), s.resolve(26))),
            Some((0xFFFF_FFFF, 0x03FF_FFFF))
        );
        assert_eq!(
            parse_layer_mask_spec(["candle_eval", "--layer-mask=0x00FF_0000"])
                .map(|s| s.resolve(32)),
            Some(0x00FF_0000)
        );
        assert_eq!(
            parse_layer_mask_spec(["candle_eval", "--layer-mask=0x0"]),
            None
        );

        assert_eq!(parse_phase_clamp_rad(["candle_eval"]), None);
        assert_eq!(
            parse_phase_clamp_rad(["candle_eval", "--phase-clamp-rad=0.14"]),
            Some(0.14)
        );
        assert_eq!(
            parse_phase_clamp_rad(["candle_eval", "--phase-clamp-rad=0.0"]),
            None
        );
        assert_eq!(
            parse_phase_clamp_rad(["candle_eval", "--phase-clamp-rad=-0.1"]),
            None
        );
        assert_eq!(
            parse_phase_clamp_rad(["candle_eval", "--phase-clamp-rad=2.5"]),
            None
        );
        assert_eq!(
            parse_phase_clamp_rad(["candle_eval", "--phase-clamp-rad=NaN"]),
            None
        );

        assert_eq!(parse_max_tokens(["candle_eval"]), DEFAULT_MAX_GEN_TOKENS);
        assert_eq!(parse_max_tokens(["candle_eval", "--max-tokens=0"]), 1);
        assert_eq!(parse_max_tokens(["candle_eval", "--max-tokens=128"]), 128);
        assert_eq!(parse_max_tokens(["candle_eval", "--max-tokens=9999"]), 512);
    }

    #[test]
    fn extract_device_argmax_token_matches_cpu_argmax_for_f32_and_bf16() {
        use crate::model::extract_device_argmax_token;
        use candle_core::{DType, Device, Tensor};

        let logits = vec![-2.5_f32, 0.75, 4.25, 1.5, -0.5, 3.0];
        let expected = argmax(&logits);

        let t_f32 = Tensor::from_slice(&logits, (logits.len(),), &Device::Cpu).expect("f32 tensor");
        assert_eq!(
            extract_device_argmax_token(&t_f32).expect("f32 argmax"),
            expected
        );

        let t_bf16 = t_f32.to_dtype(DType::BF16).expect("bf16 cast");
        let bf16_cpu = t_bf16
            .to_dtype(DType::F32)
            .expect("bf16->f32")
            .to_vec1::<f32>()
            .expect("vec1");
        assert_eq!(
            extract_device_argmax_token(&t_bf16).expect("bf16 argmax"),
            argmax(&bf16_cpu)
        );
    }
}
