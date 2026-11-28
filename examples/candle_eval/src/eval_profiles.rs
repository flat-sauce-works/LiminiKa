// name: examples/candle_eval/src/eval_profiles.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CLI profile enums, argument parsers, and cognitive circuit / Ego profile builders for
//! `candle_eval`.

use std::path::{Path, PathBuf};

use candle_core::quantized::GgmlDType;
use liminika_adapter::{LmnkTimeline, ModelSteeringAdapter};
use liminika_core::abi::lmnk_q7_t;
use liminika_core::drivers::LiminalEgoPreset;
use liminika_core::stigmergy::{
    CognitiveOpKind, CognitiveStepOp, CognitiveTrajectoryLedger, StigmergicScaleGranularity,
};

use crate::model::{
    ChatTemplateArg, ModelArchArg, ProbeCoordinateFrame, active_layers_mask, clip_layer_mask,
    top_half_layer_mask,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum LayerMaskSpec {
    #[default]
    Auto,
    TopHalf,
    All,
    Fixed(u64),
}

impl LayerMaskSpec {
    pub(crate) const fn is_auto(self) -> bool {
        matches!(self, Self::Auto)
    }

    pub(crate) const fn resolve(self, num_layers: usize) -> u64 {
        match self {
            Self::Auto | Self::TopHalf => top_half_layer_mask(num_layers),
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
            if let Some(blend_str) = val.strip_prefix("hybrid:")
                && let Ok(blend) = blend_str.parse::<u8>()
            {
                return CircuitProfileMode::Hybrid(blend.min(128));
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
        if let Some(val) = arg.strip_prefix("--kv-min-res=")
            && let Ok(parsed) = val.parse::<i8>()
        {
            return Some(parsed.clamp(-127, 127));
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
        if let Some(val) = arg.strip_prefix("--ego-max-hops=")
            && let Ok(parsed) = val.parse::<u8>()
        {
            return parsed.clamp(1, 4);
        }
    }
    1
}

pub(crate) fn parse_temperature<'a>(args: impl IntoIterator<Item = &'a str>) -> f32 {
    for arg in args {
        if let Some(val) = arg.strip_prefix("--temp=")
            && let Ok(parsed) = val.parse::<f32>()
            && parsed.is_finite()
            && parsed >= 0.0
        {
            return parsed;
        }
    }
    0.0
}

pub(crate) fn parse_rng_seed<'a>(args: impl IntoIterator<Item = &'a str>) -> u64 {
    for arg in args {
        if let Some(val) = arg.strip_prefix("--seed=")
            && let Ok(parsed) = val.parse::<u64>()
        {
            return parsed.max(1);
        }
    }
    42
}

pub(crate) const DEFAULT_LOGITS_OBSERVE_STRIDE: usize = 1;
pub(crate) const DEFAULT_GUARD_HOLD_TOKENS: u8 = ModelSteeringAdapter::DEFAULT_GUARD_HOLD_TOKENS;
pub(crate) const DEFAULT_MAX_GEN_TOKENS: usize = 64;
pub(crate) const DEFAULT_PREFILL_CHUNK_SIZE: usize = crate::model::DEFAULT_PREFILL_CHUNK_SIZE;

pub(crate) fn parse_prefill_chunk_size<'a>(args: impl IntoIterator<Item = &'a str>) -> usize {
    for arg in args {
        if let Some(raw) = arg.strip_prefix("--prefill-chunk-size=")
            && let Ok(parsed) = raw.trim().parse::<usize>()
        {
            return parsed.clamp(16, 4096);
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
            if val.eq_ignore_ascii_case("auto") {
                return Some(LayerMaskSpec::Auto);
            }
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
                if let Ok(parsed) = u64::from_str_radix(&cleaned, 16)
                    && parsed != 0
                {
                    return Some(LayerMaskSpec::Fixed(parsed));
                }
            }
        }
    }
    None
}

pub(crate) fn parse_phase_clamp_rad<'a>(args: impl IntoIterator<Item = &'a str>) -> Option<f32> {
    for arg in args {
        if let Some(raw) = arg.strip_prefix("--phase-clamp-rad=")
            && let Ok(parsed) = raw.trim().parse::<f32>()
            && parsed.is_finite()
            && parsed > 0.0
            && parsed <= std::f32::consts::FRAC_PI_2
        {
            return Some(parsed);
        }
    }
    None
}

pub(crate) fn parse_max_tokens<'a>(args: impl IntoIterator<Item = &'a str>) -> usize {
    for arg in args {
        if let Some(raw) = arg.strip_prefix("--max-tokens=")
            && let Ok(parsed) = raw.trim().parse::<usize>()
        {
            return parsed.clamp(1, 512);
        }
    }
    DEFAULT_MAX_GEN_TOKENS
}

pub(crate) fn parse_logits_observe_stride<'a>(args: impl IntoIterator<Item = &'a str>) -> usize {
    for arg in args {
        if let Some(val) = arg.strip_prefix("--logits-observe-stride=")
            && let Ok(parsed) = val.parse::<usize>()
        {
            return parsed.max(1);
        }
    }
    DEFAULT_LOGITS_OBSERVE_STRIDE
}

pub(crate) fn parse_guard_hold_tokens<'a>(args: impl IntoIterator<Item = &'a str>) -> u8 {
    for arg in args {
        if let Some(val) = arg.strip_prefix("--guard-hold-tokens=")
            && let Ok(parsed) = val.parse::<u8>()
        {
            return parsed;
        }
    }
    DEFAULT_GUARD_HOLD_TOKENS
}

pub(crate) fn parse_bench_trials<'a>(args: impl IntoIterator<Item = &'a str>) -> usize {
    for arg in args {
        if let Some(val) = arg.strip_prefix("--bench-trials=")
            && let Ok(parsed) = val.parse::<usize>()
        {
            return parsed.max(1);
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
