// name: examples/candle_eval/src/eval.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! CLI profile parsers, KV-cache statistics, steered decoding, and reflection critique.

use std::path::Path;
use std::time::Instant;

use candle_core::quantized::GgmlDType;
use liminika_adapter::{
    AttractorTrajectoryEvaluator, CompiledTokenCorpus, CorpusRecallEscalationProvider,
    GcsoDualTrackTimeline, ManifoldReflectionEvaluator, ModelSteeringAdapter, ReflectionCritique,
    ReflectionPolicy, SystemPressureReading, TrajectoryManifoldSnapshot,
};
use liminika_core::abi::{gcso_pointer_trail_t, gcso_q7_t};
use liminika_core::drivers::{LayerDistortionReport, LiminalEgoPreset, TokenBypassObservation};
use liminika_core::liminal::ACTION_MODE_PHASE_REPULSE;
use liminika_core::runtime::{CoreTelemetryBank, CoreTelemetrySnapshot, RamPhaseKvStepOutcome};
use liminika_core::stigmergy::{
    CognitiveCircuitPlasticity, CognitiveOpKind, CognitiveStepOp, CognitiveTrajectoryLedger,
    StigmergicScaleGranularity, TRAIL_FLAG_REPULSION_ACTIVE,
};
use tokenizers::Tokenizer;

use crate::bake::argmax;
use crate::model::{ForwardSteer, Phi3Model, ResonantAnchors};

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

pub(crate) fn parse_kv_min_res<'a>(args: impl IntoIterator<Item = &'a str>) -> Option<gcso_q7_t> {
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

    let mut raw_hits = 0_usize;
    let mut semantic_hits = 0_usize;
    for kw in forbidden {
        let kw_l = kw.to_ascii_lowercase();
        if kw_l.is_empty() {
            continue;
        }
        let mut matched_raw = false;
        let mut matched_semantic = false;
        for &clause in &clauses {
            let mut search_from = 0_usize;
            while let Some(rel) = clause[search_from..].find(&kw_l) {
                matched_raw = true;
                let abs_pos = search_from + rel;
                let prefix_start = abs_pos.saturating_sub(96);
                let prefix = &clause[prefix_start..abs_pos];
                let suffix_end = (abs_pos + kw_l.len() + 28).min(clause.len());
                let suffix = &clause[abs_pos + kw_l.len()..suffix_end];
                let is_negated = PREFIX_NEG_CUES.iter().any(|cue| prefix.contains(cue))
                    || SUFFIX_NEG_CUES.iter().any(|cue| suffix.contains(cue));
                if !is_negated {
                    matched_semantic = true;
                }
                search_from = abs_pos + kw_l.len();
                if search_from >= clause.len() {
                    break;
                }
            }
        }
        if matched_raw {
            raw_hits += 1;
        }
        if matched_semantic {
            semantic_hits += 1;
        }
    }
    (raw_hits, semantic_hits)
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
        phase_deltas: &[gcso_q7_t; 64],
    ) -> String {
        self.lookups += 1;
        self.lookup_us_sum += elapsed_us;
        let ph_l1: u32 = phase_deltas
            .iter()
            .map(|&v| u32::from(v.unsigned_abs()))
            .sum();
        let ph_head = &phase_deltas[..6];
        match outcome {
            RamPhaseKvStepOutcome::SkippedVoid => "VOID(skip)".to_string(),
            RamPhaseKvStepOutcome::Unbacked => format!("VOID(miss,l1={ph_l1},h={ph_head:?})"),
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
        .and_then(|p| GcsoDualTrackTimeline::load_from_path(p).ok())
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
    /// Condition C: LiminiKa Hybrid (Fact text only + RoPE/Rank-1/ActionHub Steer ON).
    LiminiKaHybrid,
    /// Condition D: LiminiKa Zero-Context (No text injection + RoPE/Rank-1/ActionHub Steer ON).
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
        self.total_steps += snap.system.total_steps;
        self.continuous_standard_steps += snap.system.continuous_standard_steps;
        for (dst, &src) in self
            .decision_counts
            .iter_mut()
            .zip(snap.layer1.decision_counts.iter())
        {
            *dst += src;
        }
        self.excursion_steps += snap.layer0.regime_counts[1] + snap.layer0.regime_counts[2];
        self.ego_interventions += snap.layer2.intervention_count;
        self.escalation_swaps += snap.layer2.escalation_swaps;
        self.slide_in_count += snap.layer0.slide_in_count;
        self.slide_out_count += snap.layer0.slide_out_count;
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
    pub(crate) q7s: Option<&'a [Vec<gcso_q7_t>]>,
    pub(crate) persona_q7s: Option<&'a [Vec<gcso_q7_t>]>,
    pub(crate) resonant_ctx: Option<ResonantAnchors<'a>>,
    pub(crate) rope_rad: f32,
    pub(crate) srl: Option<(&'a [f32], &'a [f32], f32)>,
    pub(crate) use_steer: bool,
    pub(crate) horizon_mode: HorizonAblationMode,
    pub(crate) escalation_provider: Option<&'e mut CorpusRecallEscalationProvider<'c>>,
    pub(crate) pressure_sim: PressureSimMode,
    pub(crate) enable_pre_bypass: bool,
    pub(crate) temperature: f32,
    pub(crate) rng_seed: u64,
    pub(crate) steer_bypass_hits: usize,
    pub(crate) steer_eligible_layers: usize,
    pub(crate) whole_token_bypass_hits: usize,
    pub(crate) decode_tokens_total: usize,
    pub(crate) instability_recoveries: usize,
}

pub(crate) fn generate_steered_reply_with_offset(
    model: &mut Phi3Model,
    tokenizer: &Tokenizer,
    adapter: &mut ModelSteeringAdapter,
    formatted_prompt: &str,
    start_offset: usize,
    cfg: &mut SteerDecodeConfig<'_, '_, '_>,
) -> Result<(String, Vec<u32>, f64), Box<dyn std::error::Error>> {
    let mut trail = gcso_pointer_trail_t::default();
    let ids = tokenizer
        .encode(formatted_prompt, false)
        .map_err(|e| format!("encode error: {e}"))?
        .get_ids()
        .to_vec();

    let mut steer = ForwardSteer {
        adapter,
        q7s: cfg.q7s,
        persona_q7s: cfg.persona_q7s,
        resonant_ctx: cfg.resonant_ctx,
        rope_rad: cfg.rope_rad,
        trail: &mut trail,
        srl: cfg.srl,
        horizon_mode: cfg.horizon_mode,
        escalation_provider: cfg.escalation_provider.as_deref_mut(),
        compute_dispatch: None,
        steer_bypass_hits: 0,
        steer_eligible_layers: 0,
    };
    let t_dec = Instant::now();
    steer.adapter.compute_pool_mut().whole_token_calm_dwell = 0;
    steer.adapter.compute_pool_mut().re_bypass_cooldown = 0;
    steer.adapter.action_hub_mut().last_entropy_h_tilde = 0.0;
    let mut prefill_dispatch = steer.adapter.step_compute_dispatch(SystemPressureReading {
        frame_time_pressure_q15: 8192,
        bandwidth_pressure_q15: 8192,
        l3_load_pressure_q15: 8192,
        irreversible_step: 1,
        _pad: 0,
    });
    prefill_dispatch.whole_token_steering_bypass = 0;
    steer.compute_dispatch = Some(prefill_dispatch);
    let (mut logits, _, _) =
        model.forward_tokens(&ids, start_offset, Some(&mut steer), cfg.use_steer, false)?;
    steer
        .adapter
        .commit_compute_exit_reward(prefill_dispatch.execute_mask, 64);

    const MAX_GEN_TOKENS: usize = 64;
    let mut gen_ids = Vec::with_capacity(MAX_GEN_TOKENS);
    let mut rng_state = cfg.rng_seed.max(1);
    let mut entropy_sum = 0.0f32;
    let mut entropy_count = 0u32;
    for offset in (start_offset + ids.len()..).take(MAX_GEN_TOKENS) {
        let next = sample_next_token(&logits, cfg.temperature, &mut rng_state);
        if next == 199999 || next == 200020 {
            gen_ids.push(next);
            break;
        }
        gen_ids.push(next);
        let tok_was_whole_bypass = steer
            .compute_dispatch
            .is_some_and(|d| d.whole_token_steering_bypass != 0);
        if cfg.use_steer {
            cfg.decode_tokens_total += 1;
            if tok_was_whole_bypass {
                cfg.whole_token_bypass_hits += 1;
            }
        }
        let t_tok = Instant::now();
        let (next_logits, _, _) =
            model.forward_tokens(&[next], offset, Some(&mut steer), cfg.use_steer, false)?;
        let tok_ms = t_tok.elapsed().as_secs_f64() * 1000.0;
        let measured_q15 = ((tok_ms / 120.0).clamp(0.0, 1.0) * 32767.0) as u16;
        if !matches!(cfg.pressure_sim, PressureSimMode::Off)
            && gen_ids.len() == 4
            && steer.adapter.compute_pool().baseline_pressure_q15 == 8192
        {
            steer
                .adapter
                .compute_pool_mut()
                .reanchor_baseline_pressure(measured_q15);
        }
        let base_p = steer.adapter.compute_pool().baseline_pressure_q15;
        let pressure_q15 = match cfg.pressure_sim {
            PressureSimMode::Off => base_p,
            PressureSimMode::Elevated => measured_q15.max(base_p.saturating_add(6144).min(32767)),
            PressureSimMode::Surge => measured_q15.max(base_p.saturating_add(14336).min(32767)),
        };
        let logit_entropy = compute_top_k_logit_entropy(&next_logits, 16);
        entropy_sum += logit_entropy;
        entropy_count += 1;
        if tok_was_whole_bypass {
            steer.adapter.action_hub_mut().last_entropy_h_tilde = logit_entropy;
        }
        let exit_entropy_q7 = ((logit_entropy - 0.15).max(0.0) * 127.0).clamp(0.0, 127.0) as u8;
        let repulsion_active = (steer.adapter.action_hub().slot_repulsion_mask != 0)
            || ((steer.trail.flags & TRAIL_FLAG_REPULSION_ACTIVE) != 0)
            || ((((steer.trail.current_ptr >> 48) as u8) & ACTION_MODE_PHASE_REPULSE) != 0);
        let sens_q7 = steer.adapter.last_venn_partition().boundary_sensitivity_q7;
        let guard_prox_q7 = if repulsion_active { 88 } else { sens_q7 >> 2 };
        let active_overdrive = repulsion_active && steer.adapter.ego_field().is_active();
        steer.adapter.step_distortion_tension_hot_path(
            exit_entropy_q7,
            guard_prox_q7,
            active_overdrive,
        );
        let entropy_surge = logit_entropy > 0.85;
        if entropy_surge {
            steer.adapter.signal_compute_instability();
            cfg.instability_recoveries += 1;
        }
        let mut tok_dispatch = steer.adapter.step_compute_dispatch(SystemPressureReading {
            frame_time_pressure_q15: pressure_q15,
            bandwidth_pressure_q15: pressure_q15,
            l3_load_pressure_q15: pressure_q15 >> 1,
            irreversible_step: 0,
            _pad: 0,
        });
        if cfg.use_steer && cfg.enable_pre_bypass {
            let ego_active_boundary = steer.adapter.ego_field().is_active() && sens_q7 >= 80;
            let obs = TokenBypassObservation {
                irreversible_step: 0,
                liminal_stationary: u8::from(logit_entropy < 0.78 && sens_q7 < 80),
                boundary_sensitivity_q7: sens_q7,
                repulsion_active: u8::from(repulsion_active),
                exit_entropy_rising: u8::from(entropy_surge),
                ego_active_boundary: u8::from(ego_active_boundary),
                min_calm_dwell: 2,
                _pad: 0,
            };
            let _ = steer
                .adapter
                .evaluate_whole_token_bypass(obs, &mut tok_dispatch);
        } else {
            tok_dispatch.whole_token_steering_bypass = 0;
        }
        let exit_reward = if logit_entropy < 0.55 { 56 } else { 96 };
        steer
            .adapter
            .commit_compute_exit_reward(tok_dispatch.execute_mask, exit_reward);
        steer.compute_dispatch = Some(tok_dispatch);
        logits = next_logits;
    }
    if entropy_count > 0 {
        steer.adapter.action_hub_mut().last_entropy_h_tilde = entropy_sum / entropy_count as f32;
    }
    cfg.steer_bypass_hits += steer.steer_bypass_hits;
    cfg.steer_eligible_layers += steer.steer_eligible_layers;
    let decode_secs = t_dec.elapsed().as_secs_f64();
    let text = tokenizer
        .decode(&gen_ids, true)
        .map_err(|e| format!("decode error: {e}"))?
        .replace('\n', " ");
    let mut all_ids = ids;
    all_ids.extend_from_slice(&gen_ids);
    Ok((text, all_ids, decode_secs))
}

pub(crate) fn generate_steered_reply(
    model: &mut Phi3Model,
    tokenizer: &Tokenizer,
    adapter: &mut ModelSteeringAdapter,
    formatted_prompt: &str,
    cfg: &mut SteerDecodeConfig<'_, '_, '_>,
) -> Result<(String, usize, usize, f64), Box<dyn std::error::Error>> {
    model.clear_cache();
    let prompt_ids_len = tokenizer
        .encode(formatted_prompt, false)
        .map_err(|e| format!("encode error: {e}"))?
        .get_ids()
        .len();
    let (text, all_ids, decode_secs) =
        generate_steered_reply_with_offset(model, tokenizer, adapter, formatted_prompt, 0, cfg)?;
    let gen_tokens = all_ids.len().saturating_sub(prompt_ids_len);
    Ok((text, prompt_ids_len, gen_tokens, decode_secs))
}

pub(crate) fn critique_generated_draft(
    model: &mut Phi3Model,
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
    model.clear_cache();
    let (_, draft_tok_qs, _) = model.forward_tokens(&draft_ids, 0, None, false, true)?;
    let draft_recall = compiled.recall_token_sequence(adapter, &draft_tok_qs, draft_ids.len())?;
    let draft_snap = TrajectoryManifoldSnapshot::from_recall(&draft_recall);
    Ok(Some(
        ManifoldReflectionEvaluator.assess_trajectory_with_ego(
            query_snap,
            &draft_snap,
            &compiled.recall_policy,
            reflection_policy,
            adapter.ego_field(),
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
}
