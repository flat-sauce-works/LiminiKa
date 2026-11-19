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
use liminika_core::traits::gateway::{
    CognitiveCircuitPlasticity, CognitiveOpKind, CognitiveStepOp, CognitiveTrajectoryLedger,
    CoreTelemetryBank, CoreTelemetrySnapshot, LayerDistortionReport, LiminalEgoPreset,
    RamPhaseKvStepOutcome, StigmergicScaleGranularity,
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
) -> Option<Vec<(FairBenchCondition, bool)>> {
    let mut fair_all = false;
    let mut selected: Vec<(FairBenchCondition, bool)> = Vec::new();
    for arg in args {
        if arg == "--fair-bench" {
            fair_all = true;
        } else if let Some(val) = arg.strip_prefix("--bench-cond=") {
            for tok in val.split(',').map(str::trim) {
                match tok.to_ascii_lowercase().as_str() {
                    "a" | "vanilla-full" => {
                        selected.push((FairBenchCondition::VanillaFullRag, true))
                    }
                    "b" | "vanilla-min" => selected.push((FairBenchCondition::VanillaMinRag, true)),
                    "c" | "hybrid" => {
                        selected.push((FairBenchCondition::LiminiKaHybrid, default_no_reflect))
                    }
                    "c-noreflect" | "c-off" => {
                        selected.push((FairBenchCondition::LiminiKaHybrid, true))
                    }
                    "c-reflect" | "c-on" => {
                        selected.push((FairBenchCondition::LiminiKaHybrid, false))
                    }
                    "d" | "zero-ctx" => {
                        selected.push((FairBenchCondition::LiminiKaZeroContext, true))
                    }
                    "all" => fair_all = true,
                    _ => {}
                }
            }
        }
    }
    if fair_all {
        Some(vec![
            (FairBenchCondition::VanillaFullRag, true),
            (FairBenchCondition::VanillaMinRag, true),
            (FairBenchCondition::LiminiKaHybrid, true),
            (FairBenchCondition::LiminiKaHybrid, false),
            (FairBenchCondition::LiminiKaZeroContext, true),
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
    pub(crate) steer_bypass_hits: usize,
    pub(crate) steer_eligible_layers: usize,
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
    let prefill_dispatch = steer.adapter.step_compute_dispatch(SystemPressureReading {
        frame_time_pressure_q15: 8192,
        bandwidth_pressure_q15: 8192,
        l3_load_pressure_q15: 8192,
        irreversible_step: 1,
        _pad: 0,
    });
    steer.compute_dispatch = Some(prefill_dispatch);
    let (mut logits, _, _) =
        model.forward_tokens(&ids, start_offset, Some(&mut steer), cfg.use_steer, false)?;
    steer
        .adapter
        .commit_compute_exit_reward(prefill_dispatch.execute_mask, 64);

    const MAX_GEN_TOKENS: usize = 64;
    let mut gen_ids = Vec::with_capacity(MAX_GEN_TOKENS);
    for offset in (start_offset + ids.len()..).take(MAX_GEN_TOKENS) {
        let next = argmax(&logits);
        if next == 199999 || next == 200020 {
            gen_ids.push(next);
            break;
        }
        gen_ids.push(next);
        let t_tok = Instant::now();
        let (next_logits, _, _) =
            model.forward_tokens(&[next], offset, Some(&mut steer), cfg.use_steer, false)?;
        let tok_ms = t_tok.elapsed().as_secs_f64() * 1000.0;
        let measured_q15 = ((tok_ms / 120.0).clamp(0.0, 1.0) * 32767.0) as u16;
        if gen_ids.len() == 4 && steer.adapter.compute_pool().baseline_pressure_q15 == 8192 {
            steer
                .adapter
                .compute_pool_mut()
                .reanchor_baseline_pressure(measured_q15);
        }
        let base_p = steer.adapter.compute_pool().baseline_pressure_q15;
        let pressure_q15 = match cfg.pressure_sim {
            PressureSimMode::Off => measured_q15,
            PressureSimMode::Elevated => measured_q15.max(base_p.saturating_add(6144).min(32767)),
            PressureSimMode::Surge => measured_q15.max(base_p.saturating_add(14336).min(32767)),
        };
        let exit_entropy = steer.adapter.action_hub().last_entropy_h_tilde;
        if exit_entropy > 0.65 {
            steer.adapter.signal_compute_instability();
            cfg.instability_recoveries += 1;
        }
        let tok_dispatch = steer.adapter.step_compute_dispatch(SystemPressureReading {
            frame_time_pressure_q15: pressure_q15,
            bandwidth_pressure_q15: pressure_q15,
            l3_load_pressure_q15: pressure_q15 >> 1,
            irreversible_step: 0,
            _pad: 0,
        });
        let exit_reward = if exit_entropy < 0.55 { 56 } else { 96 };
        steer
            .adapter
            .commit_compute_exit_reward(tok_dispatch.execute_mask, exit_reward);
        steer.compute_dispatch = Some(tok_dispatch);
        logits = next_logits;
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
