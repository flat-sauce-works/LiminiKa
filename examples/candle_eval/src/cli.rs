// name: examples/candle_eval/src/cli.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Command-line configuration parsing, run-spec construction, and text-similarity metrics.

use std::collections::HashSet;
use std::path::PathBuf;

use crate::eval::{
    parse_circuit_mode, parse_ego_max_hops, parse_ego_mode, parse_fair_bench_conditions,
    parse_horizon_mode, parse_kv_min_res, parse_pressure_sim_mode, parse_quant_mode,
    parse_reflect_mode, parse_rng_seed, parse_temperature, CircuitProfileMode, EgoProfileMode,
    FairBenchCondition, HorizonAblationMode, PressureSimMode, ReflectMode,
};
use candle_core::quantized::GgmlDType;
use liminika_core::abi::lmnk_q7_t;
use liminika_core::drivers::LiminalEgoPreset;

pub(crate) struct EvalCliConfig {
    pub(crate) recall_only: bool,
    pub(crate) use_base_prior: bool,
    pub(crate) no_reflect: bool,
    pub(crate) reflect_mode: ReflectMode,
    pub(crate) enable_pre_bypass: bool,
    pub(crate) ablate_dual_axis: bool,
    pub(crate) ego_max_hops: u8,
    pub(crate) temperature: f32,
    pub(crate) rng_seed: u64,
    pub(crate) multi_turn: bool,
    pub(crate) quant_mode: Option<GgmlDType>,
    pub(crate) kv_min_res: Option<lmnk_q7_t>,
    pub(crate) circuit_mode: CircuitProfileMode,
    pub(crate) ego_mode: EgoProfileMode,
    pub(crate) ablate_ego: bool,
    pub(crate) horizon_mode: HorizonAblationMode,
    pub(crate) ablate_horizon: bool,
    pub(crate) fair_bench_conditions: Option<Vec<(FairBenchCondition, bool, Option<bool>)>>,
    pub(crate) pressure_sim_mode: PressureSimMode,
    pub(crate) step2_only: bool,
    pub(crate) persona_container_in: Option<PathBuf>,
    pub(crate) persona_container_out: Option<PathBuf>,
}

impl EvalCliConfig {
    pub(crate) fn parse_from_env() -> Self {
        let cli_args: Vec<String> = std::env::args().collect();
        let recall_only = cli_args.iter().any(|a| a == "--recall-only");
        let use_base_prior = cli_args.iter().any(|a| a == "--base-prior");
        let no_reflect = cli_args.iter().any(|a| a == "--no-reflect");
        let reflect_mode = parse_reflect_mode(cli_args.iter().map(String::as_str));
        let enable_pre_bypass = !cli_args.iter().any(|a| a == "--no-pre-bypass");
        let ablate_dual_axis = cli_args.iter().any(|a| a == "--ablate-dual-axis");
        let ego_max_hops = parse_ego_max_hops(cli_args.iter().map(String::as_str));
        let temperature = parse_temperature(cli_args.iter().map(String::as_str));
        let rng_seed = parse_rng_seed(cli_args.iter().map(String::as_str));
        let multi_turn = cli_args.iter().any(|a| a == "--multi-turn");
        let quant_mode = parse_quant_mode(cli_args.iter().map(String::as_str));
        let kv_min_res = parse_kv_min_res(cli_args.iter().map(String::as_str));
        let circuit_mode = parse_circuit_mode(cli_args.iter().map(String::as_str));
        let ego_mode = parse_ego_mode(cli_args.iter().map(String::as_str));
        let ablate_ego = cli_args.iter().any(|a| a == "--ablate-ego");
        let horizon_mode = parse_horizon_mode(cli_args.iter().map(String::as_str));
        let ablate_horizon = cli_args.iter().any(|a| a == "--ablate-horizon");
        let fair_bench_conditions =
            parse_fair_bench_conditions(cli_args.iter().map(String::as_str), no_reflect);
        let pressure_sim_mode = parse_pressure_sim_mode(cli_args.iter().map(String::as_str));
        let step2_only = cli_args.iter().any(|a| a == "--step2-only")
            || recall_only
            || fair_bench_conditions.is_some()
            || ablate_dual_axis;
        let persona_container_in = cli_args
            .iter()
            .find_map(|a| {
                a.strip_prefix("--persona-container=")
                    .or_else(|| a.strip_prefix("--persona-gcso="))
            })
            .map(PathBuf::from);
        let persona_container_out = cli_args
            .iter()
            .find_map(|a| {
                a.strip_prefix("--save-persona-container=")
                    .or_else(|| a.strip_prefix("--save-persona-gcso="))
            })
            .map(PathBuf::from)
            .or_else(|| persona_container_in.clone());

        Self {
            recall_only,
            use_base_prior,
            no_reflect,
            reflect_mode,
            enable_pre_bypass,
            ablate_dual_axis,
            ego_max_hops,
            temperature,
            rng_seed,
            multi_turn,
            quant_mode,
            kv_min_res,
            circuit_mode,
            ego_mode,
            ablate_ego,
            horizon_mode,
            ablate_horizon,
            fair_bench_conditions,
            pressure_sim_mode,
            step2_only,
            persona_container_in,
            persona_container_out,
        }
    }
}

#[derive(Clone)]
pub(crate) struct Step2RunSpec {
    pub(crate) label: String,
    pub(crate) axis: u8,
    pub(crate) ego_mode: EgoProfileMode,
    pub(crate) ego_max_hops: u8,
    pub(crate) horizon_mode: HorizonAblationMode,
    pub(crate) cond: FairBenchCondition,
    pub(crate) reflect_mode: ReflectMode,
    pub(crate) enable_pre_bypass: bool,
    pub(crate) temperature: f32,
    pub(crate) rng_seed: u64,
}

#[allow(dead_code)]
pub(crate) struct Step2PassResult {
    pub(crate) label: String,
    pub(crate) axis: u8,
    pub(crate) ego_mode: EgoProfileMode,
    pub(crate) ego_max_hops: u8,
    pub(crate) cond: FairBenchCondition,
    pub(crate) reflect_mode: ReflectMode,
    pub(crate) enable_pre_bypass: bool,
    pub(crate) temperature: f32,
    pub(crate) exp_hits: usize,
    pub(crate) exp_total: usize,
    pub(crate) fact_exp_hits: usize,
    pub(crate) fact_exp_total: usize,
    pub(crate) persona_exp_hits: usize,
    pub(crate) persona_exp_total: usize,
    pub(crate) guard_sem_forb_hits: usize,
    pub(crate) guard_forb_total: usize,
    pub(crate) raw_forb_hits: usize,
    pub(crate) sem_forb_hits: usize,
    pub(crate) forb_total: usize,
    pub(crate) ood_bleed_hits: usize,
    pub(crate) ood_total: usize,
    pub(crate) in_tokens: usize,
    pub(crate) decode_tps: f64,
    pub(crate) whole_bypass_pct: f64,
    pub(crate) passive_pct: f32,
    pub(crate) active_pct: f32,
    pub(crate) ego_delta_norm_q7: u8,
    pub(crate) reflect_triggers: usize,
    pub(crate) reflect_adopted: usize,
    pub(crate) reflect_skipped: usize,
    pub(crate) ext_rescues: usize,
    pub(crate) sandbox_promos: u64,
    pub(crate) replies: Vec<String>,
}

pub(crate) fn text_bigram_jaccard(a: &str, b: &str) -> f32 {
    let ta: Vec<String> = a
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_ascii_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect();
    let tb: Vec<String> = b
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_ascii_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect();
    if ta.is_empty() && tb.is_empty() {
        return 1.0;
    }
    if ta.is_empty() || tb.is_empty() {
        return 0.0;
    }
    let set_a: HashSet<(String, String)> = if ta.len() >= 2 {
        ta.windows(2)
            .map(|w| (w[0].clone(), w[1].clone()))
            .collect()
    } else {
        ta.iter().map(|w| (w.clone(), String::new())).collect()
    };
    let set_b: HashSet<(String, String)> = if tb.len() >= 2 {
        tb.windows(2)
            .map(|w| (w[0].clone(), w[1].clone()))
            .collect()
    } else {
        tb.iter().map(|w| (w.clone(), String::new())).collect()
    };
    let inter = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        1.0
    } else {
        inter as f32 / union as f32
    }
}

pub(crate) fn mean_reply_bigram_jaccard(a: &[String], b: &[String]) -> (f32, f32) {
    let n = a.len().min(b.len());
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut sum_j = 0.0_f32;
    let mut exact = 0_usize;
    for (ra, rb) in a.iter().zip(b.iter()) {
        sum_j += text_bigram_jaccard(ra, rb);
        if ra.trim() == rb.trim() {
            exact += 1;
        }
    }
    (sum_j / n as f32, exact as f32 / n as f32 * 100.0)
}

pub(crate) fn build_step2_run_specs(cli: &EvalCliConfig) -> Vec<Step2RunSpec> {
    let ablate_dual_axis = cli.ablate_dual_axis;
    let ablate_ego = cli.ablate_ego;
    let ablate_horizon = cli.ablate_horizon;
    let fair_bench_conditions = &cli.fair_bench_conditions;
    let horizon_mode = cli.horizon_mode;
    let ego_mode = &cli.ego_mode;
    let ego_max_hops = cli.ego_max_hops;
    let reflect_mode = cli.reflect_mode;
    let enable_pre_bypass = cli.enable_pre_bypass;
    let temperature = cli.temperature;
    let rng_seed = cli.rng_seed;
    if ablate_dual_axis {
        vec![
            Step2RunSpec {
                label: "A:Vanilla+FullRAG".to_string(),
                axis: 1,
                ego_mode: EgoProfileMode::Custom,
                ego_max_hops: 1,
                horizon_mode,
                cond: FairBenchCondition::VanillaFullRag,
                reflect_mode: ReflectMode::Off,
                enable_pre_bypass: false,
                temperature: 0.0,
                rng_seed: 42,
            },
            Step2RunSpec {
                label: "B:Vanilla+MinRAG".to_string(),
                axis: 1,
                ego_mode: EgoProfileMode::Custom,
                ego_max_hops: 1,
                horizon_mode,
                cond: FairBenchCondition::VanillaMinRag,
                reflect_mode: ReflectMode::Off,
                enable_pre_bypass: false,
                temperature: 0.0,
                rng_seed: 42,
            },
            Step2RunSpec {
                label: "C:Hybrid(ref=off,byp=off)".to_string(),
                axis: 1,
                ego_mode: EgoProfileMode::Custom,
                ego_max_hops: 1,
                horizon_mode,
                cond: FairBenchCondition::LiminiKaHybrid,
                reflect_mode: ReflectMode::Off,
                enable_pre_bypass: false,
                temperature: 0.0,
                rng_seed: 42,
            },
            Step2RunSpec {
                label: "C:Hybrid(ref=off,byp=on)".to_string(),
                axis: 1,
                ego_mode: EgoProfileMode::Custom,
                ego_max_hops: 1,
                horizon_mode,
                cond: FairBenchCondition::LiminiKaHybrid,
                reflect_mode: ReflectMode::Off,
                enable_pre_bypass: true,
                temperature: 0.0,
                rng_seed: 42,
            },
            Step2RunSpec {
                label: "C:Hybrid(ref=cond,byp=on)".to_string(),
                axis: 1,
                ego_mode: EgoProfileMode::Custom,
                ego_max_hops: 1,
                horizon_mode,
                cond: FairBenchCondition::LiminiKaHybrid,
                reflect_mode: ReflectMode::Conditional,
                enable_pre_bypass: true,
                temperature: 0.0,
                rng_seed: 42,
            },
            Step2RunSpec {
                label: "C:Hybrid(ref=on,byp=on)".to_string(),
                axis: 1,
                ego_mode: EgoProfileMode::Custom,
                ego_max_hops: 1,
                horizon_mode,
                cond: FairBenchCondition::LiminiKaHybrid,
                reflect_mode: ReflectMode::Always,
                enable_pre_bypass: true,
                temperature: 0.0,
                rng_seed: 42,
            },
            Step2RunSpec {
                label: "Ego=off(f_sys)#2".to_string(),
                axis: 2,
                ego_mode: EgoProfileMode::Custom,
                ego_max_hops: 1,
                horizon_mode,
                cond: FairBenchCondition::LiminiKaHybrid,
                reflect_mode: ReflectMode::Off,
                enable_pre_bypass: true,
                temperature: 0.0,
                rng_seed: 42,
            },
            Step2RunSpec {
                label: "Ego=med(1-hop)#1".to_string(),
                axis: 2,
                ego_mode: EgoProfileMode::Preset(LiminalEgoPreset::Medium),
                ego_max_hops: 1,
                horizon_mode,
                cond: FairBenchCondition::LiminiKaHybrid,
                reflect_mode: ReflectMode::Off,
                enable_pre_bypass: true,
                temperature: 0.0,
                rng_seed: 42,
            },
            Step2RunSpec {
                label: "Ego=med(1-hop)#2".to_string(),
                axis: 2,
                ego_mode: EgoProfileMode::Preset(LiminalEgoPreset::Medium),
                ego_max_hops: 1,
                horizon_mode,
                cond: FairBenchCondition::LiminiKaHybrid,
                reflect_mode: ReflectMode::Off,
                enable_pre_bypass: true,
                temperature: 0.0,
                rng_seed: 42,
            },
            Step2RunSpec {
                label: "Ego=med(3-hop)#1".to_string(),
                axis: 2,
                ego_mode: EgoProfileMode::Preset(LiminalEgoPreset::Medium),
                ego_max_hops: 3,
                horizon_mode,
                cond: FairBenchCondition::LiminiKaHybrid,
                reflect_mode: ReflectMode::Off,
                enable_pre_bypass: true,
                temperature: 0.0,
                rng_seed: 42,
            },
            Step2RunSpec {
                label: "Ego=med(3-hop)#2".to_string(),
                axis: 2,
                ego_mode: EgoProfileMode::Preset(LiminalEgoPreset::Medium),
                ego_max_hops: 3,
                horizon_mode,
                cond: FairBenchCondition::LiminiKaHybrid,
                reflect_mode: ReflectMode::Off,
                enable_pre_bypass: true,
                temperature: 0.0,
                rng_seed: 42,
            },
            Step2RunSpec {
                label: "Ego=high(1-hop)".to_string(),
                axis: 2,
                ego_mode: EgoProfileMode::Preset(LiminalEgoPreset::High),
                ego_max_hops: 1,
                horizon_mode,
                cond: FairBenchCondition::LiminiKaHybrid,
                reflect_mode: ReflectMode::Off,
                enable_pre_bypass: true,
                temperature: 0.0,
                rng_seed: 42,
            },
            Step2RunSpec {
                label: "RandCtrl(T=0.8,s42)".to_string(),
                axis: 2,
                ego_mode: EgoProfileMode::Custom,
                ego_max_hops: 1,
                horizon_mode,
                cond: FairBenchCondition::LiminiKaHybrid,
                reflect_mode: ReflectMode::Off,
                enable_pre_bypass: true,
                temperature: 0.8,
                rng_seed: 42,
            },
            Step2RunSpec {
                label: "RandCtrl(T=0.8,s43)".to_string(),
                axis: 2,
                ego_mode: EgoProfileMode::Custom,
                ego_max_hops: 1,
                horizon_mode,
                cond: FairBenchCondition::LiminiKaHybrid,
                reflect_mode: ReflectMode::Off,
                enable_pre_bypass: true,
                temperature: 0.8,
                rng_seed: 43,
            },
            Step2RunSpec {
                label: "D:ZeroCtx(byp=on)".to_string(),
                axis: 1,
                ego_mode: EgoProfileMode::Custom,
                ego_max_hops: 1,
                horizon_mode,
                cond: FairBenchCondition::LiminiKaZeroContext,
                reflect_mode: ReflectMode::Off,
                enable_pre_bypass: true,
                temperature: 0.0,
                rng_seed: 42,
            },
        ]
    } else if let Some(ref conds) = fair_bench_conditions {
        conds
            .iter()
            .map(|&(cond, cond_no_reflect, byp_override)| {
                let run_ref_mode = if cond_no_reflect {
                    ReflectMode::Off
                } else {
                    reflect_mode
                };
                let run_pre_bypass = byp_override.unwrap_or(enable_pre_bypass);
                let lbl = if matches!(cond, FairBenchCondition::LiminiKaHybrid) {
                    format!(
                        "C:Hybrid(ref={},byp={})",
                        run_ref_mode.label(),
                        if run_pre_bypass { "on" } else { "off" }
                    )
                } else {
                    cond.label().to_string()
                };
                Step2RunSpec {
                    label: lbl,
                    axis: 1,
                    ego_mode: ego_mode.clone(),
                    ego_max_hops,
                    horizon_mode,
                    cond,
                    reflect_mode: run_ref_mode,
                    enable_pre_bypass: run_pre_bypass,
                    temperature,
                    rng_seed,
                }
            })
            .collect()
    } else if ablate_horizon {
        vec![
            Step2RunSpec {
                label: format!("{:?}:Core3", ego_mode),
                axis: 0,
                ego_mode: ego_mode.clone(),
                ego_max_hops,
                horizon_mode: HorizonAblationMode::Core3,
                cond: FairBenchCondition::LiminiKaHybrid,
                reflect_mode,
                enable_pre_bypass,
                temperature,
                rng_seed,
            },
            Step2RunSpec {
                label: format!("{:?}:Spillover", ego_mode),
                axis: 0,
                ego_mode: ego_mode.clone(),
                ego_max_hops,
                horizon_mode: HorizonAblationMode::Spillover,
                cond: FairBenchCondition::LiminiKaHybrid,
                reflect_mode,
                enable_pre_bypass,
                temperature,
                rng_seed,
            },
            Step2RunSpec {
                label: format!("{:?}:SlowLoop", ego_mode),
                axis: 0,
                ego_mode: ego_mode.clone(),
                ego_max_hops,
                horizon_mode: HorizonAblationMode::SlowLoop,
                cond: FairBenchCondition::LiminiKaHybrid,
                reflect_mode,
                enable_pre_bypass,
                temperature,
                rng_seed,
            },
        ]
    } else if ablate_ego {
        let active_ego = match &ego_mode {
            EgoProfileMode::Custom => EgoProfileMode::Preset(LiminalEgoPreset::Medium),
            other => (*other).clone(),
        };
        vec![
            Step2RunSpec {
                label: "NoEgo".to_string(),
                axis: 2,
                ego_mode: EgoProfileMode::Custom,
                ego_max_hops,
                horizon_mode,
                cond: FairBenchCondition::LiminiKaHybrid,
                reflect_mode,
                enable_pre_bypass,
                temperature,
                rng_seed,
            },
            Step2RunSpec {
                label: "Ego".to_string(),
                axis: 2,
                ego_mode: active_ego,
                ego_max_hops,
                horizon_mode,
                cond: FairBenchCondition::LiminiKaHybrid,
                reflect_mode,
                enable_pre_bypass,
                temperature,
                rng_seed,
            },
        ]
    } else {
        vec![Step2RunSpec {
            label: horizon_mode.label().to_string(),
            axis: 0,
            ego_mode: ego_mode.clone(),
            ego_max_hops,
            horizon_mode,
            cond: FairBenchCondition::LiminiKaHybrid,
            reflect_mode,
            enable_pre_bypass,
            temperature,
            rng_seed,
        }]
    }
}

use crate::bake::BakedModelProfile;
use crate::model::{Phi3Config, Phi3Model};
use liminika_adapter::ModelSteeringAdapter;
use liminika_core::stigmergy::StandardCognitiveBank;
use tokenizers::Tokenizer;

pub(crate) struct EvalRuntime<'a> {
    pub(crate) cfg: &'a Phi3Config,
    pub(crate) tokenizer: &'a Tokenizer,
    pub(crate) model: &'a mut Phi3Model,
    pub(crate) adapter: &'a mut ModelSteeringAdapter,
    pub(crate) baked_opt: Option<&'a BakedModelProfile>,
    pub(crate) active_persona_bank: &'a mut Option<StandardCognitiveBank>,
}
