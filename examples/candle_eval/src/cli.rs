// name: examples/candle_eval/src/cli.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Command-line configuration parsing, run-spec construction, and text-similarity metrics.

use std::collections::HashSet;
use std::path::PathBuf;

use crate::eval::{
    parse_bench_trials, parse_chat_template, parse_circuit_mode, parse_ego_max_hops,
    parse_ego_mode, parse_fair_bench_conditions, parse_fixture_path, parse_guard_hold_tokens,
    parse_horizon_mode, parse_kv_min_res, parse_layer_mask_spec, parse_logits_observe_stride,
    parse_long_doc_bench_path, parse_max_tokens, parse_model_arch, parse_model_path,
    parse_phase_clamp_rad, parse_prefill_chunk_size, parse_pressure_sim_mode, parse_probe_frame,
    parse_quant_mode, parse_reflect_mode, parse_rng_seed, parse_temperature, parse_tokenizer_path,
    CircuitProfileMode, EgoProfileMode, FairBenchCondition, HorizonAblationMode, LayerMaskSpec,
    PressureSimMode, ReflectMode, DEFAULT_LAYER_STEER_MASK,
};
use crate::model::{
    clip_layer_mask, ChatTemplate, ChatTemplateArg, ModelArchArg, ModelConfig, ProbeCoordinateFrame,
};
use candle_core::quantized::GgmlDType;
use liminika_core::abi::lmnk_q7_t;
use liminika_core::drivers::LiminalEgoPreset;

pub(crate) struct EvalCliConfig {
    pub(crate) model_path: Option<PathBuf>,
    pub(crate) model_arch: ModelArchArg,
    pub(crate) tokenizer_path: Option<PathBuf>,
    pub(crate) chat_template: ChatTemplateArg,
    pub(crate) smoke_test: bool,
    pub(crate) recall_only: bool,
    pub(crate) use_base_prior: bool,
    pub(crate) no_reflect: bool,
    pub(crate) reflect_mode: ReflectMode,
    pub(crate) enable_pre_bypass: bool,
    pub(crate) ablate_dual_axis: bool,
    pub(crate) ego_max_hops: u8,
    pub(crate) temperature: f32,
    pub(crate) rng_seed: u64,
    pub(crate) logits_observe_stride: usize,
    pub(crate) guard_hold_tokens: u8,
    pub(crate) bench_trials: usize,
    pub(crate) fixture_path: Option<PathBuf>,
    pub(crate) multi_turn: bool,
    pub(crate) quant_mode: Option<GgmlDType>,
    pub(crate) probe_frame: ProbeCoordinateFrame,
    pub(crate) layer_mask_spec: Option<LayerMaskSpec>,
    pub(crate) phase_clamp_rad: Option<f32>,
    pub(crate) random_push_control: bool,
    pub(crate) max_tokens: usize,
    pub(crate) prefill_chunk_size: usize,
    pub(crate) long_doc_bench: Option<PathBuf>,
    pub(crate) equilibrium_bench: bool,
    pub(crate) physics_eval: bool,
    pub(crate) extended_eval: bool,
    pub(crate) ja_eval: bool,
    pub(crate) kv_min_res: Option<lmnk_q7_t>,
    pub(crate) circuit_mode: CircuitProfileMode,
    pub(crate) ego_mode: EgoProfileMode,
    pub(crate) ablate_ego: bool,
    pub(crate) horizon_mode: HorizonAblationMode,
    pub(crate) ablate_horizon: bool,
    pub(crate) fair_bench_conditions: Option<Vec<(FairBenchCondition, bool, Option<bool>)>>,
    pub(crate) pressure_sim_mode: PressureSimMode,
    pub(crate) step2_only: bool,
    pub(crate) persona_lmnk_in: Option<PathBuf>,
    pub(crate) persona_lmnk_out: Option<PathBuf>,
}

impl EvalCliConfig {
    pub(crate) fn resolve_layer_mask(&self, num_layers: usize) -> u64 {
        match self.layer_mask_spec {
            Some(spec) => spec.resolve(num_layers),
            None => clip_layer_mask(DEFAULT_LAYER_STEER_MASK, num_layers),
        }
    }

    pub(crate) fn resolve_layer_mask_override(&self, num_layers: usize) -> Option<u64> {
        self.layer_mask_spec.map(|spec| spec.resolve(num_layers))
    }
}

pub(crate) fn wants_help<'a>(args: impl IntoIterator<Item = &'a str>) -> bool {
    args.into_iter().any(|a| a == "--help" || a == "-h")
}

pub(crate) fn print_help() {
    println!("LiminiKa Candle Eval v{}", env!("CARGO_PKG_VERSION"));
    println!("Usage: candle_eval [OPTIONS]");
    println!("Options:");
    println!("  -h, --help                         Print this help message and exit");
    println!("  --model=<path>                     Override model directory (default: models/Phi-4-mini-instruct)");
    println!(
        "  --model-arch=<auto|phi4|mistral3>  Override model architecture loader (default: auto)"
    );
    println!("  --tokenizer=<path>                 Override tokenizer.json path (default: <model>/tokenizer.json)");
    println!(
        "  --chat-template=<auto|phi4|mistral3> Override chat template formatting (default: auto)"
    );
    println!(
        "  --smoke-test                       Run Steer-OFF greedy generation smoke test and exit"
    );
    println!("  --fixture=<path>                   Override evaluation fixture file (default: tests/fixtures/gcso_eval_samples.txt)");
    println!("  --bench-trials=<N>                 Run N independent benchmark trials after warmup and report median/spread (default: 1)");
    println!("  --step2-only                       Skip Step 1 sweeps and run Step 2 (+ Step 3) directly");
    println!("  --recall-only                      Run recall/timeline evaluation without LLM weight inference");
    println!("  --base-prior                       Enable 1-pass neutral base-model prior calibration (default: on)");
    println!("  --no-base-prior                    Disable startup base-model prior calibration (A/B opt-out)");
    println!("  --no-reflect                       Disable reflection critique retries");
    println!("  --reflect-mode=<off|conditional|always>");
    println!("  --no-pre-bypass                    Disable whole-token pre-inference bypass");
    println!("  --logits-observe-stride=<N>        Dynamic GPU->CPU logits observe max stride (default: 1, clamp 0->1)");
    println!("  --guard-hold-tokens=<N>            Transient contradiction-guard hold window before relaxation (default: 0 = full turn)");
    println!("  --multi-turn                       Run 8-turn persistent KV dialogue evaluation (Step 2B)");
    println!(
        "  --quant=<bf16|q4_0|q4k|q8_0>       Model quantization mode (default: bf16 on CUDA)"
    );
    println!(
        "  --probe-frame=<pre-rope|post-rope> Procrustes probe coordinate frame (default: pre-rope)"
    );
    println!(
        "  --layer-mask=<preset|0xHEX>        RoPE steering layer window: top16, top-half, mid12, mid16, upper-mid12, all, or 0xHEX (default: top16)"
    );
    println!(
        "  --phase-clamp-rad=<f32>            Override eval-harness PhaseClamp arc in (0, pi/2] rad (default: core GatewayFrame default)"
    );
    println!(
        "  --random-push-control              Inject norm-matched deterministic random Q7 phase perturbation through the closed-loop pipeline"
    );
    println!(
        "  --max-tokens=<N>                   Maximum generated decode tokens per turn in [1, 512] (default: 64)"
    );
    println!(
        "  --prefill-chunk-size=<N>           Eval-only chunked prefill width in [16, 4096] (default: 256)"
    );
    println!(
        "  --long-doc-bench=<path>            Run streaming distillation, anchor paging, and chunked prefill bench"
    );
    println!(
        "  --equilibrium-bench                Run cross-restart equilibrium & .lmnkp immutability E2E bench"
    );
    println!(
        "  --physics-eval                     Run the 4-axis deterministic steering-physics evaluation suite in a single model load"
    );
    println!(
        "  --extended-eval                    Run extended English boundary (54) and guard (24) probes in an isolated state"
    );
    println!(
        "  --ja-eval                          Run fictional Japanese multilingual corpus and 28-probe suite in an isolated state"
    );
    println!(
        "  --kv-min-res=<i8>                  Override RAM Phase-KV minimum resonance Q7 threshold"
    );
    println!("  --circuit=<analytical|integrative|hybrid[:N]>");
    println!("  --ego=<off|low|medium|high|desc:...>");
    println!("  --ego-max-hops=<1..4>              Maximum Ego P-Frame hop count (default: 1)");
    println!("  --temp=<f32>                       Sampling temperature (default: 0.0 greedy)");
    println!(
        "  --seed=<u64>                       PRNG seed for non-greedy sampling (default: 42)"
    );
    println!(
        "  --ablate-dual-axis                 Run full Dual-Axis SystemSpine + Ego ablation matrix"
    );
    println!("  --ablate-ego                       Run NoEgo vs Ego ablation");
    println!("  --ablate-horizon                   Run Core3 vs Spillover vs SlowLoop ablation");
    println!("  --horizon-mode=<core3|spillover|slow-loop>");
    println!("  --fair-bench                       Run all Fair Benchmark conditions (A, B, C-off, C-on, D)");
    println!("  --bench-cond=<list>                Comma-separated conditions: a,b,c,c-byp-off,c-byp-on,c-reflect,d,all");
    println!("  --pressure-sim=<off|elevated|surge>");
    println!(
        "  --persona-lmnk=<path>              Load persona cognitive circuit from .lmnkc/.lmnkp"
    );
    println!(
        "  --save-persona-lmnk=<path>         Save updated persona cognitive circuit to .lmnkc"
    );
}

pub(crate) fn find_unrecognized_args<'a>(args: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    const BOOL_FLAGS: &[&str] = &[
        "--help",
        "-h",
        "--smoke-test",
        "--recall-only",
        "--base-prior",
        "--no-base-prior",
        "--no-reflect",
        "--no-pre-bypass",
        "--ablate-dual-axis",
        "--multi-turn",
        "--ablate-ego",
        "--ablate-horizon",
        "--fair-bench",
        "--step2-only",
        "--random-push-control",
        "--equilibrium-bench",
        "--physics-eval",
        "--extended-eval",
        "--ja-eval",
    ];
    const PREFIX_FLAGS: &[&str] = &[
        "--model=",
        "--model-arch=",
        "--tokenizer=",
        "--chat-template=",
        "--reflect-mode=",
        "--ego-max-hops=",
        "--temp=",
        "--seed=",
        "--logits-observe-stride=",
        "--guard-hold-tokens=",
        "--bench-trials=",
        "--fixture=",
        "--quant=",
        "--probe-frame=",
        "--layer-mask=",
        "--phase-clamp-rad=",
        "--max-tokens=",
        "--prefill-chunk-size=",
        "--long-doc-bench=",
        "--kv-min-res=",
        "--circuit=",
        "--ego=",
        "--horizon-mode=",
        "--bench-cond=",
        "--pressure-sim=",
        "--persona-lmnk=",
        "--save-persona-lmnk=",
    ];
    args.into_iter()
        .filter(|arg| {
            !BOOL_FLAGS.contains(arg) && !PREFIX_FLAGS.iter().any(|pref| arg.starts_with(pref))
        })
        .map(ToString::to_string)
        .collect()
}

impl EvalCliConfig {
    pub(crate) fn parse_from_args(cli_args: &[String]) -> Self {
        let model_path = parse_model_path(cli_args.iter().map(String::as_str));
        let model_arch = parse_model_arch(cli_args.iter().map(String::as_str));
        let tokenizer_path = parse_tokenizer_path(cli_args.iter().map(String::as_str));
        let chat_template = parse_chat_template(cli_args.iter().map(String::as_str));
        let smoke_test = cli_args.iter().any(|a| a == "--smoke-test");
        let recall_only = cli_args.iter().any(|a| a == "--recall-only");
        let mut use_base_prior = true;
        for arg in cli_args {
            if arg == "--no-base-prior" {
                use_base_prior = false;
            } else if arg == "--base-prior" {
                use_base_prior = true;
            }
        }
        let no_reflect = cli_args.iter().any(|a| a == "--no-reflect");
        let reflect_mode = parse_reflect_mode(cli_args.iter().map(String::as_str));
        let enable_pre_bypass = !cli_args.iter().any(|a| a == "--no-pre-bypass");
        let ablate_dual_axis = cli_args.iter().any(|a| a == "--ablate-dual-axis");
        let ego_max_hops = parse_ego_max_hops(cli_args.iter().map(String::as_str));
        let temperature = parse_temperature(cli_args.iter().map(String::as_str));
        let rng_seed = parse_rng_seed(cli_args.iter().map(String::as_str));
        let logits_observe_stride =
            parse_logits_observe_stride(cli_args.iter().map(String::as_str));
        let guard_hold_tokens = parse_guard_hold_tokens(cli_args.iter().map(String::as_str));
        let bench_trials = parse_bench_trials(cli_args.iter().map(String::as_str));
        let fixture_path = parse_fixture_path(cli_args.iter().map(String::as_str));
        let multi_turn = cli_args.iter().any(|a| a == "--multi-turn");
        let quant_mode = parse_quant_mode(cli_args.iter().map(String::as_str));
        let probe_frame = parse_probe_frame(cli_args.iter().map(String::as_str));
        let layer_mask_spec = parse_layer_mask_spec(cli_args.iter().map(String::as_str));
        let phase_clamp_rad = parse_phase_clamp_rad(cli_args.iter().map(String::as_str));
        let random_push_control = cli_args.iter().any(|a| a == "--random-push-control");
        let max_tokens = parse_max_tokens(cli_args.iter().map(String::as_str));
        let prefill_chunk_size = parse_prefill_chunk_size(cli_args.iter().map(String::as_str));
        let long_doc_bench = parse_long_doc_bench_path(cli_args.iter().map(String::as_str));
        let equilibrium_bench = cli_args.iter().any(|a| a == "--equilibrium-bench");
        let physics_eval = cli_args.iter().any(|a| a == "--physics-eval");
        let extended_eval = cli_args.iter().any(|a| a == "--extended-eval");
        let ja_eval = cli_args.iter().any(|a| a == "--ja-eval");
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
        let persona_lmnk_in = cli_args
            .iter()
            .find_map(|a| a.strip_prefix("--persona-lmnk="))
            .map(PathBuf::from);
        let persona_lmnk_out = cli_args
            .iter()
            .find_map(|a| a.strip_prefix("--save-persona-lmnk="))
            .map(PathBuf::from)
            .or_else(|| persona_lmnk_in.clone());

        Self {
            model_path,
            model_arch,
            tokenizer_path,
            chat_template,
            smoke_test,
            recall_only,
            use_base_prior,
            no_reflect,
            reflect_mode,
            enable_pre_bypass,
            ablate_dual_axis,
            ego_max_hops,
            temperature,
            rng_seed,
            logits_observe_stride,
            guard_hold_tokens,
            bench_trials,
            fixture_path,
            multi_turn,
            quant_mode,
            probe_frame,
            layer_mask_spec,
            phase_clamp_rad,
            random_push_control,
            max_tokens,
            prefill_chunk_size,
            long_doc_bench,
            equilibrium_bench,
            physics_eval,
            extended_eval,
            ja_eval,
            kv_min_res,
            circuit_mode,
            ego_mode,
            ablate_ego,
            horizon_mode,
            ablate_horizon,
            fair_bench_conditions,
            pressure_sim_mode,
            step2_only,
            persona_lmnk_in,
            persona_lmnk_out,
        }
    }

    pub(crate) fn parse_from_env() -> Self {
        let cli_args: Vec<String> = std::env::args().collect();
        for unknown in find_unrecognized_args(cli_args.iter().skip(1).map(String::as_str)) {
            eprintln!("Warning: unrecognized argument: {unknown}");
        }
        Self::parse_from_args(&cli_args)
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

pub(crate) struct Step2PassResult {
    pub(crate) label: String,
    pub(crate) axis: u8,
    pub(crate) ego_max_hops: u8,
    pub(crate) cond: FairBenchCondition,
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
    pub(crate) ego_delta_norm_q7: u8,
    pub(crate) reflect_skipped: usize,
    pub(crate) ext_rescues: usize,
    pub(crate) replies: Vec<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct TrialAggregateStats {
    pub(crate) median: f64,
    pub(crate) min: f64,
    pub(crate) max: f64,
    pub(crate) spread_pct: f64,
}

pub(crate) fn compute_trial_stats(values: &[f64]) -> TrialAggregateStats {
    if values.is_empty() {
        return TrialAggregateStats::default();
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let mid = sorted.len() / 2;
    let median = if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    };
    let spread_pct = if median > 1e-12 {
        (max - min) / median * 100.0
    } else {
        0.0
    };
    TrialAggregateStats {
        median,
        min,
        max,
        spread_pct,
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TrialPassMetrics {
    pub(crate) label: String,
    pub(crate) pure_decode_tps: f64,
    pub(crate) e2e_tps: f64,
    pub(crate) prefill_total_ms: f64,
    pub(crate) prefill_mean_ms: f64,
    pub(crate) exp_hits: usize,
    pub(crate) exp_total: usize,
    pub(crate) sem_forb_hits: usize,
    pub(crate) forb_total: usize,
    pub(crate) ood_bleed_hits: usize,
    pub(crate) ood_total: usize,
    pub(crate) whole_bypass_hits: usize,
    pub(crate) decode_tokens_steered: usize,
}

pub(crate) fn print_bench_trials_summary(trials: &[Vec<TrialPassMetrics>]) {
    if trials.is_empty() || trials[0].is_empty() {
        return;
    }
    let n_trials = trials.len();
    println!("\n=== [Bench Trials Summary (N={n_trials})] ===");
    let mut b_med_pure: Option<f64> = None;
    let mut b_med_e2e: Option<f64> = None;
    let mut c_rows: Vec<(String, f64, f64)> = Vec::new();

    for (cond_idx, first) in trials[0].iter().enumerate() {
        let mut pure_vals = Vec::with_capacity(n_trials);
        let mut e2e_vals = Vec::with_capacity(n_trials);
        let mut pf_tot_vals = Vec::with_capacity(n_trials);
        let mut pf_mean_vals = Vec::with_capacity(n_trials);
        let mut det_ok = true;
        for trial in trials {
            if let Some(m) = trial.get(cond_idx) {
                pure_vals.push(m.pure_decode_tps);
                e2e_vals.push(m.e2e_tps);
                pf_tot_vals.push(m.prefill_total_ms);
                pf_mean_vals.push(m.prefill_mean_ms);
                if (
                    m.exp_hits,
                    m.exp_total,
                    m.sem_forb_hits,
                    m.forb_total,
                    m.ood_bleed_hits,
                    m.ood_total,
                    m.whole_bypass_hits,
                    m.decode_tokens_steered,
                ) != (
                    first.exp_hits,
                    first.exp_total,
                    first.sem_forb_hits,
                    first.forb_total,
                    first.ood_bleed_hits,
                    first.ood_total,
                    first.whole_bypass_hits,
                    first.decode_tokens_steered,
                ) {
                    det_ok = false;
                }
            }
        }
        let pure_st = compute_trial_stats(&pure_vals);
        let e2e_st = compute_trial_stats(&e2e_vals);
        let pf_tot_st = compute_trial_stats(&pf_tot_vals);
        let pf_mean_st = compute_trial_stats(&pf_mean_vals);
        let exp_pct = if first.exp_total == 0 {
            0.0
        } else {
            first.exp_hits as f64 / first.exp_total as f64 * 100.0
        };
        let forb_pct = if first.forb_total == 0 {
            0.0
        } else {
            first.sem_forb_hits as f64 / first.forb_total as f64 * 100.0
        };
        let ood_pct = if first.ood_total == 0 {
            0.0
        } else {
            first.ood_bleed_hits as f64 / first.ood_total as f64 * 100.0
        };
        let byp_pct = if first.decode_tokens_steered == 0 {
            0.0
        } else {
            first.whole_bypass_hits as f64 / first.decode_tokens_steered as f64 * 100.0
        };
        println!(
            "  {:<26} | pure_med={:6.2} [{:6.2}..{:6.2}] ({:4.1}%) | e2e_med={:6.2} [{:6.2}..{:6.2}] ({:4.1}%) | prefill_med={:6.1}ms ({:5.2}ms/req) | hit={:5.1}% sem_forb={:4.1}% ood={:4.1}% byp={:4.1}% det={}",
            first.label,
            pure_st.median,
            pure_st.min,
            pure_st.max,
            pure_st.spread_pct,
            e2e_st.median,
            e2e_st.min,
            e2e_st.max,
            e2e_st.spread_pct,
            pf_tot_st.median,
            pf_mean_st.median,
            exp_pct,
            forb_pct,
            ood_pct,
            byp_pct,
            det_ok,
        );
        if first.label.starts_with("B:Vanilla") {
            b_med_pure = Some(pure_st.median);
            b_med_e2e = Some(e2e_st.median);
        } else if first.label.starts_with("C:Hybrid") {
            c_rows.push((first.label.clone(), pure_st.median, e2e_st.median));
        }
    }

    if let (Some(b_pure), Some(b_e2e)) = (b_med_pure, b_med_e2e) {
        for (lbl, c_pure, c_e2e) in c_rows {
            if b_pure > 1e-6 && b_e2e > 1e-6 {
                println!(
                    "  [Calibrated C/B Median] {lbl} vs B:Vanilla+MinRAG | pure_decode={:.3}x ({:+.2}%) | e2e={:.3}x ({:+.2}%)",
                    c_pure / b_pure,
                    (c_pure - b_pure) / b_pure * 100.0,
                    c_e2e / b_e2e,
                    (c_e2e - b_e2e) / b_e2e * 100.0,
                );
            }
        }
    }
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
use crate::model::CausalLmModel;
use liminika_adapter::{CompiledTokenCorpus, ModelSteeringAdapter};
use liminika_core::stigmergy::StandardCognitiveBank;
use tokenizers::Tokenizer;

pub(crate) struct EvalRuntime<'a> {
    pub(crate) cfg: &'a ModelConfig,
    pub(crate) chat: &'a ChatTemplate,
    pub(crate) active_layer_mask: u64,
    pub(crate) tokenizer: &'a Tokenizer,
    pub(crate) model: &'a mut CausalLmModel,
    pub(crate) adapter: &'a mut ModelSteeringAdapter,
    pub(crate) baked_opt: Option<&'a BakedModelProfile>,
    pub(crate) compiled: &'a CompiledTokenCorpus,
    pub(crate) compile_ms: f64,
    pub(crate) active_persona_bank: &'a mut Option<StandardCognitiveBank>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_help_fixture_bench_trials_and_unknown_flag_detection() {
        assert!(wants_help(["--help"]));
        assert!(wants_help(["--quant=q4k", "-h"]));
        assert!(!wants_help(["--step2-only", "--multi-turn"]));

        let args = vec![
            "candle_eval".to_string(),
            "--step2-only".to_string(),
            "--bench-trials=0".to_string(),
            "--fixture=tests/fixtures/custom.txt".to_string(),
            "--quant=q4k".to_string(),
            "--probe-frame=post-rope".to_string(),
            "--layer-mask=mid16".to_string(),
            "--phase-clamp-rad=0.24".to_string(),
            "--random-push-control".to_string(),
            "--max-tokens=128".to_string(),
            "--prefill-chunk-size=128".to_string(),
            "--long-doc-bench=docs/paper/en/GCSO.md".to_string(),
            "--physics-eval".to_string(),
            "--extended-eval".to_string(),
            "--ja-eval".to_string(),
        ];
        assert!(find_unrecognized_args(args.iter().skip(1).map(String::as_str)).is_empty());
        let parsed = EvalCliConfig::parse_from_args(&args);
        assert_eq!(parsed.bench_trials, 1);
        assert_eq!(parsed.probe_frame, ProbeCoordinateFrame::PostRope);
        assert_eq!(parsed.resolve_layer_mask(32), 0x03FF_FC00);
        assert_eq!(parsed.phase_clamp_rad, Some(0.24));
        assert!(parsed.random_push_control);
        assert_eq!(parsed.max_tokens, 128);
        assert_eq!(parsed.prefill_chunk_size, 128);
        assert_eq!(
            parsed.long_doc_bench,
            Some(PathBuf::from("docs/paper/en/GCSO.md"))
        );
        assert!(parsed.physics_eval);
        assert!(parsed.extended_eval);
        assert!(parsed.ja_eval);
        assert_eq!(
            parsed.fixture_path,
            Some(PathBuf::from("tests/fixtures/custom.txt"))
        );

        let args_m = vec![
            "candle_eval".to_string(),
            "--model=models/Ministral-3-3B-Instruct-2512-BF16".to_string(),
            "--model-arch=mistral3".to_string(),
            "--tokenizer=models/Ministral-3-3B-Instruct-2512-BF16/tokenizer.json".to_string(),
            "--chat-template=mistral3".to_string(),
            "--layer-mask=top-half".to_string(),
            "--smoke-test".to_string(),
        ];
        assert!(find_unrecognized_args(args_m.iter().skip(1).map(String::as_str)).is_empty());
        let parsed_m = EvalCliConfig::parse_from_args(&args_m);
        assert_eq!(parsed_m.model_arch, ModelArchArg::Mistral3);
        assert_eq!(parsed_m.chat_template, ChatTemplateArg::Mistral3);
        assert!(parsed_m.smoke_test);
        assert_eq!(parsed_m.resolve_layer_mask(32), 0xFFFF_0000);
        assert_eq!(parsed_m.resolve_layer_mask(26), 0x03FF_E000);

        let args_3 = vec!["candle_eval".to_string(), "--bench-trials=3".to_string()];
        assert_eq!(EvalCliConfig::parse_from_args(&args_3).bench_trials, 3);
        let defaults = EvalCliConfig::parse_from_args(&["candle_eval".to_string()]);
        assert_eq!(defaults.bench_trials, 1);
        assert!(defaults.use_base_prior);
        assert_eq!(defaults.probe_frame, ProbeCoordinateFrame::PreRope);
        assert_eq!(defaults.layer_mask_spec, None);
        assert_eq!(defaults.resolve_layer_mask(32), 0xFFFF_0000);
        assert_eq!(defaults.resolve_layer_mask(26), 0x03FF_0000);
        assert_eq!(defaults.phase_clamp_rad, None);
        assert!(!defaults.random_push_control);
        assert_eq!(defaults.max_tokens, 64);
        assert!(!defaults.physics_eval);
        assert!(!defaults.extended_eval);
        assert!(!defaults.ja_eval);

        assert!(find_unrecognized_args(["--no-base-prior", "--base-prior"]).is_empty());
        let no_prior = EvalCliConfig::parse_from_args(&[
            "candle_eval".to_string(),
            "--no-base-prior".to_string(),
        ]);
        assert!(!no_prior.use_base_prior);
        let re_enabled = EvalCliConfig::parse_from_args(&[
            "candle_eval".to_string(),
            "--no-base-prior".to_string(),
            "--base-prior".to_string(),
        ]);
        assert!(re_enabled.use_base_prior);

        let bad = find_unrecognized_args([
            "--step2-only",
            "--unknown-flag",
            "--bench-trials=3",
            "stray_positional",
        ]);
        assert_eq!(
            bad,
            vec!["--unknown-flag".to_string(), "stray_positional".to_string()]
        );
    }

    #[test]
    fn trial_aggregate_stats_computes_median_min_max_and_spread() {
        let single = compute_trial_stats(&[42.0]);
        assert!((single.median - 42.0).abs() < 1e-9);
        assert!((single.min - 42.0).abs() < 1e-9);
        assert!((single.max - 42.0).abs() < 1e-9);
        assert!((single.spread_pct - 0.0).abs() < 1e-9);

        let odd = compute_trial_stats(&[55.0, 40.0, 50.0]);
        assert!((odd.median - 50.0).abs() < 1e-9);
        assert!((odd.min - 40.0).abs() < 1e-9);
        assert!((odd.max - 55.0).abs() < 1e-9);
        assert!((odd.spread_pct - 30.0).abs() < 1e-9);

        let even = compute_trial_stats(&[30.0, 50.0, 42.0, 38.0]);
        assert!((even.median - 40.0).abs() < 1e-9);
        assert!((even.min - 30.0).abs() < 1e-9);
        assert!((even.max - 50.0).abs() < 1e-9);
        assert!((even.spread_pct - 50.0).abs() < 1e-9);
    }
}
