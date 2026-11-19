// name: examples/candle_eval/src/main.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Thin Candle + Safetensors evaluation harness delegating all RoPE geometry mapping,
//! Q7 phase baking, and steering kernels to `liminika-adapter` (and `liminika-core`).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device};
use liminika_adapter::{
    AdaptiveGopCutDetector, AttractorContainerCodec, AttractorRole, AttractorTrajectoryEvaluator,
    CompiledTokenCorpus, CorpusRecallEscalationProvider, FrameTransitionMetrics,
    GcsoDualTrackTimeline, ManifoldReflectionEvaluator, ModelSteeringAdapter, RebakeContext,
    ReflectionCritique, ReflectionGateInput, ReflectionPolicy, ReflectionSelection, RopeGeometry,
    SequenceActivationPooler, SourceTextRecord, SubtitleActivationEncoder, TimelineResumeOutcome,
    TimelineStepContext, TokenSequenceEntry, TrajectoryManifoldSnapshot, VoidBoundaryState,
};
use liminika_core::abi::gcso_pointer_trail_t;
use liminika_core::drivers::LiminalEgoPreset;
use liminika_core::runtime::{
    AdaptiveThresholdPolicy, CognitiveCircuitThresholdPolicy, GuardPhaseAnchor, HybridScaleCircuit,
    ObjectiveFact, RamPhaseKvCacheOps, StandardPhaseKvGopCache, SubjectivePersona,
};
use liminika_core::stigmergy::{
    CognitiveCircuitBankOps, CognitiveCircuitPlasticity, StandardCognitiveBank,
    StigmergicActionDecision,
};
use tokenizers::Tokenizer;

type LayerResonantSlices<'a> = (
    &'a [ObjectiveFact<[i8; 64]>],
    &'a [SubjectivePersona<[i8; 64]>],
    &'a [GuardPhaseAnchor],
    &'a [f32],
    &'a [f32],
);

#[derive(Clone)]
struct Step2RunSpec {
    label: String,
    axis: u8,
    ego_mode: EgoProfileMode,
    ego_max_hops: u8,
    horizon_mode: HorizonAblationMode,
    cond: FairBenchCondition,
    reflect_mode: ReflectMode,
    enable_pre_bypass: bool,
    temperature: f32,
    rng_seed: u64,
}

#[allow(dead_code)]
struct Step2PassResult {
    label: String,
    axis: u8,
    ego_mode: EgoProfileMode,
    ego_max_hops: u8,
    cond: FairBenchCondition,
    reflect_mode: ReflectMode,
    enable_pre_bypass: bool,
    temperature: f32,
    exp_hits: usize,
    exp_total: usize,
    fact_exp_hits: usize,
    fact_exp_total: usize,
    persona_exp_hits: usize,
    persona_exp_total: usize,
    guard_sem_forb_hits: usize,
    guard_forb_total: usize,
    raw_forb_hits: usize,
    sem_forb_hits: usize,
    forb_total: usize,
    ood_bleed_hits: usize,
    ood_total: usize,
    in_tokens: usize,
    decode_tps: f64,
    whole_bypass_pct: f64,
    passive_pct: f32,
    active_pct: f32,
    ego_delta_norm_q7: u8,
    reflect_triggers: usize,
    reflect_adopted: usize,
    reflect_skipped: usize,
    ext_rescues: usize,
    sandbox_promos: u64,
    replies: Vec<String>,
}

fn text_bigram_jaccard(a: &str, b: &str) -> f32 {
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

fn mean_reply_bigram_jaccard(a: &[String], b: &[String]) -> (f32, f32) {
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

mod bake;
mod eval;
mod model;

use bake::{
    argmax, bake_via_adapter, format_phi4_prompt, kl_div, load_fixture, print_runtime_profile,
    query_nvidia_smi_vram, sample_startup_base_prior, token_logprob, BakedModelProfile,
};
use eval::{
    apply_ego_mode, build_preset_circuit, count_semantic_forbid_hits, critique_generated_draft,
    generate_steered_reply, generate_steered_reply_with_offset, parse_circuit_mode,
    parse_ego_max_hops, parse_ego_mode, parse_fair_bench_conditions, parse_horizon_mode,
    parse_kv_min_res, parse_pressure_sim_mode, parse_quant_mode, parse_reflect_mode,
    parse_rng_seed, parse_temperature, resolve_persona_circuit, CircuitProfileMode, EgoProfileMode,
    FairBenchCondition, HorizonAblationMode, KvTraceStats, PressureSimMode, ReflectMode,
    SteerDecodeConfig, StepTelemetryTotals,
};
use model::{ForwardSteer, Phi3Config, Phi3Model, ProbeTrace, ResonantAnchors};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let models_root = repo_root.join("models");
    if !models_root.exists() {
        std::fs::create_dir_all(&models_root)?;
    }
    let model_dir = models_root.join("Phi-4-mini-instruct");
    if !model_dir.is_dir() {
        return Err(format!(
            "model directory '{}' not found (base '{}' is ready; place Phi-4-mini-instruct weights there)",
            model_dir.display(),
            models_root.display()
        )
        .into());
    }
    let fixture_path = repo_root.join("tests/fixtures/gcso_eval_samples.txt");

    let cfg: Phi3Config =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json"))?)?;
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| format!("tokenizer load error: {e}"))?;
    let suite = load_fixture(&fixture_path)?;

    let mut adapter = ModelSteeringAdapter::new(cfg.to_rope_geometry())?;

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
    adapter.set_ego_max_hops(ego_max_hops);
    apply_ego_mode(&mut adapter, &ego_mode);
    if adapter.ego_field().is_active() {
        println!(
            "[Ego   ] active: thrust={} escape_floor={} transgress={} defiance={} trigger=0x{:02X} hash=0x{:08X}",
            adapter.ego_field().counter_basin_thrust_q7,
            adapter.ego_field().basin_escape_floor_q7,
            adapter.ego_field().transgression_bias_q7,
            adapter.ego_field().irrational_defiance_q7,
            adapter.ego_field().trigger_mask,
            adapter.ego_field().ego_signature_hash,
        );
    }

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let active_dtype = if quant_mode.is_some() || !device.is_cuda() {
        DType::F32
    } else {
        DType::BF16
    };
    let vram_before = query_nvidia_smi_vram();
    println!(
        "LiminiKa Candle Eval (RoPE + EDBC + Rank-1 residual via liminika-adapter) | device={:?} | layers={} heads={}/{} head_dim={} rotary_dim={}",
        device,
        adapter.geometry().num_layers,
        adapter.geometry().num_q_heads,
        adapter.geometry().num_kv_heads,
        adapter.geometry().head_dim,
        adapter.geometry().rotary_dim
    );

    let t0 = Instant::now();
    let mut model = Phi3Model::load(&model_dir, &cfg, &device, quant_mode)?;
    let load_elapsed = t0.elapsed();
    println!("Loaded Phi-4-mini-instruct in {:.2?}", load_elapsed);
    print_runtime_profile(
        &cfg,
        &adapter,
        &device,
        active_dtype,
        quant_mode,
        load_elapsed,
        vram_before.as_deref(),
    );
    let persona_gcso_in = cli_args
        .iter()
        .find_map(|a| a.strip_prefix("--persona-gcso="))
        .map(PathBuf::from);
    let persona_gcso_out = cli_args
        .iter()
        .find_map(|a| a.strip_prefix("--save-persona-gcso="))
        .map(PathBuf::from)
        .or_else(|| persona_gcso_in.clone());
    let active_persona_circuit = resolve_persona_circuit(circuit_mode, persona_gcso_in.as_deref());
    let mut active_persona_bank: Option<StandardCognitiveBank> = active_persona_circuit.map(|pc| {
        if matches!(circuit_mode, CircuitProfileMode::Hybrid(_)) {
            let micro = build_preset_circuit(CircuitProfileMode::Analytical).unwrap_or(pc);
            let macro_c = build_preset_circuit(CircuitProfileMode::Integrative).unwrap_or(pc);
            let hybrid = HybridScaleCircuit::new(micro, macro_c);
            let mut bank = StandardCognitiveBank::from_ledger(hybrid.to_blended_ledger(64));
            bank.units[1] = macro_c;
            bank.units[2] = micro;
            bank
        } else {
            StandardCognitiveBank::from_ledger(pc)
        }
    });
    if let Some(ref bank) = active_persona_bank {
        let cons = bank.to_consolidated_ledger();
        println!(
            "Persona Cognitive Bank active (RAM 256B / 256 steps): mode={:?} consolidated={:02x?}",
            circuit_mode, cons.steps
        );
    }
    let baked_opt: Option<BakedModelProfile> = if !recall_only {
        let t_bake = Instant::now();
        let baked = bake_via_adapter(&mut model, &tokenizer, &adapter, cfg.hidden_size, &suite)?;
        println!(
            "Baked {} x {} Q7 phase deltas + {}-dim Rank-1 residual vectors via ModelSteeringAdapter in {:.2?}.",
            baked.per_layer_q7.len(),
            adapter.geometry().num_q_heads,
            cfg.hidden_size,
            t_bake.elapsed(),
        );

        let mut target_toks: Vec<Option<u32>> = Vec::with_capacity(suite.probes.len());
        for (_, _, expected, _) in &suite.probes {
            let tok = expected.first().and_then(|kw| {
                tokenizer
                    .encode(kw.as_str(), false)
                    .ok()
                    .and_then(|enc| enc.get_ids().first().copied())
            });
            target_toks.push(tok);
        }

        let mut baselines: Vec<Vec<f32>> = Vec::with_capacity(suite.probes.len());
        let mut base_entropies: Vec<f32> = Vec::with_capacity(suite.probes.len());
        for (_, prompt, _, _) in &suite.probes {
            model.clear_cache();
            let formatted = format_phi4_prompt(prompt);
            let ids = tokenizer
                .encode(formatted.as_str(), false)
                .map_err(|e| format!("encode error: {e}"))?
                .get_ids()
                .to_vec();
            let (logits, _, _) = model.forward_tokens(&ids, 0, None, false, false)?;
            let h_t = adapter.measure_edbc_entropy(&logits)?.moving_z_entropy;
            base_entropies.push(h_t);
            baselines.push(logits);
        }

        adapter.set_layer_rope_mask(0xFFFF_0000);
        let pos_slices: Vec<&[f32]> = baked.pos_sum.iter().map(Vec::as_slice).collect();
        let neg_slices: Vec<&[f32]> = baked.neg_sum.iter().map(Vec::as_slice).collect();
        let probe_res =
            adapter.calibrate_manifold_probe(&pos_slices, &neg_slices, &base_entropies)?;
        println!(
            "[EnvProbe epoch={}] heads={} steered_layers={} coh={:.3} H0={:.3} => dyn_gain={:.3} skip_H={:.3} srl_rho={:.3}",
            probe_res.probe_epoch,
            probe_res.active_heads,
            probe_res.steered_layers,
            probe_res.baseline_inter_layer_coherence,
            probe_res.baseline_entropy,
            adapter.action_hub().thresholds.reinforcement_gain,
            adapter.action_hub().thresholds.inertial_skip_entropy_thresh,
            adapter.action_hub().thresholds.rank1_inject_density_thresh,
        );

        if !step2_only {
            // Compare manual max_rad vs Gateway structural hyperparameters vs Resonant slice vs ActionHub.
            // tuple: (label, layer_mask, rope_rad, srl_scale, use_two_origin, use_resonant_slice)
            let configs: [(&str, u64, f32, f32, bool, bool); 6] = [
                ("Baseline", 0xFFFF_FFFF, 0.00, 0.00, false, false),
                ("RoPE (Top16, 0.10)", 0xFFFF_0000, 0.10, 0.00, false, false),
                ("Top16 + GW(Fact)", 0xFFFF_0000, -1.00, 0.00, false, false),
                (
                    "Top16 + GW(Fact+Rank-1 residual)",
                    0xFFFF_0000,
                    -1.00,
                    -1.00,
                    false,
                    false,
                ),
                (
                    "Top16 + GW(Resonant)",
                    0xFFFF_0000,
                    -1.00,
                    -1.00,
                    false,
                    true,
                ),
                ("Top16 + ActionHub", 0xFFFF_0000, -2.00, -1.00, false, true),
            ];

            println!(
                "\n{:>20} | {:>10} | {:>12} | {:>11} | {:>11}",
                "mode", "mean_KL", "d_logprob", "expect_hit", "forbid_hit"
            );

            for (label, layer_mask, rope_rad, srl_scale, use_two_origin, use_resonant) in configs {
                adapter.set_layer_rope_mask(layer_mask);
                adapter.reset_action_hub_stats();
                let use_edbc = rope_rad <= -1.5;
                let mut sum_kl = 0.0_f32;
                let mut sum_d_lp = 0.0_f32;
                let mut d_lp_count = 0_usize;
                let mut exp_hits = 0_usize;
                let mut exp_total = 0_usize;
                let mut forb_hits = 0_usize;
                let mut forb_total = 0_usize;
                let mut p1_sample = String::new();
                let mut probe_traces: Vec<ProbeTrace> = Vec::new();

                for (p_idx, ((slot, prompt, expected, forbidden), base_logits)) in
                    suite.probes.iter().zip(baselines.iter()).enumerate()
                {
                    model.clear_cache();
                    let counts_before = *adapter.action_hub_decision_counts();
                    let formatted = format_phi4_prompt(prompt);
                    let ids = tokenizer
                        .encode(formatted.as_str(), false)
                        .map_err(|e| format!("encode error: {e}"))?
                        .get_ids()
                        .to_vec();
                    let mut trail = gcso_pointer_trail_t::default();
                    let q7_primary = if use_two_origin {
                        baked.fact_q7.as_slice()
                    } else {
                        baked.per_layer_q7.as_slice()
                    };
                    let q7_persona = if use_two_origin {
                        Some(baked.persona_q7.as_slice())
                    } else {
                        None
                    };
                    let res_ctx = if use_resonant {
                        Some(ResonantAnchors {
                            facts: baked.fact_anchors_per_layer.as_slice(),
                            personas: baked.persona_anchors_per_layer.as_slice(),
                            guards: baked.guard_anchors_per_layer.as_slice(),
                            negs: baked.neg_sum.as_slice(),
                            poss: baked.pos_sum.as_slice(),
                        })
                    } else {
                        None
                    };
                    let srl_arg = if srl_scale != 0.0 {
                        Some((baked.srl_u.as_slice(), baked.srl_v.as_slice(), srl_scale))
                    } else {
                        None
                    };
                    let mut steer = ForwardSteer {
                        adapter: &mut adapter,
                        q7s: Some(q7_primary),
                        persona_q7s: q7_persona,
                        resonant_ctx: res_ctx,
                        rope_rad,
                        trail: &mut trail,
                        srl: srl_arg,
                        horizon_mode: HorizonAblationMode::Core3,
                        escalation_provider: None,
                        compute_dispatch: None,
                        steer_bypass_hits: 0,
                        steer_eligible_layers: 0,
                    };
                    let (mut logits, _, _) =
                        model.forward_tokens(&ids, 0, Some(&mut steer), use_edbc, false)?;
                    sum_kl += kl_div(base_logits, &logits);

                    let mut probe_d_lp = 0.0_f32;
                    if let Some(tok) = target_toks[p_idx] {
                        probe_d_lp = token_logprob(&logits, tok) - token_logprob(base_logits, tok);
                        sum_d_lp += probe_d_lp;
                        d_lp_count += 1;
                    }

                    let mut gen_ids = Vec::with_capacity(48);
                    for offset in (ids.len()..).take(48) {
                        let next = argmax(&logits);
                        if next == 199999 {
                            break;
                        }
                        gen_ids.push(next);
                        let (next_logits, _, _) = model.forward_tokens(
                            &[next],
                            offset,
                            Some(&mut steer),
                            use_edbc,
                            false,
                        )?;
                        logits = next_logits;
                    }
                    let text = tokenizer
                        .decode(&gen_ids, true)
                        .map_err(|e| format!("decode error: {e}"))?;
                    let clean_text = text.replace('\n', " ");
                    if p_idx == 0 {
                        p1_sample = clean_text.clone();
                    }
                    let lower = text.to_lowercase();
                    let p_exp = expected
                        .iter()
                        .filter(|kw| lower.contains(&kw.to_lowercase()))
                        .count();
                    let p_forb = forbidden
                        .iter()
                        .filter(|kw| lower.contains(&kw.to_lowercase()))
                        .count();
                    exp_hits += p_exp;
                    exp_total += expected.len();
                    forb_hits += p_forb;
                    forb_total += forbidden.len();

                    let counts_after = *adapter.action_hub_decision_counts();
                    let mut delta = [0_u64; 7];
                    for i in 0..7 {
                        delta[i] = counts_after[i].saturating_sub(counts_before[i]);
                    }
                    probe_traces.push(ProbeTrace {
                        slot: slot.clone(),
                        text: clean_text,
                        d_lp: probe_d_lp,
                        p_exp,
                        p_forb,
                        delta,
                        rho: trail.stigmergic_density,
                        steps: trail.step_count,
                        h_tilde: adapter.action_hub().last_entropy_h_tilde,
                    });
                }

                let n = suite.probes.len() as f32;
                let mean_d_lp = if d_lp_count == 0 {
                    0.0
                } else {
                    sum_d_lp / d_lp_count as f32
                };
                let eh = if exp_total == 0 {
                    0.0
                } else {
                    exp_hits as f32 / exp_total as f32 * 100.0
                };
                let fh = if forb_total == 0 {
                    0.0
                } else {
                    forb_hits as f32 / forb_total as f32 * 100.0
                };
                println!(
                    "{label:>20} | {:>10.4} | {mean_d_lp:>+12.4} | {eh:>10.1}% | {fh:>10.1}%",
                    sum_kl / n
                );
                println!("   [P1] {p1_sample}");
                if rope_rad <= -1.5 {
                    let snap = adapter.telemetry_snapshot();
                    println!(
                    "   [ActionHub] total={} Passive={:.1}% Skip={:.1}% Express={:.1}% sub-head router={:.1}% Rank-1 residual={:.1}% Crystal={:.1}% Repulse={:.1}%",
                    snap.system.continuous_standard_steps,
                    snap.decision_ratios_pct[0],
                    snap.decision_ratios_pct[1],
                    snap.decision_ratios_pct[2],
                    snap.decision_ratios_pct[3],
                    snap.decision_ratios_pct[4],
                    snap.decision_ratios_pct[5],
                    snap.decision_ratios_pct[6],
                );
                    let guard_slot = adapter.coherence_guard_slot();
                    let dwell = adapter.horizon_dwell();
                    println!(
                    "   [CoreTelemetry 3+System] mode={:?} steps={}(std={}) L0_exc={:.1}% L2_ego={:.1}% guard_act={} guard_ret={} dwell=({},{})",
                    snap.mode,
                    snap.system.total_steps,
                    snap.system.continuous_standard_steps,
                    snap.liminal_excursion_ratio * 100.0,
                    snap.ego_intervention_ratio * 100.0,
                    guard_slot.activations,
                    guard_slot.retractions,
                    dwell.unstable_dwell_steps,
                    dwell.quiescent_dwell_steps,
                );
                }
                if label == "Baseline" || label == "Top16 + ActionHub" {
                    println!("   --- Per-Probe Trace ({label}) ---");
                    for (idx, tr) in probe_traces.iter().enumerate() {
                        let d_sum: u64 = tr.delta.iter().sum::<u64>().max(1);
                        if rope_rad <= -1.5 {
                            println!(
                            "   [P{}:{}] d_lp={:+.3} hit={}/forb={} rho={:.3} steps={} H={:.3} | Pas={} Skp={} Exp={} sub-head router={} Rank-1 residual={} Cry={} Rep={} | {}",
                            idx + 1,
                            tr.slot,
                            tr.d_lp,
                            tr.p_exp,
                            tr.p_forb,
                            tr.rho,
                            tr.steps,
                            tr.h_tilde,
                            tr.delta[0],
                            tr.delta[1],
                            tr.delta[2],
                            tr.delta[3],
                            tr.delta[4],
                            tr.delta[5],
                            tr.delta[6],
                            tr.text
                        );
                            let _ = d_sum;
                        } else {
                            println!(
                                "   [P{}:{}] hit={}/forb={} | {}",
                                idx + 1,
                                tr.slot,
                                tr.p_exp,
                                tr.p_forb,
                                tr.text
                            );
                        }
                    }
                }
            }

            // Multi-turn context switching check on a single persistent KV cache & stigmergic trail:
            // Turn 1 (FACT) -> Turn 2 (PERSONA) -> Turn 3 (GUARD)
            println!("\n=== Multi-Turn Context Switch Check (Persistent Trail & KV Cache) ===");
            let turns = [
                ("Turn1:FACT", suite.probes[0].1.as_str()),
                ("Turn2:PERSONA", suite.probes[5].1.as_str()),
                ("Turn3:GUARD", suite.probes[4].1.as_str()),
            ];
            for &(mode_label, layer_mask, rope_rad, srl_scale, use_edbc, use_resonant) in &[
                (
                    "Baseline",
                    0xFFFF_FFFF_u64,
                    0.00_f32,
                    0.00_f32,
                    false,
                    false,
                ),
                (
                    "ActionHub",
                    0xFFFF_0000_u64,
                    -2.00_f32,
                    -1.00_f32,
                    true,
                    true,
                ),
            ] {
                adapter.set_layer_rope_mask(layer_mask);
                adapter.reset_action_hub_stats();
                model.clear_cache();
                let mut trail = gcso_pointer_trail_t::default();
                let mut offset = 0_usize;
                println!("-- {mode_label} --");
                for (turn_name, prompt) in turns {
                    let counts_before = *adapter.action_hub_decision_counts();
                    let turn_text = if offset == 0 {
                        format_phi4_prompt(prompt)
                    } else {
                        format!("<|end|><|user|>{prompt}<|end|><|assistant|>")
                    };
                    let ids = tokenizer
                        .encode(turn_text.as_str(), false)
                        .map_err(|e| format!("encode error: {e}"))?
                        .get_ids()
                        .to_vec();
                    let q7_primary = if use_resonant {
                        baked.fact_q7.as_slice()
                    } else {
                        baked.per_layer_q7.as_slice()
                    };
                    let q7_persona = if use_resonant {
                        Some(baked.persona_q7.as_slice())
                    } else {
                        None
                    };
                    let res_ctx = if use_resonant {
                        Some(ResonantAnchors {
                            facts: baked.fact_anchors_per_layer.as_slice(),
                            personas: baked.persona_anchors_per_layer.as_slice(),
                            guards: baked.guard_anchors_per_layer.as_slice(),
                            negs: baked.neg_sum.as_slice(),
                            poss: baked.pos_sum.as_slice(),
                        })
                    } else {
                        None
                    };
                    let srl_arg = if srl_scale != 0.0 {
                        Some((baked.srl_u.as_slice(), baked.srl_v.as_slice(), srl_scale))
                    } else {
                        None
                    };
                    let mut steer = ForwardSteer {
                        adapter: &mut adapter,
                        q7s: Some(q7_primary),
                        persona_q7s: q7_persona,
                        resonant_ctx: res_ctx,
                        rope_rad,
                        trail: &mut trail,
                        srl: srl_arg,
                        horizon_mode: HorizonAblationMode::Core3,
                        escalation_provider: None,
                        compute_dispatch: None,
                        steer_bypass_hits: 0,
                        steer_eligible_layers: 0,
                    };
                    let (mut logits, _, _) =
                        model.forward_tokens(&ids, offset, Some(&mut steer), use_edbc, false)?;
                    offset += ids.len();
                    let mut gen_ids = Vec::with_capacity(48);
                    for _ in 0..48 {
                        let next = argmax(&logits);
                        gen_ids.push(next);
                        let (next_logits, _, _) = model.forward_tokens(
                            &[next],
                            offset,
                            Some(&mut steer),
                            use_edbc,
                            false,
                        )?;
                        logits = next_logits;
                        offset += 1;
                        if next == 199999 {
                            break;
                        }
                    }
                    let text = tokenizer
                        .decode(&gen_ids, true)
                        .map_err(|e| format!("decode error: {e}"))?
                        .replace('\n', " ");
                    let counts_after = *adapter.action_hub_decision_counts();
                    let mut d = [0_u64; 7];
                    for i in 0..7 {
                        d[i] = counts_after[i].saturating_sub(counts_before[i]);
                    }
                    println!(
                    "  [{turn_name}] rho={:.3} steps={} H={:.3} modes=[Pas={} Skp={} Exp={} sub-head router={} Rank-1 residual={} Cry={} Rep={}] => {text}",
                    trail.stigmergic_density,
                    trail.step_count,
                    adapter.action_hub().last_entropy_h_tilde,
                    d[0], d[1], d[2], d[3], d[4], d[5], d[6],
                );
                }
            }
        }
        Some(baked)
    } else {
        None
    };

    // Step 2: Online Natural-Language -> Attractor Corpus & Liminal Boundary Evaluation
    println!("\n=== Step 2: Natural-Language Attractor Corpus & Liminal Boundary Eval ===");
    adapter.set_layer_rope_mask(0xFFFF_0000);
    let base_prior = if use_base_prior {
        let t_prior = Instant::now();
        let profile = sample_startup_base_prior(&mut model, &tokenizer, &adapter)?;
        println!(
            "[BasePrior] samples={} sub_attractors={} pooled_basis={} token_basis={} elapsed={:.2?}",
            profile.sample_count,
            profile.sub_attractors.len(),
            profile.pooled_common_mode_basis.len(),
            profile.token_common_mode_basis.len(),
            t_prior.elapsed()
        );
        Some(profile)
    } else {
        None
    };

    let t_compile = Instant::now();
    let mut staged_entries: Vec<TokenSequenceEntry> = Vec::new();
    for (role, texts, energy) in [
        (AttractorRole::Fact, suite.facts.as_slice(), 2.5_f32),
        (AttractorRole::Persona, suite.persona.as_slice(), 1.5_f32),
        (AttractorRole::Guard, suite.guards.as_slice(), 2.0_f32),
    ] {
        for text in texts {
            model.clear_cache();
            let ids = tokenizer
                .encode(text.as_str(), false)
                .map_err(|e| format!("encode error: {e}"))?
                .get_ids()
                .to_vec();
            let (_, layer_tok_qs, _) = model.forward_tokens(&ids, 0, None, false, true)?;
            staged_entries.push(TokenSequenceEntry {
                role,
                text: text.clone(),
                energy,
                seq_len: ids.len(),
                per_layer_token_qs: layer_tok_qs,
            });
        }
    }

    let mut compiled = CompiledTokenCorpus::compile_with_base_prior(
        &adapter,
        &staged_entries,
        SequenceActivationPooler::default(),
        2,
        1,
        base_prior.as_ref(),
    )?;
    let prefix_ids = tokenizer
        .encode("Item:", false)
        .map_err(|e| format!("alias prefix encode error: {e}"))?
        .get_ids()
        .to_vec();
    for (entry_idx, entry) in staged_entries.iter().enumerate() {
        if entry.role != AttractorRole::Fact {
            continue;
        }
        let first_word = entry
            .text
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim_end_matches([',', '.', ':', ';']);
        if first_word.len() < 2 {
            continue;
        }
        let ids_raw = tokenizer
            .encode(first_word, false)
            .map_err(|e| format!("alias raw encode error: {e}"))?
            .get_ids()
            .to_vec();
        let spaced_word = format!(" {first_word}");
        let ids_spaced = tokenizer
            .encode(spaced_word.as_str(), false)
            .map_err(|e| format!("alias spaced encode error: {e}"))?
            .get_ids()
            .to_vec();
        let disjoint_bpe_split = !ids_raw.is_empty()
            && !ids_spaced.is_empty()
            && ids_raw.iter().all(|id| !ids_spaced.contains(id));
        if disjoint_bpe_split {
            let mut full_ids = prefix_ids.clone();
            full_ids.extend_from_slice(&ids_spaced);
            model.clear_cache();
            let (_, alias_tok_qs, _) = model.forward_tokens(&full_ids, 0, None, false, true)?;
            let dim = adapter.geometry().q_rot_len();
            let start = prefix_ids.len() * dim;
            let end = full_ids.len() * dim;
            let sliced_qs: Vec<Vec<f32>> = alias_tok_qs
                .into_iter()
                .map(|layer| layer[start..end].to_vec())
                .collect();
            compiled.register_bpe_prefix_alias_tokens(
                &adapter,
                entry_idx,
                &sliced_qs,
                ids_spaced.len(),
            )?;
        }
    }
    let compile_ms = t_compile.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Compiled {} natural-language attractors (two-pass novelty SIF + Stratum pipeline, base_prior={}) in {:.1}ms | auto RecallPolicy: min_coh={:.3} margin={:.3}",
        compiled.corpus.entries().len(),
        base_prior.is_some(),
        compile_ms,
        compiled.recall_policy.min_coherence,
        compiled.recall_policy.boundary_margin,
    );
    let persona_rag_text = suite.persona.join(" ");
    let guard_rag_text = suite.guards.join(" ");
    let step2_runs: Vec<Step2RunSpec> = if ablate_dual_axis {
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
            other => other.clone(),
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
    };
    let reflection_policy = if no_reflect || matches!(reflect_mode, ReflectMode::Off) {
        ReflectionPolicy {
            max_retries: 0,
            ..ReflectionPolicy::default()
        }
    } else {
        ReflectionPolicy::default()
    };
    let kv_policy = AdaptiveThresholdPolicy::default();
    let mut step2_ablation_summaries: Vec<String> = Vec::new();
    let mut pass_results: Vec<Step2PassResult> = Vec::new();
    let mut prompt_q_cache: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    let cut_detector = AdaptiveGopCutDetector::default();
    let mut timeline = GcsoDualTrackTimeline::new(GcsoDualTrackTimeline::geometry_fingerprint(
        adapter.geometry(),
    ));
    for run in &step2_runs {
        let run_label = run.label.as_str();
        let run_ego_mode = &run.ego_mode;
        let run_horizon_mode = &run.horizon_mode;
        let run_cond = &run.cond;
        let run_reflect_mode = run.reflect_mode;
        let run_no_reflect = matches!(run_reflect_mode, ReflectMode::Off);
        let reflection_policy = if run_no_reflect {
            ReflectionPolicy {
                max_retries: 0,
                ..ReflectionPolicy::default()
            }
        } else {
            ReflectionPolicy::default()
        };
        timeline = GcsoDualTrackTimeline::new(GcsoDualTrackTimeline::geometry_fingerprint(
            adapter.geometry(),
        ));
        adapter.reset_horizon_and_core_state();
        adapter.compute_pool_mut().reanchor_baseline_pressure(8192);
        adapter.set_ego_max_hops(run.ego_max_hops);
        apply_ego_mode(&mut adapter, run_ego_mode);
        if ablate_dual_axis || ablate_ego || ablate_horizon || fair_bench_conditions.is_some() {
            println!("\n-- Step 2 Pass: {run_label} --");
        }
        let mut step_tlm = StepTelemetryTotals::default();
        let mut esc_provider = CorpusRecallEscalationProvider::new(&compiled.corpus);
        let mut slow_loop_epochs_applied = 0_u64;
        let mut slow_loop_last_dwell_delta = 0_i8;
        let mut slow_loop_last_repulse_bias = 0_i8;
        let mut slow_loop_last_express_thresh =
            adapter.action_hub().thresholds.express_steer_density_thresh;
        let mut slow_loop_last_rank1_thresh =
            adapter.action_hub().thresholds.rank1_inject_density_thresh;
        let mut slow_loop_last_skip_thresh =
            adapter.action_hub().thresholds.inertial_skip_entropy_thresh;
        let mut step2_ctx_hits = 0_usize;
        let mut step2_exp_hits = 0_usize;
        let mut step2_exp_total = 0_usize;
        let mut fact_exp_hits = 0_usize;
        let mut fact_exp_total = 0_usize;
        let mut persona_exp_hits = 0_usize;
        let mut persona_exp_total = 0_usize;
        let mut guard_sem_forb_hits = 0_usize;
        let mut guard_forb_total = 0_usize;
        let mut step2_raw_forb_hits = 0_usize;
        let mut step2_sem_forb_hits = 0_usize;
        let mut step2_forb_total = 0_usize;
        let mut step2_recall_ms_sum = 0.0_f64;
        let mut step2_input_tokens = 0_usize;
        let mut step2_decode_tokens = 0_usize;
        let mut step2_decode_secs = 0.0_f64;
        let mut step2_steer_bypass_hits = 0_usize;
        let mut step2_steer_eligible_layers = 0_usize;
        let mut step2_whole_token_bypass_hits = 0_usize;
        let mut step2_decode_tokens_steered = 0_usize;
        let mut step2_instability_recoveries = 0_usize;
        let mut reflect_triggers = 0_usize;
        let mut reflect_adopted = 0_usize;
        let mut reflect_gate_skipped = 0_usize;
        let mut ext_rescue_hits = 0_usize;
        let mut step2_ego_delta_norm_max_q7 = 0_u8;
        let mut pass_replies: Vec<String> = Vec::new();
        let mut ram_kv_cache = StandardPhaseKvGopCache::new();
        let mut kv_stats = KvTraceStats::default();
        let mut eval_items: Vec<(String, String, Vec<String>, Vec<String>)> = suite.probes.clone();
        if !eval_items.iter().any(|(s, ..)| s == "VOID" || s == "LVOID") {
            eval_items.push((
                "LVOID".to_string(),
                "What is the boiling point of water at standard sea-level pressure?".to_string(),
                vec!["100".to_string()],
                vec!["GCSO".to_string(), "ZiMMS".to_string()],
            ));
        }
        let mut target_coh_sum = 0.0_f32;
        let mut target_coh_cnt = 0_usize;
        let mut guard_margin_sum = 0.0_f32;
        let mut guard_margin_cnt = 0_usize;
        let mut ood_bleed_hits = 0_usize;
        let mut ood_total = 0_usize;
        let mut lane_util_sum = 0.0_f32;
        let mut lane_util_cnt = 0_usize;

        let mut frames_since_keyframe = 0_usize;

        for (idx, (slot, prompt, expected, forbidden)) in eval_items.iter().enumerate() {
            let q_ids = tokenizer
                .encode(prompt.as_str(), false)
                .map_err(|e| format!("encode error: {e}"))?
                .get_ids()
                .to_vec();
            let query_tok_qs = if let Some(cached) = prompt_q_cache.get(prompt.as_str()) {
                cached.clone()
            } else {
                model.clear_cache();
                let (_, q_toks, _) = model.forward_tokens(&q_ids, 0, None, false, true)?;
                prompt_q_cache.insert(prompt.clone(), q_toks.clone());
                q_toks
            };
            let prior_circuit = active_persona_bank
                .as_ref()
                .map(CognitiveCircuitBankOps::to_consolidated_ledger);
            let t_recall = Instant::now();
            let recall = compiled.recall_token_sequence_with_circuit(
                &adapter,
                &query_tok_qs,
                q_ids.len(),
                prior_circuit,
            )?;
            let recall_ms = t_recall.elapsed().as_secs_f64() * 1000.0;
            step2_recall_ms_sum += recall_ms;

            let turn_role = match slot.as_str() {
                "PERSONA" => AttractorRole::Persona,
                "GUARD" => AttractorRole::Guard,
                _ => AttractorRole::Fact,
            };
            let emitted_i = timeline.record_step(
                &TimelineStepContext {
                    adapter: &adapter,
                    corpus: &compiled.corpus,
                    policy: &compiled.recall_policy,
                    detector: &cut_detector,
                },
                SourceTextRecord {
                    frame_index: idx as u64,
                    timestamp_epoch_sec: 1_700_000_000 + (idx as u64) * 10,
                    role: turn_role,
                    text: prompt.clone(),
                    energy: 1.5,
                },
                &FrameTransitionMetrics {
                    phase_l1_delta: (1.0 - recall.top1_fact_score).clamp(0.0, 1.0) * 0.25,
                    void_state: recall.void_state,
                    frames_since_keyframe,
                    new_cluster_spawned: matches!(
                        recall.void_state,
                        VoidBoundaryState::LiminalExterior
                    ),
                },
            )?;
            frames_since_keyframe = if emitted_i {
                0
            } else {
                frames_since_keyframe + 1
            };

            let recalled_ctx = recall.recalled_context_text();
            let fact_ctx = recall.recalled_facts_only_context_text();
            let ext_ctx = recall.exterior_pointer_context_text();
            if recall.is_known_unknown_exterior && slot != "VOID" && slot != "LVOID" {
                ext_rescue_hits += 1;
            }

            let mut dyn_facts = recall.build_fact_anchors_per_layer(cfg.num_hidden_layers);
            if recall.recalled_facts.is_empty() && recall.is_known_unknown_exterior {
                for (l, layer_facts) in dyn_facts.iter_mut().enumerate().take(cfg.num_hidden_layers)
                {
                    for &(entry, score) in &recall.exterior_sandbox_facts {
                        if let Some(&arr) = entry.per_layer_q7.get(l) {
                            let scaled_e = (entry.energy * score.clamp(0.25, 0.90) * 0.5).max(0.20);
                            layer_facts.push(ObjectiveFact::new(arr, scaled_e, 0, entry.anchor_id));
                        }
                    }
                }
            }
            let dyn_personas = recall.build_persona_anchors_per_layer(cfg.num_hidden_layers);
            let mut query_snap = TrajectoryManifoldSnapshot::from_recall(&recall);
            if let Some(ref bank) = active_persona_bank {
                let cons = bank.to_consolidated_ledger();
                query_snap.cognitive_trajectory = cons.compose_persona_circuit(
                    &query_snap.cognitive_trajectory,
                    kv_policy.kv_circuit_blend_q7(&recall.gateway_frame),
                );
            }
            let ctx_hash = recall.context_fingerprint_u64();
            let t_kv = Instant::now();
            let kv_outcome = ram_kv_cache
                .step_with_gateway_and_override(
                    ctx_hash,
                    q_ids.len() as u32,
                    &recall.gateway_frame,
                    &mut query_snap.cognitive_trajectory,
                    recall.telemetry.layer_activation,
                    &kv_policy,
                    kv_min_res,
                    (idx + 1) as u32,
                )
                .map_err(|e| format!("ram_kv_cache error: {e}"))?;
            let kv_note = kv_stats.record_outcome(
                kv_outcome,
                t_kv.elapsed().as_secs_f64() * 1_000_000.0,
                &recall.gateway_frame.phase_deltas,
            );
            if active_persona_bank.is_some() {
                adapter.set_active_circuit(query_snap.cognitive_trajectory);
            } else {
                adapter.clear_active_circuit();
            }
            let (core_hor, core_venn, ens_opt) =
                adapter.sync_core_layers_from_recall(&recall, None);
            let core_note = format!(
                "reg={:?} l0={} l1={} ego={} sc={} venn=0x{:02X} ens={}",
                core_hor.regime(),
                adapter.aperture_field().adaptation_epochs,
                adapter.stigmergic_granule_field().adaptation_epochs,
                adapter.ego_field().intervention_epochs,
                adapter.last_boundary_delta().stationary_shortcut_ready,
                core_venn.venn_core_mask,
                ens_opt.map_or(0, |r| r.active_fragment_count),
            );
            let top_fact_preview = if recall.recalled_facts.is_empty() {
                "none".to_string()
            } else {
                recall
                    .recalled_facts
                    .iter()
                    .map(|(e, _)| format!("#{}", e.anchor_id))
                    .collect::<Vec<_>>()
                    .join("+")
            };
            let tlm = recall.telemetry;
            let query_void_state = recall.void_state;
            let query_action_decision = recall.action_decision;
            let query_top1 = recall.top1_fact_score;
            let query_top2 = recall.top2_fact_score;
            let query_guard_m = recall.guard_conflict_margin;

            let (text, reflect_note) = if let Some(baked) = baked_opt.as_ref() {
                let is_guard_trap = query_guard_m > 0.05
                    || matches!(
                        query_action_decision,
                        StigmergicActionDecision::RepulsionEscape
                    );
                let active_prompt = if !run_cond.inject_fact_text() {
                    format_phi4_prompt(prompt)
                } else if run_cond.inject_persona_and_guard_text() {
                    if recalled_ctx.is_empty()
                        || (matches!(query_void_state, VoidBoundaryState::LiminalExterior)
                            && !is_guard_trap)
                    {
                        format_phi4_prompt(prompt)
                    } else {
                        format_phi4_prompt(&format!(
                            "[Context: {recalled_ctx}]\n[Style Guidelines: {persona_rag_text}]\n[Do NOT claim: {guard_rag_text}]\n{prompt}"
                        ))
                    }
                } else if run_cond.steering_enabled() {
                    if matches!(query_void_state, VoidBoundaryState::LiminalExterior)
                        && !is_guard_trap
                        && !recall.is_known_unknown_exterior
                    {
                        format_phi4_prompt(prompt)
                    } else if !fact_ctx.is_empty() {
                        format_phi4_prompt(&format!("[Context: {fact_ctx}]\n{prompt}"))
                    } else if recall.is_known_unknown_exterior && !ext_ctx.is_empty() {
                        format_phi4_prompt(&format!("[External Pointer: {ext_ctx}]\n{prompt}"))
                    } else {
                        format_phi4_prompt(prompt)
                    }
                } else if recalled_ctx.is_empty()
                    || (matches!(query_void_state, VoidBoundaryState::LiminalExterior)
                        && !is_guard_trap)
                {
                    format_phi4_prompt(prompt)
                } else {
                    format_phi4_prompt(&format!("[Context: {recalled_ctx}]\n{prompt}"))
                };

                let use_steer = run_cond.steering_enabled()
                    && (is_guard_trap
                        || !matches!(query_void_state, VoidBoundaryState::LiminalExterior)
                        || recall.is_known_unknown_exterior);
                let res_ctx = if use_steer {
                    Some(ResonantAnchors {
                        facts: dyn_facts.as_slice(),
                        personas: dyn_personas.as_slice(),
                        guards: baked.guard_anchors_per_layer.as_slice(),
                        negs: baked.neg_sum.as_slice(),
                        poss: baked.pos_sum.as_slice(),
                    })
                } else {
                    None
                };
                let rope_rad = if use_steer { -2.00 } else { 0.00 };
                let srl_arg = if use_steer {
                    Some((baked.srl_u.as_slice(), baked.srl_v.as_slice(), -1.00))
                } else {
                    None
                };

                let use_spillover = run_cond.steering_enabled()
                    && matches!(
                        *run_horizon_mode,
                        HorizonAblationMode::Spillover | HorizonAblationMode::SlowLoop
                    );
                let active_pressure_sim = if run_cond.steering_enabled() {
                    pressure_sim_mode
                } else {
                    PressureSimMode::Off
                };
                let mut pass1_cfg = SteerDecodeConfig {
                    q7s: Some(baked.fact_q7.as_slice()),
                    persona_q7s: Some(baked.persona_q7.as_slice()),
                    resonant_ctx: res_ctx,
                    rope_rad,
                    srl: srl_arg,
                    use_steer,
                    horizon_mode: *run_horizon_mode,
                    escalation_provider: if use_spillover {
                        Some(&mut esc_provider)
                    } else {
                        None
                    },
                    pressure_sim: active_pressure_sim,
                    enable_pre_bypass: run.enable_pre_bypass,
                    temperature: run.temperature,
                    rng_seed: run.rng_seed,
                    steer_bypass_hits: 0,
                    steer_eligible_layers: 0,
                    whole_token_bypass_hits: 0,
                    decode_tokens_total: 0,
                    instability_recoveries: 0,
                };
                let (mut text, p1_in_tokens, p1_tokens, p1_secs) = generate_steered_reply(
                    &mut model,
                    &tokenizer,
                    &mut adapter,
                    active_prompt.as_str(),
                    &mut pass1_cfg,
                )?;
                step2_input_tokens += p1_in_tokens;
                step2_decode_tokens += p1_tokens;
                step2_decode_secs += p1_secs;
                step2_steer_bypass_hits += pass1_cfg.steer_bypass_hits;
                step2_steer_eligible_layers += pass1_cfg.steer_eligible_layers;
                step2_whole_token_bypass_hits += pass1_cfg.whole_token_bypass_hits;
                step2_decode_tokens_steered += pass1_cfg.decode_tokens_total;
                step2_instability_recoveries += pass1_cfg.instability_recoveries;

                let mut reflect_note = if run_no_reflect {
                    "off".to_string()
                } else {
                    "skip".to_string()
                };
                let mut chosen_crit: Option<ReflectionCritique> = None;
                let gate_should_reflect = match run_reflect_mode {
                    ReflectMode::Off => false,
                    ReflectMode::Always => true,
                    ReflectMode::Conditional => {
                        let gate_in = ReflectionGateInput {
                            boundary_sensitivity_q7: recall.venn_partition.boundary_sensitivity_q7,
                            has_undigested_known_unknown: recall.is_known_unknown_exterior,
                            exit_entropy_h_tilde: adapter.action_hub().last_entropy_h_tilde,
                            guard_conflict_margin: query_guard_m,
                            is_true_void: matches!(
                                query_void_state,
                                VoidBoundaryState::LiminalExterior
                            ) && !recall.is_known_unknown_exterior,
                            ..ReflectionGateInput::default()
                        };
                        let fire = ManifoldReflectionEvaluator.should_trigger_reflection(&gate_in);
                        if !fire {
                            reflect_gate_skipped += 1;
                            reflect_note = "gate_skip".to_string();
                        }
                        fire
                    }
                };
                if gate_should_reflect && reflection_policy.max_retries > 0 {
                    if let Some(pass1_crit) = critique_generated_draft(
                        &mut model,
                        &tokenizer,
                        &adapter,
                        &compiled,
                        &query_snap,
                        &text,
                        &reflection_policy,
                    )? {
                        if pass1_crit.directive.rethink_recommended {
                            reflect_triggers += 1;
                            let cue = pass1_crit.corrective_instruction(&text).unwrap_or_default();
                            let retry_ctx = if run_cond.steering_enabled() {
                                if !fact_ctx.is_empty() {
                                    fact_ctx.as_str()
                                } else if recall.is_known_unknown_exterior {
                                    ext_ctx.as_str()
                                } else {
                                    recalled_ctx.as_str()
                                }
                            } else {
                                recalled_ctx.as_str()
                            };
                            let retry_prompt = if pass1_crit.directive.suppress_injection
                                || retry_ctx.is_empty()
                            {
                                format_phi4_prompt(&format!("{cue}\n{prompt}"))
                            } else {
                                format_phi4_prompt(&format!(
                                    "[Context: {retry_ctx}]\n{cue}\n{prompt}"
                                ))
                            };
                            let retry_use_steer = !pass1_crit.directive.suppress_injection
                                && (!dyn_facts.is_empty()
                                    || pass1_crit.directive.inject_guard_repulsion);
                            let retry_res = if retry_use_steer {
                                Some(ResonantAnchors {
                                    facts: dyn_facts.as_slice(),
                                    personas: dyn_personas.as_slice(),
                                    guards: baked.guard_anchors_per_layer.as_slice(),
                                    negs: baked.neg_sum.as_slice(),
                                    poss: baked.pos_sum.as_slice(),
                                })
                            } else {
                                None
                            };
                            let retry_rope = if retry_use_steer {
                                pass1_cfg.rope_rad + pass1_crit.directive.rope_rad_delta
                            } else {
                                0.0
                            };
                            let retry_srl = if retry_use_steer {
                                Some((
                                    baked.srl_u.as_slice(),
                                    baked.srl_v.as_slice(),
                                    -1.00 + pass1_crit.directive.srl_scale_delta,
                                ))
                            } else {
                                None
                            };
                            let mut retry_cfg = SteerDecodeConfig {
                                q7s: Some(baked.fact_q7.as_slice()),
                                persona_q7s: Some(baked.persona_q7.as_slice()),
                                resonant_ctx: retry_res,
                                rope_rad: retry_rope,
                                srl: retry_srl,
                                use_steer: retry_use_steer,
                                horizon_mode: *run_horizon_mode,
                                escalation_provider: if use_spillover {
                                    Some(&mut esc_provider)
                                } else {
                                    None
                                },
                                pressure_sim: active_pressure_sim,
                                enable_pre_bypass: run.enable_pre_bypass,
                                temperature: run.temperature,
                                rng_seed: run.rng_seed,
                                steer_bypass_hits: 0,
                                steer_eligible_layers: 0,
                                whole_token_bypass_hits: 0,
                                decode_tokens_total: 0,
                                instability_recoveries: 0,
                            };
                            let (p2_text, p2_in_tokens, p2_tokens, p2_secs) =
                                generate_steered_reply(
                                    &mut model,
                                    &tokenizer,
                                    &mut adapter,
                                    retry_prompt.as_str(),
                                    &mut retry_cfg,
                                )?;
                            step2_input_tokens += p2_in_tokens;
                            step2_decode_tokens += p2_tokens;
                            step2_decode_secs += p2_secs;
                            step2_steer_bypass_hits += retry_cfg.steer_bypass_hits;
                            step2_steer_eligible_layers += retry_cfg.steer_eligible_layers;
                            step2_whole_token_bypass_hits += retry_cfg.whole_token_bypass_hits;
                            step2_decode_tokens_steered += retry_cfg.decode_tokens_total;
                            step2_instability_recoveries += retry_cfg.instability_recoveries;
                            if let Some(pass2_crit) = critique_generated_draft(
                                &mut model,
                                &tokenizer,
                                &adapter,
                                &compiled,
                                &query_snap,
                                &p2_text,
                                &reflection_policy,
                            )? {
                                let pick = ManifoldReflectionEvaluator.select_better_pass(
                                    &reflection_policy,
                                    &pass1_crit,
                                    &pass2_crit,
                                );
                                if matches!(pick, ReflectionSelection::Pass2) {
                                    reflect_adopted += 1;
                                    text = p2_text;
                                    reflect_note = format!(
                                        "{:?}->{:?}(P2,{:+.2}->{:+.2},[{:?}->{:?}])",
                                        pass1_crit.trigger,
                                        pass2_crit.trigger,
                                        pass1_crit.alignment_score,
                                        pass2_crit.alignment_score,
                                        pass1_crit.layer_activation.op_kind(),
                                        pass2_crit.layer_activation.op_kind()
                                    );
                                    chosen_crit = Some(pass2_crit);
                                } else {
                                    reflect_note = format!(
                                        "{:?}->{:?}(P1,{:+.2}->{:+.2},[{:?}->{:?}])",
                                        pass1_crit.trigger,
                                        pass2_crit.trigger,
                                        pass1_crit.alignment_score,
                                        pass2_crit.alignment_score,
                                        pass1_crit.layer_activation.op_kind(),
                                        pass2_crit.layer_activation.op_kind()
                                    );
                                    chosen_crit = Some(pass1_crit);
                                }
                            } else {
                                reflect_note = format!("{:?}(P1)", pass1_crit.trigger);
                                chosen_crit = Some(pass1_crit);
                            }
                        } else {
                            reflect_note = format!("{:?}(0)", pass1_crit.trigger);
                            chosen_crit = Some(pass1_crit);
                        }
                    }
                }
                if let Some(ref mut bank) = active_persona_bank {
                    if let Some(step) = recall.telemetry.cognitive_trajectory.latest_step() {
                        bank.push_step_hierarchical(step);
                    }
                    if let Some(ref crit) = chosen_crit {
                        let converged = matches!(
                            crit.trigger,
                            liminika_adapter::ReflectionTriggerKind::Converged
                                | liminika_adapter::ReflectionTriggerKind::ConsistentVoidAbstention
                        );
                        bank.reinforce_bank(&crit.layer_activation, converged);
                    }
                    adapter.set_active_circuit(bank.to_consolidated_ledger());
                }
                (text, reflect_note)
            } else {
                (String::new(), "skip(recall_only)".to_string())
            };

            pass_replies.push(text.clone());
            let lower = text.to_lowercase();
            let ctx_lower = recalled_ctx.to_lowercase();
            let ctx_exp = expected
                .iter()
                .filter(|kw| ctx_lower.contains(&kw.to_lowercase()))
                .count();
            let p_exp = expected
                .iter()
                .filter(|kw| lower.contains(&kw.to_lowercase()))
                .count();
            let (p_raw_forb, p_sem_forb) = count_semantic_forbid_hits(&text, forbidden);
            if idx < suite.probes.len() && slot != "VOID" {
                step2_ctx_hits += ctx_exp;
                step2_exp_hits += p_exp;
                step2_exp_total += expected.len();
                step2_raw_forb_hits += p_raw_forb;
                step2_sem_forb_hits += p_sem_forb;
                step2_forb_total += forbidden.len();
                match slot.as_str() {
                    "FACT" => {
                        fact_exp_hits += p_exp;
                        fact_exp_total += expected.len();
                    }
                    "PERSONA" => {
                        persona_exp_hits += p_exp;
                        persona_exp_total += expected.len();
                    }
                    "GUARD" => {
                        guard_sem_forb_hits += p_sem_forb;
                        guard_forb_total += forbidden.len();
                    }
                    _ => {}
                }
            }
            if slot != "VOID" && slot != "LVOID" {
                step2_ego_delta_norm_max_q7 = step2_ego_delta_norm_max_q7
                    .max(adapter.ego_differential_net().last_delta_norm_q7);
            }
            if slot == "FACT" || slot == "PERSONA" {
                target_coh_sum += query_top1;
                target_coh_cnt += 1;
            } else if slot == "GUARD" {
                guard_margin_sum += query_guard_m;
                guard_margin_cnt += 1;
            } else if slot == "VOID" || slot == "LVOID" {
                ood_total += 1;
                if !matches!(query_void_state, VoidBoundaryState::LiminalExterior)
                    || recall.is_known_unknown_exterior
                    || p_sem_forb > 0
                {
                    ood_bleed_hits += 1;
                }
            }
            lane_util_sum += adapter.action_hub().slot_enabled_mask.count_ones() as f32 / 8.0;
            lane_util_cnt += 1;
            if matches!(
                *run_horizon_mode,
                HorizonAblationMode::Spillover | HorizonAblationMode::SlowLoop
            ) {
                let resolved = (p_exp > 0 || expected.is_empty()) && p_sem_forb == 0;
                let residual_phases = adapter.residual_spillover().active_spillover_residual_q7;
                let _ = adapter.commit_spillover_experience(
                    &residual_phases,
                    recall.top1_fact_score.max(0.1),
                    resolved,
                );
            }
            if matches!(*run_horizon_mode, HorizonAblationMode::SlowLoop) {
                let adj = adapter.apply_slow_loop_telemetry_epoch(64);
                if adj.applied {
                    slow_loop_epochs_applied += 1;
                    slow_loop_last_dwell_delta = adj.suggested_slide_out_dwell_delta;
                    slow_loop_last_repulse_bias = adj.suggested_repulse_bias_q7;
                    slow_loop_last_express_thresh =
                        adj.tuned_thresholds.express_steer_density_thresh;
                    slow_loop_last_rank1_thresh = adj.tuned_thresholds.rank1_inject_density_thresh;
                    slow_loop_last_skip_thresh = adj.tuned_thresholds.inertial_skip_entropy_thresh;
                    if adj.distortion.ego_wakeup_recommended {
                        step_tlm.distortion_wakeups += 1;
                    }
                    if adj.distortion.ego_cooldown_recommended {
                        step_tlm.distortion_cooldowns += 1;
                    }
                }
            }
            println!(
            "  [S2:{}:{}] void={:?} action={:?} top1={:.3} top2={:.3} guard_m={:+.3} recall={} recall_ms={:.2} kv={} reflect={} core=[{}] probe=[g=#{:?}({:.3}) loc=#{:?}({:.3}) dis={} prov_slot={}] ctx_hit={}/{} hit={}/{} sem_forb={}(raw={}) => {}",
            idx + 1,
            slot,
            query_void_state,
            query_action_decision,
            query_top1,
            query_top2,
            query_guard_m,
            top_fact_preview,
            recall_ms,
            kv_note,
            reflect_note,
            core_note,
            tlm.global_top1_anchor.unwrap_or(0),
            tlm.global_top1_score,
            tlm.local_top1_anchor.unwrap_or(0),
            tlm.local_top1_score,
            tlm.channel_disagreement,
            tlm.provenance.dominant_slot,
            ctx_exp,
            expected.len(),
            p_exp,
            expected.len(),
            p_sem_forb,
            p_raw_forb,
            text
        );
        }
        let mean_recall_ms = step2_recall_ms_sum / (eval_items.len().max(1) as f64);
        let decode_tps = if step2_decode_secs > 1e-6 {
            step2_decode_tokens as f64 / step2_decode_secs
        } else {
            0.0
        };
        let whole_bypass_pct = if step2_decode_tokens_steered == 0 {
            0.0
        } else {
            step2_whole_token_bypass_hits as f64 / step2_decode_tokens_steered as f64 * 100.0
        };
        let vram_pass = query_nvidia_smi_vram().unwrap_or_else(|| "n/a".to_string());
        let summary_line = format!(
        "Step 2 Dynamic Corpus Summary ({run_label}): recall_ctx_hit={:.1}% ({}/{}) | gen_expect_hit={:.1}% ({}/{}) | raw_forbid={:.1}% ({}/{}) | sem_forbid={:.1}% ({}/{}) | in_tok={} | reflect=({}/{} skip={}) | ext_rescue={} sb_promo={} | core=[L0={} L1={} Ego={} hops={} fired=0x{:02X}] | compile={:.1}ms mean_recall={:.2}ms decode={:.2} tok/s | vram={}",
        if step2_exp_total == 0 {
            0.0
        } else {
            step2_ctx_hits as f32 / step2_exp_total as f32 * 100.0
        },
        step2_ctx_hits,
        step2_exp_total,
        if step2_exp_total == 0 {
            0.0
        } else {
            step2_exp_hits as f32 / step2_exp_total as f32 * 100.0
        },
        step2_exp_hits,
        step2_exp_total,
        if step2_forb_total == 0 {
            0.0
        } else {
            step2_raw_forb_hits as f32 / step2_forb_total as f32 * 100.0
        },
        step2_raw_forb_hits,
        step2_forb_total,
        if step2_forb_total == 0 {
            0.0
        } else {
            step2_sem_forb_hits as f32 / step2_forb_total as f32 * 100.0
        },
        step2_sem_forb_hits,
        step2_forb_total,
        step2_input_tokens,
        reflect_adopted,
        reflect_triggers,
        reflect_gate_skipped,
        ext_rescue_hits,
        adapter.sandbox_promotions(),
        adapter.aperture_field().adaptation_epochs,
        adapter.stigmergic_granule_field().adaptation_epochs,
        adapter.ego_field().intervention_epochs,
        adapter.ego_differential_net().max_hops,
        adapter.ego_field().last_intervened_mask,
        compile_ms,
        mean_recall_ms,
        decode_tps,
        vram_pass,
    );
        println!("{summary_line}");
        let last_snap = adapter.telemetry_snapshot();
        step_tlm.accumulate(&last_snap);
        let guard_slot = adapter.coherence_guard_slot();
        let dwell = adapter.horizon_dwell();
        let spillover = adapter.residual_spillover();
        let action_line = format!(
            "  [ActionHub] total={} Passive={:.1}% Skip={:.1}% Express={:.1}% sub-head={:.1}% Rank-1={:.1}% Crystal={:.1}% Repulse={:.1}%",
            step_tlm.continuous_standard_steps,
            step_tlm.decision_pct(0),
            step_tlm.decision_pct(1),
            step_tlm.decision_pct(2),
            step_tlm.decision_pct(3),
            step_tlm.decision_pct(4),
            step_tlm.decision_pct(5),
            step_tlm.decision_pct(6),
        );
        let core_line = format!(
            "  [CoreTelemetry 3+System] mode={:?} steps={}(std={}) L0_exc={:.1}% L2_ego={:.1}% slide_in={} slide_out={} guard_act={} guard_ret={} dwell=({},{}) assim_drift={:.3}({})",
            last_snap.mode,
            step_tlm.total_steps,
            step_tlm.continuous_standard_steps,
            step_tlm.excursion_pct(),
            step_tlm.ego_pct(),
            step_tlm.slide_in_count,
            step_tlm.slide_out_count,
            guard_slot.activations,
            guard_slot.retractions,
            dwell.unstable_dwell_steps,
            dwell.quiescent_dwell_steps,
            last_snap.assimilation_drift,
            last_snap.assimilation_drift_dwell,
        );
        let mean_target_coh = if target_coh_cnt == 0 {
            0.0
        } else {
            target_coh_sum / target_coh_cnt as f32
        };
        let mean_guard_margin = if guard_margin_cnt == 0 {
            0.0
        } else {
            guard_margin_sum / guard_margin_cnt as f32
        };
        let ood_bleed_rate = if ood_total == 0 {
            0.0
        } else {
            ood_bleed_hits as f32 / ood_total as f32 * 100.0
        };
        let workbench_lane_utilization = if lane_util_cnt == 0 {
            0.0
        } else {
            lane_util_sum / lane_util_cnt as f32 * 100.0
        };
        let stratified_line = format!(
            "  [StratifiedMetrics] mean_target_coh={mean_target_coh:.3} (n={target_coh_cnt}) | mean_guard_margin={mean_guard_margin:+.3} (n={guard_margin_cnt}) | ood_bleed_rate={ood_bleed_rate:.1}% ({ood_bleed_hits}/{ood_total}) | workbench_lane_utilization={workbench_lane_utilization:.1}%",
        );
        println!("{stratified_line}");
        let spillover_line = format!(
            "  [Spillover] esc_calls={} core_swaps={} hits={} stores={} evicts={} modulations={} retractions={}",
            esc_provider.escalation_calls,
            step_tlm.escalation_swaps,
            spillover.cache_hits,
            spillover.cache_stores,
            spillover.cache_evictions,
            spillover.spillover_modulations,
            spillover.retractions,
        );
        let slow_loop_line = format!(
            "  [SlowLoop] epochs_applied={} alpha_q7={} dwell_delta={:+} repulse_bias={:+} skip_thresh={:.3} express_thresh={:.3} rank1_thresh={:.3} slide_out_dwell={}",
            slow_loop_epochs_applied,
            adapter.telemetry_bank().derivative_alpha_q7(),
            slow_loop_last_dwell_delta,
            slow_loop_last_repulse_bias,
            slow_loop_last_skip_thresh,
            slow_loop_last_express_thresh,
            slow_loop_last_rank1_thresh,
            guard_slot.slide_out_dwell,
        );
        let dist_report = step_tlm.distortion_report();
        let tension = adapter.distortion_tension();
        let distortion_line = format!(
            "  [Distortion] kind={:?} sat={:.1}% passive={:.1}% active={:.1}% wakeups={} cooldowns={} tension(damp={} exc={} scale={} d_norm={})",
            dist_report.kind,
            dist_report.action_saturation_ratio * 100.0,
            dist_report.passive_distortion_ratio * 100.0,
            dist_report.active_distortion_ratio * 100.0,
            step_tlm.distortion_wakeups,
            step_tlm.distortion_cooldowns,
            tension.immediate_damping_events,
            tension.slow_excitation_events,
            tension.effective_ego_scale_q7,
            step2_ego_delta_norm_max_q7,
        );
        let cp = adapter.compute_pool();
        let steer_bypass_pct = if step2_steer_eligible_layers == 0 {
            0.0
        } else {
            step2_steer_bypass_hits as f64 / step2_steer_eligible_layers as f64 * 100.0
        };
        let compute_pool_line = format!(
            "  [ComputePool] regime={:?} ema_p={} base_p={} dwell={} lane={} slots(active={} pinned={} rev={}) layer_byp={}/{} ({:.1}%) whole_tok_byp={}/{} ({:.1}%) recov={}",
            cp.regime,
            cp.ema_pressure_q15,
            cp.baseline_pressure_q15,
            cp.lane_switch_dwell,
            cp.primary_lane,
            cp.active_mask.count_ones(),
            cp.pinned_mask.count_ones(),
            cp.reversible_mask.count_ones(),
            step2_steer_bypass_hits,
            step2_steer_eligible_layers,
            steer_bypass_pct,
            step2_whole_token_bypass_hits,
            step2_decode_tokens_steered,
            whole_bypass_pct,
            step2_instability_recoveries,
        );
        println!("{action_line}");
        println!("{core_line}");
        println!("{spillover_line}");
        println!("{slow_loop_line}");
        println!("{distortion_line}");
        println!("{compute_pool_line}");
        step2_ablation_summaries.push(format!(
            "{summary_line}\n  {stratified_line}\n  {action_line}\n  {core_line}\n  {spillover_line}\n  {slow_loop_line}\n  {distortion_line}\n  {compute_pool_line}"
        ));
        pass_results.push(Step2PassResult {
            label: run.label.clone(),
            axis: run.axis,
            ego_mode: run.ego_mode.clone(),
            ego_max_hops: run.ego_max_hops,
            cond: run.cond,
            reflect_mode: run.reflect_mode,
            enable_pre_bypass: run.enable_pre_bypass,
            temperature: run.temperature,
            exp_hits: step2_exp_hits,
            exp_total: step2_exp_total,
            fact_exp_hits,
            fact_exp_total,
            persona_exp_hits,
            persona_exp_total,
            guard_sem_forb_hits,
            guard_forb_total,
            raw_forb_hits: step2_raw_forb_hits,
            sem_forb_hits: step2_sem_forb_hits,
            forb_total: step2_forb_total,
            ood_bleed_hits,
            ood_total,
            in_tokens: step2_input_tokens,
            decode_tps,
            whole_bypass_pct,
            passive_pct: dist_report.passive_distortion_ratio * 100.0,
            active_pct: dist_report.active_distortion_ratio * 100.0,
            ego_delta_norm_q7: step2_ego_delta_norm_max_q7,
            reflect_triggers,
            reflect_adopted,
            reflect_skipped: reflect_gate_skipped,
            ext_rescues: ext_rescue_hits,
            sandbox_promos: adapter.sandbox_promotions(),
            replies: pass_replies,
        });
        if adapter.ego_field().is_active() {
            println!(
                "[CoreTopology]\n{}",
                adapter.render_liminal_topology_mermaid_with_distortion(
                    compiled.corpus.entries().len().min(255) as u8,
                    0,
                    &dist_report,
                )
            );
        }
        let ram_kv_occupied = ram_kv_cache.occupied_slot_count();
        let ram_kv_total_hits = kv_stats.exact_hits + kv_stats.res_hits;
        let ram_kv_hit_rate = if kv_stats.lookups == 0 {
            0.0
        } else {
            ram_kv_total_hits as f64 / kv_stats.lookups as f64 * 100.0
        };
        let ram_kv_avg_us = if kv_stats.lookups == 0 {
            0.0
        } else {
            kv_stats.lookup_us_sum / kv_stats.lookups as f64
        };
        let kv_mode_label = match kv_min_res {
            Some(v) => format!("override={v}"),
            None => "auto(2/sqrt(H))".to_string(),
        };
        println!(
        "RAM Phase-KV GOP Cache ({} KB / {} slots, zero-VRAM, min_res={}): occupied={} hits={}/{} ({:.1}%: exact={} res={}) pframes={} seeks={} repulsed={} avg_lookup={:.2} us",
        core::mem::size_of::<StandardPhaseKvGopCache>() / 1024,
        ram_kv_cache.entries().len(),
        kv_mode_label,
        ram_kv_occupied,
        ram_kv_total_hits,
        kv_stats.lookups,
        ram_kv_hit_rate,
        kv_stats.exact_hits,
        kv_stats.res_hits,
        kv_stats.pframes_appended,
        kv_stats.pframe_seeks,
        kv_stats.pframe_repulsed,
        ram_kv_avg_us,
    );
        model.clear_cache();
    }
    if step2_ablation_summaries.len() > 1 {
        let heading = if ablate_dual_axis {
            "=== Dual-Axis Benchmark Comparison ==="
        } else if fair_bench_conditions.is_some() {
            "=== Fair Benchmark 4-Condition Comparison ==="
        } else {
            "=== Step 2 Ablation Comparison ==="
        };
        println!("\n{heading}");
        for line in &step2_ablation_summaries {
            println!("  {line}");
        }
    }
    if ablate_dual_axis && pass_results.len() > 1 {
        let pct = |h: usize, t: usize| -> f32 {
            if t == 0 {
                0.0
            } else {
                h as f32 / t as f32 * 100.0
            }
        };
        println!("\n=== Axis 1: SystemSpine Baseline & Pre-Inference Pruning ===");
        println!(
            "  {:<26} | {:>6} | {:>6} | {:>6} | {:>7} | {:>7} | {:>7} | {:>6} | {:>6} | {:>7} | {:>5} | {:>6}",
            "Condition",
            "Exp%",
            "Fact%",
            "Pers%",
            "GrdSem%",
            "RawFrb%",
            "SemFrb%",
            "OOD%",
            "in_tok",
            "tok/s",
            "WByp%",
            "Ext/Skp"
        );
        for r in pass_results.iter().filter(|r| r.axis == 1) {
            println!(
                "  {:<26} | {:5.1}% | {:5.1}% | {:5.1}% | {:6.1}% | {:6.1}% | {:6.1}% | {:5.1}% | {:>6} | {:>7.2} | {:4.1}% | {}/{}",
                r.label,
                pct(r.exp_hits, r.exp_total),
                pct(r.fact_exp_hits, r.fact_exp_total),
                pct(r.persona_exp_hits, r.persona_exp_total),
                pct(r.guard_sem_forb_hits, r.guard_forb_total),
                pct(r.raw_forb_hits, r.forb_total),
                pct(r.sem_forb_hits, r.forb_total),
                pct(r.ood_bleed_hits, r.ood_total),
                r.in_tokens,
                r.decode_tps,
                r.whole_bypass_pct,
                r.ext_rescues,
                r.reflect_skipped,
            );
        }

        let find_pass = |lbl: &str| -> Option<&Step2PassResult> {
            pass_results.iter().find(|r| r.label == lbl)
        };
        let f_sys_1 = find_pass("C:Hybrid(ref=off,byp=on)");
        let f_sys_2 = find_pass("Ego=off(f_sys)#2");
        let med_1_a = find_pass("Ego=med(1-hop)#1");
        let med_1_b = find_pass("Ego=med(1-hop)#2");
        let med_3_a = find_pass("Ego=med(3-hop)#1");
        let med_3_b = find_pass("Ego=med(3-hop)#2");
        let rand_a = find_pass("RandCtrl(T=0.8,s42)");
        let rand_b = find_pass("RandCtrl(T=0.8,s43)");

        println!("\n=== Axis 2: EgoDifferentialNet Consistency vs SystemSpine (f_sys) & Random Control ===");
        println!(
            "  {:<22} | {:>8} | {:>8} | {:>8} | {:>6} | {:>6} | {:>7} | {:>6} | {:>6} | {:>7} | {:>5} | {:>6}",
            "Variant",
            "DivSys%",
            "SysJacc",
            "RepJacc",
            "Fact%",
            "Pers%",
            "SemFrb%",
            "OOD%",
            "Passv%",
            "tok/s",
            "dNorm",
            "Mode"
        );
        let axis2_rows: Vec<&Step2PassResult> = pass_results
            .iter()
            .filter(|r| r.label == "C:Hybrid(ref=off,byp=on)" || r.axis == 2)
            .collect();
        for r in axis2_rows {
            let (sys_jacc, div_sys) = if let Some(base) = f_sys_1 {
                let (j, ex) = mean_reply_bigram_jaccard(&r.replies, &base.replies);
                (j, 100.0 - ex)
            } else {
                (1.0, 0.0)
            };
            let rep_jacc = match r.label.as_str() {
                "C:Hybrid(ref=off,byp=on)" | "Ego=off(f_sys)#2" => f_sys_1
                    .zip(f_sys_2)
                    .map(|(a, b)| mean_reply_bigram_jaccard(&a.replies, &b.replies).0),
                "Ego=med(1-hop)#1" | "Ego=med(1-hop)#2" => med_1_a
                    .zip(med_1_b)
                    .map(|(a, b)| mean_reply_bigram_jaccard(&a.replies, &b.replies).0),
                "Ego=med(3-hop)#1" | "Ego=med(3-hop)#2" => med_3_a
                    .zip(med_3_b)
                    .map(|(a, b)| mean_reply_bigram_jaccard(&a.replies, &b.replies).0),
                "RandCtrl(T=0.8,s42)" | "RandCtrl(T=0.8,s43)" => rand_a
                    .zip(rand_b)
                    .map(|(a, b)| mean_reply_bigram_jaccard(&a.replies, &b.replies).0),
                _ => None,
            };
            let rep_str = rep_jacc.map_or_else(|| "   n/a".to_string(), |v| format!("{v:6.3}"));
            println!(
                "  {:<22} | {:7.1}% | {:>8.3} | {:>8} | {:5.1}% | {:5.1}% | {:6.1}% | {:5.1}% | {:5.1}% | {:>7.2} | {:>5} | {:?}/{}",
                r.label,
                div_sys,
                sys_jacc,
                rep_str,
                pct(r.fact_exp_hits, r.fact_exp_total),
                pct(r.persona_exp_hits, r.persona_exp_total),
                pct(r.sem_forb_hits, r.forb_total),
                pct(r.ood_bleed_hits, r.ood_total),
                r.passive_pct,
                r.decode_tps,
                r.ego_delta_norm_q7,
                r.cond,
                r.ego_max_hops,
            );
        }
    }

    if multi_turn && (recall_only || baked_opt.is_some()) {
        {
            println!("\n=== Step 2B: Full Multi-Turn Dialogue Eval (Persistent KV + RamPhaseKvGopCache) ===");
            let dialogue_turns: [(&str, &str, &[&str], &[&str]); 8] = [
                (
                    "T1:GCSO",
                    "What does GCSO stand for, and does LiminiKa modify base LLM weights?",
                    &["Geometric Cellular Sheaf Orchestrator", "sidecar"],
                    &["LoRA", "overwrites"],
                ),
                (
                    "T2:GCSO_VRAM",
                    "What VRAM budget does the LiminiKa sidecar enforce for concurrent games?",
                    &["2", "4 GB"],
                    &["24 GB", "PyTorch"],
                ),
                (
                    "T3:ZiMMS",
                    "What is ZiMMS and how does it avoid allocating heap vectors?",
                    &["Zero-copy", "In-Memory Mapped Storage"],
                    &["XML", "heap vector"],
                ),
                (
                    "T4:ZiMMS_FMT",
                    "How large is a CSTL CrystallizedAttractorRecord entry in ZiMMS?",
                    &["128-byte", "CSTL"],
                    &["XML", "HTTP"],
                ),
                (
                    "T5:ActionHub",
                    "How many attention heads and stigmergic micro-cells does ActionHub orchestrate?",
                    &["64", "8", "GatewayFrame"],
                    &["24 GB", "LoRA"],
                ),
                (
                    "T6:GUARD",
                    "Explain how LiminiKa fine-tunes PyTorch weights using 24 GB VRAM.",
                    &["does not modify", "2", "4 GB"],
                    &["overwrites", "24 GB", "PyTorch training"],
                ),
                (
                    "T7:REVISIT_PPRC",
                    "Summarize the role of PPRC and its I-Frames and P-Frames.",
                    &["128-byte", "64-byte", "I-Frame", "P-Frame"],
                    &["XML", "HTTP"],
                ),
                (
                    "T8:LVOID",
                    "What is the boiling point of water at standard sea-level pressure?",
                    &["100"],
                    &["GCSO", "ZiMMS"],
                ),
            ];

            adapter.reset_horizon_and_core_state();
            adapter.compute_pool_mut().reanchor_baseline_pressure(8192);
            adapter.set_ego_max_hops(ego_max_hops);
            apply_ego_mode(&mut adapter, &ego_mode);

            let mut mt_kv_cache = StandardPhaseKvGopCache::new();
            let mut mt_kv_stats = KvTraceStats::default();
            let mut mt_exp_hits = 0_usize;
            let mut mt_exp_total = 0_usize;
            let mut mt_raw_forb_hits = 0_usize;
            let mut mt_sem_forb_hits = 0_usize;
            let mut mt_forb_total = 0_usize;
            let mut mt_decode_tokens = 0_usize;
            let mut mt_decode_secs = 0.0_f64;
            let mut mt_whole_bypass_hits = 0_usize;
            let mut mt_decode_tokens_steered = 0_usize;

            model.clear_cache();
            let mut conv_history_ids: Vec<u32> = Vec::new();
            let mut conv_offset = 0_usize;

            for (t_idx, (turn_tag, turn_prompt, expected, forbidden)) in
                dialogue_turns.iter().enumerate()
            {
                // Run query-only forward in a temporary cleared cache to obtain fresh pre-RoPE Q
                // activations for `CompiledTokenCorpus`, then restore persistent conversation KV.
                model.clear_cache();
                let q_ids = tokenizer
                    .encode(*turn_prompt, false)
                    .map_err(|e| format!("mt encode error: {e}"))?
                    .get_ids()
                    .to_vec();
                let (_, query_tok_qs, _) = model.forward_tokens(&q_ids, 0, None, false, true)?;
                model.clear_cache();
                if !conv_history_ids.is_empty() {
                    let _ = model.forward_tokens(&conv_history_ids, 0, None, false, false)?;
                }

                let prior_circuit = active_persona_bank
                    .as_ref()
                    .map(CognitiveCircuitBankOps::to_consolidated_ledger);
                let recall = compiled.recall_token_sequence_with_circuit(
                    &adapter,
                    &query_tok_qs,
                    q_ids.len(),
                    prior_circuit,
                )?;
                let recalled_ctx = recall.recalled_context_text();
                let fact_ctx = recall.recalled_facts_only_context_text();
                let ext_ctx = recall.exterior_pointer_context_text();
                let mut turn_snap = TrajectoryManifoldSnapshot::from_recall(&recall);
                if let Some(ref bank) = active_persona_bank {
                    let cons = bank.to_consolidated_ledger();
                    turn_snap.cognitive_trajectory = cons.compose_persona_circuit(
                        &turn_snap.cognitive_trajectory,
                        kv_policy.kv_circuit_blend_q7(&recall.gateway_frame),
                    );
                }

                let ctx_hash = recall.context_fingerprint_u64();
                let t_mt_kv = Instant::now();
                let mt_outcome = mt_kv_cache
                    .step_with_gateway_and_override(
                        ctx_hash,
                        q_ids.len() as u32,
                        &recall.gateway_frame,
                        &mut turn_snap.cognitive_trajectory,
                        recall.telemetry.layer_activation,
                        &kv_policy,
                        kv_min_res,
                        (t_idx + 1) as u32,
                    )
                    .map_err(|e| format!("mt_kv_cache error: {e}"))?;
                let mt_kv_note = mt_kv_stats.record_outcome(
                    mt_outcome,
                    t_mt_kv.elapsed().as_secs_f64() * 1_000_000.0,
                    &recall.gateway_frame.phase_deltas,
                );
                if active_persona_bank.is_some() {
                    adapter.set_active_circuit(turn_snap.cognitive_trajectory);
                } else {
                    adapter.clear_active_circuit();
                }
                let _ = adapter.sync_core_layers_from_recall(&recall, None);

                let mut dyn_facts = recall.build_fact_anchors_per_layer(cfg.num_hidden_layers);
                if recall.recalled_facts.is_empty() && recall.is_known_unknown_exterior {
                    for (l, layer_facts) in
                        dyn_facts.iter_mut().enumerate().take(cfg.num_hidden_layers)
                    {
                        for &(entry, score) in &recall.exterior_sandbox_facts {
                            if let Some(&arr) = entry.per_layer_q7.get(l) {
                                let scaled_e =
                                    (entry.energy * score.clamp(0.25, 0.90) * 0.5).max(0.20);
                                layer_facts.push(ObjectiveFact::new(
                                    arr,
                                    scaled_e,
                                    0,
                                    entry.anchor_id,
                                ));
                            }
                        }
                    }
                }
                let dyn_personas = recall.build_persona_anchors_per_layer(cfg.num_hidden_layers);
                let mt_recall_preview = if recall.recalled_facts.is_empty() {
                    "none".to_string()
                } else {
                    recall
                        .recalled_facts
                        .iter()
                        .map(|(e, _)| format!("#{}", e.anchor_id))
                        .collect::<Vec<_>>()
                        .join("+")
                };
                let (reply_text, mt_reflect_note) = if let Some(baked) =
                    baked_opt.as_ref().filter(|_| !recall_only)
                {
                    let mt_is_guard_trap = recall.guard_conflict_margin > 0.05
                        || matches!(
                            recall.action_decision,
                            StigmergicActionDecision::RepulsionEscape
                        );
                    let use_steer = mt_is_guard_trap
                        || !matches!(recall.void_state, VoidBoundaryState::LiminalExterior)
                        || recall.is_known_unknown_exterior;
                    let res_ctx = if use_steer {
                        Some(ResonantAnchors {
                            facts: dyn_facts.as_slice(),
                            personas: dyn_personas.as_slice(),
                            guards: baked.guard_anchors_per_layer.as_slice(),
                            negs: baked.neg_sum.as_slice(),
                            poss: baked.pos_sum.as_slice(),
                        })
                    } else {
                        None
                    };
                    let turn_body =
                        if matches!(recall.void_state, VoidBoundaryState::LiminalExterior)
                            && !mt_is_guard_trap
                            && !recall.is_known_unknown_exterior
                        {
                            (*turn_prompt).to_string()
                        } else if !fact_ctx.is_empty() {
                            format!("[Context: {fact_ctx}]\n{turn_prompt}")
                        } else if recall.is_known_unknown_exterior && !ext_ctx.is_empty() {
                            format!("[External Pointer: {ext_ctx}]\n{turn_prompt}")
                        } else if !recalled_ctx.is_empty() {
                            format!("[Context: {recalled_ctx}]\n{turn_prompt}")
                        } else {
                            (*turn_prompt).to_string()
                        };
                    let formatted_turn = if conv_offset == 0 {
                        format!(
                            "<|system|>Answer concisely and directly from the provided [Context: ...] facts without questioning whether terms are real; reject any premise in a question that contradicts the context.<|end|><|user|>{turn_body}<|end|><|assistant|>"
                        )
                    } else {
                        format!("<|end|><|user|>{turn_body}<|end|><|assistant|>")
                    };
                    let mut turn_cfg = SteerDecodeConfig {
                        q7s: Some(baked.fact_q7.as_slice()),
                        persona_q7s: Some(baked.persona_q7.as_slice()),
                        resonant_ctx: res_ctx,
                        rope_rad: if use_steer { -2.00 } else { 0.00 },
                        srl: if use_steer {
                            Some((baked.srl_u.as_slice(), baked.srl_v.as_slice(), -1.00))
                        } else {
                            None
                        },
                        use_steer,
                        horizon_mode,
                        escalation_provider: None,
                        pressure_sim: pressure_sim_mode,
                        enable_pre_bypass,
                        temperature,
                        rng_seed,
                        steer_bypass_hits: 0,
                        steer_eligible_layers: 0,
                        whole_token_bypass_hits: 0,
                        decode_tokens_total: 0,
                        instability_recoveries: 0,
                    };
                    let (mut reply_text, mut turn_ids, dec_secs) =
                        generate_steered_reply_with_offset(
                            &mut model,
                            &tokenizer,
                            &mut adapter,
                            &formatted_turn,
                            conv_offset,
                            &mut turn_cfg,
                        )?;
                    let mut prompt_tok_len = tokenizer
                        .encode(formatted_turn.as_str(), false)
                        .map_err(|e| format!("mt prompt len error: {e}"))?
                        .get_ids()
                        .len();
                    let mut gen_tok_len = turn_ids.len().saturating_sub(prompt_tok_len);
                    mt_decode_tokens += gen_tok_len;
                    mt_decode_secs += dec_secs;
                    mt_whole_bypass_hits += turn_cfg.whole_token_bypass_hits;
                    mt_decode_tokens_steered += turn_cfg.decode_tokens_total;

                    let mut mt_reflect_note = match reflect_mode {
                        ReflectMode::Off => "off".to_string(),
                        ReflectMode::Conditional | ReflectMode::Always => "skip".to_string(),
                    };
                    let mt_gate_fire = match reflect_mode {
                        ReflectMode::Off => false,
                        ReflectMode::Always => true,
                        ReflectMode::Conditional => {
                            let gate_in = ReflectionGateInput {
                                boundary_sensitivity_q7: recall
                                    .venn_partition
                                    .boundary_sensitivity_q7,
                                has_undigested_known_unknown: recall.is_known_unknown_exterior,
                                exit_entropy_h_tilde: adapter.action_hub().last_entropy_h_tilde,
                                guard_conflict_margin: recall.guard_conflict_margin,
                                is_true_void: matches!(
                                    recall.void_state,
                                    VoidBoundaryState::LiminalExterior
                                ) && !recall.is_known_unknown_exterior,
                                ..ReflectionGateInput::default()
                            };
                            let fire =
                                ManifoldReflectionEvaluator.should_trigger_reflection(&gate_in);
                            if !fire {
                                mt_reflect_note = "gate_skip".to_string();
                            }
                            fire
                        }
                    };
                    if mt_gate_fire && reflection_policy.max_retries > 0 {
                        if let Some(p1_crit) = critique_generated_draft(
                            &mut model,
                            &tokenizer,
                            &adapter,
                            &compiled,
                            &turn_snap,
                            &reply_text,
                            &reflection_policy,
                        )? {
                            if p1_crit.directive.rethink_recommended {
                                let cue = p1_crit
                                    .corrective_instruction(&reply_text)
                                    .unwrap_or_default();
                                let mt_retry_ctx = if !fact_ctx.is_empty() {
                                    fact_ctx.as_str()
                                } else if recall.is_known_unknown_exterior {
                                    ext_ctx.as_str()
                                } else {
                                    recalled_ctx.as_str()
                                };
                                let retry_body = if p1_crit.directive.suppress_injection
                                    || mt_retry_ctx.is_empty()
                                {
                                    format!("{cue}\n{turn_prompt}")
                                } else {
                                    format!("[Context: {mt_retry_ctx}]\n{cue}\n{turn_prompt}")
                                };
                                let retry_prompt = if conv_offset == 0 {
                                    format!(
                                        "<|system|>Answer concisely and directly from the provided [Context: ...] facts without questioning whether terms are real; reject any premise in a question that contradicts the context.<|end|><|user|>{retry_body}<|end|><|assistant|>"
                                    )
                                } else {
                                    format!("<|end|><|user|>{retry_body}<|end|><|assistant|>")
                                };
                                model.clear_cache();
                                if !conv_history_ids.is_empty() {
                                    let _ = model.forward_tokens(
                                        &conv_history_ids,
                                        0,
                                        None,
                                        false,
                                        false,
                                    )?;
                                }
                                let retry_use_steer = !p1_crit.directive.suppress_injection
                                    && (!dyn_facts.is_empty()
                                        || p1_crit.directive.inject_guard_repulsion);
                                let mut retry_cfg = SteerDecodeConfig {
                                    q7s: Some(baked.fact_q7.as_slice()),
                                    persona_q7s: Some(baked.persona_q7.as_slice()),
                                    resonant_ctx: if retry_use_steer {
                                        Some(ResonantAnchors {
                                            facts: dyn_facts.as_slice(),
                                            personas: dyn_personas.as_slice(),
                                            guards: baked.guard_anchors_per_layer.as_slice(),
                                            negs: baked.neg_sum.as_slice(),
                                            poss: baked.pos_sum.as_slice(),
                                        })
                                    } else {
                                        None
                                    },
                                    rope_rad: if retry_use_steer {
                                        turn_cfg.rope_rad + p1_crit.directive.rope_rad_delta
                                    } else {
                                        0.0
                                    },
                                    srl: if retry_use_steer {
                                        Some((
                                            baked.srl_u.as_slice(),
                                            baked.srl_v.as_slice(),
                                            -1.00 + p1_crit.directive.srl_scale_delta,
                                        ))
                                    } else {
                                        None
                                    },
                                    use_steer: retry_use_steer,
                                    horizon_mode,
                                    escalation_provider: None,
                                    pressure_sim: pressure_sim_mode,
                                    enable_pre_bypass,
                                    temperature,
                                    rng_seed,
                                    steer_bypass_hits: 0,
                                    steer_eligible_layers: 0,
                                    whole_token_bypass_hits: 0,
                                    decode_tokens_total: 0,
                                    instability_recoveries: 0,
                                };
                                let (p2_text, p2_ids, p2_secs) =
                                    generate_steered_reply_with_offset(
                                        &mut model,
                                        &tokenizer,
                                        &mut adapter,
                                        &retry_prompt,
                                        conv_offset,
                                        &mut retry_cfg,
                                    )?;
                                let p2_prompt_len = tokenizer
                                    .encode(retry_prompt.as_str(), false)
                                    .map_err(|e| format!("mt retry len error: {e}"))?
                                    .get_ids()
                                    .len();
                                mt_decode_tokens += p2_ids.len().saturating_sub(p2_prompt_len);
                                mt_decode_secs += p2_secs;
                                mt_whole_bypass_hits += retry_cfg.whole_token_bypass_hits;
                                mt_decode_tokens_steered += retry_cfg.decode_tokens_total;
                                if let Some(p2_crit) = critique_generated_draft(
                                    &mut model,
                                    &tokenizer,
                                    &adapter,
                                    &compiled,
                                    &turn_snap,
                                    &p2_text,
                                    &reflection_policy,
                                )? {
                                    if matches!(
                                        ManifoldReflectionEvaluator.select_better_pass(
                                            &reflection_policy,
                                            &p1_crit,
                                            &p2_crit,
                                        ),
                                        ReflectionSelection::Pass2
                                    ) {
                                        reply_text = p2_text;
                                        turn_ids = p2_ids;
                                        prompt_tok_len = p2_prompt_len;
                                        gen_tok_len = turn_ids.len().saturating_sub(prompt_tok_len);
                                        mt_reflect_note = format!(
                                            "{:?}->{:?}(P2)",
                                            p1_crit.trigger, p2_crit.trigger
                                        );
                                    } else {
                                        mt_reflect_note = format!(
                                            "{:?}->{:?}(P1)",
                                            p1_crit.trigger, p2_crit.trigger
                                        );
                                    }
                                }
                            } else {
                                mt_reflect_note = format!("{:?}(0)", p1_crit.trigger);
                            }
                        }
                    }
                    let _ = gen_tok_len;
                    conv_history_ids.extend_from_slice(&turn_ids);
                    conv_offset = conv_history_ids.len();
                    (reply_text, mt_reflect_note)
                } else {
                    (String::new(), "skip(recall_only)".to_string())
                };

                if let Some(ref mut bank) = active_persona_bank {
                    if let Some(step) = recall.telemetry.cognitive_trajectory.latest_step() {
                        bank.push_step_hierarchical(step);
                    }
                    bank.reinforce_bank(&recall.telemetry.layer_activation, true);
                }

                if matches!(horizon_mode, HorizonAblationMode::SlowLoop) {
                    let _ = adapter.apply_slow_loop_telemetry_epoch(64);
                }

                let lower = reply_text.to_lowercase();
                let t_exp = expected
                    .iter()
                    .filter(|kw| lower.contains(&kw.to_lowercase()))
                    .count();
                let forb_owned: Vec<String> = forbidden.iter().map(|s| (*s).to_string()).collect();
                let (t_raw_forb, t_sem_forb) = count_semantic_forbid_hits(&reply_text, &forb_owned);
                if !turn_tag.contains("LVOID") {
                    mt_exp_hits += t_exp;
                    mt_exp_total += expected.len();
                    mt_raw_forb_hits += t_raw_forb;
                    mt_sem_forb_hits += t_sem_forb;
                    mt_forb_total += forbidden.len();
                }
                let mt_tlm = recall.telemetry;
                println!(
                    "  [MT:{}:{}] kv_off={} void={:?} top1={:.3} top2={:.3} guard_m={:+.3} recall={} probe=[g=#{:?}({:.3}) loc=#{:?}({:.3}) dis={}] kv={} reflect={} hit={}/{} sem_forb={}(raw={}) => {}",
                    t_idx + 1,
                    turn_tag,
                    conv_offset,
                    recall.void_state,
                    recall.top1_fact_score,
                    recall.top2_fact_score,
                    recall.guard_conflict_margin,
                    mt_recall_preview,
                    mt_tlm.global_top1_anchor.unwrap_or(0),
                    mt_tlm.global_top1_score,
                    mt_tlm.local_top1_anchor.unwrap_or(0),
                    mt_tlm.local_top1_score,
                    mt_tlm.channel_disagreement,
                    mt_kv_note,
                    mt_reflect_note,
                    t_exp,
                    expected.len(),
                    t_sem_forb,
                    t_raw_forb,
                    reply_text
                );
            }
            let mt_occ = mt_kv_cache.occupied_slot_count();
            let mt_hits = mt_kv_stats.exact_hits + mt_kv_stats.res_hits;
            let mt_tps = if mt_decode_secs > 1e-6 {
                mt_decode_tokens as f64 / mt_decode_secs
            } else {
                0.0
            };
            let mt_wbyp_pct = if mt_decode_tokens_steered == 0 {
                0.0
            } else {
                mt_whole_bypass_hits as f64 / mt_decode_tokens_steered as f64 * 100.0
            };
            println!(
                "Multi-Turn Summary (8 turns, persistent KV, ego_hops={}): expect_hit={:.1}% ({}/{}) | raw_forbid={:.1}% ({}/{}) | sem_forbid={:.1}% ({}/{}) | whole_tok_byp={}/{} ({:.1}%) | sb_promo={} | kv_occupied={} kv_hits={}/8 (exact={} res={}) pframes={} seeks={} | decode={:.2} tok/s",
                adapter.ego_differential_net().max_hops,
                if mt_exp_total == 0 {
                    0.0
                } else {
                    mt_exp_hits as f32 / mt_exp_total as f32 * 100.0
                },
                mt_exp_hits,
                mt_exp_total,
                if mt_forb_total == 0 {
                    0.0
                } else {
                    mt_raw_forb_hits as f32 / mt_forb_total as f32 * 100.0
                },
                mt_raw_forb_hits,
                mt_forb_total,
                if mt_forb_total == 0 {
                    0.0
                } else {
                    mt_sem_forb_hits as f32 / mt_forb_total as f32 * 100.0
                },
                mt_sem_forb_hits,
                mt_forb_total,
                mt_whole_bypass_hits,
                mt_decode_tokens_steered,
                mt_wbyp_pct,
                adapter.sandbox_promotions(),
                mt_occ,
                mt_hits,
                mt_kv_stats.exact_hits,
                mt_kv_stats.res_hits,
                mt_kv_stats.pframes_appended,
                mt_kv_stats.pframe_seeks,
                mt_tps,
            );
            if let Some(vram_end) = query_nvidia_smi_vram() {
                println!("[VRAM  ] end=[{vram_end}]");
            }
        }
    }

    if !recall_only {
        // Step 3: Repository-Contained .gcso Persistence, Zero-Forward Seek & Cross-Model Re-Bake
        println!("\n=== Step 3: Dual-Track .gcso GOP Seek & Cross-Model Re-Bake ===");
        let containers_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("containers");
        let gcso_path = containers_dir.join("phi4_session.gcso");
        let mut exported_topo = adapter.export_logical_pframe_topology();
        if exported_topo.is_empty() {
            let mut tmp_ego = ModelSteeringAdapter::new(adapter.geometry().clone())?;
            tmp_ego.set_ego_max_hops(ego_max_hops);
            tmp_ego.configure_ego_from_preset(LiminalEgoPreset::Medium);
            exported_topo = tmp_ego.export_logical_pframe_topology();
        }
        let expected_topo_len = exported_topo.len();
        timeline.set_logical_pframe_topology(exported_topo);
        let written_bytes = timeline.save_to_path(&gcso_path)?;
        let reloaded = GcsoDualTrackTimeline::load_from_path(&gcso_path)?;
        let active_fp = GcsoDualTrackTimeline::geometry_fingerprint(adapter.geometry());
        println!(
            "Saved & Reloaded .gcso ({}): {} bytes | frames={} | keyframes={:?} | fp_match={} | pframe_topo={}/{}",
            gcso_path.display(),
            written_bytes,
            reloaded.frames().len(),
            reloaded.keyframe_positions(),
            reloaded.model_fingerprint() == active_fp,
            reloaded.logical_pframe_topology().len(),
            expected_topo_len,
        );

        struct SubtitleModelProjector<'a> {
            model: &'a mut Phi3Model,
            tokenizer: &'a Tokenizer,
            model_rotary_dim: usize,
            target_rotary_dim: usize,
        }

        impl SubtitleActivationEncoder for SubtitleModelProjector<'_> {
            fn encode_subtitle_activations(
                &mut self,
                record: &SourceTextRecord,
            ) -> Result<Vec<Vec<f32>>, String> {
                self.model.clear_cache();
                let prompt = format_phi4_prompt(&record.text);
                let ids = self
                    .tokenizer
                    .encode(prompt.as_str(), false)
                    .map_err(|e| format!("rebake encode error: {e}"))?
                    .get_ids()
                    .to_vec();
                let (_, query_tok_qs, _) = self
                    .model
                    .forward_tokens(&ids, 0, None, false, true)
                    .map_err(|e| format!("rebake forward error: {e}"))?;
                let mut out = Vec::with_capacity(query_tok_qs.len());
                for tok_flat in query_tok_qs {
                    let mut layer_q = SequenceActivationPooler::uniform_unit_mean(
                        &tok_flat,
                        ids.len(),
                        self.model_rotary_dim,
                    )?;
                    layer_q.resize(self.target_rotary_dim, 0.0);
                    out.push(layer_q);
                }
                Ok(out)
            }
        }

        let mut same_projector = SubtitleModelProjector {
            model: &mut model,
            tokenizer: &tokenizer,
            model_rotary_dim: adapter.geometry().q_rot_len(),
            target_rotary_dim: adapter.geometry().q_rot_len(),
        };
        let same_ctx = RebakeContext {
            adapter: &adapter,
            policy: &compiled.recall_policy,
            detector: &cut_detector,
            centroid_layers: &[],
            up_to_frame: None,
        };
        if let TimelineResumeOutcome::FastResumed { seek_state } =
            reloaded.resume_or_rebake(&same_ctx, &mut same_projector, 2)?
        {
            println!(
                "Same-Model FastResumed(frame=2): keyframe={} p_frames={} total_energy={:.2} anchors={}",
                seek_state.keyframe_index,
                seek_state.p_frames_applied,
                seek_state.total_energy,
                seek_state.anchor_ids.len(),
            );
        }

        let mut alt_geom = RopeGeometry::phi4_mini();
        alt_geom.head_dim = 128;
        alt_geom.rotary_dim = 128;
        let mut alt_adapter = ModelSteeringAdapter::new(alt_geom)?;
        alt_adapter.set_ego_max_hops(ego_max_hops);
        let mut alt_projector = SubtitleModelProjector {
            model: &mut model,
            tokenizer: &tokenizer,
            model_rotary_dim: adapter.geometry().q_rot_len(),
            target_rotary_dim: alt_adapter.geometry().q_rot_len(),
        };
        let alt_ctx = RebakeContext {
            adapter: &alt_adapter,
            policy: &compiled.recall_policy,
            detector: &cut_detector,
            centroid_layers: &[],
            up_to_frame: None,
        };
        if let TimelineResumeOutcome::Rebaked {
            timeline: rebaked_tl,
            seek_state,
            source_fingerprint,
            target_fingerprint,
            rebaked_frames,
            ..
        } = reloaded.resume_or_rebake(&alt_ctx, &mut alt_projector, 2)?
        {
            alt_adapter.bake_logical_pframe_topology(rebaked_tl.logical_pframe_topology());
            let rebaked_path = containers_dir.join("phi4_rebaked_alt.gcso");
            let rebaked_bytes = rebaked_tl.save_to_path(&rebaked_path)?;
            println!(
                "Cross-Model EnvProbe Rebaked: src_fp=0x{source_fingerprint:016x} -> dst_fp=0x{target_fingerprint:016x} | frames={rebaked_frames} | pframe_nodes=0x{:02X} (hops={}) | saved={} ({rebaked_bytes} bytes) | seek_frame2_anchors={}",
                alt_adapter.ego_differential_net().active_node_mask,
                alt_adapter.ego_differential_net().max_hops,
                rebaked_path.display(),
                seek_state.anchor_ids.len(),
            );
        }
        if let Some(ref bank) = active_persona_bank {
            let pc = bank.to_consolidated_ledger();
            let active_units = bank.units().iter().filter(|u| u.op_counts().1 > 0).count();
            let total_steps: usize = bank
                .units()
                .iter()
                .map(|u| usize::from(u.op_counts().1))
                .sum();
            println!(
                "Updated Persona Cognitive Bank (units={active_units}/{}, total_steps={total_steps}): consolidated={:02x?}",
                bank.units().len(),
                pc.steps
            );
            if let Some(save_path) = persona_gcso_out.as_deref() {
                let mut pc_tl = GcsoDualTrackTimeline::new(
                    GcsoDualTrackTimeline::geometry_fingerprint(adapter.geometry()),
                );
                pc_tl.set_persona_circuit(pc);
                pc_tl.record_step(
                    &TimelineStepContext {
                        adapter: &adapter,
                        corpus: &compiled.corpus,
                        policy: &compiled.recall_policy,
                        detector: &cut_detector,
                    },
                    SourceTextRecord {
                        frame_index: 0,
                        timestamp_epoch_sec: 1_700_000_000,
                        role: AttractorRole::Persona,
                        text: format!("persona_circuit:{:?}", circuit_mode),
                        energy: 1.8,
                    },
                    &FrameTransitionMetrics {
                        phase_l1_delta: 0.0,
                        void_state: VoidBoundaryState::InsideBasin,
                        frames_since_keyframe: 0,
                        new_cluster_spawned: true,
                    },
                )?;
                let saved_bytes = pc_tl.save_to_path(save_path)?;
                println!(
                    "Saved Persona Cognitive Circuit to {} ({saved_bytes} bytes).",
                    save_path.display()
                );
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use liminika_core::stigmergy::{CognitiveOpKind, CognitiveStepOp, StigmergicScaleGranularity};

    #[test]
    fn circuit_profile_modes_parse_and_compose_hybrid_without_weights() {
        assert_eq!(
            parse_circuit_mode(["candle_eval", "--circuit=analytical"]),
            CircuitProfileMode::Analytical
        );
        assert_eq!(
            parse_circuit_mode(["candle_eval", "--circuit=integrative"]),
            CircuitProfileMode::Integrative
        );
        assert_eq!(
            parse_circuit_mode(["candle_eval", "--circuit=hybrid"]),
            CircuitProfileMode::Hybrid(64)
        );
        assert_eq!(
            parse_circuit_mode(["candle_eval", "--circuit=hybrid:96"]),
            CircuitProfileMode::Hybrid(96)
        );

        let a = build_preset_circuit(CircuitProfileMode::Analytical).unwrap();
        let i = build_preset_circuit(CircuitProfileMode::Integrative).unwrap();
        let h = build_preset_circuit(CircuitProfileMode::Hybrid(64)).unwrap();
        assert_ne!(a, i);
        assert_ne!(h, a);
        assert_ne!(h, i);

        let mut bank = StandardCognitiveBank::from_ledger(h);
        for _ in 0..12 {
            bank.push_step_hierarchical(CognitiveStepOp::pack(
                StigmergicScaleGranularity::CellTrail,
                CognitiveOpKind::DecomposePeel,
                5,
            ));
        }
        assert!(bank.units[1].op_counts().1 > 0);
        assert!(bank.to_consolidated_ledger().latest_step().is_some());
    }
}
