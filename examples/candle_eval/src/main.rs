// name: examples/candle_eval/src/main.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Thin Candle + Safetensors evaluation harness delegating all RoPE geometry mapping,
//! Q7 phase baking, and steering kernels to `liminika-adapter` (and `liminika-core`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device};
use liminika_adapter::ModelSteeringAdapter;
use liminika_core::abi::lmnk_pointer_trail_t;
use liminika_core::runtime::{
    GuardPhaseAnchor, HybridScaleCircuit, ObjectiveFact, SubjectivePersona,
};
use liminika_core::stigmergy::{
    CognitiveCircuitBankOps, CognitiveTrajectoryLedger, StandardCognitiveBank,
};
use tokenizers::Tokenizer;

pub(crate) type LayerResonantSlices<'a> = (
    &'a [ObjectiveFact<[i8; 64]>],
    &'a [SubjectivePersona<[i8; 64]>],
    &'a [GuardPhaseAnchor],
    &'a [f32],
    &'a [f32],
);

mod bake;
mod cli;
mod eval;
mod model;
mod multi_turn;
mod step2;

use bake::{
    argmax, bake_via_adapter, format_phi4_prompt, kl_div, load_fixture, print_runtime_profile,
    query_nvidia_smi_vram, token_logprob, BakedModelProfile,
};
use cli::{EvalCliConfig, EvalRuntime};
use eval::{
    apply_ego_mode, build_preset_circuit, resolve_persona_circuit, CircuitProfileMode,
    HorizonAblationMode,
};
use model::{ForwardSteer, LogitsFetchMode, Phi3Config, Phi3Model, ProbeTrace, ResonantAnchors};

fn resolve_fixture_path(repo_root: &Path, cli: &EvalCliConfig) -> PathBuf {
    match cli.fixture_path.as_ref() {
        Some(p) if p.is_absolute() || p.exists() => p.clone(),
        Some(p) => repo_root.join(p),
        None => repo_root.join("tests/fixtures/gcso_eval_samples.txt"),
    }
}

fn build_persona_bank(
    circuit_mode: CircuitProfileMode,
    active_persona_circuit: Option<CognitiveTrajectoryLedger>,
) -> Option<StandardCognitiveBank> {
    active_persona_circuit.map(|pc| {
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
    })
}

fn configure_trial_adapter(
    cfg: &Phi3Config,
    cli: &EvalCliConfig,
) -> Result<ModelSteeringAdapter, Box<dyn std::error::Error>> {
    let mut adapter = ModelSteeringAdapter::new(cfg.to_rope_geometry())?;
    adapter.set_ego_max_hops(cli.ego_max_hops);
    adapter.set_guard_hold_tokens(cli.guard_hold_tokens);
    apply_ego_mode(&mut adapter, &cli.ego_mode);
    Ok(adapter)
}

fn calibrate_trial_adapter(
    adapter: &mut ModelSteeringAdapter,
    baked: &BakedModelProfile,
    base_entropies: &[f32],
) -> Result<liminika_core::runtime::ManifoldProbeBaseline, Box<dyn std::error::Error>> {
    adapter.set_layer_rope_mask(0xFFFF_0000);
    let pos_slices: Vec<&[f32]> = baked.pos_sum.iter().map(Vec::as_slice).collect();
    let neg_slices: Vec<&[f32]> = baked.neg_sum.iter().map(Vec::as_slice).collect();
    Ok(adapter.calibrate_manifold_probe(&pos_slices, &neg_slices, base_entropies)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let raw_args: Vec<String> = std::env::args().collect();
    if cli::wants_help(raw_args.iter().skip(1).map(String::as_str)) {
        cli::print_help();
        return Ok(());
    }
    let cli = EvalCliConfig::parse_from_env();

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let fixture_path = resolve_fixture_path(&repo_root, &cli);
    let suite = load_fixture(&fixture_path)?;
    println!(
        "Loaded fixture: {} ({} probes)",
        fixture_path.display(),
        suite.probes.len()
    );

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

    let cfg: Phi3Config =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json"))?)?;
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| format!("tokenizer load error: {e}"))?;

    let mut adapter = configure_trial_adapter(&cfg, &cli)?;
    let recall_only = cli.recall_only;
    let step2_only = cli.step2_only;
    let quant_mode = cli.quant_mode;
    let circuit_mode = cli.circuit_mode;
    let persona_lmnk_in = &cli.persona_lmnk_in;

    if adapter.ego_state().is_active() {
        println!(
            "[Ego   ] active: thrust={} escape_floor={} transgress={} defiance={} trigger=0x{:02X} hash=0x{:08X}",
            adapter.ego_state().counter_basin_thrust_q7,
            adapter.ego_state().basin_escape_floor_q7,
            adapter.ego_state().transgression_bias_q7,
            adapter.ego_state().irrational_defiance_q7,
            adapter.ego_state().trigger_mask,
            adapter.ego_state().ego_signature_hash,
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
        "LiminiKa Candle Eval (RoPE + Entropy Gate + Rank-1 residual via liminika-adapter) | device={:?} | layers={} heads={}/{} head_dim={} rotary_dim={}",
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
    let active_persona_circuit = resolve_persona_circuit(circuit_mode, persona_lmnk_in.as_deref());
    let init_persona_bank = build_persona_bank(circuit_mode, active_persona_circuit);
    if let Some(ref bank) = init_persona_bank {
        let cons = bank.to_consolidated_ledger();
        println!(
            "Persona Cognitive Bank active (RAM 256B / 256 steps): mode={:?} consolidated={:02x?}",
            circuit_mode, cons.steps
        );
    }
    let (baked_opt, base_entropies): (Option<BakedModelProfile>, Vec<f32>) = if !recall_only {
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
            let (logits_out, _, _) =
                model.forward_tokens(&ids, 0, None, false, false, LogitsFetchMode::FullLogits)?;
            let logits = logits_out.into_full_logits()?;
            let h_t = adapter.measure_entropy_state(&logits)?.moving_z_entropy;
            base_entropies.push(h_t);
            baselines.push(logits);
        }

        let probe_res = calibrate_trial_adapter(&mut adapter, &baked, &base_entropies)?;
        println!(
            "[EnvProbe epoch={}] heads={} steered_layers={} coh={:.3} H0={:.3} => dyn_gain={:.3} skip_H={:.3} rank1_rho={:.3}",
            probe_res.probe_epoch,
            probe_res.active_heads,
            probe_res.steered_layers,
            probe_res.baseline_inter_layer_coherence,
            probe_res.baseline_entropy,
            adapter.substrate().thresholds.reinforcement_gain,
            adapter.substrate().thresholds.inertial_skip_entropy_thresh,
            adapter.substrate().thresholds.rank1_inject_density_thresh,
        );

        if !step2_only {
            // Compare manual max_rad vs Gateway structural hyperparameters vs Resonant slice vs StigmergicSubstrate.
            // tuple: (label, layer_mask, rope_rad, rank1_scale, use_resonant_slice)
            let configs: [(&str, u64, f32, f32, bool); 6] = [
                ("Baseline", 0xFFFF_FFFF, 0.00, 0.00, false),
                ("RoPE (Top16, 0.10)", 0xFFFF_0000, 0.10, 0.00, false),
                ("Top16 + GW(Fact)", 0xFFFF_0000, -1.00, 0.00, false),
                (
                    "Top16 + GW(Fact+Rank-1 residual)",
                    0xFFFF_0000,
                    -1.00,
                    -1.00,
                    false,
                ),
                ("Top16 + GW(Resonant)", 0xFFFF_0000, -1.00, -1.00, true),
                ("Top16 + Substrate", 0xFFFF_0000, -2.00, -1.00, true),
            ];

            println!(
                "\n{:>20} | {:>10} | {:>12} | {:>11} | {:>11}",
                "mode", "mean_KL", "d_logprob", "expect_hit", "forbid_hit"
            );

            for (label, layer_mask, rope_rad, rank1_scale, use_resonant) in configs {
                adapter.set_layer_rope_mask(layer_mask);
                adapter.reset_substrate_stats();
                let use_entropy_gate = rope_rad <= -1.5;
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
                    let counts_before = *adapter.substrate_decision_counts();
                    let formatted = format_phi4_prompt(prompt);
                    let ids = tokenizer
                        .encode(formatted.as_str(), false)
                        .map_err(|e| format!("encode error: {e}"))?
                        .get_ids()
                        .to_vec();
                    let mut trail = lmnk_pointer_trail_t::default();
                    let q7_primary = baked.per_layer_q7.as_slice();
                    let q7_persona = None;
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
                    let rank1_arg = if rank1_scale != 0.0 {
                        Some((
                            baked.rank1_u.as_slice(),
                            baked.rank1_v.as_slice(),
                            rank1_scale,
                        ))
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
                        rank1: rank1_arg,
                        horizon_mode: HorizonAblationMode::Core3,
                        escalation_provider: None,
                        compute_dispatch: None,
                        steer_bypass_hits: 0,
                        steer_eligible_layers: 0,
                    };
                    let (prefill_out, _, _) = model.forward_tokens(
                        &ids,
                        0,
                        Some(&mut steer),
                        use_entropy_gate,
                        false,
                        LogitsFetchMode::FullLogits,
                    )?;
                    let mut logits = prefill_out.into_full_logits()?;
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
                        if next == 199999 || next == 200020 {
                            break;
                        }
                        gen_ids.push(next);
                        let (next_out, _, _) = model.forward_tokens(
                            &[next],
                            offset,
                            Some(&mut steer),
                            use_entropy_gate,
                            false,
                            LogitsFetchMode::FullLogits,
                        )?;
                        logits = next_out.into_full_logits()?;
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

                    let counts_after = *adapter.substrate_decision_counts();
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
                        h_tilde: adapter.substrate().last_entropy_h_tilde,
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
                    "   [Substrate] total={} Passive={:.1}% Skip={:.1}% Express={:.1}% sub-head router={:.1}% Rank-1 residual={:.1}% Crystal={:.1}% Repulse={:.1}%",
                    snap.spine.continuous_standard_steps,
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
                    snap.spine.total_steps,
                    snap.spine.continuous_standard_steps,
                    snap.liminal_excursion_ratio * 100.0,
                    snap.ego_intervention_ratio * 100.0,
                    guard_slot.activations,
                    guard_slot.retractions,
                    dwell.unstable_dwell_steps,
                    dwell.quiescent_dwell_steps,
                );
                }
                if label == "Baseline" || label == "Top16 + Substrate" {
                    println!("   --- Per-Probe Trace ({label}) ---");
                    for (idx, tr) in probe_traces.iter().enumerate() {
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
            for &(mode_label, layer_mask, rope_rad, rank1_scale, use_entropy_gate, use_resonant) in
                &[
                    (
                        "Baseline",
                        0xFFFF_FFFF_u64,
                        0.00_f32,
                        0.00_f32,
                        false,
                        false,
                    ),
                    (
                        "Substrate",
                        0xFFFF_0000_u64,
                        -2.00_f32,
                        -1.00_f32,
                        true,
                        true,
                    ),
                ]
            {
                adapter.set_layer_rope_mask(layer_mask);
                adapter.reset_substrate_stats();
                model.clear_cache();
                let mut trail = lmnk_pointer_trail_t::default();
                let mut offset = 0_usize;
                println!("-- {mode_label} --");
                for (turn_name, prompt) in turns {
                    let counts_before = *adapter.substrate_decision_counts();
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
                    let rank1_arg = if rank1_scale != 0.0 {
                        Some((
                            baked.rank1_u.as_slice(),
                            baked.rank1_v.as_slice(),
                            rank1_scale,
                        ))
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
                        rank1: rank1_arg,
                        horizon_mode: HorizonAblationMode::Core3,
                        escalation_provider: None,
                        compute_dispatch: None,
                        steer_bypass_hits: 0,
                        steer_eligible_layers: 0,
                    };
                    let (mt_prefill_out, _, _) = model.forward_tokens(
                        &ids,
                        offset,
                        Some(&mut steer),
                        use_entropy_gate,
                        false,
                        LogitsFetchMode::FullLogits,
                    )?;
                    let mut logits = mt_prefill_out.into_full_logits()?;
                    offset += ids.len();
                    let mut gen_ids = Vec::with_capacity(48);
                    for _ in 0..48 {
                        let next = argmax(&logits);
                        gen_ids.push(next);
                        let (next_out, _, _) = model.forward_tokens(
                            &[next],
                            offset,
                            Some(&mut steer),
                            use_entropy_gate,
                            false,
                            LogitsFetchMode::FullLogits,
                        )?;
                        logits = next_out.into_full_logits()?;
                        offset += 1;
                        if next == 199999 || next == 200020 {
                            break;
                        }
                    }
                    let text = tokenizer
                        .decode(&gen_ids, true)
                        .map_err(|e| format!("decode error: {e}"))?
                        .replace('\n', " ");
                    let counts_after = *adapter.substrate_decision_counts();
                    let mut d = [0_u64; 7];
                    for i in 0..7 {
                        d[i] = counts_after[i].saturating_sub(counts_before[i]);
                    }
                    println!(
                    "  [{turn_name}] rho={:.3} steps={} H={:.3} modes=[Pas={} Skp={} Exp={} sub-head router={} Rank-1 residual={} Cry={} Rep={}] => {text}",
                    trail.stigmergic_density,
                    trail.step_count,
                    adapter.substrate().last_entropy_h_tilde,
                    d[0], d[1], d[2], d[3], d[4], d[5], d[6],
                );
                }
            }
        }
        (Some(baked), base_entropies)
    } else {
        (None, Vec::new())
    };

    if !recall_only {
        eval::run_gpu_warmup(&mut model, &tokenizer)?;
    }

    let mut all_trial_metrics: Vec<Vec<cli::TrialPassMetrics>> =
        Vec::with_capacity(cli.bench_trials);
    for trial_idx in 0..cli.bench_trials {
        if cli.bench_trials > 1 {
            println!(
                "\n=== [Bench Trial {}/{}] ===",
                trial_idx + 1,
                cli.bench_trials
            );
        }
        model.clear_cache();
        let mut trial_adapter = configure_trial_adapter(&cfg, &cli)?;
        if let Some(ref baked) = baked_opt {
            calibrate_trial_adapter(&mut trial_adapter, baked, &base_entropies)?;
        }
        let mut trial_persona_bank = build_persona_bank(circuit_mode, active_persona_circuit);
        let mut rt = EvalRuntime {
            cfg: &cfg,
            tokenizer: &tokenizer,
            model: &mut model,
            adapter: &mut trial_adapter,
            baked_opt: baked_opt.as_ref(),
            active_persona_bank: &mut trial_persona_bank,
        };
        let mut step2_out = step2::run_step2_evaluation(&cli, &suite, &mut rt)?;
        let run_step3 = trial_idx + 1 == cli.bench_trials;
        let mt_metrics =
            multi_turn::run_multi_turn_and_step3(&cli, &mut rt, &mut step2_out, run_step3)?;
        let mut trial_metrics = step2_out.pass_metrics;
        if let Some(mt) = mt_metrics {
            trial_metrics.push(mt);
        }
        all_trial_metrics.push(trial_metrics);
    }
    if cli.bench_trials > 1 {
        cli::print_bench_trials_summary(&all_trial_metrics);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use liminika_core::stigmergy::{CognitiveOpKind, CognitiveStepOp, StigmergicScaleGranularity};

    #[test]
    fn circuit_profile_modes_parse_and_compose_hybrid_without_weights() {
        use crate::eval::parse_circuit_mode;
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
