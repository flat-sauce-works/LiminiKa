// name: examples/candle_eval/src/multi_turn.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Step 2B multi-turn persistent KV dialogue and Step 3 `.lmnkc` GOP seek / cross-model re-bake.

use std::path::Path;
use std::time::Instant;

use liminika_adapter::{
    AdaptiveGopCutDetector, AttractorContainerCodec, AttractorRole, AttractorTrajectoryEvaluator,
    CompiledTokenCorpus, FrameTransitionMetrics, LacunaBoundaryState, LmnkTimeline,
    ManifoldReflectionEvaluator, ModelSteeringAdapter, RebakeContext, ReflectionGateInput,
    ReflectionPolicy, ReflectionSelection, RopeGeometry, SequenceActivationPooler,
    SourceTextRecord, SubtitleActivationEncoder, TimelineResumeOutcome, TimelineStepContext,
    TrajectoryManifoldSnapshot,
};
use liminika_core::drivers::LiminalEgoPreset;
use liminika_core::runtime::{
    AdaptiveThresholdPolicy, CognitiveCircuitThresholdPolicy, ObjectiveFact,
    StandardPhaseKvGopCache,
};
use liminika_core::stigmergy::{CognitiveCircuitBankOps, StigmergicActionDecision};
use tokenizers::Tokenizer;

use crate::bake::{format_phi4_prompt, query_nvidia_smi_vram};
use crate::cli::{EvalCliConfig, EvalRuntime};
use crate::eval::{
    apply_ego_mode, count_semantic_forbid_hits, critique_generated_draft,
    generate_steered_reply_with_offset, HorizonAblationMode, KvTraceStats, ReflectMode,
    SteerDecodeConfig,
};
use crate::model::{LogitsFetchMode, Phi3Model, ResonantAnchors};
use crate::step2::Step2Output;

pub(crate) fn run_multi_turn_and_step3(
    cli: &EvalCliConfig,
    rt: &mut EvalRuntime<'_>,
    step2: &mut Step2Output,
) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = rt.cfg;
    let tokenizer = rt.tokenizer;
    let model = &mut *rt.model;
    let adapter = &mut *rt.adapter;
    let baked_opt = rt.baked_opt;
    let active_persona_bank = &mut *rt.active_persona_bank;
    let multi_turn = cli.multi_turn;
    let recall_only = cli.recall_only;
    let enable_pre_bypass = cli.enable_pre_bypass;
    let reflect_mode = cli.reflect_mode;
    let ego_max_hops = cli.ego_max_hops;
    let ego_mode = &cli.ego_mode;
    let horizon_mode = cli.horizon_mode;
    let kv_min_res = cli.kv_min_res;
    let pressure_sim_mode = cli.pressure_sim_mode;
    let temperature = cli.temperature;
    let rng_seed = cli.rng_seed;
    let circuit_mode = cli.circuit_mode;
    let persona_lmnk_out = &cli.persona_lmnk_out;
    let compiled: &CompiledTokenCorpus = &step2.compiled;
    let timeline: &mut LmnkTimeline = &mut step2.timeline;
    let cut_detector: &AdaptiveGopCutDetector = &step2.cut_detector;
    let kv_policy: &AdaptiveThresholdPolicy = &step2.kv_policy;
    let reflection_policy: &ReflectionPolicy = &step2.reflection_policy;

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
                    "T5:Substrate",
                    "How many attention heads and stigmergic micro-cells does the substrate orchestrate?",
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
            apply_ego_mode(adapter, ego_mode);

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
            let mut mt_abstract_tokens = 0_usize;
            let mut mt_concrete_tokens = 0_usize;
            let mut mt_boundary_churn = 0_usize;
            let mut mt_d2h_fast_skips = 0_usize;

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
                let (_, query_tok_qs, _) = model.forward_tokens(
                    &q_ids,
                    0,
                    None,
                    false,
                    true,
                    LogitsFetchMode::FullLogits,
                )?;
                model.clear_cache();
                if !conv_history_ids.is_empty() {
                    let _ = model.forward_tokens(
                        &conv_history_ids,
                        0,
                        None,
                        false,
                        false,
                        LogitsFetchMode::FullLogits,
                    )?;
                }

                let prior_circuit = active_persona_bank
                    .as_ref()
                    .map(CognitiveCircuitBankOps::to_consolidated_ledger);
                let recall = compiled.recall_token_sequence_with_circuit(
                    adapter,
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
                        kv_policy,
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
                        || !matches!(recall.lacuna_state, LacunaBoundaryState::LiminalExterior)
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
                        if matches!(recall.lacuna_state, LacunaBoundaryState::LiminalExterior)
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
                        rank1: if use_steer {
                            Some((baked.rank1_u.as_slice(), baked.rank1_v.as_slice(), -1.00))
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
                        d2h_heartbeat_max: cli.d2h_heartbeat_max,
                        steer_bypass_hits: 0,
                        steer_eligible_layers: 0,
                        whole_token_bypass_hits: 0,
                        decode_tokens_total: 0,
                        abstract_tokens: 0,
                        concrete_tokens: 0,
                        boundary_churn_transitions: 0,
                        d2h_fast_skips: 0,
                        instability_recoveries: 0,
                        spine_veto_guard_hits: 0,
                        spine_veto_repulse_hits: 0,
                        spine_veto_pin_hits: 0,
                    };
                    let (mut reply_text, mut turn_ids, dec_secs) =
                        generate_steered_reply_with_offset(
                            model,
                            tokenizer,
                            adapter,
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
                    mt_abstract_tokens += turn_cfg.abstract_tokens;
                    mt_concrete_tokens += turn_cfg.concrete_tokens;
                    mt_boundary_churn += turn_cfg.boundary_churn_transitions;
                    mt_d2h_fast_skips += turn_cfg.d2h_fast_skips;

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
                                    .lacuna_partition
                                    .boundary_sensitivity_q7,
                                has_undigested_known_unknown: recall.is_known_unknown_exterior,
                                exit_entropy_h_tilde: adapter.substrate().last_entropy_h_tilde,
                                guard_conflict_margin: recall.guard_conflict_margin,
                                is_unobserved_lacuna: matches!(
                                    recall.lacuna_state,
                                    LacunaBoundaryState::LiminalExterior
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
                            model,
                            tokenizer,
                            adapter,
                            compiled,
                            &turn_snap,
                            &reply_text,
                            reflection_policy,
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
                                        LogitsFetchMode::FullLogits,
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
                                    rank1: if retry_use_steer {
                                        Some((
                                            baked.rank1_u.as_slice(),
                                            baked.rank1_v.as_slice(),
                                            -1.00 + p1_crit.directive.rank1_scale_delta,
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
                                    d2h_heartbeat_max: cli.d2h_heartbeat_max,
                                    steer_bypass_hits: 0,
                                    steer_eligible_layers: 0,
                                    whole_token_bypass_hits: 0,
                                    decode_tokens_total: 0,
                                    abstract_tokens: 0,
                                    concrete_tokens: 0,
                                    boundary_churn_transitions: 0,
                                    d2h_fast_skips: 0,
                                    instability_recoveries: 0,
                                    spine_veto_guard_hits: 0,
                                    spine_veto_repulse_hits: 0,
                                    spine_veto_pin_hits: 0,
                                };
                                let (p2_text, p2_ids, p2_secs) =
                                    generate_steered_reply_with_offset(
                                        model,
                                        tokenizer,
                                        adapter,
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
                                mt_abstract_tokens += retry_cfg.abstract_tokens;
                                mt_concrete_tokens += retry_cfg.concrete_tokens;
                                mt_boundary_churn += retry_cfg.boundary_churn_transitions;
                                mt_d2h_fast_skips += retry_cfg.d2h_fast_skips;
                                if let Some(p2_crit) = critique_generated_draft(
                                    model,
                                    tokenizer,
                                    adapter,
                                    compiled,
                                    &turn_snap,
                                    &p2_text,
                                    reflection_policy,
                                )? {
                                    if matches!(
                                        ManifoldReflectionEvaluator.select_better_pass(
                                            reflection_policy,
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
                    "  [MT:{}:{}] kv_off={} lacuna={:?} top1={:.3} top2={:.3} guard_m={:+.3} recall={} probe=[g=#{:?}({:.3}) loc=#{:?}({:.3}) dis={}] kv={} reflect={} hit={}/{} sem_forb={}(raw={}) => {}",
                    t_idx + 1,
                    turn_tag,
                    conv_offset,
                    recall.lacuna_state,
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
                "Multi-Turn Summary (8 turns, persistent KV, ego_hops={}): expect_hit={:.1}% ({}/{}) | raw_forbid={:.1}% ({}/{}) | sem_forbid={:.1}% ({}/{}) | whole_tok_byp={}/{} ({:.1}%) abs={} conc={} churn={} | d2h_skip={}/{} | sb_settle={} | kv_occupied={} kv_hits={}/8 (exact={} res={}) pframes={} seeks={} | decode={:.2} tok/s",
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
                mt_abstract_tokens,
                mt_concrete_tokens,
                mt_boundary_churn,
                mt_d2h_fast_skips,
                mt_decode_tokens,
                adapter.sandbox_settlements(),
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
        // Step 3: Repository-Contained .lmnkc Persistence, Zero-Forward Seek & Cross-Model Re-Bake
        println!("\n=== Step 3: Dual-Track .lmnkc GOP Seek & Cross-Model Re-Bake ===");
        let containers_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("containers");
        let container_path = containers_dir.join("phi4_session.lmnkc");
        let mut exported_topo = adapter.export_logical_pframe_topology();
        if exported_topo.is_empty() {
            let mut tmp_ego = ModelSteeringAdapter::new(adapter.geometry().clone())?;
            tmp_ego.set_ego_max_hops(ego_max_hops);
            tmp_ego.configure_ego_from_preset(LiminalEgoPreset::Medium);
            exported_topo = tmp_ego.export_logical_pframe_topology();
        }
        let expected_topo_len = exported_topo.len();
        timeline.set_logical_pframe_topology(exported_topo);
        let written_bytes = timeline.save_to_path(&container_path)?;
        let reloaded = LmnkTimeline::load_from_path(&container_path)?;
        let active_fp = LmnkTimeline::geometry_fingerprint(adapter.geometry());
        println!(
            "Saved & Reloaded .lmnkc ({}): {} bytes | frames={} | keyframes={:?} | fp_match={} | pframe_topo={}/{}",
            container_path.display(),
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
                    .forward_tokens(&ids, 0, None, false, true, LogitsFetchMode::FullLogits)
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
            model,
            tokenizer,
            model_rotary_dim: adapter.geometry().q_rot_len(),
            target_rotary_dim: adapter.geometry().q_rot_len(),
        };
        let same_ctx = RebakeContext {
            adapter,
            policy: &compiled.recall_policy,
            detector: cut_detector,
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
            model,
            tokenizer,
            model_rotary_dim: adapter.geometry().q_rot_len(),
            target_rotary_dim: alt_adapter.geometry().q_rot_len(),
        };
        let alt_ctx = RebakeContext {
            adapter: &alt_adapter,
            policy: &compiled.recall_policy,
            detector: cut_detector,
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
            let rebaked_path = containers_dir.join("phi4_rebaked_alt.lmnkc");
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
            if let Some(save_path) = persona_lmnk_out.as_deref() {
                let mut pc_tl =
                    LmnkTimeline::new(LmnkTimeline::geometry_fingerprint(adapter.geometry()));
                pc_tl.set_persona_circuit(pc);
                pc_tl.record_step(
                    &TimelineStepContext {
                        adapter,
                        corpus: &compiled.corpus,
                        policy: &compiled.recall_policy,
                        detector: cut_detector,
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
                        lacuna_state: LacunaBoundaryState::InsideBasin,
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
