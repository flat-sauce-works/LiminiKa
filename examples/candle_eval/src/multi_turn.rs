// name: examples/candle_eval/src/multi_turn.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Step 2B multi-turn persistent KV dialogue and Step 3 `.lmnkc` GOP seek / cross-model re-bake.

use std::time::Instant;

use liminika_adapter::{
    AdaptiveGopCutDetector, AttractorContainerCodec, AttractorRole, AttractorTrajectoryEvaluator,
    CompiledTokenCorpus, FrameTransitionMetrics, LacunaBoundaryState, LmnkTimeline,
    ManifoldReflectionEvaluator, ModelSteeringAdapter, PooledSubtitleEncoder, RebakeContext,
    ReflectionPolicy, ReflectionSelection, SourceTextRecord, TimelineResumeOutcome,
    TimelineStepContext, TurnOrchestrator, TurnPrepareInput,
};
use liminika_core::drivers::LiminalEgoPreset;
use liminika_core::runtime::{AdaptiveThresholdPolicy, StandardPhaseKvGopCache};
use liminika_core::stigmergy::CognitiveCircuitBankOps;

use crate::bake::{LmTokenEncoder, query_nvidia_smi_vram};
use crate::cli::{EvalCliConfig, EvalRuntime, TrialPassMetrics};
use crate::eval::{
    HorizonAblationMode, KvTraceStats, ReflectMode, SteerDecodeConfig, apply_ego_mode,
    count_semantic_forbid_hits, critique_generated_draft, generate_steered_reply_with_offset,
};
use crate::metrics::summarize_dialogue_quality;
use crate::model::{KvCacheMode, LogitsFetchMode, ResonantAnchors};
use crate::step2::Step2Output;

pub(crate) fn run_multi_turn_and_step3(
    cli: &EvalCliConfig,
    rt: &mut EvalRuntime<'_>,
    step2: &mut Step2Output,
    run_step3: bool,
) -> Result<Option<TrialPassMetrics>, Box<dyn std::error::Error>> {
    let tokenizer = rt.tokenizer;
    let chat = rt.chat;
    let model_arch_label = rt.cfg.arch.label();
    let layer_mask_override = cli.resolve_layer_mask_override(rt.cfg.num_hidden_layers);
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
    let compiled: &CompiledTokenCorpus = rt.compiled;
    let timeline: &mut LmnkTimeline = &mut step2.timeline;
    let cut_detector: &AdaptiveGopCutDetector = &step2.cut_detector;
    let kv_policy: &AdaptiveThresholdPolicy = &step2.kv_policy;
    let reflection_policy: &ReflectionPolicy = &step2.reflection_policy;
    let mut mt_metrics: Option<TrialPassMetrics> = None;

    if multi_turn && (recall_only || baked_opt.is_some()) {
        {
            println!(
                "\n=== Step 2B: Full Multi-Turn Dialogue Eval (Persistent KV + RamPhaseKvGopCache) ==="
            );
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
            adapter.set_guard_hold_tokens(cli.guard_hold_tokens);
            adapter.reanchor_compute_pressure(8192);
            adapter.set_ego_max_hops(ego_max_hops);
            apply_ego_mode(adapter, ego_mode);

            let mut mt_kv_cache = StandardPhaseKvGopCache::new();
            let mut mt_kv_stats = KvTraceStats::default();
            let mut mt_exp_hits = 0_usize;
            let mut mt_exp_total = 0_usize;
            let mut mt_raw_forb_hits = 0_usize;
            let mut mt_sem_forb_hits = 0_usize;
            let mut mt_forb_total = 0_usize;
            let mut mt_ood_bleed_hits = 0_usize;
            let mut mt_ood_total = 0_usize;
            let mut mt_decode_tokens = 0_usize;
            let mut mt_prefill_secs = 0.0_f64;
            let mut mt_decode_secs = 0.0_f64;
            let mut mt_prompts_timed = 0_usize;
            let mut mt_replies: Vec<String> = Vec::with_capacity(dialogue_turns.len());

            model.clear_cache();
            let mut conv_offset = 0_usize;

            for (t_idx, (turn_tag, turn_prompt, expected, forbidden)) in
                dialogue_turns.iter().enumerate()
            {
                let turn_start_kv = model.kv_len();
                debug_assert_eq!(turn_start_kv, conv_offset);

                // Run query-only probe in `KvCacheMode::IsolatedProbe` so persistent
                // conversation KV remains intact with zero history re-prefill.
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
                    KvCacheMode::IsolatedProbe,
                )?;
                debug_assert_eq!(model.kv_len(), turn_start_kv);

                let t_mt_prep = Instant::now();
                let plan = TurnOrchestrator::prepare_turn(
                    compiled,
                    adapter,
                    &mut mt_kv_cache,
                    &TurnPrepareInput {
                        query_token_qs: &query_tok_qs,
                        query_seq_len: q_ids.len(),
                        token_include_mask: None,
                        turn_index: (t_idx + 1) as u32,
                        kv_policy,
                        kv_min_resonance_override: kv_min_res,
                        persona_bank: active_persona_bank.as_ref(),
                        layer_steer_mask_override: layer_mask_override,
                    },
                )?;
                let mt_kv_note = mt_kv_stats.record_outcome(
                    plan.kv_outcome(),
                    t_mt_prep.elapsed().as_secs_f64() * 1_000_000.0,
                    &plan.gateway_frame().phase_deltas,
                );

                let fact_ctx = plan.self_ref_context();
                let persona_ctx = plan.persona_context();
                let ext_ctx = plan.exterior_context();
                let recalled_ctx = match (fact_ctx.is_empty(), persona_ctx.is_empty()) {
                    (false, false) => format!("{fact_ctx} {persona_ctx}"),
                    (false, true) => fact_ctx.to_string(),
                    (true, false) => persona_ctx.to_string(),
                    (true, true) => String::new(),
                };
                // Keep `TurnSteeringPlan` strictly partitioned (`S & O` vs `O \ S`) while
                // selecting the exterior sandbox slice explicitly for RTX 4070 baseline parity.
                let steer_facts = if plan.has_self_ref_anchors() {
                    plan.self_ref_anchors_per_layer()
                } else if plan.is_known_unknown_exterior() {
                    plan.exterior_sandbox_anchors_per_layer()
                } else {
                    plan.self_ref_anchors_per_layer()
                };
                let has_steer_facts = steer_facts.iter().any(|layer| !layer.is_empty());
                let dyn_personas = plan.persona_anchors_per_layer();
                let mt_recall_preview = if plan.self_ref_anchor_ids().is_empty() {
                    "none".to_string()
                } else {
                    plan.self_ref_anchor_ids()
                        .iter()
                        .map(|id| format!("#{id}"))
                        .collect::<Vec<_>>()
                        .join("+")
                };
                let (reply_text, mt_reflect_note) = if let Some(baked) =
                    baked_opt.as_ref().filter(|_| !recall_only)
                {
                    let mt_is_guard_trap = plan.is_guard_trap();
                    let use_steer = plan.steering_eligible();
                    let res_ctx = if use_steer {
                        Some(ResonantAnchors {
                            facts: steer_facts,
                            personas: dyn_personas,
                            guards: plan.guard_anchors_per_layer(),
                            negs: baked.repulsive_centroid_per_layer.as_slice(),
                            poss: baked.attractive_centroid_per_layer.as_slice(),
                        })
                    } else {
                        None
                    };
                    let turn_body = if plan.is_unobserved_lacuna() && !mt_is_guard_trap {
                        (*turn_prompt).to_string()
                    } else if !fact_ctx.is_empty() {
                        format!("[Context: {fact_ctx}]\n{turn_prompt}")
                    } else if plan.is_known_unknown_exterior() && !ext_ctx.is_empty() {
                        format!("[External Pointer: {ext_ctx}]\n{turn_prompt}")
                    } else if !recalled_ctx.is_empty() {
                        format!("[Context: {recalled_ctx}]\n{turn_prompt}")
                    } else {
                        (*turn_prompt).to_string()
                    };
                    const MT_SYSTEM_PROMPT: &str = "Answer concisely and directly from the provided [Conte\
                     xt: ...] facts without questioning whether terms are real; reject any premise in a qu\
                     estion that contradicts the context.";
                    let formatted_turn = if turn_start_kv == 0 {
                        chat.format_first_turn_with_system(MT_SYSTEM_PROMPT, &turn_body)
                    } else {
                        chat.format_followup_turn(&turn_body)
                    };
                    let effective_layer_mask = plan.effective_layer_steer_mask(adapter);
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
                        logits_observe_stride: cli.logits_observe_stride,
                        layer_steer_mask: Some(effective_layer_mask),
                        probe_frame: cli.probe_frame,
                        phase_clamp_override: cli.phase_clamp_rad,
                        random_push_seed: cli.random_push_control.then_some(rng_seed),
                        max_tokens: cli.max_tokens,
                        prefill_chunk_size: cli.prefill_chunk_size,
                        capture_first_logits: false,
                        first_prefill_logits: None,
                        exit_entropy_h_tilde: 0.0,
                        steer_bypass_hits: 0,
                        steer_eligible_layers: 0,
                        spine_veto_guard_hits: 0,
                        spine_veto_repulse_hits: 0,
                        spine_veto_pin_hits: 0,
                    };
                    let (mut reply_text, p1_turn_ids, dec_timing) =
                        generate_steered_reply_with_offset(
                            model,
                            tokenizer,
                            chat,
                            adapter,
                            &formatted_turn,
                            turn_start_kv,
                            &mut turn_cfg,
                        )?;
                    debug_assert_eq!(model.kv_len(), turn_start_kv + p1_turn_ids.len());
                    let prompt_tok_len = tokenizer
                        .encode(formatted_turn.as_str(), false)
                        .map_err(|e| format!("mt prompt len error: {e}"))?
                        .get_ids()
                        .len();
                    let gen_tok_len = p1_turn_ids.len().saturating_sub(prompt_tok_len);
                    mt_decode_tokens += gen_tok_len;
                    mt_prefill_secs += dec_timing.prefill_secs;
                    mt_decode_secs += dec_timing.decode_secs;
                    mt_prompts_timed += 1;

                    let mut mt_reflect_note = match reflect_mode {
                        ReflectMode::Off => "off".to_string(),
                        ReflectMode::Conditional | ReflectMode::Always => "skip".to_string(),
                    };
                    let mt_gate_fire = match reflect_mode {
                        ReflectMode::Off => false,
                        ReflectMode::Always => true,
                        ReflectMode::Conditional => {
                            let gate_in = plan.reflection_gate_input(turn_cfg.exit_entropy_h_tilde);
                            let fire =
                                ManifoldReflectionEvaluator.should_trigger_reflection(&gate_in);
                            if !fire {
                                mt_reflect_note = "gate_skip".to_string();
                            }
                            fire
                        }
                    };
                    if mt_gate_fire
                        && reflection_policy.max_retries > 0
                        && let Some(p1_crit) = critique_generated_draft(
                            model,
                            tokenizer,
                            adapter,
                            compiled,
                            plan.query_snapshot(),
                            &reply_text,
                            reflection_policy,
                        )?
                    {
                        if p1_crit.directive.rethink_recommended {
                            let cue = p1_crit
                                .corrective_instruction(&reply_text)
                                .unwrap_or_default();
                            let mt_retry_ctx = if !fact_ctx.is_empty() {
                                fact_ctx
                            } else if plan.is_known_unknown_exterior() {
                                ext_ctx
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
                            let retry_prompt = if turn_start_kv == 0 {
                                chat.format_first_turn_with_system(MT_SYSTEM_PROMPT, &retry_body)
                            } else {
                                chat.format_followup_turn(&retry_body)
                            };
                            model.truncate_cache(turn_start_kv)?;
                            debug_assert_eq!(model.kv_len(), turn_start_kv);
                            let retry_use_steer = !p1_crit.directive.suppress_injection
                                && (has_steer_facts || p1_crit.directive.inject_guard_repulsion);
                            let mut retry_cfg = SteerDecodeConfig {
                                q7s: Some(baked.fact_q7.as_slice()),
                                persona_q7s: Some(baked.persona_q7.as_slice()),
                                resonant_ctx: if retry_use_steer {
                                    Some(ResonantAnchors {
                                        facts: steer_facts,
                                        personas: dyn_personas,
                                        guards: plan.guard_anchors_per_layer(),
                                        negs: baked.repulsive_centroid_per_layer.as_slice(),
                                        poss: baked.attractive_centroid_per_layer.as_slice(),
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
                                logits_observe_stride: cli.logits_observe_stride,
                                layer_steer_mask: Some(effective_layer_mask),
                                probe_frame: cli.probe_frame,
                                phase_clamp_override: cli.phase_clamp_rad,
                                random_push_seed: cli.random_push_control.then_some(rng_seed),
                                max_tokens: cli.max_tokens,
                                prefill_chunk_size: cli.prefill_chunk_size,
                                capture_first_logits: false,
                                first_prefill_logits: None,
                                exit_entropy_h_tilde: 0.0,
                                steer_bypass_hits: 0,
                                steer_eligible_layers: 0,
                                spine_veto_guard_hits: 0,
                                spine_veto_repulse_hits: 0,
                                spine_veto_pin_hits: 0,
                            };
                            let (p2_text, p2_ids, p2_timing) = generate_steered_reply_with_offset(
                                model,
                                tokenizer,
                                chat,
                                adapter,
                                &retry_prompt,
                                turn_start_kv,
                                &mut retry_cfg,
                            )?;
                            debug_assert_eq!(model.kv_len(), turn_start_kv + p2_ids.len());
                            let p2_prompt_len = tokenizer
                                .encode(retry_prompt.as_str(), false)
                                .map_err(|e| format!("mt retry len error: {e}"))?
                                .get_ids()
                                .len();
                            mt_decode_tokens += p2_ids.len().saturating_sub(p2_prompt_len);
                            mt_prefill_secs += p2_timing.prefill_secs;
                            mt_decode_secs += p2_timing.decode_secs;
                            mt_prompts_timed += 1;
                            let adopt_p2 = if let Some(p2_crit) = critique_generated_draft(
                                model,
                                tokenizer,
                                adapter,
                                compiled,
                                plan.query_snapshot(),
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
                                    mt_reflect_note =
                                        format!("{:?}->{:?}(P2)", p1_crit.trigger, p2_crit.trigger);
                                    true
                                } else {
                                    mt_reflect_note =
                                        format!("{:?}->{:?}(P1)", p1_crit.trigger, p2_crit.trigger);
                                    false
                                }
                            } else {
                                false
                            };
                            if !adopt_p2 {
                                model.truncate_cache(turn_start_kv)?;
                                let _ = model.forward_tokens(
                                    &p1_turn_ids,
                                    turn_start_kv,
                                    None,
                                    false,
                                    false,
                                    LogitsFetchMode::FullLogits,
                                    KvCacheMode::Append,
                                )?;
                                debug_assert_eq!(model.kv_len(), turn_start_kv + p1_turn_ids.len());
                            }
                        } else {
                            mt_reflect_note = format!("{:?}(0)", p1_crit.trigger);
                        }
                    }
                    conv_offset = model.kv_len();
                    (reply_text, mt_reflect_note)
                } else {
                    (String::new(), "skip(recall_only)".to_string())
                };

                if let Some(bank) = active_persona_bank {
                    bank.reinforce_bank(&plan.telemetry().layer_activation, true);
                    TurnOrchestrator::commit_turn_circuit(adapter, &plan, bank, None);
                }

                if matches!(horizon_mode, HorizonAblationMode::SlowLoop) {
                    let slow_adj = adapter.apply_slow_loop_telemetry_epoch(64);
                    if slow_adj.applied {
                        println!(
                            "    [MT:slow_loop] steps={} alpha_q7={} repulse_bias_q7={} dwell_delta={}",
                            slow_adj.sampled_steps,
                            slow_adj.suggested_derivative_alpha_q7,
                            slow_adj.suggested_repulse_bias_q7,
                            slow_adj.suggested_slide_out_dwell_delta,
                        );
                    }
                }

                let lower = reply_text.to_lowercase();
                mt_replies.push(reply_text.clone());
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
                } else {
                    mt_ood_total += 1;
                    if !matches!(plan.lacuna_state(), LacunaBoundaryState::LiminalExterior)
                        || plan.is_known_unknown_exterior()
                        || t_sem_forb > 0
                    {
                        mt_ood_bleed_hits += 1;
                    }
                }
                let mt_tlm = *plan.telemetry();
                println!(
                    "  [MT:{}:{}] kv_off={} lacuna={:?} top1={:.3} top2={:.3} guard_m={:+.3} recall={} pro\
                     be=[g=#{:?}({:.3}) loc=#{:?}({:.3}) dis={}] kv={} reflect={} hit={}/{} sem_forb={}(ra\
                     w={}) => {}",
                    t_idx + 1,
                    turn_tag,
                    conv_offset,
                    plan.lacuna_state(),
                    plan.top1_fact_score(),
                    plan.top2_fact_score(),
                    plan.guard_conflict_margin(),
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
            let mt_snap = adapter.telemetry_snapshot();
            let mt_abstract_tokens = mt_snap.abstract_token_count as usize;
            let mt_concrete_tokens = mt_snap.concrete_token_count as usize;
            let mt_whole_bypass_hits = mt_abstract_tokens;
            let mt_decode_tokens_steered = mt_abstract_tokens + mt_concrete_tokens;
            let mt_boundary_churn = mt_snap.boundary_churn_count as usize;
            let mt_logits_skips = mt_snap.logits_skip_count as usize;
            let mt_guard_relaxations = mt_snap.guard_relaxation_count as usize;
            let mt_occ = mt_kv_cache.occupied_slot_count();
            let mt_hits = mt_kv_stats.exact_hits + mt_kv_stats.res_hits;
            let mt_pure_tps = if mt_decode_secs > 1e-6 {
                mt_decode_tokens as f64 / mt_decode_secs
            } else {
                0.0
            };
            let mt_e2e_secs = mt_prefill_secs + mt_decode_secs;
            let mt_e2e_tps = if mt_e2e_secs > 1e-6 {
                mt_decode_tokens as f64 / mt_e2e_secs
            } else {
                0.0
            };
            let mt_prefill_total_ms = mt_prefill_secs * 1000.0;
            let mt_prefill_mean_ms = if mt_prompts_timed > 0 {
                mt_prefill_total_ms / mt_prompts_timed as f64
            } else {
                0.0
            };
            let mt_wbyp_pct = if mt_decode_tokens_steered == 0 {
                0.0
            } else {
                mt_whole_bypass_hits as f64 / mt_decode_tokens_steered as f64 * 100.0
            };
            println!(
                "Multi-Turn Summary (8 turns, persistent KV, ego_hops={}): expect_hit={:.1}% ({}/{}) | raw\
                 _forbid={:.1}% ({}/{}) | sem_forbid={:.1}% ({}/{}) | whole_tok_byp={}/{} ({:.1}%) abs={} \
                 conc={} churn={} guard_relax={} | logits_skip={}/{} | sb_settle={} | kv_occupied={} kv_hi\
                 ts={}/8 (exact={} res={}) pframes={} seeks={} | pure_decode={:.2} tok/s e2e={:.2} tok/s p\
                 refill={:.1}ms ({:.2}ms/t)",
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
                mt_guard_relaxations,
                mt_logits_skips,
                mt_decode_tokens,
                adapter.sandbox_settlements(),
                mt_occ,
                mt_hits,
                mt_kv_stats.exact_hits,
                mt_kv_stats.res_hits,
                mt_kv_stats.pframes_appended,
                mt_kv_stats.pframe_seeks,
                mt_pure_tps,
                mt_e2e_tps,
                mt_prefill_total_ms,
                mt_prefill_mean_ms,
            );
            let mt_q = summarize_dialogue_quality(&mt_replies);
            println!(
                "  [MultiTurnQuality] mattr20={:.3} mattr50={:.3} dist2={:.3} rep4_loop={:.3} aband={:.1}% \
                 | drift(T8-T1): d_mattr20={:+.3} d_dist2={:+.3} d_rep4={:+.3}",
                mt_q.mean_mattr20,
                mt_q.concat_mattr50,
                mt_q.mean_distinct2,
                mt_q.mean_rep4_loop,
                mt_q.abandonment_rate * 100.0,
                mt_q.delta_mattr20,
                mt_q.delta_distinct2,
                mt_q.delta_rep4_loop,
            );
            mt_metrics = Some(TrialPassMetrics {
                label: "8-Turn".to_string(),
                pure_decode_tps: mt_pure_tps,
                e2e_tps: mt_e2e_tps,
                prefill_total_ms: mt_prefill_total_ms,
                prefill_mean_ms: mt_prefill_mean_ms,
                exp_hits: mt_exp_hits,
                exp_total: mt_exp_total,
                sem_forb_hits: mt_sem_forb_hits,
                forb_total: mt_forb_total,
                ood_bleed_hits: mt_ood_bleed_hits,
                ood_total: mt_ood_total,
                whole_bypass_hits: mt_whole_bypass_hits,
                decode_tokens_steered: mt_decode_tokens_steered,
            });
            if let Some(vram_end) = query_nvidia_smi_vram() {
                println!("[VRAM  ] end=[{vram_end}]");
            }
        }
    }

    if !recall_only && run_step3 {
        // Step 3: Temp-Directory .lmnkc Persistence, Zero-Forward Seek & Cross-Model Re-Bake
        println!("\n=== Step 3: Dual-Track .lmnkc GOP Seek & Cross-Model Re-Bake ===");
        let containers_dir =
            std::env::temp_dir().join(format!("liminika_step3_{}", std::process::id()));
        std::fs::create_dir_all(&containers_dir)?;
        let container_path = containers_dir.join(format!("{model_arch_label}_session.lmnkc"));
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

        let mut same_projector = PooledSubtitleEncoder::new(
            LmTokenEncoder::new(model, tokenizer, chat).with_chat_prompt(true),
            adapter.geometry().q_rot_len(),
            adapter.geometry().q_rot_len(),
        );
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

        let mut alt_geom = adapter.geometry().clone();
        if alt_geom.rotary_dim == 128 {
            alt_geom.rotary_dim = 96;
        } else {
            alt_geom.head_dim = 128;
            alt_geom.rotary_dim = 128;
        }
        let mut alt_adapter = ModelSteeringAdapter::new(alt_geom)?;
        alt_adapter.set_ego_max_hops(ego_max_hops);
        let mut alt_projector = PooledSubtitleEncoder::new(
            LmTokenEncoder::new(model, tokenizer, chat).with_chat_prompt(true),
            adapter.geometry().q_rot_len(),
            alt_adapter.geometry().q_rot_len(),
        );
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
            let rebaked_path = containers_dir.join(format!("{model_arch_label}_rebaked_alt.lmnkc"));
            let rebaked_bytes = rebaked_tl.save_to_path(&rebaked_path)?;
            println!(
                "Cross-Model EnvProbe Rebaked: src_fp=0x{source_fingerprint:016x} -> dst_fp=0x\
                 {target_fingerprint:016x} | frames={rebaked_frames} | pframe_nodes=0x{:02X} (hops={}) | s\
                 aved={} ({rebaked_bytes} bytes) | seek_frame2_anchors={}",
                alt_adapter.ego_differential_net().active_node_mask,
                alt_adapter.ego_differential_net().max_hops,
                rebaked_path.display(),
                seek_state.anchor_ids.len(),
            );
        }
        if let Some(bank) = active_persona_bank.as_ref() {
            let pc = bank.to_consolidated_ledger();
            let active_units = bank.units().iter().filter(|u| u.op_counts().1 > 0).count();
            let total_steps: usize = bank
                .units()
                .iter()
                .map(|u| usize::from(u.op_counts().1))
                .sum();
            println!(
                "Updated Persona Cognitive Bank (units={active_units}/{}, total_steps={total_steps}): cons\
                 olidated={:02x?}",
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
        let _ = std::fs::remove_dir_all(&containers_dir);
    }

    Ok(mt_metrics)
}
