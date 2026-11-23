// name: examples/candle_eval/src/step2.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Step 2 natural-language attractor compilation, probe evaluation, and ablation reporting.

use std::collections::HashMap;
use std::time::Instant;

use crate::bake::{
    format_phi4_prompt, query_nvidia_smi_vram, sample_startup_base_prior, FixtureSuite,
};
use crate::cli::{
    build_step2_run_specs, mean_reply_bigram_jaccard, EvalCliConfig, EvalRuntime, Step2PassResult,
    Step2RunSpec, TrialPassMetrics,
};
use crate::eval::{
    apply_ego_mode, count_semantic_forbid_hits, critique_generated_draft, generate_steered_reply,
    HorizonAblationMode, KvTraceStats, PressureSimMode, ReflectMode, SteerDecodeConfig,
    StepTelemetryTotals,
};
use crate::model::{LogitsFetchMode, ResonantAnchors};
use liminika_adapter::{
    AdaptiveGopCutDetector, AttractorContainerCodec, AttractorRole, AttractorTrajectoryEvaluator,
    CompiledTokenCorpus, CorpusRecallEscalationProvider, FrameTransitionMetrics,
    LacunaBoundaryState, LmnkTimeline, ManifoldReflectionEvaluator, ReflectionCritique,
    ReflectionGateInput, ReflectionPolicy, ReflectionSelection, SequenceActivationPooler,
    SourceTextRecord, TimelineStepContext, TokenSequenceEntry, TrajectoryManifoldSnapshot,
};
use liminika_core::runtime::{
    AdaptiveThresholdPolicy, CognitiveCircuitThresholdPolicy, ObjectiveFact, RamPhaseKvCacheOps,
    StandardPhaseKvGopCache,
};
use liminika_core::stigmergy::{CognitiveCircuitBankOps, StigmergicActionDecision};

pub(crate) struct Step2Output {
    pub(crate) compiled: CompiledTokenCorpus,
    pub(crate) timeline: LmnkTimeline,
    pub(crate) cut_detector: AdaptiveGopCutDetector,
    pub(crate) kv_policy: AdaptiveThresholdPolicy,
    pub(crate) reflection_policy: ReflectionPolicy,
    pub(crate) pass_metrics: Vec<TrialPassMetrics>,
}

pub(crate) fn run_step2_evaluation(
    cli: &EvalCliConfig,
    suite: &FixtureSuite,
    rt: &mut EvalRuntime<'_>,
) -> Result<Step2Output, Box<dyn std::error::Error>> {
    let cfg = rt.cfg;
    let tokenizer = rt.tokenizer;
    let model = &mut *rt.model;
    let adapter = &mut *rt.adapter;
    let baked_opt = rt.baked_opt;
    let active_persona_bank = &mut *rt.active_persona_bank;
    let use_base_prior = cli.use_base_prior;
    let no_reflect = cli.no_reflect;
    let reflect_mode = cli.reflect_mode;
    let ablate_dual_axis = cli.ablate_dual_axis;
    let ablate_ego = cli.ablate_ego;
    let ablate_horizon = cli.ablate_horizon;
    let fair_bench_conditions = &cli.fair_bench_conditions;
    let kv_min_res = cli.kv_min_res;
    let pressure_sim_mode = cli.pressure_sim_mode;

    // Step 2: Online Natural-Language -> Attractor Corpus & Liminal Boundary Evaluation
    println!("\n=== Step 2: Natural-Language Attractor Corpus & Liminal Boundary Eval ===");
    adapter.set_layer_rope_mask(0xFFFF_0000);
    let base_prior = if use_base_prior {
        let t_prior = Instant::now();
        let profile = sample_startup_base_prior(model, tokenizer, adapter)?;
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
            let (_, layer_tok_qs, _) =
                model.forward_tokens(&ids, 0, None, false, true, LogitsFetchMode::FullLogits)?;
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
        adapter,
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
            let (_, alias_tok_qs, _) = model.forward_tokens(
                &full_ids,
                0,
                None,
                false,
                true,
                LogitsFetchMode::FullLogits,
            )?;
            let dim = adapter.geometry().q_rot_len();
            let start = prefix_ids.len() * dim;
            let end = full_ids.len() * dim;
            let sliced_qs: Vec<Vec<f32>> = alias_tok_qs
                .into_iter()
                .map(|layer| layer[start..end].to_vec())
                .collect();
            compiled.register_bpe_prefix_alias_tokens(
                adapter,
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
    let step2_runs: Vec<Step2RunSpec> = build_step2_run_specs(cli);
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
    let mut pass_metrics: Vec<TrialPassMetrics> = Vec::new();
    let mut prompt_q_cache: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    let cut_detector = AdaptiveGopCutDetector::default();
    let mut timeline = LmnkTimeline::new(LmnkTimeline::geometry_fingerprint(adapter.geometry()));
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
        timeline = LmnkTimeline::new(LmnkTimeline::geometry_fingerprint(adapter.geometry()));
        adapter.reset_horizon_and_core_state();
        adapter.set_guard_hold_tokens(cli.guard_hold_tokens);
        adapter.compute_pool_mut().reanchor_baseline_pressure(8192);
        adapter.set_ego_max_hops(run.ego_max_hops);
        apply_ego_mode(adapter, run_ego_mode);
        if ablate_dual_axis || ablate_ego || ablate_horizon || fair_bench_conditions.is_some() {
            println!("\n-- Step 2 Pass: {run_label} --");
        }
        let mut step_tlm = StepTelemetryTotals::default();
        let mut esc_provider = CorpusRecallEscalationProvider::new(&compiled.corpus);
        let mut slow_loop_epochs_applied = 0_u64;
        let mut slow_loop_last_dwell_delta = 0_i8;
        let mut slow_loop_last_repulse_bias = 0_i8;
        let mut slow_loop_last_express_thresh =
            adapter.substrate().thresholds.express_steer_density_thresh;
        let mut slow_loop_last_rank1_thresh =
            adapter.substrate().thresholds.rank1_inject_density_thresh;
        let mut slow_loop_last_skip_thresh =
            adapter.substrate().thresholds.inertial_skip_entropy_thresh;
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
        let mut step2_prefill_secs = 0.0_f64;
        let mut step2_decode_secs = 0.0_f64;
        let mut step2_prompts_timed = 0_usize;
        let mut step2_steer_bypass_hits = 0_usize;
        let mut step2_steer_eligible_layers = 0_usize;
        let mut step2_whole_token_bypass_hits = 0_usize;
        let mut step2_decode_tokens_steered = 0_usize;
        let mut step2_abstract_tokens = 0_usize;
        let mut step2_concrete_tokens = 0_usize;
        let mut step2_boundary_churn = 0_usize;
        let mut step2_logits_skips = 0_usize;
        let mut step2_instability_recoveries = 0_usize;
        let mut step2_guard_relaxations = 0_usize;
        let mut step2_spine_veto_guard = 0_usize;
        let mut step2_spine_veto_repulse = 0_usize;
        let mut step2_spine_veto_pin = 0_usize;
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
                let (_, q_toks, _) = model.forward_tokens(
                    &q_ids,
                    0,
                    None,
                    false,
                    true,
                    LogitsFetchMode::FullLogits,
                )?;
                prompt_q_cache.insert(prompt.clone(), q_toks.clone());
                q_toks
            };
            let prior_circuit = active_persona_bank
                .as_ref()
                .map(CognitiveCircuitBankOps::to_consolidated_ledger);
            let t_recall = Instant::now();
            let recall = compiled.recall_token_sequence_with_circuit(
                adapter,
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
                    adapter,
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
                    lacuna_state: recall.lacuna_state,
                    frames_since_keyframe,
                    new_cluster_spawned: matches!(
                        recall.lacuna_state,
                        LacunaBoundaryState::LiminalExterior
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
            let (core_hor, core_overlap, ens_opt) =
                adapter.sync_core_layers_from_recall(&recall, None);
            let core_note = format!(
                "reg={:?} l0={} l1={} ego={} sc={} ovlp=0x{:02X} ens={}",
                core_hor.regime(),
                adapter.threshold_state().adaptation_epochs,
                adapter.stigmergic_granule_state().adaptation_epochs,
                adapter.ego_state().intervention_epochs,
                adapter.last_boundary_delta().stationary_shortcut_ready,
                core_overlap.overlap_core_mask,
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
            let query_lacuna_state = recall.lacuna_state;
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
                        || (matches!(query_lacuna_state, LacunaBoundaryState::LiminalExterior)
                            && !is_guard_trap)
                    {
                        format_phi4_prompt(prompt)
                    } else {
                        format_phi4_prompt(&format!(
                            "[Context: {recalled_ctx}]\n[Style Guidelines: {persona_rag_text}]\n[Do NOT claim: {guard_rag_text}]\n{prompt}"
                        ))
                    }
                } else if run_cond.steering_enabled() {
                    if matches!(query_lacuna_state, LacunaBoundaryState::LiminalExterior)
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
                    || (matches!(query_lacuna_state, LacunaBoundaryState::LiminalExterior)
                        && !is_guard_trap)
                {
                    format_phi4_prompt(prompt)
                } else {
                    format_phi4_prompt(&format!("[Context: {recalled_ctx}]\n{prompt}"))
                };

                let use_steer = run_cond.steering_enabled()
                    && (is_guard_trap
                        || !matches!(query_lacuna_state, LacunaBoundaryState::LiminalExterior)
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
                let rank1_arg = if use_steer {
                    Some((baked.rank1_u.as_slice(), baked.rank1_v.as_slice(), -1.00))
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
                    rank1: rank1_arg,
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
                    logits_observe_stride: cli.logits_observe_stride,
                    steer_bypass_hits: 0,
                    steer_eligible_layers: 0,
                    whole_token_bypass_hits: 0,
                    decode_tokens_total: 0,
                    abstract_tokens: 0,
                    concrete_tokens: 0,
                    boundary_churn_transitions: 0,
                    logits_skips: 0,
                    instability_recoveries: 0,
                    guard_relaxations: 0,
                    spine_veto_guard_hits: 0,
                    spine_veto_repulse_hits: 0,
                    spine_veto_pin_hits: 0,
                };
                let (mut text, p1_in_tokens, p1_tokens, p1_timing) = generate_steered_reply(
                    model,
                    tokenizer,
                    adapter,
                    active_prompt.as_str(),
                    &mut pass1_cfg,
                )?;
                step2_input_tokens += p1_in_tokens;
                step2_decode_tokens += p1_tokens;
                step2_prefill_secs += p1_timing.prefill_secs;
                step2_decode_secs += p1_timing.decode_secs;
                step2_prompts_timed += 1;
                step2_steer_bypass_hits += pass1_cfg.steer_bypass_hits;
                step2_steer_eligible_layers += pass1_cfg.steer_eligible_layers;
                step2_whole_token_bypass_hits += pass1_cfg.whole_token_bypass_hits;
                step2_decode_tokens_steered += pass1_cfg.decode_tokens_total;
                step2_abstract_tokens += pass1_cfg.abstract_tokens;
                step2_concrete_tokens += pass1_cfg.concrete_tokens;
                step2_boundary_churn += pass1_cfg.boundary_churn_transitions;
                step2_logits_skips += pass1_cfg.logits_skips;
                step2_instability_recoveries += pass1_cfg.instability_recoveries;
                step2_guard_relaxations += pass1_cfg.guard_relaxations;
                step2_spine_veto_guard += pass1_cfg.spine_veto_guard_hits;
                step2_spine_veto_repulse += pass1_cfg.spine_veto_repulse_hits;
                step2_spine_veto_pin += pass1_cfg.spine_veto_pin_hits;

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
                            boundary_sensitivity_q7: recall
                                .lacuna_partition
                                .boundary_sensitivity_q7,
                            has_undigested_known_unknown: recall.is_known_unknown_exterior,
                            exit_entropy_h_tilde: adapter.substrate().last_entropy_h_tilde,
                            guard_conflict_margin: query_guard_m,
                            is_unobserved_lacuna: matches!(
                                query_lacuna_state,
                                LacunaBoundaryState::LiminalExterior
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
                        model,
                        tokenizer,
                        adapter,
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
                            let retry_rank1 = if retry_use_steer {
                                Some((
                                    baked.rank1_u.as_slice(),
                                    baked.rank1_v.as_slice(),
                                    -1.00 + pass1_crit.directive.rank1_scale_delta,
                                ))
                            } else {
                                None
                            };
                            let mut retry_cfg = SteerDecodeConfig {
                                q7s: Some(baked.fact_q7.as_slice()),
                                persona_q7s: Some(baked.persona_q7.as_slice()),
                                resonant_ctx: retry_res,
                                rope_rad: retry_rope,
                                rank1: retry_rank1,
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
                                logits_observe_stride: cli.logits_observe_stride,
                                steer_bypass_hits: 0,
                                steer_eligible_layers: 0,
                                whole_token_bypass_hits: 0,
                                decode_tokens_total: 0,
                                abstract_tokens: 0,
                                concrete_tokens: 0,
                                boundary_churn_transitions: 0,
                                logits_skips: 0,
                                instability_recoveries: 0,
                                guard_relaxations: 0,
                                spine_veto_guard_hits: 0,
                                spine_veto_repulse_hits: 0,
                                spine_veto_pin_hits: 0,
                            };
                            let (p2_text, p2_in_tokens, p2_tokens, p2_timing) =
                                generate_steered_reply(
                                    model,
                                    tokenizer,
                                    adapter,
                                    retry_prompt.as_str(),
                                    &mut retry_cfg,
                                )?;
                            step2_input_tokens += p2_in_tokens;
                            step2_decode_tokens += p2_tokens;
                            step2_prefill_secs += p2_timing.prefill_secs;
                            step2_decode_secs += p2_timing.decode_secs;
                            step2_prompts_timed += 1;
                            step2_steer_bypass_hits += retry_cfg.steer_bypass_hits;
                            step2_steer_eligible_layers += retry_cfg.steer_eligible_layers;
                            step2_whole_token_bypass_hits += retry_cfg.whole_token_bypass_hits;
                            step2_decode_tokens_steered += retry_cfg.decode_tokens_total;
                            step2_abstract_tokens += retry_cfg.abstract_tokens;
                            step2_concrete_tokens += retry_cfg.concrete_tokens;
                            step2_boundary_churn += retry_cfg.boundary_churn_transitions;
                            step2_logits_skips += retry_cfg.logits_skips;
                            step2_instability_recoveries += retry_cfg.instability_recoveries;
                            step2_guard_relaxations += retry_cfg.guard_relaxations;
                            step2_spine_veto_guard += retry_cfg.spine_veto_guard_hits;
                            step2_spine_veto_repulse += retry_cfg.spine_veto_repulse_hits;
                            step2_spine_veto_pin += retry_cfg.spine_veto_pin_hits;
                            if let Some(pass2_crit) = critique_generated_draft(
                                model,
                                tokenizer,
                                adapter,
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
                                | liminika_adapter::ReflectionTriggerKind::ConsistentLacunaAbstention
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
                if !matches!(query_lacuna_state, LacunaBoundaryState::LiminalExterior)
                    || recall.is_known_unknown_exterior
                    || p_sem_forb > 0
                {
                    ood_bleed_hits += 1;
                }
            }
            lane_util_sum += adapter.substrate().slot_enabled_mask.count_ones() as f32 / 8.0;
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
            "  [S2:{}:{}] lacuna={:?} action={:?} top1={:.3} top2={:.3} guard_m={:+.3} recall={} recall_ms={:.2} kv={} reflect={} core=[{}] probe=[g=#{:?}({:.3}) loc=#{:?}({:.3}) dis={} prov_slot={}] ctx_hit={}/{} hit={}/{} sem_forb={}(raw={}) => {}",
            idx + 1,
            slot,
            query_lacuna_state,
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
        let e2e_secs = step2_prefill_secs + step2_decode_secs;
        let e2e_tps = if e2e_secs > 1e-6 {
            step2_decode_tokens as f64 / e2e_secs
        } else {
            0.0
        };
        let prefill_total_ms = step2_prefill_secs * 1000.0;
        let prefill_mean_ms = if step2_prompts_timed > 0 {
            prefill_total_ms / step2_prompts_timed as f64
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
        "Step 2 Dynamic Corpus Summary ({run_label}): recall_ctx_hit={:.1}% ({}/{}) | gen_expect_hit={:.1}% ({}/{}) | raw_forbid={:.1}% ({}/{}) | sem_forbid={:.1}% ({}/{}) | in_tok={} | reflect=({}/{} skip={}) | ext_rescue={} sb_settle={} logits_skip={}/{} | core=[L0={} L1={} Ego={} hops={} fired=0x{:02X}] | compile={:.1}ms mean_recall={:.2}ms pure_decode={:.2} tok/s e2e={:.2} tok/s prefill={:.1}ms ({:.2}ms/p) | vram={}",
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
        adapter.sandbox_settlements(),
        step2_logits_skips,
        step2_decode_tokens,
        adapter.threshold_state().adaptation_epochs,
        adapter.stigmergic_granule_state().adaptation_epochs,
        adapter.ego_state().intervention_epochs,
        adapter.ego_differential_net().max_hops,
        adapter.ego_state().last_intervened_mask,
        compile_ms,
        mean_recall_ms,
        decode_tps,
        e2e_tps,
        prefill_total_ms,
        prefill_mean_ms,
        vram_pass,
    );
        println!("{summary_line}");
        let last_snap = adapter.telemetry_snapshot();
        step_tlm.accumulate(&last_snap);
        let guard_slot = adapter.coherence_guard_slot();
        let dwell = adapter.horizon_dwell();
        let spillover = adapter.residual_spillover();
        let action_line = format!(
            "  [Substrate] total={} Passive={:.1}% Skip={:.1}% Express={:.1}% sub-head={:.1}% Rank-1={:.1}% Crystal={:.1}% Repulse={:.1}%",
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
            "  [ComputePool] regime={:?} ema_p={} base_p={} dwell={} lane={} slots(active={} pinned={} rev={}) layer_byp={}/{} ({:.1}%) whole_tok_byp={}/{} ({:.1}%) abs={} conc={} churn={} recov={} guard_relax={} spine_veto=(g={},r={},p={})",
            cp.regime,
            cp.ema_pressure_q15,
            cp.baseline_pressure_q15,
            cp.lane_switch_dwell,
            cp.primary_lane,
            cp.active_mask.count_ones(),
            adapter.system_spine().pinned_compute_mask.count_ones(),
            cp.reversible_mask.count_ones(),
            step2_steer_bypass_hits,
            step2_steer_eligible_layers,
            steer_bypass_pct,
            step2_whole_token_bypass_hits,
            step2_decode_tokens_steered,
            whole_bypass_pct,
            step2_abstract_tokens,
            step2_concrete_tokens,
            step2_boundary_churn,
            step2_instability_recoveries,
            step2_guard_relaxations,
            step2_spine_veto_guard,
            step2_spine_veto_repulse,
            step2_spine_veto_pin,
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
        pass_metrics.push(TrialPassMetrics {
            label: run.label.clone(),
            pure_decode_tps: decode_tps,
            e2e_tps,
            prefill_total_ms,
            prefill_mean_ms,
            exp_hits: step2_exp_hits,
            exp_total: step2_exp_total,
            sem_forb_hits: step2_sem_forb_hits,
            forb_total: step2_forb_total,
            ood_bleed_hits,
            ood_total,
            whole_bypass_hits: step2_whole_token_bypass_hits,
            decode_tokens_steered: step2_decode_tokens_steered,
        });
        pass_results.push(Step2PassResult {
            label: run.label.clone(),
            axis: run.axis,
            ego_max_hops: run.ego_max_hops,
            cond: run.cond,
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
            ego_delta_norm_q7: step2_ego_delta_norm_max_q7,
            reflect_skipped: reflect_gate_skipped,
            ext_rescues: ext_rescue_hits,
            replies: pass_replies,
        });
        if adapter.ego_state().is_active() {
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

    Ok(Step2Output {
        compiled,
        timeline,
        cut_detector,
        kv_policy,
        reflection_policy,
        pass_metrics,
    })
}
