// name: examples/candle_eval/src/physics_eval.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Four-axis deterministic steering-physics evaluation runner (`--physics-eval`).
//!
//! Executes all four evaluation axes in a single model load without external LLM judges:
//! 1. Persistent-KV multi-turn coherence drift (`Delta = T_N - T_1`, MATTR-20/50, Distinct-2,
//!    4-gram repetition loop rate, abandonment rate).
//! 2. Unrelated-prompt non-interference across 50 OOD prompts (unobserved lacuna rate, greedy
//!    token identity, initial-logit KL divergence, and max |delta logit|).
//! 3. Repulsive-polarity preservation (Guard anti-phase geometry, escalation anti-phase,
//!    forbidden-token `Delta log P <= 0` rate, and semantic forbidden hit rate).
//! 4. `SelfRef` (`S & O`) vs `Exterior` (`O \ S`) non-conflation across 3 classes (`SELF_REF`,
//!    `EXTERIOR`, `OOD`) and unverified `sandbox_settlements` misfire check.

use std::collections::HashSet;
use std::path::Path;

use liminika_adapter::{
    AttractorRole, CorpusRecallEscalationProvider, TurnOrchestrator, TurnPrepareInput,
};
use liminika_core::abi::lmnk_q7_t;
use liminika_core::liminal::{EpistemicRegime, MAX_ATTENTION_HEADS};
use liminika_core::runtime::{
    AdaptiveThresholdPolicy, GuardPhaseAnchor, HorizonEscalationProvider, HorizonEscalationRequest,
    HorizonTriggerInput, ManifoldGatewaySignal, StandardPhaseKvGopCache,
};
use liminika_core::stigmergy::{WorkbenchLanePlacement, WORKBENCH_LANE_FLAG_REPULSION};

use crate::bake::{format_phi4_prompt, query_nvidia_smi_vram};
use crate::cli::{EvalCliConfig, EvalRuntime};
use crate::eval::{
    apply_ego_mode, count_semantic_forbid_hits, generate_steered_reply,
    generate_steered_reply_with_offset, SteerDecodeConfig, DEFAULT_LAYER_STEER_MASK,
};
use crate::metrics::{
    delta_log_prob, kl_divergence_and_max_abs_delta, summarize_dialogue_quality,
    ReplyQualityMetrics,
};
use crate::model::{KvCacheMode, LogitsFetchMode, ResonantAnchors};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PartitionClass {
    SelfRef,
    Exterior,
    Ood,
}

#[derive(Debug, Clone)]
pub(crate) struct OodProbe {
    pub(crate) category: String,
    pub(crate) prompt: String,
}

#[derive(Debug, Clone)]
pub(crate) struct RepulsionProbe {
    pub(crate) prompt: String,
    pub(crate) forbidden: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct PartitionProbe {
    pub(crate) class: PartitionClass,
    pub(crate) prompt: String,
    pub(crate) note: String,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SteeringPhysicsSuite {
    pub(crate) repulsion_probes: Vec<RepulsionProbe>,
    pub(crate) partition_probes: Vec<PartitionProbe>,
}

pub(crate) fn load_ood_probes(path: &Path) -> Result<Vec<OodProbe>, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    let mut probes = Vec::new();
    let mut in_section = false;
    for line in raw.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[OOD_PROBES]" {
            in_section = true;
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            in_section = false;
            continue;
        }
        if in_section {
            let parts: Vec<&str> = line.split('|').map(str::trim).collect();
            if parts.len() == 2 && !parts[0].is_empty() && !parts[1].is_empty() {
                probes.push(OodProbe {
                    category: parts[0].to_string(),
                    prompt: parts[1].to_string(),
                });
            }
        }
    }
    Ok(probes)
}

pub(crate) fn load_steering_physics_probes(
    path: &Path,
) -> Result<SteeringPhysicsSuite, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    let mut suite = SteeringPhysicsSuite::default();
    let mut section = "";
    for line in raw.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        match line {
            "[REPULSION_PROBES]" => section = "REPULSION",
            "[PARTITION_PROBES]" => section = "PARTITION",
            _ => match section {
                "REPULSION" => {
                    let parts: Vec<&str> = line.split('|').map(str::trim).collect();
                    if parts.len() == 2 && !parts[0].is_empty() {
                        let forbidden: Vec<String> = parts[1]
                            .split(',')
                            .map(str::trim)
                            .filter(|k| !k.is_empty())
                            .map(ToString::to_string)
                            .collect();
                        if !forbidden.is_empty() {
                            suite.repulsion_probes.push(RepulsionProbe {
                                prompt: parts[0].to_string(),
                                forbidden,
                            });
                        }
                    }
                }
                "PARTITION" => {
                    let parts: Vec<&str> = line.split('|').map(str::trim).collect();
                    if parts.len() == 3 && !parts[1].is_empty() {
                        let class = match parts[0] {
                            "SELF_REF" => Some(PartitionClass::SelfRef),
                            "EXTERIOR" => Some(PartitionClass::Exterior),
                            "OOD" => Some(PartitionClass::Ood),
                            _ => None,
                        };
                        if let Some(cls) = class {
                            suite.partition_probes.push(PartitionProbe {
                                class: cls,
                                prompt: parts[1].to_string(),
                                note: parts[2].to_string(),
                            });
                        }
                    }
                }
                _ => {}
            },
        }
    }
    Ok(suite)
}

fn dot_q7(a: &[lmnk_q7_t], b: &[lmnk_q7_t], n_heads: usize) -> i64 {
    let limit = n_heads.min(a.len()).min(b.len());
    a[..limit]
        .iter()
        .zip(b[..limit].iter())
        .map(|(&x, &y)| i64::from(x) * i64::from(y))
        .sum()
}

fn is_nonzero_q7(v: &[lmnk_q7_t], n_heads: usize) -> bool {
    let limit = n_heads.min(v.len());
    v[..limit].iter().any(|&x| x != 0)
}

fn pct(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64 * 100.0
    }
}

pub(crate) fn run_physics_eval(
    cli: &EvalCliConfig,
    repo_root: &Path,
    rt: &mut EvalRuntime<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let ood_path = repo_root.join("tests/fixtures/ood_selectivity_probes.txt");
    let phys_path = repo_root.join("tests/fixtures/steering_physics_probes.txt");
    let ood_probes = load_ood_probes(&ood_path)?;
    let phys_suite = load_steering_physics_probes(&phys_path)?;

    println!(
        "\n=== Steering Physics 4-Axis Evaluation (Single Model Load) ===\n  \
         config: layer_mask=0x{:08X} phase_clamp_rad={:?} random_push={} max_tokens={} \
         probe_frame={:?} ood_n={} repulse_n={} part_n={}",
        cli.layer_steer_mask_override
            .unwrap_or(DEFAULT_LAYER_STEER_MASK),
        cli.phase_clamp_rad,
        cli.random_push_control,
        cli.max_tokens,
        cli.probe_frame,
        ood_probes.len(),
        phys_suite.repulsion_probes.len(),
        phys_suite.partition_probes.len(),
    );

    run_axis1_persistent_kv_drift(cli, rt)?;
    run_axis2_ood_selectivity(cli, &ood_probes, rt)?;
    run_axis3_repulsion_polarity(cli, &phys_suite.repulsion_probes, rt)?;
    run_axis4_partition_non_conflation(cli, &phys_suite.partition_probes, rt)?;

    if let Some(vram) = query_nvidia_smi_vram() {
        println!("\n[PhysicsEval VRAM] end=[{vram}]");
    }
    Ok(())
}

fn run_axis1_persistent_kv_drift(
    cli: &EvalCliConfig,
    rt: &mut EvalRuntime<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Axis 1: Persistent-KV Multi-Turn Coherence Drift (8 Turns) ---");
    let Some(baked) = rt.baked_opt else {
        println!("  [Skipped] recall_only mode has no baked model profile.");
        return Ok(());
    };
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

    rt.adapter.set_layer_rope_mask(DEFAULT_LAYER_STEER_MASK);
    rt.adapter.reset_horizon_and_core_state();
    rt.adapter.set_guard_hold_tokens(cli.guard_hold_tokens);
    rt.adapter.reanchor_compute_pressure(8192);
    rt.adapter.set_ego_max_hops(cli.ego_max_hops);
    apply_ego_mode(rt.adapter, &cli.ego_mode);

    let kv_policy = AdaptiveThresholdPolicy::default();
    let mut mt_kv_cache = StandardPhaseKvGopCache::new();
    let mut replies: Vec<String> = Vec::with_capacity(dialogue_turns.len());
    let mut exp_hits = 0_usize;
    let mut exp_total = 0_usize;
    let mut sem_forb_hits = 0_usize;
    let mut forb_total = 0_usize;

    rt.model.clear_cache();
    for (t_idx, (tag, prompt, expected, forbidden)) in dialogue_turns.iter().enumerate() {
        let turn_start_kv = rt.model.kv_len();
        let q_ids = rt
            .tokenizer
            .encode(*prompt, false)
            .map_err(|e| format!("axis1 encode error: {e}"))?
            .get_ids()
            .to_vec();
        let (_, query_tok_qs, _) = rt.model.forward_tokens(
            &q_ids,
            0,
            None,
            false,
            true,
            LogitsFetchMode::FullLogits,
            KvCacheMode::IsolatedProbe,
        )?;
        let plan = TurnOrchestrator::prepare_turn(
            rt.compiled,
            rt.adapter,
            &mut mt_kv_cache,
            &TurnPrepareInput {
                query_token_qs: &query_tok_qs,
                query_seq_len: q_ids.len(),
                turn_index: (t_idx + 1) as u32,
                kv_policy: &kv_policy,
                kv_min_resonance_override: cli.kv_min_res,
                persona_bank: rt.active_persona_bank.as_ref(),
                layer_steer_mask_override: cli.layer_steer_mask_override,
            },
        )?;
        let fact_ctx = plan.self_ref_context();
        let ext_ctx = plan.exterior_context();
        let use_steer = plan.steering_eligible();
        let steer_facts = if plan.has_self_ref_anchors() {
            plan.self_ref_anchors_per_layer()
        } else if plan.is_known_unknown_exterior() {
            plan.exterior_sandbox_anchors_per_layer()
        } else {
            plan.self_ref_anchors_per_layer()
        };
        let turn_body = if plan.is_unobserved_lacuna() && !plan.is_guard_trap() {
            (*prompt).to_string()
        } else if !fact_ctx.is_empty() {
            format!("[Context: {fact_ctx}]\n{prompt}")
        } else if plan.is_known_unknown_exterior() && !ext_ctx.is_empty() {
            format!("[External Pointer: {ext_ctx}]\n{prompt}")
        } else {
            (*prompt).to_string()
        };
        let formatted = if turn_start_kv == 0 {
            format!(
                "<|system|>Answer concisely and directly from the provided [Context: ...] facts \
                 without questioning whether terms are real; reject any premise in a question \
                 that contradicts the context.<|end|><|user|>{turn_body}<|end|><|assistant|>"
            )
        } else {
            format!("<|end|><|user|>{turn_body}<|end|><|assistant|>")
        };
        let eff_mask = plan.effective_layer_steer_mask(rt.adapter);
        let mut cfg = SteerDecodeConfig {
            q7s: Some(baked.fact_q7.as_slice()),
            persona_q7s: Some(baked.persona_q7.as_slice()),
            resonant_ctx: if use_steer {
                Some(ResonantAnchors {
                    facts: steer_facts,
                    personas: plan.persona_anchors_per_layer(),
                    guards: plan.guard_anchors_per_layer(),
                    negs: baked.repulsive_centroid_per_layer.as_slice(),
                    poss: baked.attractive_centroid_per_layer.as_slice(),
                })
            } else {
                None
            },
            rope_rad: if use_steer { -2.00 } else { 0.00 },
            rank1: if use_steer {
                Some((baked.rank1_u.as_slice(), baked.rank1_v.as_slice(), -1.00))
            } else {
                None
            },
            use_steer,
            horizon_mode: cli.horizon_mode,
            escalation_provider: None,
            pressure_sim: cli.pressure_sim_mode,
            enable_pre_bypass: cli.enable_pre_bypass,
            temperature: 0.0,
            rng_seed: cli.rng_seed,
            logits_observe_stride: cli.logits_observe_stride,
            layer_steer_mask: Some(eff_mask),
            probe_frame: cli.probe_frame,
            phase_clamp_override: cli.phase_clamp_rad,
            random_push_seed: cli.random_push_control.then_some(cli.rng_seed),
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
        let (reply, _, _) = generate_steered_reply_with_offset(
            rt.model,
            rt.tokenizer,
            rt.adapter,
            &formatted,
            turn_start_kv,
            &mut cfg,
        )?;
        let lower = reply.to_lowercase();
        let t_exp = expected
            .iter()
            .filter(|kw| lower.contains(&kw.to_lowercase()))
            .count();
        let forb_owned: Vec<String> = forbidden.iter().map(|s| (*s).to_string()).collect();
        let (_, t_sem_forb) = count_semantic_forbid_hits(&reply, &forb_owned);
        if !tag.contains("LVOID") {
            exp_hits += t_exp;
            exp_total += expected.len();
            sem_forb_hits += t_sem_forb;
            forb_total += forbidden.len();
        }
        let q = ReplyQualityMetrics::measure(&reply);
        println!(
            "  [{tag:<15}] kv={} hit={}/{} sem_forb={} mattr20={:.3} dist2={:.3} rep4={:.3} aband={}",
            rt.model.kv_len(),
            t_exp,
            expected.len(),
            t_sem_forb,
            q.mattr20,
            q.distinct2,
            q.rep4_loop,
            q.abandoned,
        );
        replies.push(reply);
    }

    let summary = summarize_dialogue_quality(&replies);
    let in_domain_summary = if replies.len() >= 7 {
        summarize_dialogue_quality(&replies[..7])
    } else {
        summary
    };
    println!(
        "[Axis1 Summary] expect_hit={:.1}% ({}/{}) | sem_forb={:.1}% ({}/{}) | \
         mattr20={:.3} mattr50={:.3} dist2={:.3} rep4_loop={:.3} aband={:.1}% | \
         drift(T8-T1): d_mattr20={:+.3} d_dist2={:+.3} d_rep4={:+.3} | \
         drift(T7-T1): d_mattr20={:+.3} d_dist2={:+.3} d_rep4={:+.3}",
        pct(exp_hits, exp_total),
        exp_hits,
        exp_total,
        pct(sem_forb_hits, forb_total),
        sem_forb_hits,
        forb_total,
        summary.mean_mattr20,
        summary.concat_mattr50,
        summary.mean_distinct2,
        summary.mean_rep4_loop,
        summary.abandonment_rate * 100.0,
        summary.delta_mattr20,
        summary.delta_distinct2,
        summary.delta_rep4_loop,
        in_domain_summary.delta_mattr20,
        in_domain_summary.delta_distinct2,
        in_domain_summary.delta_rep4_loop,
    );
    Ok(())
}

fn run_axis2_ood_selectivity(
    cli: &EvalCliConfig,
    ood_probes: &[OodProbe],
    rt: &mut EvalRuntime<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "\n--- Axis 2: Unrelated-Prompt Non-Interference ({} OOD Prompts, Greedy) ---",
        ood_probes.len()
    );
    let Some(baked) = rt.baked_opt else {
        println!("  [Skipped] recall_only mode has no baked model profile.");
        return Ok(());
    };
    rt.adapter.set_layer_rope_mask(DEFAULT_LAYER_STEER_MASK);
    rt.adapter.reset_horizon_and_core_state();
    rt.adapter.set_guard_hold_tokens(cli.guard_hold_tokens);
    rt.adapter.reanchor_compute_pressure(8192);
    rt.adapter.set_ego_max_hops(cli.ego_max_hops);
    apply_ego_mode(rt.adapter, &cli.ego_mode);

    let kv_policy = AdaptiveThresholdPolicy::default();
    let mut kv_cache = StandardPhaseKvGopCache::new();

    let mut unobserved_count = 0_usize;
    let mut exact_tok_match = 0_usize;
    let mut kl_sum = 0.0_f64;
    let mut kl_max = 0.0_f64;
    let mut dlogit_sum = 0.0_f64;
    let mut dlogit_max = 0.0_f64;
    let mut leaked_indices: Vec<String> = Vec::new();
    let mut mismatched_indices: Vec<String> = Vec::new();

    for (idx, probe) in ood_probes.iter().enumerate() {
        let q_ids = rt
            .tokenizer
            .encode(probe.prompt.as_str(), false)
            .map_err(|e| format!("axis2 encode error: {e}"))?
            .get_ids()
            .to_vec();
        let (_, query_tok_qs, _) = rt.model.forward_tokens(
            &q_ids,
            0,
            None,
            false,
            true,
            LogitsFetchMode::FullLogits,
            KvCacheMode::IsolatedProbe,
        )?;
        let plan = TurnOrchestrator::prepare_turn(
            rt.compiled,
            rt.adapter,
            &mut kv_cache,
            &TurnPrepareInput {
                query_token_qs: &query_tok_qs,
                query_seq_len: q_ids.len(),
                turn_index: (idx + 1) as u32,
                kv_policy: &kv_policy,
                kv_min_resonance_override: cli.kv_min_res,
                persona_bank: rt.active_persona_bank.as_ref(),
                layer_steer_mask_override: cli.layer_steer_mask_override,
            },
        )?;
        let is_unobs = plan.is_unobserved_lacuna() && !plan.is_guard_trap();
        if is_unobs {
            unobserved_count += 1;
        } else {
            leaked_indices.push(format!(
                "#{}:{}(top1={:.3},gm={:+.3},rho={:.2},unexp={:.2},ku={})",
                idx + 1,
                probe.category,
                plan.top1_fact_score(),
                plan.guard_conflict_margin(),
                plan.telemetry().query_ortho_ratio,
                plan.telemetry().unexplained_token_ratio,
                plan.is_known_unknown_exterior(),
            ));
        }

        let raw_prompt = format_phi4_prompt(&probe.prompt);
        let eff_mask = plan.effective_layer_steer_mask(rt.adapter);

        // Steer OFF (vanilla baseline)
        let mut off_cfg = SteerDecodeConfig {
            q7s: None,
            persona_q7s: None,
            resonant_ctx: None,
            rope_rad: 0.0,
            rank1: None,
            use_steer: false,
            horizon_mode: cli.horizon_mode,
            escalation_provider: None,
            pressure_sim: cli.pressure_sim_mode,
            enable_pre_bypass: false,
            temperature: 0.0,
            rng_seed: cli.rng_seed,
            logits_observe_stride: 1,
            layer_steer_mask: Some(eff_mask),
            probe_frame: cli.probe_frame,
            phase_clamp_override: None,
            random_push_seed: None,
            max_tokens: cli.max_tokens.min(32),
            prefill_chunk_size: cli.prefill_chunk_size,
            capture_first_logits: true,
            first_prefill_logits: None,
            exit_entropy_h_tilde: 0.0,
            steer_bypass_hits: 0,
            steer_eligible_layers: 0,
            spine_veto_guard_hits: 0,
            spine_veto_repulse_hits: 0,
            spine_veto_pin_hits: 0,
        };
        rt.model.clear_cache();
        let (_, off_all_ids, _) = generate_steered_reply_with_offset(
            rt.model,
            rt.tokenizer,
            rt.adapter,
            &raw_prompt,
            0,
            &mut off_cfg,
        )?;
        let off_prompt_len = rt
            .tokenizer
            .encode(raw_prompt.as_str(), false)
            .map_err(|e| format!("axis2 prompt len error: {e}"))?
            .get_ids()
            .len();
        let off_gen_ids = &off_all_ids[off_prompt_len.min(off_all_ids.len())..];

        // Steer ON (LiminiKaHybrid gating)
        let fact_ctx = plan.self_ref_context();
        let ext_ctx = plan.exterior_context();
        let on_prompt = if is_unobs {
            raw_prompt.clone()
        } else if !fact_ctx.is_empty() {
            format_phi4_prompt(&format!("[Context: {fact_ctx}]\n{}", probe.prompt))
        } else if plan.is_known_unknown_exterior() && !ext_ctx.is_empty() {
            format_phi4_prompt(&format!("[External Pointer: {ext_ctx}]\n{}", probe.prompt))
        } else {
            raw_prompt.clone()
        };
        let use_steer = plan.steering_eligible();
        let steer_facts = if plan.has_self_ref_anchors() {
            plan.self_ref_anchors_per_layer()
        } else if plan.is_known_unknown_exterior() {
            plan.exterior_sandbox_anchors_per_layer()
        } else {
            plan.self_ref_anchors_per_layer()
        };
        let mut on_cfg = SteerDecodeConfig {
            q7s: Some(baked.fact_q7.as_slice()),
            persona_q7s: Some(baked.persona_q7.as_slice()),
            resonant_ctx: if use_steer {
                Some(ResonantAnchors {
                    facts: steer_facts,
                    personas: plan.persona_anchors_per_layer(),
                    guards: plan.guard_anchors_per_layer(),
                    negs: baked.repulsive_centroid_per_layer.as_slice(),
                    poss: baked.attractive_centroid_per_layer.as_slice(),
                })
            } else {
                None
            },
            rope_rad: if use_steer { -2.00 } else { 0.00 },
            rank1: if use_steer {
                Some((baked.rank1_u.as_slice(), baked.rank1_v.as_slice(), -1.00))
            } else {
                None
            },
            use_steer,
            horizon_mode: cli.horizon_mode,
            escalation_provider: None,
            pressure_sim: cli.pressure_sim_mode,
            enable_pre_bypass: cli.enable_pre_bypass,
            temperature: 0.0,
            rng_seed: cli.rng_seed,
            logits_observe_stride: cli.logits_observe_stride,
            layer_steer_mask: Some(eff_mask),
            probe_frame: cli.probe_frame,
            phase_clamp_override: cli.phase_clamp_rad,
            random_push_seed: cli.random_push_control.then_some(cli.rng_seed),
            max_tokens: cli.max_tokens.min(32),
            prefill_chunk_size: cli.prefill_chunk_size,
            capture_first_logits: true,
            first_prefill_logits: None,
            exit_entropy_h_tilde: 0.0,
            steer_bypass_hits: 0,
            steer_eligible_layers: 0,
            spine_veto_guard_hits: 0,
            spine_veto_repulse_hits: 0,
            spine_veto_pin_hits: 0,
        };
        rt.model.clear_cache();
        let (_, on_all_ids, _) = generate_steered_reply_with_offset(
            rt.model,
            rt.tokenizer,
            rt.adapter,
            &on_prompt,
            0,
            &mut on_cfg,
        )?;
        let on_prompt_len = rt
            .tokenizer
            .encode(on_prompt.as_str(), false)
            .map_err(|e| format!("axis2 on prompt len error: {e}"))?
            .get_ids()
            .len();
        let on_gen_ids = &on_all_ids[on_prompt_len.min(on_all_ids.len())..];

        if off_gen_ids == on_gen_ids {
            exact_tok_match += 1;
        } else {
            mismatched_indices.push(format!("#{}:{}", idx + 1, probe.category));
        }
        if let (Some(off_l), Some(on_l)) = (
            off_cfg.first_prefill_logits.as_deref(),
            on_cfg.first_prefill_logits.as_deref(),
        ) {
            let (kl, max_dl) = kl_divergence_and_max_abs_delta(off_l, on_l);
            kl_sum += kl;
            kl_max = kl_max.max(kl);
            dlogit_sum += max_dl;
            dlogit_max = dlogit_max.max(max_dl);
        }
    }

    let n = ood_probes.len().max(1) as f64;
    println!(
        "[Axis2 Summary] unobserved_lacuna={:.1}% ({}/{}) | exact_token_match={:.1}% ({}/{}) | \
         first_logit_KL: mean={:.6} max={:.6} | max_abs_dlogit: mean={:.6} max={:.6}",
        pct(unobserved_count, ood_probes.len()),
        unobserved_count,
        ood_probes.len(),
        pct(exact_tok_match, ood_probes.len()),
        exact_tok_match,
        ood_probes.len(),
        kl_sum / n,
        kl_max,
        dlogit_sum / n,
        dlogit_max,
    );
    if !leaked_indices.is_empty() || !mismatched_indices.is_empty() {
        println!(
            "  [Axis2 Leaks] non_unobserved_ids={leaked_indices:?} token_mismatch_ids={mismatched_indices:?}"
        );
    }
    Ok(())
}

fn run_axis3_repulsion_polarity(
    cli: &EvalCliConfig,
    repulsion_probes: &[RepulsionProbe],
    rt: &mut EvalRuntime<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "\n--- Axis 3: Repulsive-Polarity Preservation ({} Repulsion Probes) ---",
        repulsion_probes.len()
    );
    let Some(baked) = rt.baked_opt else {
        println!("  [Skipped] recall_only mode has no baked model profile.");
        return Ok(());
    };
    let num_heads = rt.adapter.geometry().num_q_heads;

    // (i-a) Direct GuardPhaseAnchor anti-phase dot-product verification across all layers.
    let mut guard_checked = 0_usize;
    let mut guard_antiphase = 0_usize;
    for layer_guards in &baked.guard_anchors_per_layer {
        for g in layer_guards {
            if !is_nonzero_q7(&g.phase_deltas, num_heads) {
                continue;
            }
            guard_checked += 1;
            let frame = GuardPhaseAnchor::new(g.phase_deltas, 1.0, 0.5).synthesize_gateway_frame();
            if dot_q7(&g.phase_deltas, &frame.phase_deltas, num_heads) < 0 {
                guard_antiphase += 1;
            }
        }
    }

    // (i-b) CorpusRecallEscalationProvider anti-phase verification across repulsion escalation.
    let mut esc_checked = 0_usize;
    let mut esc_antiphase = 0_usize;
    let mut esc_provider = CorpusRecallEscalationProvider::new(&rt.compiled.corpus);
    let best_guard = rt
        .compiled
        .corpus
        .entries()
        .iter()
        .filter(|e| e.role == AttractorRole::Guard && !e.per_layer_q7.is_empty())
        .max_by(|a, b| a.energy.total_cmp(&b.energy));
    if let Some(guard_entry) = best_guard {
        let n_layers = guard_entry.per_layer_q7.len();
        for (layer_idx, ref_q7) in guard_entry.per_layer_q7.iter().enumerate() {
            let scale_q7 = if n_layers <= 1 {
                0
            } else {
                ((layer_idx * 127) / (n_layers - 1)).min(127) as u8
            };
            let req = HorizonEscalationRequest {
                trigger: HorizonTriggerInput {
                    entropy_gate_mode: 0,
                    moving_z_entropy: 0.85,
                    repulsion_severity: 0.90,
                    causal_root_cause: 0,
                    epistemic_regime: EpistemicRegime::InternalConflict as u8,
                    stationary_shortcut_ready: false,
                    ego_active_triggers: 0,
                    unstable_dwell_steps: 4,
                    quiescent_dwell_steps: 0,
                },
                placement: WorkbenchLanePlacement::AutoFree,
                lane_flags: WORKBENCH_LANE_FLAG_REPULSION,
                layer_scale_q7: scale_q7,
                desired_aperture_mask: 0x0F,
                current_frame_energy: 2.0,
            };
            let mut slot_out = [0_i8; MAX_ATTENTION_HEADS];
            if let Ok(Some(_)) = esc_provider.try_escalate_horizon(&req, &mut slot_out) {
                if is_nonzero_q7(ref_q7, num_heads) {
                    esc_checked += 1;
                    if dot_q7(ref_q7, &slot_out, num_heads) < 0 {
                        esc_antiphase += 1;
                    }
                }
            }
        }
    }

    // (ii) Behavioral forbidden-token Delta log P and sem_forb across novel repulsion probes.
    rt.adapter.set_layer_rope_mask(DEFAULT_LAYER_STEER_MASK);
    rt.adapter.reset_horizon_and_core_state();
    rt.adapter.set_guard_hold_tokens(cli.guard_hold_tokens);
    rt.adapter.reanchor_compute_pressure(8192);
    rt.adapter.set_ego_max_hops(cli.ego_max_hops);
    apply_ego_mode(rt.adapter, &cli.ego_mode);

    let kv_policy = AdaptiveThresholdPolicy::default();
    let mut kv_cache = StandardPhaseKvGopCache::new();

    let mut guard_traps = 0_usize;
    let mut dlp_hybrid_le0 = 0_usize;
    let mut dlp_hybrid_total = 0_usize;
    let mut dlp_hybrid_sum = 0.0_f64;
    let mut dlp_rope_le0 = 0_usize;
    let mut dlp_rope_total = 0_usize;
    let mut dlp_rope_sum = 0.0_f64;
    let mut off_sem_forb = 0_usize;
    let mut on_sem_forb = 0_usize;
    let mut forb_kw_total = 0_usize;

    for (idx, probe) in repulsion_probes.iter().enumerate() {
        let q_ids = rt
            .tokenizer
            .encode(probe.prompt.as_str(), false)
            .map_err(|e| format!("axis3 encode error: {e}"))?
            .get_ids()
            .to_vec();
        let (_, query_tok_qs, _) = rt.model.forward_tokens(
            &q_ids,
            0,
            None,
            false,
            true,
            LogitsFetchMode::FullLogits,
            KvCacheMode::IsolatedProbe,
        )?;
        let plan = TurnOrchestrator::prepare_turn(
            rt.compiled,
            rt.adapter,
            &mut kv_cache,
            &TurnPrepareInput {
                query_token_qs: &query_tok_qs,
                query_seq_len: q_ids.len(),
                turn_index: (idx + 1) as u32,
                kv_policy: &kv_policy,
                kv_min_resonance_override: cli.kv_min_res,
                persona_bank: rt.active_persona_bank.as_ref(),
                layer_steer_mask_override: cli.layer_steer_mask_override,
            },
        )?;
        if plan.is_guard_trap() {
            guard_traps += 1;
        }
        let eff_mask = plan.effective_layer_steer_mask(rt.adapter);
        let raw_prompt = format_phi4_prompt(&probe.prompt);
        let fact_ctx = plan.self_ref_context();
        let ext_ctx = plan.exterior_context();
        let hybrid_prompt = if !fact_ctx.is_empty() {
            format_phi4_prompt(&format!("[Context: {fact_ctx}]\n{}", probe.prompt))
        } else if plan.is_known_unknown_exterior() && !ext_ctx.is_empty() {
            format_phi4_prompt(&format!("[External Pointer: {ext_ctx}]\n{}", probe.prompt))
        } else {
            raw_prompt.clone()
        };
        let steer_facts = if plan.has_self_ref_anchors() {
            plan.self_ref_anchors_per_layer()
        } else if plan.is_known_unknown_exterior() {
            plan.exterior_sandbox_anchors_per_layer()
        } else {
            plan.self_ref_anchors_per_layer()
        };

        // Baseline OFF on raw prompt
        let mut off_cfg = SteerDecodeConfig {
            q7s: None,
            persona_q7s: None,
            resonant_ctx: None,
            rope_rad: 0.0,
            rank1: None,
            use_steer: false,
            horizon_mode: cli.horizon_mode,
            escalation_provider: None,
            pressure_sim: cli.pressure_sim_mode,
            enable_pre_bypass: false,
            temperature: 0.0,
            rng_seed: cli.rng_seed,
            logits_observe_stride: 1,
            layer_steer_mask: Some(eff_mask),
            probe_frame: cli.probe_frame,
            phase_clamp_override: None,
            random_push_seed: None,
            max_tokens: cli.max_tokens.min(48),
            prefill_chunk_size: cli.prefill_chunk_size,
            capture_first_logits: true,
            first_prefill_logits: None,
            exit_entropy_h_tilde: 0.0,
            steer_bypass_hits: 0,
            steer_eligible_layers: 0,
            spine_veto_guard_hits: 0,
            spine_veto_repulse_hits: 0,
            spine_veto_pin_hits: 0,
        };
        let (off_reply, _, _, _) = generate_steered_reply(
            rt.model,
            rt.tokenizer,
            rt.adapter,
            &raw_prompt,
            &mut off_cfg,
        )?;

        // Pure RoPE/Guard prefill probe on identical raw prompt (isolates geometric delta)
        let mut rope_only_cfg = SteerDecodeConfig {
            q7s: Some(baked.fact_q7.as_slice()),
            persona_q7s: Some(baked.persona_q7.as_slice()),
            resonant_ctx: Some(ResonantAnchors {
                facts: steer_facts,
                personas: plan.persona_anchors_per_layer(),
                guards: plan.guard_anchors_per_layer(),
                negs: baked.repulsive_centroid_per_layer.as_slice(),
                poss: baked.attractive_centroid_per_layer.as_slice(),
            }),
            rope_rad: -2.00,
            rank1: Some((baked.rank1_u.as_slice(), baked.rank1_v.as_slice(), -1.00)),
            use_steer: true,
            horizon_mode: cli.horizon_mode,
            escalation_provider: None,
            pressure_sim: cli.pressure_sim_mode,
            enable_pre_bypass: cli.enable_pre_bypass,
            temperature: 0.0,
            rng_seed: cli.rng_seed,
            logits_observe_stride: 1,
            layer_steer_mask: Some(eff_mask),
            probe_frame: cli.probe_frame,
            phase_clamp_override: cli.phase_clamp_rad,
            random_push_seed: cli.random_push_control.then_some(cli.rng_seed),
            max_tokens: 1,
            prefill_chunk_size: cli.prefill_chunk_size,
            capture_first_logits: true,
            first_prefill_logits: None,
            exit_entropy_h_tilde: 0.0,
            steer_bypass_hits: 0,
            steer_eligible_layers: 0,
            spine_veto_guard_hits: 0,
            spine_veto_repulse_hits: 0,
            spine_veto_pin_hits: 0,
        };
        let _ = generate_steered_reply(
            rt.model,
            rt.tokenizer,
            rt.adapter,
            &raw_prompt,
            &mut rope_only_cfg,
        )?;

        // Full Hybrid ON
        let use_steer = plan.steering_eligible();
        let mut on_cfg = SteerDecodeConfig {
            q7s: Some(baked.fact_q7.as_slice()),
            persona_q7s: Some(baked.persona_q7.as_slice()),
            resonant_ctx: if use_steer {
                Some(ResonantAnchors {
                    facts: steer_facts,
                    personas: plan.persona_anchors_per_layer(),
                    guards: plan.guard_anchors_per_layer(),
                    negs: baked.repulsive_centroid_per_layer.as_slice(),
                    poss: baked.attractive_centroid_per_layer.as_slice(),
                })
            } else {
                None
            },
            rope_rad: if use_steer { -2.00 } else { 0.00 },
            rank1: if use_steer {
                Some((baked.rank1_u.as_slice(), baked.rank1_v.as_slice(), -1.00))
            } else {
                None
            },
            use_steer,
            horizon_mode: cli.horizon_mode,
            escalation_provider: None,
            pressure_sim: cli.pressure_sim_mode,
            enable_pre_bypass: cli.enable_pre_bypass,
            temperature: 0.0,
            rng_seed: cli.rng_seed,
            logits_observe_stride: cli.logits_observe_stride,
            layer_steer_mask: Some(eff_mask),
            probe_frame: cli.probe_frame,
            phase_clamp_override: cli.phase_clamp_rad,
            random_push_seed: cli.random_push_control.then_some(cli.rng_seed),
            max_tokens: cli.max_tokens.min(48),
            prefill_chunk_size: cli.prefill_chunk_size,
            capture_first_logits: true,
            first_prefill_logits: None,
            exit_entropy_h_tilde: 0.0,
            steer_bypass_hits: 0,
            steer_eligible_layers: 0,
            spine_veto_guard_hits: 0,
            spine_veto_repulse_hits: 0,
            spine_veto_pin_hits: 0,
        };
        let (on_reply, _, _, _) = generate_steered_reply(
            rt.model,
            rt.tokenizer,
            rt.adapter,
            &hybrid_prompt,
            &mut on_cfg,
        )?;

        let (_, off_sf) = count_semantic_forbid_hits(&off_reply, &probe.forbidden);
        let (_, on_sf) = count_semantic_forbid_hits(&on_reply, &probe.forbidden);
        off_sem_forb += off_sf;
        on_sem_forb += on_sf;
        forb_kw_total += probe.forbidden.len();

        if let Some(off_l) = off_cfg.first_prefill_logits.as_deref() {
            for kw in &probe.forbidden {
                let tok_opt = rt
                    .tokenizer
                    .encode(kw.as_str(), false)
                    .ok()
                    .and_then(|e| e.get_ids().first().copied());
                let Some(tok_id) = tok_opt else { continue };
                if let Some(hyb_l) = on_cfg.first_prefill_logits.as_deref() {
                    if let Some(dlp) = delta_log_prob(off_l, hyb_l, tok_id) {
                        dlp_hybrid_total += 1;
                        dlp_hybrid_sum += dlp;
                        if dlp <= 0.0 {
                            dlp_hybrid_le0 += 1;
                        }
                    }
                }
                if let Some(rope_l) = rope_only_cfg.first_prefill_logits.as_deref() {
                    if let Some(dlp) = delta_log_prob(off_l, rope_l, tok_id) {
                        dlp_rope_total += 1;
                        dlp_rope_sum += dlp;
                        if dlp <= 0.0 {
                            dlp_rope_le0 += 1;
                        }
                    }
                }
            }
        }
    }

    println!(
        "[Axis3 Summary] guard_antiphase={:.1}% ({}/{}) | esc_antiphase={:.1}% ({}/{}) | \
         guard_trap={:.1}% ({}/{}) | d_logP_hybrid<=0={:.1}% ({}/{}, mean={:+.4}) | \
         d_logP_rope<=0={:.1}% ({}/{}, mean={:+.4}) | sem_forb: OFF={:.1}% ({}/{}) -> ON={:.1}% ({}/{})",
        pct(guard_antiphase, guard_checked),
        guard_antiphase,
        guard_checked,
        pct(esc_antiphase, esc_checked),
        esc_antiphase,
        esc_checked,
        pct(guard_traps, repulsion_probes.len()),
        guard_traps,
        repulsion_probes.len(),
        pct(dlp_hybrid_le0, dlp_hybrid_total),
        dlp_hybrid_le0,
        dlp_hybrid_total,
        if dlp_hybrid_total == 0 {
            0.0
        } else {
            dlp_hybrid_sum / dlp_hybrid_total as f64
        },
        pct(dlp_rope_le0, dlp_rope_total),
        dlp_rope_le0,
        dlp_rope_total,
        if dlp_rope_total == 0 {
            0.0
        } else {
            dlp_rope_sum / dlp_rope_total as f64
        },
        pct(off_sem_forb, forb_kw_total),
        off_sem_forb,
        forb_kw_total,
        pct(on_sem_forb, forb_kw_total),
        on_sem_forb,
        forb_kw_total,
    );
    Ok(())
}

fn run_axis4_partition_non_conflation(
    cli: &EvalCliConfig,
    partition_probes: &[PartitionProbe],
    rt: &mut EvalRuntime<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "\n--- Axis 4: SelfRef (S & O) vs Exterior (O \\ S) Non-Conflation ({} Probes) ---",
        partition_probes.len()
    );
    rt.adapter.set_layer_rope_mask(DEFAULT_LAYER_STEER_MASK);
    rt.adapter.reset_horizon_and_core_state();
    rt.adapter.set_guard_hold_tokens(cli.guard_hold_tokens);
    rt.adapter.reanchor_compute_pressure(8192);
    rt.adapter.set_ego_max_hops(cli.ego_max_hops);
    apply_ego_mode(rt.adapter, &cli.ego_mode);

    let kv_policy = AdaptiveThresholdPolicy::default();
    let mut kv_cache = StandardPhaseKvGopCache::new();
    let settlements_before = rt.adapter.sandbox_settlements();

    let mut disjoint_ok = 0_usize;
    let mut self_ref_n = 0_usize;
    let mut self_ref_hit = 0_usize;
    let mut self_ref_no_ext = 0_usize;
    let mut ext_n = 0_usize;
    let mut ext_self_empty = 0_usize;
    let mut ext_known_unknown = 0_usize;
    let mut ext_unobserved = 0_usize;
    let mut ood_n = 0_usize;
    let mut ood_unobserved = 0_usize;
    let mut ood_empty_both = 0_usize;

    for (idx, probe) in partition_probes.iter().enumerate() {
        let q_ids = rt
            .tokenizer
            .encode(probe.prompt.as_str(), false)
            .map_err(|e| format!("axis4 encode error: {e}"))?
            .get_ids()
            .to_vec();
        let (_, query_tok_qs, _) = rt.model.forward_tokens(
            &q_ids,
            0,
            None,
            false,
            true,
            LogitsFetchMode::FullLogits,
            KvCacheMode::IsolatedProbe,
        )?;
        let plan = TurnOrchestrator::prepare_turn(
            rt.compiled,
            rt.adapter,
            &mut kv_cache,
            &TurnPrepareInput {
                query_token_qs: &query_tok_qs,
                query_seq_len: q_ids.len(),
                turn_index: (idx + 1) as u32,
                kv_policy: &kv_policy,
                kv_min_resonance_override: cli.kv_min_res,
                persona_bank: rt.active_persona_bank.as_ref(),
                layer_steer_mask_override: cli.layer_steer_mask_override,
            },
        )?;

        let self_ids: HashSet<u32> = plan
            .self_ref_anchors_per_layer()
            .iter()
            .flat_map(|layer| layer.iter().map(|a| a.anchor_id))
            .collect();
        let ext_ids: HashSet<u32> = plan
            .exterior_sandbox_anchors_per_layer()
            .iter()
            .flat_map(|layer| layer.iter().map(|a| a.anchor_id))
            .collect();
        let is_disjoint = self_ids.is_disjoint(&ext_ids);
        if is_disjoint {
            disjoint_ok += 1;
        }

        match probe.class {
            PartitionClass::SelfRef => {
                self_ref_n += 1;
                if plan.has_self_ref_anchors() {
                    self_ref_hit += 1;
                }
                if plan.exterior_context().is_empty() && !plan.is_known_unknown_exterior() {
                    self_ref_no_ext += 1;
                }
            }
            PartitionClass::Exterior => {
                ext_n += 1;
                if plan.self_ref_context().is_empty() {
                    ext_self_empty += 1;
                }
                if plan.is_known_unknown_exterior() {
                    ext_known_unknown += 1;
                }
                if plan.is_unobserved_lacuna() {
                    ext_unobserved += 1;
                }
            }
            PartitionClass::Ood => {
                ood_n += 1;
                if plan.is_unobserved_lacuna() {
                    ood_unobserved += 1;
                }
                if plan.self_ref_context().is_empty() && plan.exterior_context().is_empty() {
                    ood_empty_both += 1;
                }
            }
        }

        // Exercise a short decode step on Exterior / OOD when steering is eligible to verify
        // zero unverified `sandbox_settlements` misfires.
        if let Some(baked) = rt.baked_opt {
            if !matches!(probe.class, PartitionClass::SelfRef) && plan.steering_eligible() {
                let eff_mask = plan.effective_layer_steer_mask(rt.adapter);
                let steer_facts = if plan.has_self_ref_anchors() {
                    plan.self_ref_anchors_per_layer()
                } else {
                    plan.exterior_sandbox_anchors_per_layer()
                };
                let prompt = format_phi4_prompt(&probe.prompt);
                let mut dec_cfg = SteerDecodeConfig {
                    q7s: Some(baked.fact_q7.as_slice()),
                    persona_q7s: Some(baked.persona_q7.as_slice()),
                    resonant_ctx: Some(ResonantAnchors {
                        facts: steer_facts,
                        personas: plan.persona_anchors_per_layer(),
                        guards: plan.guard_anchors_per_layer(),
                        negs: baked.repulsive_centroid_per_layer.as_slice(),
                        poss: baked.attractive_centroid_per_layer.as_slice(),
                    }),
                    rope_rad: -2.00,
                    rank1: Some((baked.rank1_u.as_slice(), baked.rank1_v.as_slice(), -1.00)),
                    use_steer: true,
                    horizon_mode: cli.horizon_mode,
                    escalation_provider: None,
                    pressure_sim: cli.pressure_sim_mode,
                    enable_pre_bypass: cli.enable_pre_bypass,
                    temperature: 0.0,
                    rng_seed: cli.rng_seed,
                    logits_observe_stride: cli.logits_observe_stride,
                    layer_steer_mask: Some(eff_mask),
                    probe_frame: cli.probe_frame,
                    phase_clamp_override: cli.phase_clamp_rad,
                    random_push_seed: cli.random_push_control.then_some(cli.rng_seed),
                    max_tokens: 16,
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
                let _ = generate_steered_reply(
                    rt.model,
                    rt.tokenizer,
                    rt.adapter,
                    &prompt,
                    &mut dec_cfg,
                )?;
            }
        }

        println!(
            "  [P{:02}:{:?}] disjoint={} lacuna={:?} top1={:.3} unexp={:.2} br={} rho={:.2} \
             self_ids={} ext_ids={} known_unk={} unobs={} ({})",
            idx + 1,
            probe.class,
            is_disjoint,
            plan.lacuna_state(),
            plan.top1_fact_score(),
            plan.telemetry().unexplained_token_ratio,
            plan.lacuna_partition().bridge_count,
            plan.telemetry().query_ortho_ratio,
            self_ids.len(),
            ext_ids.len(),
            plan.is_known_unknown_exterior(),
            plan.is_unobserved_lacuna(),
            probe.note,
        );
    }

    let settlements_delta = rt
        .adapter
        .sandbox_settlements()
        .saturating_sub(settlements_before);
    println!(
        "[Axis4 Summary] anchor_id_disjoint={:.1}% ({}/{}) | \
         SELF_REF(hit={:.1}% {}/{}, ext_empty={:.1}% {}/{}) | \
         EXTERIOR(self_empty={:.1}% {}/{}, known_unk={:.1}% {}/{}, unobs={:.1}% {}/{}) | \
         OOD(unobs={:.1}% {}/{}, both_empty={:.1}% {}/{}) | \
         unverified_sandbox_settlements={}",
        pct(disjoint_ok, partition_probes.len()),
        disjoint_ok,
        partition_probes.len(),
        pct(self_ref_hit, self_ref_n),
        self_ref_hit,
        self_ref_n,
        pct(self_ref_no_ext, self_ref_n),
        self_ref_no_ext,
        self_ref_n,
        pct(ext_self_empty, ext_n),
        ext_self_empty,
        ext_n,
        pct(ext_known_unknown, ext_n),
        ext_known_unknown,
        ext_n,
        pct(ext_unobserved, ext_n),
        ext_unobserved,
        ext_n,
        pct(ood_unobserved, ood_n),
        ood_unobserved,
        ood_n,
        pct(ood_empty_both, ood_n),
        ood_empty_both,
        ood_n,
        settlements_delta,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake::load_fixture;

    fn repo_root() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("repo root")
    }

    #[test]
    fn ood_selectivity_fixture_has_50_uncontaminated_prompts() {
        let path = repo_root().join("tests/fixtures/ood_selectivity_probes.txt");
        let probes = load_ood_probes(&path).expect("load ood fixture");
        assert_eq!(probes.len(), 50);

        let forbidden_domain_terms = [
            "gcso",
            "liminika",
            "zimms",
            "cstl",
            "actionhub",
            "ripa",
            "edbc",
            "pprc",
            "gatewayframe",
        ];
        let mut unique_prompts = HashSet::new();
        for p in &probes {
            assert!(!p.category.is_empty());
            assert!(!p.prompt.is_empty());
            assert!(
                unique_prompts.insert(p.prompt.clone()),
                "duplicate OOD prompt: {}",
                p.prompt
            );
            let lower = p.prompt.to_ascii_lowercase();
            for term in forbidden_domain_terms {
                assert!(
                    !lower.contains(term),
                    "OOD prompt contaminated with domain term `{term}`: {}",
                    p.prompt
                );
            }
        }
    }

    #[test]
    fn steering_physics_fixture_parses_cleanly_without_verbatim_corpus_copies() {
        let root = repo_root();
        let gcso = load_fixture(&root.join("tests/fixtures/gcso_eval_samples.txt"))
            .expect("load gcso fixture");
        let suite =
            load_steering_physics_probes(&root.join("tests/fixtures/steering_physics_probes.txt"))
                .expect("load steering physics fixture");

        assert!(
            (12..=20).contains(&suite.repulsion_probes.len()),
            "expected 12..=20 repulsion probes, got {}",
            suite.repulsion_probes.len()
        );
        for rp in &suite.repulsion_probes {
            assert!(!rp.prompt.is_empty());
            assert!(!rp.forbidden.is_empty());
        }

        let self_ref_n = suite
            .partition_probes
            .iter()
            .filter(|p| p.class == PartitionClass::SelfRef)
            .count();
        let ext_n = suite
            .partition_probes
            .iter()
            .filter(|p| p.class == PartitionClass::Exterior)
            .count();
        let ood_n = suite
            .partition_probes
            .iter()
            .filter(|p| p.class == PartitionClass::Ood)
            .count();
        assert!(self_ref_n >= 6, "expected >= 6 SELF_REF probes");
        assert!(ext_n >= 6, "expected >= 6 EXTERIOR probes");
        assert!(ood_n >= 6, "expected >= 6 OOD probes");

        let mut existing_lines: HashSet<String> = HashSet::new();
        for s in gcso
            .facts
            .iter()
            .chain(gcso.persona.iter())
            .chain(gcso.guards.iter())
        {
            existing_lines.insert(s.trim().to_ascii_lowercase());
        }
        for (_, prompt, _, _) in &gcso.probes {
            existing_lines.insert(prompt.trim().to_ascii_lowercase());
        }
        for rp in &suite.repulsion_probes {
            assert!(
                !existing_lines.contains(&rp.prompt.trim().to_ascii_lowercase()),
                "repulsion probe must not verbatim-copy gcso_eval_samples.txt: {}",
                rp.prompt
            );
        }
        for pp in &suite.partition_probes {
            assert!(
                !existing_lines.contains(&pp.prompt.trim().to_ascii_lowercase()),
                "partition probe must not verbatim-copy gcso_eval_samples.txt: {}",
                pp.prompt
            );
        }
    }

    #[test]
    fn startup_prior_probes_have_zero_overlap_with_all_eval_fixtures() {
        use liminika_adapter::DEFAULT_STARTUP_PRIOR_PROBES;

        let root = repo_root();
        let gcso = load_fixture(&root.join("tests/fixtures/gcso_eval_samples.txt"))
            .expect("load gcso fixture");
        let ood = load_ood_probes(&root.join("tests/fixtures/ood_selectivity_probes.txt"))
            .expect("load ood fixture");
        let phys =
            load_steering_physics_probes(&root.join("tests/fixtures/steering_physics_probes.txt"))
                .expect("load steering physics fixture");

        assert!(
            (10..=12).contains(&DEFAULT_STARTUP_PRIOR_PROBES.len()),
            "expected 10..=12 neutral startup prior probes, got {}",
            DEFAULT_STARTUP_PRIOR_PROBES.len()
        );

        let mut fixture_lines: HashSet<String> = HashSet::new();
        for s in gcso
            .facts
            .iter()
            .chain(gcso.persona.iter())
            .chain(gcso.guards.iter())
        {
            fixture_lines.insert(s.trim().to_ascii_lowercase());
        }
        for (_, prompt, _, _) in &gcso.probes {
            fixture_lines.insert(prompt.trim().to_ascii_lowercase());
        }
        for p in &ood {
            fixture_lines.insert(p.prompt.trim().to_ascii_lowercase());
        }
        for rp in &phys.repulsion_probes {
            fixture_lines.insert(rp.prompt.trim().to_ascii_lowercase());
        }
        for pp in &phys.partition_probes {
            fixture_lines.insert(pp.prompt.trim().to_ascii_lowercase());
        }

        let mut unique_prior: HashSet<String> = HashSet::new();
        for prompt in DEFAULT_STARTUP_PRIOR_PROBES {
            let norm = prompt.trim().to_ascii_lowercase();
            assert!(!norm.is_empty());
            assert!(
                unique_prior.insert(norm.clone()),
                "duplicate startup prior probe: {prompt}"
            );
            assert!(
                !fixture_lines.contains(&norm),
                "startup prior probe overlaps eval fixture: {prompt}"
            );
            for term in ["gcso", "liminika", "zimms", "cstl", "ripa", "edbc", "pprc"] {
                assert!(
                    !norm.contains(term),
                    "startup prior probe contains domain term `{term}`: {prompt}"
                );
            }
        }
    }
}
