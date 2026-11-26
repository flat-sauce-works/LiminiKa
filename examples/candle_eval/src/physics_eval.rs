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

use std::path::Path;

use liminika_adapter::{TurnOrchestrator, TurnPrepareInput};
use liminika_core::runtime::{AdaptiveThresholdPolicy, StandardPhaseKvGopCache};

use crate::bake::query_nvidia_smi_vram;
use crate::cli::{EvalCliConfig, EvalRuntime};
use crate::eval::{
    apply_ego_mode, count_semantic_forbid_hits, generate_steered_reply_with_offset,
    SteerDecodeConfig,
};
use crate::eval_common::pct;
use crate::eval_suites::{
    run_ood_selectivity_eval, run_partition_eval, run_repulsion_eval, ForbiddenTokenMode,
    OodSelectivitySpec, PartitionEvalSpec, PartitionProbeRef, RepulsionEvalSpec, RepulsionProbeRef,
};
use crate::metrics::{summarize_dialogue_quality, ReplyQualityMetrics};
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
        rt.active_layer_mask,
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

    rt.adapter.set_layer_rope_mask(rt.active_layer_mask);
    rt.adapter.reset_horizon_and_core_state();
    rt.adapter.set_guard_hold_tokens(cli.guard_hold_tokens);
    rt.adapter.reanchor_compute_pressure(8192);
    rt.adapter.set_ego_max_hops(cli.ego_max_hops);
    apply_ego_mode(rt.adapter, &cli.ego_mode);

    let layer_mask_override = cli.resolve_layer_mask_override(rt.cfg.num_hidden_layers);
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
                layer_steer_mask_override: layer_mask_override,
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
        let sys_prompt = "Answer concisely and directly from the provided [Context: ...] facts \
            without questioning whether terms are real; reject any premise in a question \
            that contradicts the context.";
        let formatted = if turn_start_kv == 0 {
            rt.chat
                .format_first_turn_with_system(sys_prompt, &turn_body)
        } else {
            rt.chat.format_followup_turn(&turn_body)
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
            rt.chat,
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
    let probe_pairs: Vec<(&str, &str)> = ood_probes
        .iter()
        .map(|p| (p.category.as_str(), p.prompt.as_str()))
        .collect();
    run_ood_selectivity_eval(
        &OodSelectivitySpec {
            header: format!(
                "\n--- Axis 2: Unrelated-Prompt Non-Interference ({} OOD Prompts, Greedy) ---",
                ood_probes.len()
            ),
            error_prefix: "axis2",
            suite_tag: None,
            max_tokens: cli.max_tokens.min(32),
            probes: &probe_pairs,
        },
        cli,
        rt,
    )
}

fn run_axis3_repulsion_polarity(
    cli: &EvalCliConfig,
    repulsion_probes: &[RepulsionProbe],
    rt: &mut EvalRuntime<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let probe_refs: Vec<RepulsionProbeRef<'_>> = repulsion_probes
        .iter()
        .map(|p| RepulsionProbeRef {
            style: None,
            prompt: p.prompt.as_str(),
            forbidden: p.forbidden.as_slice(),
        })
        .collect();
    run_repulsion_eval(
        &RepulsionEvalSpec {
            header: format!(
                "\n--- Axis 3: Repulsive-Polarity Preservation ({} Repulsion Probes) ---",
                repulsion_probes.len()
            ),
            error_prefix: "axis3",
            suite_tag: None,
            max_tokens: cli.max_tokens.min(48),
            token_mode: ForbiddenTokenMode::FirstSubword,
            extra_neg_cues: &[],
            probes: &probe_refs,
        },
        cli,
        rt,
    )
}

fn run_axis4_partition_non_conflation(
    cli: &EvalCliConfig,
    partition_probes: &[PartitionProbe],
    rt: &mut EvalRuntime<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let probe_refs: Vec<PartitionProbeRef<'_>> = partition_probes
        .iter()
        .map(|p| PartitionProbeRef {
            class: p.class,
            style: None,
            prompt: p.prompt.as_str(),
            note: p.note.as_str(),
        })
        .collect();
    run_partition_eval(
        &PartitionEvalSpec {
            header: format!(
                "\n--- Axis 4: SelfRef (S & O) vs Exterior (O \\ S) Non-Conflation ({} Probes) ---",
                partition_probes.len()
            ),
            error_prefix: "axis4",
            suite_tag: None,
            probes: &probe_refs,
        },
        cli,
        rt,
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

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
        let archive_fixture = load_fixture(&root.join("tests/fixtures/gcso_eval_samples.txt"))
            .expect("load archive fixture");
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
        for s in archive_fixture
            .facts
            .iter()
            .chain(archive_fixture.persona.iter())
            .chain(archive_fixture.guards.iter())
        {
            existing_lines.insert(s.trim().to_ascii_lowercase());
        }
        for (_, prompt, _, _) in &archive_fixture.probes {
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
        let archive_fixture = load_fixture(&root.join("tests/fixtures/gcso_eval_samples.txt"))
            .expect("load archive fixture");
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
        for s in archive_fixture
            .facts
            .iter()
            .chain(archive_fixture.persona.iter())
            .chain(archive_fixture.guards.iter())
        {
            fixture_lines.insert(s.trim().to_ascii_lowercase());
        }
        for (_, prompt, _, _) in &archive_fixture.probes {
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
