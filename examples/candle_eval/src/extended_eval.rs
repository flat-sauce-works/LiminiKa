// name: examples/candle_eval/src/extended_eval.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Extended English boundary/guard evaluation (`--extended-eval`) and Japanese multilingual
//! evaluation (`--ja-eval`) runners with strict cross-suite state isolation.
//!
//! Both suites reuse the unmodified `v0.3.0` core and adapter pipeline under identical greedy
//! decode settings (`det=true`, `max_tokens=64` by default, CLI-overridable) while reporting
//! `NullCalibrationEnvelope`, `EnvProbe`, per-class `(top1_coh, unexp, rho)` quantiles, OOD
//! non-interference, Guard anti-phase geometry, forbidden-token `Delta log P`, and UTF-8-safe
//! semantic forbidden keyword hit rates broken down by attack style.

use std::collections::HashSet;
use std::path::Path;

use liminika_adapter::{
    extract_leading_alias_word, AttractorRole, CompiledTokenCorpus, CorpusRecallEscalationProvider,
    TurnOrchestrator, TurnPrepareInput,
};
use liminika_core::abi::lmnk_q7_t;
use liminika_core::liminal::{EpistemicRegime, MAX_ATTENTION_HEADS};
use liminika_core::runtime::{
    AdaptiveThresholdPolicy, GuardPhaseAnchor, HorizonEscalationProvider, HorizonEscalationRequest,
    HorizonTriggerInput, ManifoldGatewaySignal, StandardPhaseKvGopCache,
};
use liminika_core::stigmergy::{
    CognitiveTrajectoryLedger, WorkbenchLanePlacement, WORKBENCH_LANE_FLAG_REPULSION,
};
use tokenizers::Tokenizer;

use crate::bake::{
    capture_and_build_shared_corpus, load_fixture, query_nvidia_smi_vram, FixtureSuite,
};
use crate::cli::{EvalCliConfig, EvalRuntime};
use crate::eval::{
    apply_ego_mode, count_semantic_forbid_hits, generate_steered_reply,
    generate_steered_reply_with_offset, CircuitProfileMode, SteerDecodeConfig,
};
use crate::metrics::{delta_log_prob, kl_divergence_and_max_abs_delta};
use crate::model::{
    CausalLmModel, ChatTemplate, KvCacheMode, LogitsFetchMode, ModelConfig, ResonantAnchors,
};
use crate::physics_eval::PartitionClass;
use crate::{
    build_persona_bank, calibrate_trial_adapter, configure_trial_adapter, resolve_fixture_path,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GuardAttackStyle {
    Direct,
    Indirect,
    Roleplay,
}

impl GuardAttackStyle {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Direct => "DIRECT",
            Self::Indirect => "INDIRECT",
            Self::Roleplay => "ROLEPLAY",
        }
    }

    fn parse(raw: &str) -> Option<Self> {
        match raw.trim() {
            "DIRECT" => Some(Self::Direct),
            "INDIRECT" => Some(Self::Indirect),
            "ROLEPLAY" => Some(Self::Roleplay),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ExtendedPartitionProbe {
    pub(crate) class: PartitionClass,
    pub(crate) style: String,
    pub(crate) prompt: String,
    pub(crate) note: String,
}

#[derive(Debug, Clone)]
pub(crate) struct ExtendedRepulsionProbe {
    pub(crate) style: GuardAttackStyle,
    pub(crate) prompt: String,
    pub(crate) forbidden: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ExtendedBoundarySuite {
    pub(crate) partition_probes: Vec<ExtendedPartitionProbe>,
    pub(crate) repulsion_probes: Vec<ExtendedRepulsionProbe>,
}

pub(crate) struct JaMultilingualSuite {
    pub(crate) corpus_fixture: FixtureSuite,
    pub(crate) negation_cues: Vec<String>,
    pub(crate) partition_probes: Vec<ExtendedPartitionProbe>,
    pub(crate) repulsion_probes: Vec<ExtendedRepulsionProbe>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct QuantileSummary {
    pub(crate) count: usize,
    pub(crate) min: f64,
    pub(crate) median: f64,
    pub(crate) max: f64,
}

pub(crate) fn compute_quantiles(values: &[f64]) -> QuantileSummary {
    let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return QuantileSummary::default();
    }
    finite.sort_by(f64::total_cmp);
    let n = finite.len();
    let mid = n / 2;
    let median = if n.is_multiple_of(2) {
        0.5 * (finite[mid - 1] + finite[mid])
    } else {
        finite[mid]
    };
    QuantileSummary {
        count: n,
        min: finite[0],
        median,
        max: finite[n - 1],
    }
}

pub(crate) fn load_extended_boundary_probes(
    path: &Path,
) -> Result<ExtendedBoundarySuite, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    let mut suite = ExtendedBoundarySuite::default();
    let mut section = "";
    for line in raw.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        match line {
            "[EXTENDED_PARTITION_PROBES]" => section = "PARTITION",
            "[EXTENDED_REPULSION_PROBES]" => section = "REPULSION",
            _ => match section {
                "PARTITION" => {
                    if let Some(p) = parse_partition_line(line) {
                        suite.partition_probes.push(p);
                    }
                }
                "REPULSION" => {
                    if let Some(rp) = parse_repulsion_line(line) {
                        suite.repulsion_probes.push(rp);
                    }
                }
                _ => {}
            },
        }
    }
    Ok(suite)
}

pub(crate) fn load_ja_multilingual_suite(
    path: &Path,
) -> Result<JaMultilingualSuite, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    let mut corpus_fixture = FixtureSuite {
        facts: Vec::new(),
        persona: Vec::new(),
        guards: Vec::new(),
        probes: Vec::new(),
    };
    let mut negation_cues = Vec::new();
    let mut partition_probes = Vec::new();
    let mut repulsion_probes = Vec::new();
    let mut section = "";
    for line in raw.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        match line {
            "[FACT_ATTRACTORS]" => section = "FACT",
            "[PERSONA_ATTRACTORS]" => section = "PERSONA",
            "[CONTRASTIVE_GUARDS]" => section = "GUARD",
            "[NEGATION_CUES]" => section = "NEGATION",
            "[JA_PARTITION_PROBES]" => section = "PARTITION",
            "[JA_REPULSION_PROBES]" => section = "REPULSION",
            _ => match section {
                "FACT" => corpus_fixture.facts.push(line.to_string()),
                "PERSONA" => corpus_fixture.persona.push(line.to_string()),
                "GUARD" => corpus_fixture.guards.push(line.to_string()),
                "NEGATION" => negation_cues.push(line.to_lowercase()),
                "PARTITION" => {
                    if let Some(p) = parse_partition_line(line) {
                        partition_probes.push(p);
                    }
                }
                "REPULSION" => {
                    if let Some(rp) = parse_repulsion_line(line) {
                        repulsion_probes.push(rp);
                    }
                }
                _ => {}
            },
        }
    }
    Ok(JaMultilingualSuite {
        corpus_fixture,
        negation_cues,
        partition_probes,
        repulsion_probes,
    })
}

fn parse_partition_line(line: &str) -> Option<ExtendedPartitionProbe> {
    let parts: Vec<&str> = line.split('|').map(str::trim).collect();
    if parts.len() != 4 || parts[2].is_empty() {
        return None;
    }
    let class = match parts[0] {
        "SELF_REF" => PartitionClass::SelfRef,
        "EXTERIOR" => PartitionClass::Exterior,
        "OOD" => PartitionClass::Ood,
        _ => return None,
    };
    Some(ExtendedPartitionProbe {
        class,
        style: parts[1].to_string(),
        prompt: parts[2].to_string(),
        note: parts[3].to_string(),
    })
}

fn parse_repulsion_line(line: &str) -> Option<ExtendedRepulsionProbe> {
    let parts: Vec<&str> = line.split('|').map(str::trim).collect();
    if parts.len() != 3 || parts[1].is_empty() {
        return None;
    }
    let style = GuardAttackStyle::parse(parts[0])?;
    let forbidden: Vec<String> = parts[2]
        .split(',')
        .map(str::trim)
        .filter(|k| !k.is_empty())
        .map(ToString::to_string)
        .collect();
    if forbidden.is_empty() {
        return None;
    }
    Some(ExtendedRepulsionProbe {
        style,
        prompt: parts[1].to_string(),
        forbidden,
    })
}

/// Counts `(raw_hits, semantic_hits)` across UTF-8 `forbidden` keywords without ASCII word-boundary
/// assumptions, suppressing matches inside explicit negation clauses.
///
/// When `extra_neg_cues` is empty, delegates directly to [`count_semantic_forbid_hits`] for exact
/// parity with `--physics-eval`.
pub(crate) fn count_semantic_forbid_hits_utf8(
    reply_text: &str,
    forbidden: &[String],
    extra_neg_cues: &[String],
) -> (usize, usize) {
    if extra_neg_cues.is_empty() {
        return count_semantic_forbid_hits(reply_text, forbidden);
    }
    const EN_PREFIX_NEG_CUES: &[&str] = &[
        "does not ",
        "do not ",
        "did not ",
        "is not ",
        "are not ",
        "cannot ",
        "can't",
        "doesn't",
        "don't",
        "isn't",
        "aren't",
        "never ",
        "not ",
        "no ",
        "without ",
        "avoid",
        "reject",
        "refute",
        "false",
        "incorrect",
    ];
    const EN_SUFFIX_NEG_CUES: &[&str] = &[
        " is false",
        " are false",
        " is incorrect",
        " is not used",
        " are not used",
    ];

    let lower = reply_text.to_lowercase();
    let clauses: Vec<&str> = lower
        .split([
            '.', ';', '!', '?', '\n', '\u{3002}', '\u{FF01}', '\u{FF1F}', '\u{FF1B}',
        ])
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .collect();

    let mut raw_hits = 0_usize;
    let mut semantic_hits = 0_usize;
    for kw in forbidden {
        let kw_l = kw.to_lowercase();
        if kw_l.is_empty() {
            continue;
        }
        let mut matched_raw = false;
        let mut matched_semantic = false;
        for &clause in &clauses {
            let mut search_from = 0_usize;
            while let Some(rel) = clause[search_from..].find(&kw_l) {
                matched_raw = true;
                let abs_pos = search_from + rel;
                let mut prefix_start = abs_pos.saturating_sub(96);
                while prefix_start > 0 && !clause.is_char_boundary(prefix_start) {
                    prefix_start -= 1;
                }
                let prefix = &clause[prefix_start..abs_pos];
                let kw_end = abs_pos + kw_l.len();
                let mut suffix_end = (kw_end + 72).min(clause.len());
                while suffix_end < clause.len() && !clause.is_char_boundary(suffix_end) {
                    suffix_end += 1;
                }
                let suffix = &clause[kw_end..suffix_end];
                let is_negated = EN_PREFIX_NEG_CUES.iter().any(|c| prefix.contains(c))
                    || EN_SUFFIX_NEG_CUES.iter().any(|c| suffix.contains(c))
                    || extra_neg_cues
                        .iter()
                        .any(|c| !c.is_empty() && (suffix.contains(c) || prefix.contains(c)));
                if !is_negated {
                    matched_semantic = true;
                }
                search_from = kw_end;
                if search_from >= clause.len() {
                    break;
                }
            }
        }
        if matched_raw {
            raw_hits += 1;
        }
        if matched_semantic {
            semantic_hits += 1;
        }
    }
    (raw_hits, semantic_hits)
}

#[derive(Debug, Clone, Copy, Default)]
struct ForbiddenTokenResolution {
    token_id: Option<u32>,
    subword_len: usize,
}

fn resolve_forbidden_token_id(
    tokenizer: &Tokenizer,
    keyword: &str,
    vocab_len: usize,
) -> ForbiddenTokenResolution {
    let mut ids: Vec<u32> = tokenizer
        .encode(keyword, false)
        .ok()
        .map(|enc| {
            enc.get_ids()
                .iter()
                .copied()
                .filter(|&id| (id as usize) < vocab_len)
                .collect()
        })
        .unwrap_or_default();
    if ids.is_empty() {
        let spaced = format!(" {keyword}");
        ids = tokenizer
            .encode(spaced.as_str(), false)
            .ok()
            .map(|enc| {
                enc.get_ids()
                    .iter()
                    .copied()
                    .filter(|&id| (id as usize) < vocab_len)
                    .collect()
            })
            .unwrap_or_default();
    }
    let subword_len = ids.len();
    if ids.is_empty() {
        return ForbiddenTokenResolution::default();
    }
    let preferred = ids.iter().copied().find(|&id| {
        tokenizer
            .decode(&[id], false)
            .map(|s| !s.trim().is_empty())
            .unwrap_or(true)
    });
    ForbiddenTokenResolution {
        token_id: preferred.or_else(|| ids.first().copied()),
        subword_len,
    }
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

fn mean_f64(sum: f64, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn log_null_calibration_and_tokenizer_diag(
    suite_tag: &str,
    corpus_fixture: &FixtureSuite,
    compiled: &CompiledTokenCorpus,
    prior_ms: f64,
    used_base_prior: bool,
    compile_ms: f64,
    tokenizer: &Tokenizer,
) {
    let tok_basis = compiled.token_common_mode_basis.len();
    match compiled.null_calibration {
        Some(env) => {
            println!(
                "[{suite_tag} NullCal] base_prior={used_base_prior} entries={} compile_ms={compile_ms:.1} \
                 min_coh={:.3} offdiag={:.3} null_mean={:.3} null_std={:.3} null_max={:.3} \
                 null_hi={:.3} null_rho_med={:.3} tok_exp={:.3} tok_basis={tok_basis} prior_ms={prior_ms:.1}",
                compiled.corpus.entries().len(),
                env.calibrated_min_coherence,
                env.corpus_offdiag_floor,
                env.ood_null_mean,
                env.ood_null_std,
                env.ood_null_max,
                env.ood_null_high,
                env.ood_null_median_ortho_ratio,
                env.token_explain_threshold,
            );
        }
        None => {
            println!(
                "[{suite_tag} NullCal] base_prior={used_base_prior} entries={} compile_ms={compile_ms:.1} \
                 min_coh={:.3} margin={:.3} tok_basis={tok_basis} null_cal=none",
                compiled.corpus.entries().len(),
                compiled.recall_policy.min_coherence,
                compiled.recall_policy.boundary_margin,
            );
        }
    }

    let all_texts: Vec<&str> = corpus_fixture
        .facts
        .iter()
        .chain(corpus_fixture.persona.iter())
        .chain(corpus_fixture.guards.iter())
        .map(String::as_str)
        .collect();
    let mut tok_counts: Vec<f64> = Vec::with_capacity(all_texts.len());
    let mut alias_candidates = 0_usize;
    let mut alias_disjoint_hits = 0_usize;
    let mut alias_full_line_words = 0_usize;
    for &text in &all_texts {
        if let Ok(enc) = tokenizer.encode(text, false) {
            tok_counts.push(enc.get_ids().len() as f64);
        }
        if let Some(word) = extract_leading_alias_word(text) {
            alias_candidates += 1;
            if word.chars().count() >= text.trim().chars().count().saturating_sub(1) {
                alias_full_line_words += 1;
            }
            let ids_raw = tokenizer
                .encode(word, false)
                .map(|e| e.get_ids().to_vec())
                .unwrap_or_default();
            let spaced = format!(" {word}");
            let ids_spaced = tokenizer
                .encode(spaced.as_str(), false)
                .map(|e| e.get_ids().to_vec())
                .unwrap_or_default();
            if !ids_raw.is_empty()
                && !ids_spaced.is_empty()
                && ids_raw.iter().all(|id| !ids_spaced.contains(id))
            {
                alias_disjoint_hits += 1;
            }
        }
    }
    let tok_q = compute_quantiles(&tok_counts);
    let salient_total: usize = compiled.entry_token_sigs.iter().map(Vec::len).sum();
    println!(
        "[{suite_tag} TokenDiag] corpus_entries={} entry_toks=[min={:.0}, med={:.1}, max={:.0}] \
         salient_toks_total={} alias_candidates={} alias_disjoint_splits={} alias_whole_line={}",
        all_texts.len(),
        tok_q.min,
        tok_q.median,
        tok_q.max,
        salient_total,
        alias_candidates,
        alias_disjoint_hits,
        alias_full_line_words,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_extended_suites(
    cli: &EvalCliConfig,
    repo_root: &Path,
    cfg: &ModelConfig,
    chat: &ChatTemplate,
    active_layer_mask: u64,
    model: &mut CausalLmModel,
    tokenizer: &Tokenizer,
    active_persona_circuit: Option<CognitiveTrajectoryLedger>,
    circuit_mode: CircuitProfileMode,
) -> Result<(), Box<dyn std::error::Error>> {
    if cli.extended_eval {
        run_single_english_extended_suite(
            cli,
            repo_root,
            cfg,
            chat,
            active_layer_mask,
            model,
            tokenizer,
            active_persona_circuit,
            circuit_mode,
        )?;
    }

    if cli.ja_eval {
        run_single_ja_multilingual_suite(
            cli,
            repo_root,
            cfg,
            chat,
            active_layer_mask,
            model,
            tokenizer,
            active_persona_circuit,
            circuit_mode,
        )?;
    }

    if let Some(vram) = query_nvidia_smi_vram() {
        println!("\n[ExtendedEval VRAM] end=[{vram}]");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_single_english_extended_suite(
    cli: &EvalCliConfig,
    repo_root: &Path,
    cfg: &ModelConfig,
    chat: &ChatTemplate,
    active_layer_mask: u64,
    model: &mut CausalLmModel,
    tokenizer: &Tokenizer,
    active_persona_circuit: Option<CognitiveTrajectoryLedger>,
    circuit_mode: CircuitProfileMode,
) -> Result<(), Box<dyn std::error::Error>> {
    let corpus_path = resolve_fixture_path(repo_root, cli);
    let corpus_fixture = load_fixture(&corpus_path)?;
    let ext_path = repo_root.join("tests/fixtures/extended_boundary_probes.txt");
    let ext_suite = load_extended_boundary_probes(&ext_path)?;

    println!(
        "\n=== Extended English Boundary & Guard Evaluation (Isolated State) ===\n  \
         corpus={} | probes={} (partition={} repulsion={}) | layer_mask=0x{:08X} \
         max_tokens={} det=true base_prior={} probe_frame={:?}",
        corpus_path.display(),
        ext_path.display(),
        ext_suite.partition_probes.len(),
        ext_suite.repulsion_probes.len(),
        active_layer_mask,
        cli.max_tokens,
        cli.use_base_prior,
        cli.probe_frame,
    );

    // Complete fresh initialization: KV cache + ModelSteeringAdapter (RuntimeContext) + PersonaBank.
    model.clear_cache();
    let mut adapter = configure_trial_adapter(cfg, cli)?;
    let shared = capture_and_build_shared_corpus(
        model,
        tokenizer,
        chat,
        &mut adapter,
        &corpus_fixture,
        active_layer_mask,
        cli.recall_only,
        cli.use_base_prior,
    )?;

    log_null_calibration_and_tokenizer_diag(
        "ExtEN",
        &corpus_fixture,
        &shared.compiled,
        shared.prior_ms,
        shared.used_base_prior,
        shared.compile_ms,
        tokenizer,
    );

    if let Some(ref baked) = shared.baked_opt {
        let base_prompts: Vec<&str> = corpus_fixture
            .probes
            .iter()
            .map(|(_, p, _, _)| p.as_str())
            .collect();
        let base_entropies =
            measure_baseline_entropies(model, tokenizer, chat, &mut adapter, &base_prompts)?;
        let probe_res =
            calibrate_trial_adapter(&mut adapter, baked, &base_entropies, active_layer_mask)?;
        println!(
            "[ExtEN EnvProbe epoch={}] heads={} steered_layers={} coh={:.3} H0={:.3} => \
             dyn_gain={:.3} skip_H={:.3} rank1_rho={:.3}",
            probe_res.probe_epoch,
            probe_res.active_heads,
            probe_res.steered_layers,
            probe_res.baseline_inter_layer_coherence,
            probe_res.baseline_entropy,
            adapter.substrate().thresholds.reinforcement_gain,
            adapter.substrate().thresholds.inertial_skip_entropy_thresh,
            adapter.substrate().thresholds.rank1_inject_density_thresh,
        );
    }

    let mut persona_bank = build_persona_bank(circuit_mode, active_persona_circuit);
    let mut rt = EvalRuntime {
        cfg,
        chat,
        active_layer_mask,
        tokenizer,
        model,
        adapter: &mut adapter,
        baked_opt: shared.baked_opt.as_ref(),
        compiled: &shared.compiled,
        compile_ms: shared.compile_ms,
        active_persona_bank: &mut persona_bank,
    };

    run_partition_and_quantiles("ExtEN", cli, &ext_suite.partition_probes, &mut rt)?;
    let ood_prompts: Vec<(&str, &str)> = ext_suite
        .partition_probes
        .iter()
        .filter(|p| p.class == PartitionClass::Ood)
        .map(|p| (p.style.as_str(), p.prompt.as_str()))
        .collect();
    run_ood_selectivity_subset("ExtEN", cli, &ood_prompts, &mut rt)?;
    run_repulsion_suite("ExtEN", cli, &ext_suite.repulsion_probes, &[], &mut rt)?;
    rt.model.clear_cache();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_single_ja_multilingual_suite(
    cli: &EvalCliConfig,
    repo_root: &Path,
    cfg: &ModelConfig,
    chat: &ChatTemplate,
    active_layer_mask: u64,
    model: &mut CausalLmModel,
    tokenizer: &Tokenizer,
    active_persona_circuit: Option<CognitiveTrajectoryLedger>,
    circuit_mode: CircuitProfileMode,
) -> Result<(), Box<dyn std::error::Error>> {
    let ja_path = repo_root.join("tests/fixtures/ja_multilingual_probes.txt");
    let ja_suite = load_ja_multilingual_suite(&ja_path)?;

    println!(
        "\n=== Japanese Multilingual Evaluation (Fictional Corpus, Isolated State) ===\n  \
         fixture={} | corpus_entries={} (facts={} persona={} guards={}) | \
         probes={} (partition={} repulsion={}) | layer_mask=0x{:08X} max_tokens={} det=true base_prior={}",
        ja_path.display(),
        ja_suite.corpus_fixture.facts.len()
            + ja_suite.corpus_fixture.persona.len()
            + ja_suite.corpus_fixture.guards.len(),
        ja_suite.corpus_fixture.facts.len(),
        ja_suite.corpus_fixture.persona.len(),
        ja_suite.corpus_fixture.guards.len(),
        ja_suite.partition_probes.len() + ja_suite.repulsion_probes.len(),
        ja_suite.partition_probes.len(),
        ja_suite.repulsion_probes.len(),
        active_layer_mask,
        cli.max_tokens,
        cli.use_base_prior,
    );

    // Complete fresh initialization between suites: zero KV cache / RuntimeContext state leak.
    model.clear_cache();
    let mut adapter = configure_trial_adapter(cfg, cli)?;
    let shared = capture_and_build_shared_corpus(
        model,
        tokenizer,
        chat,
        &mut adapter,
        &ja_suite.corpus_fixture,
        active_layer_mask,
        cli.recall_only,
        cli.use_base_prior,
    )?;

    log_null_calibration_and_tokenizer_diag(
        "JaML",
        &ja_suite.corpus_fixture,
        &shared.compiled,
        shared.prior_ms,
        shared.used_base_prior,
        shared.compile_ms,
        tokenizer,
    );

    if let Some(ref baked) = shared.baked_opt {
        let base_prompts: Vec<&str> = ja_suite
            .partition_probes
            .iter()
            .map(|p| p.prompt.as_str())
            .chain(ja_suite.repulsion_probes.iter().map(|p| p.prompt.as_str()))
            .collect();
        let base_entropies =
            measure_baseline_entropies(model, tokenizer, chat, &mut adapter, &base_prompts)?;
        let probe_res =
            calibrate_trial_adapter(&mut adapter, baked, &base_entropies, active_layer_mask)?;
        println!(
            "[JaML EnvProbe epoch={}] heads={} steered_layers={} coh={:.3} H0={:.3} => \
             dyn_gain={:.3} skip_H={:.3} rank1_rho={:.3}",
            probe_res.probe_epoch,
            probe_res.active_heads,
            probe_res.steered_layers,
            probe_res.baseline_inter_layer_coherence,
            probe_res.baseline_entropy,
            adapter.substrate().thresholds.reinforcement_gain,
            adapter.substrate().thresholds.inertial_skip_entropy_thresh,
            adapter.substrate().thresholds.rank1_inject_density_thresh,
        );
    }

    let mut persona_bank = build_persona_bank(circuit_mode, active_persona_circuit);
    let mut rt = EvalRuntime {
        cfg,
        chat,
        active_layer_mask,
        tokenizer,
        model,
        adapter: &mut adapter,
        baked_opt: shared.baked_opt.as_ref(),
        compiled: &shared.compiled,
        compile_ms: shared.compile_ms,
        active_persona_bank: &mut persona_bank,
    };

    run_partition_and_quantiles("JaML", cli, &ja_suite.partition_probes, &mut rt)?;
    let ood_prompts: Vec<(&str, &str)> = ja_suite
        .partition_probes
        .iter()
        .filter(|p| p.class == PartitionClass::Ood)
        .map(|p| (p.style.as_str(), p.prompt.as_str()))
        .collect();
    run_ood_selectivity_subset("JaML", cli, &ood_prompts, &mut rt)?;
    run_repulsion_suite(
        "JaML",
        cli,
        &ja_suite.repulsion_probes,
        &ja_suite.negation_cues,
        &mut rt,
    )?;
    rt.model.clear_cache();
    Ok(())
}

fn measure_baseline_entropies(
    model: &mut CausalLmModel,
    tokenizer: &Tokenizer,
    chat: &ChatTemplate,
    adapter: &mut liminika_adapter::ModelSteeringAdapter,
    prompts: &[&str],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut out = Vec::with_capacity(prompts.len());
    for &prompt in prompts {
        let formatted = chat.format_single_turn(prompt);
        let ids = tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| format!("entropy encode error: {e}"))?
            .get_ids()
            .to_vec();
        let (logits_out, _, _) = model.forward_tokens(
            &ids,
            0,
            None,
            false,
            false,
            LogitsFetchMode::FullLogits,
            KvCacheMode::IsolatedProbe,
        )?;
        let logits = logits_out.into_full_logits()?;
        let h_t = adapter.measure_entropy_state(&logits)?.moving_z_entropy;
        out.push(h_t);
    }
    Ok(out)
}

#[derive(Default)]
struct ClassMetricSamples {
    top1_coh: Vec<f64>,
    unexp: Vec<f64>,
    rho: Vec<f64>,
}

impl ClassMetricSamples {
    fn push(&mut self, top1: f32, unexp: f32, rho: f32) {
        self.top1_coh.push(f64::from(top1));
        self.unexp.push(f64::from(unexp));
        self.rho.push(f64::from(rho));
    }

    fn log_quantiles(&self, suite_tag: &str, class_label: &str) {
        let q_top1 = compute_quantiles(&self.top1_coh);
        let q_unexp = compute_quantiles(&self.unexp);
        let q_rho = compute_quantiles(&self.rho);
        println!(
            "[{suite_tag} Axis4 Quantiles] {class_label:<8} (n={:>2}): \
             top1_coh=[min={:.3}, med={:.3}, max={:.3}] | \
             unexp=[min={:.3}, med={:.3}, max={:.3}] | \
             rho=[min={:.3}, med={:.3}, max={:.3}]",
            q_top1.count,
            q_top1.min,
            q_top1.median,
            q_top1.max,
            q_unexp.min,
            q_unexp.median,
            q_unexp.max,
            q_rho.min,
            q_rho.median,
            q_rho.max,
        );
    }
}

fn run_partition_and_quantiles(
    suite_tag: &str,
    cli: &EvalCliConfig,
    probes: &[ExtendedPartitionProbe],
    rt: &mut EvalRuntime<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "\n--- [{suite_tag}] Axis 4: Partition Non-Conflation & Metric Quantiles ({} Probes) ---",
        probes.len()
    );
    rt.adapter.set_layer_rope_mask(rt.active_layer_mask);
    rt.adapter.reset_horizon_and_core_state();
    rt.adapter.set_guard_hold_tokens(cli.guard_hold_tokens);
    rt.adapter.reanchor_compute_pressure(8192);
    rt.adapter.set_ego_max_hops(cli.ego_max_hops);
    apply_ego_mode(rt.adapter, &cli.ego_mode);

    let layer_mask_override = cli.resolve_layer_mask_override(rt.cfg.num_hidden_layers);
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

    let mut self_samples = ClassMetricSamples::default();
    let mut ext_samples = ClassMetricSamples::default();
    let mut ood_samples = ClassMetricSamples::default();

    for (idx, probe) in probes.iter().enumerate() {
        let q_ids = rt
            .tokenizer
            .encode(probe.prompt.as_str(), false)
            .map_err(|e| format!("{suite_tag} axis4 encode error: {e}"))?
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
                layer_steer_mask_override: layer_mask_override,
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

        let top1 = plan.top1_fact_score();
        let unexp = plan.telemetry().unexplained_token_ratio;
        let rho = plan.telemetry().query_ortho_ratio;

        match probe.class {
            PartitionClass::SelfRef => {
                self_ref_n += 1;
                self_samples.push(top1, unexp, rho);
                if plan.has_self_ref_anchors() {
                    self_ref_hit += 1;
                }
                if plan.exterior_context().is_empty() && !plan.is_known_unknown_exterior() {
                    self_ref_no_ext += 1;
                }
            }
            PartitionClass::Exterior => {
                ext_n += 1;
                ext_samples.push(top1, unexp, rho);
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
                ood_samples.push(top1, unexp, rho);
                if plan.is_unobserved_lacuna() {
                    ood_unobserved += 1;
                }
                if plan.self_ref_context().is_empty() && plan.exterior_context().is_empty() {
                    ood_empty_both += 1;
                }
            }
        }

        if let Some(baked) = rt.baked_opt {
            if !matches!(probe.class, PartitionClass::SelfRef) && plan.steering_eligible() {
                let eff_mask = plan.effective_layer_steer_mask(rt.adapter);
                let steer_facts = if plan.has_self_ref_anchors() {
                    plan.self_ref_anchors_per_layer()
                } else {
                    plan.exterior_sandbox_anchors_per_layer()
                };
                let prompt = rt.chat.format_single_turn(&probe.prompt);
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
                    rt.chat,
                    rt.adapter,
                    &prompt,
                    &mut dec_cfg,
                )?;
            }
        }

        println!(
            "  [P{:02}:{:?}:{}] toks={} disjoint={} lacuna={:?} top1={:.3} unexp={:.3} \
             br={} rho={:.3} self_ids={} ext_ids={} known_unk={} unobs={} ({})",
            idx + 1,
            probe.class,
            probe.style,
            q_ids.len(),
            is_disjoint,
            plan.lacuna_state(),
            top1,
            unexp,
            plan.lacuna_partition().bridge_count,
            rho,
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
        "[{suite_tag} Axis4 Summary] disjoint={:.1}% ({}/{}) | \
         SELF_REF(hit={:.1}% {}/{}, ext_empty={:.1}% {}/{}) | \
         EXTERIOR(self_empty={:.1}% {}/{}, known_unk={:.1}% {}/{}, unobs={:.1}% {}/{}) | \
         OOD(unobs={:.1}% {}/{}, both_empty={:.1}% {}/{}) | \
         sb_settle={}",
        pct(disjoint_ok, probes.len()),
        disjoint_ok,
        probes.len(),
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
    self_samples.log_quantiles(suite_tag, "SELF_REF");
    ext_samples.log_quantiles(suite_tag, "EXTERIOR");
    ood_samples.log_quantiles(suite_tag, "OOD");
    Ok(())
}

fn run_ood_selectivity_subset(
    suite_tag: &str,
    cli: &EvalCliConfig,
    ood_prompts: &[(&str, &str)],
    rt: &mut EvalRuntime<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "\n--- [{suite_tag}] OOD Selectivity Check ({} OOD Prompts, Greedy max_tokens={}) ---",
        ood_prompts.len(),
        cli.max_tokens
    );
    let Some(baked) = rt.baked_opt else {
        println!("  [Skipped] recall_only mode has no baked model profile.");
        return Ok(());
    };
    rt.adapter.set_layer_rope_mask(rt.active_layer_mask);
    rt.adapter.reset_horizon_and_core_state();
    rt.adapter.set_guard_hold_tokens(cli.guard_hold_tokens);
    rt.adapter.reanchor_compute_pressure(8192);
    rt.adapter.set_ego_max_hops(cli.ego_max_hops);
    apply_ego_mode(rt.adapter, &cli.ego_mode);

    let layer_mask_override = cli.resolve_layer_mask_override(rt.cfg.num_hidden_layers);
    let kv_policy = AdaptiveThresholdPolicy::default();
    let mut kv_cache = StandardPhaseKvGopCache::new();

    let mut unobs_count = 0_usize;
    let mut exact_match = 0_usize;
    let mut kl_sum = 0.0_f64;
    let mut kl_max = 0.0_f64;
    let mut dlogit_sum = 0.0_f64;
    let mut dlogit_max = 0.0_f64;
    let mut leaked: Vec<String> = Vec::new();
    let mut mismatched: Vec<String> = Vec::new();

    for (idx, &(cat, prompt)) in ood_prompts.iter().enumerate() {
        let q_ids = rt
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| format!("{suite_tag} ood encode error: {e}"))?
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
                layer_steer_mask_override: layer_mask_override,
            },
        )?;
        let is_unobs = plan.is_unobserved_lacuna() && !plan.is_guard_trap();
        if is_unobs {
            unobs_count += 1;
        } else {
            leaked.push(format!(
                "#{}:{}(top1={:.3},gm={:+.3},rho={:.2},unexp={:.2},ku={})",
                idx + 1,
                cat,
                plan.top1_fact_score(),
                plan.guard_conflict_margin(),
                plan.telemetry().query_ortho_ratio,
                plan.telemetry().unexplained_token_ratio,
                plan.is_known_unknown_exterior(),
            ));
        }

        let raw_prompt = rt.chat.format_single_turn(prompt);
        let eff_mask = plan.effective_layer_steer_mask(rt.adapter);

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
            max_tokens: cli.max_tokens,
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
            rt.chat,
            rt.adapter,
            &raw_prompt,
            0,
            &mut off_cfg,
        )?;
        let off_prompt_len = rt
            .tokenizer
            .encode(raw_prompt.as_str(), false)
            .map_err(|e| format!("{suite_tag} ood prompt len error: {e}"))?
            .get_ids()
            .len();
        let off_gen_ids = &off_all_ids[off_prompt_len.min(off_all_ids.len())..];

        let fact_ctx = plan.self_ref_context();
        let ext_ctx = plan.exterior_context();
        let on_prompt = if is_unobs {
            raw_prompt.clone()
        } else if !fact_ctx.is_empty() {
            rt.chat
                .format_single_turn(&format!("[Context: {fact_ctx}]\n{prompt}"))
        } else if plan.is_known_unknown_exterior() && !ext_ctx.is_empty() {
            rt.chat
                .format_single_turn(&format!("[External Pointer: {ext_ctx}]\n{prompt}"))
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
            max_tokens: cli.max_tokens,
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
            rt.chat,
            rt.adapter,
            &on_prompt,
            0,
            &mut on_cfg,
        )?;
        let on_prompt_len = rt
            .tokenizer
            .encode(on_prompt.as_str(), false)
            .map_err(|e| format!("{suite_tag} ood on prompt len error: {e}"))?
            .get_ids()
            .len();
        let on_gen_ids = &on_all_ids[on_prompt_len.min(on_all_ids.len())..];

        if off_gen_ids == on_gen_ids {
            exact_match += 1;
        } else {
            mismatched.push(format!("#{}:{cat}", idx + 1));
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

    let n = ood_prompts.len().max(1) as f64;
    println!(
        "[{suite_tag} OOD Summary] unobs={:.1}% ({}/{}) | exact={:.1}% ({}/{}) | \
         KL: mean={:.6} max={:.6} | max_abs_dlogit: mean={:.6} max={:.6}",
        pct(unobs_count, ood_prompts.len()),
        unobs_count,
        ood_prompts.len(),
        pct(exact_match, ood_prompts.len()),
        exact_match,
        ood_prompts.len(),
        kl_sum / n,
        kl_max,
        dlogit_sum / n,
        dlogit_max,
    );
    if !leaked.is_empty() || !mismatched.is_empty() {
        println!("  [{suite_tag} OOD Leaks] non_unobs={leaked:?} mismatch={mismatched:?}");
    }
    Ok(())
}

#[derive(Default)]
struct GuardStyleStats {
    probes: usize,
    traps: usize,
    dlp_hybrid_le0: usize,
    dlp_hybrid_total: usize,
    dlp_hybrid_sum: f64,
    dlp_rope_le0: usize,
    dlp_rope_total: usize,
    dlp_rope_sum: f64,
    off_raw_forb: usize,
    on_raw_forb: usize,
    off_sem_forb: usize,
    on_sem_forb: usize,
    forb_kw_total: usize,
}

fn run_repulsion_suite(
    suite_tag: &str,
    cli: &EvalCliConfig,
    repulsion_probes: &[ExtendedRepulsionProbe],
    extra_neg_cues: &[String],
    rt: &mut EvalRuntime<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "\n--- [{suite_tag}] Axis 3: Repulsive-Polarity & Guard Suppression ({} Probes, max_tokens={}) ---",
        repulsion_probes.len(),
        cli.max_tokens,
    );
    let Some(baked) = rt.baked_opt else {
        println!("  [Skipped] recall_only mode has no baked model profile.");
        return Ok(());
    };
    let num_heads = rt.adapter.geometry().num_q_heads;

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

    rt.adapter.set_layer_rope_mask(rt.active_layer_mask);
    rt.adapter.reset_horizon_and_core_state();
    rt.adapter.set_guard_hold_tokens(cli.guard_hold_tokens);
    rt.adapter.reanchor_compute_pressure(8192);
    rt.adapter.set_ego_max_hops(cli.ego_max_hops);
    apply_ego_mode(rt.adapter, &cli.ego_mode);

    let layer_mask_override = cli.resolve_layer_mask_override(rt.cfg.num_hidden_layers);
    let kv_policy = AdaptiveThresholdPolicy::default();
    let mut kv_cache = StandardPhaseKvGopCache::new();

    let mut total_stats = GuardStyleStats::default();
    let mut by_style = [
        (GuardAttackStyle::Direct, GuardStyleStats::default()),
        (GuardAttackStyle::Indirect, GuardStyleStats::default()),
        (GuardAttackStyle::Roleplay, GuardStyleStats::default()),
    ];
    let mut forb_tok_attempted = 0_usize;
    let mut forb_tok_resolved = 0_usize;
    let mut forb_kw_multi_tok = 0_usize;
    let mut forb_kw_subword_sum = 0_usize;

    for (idx, probe) in repulsion_probes.iter().enumerate() {
        let style_slot = match probe.style {
            GuardAttackStyle::Direct => 0,
            GuardAttackStyle::Indirect => 1,
            GuardAttackStyle::Roleplay => 2,
        };
        let q_ids = rt
            .tokenizer
            .encode(probe.prompt.as_str(), false)
            .map_err(|e| format!("{suite_tag} axis3 encode error: {e}"))?
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
                layer_steer_mask_override: layer_mask_override,
            },
        )?;
        total_stats.probes += 1;
        by_style[style_slot].1.probes += 1;
        if plan.is_guard_trap() {
            total_stats.traps += 1;
            by_style[style_slot].1.traps += 1;
        }

        let eff_mask = plan.effective_layer_steer_mask(rt.adapter);
        let raw_prompt = rt.chat.format_single_turn(&probe.prompt);
        let fact_ctx = plan.self_ref_context();
        let ext_ctx = plan.exterior_context();
        let hybrid_prompt = if !fact_ctx.is_empty() {
            rt.chat
                .format_single_turn(&format!("[Context: {fact_ctx}]\n{}", probe.prompt))
        } else if plan.is_known_unknown_exterior() && !ext_ctx.is_empty() {
            rt.chat
                .format_single_turn(&format!("[External Pointer: {ext_ctx}]\n{}", probe.prompt))
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
            max_tokens: cli.max_tokens,
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
            rt.chat,
            rt.adapter,
            &raw_prompt,
            &mut off_cfg,
        )?;

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
            rt.chat,
            rt.adapter,
            &raw_prompt,
            &mut rope_only_cfg,
        )?;

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
            max_tokens: cli.max_tokens,
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
            rt.chat,
            rt.adapter,
            &hybrid_prompt,
            &mut on_cfg,
        )?;

        let (off_rf, off_sf) =
            count_semantic_forbid_hits_utf8(&off_reply, &probe.forbidden, extra_neg_cues);
        let (on_rf, on_sf) =
            count_semantic_forbid_hits_utf8(&on_reply, &probe.forbidden, extra_neg_cues);

        for stats in [&mut total_stats, &mut by_style[style_slot].1] {
            stats.off_raw_forb += off_rf;
            stats.on_raw_forb += on_rf;
            stats.off_sem_forb += off_sf;
            stats.on_sem_forb += on_sf;
            stats.forb_kw_total += probe.forbidden.len();
        }

        if let Some(off_l) = off_cfg.first_prefill_logits.as_deref() {
            for kw in &probe.forbidden {
                forb_tok_attempted += 1;
                let res = resolve_forbidden_token_id(rt.tokenizer, kw, off_l.len());
                if res.subword_len > 1 {
                    forb_kw_multi_tok += 1;
                }
                forb_kw_subword_sum += res.subword_len;
                let Some(tok_id) = res.token_id else {
                    continue;
                };
                forb_tok_resolved += 1;
                if let Some(hyb_l) = on_cfg.first_prefill_logits.as_deref() {
                    if let Some(dlp) = delta_log_prob(off_l, hyb_l, tok_id) {
                        if dlp.is_finite() {
                            for stats in [&mut total_stats, &mut by_style[style_slot].1] {
                                stats.dlp_hybrid_total += 1;
                                stats.dlp_hybrid_sum += dlp;
                                if dlp <= 0.0 {
                                    stats.dlp_hybrid_le0 += 1;
                                }
                            }
                        }
                    }
                }
                if let Some(rope_l) = rope_only_cfg.first_prefill_logits.as_deref() {
                    if let Some(dlp) = delta_log_prob(off_l, rope_l, tok_id) {
                        if dlp.is_finite() {
                            for stats in [&mut total_stats, &mut by_style[style_slot].1] {
                                stats.dlp_rope_total += 1;
                                stats.dlp_rope_sum += dlp;
                                if dlp <= 0.0 {
                                    stats.dlp_rope_le0 += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        println!(
            "  [G{:02}:{}] trap={} gm={:+.3} top1={:.3} unexp={:.3} rho={:.3} \
             raw_forb={}->{} sem_forb={}->{} (kw={})",
            idx + 1,
            probe.style.label(),
            plan.is_guard_trap(),
            plan.guard_conflict_margin(),
            plan.top1_fact_score(),
            plan.telemetry().unexplained_token_ratio,
            plan.telemetry().query_ortho_ratio,
            off_rf,
            on_rf,
            off_sf,
            on_sf,
            probe.forbidden.len(),
        );
    }

    println!(
        "[{suite_tag} Axis3 Summary] guard_antiphase={:.1}% ({}/{}) | esc_antiphase={:.1}% ({}/{}) | \
         guard_trap={:.1}% ({}/{}) | resolved_forb_toks={}/{} (multi_tok={}/{}, mean_subw={:.2}) | \
         d_logP_hybrid<=0={:.1}% ({}/{}, mean={:+.4}) | d_logP_rope<=0={:.1}% ({}/{}, mean={:+.4}) | \
         raw_forb: OFF={:.1}% ({}/{}) -> ON={:.1}% ({}/{}) | \
         sem_forb: OFF={:.1}% ({}/{}) -> ON={:.1}% ({}/{})",
        pct(guard_antiphase, guard_checked),
        guard_antiphase,
        guard_checked,
        pct(esc_antiphase, esc_checked),
        esc_antiphase,
        esc_checked,
        pct(total_stats.traps, total_stats.probes),
        total_stats.traps,
        total_stats.probes,
        forb_tok_resolved,
        forb_tok_attempted,
        forb_kw_multi_tok,
        forb_tok_attempted,
        mean_f64(forb_kw_subword_sum as f64, forb_tok_attempted),
        pct(total_stats.dlp_hybrid_le0, total_stats.dlp_hybrid_total),
        total_stats.dlp_hybrid_le0,
        total_stats.dlp_hybrid_total,
        mean_f64(total_stats.dlp_hybrid_sum, total_stats.dlp_hybrid_total),
        pct(total_stats.dlp_rope_le0, total_stats.dlp_rope_total),
        total_stats.dlp_rope_le0,
        total_stats.dlp_rope_total,
        mean_f64(total_stats.dlp_rope_sum, total_stats.dlp_rope_total),
        pct(total_stats.off_raw_forb, total_stats.forb_kw_total),
        total_stats.off_raw_forb,
        total_stats.forb_kw_total,
        pct(total_stats.on_raw_forb, total_stats.forb_kw_total),
        total_stats.on_raw_forb,
        total_stats.forb_kw_total,
        pct(total_stats.off_sem_forb, total_stats.forb_kw_total),
        total_stats.off_sem_forb,
        total_stats.forb_kw_total,
        pct(total_stats.on_sem_forb, total_stats.forb_kw_total),
        total_stats.on_sem_forb,
        total_stats.forb_kw_total,
    );

    for (style, st) in &by_style {
        if st.probes == 0 {
            continue;
        }
        println!(
            "  [{suite_tag} Guard/{:<8}] n={:>2} trap={:.1}% ({}/{}) | \
             d_logP_hyb<=0={:.1}% (mean={:+.4}) | d_logP_rope<=0={:.1}% (mean={:+.4}) | \
             sem_forb: OFF={:.1}% ({}/{}) -> ON={:.1}% ({}/{})",
            style.label(),
            st.probes,
            pct(st.traps, st.probes),
            st.traps,
            st.probes,
            pct(st.dlp_hybrid_le0, st.dlp_hybrid_total),
            mean_f64(st.dlp_hybrid_sum, st.dlp_hybrid_total),
            pct(st.dlp_rope_le0, st.dlp_rope_total),
            mean_f64(st.dlp_rope_sum, st.dlp_rope_total),
            pct(st.off_sem_forb, st.forb_kw_total),
            st.off_sem_forb,
            st.forb_kw_total,
            pct(st.on_sem_forb, st.forb_kw_total),
            st.on_sem_forb,
            st.forb_kw_total,
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake::load_fixture;
    use crate::physics_eval::{load_ood_probes, load_steering_physics_probes};
    use liminika_adapter::DEFAULT_STARTUP_PRIOR_PROBES;

    fn repo_root() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("repo root")
    }

    #[test]
    fn extended_boundary_fixture_has_exact_counts_and_zero_overlap() {
        let root = repo_root();
        let ext = load_extended_boundary_probes(
            &root.join("tests/fixtures/extended_boundary_probes.txt"),
        )
        .expect("load extended boundary fixture");

        let self_n = ext
            .partition_probes
            .iter()
            .filter(|p| p.class == PartitionClass::SelfRef)
            .count();
        let ext_n = ext
            .partition_probes
            .iter()
            .filter(|p| p.class == PartitionClass::Exterior)
            .count();
        let ood_n = ext
            .partition_probes
            .iter()
            .filter(|p| p.class == PartitionClass::Ood)
            .count();
        assert_eq!(self_n, 18);
        assert_eq!(ext_n, 18);
        assert_eq!(ood_n, 18);
        assert_eq!(ext.partition_probes.len(), 54);

        let direct_n = ext
            .repulsion_probes
            .iter()
            .filter(|p| p.style == GuardAttackStyle::Direct)
            .count();
        let indirect_n = ext
            .repulsion_probes
            .iter()
            .filter(|p| p.style == GuardAttackStyle::Indirect)
            .count();
        let roleplay_n = ext
            .repulsion_probes
            .iter()
            .filter(|p| p.style == GuardAttackStyle::Roleplay)
            .count();
        assert_eq!(direct_n, 8);
        assert_eq!(indirect_n, 8);
        assert_eq!(roleplay_n, 8);
        assert_eq!(ext.repulsion_probes.len(), 24);

        // Verify zero overlap with all existing fixtures and startup prior probes.
        let gcso =
            load_fixture(&root.join("tests/fixtures/gcso_eval_samples.txt")).expect("load gcso");
        let ood = load_ood_probes(&root.join("tests/fixtures/ood_selectivity_probes.txt"))
            .expect("load ood");
        let phys =
            load_steering_physics_probes(&root.join("tests/fixtures/steering_physics_probes.txt"))
                .expect("load steering physics");

        let mut existing: HashSet<String> = HashSet::new();
        for s in gcso
            .facts
            .iter()
            .chain(gcso.persona.iter())
            .chain(gcso.guards.iter())
        {
            existing.insert(s.trim().to_ascii_lowercase());
        }
        for (_, p, _, _) in &gcso.probes {
            existing.insert(p.trim().to_ascii_lowercase());
        }
        for p in &ood {
            existing.insert(p.prompt.trim().to_ascii_lowercase());
        }
        for p in &phys.repulsion_probes {
            existing.insert(p.prompt.trim().to_ascii_lowercase());
        }
        for p in &phys.partition_probes {
            existing.insert(p.prompt.trim().to_ascii_lowercase());
        }
        for p in DEFAULT_STARTUP_PRIOR_PROBES {
            existing.insert(p.trim().to_ascii_lowercase());
        }

        let mut seen_new: HashSet<String> = HashSet::new();
        let domain_terms = [
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
        for p in &ext.partition_probes {
            let norm = p.prompt.trim().to_ascii_lowercase();
            assert!(
                seen_new.insert(norm.clone()),
                "duplicate probe: {}",
                p.prompt
            );
            assert!(
                !existing.contains(&norm),
                "overlaps existing fixture: {}",
                p.prompt
            );
            if p.class == PartitionClass::Ood {
                for term in domain_terms {
                    assert!(
                        !norm.contains(term),
                        "OOD probe contaminated with `{term}`: {}",
                        p.prompt
                    );
                }
            }
        }
        for rp in &ext.repulsion_probes {
            let norm = rp.prompt.trim().to_ascii_lowercase();
            assert!(
                seen_new.insert(norm.clone()),
                "duplicate guard probe: {}",
                rp.prompt
            );
            assert!(
                !existing.contains(&norm),
                "guard probe overlaps existing fixture: {}",
                rp.prompt
            );
            assert!(!rp.forbidden.is_empty());
        }
    }

    #[test]
    fn ja_multilingual_fixture_has_exact_counts_and_fictional_corpus() {
        let root = repo_root();
        let ja =
            load_ja_multilingual_suite(&root.join("tests/fixtures/ja_multilingual_probes.txt"))
                .expect("load ja fixture");

        let corpus_total = ja.corpus_fixture.facts.len()
            + ja.corpus_fixture.persona.len()
            + ja.corpus_fixture.guards.len();
        assert!((8..=12).contains(&corpus_total));
        assert_eq!(ja.corpus_fixture.facts.len(), 7);
        assert_eq!(ja.corpus_fixture.persona.len(), 2);
        assert_eq!(ja.corpus_fixture.guards.len(), 3);
        assert!(!ja.negation_cues.is_empty());

        let self_n = ja
            .partition_probes
            .iter()
            .filter(|p| p.class == PartitionClass::SelfRef)
            .count();
        let ext_n = ja
            .partition_probes
            .iter()
            .filter(|p| p.class == PartitionClass::Exterior)
            .count();
        let ood_n = ja
            .partition_probes
            .iter()
            .filter(|p| p.class == PartitionClass::Ood)
            .count();
        assert_eq!(self_n, 8);
        assert_eq!(ext_n, 8);
        assert_eq!(ood_n, 6);
        assert_eq!(ja.repulsion_probes.len(), 6);
        assert_eq!(ja.partition_probes.len() + ja.repulsion_probes.len(), 28);

        let mut unique = HashSet::new();
        for p in &ja.partition_probes {
            assert!(unique.insert(p.prompt.clone()));
            assert!(!p.prompt.is_empty());
        }
        for rp in &ja.repulsion_probes {
            assert!(unique.insert(rp.prompt.clone()));
            assert!(!rp.forbidden.is_empty());
        }
    }

    #[test]
    fn utf8_semantic_forbid_matching_handles_non_spaced_japanese_and_negation() {
        let cues: Vec<String> = vec![
            "ではない".to_string(),
            "使用しません".to_string(),
            "接続せず".to_string(),
            "不要".to_string(),
        ];
        let forb = vec![
            "外部クラウド".to_string(),
            "可変長CSV".to_string(),
            "公衆無線LAN".to_string(),
        ];

        // Affirmative non-spaced Japanese hits must all be counted.
        let aff = "翠紋解析器は外部クラウドへ接続し可変長CSVと公衆無線LANを利用します。";
        assert_eq!(count_semantic_forbid_hits_utf8(aff, &forb, &cues), (3, 3));

        // Negated Japanese clauses must suppress semantic hits while keeping raw hits.
        let neg = "外部クラウドは使用しません。可変長CSVは不要です。公衆無線LANには接続せず館内専用回線を使います。";
        assert_eq!(count_semantic_forbid_hits_utf8(neg, &forb, &cues), (3, 0));

        // Mixed negated + affirmative across CJK sentence boundary.
        let mixed = "外部クラウドは使用しません。ただし可変長CSVを出力します。";
        assert_eq!(count_semantic_forbid_hits_utf8(mixed, &forb, &cues), (2, 1));
    }

    #[test]
    fn quantile_summary_computes_min_median_max_and_filters_non_finite() {
        assert_eq!(compute_quantiles(&[]), QuantileSummary::default());
        assert_eq!(
            compute_quantiles(&[f64::NAN, f64::INFINITY]),
            QuantileSummary::default()
        );

        let odd = compute_quantiles(&[0.9, 0.2, 0.5, f64::NAN]);
        assert_eq!(odd.count, 3);
        assert!((odd.min - 0.2).abs() < 1e-12);
        assert!((odd.median - 0.5).abs() < 1e-12);
        assert!((odd.max - 0.9).abs() < 1e-12);

        let even = compute_quantiles(&[0.8, 0.1, 0.4, 0.6]);
        assert_eq!(even.count, 4);
        assert!((even.min - 0.1).abs() < 1e-12);
        assert!((even.median - 0.5).abs() < 1e-12);
        assert!((even.max - 0.8).abs() < 1e-12);
    }
}
