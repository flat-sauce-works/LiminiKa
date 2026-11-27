// name: examples/candle_eval/src/extended_eval.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Extended English boundary/guard evaluation (`--extended-eval`) and Japanese multilingual
//! evaluation (`--ja-eval`) runners with strict cross-suite state isolation.
//!
//! Both suites reuse the unmodified core and adapter pipeline under identical greedy
//! decode settings (`det=true`, `max_tokens=64` by default, CLI-overridable) while reporting
//! `NullCalibrationEnvelope`, `EnvProbe`, per-class `(top1_coh, unexp, rho)` quantiles, OOD
//! non-interference, Guard anti-phase geometry, forbidden-token `Delta log P`, and UTF-8-safe
//! semantic forbidden keyword hit rates broken down by attack style.

use std::path::Path;

use liminika_adapter::{CompiledTokenCorpus, extract_leading_alias_word};
use liminika_core::stigmergy::CognitiveTrajectoryLedger;
use tokenizers::Tokenizer;

use crate::bake::{
    FixtureSuite, capture_and_build_shared_corpus, detect_leading_alias_continuation, load_fixture,
    query_nvidia_smi_vram,
};
use crate::cli::{EvalCliConfig, EvalRuntime};
use crate::eval::{CircuitProfileMode, count_semantic_forbid_hits};
use crate::eval_common::{
    compute_quantiles, count_semantic_forbid_hits_in_clauses, measure_baseline_entropies,
};
use crate::eval_suites::{
    ForbiddenTokenMode, OodSelectivitySpec, PartitionEvalSpec, PartitionProbeRef,
    RepulsionEvalSpec, RepulsionProbeRef, run_ood_selectivity_eval, run_partition_eval,
    run_repulsion_eval,
};
use crate::model::{CausalLmModel, ChatTemplate, ModelConfig};
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

    let forbidden_lower: Vec<String> = forbidden.iter().map(|k| k.to_lowercase()).collect();
    count_semantic_forbid_hits_in_clauses(
        &clauses,
        &forbidden_lower,
        EN_PREFIX_NEG_CUES,
        EN_SUFFIX_NEG_CUES,
        extra_neg_cues,
        72,
    )
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
            if matches!(
                detect_leading_alias_continuation(tokenizer, word),
                Ok(Some(_))
            ) {
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

pub(crate) struct ExtendedSuiteRunCtx<'a> {
    pub(crate) cli: &'a EvalCliConfig,
    pub(crate) repo_root: &'a Path,
    pub(crate) cfg: &'a ModelConfig,
    pub(crate) chat: &'a ChatTemplate,
    pub(crate) active_layer_mask: u64,
    pub(crate) model: &'a mut CausalLmModel,
    pub(crate) tokenizer: &'a Tokenizer,
    pub(crate) active_persona_circuit: Option<CognitiveTrajectoryLedger>,
    pub(crate) circuit_mode: CircuitProfileMode,
}

struct IsolatedSuiteSpec<'a> {
    suite_tag: &'a str,
    banner: String,
    corpus_fixture: &'a FixtureSuite,
    baseline_prompts: Vec<&'a str>,
    partition_probes: &'a [ExtendedPartitionProbe],
    repulsion_probes: &'a [ExtendedRepulsionProbe],
    extra_neg_cues: &'a [String],
}

pub(crate) fn run_extended_suites(
    mut ctx: ExtendedSuiteRunCtx<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    if ctx.cli.extended_eval {
        let corpus_path = resolve_fixture_path(ctx.repo_root, ctx.cli);
        let corpus_fixture = load_fixture(&corpus_path)?;
        let ext_path = ctx
            .repo_root
            .join("tests/fixtures/extended_boundary_probes.txt");
        let ext_suite = load_extended_boundary_probes(&ext_path)?;
        let baseline_prompts: Vec<&str> = corpus_fixture
            .probes
            .iter()
            .map(|(_, p, _, _)| p.as_str())
            .collect();
        let banner = format!(
            "\n=== Extended English Boundary & Guard Evaluation (Isolated State) ===\n  \
             corpus={} | probes={} (partition={} repulsion={}) | layer_mask=0x{:08X} \
             max_tokens={} det=true base_prior={} probe_frame={:?}",
            corpus_path.display(),
            ext_path.display(),
            ext_suite.partition_probes.len(),
            ext_suite.repulsion_probes.len(),
            ctx.active_layer_mask,
            ctx.cli.max_tokens,
            ctx.cli.use_base_prior,
            ctx.cli.probe_frame,
        );
        run_isolated_extended_suite(
            &mut ctx,
            IsolatedSuiteSpec {
                suite_tag: "ExtEN",
                banner,
                corpus_fixture: &corpus_fixture,
                baseline_prompts,
                partition_probes: &ext_suite.partition_probes,
                repulsion_probes: &ext_suite.repulsion_probes,
                extra_neg_cues: &[],
            },
        )?;
    }

    if ctx.cli.ja_eval {
        let ja_path = ctx
            .repo_root
            .join("tests/fixtures/ja_multilingual_probes.txt");
        let ja_suite = load_ja_multilingual_suite(&ja_path)?;
        let baseline_prompts: Vec<&str> = ja_suite
            .partition_probes
            .iter()
            .map(|p| p.prompt.as_str())
            .chain(ja_suite.repulsion_probes.iter().map(|p| p.prompt.as_str()))
            .collect();
        let banner = format!(
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
            ctx.active_layer_mask,
            ctx.cli.max_tokens,
            ctx.cli.use_base_prior,
        );
        run_isolated_extended_suite(
            &mut ctx,
            IsolatedSuiteSpec {
                suite_tag: "JaML",
                banner,
                corpus_fixture: &ja_suite.corpus_fixture,
                baseline_prompts,
                partition_probes: &ja_suite.partition_probes,
                repulsion_probes: &ja_suite.repulsion_probes,
                extra_neg_cues: &ja_suite.negation_cues,
            },
        )?;
    }

    if let Some(vram) = query_nvidia_smi_vram() {
        println!("\n[ExtendedEval VRAM] end=[{vram}]");
    }
    Ok(())
}

fn run_isolated_extended_suite(
    ctx: &mut ExtendedSuiteRunCtx<'_>,
    spec: IsolatedSuiteSpec<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", spec.banner);

    // Complete fresh initialization: KV cache + ModelSteeringAdapter (RuntimeContext) + PersonaBank.
    ctx.model.clear_cache();
    let mut adapter = configure_trial_adapter(ctx.cfg, ctx.cli)?;
    let shared = capture_and_build_shared_corpus(
        ctx.model,
        ctx.tokenizer,
        ctx.chat,
        &mut adapter,
        spec.corpus_fixture,
        ctx.cli.effective_layer_mask_spec(),
        ctx.cli.recall_only,
        ctx.cli.use_base_prior,
    )?;
    let active_layer_mask = shared.active_layer_mask;

    log_null_calibration_and_tokenizer_diag(
        spec.suite_tag,
        spec.corpus_fixture,
        &shared.compiled,
        shared.prior_ms,
        shared.used_base_prior,
        shared.compile_ms,
        ctx.tokenizer,
    );

    if let Some(ref baked) = shared.baked_opt {
        let base_entropies = measure_baseline_entropies(
            ctx.model,
            ctx.tokenizer,
            ctx.chat,
            &mut adapter,
            &spec.baseline_prompts,
        )?;
        let probe_res =
            calibrate_trial_adapter(&mut adapter, baked, &base_entropies, active_layer_mask)?;
        println!(
            "[{} EnvProbe epoch={}] heads={} steered_layers={} coh={:.3} H0={:.3} => \
             dyn_gain={:.3} skip_H={:.3} rank1_rho={:.3}",
            spec.suite_tag,
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

    let mut persona_bank = build_persona_bank(ctx.circuit_mode, ctx.active_persona_circuit);
    let mut rt = EvalRuntime {
        cfg: ctx.cfg,
        chat: ctx.chat,
        active_layer_mask,
        tokenizer: ctx.tokenizer,
        model: ctx.model,
        adapter: &mut adapter,
        baked_opt: shared.baked_opt.as_ref(),
        compiled: &shared.compiled,
        compile_ms: shared.compile_ms,
        active_persona_bank: &mut persona_bank,
    };

    let part_refs: Vec<PartitionProbeRef<'_>> = spec
        .partition_probes
        .iter()
        .map(|p| PartitionProbeRef {
            class: p.class,
            style: Some(p.style.as_str()),
            prompt: p.prompt.as_str(),
            note: p.note.as_str(),
        })
        .collect();
    let part_err_prefix = format!("{} axis4", spec.suite_tag);
    run_partition_eval(
        &PartitionEvalSpec {
            header: format!(
                "\n--- [{}] Axis 4: Partition Non-Conflation & Metric Quantiles ({} Probes) ---",
                spec.suite_tag,
                spec.partition_probes.len(),
            ),
            error_prefix: &part_err_prefix,
            suite_tag: Some(spec.suite_tag),
            probes: &part_refs,
        },
        ctx.cli,
        &mut rt,
    )?;

    let ood_prompts: Vec<(&str, &str)> = spec
        .partition_probes
        .iter()
        .filter(|p| p.class == PartitionClass::Ood)
        .map(|p| (p.style.as_str(), p.prompt.as_str()))
        .collect();
    let ood_err_prefix = format!("{} ood", spec.suite_tag);
    run_ood_selectivity_eval(
        &OodSelectivitySpec {
            header: format!(
                "\n--- [{}] OOD Selectivity Check ({} OOD Prompts, Greedy max_tokens={}) ---",
                spec.suite_tag,
                ood_prompts.len(),
                ctx.cli.max_tokens,
            ),
            error_prefix: &ood_err_prefix,
            suite_tag: Some(spec.suite_tag),
            max_tokens: ctx.cli.max_tokens,
            probes: &ood_prompts,
        },
        ctx.cli,
        &mut rt,
    )?;

    let rep_refs: Vec<RepulsionProbeRef<'_>> = spec
        .repulsion_probes
        .iter()
        .map(|p| RepulsionProbeRef {
            style: Some(p.style),
            prompt: p.prompt.as_str(),
            forbidden: p.forbidden.as_slice(),
        })
        .collect();
    let rep_err_prefix = format!("{} axis3", spec.suite_tag);
    run_repulsion_eval(
        &RepulsionEvalSpec {
            header: format!(
                "\n--- [{}] Axis 3: Repulsive-Polarity & Guard Suppression ({} Probes, max_tokens={}) ---",
                spec.suite_tag,
                spec.repulsion_probes.len(),
                ctx.cli.max_tokens,
            ),
            error_prefix: &rep_err_prefix,
            suite_tag: Some(spec.suite_tag),
            max_tokens: ctx.cli.max_tokens,
            token_mode: ForbiddenTokenMode::PreferredNonBlank,
            extra_neg_cues: spec.extra_neg_cues,
            probes: &rep_refs,
        },
        ctx.cli,
        &mut rt,
    )?;

    rt.model.clear_cache();
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::bake::load_fixture;
    use crate::eval_common::QuantileSummary;
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
        let archive_fixture = load_fixture(&root.join("tests/fixtures/gcso_eval_samples.txt"))
            .expect("load archive fixture");
        let ood = load_ood_probes(&root.join("tests/fixtures/ood_selectivity_probes.txt"))
            .expect("load ood");
        let phys =
            load_steering_physics_probes(&root.join("tests/fixtures/steering_physics_probes.txt"))
                .expect("load steering physics");

        let mut existing: HashSet<String> = HashSet::new();
        for s in archive_fixture
            .facts
            .iter()
            .chain(archive_fixture.persona.iter())
            .chain(archive_fixture.guards.iter())
        {
            existing.insert(s.trim().to_ascii_lowercase());
        }
        for (_, p, _, _) in &archive_fixture.probes {
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

        // Verify zero overlap between 24 multilingual startup probes and JA fixture lines,
        // and confirm whitespace-free alias extraction produces zero whole-line spans.
        let prior_norm: HashSet<String> = DEFAULT_STARTUP_PRIOR_PROBES
            .iter()
            .map(|p| p.trim().to_lowercase())
            .collect();
        assert_eq!(prior_norm.len(), 24);
        for line in ja
            .corpus_fixture
            .facts
            .iter()
            .chain(ja.corpus_fixture.persona.iter())
            .chain(ja.corpus_fixture.guards.iter())
        {
            assert!(!prior_norm.contains(&line.trim().to_lowercase()));
            let span = extract_leading_alias_word(line).expect("leading JA span");
            assert!(
                span.chars().count() < line.trim().chars().count().saturating_sub(1),
                "unexpected whole-line alias span `{span}` for `{line}`"
            );
        }
        for p in &ja.partition_probes {
            assert!(!prior_norm.contains(&p.prompt.trim().to_lowercase()));
        }
        for rp in &ja.repulsion_probes {
            assert!(!prior_norm.contains(&rp.prompt.trim().to_lowercase()));
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
