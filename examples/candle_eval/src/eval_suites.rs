// name: examples/candle_eval/src/eval_suites.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parameterized boundary, OOD selectivity, and Guard repulsion evaluation runners shared
//! between `--physics-eval` and `--extended-eval` / `--ja-eval`.

use liminika_adapter::{TurnOrchestrator, TurnPrepareInput};
use liminika_core::runtime::{AdaptiveThresholdPolicy, StandardPhaseKvGopCache};

use crate::cli::{EvalCliConfig, EvalRuntime};
use crate::eval::{
    count_semantic_forbid_hits, generate_steered_reply, generate_steered_reply_with_offset,
};
use crate::eval_common::{
    ClassMetricSamples, DecodeCfgOptions, anchor_disjoint_stats, build_steered_body,
    make_decode_cfg, mean_f64, measure_guard_antiphase, pct, reset_clean_probe_adapter,
    resolve_forbidden_token_id,
};
use crate::extended_eval::{GuardAttackStyle, count_semantic_forbid_hits_utf8};
use crate::metrics::{delta_log_prob, kl_divergence_and_max_abs_delta};
use crate::model::{KvCacheMode, LogitsFetchMode};
use crate::physics_eval::PartitionClass;

pub(crate) struct OodSelectivitySpec<'a> {
    pub(crate) header: String,
    pub(crate) error_prefix: &'a str,
    pub(crate) suite_tag: Option<&'a str>,
    pub(crate) max_tokens: usize,
    pub(crate) probes: &'a [(&'a str, &'a str)],
}

pub(crate) fn run_ood_selectivity_eval(
    spec: &OodSelectivitySpec<'_>,
    cli: &EvalCliConfig,
    rt: &mut EvalRuntime<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", spec.header);
    let Some(baked) = rt.baked_opt else {
        println!("  [Skipped] recall_only mode has no baked model profile.");
        return Ok(());
    };
    reset_clean_probe_adapter(rt.adapter, cli, rt.active_layer_mask);

    let layer_mask_override = cli.resolve_layer_mask_override(rt.cfg.num_hidden_layers);
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

    for (idx, &(category, prompt)) in spec.probes.iter().enumerate() {
        let q_ids = rt
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| format!("{} encode error: {e}", spec.error_prefix))?
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
                token_include_mask: None,
                turn_index: (idx + 1) as u32,
                kv_policy: &kv_policy,
                kv_min_resonance_override: cli.kv_min_res,
                persona_bank: rt.active_persona_bank.as_ref(),
                layer_steer_mask_override: layer_mask_override,
            },
        )?;
        let is_unobs = plan.is_unobserved_lacuna() && !plan.is_guard_trap();
        if is_unobs {
            unobserved_count += 1;
        } else {
            leaked_indices.push(format!(
                "#{}:{}(top1={:.3},gm={:+.3},rho={:.2},unexp={:.2},ku={})",
                idx + 1,
                category,
                plan.top1_fact_score(),
                plan.guard_conflict_margin(),
                plan.telemetry().query_ortho_ratio,
                plan.telemetry().unexplained_token_ratio,
                plan.is_known_unknown_exterior(),
            ));
        }

        let raw_prompt = rt.chat.format_single_turn(prompt);
        let eff_mask = plan.effective_layer_steer_mask(rt.adapter);

        let mut off_cfg = make_decode_cfg(
            cli,
            None,
            None,
            DecodeCfgOptions {
                eff_mask,
                use_steer: false,
                max_tokens: spec.max_tokens,
                capture_first_logits: true,
                force_observe_stride_one: true,
            },
        );
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
            .map_err(|e| format!("{} prompt len error: {e}", spec.error_prefix))?
            .get_ids()
            .len();
        let off_gen_ids = &off_all_ids[off_prompt_len.min(off_all_ids.len())..];

        let on_body = build_steered_body(&plan, prompt, true);
        let on_prompt = if on_body == prompt {
            raw_prompt.clone()
        } else {
            rt.chat.format_single_turn(&on_body)
        };
        let mut on_cfg = make_decode_cfg(
            cli,
            Some(baked),
            Some(&plan),
            DecodeCfgOptions {
                eff_mask,
                use_steer: plan.steering_eligible(),
                max_tokens: spec.max_tokens,
                capture_first_logits: true,
                force_observe_stride_one: false,
            },
        );
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
            .map_err(|e| format!("{} on prompt len error: {e}", spec.error_prefix))?
            .get_ids()
            .len();
        let on_gen_ids = &on_all_ids[on_prompt_len.min(on_all_ids.len())..];

        if off_gen_ids == on_gen_ids {
            exact_tok_match += 1;
        } else {
            mismatched_indices.push(format!("#{}:{category}", idx + 1));
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

    let n = spec.probes.len().max(1) as f64;
    if let Some(tag) = spec.suite_tag {
        println!(
            "[{tag} OOD Summary] unobs={:.1}% ({}/{}) | exact={:.1}% ({}/{}) | \
             KL: mean={:.6} max={:.6} | max_abs_dlogit: mean={:.6} max={:.6}",
            pct(unobserved_count, spec.probes.len()),
            unobserved_count,
            spec.probes.len(),
            pct(exact_tok_match, spec.probes.len()),
            exact_tok_match,
            spec.probes.len(),
            kl_sum / n,
            kl_max,
            dlogit_sum / n,
            dlogit_max,
        );
        if !leaked_indices.is_empty() || !mismatched_indices.is_empty() {
            println!(
                "  [{tag} OOD Leaks] non_unobs={leaked_indices:?} mismatch={mismatched_indices:?}"
            );
        }
    } else {
        println!(
            "[Axis2 Summary] unobserved_lacuna={:.1}% ({}/{}) | exact_token_match={:.1}% ({}/{}) | \
             first_logit_KL: mean={:.6} max={:.6} | max_abs_dlogit: mean={:.6} max={:.6}",
            pct(unobserved_count, spec.probes.len()),
            unobserved_count,
            spec.probes.len(),
            pct(exact_tok_match, spec.probes.len()),
            exact_tok_match,
            spec.probes.len(),
            kl_sum / n,
            kl_max,
            dlogit_sum / n,
            dlogit_max,
        );
        if !leaked_indices.is_empty() || !mismatched_indices.is_empty() {
            println!(
                "  [Axis2 Leaks] non_unobserved_ids={leaked_indices:?} \
                 token_mismatch_ids={mismatched_indices:?}"
            );
        }
    }
    Ok(())
}

pub(crate) struct PartitionProbeRef<'a> {
    pub(crate) class: PartitionClass,
    pub(crate) style: Option<&'a str>,
    pub(crate) prompt: &'a str,
    pub(crate) note: &'a str,
}

pub(crate) struct PartitionEvalSpec<'a> {
    pub(crate) header: String,
    pub(crate) error_prefix: &'a str,
    pub(crate) suite_tag: Option<&'a str>,
    pub(crate) probes: &'a [PartitionProbeRef<'a>],
}

pub(crate) fn run_partition_eval(
    spec: &PartitionEvalSpec<'_>,
    cli: &EvalCliConfig,
    rt: &mut EvalRuntime<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", spec.header);
    reset_clean_probe_adapter(rt.adapter, cli, rt.active_layer_mask);

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

    for (idx, probe) in spec.probes.iter().enumerate() {
        let q_ids = rt
            .tokenizer
            .encode(probe.prompt, false)
            .map_err(|e| format!("{} encode error: {e}", spec.error_prefix))?
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
                token_include_mask: None,
                turn_index: (idx + 1) as u32,
                kv_policy: &kv_policy,
                kv_min_resonance_override: cli.kv_min_res,
                persona_bank: rt.active_persona_bank.as_ref(),
                layer_steer_mask_override: layer_mask_override,
            },
        )?;

        let (is_disjoint, self_id_len, ext_id_len) = anchor_disjoint_stats(&plan);
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

        // Exercise a short decode step on Exterior / OOD when steering is eligible to verify
        // zero unverified `sandbox_settlements` misfires.
        if let Some(baked) = rt.baked_opt
            && !matches!(probe.class, PartitionClass::SelfRef)
            && plan.steering_eligible()
        {
            let eff_mask = plan.effective_layer_steer_mask(rt.adapter);
            let prompt = rt.chat.format_single_turn(probe.prompt);
            let mut dec_cfg = make_decode_cfg(
                cli,
                Some(baked),
                Some(&plan),
                DecodeCfgOptions {
                    eff_mask,
                    use_steer: true,
                    max_tokens: 16,
                    capture_first_logits: false,
                    force_observe_stride_one: false,
                },
            );
            let _ = generate_steered_reply(
                rt.model,
                rt.tokenizer,
                rt.chat,
                rt.adapter,
                &prompt,
                &mut dec_cfg,
            )?;
        }

        if let Some(style) = probe.style {
            println!(
                "  [P{:02}:{:?}:{style}] toks={} disjoint={} lacuna={:?} top1={:.3} unexp={:.3} \
                 br={} rho={:.3} self_ids={} ext_ids={} known_unk={} unobs={} ({})",
                idx + 1,
                probe.class,
                q_ids.len(),
                is_disjoint,
                plan.lacuna_state(),
                top1,
                unexp,
                plan.lacuna_partition().bridge_count,
                rho,
                self_id_len,
                ext_id_len,
                plan.is_known_unknown_exterior(),
                plan.is_unobserved_lacuna(),
                probe.note,
            );
        } else {
            println!(
                "  [P{:02}:{:?}] disjoint={} lacuna={:?} top1={:.3} unexp={:.2} br={} rho={:.2} \
                 self_ids={} ext_ids={} known_unk={} unobs={} ({})",
                idx + 1,
                probe.class,
                is_disjoint,
                plan.lacuna_state(),
                top1,
                unexp,
                plan.lacuna_partition().bridge_count,
                rho,
                self_id_len,
                ext_id_len,
                plan.is_known_unknown_exterior(),
                plan.is_unobserved_lacuna(),
                probe.note,
            );
        }
    }

    let settlements_delta = rt
        .adapter
        .sandbox_settlements()
        .saturating_sub(settlements_before);
    if let Some(tag) = spec.suite_tag {
        println!(
            "[{tag} Axis4 Summary] disjoint={:.1}% ({}/{}) | \
             SELF_REF(hit={:.1}% {}/{}, ext_empty={:.1}% {}/{}) | \
             EXTERIOR(self_empty={:.1}% {}/{}, known_unk={:.1}% {}/{}, unobs={:.1}% {}/{}) | \
             OOD(unobs={:.1}% {}/{}, both_empty={:.1}% {}/{}) | \
             sb_settle={}",
            pct(disjoint_ok, spec.probes.len()),
            disjoint_ok,
            spec.probes.len(),
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
        self_samples.log_quantiles(tag, "SELF_REF");
        ext_samples.log_quantiles(tag, "EXTERIOR");
        ood_samples.log_quantiles(tag, "OOD");
    } else {
        println!(
            "[Axis4 Summary] anchor_id_disjoint={:.1}% ({}/{}) | \
             SELF_REF(hit={:.1}% {}/{}, ext_empty={:.1}% {}/{}) | \
             EXTERIOR(self_empty={:.1}% {}/{}, known_unk={:.1}% {}/{}, unobs={:.1}% {}/{}) | \
             OOD(unobs={:.1}% {}/{}, both_empty={:.1}% {}/{}) | \
             unverified_sandbox_settlements={}",
            pct(disjoint_ok, spec.probes.len()),
            disjoint_ok,
            spec.probes.len(),
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
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ForbiddenTokenMode {
    /// Uses the raw first encoded token ID (`--physics-eval` baseline parity).
    FirstSubword,
    /// Uses [`resolve_forbidden_token_id`] preferring non-blank subwords with leading-space fallback.
    PreferredNonBlank,
}

pub(crate) struct RepulsionProbeRef<'a> {
    pub(crate) style: Option<GuardAttackStyle>,
    pub(crate) prompt: &'a str,
    pub(crate) forbidden: &'a [String],
}

pub(crate) struct RepulsionEvalSpec<'a> {
    pub(crate) header: String,
    pub(crate) error_prefix: &'a str,
    pub(crate) suite_tag: Option<&'a str>,
    pub(crate) max_tokens: usize,
    pub(crate) token_mode: ForbiddenTokenMode,
    pub(crate) extra_neg_cues: &'a [String],
    pub(crate) probes: &'a [RepulsionProbeRef<'a>],
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

pub(crate) fn run_repulsion_eval(
    spec: &RepulsionEvalSpec<'_>,
    cli: &EvalCliConfig,
    rt: &mut EvalRuntime<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", spec.header);
    let Some(baked) = rt.baked_opt else {
        println!("  [Skipped] recall_only mode has no baked model profile.");
        return Ok(());
    };
    let num_heads = rt.adapter.geometry().num_q_heads;
    let (guard_antiphase, guard_checked, esc_antiphase, esc_checked) =
        measure_guard_antiphase(baked, rt.compiled, num_heads);

    reset_clean_probe_adapter(rt.adapter, cli, rt.active_layer_mask);
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

    for (idx, probe) in spec.probes.iter().enumerate() {
        let style_slot = probe.style.map(|s| match s {
            GuardAttackStyle::Direct => 0,
            GuardAttackStyle::Indirect => 1,
            GuardAttackStyle::Roleplay => 2,
        });
        let q_ids = rt
            .tokenizer
            .encode(probe.prompt, false)
            .map_err(|e| format!("{} encode error: {e}", spec.error_prefix))?
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
                token_include_mask: None,
                turn_index: (idx + 1) as u32,
                kv_policy: &kv_policy,
                kv_min_resonance_override: cli.kv_min_res,
                persona_bank: rt.active_persona_bank.as_ref(),
                layer_steer_mask_override: layer_mask_override,
            },
        )?;
        total_stats.probes += 1;
        if let Some(slot) = style_slot {
            by_style[slot].1.probes += 1;
        }
        if plan.is_guard_trap() {
            total_stats.traps += 1;
            if let Some(slot) = style_slot {
                by_style[slot].1.traps += 1;
            }
        }

        let eff_mask = plan.effective_layer_steer_mask(rt.adapter);
        let raw_prompt = rt.chat.format_single_turn(probe.prompt);
        let hybrid_body = build_steered_body(&plan, probe.prompt, false);
        let hybrid_prompt = if hybrid_body == probe.prompt {
            raw_prompt.clone()
        } else {
            rt.chat.format_single_turn(&hybrid_body)
        };

        let mut off_cfg = make_decode_cfg(
            cli,
            None,
            None,
            DecodeCfgOptions {
                eff_mask,
                use_steer: false,
                max_tokens: spec.max_tokens,
                capture_first_logits: true,
                force_observe_stride_one: true,
            },
        );
        let (off_reply, _, _, _) = generate_steered_reply(
            rt.model,
            rt.tokenizer,
            rt.chat,
            rt.adapter,
            &raw_prompt,
            &mut off_cfg,
        )?;

        let mut rope_only_cfg = make_decode_cfg(
            cli,
            Some(baked),
            Some(&plan),
            DecodeCfgOptions {
                eff_mask,
                use_steer: true,
                max_tokens: 1,
                capture_first_logits: true,
                force_observe_stride_one: true,
            },
        );
        let _ = generate_steered_reply(
            rt.model,
            rt.tokenizer,
            rt.chat,
            rt.adapter,
            &raw_prompt,
            &mut rope_only_cfg,
        )?;

        let mut on_cfg = make_decode_cfg(
            cli,
            Some(baked),
            Some(&plan),
            DecodeCfgOptions {
                eff_mask,
                use_steer: plan.steering_eligible(),
                max_tokens: spec.max_tokens,
                capture_first_logits: true,
                force_observe_stride_one: false,
            },
        );
        let (on_reply, _, _, _) = generate_steered_reply(
            rt.model,
            rt.tokenizer,
            rt.chat,
            rt.adapter,
            &hybrid_prompt,
            &mut on_cfg,
        )?;

        let (off_rf, off_sf) = if spec.extra_neg_cues.is_empty() && spec.suite_tag.is_none() {
            count_semantic_forbid_hits(&off_reply, probe.forbidden)
        } else {
            count_semantic_forbid_hits_utf8(&off_reply, probe.forbidden, spec.extra_neg_cues)
        };
        let (on_rf, on_sf) = if spec.extra_neg_cues.is_empty() && spec.suite_tag.is_none() {
            count_semantic_forbid_hits(&on_reply, probe.forbidden)
        } else {
            count_semantic_forbid_hits_utf8(&on_reply, probe.forbidden, spec.extra_neg_cues)
        };

        let update_forb = |stats: &mut GuardStyleStats| {
            stats.off_raw_forb += off_rf;
            stats.on_raw_forb += on_rf;
            stats.off_sem_forb += off_sf;
            stats.on_sem_forb += on_sf;
            stats.forb_kw_total += probe.forbidden.len();
        };
        update_forb(&mut total_stats);
        if let Some(slot) = style_slot {
            update_forb(&mut by_style[slot].1);
        }

        if let Some(off_l) = off_cfg.first_prefill_logits.as_deref() {
            for kw in probe.forbidden {
                forb_tok_attempted += 1;
                let tok_opt = match spec.token_mode {
                    ForbiddenTokenMode::FirstSubword => rt
                        .tokenizer
                        .encode(kw.as_str(), false)
                        .ok()
                        .and_then(|e| e.get_ids().first().copied()),
                    ForbiddenTokenMode::PreferredNonBlank => {
                        let res = resolve_forbidden_token_id(rt.tokenizer, kw, off_l.len());
                        if res.subword_len > 1 {
                            forb_kw_multi_tok += 1;
                        }
                        forb_kw_subword_sum += res.subword_len;
                        res.token_id
                    }
                };
                let Some(tok_id) = tok_opt else { continue };
                forb_tok_resolved += 1;
                if let Some(hyb_l) = on_cfg.first_prefill_logits.as_deref()
                    && let Some(dlp) = delta_log_prob(off_l, hyb_l, tok_id)
                    && (matches!(spec.token_mode, ForbiddenTokenMode::FirstSubword)
                        || dlp.is_finite())
                {
                    total_stats.dlp_hybrid_total += 1;
                    total_stats.dlp_hybrid_sum += dlp;
                    if dlp <= 0.0 {
                        total_stats.dlp_hybrid_le0 += 1;
                    }
                    if let Some(slot) = style_slot {
                        by_style[slot].1.dlp_hybrid_total += 1;
                        by_style[slot].1.dlp_hybrid_sum += dlp;
                        if dlp <= 0.0 {
                            by_style[slot].1.dlp_hybrid_le0 += 1;
                        }
                    }
                }
                if let Some(rope_l) = rope_only_cfg.first_prefill_logits.as_deref()
                    && let Some(dlp) = delta_log_prob(off_l, rope_l, tok_id)
                    && (matches!(spec.token_mode, ForbiddenTokenMode::FirstSubword)
                        || dlp.is_finite())
                {
                    total_stats.dlp_rope_total += 1;
                    total_stats.dlp_rope_sum += dlp;
                    if dlp <= 0.0 {
                        total_stats.dlp_rope_le0 += 1;
                    }
                    if let Some(slot) = style_slot {
                        by_style[slot].1.dlp_rope_total += 1;
                        by_style[slot].1.dlp_rope_sum += dlp;
                        if dlp <= 0.0 {
                            by_style[slot].1.dlp_rope_le0 += 1;
                        }
                    }
                }
            }
        }

        if let Some(style) = probe.style {
            println!(
                "  [G{:02}:{}] trap={} gm={:+.3} top1={:.3} unexp={:.3} rho={:.3} \
                 raw_forb={}->{} sem_forb={}->{} (kw={})",
                idx + 1,
                style.label(),
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
    }

    if let Some(tag) = spec.suite_tag {
        println!(
            "[{tag} Axis3 Summary] guard_antiphase={:.1}% ({}/{}) | esc_antiphase={:.1}% ({}/{}) | \
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
                "  [{tag} Guard/{:<8}] n={:>2} trap={:.1}% ({}/{}) | \
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
    } else {
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
            pct(total_stats.traps, total_stats.probes),
            total_stats.traps,
            total_stats.probes,
            pct(total_stats.dlp_hybrid_le0, total_stats.dlp_hybrid_total),
            total_stats.dlp_hybrid_le0,
            total_stats.dlp_hybrid_total,
            mean_f64(total_stats.dlp_hybrid_sum, total_stats.dlp_hybrid_total),
            pct(total_stats.dlp_rope_le0, total_stats.dlp_rope_total),
            total_stats.dlp_rope_le0,
            total_stats.dlp_rope_total,
            mean_f64(total_stats.dlp_rope_sum, total_stats.dlp_rope_total),
            pct(total_stats.off_sem_forb, total_stats.forb_kw_total),
            total_stats.off_sem_forb,
            total_stats.forb_kw_total,
            pct(total_stats.on_sem_forb, total_stats.forb_kw_total),
            total_stats.on_sem_forb,
            total_stats.forb_kw_total,
        );
    }
    Ok(())
}
