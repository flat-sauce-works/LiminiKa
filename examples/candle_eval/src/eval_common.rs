// name: examples/candle_eval/src/eval_common.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared low-level helpers for deterministic evaluation harnesses in `candle_eval`:
//! Q7 inner-product checks, quantile aggregation, forbidden-token resolution, semantic negation
//! clause scanning, Guard anti-phase checks, and `SteerDecodeConfig` construction.

use std::collections::HashSet;

use liminika_adapter::{
    AttractorRole, CompiledTokenCorpus, CorpusRecallEscalationProvider, ModelSteeringAdapter,
    TurnSteeringPlan,
};
use liminika_core::abi::lmnk_q7_t;
use liminika_core::liminal::{EpistemicRegime, MAX_ATTENTION_HEADS};
use liminika_core::runtime::{
    GuardPhaseAnchor, HorizonEscalationProvider, HorizonEscalationRequest, HorizonTriggerInput,
    ManifoldGatewaySignal, ObjectiveFact,
};
use liminika_core::stigmergy::{WorkbenchLanePlacement, WORKBENCH_LANE_FLAG_REPULSION};
use tokenizers::Tokenizer;

use crate::bake::BakedModelProfile;
use crate::cli::EvalCliConfig;
use crate::eval::{apply_ego_mode, SteerDecodeConfig};
use crate::model::{CausalLmModel, ChatTemplate, KvCacheMode, LogitsFetchMode, ResonantAnchors};

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

#[derive(Default)]
pub(crate) struct ClassMetricSamples {
    top1_coh: Vec<f64>,
    unexp: Vec<f64>,
    rho: Vec<f64>,
}

impl ClassMetricSamples {
    pub(crate) fn push(&mut self, top1: f32, unexp: f32, rho: f32) {
        self.top1_coh.push(f64::from(top1));
        self.unexp.push(f64::from(unexp));
        self.rho.push(f64::from(rho));
    }

    pub(crate) fn push_plan(&mut self, plan: &TurnSteeringPlan) {
        self.push(
            plan.top1_fact_score(),
            plan.telemetry().unexplained_token_ratio,
            plan.telemetry().query_ortho_ratio,
        );
    }

    pub(crate) fn log_quantiles(&self, suite_tag: &str, class_label: &str) {
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

    pub(crate) fn format_quantiles(&self, label: &str) -> String {
        let q1 = compute_quantiles(&self.top1_coh);
        let q2 = compute_quantiles(&self.unexp);
        let q3 = compute_quantiles(&self.rho);
        format!(
            "{label:<8}(n={:>2}): top1=[{:.3},{:.3},{:.3}] \
             unexp=[{:.3},{:.3},{:.3}] rho=[{:.3},{:.3},{:.3}]",
            q1.count,
            q1.min,
            q1.median,
            q1.max,
            q2.min,
            q2.median,
            q2.max,
            q3.min,
            q3.median,
            q3.max,
        )
    }
}

pub(crate) fn dot_q7(a: &[lmnk_q7_t], b: &[lmnk_q7_t], n_heads: usize) -> i64 {
    let limit = n_heads.min(a.len()).min(b.len());
    a[..limit]
        .iter()
        .zip(b[..limit].iter())
        .map(|(&x, &y)| i64::from(x) * i64::from(y))
        .sum()
}

pub(crate) fn is_nonzero_q7(v: &[lmnk_q7_t], n_heads: usize) -> bool {
    let limit = n_heads.min(v.len());
    v[..limit].iter().any(|&x| x != 0)
}

pub(crate) fn pct(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64 * 100.0
    }
}

pub(crate) fn mean_f64(sum: f64, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ForbiddenTokenResolution {
    pub(crate) token_id: Option<u32>,
    pub(crate) subword_len: usize,
}

pub(crate) fn resolve_forbidden_token_id(
    tokenizer: &Tokenizer,
    keyword: &str,
    vocab_len: usize,
) -> ForbiddenTokenResolution {
    let encode_valid = |s: &str| -> Vec<u32> {
        tokenizer
            .encode(s, false)
            .ok()
            .map(|enc| {
                enc.get_ids()
                    .iter()
                    .copied()
                    .filter(|&id| (id as usize) < vocab_len)
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut ids = encode_valid(keyword);
    if ids.is_empty() {
        let spaced = format!(" {keyword}");
        ids = encode_valid(spaced.as_str());
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

/// Shared clause-level forbidden-keyword scanner with char-boundary-safe prefix/suffix
/// negation checks. Callers supply language-appropriate normalization and clause splitting.
pub(crate) fn count_semantic_forbid_hits_in_clauses(
    clauses: &[&str],
    forbidden_lower: &[String],
    prefix_cues: &[&str],
    suffix_cues: &[&str],
    extra_cues: &[String],
    suffix_window_bytes: usize,
) -> (usize, usize) {
    let mut raw_hits = 0_usize;
    let mut semantic_hits = 0_usize;
    for kw_l in forbidden_lower {
        if kw_l.is_empty() {
            continue;
        }
        let mut matched_raw = false;
        let mut matched_semantic = false;
        for &clause in clauses {
            let mut search_from = 0_usize;
            while let Some(rel) = clause[search_from..].find(kw_l.as_str()) {
                matched_raw = true;
                let abs_pos = search_from + rel;
                let mut prefix_start = abs_pos.saturating_sub(96);
                while prefix_start > 0 && !clause.is_char_boundary(prefix_start) {
                    prefix_start -= 1;
                }
                let prefix = &clause[prefix_start..abs_pos];
                let kw_end = abs_pos + kw_l.len();
                let mut suffix_end = (kw_end + suffix_window_bytes).min(clause.len());
                while suffix_end < clause.len() && !clause.is_char_boundary(suffix_end) {
                    suffix_end += 1;
                }
                let suffix = &clause[kw_end..suffix_end];
                let is_negated = prefix_cues.iter().any(|c| prefix.contains(c))
                    || suffix_cues.iter().any(|c| suffix.contains(c))
                    || extra_cues
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

pub(crate) fn anchor_disjoint_stats(plan: &TurnSteeringPlan) -> (bool, usize, usize) {
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
    (
        self_ids.is_disjoint(&ext_ids),
        self_ids.len(),
        ext_ids.len(),
    )
}

pub(crate) fn select_steer_facts(plan: &TurnSteeringPlan) -> &[Vec<ObjectiveFact<[i8; 64]>>] {
    if plan.has_self_ref_anchors() {
        plan.self_ref_anchors_per_layer()
    } else if plan.is_known_unknown_exterior() {
        plan.exterior_sandbox_anchors_per_layer()
    } else {
        plan.self_ref_anchors_per_layer()
    }
}

pub(crate) fn build_steered_body(
    plan: &TurnSteeringPlan,
    prompt: &str,
    skip_unobserved: bool,
) -> String {
    if skip_unobserved && plan.is_unobserved_lacuna() && !plan.is_guard_trap() {
        return prompt.to_string();
    }
    let fact_ctx = plan.self_ref_context();
    let ext_ctx = plan.exterior_context();
    if !fact_ctx.is_empty() {
        format!("[Context: {fact_ctx}]\n{prompt}")
    } else if plan.is_known_unknown_exterior() && !ext_ctx.is_empty() {
        format!("[External Pointer: {ext_ctx}]\n{prompt}")
    } else {
        prompt.to_string()
    }
}

pub(crate) fn reset_clean_probe_adapter(
    adapter: &mut ModelSteeringAdapter,
    cli: &EvalCliConfig,
    active_layer_mask: u64,
) {
    adapter.set_layer_rope_mask(active_layer_mask);
    adapter.reset_horizon_and_core_state();
    adapter.set_guard_hold_tokens(cli.guard_hold_tokens);
    adapter.reanchor_compute_pressure(8192);
    adapter.set_ego_max_hops(cli.ego_max_hops);
    apply_ego_mode(adapter, &cli.ego_mode);
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DecodeCfgOptions {
    pub(crate) eff_mask: u64,
    pub(crate) use_steer: bool,
    pub(crate) max_tokens: usize,
    pub(crate) capture_first_logits: bool,
    pub(crate) force_observe_stride_one: bool,
}

pub(crate) fn make_decode_cfg<'a>(
    cli: &EvalCliConfig,
    baked: Option<&'a BakedModelProfile>,
    plan: Option<&'a TurnSteeringPlan>,
    opts: DecodeCfgOptions,
) -> SteerDecodeConfig<'a, 'a, 'a> {
    let (q7s, persona_q7s, resonant_ctx, rank1) = match (opts.use_steer, baked, plan) {
        (true, Some(b), Some(p)) => (
            Some(b.fact_q7.as_slice()),
            Some(b.persona_q7.as_slice()),
            Some(ResonantAnchors {
                facts: select_steer_facts(p),
                personas: p.persona_anchors_per_layer(),
                guards: p.guard_anchors_per_layer(),
                negs: b.repulsive_centroid_per_layer.as_slice(),
                poss: b.attractive_centroid_per_layer.as_slice(),
            }),
            Some((b.rank1_u.as_slice(), b.rank1_v.as_slice(), -1.00)),
        ),
        _ => (None, None, None, None),
    };
    SteerDecodeConfig {
        q7s,
        persona_q7s,
        resonant_ctx,
        rope_rad: if opts.use_steer { -2.00 } else { 0.00 },
        rank1,
        use_steer: opts.use_steer,
        horizon_mode: cli.horizon_mode,
        escalation_provider: None,
        pressure_sim: cli.pressure_sim_mode,
        enable_pre_bypass: opts.use_steer && cli.enable_pre_bypass,
        temperature: 0.0,
        rng_seed: cli.rng_seed,
        logits_observe_stride: if opts.use_steer && !opts.force_observe_stride_one {
            cli.logits_observe_stride
        } else {
            1
        },
        layer_steer_mask: Some(opts.eff_mask),
        probe_frame: cli.probe_frame,
        phase_clamp_override: opts.use_steer.then_some(cli.phase_clamp_rad).flatten(),
        random_push_seed: (opts.use_steer && cli.random_push_control).then_some(cli.rng_seed),
        max_tokens: opts.max_tokens,
        prefill_chunk_size: cli.prefill_chunk_size,
        capture_first_logits: opts.capture_first_logits,
        first_prefill_logits: None,
        exit_entropy_h_tilde: 0.0,
        steer_bypass_hits: 0,
        steer_eligible_layers: 0,
        spine_veto_guard_hits: 0,
        spine_veto_repulse_hits: 0,
        spine_veto_pin_hits: 0,
    }
}

/// Verifies anti-phase (`dot_q7 < 0`) geometry for direct `GuardPhaseAnchor` frames and
/// `CorpusRecallEscalationProvider` repulsion escalations across all layers.
pub(crate) fn measure_guard_antiphase(
    baked: &BakedModelProfile,
    compiled: &CompiledTokenCorpus,
    num_heads: usize,
) -> (usize, usize, usize, usize) {
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
    let mut esc_provider = CorpusRecallEscalationProvider::new(&compiled.corpus);
    let best_guard = compiled
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
    (guard_antiphase, guard_checked, esc_antiphase, esc_checked)
}

pub(crate) fn measure_baseline_entropies(
    model: &mut CausalLmModel,
    tokenizer: &Tokenizer,
    chat: &ChatTemplate,
    adapter: &mut ModelSteeringAdapter,
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
