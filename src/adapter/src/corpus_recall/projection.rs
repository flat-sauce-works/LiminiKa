// name: src/adapter/src/corpus_recall/projection.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Recall projection traits, 8-granule `LacunaPartition8` density mapping, and per-layer anchor
//! materialization (including adapter-side Q7 attenuation `+R_soft` for Layer 2 soft-caution Guard).

use super::boundary_2d::{
    DEFAULT_GUARD_HARD_MARGIN, DEFAULT_SOFT_GUARD_Q7_SCALE, GuardLayerClassification,
    GuardLayerMode, classify_guard_tier,
};
use super::{DynamicAttractorRecall, LacunaBoundaryState, TextAttractorEntry};
use crate::{ModelSteeringAdapter, StratumProbeTelemetry};
use liminika_core::abi::lmnk_q7_t;
use liminika_core::liminal::{
    LacunaPartition8, MAX_ATTENTION_HEADS, NUM_STIGMERGIC_GRANULES,
    evaluate_lacuna_partition_branchless,
};
use liminika_core::runtime::{GatewayFrame, GuardPhaseAnchor, ObjectiveFact, SubjectivePersona};
use liminika_core::stigmergy::StigmergicActionDecision;

/// Lightweight zero-copy view over a recalled attractor anchor shared by borrowed
/// ([`DynamicAttractorRecall`]) and owned ([`crate::OwnedAttractorRecall`]) recall paths.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RecalledAnchorRef<'a> {
    pub(crate) anchor_id: u32,
    pub(crate) text: &'a str,
    pub(crate) per_layer_q7: &'a [[lmnk_q7_t; MAX_ATTENTION_HEADS]],
    pub(crate) energy: f32,
    pub(crate) score: f32,
}

/// Crate-internal projection unifying borrowed [`DynamicAttractorRecall`] and owned
/// [`crate::OwnedAttractorRecall`] so core-layer sync, snapshot capture, and anchor
/// materialization share a single implementation without cloning or dual APIs.
pub(crate) trait AttractorRecallProjection {
    fn lacuna_state(&self) -> LacunaBoundaryState;
    fn top1_fact_score(&self) -> f32;
    fn top2_fact_score(&self) -> f32;
    fn top1_persona_score(&self) -> f32;
    fn guard_conflict_margin(&self) -> f32;
    fn action_decision(&self) -> StigmergicActionDecision;
    fn gateway_frame(&self) -> &GatewayFrame;
    fn telemetry(&self) -> &StratumProbeTelemetry;
    fn lacuna_partition(&self) -> LacunaPartition8;
    fn is_known_unknown_exterior(&self) -> bool;
    fn recalled_fact_len(&self) -> usize;
    fn recalled_fact_at(&self, idx: usize) -> Option<RecalledAnchorRef<'_>>;
    fn recalled_persona_len(&self) -> usize;
    fn recalled_persona_at(&self, idx: usize) -> Option<RecalledAnchorRef<'_>>;
    fn exterior_sandbox_len(&self) -> usize;
    fn exterior_sandbox_at(&self, idx: usize) -> Option<RecalledAnchorRef<'_>>;
    fn top_guard_ref(&self) -> Option<RecalledAnchorRef<'_>>;
}

/// Attenuates a baked Guard pole `+R` in Q7 space to `+R_soft` strictly inside `liminika-adapter`
/// while preserving sign and non-zero guarantees on every non-zero head so `liminika-core`'s
/// single saturating negation (`-R`) produces a gentler anti-phase rotation without double negation.
#[must_use]
pub fn attenuate_guard_q7_soft(
    q7: [lmnk_q7_t; MAX_ATTENTION_HEADS],
    scale: f32,
) -> [lmnk_q7_t; MAX_ATTENTION_HEADS] {
    let s = if scale.is_finite() {
        scale.clamp(0.05, 1.0)
    } else {
        DEFAULT_SOFT_GUARD_Q7_SCALE
    };
    let mut out = [0_i8; MAX_ATTENTION_HEADS];
    for (dst, &src) in out.iter_mut().zip(q7.iter()) {
        if src == 0 {
            *dst = 0;
        } else {
            let mag = ((f32::from(src).abs() * s).round() as i32).clamp(1, 127);
            *dst = if src > 0 { mag as i8 } else { -(mag as i8) };
        }
    }
    out
}

/// Classifies the two-tier Guard state from any [`AttractorRecallProjection`], honoring the
/// recall's recorded `guard_hard_margin` and `is_soft_guard_caution` telemetry while cleanly
/// handling mock recall sources that leave telemetry at defaults.
pub(crate) fn classify_guard_tier_from_projection<R: AttractorRecallProjection + ?Sized>(
    recall: &R,
) -> GuardLayerClassification {
    let tel = recall.telemetry();
    let hard_margin = if tel.guard_hard_margin.is_finite() && tel.guard_hard_margin > 0.0 {
        tel.guard_hard_margin
    } else {
        DEFAULT_GUARD_HARD_MARGIN
    };
    let is_repulsion_escape = matches!(
        recall.action_decision(),
        StigmergicActionDecision::RepulsionEscape
    );
    let margin = recall.guard_conflict_margin();
    let max_guard_score = recall
        .top_guard_ref()
        .map_or(f32::NEG_INFINITY, |g| g.score);
    // Recall sources from `TextAttractorCorpus` already clamp `guard_conflict_margin <= 0.0`
    // whenever `max_guard_score < guard_floor`. For mock recall sources without `top_guard`,
    // a positive margin above `hard_margin` still triggers Layer 1 hard-block.
    let guard_floor =
        if tel.is_soft_guard_caution || recall.top_guard_ref().is_some() || margin > hard_margin {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    let effective_score = if recall.top_guard_ref().is_some() {
        max_guard_score
    } else if margin > hard_margin {
        1.0
    } else {
        f32::NEG_INFINITY
    };
    let mut tier = classify_guard_tier(
        effective_score,
        guard_floor,
        margin,
        hard_margin,
        is_repulsion_escape,
    );
    if tier.mode == GuardLayerMode::SoftCaution && recall.top_guard_ref().is_none() {
        // Without an observed Guard pole, do not synthesize a phantom soft-caution Guard anchor.
        tier.mode = GuardLayerMode::None;
        tier.obstruction = 0.0;
        tier.is_soft_guard_caution = false;
    }
    tier
}

/// Computes the 8-granule 4-region [`LacunaPartition8`] (`S & O`, `S \ O`, `O \ S`, Unobserved)
/// from the winning corpus entry's Q7 profile and the query's residual energy across head granules.
pub(crate) fn compute_lacuna_partition_8(
    adapter: &ModelSteeringAdapter,
    top_any_entry: Option<(&TextAttractorEntry, f32)>,
    align_num: f32,
    min_coherence: f32,
    query_per_layer_q: &[&[f32]],
    baseline_ref_per_layer: &[&[f32]],
) -> LacunaPartition8 {
    let geom = adapter.geometry();
    let self_mask = {
        let mut g_abs = [0u32; NUM_STIGMERGIC_GRANULES];
        let heads = geom.num_q_heads.max(1);
        if let Some((entry, _)) = top_any_entry
            && let Some(q7) = entry.per_layer_q7.last()
        {
            let per_g = (heads / NUM_STIGMERGIC_GRANULES).max(1);
            for (g, g_slot) in g_abs.iter_mut().enumerate().take(NUM_STIGMERGIC_GRANULES) {
                let start = (g * per_g) % heads;
                let mut sum = 0u32;
                for off in 0..per_g {
                    let h = (start + off) % heads;
                    sum += u32::from(q7[h.min(MAX_ATTENTION_HEADS - 1)].unsigned_abs());
                }
                *g_slot = sum / (per_g as u32);
            }
        }
        let min_v = *g_abs.iter().min().unwrap_or(&0);
        let max_v = *g_abs.iter().max().unwrap_or(&0);
        if max_v == min_v {
            0x55u8
        } else {
            let sum_v: u32 = g_abs.iter().sum();
            let thresh = (sum_v * 9) / (NUM_STIGMERGIC_GRANULES as u32 * 10);
            let mut mask = 0u8;
            for (g, &val) in g_abs.iter().enumerate() {
                if val >= thresh {
                    mask |= 1u8 << g;
                }
            }
            if mask == 0 || mask == 0xFF {
                0x55u8
            } else {
                mask
            }
        }
    };

    let mut other_granule_density_q7 = [0u8; NUM_STIGMERGIC_GRANULES];
    let align_ratio = (align_num / min_coherence.max(0.12)).clamp(0.0, 1.35);
    let excess_align = (align_ratio - 0.35).max(0.0);
    if excess_align > 0.0 {
        let last_rope_layer = (0..query_per_layer_q.len())
            .rfind(|&l| geom.is_rope_layer(l))
            .unwrap_or(0);
        let mut rel_energy = [1.0_f32; NUM_STIGMERGIC_GRANULES];
        if let (Some(ql), Some(rl)) = (
            query_per_layer_q.get(last_rope_layer),
            baseline_ref_per_layer.get(last_rope_layer),
        ) {
            let heads = geom.num_q_heads.max(1);
            let rdim = geom.rotary_dim.max(1);
            if ql.len() >= heads * rdim && rl.len() >= heads * rdim {
                let per_g = (heads / NUM_STIGMERGIC_GRANULES).max(1);
                let mut raw_e = [0.0_f32; NUM_STIGMERGIC_GRANULES];
                for (g, raw_slot) in raw_e.iter_mut().enumerate().take(NUM_STIGMERGIC_GRANULES) {
                    let start_h = (g * per_g) % heads;
                    let mut acc = 0.0_f64;
                    for off in 0..per_g {
                        let h = (start_h + off) % heads;
                        let base = h * rdim;
                        for d in 0..rdim {
                            let diff = f64::from(ql[base + d] - rl[base + d]);
                            acc += diff * diff;
                        }
                    }
                    *raw_slot = (acc / ((per_g * rdim) as f64)).sqrt() as f32;
                }
                let mean_e = (raw_e.iter().sum::<f32>() / NUM_STIGMERGIC_GRANULES as f32).max(1e-6);
                for g in 0..NUM_STIGMERGIC_GRANULES {
                    rel_energy[g] = (raw_e[g] / mean_e).clamp(0.25, 1.80);
                }
            }
        }
        for g in 0..NUM_STIGMERGIC_GRANULES {
            other_granule_density_q7[g] =
                ((excess_align * rel_energy[g] * 96.0).clamp(0.0, 127.0)) as u8;
        }
    }
    let substrate_tol = adapter
        .core_primitive_state()
        .stigmergy
        .granule_noise_tolerance_q7;
    evaluate_lacuna_partition_branchless(self_mask, &other_granule_density_q7, &substrate_tol, 28)
}

pub(crate) fn build_fact_anchors_per_layer_from_projection<
    R: AttractorRecallProjection + ?Sized,
>(
    recall: &R,
    num_layers: usize,
) -> Vec<Vec<ObjectiveFact<[lmnk_q7_t; MAX_ATTENTION_HEADS]>>> {
    let n_facts = recall.recalled_fact_len();
    (0..num_layers)
        .map(|l| {
            (0..n_facts)
                .filter_map(|i| recall.recalled_fact_at(i))
                .filter_map(|anchor| {
                    anchor.per_layer_q7.get(l).map(|&arr| {
                        let scaled_e = (anchor.energy * anchor.score.clamp(0.25, 1.25)).max(0.25);
                        ObjectiveFact::new(arr, scaled_e, 0, anchor.anchor_id)
                    })
                })
                .collect()
        })
        .collect()
}

pub(crate) fn build_exterior_sandbox_anchors_per_layer_from_projection<
    R: AttractorRecallProjection + ?Sized,
>(
    recall: &R,
    num_layers: usize,
) -> Vec<Vec<ObjectiveFact<[lmnk_q7_t; MAX_ATTENTION_HEADS]>>> {
    if !recall.is_known_unknown_exterior() || recall.exterior_sandbox_len() == 0 {
        return vec![Vec::new(); num_layers];
    }
    let n_ext = recall.exterior_sandbox_len();
    (0..num_layers)
        .map(|l| {
            (0..n_ext)
                .filter_map(|i| recall.exterior_sandbox_at(i))
                .filter_map(|anchor| {
                    anchor.per_layer_q7.get(l).map(|&arr| {
                        let scaled_e =
                            (anchor.energy * anchor.score.clamp(0.25, 0.90) * 0.5).max(0.20);
                        ObjectiveFact::new(arr, scaled_e, 0, anchor.anchor_id)
                    })
                })
                .collect()
        })
        .collect()
}

pub(crate) fn build_persona_anchors_per_layer_from_projection<
    R: AttractorRecallProjection + ?Sized,
>(
    recall: &R,
    num_layers: usize,
) -> Vec<Vec<SubjectivePersona<[lmnk_q7_t; MAX_ATTENTION_HEADS]>>> {
    let n_personas = recall.recalled_persona_len();
    (0..num_layers)
        .map(|l| {
            (0..n_personas)
                .filter_map(|i| recall.recalled_persona_at(i))
                .filter_map(|anchor| {
                    anchor.per_layer_q7.get(l).map(|&arr| {
                        let scaled_e = (anchor.energy * anchor.score.clamp(0.25, 1.25)).max(0.25);
                        SubjectivePersona::new(arr, scaled_e, 0, anchor.anchor_id)
                    })
                })
                .collect()
        })
        .collect()
}

/// Materializes per-layer [`GuardPhaseAnchor`] poles according to two-tier `guard_tier`:
/// - [`GuardLayerMode::None`]: empty slices (zero KL contamination on non-Guard / OOD turns).
/// - [`GuardLayerMode::HardBlock`]: full baked `+R` (`obstruction = clamp(0.55 + margin, 0.55, 1.0)`).
/// - [`GuardLayerMode::SoftCaution`]: adapter-attenuated `+R_soft` (`0.40x` non-zero Q7,
///   `obstruction in 0.52..=0.55`), preserving `liminika-core`'s single `-R` inversion invariant.
pub(crate) fn build_guard_anchors_per_layer_from_projection<
    R: AttractorRecallProjection + ?Sized,
>(
    recall: &R,
    num_layers: usize,
    guard_tier: GuardLayerClassification,
) -> Vec<Vec<GuardPhaseAnchor>> {
    if guard_tier.mode == GuardLayerMode::None {
        return vec![Vec::new(); num_layers];
    }
    let Some(guard) = recall.top_guard_ref() else {
        return vec![Vec::new(); num_layers];
    };
    let is_soft = guard_tier.mode == GuardLayerMode::SoftCaution;
    (0..num_layers)
        .map(|l| match guard.per_layer_q7.get(l) {
            Some(&arr) => {
                let q7 = if is_soft {
                    attenuate_guard_q7_soft(arr, DEFAULT_SOFT_GUARD_Q7_SCALE)
                } else {
                    arr
                };
                vec![GuardPhaseAnchor::new(
                    q7,
                    guard_tier.obstruction,
                    guard_tier.obstruction_threshold,
                )]
            }
            None => Vec::new(),
        })
        .collect()
}

pub(crate) fn join_fact_context_from_projection<R: AttractorRecallProjection + ?Sized>(
    recall: &R,
) -> String {
    (0..recall.recalled_fact_len())
        .filter_map(|i| recall.recalled_fact_at(i).map(|a| a.text))
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn join_persona_context_from_projection<R: AttractorRecallProjection + ?Sized>(
    recall: &R,
) -> String {
    (0..recall.recalled_persona_len())
        .filter_map(|i| recall.recalled_persona_at(i).map(|a| a.text))
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn join_exterior_context_from_projection<R: AttractorRecallProjection + ?Sized>(
    recall: &R,
) -> String {
    (0..recall.exterior_sandbox_len())
        .filter_map(|i| recall.exterior_sandbox_at(i).map(|a| a.text))
        .collect::<Vec<_>>()
        .join(" ")
}

pub(crate) fn context_fingerprint_from_projection<R: AttractorRecallProjection + ?Sized>(
    recall: &R,
) -> u64 {
    if matches!(recall.lacuna_state(), LacunaBoundaryState::LiminalExterior)
        && !recall.is_known_unknown_exterior()
    {
        return 0;
    }
    let main_ctx = (0..recall.recalled_fact_len())
        .filter_map(|i| recall.recalled_fact_at(i).map(|a| a.text))
        .chain(
            (0..recall.recalled_persona_len())
                .filter_map(|i| recall.recalled_persona_at(i).map(|a| a.text)),
        )
        .collect::<Vec<_>>()
        .join(" ");
    let ctx = if main_ctx.is_empty() && recall.is_known_unknown_exterior() {
        join_exterior_context_from_projection(recall)
    } else {
        main_ctx
    };
    if ctx.is_empty() {
        return 0;
    }
    ctx.as_bytes()
        .iter()
        .fold(0xCBF2_9CE4_8422_2325_u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3)
        })
}

impl AttractorRecallProjection for DynamicAttractorRecall<'_> {
    #[inline]
    fn lacuna_state(&self) -> LacunaBoundaryState {
        self.lacuna_state
    }
    #[inline]
    fn top1_fact_score(&self) -> f32 {
        self.top1_fact_score
    }
    #[inline]
    fn top2_fact_score(&self) -> f32 {
        self.top2_fact_score
    }
    #[inline]
    fn top1_persona_score(&self) -> f32 {
        self.top1_persona_score
    }
    #[inline]
    fn guard_conflict_margin(&self) -> f32 {
        self.guard_conflict_margin
    }
    #[inline]
    fn action_decision(&self) -> StigmergicActionDecision {
        self.action_decision
    }
    #[inline]
    fn gateway_frame(&self) -> &GatewayFrame {
        &self.gateway_frame
    }
    #[inline]
    fn telemetry(&self) -> &StratumProbeTelemetry {
        &self.telemetry
    }
    #[inline]
    fn lacuna_partition(&self) -> LacunaPartition8 {
        self.lacuna_partition
    }
    #[inline]
    fn is_known_unknown_exterior(&self) -> bool {
        self.is_known_unknown_exterior
    }
    #[inline]
    fn recalled_fact_len(&self) -> usize {
        self.recalled_facts.len()
    }
    #[inline]
    fn recalled_fact_at(&self, idx: usize) -> Option<RecalledAnchorRef<'_>> {
        self.recalled_facts
            .get(idx)
            .map(|&(e, score)| RecalledAnchorRef {
                anchor_id: e.anchor_id,
                text: e.text.as_str(),
                per_layer_q7: &e.per_layer_q7,
                energy: e.energy,
                score,
            })
    }
    #[inline]
    fn recalled_persona_len(&self) -> usize {
        self.recalled_personas.len()
    }
    #[inline]
    fn recalled_persona_at(&self, idx: usize) -> Option<RecalledAnchorRef<'_>> {
        self.recalled_personas
            .get(idx)
            .map(|&(e, score)| RecalledAnchorRef {
                anchor_id: e.anchor_id,
                text: e.text.as_str(),
                per_layer_q7: &e.per_layer_q7,
                energy: e.energy,
                score,
            })
    }
    #[inline]
    fn exterior_sandbox_len(&self) -> usize {
        self.exterior_sandbox_facts.len()
    }
    #[inline]
    fn exterior_sandbox_at(&self, idx: usize) -> Option<RecalledAnchorRef<'_>> {
        self.exterior_sandbox_facts
            .get(idx)
            .map(|&(e, score)| RecalledAnchorRef {
                anchor_id: e.anchor_id,
                text: e.text.as_str(),
                per_layer_q7: &e.per_layer_q7,
                energy: e.energy,
                score,
            })
    }
    #[inline]
    fn top_guard_ref(&self) -> Option<RecalledAnchorRef<'_>> {
        self.top_guard.map(|(e, score)| RecalledAnchorRef {
            anchor_id: e.anchor_id,
            text: e.text.as_str(),
            per_layer_q7: &e.per_layer_q7,
            energy: e.energy,
            score,
        })
    }
}
