// name: src/adapter/src/pooling/null_calibration.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Non-leaking 2-fold cross-fitted OOD null calibration ([`NullCalibrationEnvelope`]) with
//! 2-pass `(top1_coh, unexp)` robust statistics and deterministic script-group conservative
//! envelopes preventing cross-script bimodal distortion.

use super::base_prior::{
    ModelBaseAttractorProfile, merge_orthonormal_basis,
    orthogonalize_prior_basis_against_discriminative,
};
use super::{
    AdaptiveTokenSpecificityCalibrator, CompiledTokenCorpus, SequenceActivationPooler,
    TokenSequenceEntry, TokenSpecificityCalibrator,
};
use crate::{
    AttractorRole, ModelSteeringAdapter, RecallEnvelopeBounds, RecallPolicy, TextAttractorCorpus,
};
use liminika_core::runtime::ManifoldProbeBaseline;

/// Minimum samples required within a single script group before applying per-script robust
/// envelope aggregation (below this floor, calibration falls back cleanly to overall statistics).
pub(super) const MIN_SCRIPT_GROUP_SAMPLES: usize = 4;

/// Safety floor on robust OOD coherence dispersion (`ood_coh_mad`) preventing division-by-zero
/// or unstable $z_c$ amplification on narrow synthetic or degenerate probe sets.
pub const OOD_COH_MAD_FLOOR: f32 = RecallEnvelopeBounds::standard().ood_coh_mad_floor;

/// Safety floor on robust OOD unexplained-token dispersion (`ood_unexp_mad`) preventing unstable
/// $z_u$ amplification when held-out OOD probes saturate near `1.0`.
pub const OOD_UNEXP_MAD_FLOOR: f32 = RecallEnvelopeBounds::standard().ood_unexp_mad_floor;

/// Fallback `token_explain_threshold` used under `--no-base-prior` or empty token calibration sets.
pub const DEFAULT_TOKEN_EXPLAIN_THRESHOLD: f32 =
    RecallEnvelopeBounds::standard().default_token_explain_threshold;

/// Coarse script-family bucket used solely for deterministic bimodal null-calibration defense.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScriptGroup {
    /// Predominantly ASCII text (`< 25%` non-ASCII non-whitespace characters).
    AsciiDominant,
    /// Non-ASCII script text (CJK, Japanese, Cyrillic, etc.; `>= 25%` non-ASCII chars).
    NonAsciiDominant,
}

/// Classifies `text` deterministically into [`ScriptGroup`] without external language detectors.
#[must_use]
pub(crate) fn classify_script_group(text: &str) -> ScriptGroup {
    let mut total = 0_usize;
    let mut non_ascii = 0_usize;
    for ch in text.chars() {
        if ch.is_whitespace() {
            continue;
        }
        total += 1;
        if !ch.is_ascii() {
            non_ascii += 1;
        }
    }
    if total > 0 && non_ascii.saturating_mul(4) >= total {
        ScriptGroup::NonAsciiDominant
    } else {
        ScriptGroup::AsciiDominant
    }
}

/// Non-leaking out-of-sample OOD null distribution envelope calibrated from startup base-prior
/// probes via 2-fold, 2-pass cross-fitting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NullCalibrationEnvelope {
    /// Number of strictly held-out OOD probes evaluated across cross-fitted folds.
    pub sample_count: usize,
    /// Empirical off-diagonal corpus floor (`mean + 1.25 * std`).
    pub corpus_offdiag_floor: f32,
    /// Mean out-of-sample OOD null peak score across held-out probes.
    pub ood_null_mean: f32,
    /// Standard deviation of out-of-sample OOD null peak scores.
    pub ood_null_std: f32,
    /// Empirical maximum out-of-sample OOD null peak score.
    pub ood_null_max: f32,
    /// Calibrated high-quantile OOD null threshold (`ood_null_high`).
    pub ood_null_high: f32,
    /// Final auto-calibrated coherence floor `max(corpus_offdiag_floor, ood_null_high)`.
    pub calibrated_min_coherence: f32,
    /// Median orthogonal residual norm ratio `rho_perp` observed on held-out OOD probes.
    pub ood_null_median_ortho_ratio: f32,
    /// Calibrated token-level corpus explanation threshold derived from held-out OOD tokens.
    pub token_explain_threshold: f32,
    /// Conservative upper-envelope median of held-out OOD Fact coherence ($\mu_c^{\mathrm{ood}}$).
    pub ood_coh_median: f32,
    /// Conservative robust MAD of held-out OOD Fact coherence (floored at [`OOD_COH_MAD_FLOOR`]).
    pub ood_coh_mad: f32,
    /// Conservative lower-envelope median of held-out OOD unexplained-token ratios ($\mu_u^{\mathrm{ood}}$).
    pub ood_unexp_median: f32,
    /// Conservative robust MAD of held-out OOD unexplained-token ratios (floored at [`OOD_UNEXP_MAD_FLOOR`]).
    pub ood_unexp_mad: f32,
    /// Conservative lower-envelope 15th percentile of held-out OOD unexplained-token ratios.
    pub ood_unexp_p15: f32,
}

/// Deterministic scalar robust statistics over finite `f32` samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct RobustScalarSummary {
    pub(super) mean: f32,
    pub(super) std: f32,
    pub(super) median: f32,
    pub(super) mad: f32,
    pub(super) p15: f32,
    pub(super) p85: f32,
    pub(super) p90: f32,
    pub(super) max: f32,
}

impl RobustScalarSummary {
    /// Summarizes finite values using `f32::total_cmp` ordering (zero RNG, bit-deterministic).
    pub(super) fn from_values(values: &[f32]) -> Option<Self> {
        let mut sorted: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
        if sorted.is_empty() {
            return None;
        }
        sorted.sort_by(f32::total_cmp);
        let n = sorted.len();
        let m = n as f64;
        let mean_f64 = sorted.iter().map(|&v| f64::from(v)).sum::<f64>() / m;
        let mean = mean_f64 as f32;
        let var = sorted
            .iter()
            .map(|&v| {
                let d = f64::from(v) - mean_f64;
                d * d
            })
            .sum::<f64>()
            / m;
        let std = var.sqrt() as f32;
        let median = sorted_median(&sorted);
        let mut abs_devs: Vec<f32> = sorted.iter().map(|&v| (v - median).abs()).collect();
        abs_devs.sort_by(f32::total_cmp);
        let mad = sorted_median(&abs_devs);
        let p15_idx = ((n.saturating_sub(1) * 15) / 100).min(n - 1);
        let p85_idx = ((n * 85) / 100).min(n - 1);
        let p90_idx = ((n.saturating_sub(1) * 9).div_ceil(10)).min(n - 1);
        Some(Self {
            mean,
            std,
            median,
            mad,
            p15: sorted[p15_idx],
            p85: sorted[p85_idx],
            p90: sorted[p90_idx],
            max: sorted[n - 1],
        })
    }

    /// Computes a high-quantile scalar bound resilient to isolated outliers while preserving
    /// compatibility with unimodal Gaussian tails (`max(robust_hi, p90 + 0.015)`).
    #[must_use]
    pub(super) fn high_bound(&self) -> f32 {
        let parametric = self.mean + 1.50 * self.std;
        let robust_sigma = 1.4826_f32 * self.mad;
        let tail_est = if robust_sigma > 1e-6 {
            (self.median + 1.50 * robust_sigma).min(parametric)
        } else {
            parametric
        };
        tail_est.max(self.p90 + 0.015)
    }
}

fn sorted_median(sorted: &[f32]) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    let mid = n / 2;
    if n.is_multiple_of(2) {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    }
}

fn split_by_script(samples: &[(ScriptGroup, f32)]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut all = Vec::with_capacity(samples.len());
    let mut ascii = Vec::new();
    let mut non_ascii = Vec::new();
    for &(group, val) in samples {
        if !val.is_finite() {
            continue;
        }
        all.push(val);
        match group {
            ScriptGroup::AsciiDominant => ascii.push(val),
            ScriptGroup::NonAsciiDominant => non_ascii.push(val),
        }
    }
    (all, ascii, non_ascii)
}

/// Resolves the calibrated `token_explain_threshold` from held-out OOD token max-cosines using
/// conservative per-script upper envelopes bounded by [`RecallEnvelopeBounds::standard`].
#[must_use]
pub(super) fn resolve_token_explain_threshold(tok_cos_samples: &[(ScriptGroup, f32)]) -> f32 {
    let bounds = RecallEnvelopeBounds::standard();
    let (all_toks, ascii_toks, non_ascii_toks) = split_by_script(tok_cos_samples);
    if ascii_toks.len() >= MIN_SCRIPT_GROUP_SAMPLES
        && non_ascii_toks.len() >= MIN_SCRIPT_GROUP_SAMPLES
    {
        let fallback_p85 = bounds.default_token_explain_threshold - 0.04;
        let a_p85 = RobustScalarSummary::from_values(&ascii_toks).map_or(fallback_p85, |s| s.p85);
        let n_p85 =
            RobustScalarSummary::from_values(&non_ascii_toks).map_or(fallback_p85, |s| s.p85);
        (a_p85.max(n_p85) + 0.04).clamp(bounds.token_explain_floor, bounds.token_explain_ceiling)
    } else if let Some(tok_sum) = RobustScalarSummary::from_values(&all_toks) {
        (tok_sum.p85 + 0.04).clamp(bounds.token_explain_floor, bounds.token_explain_ceiling)
    } else {
        bounds.default_token_explain_threshold
    }
}

/// Aggregates held-out 2-fold OOD calibration observations into a [`NullCalibrationEnvelope`]
/// using conservative script-group envelopes whenever both ASCII-dominant and non-ASCII-dominant
/// groups have at least [`MIN_SCRIPT_GROUP_SAMPLES`] samples:
/// - Coherence (`ood_coh_*`, `ood_null_high`, `token_explain_threshold`): conservative upper envelope
///   (`max`) so neither script's higher OOD tail leaks into `SelfRef`.
/// - Unexplained token ratio (`ood_unexp_median`, `ood_unexp_p15`): conservative lower envelope
///   (`min`) so neither script's lower OOD unexplained ratio bypasses `u_foreign`.
pub(super) fn aggregate_null_envelope(
    corpus_offdiag_floor: f32,
    peak_samples: &[(ScriptGroup, f32)],
    ortho_samples: &[(ScriptGroup, f32)],
    tok_cos_samples: &[(ScriptGroup, f32)],
    unexp_samples: &[(ScriptGroup, f32)],
) -> Option<(NullCalibrationEnvelope, f32)> {
    let bounds = RecallEnvelopeBounds::standard();
    let (all_peaks, ascii_peaks, non_ascii_peaks) = split_by_script(peak_samples);
    let overall_peaks = RobustScalarSummary::from_values(&all_peaks)?;

    let (ood_null_high_raw, ood_coh_median, ood_coh_mad) = if ascii_peaks.len()
        >= MIN_SCRIPT_GROUP_SAMPLES
        && non_ascii_peaks.len() >= MIN_SCRIPT_GROUP_SAMPLES
    {
        let ascii_sum = RobustScalarSummary::from_values(&ascii_peaks)?;
        let non_ascii_sum = RobustScalarSummary::from_values(&non_ascii_peaks)?;
        (
            ascii_sum.high_bound().max(non_ascii_sum.high_bound()),
            ascii_sum.median.max(non_ascii_sum.median),
            ascii_sum
                .mad
                .max(non_ascii_sum.mad)
                .max(bounds.ood_coh_mad_floor),
        )
    } else {
        (
            overall_peaks.high_bound(),
            overall_peaks.median,
            overall_peaks.mad.max(bounds.ood_coh_mad_floor),
        )
    };
    let ood_null_high = ood_null_high_raw.clamp(
        bounds.ood_null_coherence_floor,
        bounds.ood_null_coherence_ceiling,
    );
    let calibrated_min_coherence = corpus_offdiag_floor.max(ood_null_high);

    let (all_orthos, ascii_orthos, non_ascii_orthos) = split_by_script(ortho_samples);
    let ood_null_median_ortho_ratio = if ascii_orthos.len() >= MIN_SCRIPT_GROUP_SAMPLES
        && non_ascii_orthos.len() >= MIN_SCRIPT_GROUP_SAMPLES
    {
        let a_med = RobustScalarSummary::from_values(&ascii_orthos).map_or(1.0, |s| s.median);
        let n_med = RobustScalarSummary::from_values(&non_ascii_orthos).map_or(1.0, |s| s.median);
        a_med.max(n_med)
    } else {
        RobustScalarSummary::from_values(&all_orthos).map_or(1.0, |s| s.median)
    };

    let token_explain_threshold = resolve_token_explain_threshold(tok_cos_samples);

    let (all_unexp, ascii_unexp, non_ascii_unexp) = split_by_script(unexp_samples);
    let (ood_unexp_median, ood_unexp_mad, ood_unexp_p15) = if ascii_unexp.len()
        >= MIN_SCRIPT_GROUP_SAMPLES
        && non_ascii_unexp.len() >= MIN_SCRIPT_GROUP_SAMPLES
        && let (Some(a_sum), Some(n_sum)) = (
            RobustScalarSummary::from_values(&ascii_unexp),
            RobustScalarSummary::from_values(&non_ascii_unexp),
        ) {
        (
            a_sum.median.min(n_sum.median).clamp(0.0, 1.0),
            a_sum.mad.max(n_sum.mad).max(bounds.ood_unexp_mad_floor),
            a_sum.p15.min(n_sum.p15).clamp(0.0, 1.0),
        )
    } else if let Some(u_sum) = RobustScalarSummary::from_values(&all_unexp) {
        (
            u_sum.median.clamp(0.0, 1.0),
            u_sum.mad.max(bounds.ood_unexp_mad_floor),
            u_sum.p15.clamp(0.0, 1.0),
        )
    } else {
        (
            1.0_f32,
            bounds.ood_unexp_mad_floor,
            bounds.u_foreign_ceiling,
        )
    };

    let envelope = NullCalibrationEnvelope {
        sample_count: all_peaks.len(),
        corpus_offdiag_floor,
        ood_null_mean: overall_peaks.mean,
        ood_null_std: overall_peaks.std,
        ood_null_max: overall_peaks.max,
        ood_null_high,
        calibrated_min_coherence,
        ood_null_median_ortho_ratio,
        token_explain_threshold,
        ood_coh_median,
        ood_coh_mad,
        ood_unexp_median,
        ood_unexp_mad,
        ood_unexp_p15,
    };
    Some((envelope, token_explain_threshold))
}

/// Shared inputs for non-leaking 2-fold cross-fitted null calibration across both batch and
/// streaming corpus compilation paths.
pub(super) struct NullCalibrationContext<'a> {
    pub(super) adapter: &'a ModelSteeringAdapter,
    pub(super) prior: &'a ModelBaseAttractorProfile,
    pub(super) corpus_pre_prior: &'a TextAttractorCorpus,
    pub(super) corpus_tok_basis: &'a [Vec<f32>],
    pub(super) pooled_disc_basis: &'a [Vec<f32>],
    pub(super) tok_disc_basis: &'a [Vec<f32>],
    pub(super) novelty_reference_per_layer: &'a [Vec<f32>],
    pub(super) corpus_centroid_per_layer: &'a [Vec<f32>],
    pub(super) recall_policy: RecallPolicy,
    pub(super) corpus_offdiag_floor: f32,
    pub(super) pooler: SequenceActivationPooler,
}

/// Runs deterministic 2-fold, 2-pass cross-fitting over `ctx.prior.calibration_samples`:
/// - **Pass 1**: fits prior common-mode subspaces on one parity half, evaluates the strictly
///   held-out parity half through the full Stratum recall pipeline (honoring `token_include_mask`),
///   and records per-token best Fact cosines + SIF weights across both folds to finalize
///   `token_explain_threshold` without fold-ordering bias.
/// - **Pass 2**: applies the finalized `token_explain_threshold` uniformly across all held-out
///   probes to compute `ood_unexp` (`unexplained_token_ratio`) and aggregates 2D robust statistics.
pub(super) fn calibrate_null_envelope_2fold<F>(
    ctx: &NullCalibrationContext<'_>,
    build_fold_entry_toks: F,
) -> Option<(NullCalibrationEnvelope, f32)>
where
    F: Fn(&[Vec<f32>]) -> Vec<Vec<Vec<f32>>>,
{
    if ctx.prior.calibration_samples.len() < 4 {
        return None;
    }

    let n_cal = ctx.prior.calibration_samples.len();
    let mut ood_null_peaks: Vec<(ScriptGroup, f32)> = Vec::with_capacity(n_cal);
    let mut ood_ortho_ratios: Vec<(ScriptGroup, f32)> = Vec::with_capacity(n_cal);
    let mut ood_tok_max_cos: Vec<(ScriptGroup, f32)> = Vec::new();
    let mut held_out_sample_tokens: Vec<(ScriptGroup, Vec<(f32, f32)>)> = Vec::with_capacity(n_cal);

    // Pass 1: collect held-out OOD Fact peaks, ortho ratios, and per-token best Fact cosines.
    for fold in 0..2_usize {
        let fit_samples: Vec<TokenSequenceEntry> = ctx
            .prior
            .calibration_samples
            .iter()
            .enumerate()
            .filter(|(idx, _)| idx % 2 != fold)
            .map(|(_, s)| s.clone())
            .collect();
        let eval_samples: Vec<&TokenSequenceEntry> = ctx
            .prior
            .calibration_samples
            .iter()
            .enumerate()
            .filter(|(idx, _)| idx % 2 == fold)
            .map(|(_, s)| s)
            .collect();
        if fit_samples.is_empty() || eval_samples.is_empty() {
            continue;
        }
        let Ok(fold_prior) =
            ModelBaseAttractorProfile::extract(ctx.adapter, &fit_samples, ctx.pooler, 0.55)
        else {
            continue;
        };

        let mut fold_corpus = ctx.corpus_pre_prior.clone();
        let safe_fold_pooled = orthogonalize_prior_basis_against_discriminative(
            &fold_prior.pooled_common_mode_basis,
            ctx.pooled_disc_basis,
        );
        fold_corpus.extend_common_mode_basis(&safe_fold_pooled, 3);
        let safe_fold_tok = orthogonalize_prior_basis_against_discriminative(
            &fold_prior.token_common_mode_basis,
            ctx.tok_disc_basis,
        );
        let fold_tok_basis = merge_orthonormal_basis(ctx.corpus_tok_basis, &safe_fold_tok, 12);
        let fold_entry_toks = build_fold_entry_toks(&fold_tok_basis);
        let fold_stats = AdaptiveTokenSpecificityCalibrator::default()
            .calibrate_token_stats(&fold_entry_toks, fold_corpus.entries());

        let fold_compiled = CompiledTokenCorpus {
            corpus: fold_corpus,
            novelty_reference_per_layer: ctx.novelty_reference_per_layer.to_vec(),
            corpus_centroid_per_layer: ctx.corpus_centroid_per_layer.to_vec(),
            token_common_mode_basis: fold_tok_basis.clone(),
            entry_token_sigs: fold_entry_toks,
            entry_token_weights: fold_stats.entry_token_weights,
            entry_token_hub_bias: fold_stats.entry_token_hub_bias,
            global_token_hub_mean: fold_stats.global_token_hub_mean,
            probe_baseline: ManifoldProbeBaseline::default(),
            recall_policy: ctx.recall_policy,
            null_calibration: None,
            token_explain_threshold: DEFAULT_TOKEN_EXPLAIN_THRESHOLD,
            pooler: ctx.pooler,
        };
        let has_fact_docs = fold_compiled
            .corpus
            .entries()
            .iter()
            .enumerate()
            .any(|(i, e)| {
                e.role == AttractorRole::Fact && !fold_compiled.entry_token_sigs[i].is_empty()
            });

        for sample in eval_samples {
            let group = classify_script_group(&sample.text);
            let mask = sample.token_include_mask.as_deref();
            if let Ok(rec) = fold_compiled.recall_token_sequence_with_mask(
                ctx.adapter,
                &sample.per_layer_token_qs,
                sample.seq_len,
                mask,
            ) {
                // Calibrate factual OOD floor strictly from Fact resonance (excluding broad
                // stylistic Persona attractors so style hubs never inflate factual thresholds).
                if rec.top1_fact_score.is_finite() {
                    ood_null_peaks.push((group, rec.top1_fact_score));
                }
                if rec.telemetry.query_ortho_ratio.is_finite() {
                    ood_ortho_ratios.push((group, rec.telemetry.query_ortho_ratio));
                }
            }
            if has_fact_docs
                && let Ok(q_salient) =
                    CompiledTokenCorpus::extract_salient_token_sigs_indexed_masked(
                        ctx.adapter,
                        &sample.per_layer_token_qs,
                        sample.seq_len,
                        ctx.novelty_reference_per_layer,
                        &fold_tok_basis,
                        24,
                        ctx.pooler,
                        mask,
                    )
            {
                let mut sample_cos_weights = Vec::with_capacity(q_salient.len());
                for (_, q_tok, w_t) in &q_salient {
                    let mut best_fact_c = 0.0_f32;
                    for (idx, d_toks) in fold_compiled.entry_token_sigs.iter().enumerate() {
                        if fold_compiled
                            .corpus
                            .entries()
                            .get(idx)
                            .is_none_or(|e| e.role != AttractorRole::Fact)
                        {
                            continue;
                        }
                        for d_tok in d_toks {
                            let c = TextAttractorCorpus::cosine_f32(q_tok, d_tok);
                            if c > best_fact_c {
                                best_fact_c = c;
                            }
                        }
                    }
                    ood_tok_max_cos.push((group, best_fact_c));
                    if *w_t > 0.0 && w_t.is_finite() {
                        sample_cos_weights.push((best_fact_c, *w_t));
                    }
                }
                if !sample_cos_weights.is_empty() {
                    held_out_sample_tokens.push((group, sample_cos_weights));
                }
            }
        }
    }

    // Pass 2: evaluate held-out OOD unexplained-token ratios under the finalized threshold.
    let finalized_tok_thresh = resolve_token_explain_threshold(&ood_tok_max_cos);
    let mut ood_unexp_samples: Vec<(ScriptGroup, f32)> =
        Vec::with_capacity(held_out_sample_tokens.len());
    for (group, tok_cos_weights) in &held_out_sample_tokens {
        let mut unexplained_w = 0.0_f32;
        let mut total_w = 0.0_f32;
        for &(best_c, w_t) in tok_cos_weights {
            total_w += w_t;
            if best_c < finalized_tok_thresh {
                unexplained_w += w_t;
            }
        }
        if total_w > 1e-8 {
            ood_unexp_samples.push((*group, (unexplained_w / total_w).clamp(0.0, 1.0)));
        }
    }

    aggregate_null_envelope(
        ctx.corpus_offdiag_floor,
        &ood_null_peaks,
        &ood_ortho_ratios,
        &ood_tok_max_cos,
        &ood_unexp_samples,
    )
}
