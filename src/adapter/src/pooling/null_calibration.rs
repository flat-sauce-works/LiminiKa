// name: src/adapter/src/pooling/null_calibration.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Non-leaking 2-fold cross-fitted OOD null calibration ([`NullCalibrationEnvelope`]) with
//! deterministic script-group robust envelopes preventing cross-script bimodal distortion.

use super::base_prior::{ModelBaseAttractorProfile, merge_orthonormal_basis};
use super::{
    AdaptiveTokenSpecificityCalibrator, CompiledTokenCorpus, SequenceActivationPooler,
    TokenSequenceEntry, TokenSpecificityCalibrator,
};
use crate::{AttractorRole, ModelSteeringAdapter, RecallPolicy, TextAttractorCorpus};
use liminika_core::runtime::ManifoldProbeBaseline;

/// Minimum samples required within a single script group before applying per-script robust
/// envelope aggregation (below this floor, calibration falls back cleanly to overall statistics).
pub(super) const MIN_SCRIPT_GROUP_SAMPLES: usize = 4;

/// Coarse script-family bucket used solely for deterministic bimodal null-calibration defense.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ScriptGroup {
    /// Predominantly ASCII text (`< 25%` non-ASCII non-whitespace characters).
    AsciiDominant,
    /// Non-ASCII script text (CJK, Japanese, Cyrillic, etc.; `>= 25%` non-ASCII chars).
    NonAsciiDominant,
}

/// Classifies `text` deterministically into [`ScriptGroup`] without external language detectors.
#[must_use]
pub(super) fn classify_script_group(text: &str) -> ScriptGroup {
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
/// probes via 2-fold cross-fitting.
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
}

/// Deterministic scalar robust statistics over finite `f32` samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct RobustScalarSummary {
    pub(super) mean: f32,
    pub(super) std: f32,
    pub(super) median: f32,
    pub(super) mad: f32,
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
        let p85_idx = ((n * 85) / 100).min(n - 1);
        let p90_idx = ((n.saturating_sub(1) * 9).div_ceil(10)).min(n - 1);
        Some(Self {
            mean,
            std,
            median,
            mad,
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

/// Aggregates held-out 2-fold OOD calibration observations into a [`NullCalibrationEnvelope`]
/// using a conservative script-group envelope whenever both ASCII-dominant and non-ASCII-dominant
/// groups have at least [`MIN_SCRIPT_GROUP_SAMPLES`] samples.
///
/// Computing per-script robust bounds and taking their conservative upper envelope (`max`) avoids
/// two complementary cross-script bimodal traps:
/// 1. Inter-script variance inflation (`mean + 1.5 * std` blowing up when one script sits far from
///    a monolingual corpus).
/// 2. Quantile dilution (`p85`/`p90` dropping when half the probes are low-scoring foreign script).
pub(super) fn aggregate_null_envelope(
    corpus_offdiag_floor: f32,
    peak_samples: &[(ScriptGroup, f32)],
    ortho_samples: &[(ScriptGroup, f32)],
    tok_cos_samples: &[(ScriptGroup, f32)],
) -> Option<(NullCalibrationEnvelope, f32)> {
    let (all_peaks, ascii_peaks, non_ascii_peaks) = split_by_script(peak_samples);
    let overall_peaks = RobustScalarSummary::from_values(&all_peaks)?;

    let ood_null_high_raw = if ascii_peaks.len() >= MIN_SCRIPT_GROUP_SAMPLES
        && non_ascii_peaks.len() >= MIN_SCRIPT_GROUP_SAMPLES
    {
        let ascii_sum = RobustScalarSummary::from_values(&ascii_peaks)?;
        let non_ascii_sum = RobustScalarSummary::from_values(&non_ascii_peaks)?;
        ascii_sum.high_bound().max(non_ascii_sum.high_bound())
    } else {
        overall_peaks.high_bound()
    };
    let ood_null_high = ood_null_high_raw.clamp(0.15, 0.36);
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

    let (all_toks, ascii_toks, non_ascii_toks) = split_by_script(tok_cos_samples);
    let token_explain_threshold = if ascii_toks.len() >= MIN_SCRIPT_GROUP_SAMPLES
        && non_ascii_toks.len() >= MIN_SCRIPT_GROUP_SAMPLES
    {
        let a_p85 = RobustScalarSummary::from_values(&ascii_toks).map_or(0.35, |s| s.p85);
        let n_p85 = RobustScalarSummary::from_values(&non_ascii_toks).map_or(0.35, |s| s.p85);
        (a_p85.max(n_p85) + 0.05).clamp(0.35, 0.56)
    } else if let Some(tok_sum) = RobustScalarSummary::from_values(&all_toks) {
        (tok_sum.p85 + 0.05).clamp(0.35, 0.56)
    } else {
        0.40_f32
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
    pub(super) novelty_reference_per_layer: &'a [Vec<f32>],
    pub(super) corpus_centroid_per_layer: &'a [Vec<f32>],
    pub(super) recall_policy: RecallPolicy,
    pub(super) corpus_offdiag_floor: f32,
    pub(super) pooler: SequenceActivationPooler,
}

/// Runs deterministic 2-fold cross-fitting over `ctx.prior.calibration_samples`: fits prior
/// common-mode subspaces on one parity half and evaluates the strictly held-out parity half
/// through the full Stratum recall pipeline (honoring per-sample `token_include_mask`).
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
        fold_corpus.extend_common_mode_basis(&fold_prior.pooled_common_mode_basis, 3);
        let fold_tok_basis = merge_orthonormal_basis(
            ctx.corpus_tok_basis,
            &fold_prior.token_common_mode_basis,
            12,
        );
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
            token_explain_threshold: 0.40,
            pooler: ctx.pooler,
        };

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
            if let Ok(q_salient) = CompiledTokenCorpus::extract_salient_token_sigs_indexed_masked(
                ctx.adapter,
                &sample.per_layer_token_qs,
                sample.seq_len,
                ctx.novelty_reference_per_layer,
                &fold_tok_basis,
                16,
                ctx.pooler,
                mask,
            ) {
                for (_, q_tok, _) in &q_salient {
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
                }
            }
        }
    }

    aggregate_null_envelope(
        ctx.corpus_offdiag_floor,
        &ood_null_peaks,
        &ood_ortho_ratios,
        &ood_tok_max_cos,
    )
}
