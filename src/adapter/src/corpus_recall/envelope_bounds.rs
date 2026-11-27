// name: src/adapter/src/corpus_recall/envelope_bounds.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Adapter-side viability envelope bounds ([`RecallEnvelopeBounds`]) consolidating all Stage 1
//! recall, null-calibration, 2D boundary, and coalition safety floors/ceilings in one SSOT.
//!
//! Keeps [`liminika_core::drivers::EnvelopeBounds`] strictly semantic-free while eliminating
//! duplicated clamping literals across `pooling`, `corpus_recall`, and `stratum`.

/// Single-source-of-truth Stage 1 viability bounds for LLM geometric recall calibration,
/// 2D `(top1_coh, unexp)` boundary classification, two-tier Guard margins, and coalition scaling.
///
/// All Stage 2 model-calibrated parameters operate strictly inside these viability bounds so
/// cross-model distributions can adapt freely without collapsing recall or relaxing OOD safety.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecallEnvelopeBounds {
    /// Minimum empirical corpus off-diagonal coherence floor (`0.15`).
    pub corpus_coherence_floor: f32,
    /// Maximum empirical corpus off-diagonal coherence ceiling (`0.75`).
    pub corpus_coherence_ceiling: f32,
    /// Minimum top-1 minus top-2 liminal boundary margin (`0.01`).
    pub boundary_margin_floor: f32,
    /// Maximum top-1 minus top-2 liminal boundary margin (`0.12`).
    pub boundary_margin_ceiling: f32,
    /// Minimum out-of-sample OOD null coherence floor (`0.15`).
    pub ood_null_coherence_floor: f32,
    /// Maximum out-of-sample OOD null coherence ceiling (`0.50`, $\cos(60^\circ)$ acute-cone
    /// viability bound replacing the legacy `0.36` cap while preserving 2D recall in high-baseline
    /// anisotropic scripts).
    pub ood_null_coherence_ceiling: f32,
    /// Safety floor on robust OOD coherence MAD (`0.025`).
    pub ood_coh_mad_floor: f32,
    /// Safety floor on robust OOD unexplained-token MAD (`0.04`).
    pub ood_unexp_mad_floor: f32,
    /// Minimum calibrated token-level corpus explanation cosine threshold (`0.22`).
    pub token_explain_floor: f32,
    /// Fallback token-level corpus explanation threshold under `--no-base-prior` (`0.40`).
    pub default_token_explain_threshold: f32,
    /// Maximum calibrated token-level corpus explanation cosine threshold (`0.62`).
    pub token_explain_ceiling: f32,
    /// Minimum calibrated foreign-OOD unexplained-token cutoff (`0.65`).
    pub u_foreign_floor: f32,
    /// Maximum Stage 1 foreign-OOD unexplained-token viability ceiling (`0.80`; queries with
    /// `>= 80%` unexplained salient tokens are strictly foreign OOD and never admitted).
    pub u_foreign_ceiling: f32,
    /// Minimum calibrated semi-known (`Exterior`) unexplained-token cutoff (`0.45`).
    pub u_semi_floor: f32,
    /// Minimum separation maintained between `u_semi` and `u_foreign` (`0.12`).
    pub u_semi_foreign_min_gap: f32,
    /// Minimum relative coherence dispersion scale (`0.60`).
    pub coherence_scale_floor: f32,
    /// Maximum relative coherence dispersion scale (`1.60`).
    pub coherence_scale_ceiling: f32,
    /// Minimum Layer 1 hard-block Guard conflict margin (`0.03`).
    pub guard_hard_margin_floor: f32,
    /// Maximum Layer 1 hard-block Guard conflict margin (`0.08`).
    pub guard_hard_margin_ceiling: f32,
    /// Minimum coalition unexplained-token cutoff (`0.55`).
    pub u_coalition_floor: f32,
    /// Maximum coalition unexplained-token cutoff (`0.82`).
    pub u_coalition_ceiling: f32,
    /// Minimum token-only basin-support unexplained-token cutoff (`0.50`).
    pub u_basin_support_floor: f32,
    /// Maximum token-only basin-support unexplained-token cutoff (`0.78`).
    pub u_basin_support_ceiling: f32,
}

impl RecallEnvelopeBounds {
    /// Returns the canonical Stage 1 viability envelope shared across `pooling`, `corpus_recall`,
    /// and `stratum`.
    #[inline]
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            corpus_coherence_floor: 0.15,
            corpus_coherence_ceiling: 0.75,
            boundary_margin_floor: 0.01,
            boundary_margin_ceiling: 0.12,
            ood_null_coherence_floor: 0.15,
            ood_null_coherence_ceiling: 0.50,
            ood_coh_mad_floor: 0.025,
            ood_unexp_mad_floor: 0.04,
            token_explain_floor: 0.22,
            default_token_explain_threshold: 0.40,
            token_explain_ceiling: 0.62,
            u_foreign_floor: 0.65,
            u_foreign_ceiling: 0.80,
            u_semi_floor: 0.45,
            u_semi_foreign_min_gap: 0.12,
            coherence_scale_floor: 0.60,
            coherence_scale_ceiling: 1.60,
            guard_hard_margin_floor: 0.03,
            guard_hard_margin_ceiling: 0.08,
            u_coalition_floor: 0.55,
            u_coalition_ceiling: 0.82,
            u_basin_support_floor: 0.50,
            u_basin_support_ceiling: 0.78,
        }
    }

    /// Verifies that every bound is finite and all floor/ceiling/gap ordering invariants hold.
    #[must_use]
    pub fn validate_invariants(&self) -> bool {
        let all_finite = [
            self.corpus_coherence_floor,
            self.corpus_coherence_ceiling,
            self.boundary_margin_floor,
            self.boundary_margin_ceiling,
            self.ood_null_coherence_floor,
            self.ood_null_coherence_ceiling,
            self.ood_coh_mad_floor,
            self.ood_unexp_mad_floor,
            self.token_explain_floor,
            self.default_token_explain_threshold,
            self.token_explain_ceiling,
            self.u_foreign_floor,
            self.u_foreign_ceiling,
            self.u_semi_floor,
            self.u_semi_foreign_min_gap,
            self.coherence_scale_floor,
            self.coherence_scale_ceiling,
            self.guard_hard_margin_floor,
            self.guard_hard_margin_ceiling,
            self.u_coalition_floor,
            self.u_coalition_ceiling,
            self.u_basin_support_floor,
            self.u_basin_support_ceiling,
        ]
        .iter()
        .all(|v| v.is_finite());

        all_finite
            && self.corpus_coherence_floor > 0.0
            && self.corpus_coherence_floor <= self.corpus_coherence_ceiling
            && self.corpus_coherence_ceiling < 1.0
            && self.boundary_margin_floor > 0.0
            && self.boundary_margin_floor <= self.boundary_margin_ceiling
            && self.boundary_margin_ceiling < 1.0
            && self.ood_null_coherence_floor > 0.0
            && self.ood_null_coherence_floor <= self.ood_null_coherence_ceiling
            && self.ood_null_coherence_ceiling <= self.corpus_coherence_ceiling
            && self.ood_coh_mad_floor > 0.0
            && self.ood_coh_mad_floor < 1.0
            && self.ood_unexp_mad_floor > 0.0
            && self.ood_unexp_mad_floor < 1.0
            && self.token_explain_floor > 0.0
            && self.token_explain_floor <= self.default_token_explain_threshold
            && self.default_token_explain_threshold <= self.token_explain_ceiling
            && self.token_explain_ceiling < 1.0
            && self.u_semi_floor > 0.0
            && self.u_semi_foreign_min_gap > 0.0
            && self.u_semi_floor + self.u_semi_foreign_min_gap <= self.u_foreign_floor
            && self.u_foreign_floor <= self.u_foreign_ceiling
            && self.u_foreign_ceiling <= 1.0
            && self.coherence_scale_floor > 0.0
            && self.coherence_scale_floor <= 1.0
            && self.coherence_scale_ceiling >= 1.0
            && self.guard_hard_margin_floor > 0.0
            && self.guard_hard_margin_floor <= self.guard_hard_margin_ceiling
            && self.guard_hard_margin_ceiling < 1.0
            && self.u_basin_support_floor > 0.0
            && self.u_basin_support_floor <= self.u_coalition_floor
            && self.u_coalition_floor <= self.u_coalition_ceiling
            && self.u_coalition_ceiling <= 1.0
            && self.u_basin_support_floor <= self.u_basin_support_ceiling
            && self.u_basin_support_ceiling <= self.u_coalition_ceiling
    }
}

impl Default for RecallEnvelopeBounds {
    #[inline]
    fn default() -> Self {
        Self::standard()
    }
}
