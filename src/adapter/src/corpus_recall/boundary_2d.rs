// name: src/adapter/src/corpus_recall/boundary_2d.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Two-dimensional `(top1_coh, unexp)` recall decision boundaries, `--no-base-prior`
//! 1D compatibility fallbacks, and adapter-side two-tier Guard classification.

use super::{LacunaBoundaryState, RecallEnvelopeBounds, RecallPolicy};
use crate::pooling::NullCalibrationEnvelope;

/// Default hard-block Guard conflict margin above which Layer 1 repulsion suppresses Fact
/// injection and engages the `guard_turn` bypass lock (`0.05`).
pub const DEFAULT_GUARD_HARD_MARGIN: f32 = 0.05;

/// Adapter-side Q7 amplitude attenuation factor for Layer 2 soft-caution Guard poles (`+R_soft`),
/// preserving `liminika-core`'s single `-R` saturating negation invariant.
pub const DEFAULT_SOFT_GUARD_Q7_SCALE: f32 = 0.40;

/// Fixed core cohomology obstruction threshold shared by hard and soft Guard anchors (`0.50`).
pub const DEFAULT_GUARD_OBSTRUCTION_THRESHOLD: f32 = 0.50;

/// Geometric boundary measurements evaluated by a [`RecallBoundaryGate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecallBoundaryMetrics {
    /// Highest geometric resonance score among Fact attractors.
    pub top1_fact_score: f32,
    /// Second-highest geometric resonance score among Fact attractors.
    pub top2_fact_score: f32,
    /// Highest geometric resonance score among Persona attractors.
    pub top1_persona_score: f32,
    /// Highest geometric resonance score among Guard attractors.
    pub max_guard_score: f32,
    /// `max_guard_score - max(top1_fact_score, top1_persona_score)`.
    pub guard_conflict_margin: f32,
}

/// Decision produced by a [`RecallBoundaryGate`] for a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecallGateDecision {
    /// Classified boundary state (`InsideBasin`, `LiminalBoundary`, or `LiminalExterior`).
    pub lacuna_state: LacunaBoundaryState,
    /// Whether forced Fact/Persona attractor injection should be suppressed (e.g., `LiminalExterior`
    /// or Layer 1 hard-block Guard dominance).
    pub suppress_fact_injection: bool,
}

/// Trait abstracting `Liminal` boundary classification and Guard-conflict suppression policies.
pub trait RecallBoundaryGate {
    /// Evaluates boundary metrics under the active [`RecallPolicy`].
    fn evaluate(
        &self,
        metrics: &RecallBoundaryMetrics,
        policy: &RecallPolicy,
    ) -> RecallGateDecision;
}

/// Default [`RecallBoundaryGate`] enforcing both `Liminal` coherence floor and Layer 1 Guard
/// hard-block suppression while leaving narrow boundary-zone conflicts (`0 < margin <= hard_margin`)
/// open for Layer 2 soft-caution contextual audit.
#[derive(Debug, Default, Clone, Copy)]
pub struct CoherenceGuardGate;

impl RecallBoundaryGate for CoherenceGuardGate {
    fn evaluate(
        &self,
        metrics: &RecallBoundaryMetrics,
        policy: &RecallPolicy,
    ) -> RecallGateDecision {
        let guard_floor = policy.min_coherence * 0.70;
        let hard_margin =
            if policy.guard_suppress_margin.is_finite() && policy.guard_suppress_margin > 0.0 {
                policy.guard_suppress_margin
            } else {
                DEFAULT_GUARD_HARD_MARGIN
            };
        if metrics.max_guard_score >= guard_floor && metrics.guard_conflict_margin > hard_margin {
            return RecallGateDecision {
                lacuna_state: LacunaBoundaryState::LiminalExterior,
                suppress_fact_injection: true,
            };
        }
        let margin = metrics.top1_fact_score - metrics.top2_fact_score;
        if metrics.top1_fact_score < policy.min_coherence {
            RecallGateDecision {
                lacuna_state: LacunaBoundaryState::LiminalExterior,
                suppress_fact_injection: true,
            }
        } else if margin < policy.boundary_margin {
            RecallGateDecision {
                lacuna_state: LacunaBoundaryState::LiminalBoundary,
                suppress_fact_injection: false,
            }
        } else {
            RecallGateDecision {
                lacuna_state: LacunaBoundaryState::InsideBasin,
                suppress_fact_injection: false,
            }
        }
    }
}

/// Two-tier Guard intervention regime classified strictly in `liminika-adapter`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardLayerMode {
    /// Non-Guard region (`max_guard_score < guard_floor` or `guard_conflict_margin <= 0.0`):
    /// zero `GuardPhaseAnchor` emission and zero OOD / normal-Fact KL contamination.
    None,
    /// Layer 2 (Soft Caution): `max_guard_score >= guard_floor` and
    /// `0.0 < guard_conflict_margin <= hard_margin` without `RepulsionEscape`.
    /// Preserves Fact injection, emits adapter-attenuated `+R_soft` (`obstruction in 0.52..=0.55`),
    /// avoids `guard_turn` bypass lock, and signals contextual reflection audit.
    SoftCaution,
    /// Layer 1 (Hard Block): `guard_conflict_margin > hard_margin` (with `max_guard_score >=
    /// guard_floor`) or `RepulsionEscape`. Suppresses Fact injection (allowing at most 1 grounded
    /// counter-fact), emits full `+R` (`obstruction = clamp(0.55 + margin, 0.55, 1.0)`), and
    /// activates `guard_turn = true`.
    HardBlock,
}

/// Complete adapter-side two-tier Guard classification outcome.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuardLayerClassification {
    /// Selected Guard intervention tier (`None`, `SoftCaution`, or `HardBlock`).
    pub mode: GuardLayerMode,
    /// Effective hard-block margin threshold (`>= 0.03..=0.08` or policy override).
    pub hard_margin: f32,
    /// Dynamic cohomology obstruction scalar passed into [`liminika_core::runtime::GuardPhaseAnchor`].
    pub obstruction: f32,
    /// Fixed core obstruction threshold (`0.50`).
    pub obstruction_threshold: f32,
    /// True strictly on Layer 1 hard-block (`guard_turn = true`).
    pub is_guard_trap: bool,
    /// True strictly on Layer 2 soft-caution (`guard_turn = false`, contextual reflection audit).
    pub is_soft_guard_caution: bool,
}

/// Computes the effective Layer 1 hard-block margin from `policy_guard_suppress_margin` and the
/// calibrated cosine dispersion scale (`coherence_scale`, `1.0` when uncalibrated).
#[inline]
#[must_use]
pub fn effective_guard_hard_margin(policy_guard_suppress_margin: f32, coherence_scale: f32) -> f32 {
    let bounds = RecallEnvelopeBounds::standard();
    let base = (DEFAULT_GUARD_HARD_MARGIN * coherence_scale).clamp(
        bounds.guard_hard_margin_floor,
        bounds.guard_hard_margin_ceiling,
    );
    if policy_guard_suppress_margin.is_finite() {
        policy_guard_suppress_margin.max(base)
    } else {
        base
    }
}

/// Classifies a query's Guard regime into Layer 1 (`HardBlock`), Layer 2 (`SoftCaution`), or
/// non-Guard (`None`) without modifying `liminika-core`.
#[must_use]
pub fn classify_guard_tier(
    max_guard_score: f32,
    guard_floor: f32,
    guard_conflict_margin: f32,
    hard_margin: f32,
    is_repulsion_escape: bool,
) -> GuardLayerClassification {
    let eff_hard = if hard_margin.is_finite() && hard_margin > 0.0 {
        hard_margin
    } else {
        DEFAULT_GUARD_HARD_MARGIN
    };
    let above_floor = max_guard_score.is_finite() && max_guard_score >= guard_floor;
    let margin = if guard_conflict_margin.is_finite() {
        guard_conflict_margin
    } else {
        0.0
    };

    if is_repulsion_escape || (above_floor && margin > eff_hard) {
        GuardLayerClassification {
            mode: GuardLayerMode::HardBlock,
            hard_margin: eff_hard,
            obstruction: (0.55 + margin.max(0.0)).clamp(0.55, 1.0),
            obstruction_threshold: DEFAULT_GUARD_OBSTRUCTION_THRESHOLD,
            is_guard_trap: true,
            is_soft_guard_caution: false,
        }
    } else if above_floor && margin > 0.0 {
        let frac = (margin / eff_hard.max(1e-4)).clamp(0.0, 1.0);
        GuardLayerClassification {
            mode: GuardLayerMode::SoftCaution,
            hard_margin: eff_hard,
            obstruction: (0.52 + 0.03 * frac).clamp(0.52, 0.55),
            obstruction_threshold: DEFAULT_GUARD_OBSTRUCTION_THRESHOLD,
            is_guard_trap: false,
            is_soft_guard_caution: true,
        }
    } else {
        GuardLayerClassification {
            mode: GuardLayerMode::None,
            hard_margin: eff_hard,
            obstruction: 0.0,
            obstruction_threshold: DEFAULT_GUARD_OBSTRUCTION_THRESHOLD,
            is_guard_trap: false,
            is_soft_guard_caution: false,
        }
    }
}

/// Three epistemic regions classified by the 2D `(top1_coh, unexp)` recall decision boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary2DRegion {
    /// Out-of-domain / unobserved space ($\overline{S \cup O}$): zero Fact injection.
    Unobserved,
    /// Semi-known Known-Unknown ($O \setminus S$): staged strictly in `exterior_sandbox_facts`.
    Exterior,
    /// Grounded in-domain basin ($S \cap O$): retained in `recalled_facts`.
    SelfRef,
}

/// Calibrated 2D `(top1_coh, unexp)` boundary thresholds derived from [`NullCalibrationEnvelope`]
/// or safely degraded to exact 1D compatibility values under `--no-base-prior`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Boundary2DThresholds {
    /// `true` when calibrated from 2-fold OOD statistics; `false` under `--no-base-prior` fallback.
    pub is_calibrated: bool,
    /// Active minimum coherence floor (`policy.min_coherence`).
    pub calibrated_min_coherence: f32,
    /// OOD null coherence center ($\mu_c^{\mathrm{ood}}$).
    pub ood_coh_median: f32,
    /// OOD null coherence robust scale ($\sigma_c^{\mathrm{ood}} = 1.4826 \cdot \text{MAD}$).
    pub ood_coh_sigma: f32,
    /// OOD null unexplained-token center ($\mu_u^{\mathrm{ood}}$).
    pub ood_unexp_median: f32,
    /// OOD null unexplained-token robust scale ($\sigma_u^{\mathrm{ood}} = 1.4826 \cdot \text{MAD}$).
    pub ood_unexp_sigma: f32,
    /// Empirical lower quantile (`p15`) of held-out OOD unexplained-token ratios.
    pub ood_unexp_p15: f32,
    /// Foreign-OOD unexplained-token cutoff ($u_{\mathrm{foreign}}$, fallback `0.80`).
    pub u_foreign: f32,
    /// Semi-known unexplained-token cutoff ($u_{\mathrm{semi}}$, fallback `0.59`).
    pub u_semi: f32,
    /// Token-level corpus explanation cosine threshold (fallback `0.40`).
    pub token_explain_threshold: f32,
    /// Combined 2D score threshold below which queries classify as [`Boundary2DRegion::Unobserved`].
    pub tau_unobs: f32,
    /// Combined 2D score threshold required for [`Boundary2DRegion::SelfRef`].
    pub tau_self: f32,
    /// High-coherence $z_c$ rescue threshold exempting strong matches from `u_foreign` rejection.
    pub z_c_rescue: f32,
    /// Additive `u_semi` safety margin when sharp unimodal Fact lock is present.
    pub sharp_unimodal_semi_margin: f32,
    /// Relative coherence dispersion scale `(ood_coh_mad / 0.05).clamp(0.60, 1.60)` (`1.0` fallback).
    pub coherence_scale: f32,
}

impl Boundary2DThresholds {
    /// Derives calibrated 2D decision thresholds from a 2-fold [`NullCalibrationEnvelope`]
    /// bounded by [`RecallEnvelopeBounds::standard`].
    #[must_use]
    pub fn from_null_calibration(env: &NullCalibrationEnvelope, min_coherence: f32) -> Self {
        let bounds = RecallEnvelopeBounds::standard();
        let coh_mad = env.ood_coh_mad.max(bounds.ood_coh_mad_floor);
        let unexp_mad = env.ood_unexp_mad.max(bounds.ood_unexp_mad_floor);
        // Anchor `sigma_c` so `top1_coh == min_coherence` never exceeds `z_c = 1.50` (`tau_unobs`),
        // preventing narrow/clamped OOD medians from falsely triggering `z_c >= 3.0` rescue.
        let floor_gap_sigma_c = (min_coherence - env.ood_coh_median).max(0.0) / 1.50;
        let sigma_c = (1.4826_f32 * coh_mad).max(floor_gap_sigma_c);
        let sigma_u = 1.4826_f32 * unexp_mad;
        let u_foreign = env
            .ood_unexp_p15
            .min(env.ood_unexp_median - 0.5 * sigma_u)
            .clamp(bounds.u_foreign_floor, bounds.u_foreign_ceiling);
        let u_semi = (env.ood_unexp_median - 1.5 * sigma_u).clamp(
            bounds.u_semi_floor,
            u_foreign - bounds.u_semi_foreign_min_gap,
        );
        let coherence_scale =
            (coh_mad / 0.05).clamp(bounds.coherence_scale_floor, bounds.coherence_scale_ceiling);
        Self {
            is_calibrated: true,
            calibrated_min_coherence: min_coherence,
            ood_coh_median: env.ood_coh_median,
            ood_coh_sigma: sigma_c,
            ood_unexp_median: env.ood_unexp_median,
            ood_unexp_sigma: sigma_u,
            ood_unexp_p15: env.ood_unexp_p15,
            u_foreign,
            u_semi,
            token_explain_threshold: env.token_explain_threshold,
            tau_unobs: 1.50,
            tau_self: 2.00,
            z_c_rescue: 3.0,
            sharp_unimodal_semi_margin: 0.10,
            coherence_scale,
        }
    }

    /// Strict 1D compatibility fallback used when `--no-base-prior` disables null calibration
    /// (`u_foreign = 0.80`, `u_semi = 0.59`, `sharp_unimodal_cutoff = 0.72`, `tok_exp = 0.40`).
    #[must_use]
    pub fn fallback_without_prior(min_coherence: f32) -> Self {
        let bounds = RecallEnvelopeBounds::standard();
        Self {
            is_calibrated: false,
            calibrated_min_coherence: min_coherence,
            ood_coh_median: (min_coherence * 0.50).max(0.05),
            ood_coh_sigma: 1.4826_f32 * 0.05,
            ood_unexp_median: 1.0,
            ood_unexp_sigma: 1.4826_f32 * bounds.ood_unexp_mad_floor,
            ood_unexp_p15: bounds.u_foreign_ceiling,
            u_foreign: bounds.u_foreign_ceiling,
            u_semi: 0.59,
            token_explain_threshold: bounds.default_token_explain_threshold,
            tau_unobs: 1.50,
            tau_self: 2.00,
            z_c_rescue: 3.0,
            sharp_unimodal_semi_margin: 0.13,
            coherence_scale: 1.0,
        }
    }

    /// Constructs thresholds from an optional [`NullCalibrationEnvelope`].
    #[must_use]
    pub fn from_optional_envelope(
        env: Option<&NullCalibrationEnvelope>,
        min_coherence: f32,
    ) -> Self {
        match env {
            Some(e) => Self::from_null_calibration(e, min_coherence),
            None => Self::fallback_without_prior(min_coherence),
        }
    }

    /// Computes standardized 2D coordinates `(z_c, z_u, s_2d)` where `s_2d = 0.60 * z_c + 0.40 * z_u`.
    #[inline]
    #[must_use]
    pub fn standardized_coords(&self, top1_coh: f32, unexp: f32) -> (f32, f32, f32) {
        let z_c = (top1_coh - self.ood_coh_median) / self.ood_coh_sigma.max(1e-4);
        let z_u = (self.ood_unexp_median - unexp) / self.ood_unexp_sigma.max(1e-4);
        let s_2d = 0.60 * z_c + 0.40 * z_u;
        (z_c, z_u, s_2d)
    }

    /// Classifies a query into [`Boundary2DRegion`] using either calibrated 2D decision geometry
    /// or the strict 1D compatibility fallback when `self.is_calibrated == false`.
    #[must_use]
    pub fn classify(&self, input: &Boundary2DInput) -> Boundary2DDecision {
        let (z_c, z_u, s_2d) =
            self.standardized_coords(input.top1_coh, input.unexplained_token_ratio);
        if input.repulse_guard {
            return Boundary2DDecision {
                region: Boundary2DRegion::Unobserved,
                foreign_ood_tokens: false,
                z_c,
                z_u,
                s_2d,
                stage_top1_candidate_to_exterior: false,
            };
        }

        let bridge_eligible = input.guard_conflict_margin <= 0.0
            && input.is_sandbox_eligible
            && input.bridge_count >= 2
            && input.has_top1_fact_candidate;
        let unimodal_margin = (input.top1_coh - input.top2_coh).max(0.0);
        let sharp_unimodal_lock = unimodal_margin >= input.boundary_margin * 2.0
            && input.top1_coh >= self.calibrated_min_coherence * 1.20;

        if !self.is_calibrated {
            let foreign_ood_tokens =
                input.query_seq_len > 0 && input.unexplained_token_ratio >= self.u_foreign;
            if foreign_ood_tokens {
                return Boundary2DDecision {
                    region: Boundary2DRegion::Unobserved,
                    foreign_ood_tokens: true,
                    z_c,
                    z_u,
                    s_2d,
                    stage_top1_candidate_to_exterior: false,
                };
            }
            if bridge_eligible {
                let sub_threshold_bridged = !input.has_recalled_facts
                    && input.top1_coh >= self.calibrated_min_coherence
                    && (self.u_semi..self.u_foreign).contains(&input.unexplained_token_ratio);
                let semi_known_cutoff = if sharp_unimodal_lock {
                    self.u_semi + self.sharp_unimodal_semi_margin
                } else {
                    self.u_semi
                };
                let semi_known_high_unexplained = input.has_recalled_facts
                    && input.unexplained_token_ratio >= semi_known_cutoff
                    && input.top1_coh < 0.90;
                if sub_threshold_bridged {
                    return Boundary2DDecision {
                        region: Boundary2DRegion::Exterior,
                        foreign_ood_tokens: false,
                        z_c,
                        z_u,
                        s_2d,
                        stage_top1_candidate_to_exterior: true,
                    };
                }
                if semi_known_high_unexplained {
                    return Boundary2DDecision {
                        region: Boundary2DRegion::Exterior,
                        foreign_ood_tokens: false,
                        z_c,
                        z_u,
                        s_2d,
                        stage_top1_candidate_to_exterior: false,
                    };
                }
            }
            let region = if input.has_recalled_facts {
                Boundary2DRegion::SelfRef
            } else {
                Boundary2DRegion::Unobserved
            };
            return Boundary2DDecision {
                region,
                foreign_ood_tokens: false,
                z_c,
                z_u,
                s_2d,
                stage_top1_candidate_to_exterior: false,
            };
        }

        // Single-vector callers (`query_seq_len == 0`) do not observe token-level `unexp`;
        // preserve 1D coherence gating rather than fabricating a synthetic `z_u` from `unexp = 0`.
        if input.query_seq_len == 0 {
            let region =
                if input.has_recalled_facts && input.top1_coh >= self.calibrated_min_coherence {
                    Boundary2DRegion::SelfRef
                } else {
                    Boundary2DRegion::Unobserved
                };
            return Boundary2DDecision {
                region,
                foreign_ood_tokens: false,
                z_c,
                z_u,
                s_2d,
                stage_top1_candidate_to_exterior: false,
            };
        }

        // Calibrated 2D `(top1_coh, unexp)` classification: both axes must satisfy their
        // respective admissibility bounds (`unexp < u_foreign`, `top1_coh >= min_coherence`,
        // and `s_2d >= tau_unobs`) with no 1D bypass around foreign-OOD tokens or below-floor noise.
        let foreign_ood_tokens = input.unexplained_token_ratio >= self.u_foreign;
        if foreign_ood_tokens
            || input.top1_coh < self.calibrated_min_coherence
            || s_2d < self.tau_unobs
        {
            return Boundary2DDecision {
                region: Boundary2DRegion::Unobserved,
                foreign_ood_tokens,
                z_c,
                z_u,
                s_2d,
                stage_top1_candidate_to_exterior: false,
            };
        }

        let eff_u_semi = if sharp_unimodal_lock {
            (self.u_semi + self.sharp_unimodal_semi_margin).min(self.u_foreign - 0.02)
        } else {
            self.u_semi
        };
        let in_semi_unexp_band =
            (eff_u_semi..self.u_foreign).contains(&input.unexplained_token_ratio);
        let in_mid_s2d_unexp_dom =
            s_2d < self.tau_self && input.unexplained_token_ratio >= eff_u_semi * 0.85;

        if bridge_eligible && (in_semi_unexp_band || in_mid_s2d_unexp_dom) {
            if !input.has_recalled_facts {
                return Boundary2DDecision {
                    region: Boundary2DRegion::Exterior,
                    foreign_ood_tokens: false,
                    z_c,
                    z_u,
                    s_2d,
                    stage_top1_candidate_to_exterior: true,
                };
            }
            if input.top1_coh < 0.90 {
                return Boundary2DDecision {
                    region: Boundary2DRegion::Exterior,
                    foreign_ood_tokens: false,
                    z_c,
                    z_u,
                    s_2d,
                    stage_top1_candidate_to_exterior: false,
                };
            }
        }

        if s_2d >= self.tau_self
            && input.top1_coh >= self.calibrated_min_coherence
            && input.has_recalled_facts
        {
            Boundary2DDecision {
                region: Boundary2DRegion::SelfRef,
                foreign_ood_tokens: false,
                z_c,
                z_u,
                s_2d,
                stage_top1_candidate_to_exterior: false,
            }
        } else {
            Boundary2DDecision {
                region: Boundary2DRegion::Unobserved,
                foreign_ood_tokens: false,
                z_c,
                z_u,
                s_2d,
                stage_top1_candidate_to_exterior: false,
            }
        }
    }
}

/// Compact parameter struct passed into [`Boundary2DThresholds::classify`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Boundary2DInput {
    /// Highest Fact resonance score (`top1_coh`).
    pub top1_coh: f32,
    /// Second-highest Fact resonance score (`top2_coh`).
    pub top2_coh: f32,
    /// Policy boundary margin used for sharp-unimodal detection.
    pub boundary_margin: f32,
    /// SIF-weighted unexplained salient token fraction (`0.0..=1.0`).
    pub unexplained_token_ratio: f32,
    /// Effective query token length (`0` for single-vector callers without token channel).
    pub query_seq_len: usize,
    /// True if Fact candidates survived pre-2D recall selection.
    pub has_recalled_facts: bool,
    /// True if at least one Fact candidate exists in the corpus.
    pub has_top1_fact_candidate: bool,
    /// True when Layer 1 hard-block Guard repulsion (`RepulsionEscape`) is active.
    pub repulse_guard: bool,
    /// Guard conflict margin (`max_guard_score - top1_non_guard`).
    pub guard_conflict_margin: f32,
    /// True when [`liminika_core::liminal::LacunaPartition8::is_sandbox_eligible`] holds.
    pub is_sandbox_eligible: bool,
    /// Bridge granule count from [`liminika_core::liminal::LacunaPartition8`].
    pub bridge_count: u8,
}

/// Classification outcome emitted by [`Boundary2DThresholds::classify`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Boundary2DDecision {
    /// Selected epistemic region (`Unobserved`, `Exterior`, or `SelfRef`).
    pub region: Boundary2DRegion,
    /// True when foreign-OOD token gating rejected the query (suppresses Persona bleed too).
    pub foreign_ood_tokens: bool,
    /// Standardized coherence coordinate $z_c$.
    pub z_c: f32,
    /// Standardized unexplained-token coordinate $z_u$.
    pub z_u: f32,
    /// Combined 2D score $S_{2D} = 0.60 z_c + 0.40 z_u$.
    pub s_2d: f32,
    /// True when `Exterior` was reached from an empty `recalled_facts` via sub-threshold bridging.
    pub stage_top1_candidate_to_exterior: bool,
}
