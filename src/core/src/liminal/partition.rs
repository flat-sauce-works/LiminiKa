// name: src/core/src/liminal/partition.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`EpistemicRegime`] classification and branchless 16-byte 4-region Venn epistemic partitioning
//! ($S \setminus O$, $S \cap O$, $O \setminus S$, $\overline{S \cup O}$) aligned with [`BitCut8`].

#![forbid(unsafe_code)]

use crate::liminal::bitcut::BitCut8;
use crate::liminal::constants::{AttributionTag, NUM_STIGMERGIC_GRANULES};
use crate::liminal::select::branchless_select_u8;

/// Epistemic regime classification separating internal memory conflict from out-of-memory unknowability.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicRegime {
    /// Query falls inside the carved memory horizon with coherent dominant alignment ($H^0$).
    KnownCoherent = 0,
    /// Query falls inside the carved memory horizon with competing / opposing attractors ($H^1 \neq 0$).
    InternalConflict = 1,
    /// Query lies outside the carved memory horizon (*Limina* between cognition and the unknowable).
    Liminal = 2,
}

impl EpistemicRegime {
    /// Decodes a compact `u8` regime code (`0`, `1`, `2`) into [`EpistemicRegime`].
    #[inline(always)]
    #[must_use]
    pub const fn from_code(code: u8) -> Self {
        match code {
            1 => Self::InternalConflict,
            2 => Self::Liminal,
            _ => Self::KnownCoherent,
        }
    }
}

/// Scope region bit `0`: trigger on $S \cap O$ (`shared_core_mask`).
pub const SCOPE_REGION_SHARED_CORE: u8 = 1 << 0;

/// Scope region bit `1`: trigger on $S \setminus O$ (`self_exclusive_mask`).
pub const SCOPE_REGION_SELF_EXCLUSIVE: u8 = 1 << 1;

/// Scope region bit `2`: trigger on validated $O \setminus S$ (`known_unknown_mask`).
pub const SCOPE_REGION_KNOWN_UNKNOWN: u8 = 1 << 2;

/// Branchless 16-byte (`#[repr(C, align(16))]`) 4-region Venn partition across 8 SWAR granules
/// separating $S \cap O$ (shared ground), $S \setminus O$ (self-exclusive gauge origin),
/// validated $O \setminus S$ (known unknown eligible only for non-copy sandbox routing), and
/// $\overline{S \cup O}$ (True Void structural threshold 0).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EpistemicPartition {
    /// 8-granule cognitive boundary bitmask $S$ of self/active memory (`offset 0`).
    pub self_mask: u8,
    /// 8-granule noise-filtered external bitmask $O$ (`density > noise_tolerance`, `offset 1`).
    pub other_mask: u8,
    /// $S \cap O$: shared common ground eligible for steady shortcut routing (`offset 2`).
    pub shared_core_mask: u8,
    /// $S \setminus O$: protected self-specific domain / gauge origin (`offset 3`).
    pub self_exclusive_mask: u8,
    /// Validated $O \setminus S$: known unknown routed strictly to isolated sandbox (`offset 4`).
    pub known_unknown_mask: u8,
    /// $\overline{S \cup O}$ (plus rejected isolated single-granule noise): True Void (`offset 5`).
    pub true_void_mask: u8,
    /// Number of $S \cap O$ bridge granules (`popcnt(shared_core_mask)`, `offset 6`).
    pub bridge_count: u8,
    /// Local concentration in Q0.7 (`0..=127`) across exterior granules (`offset 7`).
    pub other_concentration_q7: u8,
    /// Branchless $4p(1-p)$ boundary sensitivity in Q0.7 (`0..=127`, `0 fdiv`, `offset 8`).
    pub boundary_sensitivity_q7: u8,
    /// Mean excess Q0.7 density across validated $O \setminus S$ granules (`offset 9`).
    pub known_unknown_density_q7: u8,
    /// `1` when validated $O \setminus S$ exterior pointers require isolated sandbox staging (`offset 10`).
    pub sandbox_eligible: u8,
    /// `1` when both $S \cap O$ and validated $O \setminus S$ are empty (zero-threshold Void lock, `offset 11`).
    pub true_void_aperture_zero: u8,
    /// 2-bit [`AttributionTag`] (`Liminal`, `System` or `Ego`, `offset 12`).
    pub attribution_tag: u8,
    /// Explicit C alignment padding (`offset 13..16`).
    pub reserved_align: [u8; 3],
}

impl EpistemicPartition {
    /// Returns `true` when both shared bridge and validated exterior regions are empty
    /// (zero-aperture non-observation).
    #[inline(always)]
    #[must_use]
    pub const fn is_void(&self) -> bool {
        self.true_void_aperture_zero != 0
    }

    /// Returns `true` when shared bridge support exists with zero validated exterior gap.
    #[inline(always)]
    #[must_use]
    pub const fn is_self_aligned(&self) -> bool {
        (self.shared_core_mask != 0) & (self.known_unknown_mask == 0)
    }

    /// Returns `true` when validated exterior granules ($O \setminus S$) are present.
    #[inline(always)]
    #[must_use]
    pub const fn has_exterior_gap(&self) -> bool {
        self.known_unknown_mask != 0
    }

    /// Returns `true` when self-exclusive interior granules ($S \setminus O$) are present.
    #[inline(always)]
    #[must_use]
    pub const fn has_interior_excess(&self) -> bool {
        self.self_exclusive_mask != 0
    }

    /// Returns the raw 4-quadrant atomic [`BitCut8`] refinement of `(self_mask, other_mask)`:
    /// `((S & O, S & !O), (!S & O, !S & !O))`.
    #[inline(always)]
    #[must_use]
    pub const fn quadrant_cuts(&self) -> (BitCut8, BitCut8) {
        BitCut8::split(0xFF, self.self_mask).refine(self.other_mask)
    }
}

/// Evaluates the branchless 4-region Venn partition across 8 SWAR granules (`0 alloc`, `0 fdiv`,
/// `0` data-dependent branches).
///
/// Validation rule for $O \setminus S$ (Known Unknown):
/// - Each granule `g` joins `other_mask` only if `other_granule_density_q7[g] > granule_noise_tolerance_q7[g]`.
/// - Unbridged `O \ S` (`bridge_count == 0`) requires at least `2` active exterior granules AND
///   `other_concentration_q7 >= concentration_floor_q7`; otherwise isolated single-granule noise
///   drops into `true_void_mask` ($\overline{S \cup O}$) with `known_unknown_mask = 0`.
#[inline]
#[must_use]
pub fn evaluate_epistemic_partition_branchless(
    self_mask: u8,
    other_granule_density_q7: &[u8; NUM_STIGMERGIC_GRANULES],
    granule_noise_tolerance_q7: &[u8; NUM_STIGMERGIC_GRANULES],
    concentration_floor_q7: u8,
) -> EpistemicPartition {
    let mut other_mask = 0u8;
    let mut g = 0usize;
    while g < NUM_STIGMERGIC_GRANULES {
        let bit = 1u8 << g;
        let above = other_granule_density_q7[g] > granule_noise_tolerance_q7[g];
        other_mask |= branchless_select_u8(above, bit, 0);
        g += 1;
    }

    let self_cut = BitCut8::split(0xFF, self_mask);
    let other_cut = BitCut8::split(0xFF, other_mask);
    let shared_core_mask = self_cut.intersect(other_cut);
    let self_exclusive_mask = self_cut.excess_over(other_cut);
    let raw_ext_mask = other_cut.excess_over(self_cut);
    let bridge_count = shared_core_mask.count_ones() as u8;
    let ext_cnt = raw_ext_mask.count_ones() as u8;

    let mut peak_excess = 0u8;
    let mut sum_excess = 0u16;
    let mut g2 = 0usize;
    while g2 < NUM_STIGMERGIC_GRANULES {
        let in_ext = ((raw_ext_mask >> g2) & 1) != 0;
        let ex = other_granule_density_q7[g2].saturating_sub(granule_noise_tolerance_q7[g2]);
        let eff_ex = branchless_select_u8(in_ext, ex, 0);
        peak_excess = branchless_select_u8(eff_ex > peak_excess, eff_ex, peak_excess);
        sum_excess = sum_excess.wrapping_add(u16::from(eff_ex));
        g2 += 1;
    }

    // Branchless fixed reciprocal multiply by `1/k` scaled by 128 (`>> 7`) to avoid hardware `div`
    const INV_CNT_Q7: [u16; 9] = [0, 128, 64, 43, 32, 26, 21, 18, 16];
    let mean_ext_excess =
        (((sum_excess * INV_CNT_Q7[usize::from(ext_cnt.min(8))]) >> 7).min(127)) as u8;

    let unbridged_conc = ((sum_excess >> 1).min(127) as u8).max(peak_excess.min(127));
    let other_concentration_q7 = branchless_select_u8(
        bridge_count >= 1,
        peak_excess.min(127).max(mean_ext_excess),
        branchless_select_u8(ext_cnt >= 2, unbridged_conc, 0),
    );

    let ext_valid = (raw_ext_mask != 0)
        & ((bridge_count >= 1) | (other_concentration_q7 >= concentration_floor_q7));
    let known_unknown_mask = branchless_select_u8(ext_valid, raw_ext_mask, 0);
    let known_unknown_density_q7 = branchless_select_u8(ext_valid, mean_ext_excess, 0);

    let valid_other_cut = BitCut8::split(0xFF, shared_core_mask | known_unknown_mask);
    let true_void_mask = self_cut.blindspot_mask(valid_other_cut);

    // 4p(1-p) in Q0.7 (`0 fdiv`): p_q7 = active_cnt * 16 in [0..=128], sensitivity = (p_q7 * (128 - p_q7)) >> 5
    let active_other_cnt = (shared_core_mask | known_unknown_mask).count_ones();
    let p_q7 = (active_other_cnt << 4).min(128);
    let sens_raw = (p_q7.saturating_mul(128u32.saturating_sub(p_q7))) >> 5;
    let boundary_sensitivity_q7 = sens_raw.min(127) as u8;

    let sandbox_eligible = u8::from(known_unknown_mask != 0);
    let true_void_aperture_zero = u8::from((shared_core_mask == 0) & (known_unknown_mask == 0));

    EpistemicPartition {
        self_mask,
        other_mask,
        shared_core_mask,
        self_exclusive_mask,
        known_unknown_mask,
        true_void_mask,
        bridge_count,
        other_concentration_q7,
        boundary_sensitivity_q7,
        known_unknown_density_q7,
        sandbox_eligible,
        true_void_aperture_zero,
        attribution_tag: AttributionTag::LIMINAL_SYSTEM.raw(),
        reserved_align: [0; 3],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epistemic_partition_four_regions_and_noise_deadband() {
        let noise_tol = [10u8; NUM_STIGMERGIC_GRANULES];

        // 1. Isolated single-granule OOD noise collapses to True Void (`known_unknown_mask == 0`).
        let mut single_noise = [0u8; NUM_STIGMERGIC_GRANULES];
        single_noise[5] = 28;
        let vp_noise = evaluate_epistemic_partition_branchless(0x03, &single_noise, &noise_tol, 40);
        assert_eq!(vp_noise.shared_core_mask, 0);
        assert_eq!(vp_noise.self_exclusive_mask, 0x03);
        assert_eq!(vp_noise.known_unknown_mask, 0);
        assert_eq!(vp_noise.true_void_mask, !0x03);
        assert_eq!(vp_noise.true_void_aperture_zero, 1);
        assert_eq!(vp_noise.sandbox_eligible, 0);
        assert_eq!(vp_noise.boundary_sensitivity_q7, 0);
        assert!(vp_noise.is_void());
        assert!(!vp_noise.is_self_aligned());
        assert!(!vp_noise.has_exterior_gap());
        assert!(vp_noise.has_interior_excess());

        // 2. Bridged `S & O` + `O \ S` stages validated Known Unknown into sandbox without copy.
        let bridged = [80, 72, 0, 0, 92, 88, 0, 0];
        let vp_ext = evaluate_epistemic_partition_branchless(0x03, &bridged, &noise_tol, 40);
        assert_eq!(vp_ext.shared_core_mask, 0x03);
        assert_eq!(vp_ext.self_exclusive_mask, 0);
        assert_eq!(vp_ext.known_unknown_mask, 0x30);
        assert_eq!(vp_ext.bridge_count, 2);
        assert_eq!(vp_ext.sandbox_eligible, 1);
        assert_eq!(vp_ext.true_void_aperture_zero, 0);
        assert!(!vp_ext.is_void());
        assert!(!vp_ext.is_self_aligned());
        assert!(vp_ext.has_exterior_gap());
        assert!(!vp_ext.has_interior_excess());
        // 4/8 active granules -> 4 * 0.5 * 0.5 = 1.0 -> 127 in Q0.7.
        assert_eq!(vp_ext.boundary_sensitivity_q7, 127);
        let (in_self, out_self) = vp_ext.quadrant_cuts();
        assert_eq!(in_self.inside, vp_ext.shared_core_mask);
        assert_eq!(in_self.outside, vp_ext.self_exclusive_mask);
        assert_eq!(out_self.inside, vp_ext.known_unknown_mask);
        assert_eq!(out_self.outside, vp_ext.true_void_mask);

        // 3. 8/8 saturation -> p = 1.0 -> 4p(1-p) == 0 boundary sensitivity, self-aligned.
        let full = [90u8; NUM_STIGMERGIC_GRANULES];
        let vp_full = evaluate_epistemic_partition_branchless(0xFF, &full, &noise_tol, 40);
        assert_eq!(vp_full.boundary_sensitivity_q7, 0);
        assert!(vp_full.is_self_aligned());
    }
}
