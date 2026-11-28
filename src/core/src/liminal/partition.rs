// name: src/core/src/liminal/partition.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`EpistemicRegime`] classification and branchless 16-byte 4-state [`LacunaPartition8`]
//! ($S \cap O \setminus \bot$, $S \setminus O \setminus \bot$, $O \setminus S \setminus \bot$,
//! $\overline{S \cup O} \setminus \bot$, $\bot$) aligned with [`BitCut8`] and [`Lacuna`].

#![forbid(unsafe_code)]

use core::mem::{align_of, offset_of, size_of};

use crate::liminal::axiom::Lacuna;
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
    /// Query lies outside the carved memory horizon (boundary between settled state and unobserved region).
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

/// Scope region bit `0`: trigger on $S \cap O \setminus \bot$ (`self_ref_mask`).
pub const SCOPE_REGION_SELF_REF: u8 = 1 << 0;

/// Scope region bit `1`: trigger on $S \setminus O \setminus \bot$ (`interior_mask`).
pub const SCOPE_REGION_INTERIOR: u8 = 1 << 1;

/// Scope region bit `2`: trigger on validated $O \setminus S \setminus \bot$ (`exterior_mask`).
pub const SCOPE_REGION_EXTERIOR: u8 = 1 << 2;

/// Partition flag (`bit 0`): validated exterior granules require isolated non-copy sandbox staging.
pub const LACUNA_PARTITION_FLAG_SANDBOX_ELIGIBLE: u8 = 1 << 0;

/// Partition flag (`bit 1`): zero-aperture non-observation (`self_ref_mask == 0`, `exterior_mask == 0`,
/// `undefined_mask == 0`).
pub const LACUNA_PARTITION_FLAG_UNOBSERVED_APERTURE_ZERO: u8 = 1 << 1;

/// Partition flag (`bit 2`): at least one granule encountered a non-finite or Q0.7 bit-7 fault (`undefined_mask != 0`).
pub const LACUNA_PARTITION_FLAG_HAS_UNDEFINED_BOTTOM: u8 = 1 << 2;

/// Partition flag (`bit 3`): self-aligned (`self_ref_mask != 0`, `exterior_mask == 0`, `undefined_mask == 0`).
pub const LACUNA_PARTITION_FLAG_SELF_ALIGNED: u8 = 1 << 3;

/// Spreads the 8 bits of `m` into even bit positions (`0, 2, ..., 14`) of a `u16` without loops or branches.
#[inline(always)]
#[must_use]
const fn spread_bits_u8_to_u16(m: u8) -> u16 {
    let mut x = m as u16;
    x = (x | (x << 4)) & 0x0F0F;
    x = (x | (x << 2)) & 0x3333;
    x = (x | (x << 1)) & 0x5555;
    x
}

/// Packs disjoint 8-bit region masks into 8x 2-bit [`Lacuna`] codes (`bits 2g..=2g+1`):
/// - `0b00` (`Unobserved`): `unobserved_mask` ($\overline{S \cup O} \setminus \bot$) **and** `interior_mask`
///   ($S \setminus O \setminus \bot$)
/// - `0b01` (`SelfRef`): `self_ref_mask` ($S \cap O \setminus \bot$)
/// - `0b10` (`Exterior`): `exterior_mask` (validated $O \setminus S \setminus \bot$)
/// - `0b11` (`Undefined`): `undefined_mask` ($\bot$)
#[inline(always)]
#[must_use]
const fn pack_lacuna_codes_u16(self_ref_mask: u8, exterior_mask: u8, undefined_mask: u8) -> u16 {
    let lo_bits = self_ref_mask | undefined_mask;
    let hi_bits = exterior_mask | undefined_mask;
    spread_bits_u8_to_u16(lo_bits) | (spread_bits_u8_to_u16(hi_bits) << 1)
}

/// Branchless 16-byte (`#[repr(C, align(16))]`, zero implicit padding) 8-granule boundary partition
/// encoding the 5 mutually disjoint regions (`self_ref`, `interior`, `exterior`, `unobserved`,
/// `undefined`) together with their packed 2-bit [`Lacuna`] truth table.
///
/// Case 2 (`fault = 0`, `self = 1`, `observed = 0`, i.e., $S \setminus O$) populates
/// `interior_mask` and maps to `0b00` (`Lacuna::Unobserved`) in `packed_lacuna_u16` so unverified
/// interior state never bleeds into out-of-distribution triggers.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LacunaPartition8 {
    /// 8-granule cognitive boundary bitmask $S$ of self/active memory (`offset 0`).
    pub self_mask: u8,
    /// 8-granule noise-filtered external bitmask $O$ (`self_ref_mask | exterior_mask`, disjoint from $\bot$,
    /// `offset 1`).
    pub observed_mask: u8,
    /// $S \cap O \setminus \bot$: observed self-reference bridge granules (`0b01 = SelfRef`, `offset 2`).
    pub self_ref_mask: u8,
    /// $S \setminus O \setminus \bot$: protected self-specific interior granules (`0b00 = Unobserved`, `offset 3`).
    pub interior_mask: u8,
    /// Validated $O \setminus S \setminus \bot$: exterior observations routed to isolated sandbox (`0b10 =
    /// Exterior`, `offset 4`).
    pub exterior_mask: u8,
    /// $\overline{S \cup O} \setminus \bot$ (plus rejected isolated single-granule noise, `0b00 = Unobserved`,
    /// `offset 5`).
    pub unobserved_mask: u8,
    /// $\bot$: non-finite or Q0.7 `bit 7 != 0` fault granules (`0b11 = Undefined`, `offset 6`).
    pub undefined_mask: u8,
    /// Number of $S \cap O \setminus \bot$ bridge granules (`popcnt(self_ref_mask)`, `offset 7`).
    pub bridge_count: u8,
    /// Branchless $4p(1-p)$ boundary sensitivity in Q0.7 (`0..=127`, `0 fdiv`, `offset 8`).
    pub boundary_sensitivity_q7: u8,
    /// Local concentration in Q0.7 (`0..=127`) across exterior granules (`offset 9`).
    pub exterior_concentration_q7: u8,
    /// Mean excess Q0.7 density across validated $O \setminus S \setminus \bot$ granules (`offset 10`).
    pub exterior_density_q7: u8,
    /// Bitmask of `LACUNA_PARTITION_FLAG_*` summary flags (`offset 11`).
    pub partition_flags: u8,
    /// Packed 2-bit [`Lacuna`] codes across all 8 granules (lane `g` at bits `2g..=2g+1`, `offset 12..14`).
    pub packed_lacuna_u16: u16,
    /// Dominant [`Lacuna`] discriminant by priority `Undefined > Exterior > SelfRef > Unobserved` (`offset 14`).
    pub dominant_lacuna: u8,
    /// 2-bit [`AttributionTag`] (`Liminal`, `System` or `Ego`, `offset 15`).
    pub attribution_tag: u8,
}

const _: () = {
    assert!(size_of::<LacunaPartition8>() == 16);
    assert!(align_of::<LacunaPartition8>() == 16);
    assert!(offset_of!(LacunaPartition8, self_mask) == 0);
    assert!(offset_of!(LacunaPartition8, observed_mask) == 1);
    assert!(offset_of!(LacunaPartition8, self_ref_mask) == 2);
    assert!(offset_of!(LacunaPartition8, interior_mask) == 3);
    assert!(offset_of!(LacunaPartition8, exterior_mask) == 4);
    assert!(offset_of!(LacunaPartition8, unobserved_mask) == 5);
    assert!(offset_of!(LacunaPartition8, undefined_mask) == 6);
    assert!(offset_of!(LacunaPartition8, bridge_count) == 7);
    assert!(offset_of!(LacunaPartition8, boundary_sensitivity_q7) == 8);
    assert!(offset_of!(LacunaPartition8, exterior_concentration_q7) == 9);
    assert!(offset_of!(LacunaPartition8, exterior_density_q7) == 10);
    assert!(offset_of!(LacunaPartition8, partition_flags) == 11);
    assert!(offset_of!(LacunaPartition8, packed_lacuna_u16) == 12);
    assert!(offset_of!(LacunaPartition8, dominant_lacuna) == 14);
    assert!(offset_of!(LacunaPartition8, attribution_tag) == 15);
};

impl LacunaPartition8 {
    /// Returns `true` when zero-aperture non-observation holds (no `SelfRef`, no `Exterior`, no `Undefined`).
    #[inline(always)]
    #[must_use]
    pub const fn is_unobserved_aperture_zero(&self) -> bool {
        (self.partition_flags & LACUNA_PARTITION_FLAG_UNOBSERVED_APERTURE_ZERO) != 0
    }

    /// Returns `true` when shared bridge support exists with zero validated exterior gap and zero faults.
    #[inline(always)]
    #[must_use]
    pub const fn is_self_aligned(&self) -> bool {
        (self.partition_flags & LACUNA_PARTITION_FLAG_SELF_ALIGNED) != 0
    }

    /// Returns `true` when validated exterior granules ($O \setminus S \setminus \bot$) are present.
    #[inline(always)]
    #[must_use]
    pub const fn has_exterior_gap(&self) -> bool {
        self.exterior_mask != 0
    }

    /// Returns `true` when self-exclusive interior granules ($S \setminus O \setminus \bot$) are present.
    #[inline(always)]
    #[must_use]
    pub const fn has_interior_excess(&self) -> bool {
        self.interior_mask != 0
    }

    /// Returns `true` when any granule encountered a fault ($\bot$).
    #[inline(always)]
    #[must_use]
    pub const fn has_undefined_bottom(&self) -> bool {
        self.undefined_mask != 0
    }

    /// Returns `true` when exterior granules require isolated non-copy sandbox staging.
    #[inline(always)]
    #[must_use]
    pub const fn is_sandbox_eligible(&self) -> bool {
        (self.partition_flags & LACUNA_PARTITION_FLAG_SANDBOX_ELIGIBLE) != 0
    }

    /// Decodes the [`Lacuna`] state for granule lane `g` (`0..8`) from `packed_lacuna_u16`.
    #[inline(always)]
    #[must_use]
    pub const fn lane_lacuna(&self, g: usize) -> Lacuna {
        let shift = (g & 7) << 1;
        Lacuna::from_code(((self.packed_lacuna_u16 >> shift) & 0b11) as u8)
    }

    /// Decodes `dominant_lacuna` into [`Lacuna`].
    #[inline(always)]
    #[must_use]
    pub const fn dominant(&self) -> Lacuna {
        Lacuna::from_code(self.dominant_lacuna)
    }

    /// Returns the raw 4-quadrant atomic [`BitCut8`] refinement of `(self_mask, observed_mask)`
    /// restricted to the fault-free universe `!undefined_mask`:
    /// `((S & O \ bot, S \ O \ bot), (O \ S \ bot, !(S | O) \ bot))`.
    #[inline(always)]
    #[must_use]
    pub const fn quadrant_cuts(&self) -> (BitCut8, BitCut8) {
        BitCut8::split(!self.undefined_mask, self.self_mask).refine(self.observed_mask)
    }
}

/// Evaluates the branchless [`LacunaPartition8`] truth table with explicit `fault_mask` (`0 alloc`,
/// `0 fdiv`, `0` data-dependent branches).
///
/// Fault detection & validation rules:
/// - Any granule `g` with `(fault_mask >> g) & 1 != 0` OR Q0.7 `bit 7 != 0` in either
///   `other_granule_density_q7[g]` or `granule_noise_tolerance_q7[g]` enters `undefined_mask` ($\bot$)
///   with `0b11` (`Lacuna::Undefined`).
/// - Non-fault granules join raw observation only if `other_granule_density_q7[g] > granule_noise_tolerance_q7[g]`.
/// - Unbridged `O \ S` (`bridge_count == 0`) requires at least `2` active exterior granules AND
///   `exterior_concentration_q7 >= concentration_floor_q7`; otherwise isolated single-granule noise
///   drops into `unobserved_mask` with `exterior_mask = 0`.
#[inline]
#[must_use]
pub const fn evaluate_lacuna_partition_with_faults_branchless(
    self_mask: u8,
    other_granule_density_q7: &[u8; NUM_STIGMERGIC_GRANULES],
    granule_noise_tolerance_q7: &[u8; NUM_STIGMERGIC_GRANULES],
    concentration_floor_q7: u8,
    fault_mask: u8,
) -> LacunaPartition8 {
    let mut undefined_mask = 0u8;
    let mut raw_other_mask = 0u8;
    let mut g = 0usize;
    while g < NUM_STIGMERGIC_GRANULES {
        let bit = 1u8 << g;
        let q7_fault = ((other_granule_density_q7[g] | granule_noise_tolerance_q7[g]) & 0x80) != 0;
        let is_fault = (((fault_mask >> g) & 1) != 0) | q7_fault;
        let above = (!is_fault) & (other_granule_density_q7[g] > granule_noise_tolerance_q7[g]);
        undefined_mask |= branchless_select_u8(is_fault, bit, 0);
        raw_other_mask |= branchless_select_u8(above, bit, 0);
        g += 1;
    }

    let defined_universe = !undefined_mask;
    let self_cut = BitCut8::split(defined_universe, self_mask);
    let raw_other_cut = BitCut8::split(defined_universe, raw_other_mask);
    let self_ref_mask = self_cut.intersect(raw_other_cut);
    let interior_mask = self_cut.excess_over(raw_other_cut);
    let raw_ext_mask = raw_other_cut.excess_over(self_cut);
    let bridge_count = self_ref_mask.count_ones() as u8;
    let ext_cnt = raw_ext_mask.count_ones() as u8;

    let mut peak_excess = 0u8;
    let mut sum_excess = 0u16;
    let mut g2 = 0usize;
    while g2 < NUM_STIGMERGIC_GRANULES {
        let in_ext = ((raw_ext_mask >> g2) & 1) != 0;
        let ex = other_granule_density_q7[g2].saturating_sub(granule_noise_tolerance_q7[g2]);
        let eff_ex = branchless_select_u8(in_ext, ex, 0);
        peak_excess = branchless_select_u8(eff_ex > peak_excess, eff_ex, peak_excess);
        sum_excess = sum_excess.wrapping_add(eff_ex as u16);
        g2 += 1;
    }

    // Branchless fixed reciprocal multiply by `1/k` scaled by 128 (`>> 7`) to avoid hardware `div`
    const INV_CNT_Q7: [u16; 9] = [0, 128, 64, 43, 32, 26, 21, 18, 16];
    let ext_idx = if ext_cnt > 8 { 8 } else { ext_cnt } as usize;
    let mean_raw = (sum_excess * INV_CNT_Q7[ext_idx]) >> 7;
    let mean_ext_excess = if mean_raw > 127 { 127 } else { mean_raw as u8 };

    let half_sum = sum_excess >> 1;
    let half_sum_q7 = if half_sum > 127 { 127 } else { half_sum as u8 };
    let peak_q7 = if peak_excess > 127 { 127 } else { peak_excess };
    let unbridged_conc = if half_sum_q7 > peak_q7 {
        half_sum_q7
    } else {
        peak_q7
    };
    let bridged_conc = if peak_q7 > mean_ext_excess {
        peak_q7
    } else {
        mean_ext_excess
    };
    let exterior_concentration_q7 = branchless_select_u8(
        bridge_count >= 1,
        bridged_conc,
        branchless_select_u8(ext_cnt >= 2, unbridged_conc, 0),
    );

    let ext_valid = (raw_ext_mask != 0)
        & ((bridge_count >= 1) | (exterior_concentration_q7 >= concentration_floor_q7));
    let exterior_mask = branchless_select_u8(ext_valid, raw_ext_mask, 0);
    let exterior_density_q7 = branchless_select_u8(ext_valid, mean_ext_excess, 0);
    let observed_mask = self_ref_mask | exterior_mask;

    let observed_cut = BitCut8::split(defined_universe, observed_mask);
    let unobserved_mask = self_cut.blindspot_mask(observed_cut);

    // 4p(1-p) in Q0.7 (`0 fdiv`): p_q7 = active_cnt * 16 in [0..=128], sensitivity = (p_q7 * (128 - p_q7)) >> 5
    let active_other_cnt = observed_mask.count_ones();
    let p_raw = active_other_cnt << 4;
    let p_q7 = if p_raw > 128 { 128 } else { p_raw };
    let sens_raw = (p_q7.saturating_mul(128u32.saturating_sub(p_q7))) >> 5;
    let boundary_sensitivity_q7 = if sens_raw > 127 { 127 } else { sens_raw as u8 };

    let sandbox_eligible = exterior_mask != 0;
    let unobserved_aperture_zero =
        (self_ref_mask == 0) & (exterior_mask == 0) & (undefined_mask == 0);
    let has_undefined = undefined_mask != 0;
    let self_aligned = (self_ref_mask != 0) & (exterior_mask == 0) & (undefined_mask == 0);

    let partition_flags =
        branchless_select_u8(sandbox_eligible, LACUNA_PARTITION_FLAG_SANDBOX_ELIGIBLE, 0)
            | branchless_select_u8(
                unobserved_aperture_zero,
                LACUNA_PARTITION_FLAG_UNOBSERVED_APERTURE_ZERO,
                0,
            )
            | branchless_select_u8(has_undefined, LACUNA_PARTITION_FLAG_HAS_UNDEFINED_BOTTOM, 0)
            | branchless_select_u8(self_aligned, LACUNA_PARTITION_FLAG_SELF_ALIGNED, 0);

    let packed_lacuna_u16 = pack_lacuna_codes_u16(self_ref_mask, exterior_mask, undefined_mask);

    let mut dominant_lacuna = Lacuna::Unobserved.code();
    dominant_lacuna =
        branchless_select_u8(self_ref_mask != 0, Lacuna::SelfRef.code(), dominant_lacuna);
    dominant_lacuna =
        branchless_select_u8(exterior_mask != 0, Lacuna::Exterior.code(), dominant_lacuna);
    dominant_lacuna =
        branchless_select_u8(has_undefined, Lacuna::Undefined.code(), dominant_lacuna);

    LacunaPartition8 {
        self_mask,
        observed_mask,
        self_ref_mask,
        interior_mask,
        exterior_mask,
        unobserved_mask,
        undefined_mask,
        bridge_count,
        boundary_sensitivity_q7,
        exterior_concentration_q7,
        exterior_density_q7,
        partition_flags,
        packed_lacuna_u16,
        dominant_lacuna,
        attribution_tag: AttributionTag::LIMINAL_SYSTEM.raw(),
    }
}

/// Evaluates the branchless [`LacunaPartition8`] truth table across 8 SWAR granules, automatically
/// classifying any Q0.7 `bit 7 != 0` input entry into `undefined_mask` ($\bot$).
#[inline]
#[must_use]
pub const fn evaluate_lacuna_partition_branchless(
    self_mask: u8,
    other_granule_density_q7: &[u8; NUM_STIGMERGIC_GRANULES],
    granule_noise_tolerance_q7: &[u8; NUM_STIGMERGIC_GRANULES],
    concentration_floor_q7: u8,
) -> LacunaPartition8 {
    evaluate_lacuna_partition_with_faults_branchless(
        self_mask,
        other_granule_density_q7,
        granule_noise_tolerance_q7,
        concentration_floor_q7,
        0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_partition_bitwise_invariants(p: &LacunaPartition8) {
        let masks = [
            p.self_ref_mask,
            p.interior_mask,
            p.exterior_mask,
            p.unobserved_mask,
            p.undefined_mask,
        ];
        // 1. Pairwise disjointness among all 5 region masks.
        for i in 0..masks.len() {
            for j in (i + 1)..masks.len() {
                assert_eq!(masks[i] & masks[j], 0, "masks {i} and {j} overlap");
            }
        }
        // 2. Exact covering of all 8 granules.
        assert_eq!(
            p.self_ref_mask
                | p.interior_mask
                | p.exterior_mask
                | p.unobserved_mask
                | p.undefined_mask,
            0xFF
        );
        // 3. Exact decomposition of `self_mask` and `observed_mask` outside `undefined_mask`.
        assert_eq!(
            p.self_ref_mask | p.interior_mask,
            p.self_mask & !p.undefined_mask
        );
        assert_eq!(p.self_ref_mask | p.exterior_mask, p.observed_mask);
        assert_eq!(p.observed_mask & p.undefined_mask, 0);
        // 4. Branchless bijection between `(packed_lacuna_u16, self_mask)` and all 5 masks.
        let mut rec_self_ref = 0u8;
        let mut rec_interior = 0u8;
        let mut rec_exterior = 0u8;
        let mut rec_unobserved = 0u8;
        let mut rec_undefined = 0u8;
        for g in 0..NUM_STIGMERGIC_GRANULES {
            let bit = 1u8 << g;
            let in_self = (p.self_mask & bit) != 0;
            match p.lane_lacuna(g) {
                Lacuna::Unobserved => {
                    if in_self {
                        rec_interior |= bit;
                    } else {
                        rec_unobserved |= bit;
                    }
                }
                Lacuna::SelfRef => rec_self_ref |= bit,
                Lacuna::Exterior => rec_exterior |= bit,
                Lacuna::Undefined => rec_undefined |= bit,
            }
        }
        assert_eq!(rec_self_ref, p.self_ref_mask);
        assert_eq!(rec_interior, p.interior_mask);
        assert_eq!(rec_exterior, p.exterior_mask);
        assert_eq!(rec_unobserved, p.unobserved_mask);
        assert_eq!(rec_undefined, p.undefined_mask);
    }

    #[test]
    fn test_lacuna_partition8_five_cases_and_noise_deadband() {
        let noise_tol = [10u8; NUM_STIGMERGIC_GRANULES];

        // 1. Isolated single-granule OOD noise collapses to Unobserved (`exterior_mask == 0`),
        //    preserving Case 2 (`S \ O` -> `interior_mask` + `Lacuna::Unobserved`).
        let mut single_noise = [0u8; NUM_STIGMERGIC_GRANULES];
        single_noise[5] = 28;
        let lp_noise = evaluate_lacuna_partition_branchless(0x03, &single_noise, &noise_tol, 40);
        assert_partition_bitwise_invariants(&lp_noise);
        assert_eq!(lp_noise.self_ref_mask, 0);
        assert_eq!(lp_noise.interior_mask, 0x03);
        assert_eq!(lp_noise.exterior_mask, 0);
        assert_eq!(lp_noise.unobserved_mask, !0x03);
        assert_eq!(lp_noise.undefined_mask, 0);
        assert_eq!(lp_noise.dominant(), Lacuna::Unobserved);
        assert_eq!(lp_noise.lane_lacuna(0), Lacuna::Unobserved); // Case 2: F=0, S=1, O=0
        assert_eq!(lp_noise.lane_lacuna(7), Lacuna::Unobserved); // Case 1: F=0, S=0, O=0
        assert!(lp_noise.is_unobserved_aperture_zero());
        assert!(!lp_noise.is_self_aligned());
        assert!(!lp_noise.has_exterior_gap());
        assert!(lp_noise.has_interior_excess());
        assert!(!lp_noise.has_undefined_bottom());

        // 2. Bridged `S & O` + validated `O \ S` across all non-fault cases.
        let bridged = [80, 72, 0, 0, 92, 88, 0, 0];
        let lp_ext = evaluate_lacuna_partition_branchless(0x07, &bridged, &noise_tol, 40);
        assert_partition_bitwise_invariants(&lp_ext);
        assert_eq!(lp_ext.self_ref_mask, 0x03);
        assert_eq!(lp_ext.interior_mask, 0x04);
        assert_eq!(lp_ext.exterior_mask, 0x30);
        assert_eq!(lp_ext.unobserved_mask, 0xC8);
        assert_eq!(lp_ext.bridge_count, 2);
        assert!(lp_ext.is_sandbox_eligible());
        assert!(!lp_ext.is_unobserved_aperture_zero());
        assert!(!lp_ext.is_self_aligned());
        assert!(lp_ext.has_exterior_gap());
        assert!(lp_ext.has_interior_excess());
        assert_eq!(lp_ext.dominant(), Lacuna::Exterior);
        assert_eq!(lp_ext.lane_lacuna(0), Lacuna::SelfRef); // Case 3
        assert_eq!(lp_ext.lane_lacuna(2), Lacuna::Unobserved); // Case 2
        assert_eq!(lp_ext.lane_lacuna(4), Lacuna::Exterior); // Case 4
        assert_eq!(lp_ext.lane_lacuna(7), Lacuna::Unobserved); // Case 1
        assert_eq!(lp_ext.boundary_sensitivity_q7, 127);

        let (in_self, out_self) = lp_ext.quadrant_cuts();
        assert_eq!(in_self.inside, lp_ext.self_ref_mask);
        assert_eq!(in_self.outside, lp_ext.interior_mask);
        assert_eq!(out_self.inside, lp_ext.exterior_mask);
        assert_eq!(out_self.outside, lp_ext.unobserved_mask);

        // 3. 8/8 saturation -> p = 1.0 -> 4p(1-p) == 0 boundary sensitivity, self-aligned.
        let full = [90u8; NUM_STIGMERGIC_GRANULES];
        let lp_full = evaluate_lacuna_partition_branchless(0xFF, &full, &noise_tol, 40);
        assert_partition_bitwise_invariants(&lp_full);
        assert_eq!(lp_full.boundary_sensitivity_q7, 0);
        assert!(lp_full.is_self_aligned());
        assert_eq!(lp_full.dominant(), Lacuna::SelfRef);

        // 4. Case 5: Q0.7 bit 7 fault (`0x80`..`0xFF`) or explicit `fault_mask` enters `undefined_mask`.
        let mut faulty = [80, 72, 0x80, 0, 92, 88, 0, 0];
        faulty[6] = 0xFF;
        let lp_fault = evaluate_lacuna_partition_with_faults_branchless(
            0x0F,
            &faulty,
            &noise_tol,
            40,
            0b1000_0000,
        );
        assert_partition_bitwise_invariants(&lp_fault);
        assert_eq!(lp_fault.undefined_mask, 0b1100_0100);
        assert!(lp_fault.has_undefined_bottom());
        assert!(!lp_fault.is_unobserved_aperture_zero());
        assert!(!lp_fault.is_self_aligned());
        assert_eq!(lp_fault.dominant(), Lacuna::Undefined);
        assert_eq!(lp_fault.lane_lacuna(2), Lacuna::Undefined);
        assert_eq!(lp_fault.lane_lacuna(6), Lacuna::Undefined);
        assert_eq!(lp_fault.lane_lacuna(7), Lacuna::Undefined);
    }
}
