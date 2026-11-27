// name: src/core/src/liminal/boundary.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Domain-agnostic 32-byte boundary signal (`BoundarySignal`) carrying an 8-granule
//! [`LacunaPartition8`], normalized Q0.15 pressure/delta readings, Q0.7 contraction/chatter
//! indicators, and structural [`AttributionTag`] provenance.
//!
//! Strictly `#![forbid(unsafe_code)]`, depending only on `core` and non-axiom `liminal` modules.

#![forbid(unsafe_code)]

use core::mem::{align_of, offset_of, size_of};

use crate::liminal::constants::AttributionTag;
use crate::liminal::partition::LacunaPartition8;
use crate::liminal::select::branchless_select_u8;

/// Signal flag (`bit 0`): observation channel produced a valid non-empty reading.
pub const BOUNDARY_SIGNAL_FLAG_OBSERVED: u8 = 1 << 0;
/// Signal flag (`bit 1`): non-finite measurement or `Undefined` fault detected (`fail-closed`).
pub const BOUNDARY_SIGNAL_FLAG_FAULT: u8 = 1 << 1;
/// Signal flag (`bit 2`): validated exterior excursion or repulsion alert active.
pub const BOUNDARY_SIGNAL_FLAG_EXTERIOR_ALERT: u8 = 1 << 2;
/// Signal flag (`bit 3`): stationary self-aligned state eligible for low-cost bypass.
pub const BOUNDARY_SIGNAL_FLAG_STATIONARY_READY: u8 = 1 << 3;
/// Signal flag (`bit 4`): irreversible transition intent requiring strict pin preservation.
pub const BOUNDARY_SIGNAL_FLAG_IRREVERSIBLE_INTENT: u8 = 1 << 4;
/// Bitmask of all defined `BOUNDARY_SIGNAL_FLAG_*` bits (`0x1F`).
pub const BOUNDARY_SIGNAL_KNOWN_FLAGS: u8 = BOUNDARY_SIGNAL_FLAG_OBSERVED
    | BOUNDARY_SIGNAL_FLAG_FAULT
    | BOUNDARY_SIGNAL_FLAG_EXTERIOR_ALERT
    | BOUNDARY_SIGNAL_FLAG_STATIONARY_READY
    | BOUNDARY_SIGNAL_FLAG_IRREVERSIBLE_INTENT;

/// Canonical `Unobserved` discriminant mirroring `Lacuna::Unobserved.code()` (`0`).
const LACUNA_CODE_UNOBSERVED: u8 = 0;
/// Canonical `SelfRef` discriminant mirroring `Lacuna::SelfRef.code()` (`1`).
const LACUNA_CODE_SELF_REF: u8 = 1;
/// Canonical `Exterior` discriminant mirroring `Lacuna::Exterior.code()` (`2`).
const LACUNA_CODE_EXTERIOR: u8 = 2;
/// Canonical `Undefined` fault discriminant mirroring `Lacuna::Undefined.code()` (`3`).
const LACUNA_CODE_UNDEFINED: u8 = 3;

/// Portable 32-byte (`#[repr(C, align(32))]`, zero implicit padding) domain-agnostic boundary
/// observation packet bridging Lv.0 partitions, Lv.2 anchors, Lv.3 drivers, and Lv.4 orchestration.
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundarySignal {
    /// 16-byte 8-granule 4-state boundary partition (`offset 0..16`).
    pub partition: LacunaPartition8,
    /// Normalized system/signal pressure in Q0.15 (`0..=32768`, `offset 16..18`).
    pub pressure_q15: u16,
    /// Normalized deviation/dispersion delta in Q0.15 (`0..=32768`, `offset 18..20`).
    pub delta_q15: u16,
    /// Packed 3-bit [`AttributionTag`] (`2 Primitives x 3 Drivers`, `0..=5`, `offset 20`).
    pub attribution_tag: u8,
    /// Requested or observed aperture contraction scale in Q0.7 (`0..=128`, `128 = 1.0`, `offset 21`).
    pub contraction_q7: u8,
    /// Observed boundary chatter intensity in Q0.7 (`0..=128`, `offset 22`).
    pub chatter_q7: u8,
    /// Bitwise OR of `BOUNDARY_SIGNAL_FLAG_*` summary flags (`offset 23`).
    pub signal_flags: u8,
    /// Dominant 3-bit working lane index (`0..=7`, `offset 24`).
    pub dominant_lane: u8,
    /// Primary root-cause classification code (`0` = nominal, `offset 25`).
    pub root_cause_code: u8,
    /// Explicit zero-initialized extension space (`offset 26..28`).
    pub reserved: [u8; 2],
    /// Monotonic logical step/epoch identifier (`offset 28..32`).
    pub epoch_id: u32,
}

const _: () = {
    assert!(size_of::<BoundarySignal>() == 32);
    assert!(align_of::<BoundarySignal>() == 32);
    assert!(offset_of!(BoundarySignal, partition) == 0);
    assert!(offset_of!(BoundarySignal, pressure_q15) == 16);
    assert!(offset_of!(BoundarySignal, delta_q15) == 18);
    assert!(offset_of!(BoundarySignal, attribution_tag) == 20);
    assert!(offset_of!(BoundarySignal, contraction_q7) == 21);
    assert!(offset_of!(BoundarySignal, chatter_q7) == 22);
    assert!(offset_of!(BoundarySignal, signal_flags) == 23);
    assert!(offset_of!(BoundarySignal, dominant_lane) == 24);
    assert!(offset_of!(BoundarySignal, root_cause_code) == 25);
    assert!(offset_of!(BoundarySignal, reserved) == 26);
    assert!(offset_of!(BoundarySignal, epoch_id) == 28);
};

impl Default for BoundarySignal {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl BoundarySignal {
    /// Constructs a neutral unobserved 32-byte [`BoundarySignal`] with unit contraction (`128`).
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            partition: LacunaPartition8 {
                self_mask: 0,
                observed_mask: 0,
                self_ref_mask: 0,
                interior_mask: 0,
                exterior_mask: 0,
                unobserved_mask: 0,
                undefined_mask: 0,
                bridge_count: 0,
                boundary_sensitivity_q7: 0,
                exterior_concentration_q7: 0,
                exterior_density_q7: 0,
                partition_flags: 0,
                packed_lacuna_u16: 0,
                dominant_lacuna: 0,
                attribution_tag: AttributionTag::LIMINAL_SYSTEM.raw(),
            },
            pressure_q15: 0,
            delta_q15: 0,
            attribution_tag: AttributionTag::LIMINAL_SYSTEM.raw(),
            contraction_q7: 128,
            chatter_q7: 0,
            signal_flags: 0,
            dominant_lane: 0,
            root_cause_code: 0,
            reserved: [0; 2],
            epoch_id: 0,
        }
    }

    /// Projects a [`LacunaPartition8`] and scalar boundary metrics into a 32-byte [`BoundarySignal`]
    /// strictly branchlessly (`0 alloc`, `0 fdiv`).
    #[inline(always)]
    #[must_use]
    pub const fn from_partition(
        partition: LacunaPartition8,
        pressure_q15: u16,
        delta_q15: u16,
        chatter_q7: u8,
        epoch_id: u32,
    ) -> Self {
        let is_fault =
            partition.has_undefined_bottom() | (partition.dominant_lacuna == LACUNA_CODE_UNDEFINED);
        let is_obs = (!is_fault) & (partition.observed_mask != 0);
        let is_ext = (!is_fault) & partition.has_exterior_gap();
        let is_stat = (!is_fault) & partition.is_self_aligned();
        let signal_flags = branchless_select_u8(is_obs, BOUNDARY_SIGNAL_FLAG_OBSERVED, 0)
            | branchless_select_u8(is_fault, BOUNDARY_SIGNAL_FLAG_FAULT, 0)
            | branchless_select_u8(is_ext, BOUNDARY_SIGNAL_FLAG_EXTERIOR_ALERT, 0)
            | branchless_select_u8(is_stat, BOUNDARY_SIGNAL_FLAG_STATIONARY_READY, 0);
        let clamped_chatter = if chatter_q7 > 128 { 128 } else { chatter_q7 };

        Self {
            partition,
            pressure_q15,
            delta_q15,
            attribution_tag: partition.attribution_tag & 0x07,
            contraction_q7: 128,
            chatter_q7: clamped_chatter,
            signal_flags,
            dominant_lane: 0,
            root_cause_code: 0,
            reserved: [0; 2],
            epoch_id,
        }
    }

    /// Branchlessly derives the canonical 2-bit `Lacuna` state code (`0=Unobserved`, `1=SelfRef`,
    /// `2=Exterior`, `3=Undefined`) directly from `signal_flags` and `partition` (`0` redundancy).
    #[inline(always)]
    #[must_use]
    pub const fn lacuna_code(&self) -> u8 {
        let dom = self.partition.dominant_lacuna & 0x03;
        let is_fault = ((self.signal_flags & BOUNDARY_SIGNAL_FLAG_FAULT) != 0)
            | self.partition.has_undefined_bottom()
            | (dom == LACUNA_CODE_UNDEFINED);
        let is_ext = ((self.signal_flags & BOUNDARY_SIGNAL_FLAG_EXTERIOR_ALERT) != 0)
            | self.partition.has_exterior_gap()
            | (dom == LACUNA_CODE_EXTERIOR);
        let is_self = ((self.signal_flags & BOUNDARY_SIGNAL_FLAG_OBSERVED) != 0)
            | (self.partition.self_ref_mask != 0)
            | (dom == LACUNA_CODE_SELF_REF);
        let c1 = branchless_select_u8(is_self, LACUNA_CODE_SELF_REF, LACUNA_CODE_UNOBSERVED);
        let c2 = branchless_select_u8(is_ext, LACUNA_CODE_EXTERIOR, c1);
        branchless_select_u8(is_fault, LACUNA_CODE_UNDEFINED, c2)
    }

    /// Returns `true` when [`BOUNDARY_SIGNAL_FLAG_OBSERVED`] is set.
    #[inline(always)]
    #[must_use]
    pub const fn is_observed(&self) -> bool {
        (self.signal_flags & BOUNDARY_SIGNAL_FLAG_OBSERVED) != 0
    }

    /// Returns `true` when a fault or `Undefined` bottom state is indicated.
    #[inline(always)]
    #[must_use]
    pub const fn is_fault(&self) -> bool {
        ((self.signal_flags & BOUNDARY_SIGNAL_FLAG_FAULT) != 0)
            | self.partition.has_undefined_bottom()
            | ((self.partition.dominant_lacuna & 0x03) == LACUNA_CODE_UNDEFINED)
    }

    /// Returns `true` when [`BOUNDARY_SIGNAL_FLAG_EXTERIOR_ALERT`] is set or `partition.has_exterior_gap()`.
    #[inline(always)]
    #[must_use]
    pub const fn has_exterior_alert(&self) -> bool {
        ((self.signal_flags & BOUNDARY_SIGNAL_FLAG_EXTERIOR_ALERT) != 0)
            | self.partition.has_exterior_gap()
    }

    /// Returns `true` when [`BOUNDARY_SIGNAL_FLAG_STATIONARY_READY`] is set without faults.
    #[inline(always)]
    #[must_use]
    pub const fn is_stationary_ready(&self) -> bool {
        (!self.is_fault()) & ((self.signal_flags & BOUNDARY_SIGNAL_FLAG_STATIONARY_READY) != 0)
    }

    /// Returns `true` when [`BOUNDARY_SIGNAL_FLAG_IRREVERSIBLE_INTENT`] is set.
    #[inline(always)]
    #[must_use]
    pub const fn is_irreversible_intent(&self) -> bool {
        (self.signal_flags & BOUNDARY_SIGNAL_FLAG_IRREVERSIBLE_INTENT) != 0
    }

    /// Returns the decoded [`AttributionTag`].
    #[inline(always)]
    #[must_use]
    pub const fn attribution(&self) -> AttributionTag {
        AttributionTag(self.attribution_tag & 0x07)
    }

    /// Sets or clears [`BOUNDARY_SIGNAL_FLAG_IRREVERSIBLE_INTENT`] and returns the updated signal.
    #[inline(always)]
    #[must_use]
    pub const fn with_irreversible_intent(mut self, irreversible: bool) -> Self {
        if irreversible {
            self.signal_flags |= BOUNDARY_SIGNAL_FLAG_IRREVERSIBLE_INTENT;
        } else {
            self.signal_flags &= !BOUNDARY_SIGNAL_FLAG_IRREVERSIBLE_INTENT;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liminal::partition::evaluate_lacuna_partition_branchless;

    #[test]
    fn test_boundary_signal_layout_and_partition_projection() {
        assert_eq!(size_of::<BoundarySignal>(), 32);
        assert_eq!(align_of::<BoundarySignal>(), 32);

        let tol = [10u8; 8];
        let aligned_obs = [80, 72, 0, 0, 0, 0, 0, 0];
        let lp = evaluate_lacuna_partition_branchless(0x03, &aligned_obs, &tol, 40);
        let sig = BoundarySignal::from_partition(lp, 12_000, 1_024, 16, 42);
        assert!(sig.is_observed());
        assert!(!sig.is_fault());
        assert!(!sig.has_exterior_alert());
        assert!(sig.is_stationary_ready());
        assert!(!sig.is_irreversible_intent());
        assert_eq!(sig.lacuna_code(), 1);
        assert_eq!(sig.contraction_q7, 128);
        assert_eq!(sig.chatter_q7, 16);
        assert_eq!(sig.reserved, [0; 2]);
        assert_eq!(sig.epoch_id, 42);

        let irrev = sig.with_irreversible_intent(true);
        assert!(irrev.is_irreversible_intent());

        let ext_obs = [80, 72, 0, 0, 92, 88, 0, 0];
        let lp_ext = evaluate_lacuna_partition_branchless(0x03, &ext_obs, &tol, 40);
        let sig_ext = BoundarySignal::from_partition(lp_ext, 15_000, 2_000, 32, 43);
        assert!(sig_ext.has_exterior_alert());
        assert_eq!(sig_ext.lacuna_code(), 2);

        let empty_sig = BoundarySignal::new();
        assert_eq!(empty_sig.lacuna_code(), 0);
        assert!(!empty_sig.is_fault());

        let faulty_obs = [80, 0x80, 0, 0, 0, 0, 0, 0];
        let lp_fault = evaluate_lacuna_partition_branchless(0x03, &faulty_obs, &tol, 40);
        let sig_fault = BoundarySignal::from_partition(lp_fault, 20_000, 5_000, 99, 44);
        assert!(sig_fault.is_fault());
        assert!(!sig_fault.is_stationary_ready());
        assert_eq!(sig_fault.lacuna_code(), 3);
    }
}
