// name: src/core/src/liminal/axiom/lacuna.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pure finite observation gap classification (`Unobserved`, `SelfRef`, `Exterior`, `Undefined`)
//! and Hot-Path register-packed 8-granule reading ([`PackedLacuna8`]).
//!
//! This module depends strictly on `core` and Layer 0 constants/selectors; it carries no
//! dependency on boundary typestate markers or bit-partition wrappers.

#![forbid(unsafe_code)]

use crate::liminal::constants::NUM_STIGMERGIC_GRANULES;
use crate::liminal::select::branchless_select_u8;

/// Structured 4-state finite observation reading distinguishing normal non-observation
/// (`Unobserved`) from non-finite arithmetic breakdown (`Undefined`), alongside observed
/// stationary self-reference (`SelfRef`) and exterior difference (`Exterior`).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Lacuna {
    /// Well-defined zero-aperture non-observation (`observed == false` with finite `delta`).
    #[default]
    Unobserved = 0,
    /// Observed zero/deadband difference denoting stationary self-reference (`observed == true && delta <= eps_self`).
    SelfRef = 1,
    /// Observed exterior difference beyond self deadband (`observed == true && delta > eps_self`).
    Exterior = 2,
    /// Non-finite (`NaN` / `±inf`) or out-of-range undefined state requiring immediate isolation.
    Undefined = 3,
}

impl Lacuna {
    /// Decodes a compact `u8` discriminant (`0` = `Unobserved`, `1` = `SelfRef`, `2` = `Exterior`,
    /// `3` or any unrecognized code = `Undefined`).
    #[inline(always)]
    #[must_use]
    pub const fn from_code(code: u8) -> Self {
        match code {
            0 => Self::Unobserved,
            1 => Self::SelfRef,
            2 => Self::Exterior,
            _ => Self::Undefined,
        }
    }

    /// Returns `true` when the gap was observed (either [`Self::SelfRef`] or [`Self::Exterior`]).
    #[inline(always)]
    #[must_use]
    pub const fn observed(self) -> bool {
        matches!(self, Self::SelfRef | Self::Exterior)
    }

    /// Returns `true` when the reading is well-defined (not [`Self::Undefined`]).
    #[inline(always)]
    #[must_use]
    pub const fn is_defined(self) -> bool {
        !matches!(self, Self::Undefined)
    }

    /// Classifies `(observed, delta_q7)` against self-reference deadband `eps_self_q7` in Q0.7 (`0..=127`).
    ///
    /// Integer Q0.7 readings are always finite and therefore map strictly to `Unobserved`,
    /// `SelfRef`, or `Exterior`.
    #[inline(always)]
    #[must_use]
    pub const fn from_q7(observed: bool, delta_q7: u8, eps_self_q7: u8) -> Self {
        Self::from_code(classify_lacuna_code_q7(observed, delta_q7, eps_self_q7))
    }
}

/// Classifies a scalar boundary measurement `(observed, delta)` against `eps_self`.
///
/// Evaluation order enforces strict fault isolation:
/// 1. `!delta.is_finite()` -> [`Lacuna::Undefined`] (regardless of `observed`)
/// 2. `!observed` -> [`Lacuna::Unobserved`]
/// 3. `delta.abs() <= eps_self` -> [`Lacuna::SelfRef`]
/// 4. `delta.abs() > eps_self` -> [`Lacuna::Exterior`]
#[inline(always)]
#[must_use]
pub fn classify_lacuna(observed: bool, delta: f32, eps_self: f32) -> Lacuna {
    if !delta.is_finite() {
        return Lacuna::Undefined;
    }
    if !observed {
        return Lacuna::Unobserved;
    }
    let clean_eps = if eps_self.is_finite() && eps_self >= 0.0 {
        eps_self
    } else {
        0.0
    };
    if delta.abs() <= clean_eps {
        Lacuna::SelfRef
    } else {
        Lacuna::Exterior
    }
}

/// Branchlessly encodes `(observed, delta_q7, eps_self_q7)` into a `Lacuna` discriminant byte
/// (`0` = `Unobserved`, `1` = `SelfRef`, `2` = `Exterior`).
#[inline(always)]
#[must_use]
pub const fn classify_lacuna_code_q7(observed: bool, delta_q7: u8, eps_self_q7: u8) -> u8 {
    let active_code = branchless_select_u8(
        delta_q7 <= eps_self_q7,
        Lacuna::SelfRef as u8,
        Lacuna::Exterior as u8,
    );
    branchless_select_u8(observed, active_code, Lacuna::Unobserved as u8)
}

/// Hot-Path register-friendly packed representation of 8 granule [`Lacuna`] readings.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PackedLacuna8 {
    /// Bitmask of observed granules (`bit g == 1` means granule `g` was observed).
    pub observed_mask: u8,
    /// Per-granule unsigned distance from gauge origin in Q0.7 (`0..=127`).
    pub delta_q7: [u8; NUM_STIGMERGIC_GRANULES],
}

impl PackedLacuna8 {
    /// Constructs a new packed 8-granule lacuna reading.
    #[inline(always)]
    #[must_use]
    pub const fn new(observed_mask: u8, delta_q7: [u8; NUM_STIGMERGIC_GRANULES]) -> Self {
        Self {
            observed_mask,
            delta_q7,
        }
    }

    /// Returns `true` when granule `granule_idx` (`0..8`) is marked as observed.
    #[inline(always)]
    #[must_use]
    pub const fn is_observed(&self, granule_idx: usize) -> bool {
        let bit = 1u8 << (granule_idx & (NUM_STIGMERGIC_GRANULES - 1));
        (self.observed_mask & bit) != 0
    }

    /// Classifies granule `granule_idx` (`0..8`) against Q0.7 self-reference deadband `eps_self_q7`.
    #[inline(always)]
    #[must_use]
    pub const fn classify_granule(&self, granule_idx: usize, eps_self_q7: u8) -> Lacuna {
        let idx = granule_idx & (NUM_STIGMERGIC_GRANULES - 1);
        Lacuna::from_q7(self.is_observed(idx), self.delta_q7[idx], eps_self_q7)
    }

    /// Branchlessly partitions all 8 granules into `(unobserved_mask, self_ref_mask, exterior_mask)`.
    #[inline]
    #[must_use]
    pub const fn measure_masks(&self, eps_self_q7: u8) -> (u8, u8, u8) {
        let mut self_ref_mask = 0u8;
        let mut exterior_mask = 0u8;
        let mut g = 0usize;
        while g < NUM_STIGMERGIC_GRANULES {
            let bit = 1u8 << g;
            let obs = (self.observed_mask & bit) != 0;
            let is_self = obs & (self.delta_q7[g] <= eps_self_q7);
            let is_ext = obs & (self.delta_q7[g] > eps_self_q7);
            self_ref_mask |= branchless_select_u8(is_self, bit, 0);
            exterior_mask |= branchless_select_u8(is_ext, bit, 0);
            g += 1;
        }
        let unobserved_mask = !self.observed_mask;
        (unobserved_mask, self_ref_mask, exterior_mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lacuna_four_state_classification() {
        assert_eq!(classify_lacuna(false, 0.0, 0.05), Lacuna::Unobserved);
        assert_eq!(classify_lacuna(false, f32::NAN, 0.05), Lacuna::Undefined);
        assert_eq!(classify_lacuna(true, f32::NAN, 0.05), Lacuna::Undefined);
        assert_eq!(
            classify_lacuna(true, f32::INFINITY, 0.05),
            Lacuna::Undefined
        );
        assert_eq!(
            classify_lacuna(false, f32::NEG_INFINITY, 0.05),
            Lacuna::Undefined
        );
        assert_eq!(classify_lacuna(true, 0.0, 0.05), Lacuna::SelfRef);
        assert_eq!(classify_lacuna(true, -0.05, 0.05), Lacuna::SelfRef);
        assert_eq!(classify_lacuna(true, 0.25, 0.05), Lacuna::Exterior);

        assert!(!Lacuna::Unobserved.observed());
        assert!(Lacuna::Unobserved.is_defined());
        assert!(Lacuna::SelfRef.observed());
        assert!(Lacuna::SelfRef.is_defined());
        assert!(Lacuna::Exterior.observed());
        assert!(Lacuna::Exterior.is_defined());
        assert!(!Lacuna::Undefined.observed());
        assert!(!Lacuna::Undefined.is_defined());

        assert_eq!(Lacuna::from_code(0), Lacuna::Unobserved);
        assert_eq!(Lacuna::from_code(1), Lacuna::SelfRef);
        assert_eq!(Lacuna::from_code(2), Lacuna::Exterior);
        assert_eq!(Lacuna::from_code(3), Lacuna::Undefined);
        assert_eq!(Lacuna::from_code(255), Lacuna::Undefined);
    }

    #[test]
    fn test_packed_lacuna8_masks() {
        let reading = PackedLacuna8::new(0b0011_0101, [2, 10, 4, 80, 3, 20, 0, 12]);
        assert_eq!(reading.classify_granule(0, 4), Lacuna::SelfRef);
        assert_eq!(reading.classify_granule(1, 4), Lacuna::Unobserved);
        assert_eq!(reading.classify_granule(5, 4), Lacuna::Exterior);

        let (unobs_m, self_m, ext_m) = reading.measure_masks(4);
        assert_eq!(unobs_m, 0b1100_1010);
        assert_eq!(self_m, 0b0001_0101);
        assert_eq!(ext_m, 0b0010_0000);
        assert_eq!(unobs_m | self_m | ext_m, 0xFF);
        assert_eq!(unobs_m & self_m, 0);
        assert_eq!(self_m & ext_m, 0);
        assert_eq!(unobs_m & ext_m, 0);
    }
}
