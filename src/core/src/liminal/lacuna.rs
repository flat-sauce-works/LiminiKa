// name: src/core/src/liminal/lacuna.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Canonical 3-state [`Lacuna`] boundary reading (`Void`, `SelfRef`, `Exterior`) backed by
//! `(observed, delta)` and Hot-Path packed `(observed_mask: u8, delta_q7: [u8; 8])`.

#![forbid(unsafe_code)]

use crate::liminal::constants::NUM_STIGMERGIC_GRANULES;
use crate::liminal::select::branchless_select_u8;

/// Structured 3-state boundary reading encoding whether a candidate gap is unmapped (`Void`),
/// an observed zero-difference self-reference (`SelfRef`), or an observed exterior difference (`Exterior`).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Lacuna {
    /// Unobserved or non-finite reading outside cognitive scope (`observed == false` or `!delta.is_finite()`).
    #[default]
    Void = 0,
    /// Observed zero/deadband difference denoting stationary self-reference (`observed == true && delta <= eps_self`).
    SelfRef = 1,
    /// Observed exterior difference beyond self deadband (`observed == true && delta > eps_self`).
    Exterior = 2,
}

impl Lacuna {
    /// Decodes a compact `u8` discriminant (`0` = `Void`, `1` = `SelfRef`, `2` = `Exterior`).
    #[inline(always)]
    #[must_use]
    pub const fn from_code(code: u8) -> Self {
        match code {
            1 => Self::SelfRef,
            2 => Self::Exterior,
            _ => Self::Void,
        }
    }

    /// Returns `true` when the gap was observed (either [`Self::SelfRef`] or [`Self::Exterior`]).
    #[inline(always)]
    #[must_use]
    pub const fn observed(self) -> bool {
        !matches!(self, Self::Void)
    }

    /// Classifies `(observed, delta)` against self-reference deadband `eps_self` in Q0.7 (`0..=127`).
    #[inline(always)]
    #[must_use]
    pub const fn from_q7(observed: bool, delta_q7: u8, eps_self_q7: u8) -> Self {
        Self::from_code(classify_lacuna_code_q7(observed, delta_q7, eps_self_q7))
    }
}

/// Classifies a scalar boundary measurement `(observed, delta)` against `eps_self`.
///
/// Negative `delta` values are treated by magnitude if finite, matching distance semantics:
/// - `!observed || !delta.is_finite()` -> [`Lacuna::Void`]
/// - `delta.abs() <= eps_self` -> [`Lacuna::SelfRef`]
/// - `delta.abs() > eps_self` -> [`Lacuna::Exterior`]
#[inline(always)]
#[must_use]
pub fn classify_lacuna(observed: bool, delta: f32, eps_self: f32) -> Lacuna {
    if !observed || !delta.is_finite() {
        return Lacuna::Void;
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
/// (`0` = `Void`, `1` = `SelfRef`, `2` = `Exterior`).
#[inline(always)]
#[must_use]
pub const fn classify_lacuna_code_q7(observed: bool, delta_q7: u8, eps_self_q7: u8) -> u8 {
    let active_code = branchless_select_u8(
        delta_q7 <= eps_self_q7,
        Lacuna::SelfRef as u8,
        Lacuna::Exterior as u8,
    );
    branchless_select_u8(observed, active_code, Lacuna::Void as u8)
}

/// Hot-Path register-friendly packed representation of 8 granule [`Lacuna`] readings.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PackedLacunaField {
    /// Bitmask of observed granules (`bit g == 1` means granule `g` was observed).
    pub observed_mask: u8,
    /// Per-granule unsigned distance from gauge origin in Q0.7 (`0..=127`).
    pub delta_q7: [u8; NUM_STIGMERGIC_GRANULES],
}

impl PackedLacunaField {
    /// Constructs a new packed 8-granule lacuna field.
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

    /// Branchlessly partitions all 8 granules into `(void_mask, self_ref_mask, exterior_mask)`.
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
        let void_mask = !self.observed_mask;
        (void_mask, self_ref_mask, exterior_mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lacuna_three_state_classification() {
        assert_eq!(classify_lacuna(false, 0.0, 0.05), Lacuna::Void);
        assert_eq!(classify_lacuna(true, f32::NAN, 0.05), Lacuna::Void);
        assert_eq!(classify_lacuna(true, f32::INFINITY, 0.05), Lacuna::Void);
        assert_eq!(classify_lacuna(true, 0.0, 0.05), Lacuna::SelfRef);
        assert_eq!(classify_lacuna(true, -0.05, 0.05), Lacuna::SelfRef);
        assert_eq!(classify_lacuna(true, 0.25, 0.05), Lacuna::Exterior);

        assert!(!Lacuna::Void.observed());
        assert!(Lacuna::SelfRef.observed());
        assert!(Lacuna::Exterior.observed());
    }

    #[test]
    fn test_packed_lacuna_field_masks() {
        let field = PackedLacunaField::new(0b0011_0101, [2, 10, 4, 80, 3, 20, 0, 12]);
        assert_eq!(field.classify_granule(0, 4), Lacuna::SelfRef);
        assert_eq!(field.classify_granule(1, 4), Lacuna::Void);
        assert_eq!(field.classify_granule(5, 4), Lacuna::Exterior);

        let (void_m, self_m, ext_m) = field.measure_masks(4);
        assert_eq!(void_m, 0b1100_1010);
        assert_eq!(self_m, 0b0001_0101);
        assert_eq!(ext_m, 0b0010_0000);
        assert_eq!(void_m | self_m | ext_m, 0xFF);
        assert_eq!(void_m & self_m, 0);
        assert_eq!(self_m & ext_m, 0);
        assert_eq!(void_m & ext_m, 0);
    }
}
