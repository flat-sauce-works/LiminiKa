// name: src/core/src/liminal/axiom/lacuna.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pure finite observation gap classification (`Unobserved`, `SelfRef`, `Exterior`, `Undefined`).
//!
//! This module depends strictly on `core` and Layer 0 selectors; it carries no
//! dependency on boundary typestate markers or bit-partition wrappers.

#![forbid(unsafe_code)]

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

    /// Returns the compact `u8` discriminant (`0` = `Unobserved`, `1` = `SelfRef`, `2` = `Exterior`, `3` =
    /// `Undefined`).
    #[inline(always)]
    #[must_use]
    pub const fn code(self) -> u8 {
        self as u8
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
    let abs_delta = f32::from_bits(delta.to_bits() & 0x7FFF_FFFF);
    if abs_delta <= clean_eps {
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
}
