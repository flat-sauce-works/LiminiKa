// name: src/core/src/liminal/axiom/lacuna.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structured boundary gap/difference reading (`Void`, `SelfRef`, `Exterior`), where `Void`
//! encodes well-defined zero-aperture non-observation rather than an undefined or invalid null
//! state.
//!
//! Combines the canonical 3-state [`Lacuna`] classifier backed by `(observed, delta)` and
//! Hot-Path [`PackedLacunaField`] with fractal [`ScopedCut`] transitions over [`Scoped<O, T>`].
//!
//! # Compile-Time Soundness Guarantees
//!
//! Verified boundary alignment and bridged discoveries seal into [`Grounded`] cleanly:
//!
//! ```
//! use liminika_core::liminal::{Grounded, GroundedScope, Scoped, ScopedCut};
//!
//! fn accept_grounded<B: GroundedScope, T>(v: Scoped<B, T>) -> Scoped<Grounded, T> {
//!     v.seal_grounded()
//! }
//!
//! let aligned = match Scoped::from_boundary(7u32).cut_by_lacuna_q7(true, 0, 4) {
//!     ScopedCut::SelfRef(s) => accept_grounded(s),
//!     _ => unreachable!(),
//! };
//! assert_eq!(*aligned.grounded_ref(), 7);
//!
//! let bridged = match Scoped::from_delta(9u32).cut_by_lacuna_q7(true, 24, 4) {
//!     ScopedCut::Exterior(spec) => accept_grounded(spec.bridge_to_shared(1).unwrap()),
//!     _ => unreachable!(),
//! };
//! assert_eq!(*bridged.grounded_ref(), 9);
//! ```
//!
//! 1. [`Void`] cannot enter [`super::cut::ActiveScope`] compute gates:
//!
//! ```compile_fail,E0277
//! use liminika_core::liminal::{ActiveScope, Scoped};
//!
//! fn require_active<B: ActiveScope, T>(_: Scoped<B, T>) {}
//!
//! require_active(Scoped::void(1u32));
//! ```
//!
//! 2. [`Void`] cannot be re-cut by [`Scoped::cut_by_lacuna_q7`]:
//!
//! ```compile_fail,E0599
//! use liminika_core::liminal::Scoped;
//!
//! let _ = Scoped::void(1u32).cut_by_lacuna_q7(true, 0, 4);
//! ```
//!
//! 3. [`Void`] absorption prevents laundering through `Refine<Void, Shared>`:
//!
//! ```compile_fail,E0277
//! use liminika_core::liminal::{ActiveScope, Refine, Scoped, Shared, Void};
//!
//! fn require_active<B: ActiveScope, T>(_: Scoped<B, T>) {}
//!
//! fn deny_void_launder(x: Scoped<Refine<Void, Shared>, u32>) {
//!     require_active(x);
//! }
//! ```
//!
//! 4. Unclassified origins (`FromBoundary`) cannot bypass classification into [`super::cut::ActiveScope`]:
//!
//! ```compile_fail,E0277
//! use liminika_core::liminal::{ActiveScope, Scoped};
//!
//! fn require_active<B: ActiveScope, T>(_: Scoped<B, T>) {}
//!
//! require_active(Scoped::from_boundary(1u32));
//! ```
//!
//! 5. [`super::cut::BoundaryViolation`] cannot enter [`super::cut::GroundedScope`] crystallization gates:
//!
//! ```compile_fail,E0277
//! use liminika_core::liminal::{BoundaryViolation, GroundedScope, Scoped};
//!
//! fn require_grounded<B: GroundedScope, T>(_: Scoped<B, T>) {}
//!
//! fn deny_violation(x: BoundaryViolation<u32>) {
//!     require_grounded(x);
//! }
//! ```
//!
//! 6. Unbridged [`super::cut::SpeculativeDiscovery`] cannot enter [`super::cut::GroundedScope`] without bridging:
//!
//! ```compile_fail,E0277
//! use liminika_core::liminal::{GroundedScope, Scoped, SpeculativeDiscovery};
//!
//! fn require_grounded<B: GroundedScope, T>(_: Scoped<B, T>) {}
//!
//! fn deny_unbridged(x: SpeculativeDiscovery<u32>) {
//!     require_grounded(x);
//! }
//! ```

#![forbid(unsafe_code)]

use crate::liminal::constants::NUM_STIGMERGIC_GRANULES;
use crate::liminal::select::branchless_select_u8;

use super::cut::{
    Active, Exterior, Grounded, GroundedOrigin, NonVoid, Refine, Scoped, Shared, Void,
};

/// Structured 3-state boundary reading encoding whether a candidate gap is unmapped (`Void`),
/// an observed zero-difference self-reference (`SelfRef`), or an observed exterior difference (`Exterior`).
///
/// `Void` represents well-defined zero-aperture non-observation outside cognitive scope rather
/// than an invalid or undefined null state.
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

/// Fractal 3-way scope refinement produced by cutting a [`NonVoid`] [`Scoped<O, T>`] through a
/// [`Lacuna`] observation witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScopedCut<O: NonVoid, T> {
    /// Zero-aperture non-observation outside cognitive scope.
    Void(Scoped<Refine<O, Void>, T>),
    /// Observed zero/deadband difference aligned with the shared gauge origin.
    SelfRef(Scoped<Refine<O, Shared>, T>),
    /// Observed exterior difference isolated from direct grounding.
    Exterior(Scoped<Refine<O, Exterior>, T>),
}

impl<O: NonVoid, T> Scoped<O, T> {
    /// Cuts `self` along a scalar `(observed, delta, eps_self)` [`Lacuna`] reading into a typed
    /// [`ScopedCut<O, T>`].
    #[inline(always)]
    #[must_use]
    pub fn cut_by_lacuna(self, observed: bool, delta: f32, eps_self: f32) -> ScopedCut<O, T> {
        let value = self.into_inner();
        match classify_lacuna(observed, delta, eps_self) {
            Lacuna::Void => ScopedCut::Void(Scoped::from_raw(value)),
            Lacuna::SelfRef => ScopedCut::SelfRef(Scoped::from_raw(value)),
            Lacuna::Exterior => ScopedCut::Exterior(Scoped::from_raw(value)),
        }
    }

    /// Cuts `self` along a Q0.7 `(observed, delta_q7, eps_self_q7)` [`Lacuna`] reading into a
    /// typed [`ScopedCut<O, T>`].
    #[inline(always)]
    #[must_use]
    pub fn cut_by_lacuna_q7(
        self,
        observed: bool,
        delta_q7: u8,
        eps_self_q7: u8,
    ) -> ScopedCut<O, T> {
        let value = self.into_inner();
        match Lacuna::from_q7(observed, delta_q7, eps_self_q7) {
            Lacuna::Void => ScopedCut::Void(Scoped::from_raw(value)),
            Lacuna::SelfRef => ScopedCut::SelfRef(Scoped::from_raw(value)),
            Lacuna::Exterior => ScopedCut::Exterior(Scoped::from_raw(value)),
        }
    }
}

impl<O: NonVoid, T> ScopedCut<O, T> {
    /// Returns `true` when this cut landed in the zero-aperture [`ScopedCut::Void`] branch.
    #[inline(always)]
    #[must_use]
    pub const fn is_void(&self) -> bool {
        matches!(self, Self::Void(_))
    }

    /// Returns `true` when this cut landed in the shared [`ScopedCut::SelfRef`] branch.
    #[inline(always)]
    #[must_use]
    pub const fn is_self_ref(&self) -> bool {
        matches!(self, Self::SelfRef(_))
    }

    /// Returns `true` when this cut landed in the exterior [`ScopedCut::Exterior`] branch.
    #[inline(always)]
    #[must_use]
    pub const fn is_exterior(&self) -> bool {
        matches!(self, Self::Exterior(_))
    }

    /// Seals any non-void branch (`SelfRef` or `Exterior`) into the flat [`Active`] closure,
    /// returning `None` on [`ScopedCut::Void`].
    #[inline(always)]
    #[must_use]
    pub fn into_active(self) -> Option<Scoped<Active, T>> {
        match self {
            Self::Void(_) => None,
            Self::SelfRef(s) => Some(s.seal_active()),
            Self::Exterior(e) => Some(e.seal_active()),
        }
    }
}

impl<O: GroundedOrigin, T> ScopedCut<O, T> {
    /// Seals the [`ScopedCut::SelfRef`] branch into the flat [`Grounded`] closure, returning
    /// `None` for [`ScopedCut::Void`] and [`ScopedCut::Exterior`].
    #[inline(always)]
    #[must_use]
    pub fn into_grounded(self) -> Option<Scoped<Grounded, T>> {
        match self {
            Self::SelfRef(s) => Some(s.seal_grounded()),
            Self::Void(_) | Self::Exterior(_) => None,
        }
    }
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
    use crate::liminal::axiom::cut::{
        BoundaryViolation, BridgedDiscovery, SpeculativeDiscovery, VerifiedAligned,
    };

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
    fn test_scoped_cut_closures_and_bridging() {
        let cut_void = Scoped::from_boundary(11u32).cut_by_lacuna(false, 0.0, 0.05);
        assert!(cut_void.is_void());
        assert!(cut_void.into_active().is_none());

        let cut_self = Scoped::from_boundary(22u32).cut_by_lacuna_q7(true, 2, 4);
        assert!(cut_self.is_self_ref());
        let grounded = cut_self.into_grounded().unwrap();
        assert_eq!(*grounded.grounded_ref(), 22);

        let cut_violation = Scoped::from_boundary(33u32).cut_by_lacuna_q7(true, 16, 4);
        assert!(cut_violation.is_exterior());
        if let ScopedCut::Exterior(v) = cut_violation {
            let v_typed: BoundaryViolation<u32> = v;
            assert_eq!(*v_typed.seal_active().active_ref(), 33);
        } else {
            panic!("expected exterior boundary violation");
        }

        let cut_spec = Scoped::from_delta(44u32).cut_by_lacuna_q7(true, 20, 4);
        if let ScopedCut::Exterior(spec) = cut_spec {
            let spec_typed: SpeculativeDiscovery<u32> = spec;
            let unbridged = spec_typed.bridge_to_shared(0).unwrap_err();
            let bridged: BridgedDiscovery<u32> = unbridged.bridge_to_shared(1).unwrap();
            assert_eq!(*bridged.seal_grounded().grounded_ref(), 44);
        } else {
            panic!("expected speculative discovery");
        }

        let cut_aligned = Scoped::from_boundary(55u32).cut_by_lacuna(true, 0.01, 0.05);
        if let ScopedCut::SelfRef(aligned) = cut_aligned {
            let aligned_typed: VerifiedAligned<u32> = aligned;
            assert_eq!(*aligned_typed.seal_grounded().grounded_ref(), 55);
        } else {
            panic!("expected verified aligned");
        }
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
