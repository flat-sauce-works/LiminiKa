// name: src/core/src/liminal/axiom/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Isolated Level 0 boundary axiom kernel uniting orthogonal [`cut`] (unbounded spatial boundary
//! algebra) and [`lacuna`] (finite 4-state observation gap readings).
//!
//! `cut.rs` and `lacuna.rs` have zero mutual dependency. This module is the sole bridge where
//! finite [`Lacuna`] readings drive [`ScopedCut`] transitions over [`Cuttable`] [`Scoped<O, T>`]
//! coordinates and [`BitCut8`] granule partitions.
//!
//! # Compile-Time Soundness Guarantees
//!
//! Verified boundary alignment and bridged discoveries seal into [`Grounded`] cleanly, and
//! [`Unmarked`] space admits further spatial cutting without authorizing force:
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
//!
//! // Cutting inside Unmarked space is spatially permitted (remains non-Definite).
//! let unmarked_cut = Scoped::unmarked(11u32).cut_by_lacuna_q7(true, 0, 4);
//! assert!(unmarked_cut.is_self_ref());
//! ```
//!
//! 1. [`Unmarked`] cannot enter [`ActiveScope`] compute gates:
//!
//! ```compile_fail,E0277
//! use liminika_core::liminal::{ActiveScope, Scoped};
//!
//! fn require_active<B: ActiveScope, T>(_: Scoped<B, T>) {}
//!
//! require_active(Scoped::unmarked(1u32));
//! ```
//!
//! 2. [`Bottom`] cannot enter [`ActiveScope`] compute gates:
//!
//! ```compile_fail,E0277
//! use liminika_core::liminal::{ActiveScope, Scoped};
//!
//! fn require_active<B: ActiveScope, T>(_: Scoped<B, T>) {}
//!
//! require_active(Scoped::bottom(1u32));
//! ```
//!
//! 3. [`Bottom`] cannot be cut (`Bottom: !Cuttable`):
//!
//! ```compile_fail,E0599
//! use liminika_core::liminal::Scoped;
//!
//! let _ = Scoped::bottom(1u32).cut_by_lacuna_q7(true, 0, 4);
//! ```
//!
//! 4. `Refine<FromBoundary, Bottom>` cannot be re-cut (`Bottom` propagation halts cutting):
//!
//! ```compile_fail,E0599
//! use liminika_core::liminal::{Bottom, FromBoundary, Refine, Scoped};
//!
//! fn deny_recut_bottom(x: Scoped<Refine<FromBoundary, Bottom>, u32>) {
//!     let _ = x.cut_by_lacuna_q7(true, 0, 4);
//! }
//! ```
//!
//! 5. Sealed [`Active`] closure cannot be re-cut (`Active: !Cuttable`):
//!
//! ```compile_fail,E0599
//! use liminika_core::liminal::Scoped;
//!
//! let _ = Scoped::interior(1u32).seal_active().cut_by_lacuna_q7(true, 0, 4);
//! ```
//!
//! 6. Sealed [`Grounded`] closure cannot be re-cut (`Grounded: !Cuttable`):
//!
//! ```compile_fail,E0599
//! use liminika_core::liminal::Scoped;
//!
//! let _ = Scoped::interior(1u32).seal_grounded().cut_by_lacuna_q7(true, 0, 4);
//! ```
//!
//! 7. Cutting [`Unmarked`] space cannot flow into [`ScopedCut::into_active`] (`Unmarked: !Definite`):
//!
//! ```compile_fail,E0599
//! use liminika_core::liminal::Scoped;
//!
//! let _ = Scoped::unmarked(1u32).cut_by_lacuna_q7(true, 0, 4).into_active();
//! ```
//!
//! 8. [`Unmarked`] absorption prevents laundering through `Refine<Unmarked, Shared>`:
//!
//! ```compile_fail,E0277
//! use liminika_core::liminal::{ActiveScope, Refine, Scoped, Shared, Unmarked};
//!
//! fn require_active<B: ActiveScope, T>(_: Scoped<B, T>) {}
//!
//! fn deny_unmarked_launder(x: Scoped<Refine<Unmarked, Shared>, u32>) {
//!     require_active(x);
//! }
//! ```
//!
//! 9. Unclassified origins (`FromBoundary`) cannot bypass classification into [`ActiveScope`]:
//!
//! ```compile_fail,E0277
//! use liminika_core::liminal::{ActiveScope, Scoped};
//!
//! fn require_active<B: ActiveScope, T>(_: Scoped<B, T>) {}
//!
//! require_active(Scoped::from_boundary(1u32));
//! ```
//!
//! 10. [`BoundaryViolation`] cannot enter [`GroundedScope`] crystallization gates:
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
//! 11. Unbridged [`SpeculativeDiscovery`] cannot enter [`GroundedScope`] without bridging:
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
#![deny(missing_docs)]

pub mod cut;
pub mod lacuna;

pub use self::cut::*;
pub use self::lacuna::*;

/// Fractal 4-way scope refinement produced by cutting a [`Cuttable`] [`Scoped<O, T>`] through a
/// finite [`Lacuna`] observation witness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScopedCut<O: Cuttable, T> {
    /// Non-finite or undefined breakdown isolating the payload into terminal [`Bottom`].
    Bottom(Scoped<Refine<O, Bottom>, T>),
    /// Zero-aperture non-observation placing the payload into [`Unmarked`] space.
    Unmarked(Scoped<Refine<O, Unmarked>, T>),
    /// Observed zero/deadband difference aligned with the shared gauge origin ([`Shared`]).
    SelfRef(Scoped<Refine<O, Shared>, T>),
    /// Observed exterior difference isolated from direct grounding ([`Exterior`]).
    Exterior(Scoped<Refine<O, Exterior>, T>),
}

impl<O: Cuttable, T> Scoped<O, T> {
    /// Cuts `self` along a scalar `(observed, delta, eps_self)` [`Lacuna`] reading into a typed
    /// [`ScopedCut<O, T>`].
    #[inline(always)]
    #[must_use]
    pub fn cut_by_lacuna(self, observed: bool, delta: f32, eps_self: f32) -> ScopedCut<O, T> {
        let value = self.into_inner();
        match classify_lacuna(observed, delta, eps_self) {
            Lacuna::Undefined => ScopedCut::Bottom(Scoped::from_raw(value)),
            Lacuna::Unobserved => ScopedCut::Unmarked(Scoped::from_raw(value)),
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
            Lacuna::Undefined => ScopedCut::Bottom(Scoped::from_raw(value)),
            Lacuna::Unobserved => ScopedCut::Unmarked(Scoped::from_raw(value)),
            Lacuna::SelfRef => ScopedCut::SelfRef(Scoped::from_raw(value)),
            Lacuna::Exterior => ScopedCut::Exterior(Scoped::from_raw(value)),
        }
    }
}

impl<O: Cuttable, T> ScopedCut<O, T> {
    /// Returns `true` when this cut landed in the terminal [`ScopedCut::Bottom`] branch.
    #[inline(always)]
    #[must_use]
    pub const fn is_bottom(&self) -> bool {
        matches!(self, Self::Bottom(_))
    }

    /// Returns `true` when this cut landed in the zero-aperture [`ScopedCut::Unmarked`] branch.
    #[inline(always)]
    #[must_use]
    pub const fn is_unmarked(&self) -> bool {
        matches!(self, Self::Unmarked(_))
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
}

impl<O: Cuttable + Definite, T> ScopedCut<O, T> {
    /// Seals any definite observed branch (`SelfRef` or `Exterior`) into the flat [`Active`]
    /// closure, returning `None` on [`ScopedCut::Bottom`] and [`ScopedCut::Unmarked`].
    #[inline(always)]
    #[must_use]
    pub fn into_active(self) -> Option<Scoped<Active, T>> {
        match self {
            Self::Bottom(_) | Self::Unmarked(_) => None,
            Self::SelfRef(s) => Some(s.seal_active()),
            Self::Exterior(e) => Some(e.seal_active()),
        }
    }
}

impl<O: Cuttable + GroundedOrigin, T> ScopedCut<O, T> {
    /// Seals the [`ScopedCut::SelfRef`] branch into the flat [`Grounded`] closure, returning
    /// `None` for [`ScopedCut::Bottom`], [`ScopedCut::Unmarked`], and [`ScopedCut::Exterior`].
    #[inline(always)]
    #[must_use]
    pub fn into_grounded(self) -> Option<Scoped<Grounded, T>> {
        match self {
            Self::SelfRef(s) => Some(s.seal_grounded()),
            Self::Bottom(_) | Self::Unmarked(_) | Self::Exterior(_) => None,
        }
    }
}

impl<T> SpeculativeDiscovery<T> {
    /// Attempts to promote a delta-originated exterior discovery into [`BridgedDiscovery`]
    /// when supported by at least one shared bridge granule (`bridge_count >= 1`).
    ///
    /// # Errors
    /// Returns `Err(self)` unchanged when `bridge_count == 0`.
    #[inline(always)]
    pub fn bridge_to_shared(self, bridge_count: u8) -> Result<BridgedDiscovery<T>, Self> {
        if bridge_count >= 1 {
            Ok(Scoped::from_raw(self.into_inner()))
        } else {
            Err(self)
        }
    }
}

/// Splits all 8 granules into `(observed, unobserved)` via [`BitCut8`].
#[inline(always)]
#[must_use]
pub const fn cut_observed_granules(field: &PackedLacunaField) -> BitCut8 {
    BitCut8::split(0xFF, field.observed_mask)
}

/// Refines the observed granule cut into `((self_ref, exterior), (0, unobserved))` via [`BitCut8`].
#[inline]
#[must_use]
pub const fn cut_self_vs_exterior_granules(
    field: &PackedLacunaField,
    eps_self_q7: u8,
) -> (BitCut8, BitCut8) {
    let (_unobs_mask, self_ref_mask, _ext_mask) = field.measure_masks(eps_self_q7);
    cut_observed_granules(field).refine(self_ref_mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scoped_cut_closures_bottom_unmarked_and_bridging() {
        let cut_bottom = Scoped::from_boundary(10u32).cut_by_lacuna(false, f32::NAN, 0.05);
        assert!(cut_bottom.is_bottom());
        assert!(cut_bottom.into_active().is_none());

        let cut_unmarked = Scoped::from_boundary(11u32).cut_by_lacuna(false, 0.0, 0.05);
        assert!(cut_unmarked.is_unmarked());
        assert!(cut_unmarked.into_active().is_none());

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
    fn test_packed_lacuna_bitcut_helpers() {
        let field = PackedLacunaField::new(0b0011_0101, [2, 10, 4, 80, 3, 20, 0, 12]);
        let obs_cut = cut_observed_granules(&field);
        assert_eq!(obs_cut.inside, 0b0011_0101);
        assert_eq!(obs_cut.outside, 0b1100_1010);

        let (in_obs, out_obs) = cut_self_vs_exterior_granules(&field, 4);
        assert_eq!(in_obs.inside, 0b0001_0101);
        assert_eq!(in_obs.outside, 0b0010_0000);
        assert_eq!(out_obs.inside, 0);
        assert_eq!(out_obs.outside, 0b1100_1010);
    }
}
