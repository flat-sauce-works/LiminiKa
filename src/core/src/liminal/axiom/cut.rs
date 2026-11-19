// name: src/core/src/liminal/axiom/cut.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fractal boundary composition axioms (`Refine<O, I>`, `BitCut8`, `BitCut32`), sealed scope
//! trait algebra (`NonVoid`, `ActiveScope`, `GroundedOrigin`, `GroundedScope`), zero-overhead
//! [`Scoped<B, T>`] coordinate wrapper, and non-generic closure markers (`Active`, `Grounded`).
//!
//! # Two-Axis Coordinate Model
//!
//! Level 0 (`liminal`) separates boundary structure from payload semantics:
//! - **Axis 1 (Boundary Generation)**: Zero-sized markers (`FromBoundary`, `FromDelta`,
//!   `Interior`, `Shared`, `Exterior`, `Void`) composed recursively via [`Refine<O, I>`] (0 bytes),
//!   plus branchless runtime bit-cuts ([`BitCut8`], [`BitCut32`]). Nesting order is non-commutative:
//!   `Refine<FromBoundary, Exterior>` ([`BoundaryViolation`]) and `Refine<FromDelta, Exterior>`
//!   ([`SpeculativeDiscovery`]) encode distinct epistemic origins at zero runtime cost.
//! - **Axis 2 (Payload Binding)**: [`Scoped<B, T>`] (`#[repr(transparent)]`) binds an arbitrary
//!   value `T` to boundary type `B`. Trait bounds select permitted operations at compile time.
//!
//! # `Void` vs. `null` Semantics
//!
//! [`Void`] and `Scoped<Void, T>` do not represent an invalid pointer, uninitialized memory, or
//! runtime error (`null`). They axiomatically encode well-defined zero-aperture non-observation
//! outside the system's cognitive scope: no steering force is applied and execution passes through
//! safely.
//!
//! # Closure & Monomorphization Control
//!
//! Arbitrary fractal nesting depths fold into flat canonical closure markers via
//! [`Scoped::seal_active`] (`-> Scoped<Active, T>`) and [`Scoped::seal_grounded`]
//! (`-> Scoped<Grounded, T>`), allowing downstream kernels to remain non-generic with zero L1I
//! code duplication while preserving compile-time isolation of [`Void`] and unverified [`Exterior`].

#![forbid(unsafe_code)]

use core::marker::PhantomData;
use core::mem::size_of;

mod sealed {
    pub trait Sealed {}
}

// ===================================================================
// 1. Zero-Sized Origin, Region, Closure & Fractal Markers
// ===================================================================

/// Origin marker indicating evaluation started from an established boundary anchor before
/// measuring difference (`anchor -> delta`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FromBoundary;

/// Origin marker indicating evaluation started from an observed difference or fluctuation before
/// proposing a provisional boundary (`delta -> boundary`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FromDelta;

/// Region marker for self-exclusive interior state ($S \setminus O$).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Interior;

/// Region marker for shared bridge state ($S \cap O$).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Shared;

/// Region marker for observed exterior state ($O \setminus S$, sandbox-isolated).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Exterior;

/// Region marker for well-defined zero-aperture non-observation outside cognitive scope
/// ($\overline{S \cup O}$).
///
/// Unlike a `null` sentinel denoting system failure or undefined state, `Void` explicitly marks
/// an unobserved domain with zero aperture and zero applied force.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Void;

/// Closure marker sealing any [`ActiveScope`] refinement chain into a flat, non-generic compute
/// permission token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Active;

/// Closure marker sealing any [`GroundedScope`] refinement chain into a flat, non-generic
/// persistence / crystallization permission token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Grounded;

/// Zero-sized non-commutative fractal boundary refinement composing outer scope `O` with inner
/// cut `I`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Refine<O, I>(PhantomData<(O, I)>);

impl sealed::Sealed for FromBoundary {}
impl sealed::Sealed for FromDelta {}
impl sealed::Sealed for Interior {}
impl sealed::Sealed for Shared {}
impl sealed::Sealed for Exterior {}
impl sealed::Sealed for Void {}
impl sealed::Sealed for Active {}
impl sealed::Sealed for Grounded {}
impl<O, I> sealed::Sealed for Refine<O, I> {}

// ===================================================================
// 2. Three-Tier Sealed Trait Algebra (Void & Exterior Absorption)
// ===================================================================

/// Sealed marker trait for scopes guaranteed free of [`Void`] at every nesting level.
///
/// Because [`Void`] never implements `NonVoid`, any `Refine<O, I>` containing `Void` in either
/// position fails to satisfy `NonVoid`, preventing void-to-active laundering.
pub trait NonVoid: sealed::Sealed {}

impl NonVoid for FromBoundary {}
impl NonVoid for FromDelta {}
impl NonVoid for Interior {}
impl NonVoid for Shared {}
impl NonVoid for Exterior {}
impl NonVoid for Active {}
impl NonVoid for Grounded {}
impl<O: NonVoid, I: NonVoid> NonVoid for Refine<O, I> {}

/// Sealed marker trait for scopes authorized to enter active compute kernels.
///
/// Excludes [`Void`], any refinement containing [`Void`], and unclassified origin markers
/// ([`FromBoundary`], [`FromDelta`]) prior to boundary cutting.
pub trait ActiveScope: NonVoid {}

impl ActiveScope for Interior {}
impl ActiveScope for Shared {}
impl ActiveScope for Exterior {}
impl ActiveScope for Active {}
impl ActiveScope for Grounded {}
impl<O: NonVoid, I: ActiveScope> ActiveScope for Refine<O, I> {}

/// Sealed marker trait for outer origins eligible to participate in grounded persistence.
///
/// Excludes [`Void`], [`Exterior`], and the erased compute-only closure [`Active`].
pub trait GroundedOrigin: NonVoid {}

impl GroundedOrigin for FromBoundary {}
impl GroundedOrigin for FromDelta {}
impl GroundedOrigin for Interior {}
impl GroundedOrigin for Shared {}
impl GroundedOrigin for Grounded {}
impl<O: GroundedOrigin, I: GroundedOrigin> GroundedOrigin for Refine<O, I> {}

/// Sealed marker trait for scopes authorized to crystallize or persist into long-term memory.
///
/// Because [`Exterior`] does not implement [`GroundedOrigin`] or [`GroundedScope`], neither
/// [`BoundaryViolation`] nor unbridged [`SpeculativeDiscovery`] can be promoted by repeated
/// refinement.
pub trait GroundedScope: ActiveScope + GroundedOrigin {}

impl GroundedScope for Interior {}
impl GroundedScope for Shared {}
impl GroundedScope for Grounded {}
impl<O: GroundedOrigin, I: GroundedScope> GroundedScope for Refine<O, I> {}

// ===================================================================
// 3. Non-Commutative Two-Stage Aliases & Transparent Scoped Wrapper
// ===================================================================

/// Boundary-originated reading confirmed within the shared bridge ($S \cap O$).
pub type VerifiedAligned<T> = Scoped<Refine<FromBoundary, Shared>, T>;

/// Boundary-originated reading that diverged into the exterior ($O \setminus S$).
///
/// Permanently excluded from [`GroundedScope`]; cannot be bridged into long-term memory.
pub type BoundaryViolation<T> = Scoped<Refine<FromBoundary, Exterior>, T>;

/// Delta-originated exterior reading ($O \setminus S$) pending shared-bridge verification.
///
/// Excluded from [`GroundedScope`] until verified through
/// [`SpeculativeDiscovery::bridge_to_shared`].
pub type SpeculativeDiscovery<T> = Scoped<Refine<FromDelta, Exterior>, T>;

/// Delta-originated discovery verified across at least one shared bridge granule ($S \cap O$).
pub type BridgedDiscovery<T> = Scoped<Refine<FromDelta, Shared>, T>;

/// Zero-overhead `#[repr(transparent)]` coordinate binding payload `T` to boundary scope `B`.
///
/// To prevent typestate forgery and post-classification tampering:
/// - `Default` is intentionally not implemented.
/// - Mutable payload access (`as_mut` / `AsMut<T>`) is intentionally withheld.
/// - Direct raw construction (`Scoped::from_raw`) is restricted to `liminal::axiom`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Scoped<B, T> {
    value: T,
    _scope: PhantomData<B>,
}

impl<B, T> Scoped<B, T> {
    /// Internal axiom-only constructor used by boundary cutting and closure transitions.
    #[inline(always)]
    #[must_use]
    pub(super) const fn from_raw(value: T) -> Self {
        Self {
            value,
            _scope: PhantomData,
        }
    }

    /// Consumes the scope wrapper and returns the inner `T`.
    #[inline(always)]
    #[must_use]
    pub fn into_inner(self) -> T {
        self.value
    }

    /// Borrows the wrapped value immutably.
    #[inline(always)]
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub const fn as_ref(&self) -> &T {
        &self.value
    }

    /// Safely discards any scoped payload into [`Void`] (zero-aperture non-observation).
    #[inline(always)]
    #[must_use]
    pub fn discard_to_void(self) -> Scoped<Void, T> {
        Scoped::from_raw(self.value)
    }
}

impl<B, T> AsRef<T> for Scoped<B, T> {
    #[inline(always)]
    fn as_ref(&self) -> &T {
        &self.value
    }
}

impl<T> Scoped<Interior, T> {
    /// Constructs a payload scoped strictly to self-exclusive [`Interior`] ($S \setminus O$).
    #[inline(always)]
    #[must_use]
    pub const fn interior(value: T) -> Self {
        Self::from_raw(value)
    }
}

impl<T> Scoped<Void, T> {
    /// Constructs a payload scoped to zero-aperture [`Void`] ($\overline{S \cup O}$).
    #[inline(always)]
    #[must_use]
    pub const fn void(value: T) -> Self {
        Self::from_raw(value)
    }
}

impl<T> Scoped<FromBoundary, T> {
    /// Tags a payload at the [`FromBoundary`] origin prior to boundary classification.
    #[inline(always)]
    #[must_use]
    pub const fn from_boundary(value: T) -> Self {
        Self::from_raw(value)
    }
}

impl<T> Scoped<FromDelta, T> {
    /// Tags a payload at the [`FromDelta`] origin prior to boundary classification.
    #[inline(always)]
    #[must_use]
    pub const fn from_delta(value: T) -> Self {
        Self::from_raw(value)
    }
}

impl<B: ActiveScope, T> Scoped<B, T> {
    /// Borrows a payload statically guaranteed to satisfy [`ActiveScope`].
    #[inline(always)]
    #[must_use]
    pub const fn active_ref(&self) -> &T {
        &self.value
    }

    /// Folds an arbitrary [`ActiveScope`] refinement chain into the flat [`Active`] closure
    /// for non-generic downstream kernels.
    #[inline(always)]
    #[must_use]
    pub fn seal_active(self) -> Scoped<Active, T> {
        Scoped::from_raw(self.value)
    }
}

impl<B: GroundedScope, T> Scoped<B, T> {
    /// Borrows a payload statically guaranteed to satisfy [`GroundedScope`].
    #[inline(always)]
    #[must_use]
    pub const fn grounded_ref(&self) -> &T {
        &self.value
    }

    /// Folds an arbitrary [`GroundedScope`] refinement chain into the flat [`Grounded`] closure
    /// for non-generic crystallization and persistence kernels.
    #[inline(always)]
    #[must_use]
    pub fn seal_grounded(self) -> Scoped<Grounded, T> {
        Scoped::from_raw(self.value)
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
            Ok(Scoped::from_raw(self.value))
        } else {
            Err(self)
        }
    }
}

// ===================================================================
// 4. Branchless Runtime Atomic Bit-Cuts (`BitCut8` & `BitCut32`)
// ===================================================================

/// Branchless 8-bit atomic partition separating a bit-universe into disjoint `inside` and
/// `outside` subsets (`2` bitwise instructions, `0` allocations).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BitCut8 {
    /// Subset retained inside the cut (`universe & mask`).
    pub inside: u8,
    /// Complement excluded outside the cut (`universe & !mask`).
    pub outside: u8,
}

impl BitCut8 {
    /// Splits `universe` branchlessly along `mask` into `(universe & mask, universe & !mask)`.
    #[inline(always)]
    #[must_use]
    pub const fn split(universe: u8, mask: u8) -> Self {
        Self {
            inside: universe & mask,
            outside: universe & !mask,
        }
    }

    /// Recursively refines both `inside` and `outside` halves by `next_mask`, yielding four
    /// mutually disjoint quadrants.
    #[inline(always)]
    #[must_use]
    pub const fn refine(self, next_mask: u8) -> (Self, Self) {
        (
            Self::split(self.inside, next_mask),
            Self::split(self.outside, next_mask),
        )
    }
}

/// Branchless 32-bit atomic partition separating a bit-universe into disjoint `inside` and
/// `outside` subsets (`2` bitwise instructions, `0` allocations).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BitCut32 {
    /// Subset retained inside the cut (`universe & mask`).
    pub inside: u32,
    /// Complement excluded outside the cut (`universe & !mask`).
    pub outside: u32,
}

impl BitCut32 {
    /// Splits `universe` branchlessly along `mask` into `(universe & mask, universe & !mask)`.
    #[inline(always)]
    #[must_use]
    pub const fn split(universe: u32, mask: u32) -> Self {
        Self {
            inside: universe & mask,
            outside: universe & !mask,
        }
    }

    /// Recursively refines both `inside` and `outside` halves by `next_mask`, yielding four
    /// mutually disjoint quadrants.
    #[inline(always)]
    #[must_use]
    pub const fn refine(self, next_mask: u32) -> (Self, Self) {
        (
            Self::split(self.inside, next_mask),
            Self::split(self.outside, next_mask),
        )
    }
}

const _: () = {
    assert!(size_of::<FromBoundary>() == 0);
    assert!(size_of::<FromDelta>() == 0);
    assert!(size_of::<Interior>() == 0);
    assert!(size_of::<Shared>() == 0);
    assert!(size_of::<Exterior>() == 0);
    assert!(size_of::<Void>() == 0);
    assert!(size_of::<Active>() == 0);
    assert!(size_of::<Grounded>() == 0);
    assert!(size_of::<Refine<FromBoundary, Shared>>() == 0);
    assert!(size_of::<Refine<Refine<FromDelta, Shared>, Interior>>() == 0);
    assert!(size_of::<Scoped<Interior, u64>>() == size_of::<u64>());
    assert!(size_of::<Scoped<Void, u64>>() == size_of::<u64>());
    assert!(size_of::<Scoped<Active, u64>>() == size_of::<u64>());
    assert!(size_of::<Scoped<Grounded, u64>>() == size_of::<u64>());
    assert!(size_of::<VerifiedAligned<u64>>() == size_of::<u64>());
    assert!(size_of::<BitCut8>() == 2);
    assert!(size_of::<BitCut32>() == 8);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scoped_closures_and_discard() {
        let interior = Scoped::interior(10u32);
        assert_eq!(*interior.active_ref(), 10);
        assert_eq!(*interior.grounded_ref(), 10);

        let sealed_active: Scoped<Active, u32> = interior.seal_active();
        assert_eq!(*sealed_active.active_ref(), 10);

        let sealed_grounded: Scoped<Grounded, u32> = Scoped::interior(42u32).seal_grounded();
        assert_eq!(*sealed_grounded.grounded_ref(), 42);

        let voided: Scoped<Void, u32> = sealed_grounded.discard_to_void();
        assert_eq!(*voided.as_ref(), 42);
        assert_eq!(voided.into_inner(), 42);
    }

    #[test]
    fn test_bit_cut8_and_bit_cut32_quadrant_disjointness() {
        let (in_self, out_self) = BitCut8::split(0xFF, 0b0011_0101).refine(0b0101_0011);
        assert_eq!(in_self.inside, 0b0001_0001);
        assert_eq!(in_self.outside, 0b0010_0100);
        assert_eq!(out_self.inside, 0b0100_0010);
        assert_eq!(out_self.outside, 0b1000_1000);
        assert_eq!(
            in_self.inside | in_self.outside | out_self.inside | out_self.outside,
            0xFF
        );
        assert_eq!(in_self.inside & in_self.outside, 0);
        assert_eq!(out_self.inside & out_self.outside, 0);
        assert_eq!(
            (in_self.inside | in_self.outside) & (out_self.inside | out_self.outside),
            0
        );

        let (c_in, c_out) = BitCut32::split(0xFFFF_00FF, 0x0F0F_0F0F).refine(0x00FF_00FF);
        assert_eq!(
            c_in.inside | c_in.outside | c_out.inside | c_out.outside,
            0xFFFF_00FF
        );
        assert_eq!(c_in.inside & c_in.outside, 0);
        assert_eq!(c_out.inside & c_out.outside, 0);
    }
}
