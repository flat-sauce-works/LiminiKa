// name: src/core/src/liminal/axiom/cut.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pure unbounded boundary composition axioms (`Refine<O, I>`, `BitCut8`, `BitCut32`),
//! two-axis sealed scope trait algebra (`Cuttable` x `Definite`, `Activatable`, `GroundedOrigin`,
//! `Groundable`, `IndivisiblePhase`), zero-overhead [`Scoped<O, T>`] coordinate wrapper, and
//! non-cuttable [`Indivisible<C, T>`] closure wrapper (`Active`, `Grounded`).
//!
//! # Two-Axis Coordinate Model
//!
//! Level 0 (`liminal`) separates spatial boundary structure from force authorization and
//! distinguishes cuttable entity phase from non-cuttable converged phase:
//! - **Axis 1 (`Cuttable` — Spatial Partitioning)**: Zero-sized markers (`FromBoundary`,
//!   `FromDelta`, `Interior`, `Shared`, `Exterior`, `Unmarked`) composed recursively
//!   via [`Refine<O, I>`] (`0` bytes), plus branchless runtime bit-cuts ([`BitCut8`],
//!   [`BitCut32`]). [`Unmarked`] permits further spatial cutting while withholding force;
//!   [`Bottom`] and sealed closures ([`Active`], [`Grounded`]) are not [`Cuttable`] and cannot
//!   inhabit [`Scoped<O, T>`].
//! - **Axis 2 (`Definite` — Force Authorization)**: [`Scoped<O, T>`] (`#[repr(transparent)]`,
//!   `O: Cuttable`) binds an arbitrary payload `T` to a cuttable boundary type `O`. Convergence
//!   bounds ([`Activatable`], [`Groundable`]) exclude [`Unmarked`], [`Bottom`], and unclassified
//!   origins at every nesting depth.
//!
//! # Closure, Monomorphization Control & Observer-Relative Re-Gauging
//!
//! Arbitrary fractal nesting depths fold into flat canonical non-cuttable wrappers via
//! [`Scoped::into_active`] (`-> Indivisible<Active, T>`) and [`Scoped::into_grounded`]
//! (`-> Indivisible<Grounded, T>`), keeping downstream kernels non-generic with zero L1I code
//! duplication while preventing re-cutting of converged values in the current observer's frame.
//!
//! An [`Indivisible<C, T>`] value is non-cuttable in the current observer's frame, while
//! remaining zero-cost re-observable as a cuttable entity from another observer's gauge origin via
//! [`Indivisible::rescope_from_boundary`] and [`Indivisible::rescope_from_delta`].

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

/// Region marker for unmarked, zero-aperture space outside active cognitive scope.
///
/// `Unmarked` remains spatially [`Cuttable`] so finer boundary cuts may still be drawn,
/// but it does not implement [`Definite`], so no compute or persistence force can flow
/// through it or through any [`Refine`] containing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Unmarked;

/// Terminal failure/undefined boundary marker ($\bot$).
///
/// `Bottom` implements neither [`Cuttable`], [`Definite`], nor [`IndivisiblePhase`]: it cannot
/// inhabit either [`Scoped<O, T>`] or [`Indivisible<C, T>`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Bottom;

/// Closure marker sealing any [`Activatable`] refinement chain into a flat, non-generic
/// [`Indivisible<Active, T>`] compute permission token. Intentionally not [`Cuttable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Active;

/// Closure marker sealing any [`Groundable`] refinement chain into a flat, non-generic
/// [`Indivisible<Grounded, T>`] persistence / crystallization permission token. Intentionally
/// not [`Cuttable`].
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
impl sealed::Sealed for Unmarked {}
impl sealed::Sealed for Bottom {}
impl sealed::Sealed for Active {}
impl sealed::Sealed for Grounded {}
impl<O, I> sealed::Sealed for Refine<O, I> {}

// ===================================================================
// 2. Two-Axis Sealed Trait Algebra (`Cuttable` x `Definite`)
// ===================================================================

/// Sealed marker trait for spaces that still admit further boundary cutting.
///
/// Intentionally not implemented for [`Bottom`] (terminal undefinedness) or for the sealed
/// abstractions [`Active`] and [`Grounded`] (preventing re-cutting of closed tokens).
pub trait Cuttable: sealed::Sealed {}

impl Cuttable for FromBoundary {}
impl Cuttable for FromDelta {}
impl Cuttable for Interior {}
impl Cuttable for Shared {}
impl Cuttable for Exterior {}
impl Cuttable for Unmarked {}
impl<O: Cuttable, I: Cuttable> Cuttable for Refine<O, I> {}

/// Sealed marker trait for definite scopes guaranteed free of [`Unmarked`] and [`Bottom`] at
/// every nesting level.
///
/// Because neither [`Unmarked`] nor [`Bottom`] implements `Definite`, any `Refine<O, I>`
/// containing either marker in either position fails to satisfy `Definite`, blocking
/// force laundering.
pub trait Definite: sealed::Sealed {}

impl Definite for FromBoundary {}
impl Definite for FromDelta {}
impl Definite for Interior {}
impl Definite for Shared {}
impl Definite for Exterior {}
impl Definite for Active {}
impl Definite for Grounded {}
impl<O: Definite, I: Definite> Definite for Refine<O, I> {}

/// Sealed marker trait for converged non-cuttable phases accepted by [`Indivisible<C, T>`].
///
/// Implemented strictly for [`Active`] and [`Grounded`], preventing [`Bottom`] or cuttable
/// markers such as [`FromBoundary`] from inhabiting [`Indivisible`].
pub trait IndivisiblePhase: sealed::Sealed {}

impl IndivisiblePhase for Active {}
impl IndivisiblePhase for Grounded {}

/// Sealed marker trait for cuttable scopes authorized to converge into [`Indivisible<Active, T>`].
///
/// Excludes [`Unmarked`], [`Bottom`], any refinement containing them, unclassified origin markers
/// ([`FromBoundary`], [`FromDelta`]) prior to boundary cutting, and already-converged closures
/// ([`Active`], [`Grounded`]).
pub trait Activatable: Cuttable + Definite {}

impl Activatable for Interior {}
impl Activatable for Shared {}
impl Activatable for Exterior {}
impl<O: Cuttable + Definite, I: Activatable> Activatable for Refine<O, I> {}

/// Sealed marker trait for outer origins eligible to participate in grounded persistence.
///
/// Excludes [`Unmarked`], [`Bottom`], [`Exterior`], and the erased compute-only closure
/// [`Active`].
pub trait GroundedOrigin: Definite {}

impl GroundedOrigin for FromBoundary {}
impl GroundedOrigin for FromDelta {}
impl GroundedOrigin for Interior {}
impl GroundedOrigin for Shared {}
impl GroundedOrigin for Grounded {}
impl<O: GroundedOrigin, I: GroundedOrigin> GroundedOrigin for Refine<O, I> {}

/// Sealed marker trait for cuttable scopes authorized to converge into [`Indivisible<Grounded, T>`]
/// for crystallization or persistence into long-term memory.
///
/// Because [`Exterior`] does not implement [`GroundedOrigin`] or [`Groundable`], neither
/// [`BoundaryViolation`] nor unbridged [`SpeculativeDiscovery`] can be promoted by repeated
/// refinement.
pub trait Groundable: Activatable + GroundedOrigin {}

impl Groundable for Interior {}
impl Groundable for Shared {}
impl<O: Cuttable + GroundedOrigin, I: Groundable> Groundable for Refine<O, I> {}

// ===================================================================
// 3. Non-Commutative Two-Stage Aliases, Scoped & Indivisible Wrappers
// ===================================================================

/// Boundary-originated reading confirmed within the shared bridge ($S \cap O$).
pub type VerifiedAligned<T> = Scoped<Refine<FromBoundary, Shared>, T>;

/// Boundary-originated reading that diverged into the exterior ($O \setminus S$).
///
/// Permanently excluded from [`Groundable`]; cannot be bridged into long-term memory.
pub type BoundaryViolation<T> = Scoped<Refine<FromBoundary, Exterior>, T>;

/// Delta-originated exterior reading ($O \setminus S$) pending shared-bridge verification.
///
/// Excluded from [`Groundable`] until promoted via `bridge_to_shared` in `liminal::axiom`.
pub type SpeculativeDiscovery<T> = Scoped<Refine<FromDelta, Exterior>, T>;

/// Delta-originated discovery verified across at least one shared bridge granule ($S \cap O$).
pub type BridgedDiscovery<T> = Scoped<Refine<FromDelta, Shared>, T>;

/// Zero-overhead `#[repr(transparent)]` coordinate binding payload `T` to a [`Cuttable`] boundary
/// scope `O`.
///
/// Every live `Scoped<O, T>` value is guaranteed cuttable (`O: Cuttable`); terminal [`Bottom`]
/// and converged phases ([`Active`], [`Grounded`]) cannot inhabit `Scoped`.
///
/// To prevent typestate forgery and post-classification tampering:
/// - `Default` is intentionally not implemented.
/// - Mutable payload access (`as_mut` / `AsMut<T>`) is intentionally withheld.
/// - Direct raw construction (`Scoped::from_raw`) is restricted to `liminal::axiom`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Scoped<O: Cuttable, T> {
    value: T,
    _scope: PhantomData<O>,
}

impl<O: Cuttable, T> Scoped<O, T> {
    /// Internal axiom-only constructor used by boundary cutting and re-gauging transitions.
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

    /// Safely discards any cuttable scoped payload into [`Unmarked`] (zero-aperture
    /// non-observation).
    #[inline(always)]
    #[must_use]
    pub fn discard_to_unmarked(self) -> Scoped<Unmarked, T> {
        Scoped::from_raw(self.value)
    }
}

impl<O: Cuttable, T> AsRef<T> for Scoped<O, T> {
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

impl<T> Scoped<Unmarked, T> {
    /// Constructs a payload scoped to zero-aperture [`Unmarked`] space.
    #[inline(always)]
    #[must_use]
    pub const fn unmarked(value: T) -> Self {
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

impl<O: Activatable, T> Scoped<O, T> {
    /// Borrows a payload statically guaranteed to satisfy [`Activatable`].
    #[inline(always)]
    #[must_use]
    pub const fn active_ref(&self) -> &T {
        &self.value
    }

    /// Converges an arbitrary [`Activatable`] refinement chain into a flat, non-cuttable
    /// [`Indivisible<Active, T>`] token for non-generic downstream compute kernels.
    #[inline(always)]
    #[must_use]
    pub fn into_active(self) -> Indivisible<Active, T> {
        Indivisible::from_raw(self.value)
    }
}

impl<O: Groundable, T> Scoped<O, T> {
    /// Borrows a payload statically guaranteed to satisfy [`Groundable`].
    #[inline(always)]
    #[must_use]
    pub const fn grounded_ref(&self) -> &T {
        &self.value
    }

    /// Converges an arbitrary [`Groundable`] refinement chain into a flat, non-cuttable
    /// [`Indivisible<Grounded, T>`] token for non-generic crystallization and persistence kernels.
    #[inline(always)]
    #[must_use]
    pub fn into_grounded(self) -> Indivisible<Grounded, T> {
        Indivisible::from_raw(self.value)
    }
}

/// Zero-overhead `#[repr(transparent)]` converged payload bound to a non-cuttable
/// [`IndivisiblePhase`] (`Active` or `Grounded`).
///
/// An `Indivisible` value is non-cuttable in the current observer's frame (it exposes zero
/// boundary-cutting methods and does not implement `Default`, `AsMut`, or `Pod`), while
/// remaining zero-cost re-observable as a cuttable entity from another observer's gauge origin
/// via [`Indivisible::rescope_from_boundary`] and [`Indivisible::rescope_from_delta`].
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Indivisible<C: IndivisiblePhase, T> {
    value: T,
    _phase: PhantomData<C>,
}

impl<C: IndivisiblePhase, T> Indivisible<C, T> {
    /// Internal axiom-only constructor used by convergence transitions.
    #[inline(always)]
    #[must_use]
    pub(super) const fn from_raw(value: T) -> Self {
        Self {
            value,
            _phase: PhantomData,
        }
    }

    /// Borrows the converged payload immutably.
    #[inline(always)]
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub const fn as_ref(&self) -> &T {
        &self.value
    }

    /// Consumes the converged wrapper and returns the inner `T`.
    #[inline(always)]
    #[must_use]
    pub fn into_inner(self) -> T {
        self.value
    }

    /// Zero-cost observer-relative re-gauging transition re-entering the cuttable entity phase
    /// under a new [`FromBoundary`] observer origin.
    #[inline(always)]
    #[must_use]
    pub fn rescope_from_boundary(self) -> Scoped<FromBoundary, T> {
        Scoped::from_raw(self.value)
    }

    /// Zero-cost observer-relative re-gauging transition re-entering the cuttable entity phase
    /// under a new [`FromDelta`] observer origin.
    #[inline(always)]
    #[must_use]
    pub fn rescope_from_delta(self) -> Scoped<FromDelta, T> {
        Scoped::from_raw(self.value)
    }
}

impl<C: IndivisiblePhase, T> AsRef<T> for Indivisible<C, T> {
    #[inline(always)]
    fn as_ref(&self) -> &T {
        &self.value
    }
}

impl<T> Indivisible<Grounded, T> {
    /// Relaxes a [`Grounded`] converged token into an [`Active`] compute token without
    /// re-entering boundary cutting.
    #[inline(always)]
    #[must_use]
    pub fn into_active(self) -> Indivisible<Active, T> {
        Indivisible::from_raw(self.value)
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

    /// Returns the shared intersection of two cuts (`self.inside & other.inside`).
    #[inline(always)]
    #[must_use]
    pub const fn intersect(self, other: Self) -> u8 {
        self.inside & other.inside
    }

    /// Returns the bits inside `self` that lie outside `other` (`self.inside & other.outside`).
    #[inline(always)]
    #[must_use]
    pub const fn excess_over(self, other: Self) -> u8 {
        self.inside & other.outside
    }

    /// Returns the symmetric boundary difference (`self.inside ^ other.inside`).
    #[inline(always)]
    #[must_use]
    pub const fn symmetric_diff(self, other: Self) -> u8 {
        self.inside ^ other.inside
    }

    /// Returns `true` when `self.inside` differs from `prev.inside` on at least one bit.
    #[inline(always)]
    #[must_use]
    pub const fn has_novel_boundary(self, prev: Self) -> bool {
        (self.inside ^ prev.inside) != 0
    }

    /// Returns the joint blindspot excluded by both cuts (`self.outside & other.outside`).
    #[inline(always)]
    #[must_use]
    pub const fn blindspot_mask(self, other: Self) -> u8 {
        self.outside & other.outside
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

    /// Returns the shared intersection of two cuts (`self.inside & other.inside`).
    #[inline(always)]
    #[must_use]
    pub const fn intersect(self, other: Self) -> u32 {
        self.inside & other.inside
    }

    /// Returns the bits inside `self` that lie outside `other` (`self.inside & other.outside`).
    #[inline(always)]
    #[must_use]
    pub const fn excess_over(self, other: Self) -> u32 {
        self.inside & other.outside
    }

    /// Returns the symmetric boundary difference (`self.inside ^ other.inside`).
    #[inline(always)]
    #[must_use]
    pub const fn symmetric_diff(self, other: Self) -> u32 {
        self.inside ^ other.inside
    }

    /// Returns `true` when `self.inside` differs from `prev.inside` on at least one bit.
    #[inline(always)]
    #[must_use]
    pub const fn has_novel_boundary(self, prev: Self) -> bool {
        (self.inside ^ prev.inside) != 0
    }

    /// Returns the joint blindspot excluded by both cuts (`self.outside & other.outside`).
    #[inline(always)]
    #[must_use]
    pub const fn blindspot_mask(self, other: Self) -> u32 {
        self.outside & other.outside
    }
}

const _: () = {
    assert!(size_of::<FromBoundary>() == 0);
    assert!(size_of::<FromDelta>() == 0);
    assert!(size_of::<Interior>() == 0);
    assert!(size_of::<Shared>() == 0);
    assert!(size_of::<Exterior>() == 0);
    assert!(size_of::<Unmarked>() == 0);
    assert!(size_of::<Bottom>() == 0);
    assert!(size_of::<Active>() == 0);
    assert!(size_of::<Grounded>() == 0);
    assert!(size_of::<Refine<FromBoundary, Shared>>() == 0);
    assert!(size_of::<Refine<Refine<FromDelta, Shared>, Interior>>() == 0);
    assert!(size_of::<Scoped<Interior, u64>>() == size_of::<u64>());
    assert!(size_of::<Scoped<Unmarked, u64>>() == size_of::<u64>());
    assert!(size_of::<Indivisible<Active, u64>>() == size_of::<u64>());
    assert!(size_of::<Indivisible<Grounded, u64>>() == size_of::<u64>());
    assert!(size_of::<VerifiedAligned<u64>>() == size_of::<u64>());
    assert!(size_of::<BitCut8>() == 2);
    assert!(size_of::<BitCut32>() == 8);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scoped_indivisible_closures_and_rescoping() {
        let interior = Scoped::interior(10u32);
        assert_eq!(*interior.active_ref(), 10);
        assert_eq!(*interior.grounded_ref(), 10);

        let active: Indivisible<Active, u32> = interior.into_active();
        assert_eq!(*active.as_ref(), 10);
        let rescoped_b: Scoped<FromBoundary, u32> = active.rescope_from_boundary();
        assert_eq!(*rescoped_b.as_ref(), 10);

        let grounded: Indivisible<Grounded, u32> = Scoped::interior(42u32).into_grounded();
        assert_eq!(*grounded.as_ref(), 42);
        let relaxed: Indivisible<Active, u32> = grounded.into_active();
        assert_eq!(*relaxed.as_ref(), 42);
        let rescoped_d: Scoped<FromDelta, u32> = grounded.rescope_from_delta();
        assert_eq!(*rescoped_d.as_ref(), 42);

        let unmarked: Scoped<Unmarked, u32> = rescoped_d.discard_to_unmarked();
        assert_eq!(*unmarked.as_ref(), 42);
        assert_eq!(unmarked.into_inner(), 42);
        assert_eq!(relaxed.into_inner(), 42);
    }

    #[test]
    fn test_bit_cut8_and_bit_cut32_quadrant_and_cross_ops() {
        let cut_a = BitCut8::split(0xFF, 0b0011_0101);
        let cut_b = BitCut8::split(0xFF, 0b0101_0011);

        assert_eq!(cut_a.intersect(cut_b), 0b0001_0001);
        assert_eq!(cut_a.excess_over(cut_b), 0b0010_0100);
        assert_eq!(cut_b.excess_over(cut_a), 0b0100_0010);
        assert_eq!(cut_a.blindspot_mask(cut_b), 0b1000_1000);
        assert_eq!(cut_a.symmetric_diff(cut_b), 0b0110_0110);
        assert!(cut_a.has_novel_boundary(cut_b));
        assert!(!cut_a.has_novel_boundary(cut_a));

        let (in_self, out_self) = cut_a.refine(0b0101_0011);
        assert_eq!(in_self.inside, cut_a.intersect(cut_b));
        assert_eq!(in_self.outside, cut_a.excess_over(cut_b));
        assert_eq!(out_self.inside, cut_b.excess_over(cut_a));
        assert_eq!(out_self.outside, cut_a.blindspot_mask(cut_b));
        assert_eq!(
            in_self.inside | in_self.outside | out_self.inside | out_self.outside,
            0xFF
        );

        let c32_a = BitCut32::split(0xFFFF_00FF, 0x0F0F_0F0F);
        let c32_b = BitCut32::split(0xFFFF_00FF, 0x00FF_00FF);
        assert_eq!(c32_a.intersect(c32_b), 0x000F_000F);
        assert_eq!(c32_a.excess_over(c32_b), 0x0F00_0000);
        assert_eq!(c32_b.excess_over(c32_a), 0x00F0_00F0);
        assert_eq!(c32_a.blindspot_mask(c32_b), 0xF000_0000);
        assert_eq!(c32_a.symmetric_diff(c32_b), 0x0FF0_00F0);
        assert!(c32_a.has_novel_boundary(c32_b));
        assert!(!c32_a.has_novel_boundary(c32_a));

        let (c_in, c_out) = c32_a.refine(0x00FF_00FF);
        assert_eq!(
            c_in.inside | c_in.outside | c_out.inside | c_out.outside,
            0xFFFF_00FF
        );
        assert_eq!(c_in.inside & c_in.outside, 0);
        assert_eq!(c_out.inside & c_out.outside, 0);
    }
}
