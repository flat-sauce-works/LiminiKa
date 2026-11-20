// name: src/core/src/liminal/axiom/cut.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pure unbounded boundary composition axioms (`Refine<O, I>`, `BitCut8`, `BitCut32`),
//! two-axis sealed scope trait algebra (`Cuttable` x `Definite`, `ActiveScope`, `GroundedOrigin`,
//! `GroundedScope`), zero-overhead [`Scoped<B, T>`] coordinate wrapper, and non-generic closure
//! markers (`Active`, `Grounded`).
//!
//! # Two-Axis Coordinate Model
//!
//! Level 0 (`liminal`) separates spatial boundary structure from force authorization:
//! - **Axis 1 (`Cuttable` — Spatial Partitioning)**: Zero-sized markers (`FromBoundary`,
//!   `FromDelta`, `Interior`, `Shared`, `Exterior`, `Unmarked`, `Bottom`) composed recursively
//!   via [`Refine<O, I>`] (`0` bytes), plus branchless runtime bit-cuts ([`BitCut8`],
//!   [`BitCut32`]). [`Unmarked`] permits further spatial cutting while withholding force;
//!   [`Bottom`] and sealed closures ([`Active`], [`Grounded`]) terminate further cutting at
//!   compile time.
//! - **Axis 2 (`Definite` — Force Authorization)**: [`Scoped<B, T>`] (`#[repr(transparent)]`)
//!   binds an arbitrary payload `T` to boundary type `B`. Trait bounds ([`ActiveScope`],
//!   [`GroundedScope`]) exclude both [`Unmarked`] and [`Bottom`] at every nesting depth.
//!
//! # Closure & Monomorphization Control
//!
//! Arbitrary fractal nesting depths fold into flat canonical closure markers via
//! [`Scoped::seal_active`] (`-> Scoped<Active, T>`) and [`Scoped::seal_grounded`]
//! (`-> Scoped<Grounded, T>`), keeping downstream kernels non-generic with zero L1I code
//! duplication while preventing re-cutting of sealed abstractions.

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
/// `Bottom` implements neither [`Cuttable`] nor [`Definite`]: once a scope enters `Bottom`
/// (directly or inside any [`Refine`]), both further cutting and force propagation stop at
/// compile time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Bottom;

/// Closure marker sealing any [`ActiveScope`] refinement chain into a flat, non-generic compute
/// permission token. Intentionally not [`Cuttable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Active;

/// Closure marker sealing any [`GroundedScope`] refinement chain into a flat, non-generic
/// persistence / crystallization permission token. Intentionally not [`Cuttable`].
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

/// Sealed marker trait for scopes authorized to enter active compute kernels.
///
/// Excludes [`Unmarked`], [`Bottom`], any refinement containing them, and unclassified origin
/// markers ([`FromBoundary`], [`FromDelta`]) prior to boundary cutting.
pub trait ActiveScope: Definite {}

impl ActiveScope for Interior {}
impl ActiveScope for Shared {}
impl ActiveScope for Exterior {}
impl ActiveScope for Active {}
impl ActiveScope for Grounded {}
impl<O: Definite, I: ActiveScope> ActiveScope for Refine<O, I> {}

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
/// Excluded from [`GroundedScope`] until promoted via `bridge_to_shared` in `liminal::axiom`.
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

    /// Safely discards any scoped payload into [`Unmarked`] (zero-aperture non-observation).
    #[inline(always)]
    #[must_use]
    pub fn discard_to_unmarked(self) -> Scoped<Unmarked, T> {
        Scoped::from_raw(self.value)
    }

    /// Terminally discards any scoped payload into [`Bottom`] (non-cuttable undefined state).
    #[inline(always)]
    #[must_use]
    pub fn discard_to_bottom(self) -> Scoped<Bottom, T> {
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

impl<T> Scoped<Unmarked, T> {
    /// Constructs a payload scoped to zero-aperture [`Unmarked`] space.
    #[inline(always)]
    #[must_use]
    pub const fn unmarked(value: T) -> Self {
        Self::from_raw(value)
    }
}

impl<T> Scoped<Bottom, T> {
    /// Constructs a payload scoped to terminal non-cuttable [`Bottom`].
    #[inline(always)]
    #[must_use]
    pub const fn bottom(value: T) -> Self {
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
    assert!(size_of::<Scoped<Bottom, u64>>() == size_of::<u64>());
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
    fn test_scoped_closures_and_discards() {
        let interior = Scoped::interior(10u32);
        assert_eq!(*interior.active_ref(), 10);
        assert_eq!(*interior.grounded_ref(), 10);

        let sealed_active: Scoped<Active, u32> = interior.seal_active();
        assert_eq!(*sealed_active.active_ref(), 10);

        let sealed_grounded: Scoped<Grounded, u32> = Scoped::interior(42u32).seal_grounded();
        assert_eq!(*sealed_grounded.grounded_ref(), 42);

        let unmarked: Scoped<Unmarked, u32> = sealed_grounded.discard_to_unmarked();
        assert_eq!(*unmarked.as_ref(), 42);
        let bottom: Scoped<Bottom, u32> = unmarked.discard_to_bottom();
        assert_eq!(*bottom.as_ref(), 42);
        assert_eq!(bottom.into_inner(), 42);
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
