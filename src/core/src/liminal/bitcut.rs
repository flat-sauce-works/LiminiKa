// name: src/core/src/liminal/bitcut.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Branchless atomic bit-partition utilities ([`BitCut<T>`], [`BitCut8`], [`BitCut32`], [`BitCut64`]).
//!
//! Provides zero-allocation, constant-time set-difference, intersection, refinement, and De Morgan
//! prohibition composition primitives across 8-, 32-, and 64-bit domains.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use core::mem::size_of;

/// Branchless atomic partition separating a bit-universe into disjoint `inside` and `outside`
/// subsets (`2` bitwise instructions, `0` allocations).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BitCut<T> {
    /// Subset retained inside the cut (`universe & mask`).
    pub inside: T,
    /// Complement excluded outside the cut (`universe & !mask`).
    pub outside: T,
}

macro_rules! impl_bitcut_uint {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl BitCut<$ty> {
                /// Splits `universe` branchlessly along `mask` into `(universe & mask, universe & !mask)`.
                #[inline(always)]
                #[must_use]
                pub const fn split(universe: $ty, mask: $ty) -> Self {
                    Self {
                        inside: universe & mask,
                        outside: universe & !mask,
                    }
                }

                /// Recursively refines both `inside` and `outside` halves by `next_mask`, yielding
                /// four mutually disjoint quadrants.
                #[inline(always)]
                #[must_use]
                pub const fn refine(self, next_mask: $ty) -> (Self, Self) {
                    (
                        Self::split(self.inside, next_mask),
                        Self::split(self.outside, next_mask),
                    )
                }

                /// Returns the shared intersection of two cuts (`self.inside & other.inside`).
                #[inline(always)]
                #[must_use]
                pub const fn intersect(self, other: Self) -> $ty {
                    self.inside & other.inside
                }

                /// Returns the bits inside `self` that lie outside `other` (`self.inside & other.outside`).
                #[inline(always)]
                #[must_use]
                pub const fn excess_over(self, other: Self) -> $ty {
                    self.inside & other.outside
                }

                /// Returns the symmetric boundary difference (`self.inside ^ other.inside`).
                #[inline(always)]
                #[must_use]
                pub const fn symmetric_diff(self, other: Self) -> $ty {
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
                pub const fn blindspot_mask(self, other: Self) -> $ty {
                    self.outside & other.outside
                }

                /// Composes two same-origin prohibition cuts via De Morgan duality
                /// (`inside = self.inside | other.inside`, `outside = self.outside & other.outside`).
                #[inline(always)]
                #[must_use]
                pub const fn union_prohibited(self, other: Self) -> Self {
                    Self {
                        inside: self.inside | other.inside,
                        outside: self.outside & other.outside,
                    }
                }
            }
        )+
    };
}

impl_bitcut_uint!(u8, u32, u64);

/// Branchless 8-bit atomic partition (`2` bytes).
pub type BitCut8 = BitCut<u8>;

/// Branchless 32-bit atomic partition (`8` bytes).
pub type BitCut32 = BitCut<u32>;

/// Branchless 64-bit atomic partition (`16` bytes).
pub type BitCut64 = BitCut<u64>;

const _: () = {
    assert!(size_of::<BitCut8>() == 2);
    assert!(size_of::<BitCut32>() == 8);
    assert!(size_of::<BitCut64>() == 16);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_cut_8_32_64_quadrant_and_cross_ops() {
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

        let composed8 = cut_a.union_prohibited(cut_b);
        assert_eq!(composed8.inside, 0b0111_0111);
        assert_eq!(composed8.outside, 0b1000_1000);

        let composed32 = c32_a.union_prohibited(c32_b);
        assert_eq!(composed32.inside, 0x0FFF_00FF);
        assert_eq!(composed32.outside, 0xF000_0000);

        let c64_a = BitCut64::split(0xFFFF_0000_FFFF_00FF, 0x0F0F_0F0F_0F0F_0F0F);
        let c64_b = BitCut64::split(0xFFFF_0000_FFFF_00FF, 0x00FF_00FF_00FF_00FF);
        assert_eq!(c64_a.intersect(c64_b), 0x000F_0000_000F_000F);
        assert_eq!(c64_a.excess_over(c64_b), 0x0F00_0000_0F00_0000);
        assert_eq!(c64_b.excess_over(c64_a), 0x00F0_0000_00F0_00F0);
        assert_eq!(c64_a.blindspot_mask(c64_b), 0xF000_0000_F000_0000);
        assert_eq!(c64_a.symmetric_diff(c64_b), 0x0FF0_0000_0FF0_00F0);
        assert!(c64_a.has_novel_boundary(c64_b));
        assert!(!c64_a.has_novel_boundary(c64_a));

        let (c64_in, c64_out) = c64_a.refine(0x00FF_00FF_00FF_00FF);
        assert_eq!(
            c64_in.inside | c64_in.outside | c64_out.inside | c64_out.outside,
            0xFFFF_0000_FFFF_00FF
        );
        assert_eq!(c64_in.inside & c64_in.outside, 0);
        assert_eq!(c64_out.inside & c64_out.outside, 0);

        let composed64 = c64_a.union_prohibited(c64_b);
        assert_eq!(composed64.inside, 0x0FFF_0000_0FFF_00FF);
        assert_eq!(composed64.outside, 0xF000_0000_F000_0000);
    }
}
