// name: src/core/src/foundation/aligned/disjoint.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Type-level non-aliasing (`LMNK_RESTRICT`) disjoint aligned slice pairs and triples.

use super::*;

// ===================================================================
// 3. Type-Level LMNK_RESTRICT Non-Aliasing Guards (Disjoint*)
// ===================================================================

/// Type-level `LMNK_RESTRICT` guard holding an immutable input [`AlignedSlice`] and a mutually
/// non-overlapping mutable output [`AlignedSliceMut`].
#[derive(Debug, PartialEq, Eq)]
pub struct DisjointRefMut<'a, T, const ALIGN_IN: usize = 32, const ALIGN_OUT: usize = 32> {
    input: AlignedSlice<'a, T, ALIGN_IN>,
    output: AlignedSliceMut<'a, T, ALIGN_OUT>,
}

impl<'a, T, const ALIGN_IN: usize, const ALIGN_OUT: usize>
    DisjointRefMut<'a, T, ALIGN_IN, ALIGN_OUT>
{
    /// Validates that `input` and `output` are mutually non-overlapping in memory.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if the two memory ranges overlap.
    #[inline(always)]
    pub fn new(
        input: AlignedSlice<'a, T, ALIGN_IN>,
        output: AlignedSliceMut<'a, T, ALIGN_OUT>,
    ) -> LmnkResult<Self> {
        output.ensure_disjoint(&input)?;
        Ok(Self { input, output })
    }

    /// Constructs a `DisjointRefMut` directly from raw FFI input/output pointers and lengths.
    ///
    /// # Safety
    /// Caller must ensure `ptr_in` points to `len_in` valid elements of `T` and `ptr_out` points
    /// to `len_out` valid writable elements of `T` for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_NULL_POINTER`] if a non-empty pointer is null.
    /// Returns [`LMNK_ERR_ALIGNMENT`] if either pointer violates its alignment requirement.
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if the two memory regions overlap.
    #[inline(always)]
    pub unsafe fn from_raw_parts(
        ptr_in: *const T,
        len_in: usize,
        ptr_out: *mut T,
        len_out: usize,
    ) -> LmnkResult<Self> {
        let () = TypeAlignCheck::<T, ALIGN_IN>::VALID;
        let () = TypeAlignCheck::<T, ALIGN_OUT>::VALID;
        let in_bytes = if len_in == 0 {
            0
        } else {
            validate_raw_slice_bounds(ptr_in, len_in)?
        };
        let out_bytes = if len_out == 0 {
            0
        } else {
            validate_raw_slice_bounds(ptr_out, len_out)?
        };

        if raw_byte_ranges_overlap(ptr_in.addr(), in_bytes, ptr_out.addr(), out_bytes) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        // SAFETY: Disjointness and slice bounds are verified above; `from_raw_parts[_mut]` verify non-nullness
        // and alignment.
        let (input, output) = unsafe {
            (
                AlignedSlice::from_raw_parts(ptr_in, len_in)?,
                AlignedSliceMut::from_raw_parts_mut(ptr_out, len_out)?,
            )
        };
        Ok(Self { input, output })
    }

    /// Reborrows this non-aliasing guard for a shorter lifetime without re-running runtime overlap checks.
    #[inline(always)]
    #[must_use]
    pub fn reborrow(&mut self) -> DisjointRefMut<'_, T, ALIGN_IN, ALIGN_OUT> {
        DisjointRefMut {
            input: self.input.reborrow(),
            output: self.output.reborrow(),
        }
    }

    /// Returns an immutable reference to the input slice.
    #[inline(always)]
    #[must_use]
    pub const fn input(&self) -> &AlignedSlice<'a, T, ALIGN_IN> {
        &self.input
    }

    /// Returns an immutable reference to the output slice.
    #[inline(always)]
    #[must_use]
    pub const fn output(&self) -> &AlignedSliceMut<'a, T, ALIGN_OUT> {
        &self.output
    }

    /// Borrows both the immutable input slice and mutable output slice simultaneously.
    #[inline(always)]
    #[must_use]
    pub fn as_ref_mut(
        &mut self,
    ) -> (
        &AlignedSlice<'a, T, ALIGN_IN>,
        &mut AlignedSliceMut<'a, T, ALIGN_OUT>,
    ) {
        (&self.input, &mut self.output)
    }
}

/// Type-level `LMNK_RESTRICT` guard holding two aligned, mutually non-overlapping mutable tensor slices.
#[derive(Debug, PartialEq, Eq)]
pub struct DisjointMutPair<'a, T, const ALIGN: usize = 32> {
    /// First non-overlapping aligned mutable slice.
    pub first: AlignedSliceMut<'a, T, ALIGN>,
    /// Second non-overlapping aligned mutable slice.
    pub second: AlignedSliceMut<'a, T, ALIGN>,
}

impl<'a, T, const ALIGN: usize> DisjointMutPair<'a, T, ALIGN> {
    /// Validates that `first` and `second` are both `ALIGN`-byte aligned and do not overlap in memory.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if the two memory ranges overlap.
    #[inline(always)]
    pub fn new(
        first: AlignedSliceMut<'a, T, ALIGN>,
        second: AlignedSliceMut<'a, T, ALIGN>,
    ) -> LmnkResult<Self> {
        first.ensure_disjoint_mut(&second)?;
        Ok(Self { first, second })
    }

    /// Constructs a `DisjointMutPair` directly from raw FFI pointers of equal length `len`.
    ///
    /// # Safety
    /// Caller must ensure `ptr_a` and `ptr_b` point to valid mutable memory of length `len` for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_NULL_POINTER`], [`LMNK_ERR_ALIGNMENT`], or
    /// [`LMNK_ERR_INVALID_ARGUMENT`] if overlapping.
    #[inline(always)]
    pub unsafe fn from_raw_parts_mut(ptr_a: *mut T, ptr_b: *mut T, len: usize) -> LmnkResult<Self> {
        // SAFETY: Delegated to `from_raw_parts_mut_hetero` with equal lengths.
        unsafe { Self::from_raw_parts_mut_hetero(ptr_a, len, ptr_b, len) }
    }

    /// Constructs a `DisjointMutPair` directly from raw FFI pointers with potentially distinct lengths.
    ///
    /// # Safety
    /// Caller must ensure `ptr_a` points to `len_a` valid elements and `ptr_b` points to `len_b`
    /// valid elements for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_NULL_POINTER`], [`LMNK_ERR_ALIGNMENT`], or
    /// [`LMNK_ERR_INVALID_ARGUMENT`] if overlapping.
    #[inline(always)]
    pub unsafe fn from_raw_parts_mut_hetero(
        ptr_a: *mut T,
        len_a: usize,
        ptr_b: *mut T,
        len_b: usize,
    ) -> LmnkResult<Self> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        let bytes_a = if len_a == 0 {
            0
        } else {
            validate_raw_slice_bounds(ptr_a, len_a)?
        };
        let bytes_b = if len_b == 0 {
            0
        } else {
            validate_raw_slice_bounds(ptr_b, len_b)?
        };

        if raw_byte_ranges_overlap(ptr_a.addr(), bytes_a, ptr_b.addr(), bytes_b) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        // SAFETY: Pointers are verified non-overlapping and in-bounds above; `from_raw_parts_mut` checks
        // non-null & alignment.
        let (first, second) = unsafe {
            (
                AlignedSliceMut::from_raw_parts_mut(ptr_a, len_a)?,
                AlignedSliceMut::from_raw_parts_mut(ptr_b, len_b)?,
            )
        };
        Ok(Self { first, second })
    }

    /// Reborrows this non-aliasing pair for a shorter lifetime without re-running runtime overlap checks.
    #[inline(always)]
    #[must_use]
    pub fn reborrow(&mut self) -> DisjointMutPair<'_, T, ALIGN> {
        DisjointMutPair {
            first: self.first.reborrow(),
            second: self.second.reborrow(),
        }
    }

    /// Returns an immutable reference to the first aligned slice.
    #[inline(always)]
    #[must_use]
    pub const fn first(&self) -> &AlignedSliceMut<'a, T, ALIGN> {
        &self.first
    }

    /// Returns an immutable reference to the second aligned slice.
    #[inline(always)]
    #[must_use]
    pub const fn second(&self) -> &AlignedSliceMut<'a, T, ALIGN> {
        &self.second
    }

    /// Splits the guard into mutable references to both disjoint aligned slices.
    #[inline(always)]
    #[must_use]
    pub fn as_mut_pair(
        &mut self,
    ) -> (
        &mut AlignedSliceMut<'a, T, ALIGN>,
        &mut AlignedSliceMut<'a, T, ALIGN>,
    ) {
        (&mut self.first, &mut self.second)
    }
}

/// Type-level `LMNK_RESTRICT` guard holding three aligned, mutually non-overlapping mutable tensor slices
/// (e.g., Q/K/V fused steering or 3-term Chebyshev polynomial recurrence buffers).
#[derive(Debug, PartialEq, Eq)]
pub struct DisjointMutTriple<'a, T, const ALIGN: usize = 32> {
    /// First non-overlapping aligned mutable slice.
    pub first: AlignedSliceMut<'a, T, ALIGN>,
    /// Second non-overlapping aligned mutable slice.
    pub second: AlignedSliceMut<'a, T, ALIGN>,
    /// Third non-overlapping aligned mutable slice.
    pub third: AlignedSliceMut<'a, T, ALIGN>,
}

impl<'a, T, const ALIGN: usize> DisjointMutTriple<'a, T, ALIGN> {
    /// Validates that `first`, `second`, and `third` are all `ALIGN`-byte aligned and pairwise non-overlapping.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if any two memory ranges overlap.
    #[inline(always)]
    pub fn new(
        first: AlignedSliceMut<'a, T, ALIGN>,
        second: AlignedSliceMut<'a, T, ALIGN>,
        third: AlignedSliceMut<'a, T, ALIGN>,
    ) -> LmnkResult<Self> {
        first.ensure_disjoint_mut(&second)?;
        first.ensure_disjoint_mut(&third)?;
        second.ensure_disjoint_mut(&third)?;
        Ok(Self {
            first,
            second,
            third,
        })
    }

    /// Constructs a `DisjointMutTriple` directly from three raw FFI pointers of equal length `len`.
    ///
    /// # Safety
    /// Caller must ensure `ptr_a`, `ptr_b`, and `ptr_c` point to valid mutable memory of length `len` for
    /// lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_NULL_POINTER`], [`LMNK_ERR_ALIGNMENT`], or
    /// [`LMNK_ERR_INVALID_ARGUMENT`] if any two regions overlap.
    #[inline(always)]
    pub unsafe fn from_raw_parts_mut(
        ptr_a: *mut T,
        ptr_b: *mut T,
        ptr_c: *mut T,
        len: usize,
    ) -> LmnkResult<Self> {
        // SAFETY: Delegated to `from_raw_parts_mut_hetero` with equal lengths.
        unsafe { Self::from_raw_parts_mut_hetero(ptr_a, len, ptr_b, len, ptr_c, len) }
    }

    /// Constructs a `DisjointMutTriple` directly from three raw FFI pointers with potentially distinct lengths.
    ///
    /// # Safety
    /// Caller must ensure `ptr_a`, `ptr_b`, and `ptr_c` point to valid mutable memory of lengths
    /// `len_a`, `len_b`, and `len_c` for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_NULL_POINTER`], [`LMNK_ERR_ALIGNMENT`], or
    /// [`LMNK_ERR_INVALID_ARGUMENT`] if any two regions overlap.
    #[inline(always)]
    pub unsafe fn from_raw_parts_mut_hetero(
        ptr_a: *mut T,
        len_a: usize,
        ptr_b: *mut T,
        len_b: usize,
        ptr_c: *mut T,
        len_c: usize,
    ) -> LmnkResult<Self> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        let bytes_a = if len_a == 0 {
            0
        } else {
            validate_raw_slice_bounds(ptr_a, len_a)?
        };
        let bytes_b = if len_b == 0 {
            0
        } else {
            validate_raw_slice_bounds(ptr_b, len_b)?
        };
        let bytes_c = if len_c == 0 {
            0
        } else {
            validate_raw_slice_bounds(ptr_c, len_c)?
        };

        let addr_a = ptr_a.addr();
        let addr_b = ptr_b.addr();
        let addr_c = ptr_c.addr();

        if raw_byte_ranges_overlap(addr_a, bytes_a, addr_b, bytes_b)
            | raw_byte_ranges_overlap(addr_a, bytes_a, addr_c, bytes_c)
            | raw_byte_ranges_overlap(addr_b, bytes_b, addr_c, bytes_c)
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }

        // SAFETY: Pairwise disjointness and bounds verified above; `from_raw_parts_mut` checks non-null & alignment.
        let (first, second, third) = unsafe {
            (
                AlignedSliceMut::from_raw_parts_mut(ptr_a, len_a)?,
                AlignedSliceMut::from_raw_parts_mut(ptr_b, len_b)?,
                AlignedSliceMut::from_raw_parts_mut(ptr_c, len_c)?,
            )
        };
        Ok(Self {
            first,
            second,
            third,
        })
    }

    /// Reborrows this non-aliasing triple for a shorter lifetime without re-running runtime overlap checks.
    #[inline(always)]
    #[must_use]
    pub fn reborrow(&mut self) -> DisjointMutTriple<'_, T, ALIGN> {
        DisjointMutTriple {
            first: self.first.reborrow(),
            second: self.second.reborrow(),
            third: self.third.reborrow(),
        }
    }

    /// Returns an immutable reference to the first aligned slice.
    #[inline(always)]
    #[must_use]
    pub const fn first(&self) -> &AlignedSliceMut<'a, T, ALIGN> {
        &self.first
    }

    /// Returns an immutable reference to the second aligned slice.
    #[inline(always)]
    #[must_use]
    pub const fn second(&self) -> &AlignedSliceMut<'a, T, ALIGN> {
        &self.second
    }

    /// Returns an immutable reference to the third aligned slice.
    #[inline(always)]
    #[must_use]
    pub const fn third(&self) -> &AlignedSliceMut<'a, T, ALIGN> {
        &self.third
    }
}
