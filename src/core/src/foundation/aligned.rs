// name: src/core/src/foundation/aligned.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lv.0 Foundation 2: Const-generic zero-copy hardware-aligned slice wrappers, type-level
//! non-aliasing (`LMNK_RESTRICT`) guards, $\mathcal{O}(1)$ Attractor Slot / 2D Slot Bank /
//! Micro-Swarm cell accessors, and stack-allocated hardware-aligned buffers (`16..512`B)
//! for the LiminiKa.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]

use core::marker::PhantomData;
use core::mem::{align_of, size_of};
use core::ops::{Deref, DerefMut, Index, IndexMut, Range};
use core::slice::{self, SliceIndex};

use crate::foundation::pod::{raw_byte_ranges_overlap, Pod};
use crate::liminal::{
    LmnkResult, LmnkStatus, LMNK_ERR_ALIGNMENT, LMNK_ERR_INVALID_ARGUMENT, LMNK_ERR_NULL_POINTER,
};

/// Compile-time helper verifying that `ALIGN` is a non-zero power of two and at least `align_of::<T>()`.
#[allow(dead_code)]
struct TypeAlignCheck<T, const ALIGN: usize>(PhantomData<T>);
impl<T, const ALIGN: usize> TypeAlignCheck<T, ALIGN> {
    const VALID: () = assert!(
        ALIGN > 0 && ALIGN.is_power_of_two() && ALIGN >= align_of::<T>(),
        "ALIGN must be a non-zero power of two and >= align_of::<T>()"
    );
}

/// Compile-time helper verifying that a fixed chunk or slot length `N` is strictly positive.
#[allow(dead_code)]
struct ChunkCheck<const N: usize>;
impl<const N: usize> ChunkCheck<N> {
    const VALID: () = assert!(N > 0, "Chunk size N must be strictly greater than 0");
}

/// Compile-time helper verifying that a 2D slot bank `(SLOTS, SLOT_LEN)` has strictly positive
/// dimensions and does not overflow `usize`.
#[allow(dead_code)]
struct SlotBankCheck<const SLOTS: usize, const SLOT_LEN: usize>;
impl<const SLOTS: usize, const SLOT_LEN: usize> SlotBankCheck<SLOTS, SLOT_LEN> {
    const VALID: () = assert!(
        SLOTS > 0 && SLOT_LEN > 0 && SLOTS.checked_mul(SLOT_LEN).is_some(),
        "SLOTS and SLOT_LEN must be > 0 and SLOTS * SLOT_LEN must not overflow usize"
    );
}

#[inline(always)]
const fn validate_cast_layout<T, U>(len: usize) -> LmnkResult<usize> {
    let u_size = size_of::<U>();
    if u_size == 0 {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    let Some(total_bytes) = len.checked_mul(size_of::<T>()) else {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    };
    if (total_bytes % u_size) != 0 {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    Ok(total_bytes / u_size)
}

#[inline(always)]
const fn validate_slot_offset<T, const ALIGN: usize, const SLOT_LEN: usize>(
    base_addr: usize,
    len: usize,
    slot_idx: usize,
) -> LmnkResult<usize> {
    let () = TypeAlignCheck::<T, ALIGN>::VALID;
    let () = ChunkCheck::<SLOT_LEN>::VALID;
    let Some(offset) = slot_idx.checked_mul(SLOT_LEN) else {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    };
    let Some(end) = offset.checked_add(SLOT_LEN) else {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    };
    if end > len {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    let Some(byte_off) = offset.checked_mul(size_of::<T>()) else {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    };
    if ((base_addr.wrapping_add(byte_off)) & (ALIGN - 1)) != 0 {
        return Err(LMNK_ERR_ALIGNMENT);
    }
    Ok(offset)
}

// ===================================================================
// 1. Immutable Zero-Copy Aligned Slice Wrapper (AlignedSlice)
// ===================================================================

/// Zero-copy SIMD/Cacheline aligned immutable slice wrapper preventing unaligned memory access in the Hot Path.
#[repr(transparent)]
#[derive(Debug, PartialEq, Eq)]
pub struct AlignedSlice<'a, T, const ALIGN: usize = 32> {
    slice: &'a [T],
}

impl<'a, T, const ALIGN: usize> Copy for AlignedSlice<'a, T, ALIGN> {}

impl<'a, T, const ALIGN: usize> Clone for AlignedSlice<'a, T, ALIGN> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T, const ALIGN: usize> AlignedSlice<'a, T, ALIGN> {
    /// Required byte alignment boundary constant.
    pub const ALIGNMENT: usize = ALIGN;

    /// Constructs an empty `AlignedSlice` satisfying the compile-time alignment constraint.
    #[inline(always)]
    #[must_use]
    pub const fn empty() -> Self {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        Self { slice: &[] }
    }

    /// Creates a new aligned slice wrapper after validating pointer memory alignment.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_ALIGNMENT`] if `slice` is non-empty and its base address
    /// is not a multiple of `ALIGN`.
    #[inline(always)]
    pub fn new(slice: &'a [T]) -> LmnkResult<Self> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        let is_valid = slice.is_empty() | (((slice.as_ptr() as usize) & (ALIGN - 1)) == 0);
        if is_valid {
            Ok(Self { slice })
        } else {
            Err(LMNK_ERR_ALIGNMENT)
        }
    }

    /// Constructs an `AlignedSlice` without runtime alignment checks.
    ///
    /// # Safety
    /// The caller must guarantee that `slice` is either empty or its base address is a multiple of `ALIGN`.
    #[inline(always)]
    #[must_use]
    pub const unsafe fn new_unchecked(slice: &'a [T]) -> Self {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        Self { slice }
    }

    /// Creates a new aligned slice wrapper from a single reference after validating memory alignment.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_ALIGNMENT`] if the reference address is not `ALIGN`-byte aligned.
    #[inline(always)]
    pub fn from_ref(value: &'a T) -> LmnkResult<Self> {
        Self::new(slice::from_ref(value))
    }

    /// Safely constructs an `AlignedSlice` from a raw pointer and element count across FFI boundaries.
    ///
    /// # Safety
    /// The caller must guarantee that `ptr` points to at least `len` valid initialized instances of `T`,
    /// that the total byte size does not exceed `isize::MAX`, and that the memory is not mutated for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_NULL_POINTER`] if `ptr` is null when `len > 0`.
    /// Returns [`LMNK_ERR_ALIGNMENT`] if `ptr` violates the `ALIGN` byte boundary.
    #[inline(always)]
    pub unsafe fn from_raw_parts(ptr: *const T, len: usize) -> LmnkResult<Self> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        if len == 0 {
            return Ok(Self::empty());
        }
        if ptr.is_null() {
            return Err(LMNK_ERR_NULL_POINTER);
        }
        if ((ptr as usize) & (ALIGN - 1)) != 0 {
            return Err(LMNK_ERR_ALIGNMENT);
        }
        // SAFETY: Pointer non-nullness and alignment (>= align_of::<T>()) verified above;
        // caller guarantees `len` initialized elements for lifetime `'a`.
        let slice = unsafe { slice::from_raw_parts(ptr, len) };
        Ok(Self { slice })
    }

    /// Safely reinterprets the aligned slice as a slice of another `Pod` type `U` with the same `ALIGN`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `U` is zero-sized or the total byte length of `self`
    /// is not an exact multiple of `size_of::<U>()`.
    #[inline(always)]
    pub fn try_cast<U: Pod>(&self) -> LmnkResult<AlignedSlice<'a, U, ALIGN>>
    where
        T: Pod,
    {
        let () = TypeAlignCheck::<U, ALIGN>::VALID;
        let new_len = validate_cast_layout::<T, U>(self.slice.len())?;
        // SAFETY: `self.slice` alignment >= `ALIGN >= align_of::<U>()`, `new_len * size_of::<U>() == total_bytes`.
        let new_slice =
            unsafe { core::slice::from_raw_parts(self.slice.as_ptr().cast::<U>(), new_len) };
        Ok(AlignedSlice { slice: new_slice })
    }

    /// Downcasts the alignment constraint from `ALIGN` to a smaller or equal power-of-two `SUB_ALIGN`.
    #[inline(always)]
    #[must_use]
    pub const fn relax_alignment<const SUB_ALIGN: usize>(self) -> AlignedSlice<'a, T, SUB_ALIGN> {
        let () = TypeAlignCheck::<T, SUB_ALIGN>::VALID;
        assert!(
            SUB_ALIGN <= ALIGN,
            "Target SUB_ALIGN must be less than or equal to source ALIGN"
        );
        AlignedSlice { slice: self.slice }
    }

    /// Returns the underlying immutable slice reference.
    #[inline(always)]
    #[must_use]
    pub const fn as_slice(&self) -> &'a [T] {
        self.slice
    }

    /// Views the prefix of the slice as a 2D slot bank `&'a [[T; SLOT_LEN]; SLOTS]` in $\mathcal{O}(1)$ zero-copy mode.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `self.len() < SLOTS * SLOT_LEN`.
    #[inline(always)]
    pub const fn as_slot_bank<const SLOTS: usize, const SLOT_LEN: usize>(
        &self,
    ) -> LmnkResult<&'a [[T; SLOT_LEN]; SLOTS]> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        let () = SlotBankCheck::<SLOTS, SLOT_LEN>::VALID;
        if self.slice.len() < SLOTS * SLOT_LEN {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        // SAFETY: `self.slice.len() >= SLOTS * SLOT_LEN` and aligned to `ALIGN >= align_of::<T>()`.
        Ok(unsafe { &*self.slice.as_ptr().cast::<[[T; SLOT_LEN]; SLOTS]>() })
    }

    /// Reborrows `self` for a shorter lifetime `'b`.
    #[inline(always)]
    #[must_use]
    pub const fn reborrow<'b>(&'b self) -> AlignedSlice<'b, T, ALIGN> {
        AlignedSlice { slice: self.slice }
    }

    /// Returns raw immutable pointer to the underlying buffer.
    #[inline(always)]
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.slice.as_ptr()
    }

    /// Returns the start and end raw pointer range for memory boundary checks.
    #[inline(always)]
    #[must_use]
    pub fn as_ptr_range(&self) -> Range<*const T> {
        self.slice.as_ptr_range()
    }

    /// Branchlessly checks whether a given raw pointer falls within the address range of this slice.
    #[inline(always)]
    #[must_use]
    pub fn contains_ptr(&self, ptr: *const T) -> bool {
        let range = self.as_ptr_range();
        let addr = ptr as usize;
        (addr >= (range.start as usize)) & (addr < (range.end as usize))
    }

    /// Branchlessly checks whether the byte range of this slice overlaps with another aligned slice.
    #[inline(always)]
    #[must_use]
    pub fn overlaps<const OTHER_ALIGN: usize>(
        &self,
        other: &AlignedSlice<'_, T, OTHER_ALIGN>,
    ) -> bool {
        raw_byte_ranges_overlap(
            self.as_ptr() as usize,
            self.byte_len(),
            other.as_ptr() as usize,
            other.byte_len(),
        )
    }

    /// Enforces `LMNK_RESTRICT` non-aliasing invariant against another slice.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if the two slices overlap in memory.
    #[inline(always)]
    pub fn ensure_disjoint<const OTHER_ALIGN: usize>(
        &self,
        other: &AlignedSlice<'_, T, OTHER_ALIGN>,
    ) -> LmnkResult<()> {
        if self.overlaps(other) {
            Err(LMNK_ERR_INVALID_ARGUMENT)
        } else {
            Ok(())
        }
    }

    /// Returns the required alignment byte boundary constraint.
    #[inline(always)]
    #[must_use]
    pub const fn align(&self) -> usize {
        ALIGN
    }

    /// Returns the number of elements in the slice.
    #[inline(always)]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.slice.len()
    }

    /// Returns `true` if the slice contains zero elements.
    #[inline(always)]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    /// Returns the total byte length of the underlying slice.
    #[inline(always)]
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.slice.len() * size_of::<T>()
    }

    /// Returns the first element of the slice, or `None` if empty.
    #[inline(always)]
    #[must_use]
    pub const fn first(&self) -> Option<&'a T> {
        self.slice.first()
    }

    /// Returns the last element of the slice, or `None` if empty.
    #[inline(always)]
    #[must_use]
    pub const fn last(&self) -> Option<&'a T> {
        self.slice.last()
    }

    /// Returns a reference to an element at the given index, or `None` if out of bounds.
    #[inline(always)]
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&'a T> {
        self.slice.get(index)
    }

    /// Returns an iterator over the aligned slice elements.
    #[inline(always)]
    pub fn iter(&self) -> core::slice::Iter<'a, T> {
        self.slice.iter()
    }

    /// Returns an iterator over exact non-overlapping chunks of length `chunk_size`.
    #[inline(always)]
    pub fn chunks_exact(&self, chunk_size: usize) -> core::slice::ChunksExact<'a, T> {
        self.slice.chunks_exact(chunk_size)
    }

    /// Views the underlying slice as a byte slice safely guarded against ZST issues.
    #[inline(always)]
    #[must_use]
    pub fn as_bytes(&self) -> &'a [u8]
    where
        T: Pod,
    {
        T::slice_as_bytes(self.slice)
    }

    /// Branchlessly checks if a slice pointer satisfies the const alignment requirement.
    #[inline(always)]
    #[must_use]
    pub fn is_aligned(slice: &[T]) -> bool {
        (ALIGN != 0)
            & ALIGN.is_power_of_two()
            & (slice.is_empty() | (((slice.as_ptr() as usize) & (ALIGN - 1)) == 0))
    }

    /// Extracts a fixed-capacity slot `&'a [T; SLOT_LEN]` at `slot_idx` (`offset = slot_idx * SLOT_LEN`)
    /// in $\mathcal{O}(1)$ zero-copy mode.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `(slot_idx + 1) * SLOT_LEN` exceeds slice length.
    /// Returns [`LMNK_ERR_ALIGNMENT`] if the slot start address violates `ALIGN`.
    #[inline(always)]
    pub fn slot_exact<const SLOT_LEN: usize>(
        &self,
        slot_idx: usize,
    ) -> LmnkResult<&'a [T; SLOT_LEN]> {
        let offset = validate_slot_offset::<T, ALIGN, SLOT_LEN>(
            self.slice.as_ptr() as usize,
            self.slice.len(),
            slot_idx,
        )?;
        // SAFETY: Bounds and `ALIGN` (>= align_of::<T>()) verified by `validate_slot_offset`.
        Ok(unsafe { &*self.slice.as_ptr().add(offset).cast::<[T; SLOT_LEN]>() })
    }
}

impl<'a, T, const ALIGN: usize> Default for AlignedSlice<'a, T, ALIGN> {
    #[inline(always)]
    fn default() -> Self {
        Self::empty()
    }
}

impl<'a, T, const ALIGN: usize> Deref for AlignedSlice<'a, T, ALIGN> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.slice
    }
}

impl<'a, T, const ALIGN: usize> AsRef<[T]> for AlignedSlice<'a, T, ALIGN> {
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        self.slice
    }
}

impl<'a, T, I: SliceIndex<[T]>, const ALIGN: usize> Index<I> for AlignedSlice<'a, T, ALIGN> {
    type Output = I::Output;
    #[inline(always)]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(self.slice, index)
    }
}

impl<'a, T, const ALIGN: usize> IntoIterator for AlignedSlice<'a, T, ALIGN> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.slice.iter()
    }
}

impl<'a, 'b, T, const ALIGN: usize> IntoIterator for &'b AlignedSlice<'a, T, ALIGN> {
    type Item = &'b T;
    type IntoIter = core::slice::Iter<'b, T>;
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.slice.iter()
    }
}

impl<'a, T, const ALIGN: usize> TryFrom<&'a [T]> for AlignedSlice<'a, T, ALIGN> {
    type Error = LmnkStatus;
    #[inline(always)]
    fn try_from(slice: &'a [T]) -> Result<Self, Self::Error> {
        Self::new(slice)
    }
}

// ===================================================================
// 2. Mutable Zero-Copy Aligned Slice Wrapper (AlignedSliceMut)
// ===================================================================

/// Zero-copy SIMD/Cacheline aligned mutable slice wrapper.
#[repr(transparent)]
#[derive(Debug, PartialEq, Eq)]
pub struct AlignedSliceMut<'a, T, const ALIGN: usize = 32> {
    slice: &'a mut [T],
}

impl<'a, T, const ALIGN: usize> AlignedSliceMut<'a, T, ALIGN> {
    /// Required byte alignment boundary constant.
    pub const ALIGNMENT: usize = ALIGN;

    /// Constructs an empty `AlignedSliceMut` satisfying the compile-time alignment constraint.
    #[inline(always)]
    #[must_use]
    pub fn empty() -> Self {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        Self { slice: &mut [] }
    }

    /// Creates a new mutable aligned slice wrapper after validating memory alignment.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_ALIGNMENT`] if `slice` is non-empty and its base address
    /// is not a multiple of `ALIGN`.
    #[inline(always)]
    pub fn new(slice: &'a mut [T]) -> LmnkResult<Self> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        let is_valid = slice.is_empty() | (((slice.as_ptr() as usize) & (ALIGN - 1)) == 0);
        if is_valid {
            Ok(Self { slice })
        } else {
            Err(LMNK_ERR_ALIGNMENT)
        }
    }

    /// Constructs an `AlignedSliceMut` without runtime alignment checks.
    ///
    /// # Safety
    /// The caller must guarantee that `slice` is either empty or its base address is a multiple of `ALIGN`.
    #[inline(always)]
    #[must_use]
    pub unsafe fn new_unchecked(slice: &'a mut [T]) -> Self {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        Self { slice }
    }

    /// Safely constructs an `AlignedSliceMut` from a raw mutable pointer and element count across FFI boundaries.
    ///
    /// # Safety
    /// The caller must guarantee that `ptr` points to at least `len` valid initialized instances of `T`,
    /// that total byte size does not exceed `isize::MAX`, and that no aliasing pointers access the
    /// memory for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_NULL_POINTER`] if `ptr` is null when `len > 0`.
    /// Returns [`LMNK_ERR_ALIGNMENT`] if `ptr` violates the `ALIGN` byte boundary.
    #[inline(always)]
    pub unsafe fn from_raw_parts_mut(ptr: *mut T, len: usize) -> LmnkResult<Self> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        if len == 0 {
            return Ok(Self::empty());
        }
        if ptr.is_null() {
            return Err(LMNK_ERR_NULL_POINTER);
        }
        if ((ptr as usize) & (ALIGN - 1)) != 0 {
            return Err(LMNK_ERR_ALIGNMENT);
        }
        // SAFETY: Pointer non-nullness and alignment (>= align_of::<T>()) verified above;
        // caller guarantees exclusive access to `len` initialized elements for lifetime `'a`.
        let slice = unsafe { slice::from_raw_parts_mut(ptr, len) };
        Ok(Self { slice })
    }

    /// Safely constructs an optional `AlignedSliceMut` from a potentially null mutable FFI pointer.
    ///
    /// # Safety
    /// If `ptr` is non-null, caller must guarantee exclusive access to `len` valid initialized instances of `T`
    /// for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_ALIGNMENT`] if `ptr` is non-null and misaligned when `len > 0`.
    #[inline(always)]
    pub unsafe fn from_raw_parts_mut_opt(ptr: *mut T, len: usize) -> LmnkResult<Option<Self>> {
        if ptr.is_null() {
            Ok(None)
        } else {
            // SAFETY: Non-null pointer is delegated directly to `from_raw_parts_mut`.
            unsafe { Self::from_raw_parts_mut(ptr, len).map(Some) }
        }
    }

    /// Safely reinterprets the mutable aligned slice as a mutable slice of another `Pod` type `U`
    /// while maintaining `ALIGN` invariants.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `U` is zero-sized or total byte length is not a multiple of `size_of::<U>()`.
    #[inline(always)]
    pub fn try_cast_mut<U: Pod>(&mut self) -> LmnkResult<AlignedSliceMut<'_, U, ALIGN>>
    where
        T: Pod,
    {
        let () = TypeAlignCheck::<U, ALIGN>::VALID;
        let new_len = validate_cast_layout::<T, U>(self.slice.len())?;
        // SAFETY: `self.slice` alignment >= `ALIGN >= align_of::<U>()`, `new_len * size_of::<U>() == total_bytes`.
        let new_slice = unsafe {
            core::slice::from_raw_parts_mut(self.slice.as_mut_ptr().cast::<U>(), new_len)
        };
        Ok(AlignedSliceMut { slice: new_slice })
    }

    /// Downcasts the alignment constraint from `ALIGN` to a smaller or equal power-of-two `SUB_ALIGN`.
    #[inline(always)]
    #[must_use]
    pub fn relax_alignment<const SUB_ALIGN: usize>(self) -> AlignedSliceMut<'a, T, SUB_ALIGN> {
        let () = TypeAlignCheck::<T, SUB_ALIGN>::VALID;
        assert!(
            SUB_ALIGN <= ALIGN,
            "Target SUB_ALIGN must be less than or equal to source ALIGN"
        );
        AlignedSliceMut { slice: self.slice }
    }

    /// Returns the underlying immutable slice reference.
    #[inline(always)]
    #[must_use]
    pub const fn as_slice(&self) -> &[T] {
        self.slice
    }

    /// Returns the underlying mutable slice reference.
    #[inline(always)]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.slice
    }

    /// Views the prefix of the slice as a 2D slot bank `&[[T; SLOT_LEN]; SLOTS]` in $\mathcal{O}(1)$ zero-copy mode.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `self.len() < SLOTS * SLOT_LEN`.
    #[inline(always)]
    pub const fn as_slot_bank<const SLOTS: usize, const SLOT_LEN: usize>(
        &self,
    ) -> LmnkResult<&[[T; SLOT_LEN]; SLOTS]> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        let () = SlotBankCheck::<SLOTS, SLOT_LEN>::VALID;
        if self.slice.len() < SLOTS * SLOT_LEN {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        // SAFETY: `self.slice.len() >= SLOTS * SLOT_LEN` and aligned to `ALIGN >= align_of::<T>()`.
        Ok(unsafe { &*self.slice.as_ptr().cast::<[[T; SLOT_LEN]; SLOTS]>() })
    }

    /// Consumes `self` and views the prefix of the mutable slice as a mutable 2D slot bank
    /// `&'a mut [[T; SLOT_LEN]; SLOTS]` with lifetime `'a` in $\mathcal{O}(1)$ zero-copy mode.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `self.len() < SLOTS * SLOT_LEN`.
    #[inline(always)]
    pub fn into_slot_bank_mut<const SLOTS: usize, const SLOT_LEN: usize>(
        self,
    ) -> LmnkResult<&'a mut [[T; SLOT_LEN]; SLOTS]> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        let () = SlotBankCheck::<SLOTS, SLOT_LEN>::VALID;
        if self.slice.len() < SLOTS * SLOT_LEN {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        // SAFETY: `self.slice.len() >= SLOTS * SLOT_LEN`, uniquely borrowed for `'a`, and aligned to `ALIGN >= align_of::<T>()`.
        Ok(unsafe { &mut *self.slice.as_mut_ptr().cast::<[[T; SLOT_LEN]; SLOTS]>() })
    }

    /// Views the prefix of the mutable slice as a mutable 2D slot bank `&mut [[T; SLOT_LEN]; SLOTS]`
    /// in $\mathcal{O}(1)$ zero-copy mode.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `self.len() < SLOTS * SLOT_LEN`.
    #[inline(always)]
    pub fn as_slot_bank_mut<const SLOTS: usize, const SLOT_LEN: usize>(
        &mut self,
    ) -> LmnkResult<&mut [[T; SLOT_LEN]; SLOTS]> {
        self.reborrow().into_slot_bank_mut::<SLOTS, SLOT_LEN>()
    }

    /// Reborrows the mutable slice as an immutable `AlignedSlice`.
    #[inline(always)]
    #[must_use]
    pub const fn as_aligned_slice(&self) -> AlignedSlice<'_, T, ALIGN> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        AlignedSlice { slice: self.slice }
    }

    /// Reborrows `self` for a shorter lifetime.
    #[inline(always)]
    #[must_use]
    pub fn reborrow(&mut self) -> AlignedSliceMut<'_, T, ALIGN> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        AlignedSliceMut {
            slice: &mut *self.slice,
        }
    }

    /// Enforces `LMNK_RESTRICT` non-aliasing invariant against another immutable aligned slice.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if memory ranges overlap.
    #[inline(always)]
    pub fn ensure_disjoint<const OTHER_ALIGN: usize>(
        &self,
        other: &AlignedSlice<'_, T, OTHER_ALIGN>,
    ) -> LmnkResult<()> {
        self.as_aligned_slice().ensure_disjoint(other)
    }

    /// Enforces `LMNK_RESTRICT` non-aliasing invariant against another mutable aligned slice.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if memory ranges overlap.
    #[inline(always)]
    pub fn ensure_disjoint_mut<const OTHER_ALIGN: usize>(
        &self,
        other: &AlignedSliceMut<'_, T, OTHER_ALIGN>,
    ) -> LmnkResult<()> {
        self.as_aligned_slice()
            .ensure_disjoint(&other.as_aligned_slice())
    }

    /// Returns raw mutable pointer to the underlying buffer.
    #[inline(always)]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.slice.as_mut_ptr()
    }

    /// Returns raw immutable pointer to the underlying buffer.
    #[inline(always)]
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.slice.as_ptr()
    }

    /// Branchlessly checks whether a given raw pointer falls within the address range of this mutable slice.
    #[inline(always)]
    #[must_use]
    pub fn contains_ptr(&self, ptr: *const T) -> bool {
        self.as_aligned_slice().contains_ptr(ptr)
    }

    /// Returns the required alignment byte boundary constraint.
    #[inline(always)]
    #[must_use]
    pub const fn align(&self) -> usize {
        ALIGN
    }

    /// Returns the number of elements in the slice.
    #[inline(always)]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.slice.len()
    }

    /// Returns `true` if the slice contains zero elements.
    #[inline(always)]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    /// Returns the total byte length of the underlying slice.
    #[inline(always)]
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.slice.len() * size_of::<T>()
    }

    /// Returns the first element of the slice, or `None` if empty.
    #[inline(always)]
    #[must_use]
    pub const fn first(&self) -> Option<&T> {
        self.slice.first()
    }

    /// Returns the last element of the slice, or `None` if empty.
    #[inline(always)]
    #[must_use]
    pub const fn last(&self) -> Option<&T> {
        self.slice.last()
    }

    /// Returns an immutable iterator over the aligned mutable slice.
    #[inline(always)]
    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.slice.iter()
    }

    /// Returns a mutable iterator over the aligned mutable slice.
    #[inline(always)]
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.slice.iter_mut()
    }

    /// Returns an iterator over exact non-overlapping mutable chunks of length `chunk_size`.
    #[inline(always)]
    pub fn chunks_exact_mut(&mut self, chunk_size: usize) -> core::slice::ChunksExactMut<'_, T> {
        self.slice.chunks_exact_mut(chunk_size)
    }

    /// Fills the aligned mutable slice with a uniform value in zero-allocation mode.
    #[inline(always)]
    pub fn fill(&mut self, value: T)
    where
        T: Copy,
    {
        self.slice.fill(value);
    }

    /// Views the underlying mutable slice as a mutable byte slice when `T: Pod`.
    #[inline(always)]
    #[must_use]
    pub fn as_bytes_mut(&mut self) -> &mut [u8]
    where
        T: Pod,
    {
        T::slice_as_bytes_mut(self.slice)
    }

    /// Branchlessly checks if a slice pointer satisfies the const alignment requirement.
    #[inline(always)]
    #[must_use]
    pub fn is_aligned(slice: &[T]) -> bool {
        AlignedSlice::<T, ALIGN>::is_aligned(slice)
    }

    /// Extracts a mutable fixed-capacity slot `&mut [T; SLOT_LEN]` at `slot_idx` (`offset = slot_idx * SLOT_LEN`)
    /// in $\mathcal{O}(1)$ zero-copy mode.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `(slot_idx + 1) * SLOT_LEN` exceeds slice length.
    /// Returns [`LMNK_ERR_ALIGNMENT`] if the slot start address violates `ALIGN`.
    #[inline(always)]
    pub fn slot_exact_mut<const SLOT_LEN: usize>(
        &mut self,
        slot_idx: usize,
    ) -> LmnkResult<&mut [T; SLOT_LEN]> {
        let offset = validate_slot_offset::<T, ALIGN, SLOT_LEN>(
            self.slice.as_ptr() as usize,
            self.slice.len(),
            slot_idx,
        )?;
        // SAFETY: Bounds and `ALIGN` verified by `validate_slot_offset`.
        Ok(unsafe { &mut *self.slice.as_mut_ptr().add(offset).cast::<[T; SLOT_LEN]>() })
    }

    /// Consumes `self` and splits the mutable slice at `mid`, preserving the original lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `mid` exceeds slice length.
    /// Returns [`LMNK_ERR_ALIGNMENT`] if the right subslice start address violates `ALIGN`.
    #[inline(always)]
    pub fn into_split_at_mut(
        self,
        mid: usize,
    ) -> LmnkResult<(AlignedSliceMut<'a, T, ALIGN>, AlignedSliceMut<'a, T, ALIGN>)> {
        if mid > self.slice.len() {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let (left, right) = self.slice.split_at_mut(mid);
        // SAFETY: `left` starts at the base address of `self.slice` which is already aligned to `ALIGN`.
        Ok((
            unsafe { AlignedSliceMut::new_unchecked(left) },
            AlignedSliceMut::new(right)?,
        ))
    }

    /// Splits the mutable slice by reborrowing `self` at `mid` if the right subslice
    /// satisfies `ALIGN` constraints.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `mid` exceeds slice length.
    /// Returns [`LMNK_ERR_ALIGNMENT`] if the right subslice start address violates `ALIGN`.
    #[inline(always)]
    pub fn split_at_mut(
        &mut self,
        mid: usize,
    ) -> LmnkResult<(AlignedSliceMut<'_, T, ALIGN>, AlignedSliceMut<'_, T, ALIGN>)> {
        self.reborrow().into_split_at_mut(mid)
    }

    /// Copies elements from an unaligned slice into `self` in zero-allocation mode.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if slice lengths do not match.
    #[inline(always)]
    pub fn copy_from_slice(&mut self, src: &[T]) -> LmnkResult<()>
    where
        T: Copy,
    {
        if self.slice.len() != src.len() {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        self.slice.copy_from_slice(src);
        Ok(())
    }
}

impl<'a, T, const ALIGN: usize> Default for AlignedSliceMut<'a, T, ALIGN> {
    #[inline(always)]
    fn default() -> Self {
        Self::empty()
    }
}

impl<'a, T, const ALIGN: usize> Deref for AlignedSliceMut<'a, T, ALIGN> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.slice
    }
}

impl<'a, T, const ALIGN: usize> DerefMut for AlignedSliceMut<'a, T, ALIGN> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slice
    }
}

impl<'a, T, const ALIGN: usize> AsRef<[T]> for AlignedSliceMut<'a, T, ALIGN> {
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        self.slice
    }
}

impl<'a, T, const ALIGN: usize> AsMut<[T]> for AlignedSliceMut<'a, T, ALIGN> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [T] {
        self.slice
    }
}

impl<'a, T, I: SliceIndex<[T]>, const ALIGN: usize> Index<I> for AlignedSliceMut<'a, T, ALIGN> {
    type Output = I::Output;
    #[inline(always)]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&*self.slice, index)
    }
}

impl<'a, T, I: SliceIndex<[T]>, const ALIGN: usize> IndexMut<I> for AlignedSliceMut<'a, T, ALIGN> {
    #[inline(always)]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(self.slice, index)
    }
}

impl<'a, T, const ALIGN: usize> IntoIterator for AlignedSliceMut<'a, T, ALIGN> {
    type Item = &'a mut T;
    type IntoIter = core::slice::IterMut<'a, T>;
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.slice.iter_mut()
    }
}

impl<'a, 'b, T, const ALIGN: usize> IntoIterator for &'b mut AlignedSliceMut<'a, T, ALIGN> {
    type Item = &'b mut T;
    type IntoIter = core::slice::IterMut<'b, T>;
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.slice.iter_mut()
    }
}

impl<'a, T, const ALIGN: usize> TryFrom<&'a mut [T]> for AlignedSliceMut<'a, T, ALIGN> {
    type Error = LmnkStatus;
    #[inline(always)]
    fn try_from(slice: &'a mut [T]) -> Result<Self, Self::Error> {
        Self::new(slice)
    }
}

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
        let in_bytes = len_in
            .checked_mul(size_of::<T>())
            .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        let out_bytes = len_out
            .checked_mul(size_of::<T>())
            .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;

        if raw_byte_ranges_overlap(ptr_in as usize, in_bytes, ptr_out as usize, out_bytes) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        // SAFETY: Disjointness is verified above; `from_raw_parts` and `from_raw_parts_mut` verify non-nullness and alignment.
        let input = unsafe { AlignedSlice::from_raw_parts(ptr_in, len_in)? };
        // SAFETY: Guaranteed disjoint from `input` above.
        let output = unsafe { AlignedSliceMut::from_raw_parts_mut(ptr_out, len_out)? };
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
        let bytes_a = len_a
            .checked_mul(size_of::<T>())
            .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        let bytes_b = len_b
            .checked_mul(size_of::<T>())
            .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;

        if raw_byte_ranges_overlap(ptr_a as usize, bytes_a, ptr_b as usize, bytes_b) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        // SAFETY: Pointers are verified non-overlapping above; `from_raw_parts_mut` checks non-null & alignment.
        let first = unsafe { AlignedSliceMut::from_raw_parts_mut(ptr_a, len_a)? };
        // SAFETY: Guaranteed disjoint from `first` above.
        let second = unsafe { AlignedSliceMut::from_raw_parts_mut(ptr_b, len_b)? };
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
    /// Caller must ensure `ptr_a`, `ptr_b`, and `ptr_c` point to valid mutable memory of length `len` for lifetime `'a`.
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
        let bytes_a = len_a
            .checked_mul(size_of::<T>())
            .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        let bytes_b = len_b
            .checked_mul(size_of::<T>())
            .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        let bytes_c = len_c
            .checked_mul(size_of::<T>())
            .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;

        let addr_a = ptr_a as usize;
        let addr_b = ptr_b as usize;
        let addr_c = ptr_c as usize;

        if raw_byte_ranges_overlap(addr_a, bytes_a, addr_b, bytes_b)
            | raw_byte_ranges_overlap(addr_a, bytes_a, addr_c, bytes_c)
            | raw_byte_ranges_overlap(addr_b, bytes_b, addr_c, bytes_c)
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }

        // SAFETY: Pairwise disjointness verified above; `from_raw_parts_mut` checks non-null & alignment.
        let first = unsafe { AlignedSliceMut::from_raw_parts_mut(ptr_a, len_a)? };
        // SAFETY: Guaranteed disjoint from `first` and `third`.
        let second = unsafe { AlignedSliceMut::from_raw_parts_mut(ptr_b, len_b)? };
        // SAFETY: Guaranteed disjoint from `first` and `second`.
        let third = unsafe { AlignedSliceMut::from_raw_parts_mut(ptr_c, len_c)? };
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

// ===================================================================
// 4. Hardware-Aligned Fixed-Capacity Stack Buffers (AlignedBuffer*)
// ===================================================================

macro_rules! define_aligned_buffer {
    ($name:ident, $align:literal, $doc:expr) => {
        #[doc = $doc]
        #[repr(C, align($align))]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $name<T, const N: usize> {
            /// Contiguous hardware-aligned element array.
            pub data: [T; N],
        }

        impl<T, const N: usize> $name<T, N> {
            /// Guaranteed hardware byte alignment of this buffer.
            pub const ALIGNMENT: usize = $align;
            /// Fixed element capacity `N` of this buffer.
            pub const LEN: usize = N;
            /// Total payload size in bytes (`N * size_of::<T>()`).
            pub const BYTE_SIZE: usize = N * size_of::<T>();

            /// Constructs a new hardware-aligned stack buffer from a fixed-size array.
            #[inline(always)]
            #[must_use]
            pub const fn new(data: [T; N]) -> Self {
                let () = TypeAlignCheck::<T, $align>::VALID;
                Self { data }
            }

            /// Constructs a new hardware-aligned stack buffer filled with `val`.
            #[inline(always)]
            #[must_use]
            pub const fn splat(val: T) -> Self
            where
                T: Copy,
            {
                let () = TypeAlignCheck::<T, $align>::VALID;
                Self { data: [val; N] }
            }

            /// Constructs a zero-initialized hardware-aligned stack buffer at compile time or runtime.
            #[inline(always)]
            #[must_use]
            pub const fn zeroed() -> Self
            where
                T: Pod,
            {
                let () = TypeAlignCheck::<T, $align>::VALID;
                // SAFETY: `T: Pod` guarantees all-zero bit patterns are valid for `[T; N]`.
                unsafe { core::mem::zeroed() }
            }

            /// Returns an immutable slice over the buffer contents.
            #[inline(always)]
            #[must_use]
            pub const fn as_slice(&self) -> &[T] {
                &self.data
            }

            /// Returns a mutable slice over the buffer contents.
            #[inline(always)]
            #[must_use]
            pub fn as_mut_slice(&mut self) -> &mut [T] {
                &mut self.data
            }

            /// Views the buffer contents as an immutable byte slice when `T: Pod`.
            #[inline(always)]
            #[must_use]
            pub fn as_bytes(&self) -> &[u8]
            where
                T: Pod,
            {
                T::slice_as_bytes(&self.data)
            }

            /// Views the buffer contents as a mutable byte slice when `T: Pod`.
            #[inline(always)]
            #[must_use]
            pub fn as_bytes_mut(&mut self) -> &mut [u8]
            where
                T: Pod,
            {
                T::slice_as_bytes_mut(&mut self.data)
            }

            /// Returns a raw immutable pointer to the aligned buffer base address.
            #[inline(always)]
            #[must_use]
            pub const fn as_ptr(&self) -> *const T {
                self.data.as_ptr()
            }

            /// Returns a raw mutable pointer to the aligned buffer base address.
            #[inline(always)]
            #[must_use]
            pub fn as_mut_ptr(&mut self) -> *mut T {
                self.data.as_mut_ptr()
            }

            /// Borrows the buffer as an [`AlignedSlice`] infallibly without runtime branch checks.
            #[inline(always)]
            #[must_use]
            pub const fn as_aligned_slice(&self) -> AlignedSlice<'_, T, $align> {
                let () = TypeAlignCheck::<T, $align>::VALID;
                // SAFETY: `#[repr(C, align($align))]` guarantees base address alignment at compile time.
                unsafe { AlignedSlice::new_unchecked(&self.data) }
            }

            /// Borrows the buffer as an [`AlignedSliceMut`] infallibly without runtime branch checks.
            #[inline(always)]
            #[must_use]
            pub fn as_aligned_slice_mut(&mut self) -> AlignedSliceMut<'_, T, $align> {
                let () = TypeAlignCheck::<T, $align>::VALID;
                // SAFETY: `#[repr(C, align($align))]` guarantees base address alignment at compile time.
                unsafe { AlignedSliceMut::new_unchecked(&mut self.data) }
            }

            /// Borrows the buffer as an [`AlignedSlice`] with a relaxed power-of-two alignment `SUB_ALIGN`.
            #[inline(always)]
            #[must_use]
            pub const fn relax_slice<const SUB_ALIGN: usize>(
                &self,
            ) -> AlignedSlice<'_, T, SUB_ALIGN> {
                self.as_aligned_slice().relax_alignment::<SUB_ALIGN>()
            }

            /// Borrows the buffer as an [`AlignedSliceMut`] with a relaxed power-of-two alignment `SUB_ALIGN`.
            #[inline(always)]
            #[must_use]
            pub fn relax_slice_mut<const SUB_ALIGN: usize>(
                &mut self,
            ) -> AlignedSliceMut<'_, T, SUB_ALIGN> {
                self.as_aligned_slice_mut().relax_alignment::<SUB_ALIGN>()
            }

            /// Views the prefix of the buffer as an immutable 2D slot bank `&[[T; SLOT_LEN]; SLOTS]`.
            ///
            /// # Errors
            /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `N < SLOTS * SLOT_LEN`.
            #[inline(always)]
            pub const fn as_slot_bank<const SLOTS: usize, const SLOT_LEN: usize>(
                &self,
            ) -> LmnkResult<&[[T; SLOT_LEN]; SLOTS]> {
                let () = TypeAlignCheck::<T, $align>::VALID;
                let () = SlotBankCheck::<SLOTS, SLOT_LEN>::VALID;
                if N < SLOTS * SLOT_LEN {
                    return Err(LMNK_ERR_INVALID_ARGUMENT);
                }
                // SAFETY: `N >= SLOTS * SLOT_LEN` and `self.data` is contiguous and aligned to `$align >= align_of::<T>()`.
                Ok(unsafe { &*self.data.as_ptr().cast::<[[T; SLOT_LEN]; SLOTS]>() })
            }

            /// Views the prefix of the buffer as a mutable 2D slot bank `&mut [[T; SLOT_LEN]; SLOTS]`.
            ///
            /// # Errors
            /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `N < SLOTS * SLOT_LEN`.
            #[inline(always)]
            pub fn as_slot_bank_mut<const SLOTS: usize, const SLOT_LEN: usize>(
                &mut self,
            ) -> LmnkResult<&mut [[T; SLOT_LEN]; SLOTS]> {
                let () = TypeAlignCheck::<T, $align>::VALID;
                let () = SlotBankCheck::<SLOTS, SLOT_LEN>::VALID;
                if N < SLOTS * SLOT_LEN {
                    return Err(LMNK_ERR_INVALID_ARGUMENT);
                }
                // SAFETY: `N >= SLOTS * SLOT_LEN`, `self.data` is uniquely borrowed, and aligned to `$align >= align_of::<T>()`.
                Ok(unsafe { &mut *self.data.as_mut_ptr().cast::<[[T; SLOT_LEN]; SLOTS]>() })
            }

            /// Extracts an aligned fixed-capacity slot `&[T; SLOT_LEN]` at `slot_idx`.
            ///
            /// # Errors
            /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `(slot_idx + 1) * SLOT_LEN > N`.
            /// Returns [`LMNK_ERR_ALIGNMENT`] if the slot start address violates `$align`.
            #[inline(always)]
            pub fn slot_exact<const SLOT_LEN: usize>(
                &self,
                slot_idx: usize,
            ) -> LmnkResult<&[T; SLOT_LEN]> {
                self.as_aligned_slice().slot_exact::<SLOT_LEN>(slot_idx)
            }

            /// Extracts an aligned mutable fixed-capacity slot `&mut [T; SLOT_LEN]` at `slot_idx`.
            ///
            /// # Errors
            /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `(slot_idx + 1) * SLOT_LEN > N`.
            /// Returns [`LMNK_ERR_ALIGNMENT`] if the slot start address violates `$align`.
            #[inline(always)]
            pub fn slot_exact_mut<const SLOT_LEN: usize>(
                &mut self,
                slot_idx: usize,
            ) -> LmnkResult<&mut [T; SLOT_LEN]> {
                let offset = validate_slot_offset::<T, $align, SLOT_LEN>(
                    self.data.as_ptr() as usize,
                    N,
                    slot_idx,
                )?;
                // SAFETY: Bounds and `$align` verified by `validate_slot_offset`.
                Ok(unsafe { &mut *self.data.as_mut_ptr().add(offset).cast::<[T; SLOT_LEN]>() })
            }
        }

        impl<T: Default + Copy, const N: usize> Default for $name<T, N> {
            #[inline(always)]
            fn default() -> Self {
                Self::splat(T::default())
            }
        }

        impl<T, const N: usize> Deref for $name<T, N> {
            type Target = [T; N];
            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                &self.data
            }
        }

        impl<T, const N: usize> DerefMut for $name<T, N> {
            #[inline(always)]
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.data
            }
        }

        impl<T, const N: usize> AsRef<[T]> for $name<T, N> {
            #[inline(always)]
            fn as_ref(&self) -> &[T] {
                &self.data
            }
        }

        impl<T, const N: usize> AsMut<[T]> for $name<T, N> {
            #[inline(always)]
            fn as_mut(&mut self) -> &mut [T] {
                &mut self.data
            }
        }

        impl<T, I: SliceIndex<[T]>, const N: usize> Index<I> for $name<T, N> {
            type Output = I::Output;
            #[inline(always)]
            fn index(&self, index: I) -> &Self::Output {
                Index::index(self.data.as_slice(), index)
            }
        }

        impl<T, I: SliceIndex<[T]>, const N: usize> IndexMut<I> for $name<T, N> {
            #[inline(always)]
            fn index_mut(&mut self, index: I) -> &mut Self::Output {
                IndexMut::index_mut(self.data.as_mut_slice(), index)
            }
        }

        // SAFETY: `[T; N]` where `T: Pod` has valid all-zero bit patterns and no drop glue.
        unsafe impl<T: Pod, const N: usize> Pod for $name<T, N> {}
    };
}

define_aligned_buffer!(
    AlignedBuffer16,
    16,
    "16-byte hardware-aligned fixed-capacity buffer for Q7 phase delta vectors."
);
define_aligned_buffer!(
    AlignedBuffer32,
    32,
    "32-byte hardware-aligned fixed-capacity buffer for AVX2 / Warp SIMD tensors."
);
define_aligned_buffer!(
    AlignedBuffer64,
    64,
    "64-byte hardware-aligned fixed-capacity buffer for CPU/GPU cacheline structures."
);
define_aligned_buffer!(
    AlignedBuffer128,
    128,
    "128-byte hardware-aligned fixed-capacity buffer for dual-cacheline pointer trails."
);
define_aligned_buffer!(
    AlignedBuffer256,
    256,
    "256-byte hardware-aligned fixed-capacity buffer for AVX-512 / Tensor Core tiles."
);
define_aligned_buffer!(
    AlignedBuffer512,
    512,
    "512-byte hardware-aligned fixed-capacity buffer for Direct-DMA page transfers."
);

// ===================================================================
// 5. Type Aliases & Standalone Alignment Inspection Functions
// ===================================================================

/// 16-byte aligned immutable slice wrapper for phase steering arrays.
pub type AlignedSlice16<'a, T> = AlignedSlice<'a, T, 16>;
/// 16-byte aligned mutable slice wrapper for phase steering arrays.
pub type AlignedSliceMut16<'a, T> = AlignedSliceMut<'a, T, 16>;

/// 32-byte aligned immutable slice wrapper for SIMD/AVX vectors and tensor buffers.
pub type AlignedSlice32<'a, T> = AlignedSlice<'a, T, 32>;
/// 32-byte aligned mutable slice wrapper for SIMD/AVX vectors and tensor buffers.
pub type AlignedSliceMut32<'a, T> = AlignedSliceMut<'a, T, 32>;

/// 64-byte aligned immutable slice wrapper for CPU/GPU cache line structures.
pub type AlignedSlice64<'a, T> = AlignedSlice<'a, T, 64>;
/// 64-byte aligned mutable slice wrapper for CPU/GPU cache line structures.
pub type AlignedSliceMut64<'a, T> = AlignedSliceMut<'a, T, 64>;

/// 128-byte aligned immutable slice wrapper for dual cache line pointer trails.
pub type AlignedSlice128<'a, T> = AlignedSlice<'a, T, 128>;
/// 128-byte aligned mutable slice wrapper for dual cache line pointer trails.
pub type AlignedSliceMut128<'a, T> = AlignedSliceMut<'a, T, 128>;

/// 256-byte aligned immutable slice wrapper for AVX-512 and GPU Tensor Core tile boundaries.
pub type AlignedSlice256<'a, T> = AlignedSlice<'a, T, 256>;
/// 256-byte aligned mutable slice wrapper for AVX-512 and GPU Tensor Core tile boundaries.
pub type AlignedSliceMut256<'a, T> = AlignedSliceMut<'a, T, 256>;

/// 512-byte aligned immutable slice wrapper for GPU Tensor Core matrix page and Direct-DMA memory transfers.
pub type AlignedSlice512<'a, T> = AlignedSlice<'a, T, 512>;
/// 512-byte aligned mutable slice wrapper for GPU Tensor Core matrix page and Direct-DMA memory transfers.
pub type AlignedSliceMut512<'a, T> = AlignedSliceMut<'a, T, 512>;

/// Branchlessly checks slice pointer alignment for a runtime power-of-two byte boundary.
#[inline(always)]
#[must_use]
pub fn is_aligned_to<T>(slice: &[T], align: usize) -> bool {
    (align != 0)
        & align.is_power_of_two()
        & (slice.is_empty() | (((slice.as_ptr() as usize) & (align - 1)) == 0))
}

/// Branchlessly checks raw pointer non-nullness and alignment for a runtime power-of-two byte boundary.
#[inline(always)]
#[must_use]
pub fn is_ptr_aligned_to<T>(ptr: *const T, align: usize) -> bool {
    (align != 0)
        & align.is_power_of_two()
        & (!ptr.is_null())
        & (((ptr as usize) & (align - 1)) == 0)
}

const _: () = {
    assert!(size_of::<AlignedSlice<'static, f32, 32>>() == size_of::<&'static [f32]>());
    assert!(align_of::<AlignedSlice<'static, f32, 32>>() == align_of::<&'static [f32]>());
    assert!(size_of::<AlignedSliceMut<'static, f32, 32>>() == size_of::<&'static mut [f32]>());
    assert!(align_of::<AlignedSliceMut<'static, f32, 32>>() == align_of::<&'static mut [f32]>());
    assert!(align_of::<AlignedBuffer16<i8, 64>>() == 16);
    assert!(align_of::<AlignedBuffer32<f32, 32>>() == 32);
    assert!(align_of::<AlignedBuffer64<u8, 64>>() == 64);
    assert!(align_of::<AlignedBuffer128<u8, 128>>() == 128);
    assert!(align_of::<AlignedBuffer256<u8, 256>>() == 256);
    assert!(align_of::<AlignedBuffer512<u8, 512>>() == 512);
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liminal::{PhaseQ7, MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS};

    #[test]
    fn test_infallible_aligned_buffers_range_indexing_and_upgrade() {
        let mut buf64 = AlignedBuffer64::<f32, 64>::splat(1.5);
        assert_eq!(&buf64[0..4], &[1.5f32; 4]);
        buf64[2..4].copy_from_slice(&[4.0, 5.0]);
        assert_eq!(buf64[2], 4.0);

        let slice = buf64.as_aligned_slice();
        assert_eq!(slice.align(), 64);
        assert_eq!(slice.len(), 64);
        assert_eq!(slice.byte_len(), 256);

        let mut slice_mut = buf64.as_aligned_slice_mut();
        slice_mut[0] = 3.0;
        let relaxed: AlignedSliceMut16<'_, f32> = slice_mut.relax_alignment::<16>();
        assert_eq!(relaxed[0], 3.0);

        let zero_buf = AlignedBuffer32::<f32, 16>::zeroed();
        assert_eq!(zero_buf[0], 0.0);
        let arr_buf = AlignedBuffer32::<f32, 16>::new([2.5f32; 16]);
        assert_eq!(arr_buf[15], 2.5);
    }

    #[test]
    fn test_disjoint_mut_pair_triple_and_ref_mut() {
        let mut buf_q = AlignedBuffer32::<f32, 32>::splat(1.0);
        let mut buf_k = AlignedBuffer32::<f32, 32>::splat(2.0);
        let mut buf_v = AlignedBuffer32::<f32, 32>::splat(3.0);

        let mut triple = DisjointMutTriple::new(
            buf_q.as_aligned_slice_mut(),
            buf_k.as_aligned_slice_mut(),
            buf_v.as_aligned_slice_mut(),
        )
        .unwrap();

        {
            let mut reborrowed = triple.reborrow();
            reborrowed.first[0] += reborrowed.second[0] + reborrowed.third[0];
        }
        assert_eq!(buf_q[0], 6.0);

        let mut ref_mut =
            DisjointRefMut::new(buf_k.as_aligned_slice(), buf_q.as_aligned_slice_mut()).unwrap();
        let (inp, out) = ref_mut.as_ref_mut();
        out[1] = inp[1] * 4.0;
        assert_eq!(buf_q[1], 8.0);
    }

    #[test]
    fn test_cast_and_split_at() {
        let mut buf = AlignedBuffer64::<f32, 32>::splat(1.0);
        let mut slice_mut = buf.as_aligned_slice_mut();
        let (mut left, mut right) = slice_mut.split_at_mut(16).unwrap();
        left[0] = 10.0;
        right[0] = 20.0;
        assert_eq!(buf[0], 10.0);
        assert_eq!(buf[16], 20.0);

        let slice_u8: AlignedSlice64<'_, u8> = buf.as_aligned_slice().try_cast::<u8>().unwrap();
        assert_eq!(slice_u8.len(), 128);
    }

    #[test]
    fn test_attractor_slot_bank_triple_and_micro_swarm_chunks() {
        let mut multi_slot_bank =
            AlignedBuffer64::<PhaseQ7, { MAX_ATTRACTOR_DOMAINS * MAX_ATTENTION_HEADS }>::zeroed();

        {
            let bank_2d_mut = multi_slot_bank
                .as_slot_bank_mut::<MAX_ATTRACTOR_DOMAINS, MAX_ATTENTION_HEADS>()
                .unwrap();
            bank_2d_mut[0][0] = 12;
        }
        assert_eq!(multi_slot_bank[0], 12);

        {
            let k_slot = multi_slot_bank
                .slot_exact_mut::<MAX_ATTENTION_HEADS>(2)
                .unwrap();
            k_slot.fill(56);
        }

        let bank_2d = multi_slot_bank
            .as_slot_bank::<MAX_ATTRACTOR_DOMAINS, MAX_ATTENTION_HEADS>()
            .unwrap();
        assert_eq!(bank_2d[0][0], 12);
        assert_eq!(bank_2d[2][0], 56);

        let imm_slice = multi_slot_bank.as_aligned_slice();
        let k_ref = imm_slice.slot_exact::<MAX_ATTENTION_HEADS>(2).unwrap();
        assert_eq!(k_ref[0], 56);
    }
}
