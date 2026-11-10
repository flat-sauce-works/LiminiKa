// name: src/core/src/traits/aligned.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lv.0 Foundation 2: Const-generic zero-copy hardware-aligned slice wrappers, type-level
//! non-aliasing (`GCSO_RESTRICT`) guards, $\mathcal{O}(1)$ Attractor Slot / 2D Slot Bank /
//! Micro-Swarm cell accessors, and stack-allocated hardware-aligned buffers (`16..512`B)
//! for the Geometric Cellular Sheaf Orchestrator (GCSO).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::inline_always)]
#![allow(clippy::manual_slice_size_calculation)]

use core::marker::PhantomData;
use core::mem::{align_of, size_of, size_of_val};
use core::ops::{Deref, DerefMut, Index, IndexMut, Range};
use core::ptr::NonNull;
use core::slice::{self, SliceIndex};

use super::primitives::{
    raw_byte_ranges_overlap, unpack_attractor_slot_from_tag, GcsoResult, Pod, MAX_ATTRACTOR_DOMAINS,
};
use crate::abi::{
    gcso_tagged_ptr_t, GcsoStatus, GCSO_ERROR_INVALID_ARGUMENT, GCSO_ERROR_MISALIGNED_POINTER,
    GCSO_ERROR_NULL_POINTER,
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

/// Shared constant-time validator for `try_cast_aligned` / `into_cast_aligned_mut`.
#[inline(always)]
const fn validate_cast_layout<U: Pod, const SRC_ALIGN: usize, const TARGET_ALIGN: usize>(
    base_addr: usize,
    total_bytes: usize,
) -> GcsoResult<usize> {
    let () = TypeAlignCheck::<U, TARGET_ALIGN>::VALID;
    let target_elem_size = size_of::<U>();
    if (target_elem_size == 0) | ((total_bytes % target_elem_size) != 0) {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    if (total_bytes > 0) & (TARGET_ALIGN > SRC_ALIGN) & ((base_addr & (TARGET_ALIGN - 1)) != 0) {
        return Err(GCSO_ERROR_MISALIGNED_POINTER);
    }
    Ok(total_bytes / target_elem_size)
}

/// Shared constant-time validator for `slot_exact` / `into_slot_exact_mut` with compile-time
/// alignment branch elision when `(SLOT_LEN * size_of::<T>()) % ALIGN == 0`.
#[inline(always)]
const fn validate_slot_offset<T, const ALIGN: usize, const SLOT_LEN: usize>(
    base_addr: usize,
    slice_len: usize,
    slot_idx: usize,
) -> GcsoResult<usize> {
    let () = TypeAlignCheck::<T, ALIGN>::VALID;
    let () = ChunkCheck::<SLOT_LEN>::VALID;
    if slot_idx >= (slice_len / SLOT_LEN) {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    let offset = slot_idx * SLOT_LEN;
    let slot_addr = base_addr.wrapping_add(offset * size_of::<T>());
    if (((SLOT_LEN * size_of::<T>()) & (ALIGN - 1)) != 0) & ((slot_addr & (ALIGN - 1)) != 0) {
        return Err(GCSO_ERROR_MISALIGNED_POINTER);
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
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `slice` is non-empty and its base address
    /// is not a multiple of `ALIGN`.
    #[inline(always)]
    pub fn new(slice: &'a [T]) -> GcsoResult<Self> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        let is_valid = slice.is_empty() | (((slice.as_ptr() as usize) & (ALIGN - 1)) == 0);
        if is_valid {
            Ok(Self { slice })
        } else {
            Err(GCSO_ERROR_MISALIGNED_POINTER)
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
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the reference address is not `ALIGN`-byte aligned.
    #[inline(always)]
    pub fn from_ref(value: &'a T) -> GcsoResult<Self> {
        Self::new(slice::from_ref(value))
    }

    /// Safely constructs an `AlignedSlice` from a raw pointer and element count across FFI boundaries.
    ///
    /// # Safety
    /// The caller must guarantee that `ptr` points to at least `len` valid initialized instances of `T`,
    /// that the total byte size does not exceed `isize::MAX`, and that the memory is not mutated for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_NULL_POINTER`] if `ptr` is null when `len > 0`.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `ptr` violates the `ALIGN` byte boundary.
    #[inline(always)]
    pub unsafe fn from_raw_parts(ptr: *const T, len: usize) -> GcsoResult<Self> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        if len == 0 {
            return Ok(Self::empty());
        }
        if ptr.is_null() {
            return Err(GCSO_ERROR_NULL_POINTER);
        }
        if ((ptr as usize) & (ALIGN - 1)) != 0 {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        // SAFETY: Pointer non-nullness and alignment (>= align_of::<T>()) verified above;
        // caller guarantees `len` initialized elements for lifetime `'a`.
        let slice = unsafe { slice::from_raw_parts(ptr, len) };
        Ok(Self { slice })
    }

    /// Safely constructs an optional `AlignedSlice` from a potentially null FFI pointer.
    ///
    /// # Safety
    /// If `ptr` is non-null, caller must guarantee it points to `len` valid initialized instances of `T`
    /// for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `ptr` is non-null and violates `ALIGN` when `len > 0`.
    #[inline(always)]
    pub unsafe fn from_raw_parts_opt(ptr: *const T, len: usize) -> GcsoResult<Option<Self>> {
        if ptr.is_null() {
            Ok(None)
        } else {
            // SAFETY: Non-null pointer is delegated directly to `from_raw_parts`.
            unsafe { Self::from_raw_parts(ptr, len).map(Some) }
        }
    }

    /// Safely reinterprets the aligned slice as a slice of another `Pod` type `U` with the same `ALIGN`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `U` is zero-sized or the total byte length of `self`
    /// is not an exact multiple of `size_of::<U>()`.
    #[inline(always)]
    pub fn try_cast<U: Pod>(&self) -> GcsoResult<AlignedSlice<'a, U, ALIGN>>
    where
        T: Pod,
    {
        self.try_cast_aligned::<U, ALIGN>()
    }

    /// Safely reinterprets the aligned slice as a slice of `U: Pod` with a target alignment `TARGET_ALIGN`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `U` is zero-sized or total byte length is not a multiple of `size_of::<U>()`.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `TARGET_ALIGN > ALIGN` and the pointer does not satisfy `TARGET_ALIGN`.
    #[inline(always)]
    pub fn try_cast_aligned<U: Pod, const TARGET_ALIGN: usize>(
        &self,
    ) -> GcsoResult<AlignedSlice<'a, U, TARGET_ALIGN>>
    where
        T: Pod,
    {
        let new_len = validate_cast_layout::<U, ALIGN, TARGET_ALIGN>(
            self.slice.as_ptr() as usize,
            size_of_val(self.slice),
        )?;
        if new_len == 0 {
            return Ok(AlignedSlice::<U, TARGET_ALIGN>::empty());
        }
        // SAFETY: Both `T` and `U` are `Pod`, and `TARGET_ALIGN` (>= align_of::<U>()) is verified above.
        let slice = unsafe { slice::from_raw_parts(self.slice.as_ptr().cast::<U>(), new_len) };
        Ok(AlignedSlice { slice })
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

    /// Attempts to upgrade the alignment guarantee to a stricter power-of-two `SUPER_ALIGN`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `SUPER_ALIGN > ALIGN` and the underlying pointer
    /// does not satisfy `SUPER_ALIGN`.
    #[inline(always)]
    pub fn try_upgrade_alignment<const SUPER_ALIGN: usize>(
        self,
    ) -> GcsoResult<AlignedSlice<'a, T, SUPER_ALIGN>> {
        let () = TypeAlignCheck::<T, SUPER_ALIGN>::VALID;
        let is_valid = (SUPER_ALIGN <= ALIGN)
            | self.slice.is_empty()
            | (((self.slice.as_ptr() as usize) & (SUPER_ALIGN - 1)) == 0);
        if is_valid {
            Ok(AlignedSlice { slice: self.slice })
        } else {
            Err(GCSO_ERROR_MISALIGNED_POINTER)
        }
    }

    /// Returns the underlying immutable slice reference.
    #[inline(always)]
    #[must_use]
    pub const fn as_slice(&self) -> &'a [T] {
        self.slice
    }

    /// Consumes the wrapper and returns the underlying immutable slice reference with lifetime `'a`.
    #[inline(always)]
    #[must_use]
    pub const fn into_slice(self) -> &'a [T] {
        self.slice
    }

    /// Views the slice as a fixed-size array reference `&'a [T; N]` at offset `0`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `self.len() != N`.
    #[inline(always)]
    pub const fn as_array<const N: usize>(&self) -> GcsoResult<&'a [T; N]> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        if self.slice.len() != N {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: `self.slice.len() == N` and base address is guaranteed aligned to `ALIGN >= align_of::<T>()`.
        Ok(unsafe { &*self.slice.as_ptr().cast::<[T; N]>() })
    }

    /// Views the prefix of the slice as a 2D slot bank `&'a [[T; SLOT_LEN]; SLOTS]` in $\mathcal{O}(1)$ zero-copy mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `self.len() < SLOTS * SLOT_LEN`.
    #[inline(always)]
    pub const fn as_slot_bank<const SLOTS: usize, const SLOT_LEN: usize>(
        &self,
    ) -> GcsoResult<&'a [[T; SLOT_LEN]; SLOTS]> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        let () = SlotBankCheck::<SLOTS, SLOT_LEN>::VALID;
        if self.slice.len() < SLOTS * SLOT_LEN {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
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

    /// Returns a non-null pointer to the underlying buffer, or `None` if the slice is empty.
    #[inline(always)]
    #[must_use]
    pub fn as_non_null_ptr(&self) -> Option<NonNull<T>> {
        if self.slice.is_empty() {
            None
        } else {
            NonNull::new(self.slice.as_ptr().cast_mut())
        }
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

    /// Enforces `GCSO_RESTRICT` non-aliasing invariant against another slice.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if the two slices overlap in memory.
    #[inline(always)]
    pub fn ensure_disjoint<const OTHER_ALIGN: usize>(
        &self,
        other: &AlignedSlice<'_, T, OTHER_ALIGN>,
    ) -> GcsoResult<()> {
        if self.overlaps(other) {
            Err(GCSO_ERROR_INVALID_ARGUMENT)
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

    /// Returns non-overlapping SIMD chunks of fixed size `N` along with remainder,
    /// enforcing `N > 0` at compile time without runtime panic branches.
    #[inline(always)]
    #[must_use]
    pub const fn as_chunks<const N: usize>(&self) -> (&'a [[T; N]], &'a [T]) {
        let () = ChunkCheck::<N>::VALID;
        let len = self.slice.len();
        let num_chunks = len / N;
        let main_len = num_chunks * N;
        // SAFETY: `N > 0` is verified at compile time, `main_len <= len`, and `[T; N]` has identical layout.
        unsafe {
            let base = self.slice.as_ptr();
            (
                slice::from_raw_parts(base.cast::<[T; N]>(), num_chunks),
                slice::from_raw_parts(base.add(main_len), len - main_len),
            )
        }
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

    /// Branchlessly checks if a raw pointer satisfies the const alignment requirement and is non-null.
    #[inline(always)]
    #[must_use]
    pub fn is_aligned_ptr(ptr: *const T) -> bool {
        (ALIGN != 0)
            & ALIGN.is_power_of_two()
            & (!ptr.is_null())
            & (((ptr as usize) & (ALIGN - 1)) == 0)
    }

    /// Creates an aligned subslice if the subslice start address satisfies `ALIGN` constraints.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if the index range is out of bounds.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the subslice start address violates `ALIGN`.
    #[inline(always)]
    pub fn subslice<R>(&self, range: R) -> GcsoResult<Self>
    where
        R: SliceIndex<[T], Output = [T]>,
    {
        Self::new(self.slice.get(range).ok_or(GCSO_ERROR_INVALID_ARGUMENT)?)
    }

    /// Extracts an aligned fixed-size sub-array `&'a [T; N]` starting at element `offset`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `offset + N` overflows or exceeds slice length.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the starting address at `offset` violates `ALIGN`.
    #[inline(always)]
    pub fn subslice_exact<const N: usize>(&self, offset: usize) -> GcsoResult<&'a [T; N]> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        let end = offset.checked_add(N).ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if end > self.slice.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: `offset + N <= self.slice.len()` verified above.
        let ptr = unsafe { self.slice.as_ptr().add(offset) };
        if (N > 0) & (((ptr as usize) & (ALIGN - 1)) != 0) {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        // SAFETY: Bounds and `ALIGN` (>= align_of::<T>()) checked above.
        Ok(unsafe { &*ptr.cast::<[T; N]>() })
    }

    /// Extracts a fixed-size chunk `&'a [T; CHUNK_LEN]` at `chunk_idx` (`offset = chunk_idx * CHUNK_LEN`)
    /// requiring only native element alignment `align_of::<T>()` in $\mathcal{O}(1)$ `const fn` mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `(chunk_idx + 1) * CHUNK_LEN` exceeds slice length.
    #[inline(always)]
    pub const fn chunk_exact<const CHUNK_LEN: usize>(
        &self,
        chunk_idx: usize,
    ) -> GcsoResult<&'a [T; CHUNK_LEN]> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        let () = ChunkCheck::<CHUNK_LEN>::VALID;
        if chunk_idx >= (self.slice.len() / CHUNK_LEN) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: `chunk_idx < self.slice.len() / CHUNK_LEN` guarantees in-bounds access aligned to `align_of::<T>()`.
        Ok(unsafe {
            &*self
                .slice
                .as_ptr()
                .add(chunk_idx * CHUNK_LEN)
                .cast::<[T; CHUNK_LEN]>()
        })
    }

    /// Extracts two fixed-size chunks `(&'a [T; CHUNK_LEN], &'a [T; CHUNK_LEN])` at `chunk_a` and `chunk_b`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if either chunk index is out of bounds.
    #[inline(always)]
    pub const fn chunk_pair_exact<const CHUNK_LEN: usize>(
        &self,
        chunk_a: usize,
        chunk_b: usize,
    ) -> GcsoResult<(&'a [T; CHUNK_LEN], &'a [T; CHUNK_LEN])> {
        let a = match self.chunk_exact::<CHUNK_LEN>(chunk_a) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        let b = match self.chunk_exact::<CHUNK_LEN>(chunk_b) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        Ok((a, b))
    }

    /// Extracts a fixed-capacity slot `&'a [T; SLOT_LEN]` at `slot_idx` (`offset = slot_idx * SLOT_LEN`)
    /// in $\mathcal{O}(1)$ zero-copy mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `(slot_idx + 1) * SLOT_LEN` exceeds slice length.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the slot start address violates `ALIGN`.
    #[inline(always)]
    pub fn slot_exact<const SLOT_LEN: usize>(
        &self,
        slot_idx: usize,
    ) -> GcsoResult<&'a [T; SLOT_LEN]> {
        let offset = validate_slot_offset::<T, ALIGN, SLOT_LEN>(
            self.slice.as_ptr() as usize,
            self.slice.len(),
            slot_idx,
        )?;
        // SAFETY: Bounds and `ALIGN` (>= align_of::<T>()) verified by `validate_slot_offset`.
        Ok(unsafe { &*self.slice.as_ptr().add(offset).cast::<[T; SLOT_LEN]>() })
    }

    /// Extracts a fixed-capacity slot at `slot_idx` directly as an `AlignedSlice<'a, T, ALIGN>`
    /// of length `SLOT_LEN` in $\mathcal{O}(1)$ zero-copy mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `(slot_idx + 1) * SLOT_LEN` exceeds slice length.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the slot start address violates `ALIGN`.
    #[inline(always)]
    pub fn slot_aligned<const SLOT_LEN: usize>(&self, slot_idx: usize) -> GcsoResult<Self> {
        let slot_arr = self.slot_exact::<SLOT_LEN>(slot_idx)?;
        // SAFETY: `slot_exact` already verified `ALIGN` on the slot's base address.
        Ok(unsafe { Self::new_unchecked(slot_arr.as_slice()) })
    }

    /// Extracts two fixed-capacity slots `(&'a [T; SLOT_LEN], &'a [T; SLOT_LEN])` for `(slot_a, slot_b)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if either slot index is out of bounds.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if either slot start address violates `ALIGN`.
    #[inline(always)]
    pub fn slot_pair_exact<const SLOT_LEN: usize>(
        &self,
        slot_a: usize,
        slot_b: usize,
    ) -> GcsoResult<(&'a [T; SLOT_LEN], &'a [T; SLOT_LEN])> {
        Ok((
            self.slot_exact::<SLOT_LEN>(slot_a)?,
            self.slot_exact::<SLOT_LEN>(slot_b)?,
        ))
    }

    /// Extracts three fixed-capacity slots `(&'a [T; SLOT_LEN], &'a [T; SLOT_LEN], &'a [T; SLOT_LEN])`
    /// for `(slot_a, slot_b, slot_c)` in $\mathcal{O}(1)$ zero-copy mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any slot index is out of bounds.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if any slot start address violates `ALIGN`.
    #[inline(always)]
    pub fn slot_triple_exact<const SLOT_LEN: usize>(
        &self,
        slot_a: usize,
        slot_b: usize,
        slot_c: usize,
    ) -> GcsoResult<(&'a [T; SLOT_LEN], &'a [T; SLOT_LEN], &'a [T; SLOT_LEN])> {
        Ok((
            self.slot_exact::<SLOT_LEN>(slot_a)?,
            self.slot_exact::<SLOT_LEN>(slot_b)?,
            self.slot_exact::<SLOT_LEN>(slot_c)?,
        ))
    }

    /// Extracts the fixed-capacity Attractor Slot `&'a [T; SLOT_LEN]` directly addressed by the
    /// 3-bit Target Attractor Slot ID (`bits 53..=55`, `0..=7`) embedded inside `tagged_ptr`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if the slice holds fewer than `(slot_id + 1) * SLOT_LEN` elements.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the target slot address violates `ALIGN`.
    #[inline(always)]
    pub fn attractor_slot_by_ptr<const SLOT_LEN: usize>(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
    ) -> GcsoResult<&'a [T; SLOT_LEN]> {
        let slot_id = unpack_attractor_slot_from_tag(tagged_ptr.unpack_tag()) as usize;
        debug_assert!(slot_id < MAX_ATTRACTOR_DOMAINS);
        self.slot_exact::<SLOT_LEN>(slot_id)
    }

    /// Extracts the fixed-capacity Attractor Slot addressed by `tagged_ptr` directly as an
    /// `AlignedSlice<'a, T, ALIGN>` of length `SLOT_LEN` without redundant alignment checks.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if the slice holds fewer than `(slot_id + 1) * SLOT_LEN` elements.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the target slot address violates `ALIGN`.
    #[inline(always)]
    pub fn attractor_slot_slice_by_ptr<const SLOT_LEN: usize>(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
    ) -> GcsoResult<Self> {
        let slot_arr = self.attractor_slot_by_ptr::<SLOT_LEN>(tagged_ptr)?;
        // SAFETY: `attractor_slot_by_ptr` already verified `ALIGN` on the slot's base address.
        Ok(unsafe { Self::new_unchecked(slot_arr.as_slice()) })
    }

    /// Splits the slice at `mid` if the right subslice satisfies `ALIGN` constraints.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `mid` exceeds slice length.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the right subslice start address violates `ALIGN`.
    #[inline(always)]
    pub fn split_at(&self, mid: usize) -> GcsoResult<(Self, Self)> {
        if mid > self.slice.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (left, right) = self.slice.split_at(mid);
        // SAFETY: `left` starts at `self.slice.as_ptr()` which is already aligned to `ALIGN`.
        Ok((unsafe { Self::new_unchecked(left) }, Self::new(right)?))
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
    type Error = GcsoStatus;
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
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `slice` is non-empty and its base address
    /// is not a multiple of `ALIGN`.
    #[inline(always)]
    pub fn new(slice: &'a mut [T]) -> GcsoResult<Self> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        let is_valid = slice.is_empty() | (((slice.as_ptr() as usize) & (ALIGN - 1)) == 0);
        if is_valid {
            Ok(Self { slice })
        } else {
            Err(GCSO_ERROR_MISALIGNED_POINTER)
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

    /// Creates a new mutable aligned slice wrapper from a single mutable reference after validating alignment.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the reference address is not `ALIGN`-byte aligned.
    #[inline(always)]
    pub fn from_mut(value: &'a mut T) -> GcsoResult<Self> {
        Self::new(slice::from_mut(value))
    }

    /// Safely constructs an `AlignedSliceMut` from a raw mutable pointer and element count across FFI boundaries.
    ///
    /// # Safety
    /// The caller must guarantee that `ptr` points to at least `len` valid initialized instances of `T`,
    /// that total byte size does not exceed `isize::MAX`, and that no aliasing pointers access the
    /// memory for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_NULL_POINTER`] if `ptr` is null when `len > 0`.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `ptr` violates the `ALIGN` byte boundary.
    #[inline(always)]
    pub unsafe fn from_raw_parts_mut(ptr: *mut T, len: usize) -> GcsoResult<Self> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        if len == 0 {
            return Ok(Self::empty());
        }
        if ptr.is_null() {
            return Err(GCSO_ERROR_NULL_POINTER);
        }
        if ((ptr as usize) & (ALIGN - 1)) != 0 {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
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
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `ptr` is non-null and misaligned when `len > 0`.
    #[inline(always)]
    pub unsafe fn from_raw_parts_mut_opt(ptr: *mut T, len: usize) -> GcsoResult<Option<Self>> {
        if ptr.is_null() {
            Ok(None)
        } else {
            // SAFETY: Non-null pointer is delegated directly to `from_raw_parts_mut`.
            unsafe { Self::from_raw_parts_mut(ptr, len).map(Some) }
        }
    }

    /// Consumes `self` and safely reinterprets the mutable aligned slice as a mutable slice of
    /// another `Pod` type `U` with lifetime `'a` while maintaining `ALIGN` invariants.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `U` is zero-sized or total byte length is not a multiple of `size_of::<U>()`.
    #[inline(always)]
    pub fn into_cast_mut<U: Pod>(self) -> GcsoResult<AlignedSliceMut<'a, U, ALIGN>>
    where
        T: Pod,
    {
        self.into_cast_aligned_mut::<U, ALIGN>()
    }

    /// Safely reinterprets the mutable aligned slice as a mutable slice of another `Pod` type `U`
    /// while maintaining `ALIGN` invariants.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `U` is zero-sized or total byte length is not a multiple of `size_of::<U>()`.
    #[inline(always)]
    pub fn try_cast_mut<U: Pod>(&mut self) -> GcsoResult<AlignedSliceMut<'_, U, ALIGN>>
    where
        T: Pod,
    {
        self.reborrow().into_cast_aligned_mut::<U, ALIGN>()
    }

    /// Consumes `self` and safely reinterprets the mutable aligned slice as a mutable slice of
    /// `U: Pod` with `TARGET_ALIGN` and lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `U` is zero-sized or total byte length is not a multiple of `size_of::<U>()`.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `TARGET_ALIGN > ALIGN` and pointer is misaligned.
    #[inline(always)]
    pub fn into_cast_aligned_mut<U: Pod, const TARGET_ALIGN: usize>(
        self,
    ) -> GcsoResult<AlignedSliceMut<'a, U, TARGET_ALIGN>>
    where
        T: Pod,
    {
        let new_len = validate_cast_layout::<U, ALIGN, TARGET_ALIGN>(
            self.slice.as_ptr() as usize,
            size_of_val(self.slice),
        )?;
        if new_len == 0 {
            return Ok(AlignedSliceMut::<U, TARGET_ALIGN>::empty());
        }
        let ptr = self.slice.as_mut_ptr().cast::<U>();
        // SAFETY: Both `T` and `U` are `Pod`, uniquely borrowed for `'a`, and `TARGET_ALIGN` is verified above.
        let slice = unsafe { slice::from_raw_parts_mut(ptr, new_len) };
        Ok(AlignedSliceMut { slice })
    }

    /// Safely reinterprets the mutable aligned slice as a mutable slice of `U: Pod` with `TARGET_ALIGN`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `U` is zero-sized or total byte length is not a multiple of `size_of::<U>()`.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `TARGET_ALIGN > ALIGN` and pointer is misaligned.
    #[inline(always)]
    pub fn try_cast_aligned_mut<U: Pod, const TARGET_ALIGN: usize>(
        &mut self,
    ) -> GcsoResult<AlignedSliceMut<'_, U, TARGET_ALIGN>>
    where
        T: Pod,
    {
        self.reborrow().into_cast_aligned_mut::<U, TARGET_ALIGN>()
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

    /// Attempts to upgrade the alignment guarantee to a stricter power-of-two `SUPER_ALIGN`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `SUPER_ALIGN > ALIGN` and the underlying pointer
    /// does not satisfy `SUPER_ALIGN`.
    #[inline(always)]
    pub fn try_upgrade_alignment<const SUPER_ALIGN: usize>(
        self,
    ) -> GcsoResult<AlignedSliceMut<'a, T, SUPER_ALIGN>> {
        let () = TypeAlignCheck::<T, SUPER_ALIGN>::VALID;
        let is_valid = (SUPER_ALIGN <= ALIGN)
            | self.slice.is_empty()
            | (((self.slice.as_ptr() as usize) & (SUPER_ALIGN - 1)) == 0);
        if is_valid {
            Ok(AlignedSliceMut { slice: self.slice })
        } else {
            Err(GCSO_ERROR_MISALIGNED_POINTER)
        }
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

    /// Consumes the wrapper and returns the underlying mutable slice reference with lifetime `'a`.
    #[inline(always)]
    #[must_use]
    pub fn into_mut_slice(self) -> &'a mut [T] {
        self.slice
    }

    /// Views the slice as an immutable fixed-size array reference `&[T; N]` at offset `0`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `self.len() != N`.
    #[inline(always)]
    pub const fn as_array<const N: usize>(&self) -> GcsoResult<&[T; N]> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        if self.slice.len() != N {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: `self.slice.len() == N` and base address is guaranteed aligned to `ALIGN >= align_of::<T>()`.
        Ok(unsafe { &*self.slice.as_ptr().cast::<[T; N]>() })
    }

    /// Consumes `self` and converts it into a mutable fixed-size array reference `&'a mut [T; N]`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `self.len() != N`.
    #[inline(always)]
    pub fn into_array_mut<const N: usize>(self) -> GcsoResult<&'a mut [T; N]> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        if self.slice.len() != N {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: `self.slice.len() == N`, uniquely owned for `'a`, and aligned to `ALIGN >= align_of::<T>()`.
        Ok(unsafe { &mut *self.slice.as_mut_ptr().cast::<[T; N]>() })
    }

    /// Views the slice as a mutable fixed-size array reference `&mut [T; N]` at offset `0`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `self.len() != N`.
    #[inline(always)]
    pub fn as_array_mut<const N: usize>(&mut self) -> GcsoResult<&mut [T; N]> {
        self.reborrow().into_array_mut::<N>()
    }

    /// Views the prefix of the slice as a 2D slot bank `&[[T; SLOT_LEN]; SLOTS]` in $\mathcal{O}(1)$ zero-copy mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `self.len() < SLOTS * SLOT_LEN`.
    #[inline(always)]
    pub const fn as_slot_bank<const SLOTS: usize, const SLOT_LEN: usize>(
        &self,
    ) -> GcsoResult<&[[T; SLOT_LEN]; SLOTS]> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        let () = SlotBankCheck::<SLOTS, SLOT_LEN>::VALID;
        if self.slice.len() < SLOTS * SLOT_LEN {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: `self.slice.len() >= SLOTS * SLOT_LEN` and aligned to `ALIGN >= align_of::<T>()`.
        Ok(unsafe { &*self.slice.as_ptr().cast::<[[T; SLOT_LEN]; SLOTS]>() })
    }

    /// Consumes `self` and views the prefix of the mutable slice as a mutable 2D slot bank
    /// `&'a mut [[T; SLOT_LEN]; SLOTS]` with lifetime `'a` in $\mathcal{O}(1)$ zero-copy mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `self.len() < SLOTS * SLOT_LEN`.
    #[inline(always)]
    pub fn into_slot_bank_mut<const SLOTS: usize, const SLOT_LEN: usize>(
        self,
    ) -> GcsoResult<&'a mut [[T; SLOT_LEN]; SLOTS]> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        let () = SlotBankCheck::<SLOTS, SLOT_LEN>::VALID;
        if self.slice.len() < SLOTS * SLOT_LEN {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: `self.slice.len() >= SLOTS * SLOT_LEN`, uniquely borrowed for `'a`, and aligned to `ALIGN >= align_of::<T>()`.
        Ok(unsafe { &mut *self.slice.as_mut_ptr().cast::<[[T; SLOT_LEN]; SLOTS]>() })
    }

    /// Views the prefix of the mutable slice as a mutable 2D slot bank `&mut [[T; SLOT_LEN]; SLOTS]`
    /// in $\mathcal{O}(1)$ zero-copy mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `self.len() < SLOTS * SLOT_LEN`.
    #[inline(always)]
    pub fn as_slot_bank_mut<const SLOTS: usize, const SLOT_LEN: usize>(
        &mut self,
    ) -> GcsoResult<&mut [[T; SLOT_LEN]; SLOTS]> {
        self.reborrow().into_slot_bank_mut::<SLOTS, SLOT_LEN>()
    }

    /// Reborrows the mutable slice as an immutable `AlignedSlice`.
    #[inline(always)]
    #[must_use]
    pub const fn as_aligned_slice(&self) -> AlignedSlice<'_, T, ALIGN> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        AlignedSlice { slice: self.slice }
    }

    /// Consumes `self` and converts it into an immutable `AlignedSlice` with lifetime `'a`.
    #[inline(always)]
    #[must_use]
    pub fn into_aligned_slice(self) -> AlignedSlice<'a, T, ALIGN> {
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

    /// Reborrows `self` for a shorter lifetime with a relaxed power-of-two alignment `SUB_ALIGN`.
    #[inline(always)]
    #[must_use]
    pub fn reborrow_relaxed<const SUB_ALIGN: usize>(
        &mut self,
    ) -> AlignedSliceMut<'_, T, SUB_ALIGN> {
        let () = TypeAlignCheck::<T, SUB_ALIGN>::VALID;
        assert!(
            SUB_ALIGN <= ALIGN,
            "Target SUB_ALIGN must be less than or equal to source ALIGN"
        );
        AlignedSliceMut {
            slice: &mut *self.slice,
        }
    }

    /// Enforces `GCSO_RESTRICT` non-aliasing invariant against another immutable aligned slice.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if memory ranges overlap.
    #[inline(always)]
    pub fn ensure_disjoint<const OTHER_ALIGN: usize>(
        &self,
        other: &AlignedSlice<'_, T, OTHER_ALIGN>,
    ) -> GcsoResult<()> {
        self.as_aligned_slice().ensure_disjoint(other)
    }

    /// Enforces `GCSO_RESTRICT` non-aliasing invariant against another mutable aligned slice.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if memory ranges overlap.
    #[inline(always)]
    pub fn ensure_disjoint_mut<const OTHER_ALIGN: usize>(
        &self,
        other: &AlignedSliceMut<'_, T, OTHER_ALIGN>,
    ) -> GcsoResult<()> {
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

    /// Returns a non-null mutable pointer to the underlying buffer, or `None` if the slice is empty.
    #[inline(always)]
    #[must_use]
    pub fn as_non_null_mut_ptr(&mut self) -> Option<NonNull<T>> {
        if self.slice.is_empty() {
            None
        } else {
            NonNull::new(self.slice.as_mut_ptr())
        }
    }

    /// Returns the start and end mutable raw pointer range for memory boundary checks.
    #[inline(always)]
    #[must_use]
    pub fn as_mut_ptr_range(&mut self) -> Range<*mut T> {
        self.slice.as_mut_ptr_range()
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

    /// Returns a mutable reference to the first element of the slice, or `None` if empty.
    #[inline(always)]
    #[must_use]
    pub fn first_mut(&mut self) -> Option<&mut T> {
        self.slice.first_mut()
    }

    /// Returns the last element of the slice, or `None` if empty.
    #[inline(always)]
    #[must_use]
    pub const fn last(&self) -> Option<&T> {
        self.slice.last()
    }

    /// Returns a mutable reference to the last element of the slice, or `None` if empty.
    #[inline(always)]
    #[must_use]
    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.slice.last_mut()
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

    /// Consumes `self` and returns non-overlapping mutable SIMD chunks of fixed size `N` along
    /// with remainder tied to lifetime `'a`, enforcing `N > 0` at compile time.
    #[inline(always)]
    #[must_use]
    pub fn into_chunks_mut<const N: usize>(self) -> (&'a mut [[T; N]], &'a mut [T]) {
        let () = ChunkCheck::<N>::VALID;
        let len = self.slice.len();
        let num_chunks = len / N;
        let main_len = num_chunks * N;
        // SAFETY: `N > 0` is verified at compile time, `main_len <= len`, and `[T; N]` has identical layout.
        unsafe {
            let base = self.slice.as_mut_ptr();
            (
                slice::from_raw_parts_mut(base.cast::<[T; N]>(), num_chunks),
                slice::from_raw_parts_mut(base.add(main_len), len - main_len),
            )
        }
    }

    /// Returns non-overlapping mutable SIMD chunks of fixed size `N` along with remainder,
    /// enforcing `N > 0` at compile time without runtime panic branches.
    #[inline(always)]
    #[must_use]
    pub fn as_chunks_mut<const N: usize>(&mut self) -> (&mut [[T; N]], &mut [T]) {
        self.reborrow().into_chunks_mut::<N>()
    }

    /// Fills the aligned mutable slice with a uniform value in zero-allocation mode.
    #[inline(always)]
    pub fn fill(&mut self, value: T)
    where
        T: Copy,
    {
        self.slice.fill(value);
    }

    /// Fills the aligned slice with default zero bytes in zero-allocation mode.
    #[inline(always)]
    pub fn zero_out(&mut self)
    where
        T: Pod,
    {
        let byte_len = size_of_val(self.slice);
        if byte_len > 0 {
            // SAFETY: `T: Pod` guarantees that all-zero byte patterns represent valid instances of `T`.
            unsafe {
                core::ptr::write_bytes(self.slice.as_mut_ptr().cast::<u8>(), 0, byte_len);
            }
        }
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

    /// Branchlessly checks if a raw pointer satisfies the const alignment requirement and is non-null.
    #[inline(always)]
    #[must_use]
    pub fn is_aligned_ptr(ptr: *const T) -> bool {
        AlignedSlice::<T, ALIGN>::is_aligned_ptr(ptr)
    }

    /// Consumes `self` and creates a mutable subslice with lifetime `'a` if the subslice start
    /// address satisfies `ALIGN` constraints.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if index range is out of bounds.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if subslice start address is misaligned.
    #[inline(always)]
    pub fn into_subslice_mut<R>(self, range: R) -> GcsoResult<AlignedSliceMut<'a, T, ALIGN>>
    where
        R: SliceIndex<[T], Output = [T]>,
    {
        AlignedSliceMut::new(
            self.slice
                .get_mut(range)
                .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?,
        )
    }

    /// Creates a mutable subslice if the subslice start address satisfies `ALIGN` constraints.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if index range is out of bounds.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if subslice start address is misaligned.
    #[inline(always)]
    pub fn subslice_mut<R>(&mut self, range: R) -> GcsoResult<AlignedSliceMut<'_, T, ALIGN>>
    where
        R: SliceIndex<[T], Output = [T]>,
    {
        self.reborrow().into_subslice_mut(range)
    }

    /// Consumes `self` and extracts an aligned mutable fixed-size sub-array `&'a mut [T; N]`
    /// starting at `offset`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `offset + N` overflows or exceeds slice length.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the starting address at `offset` violates `ALIGN`.
    #[inline(always)]
    pub fn into_subslice_exact_mut<const N: usize>(
        self,
        offset: usize,
    ) -> GcsoResult<&'a mut [T; N]> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        let end = offset.checked_add(N).ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if end > self.slice.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: `offset + N <= self.slice.len()` verified above.
        let ptr = unsafe { self.slice.as_mut_ptr().add(offset) };
        if (N > 0) & (((ptr as usize) & (ALIGN - 1)) != 0) {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        // SAFETY: Bounds, unique borrow for `'a`, and `ALIGN` (>= align_of::<T>()) checked above.
        Ok(unsafe { &mut *ptr.cast::<[T; N]>() })
    }

    /// Extracts an aligned mutable fixed-size sub-array `&mut [T; N]` starting at `offset`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `offset + N` overflows or exceeds slice length.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the starting address at `offset` violates `ALIGN`.
    #[inline(always)]
    pub fn subslice_exact_mut<const N: usize>(&mut self, offset: usize) -> GcsoResult<&mut [T; N]> {
        self.reborrow().into_subslice_exact_mut::<N>(offset)
    }

    /// Consumes `self` and extracts a mutable fixed-size chunk `&'a mut [T; CHUNK_LEN]` at `chunk_idx`
    /// (`offset = chunk_idx * CHUNK_LEN`) requiring only native element alignment `align_of::<T>()`
    /// in $\mathcal{O}(1)$ zero-copy mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `(chunk_idx + 1) * CHUNK_LEN` exceeds slice length.
    #[inline(always)]
    pub fn into_chunk_exact_mut<const CHUNK_LEN: usize>(
        self,
        chunk_idx: usize,
    ) -> GcsoResult<&'a mut [T; CHUNK_LEN]> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        let () = ChunkCheck::<CHUNK_LEN>::VALID;
        if chunk_idx >= (self.slice.len() / CHUNK_LEN) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: `chunk_idx < self.slice.len() / CHUNK_LEN` guarantees in-bounds exclusive access aligned to `align_of::<T>()`.
        Ok(unsafe {
            &mut *self
                .slice
                .as_mut_ptr()
                .add(chunk_idx * CHUNK_LEN)
                .cast::<[T; CHUNK_LEN]>()
        })
    }

    /// Extracts a mutable fixed-size chunk `&mut [T; CHUNK_LEN]` at `chunk_idx`
    /// (`offset = chunk_idx * CHUNK_LEN`) requiring only native element alignment `align_of::<T>()`
    /// in $\mathcal{O}(1)$ zero-copy mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `(chunk_idx + 1) * CHUNK_LEN` exceeds slice length.
    #[inline(always)]
    pub fn chunk_exact_mut<const CHUNK_LEN: usize>(
        &mut self,
        chunk_idx: usize,
    ) -> GcsoResult<&mut [T; CHUNK_LEN]> {
        self.reborrow().into_chunk_exact_mut::<CHUNK_LEN>(chunk_idx)
    }

    /// Consumes `self` and simultaneously borrows two distinct, non-overlapping mutable chunks
    /// `(&'a mut [T; CHUNK_LEN], &'a mut [T; CHUNK_LEN])` at `chunk_a` and `chunk_b` (`chunk_a != chunk_b`)
    /// requiring only native element alignment `align_of::<T>()` in $\mathcal{O}(1)$ zero-copy mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `chunk_a == chunk_b` or either chunk index is out of bounds.
    #[inline(always)]
    pub fn into_disjoint_chunk_pair_exact_mut<const CHUNK_LEN: usize>(
        self,
        chunk_a: usize,
        chunk_b: usize,
    ) -> GcsoResult<(&'a mut [T; CHUNK_LEN], &'a mut [T; CHUNK_LEN])> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        let () = ChunkCheck::<CHUNK_LEN>::VALID;
        let max_chunks = self.slice.len() / CHUNK_LEN;
        if (chunk_a == chunk_b) | (chunk_a >= max_chunks) | (chunk_b >= max_chunks) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let base_ptr = self.slice.as_mut_ptr();
        // SAFETY: `chunk_a != chunk_b`, `CHUNK_LEN > 0`, and both indices `< max_chunks` guarantee
        // valid, strictly non-overlapping half-open element intervals aligned to `align_of::<T>()`.
        unsafe {
            Ok((
                &mut *base_ptr.add(chunk_a * CHUNK_LEN).cast::<[T; CHUNK_LEN]>(),
                &mut *base_ptr.add(chunk_b * CHUNK_LEN).cast::<[T; CHUNK_LEN]>(),
            ))
        }
    }

    /// Simultaneously borrows two distinct, non-overlapping mutable chunks
    /// `(&mut [T; CHUNK_LEN], &mut [T; CHUNK_LEN])` at `chunk_a` and `chunk_b` (`chunk_a != chunk_b`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `chunk_a == chunk_b` or either chunk index is out of bounds.
    #[inline(always)]
    pub fn disjoint_chunk_pair_exact_mut<const CHUNK_LEN: usize>(
        &mut self,
        chunk_a: usize,
        chunk_b: usize,
    ) -> GcsoResult<(&mut [T; CHUNK_LEN], &mut [T; CHUNK_LEN])> {
        self.reborrow()
            .into_disjoint_chunk_pair_exact_mut::<CHUNK_LEN>(chunk_a, chunk_b)
    }

    /// Consumes `self` and extracts a mutable fixed-capacity slot `&'a mut [T; SLOT_LEN]` at `slot_idx`
    /// (`offset = slot_idx * SLOT_LEN`) with lifetime `'a` in $\mathcal{O}(1)$ zero-copy mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `(slot_idx + 1) * SLOT_LEN` exceeds slice length.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the slot start address violates `ALIGN`.
    #[inline(always)]
    pub fn into_slot_exact_mut<const SLOT_LEN: usize>(
        self,
        slot_idx: usize,
    ) -> GcsoResult<&'a mut [T; SLOT_LEN]> {
        let offset = validate_slot_offset::<T, ALIGN, SLOT_LEN>(
            self.slice.as_ptr() as usize,
            self.slice.len(),
            slot_idx,
        )?;
        // SAFETY: Bounds, unique borrow for `'a`, and `ALIGN` (>= align_of::<T>()) verified by `validate_slot_offset`.
        Ok(unsafe { &mut *self.slice.as_mut_ptr().add(offset).cast::<[T; SLOT_LEN]>() })
    }

    /// Extracts a mutable fixed-capacity slot `&mut [T; SLOT_LEN]` at `slot_idx` (`offset = slot_idx * SLOT_LEN`)
    /// in $\mathcal{O}(1)$ zero-copy mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `(slot_idx + 1) * SLOT_LEN` exceeds slice length.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the slot start address violates `ALIGN`.
    #[inline(always)]
    pub fn slot_exact_mut<const SLOT_LEN: usize>(
        &mut self,
        slot_idx: usize,
    ) -> GcsoResult<&mut [T; SLOT_LEN]> {
        self.reborrow().into_slot_exact_mut::<SLOT_LEN>(slot_idx)
    }

    /// Consumes `self` and extracts a mutable fixed-capacity slot at `slot_idx` directly as an
    /// `AlignedSliceMut<'a, T, ALIGN>` of length `SLOT_LEN` in $\mathcal{O}(1)$ zero-copy mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `(slot_idx + 1) * SLOT_LEN` exceeds slice length.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the slot start address violates `ALIGN`.
    #[inline(always)]
    pub fn into_slot_aligned_mut<const SLOT_LEN: usize>(
        self,
        slot_idx: usize,
    ) -> GcsoResult<AlignedSliceMut<'a, T, ALIGN>> {
        let slot_arr = self.into_slot_exact_mut::<SLOT_LEN>(slot_idx)?;
        // SAFETY: `into_slot_exact_mut` already verified `ALIGN` on the slot's base address.
        Ok(unsafe { AlignedSliceMut::new_unchecked(slot_arr.as_mut_slice()) })
    }

    /// Extracts a mutable fixed-capacity slot at `slot_idx` directly as an `AlignedSliceMut<'_, T, ALIGN>`
    /// of length `SLOT_LEN` in $\mathcal{O}(1)$ zero-copy mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `(slot_idx + 1) * SLOT_LEN` exceeds slice length.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the slot start address violates `ALIGN`.
    #[inline(always)]
    pub fn slot_aligned_mut<const SLOT_LEN: usize>(
        &mut self,
        slot_idx: usize,
    ) -> GcsoResult<AlignedSliceMut<'_, T, ALIGN>> {
        self.reborrow().into_slot_aligned_mut::<SLOT_LEN>(slot_idx)
    }

    /// Consumes `self` and simultaneously borrows two distinct, non-overlapping mutable fixed-capacity
    /// slots `(&'a mut [T; SLOT_LEN], &'a mut [T; SLOT_LEN])` at `slot_a` and `slot_b` (`slot_a != slot_b`)
    /// in $\mathcal{O}(1)$ zero-copy mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `slot_a == slot_b` or either slot index is out of bounds.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if either slot start address violates `ALIGN`.
    #[inline(always)]
    pub fn into_disjoint_slot_pair_exact_mut<const SLOT_LEN: usize>(
        self,
        slot_a: usize,
        slot_b: usize,
    ) -> GcsoResult<(&'a mut [T; SLOT_LEN], &'a mut [T; SLOT_LEN])> {
        if slot_a == slot_b {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let base_addr = self.slice.as_ptr() as usize;
        let len = self.slice.len();
        let off_a = validate_slot_offset::<T, ALIGN, SLOT_LEN>(base_addr, len, slot_a)?;
        let off_b = validate_slot_offset::<T, ALIGN, SLOT_LEN>(base_addr, len, slot_b)?;
        let base_ptr = self.slice.as_mut_ptr();
        // SAFETY: `slot_a != slot_b`, `SLOT_LEN > 0`, and validated offsets guarantee strictly disjoint aligned intervals.
        unsafe {
            Ok((
                &mut *base_ptr.add(off_a).cast::<[T; SLOT_LEN]>(),
                &mut *base_ptr.add(off_b).cast::<[T; SLOT_LEN]>(),
            ))
        }
    }

    /// Simultaneously borrows two distinct, non-overlapping mutable fixed-capacity slots
    /// `(&mut [T; SLOT_LEN], &mut [T; SLOT_LEN])` at `slot_a` and `slot_b` (`slot_a != slot_b`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `slot_a == slot_b` or either slot index is out of bounds.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if either slot start address violates `ALIGN`.
    #[inline(always)]
    pub fn disjoint_slot_pair_exact_mut<const SLOT_LEN: usize>(
        &mut self,
        slot_a: usize,
        slot_b: usize,
    ) -> GcsoResult<(&mut [T; SLOT_LEN], &mut [T; SLOT_LEN])> {
        self.reborrow()
            .into_disjoint_slot_pair_exact_mut::<SLOT_LEN>(slot_a, slot_b)
    }

    /// Consumes `self` and simultaneously borrows three distinct, pairwise non-overlapping mutable
    /// fixed-capacity slots `(&'a mut [T; SLOT_LEN], &'a mut [T; SLOT_LEN], &'a mut [T; SLOT_LEN])`
    /// at `slot_a`, `slot_b`, and `slot_c` in $\mathcal{O}(1)$ zero-copy mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any two slot indices are equal or any index is out of bounds.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if any slot start address violates `ALIGN`.
    #[inline(always)]
    pub fn into_disjoint_slot_triple_exact_mut<const SLOT_LEN: usize>(
        self,
        slot_a: usize,
        slot_b: usize,
        slot_c: usize,
    ) -> GcsoResult<(
        &'a mut [T; SLOT_LEN],
        &'a mut [T; SLOT_LEN],
        &'a mut [T; SLOT_LEN],
    )> {
        if (slot_a == slot_b) | (slot_a == slot_c) | (slot_b == slot_c) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let base_addr = self.slice.as_ptr() as usize;
        let len = self.slice.len();
        let off_a = validate_slot_offset::<T, ALIGN, SLOT_LEN>(base_addr, len, slot_a)?;
        let off_b = validate_slot_offset::<T, ALIGN, SLOT_LEN>(base_addr, len, slot_b)?;
        let off_c = validate_slot_offset::<T, ALIGN, SLOT_LEN>(base_addr, len, slot_c)?;
        let base_ptr = self.slice.as_mut_ptr();
        // SAFETY: Pairwise inequality and validated offsets guarantee strictly disjoint aligned intervals.
        unsafe {
            Ok((
                &mut *base_ptr.add(off_a).cast::<[T; SLOT_LEN]>(),
                &mut *base_ptr.add(off_b).cast::<[T; SLOT_LEN]>(),
                &mut *base_ptr.add(off_c).cast::<[T; SLOT_LEN]>(),
            ))
        }
    }

    /// Simultaneously borrows three distinct, pairwise non-overlapping mutable fixed-capacity slots
    /// `(&mut [T; SLOT_LEN], &mut [T; SLOT_LEN], &mut [T; SLOT_LEN])` at `slot_a`, `slot_b`, and `slot_c`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any two slot indices are equal or any index is out of bounds.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if any slot start address violates `ALIGN`.
    #[inline(always)]
    pub fn disjoint_slot_triple_exact_mut<const SLOT_LEN: usize>(
        &mut self,
        slot_a: usize,
        slot_b: usize,
        slot_c: usize,
    ) -> GcsoResult<(&mut [T; SLOT_LEN], &mut [T; SLOT_LEN], &mut [T; SLOT_LEN])> {
        self.reborrow()
            .into_disjoint_slot_triple_exact_mut::<SLOT_LEN>(slot_a, slot_b, slot_c)
    }

    /// Consumes `self` and extracts the mutable fixed-capacity Attractor Slot `&'a mut [T; SLOT_LEN]`
    /// directly addressed by the 3-bit Target Attractor Slot ID (`bits 53..=55`, `0..=7`) embedded
    /// inside `tagged_ptr`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if the slice holds fewer than `(slot_id + 1) * SLOT_LEN` elements.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the target slot address violates `ALIGN`.
    #[inline(always)]
    pub fn into_attractor_slot_by_ptr_mut<const SLOT_LEN: usize>(
        self,
        tagged_ptr: gcso_tagged_ptr_t,
    ) -> GcsoResult<&'a mut [T; SLOT_LEN]> {
        let slot_id = unpack_attractor_slot_from_tag(tagged_ptr.unpack_tag()) as usize;
        debug_assert!(slot_id < MAX_ATTRACTOR_DOMAINS);
        self.into_slot_exact_mut::<SLOT_LEN>(slot_id)
    }

    /// Extracts the mutable fixed-capacity Attractor Slot `&mut [T; SLOT_LEN]` directly addressed
    /// by the 3-bit Target Attractor Slot ID (`bits 53..=55`, `0..=7`) embedded inside `tagged_ptr`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if the slice holds fewer than `(slot_id + 1) * SLOT_LEN` elements.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the target slot address violates `ALIGN`.
    #[inline(always)]
    pub fn attractor_slot_by_ptr_mut<const SLOT_LEN: usize>(
        &mut self,
        tagged_ptr: gcso_tagged_ptr_t,
    ) -> GcsoResult<&mut [T; SLOT_LEN]> {
        self.reborrow()
            .into_attractor_slot_by_ptr_mut::<SLOT_LEN>(tagged_ptr)
    }

    /// Consumes `self` and extracts the mutable fixed-capacity Attractor Slot addressed by `tagged_ptr`
    /// directly as an `AlignedSliceMut<'a, T, ALIGN>` of length `SLOT_LEN` without redundant alignment checks.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if the slice holds fewer than `(slot_id + 1) * SLOT_LEN` elements.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the target slot address violates `ALIGN`.
    #[inline(always)]
    pub fn into_attractor_slot_slice_by_ptr_mut<const SLOT_LEN: usize>(
        self,
        tagged_ptr: gcso_tagged_ptr_t,
    ) -> GcsoResult<AlignedSliceMut<'a, T, ALIGN>> {
        let slot_arr = self.into_attractor_slot_by_ptr_mut::<SLOT_LEN>(tagged_ptr)?;
        // SAFETY: `into_attractor_slot_by_ptr_mut` already verified `ALIGN` on the slot's base address.
        Ok(unsafe { AlignedSliceMut::new_unchecked(slot_arr.as_mut_slice()) })
    }

    /// Extracts the mutable fixed-capacity Attractor Slot addressed by `tagged_ptr` directly as an
    /// `AlignedSliceMut<'_, T, ALIGN>` of length `SLOT_LEN` without redundant alignment checks.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if the slice holds fewer than `(slot_id + 1) * SLOT_LEN` elements.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the target slot address violates `ALIGN`.
    #[inline(always)]
    pub fn attractor_slot_slice_by_ptr_mut<const SLOT_LEN: usize>(
        &mut self,
        tagged_ptr: gcso_tagged_ptr_t,
    ) -> GcsoResult<AlignedSliceMut<'_, T, ALIGN>> {
        self.reborrow()
            .into_attractor_slot_slice_by_ptr_mut::<SLOT_LEN>(tagged_ptr)
    }

    /// Consumes `self` and splits the mutable slice at `mid`, preserving the original lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `mid` exceeds slice length.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the right subslice start address violates `ALIGN`.
    #[inline(always)]
    pub fn into_split_at_mut(
        self,
        mid: usize,
    ) -> GcsoResult<(AlignedSliceMut<'a, T, ALIGN>, AlignedSliceMut<'a, T, ALIGN>)> {
        if mid > self.slice.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
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
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `mid` exceeds slice length.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the right subslice start address violates `ALIGN`.
    #[inline(always)]
    pub fn split_at_mut(
        &mut self,
        mid: usize,
    ) -> GcsoResult<(AlignedSliceMut<'_, T, ALIGN>, AlignedSliceMut<'_, T, ALIGN>)> {
        self.reborrow().into_split_at_mut(mid)
    }

    /// Copies elements from an aligned source slice into `self` in zero-allocation mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if slice lengths do not match.
    #[inline(always)]
    pub fn copy_from_aligned_slice<const SRC_ALIGN: usize>(
        &mut self,
        src: &AlignedSlice<'_, T, SRC_ALIGN>,
    ) -> GcsoResult<()>
    where
        T: Copy,
    {
        self.copy_from_slice(src.as_slice())
    }

    /// Copies elements from an unaligned slice into `self` in zero-allocation mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if slice lengths do not match.
    #[inline(always)]
    pub fn copy_from_slice(&mut self, src: &[T]) -> GcsoResult<()>
    where
        T: Copy,
    {
        if self.slice.len() != src.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.slice.copy_from_slice(src);
        Ok(())
    }

    /// Swaps the contents of two aligned mutable slices in zero-allocation mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if slice lengths do not match.
    #[inline(always)]
    pub fn swap_with_aligned_slice<const OTHER_ALIGN: usize>(
        &mut self,
        other: &mut AlignedSliceMut<'_, T, OTHER_ALIGN>,
    ) -> GcsoResult<()> {
        if self.slice.len() != other.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.slice.swap_with_slice(other.as_mut_slice());
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
    type Error = GcsoStatus;
    #[inline(always)]
    fn try_from(slice: &'a mut [T]) -> Result<Self, Self::Error> {
        Self::new(slice)
    }
}

// ===================================================================
// 3. Type-Level GCSO_RESTRICT Non-Aliasing Guards (Disjoint*)
// ===================================================================

/// Type-level `GCSO_RESTRICT` guard holding an immutable input [`AlignedSlice`] and a mutually
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
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if the two memory ranges overlap.
    #[inline(always)]
    pub fn new(
        input: AlignedSlice<'a, T, ALIGN_IN>,
        output: AlignedSliceMut<'a, T, ALIGN_OUT>,
    ) -> GcsoResult<Self> {
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
    /// Returns [`GCSO_ERROR_NULL_POINTER`] if a non-empty pointer is null.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if either pointer violates its alignment requirement.
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if the two memory regions overlap.
    #[inline(always)]
    pub unsafe fn from_raw_parts(
        ptr_in: *const T,
        len_in: usize,
        ptr_out: *mut T,
        len_out: usize,
    ) -> GcsoResult<Self> {
        let () = TypeAlignCheck::<T, ALIGN_IN>::VALID;
        let () = TypeAlignCheck::<T, ALIGN_OUT>::VALID;
        let in_bytes = len_in
            .checked_mul(size_of::<T>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        let out_bytes = len_out
            .checked_mul(size_of::<T>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;

        if raw_byte_ranges_overlap(ptr_in as usize, in_bytes, ptr_out as usize, out_bytes) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
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

    /// Returns a mutable reference to the output slice.
    #[inline(always)]
    #[must_use]
    pub fn output_mut(&mut self) -> &mut AlignedSliceMut<'a, T, ALIGN_OUT> {
        &mut self.output
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

    /// Consumes the guard and returns the `(input, output)` pair.
    #[inline(always)]
    #[must_use]
    pub fn into_inner(
        self,
    ) -> (
        AlignedSlice<'a, T, ALIGN_IN>,
        AlignedSliceMut<'a, T, ALIGN_OUT>,
    ) {
        (self.input, self.output)
    }
}

/// Type-level `GCSO_RESTRICT` guard holding two aligned, mutually non-overlapping mutable tensor slices.
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
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if the two memory ranges overlap.
    #[inline(always)]
    pub fn new(
        first: AlignedSliceMut<'a, T, ALIGN>,
        second: AlignedSliceMut<'a, T, ALIGN>,
    ) -> GcsoResult<Self> {
        first.ensure_disjoint_mut(&second)?;
        Ok(Self { first, second })
    }

    /// Constructs a `DisjointMutPair` directly from raw FFI pointers of equal length `len`.
    ///
    /// # Safety
    /// Caller must ensure `ptr_a` and `ptr_b` point to valid mutable memory of length `len` for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_NULL_POINTER`], [`GCSO_ERROR_MISALIGNED_POINTER`], or
    /// [`GCSO_ERROR_INVALID_ARGUMENT`] if overlapping.
    #[inline(always)]
    pub unsafe fn from_raw_parts_mut(ptr_a: *mut T, ptr_b: *mut T, len: usize) -> GcsoResult<Self> {
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
    /// Returns [`GCSO_ERROR_NULL_POINTER`], [`GCSO_ERROR_MISALIGNED_POINTER`], or
    /// [`GCSO_ERROR_INVALID_ARGUMENT`] if overlapping.
    #[inline(always)]
    pub unsafe fn from_raw_parts_mut_hetero(
        ptr_a: *mut T,
        len_a: usize,
        ptr_b: *mut T,
        len_b: usize,
    ) -> GcsoResult<Self> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        let bytes_a = len_a
            .checked_mul(size_of::<T>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        let bytes_b = len_b
            .checked_mul(size_of::<T>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;

        if raw_byte_ranges_overlap(ptr_a as usize, bytes_a, ptr_b as usize, bytes_b) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
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

    /// Returns a mutable reference to the first aligned slice.
    #[inline(always)]
    #[must_use]
    pub fn first_mut(&mut self) -> &mut AlignedSliceMut<'a, T, ALIGN> {
        &mut self.first
    }

    /// Returns a mutable reference to the second aligned slice.
    #[inline(always)]
    #[must_use]
    pub fn second_mut(&mut self) -> &mut AlignedSliceMut<'a, T, ALIGN> {
        &mut self.second
    }

    /// Returns immutable references to both disjoint aligned slices.
    #[inline(always)]
    #[must_use]
    pub const fn as_pair(
        &self,
    ) -> (
        &AlignedSliceMut<'a, T, ALIGN>,
        &AlignedSliceMut<'a, T, ALIGN>,
    ) {
        (&self.first, &self.second)
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

    /// Swaps the first and second aligned slices in-place.
    #[inline(always)]
    pub fn swap_handles(&mut self) {
        core::mem::swap(&mut self.first, &mut self.second);
    }

    /// Consumes the guard and returns the pair of disjoint aligned mutable slices.
    #[inline(always)]
    #[must_use]
    pub fn into_inner(self) -> (AlignedSliceMut<'a, T, ALIGN>, AlignedSliceMut<'a, T, ALIGN>) {
        (self.first, self.second)
    }
}

/// Type-level `GCSO_RESTRICT` guard holding three aligned, mutually non-overlapping mutable tensor slices
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
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any two memory ranges overlap.
    #[inline(always)]
    pub fn new(
        first: AlignedSliceMut<'a, T, ALIGN>,
        second: AlignedSliceMut<'a, T, ALIGN>,
        third: AlignedSliceMut<'a, T, ALIGN>,
    ) -> GcsoResult<Self> {
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
    /// Returns [`GCSO_ERROR_NULL_POINTER`], [`GCSO_ERROR_MISALIGNED_POINTER`], or
    /// [`GCSO_ERROR_INVALID_ARGUMENT`] if any two regions overlap.
    #[inline(always)]
    pub unsafe fn from_raw_parts_mut(
        ptr_a: *mut T,
        ptr_b: *mut T,
        ptr_c: *mut T,
        len: usize,
    ) -> GcsoResult<Self> {
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
    /// Returns [`GCSO_ERROR_NULL_POINTER`], [`GCSO_ERROR_MISALIGNED_POINTER`], or
    /// [`GCSO_ERROR_INVALID_ARGUMENT`] if any two regions overlap.
    #[inline(always)]
    pub unsafe fn from_raw_parts_mut_hetero(
        ptr_a: *mut T,
        len_a: usize,
        ptr_b: *mut T,
        len_b: usize,
        ptr_c: *mut T,
        len_c: usize,
    ) -> GcsoResult<Self> {
        let () = TypeAlignCheck::<T, ALIGN>::VALID;
        let bytes_a = len_a
            .checked_mul(size_of::<T>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        let bytes_b = len_b
            .checked_mul(size_of::<T>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        let bytes_c = len_c
            .checked_mul(size_of::<T>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;

        let addr_a = ptr_a as usize;
        let addr_b = ptr_b as usize;
        let addr_c = ptr_c as usize;

        if raw_byte_ranges_overlap(addr_a, bytes_a, addr_b, bytes_b)
            | raw_byte_ranges_overlap(addr_a, bytes_a, addr_c, bytes_c)
            | raw_byte_ranges_overlap(addr_b, bytes_b, addr_c, bytes_c)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
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

    /// Returns a mutable reference to the first aligned slice.
    #[inline(always)]
    #[must_use]
    pub fn first_mut(&mut self) -> &mut AlignedSliceMut<'a, T, ALIGN> {
        &mut self.first
    }

    /// Returns a mutable reference to the second aligned slice.
    #[inline(always)]
    #[must_use]
    pub fn second_mut(&mut self) -> &mut AlignedSliceMut<'a, T, ALIGN> {
        &mut self.second
    }

    /// Returns a mutable reference to the third aligned slice.
    #[inline(always)]
    #[must_use]
    pub fn third_mut(&mut self) -> &mut AlignedSliceMut<'a, T, ALIGN> {
        &mut self.third
    }

    /// Returns immutable references to all three disjoint aligned slices.
    #[inline(always)]
    #[must_use]
    pub const fn as_triple(
        &self,
    ) -> (
        &AlignedSliceMut<'a, T, ALIGN>,
        &AlignedSliceMut<'a, T, ALIGN>,
        &AlignedSliceMut<'a, T, ALIGN>,
    ) {
        (&self.first, &self.second, &self.third)
    }

    /// Splits the guard into mutable references to all three disjoint aligned slices.
    #[inline(always)]
    #[must_use]
    pub fn as_mut_triple(
        &mut self,
    ) -> (
        &mut AlignedSliceMut<'a, T, ALIGN>,
        &mut AlignedSliceMut<'a, T, ALIGN>,
        &mut AlignedSliceMut<'a, T, ALIGN>,
    ) {
        (&mut self.first, &mut self.second, &mut self.third)
    }

    /// Consumes the guard and returns the triple of disjoint aligned mutable slices.
    #[inline(always)]
    #[must_use]
    pub fn into_inner(
        self,
    ) -> (
        AlignedSliceMut<'a, T, ALIGN>,
        AlignedSliceMut<'a, T, ALIGN>,
        AlignedSliceMut<'a, T, ALIGN>,
    ) {
        (self.first, self.second, self.third)
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

            /// Initializes a hardware-aligned stack buffer by copying from an existing slice of length `N`.
            ///
            /// # Errors
            /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `src.len() != N`.
            #[inline(always)]
            pub fn try_from_slice(src: &[T]) -> GcsoResult<Self>
            where
                T: Pod,
            {
                if src.len() != N {
                    return Err(GCSO_ERROR_INVALID_ARGUMENT);
                }
                let mut buf = Self::zeroed();
                buf.data.copy_from_slice(src);
                Ok(buf)
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
            /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `N < SLOTS * SLOT_LEN`.
            #[inline(always)]
            pub const fn as_slot_bank<const SLOTS: usize, const SLOT_LEN: usize>(
                &self,
            ) -> GcsoResult<&[[T; SLOT_LEN]; SLOTS]> {
                let () = TypeAlignCheck::<T, $align>::VALID;
                let () = SlotBankCheck::<SLOTS, SLOT_LEN>::VALID;
                if N < SLOTS * SLOT_LEN {
                    return Err(GCSO_ERROR_INVALID_ARGUMENT);
                }
                // SAFETY: `N >= SLOTS * SLOT_LEN` and `self.data` is contiguous and aligned to `$align >= align_of::<T>()`.
                Ok(unsafe { &*self.data.as_ptr().cast::<[[T; SLOT_LEN]; SLOTS]>() })
            }

            /// Views the prefix of the buffer as a mutable 2D slot bank `&mut [[T; SLOT_LEN]; SLOTS]`.
            ///
            /// # Errors
            /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `N < SLOTS * SLOT_LEN`.
            #[inline(always)]
            pub fn as_slot_bank_mut<const SLOTS: usize, const SLOT_LEN: usize>(
                &mut self,
            ) -> GcsoResult<&mut [[T; SLOT_LEN]; SLOTS]> {
                let () = TypeAlignCheck::<T, $align>::VALID;
                let () = SlotBankCheck::<SLOTS, SLOT_LEN>::VALID;
                if N < SLOTS * SLOT_LEN {
                    return Err(GCSO_ERROR_INVALID_ARGUMENT);
                }
                // SAFETY: `N >= SLOTS * SLOT_LEN`, `self.data` is uniquely borrowed, and aligned to `$align >= align_of::<T>()`.
                Ok(unsafe { &mut *self.data.as_mut_ptr().cast::<[[T; SLOT_LEN]; SLOTS]>() })
            }

            /// Extracts a fixed-size chunk `&[T; CHUNK_LEN]` at `chunk_idx` in $\mathcal{O}(1)$ `const fn` mode.
            ///
            /// # Errors
            /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `(chunk_idx + 1) * CHUNK_LEN > N`.
            #[inline(always)]
            pub const fn chunk_exact<const CHUNK_LEN: usize>(
                &self,
                chunk_idx: usize,
            ) -> GcsoResult<&[T; CHUNK_LEN]> {
                self.as_aligned_slice().chunk_exact::<CHUNK_LEN>(chunk_idx)
            }

            /// Extracts a mutable fixed-size chunk `&mut [T; CHUNK_LEN]` at `chunk_idx`.
            ///
            /// # Errors
            /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `(chunk_idx + 1) * CHUNK_LEN > N`.
            #[inline(always)]
            pub fn chunk_exact_mut<const CHUNK_LEN: usize>(
                &mut self,
                chunk_idx: usize,
            ) -> GcsoResult<&mut [T; CHUNK_LEN]> {
                self.as_aligned_slice_mut()
                    .into_chunk_exact_mut::<CHUNK_LEN>(chunk_idx)
            }

            /// Simultaneously borrows two distinct, non-overlapping mutable chunks at `chunk_a` and `chunk_b`.
            ///
            /// # Errors
            /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `chunk_a == chunk_b` or either index is out of bounds.
            #[inline(always)]
            pub fn disjoint_chunk_pair_exact_mut<const CHUNK_LEN: usize>(
                &mut self,
                chunk_a: usize,
                chunk_b: usize,
            ) -> GcsoResult<(&mut [T; CHUNK_LEN], &mut [T; CHUNK_LEN])> {
                self.as_aligned_slice_mut()
                    .into_disjoint_chunk_pair_exact_mut::<CHUNK_LEN>(chunk_a, chunk_b)
            }

            /// Extracts an aligned fixed-capacity slot `&[T; SLOT_LEN]` at `slot_idx`.
            ///
            /// # Errors
            /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `(slot_idx + 1) * SLOT_LEN > N`.
            /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the slot start address violates `$align`.
            #[inline(always)]
            pub fn slot_exact<const SLOT_LEN: usize>(
                &self,
                slot_idx: usize,
            ) -> GcsoResult<&[T; SLOT_LEN]> {
                self.as_aligned_slice().slot_exact::<SLOT_LEN>(slot_idx)
            }

            /// Extracts an aligned mutable fixed-capacity slot `&mut [T; SLOT_LEN]` at `slot_idx`.
            ///
            /// # Errors
            /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `(slot_idx + 1) * SLOT_LEN > N`.
            /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the slot start address violates `$align`.
            #[inline(always)]
            pub fn slot_exact_mut<const SLOT_LEN: usize>(
                &mut self,
                slot_idx: usize,
            ) -> GcsoResult<&mut [T; SLOT_LEN]> {
                self.as_aligned_slice_mut()
                    .into_slot_exact_mut::<SLOT_LEN>(slot_idx)
            }

            /// Simultaneously borrows two distinct, non-overlapping mutable slots at `slot_a` and `slot_b`.
            ///
            /// # Errors
            /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `slot_a == slot_b` or either index is out of bounds.
            /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if either slot start address violates `$align`.
            #[inline(always)]
            pub fn disjoint_slot_pair_exact_mut<const SLOT_LEN: usize>(
                &mut self,
                slot_a: usize,
                slot_b: usize,
            ) -> GcsoResult<(&mut [T; SLOT_LEN], &mut [T; SLOT_LEN])> {
                self.as_aligned_slice_mut()
                    .into_disjoint_slot_pair_exact_mut::<SLOT_LEN>(slot_a, slot_b)
            }

            /// Simultaneously borrows three distinct, pairwise non-overlapping mutable slots at `slot_a`, `slot_b`, and `slot_c`.
            ///
            /// # Errors
            /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any two slot indices are equal or out of bounds.
            /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if any slot start address violates `$align`.
            #[inline(always)]
            pub fn disjoint_slot_triple_exact_mut<const SLOT_LEN: usize>(
                &mut self,
                slot_a: usize,
                slot_b: usize,
                slot_c: usize,
            ) -> GcsoResult<(&mut [T; SLOT_LEN], &mut [T; SLOT_LEN], &mut [T; SLOT_LEN])> {
                self.as_aligned_slice_mut()
                    .into_disjoint_slot_triple_exact_mut::<SLOT_LEN>(slot_a, slot_b, slot_c)
            }

            /// Extracts the fixed-capacity Attractor Slot `&[T; SLOT_LEN]` addressed by `tagged_ptr`.
            ///
            /// # Errors
            /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `N < (slot_id + 1) * SLOT_LEN`.
            /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the target slot address violates `$align`.
            #[inline(always)]
            pub fn attractor_slot_by_ptr<const SLOT_LEN: usize>(
                &self,
                tagged_ptr: gcso_tagged_ptr_t,
            ) -> GcsoResult<&[T; SLOT_LEN]> {
                self.as_aligned_slice()
                    .attractor_slot_by_ptr::<SLOT_LEN>(tagged_ptr)
            }

            /// Extracts the mutable fixed-capacity Attractor Slot `&mut [T; SLOT_LEN]` addressed by `tagged_ptr`.
            ///
            /// # Errors
            /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `N < (slot_id + 1) * SLOT_LEN`.
            /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the target slot address violates `$align`.
            #[inline(always)]
            pub fn attractor_slot_by_ptr_mut<const SLOT_LEN: usize>(
                &mut self,
                tagged_ptr: gcso_tagged_ptr_t,
            ) -> GcsoResult<&mut [T; SLOT_LEN]> {
                self.as_aligned_slice_mut()
                    .into_attractor_slot_by_ptr_mut::<SLOT_LEN>(tagged_ptr)
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

/// 16-byte aligned immutable slice wrapper for DPSR phase steering arrays.
pub type AlignedSlice16<'a, T> = AlignedSlice<'a, T, 16>;
/// 16-byte aligned mutable slice wrapper for DPSR phase steering arrays.
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
    use crate::abi::gcso_q7_t;
    use crate::traits::primitives::{
        pack_action_slot_tag, ACTION_PRESET_FULL_ORCHESTRATION, ATTRACTOR_SLOT_KNOWLEDGE,
        ATTRACTOR_SLOT_PERSONA, ATTRACTOR_SLOT_REPULSION, MAX_ATTENTION_HEADS,
    };

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
        assert_eq!(slice.as_array::<64>().unwrap()[2], 4.0);

        let relaxed32 = slice.relax_alignment::<32>();
        let upgraded64 = relaxed32.try_upgrade_alignment::<64>().unwrap();
        assert_eq!(upgraded64.align(), 64);

        let mut slice_mut = buf64.as_aligned_slice_mut();
        slice_mut[0] = 3.0;
        let relaxed: AlignedSliceMut16<'_, f32> = slice_mut.relax_alignment::<16>();
        assert_eq!(relaxed[0], 3.0);

        let zero_buf = AlignedBuffer32::<f32, 16>::zeroed();
        assert_eq!(zero_buf[0], 0.0);
        let copied_buf = AlignedBuffer32::<f32, 16>::try_from_slice(&[2.5f32; 16]).unwrap();
        assert_eq!(copied_buf[15], 2.5);
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
            let (q, k, v) = reborrowed.as_mut_triple();
            q[0] += k[0] + v[0];
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
            AlignedBuffer64::<gcso_q7_t, { MAX_ATTRACTOR_DOMAINS * MAX_ATTENTION_HEADS }>::zeroed();

        {
            let bank_2d_mut = multi_slot_bank
                .as_slot_bank_mut::<MAX_ATTRACTOR_DOMAINS, MAX_ATTENTION_HEADS>()
                .unwrap();
            bank_2d_mut[0][0] = 12;
        }
        assert_eq!(multi_slot_bank[0], 12);

        {
            let (persona_slot, knowledge_slot, repulsion_slot) = multi_slot_bank
                .disjoint_slot_triple_exact_mut::<MAX_ATTENTION_HEADS>(
                    usize::from(ATTRACTOR_SLOT_PERSONA),
                    usize::from(ATTRACTOR_SLOT_KNOWLEDGE),
                    usize::from(ATTRACTOR_SLOT_REPULSION),
                )
                .unwrap();
            persona_slot.fill(24);
            knowledge_slot.fill(40);
            repulsion_slot.fill(-32);
        }

        assert_eq!(
            multi_slot_bank.disjoint_slot_pair_exact_mut::<MAX_ATTENTION_HEADS>(1, 1),
            Err(GCSO_ERROR_INVALID_ARGUMENT)
        );
        assert_eq!(
            multi_slot_bank.disjoint_slot_triple_exact_mut::<MAX_ATTENTION_HEADS>(1, 2, 1),
            Err(GCSO_ERROR_INVALID_ARGUMENT)
        );

        {
            let (cell1, cell3) = multi_slot_bank
                .as_aligned_slice_mut()
                .into_slot_aligned_mut::<MAX_ATTENTION_HEADS>(usize::from(ATTRACTOR_SLOT_PERSONA))
                .unwrap()
                .into_disjoint_chunk_pair_exact_mut::<8>(1, 3)
                .unwrap();
            cell1.fill(28);
            cell3.fill(36);
        }

        let ptr = gcso_tagged_ptr_t::pack(
            0x1000,
            pack_action_slot_tag(ATTRACTOR_SLOT_KNOWLEDGE, ACTION_PRESET_FULL_ORCHESTRATION),
            12,
        );

        {
            let target_slot = multi_slot_bank
                .attractor_slot_by_ptr_mut::<MAX_ATTENTION_HEADS>(ptr)
                .unwrap();
            assert_eq!(target_slot[0], 40);
            target_slot[0] = 56;
        }

        let bank_2d = multi_slot_bank
            .as_slot_bank::<MAX_ATTRACTOR_DOMAINS, MAX_ATTENTION_HEADS>()
            .unwrap();
        assert_eq!(bank_2d[0][0], 12);
        assert_eq!(bank_2d[1][8], 28);
        assert_eq!(bank_2d[1][24], 36);
        assert_eq!(bank_2d[2][0], 56);
        assert_eq!(bank_2d[3][0], -32);

        let imm_slice = multi_slot_bank.as_aligned_slice();
        let (p_ref, k_ref, r_ref) = imm_slice
            .slot_triple_exact::<MAX_ATTENTION_HEADS>(1, 2, 3)
            .unwrap();
        assert_eq!((p_ref[0], k_ref[0], r_ref[0]), (24, 56, -32));
    }
}