// name: src/core/src/traits.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core trait definitions, domain abstractions, and zero-copy alignment wrappers
//! for the Geometric Cellular Sheaf Orchestrator (GCSO) engine.
//!
//! This module serves as the single source of truth (Contract / Ground Truth) for:
//! 1. Zero-allocation Hot Path execution ($\mathcal{O}(1)$ per-token loops).
//! 2. Mezzo cellular block coordination and bitmask reductions.
//! 3. Cold Path asynchronous steering, Sheaf entropy evaluations, and ZIMMS persistence.
//!
//! All abstractions enforce static dispatch and memory alignment guarantees at the type level.

#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::inline_always)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::manual_slice_size_calculation)]

use core::ops::{Deref, DerefMut, Index, IndexMut, Range};
use core::slice;

use crate::abi::{
    gcso_anchor_type_t, gcso_capability_flags_t, gcso_config_t, gcso_daes_slot_t,
    gcso_edbc_state_t, gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_pprc_keyframe_header_t,
    gcso_pspm_config_t, gcso_q7_t, gcso_srl_descriptor_t, gcso_zimms_descriptor_t, GcsoStatus,
    GCSO_ERROR_INVALID_ARGUMENT, GCSO_ERROR_MISALIGNED_POINTER, GCSO_ERROR_NULL_POINTER,
};

/// Type alias for GCSO core result responses.
pub type GcsoResult<T> = Result<T, GcsoStatus>;

// ===================================================================
// Const-Generic Zero-Copy Aligned Slice Wrappers
// ===================================================================

/// Zero-copy SIMD/Cacheline aligned slice wrapper preventing unaligned memory access in the Hot Path.
///
/// Ensures memory alignment constraints at the type level to eliminate runtime alignment checks inside kernel loops.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct AlignedSlice<'a, T, const ALIGN: usize = 32> {
    slice: &'a [T],
}

impl<'a, T, const ALIGN: usize> AlignedSlice<'a, T, ALIGN> {
    /// Constructs an empty `AlignedSlice` satisfying any power-of-two alignment constraint.
    #[inline(always)]
    #[must_use]
    pub const fn empty() -> Self {
        Self { slice: &[] }
    }

    /// Creates a new aligned slice wrapper after validating pointer memory alignment.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `ALIGN` is zero or not a power of two.
    /// Returns `GCSO_ERROR_MISALIGNED_POINTER` if the memory address is not `ALIGN`-byte aligned.
    #[inline(always)]
    pub fn new(slice: &'a [T]) -> GcsoResult<Self> {
        if ALIGN == 0 || !ALIGN.is_power_of_two() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if slice.is_empty() || ((slice.as_ptr() as usize) & (ALIGN - 1)) == 0 {
            Ok(Self { slice })
        } else {
            Err(GCSO_ERROR_MISALIGNED_POINTER)
        }
    }

    /// Safely constructs an `AlignedSlice` from a raw pointer and element count across FFI boundaries.
    ///
    /// # Safety
    /// The caller must guarantee that `ptr` points to at least `len` valid initialized instances of `T`,
    /// and that memory is not mutated for the lifetime `'a`.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_NULL_POINTER` if `ptr` is null when `len > 0`.
    /// Returns `GCSO_ERROR_MISALIGNED_POINTER` if `ptr` violates the byte alignment constraint.
    #[inline(always)]
    pub unsafe fn from_raw_parts(ptr: *const T, len: usize) -> GcsoResult<Self> {
        if len == 0 {
            return Ok(Self::empty());
        }
        if ptr.is_null() {
            return Err(GCSO_ERROR_NULL_POINTER);
        }
        if (ptr as usize) & (ALIGN - 1) != 0 {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        // SAFETY: Pointer validity and non-null status verified above by contract.
        let slice = unsafe { slice::from_raw_parts(ptr, len) };
        Ok(Self { slice })
    }

    /// Returns the underlying immutable slice reference.
    #[inline(always)]
    #[must_use]
    pub fn as_slice(&self) -> &'a [T] {
        self.slice
    }

    /// Returns raw immutable pointer to the underlying buffer.
    #[inline(always)]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.slice.as_ptr()
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
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// Returns `true` if the slice contains zero elements.
    #[inline(always)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    /// Returns the first element of the slice, or `None` if empty.
    #[inline(always)]
    #[must_use]
    pub fn first(&self) -> Option<&'a T> {
        self.slice.first()
    }

    /// Returns the last element of the slice, or `None` if empty.
    #[inline(always)]
    #[must_use]
    pub fn last(&self) -> Option<&'a T> {
        self.slice.last()
    }

    /// Returns a reference to an element or subslice at the given index.
    #[inline(always)]
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&'a T> {
        self.slice.get(index)
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
        T: Sized,
    {
        let elem_size = core::mem::size_of::<T>();
        if elem_size == 0 || self.slice.is_empty() {
            return &[];
        }
        let byte_len = core::mem::size_of_val(self.slice);
        // SAFETY: T is Sized and initialized memory can be viewed safely as raw bytes.
        unsafe { slice::from_raw_parts(self.slice.as_ptr().cast::<u8>(), byte_len) }
    }

    /// Checks if a slice pointer satisfies the const alignment requirement.
    #[inline(always)]
    #[must_use]
    pub fn is_aligned(slice: &[T]) -> bool {
        ALIGN != 0
            && ALIGN.is_power_of_two()
            && (slice.is_empty() || ((slice.as_ptr() as usize) & (ALIGN - 1)) == 0)
    }

    /// Creates an aligned subslice if the subslice start address satisfies alignment constraints.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if index range is out of bounds.
    /// Returns `GCSO_ERROR_MISALIGNED_POINTER` if subslice start address is misaligned.
    #[inline(always)]
    pub fn subslice(&self, range: Range<usize>) -> GcsoResult<Self> {
        let sub = self.slice.get(range).ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        Self::new(sub)
    }

    /// Splits the slice at the given index if the right subslice satisfies alignment constraints.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `mid` exceeds slice length.
    /// Returns `GCSO_ERROR_MISALIGNED_POINTER` if the right subslice start address is misaligned.
    #[inline(always)]
    pub fn split_at(&self, mid: usize) -> GcsoResult<(Self, Self)> {
        if mid > self.slice.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (left, right) = self.slice.split_at(mid);
        Ok((Self::new(left)?, Self::new(right)?))
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

impl<'a, T, const ALIGN: usize> Index<usize> for AlignedSlice<'a, T, ALIGN> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.slice[index]
    }
}

impl<'a, T, const ALIGN: usize> Index<Range<usize>> for AlignedSlice<'a, T, ALIGN> {
    type Output = [T];

    #[inline(always)]
    fn index(&self, range: Range<usize>) -> &Self::Output {
        &self.slice[range]
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

/// Zero-copy SIMD/Cacheline aligned mutable slice wrapper.
#[repr(transparent)]
#[derive(Debug, PartialEq, Eq)]
pub struct AlignedSliceMut<'a, T, const ALIGN: usize = 32> {
    slice: &'a mut [T],
}

impl<'a, T, const ALIGN: usize> AlignedSliceMut<'a, T, ALIGN> {
    /// Constructs an empty `AlignedSliceMut` satisfying any power-of-two alignment constraint.
    #[inline(always)]
    #[must_use]
    pub fn empty() -> Self {
        Self { slice: &mut [] }
    }

    /// Creates a new mutable aligned slice wrapper after validating memory alignment.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `ALIGN` is zero or not a power of two.
    /// Returns `GCSO_ERROR_MISALIGNED_POINTER` if the memory address is not `ALIGN`-byte aligned.
    #[inline(always)]
    pub fn new(slice: &'a mut [T]) -> GcsoResult<Self> {
        if ALIGN == 0 || !ALIGN.is_power_of_two() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if slice.is_empty() || ((slice.as_ptr() as usize) & (ALIGN - 1)) == 0 {
            Ok(Self { slice })
        } else {
            Err(GCSO_ERROR_MISALIGNED_POINTER)
        }
    }

    /// Safely constructs an `AlignedSliceMut` from a raw mutable pointer and element count across FFI boundaries.
    ///
    /// # Safety
    /// The caller must guarantee that `ptr` points to at least `len` valid initialized instances of `T`,
    /// and that no alias pointers exist for lifetime `'a`.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_NULL_POINTER` if `ptr` is null when `len > 0`.
    /// Returns `GCSO_ERROR_MISALIGNED_POINTER` if `ptr` violates the byte alignment constraint.
    #[inline(always)]
    pub unsafe fn from_raw_parts_mut(ptr: *mut T, len: usize) -> GcsoResult<Self> {
        if len == 0 {
            return Ok(Self::empty());
        }
        if ptr.is_null() {
            return Err(GCSO_ERROR_NULL_POINTER);
        }
        if (ptr as usize) & (ALIGN - 1) != 0 {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        // SAFETY: Pointer validity and non-null status verified above by contract.
        let slice = unsafe { slice::from_raw_parts_mut(ptr, len) };
        Ok(Self { slice })
    }

    /// Returns the underlying immutable slice reference.
    #[inline(always)]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        self.slice
    }

    /// Returns the underlying mutable slice reference.
    #[inline(always)]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.slice
    }

    /// Reborrows the mutable slice as an immutable `AlignedSlice`.
    #[inline(always)]
    #[must_use]
    pub fn as_aligned_slice(&self) -> AlignedSlice<'_, T, ALIGN> {
        AlignedSlice { slice: self.slice }
    }

    /// Reborrows `self` for a shorter lifetime.
    #[inline(always)]
    pub fn reborrow(&mut self) -> AlignedSliceMut<'_, T, ALIGN> {
        AlignedSliceMut { slice: self.slice }
    }

    /// Returns raw mutable pointer to the underlying buffer.
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.slice.as_mut_ptr()
    }

    /// Returns raw immutable pointer to the underlying buffer.
    #[inline(always)]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.slice.as_ptr()
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
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// Returns `true` if the slice contains zero elements.
    #[inline(always)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    /// Returns the first element of the slice, or `None` if empty.
    #[inline(always)]
    #[must_use]
    pub fn first(&self) -> Option<&T> {
        self.slice.first()
    }

    /// Returns a mutable reference to the first element of the slice, or `None` if empty.
    #[inline(always)]
    pub fn first_mut(&mut self) -> Option<&mut T> {
        self.slice.first_mut()
    }

    /// Returns the last element of the slice, or `None` if empty.
    #[inline(always)]
    #[must_use]
    pub fn last(&self) -> Option<&T> {
        self.slice.last()
    }

    /// Returns a mutable reference to the last element of the slice, or `None` if empty.
    #[inline(always)]
    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.slice.last_mut()
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

    /// Fills the aligned slice with default zero bytes in zero-allocation mode.
    #[inline(always)]
    pub fn zero_out(&mut self)
    where
        T: Copy,
    {
        let elem_size = core::mem::size_of::<T>();
        if elem_size == 0 || self.slice.is_empty() {
            return;
        }
        let byte_len = core::mem::size_of_val(self.slice);
        // SAFETY: Raw byte zeroing over initialized T slice memory.
        unsafe {
            core::ptr::write_bytes(self.slice.as_mut_ptr().cast::<u8>(), 0, byte_len);
        }
    }

    /// Views the underlying mutable slice as a byte slice.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8]
    where
        T: Sized,
    {
        let elem_size = core::mem::size_of::<T>();
        if elem_size == 0 || self.slice.is_empty() {
            return &mut [];
        }
        let byte_len = core::mem::size_of_val(self.slice);
        // SAFETY: T is Sized and initialized memory can be viewed safely as mutable bytes.
        unsafe { slice::from_raw_parts_mut(self.slice.as_mut_ptr().cast::<u8>(), byte_len) }
    }

    /// Checks if a slice pointer satisfies the const alignment requirement.
    #[inline(always)]
    #[must_use]
    pub fn is_aligned(slice: &[T]) -> bool {
        ALIGN != 0
            && ALIGN.is_power_of_two()
            && (slice.is_empty() || ((slice.as_ptr() as usize) & (ALIGN - 1)) == 0)
    }

    /// Creates a mutable subslice if the subslice start address satisfies alignment constraints.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if index range is out of bounds.
    /// Returns `GCSO_ERROR_MISALIGNED_POINTER` if subslice start address is misaligned.
    #[inline(always)]
    pub fn subslice_mut(
        &mut self,
        range: Range<usize>,
    ) -> GcsoResult<AlignedSliceMut<'_, T, ALIGN>> {
        let sub = self
            .slice
            .get_mut(range)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        AlignedSliceMut::new(sub)
    }

    /// Splits the mutable slice at the given index if the right subslice satisfies alignment constraints.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `mid` exceeds slice length.
    /// Returns `GCSO_ERROR_MISALIGNED_POINTER` if the right subslice start address is misaligned.
    #[inline(always)]
    pub fn split_at_mut(
        &mut self,
        mid: usize,
    ) -> GcsoResult<(AlignedSliceMut<'_, T, ALIGN>, AlignedSliceMut<'_, T, ALIGN>)> {
        if mid > self.slice.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (left, right) = self.slice.split_at_mut(mid);
        Ok((AlignedSliceMut::new(left)?, AlignedSliceMut::new(right)?))
    }

    /// Copies elements from an aligned source slice into `self` in zero-allocation mode.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if slice lengths do not match.
    #[inline(always)]
    pub fn copy_from_aligned_slice(&mut self, src: &AlignedSlice<'_, T, ALIGN>) -> GcsoResult<()>
    where
        T: Copy,
    {
        if self.slice.len() != src.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.slice.copy_from_slice(src.as_slice());
        Ok(())
    }

    /// Copies elements from an unaligned slice into `self` in zero-allocation mode.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if slice lengths do not match.
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

impl<'a, T, const ALIGN: usize> Index<usize> for AlignedSliceMut<'a, T, ALIGN> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.slice[index]
    }
}

impl<'a, T, const ALIGN: usize> IndexMut<usize> for AlignedSliceMut<'a, T, ALIGN> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.slice[index]
    }
}

impl<'a, T, const ALIGN: usize> Index<Range<usize>> for AlignedSliceMut<'a, T, ALIGN> {
    type Output = [T];

    #[inline(always)]
    fn index(&self, range: Range<usize>) -> &Self::Output {
        &self.slice[range]
    }
}

impl<'a, T, const ALIGN: usize> IndexMut<Range<usize>> for AlignedSliceMut<'a, T, ALIGN> {
    #[inline(always)]
    fn index_mut(&mut self, range: Range<usize>) -> &mut Self::Output {
        &mut self.slice[range]
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

/// Helper function to check pointer alignment for a generic custom byte boundary.
#[inline(always)]
#[must_use]
pub fn is_aligned_to<T>(slice: &[T], align: usize) -> bool {
    align != 0
        && align.is_power_of_two()
        && (slice.is_empty() || ((slice.as_ptr() as usize) & (align - 1)) == 0)
}

// ===================================================================
// Boundary System Information & Capability Query Trait
// ===================================================================

/// Trait for querying system ABI versions and runtime compute capabilities.
///
/// **Boundary / System Layer Contract.**
pub trait SystemCapabilityQuery: Send + Sync {
    /// Returns the semantic ABI version tuple `(major, minor, patch)`.
    fn abi_version(&self) -> (u32, u32, u32);

    /// Returns the static ABI version string.
    fn abi_version_string(&self) -> &'static str;

    /// Queries active compute capability bitmask flags.
    fn query_capability(&self) -> gcso_capability_flags_t;

    /// Populates default runtime configuration parameters.
    fn init_default_config(&self, config: &mut gcso_config_t) -> GcsoResult<()>;
}

// ===================================================================
// Unified High-Level Runtime Context Facade Trait
// ===================================================================

/// Unified Hot-Path Runtime Context trait for token-by-token inference loops.
///
/// **Macro / High-Level Boundary Contract.**
/// Implementations MUST guarantee zero dynamic allocations (`malloc`, `Box`, `Vec`)
/// during per-token inference calls (`step_token`).
pub trait GcsoRuntimeContext: Send + Sync {
    /// Resets transient phase accumulators and step counters without freeing tables.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_STATE` if runtime context state is corrupted.
    fn reset(&mut self) -> GcsoResult<()>;

    /// Registers a natural language system prompt text anchor in phase space (Primary Baseline Endpoint).
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_ACTION_HUB_FULL` if max prompt anchors capacity is reached.
    fn set_system_prompt_anchor(&mut self, prompt: &str, weight: f32) -> GcsoResult<()>;

    /// Executes a per-token Hot Path inference step in zero-allocation mode.
    ///
    /// Applies QDPS step filtering, RIPA soft-bounded phase steering on Query registers,
    /// and populates the 128-byte pointer trail status in $\mathcal{O}(1)$ time.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_STATE` if context is uninitialized.
    /// Returns `GCSO_ERROR_MISALIGNED_POINTER` if tensor buffers violate 32-byte alignment.
    fn step_token(
        &mut self,
        token_id: u32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: Option<&mut gcso_pointer_trail_t>,
    ) -> GcsoResult<()>;

    /// Serializes active runtime context state into a binary snapshot.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_BUFFER_TOO_SMALL` if output buffer size is insufficient.
    fn serialize(&self, buffer: &mut [u8], required_size: &mut usize) -> GcsoResult<()>;

    /// Deserializes binary snapshot buffer to restore context state in-place.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_CONTAINER_CORRUPTED` if checksum or magic header validation fails.
    fn deserialize(&mut self, buffer: &[u8]) -> GcsoResult<()>;

    /// Seeks to a specific token position in the PPRC keyframe KV cache.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_PPRC_SEEK_FAILED` if `token_index` is out of bounds.
    fn seek_to_token(
        &mut self,
        token_index: u32,
        header_out: &mut gcso_pprc_keyframe_header_t,
    ) -> GcsoResult<()>;
}

// ===================================================================
// Hot-Path Traits (Zero Dynamic Allocation, Nano / Micro Level Static Dispatch Target)
// ===================================================================

/// Hot-path trait for dynamic phase-steering algorithms on Query/Key tensors (**Nano / Micro Level**).
///
/// # Performance Invariants
/// Implementations MUST NOT perform dynamic heap allocations (`malloc`, `Box`, `Vec`).
/// All methods are designed for static dispatch and MUST maintain $\mathcal{O}(d_{\mathrm{head}})$
/// time complexity per head.
pub trait PhaseSteering: Send + Sync {
    /// Returns the target head dimension ($d_{\mathrm{head}}$).
    fn head_dim(&self) -> u32;

    /// Returns the total number of attention heads.
    fn num_heads(&self) -> u32;

    /// Applies inline dynamic phase steering to a Query tensor in zero-allocation mode.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if buffer lengths or head dimensions are invalid.
    fn apply_phase_steering(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<()>;

    /// Applies fused dynamic phase steering to Query and Key tensors simultaneously in a single pass.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if tensor buffers do not match configuration.
    fn apply_phase_steering_fused(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<()>;

    /// Applies RIPA soft-bounded phase steering restricted to low-frequency channels.
    ///
    /// Clamps rotation angles strictly on low-frequency head dimensions ($d_{\mathrm{head}}/4$).
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `max_rad` is non-positive or buffers mismatch.
    fn apply_phase_steering_safe(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> GcsoResult<()>;

    /// QDPS discrete filter: cuts off phase rotation steps falling below `min_step_rad`.
    ///
    /// Prevents grid jitter and numerical oscillation under ultra-low quantization.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `min_step_rad` is negative or buffer is empty.
    fn qdps_filter_step(
        &self,
        phase_deltas: &mut AlignedSliceMut16<'_, gcso_q7_t>,
        min_step_rad: f32,
    ) -> GcsoResult<()>;

    /// Lazy Phase Unwrapping: Applies relative phase difference against context accumulator.
    ///
    /// Modulates Query registers without modifying Key-Value caches in VRAM.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if tensor buffers do not match head config.
    fn lazy_unwrap_override(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        context_accum: &AlignedSlice32<'_, f32>,
    ) -> GcsoResult<()>;

    /// Executes norm-guarded Slerp phase stabilization on state vectors in zero-allocation mode.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if bounds are invalid.
    /// Returns `GCSO_ERROR_EDBC_SINGULARITY` if floating-point vector norm is NaN or Infinite.
    fn slerp_norm_guard_stable(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        norm_lower: f32,
        norm_upper: f32,
    ) -> GcsoResult<()>;

    /// Applies fused inline logit phase shift prior to LM Head Softmax.
    ///
    /// Enforces 32-byte alignment on `logits` for SIMD/AVX-512 vectorization safety.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `logits` or `phase_deltas` are empty.
    fn fused_logit_shift(
        &self,
        logits: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<()>;

    /// Computes Procrustes phase delta alignment between source and target state representations.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if dimension mismatch or invalid scale occurs.
    fn compute_procrustes_phase_delta(
        &self,
        source: &AlignedSlice32<'_, f32>,
        target: &AlignedSlice32<'_, f32>,
        phase_out: &mut AlignedSliceMut16<'_, gcso_q7_t>,
    ) -> GcsoResult<()>;
}

/// Hot-path trait for $\mathcal{O}(1)$ tagged pointer transitions and action hub state updates (**Micro Level**).
///
/// Implements stigmergic memory traversal over sidecar pointer tables (SPT).
pub trait PointerActionHub: Send + Sync {
    /// Resets transient pointer trails and slot counters without freeing tables.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_STATE` if action hub is corrupted.
    fn reset(&mut self) -> GcsoResult<()>;

    /// Advances a tagged pointer transition in zero-allocation mode.
    ///
    /// Updates `trail_out` in-place without dynamic heap allocation.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_STATE` if context is uninitialized.
    fn step_pointer(
        &mut self,
        current_ptr: u64,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<()>;

    /// Computes direct 256-slot hash index for 64-bit pointer trail caching using SplitMix64.
    #[must_use]
    fn hash_slot256_index(&self, ptr: u64) -> u32;

    /// Pulls active pointer trails toward a macro target anchor via pull force $F_{\mathrm{pull}}$.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `pull_force` is NaN or Infinite.
    fn pull_trail_to_attractor(
        &mut self,
        trail_id: u32,
        anchor_id: u32,
        pull_force: f32,
    ) -> GcsoResult<()>;

    /// Links adjacent cellular hallucinated trails into contiguous trace graphs.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_STATE` if action hub state is invalid.
    fn link_hallucinated_trails(&mut self, src_trail_id: u32, dst_trail_id: u32) -> GcsoResult<()>;

    /// Performs bit-tree reduction across 32-byte aligned paged bitmasks in $\mathcal{O}(1)$ time.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `bitmasks` is empty.
    fn reduce_bit_tree(
        &self,
        bitmasks: &AlignedSlice32<'_, gcso_paged_bitmask_t>,
        reduced_out: &mut gcso_paged_bitmask_t,
    ) -> GcsoResult<()>;

    /// Evaluates SIMD/Warp bitmask reduction over 32-byte aligned PagedBlock KV caches.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `kv_bits` is empty.
    fn paged_block_warp_bitmask(
        &self,
        kv_bits: &AlignedSlice32<'_, u64>,
        mask_out: &mut gcso_paged_bitmask_t,
    ) -> GcsoResult<()>;

    /// Returns the maximum slot capacity of the Action Hub table.
    fn capacity(&self) -> u32;

    /// Returns the active pointer trail count currently registered.
    fn active_count(&self) -> u32;
}

/// Dynamic Adaptive Extension Scratchpad (DAES) Multi-Layer Controller.
///
/// Operates $\mathcal{O}(1)$ fast-path shortcuts and in-register telemetry push (**DAES Layer**).
pub trait DaesScratchpad: Send + Sync {
    /// Resets active telemetry mini ledger and cache counters in-place.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_STATE` if scratchpad slot is corrupted.
    fn reset_telemetry(&mut self, slot: &mut gcso_daes_slot_t) -> GcsoResult<()>;

    /// Executes fast-path shortcut lookup in zero-allocation mode.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_STATE` if slot operating mode is not 0.
    fn fast_path_lookup(&self, slot: &gcso_daes_slot_t, input_key: u64) -> GcsoResult<u64>;

    /// Pushes profiling metric byte code into the DAES telemetry ring ledger in-place.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_NULL_POINTER` if `slot` is uninitialized.
    fn telemetry_push(&mut self, slot: &mut gcso_daes_slot_t, metric_code: u8) -> GcsoResult<()>;

    /// Sets the operating mode of the DAES slot (`0` = Scratchpad, `1` = Plugin, `2` = Shared IPC Buffer).
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if mode exceeds 2.
    fn set_mode(&mut self, slot: &mut gcso_daes_slot_t, mode: u32) -> GcsoResult<()>;

    /// Evaluates telemetry ledger to auto-tune PSPM ratios, RIPA clamps, and EDBC thresholds.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_STATE` if evaluation fails.
    fn evaluate_auto_tune(
        &self,
        slot: &gcso_daes_slot_t,
        config_out: &mut gcso_config_t,
    ) -> GcsoResult<()>;

    /// Returns the cumulative cache hit count recorded in the slot.
    fn cache_hit_count(&self, slot: &gcso_daes_slot_t) -> u32;
}

/// Mezzo-path trait for cellular swarm cell chunk step processing across PagedBlocks (**Mezzo Level**).
pub trait SwarmCellChunk: Send + Sync {
    /// Updates local cellular swarm cell state across PagedBlock token chunk boundaries.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `chunk_len` is 0.
    fn step_chunk(&mut self, mask: &gcso_paged_bitmask_t, chunk_len: u32) -> GcsoResult<()>;
}

/// Hot-path trait for Sub-Head Phase Group Allocation (PSPM Router) (**Nano / Micro Level**).
///
/// Routes attention heads into Fact, Logic, and Explore sub-groups in a single pass.
pub trait PspmRouter: Send + Sync {
    /// Dispatches PSPM head-group phase profiles in a single forward pass.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `head_dim` is odd or zero.
    fn dispatch_single_pass(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        config: &gcso_pspm_config_t,
        head_dim: usize,
    ) -> GcsoResult<()>;
}

/// Hot-path trait for Sparse Residual Adapter Layer (SRL) Dynamic Rank-1 evaluations (**Nano / Micro Level**).
///
/// Evaluates $\mathbf{y} = W_{\mathrm{base}}\mathbf{x} + \mathbf{s} \odot (\mathbf{u}(\mathbf{v}^T \mathbf{x}))$.
pub trait SrlAdapter: Send + Sync {
    /// Evaluates SRL Dynamic Rank-1 outer product in zero-allocation mode.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if dimensions are zero or scalar scale is invalid.
    fn eval_rank1(
        &self,
        y_out: &mut AlignedSliceMut32<'_, f32>,
        x_in: &AlignedSlice32<'_, f32>,
        descriptor: &gcso_srl_descriptor_t,
    ) -> GcsoResult<()>;

    /// Evaluates SRL Dynamic Rank-1 outer product and accumulates directly into output tensor in-place.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if dimensions do not match.
    fn eval_rank1_fused(
        &self,
        y_inout: &mut AlignedSliceMut32<'_, f32>,
        x_in: &AlignedSlice32<'_, f32>,
        descriptor: &gcso_srl_descriptor_t,
    ) -> GcsoResult<()>;
}

// ===================================================================
// Composite Hot-Path Execution Engine Trait
// ===================================================================

/// Composite trait bundling all Hot Path components for monomorphic static dispatch.
///
/// Enforces monomorphization across `PhaseSteering`, `PointerActionHub`, `PspmRouter`, `SrlAdapter`, and `DaesScratchpad`.
pub trait HotPathExecutionEngine:
    PhaseSteering + PointerActionHub + PspmRouter + SrlAdapter + DaesScratchpad
{
}

impl<T> HotPathExecutionEngine for T where
    T: PhaseSteering + PointerActionHub + PspmRouter + SrlAdapter + DaesScratchpad
{
}

// ===================================================================
// Cold-Path Traits (Asynchronous Steering, Memory & Attractor Control)
// ===================================================================

/// Cold-path trait for entropy-driven branch evaluation and bifurcation tracking (**Macro Level**).
///
/// Evaluates Moving Z-Score Normalized Attention Entropy ($\tilde{H}$) and controls
/// Pitchfork Bifurcation mode transitions.
pub trait EntropyEvaluator: Send + Sync {
    /// Evaluates token Z-score activation entropy and updates dynamic branch mode.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_EDBC_SINGULARITY` if `token_z_score` contains `NaN` or `Inf`.
    fn eval_stateful(
        &mut self,
        token_z_score: f32,
        state_out: &mut gcso_edbc_state_t,
    ) -> GcsoResult<()>;

    /// Computes CVoid Coherent Vector Alignment Metric for Out-of-Distribution Latent Space.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `key_vector` is empty.
    /// Returns `GCSO_ERROR_EDBC_SINGULARITY` if numerical sum-of-squares is NaN/Inf.
    fn eval_cvoid_dyadic128(
        &self,
        key_vector: &AlignedSlice32<'_, f32>,
        void_score_out: &mut f32,
    ) -> GcsoResult<()>;

    /// Calculates Eyring-Kramers potential barrier height ($\Delta V = \tau_{\mathrm{eff}} / \mathcal{C}_{\mathrm{void}}$).
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_EDBC_SINGULARITY` if numerical instability is detected or $\tau_{\mathrm{eff}} \le 0$.
    fn eval_barrier(&self, void_score: f32, tau_eff: f32) -> GcsoResult<f32>;
}

/// Cold-path trait for macro attractor field steering and repulsion control (**Macro Level**).
pub trait AttractorField: Send + Sync {
    /// Returns the total number of registered topological anchors.
    fn anchor_count(&self) -> u32;

    /// Registers a system prompt anchor in phase space (Primary Baseline Endpoint).
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_ACTION_HUB_FULL` if anchor capacity is reached.
    fn add_system_prompt_anchor(&mut self, prompt: &str, weight: f32) -> GcsoResult<u32>;

    /// Registers a dense feature embedding vector as a continuous attractor anchor.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_ACTION_HUB_FULL` if anchor capacity is reached.
    fn add_embedding_anchor(
        &mut self,
        embedding: &AlignedSlice32<'_, f32>,
        anchor_type: gcso_anchor_type_t,
        weight: f32,
    ) -> GcsoResult<u32>;

    /// Registers a raw topological anchor point in the attractor field.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_ACTION_HUB_FULL` if anchor capacity is reached.
    fn add_anchor(
        &mut self,
        anchor_type: gcso_anchor_type_t,
        vec: &AlignedSlice32<'_, f32>,
        dim: usize,
    ) -> GcsoResult<u32>;

    /// Injects a phase-conjugate repulsion vector ($-\boldsymbol{\Delta\theta}$) to suppress hallucination.
    ///
    /// Converts false local minima energy valleys into repulsive potential peaks.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `gain` is `NaN` or non-positive.
    fn inject_phase_repulsion(
        &mut self,
        repulsion_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        gain: f32,
    ) -> GcsoResult<()>;

    /// Aggregates high-density pointer trails bottom-up to macro-crystallize new dynamic anchors.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_STATE` if aggregation state is invalid.
    fn aggregate_bottom_up(&mut self) -> GcsoResult<u32>;

    /// Clears all registered topological anchors from the attractor field.
    fn clear_anchors(&mut self) -> GcsoResult<()>;
}

/// Cold-path trait for dynamic persona phase modulation patches without altering base weights (**Macro Level**).
pub trait PersonaPatcher: Send + Sync {
    /// Applies binary persona phase modulation patch to runtime context.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if patch data is empty.
    fn apply_patch(&mut self, patch_data: &[u8]) -> GcsoResult<()>;
}

/// Cold-path trait for Training-Free LoRA-to-Phase SVD Projection (L2P-SVD) (**Macro / Offline Level**).
pub trait L2pSvdProjector: Send + Sync {
    /// Projects fine-tuned LoRA matrices via first-order SVD into phase profiles and Rank-1 SRL vectors.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if dimensions, rank, or outputs are invalid.
    #[allow(clippy::too_many_arguments)]
    fn project_lora(
        &self,
        lora_a: &AlignedSlice32<'_, f32>,
        lora_b: &AlignedSlice32<'_, f32>,
        rank: usize,
        dim_in: usize,
        dim_out: usize,
        srl_out: &mut gcso_srl_descriptor_t,
        phase_profile_out: &mut AlignedSliceMut16<'_, gcso_q7_t>,
    ) -> GcsoResult<()>;
}

/// Mezzo/Cold-path trait for Predictive Phase-Motion & Residual Compensation (PPRC) (**Mezzo / Storage Level**).
pub trait PprcCache: Send + Sync {
    /// Seeks to a specific token index in the keyframe KV cache without forward passes.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_PPRC_SEEK_FAILED` if `token_index` is out of bounds.
    fn seek_to_token(
        &mut self,
        token_index: u32,
        header_out: &mut gcso_pprc_keyframe_header_t,
    ) -> GcsoResult<()>;
}

/// Cold-path trait for Zero-Overhead In-Memory Mapped Storage (ZIMMS) (**Storage / Persistence Level**).
pub trait ZimmsStorage: Send + Sync {
    /// Serializes active runtime state into `.gcso` binary snapshot buffer.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_BUFFER_TOO_SMALL` if output buffer size is insufficient.
    fn serialize_snapshot(&self, buffer: &mut [u8], required_size: &mut usize) -> GcsoResult<()>;

    /// Deserializes binary snapshot buffer to restore context state.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_CONTAINER_CORRUPTED` if checksum or magic numbers fail.
    /// Returns `GCSO_ERROR_VERSION_MISMATCH` if ABI version is incompatible.
    fn deserialize_snapshot(
        &mut self,
        buffer: &[u8],
        config_out: &mut gcso_config_t,
    ) -> GcsoResult<()>;

    /// Opens memory-mapped zero-copy handle for `.gcso` container payload.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_ZIMMS_MAPPING_FAILED` if file path is invalid or mapping fails.
    fn open_mmap(
        &mut self,
        file_path: &str,
        descriptor_out: &mut gcso_zimms_descriptor_t,
    ) -> GcsoResult<()>;

    /// Unmaps zero-copy ZIMMS memory handle and releases DMA resources.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_MISALIGNED_POINTER` if descriptor is invalid.
    fn close_mmap(&mut self, descriptor: &mut gcso_zimms_descriptor_t) -> GcsoResult<()>;
}

// ===================================================================
// Unit Tests for Aligned Slice Wrappers and Traits
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_slice_valid() {
        #[repr(align(64))]
        struct AlignedBuffer([f32; 64]);
        let buf = AlignedBuffer([0.0; 64]);
        let slice = AlignedSlice32::new(&buf.0).unwrap();
        assert_eq!(slice.len(), 64);
        assert!(!slice.is_empty());
        assert_eq!(slice.align(), 32);
        assert_eq!(slice[0], 0.0);
        assert_eq!(&slice[0..16], &[0.0; 16]);
    }

    #[test]
    fn test_aligned_slice_from_raw_parts() {
        #[repr(align(64))]
        struct AlignedBuffer([f32; 32]);
        let buf = AlignedBuffer([1.5; 32]);

        // SAFETY: Pointer and length are valid for tests.
        let slice = unsafe { AlignedSlice32::from_raw_parts(buf.0.as_ptr(), 32) }.unwrap();
        assert_eq!(slice.len(), 32);
        assert_eq!(slice[0], 1.5);

        let byte_slice = slice.as_bytes();
        assert_eq!(byte_slice.len(), 32 * core::mem::size_of::<f32>());
    }

    #[test]
    fn test_aligned_slice_mut_zero_out() {
        #[repr(align(64))]
        struct AlignedBuffer([f32; 32]);
        let mut buf = AlignedBuffer([2.5; 32]);

        let mut slice_mut = AlignedSliceMut32::new(&mut buf.0).unwrap();
        assert_eq!(slice_mut[0], 2.5);

        slice_mut.zero_out();
        assert_eq!(slice_mut[0], 0.0);
        assert_eq!(slice_mut[31], 0.0);
    }

    #[test]
    fn test_aligned_slice_empty() {
        let empty: &[f32] = &[];
        let slice = AlignedSlice32::new(empty).unwrap();
        assert!(slice.is_empty());
        assert_eq!(slice.len(), 0);

        let const_empty = AlignedSlice32::<f32>::empty();
        assert!(const_empty.is_empty());
    }

    #[test]
    fn test_aligned_slice_misaligned() {
        #[repr(align(64))]
        struct AlignedBuffer([u8; 64]);
        let buf = AlignedBuffer([0; 64]);
        let unaligned = &buf.0[1..33];
        let res = AlignedSlice32::new(unaligned);
        assert_eq!(res, Err(GCSO_ERROR_MISALIGNED_POINTER));
    }

    #[test]
    fn test_aligned_slice_subslice_and_split() {
        #[repr(align(64))]
        struct AlignedBuffer([f32; 64]);
        let mut buf = AlignedBuffer([1.0; 64]);
        let mut slice_mut = AlignedSliceMut64::new(&mut buf.0).unwrap();
        assert_eq!(slice_mut.len(), 64);

        let (left, right) = slice_mut.split_at_mut(32).unwrap();
        assert_eq!(left.len(), 32);
        assert_eq!(right.len(), 32);
    }

    #[test]
    fn test_aligned_slice_copy_from_aligned() {
        #[repr(align(64))]
        struct AlignedBuffer([f32; 32]);
        let src_buf = AlignedBuffer([2.5; 32]);
        let mut dst_buf = AlignedBuffer([0.0; 32]);

        let src_slice = AlignedSlice32::new(&src_buf.0).unwrap();
        let mut dst_slice = AlignedSliceMut32::new(&mut dst_buf.0).unwrap();

        dst_slice.copy_from_aligned_slice(&src_slice).unwrap();
        assert_eq!(dst_slice[0], 2.5);
        assert_eq!(dst_slice[31], 2.5);
    }

    #[test]
    fn test_aligned_slice_copy_from_unaligned() {
        #[repr(align(64))]
        struct AlignedBuffer([f32; 32]);
        let src_unaligned = [4.2f32; 32];
        let mut dst_buf = AlignedBuffer([0.0; 32]);

        let mut dst_slice = AlignedSliceMut32::new(&mut dst_buf.0).unwrap();
        dst_slice.copy_from_slice(&src_unaligned).unwrap();
        assert_eq!(dst_slice[0], 4.2);
        assert_eq!(dst_slice[31], 4.2);
    }
}