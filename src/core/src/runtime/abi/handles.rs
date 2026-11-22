// name: src/core/src/runtime/abi/handles.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lifetime-bound immutable and mutable FFI opaque handle guards
//! ([`FfiHandleRef`], [`FfiHandleMut`]).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

use core::fmt;
use core::marker::PhantomData;
use core::mem::{align_of, size_of};
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

use crate::foundation::{raw_regions_overlap_or_same_zst, Pod, PtrAlignCheck};
use crate::liminal::{
    LmnkResult, LMNK_ERR_ALIGNMENT, LMNK_ERR_INVALID_ARGUMENT, LMNK_ERR_NULL_POINTER,
};

#[inline(always)]
fn validate_non_null_aligned_ptr<T>(raw_ptr: *mut T, align_mask: usize) -> LmnkResult<NonNull<T>> {
    if raw_ptr.is_null() {
        return Err(LMNK_ERR_NULL_POINTER);
    }
    if ((raw_ptr as usize) & align_mask) != 0 {
        return Err(LMNK_ERR_ALIGNMENT);
    }
    // SAFETY: `raw_ptr` was verified non-null and aligned to `align_mask` immediately above.
    Ok(unsafe { NonNull::new_unchecked(raw_ptr) })
}

/// Lifetime-bound immutable FFI opaque handle wrapper guaranteeing non-null and aligned pointer invariants.
#[repr(transparent)]
pub struct FfiHandleRef<'a, T> {
    ptr: NonNull<T>,
    _marker: PhantomData<&'a T>,
}

// SAFETY: `FfiHandleRef<'a, T>` is semantically equivalent to `&'a T`.
unsafe impl<'a, T: Sync> Send for FfiHandleRef<'a, T> {}
// SAFETY: Shared immutable access across threads is sound when `T: Sync`.
unsafe impl<'a, T: Sync> Sync for FfiHandleRef<'a, T> {}

impl<'a, T> Copy for FfiHandleRef<'a, T> {}

impl<'a, T> Clone for FfiHandleRef<'a, T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> fmt::Debug for FfiHandleRef<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("FfiHandleRef").field(&self.ptr).finish()
    }
}

impl<'a, T> PartialEq for FfiHandleRef<'a, T> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl<'a, T> Eq for FfiHandleRef<'a, T> {}

impl<'a, T> FfiHandleRef<'a, T> {
    /// Validates a raw pointer from C-ABI and wraps it in a non-null, naturally aligned `FfiHandleRef`.
    ///
    /// # Safety
    /// Caller must ensure `raw_ptr` points to a valid, initialized instance of `T` that is not
    /// mutated for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_NULL_POINTER`] if `raw_ptr` is null.
    /// Returns [`LMNK_ERR_ALIGNMENT`] if `raw_ptr` violates `align_of::<T>()`.
    #[inline(always)]
    pub unsafe fn from_raw(raw_ptr: *const T) -> LmnkResult<Self> {
        let ptr = validate_non_null_aligned_ptr(raw_ptr.cast_mut(), align_of::<T>() - 1)?;
        Ok(Self {
            ptr,
            _marker: PhantomData,
        })
    }

    /// Validates a raw pointer from C-ABI against an explicit power-of-two hardware alignment `ALIGN`
    /// (`ALIGN >= align_of::<T>()`).
    ///
    /// # Safety
    /// Caller must ensure `raw_ptr` points to a valid, initialized instance of `T` for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_NULL_POINTER`] if `raw_ptr` is null.
    /// Returns [`LMNK_ERR_ALIGNMENT`] if `raw_ptr` is not `ALIGN`-byte aligned.
    #[inline(always)]
    pub unsafe fn from_raw_aligned<const ALIGN: usize>(raw_ptr: *const T) -> LmnkResult<Self> {
        let () = PtrAlignCheck::<T, ALIGN>::VALID;
        let ptr = validate_non_null_aligned_ptr(raw_ptr.cast_mut(), ALIGN - 1)?;
        Ok(Self {
            ptr,
            _marker: PhantomData,
        })
    }

    /// Validates an optional raw pointer from C-ABI: returns `Ok(None)` if `raw_ptr` is null,
    /// or `Ok(Some(FfiHandleRef))` if non-null and properly aligned.
    ///
    /// # Safety
    /// If non-null, caller must ensure `raw_ptr` points to a valid instance of `T` for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_ALIGNMENT`] if `raw_ptr` is non-null and violates `align_of::<T>()`.
    #[inline(always)]
    pub unsafe fn from_raw_opt(raw_ptr: *const T) -> LmnkResult<Option<Self>> {
        if raw_ptr.is_null() {
            Ok(None)
        } else {
            // SAFETY: Delegated to `from_raw` after non-null check.
            unsafe { Self::from_raw(raw_ptr).map(Some) }
        }
    }

    /// Enforces `LMNK_RESTRICT` non-aliasing invariant against a mutable `FfiHandleMut`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if the two handles overlap in memory.
    #[inline(always)]
    pub fn ensure_disjoint<U>(&self, other: &FfiHandleMut<'_, U>) -> LmnkResult<()> {
        other.ensure_disjoint_ref(self)
    }

    /// Safely reinterprets `self` as a reference handle to another `Pod` type `U` of equal or smaller size.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `size_of::<U>() == 0` or `size_of::<U>() > size_of::<T>()`.
    /// Returns [`LMNK_ERR_ALIGNMENT`] if the pointer violates `align_of::<U>()`.
    #[inline(always)]
    pub fn try_cast<U: Pod>(&self) -> LmnkResult<FfiHandleRef<'a, U>>
    where
        T: Pod,
    {
        if U::IS_ZST || size_of::<U>() > size_of::<T>() {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        // SAFETY: Both `T` and `U` are `Pod`, `size_of::<U>() <= size_of::<T>()`, and `from_raw` checks `align_of::<U>()`.
        unsafe { FfiHandleRef::<U>::from_raw(self.ptr.as_ptr().cast::<U>()) }
    }

    /// Returns the underlying `NonNull<T>` pointer.
    #[inline(always)]
    #[must_use]
    pub const fn as_non_null(&self) -> NonNull<T> {
        self.ptr
    }

    /// Returns the underlying raw immutable pointer `*const T`.
    #[inline(always)]
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Dereferences the validated handle into an immutable reference `&'a T` with lifetime `'a`.
    #[inline(always)]
    #[must_use]
    pub const fn as_ref(&self) -> &'a T {
        // SAFETY: Validity and alignment guaranteed at construction.
        unsafe { &*self.ptr.as_ptr() }
    }

    /// Copies the underlying value out of the validated handle when `T: Copy`.
    #[inline(always)]
    #[must_use]
    pub const fn read_copy(&self) -> T
    where
        T: Copy,
    {
        // SAFETY: Validity and alignment guaranteed at construction.
        unsafe { *self.ptr.as_ptr() }
    }
}

impl<'a, T> Deref for FfiHandleRef<'a, T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<'a, T> AsRef<T> for FfiHandleRef<'a, T> {
    #[inline(always)]
    fn as_ref(&self) -> &T {
        Self::as_ref(self)
    }
}

/// Lifetime-bound mutable FFI opaque handle wrapper guaranteeing non-null, aligned, and exclusive pointer invariants.
#[repr(transparent)]
pub struct FfiHandleMut<'a, T> {
    ptr: NonNull<T>,
    _marker: PhantomData<&'a mut T>,
}

// SAFETY: `FfiHandleMut<'a, T>` is semantically equivalent to `&'a mut T`.
unsafe impl<'a, T: Send> Send for FfiHandleMut<'a, T> {}
// SAFETY: Shared references to `FfiHandleMut` only allow immutable access when `T: Sync`.
unsafe impl<'a, T: Sync> Sync for FfiHandleMut<'a, T> {}

impl<'a, T> fmt::Debug for FfiHandleMut<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("FfiHandleMut").field(&self.ptr).finish()
    }
}

impl<'a, T> PartialEq for FfiHandleMut<'a, T> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl<'a, T> Eq for FfiHandleMut<'a, T> {}

impl<'a, T> FfiHandleMut<'a, T> {
    /// Validates a raw mutable pointer from C-ABI and wraps it in a non-null, naturally aligned `FfiHandleMut`.
    ///
    /// # Safety
    /// Caller must ensure `raw_ptr` points to a valid, uniquely borrowed instance of `T` for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_NULL_POINTER`] if `raw_ptr` is null.
    /// Returns [`LMNK_ERR_ALIGNMENT`] if `raw_ptr` violates `align_of::<T>()`.
    #[inline(always)]
    pub unsafe fn from_raw(raw_ptr: *mut T) -> LmnkResult<Self> {
        let ptr = validate_non_null_aligned_ptr(raw_ptr, align_of::<T>() - 1)?;
        Ok(Self {
            ptr,
            _marker: PhantomData,
        })
    }

    /// Validates a raw mutable pointer from C-ABI against an explicit power-of-two hardware alignment `ALIGN`
    /// (`ALIGN >= align_of::<T>()`).
    ///
    /// # Safety
    /// Caller must ensure `raw_ptr` points to a valid, uniquely borrowed instance of `T` for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_NULL_POINTER`] if `raw_ptr` is null.
    /// Returns [`LMNK_ERR_ALIGNMENT`] if `raw_ptr` is not `ALIGN`-byte aligned.
    #[inline(always)]
    pub unsafe fn from_raw_aligned<const ALIGN: usize>(raw_ptr: *mut T) -> LmnkResult<Self> {
        let () = PtrAlignCheck::<T, ALIGN>::VALID;
        let ptr = validate_non_null_aligned_ptr(raw_ptr, ALIGN - 1)?;
        Ok(Self {
            ptr,
            _marker: PhantomData,
        })
    }

    /// Validates an optional raw mutable pointer from C-ABI: returns `Ok(None)` if `raw_ptr` is null,
    /// or `Ok(Some(FfiHandleMut))` if non-null and properly aligned.
    ///
    /// # Safety
    /// If non-null, caller must ensure `raw_ptr` points to a valid, uniquely borrowed instance of `T` for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_ALIGNMENT`] if `raw_ptr` is non-null and violates `align_of::<T>()`.
    #[inline(always)]
    pub unsafe fn from_raw_opt(raw_ptr: *mut T) -> LmnkResult<Option<Self>> {
        if raw_ptr.is_null() {
            Ok(None)
        } else {
            // SAFETY: Delegated to `from_raw` after non-null check.
            unsafe { Self::from_raw(raw_ptr).map(Some) }
        }
    }

    /// Validates two mutable FFI pointers and enforces `LMNK_RESTRICT` non-overlapping memory ranges.
    ///
    /// # Safety
    /// Caller must ensure `ptr_a` and `ptr_b` point to valid initialized instances of `T` and `U` for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_NULL_POINTER`] if either pointer is null.
    /// Returns [`LMNK_ERR_ALIGNMENT`] if either pointer is misaligned.
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if their memory footprints overlap.
    #[inline(always)]
    pub unsafe fn from_raw_disjoint_pair<U>(
        ptr_a: *mut T,
        ptr_b: *mut U,
    ) -> LmnkResult<(Self, FfiHandleMut<'a, U>)> {
        // SAFETY: Validated for non-nullness, alignment, and pairwise disjointness before returning.
        let handle_a = unsafe { Self::from_raw(ptr_a)? };
        // SAFETY: Validated for non-nullness and alignment, then checked against `handle_a` below.
        let handle_b = unsafe { FfiHandleMut::<U>::from_raw(ptr_b)? };
        handle_a.ensure_disjoint(&handle_b)?;
        Ok((handle_a, handle_b))
    }

    /// Validates three mutable FFI pointers and enforces pairwise `LMNK_RESTRICT` non-overlapping memory ranges.
    ///
    /// # Safety
    /// Caller must ensure `ptr_a`, `ptr_b`, and `ptr_c` point to valid initialized instances for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_NULL_POINTER`] if any pointer is null.
    /// Returns [`LMNK_ERR_ALIGNMENT`] if any pointer is misaligned.
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if any two memory footprints overlap.
    #[inline(always)]
    pub unsafe fn from_raw_disjoint_triple<U, V>(
        ptr_a: *mut T,
        ptr_b: *mut U,
        ptr_c: *mut V,
    ) -> LmnkResult<(Self, FfiHandleMut<'a, U>, FfiHandleMut<'a, V>)> {
        // SAFETY: Validated for non-nullness, alignment, and pairwise disjointness before returning.
        let handle_a = unsafe { Self::from_raw(ptr_a)? };
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and `align_of::<U>()` alignment;
        // pairwise non-aliasing with `handle_a` and `handle_c` is enforced via `ensure_disjoint` before returning.
        let handle_b = unsafe { FfiHandleMut::<U>::from_raw(ptr_b)? };
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and `align_of::<V>()` alignment;
        // pairwise non-aliasing with `handle_a` and `handle_b` is enforced via `ensure_disjoint` before returning.
        let handle_c = unsafe { FfiHandleMut::<V>::from_raw(ptr_c)? };
        handle_a.ensure_disjoint(&handle_b)?;
        handle_a.ensure_disjoint(&handle_c)?;
        handle_b.ensure_disjoint(&handle_c)?;
        Ok((handle_a, handle_b, handle_c))
    }

    /// Branchlessly checks whether the byte range of `self` overlaps with another `FfiHandleMut`.
    #[inline(always)]
    #[must_use]
    pub fn overlaps<U>(&self, other: &FfiHandleMut<'_, U>) -> bool {
        raw_regions_overlap_or_same_zst(
            self.ptr.as_ptr() as usize,
            size_of::<T>(),
            other.ptr.as_ptr() as usize,
            size_of::<U>(),
        )
    }

    /// Enforces `LMNK_RESTRICT` non-aliasing invariant against another mutable FFI handle.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if the two handles overlap in memory.
    #[inline(always)]
    pub fn ensure_disjoint<U>(&self, other: &FfiHandleMut<'_, U>) -> LmnkResult<()> {
        if self.overlaps(other) {
            Err(LMNK_ERR_INVALID_ARGUMENT)
        } else {
            Ok(())
        }
    }

    /// Enforces `LMNK_RESTRICT` non-aliasing invariant against an immutable `FfiHandleRef`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if the two handles overlap in memory.
    #[inline(always)]
    pub fn ensure_disjoint_ref<U>(&self, other: &FfiHandleRef<'_, U>) -> LmnkResult<()> {
        if raw_regions_overlap_or_same_zst(
            self.ptr.as_ptr() as usize,
            size_of::<T>(),
            other.as_ptr() as usize,
            size_of::<U>(),
        ) {
            Err(LMNK_ERR_INVALID_ARGUMENT)
        } else {
            Ok(())
        }
    }

    /// Safely reinterprets `self` as a mutable handle to another `Pod` type `U` of equal or smaller size.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `size_of::<U>() == 0` or `size_of::<U>() > size_of::<T>()`.
    /// Returns [`LMNK_ERR_ALIGNMENT`] if the pointer violates `align_of::<U>()`.
    #[inline(always)]
    pub fn try_cast_mut<U: Pod>(&mut self) -> LmnkResult<FfiHandleMut<'_, U>>
    where
        T: Pod,
    {
        if U::IS_ZST || size_of::<U>() > size_of::<T>() {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        // SAFETY: Both `T` and `U` are `Pod`, `size_of::<U>() <= size_of::<T>()`, and `from_raw` checks `align_of::<U>()`.
        unsafe { FfiHandleMut::<U>::from_raw(self.ptr.as_ptr().cast::<U>()) }
    }

    /// Returns the underlying `NonNull<T>` pointer.
    #[inline(always)]
    #[must_use]
    pub const fn as_non_null(&self) -> NonNull<T> {
        self.ptr
    }

    /// Returns the underlying raw mutable pointer `*mut T`.
    #[inline(always)]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Borrows the validated handle as an immutable `FfiHandleRef`.
    #[inline(always)]
    #[must_use]
    pub const fn as_handle_ref(&self) -> FfiHandleRef<'_, T> {
        FfiHandleRef {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }

    /// Borrows the validated handle as an immutable reference `&T`.
    #[inline(always)]
    #[must_use]
    pub const fn as_ref(&self) -> &T {
        // SAFETY: Validity and alignment guaranteed at construction.
        unsafe { &*self.ptr.as_ptr() }
    }

    /// Borrows the validated handle as a mutable reference `&mut T`.
    #[inline(always)]
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn as_mut(&mut self) -> &mut T {
        // SAFETY: Validity, alignment, and unique borrow guaranteed at construction.
        unsafe { &mut *self.ptr.as_ptr() }
    }

    /// Writes `value` directly into the validated target memory location.
    #[inline(always)]
    pub fn write(&mut self, value: T) {
        // SAFETY: Validity, alignment, and exclusive borrow guaranteed at construction.
        unsafe {
            *self.ptr.as_ptr() = value;
        }
    }

    /// Consumes the handle and converts it into a mutable reference `&'a mut T` with lifetime `'a`.
    #[inline(always)]
    #[must_use]
    pub fn into_mut(self) -> &'a mut T {
        // SAFETY: Validity, alignment, and unique borrow guaranteed at construction.
        unsafe { &mut *self.ptr.as_ptr() }
    }
}

impl<'a, T> Deref for FfiHandleMut<'a, T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<'a, T> DerefMut for FfiHandleMut<'a, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}

impl<'a, T> AsRef<T> for FfiHandleMut<'a, T> {
    #[inline(always)]
    fn as_ref(&self) -> &T {
        Self::as_ref(self)
    }
}

impl<'a, T> AsMut<T> for FfiHandleMut<'a, T> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut T {
        Self::as_mut(self)
    }
}

// ===================================================================
// Strongly-Typed Opaque Handles (v0.2.0 `0x0000_0200`)
// ===================================================================

macro_rules! define_opaque_handles {
    ($(($opaque:ident, $handle:ident, $c_alias:ident, $doc:expr)),* $(,)?) => {
        $(
            #[doc = $doc]
            #[repr(C)]
            pub struct $opaque {
                _private: [u8; 0],
            }

            #[doc = $doc]
            pub type $handle = *mut $opaque;

            #[doc = $doc]
            #[allow(non_camel_case_types)]
            pub type $c_alias = $handle;
        )*
    };
}

define_opaque_handles!(
    (
        LmnkContextOpaque,
        LmnkContextHandle,
        lmnk_context_handle_t,
        "Opaque LiminiKa runtime context handle (`lmnk_context_handle_t`)."
    ),
    (
        LmnkSubstrateOpaque,
        LmnkSubstrateHandle,
        lmnk_substrate_handle_t,
        "Opaque LiminiKa stigmergic substrate handle (`lmnk_substrate_handle_t`)."
    ),
    (
        LmnkEntropyControllerOpaque,
        LmnkEntropyControllerHandle,
        lmnk_entropy_controller_handle_t,
        "Opaque LiminiKa entropy controller handle (`lmnk_entropy_controller_handle_t`)."
    ),
    (
        LmnkPhaseSteeringOpaque,
        LmnkPhaseSteeringHandle,
        lmnk_phase_steering_handle_t,
        "Opaque LiminiKa RoPE phase-steering engine handle (`lmnk_phase_steering_handle_t`)."
    ),
);
