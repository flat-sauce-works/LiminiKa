// name: src/core/src/runtime/abi/dispatch.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Internal zero-cost FFI validation helpers using Strict Provenance `.addr()` semantics.

use super::*;

/// Branchlessly checks pointer non-nullness and natural alignment for type `T`.
#[inline(always)]
#[must_use]
pub fn is_aligned<T>(ptr: *const T) -> bool {
    let align = align_of::<T>();
    (!ptr.is_null()) & ((ptr.addr() & (align - 1)) == 0)
}

/// Validates a non-null, naturally aligned pointer or opaque handle.
#[inline(always)]
pub(super) fn validate_ptr<T>(ptr: *const T) -> LmnkResult<()> {
    if ptr.is_null() {
        Err(LMNK_ERR_NULL_POINTER)
    } else if !is_aligned(ptr) {
        Err(LMNK_ERR_ALIGNMENT)
    } else {
        Ok(())
    }
}

/// Validates a non-null pointer against an explicit power-of-two alignment boundary `align`.
#[inline(always)]
pub(super) fn validate_ptr_aligned<T>(ptr: *const T, align: usize) -> LmnkResult<()> {
    if ptr.is_null() {
        Err(LMNK_ERR_NULL_POINTER)
    } else if !is_ptr_aligned_to(ptr, align) {
        Err(LMNK_ERR_ALIGNMENT)
    } else {
        Ok(())
    }
}

/// Validates a versioned envelope's leading `struct_size` fail-closed (`< MIN || > size_of::<T>()`).
#[inline(always)]
pub(super) fn validate_envelope_size<T>(actual_size: u32) -> LmnkResult<()> {
    let expected = size_of::<T>() as u32;
    if actual_size != expected {
        Err(LMNK_ERR_INVALID_ARGUMENT)
    } else {
        Ok(())
    }
}

/// Validates batch token count (`v0.4.0` supports `1` only) and optional per-token stride.
#[inline(always)]
pub(super) fn validate_token_batch(
    num_tokens: u32,
    token_stride_elems: u32,
    dense_elems: usize,
) -> LmnkResult<()> {
    if num_tokens != 1 {
        return Err(LMNK_ERR_NOT_IMPLEMENTED);
    }
    if token_stride_elems != 0 && (token_stride_elems as usize) < dense_elems {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    Ok(())
}

/// Validates that all mutable memory regions in `muts` are pairwise disjoint and none overlap
/// any immutable input region in `reads` (zero-length entries are ignored).
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if any mutable-mutable or mutable-read byte ranges overlap.
#[inline]
pub(super) fn check_disjoint_rw(
    muts: &[(usize, usize)],
    reads: &[(usize, usize)],
) -> LmnkResult<()> {
    let mut i = 0usize;
    while i < muts.len() {
        let (a_start, a_len) = muts[i];
        if a_len > 0 {
            let mut j = i + 1;
            while j < muts.len() {
                let (b_start, b_len) = muts[j];
                if raw_byte_ranges_overlap(a_start, a_len, b_start, b_len) {
                    return Err(LMNK_ERR_INVALID_ARGUMENT);
                }
                j += 1;
            }
            let mut k = 0usize;
            while k < reads.len() {
                let (r_start, r_len) = reads[k];
                if raw_byte_ranges_overlap(a_start, a_len, r_start, r_len) {
                    return Err(LMNK_ERR_INVALID_ARGUMENT);
                }
                k += 1;
            }
        }
        i += 1;
    }
    Ok(())
}

/// Safely creates an immutable slice `&'a [T]` after validating non-nullness, natural alignment,
/// `len * size_of::<T>() <= isize::MAX`, and non-wrapping address bounds (`#![no_std]` safe).
///
/// # Safety
/// For `len > 0`, `ptr` must point to at least `len` valid, initialized instances of `T` not
/// concurrently mutated for lifetime `'a`.
#[inline(always)]
pub(super) unsafe fn checked_slice_from_raw_parts<'a, T>(
    ptr: *const T,
    len: usize,
) -> LmnkResult<&'a [T]> {
    if len == 0 {
        return Ok(&[]);
    }
    validate_ptr(ptr)?;
    let _ = validate_raw_slice_bounds(ptr, len)?;
    // SAFETY: `ptr` is non-null, naturally aligned, within `isize::MAX` bytes without address
    // wraparound, and caller guarantees valid initialized memory for lifetime `'a`.
    Ok(unsafe { slice::from_raw_parts(ptr, len) })
}

/// Safely creates a mutable slice `&'a mut [T]` after validating non-nullness, natural alignment,
/// `len * size_of::<T>() <= isize::MAX`, and non-wrapping address bounds (`#![no_std]` safe).
///
/// # Safety
/// For `len > 0`, `ptr` must point to at least `len` valid, initialized instances of `T` with
/// exclusive mutable access for lifetime `'a`.
#[inline(always)]
pub(super) unsafe fn checked_slice_from_raw_parts_mut<'a, T>(
    ptr: *mut T,
    len: usize,
) -> LmnkResult<&'a mut [T]> {
    if len == 0 {
        return Ok(&mut []);
    }
    validate_ptr(ptr)?;
    let _ = validate_raw_slice_bounds(ptr, len)?;
    // SAFETY: `ptr` is non-null, naturally aligned, within `isize::MAX` bytes without address
    // wraparound, and caller guarantees exclusive access for lifetime `'a`.
    Ok(unsafe { slice::from_raw_parts_mut(ptr, len) })
}

/// Validates RoPE geometry parameters and constructs `(PhaseSteeringEngine, RopeGeometryDescriptor, total_q_len)`.
#[inline]
pub(super) fn init_rope_geometry(
    head_dim: u32,
    rotary_dim: u32,
    num_heads: u32,
    rope_layout: u8,
) -> LmnkResult<(PhaseSteeringEngine, RopeGeometryDescriptor, usize)> {
    let hd = head_dim as usize;
    let nh = num_heads as usize;
    let rd = if rotary_dim == 0 {
        hd
    } else {
        rotary_dim as usize
    };
    if hd == 0 || (hd & 1) != 0 || nh == 0 || nh > MAX_ATTENTION_HEADS || rd > hd || (rd & 1) != 0 {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    let layout = RopePairLayout::from_u8(rope_layout)?;
    let engine = PhaseSteeringEngine::new(head_dim, num_heads)?;
    let total_q_len = hd.checked_mul(nh).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
    let geom = RopeGeometryDescriptor {
        layout,
        head_dim: hd,
        rotary_dim: rd,
        num_heads: nh,
    };
    Ok((engine, geom, total_q_len))
}
