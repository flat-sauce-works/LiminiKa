// name: src/core/src/foundation/pod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Plain Old Data (`Pod`) zero-copy safety contracts, raw range overlap checks, and Level 1
//! canonical Little-Endian binary structures (`TaggedPtr`, `ArenaSpan`, `PagedBitmask`,
//! `SubheadConfig`, `Rank1Descriptor`, `KvMotionDeltaPacket`).
//!
//! # Portable Device-Ready POD Contract
//! - **Portable Device-Ready PODs** (`TaggedPtr`, `ArenaSpan`, `PagedBitmask`, `SubheadConfig`,
//!   `Rank1Descriptor`, `KvMotionDeltaPacket`): Canonical Little-Endian byte order, fixed-width
//!   primitives only (zero `usize` / native pointers), 0 bytes implicit compiler padding,
//!   power-of-two `align`/`size`, and 64 B-aligned `[PhaseQ7; 64]` vectors directly readable as
//!   `u32x16` / `u64x8` (WGSL, Vulkan `std430`, Metal, CUDA, x86_64 64 B cache line, ARM64 /
//!   Apple Silicon 128 B cache line).

use core::marker::PhantomData;
use core::mem::{align_of, size_of, size_of_val};
use core::ptr;
use core::slice;

use crate::foundation::scalar_math::{branchless_q7_add_sat, branchless_q7_neg_sat};
use crate::liminal::{
    AttributionTag, BoundarySignal, LMNK_ERR_ALIGNMENT, LMNK_ERR_BUFFER_TOO_SMALL,
    LMNK_ERR_INVALID_ARGUMENT, LacunaPartition8, LiminalThresholdState, LmnkResult, LmnkStatus,
    MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS, PhaseQ7, RuntimeConfig,
};

/// Number of 64-bit words in a 256-bit [`PagedBitmask`] (`4`).
pub const PAGED_BITMASK_WORDS: usize = 4;

/// Total number of discrete bits in a [`PagedBitmask`] (`256`).
pub const PAGED_BITMASK_BITS: u32 = 256;

/// Bitmask isolating a valid 3-bit Semantic Provenance / Attractor Slot ID (`0..=7`).
pub const ATTRACTOR_SLOT_ID_MASK: u8 = 0x07;

/// Action Mode Bit 0 (`0x01`): Enable inline phase rotation on Query/Key registers.
pub const ACTION_MODE_PHASE_STEER: u8 = 1 << 0;

/// Action Mode Bit 1 (`0x02`): Enable intermediate layer skipping with SWAR phase compensation.
pub const ACTION_MODE_LAYER_SKIP: u8 = 1 << 1;

/// Action Mode Bit 2 (`0x04`): Enable Contrastive Phase Bloom Filter anti-phase repulsion ($-\boldsymbol{\Delta\theta}$).
pub const ACTION_MODE_PHASE_REPULSE: u8 = 1 << 2;

/// Action Mode Bit 3 (`0x08`): Enable single-pass sub-head group routing (Fact / Logic / Explore).
pub const ACTION_MODE_SUBHEAD_ROUTE: u8 = 1 << 3;

/// Action Mode Bit 4 (`0x10`): Enable Dynamic Rank-1 outer product residual injection ($\mathbf{u}\mathbf{v}^T$).
pub const ACTION_MODE_RANK1_INJECT: u8 = 1 << 4;

/// Composite Action Mode preset for passive sidecar observation (`0x00`).
pub const ACTION_PRESET_PASSIVE_SIDECAR: u8 = 0x00;

/// Composite Action Mode preset for lightweight express phase-steered sidecar execution (`0x01`).
pub const ACTION_PRESET_EXPRESS_PHASE: u8 = ACTION_MODE_PHASE_STEER;

/// Composite Action Mode preset for low-entropy inertial layer-skipping fast-forward (`0x03`).
pub const ACTION_PRESET_INERTIAL_SKIP: u8 = ACTION_MODE_PHASE_STEER | ACTION_MODE_LAYER_SKIP;

/// Composite Action Mode preset for anti-hallucination repulsion and sub-head exploration routing (`0x0D`).
pub const ACTION_PRESET_REPULSION_ESCAPE: u8 =
    ACTION_MODE_PHASE_STEER | ACTION_MODE_PHASE_REPULSE | ACTION_MODE_SUBHEAD_ROUTE;

/// Composite Action Mode preset for full-stack deep persona & knowledge orchestration (`0x19`).
pub const ACTION_PRESET_FULL_ORCHESTRATION: u8 =
    ACTION_MODE_PHASE_STEER | ACTION_MODE_SUBHEAD_ROUTE | ACTION_MODE_RANK1_INJECT;

/// Bitmask extracting the 5-bit Action Mode flags from an 8-bit [`TaggedPtr`] tag (`bits 0..=4`).
pub const TAG_ACTION_MODE_MASK: u8 = 0x1F;

/// Bit shift offset for encoding/extracting the 3-bit Target Attractor Slot ID (`0..=7`) in an 8-bit tag (`bits 5..=7`).
pub const TAG_ATTRACTOR_SLOT_SHIFT: u32 = 5;

/// Packs a 3-bit Target Attractor Slot ID (`0..=7`) and a 5-bit Action Mode mask (`0x00..=0x1F`)
/// into a single 8-bit tag byte for [`TaggedPtr`].
#[inline(always)]
#[must_use]
pub const fn pack_action_slot_tag(attractor_slot_id: u8, action_mode_flags: u8) -> u8 {
    ((attractor_slot_id & ATTRACTOR_SLOT_ID_MASK) << TAG_ATTRACTOR_SLOT_SHIFT)
        | (action_mode_flags & TAG_ACTION_MODE_MASK)
}

/// Extracts the 5-bit Action Mode bitmask (`ACTION_MODE_*`) from an 8-bit [`TaggedPtr`] tag byte.
#[inline(always)]
#[must_use]
pub const fn unpack_action_mode_from_tag(tag: u8) -> u8 {
    tag & TAG_ACTION_MODE_MASK
}

/// Extracts the 3-bit Target Attractor Slot ID (`0..=7`) from an 8-bit [`TaggedPtr`] tag byte.
#[inline(always)]
#[must_use]
pub const fn unpack_attractor_slot_from_tag(tag: u8) -> u8 {
    (tag >> TAG_ATTRACTOR_SLOT_SHIFT) & ATTRACTOR_SLOT_ID_MASK
}

/// Compile-time helper verifying that a requested alignment `ALIGN` is a non-zero power of two
/// and at least `align_of::<T>()`.
pub(crate) struct PtrAlignCheck<T, const ALIGN: usize>(PhantomData<T>);
impl<T, const ALIGN: usize> PtrAlignCheck<T, ALIGN> {
    pub(crate) const VALID: () = assert!(
        ALIGN > 0 && ALIGN.is_power_of_two() && ALIGN >= align_of::<T>(),
        "Requested ALIGN must be a non-zero power of two and >= align_of::<T>()"
    );
}

/// Validates that `len * size_of::<T>()` does not overflow `usize`, does not exceed `isize::MAX`,
/// and that `(ptr as usize) + byte_len` does not wrap around the address space.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `len * size_of::<T>()` overflows, exceeds
/// `isize::MAX`, or `(ptr as usize) + byte_len` overflows `usize`.
#[inline(always)]
pub fn validate_raw_slice_bounds<T>(ptr: *const T, len: usize) -> LmnkResult<usize> {
    let Some(byte_len) = len.checked_mul(size_of::<T>()) else {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    };
    if byte_len > isize::MAX as usize {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    if ptr.addr().checked_add(byte_len).is_none() {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    Ok(byte_len)
}

/// Branchlessly checks whether two half-open byte intervals
/// `[a_start, a_start + a_byte_len)` and `[b_start, b_start + b_byte_len)` overlap in virtual memory.
#[inline(always)]
#[must_use]
pub const fn raw_byte_ranges_overlap(
    a_start: usize,
    a_byte_len: usize,
    b_start: usize,
    b_byte_len: usize,
) -> bool {
    let non_empty = (a_byte_len > 0) & (b_byte_len > 0);
    let a_end = a_start.saturating_add(a_byte_len);
    let b_end = b_start.saturating_add(b_byte_len);
    non_empty & (a_start < b_end) & (b_start < a_end)
}

#[inline(always)]
#[must_use]
pub(crate) const fn raw_regions_overlap_or_same_zst(
    a_start: usize,
    a_size: usize,
    b_start: usize,
    b_size: usize,
) -> bool {
    let zst_same_addr = ((a_size == 0) | (b_size == 0)) & (a_start == b_start);
    raw_byte_ranges_overlap(a_start, a_size, b_start, b_size) | zst_same_addr
}

/// Marker trait for Plain Old Data (`Pod`) types that contain no uninitialized padding traps,
/// can be safely zero-initialized, viewed as raw byte slices, and cast across aligned SIMD
/// or mapped-storage memory-mapped boundaries.
///
/// # Safety
/// Implementing types must be `#[repr(C)]`, `#[repr(transparent)]`, or primitive numeric types,
/// have no invalid bit patterns (all-zero bytes and arbitrary byte patterns must represent valid
/// instances), and contain no interior mutability, pointers with provenance invariants, or drop glue.
pub unsafe trait Pod: Copy + Sized + Send + Sync + 'static {
    /// Size of `Self` in bytes (`core::mem::size_of::<Self>()`).
    const BYTE_SIZE: usize = size_of::<Self>();

    /// Required alignment of `Self` in bytes (`core::mem::align_of::<Self>()`).
    const ALIGNMENT: usize = align_of::<Self>();

    /// Compile-time flag indicating whether `Self` is a Zero-Sized Type (`BYTE_SIZE == 0`).
    const IS_ZST: bool = size_of::<Self>() == 0;

    /// Returns a zero-initialized instance of `Self`.
    #[inline(always)]
    #[must_use]
    fn zeroed() -> Self {
        // SAFETY: All-zero bit patterns are guaranteed valid by the `Pod` safety contract.
        unsafe { core::mem::zeroed() }
    }

    /// Views `self` as an immutable byte slice of length `size_of::<Self>()`.
    #[inline(always)]
    #[must_use]
    fn as_bytes(&self) -> &[u8] {
        // SAFETY: `Self` is `Pod` with no uninitialized padding traps, and `self` is a valid reference.
        unsafe { slice::from_raw_parts((self as *const Self).cast::<u8>(), size_of::<Self>()) }
    }

    /// Views `self` as a mutable byte slice of length `size_of::<Self>()`.
    #[inline(always)]
    #[must_use]
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: `Self` is `Pod` and uniquely borrowed via `&mut self`.
        unsafe { slice::from_raw_parts_mut((self as *mut Self).cast::<u8>(), size_of::<Self>()) }
    }

    /// Views a contiguous slice of `Self` as an immutable byte slice in zero-allocation mode.
    #[inline(always)]
    #[must_use]
    fn slice_as_bytes(slice: &[Self]) -> &[u8] {
        if Self::IS_ZST || slice.is_empty() {
            return &[];
        }
        // SAFETY: `Self` is `Pod` with no uninitialized padding bytes across contiguous array layouts.
        unsafe { slice::from_raw_parts(slice.as_ptr().cast::<u8>(), size_of_val(slice)) }
    }

    /// Views a contiguous mutable slice of `Self` as a mutable byte slice in zero-allocation mode.
    #[inline(always)]
    #[must_use]
    fn slice_as_bytes_mut(slice: &mut [Self]) -> &mut [u8] {
        if Self::IS_ZST || slice.is_empty() {
            return &mut [];
        }
        let byte_len = size_of_val(slice);
        // SAFETY: `Self` is `Pod` and `slice` is uniquely borrowed.
        unsafe { slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<u8>(), byte_len) }
    }

    /// Zero-copy casts an aligned byte slice at `offset` into an immutable reference `&Self`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`] if `bytes.len() < offset + size_of::<Self>()`.
    /// Returns [`LMNK_ERR_ALIGNMENT`] if the byte address at `offset` violates `align_of::<Self>()`.
    #[inline(always)]
    fn from_bytes_at(bytes: &[u8], offset: usize) -> LmnkResult<&Self> {
        let end = offset
            .checked_add(size_of::<Self>())
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
        if bytes.len() < end {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        // SAFETY: `offset <= bytes.len()` verified above.
        let ptr = unsafe { bytes.as_ptr().add(offset) };
        if (ptr.addr() & (align_of::<Self>() - 1)) != 0 {
            return Err(LMNK_ERR_ALIGNMENT);
        }
        // SAFETY: Bounds and alignment for `Self: Pod` are verified above.
        Ok(unsafe { &*ptr.cast::<Self>() })
    }

    /// Zero-copy casts an aligned mutable byte slice at `offset` into a mutable reference `&mut Self`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`] if `bytes.len() < offset + size_of::<Self>()`.
    /// Returns [`LMNK_ERR_ALIGNMENT`] if the byte address at `offset` violates `align_of::<Self>()`.
    #[inline(always)]
    fn from_bytes_mut_at(bytes: &mut [u8], offset: usize) -> LmnkResult<&mut Self> {
        let end = offset
            .checked_add(size_of::<Self>())
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
        if bytes.len() < end {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        // SAFETY: `offset <= bytes.len()` verified above.
        let ptr = unsafe { bytes.as_mut_ptr().add(offset) };
        if (ptr.addr() & (align_of::<Self>() - 1)) != 0 {
            return Err(LMNK_ERR_ALIGNMENT);
        }
        // SAFETY: Bounds, unique borrow, and alignment for `Self: Pod` are verified above.
        Ok(unsafe { &mut *ptr.cast::<Self>() })
    }

    /// Zero-copy casts an aligned byte region starting at `offset` into a slice `&[Self]` of `count` elements.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `Self` is zero-sized.
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`] if `bytes` cannot hold `count * size_of::<Self>()` bytes from `offset`.
    /// Returns [`LMNK_ERR_ALIGNMENT`] if the starting address violates `align_of::<Self>()`.
    #[inline(always)]
    fn slice_from_bytes_at(bytes: &[u8], offset: usize, count: usize) -> LmnkResult<&[Self]> {
        if Self::IS_ZST {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let byte_len = count
            .checked_mul(size_of::<Self>())
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
        let end = offset
            .checked_add(byte_len)
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
        if bytes.len() < end {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        if count == 0 {
            return Ok(&[]);
        }
        // SAFETY: `offset + byte_len <= bytes.len()` verified above.
        let ptr = unsafe { bytes.as_ptr().add(offset) };
        if (ptr.addr() & (align_of::<Self>() - 1)) != 0 {
            return Err(LMNK_ERR_ALIGNMENT);
        }
        // SAFETY: Bounds and alignment for `count` elements of `Self: Pod` are verified above.
        Ok(unsafe { slice::from_raw_parts(ptr.cast::<Self>(), count) })
    }

    /// Zero-copy casts an aligned mutable byte region starting at `offset` into `&mut [Self]` of `count` elements.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `Self` is zero-sized.
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`] if `bytes` cannot hold `count * size_of::<Self>()` bytes from `offset`.
    /// Returns [`LMNK_ERR_ALIGNMENT`] if the starting address violates `align_of::<Self>()`.
    #[inline(always)]
    fn slice_from_bytes_mut_at(
        bytes: &mut [u8],
        offset: usize,
        count: usize,
    ) -> LmnkResult<&mut [Self]> {
        if Self::IS_ZST {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let byte_len = count
            .checked_mul(size_of::<Self>())
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
        let end = offset
            .checked_add(byte_len)
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
        if bytes.len() < end {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        if count == 0 {
            return Ok(&mut []);
        }
        // SAFETY: `offset + byte_len <= bytes.len()` verified above.
        let ptr = unsafe { bytes.as_mut_ptr().add(offset) };
        if (ptr.addr() & (align_of::<Self>() - 1)) != 0 {
            return Err(LMNK_ERR_ALIGNMENT);
        }
        // SAFETY: Bounds, unique borrow, and alignment for `count` elements of `Self: Pod` verified above.
        Ok(unsafe { slice::from_raw_parts_mut(ptr.cast::<Self>(), count) })
    }

    /// Safely reads an instance of `Self` from a potentially unaligned byte slice at `offset`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`] if `bytes.len() < offset + size_of::<Self>()`.
    #[inline(always)]
    fn read_unaligned_from(bytes: &[u8], offset: usize) -> LmnkResult<Self> {
        let end = offset
            .checked_add(size_of::<Self>())
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
        if bytes.len() < end {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        // SAFETY: Bounds checked above; `Self` is `Pod` so any bit pattern is valid via `read_unaligned`.
        Ok(unsafe { ptr::read_unaligned(bytes.as_ptr().add(offset).cast::<Self>()) })
    }

    /// Safely writes `self` into a potentially unaligned byte slice at `offset`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`] if `bytes.len() < offset + size_of::<Self>()`.
    #[inline(always)]
    fn write_unaligned_to(&self, bytes: &mut [u8], offset: usize) -> LmnkResult<()> {
        let end = offset
            .checked_add(size_of::<Self>())
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
        if bytes.len() < end {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        // SAFETY: Bounds checked above; `Self` is `Pod` with no padding traps.
        unsafe {
            ptr::write_unaligned(bytes.as_mut_ptr().add(offset).cast::<Self>(), *self);
        }
        Ok(())
    }

    /// Bulk-copies `dst.len()` elements of `Self` from a potentially unaligned byte buffer at `offset`
    /// using a single `ptr::copy_nonoverlapping` (`memcpy`) operation.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`] if `bytes` has fewer than `offset + size_of_val(dst)` bytes.
    #[inline(always)]
    fn read_unaligned_slice_from(bytes: &[u8], offset: usize, dst: &mut [Self]) -> LmnkResult<()> {
        let byte_len = size_of_val(dst);
        let end = offset
            .checked_add(byte_len)
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
        if bytes.len() < end {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        if byte_len > 0 {
            // SAFETY: `bytes[offset..end]` and `dst` are valid, non-overlapping regions of `byte_len` bytes,
            // and `u8` has alignment 1 so unaligned source reads into `dst` are sound for `Self: Pod`.
            unsafe {
                ptr::copy_nonoverlapping(
                    bytes.as_ptr().add(offset),
                    dst.as_mut_ptr().cast::<u8>(),
                    byte_len,
                );
            }
        }
        Ok(())
    }

    /// Bulk-writes `src` into a potentially unaligned byte buffer at `offset` using a single
    /// `ptr::copy_nonoverlapping` (`memcpy`) operation.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`] if `bytes` has fewer than `offset + size_of_val(src)` bytes.
    #[inline(always)]
    fn write_unaligned_slice_to(src: &[Self], bytes: &mut [u8], offset: usize) -> LmnkResult<()> {
        let byte_len = size_of_val(src);
        let end = offset
            .checked_add(byte_len)
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
        if bytes.len() < end {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        if byte_len > 0 {
            // SAFETY: `src` and `bytes[offset..end]` are valid, non-overlapping regions of `byte_len` bytes,
            // and `Self: Pod` contains no uninitialized padding bytes.
            unsafe {
                ptr::copy_nonoverlapping(
                    src.as_ptr().cast::<u8>(),
                    bytes.as_mut_ptr().add(offset),
                    byte_len,
                );
            }
        }
        Ok(())
    }

    /// Safely reinterprets a slice of `Self` as a slice of another `Pod` type `U`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `U` is zero-sized or total byte length is not a multiple of `size_of::<U>()`.
    /// Returns [`LMNK_ERR_ALIGNMENT`] if the slice base pointer violates `align_of::<U>()`.
    #[inline(always)]
    fn try_cast_slice<U: Pod>(slice: &[Self]) -> LmnkResult<&[U]> {
        let total_bytes = size_of_val(slice);
        let target_size = size_of::<U>();
        if (target_size == 0) | !total_bytes.is_multiple_of(target_size) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let is_aligned = slice.is_empty() | ((slice.as_ptr().addr() & (align_of::<U>() - 1)) == 0);
        if !is_aligned {
            return Err(LMNK_ERR_ALIGNMENT);
        }
        // SAFETY: Both `Self` and `U` are `Pod`, and alignment/byte-size invariants are verified above.
        Ok(unsafe { slice::from_raw_parts(slice.as_ptr().cast::<U>(), total_bytes / target_size) })
    }

    /// Safely reinterprets a mutable slice of `Self` as a mutable slice of another `Pod` type `U`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `U` is zero-sized or total byte length is not a multiple of `size_of::<U>()`.
    /// Returns [`LMNK_ERR_ALIGNMENT`] if the slice base pointer violates `align_of::<U>()`.
    #[inline(always)]
    fn try_cast_slice_mut<U: Pod>(slice: &mut [Self]) -> LmnkResult<&mut [U]> {
        let total_bytes = size_of_val(slice);
        let target_size = size_of::<U>();
        if (target_size == 0) | !total_bytes.is_multiple_of(target_size) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let is_aligned = slice.is_empty() | ((slice.as_ptr().addr() & (align_of::<U>() - 1)) == 0);
        if !is_aligned {
            return Err(LMNK_ERR_ALIGNMENT);
        }
        // SAFETY: Both `Self` and `U` are `Pod`, and alignment/byte-size invariants are verified above.
        Ok(unsafe {
            slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<U>(), total_bytes / target_size)
        })
    }
}

macro_rules! impl_pod_for_primitives {
    ($($ty:ty),* $(,)?) => {$(
        // SAFETY: Primitive numeric and C-ABI POD structures have well-defined bit layouts
        // where all-zero bit patterns are valid and no drop glue or hidden padding exists.
        unsafe impl Pod for $ty {}
    )*};
}

/// 64-bit tagged memory pointer descriptor matching the LiminiKa specification (8 bytes).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TaggedPtr(
    /// Raw 64-bit packed tagged pointer word.
    pub u64,
);

impl TaggedPtr {
    /// Bitmask for extracting the lower 48-bit canonical raw address.
    pub const ADDR_MASK: u64 = 0x0000_FFFF_FFFF_FFFF;
    /// Bitmask for extracting the 8-bit domain/action tag (`bits 48..=55`).
    pub const TAG_MASK: u64 = 0x00FF_0000_0000_0000;
    /// Bitmask for extracting the 8-bit Q7 phase state (`bits 56..=63`).
    pub const PHASE_MASK: u64 = 0xFF00_0000_0000_0000;
    /// Bit shift offset for the 8-bit domain/action tag.
    pub const TAG_SHIFT: u32 = 48;
    /// Bit shift offset for the 8-bit Q7 phase state.
    pub const PHASE_SHIFT: u32 = 56;
    /// Null tagged pointer constant (`0`).
    pub const NULL: Self = Self(0);

    /// Packs a raw 48-bit address, 8-bit domain tag, and Q7 phase value into a 64-bit tagged pointer.
    #[inline(always)]
    #[must_use]
    pub const fn pack(raw_addr: u64, tag: u8, phase: PhaseQ7) -> Self {
        let addr = raw_addr & Self::ADDR_MASK;
        let tag_bits = (tag as u64) << Self::TAG_SHIFT;
        let phase_bits = ((phase as u8) as u64) << Self::PHASE_SHIFT;
        Self(addr | tag_bits | phase_bits)
    }

    /// Returns the raw 64-bit integer representation.
    #[inline(always)]
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Returns `true` if the 48-bit canonical address component is zero.
    #[inline(always)]
    #[must_use]
    pub const fn is_null(self) -> bool {
        (self.0 & Self::ADDR_MASK) == 0
    }

    /// Unpacks the canonical 48-bit raw address pointer.
    #[inline(always)]
    #[must_use]
    pub const fn unpack_addr(self) -> u64 {
        self.0 & Self::ADDR_MASK
    }

    /// Unpacks the 8-bit domain/action tag.
    #[inline(always)]
    #[must_use]
    pub const fn unpack_tag(self) -> u8 {
        ((self.0 & Self::TAG_MASK) >> Self::TAG_SHIFT) as u8
    }

    /// Unpacks the Q7 quantized phase state.
    #[inline(always)]
    #[must_use]
    pub const fn unpack_phase(self) -> PhaseQ7 {
        ((self.0 >> Self::PHASE_SHIFT) as u8) as PhaseQ7
    }

    /// Returns a new tagged pointer with an updated 48-bit canonical address, preserving tag and phase.
    #[inline(always)]
    #[must_use]
    pub const fn with_addr(self, new_addr: u64) -> Self {
        Self((self.0 & !Self::ADDR_MASK) | (new_addr & Self::ADDR_MASK))
    }

    /// Returns a new tagged pointer with an updated 8-bit domain/action tag, preserving address and phase.
    #[inline(always)]
    #[must_use]
    pub const fn with_tag(self, new_tag: u8) -> Self {
        Self((self.0 & !Self::TAG_MASK) | ((new_tag as u64) << Self::TAG_SHIFT))
    }

    /// Returns a new tagged pointer with an updated Q7 phase value, preserving address and tag.
    #[inline(always)]
    #[must_use]
    pub const fn with_phase(self, new_phase: PhaseQ7) -> Self {
        Self((self.0 & !Self::PHASE_MASK) | (((new_phase as u8) as u64) << Self::PHASE_SHIFT))
    }

    /// Branchlessly accumulates `delta_q7` into the embedded Q7 phase using saturating addition.
    #[inline(always)]
    #[must_use]
    pub const fn add_phase_sat(self, delta_q7: PhaseQ7) -> Self {
        self.with_phase(branchless_q7_add_sat(self.unpack_phase(), delta_q7))
    }

    /// Branchlessly negates the embedded Q7 phase into its phase-conjugate anti-phase ($-\Delta\theta$).
    #[inline(always)]
    #[must_use]
    pub const fn conjugate_phase(self) -> Self {
        self.with_phase(branchless_q7_neg_sat(self.unpack_phase()))
    }
}

/// 256-bit PagedBlock bitmask layout aligned to 32 bytes for SIMD reductions (32 bytes).
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PagedBitmask {
    /// Four 64-bit unsigned words storing 256 discrete token/block execution flags.
    pub bits: [u64; 4],
}

/// Parallel multi-head sub-head router configuration descriptor (32 bytes, 16-byte aligned).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SubheadConfig {
    /// Number of attention heads allocated to the Fact sub-group.
    pub num_fact_heads: u16,
    /// Number of attention heads allocated to the Logic sub-group.
    pub num_logic_heads: u16,
    /// Number of attention heads allocated to the Explore sub-group.
    pub num_explore_heads: u16,
    /// Bitmask flags controlling single-pass routing behavior.
    pub flags: u16,
    /// Phase/amplitude steering gain applied to the Fact head group.
    pub fact_phase_gain: f32,
    /// Phase/amplitude steering gain applied to the Logic head group.
    pub logic_phase_gain: f32,
    /// Phase/amplitude steering gain applied to the Explore head group.
    pub explore_phase_gain: f32,
    /// Explicit reserved padding for 16-byte alignment (12 bytes).
    pub reserved: [u8; 12],
}

/// Portable 8-byte (`#[repr(C)]`) offset-and-length span addressing a contiguous slice inside a
/// caller-owned arena buffer without raw host pointers (identical layout across 32-bit, 64-bit,
/// and GPU `std430`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ArenaSpan {
    /// Element offset from the start of the backing arena (`offset 0..4`).
    pub offset: u32,
    /// Number of contiguous elements referenced by this span (`offset 4..8`).
    pub len: u32,
}

impl ArenaSpan {
    /// Constructs a new [`ArenaSpan`] from element `offset` and `len`.
    #[inline(always)]
    #[must_use]
    pub const fn new(offset: u32, len: u32) -> Self {
        Self { offset, len }
    }

    /// Computes `offset + len` with 32-bit overflow protection so 32-bit and 64-bit targets agree.
    #[inline(always)]
    #[must_use]
    pub const fn checked_end(self) -> Option<usize> {
        match self.offset.checked_add(self.len) {
            Some(end) => Some(end as usize),
            None => None,
        }
    }

    /// Resolves this span to an immutable sub-slice of `arena`, returning `None` on overflow or OOB.
    #[inline(always)]
    #[must_use]
    pub fn slice<T>(self, arena: &[T]) -> Option<&[T]> {
        let end = self.checked_end()?;
        arena.get(self.offset as usize..end)
    }

    /// Resolves this span to a mutable sub-slice of `arena`, returning `None` on overflow or OOB.
    #[inline(always)]
    #[must_use]
    pub fn slice_mut<T>(self, arena: &mut [T]) -> Option<&mut [T]> {
        let end = self.checked_end()?;
        arena.get_mut(self.offset as usize..end)
    }
}

/// Sentinel value for [`Rank1Descriptor::gain_offset`] indicating no per-output diagonal gain
/// vector is applied (`u32::MAX`).
pub const RANK1_GAIN_UNUSED: u32 = u32::MAX;

/// Portable 32-byte (`#[repr(C, align(32))]`) [`Pod`] descriptor for Sparse Dynamic Rank-1
/// residual injection ($\mathbf{y} \gets \mathbf{y} + s \cdot \mathrm{diag}(\mathbf{g})\mathbf{u}(\mathbf{v}^T \mathbf{x})$),
/// addressing all factor vectors via [`ArenaSpan`] element offsets inside a caller-owned `f32` arena.
///
/// Identical layout across 32-bit (`wasm32`), 64-bit CPUs, and GPU `std430` with zero raw pointers
/// and zero implicit padding.
///
/// # Gain Length Convention
/// When `gain_offset != RANK1_GAIN_UNUSED` (`u32::MAX`), the optional diagonal gain vector occupies
/// `arena[gain_offset .. gain_offset + u_span.len]` (one gain scalar per output element, matching
/// `u_span.len == d_out`). When `gain_offset == RANK1_GAIN_UNUSED`, diagonal gain is omitted.
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rank1Descriptor {
    /// Index of target layer for Rank-1 residual injection (`offset 0..4`).
    pub layer_idx: u32,
    /// Low-rank adapter rank (must equal `1` for Rank-1 kernels, `offset 4..8`).
    pub rank: u32,
    /// Element span of left column projection vector $\mathbf{u}$ (`len == d_out`, `offset 8..16`).
    pub u_span: ArenaSpan,
    /// Element span of right row projection vector $\mathbf{v}$ (`len == d_in`, `offset 16..24`).
    pub v_span: ArenaSpan,
    /// Element offset of optional diagonal gain vector $\mathbf{g}$ of length `u_span.len`
    /// ([`RANK1_GAIN_UNUSED`] = `u32::MAX` when unused, `offset 24..28`).
    pub gain_offset: u32,
    /// Global finite scaling multiplier $s$ for Rank-1 residual injection (`offset 28..32`).
    pub scale_factor: f32,
}

impl Default for Rank1Descriptor {
    #[inline(always)]
    fn default() -> Self {
        Self {
            layer_idx: 0,
            rank: 0,
            u_span: ArenaSpan::default(),
            v_span: ArenaSpan::default(),
            gain_offset: RANK1_GAIN_UNUSED,
            scale_factor: 0.0,
        }
    }
}

/// 64-byte cacheline-aligned Plain Old Data (`Pod`) **KV-keyframe Predictive Motion Delta (P-Frame) Packet** (Section 4.3).
///
/// Stores the 64-head inter-step Q7 phase motion vector $\boldsymbol{\Delta\theta}_t \in \mathrm{Q7}^{64}$
/// (`offset 0..64`) between consecutive steps inside a Variable Group of Pictures (GOP) memory/persona stream.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvMotionDeltaPacket {
    /// 64-head Q7 predictive phase motion delta vector $\boldsymbol{\Delta\theta}_t$ (`offset 0..64`).
    pub motion_deltas: [PhaseQ7; MAX_ATTENTION_HEADS],
}

impl Default for KvMotionDeltaPacket {
    #[inline(always)]
    fn default() -> Self {
        Self::new([0; MAX_ATTENTION_HEADS])
    }
}

impl KvMotionDeltaPacket {
    /// Constructs a 64-byte aligned [`KvMotionDeltaPacket`] from `motion_deltas`.
    #[inline(always)]
    #[must_use]
    pub const fn new(motion_deltas: [PhaseQ7; MAX_ATTENTION_HEADS]) -> Self {
        Self { motion_deltas }
    }
}

impl_pod_for_primitives!(
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64,
    u128,
    i128,
    usize,
    isize,
    f32,
    f64,
    LmnkStatus,
    RuntimeConfig,
    AttributionTag,
    BoundarySignal,
    LacunaPartition8,
    LiminalThresholdState,
    TaggedPtr,
    ArenaSpan,
    PagedBitmask,
    SubheadConfig,
    Rank1Descriptor,
    KvMotionDeltaPacket,
);

// SAFETY: Fixed-size arrays of `Pod` elements have contiguous layout with no extra padding traps.
unsafe impl<T: Pod, const N: usize> Pod for [T; N] {}

/// Executes an $\mathcal{O}(1)$ cascade pointer lookup combining parent and child phase deltas.
#[inline(always)]
#[must_use]
pub fn cascade_tagged_ptr(
    parent_ptr: TaggedPtr,
    child_addr_offset: u64,
    child_phase_delta: PhaseQ7,
) -> TaggedPtr {
    let new_addr = parent_ptr.unpack_addr().wrapping_add(child_addr_offset) & TaggedPtr::ADDR_MASK;
    let combined_phase = branchless_q7_add_sat(parent_ptr.unpack_phase(), child_phase_delta);
    TaggedPtr::pack(new_addr, parent_ptr.unpack_tag(), combined_phase)
}

const _: () = {
    assert!(size_of::<TaggedPtr>() == 8 && align_of::<TaggedPtr>() == 8);
    assert!(size_of::<ArenaSpan>() == 8 && align_of::<ArenaSpan>() == 4);
    assert!(size_of::<PagedBitmask>() == 32 && align_of::<PagedBitmask>() == 32);
    assert!(size_of::<SubheadConfig>() == 32 && align_of::<SubheadConfig>() == 16);
    assert!(size_of::<Rank1Descriptor>() == 32 && align_of::<Rank1Descriptor>() == 32);
    assert!(core::mem::offset_of!(Rank1Descriptor, layer_idx) == 0);
    assert!(core::mem::offset_of!(Rank1Descriptor, rank) == 4);
    assert!(core::mem::offset_of!(Rank1Descriptor, u_span) == 8);
    assert!(core::mem::offset_of!(Rank1Descriptor, v_span) == 16);
    assert!(core::mem::offset_of!(Rank1Descriptor, gain_offset) == 24);
    assert!(core::mem::offset_of!(Rank1Descriptor, scale_factor) == 28);
    assert!(size_of::<KvMotionDeltaPacket>() == 64 && align_of::<KvMotionDeltaPacket>() == 64);
    assert!(size_of::<BoundarySignal>() == 32 && align_of::<BoundarySignal>() == 32);
    assert!(core::mem::offset_of!(BoundarySignal, partition) == 0);
    assert!(core::mem::offset_of!(BoundarySignal, pressure_q15) == 16);
    assert!(core::mem::offset_of!(BoundarySignal, delta_q15) == 18);
    assert!(core::mem::offset_of!(BoundarySignal, attribution_tag) == 20);
    assert!(core::mem::offset_of!(BoundarySignal, lacuna_code) == 21);
    assert!(core::mem::offset_of!(BoundarySignal, contraction_q7) == 22);
    assert!(core::mem::offset_of!(BoundarySignal, chatter_q7) == 23);
    assert!(core::mem::offset_of!(BoundarySignal, signal_flags) == 24);
    assert!(core::mem::offset_of!(BoundarySignal, dominant_lane) == 25);
    assert!(core::mem::offset_of!(BoundarySignal, root_cause_code) == 26);
    assert!(core::mem::offset_of!(BoundarySignal, breach_mask) == 27);
    assert!(core::mem::offset_of!(BoundarySignal, epoch_id) == 28);
    assert!((ATTRACTOR_SLOT_ID_MASK as usize) == (MAX_ATTRACTOR_DOMAINS - 1));
    assert!(pack_action_slot_tag(7, 0x1F) == 0xFF);
    assert!(unpack_attractor_slot_from_tag(pack_action_slot_tag(5, 0x13)) == 5);
    assert!(unpack_action_mode_from_tag(pack_action_slot_tag(5, 0x13)) == 0x13);
};
