// name: src/core/src/traits/primitives/packets_ffi.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! .gcso descriptor/snapshot validators, PPRC keyframe/motion-delta packets, and lifetime-bound FFI handle guards.

use super::*;
#[allow(unused_imports)]
use super::{constants_pod::*, lut_crc32::*, scalar_math::*, swar_ops::*};

// ===================================================================
// 5. Descriptor & .gcso Snapshot Validators + Branchless IEEE 802.3 CRC32
// ===================================================================

/// Constructs a canonical `gcso_descriptor_header_t` for type `T` stamped with [`GCSO_ABI_VERSION_HEX`].
#[inline(always)]
#[must_use]
pub const fn init_descriptor_header<T>() -> gcso_descriptor_header_t {
    gcso_descriptor_header_t {
        struct_size: size_of::<T>() as u32,
        abi_version: GCSO_ABI_VERSION_HEX,
    }
}

/// Validates a `gcso_descriptor_header_t` against expected structure byte size and [`GCSO_ABI_VERSION_HEX`].
///
/// # Errors
/// Returns [`GCSO_ERROR_VERSION_MISMATCH`] if `struct_size` or `abi_version` does not match.
#[inline(always)]
pub const fn validate_descriptor_header<T>(header: &gcso_descriptor_header_t) -> GcsoResult<()> {
    let valid = (header.struct_size as usize == size_of::<T>())
        & (header.abi_version == GCSO_ABI_VERSION_HEX);
    if valid {
        Ok(())
    } else {
        Err(GCSO_ERROR_VERSION_MISMATCH)
    }
}

/// Incrementally updates an IEEE 802.3 CRC32 checksum (`0xEDB88320` reflected polynomial) over `bytes`
/// starting from a previous finalized CRC value `crc_seed` (`0` for initial chunk) using the
/// 64B-aligned 256-entry lookup table (`CRC32_IEEE_LUT_256`) with 8-byte unrolled loop execution.
#[inline(always)]
#[must_use]
pub const fn update_crc32_ieee(crc_seed: u32, bytes: &[u8]) -> u32 {
    branchless_update_crc32_ieee(crc_seed, bytes)
}

/// Computes a standard IEEE 802.3 CRC32 checksum (`0xEDB88320` reflected polynomial) over `bytes`
/// using the 64B-aligned 256-entry lookup table (`CRC32_IEEE_LUT_256`) in `#![no_std]` `const fn` mode.
#[inline(always)]
#[must_use]
pub const fn compute_crc32_ieee(bytes: &[u8]) -> u32 {
    branchless_update_crc32_ieee(0, bytes)
}

/// Constructs a canonical `gcso_snapshot_header_t` initialized with magic `"GCSO"` (`0x4F534347`),
/// [`GCSO_ABI_VERSION_HEX`], and `total_size`.
#[inline(always)]
#[must_use]
pub const fn init_snapshot_header(
    total_size: u64,
    timestamp_epoch_sec: u64,
) -> gcso_snapshot_header_t {
    let hdr_size = size_of::<gcso_snapshot_header_t>() as u64;
    gcso_snapshot_header_t {
        magic: GCSO_SNAPSHOT_MAGIC_U32,
        version: GCSO_ABI_VERSION_HEX,
        total_size,
        action_hub_offset: hdr_size,
        attractor_field_offset: hdr_size,
        dpsr_state_offset: hdr_size,
        rank1_state_offset: hdr_size,
        edbc_state_offset: hdr_size,
        checksum_crc32: 0,
        ledger_slot_offset: hdr_size as u32,
        timestamp_epoch_sec,
        reserved_padding: [0; 56],
    }
}

/// Validates a `gcso_snapshot_header_t` against the `.gcso` container magic (`0x4F534347`),
/// [`GCSO_ABI_VERSION_HEX`], and payload buffer bounds.
///
/// # Errors
/// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] if `magic` is invalid, `total_size` is smaller than
/// `size_of::<gcso_snapshot_header_t>()`, `available_bytes < total_size`, or any track offset exceeds `total_size`.
/// Returns [`GCSO_ERROR_VERSION_MISMATCH`] if `version != GCSO_ABI_VERSION_HEX`.
#[inline]
pub const fn validate_snapshot_header(
    header: &gcso_snapshot_header_t,
    available_bytes: usize,
) -> GcsoResult<()> {
    if header.magic != GCSO_SNAPSHOT_MAGIC_U32 {
        return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
    }
    if header.version != GCSO_ABI_VERSION_HEX {
        return Err(GCSO_ERROR_VERSION_MISMATCH);
    }
    let min_hdr = size_of::<gcso_snapshot_header_t>() as u64;
    let total = header.total_size;
    let daes_off = header.ledger_slot_offset as u64;
    let offsets_in_bounds = (total >= min_hdr)
        & ((available_bytes as u64) >= total)
        & (header.action_hub_offset >= min_hdr)
        & (header.action_hub_offset <= total)
        & (header.attractor_field_offset >= min_hdr)
        & (header.attractor_field_offset <= total)
        & (header.dpsr_state_offset >= min_hdr)
        & (header.dpsr_state_offset <= total)
        & (header.rank1_state_offset >= min_hdr)
        & (header.rank1_state_offset <= total)
        & (header.edbc_state_offset >= min_hdr)
        & (header.edbc_state_offset <= total)
        & (daes_off >= min_hdr)
        & (daes_off <= total);
    if offsets_in_bounds {
        Ok(())
    } else {
        Err(GCSO_ERROR_CONTAINER_CORRUPTED)
    }
}

/// Validates a `.gcso` snapshot header, computes the IEEE 802.3 CRC32 checksum over the payload
/// bytes following the 128-byte header (`container_bytes[128..total_size]`), and stamps
/// `header.checksum_crc32` in-place.
///
/// # Errors
/// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] if header offsets or payload bounds are invalid.
/// Returns [`GCSO_ERROR_VERSION_MISMATCH`] if `header.version != GCSO_ABI_VERSION_HEX`.
#[inline]
pub fn finalize_snapshot_header_crc32(
    header: &mut gcso_snapshot_header_t,
    container_bytes: &[u8],
) -> GcsoResult<u32> {
    validate_snapshot_header(header, container_bytes.len())?;
    let hdr_size = size_of::<gcso_snapshot_header_t>();
    let crc = compute_crc32_ieee(&container_bytes[hdr_size..(header.total_size as usize)]);
    header.checksum_crc32 = crc;
    Ok(crc)
}

/// Validates a `.gcso` snapshot header and verifies the IEEE 802.3 CRC32 checksum over the payload
/// bytes following the 128-byte header (`container_bytes[128..total_size]`).
///
/// # Errors
/// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] if header bounds or CRC32 verification fail.
/// Returns [`GCSO_ERROR_VERSION_MISMATCH`] if `header.version != GCSO_ABI_VERSION_HEX`.
#[inline]
pub fn verify_snapshot_payload_crc32(
    header: &gcso_snapshot_header_t,
    container_bytes: &[u8],
) -> GcsoResult<()> {
    validate_snapshot_header(header, container_bytes.len())?;
    if header.checksum_crc32 != 0 {
        let hdr_size = size_of::<gcso_snapshot_header_t>();
        let computed = compute_crc32_ieee(&container_bytes[hdr_size..(header.total_size as usize)]);
        if computed != header.checksum_crc32 {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }
    }
    Ok(())
}

/// 128-byte cacheline-aligned Plain Old Data (`Pod`) **KV-keyframe Keyframe (I-Frame) Packet** (Section 4.3).
///
/// Packs the 64-byte [`gcso_kv_keyframe_header_t`] metadata header (`offset 0..64`) and the
/// uncompressed 64-head Q7 phase anchor snapshot (`offset 64..128`) for lightweight video-codec-style
/// `.gcso` memory & persona streams (Steam Workshop distribution ready).
///
/// The 24-byte `header.reserved` field encodes:
/// - `reserved[0..4]`: Little-endian IEEE 802.3 CRC32 checksum of `anchor_phases`.
/// - `reserved[4..8]`: Little-endian 32-bit `provenance_bits` causal mask.
/// - `reserved[8..12]`: Little-endian 32-bit macro Toroidal $k$-Means `cluster_id`.
/// - `reserved[12]`: 3-bit working-lane `slot_id` (`0..=7`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvKeyframePacket {
    /// 64-byte C-ABI KV-keyframe index header (`offset 0..64`, `frame_type == 0`).
    pub header: gcso_kv_keyframe_header_t,
    /// 64-head Q7 phase anchor snapshot on $SO(2)^{64}$ (`offset 64..128`).
    pub anchor_phases: [gcso_q7_t; MAX_ATTENTION_HEADS],
}

impl Default for KvKeyframePacket {
    #[inline(always)]
    fn default() -> Self {
        Self {
            header: gcso_kv_keyframe_header_t {
                frame_type: 0,
                token_index: 0,
                gop_length: 1,
                composite_vector_length: 0.0,
                von_mises_kappa: 0.0,
                sparse_scalar_residual: 0.0,
                icache_payload_offset: 0,
                pcache_payload_offset: 0,
                reserved: [0; 24],
            },
            anchor_phases: [0; MAX_ATTENTION_HEADS],
        }
    }
}

impl KvKeyframePacket {
    /// Computes the IEEE 802.3 CRC32 checksum over the 64-head `anchor_phases` payload in `const fn` mode.
    #[inline(always)]
    #[must_use]
    pub const fn compute_anchor_crc32(anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> u32 {
        let mut raw_bytes = [0u8; MAX_ATTENTION_HEADS];
        let mut idx = 0usize;
        while idx < MAX_ATTENTION_HEADS {
            raw_bytes[idx] = anchor_phases[idx] as u8;
            idx += 1;
        }
        compute_crc32_ieee(&raw_bytes)
    }

    /// Constructs a 128-byte [`KvKeyframePacket`] (I-Frame Key Anchor) stamped with IEEE 802.3
    /// CRC32 integrity protection and causal provenance metadata.
    #[inline(always)]
    #[must_use]
    pub const fn new(
        token_index: u32,
        gop_length: u32,
        composite_vector_length: f32,
        von_mises_kappa: f32,
        sparse_scalar_residual: f32,
        provenance_bits: u32,
        cluster_id: u32,
        slot_id: u8,
        anchor_phases: [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> Self {
        let crc = Self::compute_anchor_crc32(&anchor_phases);
        let crc_le = crc.to_le_bytes();
        let prov_le = provenance_bits.to_le_bytes();
        let cid_le = cluster_id.to_le_bytes();
        let mut reserved = [0u8; 24];
        reserved[0] = crc_le[0];
        reserved[1] = crc_le[1];
        reserved[2] = crc_le[2];
        reserved[3] = crc_le[3];
        reserved[4] = prov_le[0];
        reserved[5] = prov_le[1];
        reserved[6] = prov_le[2];
        reserved[7] = prov_le[3];
        reserved[8] = cid_le[0];
        reserved[9] = cid_le[1];
        reserved[10] = cid_le[2];
        reserved[11] = cid_le[3];
        reserved[12] = slot_id & ATTRACTOR_SLOT_ID_MASK;

        Self {
            header: gcso_kv_keyframe_header_t {
                frame_type: 0,
                token_index,
                gop_length: if gop_length > 0 { gop_length } else { 1 },
                composite_vector_length,
                von_mises_kappa,
                sparse_scalar_residual,
                icache_payload_offset: 64,
                pcache_payload_offset: 128,
                reserved,
            },
            anchor_phases,
        }
    }

    /// Extracts the stamped IEEE 802.3 CRC32 checksum of `anchor_phases` from `header.reserved[0..4]`.
    #[inline(always)]
    #[must_use]
    pub const fn payload_crc32(&self) -> u32 {
        u32::from_le_bytes([
            self.header.reserved[0],
            self.header.reserved[1],
            self.header.reserved[2],
            self.header.reserved[3],
        ])
    }

    /// Extracts the 32-bit `provenance_bits` causal mask from `header.reserved[4..8]`.
    #[inline(always)]
    #[must_use]
    pub const fn provenance_bits(&self) -> u32 {
        u32::from_le_bytes([
            self.header.reserved[4],
            self.header.reserved[5],
            self.header.reserved[6],
            self.header.reserved[7],
        ])
    }

    /// Extracts the 32-bit Toroidal $k$-Means `cluster_id` from `header.reserved[8..12]`.
    #[inline(always)]
    #[must_use]
    pub const fn cluster_id(&self) -> u32 {
        u32::from_le_bytes([
            self.header.reserved[8],
            self.header.reserved[9],
            self.header.reserved[10],
            self.header.reserved[11],
        ])
    }

    /// Extracts the 3-bit working-lane `slot_id` (`0..=7`) from `header.reserved[12]`.
    #[inline(always)]
    #[must_use]
    pub const fn slot_id(&self) -> u8 {
        self.header.reserved[12] & ATTRACTOR_SLOT_ID_MASK
    }

    /// Verifies that `header.frame_type == 0` (I-Frame) and that the IEEE 802.3 CRC32 checksum of
    /// `anchor_phases` matches `self.payload_crc32()`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] if `frame_type != 0` or CRC32 verification fails.
    #[inline(always)]
    pub const fn verify_integrity(&self) -> GcsoResult<()> {
        if self.header.frame_type != 0 {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }
        let expected_crc = self.payload_crc32();
        if expected_crc != 0 && Self::compute_anchor_crc32(&self.anchor_phases) != expected_crc {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }
        Ok(())
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
    pub motion_deltas: [gcso_q7_t; MAX_ATTENTION_HEADS],
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
    pub const fn new(motion_deltas: [gcso_q7_t; MAX_ATTENTION_HEADS]) -> Self {
        Self { motion_deltas }
    }
}

// ===================================================================
// 6. Type-Safe Lifetime-Bound FFI Handle Guards
// ===================================================================

#[inline(always)]
pub(super) fn validate_non_null_aligned_ptr<T>(raw_ptr: *mut T, align_mask: usize) -> GcsoResult<NonNull<T>> {
    if raw_ptr.is_null() {
        return Err(GCSO_ERROR_NULL_POINTER);
    }
    if ((raw_ptr as usize) & align_mask) != 0 {
        return Err(GCSO_ERROR_MISALIGNED_POINTER);
    }
    // SAFETY: Nullness checked above.
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
    /// Returns [`GCSO_ERROR_NULL_POINTER`] if `raw_ptr` is null.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `raw_ptr` violates `align_of::<T>()`.
    #[inline(always)]
    pub unsafe fn from_raw(raw_ptr: *const T) -> GcsoResult<Self> {
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
    /// Returns [`GCSO_ERROR_NULL_POINTER`] if `raw_ptr` is null.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `raw_ptr` is not `ALIGN`-byte aligned.
    #[inline(always)]
    pub unsafe fn from_raw_aligned<const ALIGN: usize>(raw_ptr: *const T) -> GcsoResult<Self> {
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
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `raw_ptr` is non-null and violates `align_of::<T>()`.
    #[inline(always)]
    pub unsafe fn from_raw_opt(raw_ptr: *const T) -> GcsoResult<Option<Self>> {
        if raw_ptr.is_null() {
            Ok(None)
        } else {
            // SAFETY: Delegated to `from_raw` after non-null check.
            unsafe { Self::from_raw(raw_ptr).map(Some) }
        }
    }

    /// Enforces `GCSO_RESTRICT` non-aliasing invariant against a mutable `FfiHandleMut`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if the two handles overlap in memory.
    #[inline(always)]
    pub fn ensure_disjoint<U>(&self, other: &FfiHandleMut<'_, U>) -> GcsoResult<()> {
        other.ensure_disjoint_ref(self)
    }

    /// Safely reinterprets `self` as a reference handle to another `Pod` type `U` of equal or smaller size.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `size_of::<U>() == 0` or `size_of::<U>() > size_of::<T>()`.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the pointer violates `align_of::<U>()`.
    #[inline(always)]
    pub fn try_cast<U: Pod>(&self) -> GcsoResult<FfiHandleRef<'a, U>>
    where
        T: Pod,
    {
        if U::IS_ZST || size_of::<U>() > size_of::<T>() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
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
    /// Returns [`GCSO_ERROR_NULL_POINTER`] if `raw_ptr` is null.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `raw_ptr` violates `align_of::<T>()`.
    #[inline(always)]
    pub unsafe fn from_raw(raw_ptr: *mut T) -> GcsoResult<Self> {
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
    /// Returns [`GCSO_ERROR_NULL_POINTER`] if `raw_ptr` is null.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `raw_ptr` is not `ALIGN`-byte aligned.
    #[inline(always)]
    pub unsafe fn from_raw_aligned<const ALIGN: usize>(raw_ptr: *mut T) -> GcsoResult<Self> {
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
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `raw_ptr` is non-null and violates `align_of::<T>()`.
    #[inline(always)]
    pub unsafe fn from_raw_opt(raw_ptr: *mut T) -> GcsoResult<Option<Self>> {
        if raw_ptr.is_null() {
            Ok(None)
        } else {
            // SAFETY: Delegated to `from_raw` after non-null check.
            unsafe { Self::from_raw(raw_ptr).map(Some) }
        }
    }

    /// Validates two mutable FFI pointers and enforces `GCSO_RESTRICT` non-overlapping memory ranges.
    ///
    /// # Safety
    /// Caller must ensure `ptr_a` and `ptr_b` point to valid initialized instances of `T` and `U` for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_NULL_POINTER`] if either pointer is null.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if either pointer is misaligned.
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if their memory footprints overlap.
    #[inline(always)]
    pub unsafe fn from_raw_disjoint_pair<U>(
        ptr_a: *mut T,
        ptr_b: *mut U,
    ) -> GcsoResult<(Self, FfiHandleMut<'a, U>)> {
        // SAFETY: Validated for non-nullness, alignment, and pairwise disjointness before returning.
        let handle_a = unsafe { Self::from_raw(ptr_a)? };
        // SAFETY: Validated for non-nullness and alignment, then checked against `handle_a` below.
        let handle_b = unsafe { FfiHandleMut::<U>::from_raw(ptr_b)? };
        handle_a.ensure_disjoint(&handle_b)?;
        Ok((handle_a, handle_b))
    }

    /// Validates three mutable FFI pointers and enforces pairwise `GCSO_RESTRICT` non-overlapping memory ranges.
    ///
    /// # Safety
    /// Caller must ensure `ptr_a`, `ptr_b`, and `ptr_c` point to valid initialized instances for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_NULL_POINTER`] if any pointer is null.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if any pointer is misaligned.
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any two memory footprints overlap.
    #[inline(always)]
    pub unsafe fn from_raw_disjoint_triple<U, V>(
        ptr_a: *mut T,
        ptr_b: *mut U,
        ptr_c: *mut V,
    ) -> GcsoResult<(Self, FfiHandleMut<'a, U>, FfiHandleMut<'a, V>)> {
        // SAFETY: Validated for non-nullness, alignment, and pairwise disjointness before returning.
        let handle_a = unsafe { Self::from_raw(ptr_a)? };
        // SAFETY: Checked for disjointness below.
        let handle_b = unsafe { FfiHandleMut::<U>::from_raw(ptr_b)? };
        // SAFETY: Checked for disjointness below.
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

    /// Enforces `GCSO_RESTRICT` non-aliasing invariant against another mutable FFI handle.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if the two handles overlap in memory.
    #[inline(always)]
    pub fn ensure_disjoint<U>(&self, other: &FfiHandleMut<'_, U>) -> GcsoResult<()> {
        if self.overlaps(other) {
            Err(GCSO_ERROR_INVALID_ARGUMENT)
        } else {
            Ok(())
        }
    }

    /// Enforces `GCSO_RESTRICT` non-aliasing invariant against an immutable `FfiHandleRef`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if the two handles overlap in memory.
    #[inline(always)]
    pub fn ensure_disjoint_ref<U>(&self, other: &FfiHandleRef<'_, U>) -> GcsoResult<()> {
        if raw_regions_overlap_or_same_zst(
            self.ptr.as_ptr() as usize,
            size_of::<T>(),
            other.as_ptr() as usize,
            size_of::<U>(),
        ) {
            Err(GCSO_ERROR_INVALID_ARGUMENT)
        } else {
            Ok(())
        }
    }

    /// Safely reinterprets `self` as a mutable handle to another `Pod` type `U` of equal or smaller size.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `size_of::<U>() == 0` or `size_of::<U>() > size_of::<T>()`.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the pointer violates `align_of::<U>()`.
    #[inline(always)]
    pub fn try_cast_mut<U: Pod>(&mut self) -> GcsoResult<FfiHandleMut<'_, U>>
    where
        T: Pod,
    {
        if U::IS_ZST || size_of::<U>() > size_of::<T>() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
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

