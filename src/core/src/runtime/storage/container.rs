// name: src/core/src/runtime/storage/container.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unified `.lmnk` container writer, alignment-safe [`LmnkContainerView`], and [`MappedStorageEngine`].

use core::mem::{align_of, offset_of, size_of};

use super::{
    KvKeyframeCache, KvKeyframeHeader, LMNK_CONTAINER_HEADER_BYTES, LMNK_CONTAINER_KNOWN_FLAGS,
    LMNK_CONTAINER_MAGIC, LMNK_MAX_CONTAINER_BYTES, LMNK_MAX_CONTAINER_SECTIONS,
    LMNK_SECTION_BLOOM_FILTER, LMNK_SECTION_CAUSAL_PROVENANCE, LMNK_SECTION_ENTRY_BYTES,
    LMNK_SECTION_LEDGER_SLOT, LMNK_SECTION_PHASE_ANCHORS, LMNK_SECTION_POINTER_TRAILS,
    LmnkContainerHeader, LmnkSectionEntry, init_container_header,
};
use crate::foundation::{
    AlignedSlice, Pod, compute_aligned64_payload_crc32_ieee, compute_crc32_ieee,
};
use crate::liminal::{
    LMNK_ABI_VERSION_HEX, LMNK_ERR_BUFFER_TOO_SMALL, LMNK_ERR_CONTAINER_CORRUPTED,
    LMNK_ERR_INVALID_ARGUMENT, LMNK_ERR_TOPOLOGY_MISMATCH, LMNK_ERR_VERSION_MISMATCH, LmnkResult,
    RuntimeConfig, validate_abi_version,
};
use crate::runtime::gateway::signals::CausalProvenanceReport;
use crate::runtime::topology::CORE_TOPOLOGY;
use crate::stigmergy::{
    CognitiveTrajectoryLedger, ContrastivePhaseBloomFilter, LEDGER_FLAG_COGNITIVE_TRAJECTORY,
    LedgerSlot, PhaseAnchorRecord, PointerTrail,
};

/// Cold-path trait for Zero-Overhead In-Memory Mapped Storage (mapped-storage) (**Storage / Persistence Level**).
pub trait MappedStorage: Send + Sync {
    /// Serializes active runtime state into a `.lmnk` binary snapshot buffer.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`] if `buffer.len() < *required_size`.
    fn serialize_snapshot(&self, buffer: &mut [u8], required_size: &mut usize) -> LmnkResult<()>;

    /// Deserializes a `.lmnk` binary snapshot buffer to restore context state.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_CONTAINER_CORRUPTED`] or [`LMNK_ERR_VERSION_MISMATCH`]
    /// if the snapshot header or checksum is invalid.
    fn deserialize_snapshot(
        &mut self,
        buffer: &[u8],
        config_out: &mut RuntimeConfig,
    ) -> LmnkResult<()>;

    /// Retrieves the byte offset for a specific track in a `.lmnk` container
    /// (`0` = `CORE`, `1` = `I-CACHE`, `2` = `P-CACHE`, `3` = `SNAPSHOT`).
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `track_id > 3`.
    #[inline(always)]
    fn get_track_offset(&self, track_id: u32) -> LmnkResult<u64> {
        let header_size = LMNK_CONTAINER_HEADER_BYTES as u64;
        match track_id {
            0 => Ok(header_size),
            1 => Ok(header_size + 1024),
            2 => Ok(header_size + 2048),
            3 => Ok(header_size + 3072),
            _ => Err(LMNK_ERR_INVALID_ARGUMENT),
        }
    }

    /// Reads and validates a `.lmnk` container's 64-byte [`LmnkContainerHeader`] via [`LmnkContainerView::parse`].
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`], [`LMNK_ERR_CONTAINER_CORRUPTED`],
    /// [`LMNK_ERR_VERSION_MISMATCH`], or [`LMNK_ERR_TOPOLOGY_MISMATCH`] if validation fails.
    #[inline]
    fn inspect_and_verify_container(
        &self,
        container_bytes: &[u8],
    ) -> LmnkResult<LmnkContainerHeader> {
        let view = LmnkContainerView::parse(container_bytes)?;
        Ok(*view.header())
    }

    /// Stamps `header_crc32` (and `topology_crc32`) on `header` and writes the 64-byte header
    /// into the beginning of `container_bytes`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`] if `container_bytes.len() < 64`, or
    /// [`LMNK_ERR_CONTAINER_CORRUPTED`] / [`LMNK_ERR_VERSION_MISMATCH`] if header invariants fail.
    #[inline]
    fn stamp_and_write_container_header(
        &self,
        header: &mut LmnkContainerHeader,
        container_bytes: &mut [u8],
    ) -> LmnkResult<u32> {
        if container_bytes.len() < LMNK_CONTAINER_HEADER_BYTES {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        if header.magic != LMNK_CONTAINER_MAGIC {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }
        if header.version != LMNK_ABI_VERSION_HEX {
            return Err(LMNK_ERR_VERSION_MISMATCH);
        }
        if (container_bytes.len() as u64) < header.total_size {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }
        header.topology_crc32 = CORE_TOPOLOGY.topology_crc32;
        header.header_size = LMNK_CONTAINER_HEADER_BYTES as u16;
        header.section_table_offset = LMNK_CONTAINER_HEADER_BYTES as u32;
        let crc = header.finalize_crc32();
        let le = header.to_le_bytes();
        let dst = container_bytes
            .get_mut(..LMNK_CONTAINER_HEADER_BYTES)
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
        dst.copy_from_slice(&le);
        Ok(crc)
    }

    /// Extracts a zero-copy byte slice `&'a [u8]` of length `track_len` for `track_id` (`0..=3`) from `container_bytes`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `track_id > 3`.
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`] if `container_bytes` cannot hold `offset + track_len`.
    #[inline]
    fn extract_track_slice<'a>(
        &self,
        container_bytes: &'a [u8],
        track_id: u32,
        track_len: usize,
    ) -> LmnkResult<&'a [u8]> {
        let start = usize::try_from(self.get_track_offset(track_id)?)
            .map_err(|_| LMNK_ERR_BUFFER_TOO_SMALL)?;
        let end = start
            .checked_add(track_len)
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
        container_bytes
            .get(start..end)
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)
    }

    /// Extracts a hardware-aligned zero-copy [`AlignedSlice`] (`AlignedSlice<'a, T, ALIGN>`) of `elem_count` elements
    /// for `track_id` (`0..=3`) directly from a mapped-storage `.lmnk` container buffer.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `track_id > 3` or `T` is zero-sized.
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`] if `container_bytes` is too small.
    /// Returns [`crate::liminal::LMNK_ERR_ALIGNMENT`] if the track start address violates `ALIGN`.
    #[inline]
    fn extract_aligned_track_slice<'a, T: Pod, const ALIGN: usize>(
        &self,
        container_bytes: &'a [u8],
        track_id: u32,
        elem_count: usize,
    ) -> LmnkResult<AlignedSlice<'a, T, ALIGN>> {
        let start = usize::try_from(self.get_track_offset(track_id)?)
            .map_err(|_| LMNK_ERR_BUFFER_TOO_SMALL)?;
        let typed_slice = T::slice_from_bytes_at(container_bytes, start, elem_count)?;
        AlignedSlice::<T, ALIGN>::new(typed_slice)
    }
}

/// 64-byte cacheline-aligned Plain Old Data (`Pod`) descriptor and execution engine for
/// Zero-Copy In-Memory Mapped Storage (`.lmnk` containers), KV-keyframe keyframe/delta video-stream
/// memory & persona reproduction, and 64-byte phase motion-delta packet dispatch.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MappedStorageEngine {
    /// Expected container size in bytes (`offset 0..8`).
    pub file_size_bytes: u64,
    /// Vendor-neutral storage policy bitflags (`offset 8..12`).
    pub flags: u32,
    /// Bound topology CRC32 (`offset 12..16`).
    pub topology_crc32: u32,
    /// Explicit zero padding (`offset 16..64`).
    pub reserved: [u8; 48],
}

// SAFETY: `MappedStorageEngine` is a `#[repr(C, align(64))]` `Pod` of exact size 64 bytes with zero padding.
unsafe impl Pod for MappedStorageEngine {}

impl Default for MappedStorageEngine {
    #[inline(always)]
    fn default() -> Self {
        Self::new(0, 1)
    }
}

impl MappedStorageEngine {
    /// Constructs a canonical 64-byte [`MappedStorageEngine`] initialized with `file_size_bytes` and `flags`.
    #[inline(always)]
    #[must_use]
    pub const fn new(file_size_bytes: u64, flags: u32) -> Self {
        Self {
            file_size_bytes,
            flags,
            topology_crc32: CORE_TOPOLOGY.topology_crc32,
            reserved: [0; 48],
        }
    }

    /// Validates the embedded [`MappedStorageEngine`] fields.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `flags == 0` when `file_size_bytes == 0`.
    #[inline(always)]
    pub const fn validate(&self) -> LmnkResult<()> {
        if self.file_size_bytes == 0 && self.flags == 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        Ok(())
    }
}

impl KvKeyframeCache for MappedStorageEngine {
    #[inline]
    fn seek_to_token(
        &mut self,
        token_index: u32,
        header_out: &mut KvKeyframeHeader,
    ) -> LmnkResult<()> {
        header_out.token_index = token_index;
        header_out.gop_length = 1;
        header_out.composite_vector_length = 1.0;
        Ok(())
    }
}

impl MappedStorage for MappedStorageEngine {
    #[inline]
    fn serialize_snapshot(&self, buffer: &mut [u8], required_size: &mut usize) -> LmnkResult<()> {
        *required_size = LMNK_CONTAINER_HEADER_BYTES;
        if buffer.len() < LMNK_CONTAINER_HEADER_BYTES {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        let mut hdr = init_container_header(self.flags, LMNK_CONTAINER_HEADER_BYTES as u64, 0, 0);
        let _ = self.stamp_and_write_container_header(&mut hdr, buffer)?;
        Ok(())
    }

    #[inline]
    fn deserialize_snapshot(
        &mut self,
        buffer: &[u8],
        _config_out: &mut RuntimeConfig,
    ) -> LmnkResult<()> {
        let view = LmnkContainerView::parse(buffer)?;
        self.file_size_bytes = view.header().total_size;
        self.topology_crc32 = view.header().topology_crc32;
        Ok(())
    }
}

// ===================================================================
// 8.5. Unified `.lmnk` Container Writer & Alignment-Safe Zero-Panic View
// ===================================================================

/// Borrowed section descriptor consumed by [`required_container_byte_len`] and
/// [`write_container_sections_to_bytes`].
#[derive(Debug, Clone, Copy)]
pub struct LmnkSectionInput<'a> {
    /// Section kind ID ([`super::LmnkSectionKind`] or custom `u16`).
    pub kind: u16,
    /// Per-section flags.
    pub flags: u16,
    /// Raw section payload bytes (must be non-empty).
    pub payload: &'a [u8],
}

#[inline(always)]
const fn align_up_64_checked(value: usize) -> Option<usize> {
    match value.checked_add(63) {
        Some(v) => Some(v & !63usize),
        None => None,
    }
}

/// Computes the exact 64B-aligned container byte size required for `sections`.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `sections.len() > LMNK_MAX_CONTAINER_SECTIONS`,
/// any section payload is empty, duplicate section kinds exist, or arithmetic overflows `LMNK_MAX_CONTAINER_BYTES`.
#[inline]
pub fn required_container_byte_len(sections: &[LmnkSectionInput<'_>]) -> LmnkResult<usize> {
    if sections.len() > usize::from(LMNK_MAX_CONTAINER_SECTIONS) {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    let table_bytes = sections
        .len()
        .checked_mul(LMNK_SECTION_ENTRY_BYTES)
        .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
    let table_end = LMNK_CONTAINER_HEADER_BYTES
        .checked_add(table_bytes)
        .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
    let mut cursor = align_up_64_checked(table_end).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;

    let mut i = 0usize;
    while i < sections.len() {
        let sec = sections.get(i).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        if sec.payload.is_empty() {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let mut prev = 0usize;
        while prev < i {
            let other = sections.get(prev).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
            if other.kind == sec.kind {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            }
            prev += 1;
        }
        let off = align_up_64_checked(cursor).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        let end = off
            .checked_add(sec.payload.len())
            .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        cursor = align_up_64_checked(end).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        i += 1;
    }
    if cursor > LMNK_MAX_CONTAINER_BYTES {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    Ok(cursor)
}

/// Serializes arbitrary non-empty `sections` into a unified 64B-aligned `.lmnk` container buffer.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `sections` violates uniqueness/bounds/non-empty rules,
/// or [`LMNK_ERR_BUFFER_TOO_SMALL`] if `dst_bytes` is smaller than the required container size.
#[inline]
pub fn write_container_sections_to_bytes(
    container_flags: u32,
    timestamp_epoch_sec: u64,
    sections: &[LmnkSectionInput<'_>],
    dst_bytes: &mut [u8],
) -> LmnkResult<usize> {
    let total_len = required_container_byte_len(sections)?;
    if dst_bytes.len() < total_len {
        return Err(LMNK_ERR_BUFFER_TOO_SMALL);
    }
    let buf = dst_bytes
        .get_mut(..total_len)
        .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
    buf.fill(0);

    let table_end = LMNK_CONTAINER_HEADER_BYTES + sections.len() * LMNK_SECTION_ENTRY_BYTES;
    let mut cursor = align_up_64_checked(table_end).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;

    let mut i = 0usize;
    while i < sections.len() {
        let sec = sections.get(i).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        let off = align_up_64_checked(cursor).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        let len = sec.payload.len();
        let end = off.checked_add(len).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;

        let payload_dst = buf.get_mut(off..end).ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
        payload_dst.copy_from_slice(sec.payload);
        let payload_crc32 = compute_aligned64_payload_crc32_ieee(payload_dst);

        let entry = LmnkSectionEntry {
            kind: sec.kind,
            flags: sec.flags,
            offset: u32::try_from(off).map_err(|_| LMNK_ERR_INVALID_ARGUMENT)?,
            length: u32::try_from(len).map_err(|_| LMNK_ERR_INVALID_ARGUMENT)?,
            payload_crc32,
        };
        let entry_off = LMNK_CONTAINER_HEADER_BYTES + i * LMNK_SECTION_ENTRY_BYTES;
        let entry_dst = buf
            .get_mut(entry_off..entry_off + LMNK_SECTION_ENTRY_BYTES)
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
        entry_dst.copy_from_slice(&entry.to_le_bytes());

        cursor = align_up_64_checked(end).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        i += 1;
    }

    let hdr = init_container_header(
        container_flags,
        total_len as u64,
        timestamp_epoch_sec,
        sections.len() as u16,
    );
    let hdr_dst = buf
        .get_mut(..LMNK_CONTAINER_HEADER_BYTES)
        .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
    hdr_dst.copy_from_slice(&hdr.to_le_bytes());
    Ok(total_len)
}

/// Alignment-safe, zero-panic validated view over a unified `.lmnk` binary container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LmnkContainerView<'a> {
    raw: &'a [u8],
    header: LmnkContainerHeader,
    sections: [LmnkSectionEntry; LMNK_MAX_CONTAINER_SECTIONS as usize],
}

impl<'a> LmnkContainerView<'a> {
    /// Parses and validates an untrusted `.lmnk` container byte slice without panics or
    /// base-pointer alignment assumptions.
    ///
    /// Validation sequence:
    /// 1. Length bounds (`64 <= bytes.len() <= 64 MiB`) and `header.total_size == bytes.len()`.
    /// 2. Magic (`"LMNK"`), ABI version compatibility (`MAJOR.MINOR` strict match and
    ///    `file_patch <= runtime_patch` via [`validate_abi_version`]), header geometry
    ///    (`header_size == 64`, `section_table_offset == 64`, `section_count <= 32`).
    /// 3. Header IEEE 802.3 CRC32 verification (`bytes[0..60]`).
    /// 4. Topology CRC32 verification against [`CORE_TOPOLOGY`] ([`LMNK_ERR_TOPOLOGY_MISMATCH`]).
    /// 5. Monotonic 64B-aligned non-overlapping section table parsing, unique `kind` enforcement,
    ///    per-section IEEE 802.3 CRC32 verification, and POD record structural checks.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`], [`LMNK_ERR_VERSION_MISMATCH`],
    /// [`LMNK_ERR_TOPOLOGY_MISMATCH`], or [`LMNK_ERR_CONTAINER_CORRUPTED`].
    #[inline]
    pub fn parse(bytes: &'a [u8]) -> LmnkResult<Self> {
        if bytes.len() < LMNK_CONTAINER_HEADER_BYTES {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        if bytes.len() > LMNK_MAX_CONTAINER_BYTES {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }

        let hdr_bytes = bytes
            .get(0..LMNK_CONTAINER_HEADER_BYTES)
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
        let header = LmnkContainerHeader::from_le_bytes(hdr_bytes)?;

        if header.magic != LMNK_CONTAINER_MAGIC {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }
        validate_abi_version(header.version)?;
        if header.header_size != LMNK_CONTAINER_HEADER_BYTES as u16
            || header.section_table_offset != LMNK_CONTAINER_HEADER_BYTES as u32
            || header.section_count > LMNK_MAX_CONTAINER_SECTIONS
            || header.total_size != bytes.len() as u64
            || (header.container_flags & !LMNK_CONTAINER_KNOWN_FLAGS) != 0
            || header.reserved_future != [0u8; 20]
        {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }

        let hdr_prefix = bytes.get(0..60).ok_or(LMNK_ERR_CONTAINER_CORRUPTED)?;
        if compute_crc32_ieee(hdr_prefix) != header.header_crc32 {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }

        if header.topology_crc32 != CORE_TOPOLOGY.topology_crc32 {
            return Err(LMNK_ERR_TOPOLOGY_MISMATCH);
        }

        let sec_count = usize::from(header.section_count);
        let table_bytes = sec_count
            .checked_mul(LMNK_SECTION_ENTRY_BYTES)
            .ok_or(LMNK_ERR_CONTAINER_CORRUPTED)?;
        let table_end = LMNK_CONTAINER_HEADER_BYTES
            .checked_add(table_bytes)
            .ok_or(LMNK_ERR_CONTAINER_CORRUPTED)?;
        let mut prev_end = align_up_64_checked(table_end).ok_or(LMNK_ERR_CONTAINER_CORRUPTED)?;
        if bytes.len() < prev_end {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }
        let table_pad = bytes
            .get(table_end..prev_end)
            .ok_or(LMNK_ERR_CONTAINER_CORRUPTED)?;
        if table_pad.iter().any(|&b| b != 0) {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }

        let mut sections = [LmnkSectionEntry::default(); LMNK_MAX_CONTAINER_SECTIONS as usize];

        let mut i = 0usize;
        while i < sec_count {
            let entry_off = LMNK_CONTAINER_HEADER_BYTES
                .checked_add(
                    i.checked_mul(LMNK_SECTION_ENTRY_BYTES)
                        .ok_or(LMNK_ERR_CONTAINER_CORRUPTED)?,
                )
                .ok_or(LMNK_ERR_CONTAINER_CORRUPTED)?;
            let entry_end = entry_off
                .checked_add(LMNK_SECTION_ENTRY_BYTES)
                .ok_or(LMNK_ERR_CONTAINER_CORRUPTED)?;
            let entry_slice = bytes
                .get(entry_off..entry_end)
                .ok_or(LMNK_ERR_CONTAINER_CORRUPTED)?;
            let entry = LmnkSectionEntry::from_le_bytes(entry_slice)?;

            if entry.length == 0 || entry.flags != 0 {
                return Err(LMNK_ERR_CONTAINER_CORRUPTED);
            }
            let mut prev_idx = 0usize;
            while prev_idx < i {
                let existing = sections.get(prev_idx).ok_or(LMNK_ERR_CONTAINER_CORRUPTED)?;
                if existing.kind == entry.kind {
                    return Err(LMNK_ERR_CONTAINER_CORRUPTED);
                }
                prev_idx += 1;
            }

            let sec_off =
                usize::try_from(entry.offset).map_err(|_| LMNK_ERR_CONTAINER_CORRUPTED)?;
            let sec_len =
                usize::try_from(entry.length).map_err(|_| LMNK_ERR_CONTAINER_CORRUPTED)?;
            if (sec_off & 63) != 0 || sec_off != prev_end {
                return Err(LMNK_ERR_CONTAINER_CORRUPTED);
            }
            let sec_end = sec_off
                .checked_add(sec_len)
                .ok_or(LMNK_ERR_CONTAINER_CORRUPTED)?;
            let next_aligned_end =
                align_up_64_checked(sec_end).ok_or(LMNK_ERR_CONTAINER_CORRUPTED)?;
            if next_aligned_end > bytes.len() {
                return Err(LMNK_ERR_CONTAINER_CORRUPTED);
            }
            let pad_tail = bytes
                .get(sec_end..next_aligned_end)
                .ok_or(LMNK_ERR_CONTAINER_CORRUPTED)?;
            if pad_tail.iter().any(|&b| b != 0) {
                return Err(LMNK_ERR_CONTAINER_CORRUPTED);
            }
            let payload = bytes
                .get(sec_off..sec_end)
                .ok_or(LMNK_ERR_CONTAINER_CORRUPTED)?;

            if compute_aligned64_payload_crc32_ieee(payload) != entry.payload_crc32 {
                return Err(LMNK_ERR_CONTAINER_CORRUPTED);
            }

            match entry.kind {
                LMNK_SECTION_PHASE_ANCHORS => {
                    let rec_sz = size_of::<PhaseAnchorRecord>();
                    if (sec_len & (rec_sz - 1)) != 0 {
                        return Err(LMNK_ERR_CONTAINER_CORRUPTED);
                    }
                    let mut off = 0usize;
                    while off < sec_len {
                        let rec = PhaseAnchorRecord::read_unaligned_from(payload, off)
                            .map_err(|_| LMNK_ERR_CONTAINER_CORRUPTED)?;
                        if !rec.verify_integrity() {
                            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
                        }
                        off += rec_sz;
                    }
                }
                LMNK_SECTION_POINTER_TRAILS => {
                    if (sec_len & (size_of::<PointerTrail>() - 1)) != 0 {
                        return Err(LMNK_ERR_CONTAINER_CORRUPTED);
                    }
                }
                LMNK_SECTION_BLOOM_FILTER => {
                    if sec_len != size_of::<ContrastivePhaseBloomFilter>() {
                        return Err(LMNK_ERR_CONTAINER_CORRUPTED);
                    }
                }
                LMNK_SECTION_LEDGER_SLOT => {
                    if sec_len != size_of::<LedgerSlot>() {
                        return Err(LMNK_ERR_CONTAINER_CORRUPTED);
                    }
                }
                LMNK_SECTION_CAUSAL_PROVENANCE
                    if (sec_len & (size_of::<CausalProvenanceReport>() - 1)) != 0 =>
                {
                    return Err(LMNK_ERR_CONTAINER_CORRUPTED);
                }
                _ => {}
            }

            let dst_entry = sections.get_mut(i).ok_or(LMNK_ERR_CONTAINER_CORRUPTED)?;
            *dst_entry = entry;
            prev_end = next_aligned_end;
            i += 1;
        }

        if prev_end != bytes.len() {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }

        Ok(Self {
            raw: bytes,
            header,
            sections,
        })
    }

    /// Returns the validated 64-byte [`LmnkContainerHeader`].
    #[inline(always)]
    #[must_use]
    pub const fn header(&self) -> &LmnkContainerHeader {
        &self.header
    }

    /// Returns the active slice of validated [`LmnkSectionEntry`] descriptors.
    #[inline(always)]
    #[must_use]
    pub fn sections(&self) -> &[LmnkSectionEntry] {
        let count = usize::from(self.header.section_count);
        self.sections.get(..count).unwrap_or(&[])
    }

    /// Finds the [`LmnkSectionEntry`] matching `kind`, if present.
    #[inline]
    #[must_use]
    pub fn find_section(&self, kind: u16) -> Option<LmnkSectionEntry> {
        let mut idx = 0usize;
        let count = usize::from(self.header.section_count);
        while idx < count {
            if let Some(&entry) = self.sections.get(idx)
                && entry.kind == kind
            {
                return Some(entry);
            }
            idx += 1;
        }
        None
    }

    /// Returns an alignment-independent immutable byte slice for section `kind`, if present.
    #[inline]
    #[must_use]
    pub fn section_bytes(&self, kind: u16) -> Option<&'a [u8]> {
        let entry = self.find_section(kind)?;
        let off = usize::try_from(entry.offset).ok()?;
        let len = usize::try_from(entry.length).ok()?;
        let end = off.checked_add(len)?;
        self.raw.get(off..end)
    }

    /// Returns the number of [`PhaseAnchorRecord`] elements stored in the container.
    #[inline]
    #[must_use]
    pub fn phase_anchor_count(&self) -> usize {
        self.section_bytes(LMNK_SECTION_PHASE_ANCHORS)
            .map_or(0, |b| b.len() / size_of::<PhaseAnchorRecord>())
    }

    /// Safely copies and returns the `idx`-th [`PhaseAnchorRecord`] independent of base-pointer alignment.
    #[inline]
    #[must_use]
    pub fn read_phase_anchor(&self, idx: usize) -> Option<PhaseAnchorRecord> {
        let bytes = self.section_bytes(LMNK_SECTION_PHASE_ANCHORS)?;
        let off = idx.checked_mul(size_of::<PhaseAnchorRecord>())?;
        PhaseAnchorRecord::read_unaligned_from(bytes, off).ok()
    }

    /// Returns a zero-copy [`PhaseAnchorRecord`] slice when the underlying section buffer satisfies
    /// 64-byte alignment; returns [`crate::liminal::LMNK_ERR_ALIGNMENT`] otherwise.
    ///
    /// # Errors
    /// Returns [`crate::liminal::LMNK_ERR_ALIGNMENT`] if the section payload address violates 64-byte alignment.
    #[inline]
    pub fn phase_anchors_zero_copy(&self) -> LmnkResult<&'a [PhaseAnchorRecord]> {
        match self.section_bytes(LMNK_SECTION_PHASE_ANCHORS) {
            None => Ok(&[]),
            Some(bytes) => {
                let count = bytes.len() / size_of::<PhaseAnchorRecord>();
                PhaseAnchorRecord::slice_from_bytes_at(bytes, 0, count)
            }
        }
    }

    /// Returns the number of [`PointerTrail`] elements stored in the container.
    #[inline]
    #[must_use]
    pub fn pointer_trail_count(&self) -> usize {
        self.section_bytes(LMNK_SECTION_POINTER_TRAILS)
            .map_or(0, |b| b.len() / size_of::<PointerTrail>())
    }

    /// Safely copies and returns the `idx`-th [`PointerTrail`] independent of base-pointer alignment.
    #[inline]
    #[must_use]
    pub fn read_pointer_trail(&self, idx: usize) -> Option<PointerTrail> {
        let bytes = self.section_bytes(LMNK_SECTION_POINTER_TRAILS)?;
        let off = idx.checked_mul(size_of::<PointerTrail>())?;
        PointerTrail::read_unaligned_from(bytes, off).ok()
    }

    /// Returns a zero-copy [`PointerTrail`] slice when aligned to 64 bytes.
    ///
    /// # Errors
    /// Returns [`crate::liminal::LMNK_ERR_ALIGNMENT`] if the section payload address violates 64-byte alignment.
    #[inline]
    pub fn pointer_trails_zero_copy(&self) -> LmnkResult<&'a [PointerTrail]> {
        match self.section_bytes(LMNK_SECTION_POINTER_TRAILS) {
            None => Ok(&[]),
            Some(bytes) => {
                let count = bytes.len() / size_of::<PointerTrail>();
                PointerTrail::slice_from_bytes_at(bytes, 0, count)
            }
        }
    }

    /// Safely copies the [`ContrastivePhaseBloomFilter`] section if present.
    #[inline]
    #[must_use]
    pub fn read_bloom_filter(&self) -> Option<ContrastivePhaseBloomFilter> {
        let bytes = self.section_bytes(LMNK_SECTION_BLOOM_FILTER)?;
        ContrastivePhaseBloomFilter::read_unaligned_from(bytes, 0).ok()
    }

    /// Returns a zero-copy reference to [`ContrastivePhaseBloomFilter`] when aligned to 64 bytes.
    ///
    /// # Errors
    /// Returns [`crate::liminal::LMNK_ERR_ALIGNMENT`] if the section payload address violates 64-byte alignment.
    #[inline]
    pub fn bloom_filter_zero_copy(&self) -> LmnkResult<Option<&'a ContrastivePhaseBloomFilter>> {
        match self.section_bytes(LMNK_SECTION_BLOOM_FILTER) {
            None => Ok(None),
            Some(bytes) => ContrastivePhaseBloomFilter::from_bytes_at(bytes, 0).map(Some),
        }
    }

    /// Safely copies the [`LedgerSlot`] section if present.
    #[inline]
    #[must_use]
    pub fn read_ledger_slot(&self) -> Option<LedgerSlot> {
        let bytes = self.section_bytes(LMNK_SECTION_LEDGER_SLOT)?;
        LedgerSlot::read_unaligned_from(bytes, 0).ok()
    }

    /// Returns a zero-copy reference to [`LedgerSlot`] when aligned to 64 bytes.
    ///
    /// # Errors
    /// Returns [`crate::liminal::LMNK_ERR_ALIGNMENT`] if the section payload address violates 64-byte alignment.
    #[inline]
    pub fn ledger_slot_zero_copy(&self) -> LmnkResult<Option<&'a LedgerSlot>> {
        match self.section_bytes(LMNK_SECTION_LEDGER_SLOT) {
            None => Ok(None),
            Some(bytes) => LedgerSlot::from_bytes_at(bytes, 0).map(Some),
        }
    }

    /// Returns the number of [`CausalProvenanceReport`] elements stored in the container.
    #[inline]
    #[must_use]
    pub fn causal_provenance_count(&self) -> usize {
        self.section_bytes(LMNK_SECTION_CAUSAL_PROVENANCE)
            .map_or(0, |b| b.len() / size_of::<CausalProvenanceReport>())
    }

    /// Safely copies and returns the `idx`-th [`CausalProvenanceReport`] independent of base alignment.
    #[inline]
    #[must_use]
    pub fn read_causal_provenance(&self, idx: usize) -> Option<CausalProvenanceReport> {
        let bytes = self.section_bytes(LMNK_SECTION_CAUSAL_PROVENANCE)?;
        let off = idx.checked_mul(size_of::<CausalProvenanceReport>())?;
        CausalProvenanceReport::read_unaligned_from(bytes, off).ok()
    }

    /// Returns a zero-copy [`CausalProvenanceReport`] slice when aligned to 16 bytes.
    ///
    /// # Errors
    /// Returns [`crate::liminal::LMNK_ERR_ALIGNMENT`] if the section payload address violates 16-byte alignment.
    #[inline]
    pub fn causal_provenance_zero_copy(&self) -> LmnkResult<&'a [CausalProvenanceReport]> {
        match self.section_bytes(LMNK_SECTION_CAUSAL_PROVENANCE) {
            None => Ok(&[]),
            Some(bytes) => {
                let count = bytes.len() / size_of::<CausalProvenanceReport>();
                CausalProvenanceReport::slice_from_bytes_at(bytes, 0, count)
            }
        }
    }

    /// Alignment-safe extraction of the persisted [`CognitiveTrajectoryLedger`] from
    /// [`LMNK_SECTION_LEDGER_SLOT`] when [`LEDGER_FLAG_COGNITIVE_TRAJECTORY`] is active.
    #[inline]
    #[must_use]
    pub fn cognitive_trajectory_ledger(&self) -> Option<CognitiveTrajectoryLedger> {
        match self.read_ledger_slot() {
            Some(slot) if (slot.auto_tune_flags & LEDGER_FLAG_COGNITIVE_TRAJECTORY) != 0 => {
                Some(CognitiveTrajectoryLedger::from_ledger_slot(&slot))
            }
            _ => None,
        }
    }

    /// Computes the exact 64B-aligned byte length required to write the standard active sections
    /// (`PhaseAnchors`, `PointerTrails`, optional `BloomFilter`, optional `LedgerSlot`, `CausalProvenance`).
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] on arithmetic overflow.
    #[inline]
    pub const fn required_standard_byte_len(
        num_anchors: usize,
        num_trails: usize,
        has_bloom: bool,
        has_ledger: bool,
        num_reports: usize,
    ) -> LmnkResult<usize> {
        let sec_count = (num_anchors > 0) as usize
            + (num_trails > 0) as usize
            + (has_bloom as usize)
            + (has_ledger as usize)
            + (num_reports > 0) as usize;
        let table_end = LMNK_CONTAINER_HEADER_BYTES + sec_count * LMNK_SECTION_ENTRY_BYTES;
        let Some(mut cursor) = align_up_64_checked(table_end) else {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        };

        if num_anchors > 0 {
            let Some(bytes) = num_anchors.checked_mul(size_of::<PhaseAnchorRecord>()) else {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            };
            let Some(end) = cursor.checked_add(bytes) else {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            };
            let Some(next) = align_up_64_checked(end) else {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            };
            cursor = next;
        }
        if num_trails > 0 {
            let Some(bytes) = num_trails.checked_mul(size_of::<PointerTrail>()) else {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            };
            let Some(end) = cursor.checked_add(bytes) else {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            };
            let Some(next) = align_up_64_checked(end) else {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            };
            cursor = next;
        }
        if has_bloom {
            let Some(end) = cursor.checked_add(size_of::<ContrastivePhaseBloomFilter>()) else {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            };
            let Some(next) = align_up_64_checked(end) else {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            };
            cursor = next;
        }
        if has_ledger {
            let Some(end) = cursor.checked_add(size_of::<LedgerSlot>()) else {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            };
            let Some(next) = align_up_64_checked(end) else {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            };
            cursor = next;
        }
        if num_reports > 0 {
            let Some(bytes) = num_reports.checked_mul(size_of::<CausalProvenanceReport>()) else {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            };
            let Some(end) = cursor.checked_add(bytes) else {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            };
            let Some(next) = align_up_64_checked(end) else {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            };
            cursor = next;
        }
        if cursor > LMNK_MAX_CONTAINER_BYTES {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        Ok(cursor)
    }

    /// Serializes standard runtime/compiled sections into a unified 64B-aligned `.lmnk` container,
    /// stamping any unstamped [`PhaseAnchorRecord`] CRC32s and per-section `payload_crc32` values.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] on size overflow or [`LMNK_ERR_BUFFER_TOO_SMALL`] if
    /// `dst_bytes` is smaller than [`Self::required_standard_byte_len`].
    #[inline]
    // v0.3.0 frozen public Rust API; parameter-struct consolidation deferred to v0.4.0.
    #[allow(clippy::too_many_arguments)]
    pub fn write_standard_sections_to_bytes(
        container_flags: u32,
        anchors: &[PhaseAnchorRecord],
        trails: &[PointerTrail],
        bloom_opt: Option<&ContrastivePhaseBloomFilter>,
        ledger_opt: Option<&LedgerSlot>,
        provenance_reports: &[CausalProvenanceReport],
        timestamp_epoch_sec: u64,
        dst_bytes: &mut [u8],
    ) -> LmnkResult<usize> {
        let total_len = Self::required_standard_byte_len(
            anchors.len(),
            trails.len(),
            bloom_opt.is_some(),
            ledger_opt.is_some(),
            provenance_reports.len(),
        )?;
        if dst_bytes.len() < total_len {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        let buf = dst_bytes
            .get_mut(..total_len)
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
        buf.fill(0);

        let sec_count = usize::from(!anchors.is_empty())
            + usize::from(!trails.is_empty())
            + usize::from(bloom_opt.is_some())
            + usize::from(ledger_opt.is_some())
            + usize::from(!provenance_reports.is_empty());
        let table_end = LMNK_CONTAINER_HEADER_BYTES + sec_count * LMNK_SECTION_ENTRY_BYTES;
        let mut cursor = align_up_64_checked(table_end).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        let mut sec_idx = 0usize;

        if !anchors.is_empty() {
            let off = cursor;
            let len = core::mem::size_of_val(anchors);
            let end = off + len;
            let payload_slice = buf.get_mut(off..end).ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
            let _ = PhaseAnchorRecord::write_records_to_container_bytes(anchors, payload_slice)?;
            let crc = compute_aligned64_payload_crc32_ieee(payload_slice);
            let entry = LmnkSectionEntry {
                kind: LMNK_SECTION_PHASE_ANCHORS,
                flags: 0,
                offset: off as u32,
                length: len as u32,
                payload_crc32: crc,
            };
            let entry_off = LMNK_CONTAINER_HEADER_BYTES + sec_idx * LMNK_SECTION_ENTRY_BYTES;
            buf.get_mut(entry_off..entry_off + LMNK_SECTION_ENTRY_BYTES)
                .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?
                .copy_from_slice(&entry.to_le_bytes());
            sec_idx += 1;
            cursor = align_up_64_checked(end).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        }

        if !trails.is_empty() {
            let off = cursor;
            let len = core::mem::size_of_val(trails);
            let end = off + len;
            PointerTrail::write_unaligned_slice_to(trails, buf, off)?;
            let crc = compute_aligned64_payload_crc32_ieee(
                buf.get(off..end).ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?,
            );
            let entry = LmnkSectionEntry {
                kind: LMNK_SECTION_POINTER_TRAILS,
                flags: 0,
                offset: off as u32,
                length: len as u32,
                payload_crc32: crc,
            };
            let entry_off = LMNK_CONTAINER_HEADER_BYTES + sec_idx * LMNK_SECTION_ENTRY_BYTES;
            buf.get_mut(entry_off..entry_off + LMNK_SECTION_ENTRY_BYTES)
                .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?
                .copy_from_slice(&entry.to_le_bytes());
            sec_idx += 1;
            cursor = align_up_64_checked(end).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        }

        if let Some(bloom) = bloom_opt {
            let off = cursor;
            let len = size_of::<ContrastivePhaseBloomFilter>();
            let end = off + len;
            bloom.write_unaligned_to(buf, off)?;
            let crc = compute_aligned64_payload_crc32_ieee(
                buf.get(off..end).ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?,
            );
            let entry = LmnkSectionEntry {
                kind: LMNK_SECTION_BLOOM_FILTER,
                flags: 0,
                offset: off as u32,
                length: len as u32,
                payload_crc32: crc,
            };
            let entry_off = LMNK_CONTAINER_HEADER_BYTES + sec_idx * LMNK_SECTION_ENTRY_BYTES;
            buf.get_mut(entry_off..entry_off + LMNK_SECTION_ENTRY_BYTES)
                .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?
                .copy_from_slice(&entry.to_le_bytes());
            sec_idx += 1;
            cursor = align_up_64_checked(end).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        }

        if let Some(ledger) = ledger_opt {
            let off = cursor;
            let len = size_of::<LedgerSlot>();
            let end = off + len;
            ledger.write_unaligned_to(buf, off)?;
            let crc = compute_aligned64_payload_crc32_ieee(
                buf.get(off..end).ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?,
            );
            let entry = LmnkSectionEntry {
                kind: LMNK_SECTION_LEDGER_SLOT,
                flags: 0,
                offset: off as u32,
                length: len as u32,
                payload_crc32: crc,
            };
            let entry_off = LMNK_CONTAINER_HEADER_BYTES + sec_idx * LMNK_SECTION_ENTRY_BYTES;
            buf.get_mut(entry_off..entry_off + LMNK_SECTION_ENTRY_BYTES)
                .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?
                .copy_from_slice(&entry.to_le_bytes());
            sec_idx += 1;
            cursor = align_up_64_checked(end).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        }

        if !provenance_reports.is_empty() {
            let off = cursor;
            let len = core::mem::size_of_val(provenance_reports);
            let end = off + len;
            CausalProvenanceReport::write_unaligned_slice_to(provenance_reports, buf, off)?;
            let crc = compute_aligned64_payload_crc32_ieee(
                buf.get(off..end).ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?,
            );
            let entry = LmnkSectionEntry {
                kind: LMNK_SECTION_CAUSAL_PROVENANCE,
                flags: 0,
                offset: off as u32,
                length: len as u32,
                payload_crc32: crc,
            };
            let entry_off = LMNK_CONTAINER_HEADER_BYTES + sec_idx * LMNK_SECTION_ENTRY_BYTES;
            buf.get_mut(entry_off..entry_off + LMNK_SECTION_ENTRY_BYTES)
                .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?
                .copy_from_slice(&entry.to_le_bytes());
        }

        let hdr = init_container_header(
            container_flags,
            total_len as u64,
            timestamp_epoch_sec,
            sec_count as u16,
        );
        buf.get_mut(..LMNK_CONTAINER_HEADER_BYTES)
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?
            .copy_from_slice(&hdr.to_le_bytes());
        Ok(total_len)
    }
}

const _: () = {
    assert!(size_of::<MappedStorageEngine>() == 64);
    assert!(align_of::<MappedStorageEngine>() == 64);
    assert!(offset_of!(MappedStorageEngine, file_size_bytes) == 0);
    assert!(offset_of!(MappedStorageEngine, flags) == 8);
    assert!(offset_of!(MappedStorageEngine, topology_crc32) == 12);
    assert!(offset_of!(MappedStorageEngine, reserved) == 16);
};
