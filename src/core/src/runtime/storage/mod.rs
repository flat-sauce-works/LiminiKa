// name: src/core/src/runtime/storage/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Zero-Copy In-Memory Mapped Storage (`MappedStorageEngine` 64B `Pod`,
//! [`LmnkContainerHeader`] 64B + [`LmnkSectionEntry`] 16B unified `.lmnk` container zero-copy
//! persistence & evolution ([`LmnkContainerView`]), [`KvKeyframeCache`], [`MappedStorage`],
//! phase packet codecs, and [`KvKeyframePacket`]).

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

use core::mem::{offset_of, size_of};

pub use crate::foundation::KvMotionDeltaPacket;
use crate::foundation::{Pod, compute_crc32_ieee};
use crate::liminal::{LMNK_ABI_VERSION_HEX, LMNK_ERR_BUFFER_TOO_SMALL, LmnkResult};
use crate::runtime::topology::CORE_TOPOLOGY;

mod container;
mod keyframe;

pub use container::*;
pub use keyframe::*;

/// Binary magic constant for `.lmnk` unified containers (`0x4B4E4D4C` = ASCII `"LMNK"`).
pub const LMNK_CONTAINER_MAGIC: u32 = 0x4B4E_4D4C;

/// Fixed byte length (`64` bytes) of [`LmnkContainerHeader`].
pub const LMNK_CONTAINER_HEADER_BYTES: usize = 64;

/// Fixed byte length (`16` bytes) of each [`LmnkSectionEntry`].
pub const LMNK_SECTION_ENTRY_BYTES: usize = 16;

/// Maximum number of sections permitted in a single `.lmnk` container (`32`).
pub const LMNK_MAX_CONTAINER_SECTIONS: u16 = 32;

/// Hard upper byte limit (`64 MiB`) enforced by [`LmnkContainerView::parse`].
pub const LMNK_MAX_CONTAINER_BYTES: usize = 64 * 1024 * 1024;

/// Container flag bit `0`: serialized [`crate::runtime::RuntimeContext`] snapshot.
pub const LMNK_CONTAINER_FLAG_RUNTIME_SNAPSHOT: u32 = 1 << 0;

/// Container flag bit `1`: compiled topology-dependent cache container (`.lmnkc`).
pub const LMNK_CONTAINER_FLAG_COMPILED_CACHE: u32 = 1 << 1;

/// Container flag bit `2`: portable container package (`.lmnkp`).
pub const LMNK_CONTAINER_FLAG_PORTABLE_PACKAGE: u32 = 1 << 2;

/// Bitmask of all defined [`LmnkContainerHeader::container_flags`] bits (`0x07`).
pub const LMNK_CONTAINER_KNOWN_FLAGS: u32 = LMNK_CONTAINER_FLAG_RUNTIME_SNAPSHOT
    | LMNK_CONTAINER_FLAG_COMPILED_CACHE
    | LMNK_CONTAINER_FLAG_PORTABLE_PACKAGE;

/// Section kind ID for runtime context state snapshots (`1`).
pub const LMNK_SECTION_RUNTIME_SNAPSHOT: u16 = 1;
/// Section kind ID for crystallized `PhaseAnchorRecord` arrays (`2`).
pub const LMNK_SECTION_PHASE_ANCHORS: u16 = 2;
/// Section kind ID for `PointerTrail` arrays (`3`).
pub const LMNK_SECTION_POINTER_TRAILS: u16 = 3;
/// Section kind ID for `ContrastivePhaseBloomFilter` (`4`).
pub const LMNK_SECTION_BLOOM_FILTER: u16 = 4;
/// Section kind ID for `LedgerSlot` (`5`).
pub const LMNK_SECTION_LEDGER_SLOT: u16 = 5;
/// Section kind ID for `CausalProvenanceReport` arrays (`6`).
pub const LMNK_SECTION_CAUSAL_PROVENANCE: u16 = 6;
/// Section kind ID for logical P-Frame descriptors (`7`).
pub const LMNK_SECTION_LOGICAL_PFRAMES: u16 = 7;
/// Section kind ID for recall policy and metadata (`8`).
pub const LMNK_SECTION_RECALL_METADATA: u16 = 8;
/// Reserved section kind ID for cryptographic content digests (`0xFF01`).
pub const LMNK_SECTION_CONTENT_HASH: u16 = 0xFF01;
/// Reserved section kind ID for detached container signatures (`0xFF02`).
pub const LMNK_SECTION_SIGNATURE: u16 = 0xFF02;

/// Canonical `.lmnk` section classification identifiers (`u16`).
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LmnkSectionKind {
    /// Serialized `RuntimeContext` snapshot payload (`1`).
    RuntimeSnapshot = LMNK_SECTION_RUNTIME_SNAPSHOT,
    /// Array of 128 B `PhaseAnchorRecord` entries (`2`).
    PhaseAnchors = LMNK_SECTION_PHASE_ANCHORS,
    /// Array of 128 B `PointerTrail` entries (`3`).
    PointerTrails = LMNK_SECTION_POINTER_TRAILS,
    /// 64 B `ContrastivePhaseBloomFilter` (`4`).
    BloomFilter = LMNK_SECTION_BLOOM_FILTER,
    /// 64 B `LedgerSlot` (`5`).
    LedgerSlot = LMNK_SECTION_LEDGER_SLOT,
    /// Array of 16 B `CausalProvenanceReport` entries (`6`).
    CausalProvenance = LMNK_SECTION_CAUSAL_PROVENANCE,
    /// Logical P-Frame trajectory descriptor stream (`7`).
    LogicalPFrames = LMNK_SECTION_LOGICAL_PFRAMES,
    /// Corpus recall policy / metadata payload (`8`).
    RecallMetadata = LMNK_SECTION_RECALL_METADATA,
    /// Reserved content digest section (`0xFF01`).
    ContentHash = LMNK_SECTION_CONTENT_HASH,
    /// Reserved detached signature section (`0xFF02`).
    Signature = LMNK_SECTION_SIGNATURE,
}

impl LmnkSectionKind {
    /// Returns the raw `u16` section identifier.
    #[inline(always)]
    #[must_use]
    pub const fn as_u16(self) -> u16 {
        self as u16
    }

    /// Attempts to decode a known [`LmnkSectionKind`] from a raw `u16` tag.
    #[inline(always)]
    #[must_use]
    pub const fn from_u16(raw: u16) -> Option<Self> {
        match raw {
            LMNK_SECTION_RUNTIME_SNAPSHOT => Some(Self::RuntimeSnapshot),
            LMNK_SECTION_PHASE_ANCHORS => Some(Self::PhaseAnchors),
            LMNK_SECTION_POINTER_TRAILS => Some(Self::PointerTrails),
            LMNK_SECTION_BLOOM_FILTER => Some(Self::BloomFilter),
            LMNK_SECTION_LEDGER_SLOT => Some(Self::LedgerSlot),
            LMNK_SECTION_CAUSAL_PROVENANCE => Some(Self::CausalProvenance),
            LMNK_SECTION_LOGICAL_PFRAMES => Some(Self::LogicalPFrames),
            LMNK_SECTION_RECALL_METADATA => Some(Self::RecallMetadata),
            LMNK_SECTION_CONTENT_HASH => Some(Self::ContentHash),
            LMNK_SECTION_SIGNATURE => Some(Self::Signature),
            _ => None,
        }
    }
}

/// Unified `.lmnk` binary container header (64 bytes, `#[repr(C, align(64))]`, Canonical Little-Endian).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LmnkContainerHeader {
    /// Format identifier magic constant ([`LMNK_CONTAINER_MAGIC`] = ASCII `"LMNK"`, `offset 0..4`).
    pub magic: u32,
    /// Container ABI version matching [`LMNK_ABI_VERSION_HEX`] (`offset 4..8`).
    pub version: u32,
    /// Topology CRC32 matching [`CORE_TOPOLOGY`] (`offset 8..12`).
    pub topology_crc32: u32,
    /// Container classification bitflags (`LMNK_CONTAINER_FLAG_*`, `offset 12..16`).
    pub container_flags: u32,
    /// Total container size in bytes including header, section table, padding, and payloads (`offset 16..24`).
    pub total_size: u64,
    /// Unix epoch timestamp in seconds (`offset 24..32`).
    pub timestamp_epoch_sec: u64,
    /// Byte offset to the [`LmnkSectionEntry`] table (`64`, `offset 32..36`).
    pub section_table_offset: u32,
    /// Number of section entries (`<= LMNK_MAX_CONTAINER_SECTIONS`, `offset 36..38`).
    pub section_count: u16,
    /// Header byte size (`64`, `offset 38..40`).
    pub header_size: u16,
    /// Zero-filled reserved bytes for forward ABI extensions (`offset 40..60`).
    pub reserved_future: [u8; 20],
    /// IEEE 802.3 CRC32 checksum over Canonical Little-Endian bytes `0..60` (`offset 60..64`).
    pub header_crc32: u32,
}

impl Default for LmnkContainerHeader {
    #[inline(always)]
    fn default() -> Self {
        Self {
            magic: LMNK_CONTAINER_MAGIC,
            version: LMNK_ABI_VERSION_HEX,
            topology_crc32: CORE_TOPOLOGY.topology_crc32,
            container_flags: 0,
            total_size: LMNK_CONTAINER_HEADER_BYTES as u64,
            timestamp_epoch_sec: 0,
            section_table_offset: LMNK_CONTAINER_HEADER_BYTES as u32,
            section_count: 0,
            header_size: LMNK_CONTAINER_HEADER_BYTES as u16,
            reserved_future: [0; 20],
            header_crc32: 0,
        }
    }
}

// SAFETY: `LmnkContainerHeader` is `#[repr(C, align(64))]` of exact size 64 bytes with zero implicit padding.
unsafe impl Pod for LmnkContainerHeader {}

/// Unified `.lmnk` section directory entry (16 bytes, `#[repr(C, align(16))]`, 4 entries per 64B CL).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LmnkSectionEntry {
    /// Section classification identifier (`LmnkSectionKind`, `offset 0..2`).
    pub kind: u16,
    /// Per-section flags (`offset 2..4`).
    pub flags: u16,
    /// 64-byte aligned byte offset from container start (`offset 4..8`).
    pub offset: u32,
    /// Payload byte length (`> 0` for active sections, `offset 8..12`).
    pub length: u32,
    /// IEEE 802.3 CRC32 checksum over payload bytes (`offset 12..16`).
    pub payload_crc32: u32,
}

// SAFETY: `LmnkSectionEntry` is `#[repr(C, align(16))]` of exact size 16 bytes with zero implicit padding.
unsafe impl Pod for LmnkSectionEntry {}

/// KV-keyframe index header structure (64 bytes, 32-byte aligned).
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct KvKeyframeHeader {
    /// Frame classification (`0` = I-Frame Keyframe Anchor, `1` = P-Frame Phase Motion Delta).
    pub frame_type: u32,
    /// Absolute token position index in sequence.
    pub token_index: u32,
    /// Dynamic entropy-driven Group of Pictures (GOP) length between keyframes.
    pub gop_length: u32,
    /// Mean resultant vector length $\bar{R}_c \in [0, 1]$ in circular statistics.
    pub composite_vector_length: f32,
    /// Concentration parameter $\kappa_c$ for von Mises phase distribution.
    pub von_mises_kappa: f32,
    /// Scalar residual magnitude / resultant phase angle $\bar{\theta}_c$ for sparse updates.
    pub sparse_scalar_residual: f32,
    /// Byte offset to Intra Key-Frame Cache (I-Cache) payload.
    pub icache_payload_offset: u64,
    /// Byte offset to Predictive Phase Motion Cache (P-Cache) payload.
    pub pcache_payload_offset: u64,
    /// Reserved space for future KV-keyframe extensions (24 bytes).
    pub reserved: [u8; 24],
}

// SAFETY: `KvKeyframeHeader` is `#[repr(C, align(32))]` of exact size 64 bytes with zero padding.
unsafe impl Pod for KvKeyframeHeader {}

const _: () = {
    assert!(size_of::<LmnkContainerHeader>() == 64 && align_of::<LmnkContainerHeader>() == 64);
    assert!(offset_of!(LmnkContainerHeader, magic) == 0);
    assert!(offset_of!(LmnkContainerHeader, version) == 4);
    assert!(offset_of!(LmnkContainerHeader, topology_crc32) == 8);
    assert!(offset_of!(LmnkContainerHeader, container_flags) == 12);
    assert!(offset_of!(LmnkContainerHeader, total_size) == 16);
    assert!(offset_of!(LmnkContainerHeader, timestamp_epoch_sec) == 24);
    assert!(offset_of!(LmnkContainerHeader, section_table_offset) == 32);
    assert!(offset_of!(LmnkContainerHeader, section_count) == 36);
    assert!(offset_of!(LmnkContainerHeader, header_size) == 38);
    assert!(offset_of!(LmnkContainerHeader, reserved_future) == 40);
    assert!(offset_of!(LmnkContainerHeader, header_crc32) == 60);
    assert!(size_of::<LmnkSectionEntry>() == 16 && align_of::<LmnkSectionEntry>() == 16);
    assert!(offset_of!(LmnkSectionEntry, kind) == 0);
    assert!(offset_of!(LmnkSectionEntry, flags) == 2);
    assert!(offset_of!(LmnkSectionEntry, offset) == 4);
    assert!(offset_of!(LmnkSectionEntry, length) == 8);
    assert!(offset_of!(LmnkSectionEntry, payload_crc32) == 12);
    assert!(size_of::<KvKeyframeHeader>() == 64 && align_of::<KvKeyframeHeader>() == 32);
};

#[inline(always)]
fn read_le_bytes_exact<const N: usize>(src: &[u8], off: usize) -> LmnkResult<[u8; N]> {
    let end = off.checked_add(N).ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
    let slice = src.get(off..end).ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
    <[u8; N]>::try_from(slice).map_err(|_| LMNK_ERR_BUFFER_TOO_SMALL)
}

impl LmnkContainerHeader {
    /// Serializes the protected header prefix (`bytes 0..60`) in Canonical Little-Endian order.
    #[inline(always)]
    #[must_use]
    pub const fn to_prefix_le_bytes(&self) -> [u8; 60] {
        let m = self.magic.to_le_bytes();
        let v = self.version.to_le_bytes();
        let tc = self.topology_crc32.to_le_bytes();
        let cf = self.container_flags.to_le_bytes();
        let ts = self.total_size.to_le_bytes();
        let ep = self.timestamp_epoch_sec.to_le_bytes();
        let sto = self.section_table_offset.to_le_bytes();
        let sc = self.section_count.to_le_bytes();
        let hs = self.header_size.to_le_bytes();
        [
            m[0],
            m[1],
            m[2],
            m[3],
            v[0],
            v[1],
            v[2],
            v[3],
            tc[0],
            tc[1],
            tc[2],
            tc[3],
            cf[0],
            cf[1],
            cf[2],
            cf[3],
            ts[0],
            ts[1],
            ts[2],
            ts[3],
            ts[4],
            ts[5],
            ts[6],
            ts[7],
            ep[0],
            ep[1],
            ep[2],
            ep[3],
            ep[4],
            ep[5],
            ep[6],
            ep[7],
            sto[0],
            sto[1],
            sto[2],
            sto[3],
            sc[0],
            sc[1],
            hs[0],
            hs[1],
            self.reserved_future[0],
            self.reserved_future[1],
            self.reserved_future[2],
            self.reserved_future[3],
            self.reserved_future[4],
            self.reserved_future[5],
            self.reserved_future[6],
            self.reserved_future[7],
            self.reserved_future[8],
            self.reserved_future[9],
            self.reserved_future[10],
            self.reserved_future[11],
            self.reserved_future[12],
            self.reserved_future[13],
            self.reserved_future[14],
            self.reserved_future[15],
            self.reserved_future[16],
            self.reserved_future[17],
            self.reserved_future[18],
            self.reserved_future[19],
        ]
    }

    /// Serializes the full 64-byte [`LmnkContainerHeader`] into Canonical Little-Endian bytes.
    #[inline(always)]
    #[must_use]
    pub const fn to_le_bytes(&self) -> [u8; 64] {
        let p = self.to_prefix_le_bytes();
        let c = self.header_crc32.to_le_bytes();
        [
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13],
            p[14], p[15], p[16], p[17], p[18], p[19], p[20], p[21], p[22], p[23], p[24], p[25],
            p[26], p[27], p[28], p[29], p[30], p[31], p[32], p[33], p[34], p[35], p[36], p[37],
            p[38], p[39], p[40], p[41], p[42], p[43], p[44], p[45], p[46], p[47], p[48], p[49],
            p[50], p[51], p[52], p[53], p[54], p[55], p[56], p[57], p[58], p[59], c[0], c[1], c[2],
            c[3],
        ]
    }

    /// Decodes a 64-byte [`LmnkContainerHeader`] from Canonical Little-Endian bytes without
    /// assuming base-pointer alignment.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`] if `src.len() < 64`.
    #[inline]
    pub fn from_le_bytes(src: &[u8]) -> LmnkResult<Self> {
        if src.len() < LMNK_CONTAINER_HEADER_BYTES {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        Ok(Self {
            magic: u32::from_le_bytes(read_le_bytes_exact(src, 0)?),
            version: u32::from_le_bytes(read_le_bytes_exact(src, 4)?),
            topology_crc32: u32::from_le_bytes(read_le_bytes_exact(src, 8)?),
            container_flags: u32::from_le_bytes(read_le_bytes_exact(src, 12)?),
            total_size: u64::from_le_bytes(read_le_bytes_exact(src, 16)?),
            timestamp_epoch_sec: u64::from_le_bytes(read_le_bytes_exact(src, 24)?),
            section_table_offset: u32::from_le_bytes(read_le_bytes_exact(src, 32)?),
            section_count: u16::from_le_bytes(read_le_bytes_exact(src, 36)?),
            header_size: u16::from_le_bytes(read_le_bytes_exact(src, 38)?),
            reserved_future: read_le_bytes_exact(src, 40)?,
            header_crc32: u32::from_le_bytes(read_le_bytes_exact(src, 60)?),
        })
    }

    /// Computes the IEEE 802.3 CRC32 checksum over bytes `0..60`.
    #[inline(always)]
    #[must_use]
    pub const fn compute_header_crc32(&self) -> u32 {
        compute_crc32_ieee(&self.to_prefix_le_bytes())
    }

    /// Stamps `self.header_crc32` and returns the finalized checksum.
    #[inline(always)]
    pub const fn finalize_crc32(&mut self) -> u32 {
        let crc = self.compute_header_crc32();
        self.header_crc32 = crc;
        crc
    }
}

impl LmnkSectionEntry {
    /// Serializes a 16-byte [`LmnkSectionEntry`] into Canonical Little-Endian bytes.
    #[inline(always)]
    #[must_use]
    pub const fn to_le_bytes(&self) -> [u8; 16] {
        let k = self.kind.to_le_bytes();
        let f = self.flags.to_le_bytes();
        let o = self.offset.to_le_bytes();
        let l = self.length.to_le_bytes();
        let c = self.payload_crc32.to_le_bytes();
        [
            k[0], k[1], f[0], f[1], o[0], o[1], o[2], o[3], l[0], l[1], l[2], l[3], c[0], c[1],
            c[2], c[3],
        ]
    }

    /// Decodes a 16-byte [`LmnkSectionEntry`] from Canonical Little-Endian bytes without
    /// assuming base-pointer alignment.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`] if `src.len() < 16`.
    #[inline]
    pub fn from_le_bytes(src: &[u8]) -> LmnkResult<Self> {
        if src.len() < LMNK_SECTION_ENTRY_BYTES {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        Ok(Self {
            kind: u16::from_le_bytes(read_le_bytes_exact(src, 0)?),
            flags: u16::from_le_bytes(read_le_bytes_exact(src, 2)?),
            offset: u32::from_le_bytes(read_le_bytes_exact(src, 4)?),
            length: u32::from_le_bytes(read_le_bytes_exact(src, 8)?),
            payload_crc32: u32::from_le_bytes(read_le_bytes_exact(src, 12)?),
        })
    }
}

/// Constructs a canonical [`LmnkContainerHeader`] initialized with magic `"LMNK"`,
/// [`LMNK_ABI_VERSION_HEX`], [`CORE_TOPOLOGY`], `container_flags`, `total_size`, and `section_count`.
#[inline(always)]
#[must_use]
pub const fn init_container_header(
    container_flags: u32,
    total_size: u64,
    timestamp_epoch_sec: u64,
    section_count: u16,
) -> LmnkContainerHeader {
    let mut hdr = LmnkContainerHeader {
        magic: LMNK_CONTAINER_MAGIC,
        version: LMNK_ABI_VERSION_HEX,
        topology_crc32: CORE_TOPOLOGY.topology_crc32,
        container_flags,
        total_size,
        timestamp_epoch_sec,
        section_table_offset: LMNK_CONTAINER_HEADER_BYTES as u32,
        section_count,
        header_size: LMNK_CONTAINER_HEADER_BYTES as u16,
        reserved_future: [0; 20],
        header_crc32: 0,
    };
    let _ = hdr.finalize_crc32();
    hdr
}

#[cfg(test)]
mod tests;
